"""Load the GenCast checkpoint and build the pmapped forward function.

Wrapping mirrors the notebook: ``GenCast -> InputsAndResiduals -> NaNCleaner(SST)``, JIT-ed
and pmapped over the ensemble (``dim="sample"``). On an 8-GPU node each pmap wave runs 8
ensemble members in parallel, one per H100; ``rollout.chunked_prediction_generator_multiple_runs``
loops the waves, so the member count must be a multiple of the device count.
"""
from __future__ import annotations

import os
import dataclasses
import functools

import jax
import haiku as hk
import xarray as xr

from graphcast import (checkpoint, normalization, xarray_jax,
                       gencast, nan_cleaning, casting)

from . import config as C


def _open_local_or_gcs(local_path, gcs_url: str):
    """Return a binary file object, preferring the local copy."""
    import os
    if os.path.exists(local_path):
        return open(local_path, "rb")
    import gcsfs
    return gcsfs.GCSFileSystem(token="anon").open(gcs_url, "rb")


def _load_stats() -> dict:
    out = {}
    for nm in C.STATS_FILES:
        with _open_local_or_gcs(C.STATS_DIR / nm, C.STATS_DIR_GCS + nm) as fobj:
            out[nm.removesuffix(".nc")] = xr.load_dataset(fobj).compute()
    return out


def _with_params(fn, params, state):
    # Bind params/state positionally (apply signature: params, state, rng, inputs, ...).
    # xarray_jax.pmap calls the wrapped fn positionally, so these must not be keywords.
    return functools.partial(fn, params, state)


def _drop_state(fn):
    # Accept both positional (pmap) and keyword call styles; drop the haiku state output.
    return lambda *a, **kw: fn(*a, **kw)[0]


def n_members_for_devices(requested: int, n_devices: int) -> int:
    """Round the requested ensemble size up to a multiple of the device count.

    ``chunked_prediction_generator_multiple_runs`` asserts ``num_samples % n_devices == 0``.
    """
    if requested % n_devices == 0:
        return requested
    bumped = ((requested + n_devices - 1) // n_devices) * n_devices
    print(f"[model] ensemble {requested} not a multiple of {n_devices} devices "
          f"-> using {bumped} members")
    return bumped


def load_gencast(n_members: int | None = None, params_file: str | None = None,
                 res: float = 1.0):
    """Return a bundle dict: forward (pmapped), task_config, n_members, devices.

    ``params_file`` / ``res`` select which checkpoint to load (default: the 1.0deg Mini).
    The cross-resolution experiment passes the full ``GenCast 1p0deg <2019.npz`` /
    ``GenCast 0p25deg <2019.npz`` checkpoints here. The GPU attention swap below is applied
    identically regardless of checkpoint."""
    cfg = C.model_cfg("gencast", res=res, params_file=params_file)
    devices = list(jax.local_devices())
    n_cap = os.environ.get("XRES_INFER_GPUS")
    if n_cap is not None:
        n_cap = int(n_cap)
        if 0 < n_cap < len(devices):
            devices = devices[:n_cap]
            print(f"[model] limiting JAX to {n_cap} GPU(s) via XRES_INFER_GPUS")
    n_dev = len(devices)
    requested = C.N_MEMBERS if n_members is None else n_members
    n_members = n_members_for_devices(requested, n_dev)

    with _open_local_or_gcs(C.PARAMS_DIR / cfg["params_file"],
                            C.PARAMS_DIR_GCS + cfg["params_file"]) as fobj:
        ckpt = checkpoint.load(fobj, gencast.CheckPoint)
    params, state = ckpt.params, {}
    task = ckpt.task_config
    st = _load_stats()

    # The checkpoint trained on TPU with "splash_mha" (a Pallas kernel that fails on GPU:
    # "scalar prefetch not implemented in the Triton backend"). Swap to GPU-compatible
    # dense attention before building the predictor.
    darch = dataclasses.replace(
        ckpt.denoiser_architecture_config,
        sparse_transformer_config=dataclasses.replace(
            ckpt.denoiser_architecture_config.sparse_transformer_config,
            attention_type="triblockdiag_mha", mask_type="full"))

    def _wrap():
        p = gencast.GenCast(sampler_config=ckpt.sampler_config,
                            task_config=ckpt.task_config,
                            denoiser_architecture_config=darch,
                            noise_config=ckpt.noise_config,
                            noise_encoder_config=ckpt.noise_encoder_config)
        # Optional memory-reduction: cast inputs/activations to bfloat16.
        # This may reduce GPU VRAM pressure for the 0.25° model on smaller cards.
        if os.environ.get("XRES_BF16") == "1":
            p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(
                p, diffs_stddev_by_level=st["diffs_stddev_by_level"],
                mean_by_level=st["mean_by_level"], stddev_by_level=st["stddev_by_level"])
        p = nan_cleaning.NaNCleaner(predictor=p, reintroduce_nans=True,
                                    fill_value=st["min_by_level"],
                                    var_to_clean="sea_surface_temperature")
        return p

    @hk.transform_with_state
    def _forward(inputs, targets_template, forcings):
        return _wrap()(inputs, targets_template=targets_template, forcings=forcings)

    jitted = _drop_state(_with_params(jax.jit(_forward.apply), params, state))
    serial = os.environ.get("XRES_SERIAL_INFER") == "1"
    if serial:
        forward = jitted
        print(f"[model] serial infer on {n_dev} device(s); ensemble={n_members} "
              f"({n_members} sample(s), one at a time). "
              f"task pressure levels: {len(task.pressure_levels)}")
    else:
        forward = xarray_jax.pmap(jitted, dim="sample")
        print(f"[model] GenCast loaded on {n_dev} device(s); ensemble={n_members} "
              f"({n_members // n_dev} wave(s) of {n_dev}). "
              f"task pressure levels: {len(task.pressure_levels)}")
    return dict(forward=forward, forward_jit=jitted, task_config=task,
                n_members=n_members, devices=devices, serial=serial)
