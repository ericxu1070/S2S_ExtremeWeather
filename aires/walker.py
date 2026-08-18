#!/usr/bin/env python
"""Roll one GenCast walker segment from a global, restartable 2-frame state.

Why this is not just ``xres/xinference.py``
-------------------------------------------
The xres inference path rolls a member from an ERA5 initial condition straight to the
horizon and keeps a CONUS crop. AI+RES cannot use it for two reasons, both structural:

1. **A walker must stop and restart.** Resampling interrupts every trajectory at each
   ``t_k``, clones some and kills others, then continues the survivors. That needs the
   model state at ``t_k`` on disk, not just a verification field.
2. **The restart state must be GLOBAL.** ``xres/xinference.py`` crops to CONUS *before*
   the host transfer (``.sel(lat=C.LAT, lon=C.LON)``) precisely to keep host RAM bounded,
   so the cached cubes are ~40x too small to restart from - and FCN3, which scores the
   walker, consumes the whole sphere anyway.

So this module keeps both: the **global** rolling 2-frame buffer that a restart and the
FCN3 adapter need, and the **cropped** per-step diagnostics that the observable needs. The
global field for a step is materialised once, copied into the crop, and dropped; only two
global frames (~0.70 GB) are ever resident. Writing every global frame instead would cost
~15 GB per 21-day walker and is what the crop in xres was avoiding.

Cloning
-------
Each segment is rolled with ``fold_in(fold_in(base, step), walker)``, so a cloned walker -
a new walker index continuing a parent's state - draws diffusion noise no sibling has
used, and the clones separate on their own. This is the one place where a stochastic
walker buys something the paper could not have: with a deterministic GCM, clones are
identical and must be split by hand-tuned perturbations to the spherical harmonics of
log surface pressure.

Env: the GenCast env (``moe`` here, ``my-env`` on Derecho). Rolling needs a GPU; the
``--check`` mode does not and is meant for the login node.

    # login node: validate batch construction (the 267-channel invariant) - no GPU
    PYTHONPATH=. python -m aires.walker --check --event PNW_HeatDome_2021 --steps 6

    # in an sbatch job: roll walker 0's first segment from the event's ERA5 init
    PYTHONPATH=. XRES_SERIAL_INFER=1 XRES_BF16=1 \\
        python -m aires.walker --event PNW_HeatDome_2021 --walker 0 --step 1 --steps 6
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A

STEP_H = A.WALKER_STEP_H

# ``gencast_s2s/inference.py`` prints a grid-node channel count and its comment names 267
# as the expected value. That number is informational and does not hold for an arbitrary
# segment length - targets and forcings both scale with the number of steps - so it is not
# reused here. The failure it guards against is real and worth asserting, but the precise
# statement of it is structural rather than a total: a static variable that leaks a
# ``time`` axis becomes TWO input channels instead of one, and the encoder then sees more
# features than the checkpoint was built for. :func:`check_batch_structure` asserts that
# directly, together with the per-step scaling of targets and forcings.
N_INPUT_FRAMES = 2


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #
def prognostic(ds: xr.Dataset) -> xr.Dataset:
    return ds[[v for v in ds.data_vars if "time" in ds[v].dims]]


def statics(ds: xr.Dataset) -> xr.Dataset:
    return ds[[v for v in ds.data_vars if "time" not in ds[v].dims]]


def check_state(ds: xr.Dataset, *, where: str = "state") -> None:
    """A restart state must be global, exactly 2 frames, and carry all 14 variables."""
    if ds.sizes.get("lat") != A.STATE_NLAT or ds.sizes.get("lon") != A.STATE_NLON:
        raise ValueError(
            f"{where}: need the global {A.STATE_NLAT}x{A.STATE_NLON} grid, got "
            f"{ds.sizes.get('lat')}x{ds.sizes.get('lon')}. A CONUS-cropped cube cannot "
            "restart a walker or initialise FCN3.")
    if ds.sizes.get("time") != 2:
        raise ValueError(f"{where}: GenCast needs exactly 2 input frames, got "
                         f"{ds.sizes.get('time')}")
    missing = [v for v in A.STATE_PROGNOSTIC + A.STATE_STATIC if v not in ds]
    if missing:
        raise ValueError(f"{where}: missing {missing}")
    t = pd.DatetimeIndex(np.asarray(ds["time"].values).ravel())
    dt = (t[1] - t[0]) / pd.Timedelta(hours=1)
    if abs(dt - STEP_H) > 1e-6:
        raise ValueError(f"{where}: frames are {dt} h apart, GenCast needs {STEP_H} h")


def valid_time(ds: xr.Dataset) -> pd.Timestamp:
    """The state's most recent frame - the instant a walker has reached."""
    return pd.DatetimeIndex(np.asarray(ds["time"].values).ravel())[-1]


def read_state(path) -> xr.Dataset:
    ds = xr.open_dataset(path).load()
    check_state(ds, where=str(path))
    return ds


def write_state(ds: xr.Dataset, path, *, compress: bool = True) -> Path:
    """Atomically write a restart state.

    float32, always. aires.md floats float16 for archived states to halve the ~0.70 GB
    footprint; that is unsafe here and must not be adopted without a per-variable
    encoding: ``geopotential`` at 50 hPa is ~2.0e5 m2 s-2, well past float16's 65504
    ceiling, so the archive would silently store ``inf`` for the top of every column.
    zlib on float32 is the safe halving (netCDF ``scale_factor``/``add_offset`` int16
    packing would be the principled alternative).
    """
    check_state(ds, where=str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = ({v: {"zlib": True, "complevel": 4} for v in ds.data_vars} if compress else None)
    tmp = path.with_suffix(f".nc.tmp.{os.getpid()}")
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)
    return path


def initial_state(event: str, **kw) -> xr.Dataset:
    """A walker's step-0 state: the event's ERA5 init frames, already global and 2-frame."""
    p = A.gencast_inputs_path(event, **kw)
    if not p.exists():
        raise FileNotFoundError(
            f"no init frames for {event} at {p}\n"
            "Build them with the xres prep stage on a node with internet:\n"
            f"  XRES_EVENTS_SEL={event} python run_xres.py --stage prep")
    return read_state(p)


# --------------------------------------------------------------------------- #
# Batch construction
# --------------------------------------------------------------------------- #
def build_batch(state: xr.Dataset, n_steps: int, step_h: int = STEP_H):
    """2-frame state + ``n_steps`` NaN target frames, as GenCast's rollout wants it.

    Delegates to ``gencast_s2s.inference.build_example_batch`` rather than restating it.
    That function carries two hard-won invariants in its body - ``data_vars="minimal"`` on
    the concat, and stripping any ``time`` axis that leaks onto a static - and duplicating
    them here would mean maintaining both copies. It is parameterised by (peak, lead_days)
    because the original experiment always rolled to an event peak; a walker segment is
    an arbitrary number of steps from an arbitrary state, so the two are back-computed and
    the resulting init is asserted to be the state's own last frame.
    """
    from gencast_s2s.inference import build_example_batch

    check_state(state, where="build_batch(state)")
    last = valid_time(state)
    peak = last + pd.Timedelta(hours=n_steps * step_h)
    full, init = build_example_batch(state, peak, n_steps * step_h / 24.0, step_h)
    if pd.Timestamp(init) != last:
        raise RuntimeError(f"batch init {init} != state's last frame {last}")
    return full, pd.Timestamp(init)


def _n_channels(ds: xr.Dataset) -> int:
    return int(sum(
        np.prod([ds[v].sizes[d] for d in ds[v].dims if d not in ("lat", "lon")] or [1])
        for v in ds.data_vars))


def check_batch_structure(inputs: xr.Dataset, targets: xr.Dataset, forcings: xr.Dataset,
                          n_steps: int) -> dict:
    """Assert the extracted arrays have the shape GenCast's encoder expects.

    Three independent statements, each catching a different way a batch built from an
    ARBITRARY state (rather than from ERA5 at a fixed lead) can go wrong:

    * every static contributes exactly one channel - the ``time``-axis leak;
    * every non-static input carries exactly the two frames GenCast conditions on;
    * targets and forcings span exactly the segment length, so a mis-specified
      ``target_lead_times`` slice cannot silently shorten or lengthen the rollout.
    """
    for v in A.STATE_STATIC:
        if v in inputs and inputs[v].dims != ("lat", "lon"):
            raise RuntimeError(
                f"static {v!r} has dims {inputs[v].dims}, expected ('lat', 'lon'). "
                "A leaked `time` axis turns it into two input channels and the encoder "
                "sees more features than the checkpoint was built for.")
    bad = {v: inputs[v].sizes["time"] for v in inputs.data_vars
           if "time" in inputs[v].dims and inputs[v].sizes["time"] != N_INPUT_FRAMES}
    if bad:
        raise RuntimeError(f"inputs must carry exactly {N_INPUT_FRAMES} frames; got {bad}")
    for nm, ds in (("targets", targets), ("forcings", forcings)):
        bad = {v: ds[v].sizes.get("time") for v in ds.data_vars
               if ds[v].sizes.get("time") != n_steps}
        if bad:
            raise RuntimeError(f"{nm} must span {n_steps} step(s); got {bad}")
    return dict(inputs=_n_channels(inputs), targets=_n_channels(targets),
                forcings=_n_channels(forcings))


def eval_arrays(bundle: dict, state: xr.Dataset, n_steps: int, step_h: int = STEP_H,
                verbose: bool = True):
    """(inputs, NaN targets template, forcings, init) for a segment.

    Forcings for arbitrary future timestamps come free: GenCast's are only year- and
    day-progress sin/cos, which ``data_utils`` derives from the datetime coordinate, so a
    walker can be restarted at any time without a forcing dataset. (Note that
    ``total_precipitation_12hr`` is a GenCast TARGET but not an INPUT, so it is not needed
    to restart - the state carries it anyway because the precip metric derives from it.)
    """
    from graphcast import data_utils

    full, init = build_batch(state, n_steps, step_h)
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        full, target_lead_times=slice(f"{step_h}h", f"{n_steps * step_h}h"),
        **dataclasses.asdict(bundle["task_config"]))
    nch = check_batch_structure(inputs, targets, forcings, n_steps)
    if verbose:
        print(f"  [walker] grid-node channels: inputs {nch['inputs']} + targets "
              f"{nch['targets']} + forcings {nch['forcings']}; {n_steps} step(s) "
              f"from {init}", flush=True)
    return inputs, targets * np.nan, forcings, init


# --------------------------------------------------------------------------- #
# RNG
# --------------------------------------------------------------------------- #
def segment_key(walker: int, step: int, base_seed: int = A.WALKER_BASE_SEED):
    """The diffusion noise stream for one segment. Unique per (walker, step)."""
    import jax

    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(base_seed),
                                                 int(step)), int(walker))


# --------------------------------------------------------------------------- #
# Rolling
# --------------------------------------------------------------------------- #
def _to_valid_time(ds: xr.Dataset, init) -> xr.Dataset:
    """Rollout lead-time axis -> absolute valid datetimes (state files are absolute)."""
    td = ds["time"].values.astype("timedelta64[ns]")
    return ds.assign_coords(time=("time", np.datetime64(init) + td))


def _tidy(ds: xr.Dataset) -> xr.Dataset:
    """Drop the rollout's bookkeeping coords so a state file matches the inputs schema."""
    return ds.drop_vars([c for c in ("datetime", "lead_h", "sample") if c in ds.coords],
                        errors="ignore")


def roll_segment(bundle: dict, state: xr.Dataset, n_steps: int, key, *,
                 step_h: int = STEP_H, crop: bool = True, verbose: bool = True,
                 label: str = ""):
    """Roll ``n_steps`` from ``state``; return (next global 2-frame state, diagnostics).

    Memory: the ring buffer holds two GLOBAL frames (~0.70 GB total) and nothing else
    global survives a loop iteration.
    """
    from graphcast import rollout
    from gencast_s2s import config as C

    inputs, targets_nan, forcings, init = eval_arrays(bundle, state, n_steps, step_h,
                                                      verbose=verbose)
    gen = rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=bundle["forward"], rngs=np.stack([key], axis=0),
        inputs=inputs, targets_template=targets_nan, forcings=forcings,
        num_steps_per_chunk=1, num_samples=1, pmap_devices=None, verbose=False)

    prog = _tidy(prognostic(state))
    ring = deque([prog.isel(time=[i]) for i in range(prog.sizes["time"])], maxlen=2)
    diag: list[xr.Dataset] = []

    done = 0
    for chunk in gen:
        g = chunk.isel(sample=0, drop=True) if "sample" in chunk.dims else chunk
        g = _tidy(_to_valid_time(g, init)).compute()      # the one global materialisation
        ring.append(g)
        if crop:
            # .sel with slices is BASIC indexing -> a view that pins the whole global
            # array alive. Deep-copy, or every step's global frame is retained and the
            # process grows to tens of GB - the exact failure the xres crop avoids.
            diag.append(g.sel(lat=C.LAT, lon=C.LON).copy(deep=True))
        done += 1
        if verbose:
            print(f"  [walker{label}] step {done}/{n_steps}  valid "
                  f"{pd.DatetimeIndex(g['time'].values)[-1]}", flush=True)
        del chunk, g

    if done != n_steps:
        raise RuntimeError(f"segment produced {done}/{n_steps} steps")

    nxt = _tidy(xr.concat(list(ring), dim="time", coords="minimal", compat="override"))
    nxt = xr.merge([nxt, statics(state)], compat="override")
    nxt.attrs.update(state.attrs)
    check_state(nxt, where="roll_segment(result)")

    cube = None
    if crop:
        cube = xr.concat(diag, dim="time", coords="minimal", compat="override")
        # Relative to THIS SEGMENT's start, not the event init - a walker is a chain of
        # segments and each one restarts the count. `time` is absolute and is what
        # anything spanning segments should join on.
        cube = cube.assign_coords(
            lead_h=("time", ((pd.DatetimeIndex(cube["time"].values) - init)
                             / pd.Timedelta(hours=1)).values.astype("int32")))
        cube["lead_h"].attrs["long_name"] = "hours since this segment's initial frame"
    return nxt, cube


# --------------------------------------------------------------------------- #
# Cache-aware segment driver
# --------------------------------------------------------------------------- #
def run_segment(get_bundle, event: str, walker: int, step: int, n_steps: int, *,
                parent_state: Path | None = None, base_seed: int = A.WALKER_BASE_SEED,
                step_h: int = STEP_H, save_diag: bool = True) -> dict:
    """Roll one segment, cached on disk. Re-running a finished segment is a no-op.

    ``parent_state`` is the state this segment continues; ``None`` means step 0, the
    event's ERA5 init. A CLONE is expressed by pointing a new walker index at a parent's
    state file - the differing walker index gives it its own noise stream.
    """
    A.ensure_dirs(event)
    sp, dp = A.state_path(event, walker, step), A.diag_path(event, walker, step)
    if sp.exists() and (dp.exists() or not save_diag):
        print(f"  [walker] w{walker:02d} step{step}: cached, skip")
        return dict(state=sp, diag=dp if dp.exists() else None, rolled=False)

    state = read_state(parent_state) if parent_state else initial_state(event)
    t0 = valid_time(state)
    print(f"  [walker] w{walker:02d} step{step}: {n_steps} x {step_h} h from {t0}", flush=True)

    nxt, cube = roll_segment(get_bundle(), state, n_steps,
                             segment_key(walker, step, base_seed),
                             step_h=step_h, crop=save_diag,
                             label=f" w{walker:02d} s{step}")
    nxt.attrs.update(event=event, walker=int(walker), step=int(step),
                     n_steps=int(n_steps), step_h=int(step_h), base_seed=int(base_seed),
                     parent=str(parent_state) if parent_state else "init",
                     valid_time=str(valid_time(nxt)))
    sp.parent.mkdir(parents=True, exist_ok=True)
    write_state(nxt, sp)
    print(f"  [walker] wrote {sp} ({sp.stat().st_size/1e6:.0f} MB), valid {valid_time(nxt)}")
    if cube is not None:
        cube.attrs.update(nxt.attrs)
        tmp = dp.with_suffix(f".nc.tmp.{os.getpid()}")
        cube.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4}
                                      for v in cube.data_vars})
        os.replace(tmp, dp)
        print(f"  [walker] wrote {dp} ({dp.stat().st_size/1e6:.0f} MB, "
              f"{cube.sizes['time']} steps)")
    return dict(state=sp, diag=dp if cube is not None else None, rolled=True)


# --------------------------------------------------------------------------- #
def _task_config_only(res: str = A.WALKER_RES):
    """Just the checkpoint's task config - enough for --check, and no GPU/model build."""
    from graphcast import checkpoint, gencast
    from gencast_s2s import config as C
    from gencast_s2s.model import _open_local_or_gcs

    cfg = C.model_cfg("gencast", res=0.25 if res == "0p25" else 1.0)
    with _open_local_or_gcs(C.PARAMS_DIR / cfg["params_file"],
                            C.PARAMS_DIR_GCS + cfg["params_file"]) as f:
        return checkpoint.load(f, gencast.CheckPoint).task_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default="PNW_HeatDome_2021")
    ap.add_argument("--walker", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6, help="GenCast steps in this segment")
    ap.add_argument("--parent", default=None, help="parent state.nc (default: event init)")
    ap.add_argument("--base-seed", type=int, default=A.WALKER_BASE_SEED)
    ap.add_argument("--no-diag", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="build the batch and verify the channel count; no GPU, no rollout")
    a = ap.parse_args(argv)

    if a.check:
        state = (read_state(a.parent) if a.parent else initial_state(a.event))
        print(f"[check] {a.event}: state valid {valid_time(state)}, "
              f"{state.sizes['lat']}x{state.sizes['lon']} global, "
              f"{len(A.STATE_PROGNOSTIC)} prognostic + {len(A.STATE_STATIC)} static")
        eval_arrays({"task_config": _task_config_only()}, state, a.steps)
        print(f"[check] OK - a {a.steps}-step segment builds cleanly from an arbitrary "
              f"state: statics stayed 2-D, inputs carry {N_INPUT_FRAMES} frames, and "
              f"targets/forcings span exactly {a.steps} step(s).")
        return 0

    from gencast_s2s import model as M

    bundle = {}

    def get_bundle():
        if not bundle:
            bundle.update(M.load_gencast(
                n_members=1, res=0.25 if A.WALKER_RES == "0p25" else 1.0))
        return bundle

    run_segment(get_bundle, a.event, a.walker, a.step, a.steps,
                parent_state=a.parent, base_seed=a.base_seed, save_diag=not a.no_diag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
