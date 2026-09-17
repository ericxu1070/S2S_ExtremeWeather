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
from datetime import datetime, timezone
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


# (allocator key, printed label, is it a byte count). Stable order, widest context
# first. Which of these a backend actually populates is a PROPERTY OF THE JAX BUILD, not
# of the run: jax 0.10.2 here reports only ``peak_bytes_in_use``, ``bytes_limit`` and
# ``num_allocs``. The probe used to print ``st.get(key, 0)`` for all six, so every one of
# job 1216's 208 lines carried ``in_use=0.00 GiB reserved=0.00 GiB
# largest_free_block=0.00 GiB`` - which reads as a full arena with nothing allocatable and
# misled the first analysis of that job. An ABSENT key and a key that is genuinely zero
# are different facts, so absent keys are omitted rather than printed as 0.
_MEMSTAT_KEYS = (
    ("bytes_in_use", "in_use", True),
    ("peak_bytes_in_use", "peak", True),
    ("bytes_reserved", "reserved", True),
    ("bytes_limit", "limit", True),
    ("largest_free_block_bytes", "largest_free_block", True),
    ("num_allocs", "num_allocs", False),
)


def _memstats(where: str) -> None:
    """Env-gated (``AIRES_MEMSTATS=1``) snapshot of device 0's allocator, for the acal
    array 1209 OOM investigation (acal/HANDOFF.md). No-op otherwise - and deliberately so:
    ``jax.devices()`` initialises the backend, which a login-node import must not do."""
    if not os.environ.get("AIRES_MEMSTATS"):
        return
    try:
        import jax
        st = jax.devices()[0].memory_stats() or {}
    except Exception as e:                      # CPU backend, or no device
        print(f"  [memstats {where}] unavailable ({e})", flush=True)
        return
    gib = 1.0 / 2 ** 30
    parts = [(f"{label}={st[key] * gib:.2f} GiB" if is_bytes else f"{label}={st[key]}")
             for key, label, is_bytes in _MEMSTAT_KEYS if key in st]
    print(f"  [memstats {where}] " + (" ".join(parts) if parts
                                      else "no keys reported by this backend"), flush=True)


def _stamp() -> str:
    """``AIRES_MEMSTATS``-gated UTC prefix for the walker's progress lines; ``""`` else.

    Same gate as :func:`_memstats` on purpose. A memory number is only readable against a
    clock: reconciling job 1216's allocator lines with the 5 s ``nvidia-smi`` sampler and
    with a foreign tenant's arrival (job 1215 lost all 8 cards 6 s after its guard read
    them empty) meant aligning two logs that shared no timestamp. Gating it keeps the
    DEFAULT log format byte-for-byte what every earlier run wrote, so log parsers and
    line-for-line diffs against banked runs still hold.
    """
    if not os.environ.get("AIRES_MEMSTATS"):
        return ""
    return datetime.now(timezone.utc).strftime("[%Y-%m-%dT%H:%M:%SZ] ")


def _host_materialize(ds: xr.Dataset) -> xr.Dataset:
    """Rebuild ``ds`` with every variable and coordinate pulled to host, one at a time.

    A state fresh off ``roll_segment`` is still device-resident: each prognostic
    ``DataArray``'s ``.data`` is an ``xarray_jax.JaxArrayWrapper`` around a jax array that
    has never been copied to host. ``Dataset.compute()``/``.load()`` does NOT do that copy
    - ``JaxArrayWrapper`` satisfies xarray's NEP-18 duck-array protocol (it implements
    ``__array_function__``/``__array_ufunc__``, ``xarray_jax.py:442-466``), so
    ``Variable.load()`` calls ``to_duck_array()``, sees an already-duck-typed array, and
    hands it back UNCHANGED (verified against the installed xarray: ``to_duck_array``
    special-cases ``is_chunked_array`` for dask, then returns any other duck array as-is).
    So the ``.compute()`` this replaces was a no-op for this data, and every frame stayed
    device-resident straight into ``to_netcdf``. ``.copy(deep=True)`` is the same story:
    it copies the wrapper, not the buffer behind it.

    What this is NOT. It does not prevent the OOM of acal array 1209. That 19.59 GiB
    request is made during the FIRST predictor execution, before any write - it was a BFC
    arena EXTENSION the driver refused under ``XLA_PYTHON_CLIENT_PREALLOCATE=false``, and
    ``jit_concatenate`` in the message names the first CONSUMER of the denoiser's output,
    not the allocating op (the largest concatenate on the walker/write path is 162 MB).
    The fix for that lives in ``slurm/aires_env.sh``; see acal/HANDOFF.md, 2026-09-17.
    What this function does fix is real and separate: device residency of what is written,
    and an explicit dtype at the netCDF boundary rather than whatever the roll ran in.
    """
    import jax
    from graphcast import xarray_jax

    def host(da: xr.DataArray) -> np.ndarray:
        arr = np.asarray(jax.device_get(xarray_jax.unwrap_data(da)))
        # Upcast sub-32-bit floats. The walker rolls under XRES_BF16=1, and netCDF4 has no
        # bfloat16 ("unsupported dtype for netCDF4 variable: bfloat16", raised on write) -
        # so ``write_state``'s "float32, always" has to be enforced here, not assumed.
        # bfloat16 is an ml_dtypes extension type whose numpy ``kind`` is 'V', not 'f',
        # so it is matched by name as well; float64 coords are left alone.
        if (arr.dtype.name == "bfloat16"
                or arr.dtype == np.float16
                or (arr.dtype.kind == "f" and arr.dtype.itemsize < 4)):
            arr = arr.astype(np.float32)
        return arr

    data_vars = {name: (da.dims, host(da), dict(da.attrs))
                for name, da in ds.data_vars.items()}
    coords = {name: (da.dims, host(da), dict(da.attrs))
             for name, da in ds.coords.items()}
    return xr.Dataset(data_vars, coords=coords, attrs=dict(ds.attrs))


def _write_nc(ds: xr.Dataset, path, *, compress: bool = True) -> Path:
    """Write ``ds`` atomically, leaving no partial file behind when the write fails.

    The temp name carries the pid because a shard pool writes into one shared tree. The
    ``try`` is what was missing: when a write died mid-``to_netcdf`` (acal array 1209),
    netCDF4 had already created the file, so the failure left a header-only
    ``<name>.nc.tmp.<pid>`` orphan - 25 of them, ~9.7 kB apiece (24 at exactly 9,786 B) -
    that nothing ever cleans up and no later run can use. ``BaseException``, not ``Exception``, because a walltime
    ``SIGTERM`` or a Ctrl-C mid-write leaves exactly the same litter.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = ({v: {"zlib": True, "complevel": 4} for v in ds.data_vars} if compress else None)
    tmp = path.with_suffix(f".nc.tmp.{os.getpid()}")
    try:
        ds.to_netcdf(tmp, encoding=enc)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write_state(ds: xr.Dataset, path, *, compress: bool = True) -> Path:
    """Atomically write a restart state.

    float32, always - including when the segment rolled in bf16: ``_host_materialize``
    upcasts every sub-32-bit float on the way out, which is what makes that claim true
    rather than merely intended (netCDF4 has no bfloat16 type at all). Narrower floats are
    not an option even where netCDF supports them: aires.md floats float16 for archived
    states to halve the ~0.70 GB footprint, but ``geopotential`` at 50 hPa is ~2.0e5
    m2 s-2, well past float16's 65504 ceiling, so the archive would silently store ``inf``
    for the top of every column. zlib on float32 is the safe halving (netCDF
    ``scale_factor``/``add_offset`` int16 packing would be the principled alternative).
    """
    check_state(ds, where=str(path))
    _memstats("write_state:before")
    ds = _host_materialize(ds)
    _memstats("write_state:after")
    return _write_nc(ds, path, compress=compress)


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

    Memory: each step's global frame is pulled to HOST (``_host_materialize``) and the
    ring keeps the last two of them, ~0.70 GB of host RAM; nothing else global survives a
    loop iteration. That bounds THIS function's footprint, not the device's - graphcast's
    rollout holds whatever device-side state it needs to produce the next step either way,
    so do not read the ring's 0.70 GB as a statement about GPU memory.
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
        # The one place a rollout frame leaves the device - and it has to be an explicit
        # host transfer: ``.compute()`` here was a no-op on a JaxArrayWrapper (see
        # ``_host_materialize``). It is NOT "the one global materialisation": graphcast's
        # rollout keeps its own device copies to feed the next step. What it buys is that
        # the ring and the diag crops below hold HOST arrays, so nothing retained here can
        # pin a global device buffer alive.
        g = _host_materialize(_tidy(_to_valid_time(g, init)))
        ring.append(g)
        if crop:
            # .sel with slices is BASIC indexing -> a view that pins the whole global
            # array alive. Deep-copy, or every step's global frame is retained and the
            # process grows to tens of GB - the exact failure the xres crop avoids.
            diag.append(g.sel(lat=C.LAT, lon=C.LON).copy(deep=True))
        done += 1
        if verbose:
            print(f"  {_stamp()}[walker{label}] step {done}/{n_steps}  valid "
                  f"{pd.DatetimeIndex(g['time'].values)[-1]}", flush=True)
        del chunk, g

    if done != n_steps:
        raise RuntimeError(f"segment produced {done}/{n_steps} steps")
    _memstats(f"roll_segment{label}:after-loop")

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
                step_h: int = STEP_H, save_diag: bool = True,
                state_path: Path | None = None, diag_path: Path | None = None) -> dict:
    """Roll one segment, cached on disk. Re-running a finished segment is a no-op.

    ``parent_state`` is the state this segment continues; ``None`` means step 0, the
    event's ERA5 init. A CLONE is expressed by pointing a new walker index at a parent's
    state file - the differing walker index gives it its own noise stream.

    ``state_path``/``diag_path`` override where the segment lands. They default to Gate
    3's flat tree; the AI+RES driver passes its own run tree (``aconfig.res_state_path``)
    because under resampling a walker slot is not a lineage, so writing a resampled
    population into the gate's paths would overwrite artifacts that are supposed to be
    free-running. The RNG is NOT affected: ``segment_key(walker, step)`` is what makes a
    clone diverge from its sibling, and it depends on the slot, never on the path.
    """
    A.ensure_dirs(event)
    sp = Path(state_path) if state_path else A.state_path(event, walker, step)
    dp = Path(diag_path) if diag_path else A.diag_path(event, walker, step)
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
                     valid_time=str(valid_time(nxt)),
                     # Provenance: job 1155 rolled 112 segments with the wrong checkpoint
                     # and nothing on disk said so. Now every state names its model.
                     params_file=A.walker_model_kwargs()["params_file"],
                     resolution=A.WALKER_RES)
    sp.parent.mkdir(parents=True, exist_ok=True)
    write_state(nxt, sp)
    print(f"  {_stamp()}[walker] wrote {sp} ({sp.stat().st_size/1e6:.0f} MB), "
          f"valid {valid_time(nxt)}")
    if cube is not None:
        cube.attrs.update(nxt.attrs)
        # The diag cube is the second device->netCDF write, and it was the one left
        # behind: ``.copy(deep=True)`` on the crop copies the JaxArrayWrapper, not the
        # buffer, so this reached netCDF4 exactly as device-resident as the state did.
        cube = _host_materialize(cube)
        _write_nc(cube, dp)
        print(f"  {_stamp()}[walker] wrote {dp} ({dp.stat().st_size/1e6:.0f} MB, "
              f"{cube.sizes['time']} steps)")
    return dict(state=sp, diag=dp if cube is not None else None, rolled=True)


# --------------------------------------------------------------------------- #
# Model loading - the ONE place a walker's checkpoint is chosen.
#
# There used to be two: this module's ``main`` and ``gate3.stage_walk`` each built their
# own ``load_gencast`` call. Fixing the checkpoint bug in one of them left the other
# loading the 1.0 degree Mini model, and job 1156 burned another 8-GPU allocation proving
# it. Any caller that needs a walker bundle goes through here.
# --------------------------------------------------------------------------- #
def load_bundle(n_members: int = 1, res: str | None = None) -> dict:
    """Load the walker's GenCast with the checkpoint the resolution registry names."""
    from gencast_s2s import model as M

    kw = A.walker_model_kwargs(res)
    print(f"  [walker] loading {kw['params_file']!r} ({kw['res']} deg)", flush=True)
    return M.load_gencast(n_members=n_members, **kw)


def lazy_bundle(n_members: int = 1, res: str | None = None):
    """A ``get_bundle()`` closure that loads on first use and caches thereafter.

    Lazy on purpose: a worker whose segments are all cached must not spend ~15 min on the
    load plus the XLA compile just to discover it has nothing to do.
    """
    box: dict = {}

    def get_bundle() -> dict:
        if not box:
            import time as _t
            t0 = _t.perf_counter()
            box["b"] = load_bundle(n_members, res)
            print(f"  [walker] model loaded in {_t.perf_counter() - t0:.0f} s", flush=True)
        return box["b"]

    return get_bundle


def _task_config_only(res: str = A.WALKER_RES):
    """Just the checkpoint's task config - enough for --check, and no GPU/model build."""
    from graphcast import checkpoint, gencast
    from gencast_s2s import config as C
    from gencast_s2s.model import _open_local_or_gcs

    cfg = C.model_cfg("gencast", **A.walker_model_kwargs(res))
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

    get_bundle = lazy_bundle()

    run_segment(get_bundle, a.event, a.walker, a.step, a.steps,
                parent_state=a.parent, base_seed=a.base_seed, save_diag=not a.no_diag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
