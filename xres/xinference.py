"""Inference: roll GenCast out at one resolution, CACHE THE ENTIRE output, derive metrics.

For each event we run the diffusion ensemble to the 2-week horizon and, unlike the original
pipeline (which kept only the verification-week T2m field), we stream EVERY target variable
for EVERY rollout step and EVERY member, cropped to the CONUS box, into a compressed cube:

    runs/xres/<res>/week2/cache/<event>_cube.nc
        dims (member, time, level, lat, lon); ``time`` = absolute valid datetime.

From that cube we derive the event's headline verification metric (and its free secondary)
per member:

    runs/xres/<res>/week2/verif/<event>_<metric>_members.nc   dims (member, lat, lon)

The cube is the durable artifact: any other metric (MSLP, 10 m wind, ...) can be recomputed
offline from it without re-running the model.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import xarray as xr
import jax

from graphcast import data_utils, rollout

from gencast_s2s import config as C
from gencast_s2s import model as M
from gencast_s2s.inference import build_example_batch

from . import xconfig as X
from . import xmetrics as XM
from . import xdata as XD


# --------------------------------------------------------------------------- #
# Compressed atomic NetCDF write (cubes are large; zlib roughly halves them).
# --------------------------------------------------------------------------- #
def _atomic_to_netcdf(ds: xr.Dataset, path, compress: bool = False) -> None:
    import os
    path = str(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    enc = None
    if compress:
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)


def cube_path(res: str, weeks: int, name: str):
    return X.cache_dir(res, weeks) / f"{name}_cube.nc"


def verif_path(res: str, weeks: int, name: str, metric: str):
    return X.verif_dir(res, weeks) / f"{name}_{metric}_members.nc"


# --------------------------------------------------------------------------- #
# Roll out one event and cache the full cube + derived metrics.
# --------------------------------------------------------------------------- #
def run_event(bundle: dict, res: str, weeks: int, name: str, peak: str) -> None:
    step_h = C.model_cfg("gencast")["step_h"]
    lead_days = C.lead_days_for(weeks)
    init = pd.Timestamp(peak) - pd.Timedelta(days=lead_days)
    nm = bundle["n_members"]
    X.ensure_dirs(res, weeks)

    cube_f = cube_path(res, weeks, name)
    if cube_f.exists():
        try:
            have = int(xr.open_dataset(cube_f).sizes.get("member", 0))
        except Exception:
            have = -1
        if have == nm:
            print(f"  cached cube: {cube_f.name} ({have} members)")
            _derive_metrics(res, weeks, name, peak)        # ensure verif files exist
            return
        print(f"  stale cube ({have} members != {nm}) -> re-running")

    raw = XD.load_or_build_inputs(res, weeks, name, peak)
    full, init = build_example_batch(raw, peak, lead_days, step_h)
    task = bundle["task_config"]
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        full, target_lead_times=slice(f"{step_h}h", f"{lead_days * 24}h"),
        **dataclasses.asdict(task))

    rng = jax.random.PRNGKey(0)
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(nm)], axis=0)
    gen = rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=bundle["forward"], rngs=rngs,
        inputs=eval_inputs, targets_template=eval_targets * np.nan,
        forcings=eval_forcings, num_steps_per_chunk=1,
        num_samples=nm, pmap_devices=bundle["devices"])

    kept = []   # every step, ALL variables, cropped to CONUS before leaving the device
    for chunk in gen:
        sub = chunk.sel(lat=C.LAT, lon=C.LON).compute()
        kept.append(sub)
        del chunk
    cube = xr.concat(kept, dim="time", coords="minimal", compat="override")

    # member axis + drop the singleton batch
    if "sample" in cube.dims:
        cube = cube.rename({"sample": "member"})
    if "batch" in cube.dims:
        cube = cube.isel(batch=0, drop=True)
    if "member" not in cube.dims:
        cube = cube.expand_dims(member=[0])

    # absolute valid-datetime time axis (init + lead); keep lead hours for reference
    lead_td = cube["time"].values.astype("timedelta64[ns]")
    valid = (np.datetime64(init) + lead_td)
    cube = cube.assign_coords(time=("time", valid),
                              lead_h=("time", (lead_td / np.timedelta64(1, "h")).astype("int32")))
    cube = cube.transpose("member", "time", ...)
    cube.attrs.update(event=name, peak=str(pd.Timestamp(peak)), init=str(init),
                      resolution=res, weeks=int(weeks))

    _atomic_to_netcdf(cube, cube_f, compress=True)
    print(f"  saved cube {cube_f.name}  members={cube.sizes['member']} "
          f"steps={cube.sizes['time']} vars={len(cube.data_vars)} "
          f"size={cube_f.stat().st_size / 1e6:.0f}MB")

    _derive_metrics(res, weeks, name, peak, cube=cube)


def _derive_metrics(res: str, weeks: int, name: str, peak: str,
                    cube: xr.Dataset | None = None) -> None:
    """Compute + cache the per-member headline metric (and its secondary) from the cube."""
    for metric in XD.metrics_for_event(name):
        out_f = verif_path(res, weeks, name, metric)
        if out_f.exists() and cube is None:
            continue
        if cube is None:
            cube = xr.open_dataset(cube_path(res, weeks, name)).load()
        field = XM.forecast_field(cube, peak, metric)        # (member, lat, lon)
        field = field.transpose("member", "lat", "lon")
        _atomic_to_netcdf(field.to_dataset(name=metric), out_f)
        print(f"    verif [{metric}]: members={field.sizes['member']} "
              f"mean={float(field.mean()):+.3g} -> {out_f.name}")


def run_all(res: str, weeks: int, n_members: int | None = None) -> None:
    rs = X.res_spec(res)
    requested = X.N_MEMBERS[res] if n_members is None else n_members
    bundle = M.load_gencast(n_members=requested, params_file=rs["params"], res=rs["res"])
    print(f"=== xres {res} ({rs['label']}): week{weeks}, lead {C.lead_days_for(weeks)}d, "
          f"{C.n_rollout_steps(weeks)} steps, {bundle['n_members']} members ===")
    for name, (peak, _metric) in X.events().items():
        print(f"{name} [{res} week{weeks}] init=peak-{C.lead_days_for(weeks)}d ...", flush=True)
        run_event(bundle, res, weeks, name, peak)
