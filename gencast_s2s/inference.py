"""Inference: roll GenCast out to the horizon, keep only the verification week, save.

For each (event, horizon) we build the example batch (2 real init frames + NaN target
placeholders out to the horizon), run the diffusion rollout one step per chunk, and keep
only the verification-week 2 m-temperature frames (lead days ``lead-6 .. lead``, i.e. the
7 days ending on the peak). The full trajectory is streamed and discarded. The saved
product is the per-member week-mean T2m anomaly over CONUS, ``(member, lat, lon)``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import xarray as xr
import jax

from graphcast import data_utils, rollout

from . import config as C
from . import data as D


def build_example_batch(raw: xr.Dataset, peak, lead_days: int, step_h: int):
    """Two real input frames + NaN target template out to ``lead_days``."""
    init = pd.Timestamp(peak) - pd.Timedelta(days=lead_days)
    in_dt = pd.DatetimeIndex(raw["time"].values)                  # absolute frame times
    future = pd.date_range(init + pd.Timedelta(hours=step_h), peak, freq=f"{step_h}h")
    all_dt = pd.DatetimeIndex(in_dt.append(future))
    all_td = (all_dt.values - np.datetime64(init)).astype("timedelta64[ns]")  # rel. to last input

    # NaN target template: repeat the last input frame, blank the time-varying vars.
    tmpl = raw.isel(time=[raw.sizes["time"] - 1] * len(future))
    for v in tmpl.data_vars:
        if "time" in tmpl[v].dims:
            tmpl[v] = tmpl[v] * np.nan

    # data_vars/coords="minimal": do NOT broadcast time-independent statics onto the time
    # axis — otherwise each gains a spurious time dim and an extra input channel.
    full = xr.concat([raw, tmpl], dim="time",
                     data_vars="minimal", coords="minimal", compat="override")
    full = full.assign_coords(time=("time", all_td),
                              datetime=(("batch", "time"), all_dt.values[None, :]))

    # Statics MUST stay 2-D (lat, lon). If the concat leaked a time axis onto them, each
    # static becomes 2 input channels instead of 1 and the encoder sees 269 features when
    # GenCast expects 267. Strip any stray time axis to guarantee one channel each.
    for k in ("geopotential_at_surface", "land_sea_mask"):
        if k in full.data_vars and "time" in full[k].dims:
            full[k] = full[k].isel(time=0, drop=True)
    return full, init


def to_verif_anom(t2m: xr.DataArray, init) -> xr.DataArray:
    """(.., time, lat, lon) week-mean T2m -> anomaly over CONUS using 1990-2019 clim."""
    valid = pd.DatetimeIndex(np.datetime64(init) + t2m["time"].values)
    clim_mean = D.clim_for(valid).mean("time")
    anom = t2m.mean("time") - clim_mean
    return anom.sel(lat=C.LAT, lon=C.LON)


def _load_or_build_inputs(name, peak, weeks, model="gencast", statics=None) -> xr.Dataset:
    f = C.inputs_dir(weeks) / f"{name}_{model}_inputs.nc"
    if f.exists():
        return xr.open_dataset(f).load()
    C.ensure_dirs(weeks)
    ds = D.build_raw_inputs(peak, lead_days=C.lead_days_for(weeks), model=model,
                            statics=statics, verbose=True)
    D._atomic_to_netcdf(ds, f)
    return ds


def run_event(bundle: dict, name: str, peak: str, weeks: int, model: str = "gencast"):
    cfg = C.model_cfg(model)
    step_h = cfg["step_h"]
    lead_days = C.lead_days_for(weeks)
    C.ensure_dirs(weeks)
    out_f = C.forecasts_dir(weeks) / f"{name}_{model}_verif_t2m_anom_members.nc"

    # Member-aware cache: a file from a different ensemble size must not be reused.
    if out_f.exists():
        want = bundle["n_members"]
        try:
            have = int(xr.open_dataset(out_f).sizes.get("member", 1))
        except Exception:
            have = None
        if have == want:
            print(f"  cached: {out_f.name} ({have} members)")
            return out_f
        print(f"  stale cache ({have} members != {want}) -> re-running")

    raw = _load_or_build_inputs(name, peak, weeks, model=model)
    full, init = build_example_batch(raw, peak, lead_days, step_h)
    task = bundle["task_config"]

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        full, target_lead_times=slice(f"{step_h}h", f"{lead_days * 24}h"),
        **dataclasses.asdict(task))

    # Cheap pre-flight (no GPU): count grid-node input channels. GenCast expects 267;
    # +2 almost always means a static var leaked a `time` axis.
    def _nch(d):
        return int(sum(np.prod([d[v].sizes[x] for x in d[v].dims if x not in ("lat", "lon")] or [1])
                       for v in d.data_vars))
    print(f"  grid-node input channels = {_nch(eval_inputs) + _nch(eval_targets) + _nch(eval_forcings)} "
          f"(inputs {_nch(eval_inputs)} + targets {_nch(eval_targets)} + forcings {_nch(eval_forcings)})")

    verif_start = np.timedelta64(lead_days - 6, "D")    # lead day (lead-6); start of verification week
    nm = bundle["n_members"]
    rng = jax.random.PRNGKey(0)
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(nm)], axis=0)
    gen = rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=bundle["forward"], rngs=rngs,
        inputs=eval_inputs, targets_template=eval_targets * np.nan,
        forcings=eval_forcings, num_steps_per_chunk=1,
        num_samples=nm, pmap_devices=bundle["devices"])

    kept = []   # verification-week 2m_temperature frames only, already CONUS-cropped
    for chunk in gen:
        t2m = chunk["2m_temperature"]
        mask = chunk["time"].values >= verif_start
        if mask.any():
            # Crop to CONUS *before* pulling to host RAM: keeping the global field for every
            # step x member balloons host memory ~40x and can OOM the process.
            kept.append(t2m.isel(time=np.where(mask)[0]).sel(lat=C.LAT, lon=C.LON).compute())
        del chunk

    pred = xr.concat(kept, dim="time")
    if "sample" in pred.dims:
        pred = pred.rename({"sample": "member"})
    if "member" not in pred.dims:
        pred = pred.expand_dims(member=[0])
    anom = to_verif_anom(pred, init)
    if "batch" in anom.dims:
        anom = anom.isel(batch=0, drop=True)
    anom = anom.transpose("member", "lat", "lon")
    D._atomic_to_netcdf(anom.to_dataset(name="t2m_anom"), out_f)
    print(f"  saved {out_f.name}  members={anom.sizes['member']} "
          f"spread={float(anom.std('member').mean()):.3f}K")
    return out_f


def run_all(weeks: int, n_members: int | None = None, model: str = "gencast"):
    from . import model as M
    bundle = M.load_gencast(n_members=n_members)
    lead = C.lead_days_for(weeks)
    print(f"=== week{weeks}: lead {lead}d, {C.n_rollout_steps(weeks)} steps, "
          f"{bundle['n_members']} members ===")
    for name, peak in C.EVENTS.items():
        print(f"{name} [week{weeks}] init=peak-{lead}d ...")
        run_event(bundle, name, peak, weeks, model=model)
