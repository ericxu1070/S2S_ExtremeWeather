"""June-July 2026 NE-US heat-wave: GenCast 1.0deg forecast for the Vayuh comparison.

This is a *one-off analysis* driver, separate from the horizon experiment (weeks 2/3/4)
and from xres. It exists to drop a GenCast column into the Vayuh.ai S2S report
(``Vayuh_S2S_Forecast_Analysis_July_Heatwave2026.pdf``), whose figures compare per-target-day
2 m-temperature anomaly over CONUS at subseasonal leads (14-18 d) against Vayuh + CFSv2.

Unlike ``inference.run_event`` (which keeps only the verification-week *mean*), here we keep
the *daily-mean* 2 m-temperature of every forecast day so we can reconstruct any
(target date, lead) pair the report needs:

    forecast at (target T, lead L)  ==  rollout from init (T - L days), forecast day L.

Grid / conventions are matched to the Vayuh product (``us_prob_forecast_2026.h5``):
1 deg CONUS, tmp2m anomaly vs the 1990-2019 ERA5 climatology (the same climatology the rest
of this repo uses). ERA5 for mid-2026 comes from ARCO (past the WeatherBench-2 cutoff).

Stages
------
    prep   (login node, internet)  init frames for each init date + observed daily truth
    infer  (GPU, offline)          per-init daily-mean tmp2m ensemble cube

Then ``heatwave2026_plot`` builds the Observed | Vayuh | GenCast figures.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from . import data as D

# --------------------------------------------------------------------------- #
# Experiment definition (matches the Vayuh report window).
# --------------------------------------------------------------------------- #
# Target dates the East-US time series spans, and the leads shown in the report.
TARGET_START = pd.Timestamp("2026-06-15")
TARGET_END   = pd.Timestamp("2026-07-02")
PEAK_TARGETS = pd.to_datetime(["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02"])
LEADS        = [14, 15, 16, 17, 18]

# Inits: one rollout per calendar day, 1-18 Jun 2026. This fully covers the peak-map /
# per-lead-grid pairs (targets 29 Jun-2 Jul, leads 14-18 -> inits 11-18 Jun) and gives a
# complete lead-14 East-US series (targets 15 Jun-2 Jul); longer leads cover the sub-range
# their inits reach. A rollout at (target, lead) is: init = target - lead, forecast day = lead.
INIT_START = pd.Timestamp("2026-06-01")
INIT_END   = (TARGET_END - pd.Timedelta(days=min(LEADS)))       # 2026-06-18
ROLLOUT_DAYS = max(LEADS)                                        # 18 d -> 36 steps @ 12 h

STEP_H = 12
MODEL = "gencast"

# Ensemble size (multiple of the device count is enforced by model.load_gencast).
N_MEMBERS = int(os.environ.get("HW2026_N_MEMBERS", 16))


def _inits() -> pd.DatetimeIndex:
    inits = pd.date_range(INIT_START, INIT_END, freq="D")
    sel = os.environ.get("HW2026_INITS")           # e.g. "2026-06-11,2026-06-12" for smoke
    if sel:
        want = pd.to_datetime([s.strip() for s in sel.split(",") if s.strip()])
        inits = inits[inits.isin(want)]
    maxn = os.environ.get("HW2026_MAX_INITS")
    if maxn:
        inits = inits[: int(maxn)]
    return inits


def _target_dates() -> pd.DatetimeIndex:
    return pd.date_range(TARGET_START, TARGET_END, freq="D")


# --------------------------------------------------------------------------- #
# Output layout: runs/heatwave2026/
# --------------------------------------------------------------------------- #
def root() -> "os.PathLike":
    return C.RUNS / "heatwave2026"


def inputs_dir():      return root() / "inputs"
def forecasts_dir():   return root() / "forecasts"
def obs_dir():         return root() / "observations"
def figures_dir():     return root() / "figures"


def ensure_dirs() -> None:
    C.ensure_dirs()
    for d in (root(), inputs_dir(), forecasts_dir(), obs_dir(), figures_dir()):
        d.mkdir(parents=True, exist_ok=True)


def _init_input_file(init: pd.Timestamp):
    return inputs_dir() / f"init_{init:%Y%m%d}_inputs.nc"


def _init_forecast_file(init: pd.Timestamp):
    return forecasts_dir() / f"init_{init:%Y%m%d}_daily_t2m_members.nc"


# --------------------------------------------------------------------------- #
# PREP (login node)
# --------------------------------------------------------------------------- #
def prepare_inputs() -> None:
    """Two real init frames per init date, coarsened to 1 deg -> runs/heatwave2026/inputs/."""
    ensure_dirs()
    statics = D.load_statics()
    for init in _inits():
        f = _init_input_file(init)
        if f.exists():
            print(f"[inputs] cached: {f.name}")
            continue
        # build_raw_inputs(peak, lead_days) uses init = peak - lead_days; lead_days=0 -> init.
        ds = D.build_raw_inputs(init, lead_days=0, model=MODEL, statics=statics,
                                verbose=True, res=1.0)
        D._atomic_to_netcdf(ds, f)
        print(f"[inputs] {init:%Y-%m-%d} -> {f.name}")


def _daily_mean_obs_t2m(date: pd.Timestamp):
    """ERA5 daily-mean 2 m-temperature (K) over CONUS on the 1 deg grid, from ARCO."""
    frames_t = pd.to_datetime([date + pd.Timedelta(hours=h) for h in range(0, 24, STEP_H)])
    lat_c, lon_c = D._arco_lat_lon_coords(conus=True, stride=4)
    acc = None
    for t in frames_t:
        arr = D.arco_read("2m_temperature", t, conus=True, stride=4)
        acc = arr.astype("float64") if acc is None else acc + arr
        D.release_stores(); gc.collect()
    daily = acc / len(frames_t)
    return xr.DataArray(daily, dims=("lat", "lon"),
                        coords={"lat": lat_c, "lon": lon_c}), frames_t


def prepare_truth() -> None:
    """Observed daily-mean tmp2m + anomaly + climatology for the target window.

    Saved as (date, lat, lon) cubes under runs/heatwave2026/observations/. Anomaly uses the
    shared 1990-2019 ERA5 climatology (D.clim_for), matching the rest of the repo.
    """
    ensure_dirs()
    f_abs = obs_dir() / "obs_daily_t2m.nc"
    f_anom = obs_dir() / "obs_daily_t2m_anom.nc"
    f_clim = obs_dir() / "clim_daily_t2m.nc"
    if f_abs.exists() and f_anom.exists() and f_clim.exists():
        print("[truth] cached: obs_daily_t2m{,_anom}.nc, clim_daily_t2m.nc")
        return

    dates = _target_dates()
    abs_list, anom_list, clim_list = [], [], []
    for d in dates:
        daily, frames_t = _daily_mean_obs_t2m(d)
        clim = D.clim_for(frames_t).mean("time")                    # (lat, lon), 0.25 deg
        clim = clim.sel(lat=daily.lat, lon=daily.lon, method="nearest") \
                   .assign_coords(lat=daily.lat, lon=daily.lon)
        anom = daily - clim
        abs_list.append(daily); anom_list.append(anom); clim_list.append(clim)
        print(f"[truth] {d:%Y-%m-%d}: obs-mean {float(daily.mean())-273.15:+.1f}C "
              f"anom {float(anom.mean()):+.2f}K max {float(anom.max()):+.2f}K")
        D.release_stores(); gc.collect()

    coord = {"date": dates}
    xr.concat(abs_list, dim="date").assign_coords(coord).to_dataset(name="t2m") \
        .pipe(lambda ds: D._atomic_to_netcdf(ds, f_abs))
    xr.concat(anom_list, dim="date").assign_coords(coord).to_dataset(name="t2m_anom") \
        .pipe(lambda ds: D._atomic_to_netcdf(ds, f_anom))
    xr.concat(clim_list, dim="date").assign_coords(coord).to_dataset(name="t2m_clim") \
        .pipe(lambda ds: D._atomic_to_netcdf(ds, f_clim))
    print(f"[truth] wrote {len(dates)} daily obs/anom/clim -> {obs_dir()}")


def prepare_all() -> None:
    from . import download
    download.download_model()          # checkpoint + stats + statics + clim cache
    prepare_inputs()
    prepare_truth()


# --------------------------------------------------------------------------- #
# INFER (GPU)
# --------------------------------------------------------------------------- #
def _rollout_daily_t2m(bundle: dict, raw: xr.Dataset, init: pd.Timestamp) -> xr.DataArray:
    """Roll the ensemble ROLLOUT_DAYS out and return daily-mean tmp2m (member, lead_day, lat, lon)."""
    import dataclasses
    from graphcast import data_utils, rollout
    from .inference import build_example_batch

    # Roll to ROLLOUT_DAYS days + one extra step so the last forecast day gets both its
    # 00Z and 12Z frames (a full daily mean), not just 00Z. build_example_batch derives its
    # init as peak - lead_days, so lead_days must carry the same fractional (half-day) offset.
    horizon_h = ROLLOUT_DAYS * 24 + STEP_H
    peak = init + pd.Timedelta(hours=horizon_h)
    lead_days_arg = horizon_h / 24.0
    full, init_chk = build_example_batch(raw, peak, lead_days_arg, STEP_H)
    assert pd.Timestamp(init_chk) == init, (init_chk, init)
    task = bundle["task_config"]

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        full, target_lead_times=slice(f"{STEP_H}h", f"{horizon_h}h"),
        **dataclasses.asdict(task))

    nm = bundle["n_members"]
    rng = __import__("jax").random.PRNGKey(0)
    import jax
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(nm)], axis=0)
    gen = rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=bundle["forward"], rngs=rngs,
        inputs=eval_inputs, targets_template=eval_targets * np.nan,
        forcings=eval_forcings, num_steps_per_chunk=1,
        num_samples=nm, pmap_devices=bundle["devices"])

    kept = []
    for chunk in gen:
        # crop to CONUS before host transfer (keeping the global field per step OOMs)
        kept.append(chunk["2m_temperature"].sel(lat=C.LAT, lon=C.LON).compute())
        del chunk
    pred = xr.concat(kept, dim="time")                        # (.., time, lat, lon)
    if "sample" in pred.dims:
        pred = pred.rename({"sample": "member"})
    if "member" not in pred.dims:
        pred = pred.expand_dims(member=[0])
    if "batch" in pred.dims:
        pred = pred.isel(batch=0, drop=True)

    # Aggregate the 12-hourly frames into daily means keyed by lead-day (1..ROLLOUT_DAYS).
    valid = pd.DatetimeIndex(np.datetime64(init) + pred["time"].values)
    lead_day = ((valid.normalize() - init.normalize()).days).astype(int)
    pred = pred.assign_coords(lead_day=("time", lead_day))
    daily = pred.groupby("lead_day").mean("time")             # (member, lead_day, lat, lon)
    daily = daily.sel(lead_day=[d for d in daily.lead_day.values if d >= 1])
    target_dates = [init + pd.Timedelta(days=int(d)) for d in daily.lead_day.values]
    daily = daily.assign_coords(target_date=("lead_day", np.array(target_dates, dtype="datetime64[ns]")))
    return daily.transpose("member", "lead_day", "lat", "lon")


def run_infer(n_members: int | None = None) -> None:
    from . import model as M
    ensure_dirs()
    inits = _inits()
    want_nm = M.n_members_for_devices(
        N_MEMBERS if n_members is None else n_members,
        len(__import__("jax").local_devices()))

    # Skip loading the model entirely if every init is already cached at the right size.
    todo = []
    for init in inits:
        f = _init_forecast_file(init)
        if f.exists():
            try:
                have = int(xr.open_dataset(f).sizes.get("member", 1))
            except Exception:
                have = None
            if have == want_nm:
                print(f"  cached: {f.name} ({have} members)")
                continue
            print(f"  stale cache ({have} != {want_nm} members) -> re-running {f.name}")
        todo.append(init)
    if not todo:
        print("[infer] all inits cached; nothing to do")
        return

    bundle = M.load_gencast(n_members=N_MEMBERS if n_members is None else n_members)
    print(f"=== heatwave2026: {len(todo)} init(s), {ROLLOUT_DAYS}d rollout "
          f"({ROLLOUT_DAYS * 24 // STEP_H} steps), {bundle['n_members']} members ===")
    for init in todo:
        f = _init_forecast_file(init)
        print(f"init {init:%Y-%m-%d} ...")
        raw = xr.open_dataset(_init_input_file(init)).load()
        daily = _rollout_daily_t2m(bundle, raw, init)
        D._atomic_to_netcdf(daily.to_dataset(name="t2m"), f)
        print(f"  saved {f.name}  members={daily.sizes['member']} "
              f"lead_days={daily.lead_day.values.min()}..{daily.lead_day.values.max()}")
