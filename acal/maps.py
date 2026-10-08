#!/usr/bin/env python
"""Per-gridpoint skill maps of the calibration campaign at +/-2, +/-3, +/-4 K. CPU ONLY.

`analyze.py` scores one number per case, the CONUS-mean A_L. This module asks the same
question pixel by pixel: at each 0.25 deg cell, did the AI+RES ensemble say the 7-day-mean
T2m anomaly ending at the peak would cross a threshold, and did ERA5 cross it?

The forecast at a cell
----------------------
Every case's 32 final walkers each carry a GenCast trajectory (`walkers/wNN/stepSS/
diag.nc`, CONUS crop, read through the genealogy in `compare.json["realized"]["lineage"]`).
Each is reduced with `aires.aindex.field`, the code path that produced A_L, so the CONUS
mean of a walker's field reproduces its `realized["conus"]` (asserted). The probability is
the self-normalized importance-weighted fraction

    P(x, a) = sum_i w_i 1[s_a A_i(x) >= s_a a] / sum_i w_i,     w_i = exp(-V_K,i)

with `s_a = sign(a)`, so a cold panel counts cells at or below -2 K. Importance weights
are valid for any function of the path, so the weights tuned to the CONUS mean apply at a
cell too; they are noisier there because the tilt was not aimed at it.

What is scored
--------------
Family-matched: cold thresholds use the 11 cold cases, hot ones the 31 hot cases.

  * accuracy(x, a)  = mean over cases of 1[(P >= 0.5) == o], with o = 1[ERA5 crosses a].
    At rare thresholds this is mostly "said no, nothing happened", so each panel also
    reports the always-no baseline.
  * BSS(x, a) = 1 - sum (P - o)^2 / sum (p_clim - o)^2, summed over cases. p_clim(x, a)
    is the per-cell empirical frequency over the same pool `analyze.clim_pool` uses for
    the CONUS index: ERA5 days within +/-30 d of the peak's anniversary in 2021-2026,
    minus the case's own +/-10 d.

    python -m acal.maps --stage fields    # runs/acal/analysis/maps_fields.nc (~10 min)
    python -m acal.maps --stage figures   # figures/acal/acal_map_{accuracy,bss}.png
    python -m acal.maps --stage all

CFSv2 as a second forecast source (`acal/CFS_PLAN.md` step 5)
-------------------------------------------------------------
`--stage cfs` scores NCEP CFSv2 the same way, from the 16-member lagged cubes of
`acal.cfsbase`. The 7-day field per member is `aires.aindex.field` (its CONUS mean is
asserted against the cube's own `al`), the daily fields use `cfsbase.daily_pairs`. The
probability is the equal-weight member fraction with the same tail-signed `>=` rule,

    P(x, a) = mean_i 1[s_a A_i(x) >= s_a a]

in two variants: `prob_raw` (the headline, same treatment as the walkers: CFS drift is
inside the score) and `prob_corr`, with the leave-one-year-out CFS member-mean bias of
`runs/acal/cfs/bias.nc` subtracted per cell (the sensitivity that bounds how much of any
gap is drift). Truth, climatology and family do not depend on the forecast, so they are
copied from the AI+RES files (case order asserted) rather than recomputed.

Resolution caveat, printed on every CFS figure: CFS is ~0.94 deg (T126) bilinearly
regridded to 0.25 deg, so gridpoint skill is resolution-limited. For the CONUS index that
was measured negligible; for a cell it is not.

    python -m acal.maps --stage cfs   # maps_{fields,daily}_cfs.nc, *_cfs / *_cfscorr /
                                      # *_diff_cfs figures + land-mean summary (~5 min)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import cfsbase
from acal import aprep, ccfg
from aires import aconfig as A
from aires import aindex as AI

THRESHOLDS = (-2.0, -3.0, -4.0, 2.0, 3.0, 4.0)
FIELDS = AN.OUT / "maps_fields.nc"
DAILY_FIELDS = AN.OUT / "maps_daily.nc"
CFS_FIELDS = AN.OUT / "maps_fields_cfs.nc"
CFS_DAILY_FIELDS = AN.OUT / "maps_daily_cfs.nc"
CFS_NOTE = ("CFS is ~0.94 deg (T126) bilinearly regridded to 0.25 deg: gridpoint skill is "
            "resolution-limited (unlike the CONUS index, where this was measured negligible)")
INDEX_FILES = sorted((ccfg.ACAL_ROOT / "index").glob("era5_t2m_anom_12h_*.nc"))
WINDOW_FRAMES = 13                        # [peak-6d, peak] at 12 h, both ends inclusive
CONSISTENCY_TOL = 1e-3                    # K; field CONUS mean vs compare.json A_L
YES = 0.5                                 # P >= YES is a "yes" forecast for accuracy
LAND_FRAC = 0.5                           # ERA5 land_sea_mask >= this is scored; sea is
                                          # masked: a 7-day-mean T2m over water almost
                                          # never crosses 2 K, so accuracy and BSS there
                                          # read ~1 for free


# --------------------------------------------------------------------------- #
# Stage: fields
# --------------------------------------------------------------------------- #
def walker_traj(c: AN.Case, w: int, slots: dict, peak) -> xr.Dataset:
    """Walker `w`'s T2m path through the genealogy, from the first segment that reaches
    the verification week (12 h before it, so the first daily window is complete)."""
    paths = [A.res_diag_path(c.episode_id, int(slot), int(seg), AN.TAG)
             for seg, slot in sorted(slots.items(), key=lambda kv: int(kv[0]))]
    parts = []
    for p in paths:
        with xr.open_dataset(p) as d:
            d = d[["2m_temperature"]].drop_vars("lead_h", errors="ignore")
            t = pd.DatetimeIndex(d["time"].values)
            if t.max() >= peak - pd.Timedelta(days=6):
                parts.append(d.load())
    traj = xr.concat(parts, dim="time").sortby("time")
    AI.check_window(traj, peak, "t2m_anom", where=f"{c.episode_id} w{w:02d}")
    return traj


def _check_conus(c: AN.Case, w: int, f: xr.DataArray, conus: np.ndarray) -> None:
    got = float(AI.area_mean(f))
    if abs(got - conus[w]) > CONSISTENCY_TOL:
        raise SystemExit(f"[maps] {c.episode_id} w{w:02d}: field CONUS mean {got:+.4f} != "
                         f"compare.json {conus[w]:+.4f}")


def walker_fields(c: AN.Case) -> xr.DataArray:
    """(walker, lat, lon) 7-day-mean T2m anomaly of every final walker of one case."""
    peak = pd.Timestamp(c.run["peak"])
    conus = np.asarray(c.cmp["realized"]["conus"], dtype="float64")
    out = []
    for w, slots in enumerate(c.cmp["realized"]["lineage"]):
        f = AI._squeeze(AI.field(walker_traj(c, w, slots, peak), peak, "t2m_anom"))
        _check_conus(c, w, f, conus)
        out.append(f.astype("float32"))
    return xr.concat(out, dim="walker")


def day_ends(peak) -> pd.DatetimeIndex:
    """The 7 daily windows of the verification week, by their 00Z end times."""
    return pd.date_range(pd.Timestamp(peak) - pd.Timedelta(days=6), peak, freq="D")


def walker_daily(c: AN.Case) -> xr.DataArray:
    """(walker, day, lat, lon) daily-mean T2m anomaly: each day is the mean of the two
    12 h frames ending at a 00Z in [peak-6d, peak], the same pairing as `era5_daily1`."""
    peak = pd.Timestamp(c.run["peak"])
    conus = np.asarray(c.cmp["realized"]["conus"], dtype="float64")
    ends = day_ends(peak)
    out = []
    for w, slots in enumerate(c.cmp["realized"]["lineage"]):
        traj = walker_traj(c, w, slots, peak)
        inst = AI._squeeze(AI.instantaneous_field(traj, "t2m_anom"))
        # The 13-frame mean of the same frames must still be A_L.
        _check_conus(c, w, inst.sel(time=slice(ends[0], ends[-1])).mean("time"), conus)
        d = [inst.sel(time=[e - pd.Timedelta(hours=12), e]).mean("time") for e in ends]
        out.append(xr.concat(d, dim=pd.Index(ends, name="day")).astype("float32"))
    return xr.concat(out, dim="walker")


def era5_index() -> xr.DataArray:
    """The 12-hourly ERA5 anomaly series, de-duplicated across the yearly files."""
    da = xr.concat([xr.open_dataset(p)["t2m_anom"] for p in INDEX_FILES], dim="time")
    t = pd.DatetimeIndex(da["time"].values)
    return da.isel(time=~t.duplicated(keep="first")).sortby("time").load()


def era5_daily(idx: xr.DataArray) -> xr.DataArray:
    """7-day-mean anomaly ending at each 00Z, the 13-frame A_L definition, per cell.

    Frames must be contiguous at 12 h: a gap would make a 13-frame rolling mean span
    more than 6 days, so a 00Z end-time is kept only if its window is exactly 6 days.
    """
    r = idx.rolling(time=WINDOW_FRAMES).mean()
    t = pd.DatetimeIndex(idx["time"].values)
    span = pd.Series(t).diff(WINDOW_FRAMES - 1).values
    ok = (t.hour == 0) & (span == np.timedelta64(6, "D"))
    d = r.isel(time=np.where(ok)[0])
    return d.assign_coords(time=pd.DatetimeIndex(d["time"].values).normalize())


def era5_daily1(idx: xr.DataArray) -> xr.DataArray:
    """Daily-mean anomaly ending at each 00Z: the 00Z frame and the 12Z frame before it."""
    r = idx.rolling(time=2).mean()
    t = pd.DatetimeIndex(idx["time"].values)
    span = pd.Series(t).diff(1).values
    ok = (t.hour == 0) & (span == np.timedelta64(12, "h"))
    d = r.isel(time=np.where(ok)[0])
    return d.assign_coords(time=pd.DatetimeIndex(d["time"].values).normalize())


def era5_daily_utc(idx: xr.DataArray) -> xr.DataArray:
    """Daily anomaly of each UTC day D, stamped D: the 00Z D and 12Z D frames, the pair
    a daily-mean source's day D is verified on (`s2sbase.DAILY_PAIR`, ruling C2). Kept
    only where the two frames are 12 h apart."""
    r = idx.rolling(time=2).mean()
    t = pd.DatetimeIndex(idx["time"].values)
    span = pd.Series(t).diff(1).values
    ok = (t.hour == 12) & (span == np.timedelta64(12, "h"))
    d = r.isel(time=np.where(ok)[0])
    return d.assign_coords(time=pd.DatetimeIndex(d["time"].values).normalize())


def utc_daily_truth(truth, base: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """(truth (case, day=7, lat, lon), clim (case, threshold, lat, lon)) of a daily-mean
    source's daily maps against `truth` (ruling C2). Maps day k (1..6) is UTC day
    D = peak-7+k on its (00Z D, 12Z D) pair (`era5_daily_utc` of the truth's per-cell
    index), the day the source's daily mean covers; day 0 has no UTC day in the window
    and stays NaN. clim is the frequency over the same pairs on the `analyze.clim_pool`
    days. The six days average to the truth's '12f' field (the 7-day truth of the same
    source), asserted against `truth.obs_al(eid, '12f')`. Cells outside the truth's mask
    are NaN, as in `truth_dataset`."""
    from acal import s2sbase as S2
    from acal import truth as TR
    d = era5_daily_utc(TR.index_of(truth))
    days = pd.Series(0.0, index=pd.DatetimeIndex(d["time"].values))
    for k in ("lat", "lon"):
        if not np.allclose(d[k].values, base[k].values, atol=1e-6):
            raise SystemExit(f"[maps] {truth.name} index {k} != maps grid")
    mask = truth.mask()
    valid = np.ones((base.sizes["lat"], base.sizes["lon"]), bool) if mask is None else \
        np.asarray(mask.transpose("lat", "lon").values, bool)
    peaks = {e: pd.Timestamp(p) for e, p in
             zip(aprep.episodes().episode_id, aprep.episodes().peak)}
    truth_v = np.full(base["truth"].shape, np.nan, "float32")
    clim = np.full(base["clim"].shape, np.nan, "float32")
    for i, eid in enumerate(map(str, base["case"].values)):
        peak = peaks[eid]
        obs = d.sel(time=S2.daily_days(peak))                    # (time=6, lat, lon)
        got = float(TR.area_mean(obs.mean("time"), mask))
        want = truth.obs_al(eid, "12f")
        if abs(got - want) > CONSISTENCY_TOL:
            raise SystemExit(f"[maps] {eid}: {truth.name} UTC-day mean {got:+.4f} != "
                             f"12f obs_al {want:+.4f}")
        truth_v[i, 1:] = np.where(valid, obs.values, np.nan)
        pool = d.sel(time=AN.clim_pool(days, peak).index).values
        for j, a in enumerate(base["threshold"].values):
            sg = np.sign(a)
            clim[i, j] = np.where(valid, (sg * pool >= sg * a).mean(0), np.nan)
    return truth_v, clim


def daily() -> Path:
    """Daily truth, forecast probabilities and per-cell daily climatology."""
    df, cases = AN.load_all()
    d1 = era5_daily1(era5_index())
    days = pd.Series(0.0, index=pd.DatetimeIndex(d1["time"].values))
    lat, lon = d1["lat"], d1["lon"]
    nd = 7
    prob = np.full((len(cases), nd, len(THRESHOLDS), lat.size, lon.size), np.nan, "float32")
    clim = np.full((len(cases), len(THRESHOLDS), lat.size, lon.size), np.nan, "float32")
    truth = np.full((len(cases), nd, lat.size, lon.size), np.nan, "float32")
    for i, c in enumerate(cases):
        peak = pd.Timestamp(c.run["peak"])
        truth[i] = d1.sel(time=day_ends(peak)).values
        F = walker_daily(c)
        if not (np.array_equal(F["lat"], lat) and np.array_equal(F["lon"], lon)):
            raise SystemExit(f"[maps] {c.episode_id}: walker grid != ERA5 index grid")
        w = c.weights / c.weights.sum()
        pool = d1.sel(time=AN.clim_pool(days, peak).index).values
        for j, a in enumerate(THRESHOLDS):
            sg = np.sign(a)
            prob[i, :, j] = np.tensordot(w, sg * F.values >= sg * a, axes=(0, 0))
            clim[i, j] = (sg * pool >= sg * a).mean(0)
        print(f"  {c.episode_id}  pool n={pool.shape[0]}", flush=True)

    ds = xr.Dataset(
        dict(prob=(("case", "day", "threshold", "lat", "lon"), prob),
             clim=(("case", "threshold", "lat", "lon"), clim),
             truth=(("case", "day", "lat", "lon"), truth)),
        coords=dict(case=[c.episode_id for c in cases], day=np.arange(nd),
                    threshold=list(THRESHOLDS), lat=lat.values, lon=lon.values,
                    family=("case", df.family.values)),
        attrs=dict(note="as maps_fields.nc, but for each of the 7 daily means of the "
                        "verification week (day 6 ends at the peak)"))
    ds.to_netcdf(DAILY_FIELDS)
    print(f"[maps] wrote {DAILY_FIELDS}")
    return DAILY_FIELDS


def fields() -> Path:
    """Truth, forecast probabilities and per-cell climatology for every case and threshold."""
    df, cases = AN.load_all()
    print(f"[maps] ERA5 index: {len(INDEX_FILES)} files")
    daily = era5_daily(era5_index())
    days = pd.Series(0.0, index=pd.DatetimeIndex(daily["time"].values))
    lat, lon = daily["lat"], daily["lon"]

    shape = (len(cases), len(THRESHOLDS), lat.size, lon.size)
    prob, clim = np.full(shape, np.nan, "float32"), np.full(shape, np.nan, "float32")
    truth = np.full((len(cases), lat.size, lon.size), np.nan, "float32")
    for i, c in enumerate(cases):
        peak = pd.Timestamp(c.run["peak"])
        obs = daily.sel(time=peak)
        got = float(AI.area_mean(obs))
        if abs(got - c.obs) > CONSISTENCY_TOL:
            raise SystemExit(f"[maps] {c.episode_id}: ERA5 CONUS mean {got:+.4f} != slate {c.obs:+.4f}")
        truth[i] = obs.values

        F = walker_fields(c)
        if not (np.array_equal(F["lat"], lat) and np.array_equal(F["lon"], lon)):
            raise SystemExit(f"[maps] {c.episode_id}: walker grid != ERA5 index grid")
        w = c.weights / c.weights.sum()
        pool = daily.sel(time=AN.clim_pool(days, peak).index).values
        for j, a in enumerate(THRESHOLDS):
            s = np.sign(a)
            hit = (s * F.values >= s * a)
            prob[i, j] = np.tensordot(w, hit, axes=(0, 0))
            clim[i, j] = (s * pool >= s * a).mean(0)
        print(f"  {c.episode_id}  obs {c.obs:+.2f}  pool n={pool.shape[0]}")

    ds = xr.Dataset(
        dict(prob=(("case", "threshold", "lat", "lon"), prob),
             clim=(("case", "threshold", "lat", "lon"), clim),
             truth=(("case", "lat", "lon"), truth)),
        coords=dict(case=[c.episode_id for c in cases], threshold=list(THRESHOLDS),
                    lat=lat.values, lon=lon.values,
                    family=("case", df.family.values)),
        attrs=dict(note="prob = self-normalized AI+RES P(s A(x) >= s a); clim = per-cell "
                        "climatological frequency over analyze.clim_pool; truth = ERA5 "
                        "7-day-mean T2m anomaly ending at the peak"))
    AN.OUT.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(FIELDS)
    print(f"[maps] wrote {FIELDS}")
    return FIELDS


# --------------------------------------------------------------------------- #
# Stage: figures
# --------------------------------------------------------------------------- #
def truth_dataset(base: xr.Dataset, truth, daily: bool) -> xr.Dataset:
    """`base` (an AI+RES maps file) with `truth` and `clim` rebuilt from `truth`'s per-cell
    index, exactly as `fields` / `daily` build them from the ERA5 index (same 13-frame or
    00Z + preceding 12Z windows, same `analyze.clim_pool` days), plus `valid(lat, lon)` =
    the truth's mask, which `scores` ANDs with land. The forecast probabilities are per
    cell and do not depend on the truth, so `prob` is kept. ERA5 reproduces `base`."""
    from acal import truth as TR
    idx = TR.index_of(truth)
    d = era5_daily1(idx) if daily else era5_daily(idx)
    days = pd.Series(0.0, index=pd.DatetimeIndex(d["time"].values))
    for k in ("lat", "lon"):
        if not np.allclose(d[k].values, base[k].values, atol=1e-6):
            raise SystemExit(f"[maps] {truth.name} index {k} != maps grid")
    mask = truth.mask()
    valid = np.ones((base.sizes["lat"], base.sizes["lon"]), bool) if mask is None else \
        np.asarray(mask.transpose("lat", "lon").values, bool)
    peaks = {e: pd.Timestamp(p) for e, p in
             zip(aprep.episodes().episode_id, aprep.episodes().peak)}
    truth_v = np.full(base["truth"].shape, np.nan, "float32")
    clim = np.full(base["clim"].shape, np.nan, "float32")
    for i, eid in enumerate(map(str, base["case"].values)):
        peak = peaks[eid]
        obs = d.sel(time=day_ends(peak) if daily else peak)
        if not daily:
            got = float(TR.area_mean(obs, mask))
            want = truth.obs_al(eid, "13f")
            if abs(got - want) > CONSISTENCY_TOL:
                raise SystemExit(f"[maps] {eid}: {truth.name} field mean {got:+.4f} != "
                                 f"obs_al {want:+.4f}")
        truth_v[i] = np.where(valid, obs.values, np.nan)
        pool = d.sel(time=AN.clim_pool(days, peak).index).values
        for j, a in enumerate(base["threshold"].values):
            sg = np.sign(a)
            clim[i, j] = np.where(valid, (sg * pool >= sg * a).mean(0), np.nan)
    out = base.copy()
    out["truth"] = (base["truth"].dims, truth_v)
    out["clim"] = (base["clim"].dims, clim)
    if mask is not None:
        out["valid"] = (("lat", "lon"), valid.astype("int8"))
    out.attrs["truth"] = truth.name
    out.attrs["note"] = (base.attrs.get("note", "") + f"; truth and clim rebuilt from the "
                         f"{truth.name} index" + ("; valid = the truth's mask, scored "
                                                  "cells = land AND valid" if mask is not None else ""))
    return out


def truth_base(truth, daily: bool, force: bool = False) -> Path:
    """The AI+RES maps file scored against `truth`: the published file for ERA5, else
    runs/acal/analysis/s2s/<truth>/maps_{fields,daily}.nc, built once from it."""
    src = DAILY_FIELDS if daily else FIELDS
    if truth.name == "era5":
        return src
    from acal import truth as TR
    p = TR.analysis_dir(truth) / src.name
    if p.exists() and not force:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(src) as b:
        ds = truth_dataset(b.load(), truth, daily)
    tmp = p.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    os.replace(tmp, p)
    print(f"[maps] wrote {p}")
    return p


def land_mask(ds: xr.Dataset) -> xr.DataArray:
    """ERA5 land fraction on the cube grid, from any case's (global) init frame."""
    init = next((ccfg.ACAL_ROOT / "inputs").glob("*_inputs.nc"))
    with xr.open_dataset(init) as d:
        lsm = AI._squeeze(d["land_sea_mask"]).sel(lat=ds["lat"], lon=ds["lon"]).load()
    return lsm >= LAND_FRAC


def scores(ds: xr.Dataset, prob_var: str = "prob") -> xr.Dataset:
    """Family-matched per-cell scores, one map per threshold, land cells only.

    The sample is every (case) for the 7-day fields and every (case, day) for the daily
    ones. Hit rate (POD) = hits / events and CSI = hits / (hits + misses + false alarms)
    give no credit for a correct "no", so unlike accuracy they do not rise as the
    threshold gets rarer. `prob_var` names the forecast probability to score (AI+RES
    `prob`, CFS `prob_raw` / `prob_corr`).
    """
    land = land_mask(ds)
    if "valid" in ds:               # a masked truth (HRRR): score its cells only
        land = land & ds["valid"].astype(bool)
    dims = [d for d in ("case", "day") if d in ds.dims]
    out = {k: [] for k in ("accuracy", "bss", "always_no", "n_events", "pod", "csi")}
    n_case = []
    for a in THRESHOLDS:
        fam = "heat" if a > 0 else "cold"
        d = ds.sel(threshold=a).where(ds.family == fam, drop=True)
        o = (np.sign(a) * d.truth >= np.sign(a) * a).astype("float64")
        yes = (d[prob_var] >= YES).astype("float64")
        # A source without a forecast for some (case, day) - a partial slate, or day 0 of
        # a daily-mean source - carries NaN there: leave those samples out of every sum.
        # Without NaNs nothing changes (the AI+RES / CFS numbers stay bit-identical).
        valid = d[prob_var].notnull() if bool(d[prob_var].isnull().any()) else None
        if valid is not None:
            o, yes = o.where(valid), yes.where(valid)
        hits, n_ev = (yes * o).sum(dims), o.sum(dims)
        false = (yes * (1 - o)).sum(dims)
        acc = (yes == o) if valid is None else (yes == o).where(valid)
        out["accuracy"].append(acc.mean(dims))
        out["always_no"].append((1 - o).mean(dims))
        bs = ((d[prob_var] - o) ** 2).sum(dims)
        bsc = ((d.clim - o) ** 2).sum(dims)
        out["bss"].append(1 - bs / bsc.where(bsc > 0))
        out["n_events"].append(n_ev)
        out["pod"].append(hits / n_ev.where(n_ev > 0))
        den = n_ev + false
        out["csi"].append(hits / den.where(den > 0))
        n_case.append(d.sizes["case"])
    cat = lambda x: xr.concat(x, dim=pd.Index(THRESHOLDS, name="threshold"))
    return xr.Dataset({k: cat(v) for k, v in out.items()},
                      attrs=dict(n_case=n_case, daily=int("day" in dims))).where(land)


def _map_panel(ax, da, cmap, vmin, vmax, ccrs, cfeature):
    m = ax.pcolormesh(da["lon"], da["lat"], da, cmap=cmap, vmin=vmin, vmax=vmax,
                      transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="0.35")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="0.2")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, edgecolor="0.2")
    ax.set_extent([235, 294, 24, 50], crs=ccrs.PlateCarree())
    return m


def _row_vmax(sc: xr.Dataset, var: str, ths) -> float:
    """1.0 for bounded-skill maps; else the row's 98th percentile, rounded up to 0.1."""
    if var in ("accuracy", "bss"):
        return 1.0
    v = np.nanpercentile(sc[var].sel(threshold=list(ths)).values, 98)
    return float(np.ceil(v * 10) / 10)


def _figure(sc: xr.Dataset, var: str, title: str, cbar_label: str, vmin: float,
            name: str, note: str | None = None, diff: bool = False) -> Path:
    """One 2x3 map figure. `diff=True` draws a difference of two score maps: diverging
    colormap, symmetric about 0 (a colorbar centred anywhere else would mislead), and the
    panel subtitle reports the land mean and the share of land where the first source wins.
    `note` is a footnote (the CFS resolution caveat)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    AN._style(plt)
    proj = ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.6),
                             subplot_kw=dict(projection=proj),
                             gridspec_kw=dict(wspace=0.08 if diff else 0.04, hspace=0.02))
    rows = [((-2.0, -3.0, -4.0), "Blues", "cold"), ((2.0, 3.0, 4.0), "Reds", "heat")]
    n_case = dict(zip(THRESHOLDS, sc.attrs["n_case"]))
    for r, (ths, cmap_name, fam) in enumerate(rows):
        cmap = plt.get_cmap("RdBu_r" if diff else cmap_name).copy()
        if not diff:
            cmap.set_under("0.82")             # BSS < 0: no skill over climatology
        cmap.set_bad("white")                  # sea, or BSS undefined
        if diff:
            # BSS is ill-conditioned where the climatological Brier sum is tiny (rare cells
            # give differences of 10+), so its scale is capped at 1, the skill range.
            v = np.nanpercentile(np.abs(sc[var].sel(threshold=list(ths)).values), 98)
            vmax = 1.0 if var == "bss" else max(float(np.ceil(v * 10) / 10), 0.1)
            vmin = -vmax
        else:
            vmax = _row_vmax(sc, var, ths)
        for k, a in enumerate(ths):
            ax = axes[r, k]
            da = sc[var].sel(threshold=a)
            m = _map_panel(ax, da, cmap, vmin, vmax, ccrs, cfeature)
            wts = np.cos(np.deg2rad(da["lat"]))
            if diff:
                ok = da.notnull()
                win = float((da > 0).where(ok).weighted(wts).mean())
                sub = f"median {float(da.median()):+.2f}   AI+RES better on {win:.0%} of land"
            elif var == "accuracy":
                mean = float(da.weighted(wts).mean())
                base = float(sc.always_no.sel(threshold=a).weighted(wts).mean())
                sub = f"land mean {mean:.2f}   always-no {base:.2f}"
            elif var in ("pod", "csi"):
                sub = f"land mean {float(da.weighted(wts).mean()):.2f}"
            else:
                ok = da.notnull()
                pos = float((da > 0).where(ok).weighted(wts).mean())
                sub = f"median {float(da.median()):+.2f}   BSS > 0 on {pos:.0%} of land"
            n = (f"{n_case[a]} cases x {sc.attrs.get('n_days', 7)} days" if sc.attrs["daily"]
                 else f"{n_case[a]} cases")
            # the diff subtitle is the longest: 8 pt + wider wspace keep neighbours apart
            ax.set_title(f"{a:+.0f} K  ({fam}, {n})\n{sub}", fontsize=8 if diff else 9)
        cb = fig.colorbar(m, ax=axes[r, :].tolist(), shrink=0.85, pad=0.01,
                          extend="both" if diff else "min" if var == "bss" else "neither")
        cb.set_label(cbar_label)
    fig.suptitle(title, fontsize=11, y=0.97)
    if note:
        # just under the lowest drawn element (maps shrink to their aspect inside the
        # subplot box, so a fixed y left an empty band above the note)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        y0 = min(inv.transform(a.get_tightbbox(r))[0, 1] for a in fig.axes)
        fig.text(0.5, y0 - 0.015, note, ha="center", va="top", fontsize=8, color="0.3")
    (ccfg.FIG_ROOT / name).parent.mkdir(parents=True, exist_ok=True)   # truth subdirs
    return AN._save(fig, name)


def _truth_fig(truth) -> tuple[str, str]:
    """(figure-name prefix under figures/acal/, title suffix) for a truth; None = the
    published ERA5 figures (no prefix, no suffix)."""
    if truth is None:
        return "", ""
    from acal import truth as TR
    rel = TR.fig_dir(truth).relative_to(ccfg.FIG_ROOT)
    return f"{rel}/", f" [truth: {TR.label_of(truth)}]"


def _truth_files(truth) -> tuple[Path, Path]:
    """(7-day, daily) AI+RES maps files scored against `truth` (None = published)."""
    if truth is None:
        return FIELDS, DAILY_FIELDS
    return truth_base(truth, False), truth_base(truth, True)


def figures(truth=None) -> list[Path]:
    """AI+RES map figures; a `truth` writes them under its figure dir (figures/acal/hrrr/)
    from the maps files scored against it."""
    pre, suf = _truth_fig(truth)
    f7, fd = _truth_files(truth)
    ds = xr.open_dataset(f7).load()
    sc = scores(ds)
    acc = _figure(sc, "accuracy",
                  "AI+RES accuracy per cell (yes = weighted P >= 0.5)" + suf,
                  "accuracy", 0.0, f"{pre}acal_map_accuracy.png")
    bss = _figure(sc, "bss",
                  "AI+RES Brier skill score vs climatology per cell (grey = BSS < 0)" + suf,
                  "Brier skill score", 0.0, f"{pre}acal_map_bss.png")
    out = [acc, bss]
    for src, label in ((f7, "7-day mean"), (fd, "daily mean")):
        if not src.exists():
            print(f"  skip {label}: no {src}")
            continue
        sc = scores(xr.open_dataset(src).load())
        tag = "7d" if src == f7 else "daily"
        out.append(_figure(sc, "pod", f"AI+RES hit rate per cell, {label} extremes "
                           f"(yes = weighted P >= 0.5){suf}", "hit rate (POD)", 0.0,
                           f"{pre}acal_map_pod_{tag}.png"))
        out.append(_figure(sc, "csi", f"AI+RES critical success index per cell, {label} "
                           f"extremes{suf}", "CSI", 0.0, f"{pre}acal_map_csi_{tag}.png"))
    return out


# --------------------------------------------------------------------------- #
# Stage: cfs (second forecast source)
# --------------------------------------------------------------------------- #
def cfs_prob(F: np.ndarray) -> np.ndarray:
    """(threshold, ...) equal-weight member fraction mean_i 1[s A_i >= s a] over axis 0."""
    return np.stack([(np.sign(a) * F >= np.sign(a) * a).mean(0) for a in THRESHOLDS]
                    ).astype("float32")


def source_member_fields(source: str, eid: str, peak: pd.Timestamp,
                         daily: bool) -> xr.DataArray | None:
    """`cfs_member_fields` for any registered source (`acal.s2sbase`); None if the source
    has no cube for this case.

    7-day: the source's own window reduction (`s2sbase.reduce`: 13 00/12Z frames for an
    instantaneous source, UTC days peak-6..peak-1 for a daily-mean one), its CONUS mean
    asserted against the source json's `al`. Daily: instantaneous sources use the
    `cfsbase.daily_pairs` pairing (day 6 ends at the peak); a daily-mean source's UTC day
    peak-7+k fills maps day k (k = 1..6), verified on that day's (00Z, 12Z) pair
    (`utc_daily_truth`, ruling C2); day 0 is NaN.
    """
    if source == "cfs":
        return cfs_member_fields(eid, peak, daily)
    from acal import s2sbase as S2
    src = S2.get(source)
    if not (src.cube_path(eid).exists() and src.json_path(eid).exists()):
        return None
    with xr.open_dataset(src.cube_path(eid)) as cube:
        cube = cube.load()
    if not daily:
        f = AI._squeeze(S2.reduce(cube, peak, src.window))
        al = np.asarray(json.loads(src.json_path(eid).read_text())["al"], dtype="float64")
        got = np.asarray(AI.area_mean(f), dtype="float64")
        if got.shape != al.shape or np.nanmax(np.abs(got - al)) > CONSISTENCY_TOL:
            raise SystemExit(f"[maps] {source} {eid}: field CONUS means != json al")
        return f.astype("float32")
    if S2.time_kind(cube) == "instant":
        inst = AI._squeeze(AI.instantaneous_field(cube, "t2m_anom"))
        return cfsbase.daily_pairs(inst, peak).transpose("member", "day", ...).astype("float32")
    a = S2.reduce_daily(cube, peak, per_day=True)                   # (member, time=6, ...)
    out = np.full((a.sizes["member"], 7) + a.shape[2:], np.nan, "float32")
    out[:, 1:] = a.values
    return xr.DataArray(out, dims=("member", "day", "lat", "lon"),
                        coords=dict(member=a["member"].values, day=np.arange(7),
                                    lat=a["lat"].values, lon=a["lon"].values))


def source_dataset(source: str, src: Path, daily: bool, truth_p=None) -> xr.Dataset:
    """`cfs_dataset` for any registered source: the AI+RES file's truth/clim/family plus
    the source's `prob_raw` (and `prob_corr` when it has a bias). Cases the source does
    not cover are NaN (left out by `scores`). A daily-mean source's 7-day truth is the
    12-frame mean (peak-6d 00Z .. peak-1d 12Z, ruling C9; clim stays the 13-frame
    frequency); its daily truth and clim are the UTC-day (00Z D, 12Z D) pairs
    (`utc_daily_truth`, ruling C2). Either way the AI+RES side is re-reduced on the same
    frames (`prob_aires`)."""
    if source == "cfs":
        return cfs_dataset(src, daily)
    from acal import s2sbase as S2
    from acal import truth as TR
    s = S2.get(source)
    base = xr.open_dataset(src).load()
    bias = xr.open_dataset(s.bias_nc).load() if s.bias_nc.exists() else None
    cases = {c.episode_id: c for c in AN.load_all()[1]}
    dims = ("case", "day", "threshold", "lat", "lon") if daily else \
        ("case", "threshold", "lat", "lon")
    shape = (base.sizes["case"],) + ((7,) if daily else ()) + (len(THRESHOLDS),
                                                              base.sizes["lat"], base.sizes["lon"])
    raw, corr, res12 = (np.full(shape, np.nan, "float32") for _ in range(3))
    tr = TR.get_truth("era5") if truth_p is None else truth_p
    dm = s.time_kind == "daily_mean"
    truth = base["truth"].values.copy()
    if dm and daily:
        truth, clim = utc_daily_truth(tr, base)
    for i, eid in enumerate(map(str, base["case"].values)):
        peak = pd.Timestamp(cases[eid].run["peak"])
        F = source_member_fields(source, eid, peak, daily)
        if F is None:
            print(f"  {eid}  no {source} cube", flush=True)
            continue
        if not (np.allclose(F["lat"], base["lat"]) and np.allclose(F["lon"], base["lon"])):
            raise SystemExit(f"[maps] {eid}: {source} grid != maps grid")
        pr = cfs_prob(F.values)
        pr = np.where(np.isnan(F.values).all(0)[None], np.nan, pr)   # daily day 0
        raw[i] = np.moveaxis(pr, 0, 1) if daily else pr
        if bias is not None and eid in list(map(str, bias["case"].values)):
            b = bias["bias_daily" if daily else "bias7"].sel(case=eid).values
            pc = np.where(np.isnan(pr), np.nan, cfs_prob(F.values - b[None]))
            corr[i] = np.moveaxis(pc, 0, 1) if daily else pc
        if dm and not daily:
            truth[i] = TR.frames_on(tr, eid, "12f").mean("time").values
            res12[i] = aires_prob12(eid)
        elif dm:
            res12[i] = aires_prob_daily(eid)
        print(f"  {eid}  {F.sizes['member']} members", flush=True)
    d = base.drop_vars("prob")
    d["truth"] = (base["truth"].dims, truth)
    if dm and daily:
        d["clim"] = (base["clim"].dims, clim)
    if dm:
        # AI+RES re-reduced on the same frames as this source's truth (7-day: the 12
        # frames, ruling C9; daily: the UTC-day pairs, ruling C2): the AI+RES side of
        # every comparison with a daily-mean source
        d["prob_aires"] = (dims, res12)
    d["prob_raw"] = (dims, raw)
    if bias is not None:
        d["prob_corr"] = (dims, corr)
    d.attrs["note"] = (f"prob_raw / prob_corr = equal-weight {s.label} member fraction "
                       f"P(s A >= s a) on window {s.window}; truth, clim, family from "
                       + src.name + (f"; 7-day truth replaced by the {TR.label_of(tr)} 12-frame mean"
                                     if (dm and not daily) else "")
                       + (f"; daily truth and clim replaced by the {TR.label_of(tr)} UTC-day "
                          f"pairs ({S2.DAILY_PAIR}, maps day k = UTC day peak-7+k, day 0 NaN)"
                          if (dm and daily) else ""))
    d.attrs["source"] = source
    return d


def aires_prob12(eid: str) -> np.ndarray:
    """(threshold, lat, lon) self-normalized AI+RES P(s A(x) >= s a) from the walker fields
    on the '12f' window (`s2sbase.aires_fields`, built on first use), the same estimator
    as `fields` uses on 13 frames."""
    from acal import s2sbase as S2
    F, w = S2.aires_fields(eid, "12f")
    F, w = F.values, w / w.sum()
    return np.stack([np.tensordot(w, np.sign(a) * F >= np.sign(a) * a, axes=(0, 0))
                     for a in THRESHOLDS]).astype("float32")


def aires_prob_daily(eid: str) -> np.ndarray:
    """(day=7, threshold, lat, lon) self-normalized AI+RES P(s A(x) >= s a) on the UTC-day
    pairs (`s2sbase.aires_daily`): maps day k (1..6) = UTC day peak-7+k, day 0 NaN. The
    daily partner of a daily-mean source (ruling C2), the estimator of `daily`."""
    from acal import s2sbase as S2
    F, w = S2.aires_daily(eid)                                  # (walker, day=6, ...)
    F, w = F.values, w / w.sum()
    p = np.stack([np.tensordot(w, np.sign(a) * F >= np.sign(a) * a, axes=(0, 0))
                  for a in THRESHOLDS], axis=1)                 # (day=6, threshold, ...)
    out = np.full((7,) + p.shape[1:], np.nan, "float32")
    out[1:] = p
    return out


def cfs_member_fields(eid: str, peak: pd.Timestamp, daily: bool) -> xr.DataArray:
    """(member, lat, lon) 7-day or (member, day, lat, lon) daily CFS anomaly of one case.
    The CONUS mean of the 7-day field must equal the cube's own recorded `al`."""
    with xr.open_dataset(cfsbase.cube_path(eid)) as cube:
        cube = cube.load()
    if daily:
        inst = AI._squeeze(AI.instantaneous_field(cube, "t2m_anom"))
        # daily_pairs puts `day` first; members lead everywhere else here
        return cfsbase.daily_pairs(inst, peak).transpose("member", "day", ...).astype("float32")
    f = AI._squeeze(AI.field(cube, peak, "t2m_anom"))
    al = np.asarray(json.loads(cfsbase.json_path(eid).read_text())["al"], dtype="float64")
    got = np.asarray(AI.area_mean(f), dtype="float64")
    if got.shape != al.shape or np.nanmax(np.abs(got - al)) > CONSISTENCY_TOL:
        raise SystemExit(f"[maps] {eid}: CFS field CONUS means {got} != cube al {al}")
    return f.astype("float32")


def cfs_dataset(src: Path, daily: bool) -> xr.Dataset:
    """Score-ready dataset: the AI+RES file's truth/clim/family plus CFS prob_raw/prob_corr."""
    base = xr.open_dataset(src).load()
    bias = xr.open_dataset(cfsbase.BIAS_NC).load()
    if list(bias["case"].values) != list(base["case"].values):
        raise SystemExit("[maps] bias.nc case order != maps file case order")
    for k in ("lat", "lon"):
        if not np.allclose(bias[k], base[k]):
            raise SystemExit(f"[maps] bias.nc {k} != maps file {k}")
    cases = {c.episode_id: c for c in AN.load_all()[1]}
    dims = ("case", "day", "threshold", "lat", "lon") if daily else \
        ("case", "threshold", "lat", "lon")
    shape = (base.sizes["case"],) + ((7,) if daily else ()) + (len(THRESHOLDS),
                                                              base.sizes["lat"], base.sizes["lon"])
    raw, corr = (np.full(shape, np.nan, "float32") for _ in range(2))
    for i, eid in enumerate(base["case"].values):
        peak = pd.Timestamp(cases[str(eid)].run["peak"])
        F = cfs_member_fields(str(eid), peak, daily)
        if not (np.allclose(F["lat"], base["lat"]) and np.allclose(F["lon"], base["lon"])):
            raise SystemExit(f"[maps] {eid}: CFS grid != maps grid")
        b = bias["bias_daily" if daily else "bias7"].isel(case=i).values
        # (threshold, [day], lat, lon) -> put the day axis before threshold for the daily file
        pr, pc = cfs_prob(F.values), cfs_prob(F.values - b[None])
        raw[i], corr[i] = (np.moveaxis(pr, 0, 1), np.moveaxis(pc, 0, 1)) if daily else (pr, pc)
        print(f"  {eid}  {F.sizes['member']} members", flush=True)
    d = base.drop_vars("prob")
    d["prob_raw"] = (dims, raw)
    d["prob_corr"] = (dims, corr)
    d.attrs["note"] = ("prob_raw / prob_corr = equal-weight CFSv2 member fraction "
                       "P(s A >= s a), raw and leave-one-year-out bias corrected; truth, "
                       "clim, family copied from " + src.name)
    return d


def cfs_fields() -> list[Path]:
    out = []
    for src, dst, daily in ((FIELDS, CFS_FIELDS, False), (DAILY_FIELDS, CFS_DAILY_FIELDS, True)):
        cfs_dataset(src, daily).to_netcdf(dst)
        print(f"[maps] wrote {dst}")
        out.append(dst)
    return out


def land_means(sc: xr.Dataset) -> dict:
    """Cos-latitude weighted land mean of BSS and CSI per threshold, plus the land median
    of BSS (`bss_med`): the BSS mean is dominated by a few rare-event cells."""
    w = np.cos(np.deg2rad(sc["lat"]))
    out = {v: {float(a): float(sc[v].sel(threshold=a).weighted(w).mean())
               for a in THRESHOLDS} for v in ("bss", "csi")}
    out["bss_med"] = {float(a): float(sc["bss"].sel(threshold=a).median())
                      for a in THRESHOLDS}
    return out


def cfs_figures(truth=None) -> dict:
    """CFS raw (full set), CFS corrected (BSS, CSI), AI+RES minus CFS raw (BSS, CSI), and
    the land-mean summary table of all three sources. A `truth` reads the files scored
    against it (`s2sbase --source cfs --stage maps --truth <name>` first), writes its
    figures under figures/acal/hrrr/ and the land means to
    runs/acal/analysis/s2s/<truth>/maps_land_means_cfs.json."""
    pre, n_suf = _truth_fig(truth)
    f7, fd = _truth_files(truth)
    c7, cd = CFS_FIELDS, CFS_DAILY_FIELDS
    if truth is not None:
        from acal import truth as TR
        c7, cd = (TR.analysis_dir(truth) / f"maps_{k}_cfs.nc" for k in ("fields", "daily"))
    summary = {}
    fig = lambda sc, var, title, *a, **kw: _figure(sc, var, title + n_suf, *a, **kw)  # noqa: E731
    for tag, label, res_src, cfs_src in (("7d", "7-day mean", f7, c7),
                                         ("daily", "daily mean", fd, cd)):
        res_ds = xr.open_dataset(res_src).load()
        cfs_ds = xr.open_dataset(cfs_src).load()
        s_res = scores(res_ds)
        s_raw, s_cor = scores(cfs_ds, "prob_raw"), scores(cfs_ds, "prob_corr")
        summary[tag] = {"AI+RES": land_means(s_res), "CFS raw": land_means(s_raw),
                        "CFS corrected": land_means(s_cor)}
        n = CFS_NOTE          # the truth label goes in the title (as source_figures)
        if tag == "7d":
            fig(s_raw, "accuracy", "CFSv2 accuracy per cell (yes = member fraction >= 0.5)",
                "accuracy", 0.0, f"{pre}acal_map_accuracy_cfs.png", n)
            fig(s_raw, "bss", "CFSv2 Brier skill score vs climatology per cell "
                "(grey = BSS < 0)", "Brier skill score", 0.0, f"{pre}acal_map_bss_cfs.png", n)
        fig(s_raw, "pod", f"CFSv2 hit rate per cell, {label} extremes "
            "(yes = member fraction >= 0.5)", "hit rate (POD)", 0.0,
            f"{pre}acal_map_pod_{tag}_cfs.png", n)
        fig(s_raw, "csi", f"CFSv2 critical success index per cell, {label} extremes",
            "CSI", 0.0, f"{pre}acal_map_csi_{tag}_cfs.png", n)
        if tag == "7d":
            fig(s_cor, "bss", "CFSv2 bias-corrected Brier skill score per cell, "
                "7-day mean (grey = BSS < 0)", "Brier skill score", 0.0,
                f"{pre}acal_map_bss_cfscorr.png", n)
        fig(s_cor, "csi", f"CFSv2 bias-corrected CSI per cell, {label} extremes",
            "CSI", 0.0, f"{pre}acal_map_csi_{tag}_cfscorr.png", n)
        for var, nm in (("bss", "BSS"), ("csi", "CSI")):
            dd = s_res.copy(deep=True)
            dd[var] = s_res[var] - s_raw[var]
            fig(dd, var, f"{nm} difference per cell, AI+RES minus CFSv2 (raw), {label} "
                f"extremes (red = AI+RES better)", f"{nm} difference", 0.0,
                f"{pre}acal_map_{var}_{tag}_diff_cfs.png", n, diff=True)
    if truth is not None:
        p = TR.analysis_dir(truth) / "maps_land_means_cfs.json"
        p.write_text(json.dumps({t: {k: {v: {str(a): x for a, x in d.items()}
                                          for v, d in m.items()} for k, m in per.items()}
                                 for t, per in summary.items()}, indent=1))
        print(f"[maps] wrote {p}")
    print("[maps] land mean (cos-lat weighted) BSS / CSI; bss_med = land median BSS")
    for tag, per in summary.items():
        for v in ("bss", "bss_med", "csi"):
            print(f"  {tag} {v.upper()}  " + "  ".join(f"{a:+.0f}K" for a in THRESHOLDS))
            for src, m in per.items():
                print(f"    {src:14s}" + "  ".join(f"{m[v][a]:+.3f}" for a in THRESHOLDS))
    return summary


def cfs() -> dict:
    cfs_fields()
    return cfs_figures()


def on_source_cases(ds: xr.Dataset, var: str, src_ds: xr.Dataset,
                    src_var: str = "prob_raw") -> xr.Dataset:
    """`ds` with `var` set to NaN on every (case[, day]) where the source has no forecast
    (`src_var` all-NaN there, or the case absent), so AI+RES and a partial source are
    scored on the same samples (`scores` drops NaN samples). Full coverage: `ds` unchanged."""
    p = src_ds[src_var]
    cov = p.notnull().any([d for d in p.dims if d not in ("case", "day")])
    cov = cov.reindex(case=ds["case"].values, fill_value=False)
    if "day" in cov.dims and "day" not in ds[var].dims:
        cov = cov.any("day")
    if bool(cov.all()):
        return ds
    ds = ds.copy()
    ds[var] = ds[var].where(cov)
    return ds


def source_figures(source: str, truth: str = "era5", out_dir: Path | str | None = None) -> dict:
    """`cfs_figures` for any registry source, written to `out_dir` (default
    figures/acal/s2s/<truth>/<source>/maps/), never to a published path. Reads
    `s2sbase --source <source> --stage maps --truth <truth>` output. A daily-mean source's
    7-day comparison uses AI+RES re-reduced on the same 12 frames as its truth
    (`prob_aires` in its maps file). Land means -> runs/acal/analysis/s2s/<truth>/
    maps_land_means_<source>.json."""
    from acal import s2sbase as S2
    from acal import truth as TR
    tr, src = TR.get_truth(truth), S2.get(source)
    st = S2.MODEL_STYLE.get(source, {"label": src.label, "deg": f"{src.native_deg:g} deg"})
    lab = st["label"]
    od = Path(out_dir) if out_dir is not None else S2.fig_dir(source, tr.name) / "maps"
    od.mkdir(parents=True, exist_ok=True)
    suf = f" [truth: {TR.label_of(tr)}]"
    res_note = (f"{lab} native grid {st.get('deg', f'{src.native_deg:g} deg')}, regridded "
                "to 0.25 deg: gridpoint skill is resolution-limited (unlike the CONUS index)")
    dm = src.time_kind == "daily_mean"
    summary, out = {}, []
    for tag, label, daily in (("7d", "7-day mean", False), ("daily", "daily mean", True)):
        note = res_note
        if dm and not daily:
            note += ("\n7-day: daily means of UTC days peak-6..peak-1, verified (with AI+RES) "
                     "on the 12 frames peak-6d 00Z..peak-1d 12Z")
        elif dm:
            note += ("\ndaily: UTC days peak-6..peak-1, each verified (with AI+RES) on its "
                     "00Z and 12Z frames, the pair closest to a UTC-day mean (6 days per case)")
        res_ds = xr.open_dataset(truth_base(tr, daily)).load()
        sp = S2.out_dir(tr) / f"maps_{'daily' if daily else 'fields'}_{source}.nc"
        src_ds = xr.open_dataset(sp).load()
        if dm and "prob_aires" not in src_ds:
            raise SystemExit(f"[maps] {sp} lacks prob_aires (written before rulings C9/C2): "
                             f"rerun `python -m acal.s2sbase --source {source} --stage maps "
                             f"--truth {tr.name}`")
        # AI+RES on the cases (and days) the source covers: a partial source must not be
        # compared with AI+RES on all 42 cases
        s_res = (scores(on_source_cases(src_ds, "prob_aires", src_ds), "prob_aires")
                 if "prob_aires" in src_ds else scores(on_source_cases(res_ds, "prob", src_ds)))
        s_raw = scores(src_ds, "prob_raw")
        s_cor = scores(src_ds, "prob_corr") if "prob_corr" in src_ds else None
        if dm and daily:
            for x in (s_res, s_raw, s_cor):
                if x is not None:
                    x.attrs["n_days"] = 6
        summary[tag] = {"AI+RES": land_means(s_res), f"{lab} raw": land_means(s_raw)}
        if s_cor is not None:
            summary[tag][f"{lab} corrected"] = land_means(s_cor)
        f = lambda sc, var, title, cb, name, **kw: out.append(        # noqa: E731
            _figure(sc, var, title + suf, cb, 0.0, od / name, note, **kw))
        if tag == "7d":
            f(s_raw, "accuracy", f"{lab} accuracy per cell (yes = member fraction >= 0.5)",
              "accuracy", f"map_accuracy_{source}.png")
            f(s_raw, "bss", f"{lab} Brier skill score vs climatology per cell "
              "(grey = BSS < 0)", "Brier skill score", f"map_bss_7d_{source}.png")
        f(s_raw, "pod", f"{lab} hit rate per cell, {label} extremes (yes = member fraction "
          ">= 0.5)", "hit rate (POD)", f"map_pod_{tag}_{source}.png")
        f(s_raw, "csi", f"{lab} critical success index per cell, {label} extremes", "CSI",
          f"map_csi_{tag}_{source}.png")
        if s_cor is not None:
            if tag == "7d":
                f(s_cor, "bss", f"{lab} bias-corrected Brier skill score per cell, 7-day "
                  "mean (grey = BSS < 0)", "Brier skill score", f"map_bss_7d_{source}corr.png")
            f(s_cor, "csi", f"{lab} bias-corrected CSI per cell, {label} extremes", "CSI",
              f"map_csi_{tag}_{source}corr.png")
        for var, nm in (("bss", "BSS"), ("csi", "CSI")):
            dd = s_res.copy(deep=True)
            dd[var] = s_res[var] - s_raw[var]
            f(dd, var, f"{nm} difference per cell, AI+RES minus {lab} (raw), {label} "
              "extremes (red = AI+RES better)", f"{nm} difference",
              f"map_{var}_{tag}_diff_{source}.png", diff=True)
    p = S2.out_dir(tr) / f"maps_land_means_{source}.json"
    p.write_text(json.dumps({t: {k: {v: {str(a): x for a, x in d.items()}
                                      for v, d in m.items()} for k, m in per.items()}
                             for t, per in summary.items()}, indent=1))
    print(f"[maps] {source} vs {tr.name}: {len(out)} figures -> {od}; land means -> {p}")
    for tag, per in summary.items():
        for v in ("bss", "csi"):
            print(f"  {tag} {v.upper()}  " + "  ".join(f"{a:+.0f}K" for a in THRESHOLDS))
            for k, m in per.items():
                print(f"    {k:22s}" + "  ".join(f"{m[v][a]:+.3f}" for a in THRESHOLDS))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=("fields", "daily", "figures", "cfs", "source", "all"),
                   default="all")
    p.add_argument("--source", default=None,
                   help="--stage source: per-source figures of this registry source")
    p.add_argument("--out-dir", default=None,
                   help="--stage source: figure dir (default figures/acal/s2s/<truth>/<source>/maps)")
    p.add_argument("--truth", default=None,
                   help="hrrr | hrrr_raw | era5: build the AI+RES maps files against this "
                        "truth (runs/acal/analysis/s2s/<truth>/) and draw --stage figures / "
                        "cfs from them into its figure dir; the CFS file comes from "
                        "`s2sbase --source cfs --stage maps --truth <truth>`")
    a = p.parse_args(argv)
    if a.stage == "source":
        if not a.source:
            p.error("--stage source needs --source")
        source_figures(a.source, a.truth or "era5", a.out_dir)
        return 0
    if a.truth is not None:
        from acal import truth as TR
        tr = TR.get_truth(a.truth)
        if a.stage in ("fields", "daily", "all"):
            for d in (False, True):
                truth_base(tr, d)
        if a.stage in ("figures", "all"):
            figures(tr)
        if a.stage == "cfs":
            cfs_figures(tr)
        return 0
    if a.stage in ("fields", "all"):
        fields()
    if a.stage in ("daily", "all"):
        daily()
    if a.stage in ("figures", "all"):
        figures()
    if a.stage == "cfs":
        cfs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
