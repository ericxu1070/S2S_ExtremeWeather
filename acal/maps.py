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
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import ccfg
from aires import aconfig as A
from aires import aindex as AI

THRESHOLDS = (-2.0, -3.0, -4.0, 2.0, 3.0, 4.0)
FIELDS = AN.OUT / "maps_fields.nc"
DAILY_FIELDS = AN.OUT / "maps_daily.nc"
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
def land_mask(ds: xr.Dataset) -> xr.DataArray:
    """ERA5 land fraction on the cube grid, from any case's (global) init frame."""
    init = next((ccfg.ACAL_ROOT / "inputs").glob("*_inputs.nc"))
    with xr.open_dataset(init) as d:
        lsm = AI._squeeze(d["land_sea_mask"]).sel(lat=ds["lat"], lon=ds["lon"]).load()
    return lsm >= LAND_FRAC


def scores(ds: xr.Dataset) -> xr.Dataset:
    """Family-matched per-cell scores, one map per threshold, land cells only.

    The sample is every (case) for the 7-day fields and every (case, day) for the daily
    ones. Hit rate (POD) = hits / events and CSI = hits / (hits + misses + false alarms)
    give no credit for a correct "no", so unlike accuracy they do not rise as the
    threshold gets rarer.
    """
    land = land_mask(ds)
    dims = [d for d in ("case", "day") if d in ds.dims]
    out = {k: [] for k in ("accuracy", "bss", "always_no", "n_events", "pod", "csi")}
    n_case = []
    for a in THRESHOLDS:
        fam = "heat" if a > 0 else "cold"
        d = ds.sel(threshold=a).where(ds.family == fam, drop=True)
        o = (np.sign(a) * d.truth >= np.sign(a) * a).astype("float64")
        yes = (d.prob >= YES).astype("float64")
        hits, n_ev = (yes * o).sum(dims), o.sum(dims)
        false = (yes * (1 - o)).sum(dims)
        out["accuracy"].append((yes == o).mean(dims))
        out["always_no"].append((1 - o).mean(dims))
        bs = ((d.prob - o) ** 2).sum(dims)
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
            name: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    AN._style(plt)
    proj = ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.6),
                             subplot_kw=dict(projection=proj),
                             gridspec_kw=dict(wspace=0.04, hspace=0.02))
    rows = [((-2.0, -3.0, -4.0), "Blues", "cold"), ((2.0, 3.0, 4.0), "Reds", "heat")]
    n_case = dict(zip(THRESHOLDS, sc.attrs["n_case"]))
    for r, (ths, cmap_name, fam) in enumerate(rows):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_under("0.82")                 # BSS < 0: no skill over climatology
        cmap.set_bad("white")                  # sea, or BSS undefined
        vmax = _row_vmax(sc, var, ths)
        for k, a in enumerate(ths):
            ax = axes[r, k]
            da = sc[var].sel(threshold=a)
            m = _map_panel(ax, da, cmap, vmin, vmax, ccrs, cfeature)
            wts = np.cos(np.deg2rad(da["lat"]))
            if var == "accuracy":
                mean = float(da.weighted(wts).mean())
                base = float(sc.always_no.sel(threshold=a).weighted(wts).mean())
                sub = f"land mean {mean:.2f}   always-no {base:.2f}"
            elif var in ("pod", "csi"):
                sub = f"land mean {float(da.weighted(wts).mean()):.2f}"
            else:
                ok = da.notnull()
                pos = float((da > 0).where(ok).weighted(wts).mean())
                sub = f"median {float(da.median()):+.2f}   BSS > 0 on {pos:.0%} of land"
            n = f"{n_case[a]} cases x 7 days" if sc.attrs["daily"] else f"{n_case[a]} cases"
            ax.set_title(f"{a:+.0f} K  ({fam}, {n})\n{sub}", fontsize=9)
        cb = fig.colorbar(m, ax=axes[r, :].tolist(), shrink=0.85, pad=0.01,
                          extend="min" if var == "bss" else "neither")
        cb.set_label(cbar_label)
    fig.suptitle(title, fontsize=11, y=0.97)
    return AN._save(fig, name)


def figures() -> list[Path]:
    ds = xr.open_dataset(FIELDS).load()
    sc = scores(ds)
    acc = _figure(sc, "accuracy",
                  "AI+RES accuracy per cell (yes = weighted P >= 0.5)",
                  "accuracy", 0.0, "acal_map_accuracy.png")
    bss = _figure(sc, "bss",
                  "AI+RES Brier skill score vs climatology per cell (grey = BSS < 0)",
                  "Brier skill score", 0.0, "acal_map_bss.png")
    out = [acc, bss]
    for src, label in ((FIELDS, "7-day mean"), (DAILY_FIELDS, "daily mean")):
        if not src.exists():
            print(f"  skip {label}: no {src}")
            continue
        sc = scores(xr.open_dataset(src).load())
        tag = "7d" if src == FIELDS else "daily"
        out.append(_figure(sc, "pod", f"AI+RES hit rate per cell, {label} extremes "
                           f"(yes = weighted P >= 0.5)", "hit rate (POD)", 0.0,
                           f"acal_map_pod_{tag}.png"))
        out.append(_figure(sc, "csi", f"AI+RES critical success index per cell, {label} "
                           f"extremes", "CSI", 0.0, f"acal_map_csi_{tag}.png"))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=("fields", "daily", "figures", "all"), default="all")
    a = p.parse_args(argv)
    if a.stage in ("fields", "all"):
        fields()
    if a.stage in ("daily", "all"):
        daily()
    if a.stage in ("figures", "all"):
        figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
