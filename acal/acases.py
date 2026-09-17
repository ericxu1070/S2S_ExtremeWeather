#!/usr/bin/env python
"""Build the calibration case slate: every +/-2, +/-3, +/-4 K CONUS box anomaly, 2021-2026.

Three stages, all CPU, all cache-aware:

    --stage cube   ERA5 (ARCO) 2m_temperature over the CONUS crop, 00Z and 12Z, minus the
                   1990-2019 climatology, one NetCDF per calendar year. Needs internet;
                   login node only (the a3mega GPU nodes are for GPU work).
    --stage scan   the 297-cell box lattice x every peak day -> A_L, threshold crossings,
                   declustering -> runs/acal/cases.csv. Offline.
    --stage report counts by rung, sign, season and year. Offline.

Why 12-hourly and not daily
---------------------------
`A_L` is the mean of the THIRTEEN frames in [peak-6d, peak] at a 12 h step
(`aires.aindex.check_window`). Building the scan on daily means would make the detector's
number a different statistic from the one AI+RES is later scored on, and a case selected
at +2.02 K by one definition can sit at +1.94 K under the other. Sampling 00Z/12Z and
taking the exact 13-frame mean makes the detector's value `A_L` by construction, so the
slate's rungs are the rungs the campaign verifies against.

The cost of that choice is that `A_L` is not a daily mean: it is a 13-point sample of a
6-day span, so it carries whatever the 00Z/12Z pair does to the diurnal cycle. That is
fine here precisely because the climatology is subtracted at the SAME two hours - the
diurnal term cancels to the extent that it is climatological, and what is left is the
anomaly AI+RES actually forecasts.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gencast_s2s.config as C
import gencast_s2s.data as D
from acal import ccfg


def _year_file(year: int) -> Path:
    return ccfg.ACAL_ROOT / "index" / f"era5_t2m_anom_12h_{year}.nc"


# --------------------------------------------------------------------------- #
# Stage: cube
# --------------------------------------------------------------------------- #
def build_year(year: int, *, verbose: bool = True) -> Path:
    """One calendar year of 12-hourly CONUS T2m anomaly. Idempotent."""
    out = _year_file(year)
    if out.exists():
        if verbose:
            print(f"[cube {year}] cached: {out.name}")
        return out

    # The scan needs peak-6d for the first peak of the year, so every year file carries
    # a 6-day lead-in from the previous December. The overlap is dropped on merge.
    start = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=ccfg.WINDOW_DAYS)
    # The last year of the scan is a partial one: ARCO-ERA5 runs a few days behind real
    # time, so the year file stops at PEAK_END rather than at 31 December.
    end = min(pd.Timestamp(f"{year}-12-31T12:00"), pd.Timestamp(ccfg.PEAK_END))
    if end < start:
        raise SystemExit(f"year {year} lies entirely past PEAK_END {ccfg.PEAK_END}")
    times = pd.date_range(start, end, freq="12h")
    times = times[np.isin(times.hour, ccfg.HOURS)]

    clim = D._clim_src()                                   # (hour, dayofyear, lat, lon)
    lat_c, lon_c = D._arco_lat_lon_coords(conus=True, stride=1)
    if lat_c.size != clim.lat.size or lon_c.size != clim.lon.size:
        raise SystemExit(
            f"grid mismatch: ARCO CONUS {lat_c.size}x{lon_c.size} vs climatology "
            f"{clim.lat.size}x{clim.lon.size}. See the climatology trap in CLAUDE.md - "
            f"verify the grid, not the filename.")

    frames = np.empty((len(times), lat_c.size, lon_c.size), dtype="float32")
    for i, t in enumerate(times):
        obs = D.arco_read("2m_temperature", t, conus=True, stride=1)
        cl = clim.sel(hour=int(t.hour), dayofyear=int(t.dayofyear)).values
        frames[i] = (obs - cl).astype("float32")
        if i % 200 == 0:
            D.release_stores()
            gc.collect()
            if verbose:
                print(f"[cube {year}] {i:5d}/{len(times)}  {t:%Y-%m-%d %HZ}", flush=True)

    ds = xr.Dataset(
        {"t2m_anom": (("time", "lat", "lon"), frames)},
        coords={"time": times, "lat": lat_c, "lon": lon_c},
        attrs={"source": "ARCO-ERA5 minus WB2 1990-2019 hourly climatology",
               "hours": ",".join(str(h) for h in ccfg.HOURS),
               "note": "12-hourly CONUS T2m anomaly; A_L = 13-frame mean over [peak-6d, peak]"},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    D._atomic_to_netcdf(ds, out)
    if verbose:
        print(f"[cube {year}] wrote {out.name}  {dict(ds.sizes)}")
    return out


def load_index(peak_start: str | None = None, peak_end: str | None = None) -> xr.DataArray:
    """Every cached year file, concatenated and de-overlapped."""
    lo = pd.Timestamp(peak_start or ccfg.PEAK_START) - pd.Timedelta(days=ccfg.WINDOW_DAYS)
    hi = pd.Timestamp(peak_end or ccfg.PEAK_END)
    files = sorted(p for p in (ccfg.ACAL_ROOT / "index").glob("era5_t2m_anom_12h_*.nc"))
    if not files:
        raise SystemExit("no index years built; run --stage cube first")
    das = [xr.open_dataset(f)["t2m_anom"] for f in files]
    da = xr.concat(das, dim="time")
    _, keep = np.unique(da["time"].values, return_index=True)
    da = da.isel(time=np.sort(keep)).sortby("time")
    return da.sel(time=slice(lo, hi)).load()


# --------------------------------------------------------------------------- #
# Stage: scan
# --------------------------------------------------------------------------- #
def box_series(da: xr.DataArray) -> tuple[pd.DatetimeIndex, np.ndarray, list]:
    """`A_L` for every lattice cell and every peak day.

    Returns `(peaks, A, lattice)` with `A[i, j]` the 13-frame window mean for peak
    `peaks[i]` and lattice cell `lattice[j]`.
    """
    lattice = ccfg.box_lattice()
    lat = da["lat"].values
    lon = da["lon"].values
    w = np.cos(np.deg2rad(lat)).astype("float64")

    # Box means for every cell, vectorised over time: (time, cell).
    vals = da.values.astype("float64")                       # (time, lat, lon)
    nt = vals.shape[0]
    bm = np.empty((nt, len(lattice)), dtype="float64")
    for j, (la0, la1, lo0, lo1) in enumerate(lattice):
        mi = (lat >= la0 - 1e-6) & (lat <= la1 + 1e-6)
        mj = (lon >= lo0 - 1e-6) & (lon <= lo1 + 1e-6)
        sub = vals[:, mi][:, :, mj]
        ww = w[mi][None, :, None]
        bm[:, j] = (sub * ww).sum(axis=(1, 2)) / (ww.sum() * mj.sum())

    # A_L(peak) = mean of the 13 frames ending at `peak`. A rolling mean over the frame
    # axis gives exactly that, and peaks are the 00Z frames where the window is complete.
    n = ccfg.N_WINDOW_FRAMES
    csum = np.cumsum(np.vstack([np.zeros((1, bm.shape[1])), bm]), axis=0)
    roll = (csum[n:] - csum[:-n]) / n                        # ends at frame index n-1..
    end_times = pd.DatetimeIndex(da["time"].values[n - 1:])
    is00 = end_times.hour == 0
    return end_times[is00], roll[is00], lattice


def _overlap_frac(a, b) -> float:
    """Fraction of a box's area shared by two lattice cells (both are BOX_SIZE square)."""
    la = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    lo = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return (la * lo) / (ccfg.BOX_SIZE ** 2)


def find_cases(peaks: pd.DatetimeIndex, A: np.ndarray, lattice: list) -> pd.DataFrame:
    """Threshold crossings, declustered in space and time, one row per surviving case.

    The rule, in order:
      1. every (cell, peak) whose |A_L| >= min(RUNGS) is a candidate, filed by sign;
      2. candidates are visited in descending |A_L| (greedy, so the strongest wins);
      3. a candidate is SUPPRESSED if an already-accepted case of the same sign is within
         DECLUSTER_DAYS and overlaps it by more than DECLUSTER_OVERLAP of a box area.
    Greedy peak-first declustering is what makes the representative of an event the box
    that maximises |A_L| - the same choice aires/aindex.py makes by hand per event.
    """
    rung_min = min(ccfg.RUNGS)
    ti, ci = np.where(np.abs(A) >= rung_min)
    if ti.size == 0:
        return pd.DataFrame()
    mag = np.abs(A[ti, ci])
    order = np.argsort(-mag)

    accepted: list[dict] = []
    # bucket accepted cases by sign to keep the overlap test cheap
    by_sign: dict[int, list[int]] = {1: [], -1: []}
    day = peaks.values.astype("datetime64[D]").astype("int64")

    for k in order:
        i, j = int(ti[k]), int(ci[k])
        val = float(A[i, j])
        sign = 1 if val > 0 else -1
        cell = lattice[j]
        clash = False
        for idx in by_sign[sign]:
            other = accepted[idx]
            if abs(day[i] - other["_day"]) > ccfg.DECLUSTER_DAYS:
                continue
            if _overlap_frac(cell, other["_cell"]) > ccfg.DECLUSTER_OVERLAP:
                clash = True
                break
        if clash:
            continue
        rung = max(r for r in ccfg.RUNGS if abs(val) >= r)
        by_sign[sign].append(len(accepted))
        accepted.append(dict(
            _day=int(day[i]), _cell=cell,
            peak=pd.Timestamp(peaks[i]).strftime("%Y-%m-%d"),
            box=ccfg.box_name(cell[0], cell[2]),
            lat_lo=cell[0], lat_hi=cell[1], lon_lo=cell[2], lon_hi=cell[3],
            a_l=round(val, 4), rung=rung, sign=sign,
            family="heat" if sign > 0 else "cold",
        ))

    df = pd.DataFrame(accepted).drop(columns=["_day", "_cell"])
    df = df.sort_values(["peak", "box"]).reset_index(drop=True)
    df.insert(0, "case_id", [f"c{i + 1:04d}_{r.box}_{r.peak.replace('-', '')}"
                             for i, r in enumerate(df.itertuples())])
    df["init"] = (pd.to_datetime(df["peak"]) - pd.Timedelta(days=ccfg.LEAD_DAYS)
                  ).dt.strftime("%Y-%m-%d")
    return df


def do_scan(write: bool = True) -> pd.DataFrame:
    da = load_index()
    peaks, A, lattice = box_series(da)
    print(f"[scan] {len(peaks)} peak days x {len(lattice)} lattice cells "
          f"= {len(peaks) * len(lattice):,} (box, day) pairs")
    df = find_cases(peaks, A, lattice)
    print(f"[scan] {len(df)} cases after declustering "
          f"({ccfg.DECLUSTER_DAYS} d, >{ccfg.DECLUSTER_OVERLAP:.0%} box overlap)")
    if write:
        ccfg.ensure_dirs()
        df.to_csv(ccfg.CASES_CSV, index=False)
        meta = dict(
            peak_start=ccfg.PEAK_START, peak_end=ccfg.PEAK_END,
            box_size=ccfg.BOX_SIZE, box_step=ccfg.BOX_STEP,
            n_lattice=len(lattice), rungs=list(ccfg.RUNGS),
            decluster_days=ccfg.DECLUSTER_DAYS,
            decluster_overlap=ccfg.DECLUSTER_OVERLAP,
            window_frames=ccfg.N_WINDOW_FRAMES, hours=list(ccfg.HOURS),
            lead_days=ccfg.LEAD_DAYS, n_cases=len(df),
            first_peak=str(peaks[0].date()), last_peak=str(peaks[-1].date()),
        )
        ccfg.CASES_META.write_text(json.dumps(meta, indent=2))
        print(f"[scan] wrote {ccfg.CASES_CSV} and {ccfg.CASES_META.name}")
    return df


# --------------------------------------------------------------------------- #
# Stage: report
# --------------------------------------------------------------------------- #
def do_report(df: pd.DataFrame | None = None) -> None:
    df = pd.read_csv(ccfg.CASES_CSV) if df is None else df
    print(f"\n=== {len(df)} cases, {df.peak.min()} .. {df.peak.max()} ===\n")
    piv = df.pivot_table(index="rung", columns="family", values="case_id",
                         aggfunc="count", fill_value=0)
    piv["total"] = piv.sum(axis=1)
    print("by rung and sign:"); print(piv.to_string()); print()
    yr = pd.to_datetime(df.peak).dt.year
    print("by year:"); print(df.groupby([yr, "family"]).size().unstack(fill_value=0).to_string()); print()
    mon = pd.to_datetime(df.peak).dt.month
    season = mon.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                      6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    print("by season:"); print(df.groupby([season, "family"]).size().unstack(fill_value=0).to_string())
    print(f"\n|A_L| range: {df.a_l.abs().min():.2f} .. {df.a_l.abs().max():.2f} K")
    n = len(df)
    print(f"\ncost at N={ccfg.RES_N_WALKERS}: ~{n * 31:,.0f} H100-h "
          f"= ~{n * 31 / 8:,.0f} node-h = ~{n * 31 / 8 / 5 / 24:.1f} d wall on 5 nodes")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["cube", "scan", "report", "all"], default="all")
    ap.add_argument("--year", type=int, default=None, help="cube stage: one year only")
    a = ap.parse_args(argv)

    if a.stage in ("cube", "all"):
        y0 = pd.Timestamp(ccfg.PEAK_START).year
        y1 = pd.Timestamp(ccfg.PEAK_END).year
        years = [a.year] if a.year else list(range(y0, y1 + 1))
        for y in years:
            build_year(y)
    if a.stage in ("scan", "all"):
        do_scan()
    if a.stage in ("report", "all"):
        do_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
