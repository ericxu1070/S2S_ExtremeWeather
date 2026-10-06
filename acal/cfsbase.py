#!/usr/bin/env python
"""CFSv2 operational baseline for the calibration campaign. LOGIN NODE (internet, CPU).

The question: on the same 42 cases, at the same 21 d lead, with the same CONUS `A_L`
reduction, does AI+RES put more probability on the observed tail than NCEP's operational
CFSv2? `acal/CFS_PLAN.md` is the plan; this module is the code.

The ensemble
------------
16 trailing 6-hourly cycles ending ON the AI+RES init (leads 21.0-24.75 d), built with
`aires.cfs.build` - the same fetch, regrid and `aires.aindex` reduction that produced the
aires CFS table. No member is at a shorter lead than the walkers. The 4 cycles nearest
the init (the aires convention) are the last 4 members and are reported as a subset.

Cubes go to `runs/acal/cfs/<case>_cfs16.nc`, NOT `aconfig.cfs_cube_path`: that path does
not encode the cycle count, and a 16-member cube there would replace the 4-member one.

The anomaly is CFSv2 minus the ERA5 1990-2019 climatology, uncorrected, exactly as the
walkers are scored. See the `aires/cfs.py` docstring before quoting a number.

    python -m acal.cfsbase --stage build              # all 42 cases (~45 min, ~6 GB)
    python -m acal.cfsbase --stage build --case e02_c4_20210218
    python -m acal.cfsbase --stage hind --workers 4   # bias hindcasts (~1-1.5 h)
    python -m acal.cfsbase --stage bias               # runs/acal/cfs/bias.nc (seconds)

The bias correction (sensitivity, not the headline)
---------------------------------------------------
For each case, the same 16-cycle lagged ensemble is rebuilt at the same calendar init in
every OTHER year whose verification window the ERA5 index cubes cover (2021-2026, peaks
up to 2026-08-31), so the correction is leave-one-year-out by construction. Per
(case, year) the hindcast stores the 16 member CONUS `A_L`s, ERA5's, and the CFS
member-mean minus ERA5 fields over the 7-day window and each of its 7 days. The bias is
the mean of those differences over years, and it is defined on the SCORED quantity: the
CFS 7-day field is `aires.aindex.field` (all 25 six-hourly frames, as `build` scores it)
and ERA5's is the 13-frame 12-hourly mean, so the 6 h vs 12 h sampling and the two
climatology files' offset are inside the bias rather than left as a residual. Daily
fields use only the 00Z frame and the 12Z before it, the pairing `acal.maps` uses.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import aprep, ccfg
from aires import aindex as AI
from aires import cfs

N_CYCLES = 16
N_SUBSET = 4                             # the aires convention: init-18h .. init
MIN_MEMBERS = 12                         # fewer is flagged, not dropped
CFS_ROOT = ccfg.ACAL_ROOT / "cfs"
BUILD_CSV = CFS_ROOT / "build.csv"
HIND_ROOT = CFS_ROOT / "hind"
BIAS_NC = CFS_ROOT / "bias.nc"
HIND_YEARS = range(2021, 2027)
ERA5_FIRST_PEAK = pd.Timestamp("2021-01-02")  # 2021 cube starts 2020-12-26 00Z; the daily pairs need peak-6.5d
ERA5_LAST_PEAK = pd.Timestamp(ccfg.PEAK_END)
MIN_YEARS = 4
WINDOW_FRAMES = 13


def cube_path(eid: str) -> Path:
    return CFS_ROOT / f"{eid}_cfs{N_CYCLES}.nc"


def json_path(eid: str) -> Path:
    return CFS_ROOT / f"{eid}_cfs{N_CYCLES}.json"


def build_case(row, force: bool = False, keep_grib: bool = False) -> dict:
    """Build (or reuse) one case's cube and return its per-member CONUS A_L record."""
    eid = row.episode_id
    work = CFS_ROOT / ".grib" / eid
    out = cfs.build(eid, n_cycles=N_CYCLES, mode="trailing", force=force,
                    workdir=work, out=cube_path(eid))
    with xr.open_dataset(out) as cube:
        d = cfs.indices(eid, cube)
        attrs = dict(cube.attrs)
    # acal cases are CONUS-selected, so aindex.box_for falls back to CONUS and the "box"
    # and "conus" reductions must be the same numbers. If they ever differ, the case was
    # scored on some other box and nothing downstream would mean what it says.
    if not np.allclose(d["box"], d["conus"], atol=1e-6):
        raise SystemExit(f"[cfsbase] {eid}: box A_L != CONUS A_L - not a CONUS case?")
    if not keep_grib:
        shutil.rmtree(work, ignore_errors=True)

    al = np.asarray(d["conus"], dtype=float)
    s = 1.0 if row.family == "heat" else -1.0
    obs = float(row.a_l_conus)
    rec = dict(
        episode_id=eid, family=row.family, obs=obs, sign=s,
        n_members=int(al.size), cycles=d["cycles"],
        member_lead_days=d["member_lead_days"], al=al.tolist(),
        al_mean=float(al.mean()), al_std=float(al.std(ddof=1)),
        al_sub_mean=float(al[-N_SUBSET:].mean()),
        n_reach=int(np.sum(s * al >= s * obs)),
        n_reach_sub=int(np.sum(s * al[-N_SUBSET:] >= s * obs)),
        init_anom=d["init_anom"],
        cycles_skipped=attrs.get("cycles_skipped", "none"),
        archives=attrs.get("archives", ""),
    )
    json_path(eid).write_text(json.dumps(rec, indent=2))
    flag = "" if al.size >= MIN_MEMBERS else f"  FLAG: only {al.size} members"
    print(f"[cfsbase] {eid}: obs {obs:+.2f}  CFS mean {rec['al_mean']:+.2f} "
          f"(sd {rec['al_std']:.2f})  reach {rec['n_reach']}/{al.size}{flag}")
    return rec


def build(cases: list[str] | None = None, force: bool = False,
          keep_grib: bool = False) -> pd.DataFrame:
    CFS_ROOT.mkdir(parents=True, exist_ok=True)
    df = aprep.episodes()
    if cases:
        df = df[df.episode_id.isin(cases)]
        if df.empty:
            raise SystemExit(f"[cfsbase] no such case(s): {cases}")
    recs = [build_case(r, force, keep_grib) for r in df.itertuples()]
    # Merge with any rows already on disk, so a --case rebuild does not truncate the table.
    tab = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("cycles", "member_lead_days", "al")} for r in recs])
    if BUILD_CSV.exists() and cases:
        old = pd.read_csv(BUILD_CSV)
        tab = pd.concat([old[~old.episode_id.isin(tab.episode_id)], tab])
    order = {e: i for i, e in enumerate(aprep.episodes().episode_id)}
    tab = tab.sort_values("episode_id", key=lambda c: c.map(order))
    tab.to_csv(BUILD_CSV, index=False)
    print(f"[cfsbase] wrote {BUILD_CSV} ({len(tab)} cases)")
    short = tab[tab.n_members < MIN_MEMBERS]
    if len(short):
        print(f"[cfsbase] {len(short)} case(s) under {MIN_MEMBERS} members: "
              f"{', '.join(short.episode_id)}")
    return tab


# --------------------------------------------------------------------------- #
# Stage: hind - the same lagged ensemble in the other years
# --------------------------------------------------------------------------- #
def hind_path(eid: str, year: int) -> Path:
    return HIND_ROOT / f"{eid}_{year}.nc"


def hind_jobs(cases: list[str] | None = None) -> list[tuple[str, int, pd.Timestamp, pd.Timestamp]]:
    """`(case, year, init, peak)` for every other year whose window ERA5 covers."""
    df = aprep.episodes()
    if cases:
        df = df[df.episode_id.isin(cases)]
    jobs = []
    for r in df.itertuples():
        peak = pd.Timestamp(r.peak)
        init = peak - pd.Timedelta(days=ccfg.LEAD_DAYS)
        for y in HIND_YEARS:
            if y == peak.year:
                continue
            p = peak + pd.DateOffset(years=y - peak.year)   # Feb 29 -> Feb 28
            i = p - (peak - init)
            if ERA5_FIRST_PEAK <= p <= ERA5_LAST_PEAK:
                jobs.append((r.episode_id, y, i, p))
    return jobs


def lagged_cube(init: pd.Timestamp, peak: pd.Timestamp, work: Path) -> xr.Dataset:
    """The 16-member trailing CFS ensemble for an arbitrary (init, peak), in memory.

    The same steps as `aires.cfs.build`, which is keyed by a registered event and so
    cannot take a shifted date: fetch each cycle's prefix, regrid, trim to the common
    `[init + 6 h, peak]` axis, stack on `member`.
    """
    lat, lon = cfs.target_grid()
    parts = []
    for cyc in cfs.cycles_for(init, N_CYCLES, "trailing"):
        grb = work / f"tmp2m.{cyc:%Y%m%d%H}.grb2"
        if not grb.exists():
            max_h = int(round((peak - cyc) / pd.Timedelta(hours=1)))
            try:
                cfs.fetch_window(cyc, "tmp2m", max_h, grb)
            except cfs.MissingCycle:
                print(f"    {cyc:%Y-%m-%d %HZ} ABSENT, skipping", flush=True)
                continue
        c = cfs.grib_to_cube(grb, "t2m_anom", lat, lon)
        c = c.sel(time=slice(init + pd.Timedelta(hours=cfs.CYCLE_H), peak))
        AI.check_window(c, peak, "t2m_anom", where=f"CFS {cyc:%Y-%m-%d %HZ}")
        parts.append(c.expand_dims(member=[len(parts)]))
    if len(parts) < 2:
        raise RuntimeError(f"only {len(parts)} CFS cycles for init {init}")
    return xr.concat(parts, dim="member", join="exact")


def era5_window(peak: pd.Timestamp) -> xr.DataArray:
    """ERA5 12-hourly anomaly frames [peak-6.5d, peak] from the index cube of peak's year.

    One frame more than the 13 of `A_L`: day 0 of the daily pairs ends at peak-6d and
    needs the 12Z frame before it. The 7-day mean is the LAST 13 frames.
    """
    lo = peak - pd.Timedelta(hours=156)
    parts = []
    for y in sorted({lo.year, peak.year}):
        p = ccfg.ACAL_ROOT / "index" / f"era5_t2m_anom_12h_{y}.nc"
        if p.exists():
            with xr.open_dataset(p) as d:
                parts.append(d["t2m_anom"].sel(time=slice(lo, peak)).load())
    w = xr.concat(parts, dim="time")
    t = pd.DatetimeIndex(w["time"].values)
    w = w.isel(time=~t.duplicated(keep="first")).sortby("time")
    if w.sizes["time"] != WINDOW_FRAMES + 1:
        raise RuntimeError(f"ERA5 window for {peak} has {w.sizes['time']} frames, "
                           f"want {WINDOW_FRAMES + 1}")
    return w


def daily_pairs(inst: xr.DataArray, peak: pd.Timestamp) -> xr.DataArray:
    """(…, day=7, lat, lon) mean of the 00Z frame and the 12Z before it, day 6 = peak."""
    ends = pd.date_range(peak - pd.Timedelta(days=6), peak, freq="D")
    d = [inst.sel(time=[e - pd.Timedelta(hours=12), e]).mean("time") for e in ends]
    return xr.concat(d, dim="day").assign_coords(day=np.arange(7))


def hind_one(job) -> str:
    eid, year, init, peak = job
    out = hind_path(eid, year)
    if out.exists():
        return f"{eid} {year}: cached"
    work = CFS_ROOT / ".grib" / f"{eid}_{year}"
    try:
        cube = lagged_cube(init, peak, work)
        f = AI._squeeze(AI.field(cube, peak, "t2m_anom"))            # (member, lat, lon)
        inst = AI._squeeze(AI.instantaneous_field(cube, "t2m_anom"))
        era = era5_window(peak)
        e7 = era.isel(time=slice(1, None)).mean("time")
        al_cfs = AI.area_mean(f).values.astype(float)
        al_era = float(AI.area_mean(e7))
        ds = xr.Dataset(
            dict(al_cfs=("member", al_cfs), al_era5=al_era,
                 bias7=(("lat", "lon"), (f.mean("member") - e7).values.astype("float32")),
                 bias_daily=(("day", "lat", "lon"),
                             (daily_pairs(inst.mean("member"), peak)
                              - daily_pairs(era, peak)).values.astype("float32"))),
            coords=dict(member=np.arange(al_cfs.size), day=np.arange(7),
                        lat=e7["lat"].values, lon=e7["lon"].values),
            attrs=dict(case=eid, year=year, init=str(init), peak=str(peak)))
        HIND_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp)
        tmp.replace(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return (f"{eid} {year}: n={al_cfs.size} CFS {al_cfs.mean():+.2f} ERA5 {al_era:+.2f} "
            f"bias {al_cfs.mean() - al_era:+.2f}")


def hind(cases: list[str] | None = None, workers: int = 4) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = hind_jobs(cases)
    todo = [j for j in jobs if not hind_path(j[0], j[1]).exists()]
    print(f"[cfsbase] hind: {len(jobs)} (case, year) jobs, {len(todo)} to run, "
          f"{workers} workers", flush=True)
    fails = []
    with ProcessPoolExecutor(workers) as ex:
        fut = {ex.submit(hind_one, j): j for j in todo}
        for k, f in enumerate(as_completed(fut), 1):
            try:
                print(f"  [{k}/{len(todo)}] {f.result()}", flush=True)
            except Exception as e:                       # noqa: BLE001 - report, keep going
                j = fut[f]
                fails.append(j)
                print(f"  [{k}/{len(todo)}] {j[0]} {j[1]}: FAILED {e!r}", flush=True)
    if fails:
        raise SystemExit(f"[cfsbase] {len(fails)} hindcast job(s) failed; rerun to retry")


# --------------------------------------------------------------------------- #
# Stage: bias - average the hindcasts per case
# --------------------------------------------------------------------------- #
def bias() -> Path:
    df = aprep.episodes()
    want = {}
    for eid, y, _i, _p in hind_jobs():
        want.setdefault(eid, []).append(y)
    rows, b7, bd = [], [], []
    for eid in df.episode_id:
        ds = [xr.open_dataset(hind_path(eid, y)) for y in want.get(eid, [])
              if hind_path(eid, y).exists()]
        if len(ds) < MIN_YEARS:
            raise SystemExit(f"[cfsbase] {eid}: {len(ds)} hindcast years, need {MIN_YEARS}")
        d = [float(x.al_cfs.mean() - x.al_era5) for x in ds]
        rows.append(dict(episode_id=eid, n_years=len(ds),
                         years=",".join(str(x.attrs["year"]) for x in ds),
                         bias_conus=float(np.mean(d)), bias_sd=float(np.std(d, ddof=1)),
                         spread_hind=float(np.mean([x.al_cfs.std(ddof=1) for x in ds]))))
        b7.append(xr.concat([x.bias7 for x in ds], "year").mean("year"))
        bd.append(xr.concat([x.bias_daily for x in ds], "year").mean("year"))
        for x in ds:
            x.close()
    tab = pd.DataFrame(rows)
    out = xr.Dataset(
        dict(bias_conus=("case", tab.bias_conus.values),
             bias_sd=("case", tab.bias_sd.values),
             n_years=("case", tab.n_years.values),
             bias7=xr.concat(b7, "case"), bias_daily=xr.concat(bd, "case")),
        coords=dict(case=tab.episode_id.values,
                    family=("case", df.family.values)),
        attrs=dict(note="CFS member-mean minus ERA5, averaged over the case's other "
                        "years (leave-one-year-out). Subtract from CFS to correct."))
    out.to_netcdf(BIAS_NC)
    tab.to_csv(CFS_ROOT / "bias.csv", index=False)
    print(tab.round(3).to_string(index=False))
    for fam, g in tab.merge(df[["episode_id", "family"]]).groupby("family"):
        print(f"[cfsbase] {fam}: bias_conus mean {g.bias_conus.mean():+.3f} K, "
              f"range [{g.bias_conus.min():+.2f}, {g.bias_conus.max():+.2f}]")
    print(f"[cfsbase] wrote {BIAS_NC}")
    return BIAS_NC


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("build", "hind", "bias"), default="build")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--case", action="append", help="episode id; repeatable")
    p.add_argument("--force", action="store_true", help="re-download and rebuild")
    p.add_argument("--keep-grib", action="store_true")
    a = p.parse_args(argv)
    if a.stage == "build":
        build(a.case, a.force, a.keep_grib)
    elif a.stage == "hind":
        hind(a.case, a.workers)
    elif a.stage == "bias":
        bias()
    return 0


if __name__ == "__main__":
    sys.exit(main())
