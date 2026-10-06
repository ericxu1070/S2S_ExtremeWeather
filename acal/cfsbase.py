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
    python -m acal.cfsbase --stage score              # cfs_scorecard.csv, cfs_summary.json,
                                                      #   figures/acal/acal_cfs_scorecard.png
    python -m acal.cfsbase --stage paired             # cfs_paired.csv, cfs_summary.json["paired"],
                                                      #   figures/acal/acal_cfs_paired.png
    python -m acal.cfsbase --stage all                # score + paired (CPU, seconds)

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

Scoring CFS against AI+RES (stages `score` and `paired`)
--------------------------------------------------------
Five CFS variants per case, named `<correction>_<estimator>`:

    raw_emp    16 members, ERA5-clim anomaly as is, P = mean(s*A_i >= s*a)   HEADLINE
    raw_gauss  same members, P = 1 - Phi((s*a - s*mean) / sd), sd with ddof=1
    corr_emp   members minus the leave-one-year-out CONUS bias, empirical
    corr_gauss corrected members, Gaussian
    sub_emp    the last 4 members only, raw, empirical (the earlier aires CFS table)

The empirical rule is the SAME `>=` tail-signed rule `analyze.Case.p_raw` uses, so a member
sitting exactly on the threshold counts, for both forecasts. The headline is the raw
variant because the walkers are scored raw too (same ERA5-1990-2019 anomaly, drift inside
the score); the corrected variants bound how much of any gap is CFS drift. Lift and the
PIT use the SAME climatology pool as `analyze.scorecard` (`clim_pool`, `p_clim`, `_lift`,
plus the conservative P_clim = 0 -> 1/n_clim lift). The PIT is `mean(s*A_i < s*obs)`, so
the empirical PIT is 1 - P(obs), as for AI+RES self-normalized.

The head-to-head uses AI+RES's SELF-NORMALIZED `p_sn` (a proper probability; the raw
estimate can exceed 1) against each CFS variant:

  * Brier at 2, 3 and 4 K, tail-signed: forecast P(s*A >= a), outcome o = 1[s*obs >= a].
    At 2 K every case has o = 1 by the slate's selection, so the 2 K Brier only measures
    the mass put on the observed tail. It is reported, but it is not a skill score.
  * log ratio log(P_RES(obs) / P_CFS(obs)), each side floored at 1/(N+1): N = 32 for
    AI+RES, 16 for CFS (also the Gaussian variants, same members), 4 for the subset. The
    floor keeps a zero (no member reached obs) from sending the ratio to +-inf, and it is
    the smallest probability the ensemble can resolve.
  * Win = AI+RES better (lower Brier, larger P(obs)); ties are counted apart. Wilcoxon
    signed-rank (zeros dropped) and a case-bootstrap 90% CI of the mean difference, fixed
    seed. Brier differences are CFS minus AI+RES, so positive = AI+RES better throughout.

SELECTION CAVEAT: every case has |obs| >= 2 K by construction. The head-to-head says which
forecast put more mass on what happened on this slate, NOT which is calibrated, and it is
silent on false alarms. The cases are not independent draws (one winter season shares a
circulation regime), so the bootstrap CI is optimistic.
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

from acal import analyze as AN
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


# --------------------------------------------------------------------------- #
# Stage: score - P_CFS(obs), lift and PIT per case and variant
# --------------------------------------------------------------------------- #
ANALYSIS = AN.OUT
SCORE_CSV = ANALYSIS / "cfs_scorecard.csv"
PAIRED_CSV = ANALYSIS / "cfs_paired.csv"
SUMMARY_JSON = ANALYSIS / "cfs_summary.json"
VARIANTS = ("raw_emp", "raw_gauss", "corr_emp", "corr_gauss", "sub_emp")
HEADLINE = "raw_emp"
BRIER_K = (2.0, 3.0, 4.0)
N_BOOT = 5000
BOOT_SEED = 20261006
CI = 0.90
N_RES = AN.N_WALKERS


def members(al, bias: float, variant: str) -> np.ndarray:
    """The ensemble a variant scores: corrected = minus bias, `sub` = the last 4 members."""
    a = np.asarray(al, dtype=float)
    if variant.startswith("corr"):
        a = a - bias
    if variant.startswith("sub"):
        a = a[-N_SUBSET:]
    return a


def n_floor(variant: str) -> int:
    return N_SUBSET if variant.startswith("sub") else N_CYCLES


def cfs_prob(al, a: float, sign: float, kind: str) -> float:
    """P(s*A >= s*a) from the members: empirical fraction or a Gaussian fit.

    The empirical rule is the `>=` tail-signed rule AI+RES uses (`aceiling.beyond`). The
    Gaussian uses the sample sd (ddof=1); a zero sd degenerates to the empirical step
    instead of dividing by zero.
    """
    from scipy.stats import norm
    al = np.asarray(al, dtype=float)
    emp = float(np.mean(sign * (al - a) >= 0.0))
    if kind == "emp":
        return emp
    sd = float(al.std(ddof=1))
    if sd <= 0.0:
        return emp
    return float(norm.sf((sign * a - sign * al.mean()) / sd))


def cfs_pit(al, obs: float, sign: float, kind: str) -> float:
    """Tail-signed forecast CDF at obs: `P(s*A < s*obs)` empirically, 1 - P(obs) Gaussian."""
    al = np.asarray(al, dtype=float)
    if kind == "emp":
        return float(np.mean(sign * al < sign * obs))
    return 1.0 - cfs_prob(al, obs, sign, "gauss")


def load_cfs() -> pd.DataFrame:
    """One row per case: the 16-member record joined to the bias table, slate order."""
    df = aprep.episodes()
    bias_tab = pd.read_csv(CFS_ROOT / "bias.csv").set_index("episode_id")
    rows = []
    for r in df.itertuples():
        p = json_path(r.episode_id)
        if not p.exists():
            raise SystemExit(f"[cfsbase] missing {p}; run --stage build")
        rec = json.loads(p.read_text())
        if r.episode_id not in bias_tab.index:
            raise SystemExit(f"[cfsbase] {r.episode_id} not in bias.csv; run --stage bias")
        want = 1.0 if r.family == "heat" else -1.0
        if rec["sign"] != want or abs(rec["obs"] - r.a_l_conus) > AN.CATALOG_TOL:
            raise SystemExit(f"[cfsbase] {r.episode_id}: json sign/obs disagree with the slate")
        rows.append(dict(episode_id=r.episode_id, family=r.family, rung=int(r.rung),
                         peak=r.peak, obs=rec["obs"], sign=rec["sign"], al=rec["al"],
                         bias=float(bias_tab.loc[r.episode_id, "bias_conus"])))
    return pd.DataFrame(rows)


def score_cfs_case(row, daily: pd.Series) -> dict:
    """One cfs_scorecard row. Pure in the record and the daily series (tested)."""
    pool = AN.clim_pool(daily, row["peak"])
    pc, kc, nc = AN.p_clim(pool, row["obs"], row["sign"])
    out = dict(episode_id=row["episode_id"], family=row["family"], rung=row["rung"],
               peak=row["peak"], obs=row["obs"], tail_sign=row["sign"],
               n_members=len(row["al"]), bias_conus=row["bias"],
               cfs_mean=float(np.mean(row["al"])), cfs_sd=float(np.std(row["al"], ddof=1)),
               p_clim_obs=pc, k_clim_obs=kc, n_clim=nc)
    for v in VARIANTS:
        al, kind = members(row["al"], row["bias"], v), v.split("_")[1]
        p = cfs_prob(al, row["obs"], row["sign"], kind)
        out[f"p_obs_{v}"] = p
        out[f"lift_{v}"] = AN._lift(p, pc)
        out[f"lift_cons_{v}"] = p / (max(kc, 1) / nc)
        out[f"pit_{v}"] = cfs_pit(al, row["obs"], row["sign"], kind)
        out[f"n_reach_{v}"] = int(np.sum(row["sign"] * (al - row["obs"]) >= 0.0))
    return out


def _summ(sc: pd.DataFrame, v: str) -> dict:
    keys = (f"p_obs_{v}", f"lift_{v}", f"lift_cons_{v}", f"pit_{v}", f"n_reach_{v}")

    def q(d):
        return {k.replace(f"_{v}", ""): AN._q(d[k]) for k in keys}
    heat, cold = sc.family == "heat", sc.family == "cold"
    gt = sc[f"lift_cons_{v}"] > 1
    return dict(overall=q(sc),
                by_rung={int(g): q(d) for g, d in sc.groupby("rung")},
                by_family={f: q(d) for f, d in sc.groupby("family")},
                n_zero_obs=int((sc[f"p_obs_{v}"] == 0).sum()),
                n_lift_gt1=int((sc[f"lift_{v}"] > 1).sum()),
                n_lift_defined=int(np.isfinite(sc[f"lift_{v}"]).sum()),
                n_lift_cons_gt1=int(gt.sum()),
                n_lift_cons_gt1_heat=int((gt & heat).sum()),
                n_lift_cons_gt1_cold=int((gt & cold).sum()),
                n_pit_ge_0p9=int((sc[f"pit_{v}"] >= 0.9).sum()))


def _merge_summary(update: dict) -> None:
    old = json.loads(SUMMARY_JSON.read_text()) if SUMMARY_JSON.exists() else {}
    old.update(update)
    SUMMARY_JSON.write_text(json.dumps(old, indent=2))


def score() -> pd.DataFrame:
    cfs_df, daily = load_cfs(), AN.load_daily()
    sc = pd.DataFrame([score_cfs_case(r, daily) for _, r in cfs_df.iterrows()])
    # AI+RES beside it, from the existing scorecard, so the two read off one table.
    res = pd.read_csv(AN.SCORE_OUT).set_index("episode_id")
    for c in ("p_obs_raw", "p_obs_sn", "lift_obs_raw", "lift_obs_sn", "lift_obs_raw_cons",
              "pit_sn", "n_beyond_obs"):
        sc[f"res_{c}"] = sc.episode_id.map(res[c])
    if not np.allclose(sc.p_clim_obs, sc.episode_id.map(res.p_clim_obs)):
        raise SystemExit("[cfsbase] P_clim differs from analyze scorecard - pool mismatch")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    sc.to_csv(SCORE_CSV, index=False, float_format="%.6g")
    summ = dict(
        n_cases=int(len(sc)),
        caveat="All 42 cases are selected on |obs| >= 2 K: P(obs) and lift are mass on the "
               "observed tail, not calibrated probabilities. Headline = raw_emp.",
        ai_res={k: AN._q(sc[f"res_{k}"]) for k in
                ("p_obs_raw", "p_obs_sn", "lift_obs_raw", "lift_obs_sn",
                 "lift_obs_raw_cons", "pit_sn")},
        p_clim_obs=AN._q(sc.p_clim_obs),
        variants={v: _summ(sc, v) for v in VARIANTS})
    _merge_summary(summ)
    print(f"[cfsbase] {len(sc)} cases -> {SCORE_CSV}")
    r = summ["ai_res"]["p_obs_sn"]
    print(f"  AI+RES     P(obs) sn  median {r['median']:.3f} [{r['q25']:.3f}, {r['q75']:.3f}]")
    for v in VARIANTS:
        s = summ["variants"][v]
        q, ll = s["overall"]["p_obs"], s["overall"]["lift"]
        print(f"  {v:10s} P(obs) median {q['median']:.3f} [{q['q25']:.3f}, {q['q75']:.3f}]"
              f"  lift {ll['median']:.2f} (n={ll['n']}, >1 {s['n_lift_gt1']})"
              f"  lift_cons>1 {s['n_lift_cons_gt1']}/42  P=0: {s['n_zero_obs']}")
    fig_scorecard(sc)
    return sc


# --------------------------------------------------------------------------- #
# Stage: paired - AI+RES vs CFS case by case
# --------------------------------------------------------------------------- #
def brier(p: float, o: float) -> float:
    return float((p - o) ** 2)


def log_ratio(p_res: float, p_cfs: float, n_cfs: int, n_res: int = N_RES) -> float:
    """log(P_RES / P_CFS) with each side floored at its own 1/(N+1)."""
    return float(np.log(max(p_res, 1.0 / (n_res + 1)) / max(p_cfs, 1.0 / (n_cfs + 1))))


def boot_ci(d, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = CI):
    """Case-bootstrap CI of the mean of `d` (resample cases with replacement)."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    lo, hi = np.percentile(m, [50 * (1 - level), 100 - 50 * (1 - level)])
    return (float(lo), float(hi))


def paired_stats(d) -> dict:
    """Mean, bootstrap CI, wins/ties/losses and Wilcoxon p of `d` (positive = AI+RES better)."""
    from scipy.stats import wilcoxon
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    out = dict(n=int(d.size), mean=float(d.mean()) if d.size else np.nan,
               median=float(np.median(d)) if d.size else np.nan,
               win=int((d > 0).sum()), tie=int((d == 0).sum()), loss=int((d < 0).sum()))
    out["ci_lo"], out["ci_hi"] = boot_ci(d)
    nz = d[d != 0]
    try:
        out["wilcoxon_p"] = float(wilcoxon(nz).pvalue) if nz.size >= 1 else np.nan
    except ValueError:
        out["wilcoxon_p"] = np.nan
    return out


def paired_case(row, res_case) -> dict:
    """Per-case paired record. `row` = CFS record, `res_case` = `analyze.Case`."""
    s, obs = row["sign"], row["obs"]
    out = dict(episode_id=row["episode_id"], family=row["family"], rung=row["rung"], obs=obs)
    p_res_obs = res_case.p_sn(obs)
    out["p_res_obs"] = p_res_obs
    res_p = {k: res_case.p_sn(s * k) for k in BRIER_K}
    o = {k: float(s * obs >= k) for k in BRIER_K}
    for k in BRIER_K:
        out[f"o_{k:g}K"] = o[k]
        out[f"brier_res_{k:g}K"] = brier(res_p[k], o[k])
    for v in VARIANTS:
        al, kind = members(row["al"], row["bias"], v), v.split("_")[1]
        p_obs = cfs_prob(al, obs, s, kind)
        out[f"p_cfs_obs_{v}"] = p_obs
        out[f"logratio_{v}"] = log_ratio(p_res_obs, p_obs, n_floor(v))
        for k in BRIER_K:
            b = brier(cfs_prob(al, s * k, s, kind), o[k])
            out[f"brier_{v}_{k:g}K"] = b
            out[f"dbrier_{v}_{k:g}K"] = b - out[f"brier_res_{k:g}K"]   # > 0: AI+RES better
    return out


def paired() -> pd.DataFrame:
    cfs_df = load_cfs()
    _, cases = AN.load_all()
    byid = {c.episode_id: c for c in cases}
    pc = pd.DataFrame([paired_case(r, byid[r["episode_id"]]) for _, r in cfs_df.iterrows()])
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pc.to_csv(PAIRED_CSV, index=False, float_format="%.6g")

    groups = {"all": pc, "heat": pc[pc.family == "heat"], "cold": pc[pc.family == "cold"],
              **{f"rung{g}": d for g, d in pc.groupby("rung")}}
    summ = {}
    for v in VARIANTS:
        summ[v] = {}
        for name, d in groups.items():
            summ[v][name] = {"logratio": paired_stats(d[f"logratio_{v}"])}
            for k in BRIER_K:
                st = paired_stats(d[f"dbrier_{v}_{k:g}K"])
                st["mean_brier_res"] = float(d[f"brier_res_{k:g}K"].mean())
                st["mean_brier_cfs"] = float(d[f"brier_{v}_{k:g}K"].mean())
                if k == 2.0:
                    st["note"] = ("o = 1 for every case by selection: mass on the observed "
                                  "tail, not skill")
                summ[v][name][f"brier_{k:g}K"] = st
    n3, n4 = int(pc["o_3K"].sum()), int(pc["o_4K"].sum())
    _merge_summary(dict(paired=dict(
        convention="logratio = log(P_RES/P_CFS), floors 1/33 (AI+RES) and 1/(N+1) (CFS); "
                   "brier diff = Brier_CFS - Brier_RES; positive = AI+RES better; AI+RES "
                   "probability = self-normalized; win = diff > 0, ties apart; bootstrap "
                   f"{int(CI * 100)}% CI of the mean over cases (n_boot={N_BOOT}, "
                   f"seed={BOOT_SEED}).",
        caveat="Selected on outcome (|obs| >= 2 K): which forecast put more mass on what "
               "happened, not calibration. Cases share seasons, so CIs are optimistic.",
        n_obs_ge_3K=n3, n_obs_ge_4K=n4, variants=summ)))
    print(f"[cfsbase] {len(pc)} cases -> {PAIRED_CSV}  (o=1 at 3 K: {n3}, at 4 K: {n4})")
    for v in VARIANTS:
        for name in ("all", "heat", "cold"):
            g = summ[v][name]
            parts = []
            for m, lab in (("logratio", "logR"), ("brier_3K", "dB3"), ("brier_4K", "dB4")):
                st = g[m]
                parts.append(f"{lab} {st['mean']:+.3f} W/T/L {st['win']}/{st['tie']}/"
                             f"{st['loss']} p={st['wilcoxon_p']:.3f}")
            print(f"  {v:10s} {name:5s} " + " | ".join(parts))
    fig_paired(pc, summ)
    return pc


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
VARIANT_LABEL = {"raw_emp": "raw, empirical (headline)", "raw_gauss": "raw, Gaussian",
                 "corr_emp": "bias-corrected, empirical",
                 "corr_gauss": "bias-corrected, Gaussian",
                 "sub_emp": "4 members, raw, empirical"}


def fig_scorecard(sc: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    AN._style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    fam_c, mk = AN.FAMILY_COLOR, AN.RUNG_MARKER
    pos = np.concatenate([sc[c][sc[c] > 0].values
                          for c in ("p_obs_raw_emp", "p_obs_corr_emp", "res_p_obs_sn")])
    floor = pos.min() / 3

    def fl(a):                       # zeros ride on a shaded band below the data
        return np.where(np.asarray(a) > 0, a, floor)

    # panel 1: probability of the observed tail
    ax = axes[0]
    lo, top = floor / 1.7, 1.5
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, top); ax.set_ylim(lo, top)
    ax.axhspan(lo, floor * 1.7, color="0.93", zorder=0)
    ax.axvspan(lo, floor * 1.7, color="0.93", zorder=0)
    ticks = [t for t in (1e-2, 1e-1, 1) if t > floor * 2]
    lab = ["0"] + [f"$10^{{{int(np.log10(t))}}}$" for t in ticks]
    ax.set_xticks([floor] + ticks, lab); ax.set_yticks([floor] + ticks, lab)
    ax.plot([lo, top], [lo, top], color="0.6", lw=1, ls="--", zorder=1)
    ax.text(top / 1.1, top / 1.5, "1:1", ha="right", va="top", color="0.45", fontsize=8)
    for r in sc.itertuples():
        y = float(fl(r.res_p_obs_sn))
        x0, x1 = float(fl(r.p_obs_raw_emp)), float(fl(r.p_obs_corr_emp))
        ax.plot([x0, x1], [y, y], color=fam_c[r.family], lw=0.6, alpha=0.35, zorder=2)
        ax.scatter(x1, y, s=22, marker=mk[r.rung], facecolor="none",
                   edgecolor=fam_c[r.family], linewidth=0.7, alpha=0.55, zorder=2)
        ax.scatter(x0, y, s=46, marker=mk[r.rung], color=fam_c[r.family],
                   edgecolor="white", linewidth=0.8, alpha=0.92, zorder=3)
    x, y = sc.p_obs_raw_emp.values, sc.res_p_obs_sn.values
    ax.text(0.97, 0.03,
            f"AI+RES above 1:1: {int((y > x).sum())}   below: {int((y < x).sum())}   "
            f"equal: {int((y == x).sum())}\n"
            f"CFS P = 0: {int((x == 0).sum())}   AI+RES P = 0: {int((y == 0).sum())}\n"
            "faint open marker: bias-corrected CFS",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.25",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_xlabel("CFS P(obs), 16 members, raw empirical")
    ax.set_ylabel("AI+RES P(obs), self-normalized")
    ax.set_title("probability of the observed tail", fontsize=9.5)
    ax.minorticks_off()

    # panel 2: lift over the shared climatology (conservative, defined for all 42)
    ax = axes[1]
    x, y = sc.lift_cons_raw_emp.values, sc.res_lift_obs_raw_cons.values
    lo2, top2 = 0.04, max(x.max(), y.max()) * 1.6
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo2, top2); ax.set_ylim(lo2, top2)
    ax.plot([lo2, top2], [lo2, top2], color="0.6", lw=1, ls="--", zorder=1)
    ax.axvline(1, color="0.85", lw=0.8, zorder=0); ax.axhline(1, color="0.85", lw=0.8, zorder=0)
    ax.text(top2 / 1.1, top2 / 1.5, "1:1", ha="right", va="top", color="0.45", fontsize=8)
    for r in sc.itertuples():
        ax.scatter(max(r.lift_cons_raw_emp, lo2 * 1.3), max(r.res_lift_obs_raw_cons, lo2 * 1.3),
                   s=46, marker=mk[r.rung], color=fam_c[r.family], edgecolor="white",
                   linewidth=0.8, alpha=0.92, zorder=3)
    ax.text(0.03, 0.97,
            f"AI+RES above 1:1: {int((y > x).sum())}   below: {int((y < x).sum())}\n"
            f"lift > 1: AI+RES {int((y > 1).sum())}/42, CFS {int((x > 1).sum())}/42\n"
            f"median lift: AI+RES {np.median(y):.2f}, CFS {np.median(x):.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="0.25",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_xlabel("CFS lift, raw empirical")
    ax.set_ylabel("AI+RES lift, raw")
    ax.set_title("lift over climatology (P_clim = 0 counted as 1/284)", fontsize=9.5)
    ax.minorticks_off()

    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=fam_c[f], label=f)
               for f in ("heat", "cold")]
    handles += [Line2D([], [], ls="none", marker=mk[g], ms=7, color="0.45", label=f"rung {g} K")
                for g in (2, 3, 4)]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("AI+RES vs CFSv2 (16 lagged members), 21 d lead, 42 cases "
                 "(selected on outcome)", y=1.07, fontsize=10.5)
    fig.tight_layout()
    p = AN._save(fig, "acal_cfs_scorecard.png")
    plt.close(fig)
    return p


def fig_paired(pc: pd.DataFrame, summ: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    AN._style(plt)
    fam_c, mk = AN.FAMILY_COLOR, AN.RUNG_MARKER
    n3, n4 = int(pc["o_3K"].sum()), int(pc["o_4K"].sum())
    panels = (("logratio", "logratio_%s", "log( P_RES / P_CFS ) at the observed value"),
              ("brier_3K", "dbrier_%s_3K", f"Brier(CFS) - Brier(AI+RES), 3 K  ({n3} cases o=1)"),
              ("brier_4K", "dbrier_%s_4K", f"Brier(CFS) - Brier(AI+RES), 4 K  ({n4} cases o=1)"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), sharey=True)
    rng = np.random.default_rng(1)
    nv = len(VARIANTS)
    for ax, (key, col, ttl) in zip(axes, panels):
        jit = rng.uniform(-0.28, 0.28, len(pc))
        c = col % HEADLINE
        for r, j in zip(pc.itertuples(), jit):
            ax.scatter(getattr(r, c), 1.0 + j, s=30, marker=mk[r.rung],
                       color=fam_c[r.family], edgecolor="white", linewidth=0.6,
                       alpha=0.85, zorder=3)
        ax.axvline(0, color="0.3", lw=1, zorder=1)
        for i, v in enumerate(VARIANTS):
            st = summ[v]["all"][key]
            y = -i
            col_ = "0.15" if v == HEADLINE else "0.55"
            ax.plot([st["ci_lo"], st["ci_hi"]], [y, y], color=col_,
                    lw=2.2 if v == HEADLINE else 1.4, zorder=3)
            ax.scatter(st["mean"], y, s=40, marker="D", color=col_, zorder=4)
            ax.text(1.02, y, f"{st['win']}-{st['tie']}-{st['loss']}  p={st['wilcoxon_p']:.2f}",
                    transform=ax.get_yaxis_transform(), fontsize=7.5, va="center",
                    color="0.2" if v == HEADLINE else "0.45")
        ax.axhline(0.45, color="0.8", lw=0.8)
        ax.set_ylim(-nv + 0.4, 1.45)
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel("positive = AI+RES better")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks([1.0] + [-i for i in range(nv)],
                       ["cases"] + [VARIANT_LABEL[v] for v in VARIANTS], fontsize=8)
    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=fam_c[f], label=f)
               for f in ("heat", "cold")]
    handles += [Line2D([], [], ls="none", marker=mk[g], ms=7, color="0.45", label=f"rung {g} K")
                for g in (2, 3, 4)]
    handles.append(Line2D([], [], ls="none", marker="D", ms=6, color="0.3",
                          label="mean, 90% case-bootstrap CI"))
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Paired, per case, 42 cases selected on outcome. Text at right: AI+RES "
                 "wins - ties - losses, Wilcoxon p (all cases).", y=1.075, fontsize=9.5)
    fig.subplots_adjust(right=0.9, wspace=0.6)
    p = AN._save(fig, "acal_cfs_paired.png")
    plt.close(fig)
    return p


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("build", "hind", "bias", "score", "paired", "all"), default="build")
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
    if a.stage in ("score", "all"):
        score()
    if a.stage in ("paired", "all"):
        paired()
    return 0


if __name__ == "__main__":
    sys.exit(main())
