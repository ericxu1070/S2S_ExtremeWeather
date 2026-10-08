#!/usr/bin/env python
"""Closest-member comparison, AI+RES against EVERY baseline, raw and bias corrected. CPU ONLY.

Multi-model version of `acal.sidebyside` stage `members` (that one is AI+RES vs CFSv2 raw
only). Closeness is the same: cos(lat)-weighted RMSE of the 7-day-mean T2m anomaly map over
CONUS land (`maps.land_mask`) against ERA5, and the pattern correlation `land_corr`. Each
baseline is scored on ITS OWN window (`Source.obs_window`): 13 frames for CFSv2 / GEFSv12,
12 frames (6 UTC-day means) for GEPS / EC46, with AI+RES re-reduced on the same frames
(`s2sbase.aires_fields`) and ERA5 truth on the same frames.

Fairness. Ensemble sizes differ (AI+RES 32, CFSv2 16, GEFSv12 31, GEPS 21, EC46 51 or 101)
and the closest of more draws is closer by chance alone. Every forecast is therefore
compared by its EXPECTED best of 16 (`sidebyside.expected_min`, exact order statistics over
all members), which needs every member's RMSE. The "bias corrected" baseline subtracts that
case's `bias7` field (the source's bias.nc, as `maps.source_dataset` does for `prob_corr`)
from every member; AI+RES is never bias corrected. Ensemble means: AI+RES importance
weighted, baselines equal weight. Cloned walkers share their path, so AI+RES's E[best of
16] is if anything optimistic only in the count sense; the draws are not independent.

Outputs (runs/acal/analysis/multi/, resumable per source per case, `--force` to redo):
  cache/<source>/<case>.npz      one case: per-member RMSE/r/A_L raw+corrected, closest
                                 raw member map, raw and corrected mean map
  cache/aires_<window>/<case>.npz  the same for AI+RES walkers on a window
  member_rmse_<source>.csv       one row per case x member (rmse_raw, rmse_corr, corr_raw,
                                 corr_corr, al_raw, al_corr, lead_days); also
                                 member_rmse_aires_<window>.csv (rmse_raw/corr_raw/al, weight)
  member_maps_<source>.nc        closest raw member / raw mean / corrected mean maps
  closest_members_multi.csv      per case per forecast: best, E[best of 16], median, mean
Figures (figures/acal/s2s/era5/multi/members/):
  closest_member_summary.png     one panel per baseline, AI+RES vs baseline E[best of 16]
                                 RMSE (raw triangles, corrected squares) and a second row
                                 for the ensemble-mean RMSE
  <case>.png (42)                ERA5, AI+RES closest walker, each baseline's closest raw
                                 member; AI+RES and baseline raw ensemble means; baseline
                                 bias-corrected means and the per-forecast numbers

    python -m acal.multi_members --stage rmse      # member pass, cached
    python -m acal.multi_members --stage figures   # redraw from the caches
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import maps as M
from acal import multifig as MF
from acal import s2sbase as S2
from acal import sidebyside as SB
from aires import aindex as AI

CACHE = MF.DATA_DIR / "cache"
SUBSET = SB.SUBSET                         # 16: the smallest baseline ensemble (CFSv2)
VMAX = 20.0                                # shared diverging colour range of the case pages
WINDOWS = ("13f", "12f")
WORKERS = 6


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def e16(rmses) -> float:
    """Expected best of 16 of a member RMSE vector; NaN when fewer than 16 finite members."""
    x = np.asarray(rmses, float)
    x = x[np.isfinite(x)]
    return SB.expected_min(x, SUBSET) if x.size >= SUBSET else float("nan")


def closer_count(x, y) -> tuple[int, int]:
    """(cases where x < y, cases where both are finite): AI+RES closer when x is AI+RES."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return int((x[ok] < y[ok]).sum()), int(ok.sum())


def _stamp(*paths: Path) -> str:
    """Cache key: size + mtime of every input file (a re-scored cube changes it)."""
    h = hashlib.md5()
    for p in paths:
        st = p.stat() if p.exists() else None
        h.update(f"{p.name}:{st.st_size if st else 0}:{st.st_mtime_ns if st else 0}".encode())
    return h.hexdigest()


def _cache_path(tag: str, eid: str) -> Path:
    return CACHE / tag / f"{eid}.npz"


def _fresh(path: Path, stamp: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            return str(z["stamp"]) == stamp
    except Exception:                      # truncated or foreign file: rebuild
        return False


def _write(path: Path, **arrs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp{__import__('os').getpid()}.npz")
    np.savez_compressed(tmp, **arrs)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# The member pass (one worker job = one source x case, or AI+RES x window x case)
# --------------------------------------------------------------------------- #
_CTX: dict = {}


def _ctx() -> dict:
    """Per-process: truth, land weights, cos(lat), truth base. Built once."""
    if not _CTX:
        from acal import truth as TR
        tr = TR.get_truth("era5")
        base = xr.open_dataset(M.truth_base(tr, False)).load()
        _CTX.update(tr=tr, TR=TR, base=base,
                    land=M.land_mask(base).values.astype(float),
                    coslat=np.cos(np.deg2rad(base["lat"].values)))
    return _CTX


def _obs(eid: str, window: str) -> np.ndarray:
    """ERA5 7-day-mean anomaly on `window`: the maps truth for 13f, the 12-frame mean else."""
    c = _ctx()
    if window == "13f":
        return c["base"]["truth"].sel(case=eid).values
    return np.asarray(c["TR"].frames_on(c["tr"], eid, window).mean("time").values)


def _al(F: np.ndarray, c: dict) -> np.ndarray:
    """CONUS-box cos(lat) mean (the A_L index) of each leading-axis field."""
    da = xr.DataArray(F[None] if F.ndim == 2 else F, dims=("m", "lat", "lon"),
                      coords=dict(lat=c["base"]["lat"].values, lon=c["base"]["lon"].values))
    return np.asarray(AI.area_mean(da), "float64")


def _stats(F, obs, c, w=None) -> dict:
    """Per-member RMSE / r / A_L of fields F (member, lat, lon), the closest member by RMSE,
    and the (weighted) mean field with its own RMSE / r."""
    F = np.asarray(F, "float32")
    r = SB.land_rmse(F, obs, c["land"], c["coslat"])
    k = SB.land_corr(F, obs, c["land"], c["coslat"])
    ib = int(np.nanargmin(r)) if np.isfinite(r).any() else -1
    mean = (np.tensordot(w, F, axes=(0, 0)) if w is not None else F.mean(0)).astype("float32")
    mr = SB.land_rmse(mean[None], obs, c["land"], c["coslat"])[0]
    mk = SB.land_corr(mean[None], obs, c["land"], c["coslat"])[0]
    return dict(rmse=r, corr=k, al=_al(F, c), best=ib, mean=mean, mean_rmse=mr, mean_corr=mk)


def source_job(args) -> tuple[str, str, str]:
    """Score every member of one baseline for one case, raw and bias corrected."""
    source, eid, peak, force = args
    src = S2.get(source)
    cube, js = src.cube_path(eid), src.json_path(eid)
    if not (cube.exists() and js.exists()):
        return source, eid, "no cube"
    stamp = _stamp(cube, js, src.bias_nc)
    path = _cache_path(source, eid)
    if _fresh(path, stamp) and not force:
        return source, eid, "cached"
    c = _ctx()
    obs = _obs(eid, src.obs_window)
    C = M.source_member_fields(source, eid, pd.Timestamp(peak), daily=False)
    if C is None:
        return source, eid, "no cube"
    F = C.values
    b = None
    if src.bias_nc.exists():
        with xr.open_dataset(src.bias_nc) as bd:
            if eid in list(map(str, bd["case"].values)):
                b = bd["bias7"].sel(case=eid).values.astype("float32")
    raw = _stats(F, obs, c)
    cor = _stats(F - b[None], obs, c) if b is not None else None
    nan = np.full(F.shape[0], np.nan)
    lead = C["member_lead_days"].values if "member_lead_days" in C.coords else nan
    _write(path, stamp=stamp, n=F.shape[0], lead=np.asarray(lead, float),
           rmse_raw=raw["rmse"], corr_raw=raw["corr"], al_raw=raw["al"],
           rmse_corr=cor["rmse"] if cor else nan, corr_corr=cor["corr"] if cor else nan,
           al_corr=cor["al"] if cor else nan,
           best_raw=raw["best"], best_field=F[raw["best"]],
           mean_raw=raw["mean"], mean_raw_rmse=raw["mean_rmse"], mean_raw_corr=raw["mean_corr"],
           mean_corr=cor["mean"] if cor else np.full(F.shape[1:], np.nan, "float32"),
           mean_corr_rmse=cor["mean_rmse"] if cor else np.nan,
           mean_corr_corr=cor["mean_corr"] if cor else np.nan,
           obs=obs.astype("float32"), obs_al=float(_al(obs, c)[0]))
    return source, eid, "built"


def aires_job(args) -> tuple[str, str, str]:
    """Score every AI+RES walker of one case on one window (importance-weighted mean)."""
    window, eid, force = args
    path = _cache_path(f"aires_{window}", eid)
    stamp = _stamp(S2.aires_fields_path(eid))
    if _fresh(path, stamp) and not force:
        return f"aires_{window}", eid, "cached"
    c = _ctx()
    R, w = S2.aires_fields(eid, window)
    w = w / w.sum()
    obs = _obs(eid, window)
    st = _stats(R.values, obs, c, w)
    _write(path, stamp=stamp, n=R.shape[0], weight=w, rmse_raw=st["rmse"],
           corr_raw=st["corr"], al_raw=st["al"], best_raw=st["best"],
           best_field=R.values[st["best"]].astype("float32"), mean_raw=st["mean"],
           mean_raw_rmse=st["mean_rmse"], mean_raw_corr=st["mean_corr"],
           obs=obs.astype("float32"), obs_al=float(_al(obs, c)[0]))
    return f"aires_{window}", eid, "built"


def _cases() -> tuple[pd.DataFrame, list]:
    df, cases = AN.load_all()
    return df, cases


def member_pass(force: bool = False, workers: int = WORKERS) -> None:
    """Fill the per-case caches (resumable: a cache is rebuilt only when its inputs change
    or with `force`)."""
    df, cases = _cases()
    peaks = {c.episode_id: str(c.run["peak"]) for c in cases}
    jobs = [(s, e, p, force) for s in MF.models() for e, p in peaks.items()]
    ajobs = [(w, e, force) for w in WINDOWS if any(S2.get(s).obs_window == w
                                                   for s in MF.models()) or w == "13f"
             for e in peaks]
    t0 = time.time()
    with ProcessPoolExecutor(min(workers, WORKERS)) as ex:
        futs = [ex.submit(aires_job, j) for j in ajobs] + \
               [ex.submit(source_job, j) for j in jobs]
        for f in futs:
            tag, eid, st = f.result()
            if st != "cached":
                print(f"  {tag:12s} {eid}  {st}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"[multi_members] member pass done in {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------- #
# Aggregation: csv per source, nc maps, closest_members_multi.csv
# --------------------------------------------------------------------------- #
def _load(tag: str, eid: str):
    p = _cache_path(tag, eid)
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def aggregate() -> dict:
    """Read the caches, write the csv/nc tables, return {tag: {eid: dict}} for the figures.
    A source missing from some cases is kept for the cases it has; a source with no cache at
    all is left out with a note."""
    df, cases = _cases()
    eids = [c.episode_id for c in cases]
    fam = dict(zip(eids, df.family))
    MF.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    for tag in [f"aires_{w}" for w in WINDOWS] + MF.models():
        got = {e: z for e in eids if (z := _load(tag, e)) is not None}
        if not got:
            print(f"[multi_members] {tag}: no cached cases, left out")
            continue
        data[tag] = got
        rows = []
        for e, z in got.items():
            for m in range(int(z["n"])):
                r = dict(episode_id=e, member=m, rmse_raw=z["rmse_raw"][m],
                         corr_raw=z["corr_raw"][m], al_raw=z["al_raw"][m])
                if tag.startswith("aires"):
                    r["weight"] = z["weight"][m]
                else:
                    r.update(rmse_corr=z["rmse_corr"][m], corr_corr=z["corr_corr"][m],
                             al_corr=z["al_corr"][m], lead_days=z["lead"][m])
                rows.append(r)
        pd.DataFrame(rows).to_csv(MF.DATA_DIR / f"member_rmse_{tag}.csv", index=False,
                                  float_format="%.6g")
        names = ["best_field", "mean_raw"] + ([] if tag.startswith("aires") else ["mean_corr"])
        ref = next(iter(got.values()))["obs"]
        ds = xr.Dataset({n: (("case", "lat", "lon"), np.stack([z[n] for z in got.values()]))
                         for n in names + ["obs"]},
                        coords=dict(case=list(got), lat=_ctx()["base"]["lat"].values,
                                    lon=_ctx()["base"]["lon"].values))
        ds.attrs["note"] = ("7-day-mean T2m anomaly (K); best_field = closest raw member "
                            "(AI+RES: closest walker); mean_raw / mean_corr = ensemble mean "
                            "(AI+RES importance weighted)")
        ds.to_netcdf(MF.DATA_DIR / f"member_maps_{tag}.nc")
    rows = []
    for e in eids:
        for tag, got in data.items():
            if e not in got:
                continue
            z = got[e]
            aires = tag.startswith("aires")
            rows.append(dict(
                episode_id=e, family=fam[e], forecast=tag,
                window=tag.split("_")[1] if aires else S2.get(tag).obs_window,
                n_members=int(z["n"]),
                best_raw=np.nanmin(z["rmse_raw"]), e16_raw=e16(z["rmse_raw"]),
                median_raw=np.nanmedian(z["rmse_raw"]), mean_raw=float(z["mean_raw_rmse"]),
                best_corr=np.nan if aires else np.nanmin(z["rmse_corr"]),
                e16_corr=np.nan if aires else e16(z["rmse_corr"]),
                median_corr=np.nan if aires else np.nanmedian(z["rmse_corr"]),
                mean_corr=np.nan if aires else float(z["mean_corr_rmse"])))
    out = pd.DataFrame(rows)
    out.to_csv(MF.DATA_DIR / "closest_members_multi.csv", index=False, float_format="%.6g")
    print(f"[multi_members] wrote {MF.DATA_DIR / 'closest_members_multi.csv'} "
          f"({len(out)} rows)")
    return data


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _pair_table(data: dict) -> pd.DataFrame:
    """Per baseline and case: AI+RES (on the baseline's window) vs baseline numbers."""
    rows = []
    for s in MF.models():
        if s not in data:
            continue
        aw = f"aires_{S2.get(s).obs_window}"
        for e, z in data[s].items():
            if e not in data.get(aw, {}):
                continue
            a = data[aw][e]
            rows.append(dict(source=s, episode_id=e,
                             a_e16=e16(a["rmse_raw"]), a_mean=float(a["mean_raw_rmse"]),
                             r_e16=e16(z["rmse_raw"]), c_e16=e16(z["rmse_corr"]),
                             r_mean=float(z["mean_raw_rmse"]),
                             c_mean=float(z["mean_corr_rmse"])))
    return pd.DataFrame(rows)


def summary_figure(data: dict) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    AN._style(plt)
    t = _pair_table(data)
    srcs = [s for s in MF.models() if s in set(t.source)]
    if t.empty:
        print("[multi_members] no data for the summary figure")
        return None
    n = len(srcs)
    fig, axes = plt.subplots(2, n, figsize=(3.7 * n, 8.2), squeeze=False)
    rows = (("a_e16", "r_e16", "c_e16", "closest member, expected best of 16"),
            ("a_mean", "r_mean", "c_mean", "ensemble mean"))
    for i, (xa, yr, yc, what) in enumerate(rows):
        # one limit per row (shared by its panels): closest members sit near 2-8 K, the
        # ensemble means reach ~10 K, and a common 0-11 K box would leave the top row empty
        hi = float(np.ceil(np.nanmax(t[[xa, yr, yc]].values) + 0.4))   # no point on the frame
        for j, s in enumerate(srcs):
            ax, d = axes[i, j], t[t.source == s]
            ax.plot([0, hi], [0, hi], color="0.5", lw=0.8, zorder=1)
            ax.scatter(d[xa], d[yr], marker=MF.MARKER["raw"], s=30, c=MF.color(s, raw=True),
                       edgecolor="white", linewidth=0.4, zorder=3)
            ax.scatter(d[xa], d[yc], marker=MF.MARKER["corr"], s=26, c=MF.color(s),
                       edgecolor="white", linewidth=0.4, zorder=4)
            kr, nr = closer_count(d[xa], d[yr])
            kc, nc = closer_count(d[xa], d[yc])
            ax.set_xlim(0, hi), ax.set_ylim(0, hi), ax.set_aspect("equal")
            win = S2.get(s).obs_window
            ax.set_title(f"{MF.short(s)} ({win}): {what}\nAI+RES closer in {kr}/{nr} "
                         f"(raw), {kc}/{nc} (corrected)", fontsize=8.5)
            ax.set_xlabel("AI+RES land RMSE (K)")
            ax.set_ylabel(f"{MF.short(s)} land RMSE (K)")
    handles = [Line2D([], [], marker=MF.MARKER["raw"], ls="", color="0.25", label="baseline raw"),
               Line2D([], [], marker=MF.MARKER["corr"], ls="", color="0.55",
                      label="baseline bias corrected")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.025))
    fig.suptitle("7-day-mean T2m anomaly maps vs ERA5, cases per baseline (cos(lat)-weighted "
                 "RMSE over CONUS land). Above the 1:1 line = AI+RES closer.\nTop: every "
                 "forecast scored by its expected best of 16 members; bottom: ensemble mean "
                 "(AI+RES importance weighted).", fontsize=10, y=0.985)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.11, hspace=0.42, wspace=0.3)
    fig.text(0.5, 0.008, MF.FOOT + f" {MF.by_window('12f', '/')} on 12 frames (6 UTC-day "
             "means), AI+RES re-reduced on the same frames.", ha="center", va="center", fontsize=7.5,
             color="0.3")
    return MF.save(fig, "members/closest_member_summary.png")


def _title(label, al, rmse, r, extra="") -> str:
    return f"{label}{extra}\nA_L {al:+.2f} K   RMSE {rmse:.2f} K   r {r:.2f}"


def case_figure(eid: str, family: str, peak, data: dict) -> Path:
    plt, ccrs, cfeature, proj = SB._plt()
    srcs = [s for s in MF.models() if s in data and eid in data[s]]
    a13 = data["aires_13f"][eid]
    cmap = plt.get_cmap("RdBu_r")
    ncol = 2 + len(srcs)
    fig = plt.figure(figsize=(2.75 * ncol, 6.5))
    gs = fig.add_gridspec(3, ncol + 1, width_ratios=[1] * ncol + [0.03], wspace=0.04,
                          hspace=0.36, left=0.01, right=0.975, top=0.875, bottom=0.085)
    ax = {}

    def panel(r, k, field, title):
        a = fig.add_subplot(gs[r, k], projection=proj)
        m = M._map_panel(a, xr.DataArray(field, dims=("lat", "lon"),
                         coords=dict(lat=_ctx()["base"]["lat"].values,
                                     lon=_ctx()["base"]["lon"].values)),
                         cmap, -VMAX, VMAX, ccrs, cfeature)
        a.set_title(title, fontsize=7.6, linespacing=1.25)
        return m

    land = _ctx()["land"] > 0
    m = panel(0, 0, np.where(land, a13["obs"], a13["obs"]),
              f"ERA5 observed (13 frames)\nA_L {float(a13['obs_al']):+.2f} K")
    ib = int(a13["best_raw"])
    panel(0, 1, a13["best_field"],
          _title("AI+RES closest walker", a13["al_raw"][ib], a13["rmse_raw"][ib],
                 a13["corr_raw"][ib], f" w{ib:02d}"))
    panel(1, 1, a13["mean_raw"],
          _title("AI+RES weighted mean", float(_al(a13["mean_raw"], _ctx())[0]),
                 a13["mean_raw_rmse"], a13["mean_raw_corr"]))
    for k, s in enumerate(srcs, start=2):
        z = data[s][eid]
        jb = int(z["best_raw"])
        win = S2.get(s).obs_window
        tag = "" if win == "13f" else " (12f)"
        lead = (f"  {z['lead'][jb]:.0f} d lead" if np.isfinite(z["lead"][jb]) else "")
        panel(0, k, z["best_field"],
              _title(f"{MF.short(s)} closest raw m{jb:02d}{tag}", z["al_raw"][jb],
                     z["rmse_raw"][jb], z["corr_raw"][jb], lead))
        panel(1, k, z["mean_raw"],
              _title(f"{MF.short(s)} raw mean, {int(z['n'])} mem{tag}",
                     float(_al(z["mean_raw"], _ctx())[0]), z["mean_raw_rmse"],
                     z["mean_raw_corr"]))
        if np.isfinite(z["mean_corr_rmse"]):
            panel(2, k, z["mean_corr"],
                  _title(f"{MF.short(s)} bias-corrected mean{tag}",
                         float(_al(z["mean_corr"], _ctx())[0]), z["mean_corr_rmse"],
                         z["mean_corr_corr"]))
        else:
            a = fig.add_subplot(gs[2, k])
            a.axis("off")
            a.text(0.5, 0.5, f"{MF.short(s)}: no bias field\nfor this case", ha="center",
                   va="center", fontsize=8, color="0.4")
    # column 0 rows 1-2: the colorbar and the reading key; column 1 row 2: the numbers
    cax = fig.add_subplot(gs[1, 0])
    cax.axis("off")
    cb = fig.colorbar(m, ax=cax, orientation="horizontal", fraction=0.16, pad=0.0,
                      extend="both", location="top", aspect=22)
    cb.set_label("7-day-mean T2m anomaly (K)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    key = fig.add_subplot(gs[2, 0])
    key.axis("off")
    key.text(0.0, 0.95, "Closest = smallest cos(lat)-\nweighted RMSE vs ERA5 over\n"
             "CONUS land. Rows: closest raw\nmember; raw ensemble mean;\n"
             "bias-corrected ensemble mean\n(leave-one-year-out or\nreforecast bias, per source).",
             ha="left", va="top", fontsize=7.5, transform=key.transAxes)
    box = fig.add_subplot(gs[2, 1])
    box.axis("off")
    lines = [f"{'RMSE (K)':<13s}{'med':>5s}{'best':>6s}{'E16':>6s}"]
    ra = a13["rmse_raw"]
    lines.append(f"{'AI+RES 13f':<13s}{np.median(ra):5.2f}{ra.min():6.2f}{e16(ra):6.2f}")
    if "aires_12f" in data and eid in data["aires_12f"] and any(
            S2.get(s).obs_window == "12f" for s in srcs):
        rb = data["aires_12f"][eid]["rmse_raw"]
        lines.append(f"{'AI+RES 12f':<13s}{np.median(rb):5.2f}{rb.min():6.2f}{e16(rb):6.2f}")
    for s in srcs:
        z = data[s][eid]
        for kind in ("raw", "corr"):
            v = z[f"rmse_{kind}"]
            if np.isfinite(v).any():
                lines.append(f"{MF.short(s)[:7] + ' ' + kind:<13s}{np.nanmedian(v):5.2f}"
                             f"{np.nanmin(v):6.2f}{e16(v):6.2f}")
    lines.append("E16 = expected best of 16")
    box.text(0.0, 0.97, "\n".join(lines), ha="left", va="top", fontsize=7, family="monospace",
             transform=box.transAxes)
    nm = ", ".join(f"{MF.short(s)} {int(data[s][eid]['n'])}" for s in srcs)
    fig.suptitle(f"{eid}  ({family}, week ending {pd.Timestamp(peak).date()}, ~21 d lead)   "
                 f"members: AI+RES {int(a13['n'])}, {nm}", fontsize=10.5, y=0.985)
    degs = "; ".join(f"{MF.short(s)} {S2.style(s)['deg']}" for s in srcs)
    fig.text(0.5, 0.05, f"{MF.by_window('12f', ' / ')} panels are 12-frame (6 UTC-day) means "
             f"scored against ERA5 on the same 12 frames; {MF.by_window('13f', ' / ')} use 13 "
             f"frames. Native grid spacing: {degs}; all regridded to 0.25 deg.", ha="center", va="top",
             fontsize=7.5, color="0.3")
    fig.text(0.5, 0.02, MF.FOOT, ha="center", va="top", fontsize=7.5, color="0.3")
    return MF.save(fig, f"members/{eid}.png")


def figures(data: dict | None = None) -> list[Path]:
    data = data if data is not None else aggregate()
    if "aires_13f" not in data:
        print("[multi_members] no AI+RES cache, no figures")
        return []
    df, cases = _cases()
    out = [p for p in [summary_figure(data)] if p]
    for c, fam in zip(cases, df.family):
        if c.episode_id in data["aires_13f"]:
            out.append(case_figure(c.episode_id, fam, c.run["peak"], data))
    t = _pair_table(data)
    for s in MF.models():
        d = t[t.source == s]
        if d.empty:
            continue
        print(f"  {MF.short(s):8s} n={len(d)}  AI+RES closer, E16: raw "
              f"{closer_count(d.a_e16, d.r_e16)}, corr {closer_count(d.a_e16, d.c_e16)}  | "
              f"mean: raw {closer_count(d.a_mean, d.r_mean)}, "
              f"corr {closer_count(d.a_mean, d.c_mean)}")
    return out


def main(argv=None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=("rmse", "figures", "all"), default="all")
    p.add_argument("--force", action="store_true", help="redo the cached member pass")
    p.add_argument("--workers", type=int, default=WORKERS)
    a = p.parse_args(argv)
    if a.stage in ("rmse", "all"):
        member_pass(a.force, a.workers)
    if a.stage == "rmse":
        aggregate()
        return []
    return figures()


if __name__ == "__main__":
    main()
