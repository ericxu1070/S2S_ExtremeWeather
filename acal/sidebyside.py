#!/usr/bin/env python
"""AI+RES and CFSv2 side by side: CSI per threshold, and the member closest to ERA5. CPU ONLY.

Two stages, both reading what `acal.maps` already built (no new forecast data).

csi
---
One figure per threshold (-2, -3, -4, +2, +3, +4 K), `figures/acal/sidebyside/
csi_{cold,heat}_{2,3,4}K.png`. Rows are the 7-day-mean and the daily-mean extremes, columns
are AI+RES | CFSv2 raw | AI+RES minus CFSv2 raw. The scores are `maps.scores` unchanged
(family-matched cases, land only, yes = P >= 0.5), so every panel reproduces the matching
panel of `acal_map_csi_*` / `acal_map_csi_*_cfs` / `acal_map_csi_*_diff_cfs`. The CSI colour
scale is fixed to [0, 1] and the difference to [-0.6, 0.6] in every figure, so the six
figures can be compared with each other.

members
-------
For each of the 42 cases: which single AI+RES walker and which single CFSv2 member has the
7-day-mean T2m anomaly map closest to ERA5? Closeness is the cos(lat)-weighted RMSE over
CONUS land (`maps.land_mask`, the cells the skill maps score):

    RMSE_i = sqrt( sum_x c(x) (A_i(x) - O(x))^2 / sum_x c(x) ),   c = cos(lat), x on land

The CFS member is the RAW anomaly (its cold drift counts against it, as in the headline
scores). Fields: walkers through `maps.walker_fields`, CFS members through
`maps.cfs_member_fields`, both of which assert their CONUS mean against the recorded A_L.
The AI+RES ensemble mean is importance-weighted (self-normalized w_i = exp(-V_K,i)); the CFS
mean is equal-weight.

Fairness: AI+RES has 32 walkers and CFS 16 members, and the minimum of more draws is
smaller by chance alone. `res_min16_exp` is the expected minimum walker RMSE over random
16-walker subsets (exact, from the order statistics of the 32 RMSEs); compare CFS against
that, not against the 32-walker minimum. Walkers are also not independent (cloned walkers
share their path up to the clone), which works the other way.

    python -m acal.sidebyside --stage csi       # 6 figures, ~1 min
    python -m acal.sidebyside --stage members   # ~15 min: closest_members.{csv,nc},
                                                # 42 per-case maps + summary
    python -m acal.sidebyside --stage figures   # redraw member maps from the .nc, ~2 min
"""
from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import ccfg
from acal import maps as M
from aires import aindex as AI

FIG_DIR = ccfg.FIG_ROOT / "sidebyside"
MEMBER_DIR = FIG_DIR / "members"
CLOSEST_CSV = AN.OUT / "closest_members.csv"
CLOSEST_NC = AN.OUT / "closest_members.nc"
CSI_VMAX, DIFF_VMAX = 1.0, 0.6
SUBSET = 16                               # CFS ensemble size, for the fair AI+RES minimum
C_HEAT, C_COLD = "#c0392b", "#2463a6"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    AN._style(plt)
    proj = ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45))
    return plt, ccrs, cfeature, proj


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _wmean(da: xr.DataArray) -> float:
    return float(da.weighted(np.cos(np.deg2rad(da["lat"]))).mean())


# --------------------------------------------------------------------------- #
# Stage: csi
# --------------------------------------------------------------------------- #
def csi_scores() -> dict:
    """{tag: (AI+RES scores, CFS raw scores)} for the 7-day and daily files."""
    out = {}
    for tag, res_src, cfs_src in (("7d", M.FIELDS, M.CFS_FIELDS),
                                  ("daily", M.DAILY_FIELDS, M.CFS_DAILY_FIELDS)):
        out[tag] = (M.scores(xr.open_dataset(res_src).load()),
                    M.scores(xr.open_dataset(cfs_src).load(), "prob_raw"))
    return out


def csi_figure(sc: dict, a: float) -> Path:
    plt, ccrs, cfeature, proj = _plt()
    fam = "heat" if a > 0 else "cold"
    fig = plt.figure(figsize=(14, 5.4))
    # maps | maps | cbar | gap | diff map | cbar
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 0.035, 0.09, 1, 0.035],
                          wspace=0.04, hspace=0.22, left=0.05, right=0.96, top=0.86,
                          bottom=0.06)
    axes = np.array([[fig.add_subplot(gs[r, c], projection=proj) for c in (0, 1, 4)]
                     for r in range(2)])
    seq = plt.get_cmap("Reds" if a > 0 else "Blues").copy()
    div = plt.get_cmap("RdBu_r").copy()
    for cm in (seq, div):
        cm.set_bad("white")                    # sea, or CSI undefined (no events, no alarms)
    for r, (tag, label) in enumerate((("7d", "7-day mean"), ("daily", "daily mean"))):
        s_res, s_cfs = (s["csi"].sel(threshold=a) for s in sc[tag])
        n = sc[tag][0].attrs["n_case"][M.THRESHOLDS.index(a)]
        diff = s_res - s_cfs
        win = float((diff > 0).where(diff.notnull()).weighted(
            np.cos(np.deg2rad(diff["lat"]))).mean())
        panels = (
            (s_res, seq, 0, CSI_VMAX, f"AI+RES   land mean {_wmean(s_res):.2f}"),
            (s_cfs, seq, 0, CSI_VMAX, f"CFSv2 raw   land mean {_wmean(s_cfs):.2f}"),
            (diff, div, -DIFF_VMAX, DIFF_VMAX,
             f"AI+RES minus CFSv2   median {float(diff.median()):+.2f}\n"
             f"AI+RES better on {win:.0%} of land"))
        for k, (da, cmap, lo, hi, sub) in enumerate(panels):
            m = M._map_panel(axes[r, k], da, cmap, lo, hi, ccrs, cfeature)
            axes[r, k].set_title(sub, fontsize=9)
            if k == 1:
                fig.colorbar(m, cax=fig.add_subplot(gs[r, 2])).set_label("CSI")
            if k == 2:
                fig.colorbar(m, cax=fig.add_subplot(gs[r, 5]), extend="both"
                             ).set_label("CSI difference")
        days = " x 7 days" if tag == "daily" else ""
        axes[r, 0].text(-0.04, 0.5, f"{label}\n({n} cases{days})", rotation=90,
                        transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=10)
    sym = "≥" if a > 0 else "≤"
    fig.suptitle(f"Critical success index per cell, {fam} extremes: T2m anomaly {sym} "
                 f"{a:+.0f} K   (yes = P ≥ 0.5; red in the difference = AI+RES better)",
                 fontsize=11, y=0.98)
    fig.text(0.5, 0.0, M.CFS_NOTE, ha="center", va="top", fontsize=8, color="0.3")
    return _save(fig, FIG_DIR / f"csi_{fam}_{abs(a):.0f}K.png")


def csi() -> list[Path]:
    sc = csi_scores()
    return [csi_figure(sc, a) for a in M.THRESHOLDS]


# --------------------------------------------------------------------------- #
# Stage: members
# --------------------------------------------------------------------------- #
def land_rmse(F: np.ndarray, obs: np.ndarray, land: np.ndarray, coslat: np.ndarray):
    """cos(lat)-weighted land RMSE of each leading-axis field in F against obs."""
    w = (coslat[:, None] * land)                       # (lat, lon), 0 off land
    return np.sqrt(((F - obs[None]) ** 2 * w[None]).sum((1, 2)) / w.sum())


def land_corr(F: np.ndarray, obs: np.ndarray, land: np.ndarray, coslat: np.ndarray):
    """cos(lat)-weighted centred pattern correlation over land, per leading-axis field."""
    w = (coslat[:, None] * land)
    w = w / w.sum()
    fa = F - (F * w[None]).sum((1, 2), keepdims=True)
    oa = obs - (obs * w).sum()
    cov = (fa * oa[None] * w[None]).sum((1, 2))
    return cov / np.sqrt((fa ** 2 * w[None]).sum((1, 2)) * (oa ** 2 * w).sum())


def expected_min(x: np.ndarray, k: int) -> float:
    """E[min] of a uniformly random k-subset of x: the j-th smallest (0-based) is the
    minimum in C(n-1-j, k-1) of the C(n, k) subsets."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    return float(sum(x[j] * comb(n - 1 - j, k - 1) for j in range(n - k + 1)) / comb(n, k))


def members() -> Path:
    df, cases = AN.load_all()
    base = xr.open_dataset(M.FIELDS).load()
    land = M.land_mask(base).values.astype(float)
    coslat = np.cos(np.deg2rad(base["lat"].values))
    rows, maps = [], []
    for i, c in enumerate(cases):
        eid = c.episode_id
        if str(base["case"].values[i]) != eid:
            raise SystemExit(f"[sbs] case order: {base['case'].values[i]} != {eid}")
        peak = pd.Timestamp(c.run["peak"])
        obs = base["truth"].isel(case=i).values
        R = M.walker_fields(c)
        C = M.cfs_member_fields(eid, peak, daily=False)
        w = c.weights / c.weights.sum()
        r_res = land_rmse(R.values, obs, land, coslat)
        r_cfs = land_rmse(C.values, obs, land, coslat)
        res_mean = np.tensordot(w, R.values, axes=(0, 0))
        cfs_mean = C.values.mean(0)
        ib, jb = int(np.argmin(r_res)), int(np.argmin(r_cfs))
        cr = land_corr(np.stack([R.values[ib], C.values[jb], res_mean, cfs_mean]),
                       obs, land, coslat)
        rm = land_rmse(np.stack([res_mean, cfs_mean]), obs, land, coslat)
        cmp_conus = np.asarray(c.cmp["realized"]["conus"], float)
        rows.append(dict(
            episode_id=eid, family=df.family.iloc[i],
            peak=peak.date(), obs_al=c.obs,
            res_best=ib, res_best_rmse=r_res[ib], res_best_corr=cr[0],
            res_best_al=cmp_conus[ib], res_best_weight=w[ib],
            res_best_rank_weight=int((w > w[ib]).sum()) + 1,
            res_rmse_median=float(np.median(r_res)), res_min16_exp=expected_min(r_res, SUBSET),
            res_mean_rmse=rm[0], res_mean_corr=cr[2],
            cfs_best=jb, cfs_best_rmse=r_cfs[jb], cfs_best_corr=cr[1],
            cfs_best_al=float(AI.area_mean(C.isel(member=jb))),
            cfs_best_lead_days=float(C["member_lead_days"].values[jb])
            if "member_lead_days" in C.coords else np.nan,
            cfs_rmse_median=float(np.median(r_cfs)),
            cfs_mean_rmse=rm[1], cfs_mean_corr=cr[3]))
        maps.append(np.stack([obs, R.values[ib], C.values[jb], res_mean, cfs_mean]))
        print(f"  {eid}  obs {c.obs:+.2f}  best walker w{ib:02d} {r_res[ib]:.2f} K "
              f"(E16 {rows[-1]['res_min16_exp']:.2f})  best CFS m{jb:02d} {r_cfs[jb]:.2f} K",
              flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(CLOSEST_CSV, index=False, float_format="%.6g")
    ds = xr.Dataset(
        dict(field=(("case", "panel", "lat", "lon"), np.stack(maps).astype("float32")),
             land=(("lat", "lon"), land.astype("int8"))),
        coords=dict(case=out.episode_id.values,
                    panel=["era5", "res_best", "cfs_best", "res_mean", "cfs_mean"],
                    lat=base["lat"].values, lon=base["lon"].values),
        attrs=dict(note="7-day-mean T2m anomaly (K) ending at the peak; *_best = member "
                        "with the smallest cos-lat land RMSE vs era5; res_mean importance-"
                        "weighted, cfs_mean equal-weight, CFS raw"))
    ds.to_netcdf(CLOSEST_NC)
    print(f"[sbs] wrote {CLOSEST_CSV}\n[sbs] wrote {CLOSEST_NC}")
    return CLOSEST_NC


def member_figure(ds: xr.Dataset, row: pd.Series) -> Path:
    plt, ccrs, cfeature, proj = _plt()
    f = ds["field"].sel(case=row.episode_id)
    land = ds["land"].values.astype(bool)
    vmax = float(np.ceil(np.nanpercentile(np.abs(f.sel(panel="era5").values[land]), 99)))
    vmax = max(vmax, 3.0)
    cmap = plt.get_cmap("RdBu_r")
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.035], wspace=0.05, hspace=0.28,
                          left=0.02, right=0.95, top=0.86, bottom=0.06)
    axes = np.array([[fig.add_subplot(gs[r, k], projection=proj) if (r, k) != (1, 0)
                      else fig.add_subplot(gs[r, k]) for k in range(3)] for r in range(2)])
    lead = (f", {row.cfs_best_lead_days:.2f} d lead" if np.isfinite(row.cfs_best_lead_days)
            else "")
    titles = {
        "era5": f"ERA5 observed   A_L {row.obs_al:+.2f} K",
        "res_best": (f"AI+RES closest walker w{row.res_best:02d}   A_L {row.res_best_al:+.2f} K\n"
                     f"RMSE {row.res_best_rmse:.2f} K, r {row.res_best_corr:.2f}, "
                     f"weight {row.res_best_weight:.3f} (#{row.res_best_rank_weight} of 32)"),
        "cfs_best": (f"CFSv2 closest member m{row.cfs_best:02d}   A_L {row.cfs_best_al:+.2f} K\n"
                     f"RMSE {row.cfs_best_rmse:.2f} K, r {row.cfs_best_corr:.2f}{lead}"),
        "res_mean": (f"AI+RES weighted ensemble mean\nRMSE {row.res_mean_rmse:.2f} K, "
                     f"r {row.res_mean_corr:.2f}"),
        "cfs_mean": (f"CFSv2 16-member ensemble mean (raw)\nRMSE {row.cfs_mean_rmse:.2f} K, "
                     f"r {row.cfs_mean_corr:.2f}"),
    }
    layout = {(0, 0): "era5", (0, 1): "res_best", (0, 2): "cfs_best",
              (1, 1): "res_mean", (1, 2): "cfs_mean"}
    for (r, k), p in layout.items():
        m = M._map_panel(axes[r, k], f.sel(panel=p), cmap, -vmax, vmax, ccrs, cfeature)
        axes[r, k].set_title(titles[p], fontsize=9)
    axes[1, 0].axis("off")
    better = "AI+RES" if row.res_min16_exp < row.cfs_best_rmse else "CFSv2"
    txt = (f"Closest member = smallest cos(lat)-weighted\nRMSE vs ERA5 over CONUS land.\n\n"
           f"Median member RMSE\n  AI+RES {row.res_rmse_median:.2f} K   CFSv2 {row.cfs_rmse_median:.2f} K\n"
           f"Closest member RMSE\n  AI+RES {row.res_best_rmse:.2f} K (best of 32)\n"
           f"  AI+RES {row.res_min16_exp:.2f} K (expected best of 16)\n"
           f"  CFSv2  {row.cfs_best_rmse:.2f} K (best of 16)\n"
           f"Like-for-like (16 vs 16): {better} closer")
    axes[1, 0].text(0.02, 0.5, txt, ha="left", va="center", fontsize=8.5,
                    family="monospace", transform=axes[1, 0].transAxes)
    cb = fig.colorbar(m, cax=fig.add_subplot(gs[:, 3]), extend="both")
    cb.set_label("7-day-mean T2m anomaly (K)")
    fam = "heat" if row.obs_al > 0 else "cold"
    fig.suptitle(f"{row.episode_id}  ({fam}, week ending {row.peak}, 21 d lead)",
                 fontsize=11, y=0.97)
    fig.text(0.5, 0.01, M.CFS_NOTE, ha="center", va="top", fontsize=8, color="0.3")
    return _save(fig, MEMBER_DIR / f"{row.episode_id}.png")


def summary_figure(df: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    AN._style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    col = np.where(df.family == "heat", C_HEAT, C_COLD)
    panels = (("res_best_rmse", "cfs_best_rmse", "closest member, best of 32 vs best of 16"),
              ("res_min16_exp", "cfs_best_rmse", "closest member, like-for-like (16 vs 16)"),
              ("res_mean_rmse", "cfs_mean_rmse", "ensemble mean"))
    hi = float(np.ceil(df[[p[0] for p in panels] + [p[1] for p in panels]].max().max()))
    for ax, (x, y, t) in zip(axes, panels):
        ax.scatter(df[x], df[y], c=col, s=26, edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([0, hi], [0, hi], color="0.5", lw=0.8, zorder=1)
        ax.set_xlim(0, hi), ax.set_ylim(0, hi), ax.set_aspect("equal")
        win = int((df[x] < df[y]).sum())
        ax.set_title(f"{t}\nAI+RES closer in {win} of {len(df)} cases", fontsize=9.5)
        ax.set_xlabel("AI+RES land RMSE vs ERA5 (K)")
        ax.set_ylabel("CFSv2 (raw) land RMSE vs ERA5 (K)")
        ax.text(0.04, 0.96, "above line: AI+RES closer", transform=ax.transAxes,
                fontsize=8, color="0.35", va="top")
    axes[-1].legend(handles=[Line2D([], [], marker="o", ls="", color=C_HEAT, label="heat"),
                             Line2D([], [], marker="o", ls="", color=C_COLD, label="cold")],
                    loc="lower right", frameon=False)
    fig.suptitle("7-day-mean T2m anomaly maps vs ERA5, 42 cases (cos(lat)-weighted RMSE "
                 "over CONUS land)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, FIG_DIR / "closest_member_summary.png")


def member_figures() -> list[Path]:
    df = pd.read_csv(CLOSEST_CSV)
    ds = xr.open_dataset(CLOSEST_NC).load()
    out = [member_figure(ds, r) for r in df.itertuples(index=False)]
    out.append(summary_figure(df))
    for fam, d in [("all", df)] + list(df.groupby("family")):
        print(f"  {fam:5s} n={len(d):2d}  closest RMSE median: AI+RES {d.res_best_rmse.median():.2f}"
              f" (E16 {d.res_min16_exp.median():.2f}) CFS {d.cfs_best_rmse.median():.2f}  | "
              f"AI+RES closer 32v16 {int((d.res_best_rmse < d.cfs_best_rmse).sum())}, "
              f"16v16 {int((d.res_min16_exp < d.cfs_best_rmse).sum())}, "
              f"mean {int((d.res_mean_rmse < d.cfs_mean_rmse).sum())}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=("csi", "members", "figures", "all"), default="all")
    a = p.parse_args(argv)
    if a.stage in ("csi", "all"):
        csi()
    if a.stage in ("members", "all"):
        members()
    if a.stage in ("members", "figures", "all"):
        member_figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
