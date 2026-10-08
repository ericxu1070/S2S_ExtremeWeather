#!/usr/bin/env python
"""Multi-model critical success index maps: AI+RES next to every baseline. CPU ONLY.

Replaces `sidebyside.csi_{heat,cold}_{2,3,4}K.png` (AI+RES vs CFSv2 raw only). One figure
per threshold (-2, -3, -4, +2, +3, +4 K) and per reduction:

    csi/csi_{heat,cold}_{2,3,4}K.png        7-day mean field (headline set)
    csi/csi_daily_{heat,cold}_{2,3,4}K.png  every day of the verification week, pooled

Each figure has a top row (AI+RES CSI map and a table of land-mean CSI) and one row per
baseline with four maps: raw CSI | bias-corrected CSI | AI+RES minus raw | AI+RES minus
corrected. Per cell, CSI = hits / (hits + misses + false alarms) with yes = P >= 0.5, over
the family-matched cases (heat thresholds on the 31 heat cases, cold on the 11 cold), land
only (`acal.maps.scores`). CSI gives no credit for a correct "no", so it does not rise as
the threshold gets rarer. Scales are fixed (CSI 0 to 1, difference -0.6 to 0.6, red =
AI+RES better) so every row and every figure reads on one scale.

Each baseline is scored on its OWN file: the daily-mean sources (EC46, GEPS) are verified
on 12 frames and `prob_aires` in their file is AI+RES re-reduced on those same frames, so
the difference always compares like with like. The top-row AI+RES map uses the first
13-frame baseline's file. Scores are cached under `multifig.DATA_DIR` and rebuilt when the
source file is newer. Also writes `csi_land_means.csv` (land mean of every panel).

    python -m acal.multi_csi
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import maps as M
from acal import multifig as MF
from acal import s2sbase as S2
from acal import sidebyside as SB

CSI_VMAX, DIFF_VMAX = SB.CSI_VMAX, SB.DIFF_VMAX
KINDS = (("aires", "prob_aires"), ("raw", "prob_raw"), ("corr", "prob_corr"))
TAGS = (("7d", "maps_fields"), ("daily", "maps_daily"))


def note() -> str:
    """Footer naming only the baselines in the figure (`multifig.by_window`)."""
    return ("Native grids are regridded to 0.25 deg, so gridpoint skill is resolution-limited "
            f"for the coarser baselines. Daily-mean baselines ({MF.by_window('12f')}) are "
            "verified on 12 frames, with AI+RES re-reduced on the same 12.")


def wmean(da: xr.DataArray) -> float:
    """cos(lat)-weighted mean over the non-NaN (land) cells."""
    return float(da.weighted(np.cos(np.deg2rad(da["lat"]))).mean())


def win_share(diff: xr.DataArray) -> float:
    """Land share (cos-lat weighted) where the difference is positive."""
    w = np.cos(np.deg2rad(diff["lat"]))
    return float((diff > 0).where(diff.notnull()).weighted(w).mean())


def fam_of(a: float) -> str:
    return "heat" if a > 0 else "cold"


def fig_name(tag: str, a: float) -> str:
    return f"csi/csi_{'daily_' if tag == 'daily' else ''}{fam_of(a)}_{abs(a):.0f}K.png"


def csi_maps(source: str, tag: str, stem: str) -> xr.Dataset | None:
    """Per-cell CSI (aires, raw, corr; dims threshold, lat, lon) of one source file, cached.
    None when the source's table is missing."""
    src = S2.ANALYSIS / MF.TRUTH / f"{stem}_{source}.nc"
    if not src.exists():
        print(f"  note: {src.name} missing, {source} left out of the {tag} figures")
        return None
    cache = MF.DATA_DIR / f"csi_{source}_{tag}.nc"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return xr.open_dataset(cache).load()
    print(f"  scoring {src.name} ...", flush=True)
    ds = xr.open_dataset(src).load()
    out = {}
    for k, var in KINDS:
        if var in ds:
            sc = M.scores(ds, var)
        else:
            # 13-frame sources carry no AI+RES: it is the published 13-frame AI+RES file
            # (`prob`), exactly what sidebyside scores for CFSv2
            from acal import truth as TR
            base = M.truth_base(TR.get_truth(MF.TRUTH), tag == "daily")
            sc = M.scores(xr.open_dataset(base).load())
        out[k] = sc["csi"]
        n_case = sc.attrs["n_case"]
    res = xr.Dataset(out, attrs=dict(n_case=list(n_case), source=source))
    MF.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".tmp.nc")
    res.to_netcdf(tmp)
    tmp.replace(cache)
    return res


def load_all() -> dict:
    """{tag: {source: csi dataset}} for the baselines whose files exist."""
    return {tag: {s: d for s in MF.models()
                  if (d := csi_maps(s, tag, stem)) is not None}
            for tag, stem in TAGS}


def land_means(sc: dict) -> pd.DataFrame:
    """One row per (tag, source, threshold, kind): land mean CSI, plus the AI+RES minus
    baseline median difference and win share for the baseline kinds."""
    rows = []
    for tag, per in sc.items():
        for s, d in per.items():
            for a in M.THRESHOLDS:
                aires = d["aires"].sel(threshold=a)
                for k, _ in KINDS:
                    da = d[k].sel(threshold=a)
                    row = dict(tag=tag, source=s, threshold=a, kind=k, land_mean=wmean(da),
                               n_case=d.attrs["n_case"][M.THRESHOLDS.index(a)])
                    if k != "aires":
                        diff = aires - da
                        row.update(median_diff=float(diff.median()),
                                   aires_better=win_share(diff))
                    rows.append(row)
    return pd.DataFrame(rows)


def figure(sc: dict, tag: str, a: float) -> Path:
    plt, ccrs, cfeature, proj = SB._plt()
    per = sc[tag]
    srcs = list(per)
    n = len(srcs)
    fam = fam_of(a)
    ti = M.THRESHOLDS.index(a)
    top = next((s for s in srcs if S2.get(s).obs_window == "13f"), srcs[0])
    fig = plt.figure(figsize=(15.5, 2.55 * (n + 1) + 1.3))
    # top row + one row per baseline + a slim row for the two horizontal colorbars
    gs = fig.add_gridspec(n + 2, 4, height_ratios=[1] * (n + 1) + [0.07],
                          wspace=0.03, hspace=0.30, left=0.05, right=0.99, top=0.93,
                          bottom=0.05)
    seq = plt.get_cmap("Reds" if a > 0 else "Blues").copy()
    div = plt.get_cmap("RdBu_r").copy()
    for cm in (seq, div):
        cm.set_bad("white")                    # sea, or CSI undefined (no events, no alarms)
    cases = per[top].attrs["n_case"][ti]
    days = " x 7 days" if tag == "daily" else ""

    # --- top row: AI+RES map, then a table of land means --------------------------- #
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    res = per[top]["aires"].sel(threshold=a)
    m_csi = M._map_panel(ax, res, seq, 0, CSI_VMAX, ccrs, cfeature)
    win = S2.get(top).obs_window                 # '13f' unless only daily sources remain
    ax.set_title(f"AI+RES ({win[:-1]} frames)   land mean {wmean(res):.2f}", fontsize=9)
    ax.text(-0.04, 0.5, f"AI+RES\n({cases} cases{days})", rotation=90,
            transform=ax.transAxes, ha="right", va="center", fontsize=10)
    tx = fig.add_subplot(gs[0, 1:])
    tx.axis("off")
    cell = []
    for s in srcs:
        d = per[s]
        v = {k: wmean(d[k].sel(threshold=a)) for k, _ in KINDS}
        better = {k: win_share(d["aires"].sel(threshold=a) - d[k].sel(threshold=a))
                  for k in ("raw", "corr")}
        cell.append([f"{MF.short(s)} ({S2.get(s).obs_window})", f"{v['aires']:.2f}",
                     f"{v['raw']:.2f}", f"{v['corr']:.2f}",
                     f"{better['raw']:.0%}", f"{better['corr']:.0%}"])
    tb = tx.table(cellText=cell, loc="center", cellLoc="center",
                  colLabels=["baseline (frames)", "AI+RES", "raw", "corrected",
                             "AI+RES better\nthan raw", "AI+RES better\nthan corrected"])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1.0, 1.7)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor("0.8")
        if r == 0:
            cl.set_facecolor("0.93")
            cl.set_height(cl.get_height() * 1.5)
    tx.set_title("Land-mean CSI on each baseline's own window "
                 "(AI+RES re-reduced on the same frames)", fontsize=9)

    # --- one row per baseline ------------------------------------------------------- #
    m_diff = None
    for r, s in enumerate(srcs, start=1):
        d = per[s]
        res_s = d["aires"].sel(threshold=a)
        raw, cor = (d[k].sel(threshold=a) for k in ("raw", "corr"))
        panels = (
            (raw, seq, 0, CSI_VMAX, f"{MF.short(s)} raw   land mean {wmean(raw):.2f}"),
            (cor, seq, 0, CSI_VMAX, f"{MF.short(s)} bias-corrected   "
                                    f"land mean {wmean(cor):.2f}"),
            (res_s - raw, div, -DIFF_VMAX, DIFF_VMAX, None),
            (res_s - cor, div, -DIFF_VMAX, DIFF_VMAX, None))
        for k, (da, cmap, lo, hi, sub) in enumerate(panels):
            ax = fig.add_subplot(gs[r, k], projection=proj)
            mm = M._map_panel(ax, da, cmap, lo, hi, ccrs, cfeature)
            if k >= 2:
                m_diff = mm
                what = "raw" if k == 2 else "corrected"
                sub = (f"AI+RES minus {MF.short(s)} {what}   median {float(da.median()):+.2f}"
                       f"\nAI+RES better on {win_share(da):.0%} of land")
            ax.set_title(sub, fontsize=8.5)
            if k == 0:
                ax.text(-0.04, 0.5, f"{MF.short(s)}\n{S2.style(s)['deg']}\n"
                        f"({S2.get(s).obs_window})", rotation=90,
                        transform=ax.transAxes, ha="right", va="center", fontsize=10)
    # --- shared colorbars: CSI under columns 0-1, difference under columns 2-3 ------ #
    cax1, cax2 = fig.add_subplot(gs[n + 1, 0:2]), fig.add_subplot(gs[n + 1, 2:4])
    for cax, dx in ((cax1, -0.02), (cax2, 0.02)):      # keep the two bars apart
        b = cax.get_position()
        cax.set_position([b.x0 + (0.02 if dx > 0 else 0.0),
                          b.y0, b.width - 0.02, b.height])
    c1 = fig.colorbar(m_csi, cax=cax1, orientation="horizontal")
    c1.set_label("CSI (yes = P >= 0.5)", fontsize=9)
    c2 = fig.colorbar(m_diff, cax=cax2, orientation="horizontal",
                      extend="both")
    c2.set_label("CSI difference (red = AI+RES better)", fontsize=9)
    sym = ">=" if a > 0 else "<="
    red = "daily mean, every day pooled" if tag == "daily" else "7-day mean"
    fig.suptitle(f"Critical success index per cell, {fam} extremes: T2m anomaly {sym} "
                 f"{a:+.0f} K, {red}", fontsize=12, y=0.985)
    fig.text(0.5, 0.0, note() + "\n" + MF.FOOT, ha="center", va="top", fontsize=7.5,
             color="0.3")
    return MF.save(fig, fig_name(tag, a))


def main() -> list[Path]:
    sc = load_all()
    sc = {t: p for t, p in sc.items() if p}
    if not sc:
        print("  no per-source map files: nothing to draw")
        return []
    df = land_means(sc)
    MF.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MF.DATA_DIR / "csi_land_means.csv", index=False)
    out = []
    for tag in ("7d", "daily"):
        if tag in sc:
            out += [figure(sc, tag, a) for a in M.THRESHOLDS]
    return out


if __name__ == "__main__":
    main()
