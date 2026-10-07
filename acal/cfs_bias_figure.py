#!/usr/bin/env python
"""Slide figure: how the CFSv2 leave-one-year-out bias correction works.

Reads what `acal.cfsbase --stage hind/bias/build` wrote; CPU, seconds.

    python -m acal.cfs_bias_figure                         # e02_c4_20210218
    python -m acal.cfs_bias_figure --case e14_h4_20230104
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import cfsbase as CB

FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "acal"
C_MEM = "#2b6cb0"
C_CORR = "#2f855a"
C_OBS = "#c27a12"
C_OBS_SOFT = "#f7ead3"
C_HEAT = "#c0392b"
C_COLD = "#2b6cb0"


def main(eid: str) -> Path:
    tab = pd.read_csv(CB.CFS_ROOT / "bias.csv")
    fam = pd.read_csv(CB.BUILD_CSV)[["episode_id", "family"]]
    tab = tab.merge(fam, on="episode_id")
    row = tab.set_index("episode_id").loc[eid]
    b = float(row.bias_conus)
    years = [int(y) for y in str(row.years).split(",")]
    d = json.loads(CB.json_path(eid).read_text())
    al = np.asarray(d["al"], float)
    obs, s = float(d["obs"]), float(d["sign"])
    case_year = pd.Timestamp(d["cycles"][-1]).year

    plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans"})
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.33, 6.4),
                                     gridspec_kw={"width_ratios": [1.35, 1, 1.15],
                                                  "wspace": 0.32})

    # --- 1: the same 16-run lag in every other year ---
    diffs = []
    for k, y in enumerate(years):
        with xr.open_dataset(CB.hind_path(eid, y)) as h:
            m = h.al_cfs.values.astype(float)
            e = float(h.al_era5)
        diffs.append(m.mean() - e)
        jit = np.linspace(-0.18, 0.18, m.size)
        ax.scatter(k + jit, m, s=14, color=C_MEM, alpha=0.35, lw=0)
        ax.plot([k - 0.28, k + 0.28], [m.mean()] * 2, color=C_MEM, lw=3)
        ax.plot(k, e, marker="D", color="black", ms=8, zorder=3)
        ax.annotate("", xy=(k + 0.36, m.mean()), xytext=(k + 0.36, e),
                    arrowprops=dict(arrowstyle="->", color=C_OBS, lw=2))
        diffs_txt = f"{m.mean() - e:+.2f}"
        ax.text(k, 1.0, diffs_txt, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=12, color=C_OBS)
    ax.set_xticks(range(len(years)), [str(y) for y in years], fontsize=12)
    ax.set_xlim(-0.6, len(years) - 0.4)
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_ylabel("CONUS 7-day mean T2m anomaly $A_L$ (K)", fontsize=13)
    ax.set_title(f"1. Re-run the same 16-run lag\n    at this date in the other years",
                 loc="left", fontsize=14, fontweight="bold", pad=26)
    ax.text(0.0, -0.13, f"{case_year} (the case's own year) is left out", transform=ax.transAxes,
            fontsize=11, color="0.35")
    ax.scatter([], [], s=14, color=C_MEM, alpha=0.5, label="CFS member")
    ax.plot([], [], color=C_MEM, lw=3, label="CFS mean")
    ax.plot([], [], "D", color="black", ms=7, label="ERA5")
    ax.plot([], [], color=C_OBS, lw=2, label="CFS mean - ERA5")
    ax.legend(loc="lower left", fontsize=10, frameon=False, ncol=2)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.25 * (hi - lo), hi)

    # --- 2: subtract the mean of those errors from every member ---
    corr = al - b
    n_raw = int((s * al >= s * obs).sum())
    n_cor = int((s * corr >= s * obs).sum())
    yr, yc = 1.0, 0.0
    jit = ((np.arange(al.size) % 3) - 1) * 0.07
    bx.axvspan(-30 if s < 0 else obs, obs if s < 0 else 30, color=C_OBS_SOFT, zorder=0)
    bx.axvline(obs, color=C_OBS, lw=2)
    bx.scatter(al, yr + jit, s=60, color=C_MEM, zorder=3)
    bx.scatter(corr, yc + jit, s=60, color=C_CORR, zorder=3)
    for a, c in zip(al, corr):
        bx.plot([a, c], [yr - 0.15, yc + 0.15], color="0.75", lw=0.6, zorder=1)
    bx.plot([al.mean()] * 2, [yr - 0.25, yr + 0.25], color=C_MEM, lw=3)
    bx.plot([corr.mean()] * 2, [yc - 0.25, yc + 0.25], color=C_CORR, lw=3)
    bx.set_yticks([yc, yr], [f"corrected\n{n_cor}/16 reach", f"raw\n{n_raw}/16 reach"],
                  fontsize=12)
    for lab, c in zip(bx.get_yticklabels(), (C_CORR, C_MEM)):
        lab.set_color(c)
    bx.set_ylim(-0.6, 1.75)
    lo = min(al.min(), corr.min(), obs) - 0.8
    hi = max(al.max(), corr.max(), 0) + 0.8
    bx.set_xlim(lo, hi)
    bx.text(obs, 1.62, f"ERA5 obs {obs:+.2f}", ha="center", fontsize=11.5, color=C_OBS,
            bbox=dict(fc="white", ec="none", pad=1))
    bx.set_xlabel("$A_L$ (K)", fontsize=13)
    bx.set_title(f"2. Subtract the mean error\n    bias = {b:+.2f} K  (mean of {len(years)})",
                 loc="left", fontsize=14, fontweight="bold", pad=26)
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.tick_params(axis="y", length=0)

    # --- 3: the bias over all 42 cases ---
    t = tab.sort_values("bias_conus").reset_index(drop=True)
    cols = [C_HEAT if f == "heat" else C_COLD for f in t.family]
    cx.barh(np.arange(len(t)), t.bias_conus, color=cols, height=0.8)
    k = int(t.index[t.episode_id == eid][0])
    cx.barh(k, t.bias_conus[k], color="none", edgecolor="black", lw=1.5, height=0.8)
    cx.text(t.bias_conus[k] - 0.05, k, f"{eid[:3]} ", ha="right", va="center", fontsize=10)
    cx.axvline(0, color="0.3", lw=0.8)
    mean = t.bias_conus.mean()
    cx.axvline(mean, color="black", ls="--", lw=1.2)
    cx.set_yticks([])
    cx.set_ylim(-1, len(t))
    neg = int((t.bias_conus < 0).sum())
    hm = t[t.family == "heat"].bias_conus.mean()
    cm = t[t.family == "cold"].bias_conus.mean()
    cx.text(0.98, 0.68, f"mean {mean:+.2f} K\ncold in {neg}/{len(t)}\n"
            f"heat {hm:+.2f}\ncold {cm:+.2f}", transform=cx.transAxes, ha="right",
            va="top", fontsize=11.5, bbox=dict(fc="white", ec="0.8", pad=4))
    from matplotlib.patches import Patch
    cx.legend(handles=[Patch(color=C_HEAT, label="heat case"),
                       Patch(color=C_COLD, label="cold case")],
              loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=10.5, frameon=False)
    cx.set_xlim(t.bias_conus.min() - 0.1, max(t.bias_conus.max(), 0) + 1.9)
    cx.set_xlabel("CFS bias, CONUS (K)", fontsize=13)
    cx.set_title(f"3. CFSv2 runs cold\n    ({len(t)} cases, 21-25 d lead)", loc="left",
                 fontsize=14, fontweight="bold", pad=26)
    for side in ("top", "right", "left"):
        cx.spines[side].set_visible(False)

    fig.text(0.01, 0.015,
             f"Case {eid}. Bias = mean over the other years of (16-member CFS mean - ERA5), "
             f"same calendar date and lead, leave-one-year-out. Corrected member = raw - bias.\n"
             f"Maps use the same correction per grid cell. Anomalies are vs ERA5 1990-2019, "
             f"so the bias also absorbs the warming since then.", fontsize=10.5, color="0.35")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.17)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"acal_cfs_bias_correction_{eid}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[cfs_bias_figure] wrote {out}  bias {b:+.3f}  reach raw {n_raw} corr {n_cor}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", default="e02_c4_20210218")
    main(p.parse_args().case)
