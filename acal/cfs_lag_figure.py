#!/usr/bin/env python
"""Slide figure: how the 16-member CFSv2 lagged ensemble is built, on one real case.

Reads the per-case JSON `acal.cfsbase --stage build` writes; CPU, seconds.

    python -m acal.cfs_lag_figure                          # e02_c4_20210218
    python -m acal.cfs_lag_figure --case e14_h4_20230104
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import cfsbase as CB

FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "acal"
C_MEM = "#2b6cb0"
C_SUB = "#7b3fa0"
C_WIN = "#c27a12"
C_WIN_SOFT = "#f7ead3"


def main(eid: str) -> Path:
    d = json.loads(CB.json_path(eid).read_text())
    cyc = pd.to_datetime(d["cycles"])
    lead = np.asarray(d["member_lead_days"], float)
    al = np.asarray(d["al"], float)
    obs, s = float(d["obs"]), float(d["sign"])
    init = cyc[-1]
    peak = init + pd.Timedelta(days=float(lead[-1]))
    w0 = peak - pd.Timedelta(days=6)
    n = len(al)
    sub = np.arange(n) >= n - CB.N_SUBSET
    reach = s * al >= s * obs

    plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.33, 6.4),
                                 gridspec_kw={"width_ratios": [2.1, 1], "wspace": 0.08})

    # --- left: the 16 cycles and the shared verification window ---
    y = np.arange(n)[::-1]
    ax.axvspan(w0, peak, color=C_WIN_SOFT, zorder=0)
    ax.axvline(w0, color=C_WIN, lw=1)
    ax.axvline(peak, color=C_WIN, lw=1)
    ax.axvline(init, color="0.2", lw=1.2, ls="--")
    for yi, t, sb in zip(y, cyc, sub):
        col = C_SUB if sb else C_MEM
        ax.plot([t, w0], [yi, yi], color=col, lw=1, alpha=0.45)
        ax.plot([w0, peak], [yi, yi], color=col, lw=5, solid_capstyle="butt")
        ax.plot(t, yi, "o", color=col, ms=6)
    ax.set_yticks(y, [f"{t:%m-%d %HZ}" for t in cyc], fontsize=10.5, family="monospace")
    for lab, sb in zip(ax.get_yticklabels(), sub):
        lab.set_color(C_SUB if sb else "0.25")
    ax.set_ylim(-1.9, n - 0.2 + 1.6)
    ax.set_xlim(cyc[0] - pd.Timedelta(hours=18), peak + pd.Timedelta(hours=18))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[24, 28, 1, 5, 9, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", labelsize=12)
    ax.text(init, n + 0.4, " AI+RES init", ha="left", va="bottom", fontsize=12)
    ax.text(w0 + (peak - w0) / 2, n + 0.4, "verification week", ha="center",
            va="bottom", fontsize=12, color=C_WIN)
    mid = cyc[0] + (w0 - cyc[0]) / 2
    ax.annotate("", xy=(peak, -1.35), xytext=(cyc[0], -1.35),
                arrowprops=dict(arrowstyle="<->", color="0.4", lw=1))
    ax.text(mid, -1.2, f"lead {lead[0]:.2f} d (oldest) to {lead[-1]:.1f} d (init)",
            ha="center", va="bottom", fontsize=11.5, color="0.3")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel(f"CFSv2 cycle ({cyc[0]:%Y}, UTC) = one member", fontsize=13)
    ax.set_title(f"{n} trailing 6-hourly cycles, one target week", loc="left",
                 fontsize=15, fontweight="bold")

    # --- right: member A_L on the same rows ---
    bx.axvspan(-20 if s < 0 else obs, obs if s < 0 else 20, color=C_WIN_SOFT, zorder=0)
    bx.axvline(obs, color=C_WIN, lw=2)
    bx.axvline(al.mean(), color="0.2", lw=1.2, ls="--")
    bx.axvline(0, color="0.7", lw=0.8)
    for yi, v, sb, r in zip(y, al, sub, reach):
        col = C_SUB if sb else C_MEM
        bx.plot([0, v], [yi, yi], color=col, lw=1, alpha=0.4)
        bx.plot(v, yi, "o", color=col, ms=12 if r else 9, mec=C_WIN if r else col,
                mew=3 if r else 1, zorder=3)
    lo = min(al.min(), obs) - 1.0
    hi = max(al.max(), obs, 0) + 1.0
    bx.set_xlim(lo, hi)
    bx.set_ylim(ax.get_ylim())
    bx.set_yticks([])
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.set_xlabel("CONUS 7-day mean T2m anomaly $A_L$ (K)", fontsize=13)
    bx.text(obs + 0.12, n + 0.5, f"ERA5 obs {obs:+.2f}", ha="left", va="center",
            fontsize=12, color=C_WIN, bbox=dict(fc="white", ec="none", pad=1))
    bx.text(al.mean() + 0.12, n - 0.35, f"mean {al.mean():+.2f}", ha="left", va="center",
            bbox=dict(fc="white", ec="none", pad=1),
            fontsize=11, color="0.2")
    k = int(reach.sum())
    bx.set_title(f"P(obs tail) = {k}/{n} = {k / n:.4g}", loc="left", fontsize=15,
                 fontweight="bold")

    fam = d["family"]
    fig.text(0.01, 0.015,
             f"Case {eid} ({fam}). Each member scored with the same CONUS $A_L$ as the "
             f"walkers (anomaly vs ERA5 1990-2019, no bias correction).\n"
             f"Purple = last {CB.N_SUBSET} cycles (earlier aires subset). Amber ring = "
             f"member reaches the observed tail.", fontsize=10.5, color="0.35")
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.16)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"acal_cfs_lagged_ensemble_{eid}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[cfs_lag_figure] wrote {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", default="e02_c4_20210218")
    main(p.parse_args().case)
