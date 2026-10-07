#!/usr/bin/env python
"""Where the 42 episodes sit in the 2021-2025 distribution of CONUS-wide A_L. CPU ONLY.

The population is every peak day of `runs/acal/catalog/conus_daily_2021_2025.csv`
(1,826 days, `a_l_conus`, the 7-day cos(lat)-weighted CONUS T2m anomaly vs 1990-2019).
The episodes are `conus_episodes_21d_2021_2025.csv`. A percentile here is the empirical
non-exceedance rank in that population, mid-rank for ties:

    pct(x) = 100 * (#{A_L < x} + 0.5 #{A_L == x}) / N

so a cold episode sits near 0 and a heat episode near 100. The +/-2/3/4 K rungs are drawn
as dotted lines labelled with the same pct.

Outputs: `figures/acal/acal_pdf_events.png` and `runs/acal/catalog/episode_percentiles.csv`.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from acal import ccfg

CAT = ccfg.ROOT / "runs" / "acal" / "catalog"
DAILY = CAT / "conus_daily_2021_2025.csv"
EPIS = CAT / "conus_episodes_21d_2021_2025.csv"
OUT_CSV = CAT / "episode_percentiles.csv"
OUT_FIG = ccfg.FIG_ROOT / "acal_pdf_events.png"

RUNGS = (-4.0, -3.0, -2.0, 2.0, 3.0, 4.0)
C_HEAT, C_COLD, C_PDF = "#c0392b", "#2463a6", "#444444"


def percentile(pop: np.ndarray, x) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, float))
    lt = (pop[None, :] < x[:, None]).sum(1)
    eq = (pop[None, :] == x[:, None]).sum(1)
    return 100.0 * (lt + 0.5 * eq) / pop.size


def fmt_pct(p: float) -> str:
    return f"{p:.2f}" if (p < 1 or p > 99) else f"{p:.1f}"


def plot(pop: np.ndarray, ep: pd.DataFrame) -> plt.Figure:
    lo, hi = np.floor(pop.min()) - 0.5, np.ceil(pop.max()) + 0.5
    xs = np.linspace(lo, hi, 800)
    kde = gaussian_kde(pop)
    pdf = kde(xs)

    ep = ep.sort_values("a_l_conus").reset_index(drop=True)
    n = len(ep)
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(11, 4.2 + 0.24 * n), sharex=True,
        gridspec_kw={"height_ratios": [4.2, 0.24 * n], "hspace": 0.04})

    # (a) the PDF
    ax.hist(pop, bins=np.arange(lo, hi + 0.25, 0.25), density=True,
            color="#d9d9d9", edgecolor="white", lw=0.4, label=f"daily A_L, N={pop.size}")
    ax.plot(xs, pdf, color=C_PDF, lw=1.6, label="KDE")
    ytop = pdf.max() * 1.22
    for r in RUNGS:
        p = percentile(pop, r)[0]
        for a in (ax, bx):
            a.axvline(r, color="k", ls=":", lw=1.1, zorder=1)
        ax.text(r, ytop * 0.985, f"{r:+.0f} K\np{fmt_pct(p)}", ha="center", va="top",
                fontsize=8.5, bbox=dict(fc="white", ec="none", pad=1.2))
    for fam, c in (("heat", C_HEAT), ("cold", C_COLD)):
        s = ep[ep.family == fam]
        ax.plot(s.a_l_conus, kde(s.a_l_conus), "o", ms=5, mfc=c, mec="white", mew=0.6,
                label=f"{fam} episodes ({len(s)})", zorder=4)
    ax.set_ylim(0, ytop)
    ax.set_ylabel("probability density (1/K)")
    ax.set_title("CONUS-wide 7-day T2m anomaly A_L, 2021-2025 daily distribution, "
                 "and the 42 episodes (pN = percentile of daily A_L)", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, bbox_to_anchor=(0.0, 0.86))

    # (b) one row per episode, sorted by A_L, labelled with its percentile
    for i, r in ep.iterrows():
        c = C_HEAT if r.family == "heat" else C_COLD
        bx.plot([0, r.a_l_conus], [i, i], color=c, lw=0.8, alpha=0.5)
        bx.plot(r.a_l_conus, i, "o", ms=4.5, color=c)
        txt = f"{r.peak}  {r.a_l_conus:+.2f} K  p{fmt_pct(r.pct)}"
        if r.a_l_conus < 0:
            bx.text(r.a_l_conus - 0.12, i, txt, ha="right", va="center", fontsize=7.3, zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.6))
        else:
            bx.text(r.a_l_conus + 0.12, i, txt, ha="left", va="center", fontsize=7.3, zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.6))
    bx.axvline(0, color="#999999", lw=0.6)
    bx.set_ylim(n - 0.4, -0.6)
    bx.set_yticks([])
    bx.set_xlim(lo - 3.0, hi + 3.0)
    bx.set_xlabel("A_L, CONUS-wide 7-day mean T2m anomaly vs 1990-2019 (K)")
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
    bx.spines["left"].set_visible(False)
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    daily = pd.read_csv(DAILY)
    pop = daily["a_l_conus"].to_numpy(float)
    ep = pd.read_csv(EPIS)
    # the episode CSV rounds A_L to 3 decimals; take the daily value at the peak so an
    # episode ranks against its own day exactly
    ep["a_l_conus"] = ep.peak.map(daily.set_index("date")["a_l_conus"])
    ep["pct"] = percentile(pop, ep.a_l_conus.to_numpy())
    ep[["episode_id", "family", "rung", "peak", "a_l_conus", "pct"]] \
        .sort_values("a_l_conus").to_csv(OUT_CSV, index=False, float_format="%.3f")
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    fig = plot(pop, ep)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_FIG}\nwrote {OUT_CSV}")
    for r in RUNGS:
        print(f"  {r:+.0f} K -> p{fmt_pct(percentile(pop, r)[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
