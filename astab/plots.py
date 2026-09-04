#!/usr/bin/env python
"""stage: figs - the four-panel stability curve, ``figures/astab/stability.png``.

One line PER CHAIN, never per (model, event). Grouping by (model, event) alone puts four
different leads' chains on a single line: they share ``checkpoint_day`` values, so sorting
by it interleaves them and a diverged week-8 chain gets stitched to a healthy week-10 one
into a sawtooth that is not any trajectory. ``weeks`` is what makes a chain a chain.

Every divergence is marked with a red X on EVERY panel, including the panels whose own
metric never moved. Two of job 1189's three divergences were confined to the polar
stratosphere and are invisible to every CONUS item and to the z500 spectrum; without the
marker, panel 4 reads as "all eight chains healthy to 70 d", which is false.

The walker is drawn only at the leads it was rolled to the peak at. At an FCN3-extension
lead its whole panel is one checkpoint at 3 d, and a lone blue marker sitting at x = 3
beside a 140-day FCN3 curve invites exactly the reading the sweep must not support - that
the walker was out there too. ``reduce.untested_weeks`` is the same rule the headline uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from astab import sconfig as SC
from astab.diag import BROKEN_Z500_M, HI_WAVENUMBER, MARGINAL_SEVERITY, load_bands
from astab.reduce import _all_clean_through, _model_survival, csv_path, untested_weeks
from astab.schedule import Chain


def fig_path() -> Path:
    return SC.fig_path()

MODEL_STYLE = {
    "gencast":      ("#1f77b4", "o-", "GenCast walker"),
    "fcn3_era5":    ("#9467bd", "s--", "FCN3 from an ERA5 IC (adapter-free)"),
    "fcn3_adapter": ("#2ca02c", "^:", "FCN3 from the walker at 3 d (adapter-fed)"),
}
EVENT_MARKER = {"PNW_HeatDome_2021": 1.0, "WinterStorm_Uri_2021": 0.55}


def _diverged_at(df: pd.DataFrame, model: str, event: str, weeks) -> float | None:
    """Lead at which this chain's bound excursion first exceeded ``MARGINAL_SEVERITY``."""
    c = df[(df.model == model) & (df.event == event) & (df.weeks == weeks)
           & (df.metric == "bounds_severity")]
    big = c[c.value.astype(float) > MARGINAL_SEVERITY]
    return float(big.checkpoint_day.min()) if len(big) else None


def stage_figs(chs: list[Chain]) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = csv_path()
    if not p.exists():
        print(f"[figs] ERROR: no {p}; run --stage reduce first", file=sys.stderr)
        return 1
    df = pd.read_csv(p)
    events = sorted(df.event.unique())
    untested = {m: set(untested_weeks(df, m)) for m in MODEL_STYLE}
    bands = {ev: load_bands(ev) for ev in events}
    maxcal = max((b.max_lead_days for bs in bands.values() for b in bs.values()),
                 default=21.0)

    panels = [
        ("t2m_anom_conus", "CONUS T2m anomaly  (K)",
         "1. the -26 K drift signature"),
        ("diurnal_ratio", "diurnal amplitude / climatology",
         "2. the 0.03 K collapse (a ratio, so it self-calibrates by season)"),
        ("global_z500", "global-mean z500 drift  (m)",
         "3. drift from the chain's own start; ERA5 where available"),
        ("z500_hi_wavenumber_frac", f"power fraction above k = {HI_WAVENUMBER}",
         "4. high-wavenumber pile-up (the classic AI-weather instability)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6), sharex=True)
    for ax, (metric, ylab, title) in zip(axes.ravel(), panels):
        sub = df[df.metric == metric]
        drew = False
        for model, (color, style, label) in MODEL_STYLE.items():
            for ev in events:
                seen_label = False
                # ONE LINE PER CHAIN. Grouping by (model, event) alone puts four
                # different leads' chains on a single line: they share checkpoint_day
                # values, so sorting by it interleaves them and a diverged week-8 chain
                # gets stitched to a healthy week-10 one into a sawtooth that is not any
                # trajectory. `weeks` is what makes a chain a chain.
                for wk in sorted(sub[(sub.model == model)
                                     & (sub.event == ev)].weeks.unique()):
                    if int(wk) in untested[model]:
                        # The walker at an FCN3-extension lead: three days of rollout,
                        # drawn as a lone marker at x = 3 next to curves that ran to 140 d.
                        # It is not a trajectory and it is not this figure's subject.
                        continue
                    c = sub[(sub.model == model) & (sub.event == ev)
                            & (sub.weeks == wk)].sort_values("checkpoint_day")
                    if c.empty or not np.isfinite(c.value.astype(float)).any():
                        continue
                    y = c.value.astype(float).values
                    if metric == "global_z500":
                        y = (y - y[0]) / 9.80665         # m2 s-2 -> m, from its own start
                    ax.plot(c.checkpoint_day.values, y, style, color=color, lw=1.4, ms=3.5,
                            alpha=EVENT_MARKER.get(ev, 0.8),
                            label=(None if seen_label else f"{label}  [{ev.split('_')[0]}]"))
                    seen_label = True
                    drew = True
                    # Mark where THIS chain diverged, on EVERY panel - including the ones
                    # whose own metric never moved. Two of job 1189's three divergences
                    # were confined to the polar stratosphere and are invisible to every
                    # CONUS item and to the z500 spectrum; without this marker those
                    # panels read as "all eight chains healthy to 70 d", which is false.
                    d = _diverged_at(df, model, ev, wk)
                    if d is not None:
                        yy = np.interp(d, c.checkpoint_day.values.astype(float), y)
                        ax.plot([d], [yy], "X", color="#d62728", ms=11, mew=1.4,
                                mec="white", zorder=8)
        for ev in events:
            b = bands[ev].get(metric)
            if b and np.isfinite(b.lo):
                ax.axhspan(b.lo, b.hi, color="0.75", alpha=0.18, lw=0, zorder=0)
        if metric == "global_z500":
            ax.axhline(BROKEN_Z500_M, color="#d62728", lw=1.2, ls="--", zorder=2)
            ax.text(0.99, BROKEN_Z500_M, "the broken run (-1780 m2 s-2)  ",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=8, color="#d62728")
        if metric == "diurnal_ratio":
            ax.axhline(1.0, color="0.5", lw=1.0, ls="-", zorder=1)
        # The frozen operating point, on every panel.
        ax.axvline(21.0, color="#ff7f0e", lw=1.2, ls="-.", zorder=2)
        ax.text(21.0, 0.02, " week-3: every AI+RES run to date", rotation=90, fontsize=8,
                color="#ff7f0e", va="bottom", transform=ax.get_xaxis_transform())
        ax.axvspan(maxcal, max(df.checkpoint_day.max(), 70), color="#8c564b", alpha=0.06,
                   lw=0, zorder=0)
        ax.text(maxcal + 1, 0.985, f" band measured to {maxcal:g} d; extrapolated beyond",
                transform=ax.get_xaxis_transform(), fontsize=8, color="#8c564b",
                va="top")
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        if not drew:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                    color="0.5")
    for ax in axes[1]:
        ax.set_xlabel("lead (days from each chain's own init)")
    from matplotlib.lines import Line2D
    h, lb = axes[0, 0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="X", color="#d62728", ls="none", ms=9, mec="white"))
    lb.append("chain diverged here (physical bounds)")
    axes[0, 0].legend(h, lb, fontsize=7.5, loc="best", ncol=1, framealpha=0.9)

    head = _fig_headline(df)
    ext = sorted(untested["gencast"])
    cover = (f"\nThe walker is drawn only where it was rolled to the peak; at week(s) "
             f"{ext} it rolls {SC.SCORE_LEAD_DAYS:g} d to launch the adapter-fed arm and "
             f"is NOT a curve here." if ext else "")
    fig.suptitle(
        f"AI+RES lead-time stability sweep - how far the walker and the scorer still make "
        f"an atmosphere\n{head}\n"
        f"n = 1 member of each model per lead; bands measured on the Gate 3 walkers to "
        f"{maxcal:g} d and EXTRAPOLATED beyond. Numerical stability only: this says "
        f"nothing about whether the score still RANKS.{cover}\n"
        f"TWO failure modes. LOUD (PNW wk8): every panel fires together from 42 d. "
        f"SILENT (Uri wk8/wk10): the polar stratosphere reaches 1211 K while CONUS and "
        f"z500 stay textbook-normal - only the bounds check (X) sees it.",
        fontsize=10.5, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    out = fig_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return 0


def _fig_headline(df: pd.DataFrame) -> str:
    """The same all-chains rule the printed headline uses; never a max over chains.

    And the same tested-to-the-peak rule: a week the walker never ran to cannot appear in
    ``g``, because ``_model_survival`` has already dropped it.
    """
    g = _all_clean_through(_model_survival(df, "gencast"))
    f1 = _all_clean_through(_model_survival(df, "fcn3_era5"))
    f2 = _all_clean_through(_model_survival(df, "fcn3_adapter"))
    f = min([w for w in (f1, f2) if w is not None], default=None)
    gw = sorted(_model_survival(df, "gencast"))
    fw = sorted(_model_survival(df, "fcn3_era5"))
    return (f"GenCast stable through week {g if g else '-'} (tested to wk "
            f"{max(gw) if gw else '-'}), FCN3 through week {f if f else '-'} (tested to "
            f"wk {max(fw) if fw else '-'})  -  every chain; stable is not useful")


