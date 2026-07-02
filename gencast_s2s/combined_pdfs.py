"""The three project-root "combined" T2m-anomaly PDF figures, overlaid across horizons.

Each figure is one subplot per out-of-sample event, with the observed truth (ERA5 and,
when available, HRRR) drawn against the GenCast weeks-2/3/4 forecast distributions:

    combinedpdf.png        full 24-member ensemble PDF per week
    combinedbestpdf.png    single best member (lowest weighted RMSE vs obs) per week
    combinedextremepdf.png single most-extreme member (warmest/coldest) per week

These mirror the per-week ``figures/pdf/all_events_pdf.png`` panels but overlay all three
horizons on one axis. Run as a module to (re)write the PNGs at the project root:

    python -m gencast_s2s.combined_pdfs                 # all three, weeks 2/3/4
    python -m gencast_s2s.combined_pdfs --outdir figs   # elsewhere
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

from . import config as C                  # noqa: E402
from .plotting import (                     # noqa: E402
    YMIN, _truth, _hrrr_truth, _forecast, _best_member, _extreme_member,
    _pdf_on_grid, _event_sign,
)

WEEKS = (2, 3, 4)
WEEK_COLOR = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}


def _events(model="gencast"):
    """OOS events that have a week-2 forecast and an ERA5 truth (figure row order)."""
    out = []
    for name in C.EVENTS:
        if (C.OBS_DIR / f"{name}_verif_t2m_anom.nc").exists() and \
           _forecast(name, 2, model=model) is not None:
            out.append(name)
    return out


def _week_curve(name, weeks, mode, truth, model="gencast"):
    """(values, label) for one week's forecast curve under the chosen mode, or None."""
    fc = _forecast(name, weeks, model=model)
    if fc is None:
        return None
    if mode == "ensemble":
        nm = int(fc.sizes["member"])
        return fc.values.ravel(), f"GenCast wk-{weeks} ({nm} mem)"
    if mode == "best":
        member, idx, _ = _best_member(fc, truth)
        return member.values.ravel(), f"Week {weeks} best member (#{idx})"
    if mode == "extreme":
        member, idx, _ = _extreme_member(fc, name)
        kind = "warmest" if _event_sign(name) > 0 else "coldest"
        return member.values.ravel(), f"Week {weeks} {kind} member (#{idx})"
    raise ValueError(mode)


def _panel(ax, name, mode, model="gencast"):
    truth = _truth(name)
    hrrr = _hrrr_truth(name)
    obs = {"truth": (truth.values.ravel(), dict(color="k", lw=2.5, label="Observed (ERA5)"))}
    if hrrr is not None:
        obs["hrrr"] = (hrrr.values.ravel(),
                       dict(color="tab:red", lw=2.0, label="Observed (HRRR)"))

    curves = []  # (values, color, ls, label)
    for w in WEEKS:
        res = _week_curve(name, w, mode, truth, model=model)
        if res is None:
            continue
        vals, label = res
        ls = "-" if mode == "ensemble" else "--"
        curves.append((vals, WEEK_COLOR[w], ls, label))

    allv = np.concatenate([v[np.isfinite(v)] for v, _ in obs.values()]
                          + [v[np.isfinite(v)] for v, *_ in curves])
    xgrid = np.linspace(allv.min() - 1, allv.max() + 1, 400)

    for vals, st in obs.values():
        d = np.clip(_pdf_on_grid(vals, xgrid), YMIN, None)
        ax.plot(xgrid, d, **st)
    for vals, color, ls, label in curves:
        d = np.clip(_pdf_on_grid(vals, xgrid), YMIN, None)
        ax.plot(xgrid, d, color=color, ls=ls, lw=2.0, label=label)

    ax.axvline(0, color="grey", lw=0.8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(YMIN, None)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("week T2m anomaly (K)")
    ax.set_ylabel("density")


def _figure(mode, title, fname, outdir, model="gencast", legend_each=True):
    names = _events(model=model)
    if not names:
        print(f"[combined-pdf] no events with forecasts; skipping {fname}")
        return
    cols = 3
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        _panel(ax, name, mode, model=model)
        if legend_each:
            ax.legend(fontsize=7)
    if not legend_each:
        axes.ravel()[0].legend(fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    out = Path(outdir) / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[combined-pdf] {len(names)} events -> {out}")


def make_all(outdir=None, model="gencast"):
    outdir = Path(outdir) if outdir is not None else C.ROOT
    outdir.mkdir(parents=True, exist_ok=True)
    _figure("ensemble",
            "T2m anomaly PDFs over CONUS — GenCast weeks 2/3/4 overlaid vs observed",
            "combinedpdf.png", outdir, model=model, legend_each=False)
    _figure("best",
            "CONUS T2m anomaly PDFs: observed vs. best ensemble member at weeks 2/3/4",
            "combinedbestpdf.png", outdir, model=model, legend_each=True)
    _figure("extreme",
            "CONUS T2m anomaly PDFs: observed vs. most-extreme ensemble member at weeks 2/3/4\n"
            "(heat events -> warmest member, cold events -> coldest member)",
            "combinedextremepdf.png", outdir, model=model, legend_each=True)


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Regenerate the three combined-PDF root figures.")
    p.add_argument("--outdir", default=None, help="output dir (default: project root)")
    p.add_argument("--model", default="gencast")
    args = p.parse_args(argv)
    make_all(outdir=args.outdir, model=args.model)


if __name__ == "__main__":
    main()
