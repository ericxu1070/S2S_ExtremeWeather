"""Lead-time comparison PDFs: one figure per RESOLUTION, overlaying week-2/3/4 forecasts.

The transpose of xcombined's per-week figures: instead of fixing the lead and comparing
resolutions, each figure fixes the resolution (GenCast 0.25deg or 1.0deg) and overlays,
for every event (drawn in that event's headline metric), the POOLED-member spatial PDFs
at all three lead times against the ERA5 truth on that resolution's grid:

    ERA5 truth (native 0.25deg / sampled onto the 1.0deg forecast grid)   dark,   SOLID
    GenCast week-2 forecast (all members pooled)                          light,  dotted
    GenCast week-3 forecast (all members pooled)                          medium, dash-dot
    GenCast week-4 forecast (all members pooled)                          dark,   dashed

Colors stay in the resolution's family (reds = 0.25deg, blues = 1.0deg, as everywhere in
xres) with lightness encoding the lead; the dash pattern is a redundant lead encoding for
color-vision-deficient readers. Truth is week-independent (the verification week is
anchored to the event peak), so one truth curve serves all three leads.

PDFs are pooled all-members histograms via xcombined.PDF_histogram with a bin range
shared by every curve in a panel — identical to xres_combined_pooled.png, so the two
figure families are directly comparable.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe    # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from gencast_s2s import config as C    # noqa: E402

from . import xconfig as X             # noqa: E402
from . import xmetrics as XM           # noqa: E402
from .xcombined import PDF_histogram, YMIN, _era5_1p0   # noqa: E402
from .xplotting import era5_truth, members              # noqa: E402

WEEKS_ORDER = (2, 3, 4)

# Light -> dark within each resolution's color family as the lead grows; distinct dash
# patterns as the redundant encoding (truth stays SOLID in the family's darkest shade).
LEAD_COLORS = {
    "0p25": {2: "#fcae91", 3: "#fb6a4a", 4: "#cb181d"},
    "1p0":  {2: "#9ecae1", 3: "#4292c6", 4: "#08519c"},
}
LEAD_DASHES = {2: (0, (1, 1)), 3: (0, (3, 1, 1, 1)), 4: (0, (4, 2))}


def _truth_entry(era5, res, name, metric):
    """ERA5 truth values + label on the grid matching ``res`` (native for 0.25deg;
    sampled onto the 1.0deg forecast grid — same recipe as xcombined — for 1.0deg)."""
    rs = X.res_spec(res)
    if res == "0p25":
        vals = era5.values.ravel()
        label = "ERA5 0.25deg (truth)"
    else:
        vals = _era5_1p0(era5, WEEKS_ORDER[0], name, metric).values.ravel()
        label = "ERA5 1.0deg (truth)"
    return vals, dict(color=rs["truth_color"], ls="-", lw=2.8, label=label)


def _panel(ax, name, res):
    metric = X.event_metric(name)
    sp = XM.spec(metric)
    era5 = era5_truth(name, metric)
    if era5 is None:
        ax.set_title(f"{name} (no truth)", fontsize=10)
        return

    entries = [_truth_entry(era5, res, name, metric)]
    for weeks in WEEKS_ORDER:
        mem = members(res, weeks, name, metric)
        if mem is None:
            continue
        n = mem.sizes["member"]
        # Dense dash pattern + a white under-stroke so the light short-lead curves stay
        # visible where they cross the dark truth curve and the gridlines (as xcombined).
        entries.append((mem.values.ravel(), dict(
            color=LEAD_COLORS[res][weeks], ls=LEAD_DASHES[weeks], lw=2.4,
            path_effects=[pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()],
            label=f"wk-{weeks} ({n} mem pooled)")))

    # One shared bin range per panel (mean +/- 4 sigma of everything drawn, clipped to
    # the data range) so every curve's per-bin probabilities are directly comparable.
    allv = np.concatenate([np.asarray(v)[np.isfinite(v)] for v, _ in entries])
    mu, sd = allv.mean(), allv.std()
    xmin = max(allv.min(), mu - 4 * sd)
    xmax = min(allv.max(), mu + 4 * sd)
    for vals, st in entries:
        pts, dens = PDF_histogram(vals, xmin=xmin, xmax=xmax)
        ax.plot(pts, np.clip(dens, YMIN, None), **st)

    if sp.diverging:
        ax.axvline(0, color="grey", lw=0.8, alpha=0.6)
    ax.grid(True, which="both", ls="--", color="0.85", lw=0.5)
    ax.set_yscale("log")
    ax.set_ylim(YMIN, 1.0)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel(f"{sp.label} ({sp.units})")
    ax.set_ylabel("PDF (probability per bin)")


def _figure(res, outdirs):
    names = [n for n in X.events()
             if any(members(res, w, n, X.event_metric(n)) is not None
                    for w in WEEKS_ORDER)]
    if not names:
        print(f"[xres leadpdfs] no {res} forecasts for any week; skipping")
        return
    rs = X.res_spec(res)
    cols = 3
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        _panel(ax, name, res)
        ax.legend(fontsize=6.5)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    leads = "/".join(f"{C.lead_days_for(w)}d" for w in WEEKS_ORDER)
    fig.suptitle(f"CONUS verification PDFs — ALL MEMBERS POOLED, {rs['label']}: "
                 f"week-2/3/4 leads (init = peak - {leads}) vs ERA5 truth",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fname = f"xres_pdf_leads_{res}.png"
    for outdir in outdirs:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(outdir) / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[xres leadpdfs] {len(names)} events -> {fname}")


def make_all():
    outdirs = [X.XFIG_DIR / "leads", C.ROOT / "figures" / "xres" / "leads"]
    for res in X.RES_ORDER:
        _figure(res, outdirs)


if __name__ == "__main__":
    make_all()
