"""The combined PDF figures: ERA5 (both grids) vs GenCast 0.25deg vs 1.0deg, one panel per event.

Each figure overlays, for every event (drawn in that event's headline metric), four spatial
PDFs over CONUS:

    ERA5 0.25deg truth          dark red,  SOLID
    ERA5 1.0deg truth           dark blue, SOLID   (0.25deg truth sampled onto the 1.0deg grid)
    GenCast 0.25deg forecast    light red, DOTTED
    GenCast 1.0deg forecast     light blue, DOTTED

Color families pair each forecast with the truth at its own resolution (reds = 0.25deg,
blues = 1.0deg); light-vs-dark and solid-vs-dotted encode the truth / forecast
distinction redundantly for color-vision-deficient readers.

PDFs are binned-histogram estimates via ``PDF_histogram`` — ported from
``FiguresProduction.ipynb`` of https://github.com/moeindarman77/TransferLearning-QG so both
projects generate PDFs the same way: ``np.histogram`` with 100 bins over a mean +/- 4 sigma
range, normalized as counts/N (probability per bin). The only departures are that the bin
range is shared by all curves in a panel (so the probabilities are comparable bin-for-bin)
and the x axis stays in physical units instead of the repo's x/sigma standardization.

The figures differ only in HOW each resolution's ensemble is collapsed to a single
forecast curve:

    xres_combined_mean.png      ensemble-MEAN field
    xres_combined_pooled.png    ALL members pooled into one distribution (matches the
                                 original experiment's combinedpdf.png)
    xres_combined_best.png      single BEST member (lowest latitude-weighted RMSE vs ERA5)
    xres_combined_extreme.png   single most-EXTREME member (warmest/coldest for T2m;
                                 strongest winds / heaviest precip otherwise)

Each also has a ``*_vs_era5_0p25.png`` variant where the ERA5 1.0deg curve is dropped and
the single native ERA5 0.25deg curve serves as the common baseline for both forecasts.
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
from .xplotting import era5_truth, members, _on   # noqa: E402

YMIN = 1e-6                            # floor of the log-y PDF axis
NBINS = 100                            # TransferLearning-QG default


def PDF_histogram(x, xmin=None, xmax=None, Nbins=NBINS):
    """Binned-histogram PDF, ported from TransferLearning-QG ``FiguresProduction.ipynb``.

    Returns ``(bin_centers, hist/N)`` — the probability of falling in each bin (the repo
    keeps the ``/bandwidth`` density conversion commented out, so we do too). Unlike the
    original we return physical bin centers (no ``points/sigma`` standardization) so
    curves in different units/spreads can share one axis.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    N = x.shape[0]

    mean = x.mean()
    sigma = x.std()

    if xmin is None:
        xmin = mean - 4 * sigma
    if xmax is None:
        xmax = mean + 4 * sigma

    hist, bin_edges = np.histogram(x, range=(xmin, xmax), bins=Nbins)
    density = hist / N
    points = (bin_edges[0:-1] + bin_edges[1:]) * 0.5
    return points, density


def _best_member(mem, truth):
    obs = _on(truth, mem)
    w = np.cos(np.deg2rad(mem["lat"]))
    rmse = np.sqrt(((mem - obs) ** 2).weighted(w).mean(("lat", "lon")))
    idx = int(rmse.argmin("member"))
    return mem.isel(member=idx), idx


def _extreme_member(mem, sign):
    w = np.cos(np.deg2rad(mem["lat"]))
    ma = mem.weighted(w).mean(("lat", "lon"))
    idx = int((sign * ma).argmax("member"))
    return mem.isel(member=idx), idx


def _res_curve(res, weeks, name, metric, mode, era5):
    mem = members(res, weeks, name, metric)
    if mem is None:
        return None
    n = mem.sizes["member"]
    if mode == "ensemble":
        field, tag = mem.mean("member"), f"mean ({n} members)"
    elif mode == "pooled":
        field, tag = mem, f"({n} mem pooled)"
    elif mode == "best":
        if era5 is None:
            return None
        field, idx = _best_member(mem, era5); tag = f"best #{idx} of {n}"
    elif mode == "extreme":
        field, idx = _extreme_member(mem, XM.extreme_sign(metric, name))
        tag = f"extreme #{idx} of {n}"
    else:
        raise ValueError(mode)
    return field.values.ravel(), f"{X.res_spec(res)['label']} wk-{weeks} {tag}"


def _era5_1p0(era5, weeks, name, metric):
    """ERA5 truth on the 1.0deg grid: sampled onto the 1.0deg forecast grid when it
    exists, else a stride-4 subsample of the native 0.25deg field."""
    mem = members("1p0", weeks, name, metric)
    if mem is not None:
        return _on(era5, mem)
    return era5.isel(lat=slice(None, None, 4), lon=slice(None, None, 4))


def _panel(ax, name, weeks, mode, baseline="both"):
    metric = X.event_metric(name)
    sp = XM.spec(metric)
    era5 = era5_truth(name, metric)

    r25, r10 = X.res_spec("0p25"), X.res_spec("1p0")
    entries = [
        (era5.values.ravel(),
         dict(color=r25["truth_color"], ls="-", lw=2.8,
              label="ERA5 0.25deg (truth)")),
    ]
    if baseline == "both":
        entries.append(
            (_era5_1p0(era5, weeks, name, metric).values.ravel(),
             dict(color=r10["truth_color"], ls="-", lw=2.0,
                  label="ERA5 1.0deg (truth)")))
    for res in X.RES_ORDER:
        res_c = _res_curve(res, weeks, name, metric, mode, era5)
        if res_c is None:
            continue
        vals, label = res_c
        rs = X.res_spec(res)
        # Dense dot pattern + a white under-stroke so the light forecast curves stay
        # visible where they cross the dark truth curves and the gridlines.
        entries.append((vals, dict(
            color=rs["color"], ls=(0, (1, 1)), lw=2.8,
            path_effects=[pe.Stroke(linewidth=4.4, foreground="white"),
                          pe.Normal()],
            label=label)))

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


def _figure(mode, title, fname, weeks, outdirs, baseline="both"):
    names = [n for n in X.events()
             if any(members(r, weeks, n, X.event_metric(n)) is not None for r in X.RES_ORDER)]
    if not names:
        print(f"[xres combined] no forecasts yet; skipping {fname}")
        return
    cols = 3
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        _panel(ax, name, weeks, mode, baseline=baseline)
        ax.legend(fontsize=6.5)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    for outdir in outdirs:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(outdir) / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[xres combined] {len(names)} events -> {fname}")


MODES = {
    "ensemble": ("ENSEMBLE MEAN", "mean"),
    "pooled":   ("ALL MEMBERS POOLED", "pooled"),
    "best":     ("BEST MEMBER", "best"),
    "extreme":  ("MOST-EXTREME MEMBER", "extreme"),
}


def make_all(weeks=None):
    weeks = X.WEEKS if weeks is None else weeks
    outdirs = [X.xfig_dir(weeks), C.ROOT / "figures" / "xres" / f"week{weeks}"]
    for mode, (desc, tag) in MODES.items():
        # Batch 1: truth at both grids (own-grid baseline for each resolution).
        _figure(mode,
                f"CONUS verification PDFs — {desc}, week-{weeks}: "
                "ERA5 (0.25deg + 1.0deg) vs GenCast 0.25deg vs 1.0deg",
                f"xres_combined_{tag}.png", weeks, outdirs, baseline="both")
        # Batch 2: the single ERA5 0.25deg curve as the common baseline for both.
        _figure(mode,
                f"CONUS verification PDFs — {desc}, week-{weeks}: "
                "common baseline ERA5 0.25deg vs GenCast 0.25deg vs 1.0deg",
                f"xres_combined_{tag}_vs_era5_0p25.png", weeks, outdirs,
                baseline="0p25")
