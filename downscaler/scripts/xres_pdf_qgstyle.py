#!/usr/bin/env python
"""PDFs in the TransferLearning-QG house style (Darman et al.).

Reproduces the probability-density plots from
https://github.com/moeindarman77/TransferLearning-QG (FiguresProduction.ipynb):

  * PDF_histogram() is copied VERBATIM from that notebook -- the x-axis is the field
    standardised by its own sigma (points/sigma), and y = hist / N (probability per bin,
    100 bins), i.e. the same normalisation they plot and label "PDF".
  * plotting style matches cell 74: serif/LaTeX fonts, log-scale y-axis to expose the
    tails, a dashed 0.85 grid, legend(frameon=False), and the same line convention --
        FDNS  (truth / reference)  -> black solid,  lw 3
        NoSGS (no-model baseline)  -> gray,          lw 2
        model (the NN)             -> red dashed,     lw 2

  Mapping onto the downscaler comparison (same roles):
        HRRR 3 km truth        <-> FDNS   (black solid)
        GenCast 1° forecast    <-> NoSGS  (gray)          -- the coarse, no-downscaling field
        downscaled HRRR (pred) <-> model  (red dashed)

Standardising each series by its own sigma is exactly their method: it collapses mean and
variance and asks whether the *shape / tails* agree -- i.e. whether the downscaler recovers
HRRR-like fine-scale (heavy-tailed) statistics that the smooth coarse forecast lacks.

Reads the caches built by scripts/xres_downscale_demo.py (--stage prep + infer); no GPU.

    python scripts/xres_pdf_qgstyle.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "outputs", "xres_demo_cache")


# --------------------------------------------------------------------------------------
# VERBATIM from TransferLearning-QG / FiguresProduction.ipynb (cell 59).
# --------------------------------------------------------------------------------------
def PDF_histogram(x, xmin=None, xmax=None, Nbins=100):
    """
    x is 1D numpy array with data
    How to use:
        first apply without arguments
        Then adjust xmin, xmax, Nbins
    """
    N = x.shape[0]

    mean = x.mean()
    sigma = x.std()

    if xmin is None:
        xmin = mean - 4 * sigma
    if xmax is None:
        xmax = mean + 4 * sigma

    bandwidth = (xmax - xmin) / Nbins

    hist, bin_edges = np.histogram(x, range=(xmin, xmax), bins=Nbins)

    # hist / N is probability to go into bin
    density = hist / N

    # we assign one value to each bin
    points = (bin_edges[0:-1] + bin_edges[1:]) * 0.5

    return points / sigma, density


# --------------------------------------------------------------------------------------
# style (cell 74), with a graceful fall-back to mathtext when no LaTeX is installed.
# --------------------------------------------------------------------------------------
def _set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager  # noqa: F401
    from shutil import which

    use_tex = which("latex") is not None
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "mathtext.fontset": "cm",          # Computer Modern -> LaTeX look without a TeX install
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 12,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelweight": "bold",
    })
    return plt


# variable key, display transform, whether it is signed (subtract mean, like q) or
# positive-definite (like KE/Ens: keep xmin=0, no mean removal), standardised x-label,
# panel title, physical-units x-label.
VARS = [
    ("t2m",    lambda a: a - 273.15, "signed",
     r"$(T_{2m}-\overline{T_{2m}})/\sigma$", r"$T_{2m}$", r"$T_{2m}$ [$^{\circ}$C]"),
    ("wspd",   lambda a: a,          "positive",
     r"$|\mathbf{U}_{10}|/\sigma$", r"$|\mathbf{U}_{10}|$", r"$|\mathbf{U}_{10}|$ [m s$^{-1}$]"),
    ("precip", lambda a: a,          "positive",
     r"$P/\sigma$", r"$P$", r"$P$ [mm]"),
]

# (label, prep/pred key prefix, which cache, colour, linestyle, linewidth, alpha)
SERIES = [
    ("HRRR truth (3 km)",           "truth_",  "prep", "black", "solid",  3.0, 1.0),
    ("GenCast 1° forecast",         "coarse_", "prep", "gray",  "solid",  2.0, 1.0),
    ("downscaled HRRR (predicted)", "pred_",   "pred", "red",   "dashed", 2.0, 1.0),
]


def _pdf_for(values, kind):
    """Standardised (their) treatment: signed fields -> anomaly/sigma (q-like); positive
    fields -> value/sigma with xmin=0 (KE/Ens-like). x-axis is in units of sigma."""
    if kind == "signed":
        return PDF_histogram(values - values.mean())     # mean-removed fluctuation
    return PDF_histogram(values, xmin=0.0)                # positive-definite


def PDF_histogram_phys(x, xmin, xmax, Nbins=100):
    """Non-normalised PDF: same y = hist/N convention as PDF_histogram, but the x-axis is the
    field in its PHYSICAL units (bin centres are NOT divided by sigma). A common (xmin, xmax)
    is used across the three series so the physical distributions overlay on one axis."""
    N = x.shape[0]
    hist, edges = np.histogram(x, range=(xmin, xmax), bins=Nbins)
    return 0.5 * (edges[:-1] + edges[1:]), hist / N


def main():
    plt = _set_style()

    prep = np.load(os.path.join(CACHE, "idalia_prep.npz"))
    pred = np.load(os.path.join(CACHE, "idalia_pred.npz"))
    stores = {"prep": prep, "pred": pred}
    mask = prep["mask"].astype(bool)

    nvar = len(VARS)
    # two rows: (top) standardised by sigma -- the repo's method; (bottom) physical units.
    fig, axes = plt.subplots(2, nvar, figsize=(6.2 * nvar, 9.6))

    for j, (vkey, tf, kind, xlab_n, title, xlab_p) in enumerate(VARS):
        vals = {label: tf(stores[store][prefix + vkey].astype(np.float64))[mask]
                for label, prefix, store, *_ in SERIES}

        # ---------- row 0: standardised (normalized) ----------
        ax = axes[0, j]
        for label, prefix, store, color, ls, lw, alpha in SERIES:
            x, dens = _pdf_for(vals[label], kind)
            nz = dens > 0
            ax.plot(x[nz], dens[nz], label=label, color=color, linestyle=ls, linewidth=lw, alpha=alpha)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", color="0.85")
        ax.set_xlabel(xlab_n)
        ax.set_title("Idalia 2023 wk-3: " + title, fontsize=13)
        ax.set_ylim(1e-5, 1e0)
        ax.set_xlim(-4.5, 4.5) if kind == "signed" else ax.set_xlim(0, None)

        # ---------- row 1: physical units (non-normalized) ----------
        pooled = np.concatenate(list(vals.values()))
        if kind == "signed":
            # full data extent (+ small margin) so BOTH tails of the widest series (the HRRR
            # truth: fine-scale cold high-terrain and hot desert pixels) stay inside the axis.
            lo, hi = float(pooled.min()), float(pooled.max())
            pad = 0.03 * (hi - lo)
            xmin, xmax = lo - pad, hi + pad
        else:
            xmin, xmax = 0.0, float(np.percentile(pooled, 99.9 if vkey == "precip" else 99.5))
        axp = axes[1, j]
        for label, prefix, store, color, ls, lw, alpha in SERIES:
            x, dens = PDF_histogram_phys(vals[label], xmin, xmax)
            nz = dens > 0
            axp.plot(x[nz], dens[nz], label=label, color=color, linestyle=ls, linewidth=lw, alpha=alpha)
        axp.set_yscale("log")
        axp.grid(True, which="both", ls="--", color="0.85")
        axp.set_xlabel(xlab_p)
        axp.set_ylim(1e-5, 1e0)
        axp.set_xlim(xmin, xmax)

    # row labels via the leftmost y-axis; the rest keep the bare "PDF" label
    for a in list(axes[0, 1:]) + list(axes[1, 1:]):
        a.set_ylabel("PDF")
    axes[0, 0].set_ylabel(r"PDF   (standardized, $X/\sigma$)")
    axes[1, 0].set_ylabel("PDF   (physical units)")
    axes[0, 0].legend(frameon=False, fontsize=12)
    # window-mismatch caveat, only relevant to the physical precip magnitudes
    axes[1, nvar - 1].text(0.97, 0.96, "coarse fcst $=$ 12 h accum;\nHRRR $=$ 6 h",
                           transform=axes[1, nvar - 1].transAxes, ha="right", va="top",
                           fontsize=8, color="0.4")

    fig.suptitle("Idalia 2023 wk-3 downscaling PDFs  |  standardized (top) vs physical units (bottom)",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    out = os.path.join(ROOT, "outputs", "fig3_pdfs_qgstyle.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")   # vector, like their .pdf outputs
    print(f"figure: {out}")
    print(f"figure: {out.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
