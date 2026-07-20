#!/usr/bin/env python
"""FCN3 vs GenCast comparison figures for PNW Heat Dome 2021, week-2 (0.25 deg).

Runs in `my-env` (needs gencast_s2s / xres + the ERA5 climatology + truth). Produces:

  figures/fcn3/fcn3_vs_gencast_pdf.png      pooled-member week-mean T2m-anomaly PDFs
                                            (FCN3 12 vs GenCast 12 vs ERA5 0.25 truth),
                                            same PDF_histogram / log-y style as xres.
  figures/fcn3/fcn3_vs_gencast_runtime.png  wall time to generate 12 members.

Both models' anomalies are computed by the SAME recipe (xres.xmetrics.forecast_field),
so the only difference in the PDF is the forecast, not the post-processing.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from gencast_s2s import config as C
from xres import xmetrics as XM
from xres.xcombined import PDF_histogram, YMIN

EVENT = "PNW_HeatDome_2021"
PEAK = pd.Timestamp("2021-06-28T00:00:00")
N_MATCH = 12

ROOT = Path("/glade/derecho/scratch/exu/S2S_ExtremeWeather")
FCN3_CUBE = ROOT / "runs/fcn3" / EVENT / f"{EVENT}_fcn3_cube.nc"
FCN3_TIMING = ROOT / "runs/fcn3" / EVENT / f"{EVENT}_fcn3_timing.json"
GENCAST_CUBE = ROOT / "runs/xres/0p25/week2/cache" / f"{EVENT}_cube.nc"
ERA5_TRUTH = ROOT / "runs/observations" / f"{EVENT}_verif_t2m_anom.nc"
FIGDIR = ROOT / "figures/fcn3"

# GenCast 0.25 deg week-2 runtime, from logs/6665322[0].desched1.OU (A100-40GB):
# ~80 s/step x 28 steps = ~37.3 min/member steady state + one ~5 min XLA compile.
GENCAST_MIN_PER_MEMBER = 37.3
GENCAST_COMPILE_MIN = 5.0

# FCN3 palette (greens) vs GenCast (reds, the xres 0.25 deg family); truth dark solid.
COL_FCN3 = "#238b45"
COL_GENCAST = "#cb181d"
COL_TRUTH = "#252525"


def _anom_members(cube_path: Path, nmax: int) -> np.ndarray:
    cube = xr.open_dataset(cube_path)
    if cube.sizes["member"] > nmax:
        cube = cube.isel(member=slice(0, nmax))
    anom = XM.forecast_field(cube, PEAK, "t2m_anom")  # (member, lat, lon)
    return anom.transpose("member", "lat", "lon").values


def _truth() -> np.ndarray:
    return xr.open_dataset(ERA5_TRUTH)["t2m_anom"].values


def make_pdf():
    fcn3 = _anom_members(FCN3_CUBE, N_MATCH)
    gencast = _anom_members(GENCAST_CUBE, N_MATCH)
    truth = _truth()

    allvals = np.concatenate([fcn3.ravel(), gencast.ravel(), truth.ravel()])
    allvals = allvals[np.isfinite(allvals)]
    xmin, xmax = allvals.min(), allvals.max()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    series = [
        (truth.ravel(), f"ERA5 0.25° (truth)", COL_TRUTH, "-", 2.8),
        (gencast.ravel(), f"GenCast wk-2 ({N_MATCH} mem pooled)", COL_GENCAST, (0, (1, 1)), 2.2),
        (fcn3.ravel(), f"FCN3 wk-2 ({N_MATCH} mem pooled)", COL_FCN3, (0, (4, 2)), 2.2),
    ]
    for vals, label, col, ls, lw in series:
        x, d = PDF_histogram(vals, xmin=xmin, xmax=xmax)
        d = np.where(d <= 0, np.nan, d)
        ax.plot(x, d, color=col, ls=ls, lw=lw, label=label)
    ax.set_yscale("log"); ax.set_ylim(bottom=YMIN)
    ax.set_xlabel("week-mean 2 m temperature anomaly (K)")
    ax.set_ylabel("probability (pooled grid points x members)")
    ax.set_title(f"PNW Heat Dome 2021 — week-2 (14-day lead), 0.25°\n"
                 f"FCN3 vs GenCast T2m-anomaly distribution over CONUS")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fcn3_vs_gencast_pdf.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def make_runtime():
    gencast_min = GENCAST_COMPILE_MIN + GENCAST_MIN_PER_MEMBER * N_MATCH
    if FCN3_TIMING.exists():
        t = json.loads(FCN3_TIMING.read_text())
        fcn3_min = t["minutes_total"]
        fcn3_dev = t.get("gpu") or t.get("device")
    else:
        fcn3_min, fcn3_dev = float("nan"), "(not run yet)"

    fig, ax = plt.subplots(figsize=(6.5, 5))
    labels = [f"GenCast\n(A100-40GB)", f"FCN3\n({fcn3_dev})"]
    vals = [gencast_min, fcn3_min]
    bars = ax.bar(labels, vals, color=[COL_GENCAST, COL_FCN3], width=0.6)
    ax.set_ylabel("wall time to generate 12 members (minutes)")
    ax.set_title("PNW Heat Dome 2021, week-2 0.25° — 12-member ensemble cost")
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            txt = f"{v:.0f} min" if v < 90 else f"{v/60:.1f} h"
            ax.text(b.get_x() + b.get_width() / 2, v, txt, ha="center", va="bottom", fontsize=11)
    if all(np.isfinite(vals)) and min(vals) > 0:
        ax.text(0.98, 0.95, f"speedup ×{gencast_min/fcn3_min:.0f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=12, fontweight="bold")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fcn3_vs_gencast_runtime.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}  (GenCast {gencast_min:.0f} min, FCN3 {fcn3_min} min)")


if __name__ == "__main__":
    make_pdf()
    make_runtime()
