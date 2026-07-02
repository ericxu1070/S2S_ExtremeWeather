"""The three combined PDF figures: ERA5 vs HRRR vs 0.25deg vs 1.0deg, one panel per event.

Each figure overlays, for every event (drawn in that event's headline metric), four spatial
PDFs over CONUS — Observed (ERA5, black), Observed (HRRR, red), the 0.25deg 2-week forecast
(blue) and the 1.0deg 2-week forecast (orange). The three figures differ only in HOW each
resolution's ensemble is collapsed to a single forecast curve:

    xres_combined_mean.png      ensemble-MEAN field
    xres_combined_best.png      single BEST member (lowest latitude-weighted RMSE vs ERA5)
    xres_combined_extreme.png   single most-EXTREME member (warmest/coldest for T2m;
                                 strongest winds / heaviest precip otherwise)
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from gencast_s2s import config as C    # noqa: E402
from gencast_s2s.plotting import YMIN, _pdf_on_grid   # noqa: E402

from . import xconfig as X             # noqa: E402
from . import xmetrics as XM           # noqa: E402
from .xplotting import era5_truth, hrrr_truth, members, _on   # noqa: E402


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
    if mode == "ensemble":
        field, tag = mem.mean("member"), "mean"
    elif mode == "best":
        if era5 is None:
            return None
        field, idx = _best_member(mem, era5); tag = f"best #{idx}"
    elif mode == "extreme":
        field, idx = _extreme_member(mem, XM.extreme_sign(metric, name)); tag = f"extreme #{idx}"
    else:
        raise ValueError(mode)
    return field.values.ravel(), f"{X.res_spec(res)['label']} wk-{weeks} {tag}"


def _panel(ax, name, weeks, mode):
    metric = X.event_metric(name)
    sp = XM.spec(metric)
    era5 = era5_truth(name, metric)
    hrrr = hrrr_truth(name, metric)

    obs = [(era5.values.ravel(), dict(color="k", lw=2.5, label="Observed (ERA5)"))]
    if hrrr is not None:
        obs.append((hrrr.values.ravel(),
                    dict(color="tab:red", lw=2.0, label="Observed (HRRR)")))

    curves = []   # (vals, color, ls, label)
    for res in X.RES_ORDER:
        res_c = _res_curve(res, weeks, name, metric, mode, era5)
        if res_c is None:
            continue
        vals, label = res_c
        ls = "-" if mode == "ensemble" else "--"
        curves.append((vals, X.res_spec(res)["color"], ls, label))

    allv = np.concatenate([v[np.isfinite(v)] for v, _ in obs]
                          + [v[np.isfinite(v)] for v, *_ in curves])
    xgrid = np.linspace(allv.min() - 1, allv.max() + 1, 400)
    for vals, st in obs:
        ax.plot(xgrid, np.clip(_pdf_on_grid(vals, xgrid), YMIN, None), **st)
    for vals, color, ls, label in curves:
        ax.plot(xgrid, np.clip(_pdf_on_grid(vals, xgrid), YMIN, None),
                color=color, ls=ls, lw=2.0, label=label)

    if sp.diverging:
        ax.axvline(0, color="grey", lw=0.8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(YMIN, None)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel(f"{sp.label} ({sp.units})")
    ax.set_ylabel("density")


def _figure(mode, title, fname, weeks, outdirs):
    names = [n for n in X.events()
             if any(members(r, weeks, n, X.event_metric(n)) is not None for r in X.RES_ORDER)]
    if not names:
        print(f"[xres combined] no forecasts yet; skipping {fname}")
        return
    cols = 3
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        _panel(ax, name, weeks, mode)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    for outdir in outdirs:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(outdir) / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[xres combined] {len(names)} events -> {fname}")


def make_all(weeks=None):
    weeks = X.WEEKS if weeks is None else weeks
    outdirs = [X.XFIG_DIR, C.ROOT]      # cross-res figures dir + project root
    _figure("ensemble",
            f"CONUS verification PDFs — ENSEMBLE MEAN, week-{weeks}: "
            "ERA5 vs HRRR vs GenCast 0.25deg vs 1.0deg",
            "xres_combined_mean.png", weeks, outdirs)
    _figure("best",
            f"CONUS verification PDFs — BEST MEMBER, week-{weeks}: "
            "ERA5 vs HRRR vs GenCast 0.25deg vs 1.0deg",
            "xres_combined_best.png", weeks, outdirs)
    _figure("extreme",
            f"CONUS verification PDFs — MOST-EXTREME MEMBER, week-{weeks}: "
            "ERA5 vs HRRR vs GenCast 0.25deg vs 1.0deg",
            "xres_combined_extreme.png", weeks, outdirs)
