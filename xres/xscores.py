"""Probabilistic skill figures for the cross-resolution experiment.

Three figures that summarize what the combined-PDF panels show qualitatively, all built
from the same per-member verification fields (``runs/xres/<res>/week2/verif``) and ERA5
truth. Every figure is produced in TWO verification modes:

    own-grid (no suffix)          each resolution vs ERA5 sampled onto ITS OWN grid —
                                  the 0.25deg ensemble against native 0.25deg ERA5, the
                                  1.0deg ensemble against ERA5 sampled onto the 1.0deg
                                  grid — so neither resolution is penalized for scales
                                  it cannot represent.
    common baseline (_vs_era5_0p25)  BOTH ensembles vs native 0.25deg ERA5; the 1.0deg
                                  members are nearest-neighbour upsampled onto the
                                  0.25deg grid first, so the coarse model IS penalized
                                  for the fine-scale structure it cannot produce.

    xres_brier.png        per-event (4x3) Brier score of tail exceedance vs threshold
                          percentile (75/90/95/99th of the matched-grid ERA5 truth,
                          taken in the event's extreme direction). Lower = better;
                          grey line = climatological-reference score f(1-f).
    xres_crps.png         per-event CRPS (latitude-weighted mean over CONUS), one
                          panel per metric family so units stay homogeneous.
    xres_rank_hist.png    per-event (4x3) rank histograms of ERA5 truth within the
                          24-member ensemble. Flat = calibrated, U = under-dispersed,
                          hump = over-dispersed; end spikes = biased.
    xres_rank_hist_pooled.png   ranks pooled over all events.

Documentation (math + worked examples): docs/xres_score_figures.html
All scores also land in runs/xres/figures/xres_scores.csv.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

from gencast_s2s import config as C    # noqa: E402

from . import xconfig as X             # noqa: E402
from . import xmetrics as XM           # noqa: E402
from .xplotting import era5_truth, members, _on   # noqa: E402

BRIER_PCTS = (75, 90, 95, 99)


def _mode(common):
    """(csv truth tag, title note, filename suffix) for a verification mode."""
    if common:
        return ("era5_0p25",
                "BOTH resolutions vs ERA5 0.25deg; 1.0deg upsampled to the 0.25deg grid",
                "_vs_era5_0p25")
    return ("own_grid", "each resolution vs ERA5 on its own grid", "")

# Score-figure styling: one style per resolution, redundant in hue, line style and
# marker (colorblind-safe). Dark family shades — these curves sit on white panels.
RES_STYLE = {
    "0p25": dict(color="#99000d", ls="-",  marker="o"),
    "1p0":  dict(color="#08306b", ls="--", marker="^"),
}


def _wmean(field, lat):
    """Latitude-weighted mean of a (lat, lon) numpy field, NaN-tolerant."""
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(field)
    m = np.isfinite(field)
    return float((field[m] * w[m]).sum() / w[m].sum())


def _pairs(weeks, name, metric, common=False):
    """[(res, members(member,lat,lon), truth(lat,lon))] for the available resolutions.

    common=False: own-grid verification — ERA5 sampled onto each forecast grid.
    common=True:  common-baseline verification — BOTH ensembles scored against native
                  0.25deg ERA5; the 1.0deg members are nearest-neighbour upsampled onto
                  the 0.25deg grid first.
    """
    era5 = era5_truth(name, metric)
    if era5 is None:
        return []
    out = []
    for res in X.RES_ORDER:
        mem = members(res, weeks, name, metric)
        if mem is None:
            continue
        if common:
            out.append((res, _on(mem, era5), era5))
        else:
            out.append((res, mem, _on(era5, mem)))
    return out


def _event_names(weeks):
    return [n for n in X.events()
            if _pairs(weeks, n, X.event_metric(n))]


# --------------------------------------------------------------------------- #
# Brier score of tail exceedance
# --------------------------------------------------------------------------- #
def _brier_curve(mem, truth, sign):
    """BS(p) for thresholds at the 75/90/95/99th truth percentiles, in the event's
    extreme direction (cold T2m events use the LOW tail: exceedance = below the
    mirrored percentile). Forecast probability = fraction of members exceeding."""
    e = mem.values
    o = truth.values
    lat = mem["lat"].values
    out = []
    for p in BRIER_PCTS:
        if sign >= 0:
            thr = np.nanpercentile(o, p)
            obs = (o >= thr).astype(float)
            prob = (e >= thr).mean(axis=0)
        else:
            thr = np.nanpercentile(o, 100 - p)
            obs = (o <= thr).astype(float)
            prob = (e <= thr).mean(axis=0)
        out.append(_wmean((prob - obs) ** 2, lat))
    return out


def plot_brier(weeks, outdirs, rows_out, common=False):
    names = _event_names(weeks)
    if not names:
        print("[xres scores] no data yet; skipping Brier figure")
        return
    truth_tag, note, sfx = _mode(common)
    cols = 3
    nrows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 3.4 * nrows), squeeze=False)
    ref = [(1 - p / 100) * (p / 100) for p in BRIER_PCTS]
    for ax, name in zip(axes.ravel(), names):
        metric = X.event_metric(name)
        sign = XM.extreme_sign(metric, name)
        ax.plot(BRIER_PCTS, ref, color="0.55", ls=":", lw=1.5, marker="d", ms=3.5,
                label="climatology ref f(1-f)")
        for res, mem, truth in _pairs(weeks, name, metric, common):
            bs = _brier_curve(mem, truth, sign)
            st = RES_STYLE[res]
            ax.plot(BRIER_PCTS, bs, lw=2, ms=5,
                    label=f"{X.res_spec(res)['label']}", **st)
            for p, v in zip(BRIER_PCTS, bs):
                rows_out.append(dict(event=name, metric=metric, res=res,
                                     truth=truth_tag, score=f"brier_p{p}", value=v))
        tail = "low tail" if sign < 0 else "high tail"
        ax.set_yscale("log")
        ax.set_xticks(BRIER_PCTS)
        ax.grid(True, which="both", ls="--", color="0.85", lw=0.5)
        ax.set_title(f"{name}  ({tail})", fontsize=10)
        ax.set_xlabel("truth-percentile threshold")
        ax.set_ylabel("Brier score")
        ax.legend(fontsize=6.5)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Week-{weeks} tail-exceedance Brier scores (lower = better; "
                 f"{note})", y=1.0, fontsize=12)
    fig.tight_layout()
    _save(fig, f"xres_brier{sfx}.png", outdirs)
    print(f"[xres scores] {len(names)} events -> xres_brier{sfx}.png")


# --------------------------------------------------------------------------- #
# CRPS
# --------------------------------------------------------------------------- #
def _crps(mem, truth):
    import properscoring as ps
    crps = ps.crps_ensemble(truth.values, np.moveaxis(mem.values, 0, -1))
    return _wmean(crps, mem["lat"].values)


def plot_crps(weeks, outdirs, rows_out, common=False):
    names = _event_names(weeks)
    if not names:
        print("[xres scores] no data yet; skipping CRPS figure")
        return
    truth_tag, note, sfx = _mode(common)
    fams = []                                   # metric families in event order
    for n in names:
        m = X.event_metric(n)
        if m not in fams:
            fams.append(m)
    by_fam = {m: [n for n in names if X.event_metric(n) == m] for m in fams}

    fig, axes = plt.subplots(1, len(fams),
                             figsize=(2.2 * len(names) + 2.5 * len(fams), 4.6),
                             gridspec_kw={"width_ratios": [len(by_fam[m]) for m in fams]},
                             squeeze=False)
    for ax, m in zip(axes.ravel(), fams):
        sp = XM.spec(m)
        evs = by_fam[m]
        xs = np.arange(len(evs))
        vals = {res: [] for res in X.RES_ORDER}
        for name in evs:
            got = {res: _crps(mem, truth)
                   for res, mem, truth in _pairs(weeks, name, m, common)}
            for res in X.RES_ORDER:
                v = got.get(res, np.nan)
                vals[res].append(v)
                if np.isfinite(v):
                    rows_out.append(dict(event=name, metric=m, res=res,
                                         truth=truth_tag, score="crps", value=v))
        width = 0.38
        for i, res in enumerate(X.RES_ORDER):
            st = RES_STYLE[res]
            ax.bar(xs + (i - 0.5) * width, vals[res], width,
                   color=X.res_spec(res)["color"], edgecolor=st["color"],
                   hatch="//" if res == "0p25" else None,
                   label=X.res_spec(res)["label"])
        # Winner annotation: CRPS(0.25) relative to CRPS(1.0); negative = 0.25 better.
        for x, (a, b) in enumerate(zip(vals["0p25"], vals["1p0"])):
            if np.isfinite(a) and np.isfinite(b) and b:
                ax.text(x, max(a, b) * 1.02, f"{(a - b) / b * 100:+.0f}%",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(evs, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(f"CRPS ({sp.units})")
        ax.set_title(sp.label, fontsize=10)
        ax.grid(True, axis="y", ls="--", color="0.85", lw=0.5)
        ax.legend(fontsize=7)
    fig.suptitle(f"Week-{weeks} ensemble CRPS, latitude-weighted CONUS mean "
                 f"(lower = better; % = 0.25deg relative to 1.0deg; {note})",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, f"xres_crps{sfx}.png", outdirs)
    print(f"[xres scores] {len(names)} events -> xres_crps{sfx}.png")


# --------------------------------------------------------------------------- #
# Rank histograms
# --------------------------------------------------------------------------- #
def _rank_hist(mem, truth):
    e = mem.values
    o = truth.values
    valid = np.isfinite(o) & np.isfinite(e).all(axis=0)
    below = (e < o[None]).sum(axis=0)
    return np.bincount(below[valid].ravel(), minlength=e.shape[0] + 1)


def plot_rank_hists(weeks, outdirs, rows_out, common=False):
    names = _event_names(weeks)
    if not names:
        print("[xres scores] no data yet; skipping rank histograms")
        return
    truth_tag, note, sfx = _mode(common)
    cols = 3
    nrows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 3.2 * nrows), squeeze=False)
    pooled = {res: None for res in X.RES_ORDER}
    for ax, name in zip(axes.ravel(), names):
        metric = X.event_metric(name)
        nbins = None
        for res, mem, truth in _pairs(weeks, name, metric, common):
            h = _rank_hist(mem, truth)
            nbins = len(h)
            pooled[res] = h.copy() if pooled[res] is None else pooled[res] + h
            st = RES_STYLE[res]
            ax.step(np.arange(nbins), h / h.sum(), where="mid", lw=1.8,
                    label=X.res_spec(res)["label"], **st)
            rows_out.append(dict(event=name, metric=metric, res=res,
                                 truth=truth_tag, score="rank_hist_extreme_frac",
                                 value=float((h[0] + h[-1]) / h.sum())))
        if nbins:
            ax.axhline(1 / nbins, color="k", ls=":", lw=1, label="flat (calibrated)")
        ax.grid(True, ls="--", color="0.85", lw=0.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("rank of ERA5 among members")
        ax.set_ylabel("frequency")
        ax.legend(fontsize=6.5)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Week-{weeks} rank histograms (flat = calibrated, U-shape = "
                 f"under-dispersed, end spike = biased; {note})", y=1.0, fontsize=12)
    fig.tight_layout()
    _save(fig, f"xres_rank_hist{sfx}.png", outdirs)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ref = None
    for res in X.RES_ORDER:
        h = pooled[res]
        if h is None:
            continue
        st = RES_STYLE[res]
        ax.step(np.arange(len(h)), h / h.sum(), where="mid", lw=2,
                label=X.res_spec(res)["label"], **st)
        ref = 1 / len(h)
    if ref is not None:
        ax.axhline(ref, color="k", ls=":", lw=1)
    ax.grid(True, ls="--", color="0.85", lw=0.5)
    ax.set_title(f"Week-{weeks} pooled rank histogram (all events; {note})",
                 fontsize=9)
    ax.set_xlabel("rank of ERA5 among members")
    ax.set_ylabel("frequency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, f"xres_rank_hist_pooled{sfx}.png", outdirs)
    print(f"[xres scores] {len(names)} events -> xres_rank_hist(.pooled){sfx}.png")


# --------------------------------------------------------------------------- #
def _save(fig, fname, outdirs):
    for outdir in outdirs:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(outdir) / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_all(weeks=None):
    weeks = X.WEEKS if weeks is None else weeks
    xfig = X.xfig_dir(weeks)
    outdirs = [xfig, C.ROOT / "figures" / "xres" / f"week{weeks}"]
    rows = []
    for common in (False, True):        # own-grid batch + common-0.25deg-baseline batch
        plot_brier(weeks, outdirs, rows, common=common)
        plot_crps(weeks, outdirs, rows, common=common)
        plot_rank_hists(weeks, outdirs, rows, common=common)
    if rows:
        df = pd.DataFrame(rows)
        xfig.mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(xfig) / "xres_scores.csv", index=False)
        print(f"[xres scores] scores table -> {xfig}/xres_scores.csv")
