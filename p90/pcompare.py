"""Skill figures + markdown report for the p90 T2m-anomaly experiment.

Consumes the tables written by ``p90.pmetrics.compute_all`` -
``scores_dir(model)/case_scores.csv`` (one row per scored case) and
``scores_dir(model)/pooled.npz`` (pooled rank histograms) - plus the per-case
forecast cubes and ERA5 truth fields, and turns them into a fixed set of PNGs
under ``figures/p90_t2m/<model>/`` and a ``summary.md`` under the scores dir.

Everything is pure matplotlib (NO cartopy) so it runs on any login node with no
GPU and no map backend. The probabilistic-skill numbers (Brier / BSS / reliability
decomposition / ROC-AUC) are recomputed here directly from the ``p_event`` /
``obs_event`` columns so the figures never depend on a particular pmetrics helper
signature; ``pmetrics.week_anom_members`` is used for the PDF grid when present,
falling back to the exact xres recipe (``xres.xmetrics.forecast_field``) on the
cube otherwise.

Every figure guards its own inputs: a missing CSV column, a missing cube, an empty
kind, or a missing ``pooled.npz`` prints a ``[p90 compare]`` note and skips that
figure rather than crashing, so partial runs still produce whatever is plottable.

Env vars honoured (via ``pconfig``): GENCAST_RUNS (runs root), P90_* case filters
do NOT apply here - compare always scores every case present in the CSV.

Public entry point: ``make_all(model)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

from . import pconfig as P             # noqa: E402

# --------------------------------------------------------------------------- #
# Shared style (mirrors xres/xcombined + xres/xscores conventions)
# --------------------------------------------------------------------------- #
YMIN = 1e-6                            # floor of the log-y PDF axis
COL_MODEL = "#238b45"                  # forecast / event family (green)
COL_TRUTH = "#252525"                  # ERA5 truth (near-black)
COL_CONTROL = "#969696"                # no-event controls (grey)
COL_REF_CLIM = "#807dba"               # climatology-0 reference (muted purple)
COL_REF_PERS = "#d94801"               # persistence reference (muted orange)
KIND_STYLE = {                         # per-kind scatter styling
    "event":   dict(color=COL_MODEL, marker="o", label="event"),
    "control": dict(color=COL_CONTROL, marker="s", label="control"),
}
DPI = 140
TAG = "[p90 compare]"


def _grid(ax):
    ax.grid(True, which="both", ls="--", color="0.85", lw=0.5)


def _save(fig, figdir: Path, name: str) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"{TAG} wrote {out}")


# --------------------------------------------------------------------------- #
# Data access helpers
# --------------------------------------------------------------------------- #
def _week_anom_members(model: str, case: str) -> np.ndarray | None:
    """Pooled (member, lat, lon) week-mean T2m anomaly for a case, as a flat
    finite-tolerant ndarray, via the xres recipe (``pmetrics.week_anom_members``,
    which delegates to ``xres.xmetrics.forecast_field``). Returns None when the
    case has no cached cube."""
    cube_path = P.cube_path(model, case)
    if not cube_path.exists():
        return None
    import xarray as xr
    from . import pmetrics as PM
    row = P.case_row(case)
    with xr.open_dataset(cube_path) as cube:
        da = PM.week_anom_members(cube, row["peak"]).load()
    return np.asarray(da.values, dtype=float)


def _truth_field(case: str) -> np.ndarray | None:
    tp = P.truth_path(case)
    if not tp.exists():
        return None
    import xarray as xr
    with xr.open_dataset(tp) as ds:
        return np.asarray(ds["t2m_anom"].values, dtype=float)


# --------------------------------------------------------------------------- #
# Probabilistic-skill math (recomputed from p_event / obs_event)
# --------------------------------------------------------------------------- #
def _brier_decomp(p, o, nbins: int = 10) -> dict:
    """Murphy (1973) 2-component decomposition of the Brier score over equal-width
    probability bins: BS = reliability - resolution + uncertainty. BSS is vs the
    sample climatology (uncertainty). Returns NaNs when there is no valid data."""
    p = np.asarray(p, float)
    o = np.asarray(o, float)
    m = np.isfinite(p) & np.isfinite(o)
    p, o = p[m], o[m]
    n = p.size
    if n == 0:
        return dict(brier=np.nan, bss=np.nan, rel=np.nan, res=np.nan,
                    unc=np.nan, fbar=np.nan, n=0)
    bs = float(np.mean((p - o) ** 2))
    fbar = float(o.mean())
    unc = fbar * (1.0 - fbar)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, nbins - 1)
    rel = res = 0.0
    for k in range(nbins):
        sel = idx == k
        nk = int(sel.sum())
        if nk == 0:
            continue
        pk = float(p[sel].mean())
        ok = float(o[sel].mean())
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - fbar) ** 2
    rel /= n
    res /= n
    bss = 1.0 - bs / unc if unc > 0 else np.nan
    return dict(brier=bs, bss=bss, rel=rel, res=res, unc=unc, fbar=fbar, n=n)


def _reliability_bins(p, o, nbins: int = 10):
    """Per-bin (mean forecast prob, observed frequency, count) over equal-width bins."""
    p = np.asarray(p, float)
    o = np.asarray(o, float)
    m = np.isfinite(p) & np.isfinite(o)
    p, o = p[m], o[m]
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, nbins - 1)
    mp, of, cnt = [], [], []
    for k in range(nbins):
        sel = idx == k
        nk = int(sel.sum())
        if nk == 0:
            continue
        mp.append(float(p[sel].mean()))
        of.append(float(o[sel].mean()))
        cnt.append(nk)
    return np.array(mp), np.array(of), np.array(cnt)


def _roc(p, o):
    """ROC (fpr, tpr) and AUC from scores ``p`` and binary labels ``o``.

    Delegates to ``pmetrics.roc_curve`` (a threshold sweep, ``pred = p >= thr`` then
    ``trapz(hr, far)``) rather than a cumsum over argsort order: p_event is a discrete
    fraction of members so ties are pervasive, and the sweep connects each tie block with a
    single diagonal move (the correct 0.5 credit) instead of a row-order-dependent zig-zag.
    Keeps the figure/report AUC identical to the AUC ``compute_all`` prints. Returns
    (None, None, nan) for single-class labels so the caller can note 'ROC undefined'."""
    p = np.asarray(p, float)
    o = np.asarray(o, float)
    m = np.isfinite(p) & np.isfinite(o)
    p, o = p[m], o[m]
    npos = float((o == 1).sum())
    nneg = float((o == 0).sum())
    if npos == 0 or nneg == 0:
        return None, None, np.nan
    from . import pmetrics as PM
    far, hr, auc = PM.roc_curve(p, o)          # far == fpr (x), hr == tpr (y)
    return far, hr, auc


# --------------------------------------------------------------------------- #
# Figure 1: per-event verification PDFs (pooled members vs ERA5 truth field)
# --------------------------------------------------------------------------- #
def _fig_pdf_grid(model: str, df: pd.DataFrame, figdir: Path) -> None:
    from xres.xcombined import PDF_histogram

    ev = df[df["kind"] == "event"].copy()
    if len(ev) == 0:
        print(f"{TAG} no event cases in scores; skipping pdf_grid_events.png")
        return
    ev = ev.sort_values("peak_anom_K", ascending=False)

    panels = []
    for _, r in ev.iterrows():
        case = r["case"]
        mem = _week_anom_members(model, case)
        truth = _truth_field(case)
        if mem is None or truth is None:
            continue
        panels.append((case, float(r.get("peak_anom_K", np.nan)), mem, truth))
    if not panels:
        print(f"{TAG} no cubes/truth available; skipping pdf_grid_events.png")
        return

    cols = 5
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows),
                             squeeze=False)
    for ax, (case, pk, mem, truth) in zip(axes.ravel(), panels):
        allv = np.concatenate([mem.ravel(), truth.ravel()])
        allv = allv[np.isfinite(allv)]
        xmin, xmax = float(allv.min()), float(allv.max())
        for vals, col, lw, label in (
            (truth.ravel(), COL_TRUTH, 2.6, "ERA5 truth"),
            (mem.ravel(), COL_MODEL, 2.0, f"{model} ({mem.shape[0]} mem pooled)"),
        ):
            x, d = PDF_histogram(vals, xmin=xmin, xmax=xmax)
            ax.plot(x, np.clip(d, YMIN, None), color=col, lw=lw, label=label)
        ax.axvline(0, color="grey", lw=0.8, alpha=0.6)
        ax.set_yscale("log")
        ax.set_ylim(YMIN, 1.0)
        _grid(ax)
        ttl = case if not np.isfinite(pk) else f"{case}\npeak {pk:+.1f} K"
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("week-mean T2m anom (K)", fontsize=8)
        ax.set_ylabel("PDF (probability per bin)", fontsize=8)
        ax.legend(fontsize=6)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{model}: per-event CONUS week-mean T2m-anomaly PDFs "
                 f"(pooled members vs ERA5 truth), 3-week lead", y=1.0, fontsize=12)
    fig.tight_layout()
    _save(fig, figdir, "pdf_grid_events.png")


# --------------------------------------------------------------------------- #
# Figure 2: CRPS summary
# --------------------------------------------------------------------------- #
def _fig_crps(model: str, df: pd.DataFrame, figdir: Path) -> None:
    need = {"crps", "peak_anom_K"}
    if not need.issubset(df.columns):
        print(f"{TAG} missing CRPS columns; skipping crps_summary.png")
        return
    ev = df[df["kind"] == "event"].sort_values("peak_anom_K", ascending=False)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5.2))

    # (a) per-event CRPS bars + reference markers
    if len(ev):
        xs = np.arange(len(ev))
        ax0.bar(xs, ev["crps"].values, color=COL_MODEL, width=0.7,
                label=f"{model} CRPS")
        if "crps_clim0" in ev:
            ax0.scatter(xs, ev["crps_clim0"].values, color=COL_REF_CLIM,
                        marker="_", s=180, lw=2, label="clim-0 CRPS", zorder=5)
        if "crps_persist" in ev:
            ax0.scatter(xs, ev["crps_persist"].values, color=COL_REF_PERS,
                        marker="_", s=180, lw=2, label="persistence CRPS", zorder=5)
        ax0.set_xticks(xs)
        ax0.set_xticklabels(ev["case"].values, rotation=60, ha="right", fontsize=6)
        ax0.set_ylabel("CRPS (K)")
        ax0.set_title("(a) per-event CRPS (sorted by peak anomaly)", fontsize=10)
        ax0.legend(fontsize=7)
        ax0.grid(True, axis="y", ls="--", color="0.85", lw=0.5)
    else:
        ax0.axis("off")

    # (b) CRPSS per-case strip/box by kind, one group per reference
    groups = []                                    # (label, position, values, color)
    pos = 0
    for ref, col in (("crpss_clim0", COL_REF_CLIM), ("crpss_persist", COL_REF_PERS)):
        if ref not in df.columns:
            continue
        for kind in ("event", "control"):
            v = df[df["kind"] == kind][ref].dropna().values
            groups.append((f"{ref.replace('crpss_', '')}\n{kind}", pos, v,
                           KIND_STYLE[kind]["color"]))
            pos += 1
        pos += 1                                    # gap between reference groups
    if groups:
        bx = [g[2] for g in groups if len(g[2])]
        bp = [g[1] for g in groups if len(g[2])]
        if bx:
            ax1.boxplot(bx, positions=bp, widths=0.6, showfliers=False)
        for label, x, v, col in groups:
            if len(v):
                jx = x + (np.random.RandomState(0).rand(len(v)) - 0.5) * 0.25
                ax1.scatter(jx, v, s=16, color=col, alpha=0.7, zorder=5)
        ax1.axhline(0, color="k", lw=1, ls=":")
        ax1.set_xticks([g[1] for g in groups])
        ax1.set_xticklabels([g[0] for g in groups], fontsize=7)
        ax1.set_ylabel("CRPSS (>0 = beats reference)")
        ax1.set_title("(b) CRPSS vs clim-0 and persistence, by kind", fontsize=10)
        _grid(ax1)
    else:
        ax1.axis("off")

    fig.suptitle(f"{model}: CRPS skill summary, 3-week lead", y=1.0, fontsize=12)
    fig.tight_layout()
    _save(fig, figdir, "crps_summary.png")


# --------------------------------------------------------------------------- #
# Figure 3: reliability diagram + ROC
# --------------------------------------------------------------------------- #
def _fig_reliability_roc(model: str, df: pd.DataFrame, figdir: Path) -> None:
    if not {"p_event", "obs_event"}.issubset(df.columns):
        print(f"{TAG} missing p_event/obs_event; skipping reliability_roc.png")
        return
    p = df["p_event"].values
    o = df["obs_event"].values
    ncase = int((np.isfinite(p) & np.isfinite(o)).sum())
    dec = _brier_decomp(p, o)
    fbar = dec["fbar"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.6))

    # (a) reliability diagram
    mp, of, cnt = _reliability_bins(p, o)
    ax0.plot([0, 1], [0, 1], color="0.4", ls="-", lw=1, label="perfect")
    if np.isfinite(fbar):
        ax0.axhline(fbar, color=COL_REF_CLIM, ls="--", lw=1, label="climatology")
        ax0.axline((0, fbar / 2), slope=0.5, color="0.6", ls=":", lw=1,
                   label="no-skill")
    if len(mp):
        sizes = 30 + 300 * (cnt / cnt.max())
        ax0.scatter(mp, of, s=sizes, color=COL_MODEL, edgecolor="k", lw=0.5,
                    zorder=5, label="forecast")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xlabel("forecast probability p_event")
    ax0.set_ylabel("observed event frequency")
    ax0.set_title("(a) reliability diagram", fontsize=10)
    _grid(ax0)
    if ncase >= 10 and np.isfinite(dec["brier"]):
        ax0.text(0.02, 0.98,
                 f"n={ncase}\nBrier={dec['brier']:.3f}\nBSS={dec['bss']:+.3f}\n"
                 f"rel={dec['rel']:.3f}\nres={dec['res']:.3f}\nunc={dec['unc']:.3f}",
                 transform=ax0.transAxes, va="top", ha="left", fontsize=7,
                 bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    else:
        ax0.text(0.02, 0.98, f"n={ncase} (<10: decomposition omitted)",
                 transform=ax0.transAxes, va="top", ha="left", fontsize=7)
    ax0.legend(fontsize=7, loc="lower right")

    # sharpness inset: histogram of p_event
    ins = ax0.inset_axes([0.58, 0.06, 0.38, 0.30])
    pv = p[np.isfinite(p)]
    if pv.size:
        ins.hist(pv, bins=np.linspace(0, 1, 11), color=COL_MODEL, alpha=0.8)
    ins.set_title("sharpness", fontsize=6)
    ins.tick_params(labelsize=5)

    # (b) ROC
    fpr, tpr, auc = _roc(p, o)
    ax1.plot([0, 1], [0, 1], color="0.6", ls=":", lw=1, label="no-skill")
    if fpr is not None:
        ax1.plot(fpr, tpr, color=COL_MODEL, lw=2, marker=".", ms=4,
                 label=f"ROC (AUC={auc:.3f})")
    else:
        ax1.text(0.5, 0.5, "ROC undefined\n(single-class labels)",
                 ha="center", va="center", fontsize=9, transform=ax1.transAxes)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("false-positive rate")
    ax1.set_ylabel("true-positive rate")
    ax1.set_title("(b) ROC curve", fontsize=10)
    _grid(ax1)
    ax1.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"{model}: event-probability reliability + discrimination "
                 f"(p_event = fraction of members over the p90 threshold)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    _save(fig, figdir, "reliability_roc.png")


# --------------------------------------------------------------------------- #
# Figure 4: pooled rank histograms
# --------------------------------------------------------------------------- #
def _fig_rank_hist(model: str, pooled_npz: Path, figdir: Path) -> None:
    if not pooled_npz.exists():
        print(f"{TAG} no pooled.npz; skipping rank_hist.png")
        return
    d = np.load(pooled_npz)
    M = int(d["n_members"]) if "n_members" in d else None
    fig, ax = plt.subplots(figsize=(7, 4.4))
    plotted = False
    for key, col, ls, label in (
        ("rank_hist_event", COL_MODEL, "-", "events"),
        ("rank_hist_control", COL_CONTROL, "--", "controls"),
    ):
        if key not in d:
            continue
        h = np.asarray(d[key], dtype=float)
        if h.sum() <= 0:
            continue
        ax.step(np.arange(len(h)), h / h.sum(), where="mid", lw=2,
                color=col, ls=ls, label=label)
        plotted = True
        if M is None:
            M = len(h) - 1
    if not plotted:
        print(f"{TAG} pooled.npz has no rank-hist counts; skipping rank_hist.png")
        plt.close(fig)
        return
    if M is not None and M >= 0:
        ax.axhline(1.0 / (M + 1), color="k", ls=":", lw=1,
                   label="flat 1/(M+1) (calibrated)")
    _grid(ax)
    ax.set_xlabel("rank of ERA5 truth among members")
    ax.set_ylabel("frequency")
    ax.set_title(f"{model}: pooled rank histograms (flat = calibrated, "
                 f"U = under-dispersed, spike = biased)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, figdir, "rank_hist.png")


# --------------------------------------------------------------------------- #
# Figure 5: spread vs error
# --------------------------------------------------------------------------- #
def _fig_spread_error(model: str, df: pd.DataFrame, figdir: Path) -> None:
    if not {"spread", "rmse"}.issubset(df.columns):
        print(f"{TAG} missing spread/rmse; skipping spread_error.png")
        return
    fig, ax = plt.subplots(figsize=(6.2, 6))
    lim = 0.0
    for kind, st in KIND_STYLE.items():
        sub = df[df["kind"] == kind]
        if not len(sub):
            continue
        ax.scatter(sub["spread"], sub["rmse"], s=28, alpha=0.75,
                   color=st["color"], marker=st["marker"], label=st["label"])
        lim = max(lim, np.nanmax(sub["spread"].values), np.nanmax(sub["rmse"].values))
    if lim > 0:
        ax.plot([0, lim], [0, lim], color="0.4", ls="-", lw=1, label="1:1")
        ax.set_xlim(0, lim * 1.05)
        ax.set_ylim(0, lim * 1.05)
    if "spread_error" in df.columns:
        notes = []
        for kind in ("event", "control"):
            v = df[df["kind"] == kind]["spread_error"].dropna()
            if len(v):
                notes.append(f"{kind}: mean spread/error = {v.mean():.2f}")
        if notes:
            ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes,
                    va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    _grid(ax)
    ax.set_xlabel("ensemble spread (K)")
    ax.set_ylabel("ensemble-mean RMSE (K)")
    ax.set_title(f"{model}: spread-error consistency\n"
                 f"(on the 1:1 line = well calibrated)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, figdir, "spread_error.png")


# --------------------------------------------------------------------------- #
# Figure 6: ensemble-mean deterministic skill vs event intensity
# --------------------------------------------------------------------------- #
def _fig_ensmean_skill(model: str, df: pd.DataFrame, figdir: Path) -> None:
    if "peak_anom_K" not in df.columns:
        print(f"{TAG} missing peak_anom_K; skipping ensmean_skill.png")
        return
    panels = [("rmse", "ensemble-mean RMSE (K)", None),
              ("bias", "ensemble-mean bias (K)", 0.0),
              ("acc", "anomaly correlation", 0.0)]
    have = [pn for pn in panels if pn[0] in df.columns]
    if not have:
        print(f"{TAG} no rmse/bias/acc columns; skipping ensmean_skill.png")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5 * len(have), 4.6),
                             squeeze=False)
    for ax, (col, ylabel, ref) in zip(axes.ravel(), have):
        for kind, st in KIND_STYLE.items():
            sub = df[df["kind"] == kind]
            if not len(sub):
                continue
            ax.scatter(sub["peak_anom_K"], sub[col], s=28, alpha=0.75,
                       color=st["color"], marker=st["marker"], label=st["label"])
        if ref is not None:
            ax.axhline(ref, color="k", lw=1, ls=":")
        _grid(ax)
        ax.set_xlabel("event peak T2m anomaly (K)")
        ax.set_ylabel(ylabel)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"{model}: ensemble-mean skill vs event intensity, 3-week lead",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    _save(fig, figdir, "ensmean_skill.png")


# --------------------------------------------------------------------------- #
# Figure 7: per-case timing (only if timing jsons exist)
# --------------------------------------------------------------------------- #
def _fig_timing(model: str, figdir: Path) -> None:
    tdir = P.timing_dir(model)
    if not tdir.exists():
        print(f"{TAG} no timing dir; skipping timing_summary.png")
        return
    files = sorted(tdir.glob("*_timing.json"))
    mins = []
    for f in files:
        try:
            t = json.loads(f.read_text())
        except Exception:              # noqa: BLE001 - skip unreadable timing json
            continue
        v = t.get("minutes_total", t.get("minutes"))
        if v is not None and np.isfinite(float(v)):
            mins.append(float(v))
    if not mins:
        print(f"{TAG} no timing jsons; skipping timing_summary.png")
        return
    mins = np.array(mins)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(mins, bins=min(20, max(5, len(mins))), color=COL_MODEL, alpha=0.85)
    _grid(ax)
    ax.set_xlabel("wall time per case (minutes)")
    ax.set_ylabel("number of cases")
    ax.set_title(f"{model}: per-case inference wall time", fontsize=10)
    ax.text(0.98, 0.95,
            f"cases timed: {len(mins)}\n"
            f"median: {np.median(mins):.1f} min\n"
            f"total: {mins.sum() / 60:.1f} GPU-h",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.tight_layout()
    _save(fig, figdir, "timing_summary.png")


# --------------------------------------------------------------------------- #
# Figure 8 / report: summary.md
# --------------------------------------------------------------------------- #
def _fig_list(figdir: Path) -> list[str]:
    return [f.name for f in sorted(figdir.glob("*.png"))]


def _write_summary(model: str, df: pd.DataFrame, figdir: Path) -> None:
    p = df["p_event"].values if "p_event" in df.columns else np.array([])
    o = df["obs_event"].values if "obs_event" in df.columns else np.array([])
    dec = _brier_decomp(p, o) if p.size and o.size else _brier_decomp([], [])
    _, _, auc = _roc(p, o) if p.size and o.size else (None, None, np.nan)

    n_by_kind = df["kind"].value_counts().to_dict() if "kind" in df.columns else {}

    def _mean(col):
        return df[col].mean() if col in df.columns else np.nan

    def _median(col):
        return df[col].median() if col in df.columns else np.nan

    def _frac_pos(col):
        if col not in df.columns:
            return np.nan
        v = df[col].dropna()
        return float((v > 0).mean()) if len(v) else np.nan

    # events-vs-controls p_event contrast (by whether the event was observed)
    p_obs1 = p_obs0 = np.nan
    if {"p_event", "obs_event"}.issubset(df.columns):
        p_obs1 = df.loc[df["obs_event"] == 1, "p_event"].mean()
        p_obs0 = df.loc[df["obs_event"] == 0, "p_event"].mean()

    L = []
    L.append(f"# p90 T2m-anomaly skill report - {model}")
    L.append("")
    L.append("3-week lead (init = peak - 21 d), 50-member ensemble, week-mean T2m "
             "anomaly over [peak-6d, peak] on the 0.25 deg CONUS box vs ERA5, "
             "anomalies vs the 1990-2019 climatology.")
    L.append("")
    L.append("## Cases scored")
    L.append("")
    L.append(f"- total: {len(df)}")
    for k in ("event", "control"):
        if k in n_by_kind:
            L.append(f"- {k}: {n_by_kind[k]}")
    L.append("")
    L.append("## Headline metrics")
    L.append("")
    L.append("| metric | value |")
    L.append("| --- | --- |")
    L.append(f"| mean CRPS (K) | {_mean('crps'):.4f} |")
    L.append(f"| median CRPS (K) | {_median('crps'):.4f} |")
    L.append(f"| CRPSS>0 vs clim-0 (fraction) | {_frac_pos('crpss_clim0'):.3f} |")
    L.append(f"| CRPSS>0 vs persistence (fraction) | {_frac_pos('crpss_persist'):.3f} |")
    L.append(f"| Brier (event vs p90) | {dec['brier']:.4f} |")
    L.append(f"| BSS vs climatology | {dec['bss']:+.4f} |")
    L.append(f"| reliability | {dec['rel']:.4f} |")
    L.append(f"| resolution | {dec['res']:.4f} |")
    L.append(f"| uncertainty | {dec['unc']:.4f} |")
    L.append(f"| ROC AUC | {auc:.4f} |")
    L.append(f"| mean spread/error ratio | {_mean('spread_error'):.3f} |")
    L.append(f"| mean bias (K) | {_mean('bias'):+.4f} |")
    L.append(f"| mean ACC | {_mean('acc'):.4f} |")
    L.append("")
    L.append("## Event vs control discrimination")
    L.append("")
    L.append(f"- mean p_event when event observed (obs_event=1): {p_obs1:.3f}")
    L.append(f"- mean p_event when no event (obs_event=0): {p_obs0:.3f}")
    if np.isfinite(p_obs1) and np.isfinite(p_obs0):
        L.append(f"- contrast (obs1 - obs0): {p_obs1 - p_obs0:+.3f}")
    L.append("")
    L.append("## Figures")
    L.append("")
    figs = _fig_list(figdir)
    if figs:
        for f in figs:
            L.append(f"- `{f}`")
    else:
        L.append("- (none written)")
    L.append("")
    text = "\n".join(L)

    P.scores_dir(model).mkdir(parents=True, exist_ok=True)
    out = P.scores_dir(model) / "summary.md"
    out.write_text(text)
    print(f"{TAG} wrote {out}")
    print(text)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def make_all(model: str) -> None:
    """Build all p90 skill figures + summary.md for ``model``. Reads
    ``scores_dir(model)/case_scores.csv`` (+ ``pooled.npz``); if the CSV is
    absent, prints a note pointing at compare/pmetrics and returns without error."""
    scores_csv = P.scores_dir(model) / "case_scores.csv"
    if not scores_csv.exists():
        print(f"{TAG} no scores at {scores_csv}")
        print(f"{TAG} run the compare/pmetrics step first (p90.pmetrics.compute_all) "
              f"to write case_scores.csv + pooled.npz; nothing to plot")
        return
    try:
        df = pd.read_csv(scores_csv)
    except pd.errors.EmptyDataError:
        # compute_all writes a columnless CSV when zero cases were scored (no cube/truth
        # pair yet); read_csv raises before the len() guard below, so catch it here.
        print(f"{TAG} {scores_csv} has no rows; run compare/pmetrics first; nothing to plot")
        return
    if len(df) == 0:
        print(f"{TAG} {scores_csv} is empty; run compare/pmetrics first; nothing to plot")
        return

    figdir = P.fig_dir(model)
    figdir.mkdir(parents=True, exist_ok=True)
    pooled_npz = P.scores_dir(model) / "pooled.npz"

    # Each figure guards its own inputs; the try/except is a backstop so one bad
    # figure never aborts the rest of the report.
    steps = [
        ("pdf_grid_events", lambda: _fig_pdf_grid(model, df, figdir)),
        ("crps_summary",    lambda: _fig_crps(model, df, figdir)),
        ("reliability_roc", lambda: _fig_reliability_roc(model, df, figdir)),
        ("rank_hist",       lambda: _fig_rank_hist(model, pooled_npz, figdir)),
        ("spread_error",    lambda: _fig_spread_error(model, df, figdir)),
        ("ensmean_skill",   lambda: _fig_ensmean_skill(model, df, figdir)),
        ("timing_summary",  lambda: _fig_timing(model, figdir)),
    ]
    for name, fn in steps:
        try:
            fn()
        except Exception as e:         # noqa: BLE001 - never crash on partial data
            print(f"{TAG} {name} failed on partial data ({type(e).__name__}: {e}); skipping")
    _write_summary(model, df, figdir)
    print(f"{TAG} done for model {model}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="p90 compare figures + report")
    ap.add_argument("--model", default="fcn3", choices=list(P.MODELS))
    args = ap.parse_args()
    make_all(args.model)
