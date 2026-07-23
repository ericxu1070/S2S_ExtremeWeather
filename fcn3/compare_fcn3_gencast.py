#!/usr/bin/env python
"""FCN3 vs GenCast comparison figures + scores, all six events at week-3, 0.25 deg.

Runs in `moe` / `my-env` (needs gencast_s2s + xres + the ERA5 climatology and truth); the
FCN3 rollouts themselves are produced separately by ``fcn3/run_fcn3.py`` in the `fcn3` env.

Outputs (under figures/fcn3/week3/ and runs/fcn3/week3/scores/):

  fcn3_vs_gencast_pdfs.png     6-panel grid: pooled-member verification-field PDFs,
                               FCN3 vs GenCast vs ERA5 truth, log-y (xres PDF convention)
  <event>_pdf.png              the same, one standalone figure per event
  fcn3_vs_gencast_skill.png    per-event CRPS and ensemble-mean RMSE, both models
  fcn3_vs_gencast_runtime.png  wall time / GPU cost for a 24-member ensemble
  fcn3_vs_gencast_scores.csv   every number in the figures, per event per model

Both models' verification fields are computed by the SAME recipe
(``xres.xmetrics.forecast_field``) from cubes on the SAME grid, so the only difference in
the figures is the forecast, not the post-processing. Events whose cube or truth is not
built yet are skipped with a warning rather than failing the whole run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from fcn3 import fevents as F
from p90.pmetrics import crps_field, ens_stats, wmean
from xres import xmetrics as XM
from xres.xcombined import PDF_histogram, YMIN

# FCN3 greens vs GenCast reds (the xres 0.25 deg family); truth dark solid.
COL = {"fcn3": "#238b45", "gencast": "#cb181d", "truth": "#252525"}
LABEL = {"fcn3": "FCN3", "gencast": "GenCast"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _cube_path(ev: F.Event, model: str) -> Path:
    return F.cube_path(ev) if model == "fcn3" else F.gencast_cube_path(ev)


# Each model is scored on its OWN native cadence by default: FCN3 contributes 25 six-hourly
# frames to the verification week, GenCast 13 twelve-hourly ones. For the anomaly metrics
# that is harmless -- forecast_field subtracts a climatology sampled at exactly the same
# timestamps, so the diurnal cycle cancels per model. For an ABSOLUTE metric (u850_speed)
# it does not: the two means sample the diurnal cycle differently. Set FCN3_MATCH_FRAMES=1
# to subsample FCN3 down to GenCast's 00/12Z frames for a strict same-frames check.
MATCH_FRAMES = os.environ.get("FCN3_MATCH_FRAMES", "").lower() not in ("", "0", "false", "no")


def members_field(ev: F.Event, model: str, nmax: int | None = None) -> np.ndarray | None:
    """Per-member verification field (member, lat, lon) for this event's headline metric."""
    p = _cube_path(ev, model)
    if not p.exists():
        return None
    cube = xr.open_dataset(p)
    try:
        if nmax is not None and cube.sizes["member"] > nmax:
            cube = cube.isel(member=slice(0, nmax))
        if MATCH_FRAMES and model == "fcn3":
            hours = pd.DatetimeIndex(cube["time"].values).hour
            cube = cube.isel(time=np.isin(hours, (0, 12)))
        fld = XM.forecast_field(cube, pd.Timestamp(ev.peak), ev.metric)
        return fld.transpose("member", "lat", "lon").values
    finally:
        cube.close()


def truth_field(ev: F.Event) -> np.ndarray | None:
    p = F.era5_truth_path(ev)
    if not p.exists():
        return None
    with xr.open_dataset(p) as ds:
        return ds[ev.metric].values


def _lat(ev: F.Event) -> np.ndarray:
    """Verification-grid latitudes (for cos-lat weighting)."""
    for model in ("gencast", "fcn3"):
        p = _cube_path(ev, model)
        if p.exists():
            with xr.open_dataset(p) as ds:
                return ds["lat"].values
    with xr.open_dataset(F.era5_truth_path(ev)) as ds:
        return ds["lat"].values


def n_members_available(ev: F.Event) -> int | None:
    """Smallest member count across the models that have a cube (so both are subsampled
    to a common ensemble size -- an unmatched M would bias CRPS and the PDF tails)."""
    counts = []
    for model in ("fcn3", "gencast"):
        p = _cube_path(ev, model)
        if p.exists():
            with xr.open_dataset(p) as ds:
                counts.append(int(ds.sizes["member"]))
    return min(counts) if counts else None


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def build_scores() -> pd.DataFrame:
    rows = []
    for ev in F.selected():
        truth = truth_field(ev)
        if truth is None:
            print(f"[scores] {ev.name}: no ERA5 truth ({F.era5_truth_path(ev).name}); skip")
            continue
        nmax = n_members_available(ev)
        lat = _lat(ev)
        for model in ("fcn3", "gencast"):
            mem = members_field(ev, model, nmax)
            if mem is None:
                print(f"[scores] {ev.name} [{model}]: no cube "
                      f"({_cube_path(ev, model).name}); skip")
                continue
            st = ens_stats(mem, truth, lat)
            crps = wmean(crps_field(mem, truth), lat)
            # climatology reference: for anomalies a zero forecast; for absolute metrics
            # (u850 speed) the observed cos-lat area mean of the field.
            ref = 0.0 if ev.metric.endswith("_anom") else wmean(truth, lat)
            crps_ref = wmean(np.abs(truth - ref), lat)
            rows.append(dict(
                event=ev.name, family=ev.family, label=ev.label, metric=ev.metric,
                model=model, members=int(mem.shape[0]),
                peak=ev.peak, init=ev.init.date().isoformat(),
                crps=crps, crps_ref=crps_ref,
                crpss=(1.0 - crps / crps_ref) if crps_ref > 0 else float("nan"),
                rmse=st["rmse"], bias=st["bias"], acc=st["acc"],
                spread=st["spread"], spread_error=st["spread_error"],
                ens_mean=wmean(mem.mean(axis=0), lat), truth_mean=wmean(truth, lat),
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        print("[scores] nothing to score yet")
        return df
    F.ensure_dirs()
    F.scores_path().parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(F.scores_path(), index=False)
    print(f"[scores] wrote {F.scores_path()}  ({len(df)} rows)")
    return df


# --------------------------------------------------------------------------- #
# PDFs
# --------------------------------------------------------------------------- #
def _pdf_axes(ax, ev: F.Event, nmax: int | None) -> bool:
    """Draw one event's PDF panel. Returns False if nothing could be plotted."""
    truth = truth_field(ev)
    series = []
    for model in ("fcn3", "gencast"):
        mem = members_field(ev, model, nmax)
        if mem is not None:
            series.append((model, mem.shape[0], mem.ravel()))
    if not series and truth is None:
        return False

    vals = [v for _, _, v in series] + ([truth.ravel()] if truth is not None else [])
    allv = np.concatenate(vals)
    allv = allv[np.isfinite(allv)]
    xmin, xmax = float(allv.min()), float(allv.max())

    if truth is not None:
        x, d = PDF_histogram(truth.ravel(), xmin=xmin, xmax=xmax)
        ax.plot(x, np.where(d <= 0, np.nan, d), color=COL["truth"], ls="-", lw=2.6,
                label="ERA5 0.25° (truth)")
    styles = {"gencast": (0, (1, 1)), "fcn3": (0, (4, 2))}
    for model, nmem, v in series:
        x, d = PDF_histogram(v, xmin=xmin, xmax=xmax)
        ax.plot(x, np.where(d <= 0, np.nan, d), color=COL[model], ls=styles[model], lw=2.1,
                label=f"{LABEL[model]} ({nmem} mem pooled)")

    spec = XM.spec(ev.metric)
    ax.set_yscale("log")
    ax.set_ylim(bottom=YMIN)
    ax.set_xlabel(f"{spec.label} ({spec.units})", fontsize=9)
    title = f"{ev.label}"
    if ev.cold:
        title += "  [cold event]"
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.tick_params(labelsize=8)
    return True


def make_pdfs() -> None:
    events = F.selected()
    F.ensure_dirs()

    # per-event standalone figures
    for ev in events:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        if not _pdf_axes(ax, ev, n_members_available(ev)):
            plt.close(fig)
            print(f"[pdf] {ev.name}: no data yet; skip")
            continue
        ax.set_ylabel("probability (pooled grid points x members)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"week-{F.WEEKS} ({F.LEAD_DAYS}-day lead), 0.25° — FCN3 vs GenCast",
                     fontsize=10)
        out = F.fig_dir() / f"{ev.name}_pdf.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[pdf] wrote {out}")

    # combined grid
    n = len(events)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.9 * nrow), squeeze=False)
    drew = 0
    for k, ev in enumerate(events):
        ax = axes[k // ncol][k % ncol]
        if _pdf_axes(ax, ev, n_members_available(ev)):
            drew += 1
            if k % ncol == 0:
                ax.set_ylabel("probability (pooled)", fontsize=9)
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.text(0.5, 0.5, "not built yet", ha="center", va="center",
                    transform=ax.transAxes, color="0.6")
            ax.set_title(ev.label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"FCN3 vs GenCast — week-{F.WEEKS} ({F.LEAD_DAYS}-day lead), 0.25°, "
                 f"CONUS verification-week distributions", fontsize=12)
    out = F.fig_dir() / "fcn3_vs_gencast_pdfs.png"
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[pdf] wrote {out}  ({drew}/{n} panels populated)")


# --------------------------------------------------------------------------- #
# Skill
# --------------------------------------------------------------------------- #
def make_skill(df: pd.DataFrame) -> None:
    if df.empty:
        print("[skill] no scores; skip")
        return
    events = [e for e in F.selected() if e.name in set(df.event)]
    if not events:
        return
    x = np.arange(len(events))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, col, name in ((axes[0], "crps", "CRPS"), (axes[1], "rmse", "ens-mean RMSE")):
        for j, model in enumerate(("gencast", "fcn3")):
            vals = []
            for ev in events:
                hit = df[(df.event == ev.name) & (df.model == model)]
                vals.append(float(hit[col].iloc[0]) if len(hit) else np.nan)
            ax.bar(x + (j - 0.5) * w, vals, width=w, color=COL[model], label=LABEL[model])
        ax.set_xticks(x)
        ax.set_xticklabels([e.label for e in events], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"{name} (metric units)")
        ax.set_title(f"{name} — lower is better")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(frameon=False, fontsize=9)
    # units differ per event (K vs m/s) -> say so rather than implying one scale
    metrics = sorted({e.metric for e in events})
    fig.suptitle(f"FCN3 vs GenCast skill — week-{F.WEEKS} ({F.LEAD_DAYS}-day lead), 0.25°"
                 f"   (metrics: {', '.join(metrics)}; bars are per-event units)",
                 fontsize=11)
    out = F.fig_dir() / "fcn3_vs_gencast_skill.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[skill] wrote {out}")


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
def _fcn3_timings() -> dict[str, dict]:
    out = {}
    for ev in F.selected():
        p = F.timing_path(ev)
        if p.exists():
            out[ev.name] = json.loads(p.read_text())
    return out


def _gencast_pool_timing() -> dict | None:
    p = F.gencast_pool_timing_path()
    return json.loads(p.read_text()) if p.exists() else None


def gencast_minutes_per_event(ev: F.Event, pool: dict | None) -> tuple[float, str]:
    """(node wall minutes, provenance) for a GenCast ensemble of this event.

    Both branches are the SAME hardware (one a3mega 8xH100 node), same lead, same member
    count and same code path, so they are directly comparable to the FCN3 timings:
      * events run by this experiment's pool  -> that job's measured wall / n_events
      * the three already-cached xres events  -> job 223's measured wall / 12 events
    """
    if pool and ev.name in pool.get("events", []) and pool.get("n_events"):
        return pool["minutes_total"] / float(pool["n_events"]), f"job {pool.get('job_id', '?')}"
    if ev.source == "xres":
        return F.gencast_a3mega_minutes_per_event(), f"job {F.GENCAST_A3MEGA_JOB}"
    return float("nan"), "not run"


def make_runtime() -> None:
    fcn3_t = _fcn3_timings()
    pool = _gencast_pool_timing()
    if not fcn3_t:
        print("[runtime] no FCN3 timing data yet; skip")
        return

    events = F.selected()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # -- Panel A: per-event node wall time on ONE 8xH100 a3mega node, both models measured
    ax = axes[0]
    x = np.arange(len(events))
    w = 0.38
    gc_vals = [gencast_minutes_per_event(ev, pool)[0] for ev in events]
    f3_vals = [fcn3_t.get(ev.name, {}).get("minutes_total", np.nan) for ev in events]
    ax.bar(x - 0.5 * w, gc_vals, width=w, color=COL["gencast"], label="GenCast")
    ax.bar(x + 0.5 * w, f3_vals, width=w, color=COL["fcn3"], label="FCN3")
    for xi, v in zip(x, f3_vals):
        if np.isfinite(v):
            ax.text(xi + 0.5 * w, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x, gc_vals):
        if np.isfinite(v):
            ax.text(xi - 0.5 * w, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([e.label for e in events], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"node wall time for {F.N_MEMBERS} members (min)")
    ax.set_yscale("log")
    ax.set_title("Per-event ensemble cost — one a3mega 8xH100 node\n"
                 "(both models measured on the same hardware)", fontsize=10)
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.legend(frameon=False, fontsize=8)

    # -- Panel B: head-to-head means + the cross-machine A100 reference
    ax = axes[1]
    f3_min = [v for v in f3_vals if np.isfinite(v)]
    gc_min = [v for v in gc_vals if np.isfinite(v)]
    f3_mean = float(np.mean(f3_min)) if f3_min else np.nan
    gc_mean = float(np.mean(gc_min)) if gc_min else np.nan
    bars, labels, cols = [], [], []
    if np.isfinite(gc_mean):
        bars.append(gc_mean)
        labels.append(f"GenCast\n8xH100 node\n(measured, n={len(gc_min)})")
        cols.append(COL["gencast"])
    bars.append(F.gencast_a100_minutes(F.N_MEMBERS))
    labels.append("GenCast\n1x A100-40GB\n(Derecho archive)")
    cols.append("#fb9a99")
    if np.isfinite(f3_mean):
        bars.append(f3_mean)
        labels.append(f"FCN3\n8xH100 node\n(measured, n={len(f3_min)})")
        cols.append(COL["fcn3"])
    b = ax.bar(labels, bars, color=cols, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel(f"wall time for one {F.N_MEMBERS}-member ensemble (min)")
    ax.set_title(f"Cost of one {F.N_MEMBERS}-member {F.LEAD_DAYS}-day 0.25° ensemble",
                 fontsize=10)
    for bar, v in zip(b, bars):
        txt = f"{v:.1f} min" if v < 90 else f"{v/60:.1f} h"
        ax.text(bar.get_x() + bar.get_width() / 2, v, txt, ha="center", va="bottom",
                fontsize=9)
    if np.isfinite(gc_mean) and np.isfinite(f3_mean) and f3_mean > 0:
        ax.text(0.98, 0.95, f"hardware-matched speedup  ×{gc_mean / f3_mean:.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                fontweight="bold")
    ax.grid(alpha=0.25, axis="y", which="both")

    fig.suptitle(f"FCN3 vs GenCast runtime — week-{F.WEEKS} ({F.LEAD_DAYS}-day lead), 0.25°",
                 fontsize=12)
    out = F.fig_dir() / "fcn3_vs_gencast_runtime.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[runtime] wrote {out}")


# --------------------------------------------------------------------------- #
def make_all() -> None:
    print(F.describe())
    print()
    df = build_scores()
    make_pdfs()
    make_skill(df)
    make_runtime()
    if not df.empty:
        cols = ["event", "model", "members", "crps", "crpss", "rmse", "bias", "acc",
                "spread_error"]
        print("\n" + df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nfigures -> {F.fig_dir()}")


if __name__ == "__main__":
    make_all()
