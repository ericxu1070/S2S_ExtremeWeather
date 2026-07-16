"""CONUS map grids in the downscaler fig1/fig2 style, one per (event, lead) — cached data only.

For each representative event (one per family by default) and each lead (weeks 2/3/4):

    rows    = the event's CACHED verification metrics: its headline metric plus the free
              secondary where one exists (u850_speed_max / tp_max12h). Nothing is derived
              from the cubes and no ERA5 access happens — heat/cold events get one row,
              hurricane/rain events two.
    columns = ERA5 0.25deg truth | GenCast 0.25deg ens-mean | GenCast 1.0deg ens-mean
              | error 0.25deg | error 1.0deg      (error = ens-mean - truth on the fc grid)

Style is ported from ``downscaler/scripts/xres_downscale_demo.py::_maps_figure`` (one
shared colorbar per row for the field columns, a dedicated symmetric error colorbar per
row, RMSE/bias badges, coast/border/state features); semantics follow xres conventions
(the anomaly row uses a diverging scale centered on 0; badges are cos(lat)-weighted like
xscores' CRPS/RMSE).

Usage (cwd = repo root, env ``moe``; login-node CPU only):

    python -m xres.xmapgrid                        # 4 events x 3 weeks
    python -m xres.xmapgrid --weeks 3 --events HurricaneIda_2021
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.gridspec import GridSpec     # noqa: E402
import numpy as np                           # noqa: E402

from gencast_s2s import config as C          # noqa: E402
from gencast_s2s.plotting import _to_180     # noqa: E402

from . import xconfig as X                   # noqa: E402
from . import xmetrics as XM                 # noqa: E402
from .xplotting import era5_truth, members, _on   # noqa: E402

# One event per family (heat / cold / hurricane / extreme rain); override with
# XRES_MAP_EVENTS="EventA,EventB" (must be keys of xconfig.XRES_EVENTS).
MAP_EVENTS = [
    "PNW_HeatDome_2021",
    "WinterStorm_Uri_2021",
    "HurricaneIda_2021",
    "California_AR_Floods_2023",
]
_sel = os.environ.get("XRES_MAP_EVENTS")
if _sel:
    MAP_EVENTS = [s.strip() for s in _sel.split(",") if s.strip()]

MAP_WEEKS = (2, 3, 4)

# Per-row (field cmap, error cmap). Field cmaps follow xres semantics for the anomaly and
# wind rows and the downscaler's cmocean rain map for precip; error maps follow the
# downscaler (BrBG for precip: brown = too dry, teal = too wet).
ROW_CMAPS = {
    "t2m_anom":       ("RdBu_r",   "RdBu_r"),
    "u850_speed":     ("viridis",  "RdBu_r"),
    "u850_speed_max": ("viridis",  "RdBu_r"),
    "tp_total":       ("cmo.rain", "BrBG"),
    "tp_max12h":      ("cmo.rain", "BrBG"),
}


def event_metrics(name):
    """The metrics already cached for this event: headline + free secondary."""
    m = X.event_metric(name)
    return [m] + ([XM.SECONDARY[m]] if m in XM.SECONDARY else [])


# --------------------------------------------------------------------------- #
# Drawing helpers (downscaler _maps_figure style).
# --------------------------------------------------------------------------- #
def _resolve_cmap(name):
    if isinstance(name, str) and name.startswith("cmo."):
        import cmocean
        return getattr(cmocean.cm, name[4:])
    return name


def _geo_ax(fig, gs_pos):
    import cartopy.crs as ccrs
    ax = fig.add_subplot(gs_pos, projection=ccrs.PlateCarree())
    ax.set_extent(C.CONUS_EXTENT, crs=ccrs.PlateCarree())
    return ax


def _draw(ax, da, vmin, vmax, cmap):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    da = _to_180(da)
    im = ax.pcolormesh(da.lon, da.lat, da.values, vmin=vmin, vmax=vmax,
                       cmap=_resolve_cmap(cmap), shading="auto",
                       transform=ccrs.PlateCarree(), rasterized=True)
    for feat, lw, ec in ((cfeature.COASTLINE, 0.5, "0.15"),
                         (cfeature.BORDERS, 0.4, "0.15"),
                         (cfeature.STATES, 0.25, "0.45")):
        try:
            ax.add_feature(feat, linewidth=lw, edgecolor=ec)
        except Exception:
            pass
    return im


def _wmean(da):
    w = np.cos(np.deg2rad(da["lat"]))
    return float(da.weighted(w).mean(("lat", "lon")))


def _wrmse(fc, truth_on_fc):
    w = np.cos(np.deg2rad(fc["lat"]))
    return float(np.sqrt(((fc - truth_on_fc) ** 2).weighted(w).mean(("lat", "lon"))))


def _field_range(metric, fields):
    vals = np.concatenate([f.values.ravel() for f in fields])
    vals = vals[np.isfinite(vals)]
    sp = XM.spec(metric)
    if sp.diverging:
        a = float(np.percentile(np.abs(vals), 99)) or 1.0
        return -a, a
    if metric.startswith("tp_"):
        return 0.0, max(float(np.percentile(vals, 99.5)), 1.0)
    lo, hi = np.percentile(vals, [1, 99])
    return float(lo), float(hi)


def _badge(ax, text, dark=True):
    fc, color = ("0.15", "white") if dark else ("white", "black")
    ax.text(0.98, 0.03, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc=fc, ec="none", alpha=0.7))


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def plot_event(name, weeks, outdirs):
    """One (rows x 5-map) figure from cached files; rows with missing data are dropped."""
    rows = []   # (metric, truth, {res: (n_members, ens_mean, err)})
    for metric in event_metrics(name):
        truth = era5_truth(name, metric)
        if truth is None:
            print(f"[maps3var] {name} [{metric}]: no ERA5 truth cached; dropping row")
            continue
        per_res = {}
        for res in X.RES_ORDER:
            mem = members(res, weeks, name, metric)
            if mem is None:
                print(f"[maps3var] {name} {res} week{weeks} [{metric}]: no cached "
                      f"members; dropping row")
                per_res = None
                break
            mean = mem.mean("member")
            per_res[res] = (mem.sizes["member"], mean, mean - _on(truth, mean))
        if per_res:
            rows.append((metric, truth, per_res))
    if not rows:
        print(f"[maps3var] {name} week{weeks}: nothing cached; skipping figure")
        return False

    # columns: 3 field maps | field cbar | 2 error maps | error cbar
    fig = plt.figure(figsize=(4.3 * 5 + 1.2, 3.2 * len(rows)))
    gs = GridSpec(len(rows), 7, figure=fig,
                  width_ratios=[1, 1, 1, 0.055, 1, 1, 0.055],
                  hspace=0.14, wspace=0.10)

    for r, (metric, truth, per_res) in enumerate(rows):
        sp = XM.spec(metric)
        fcmap, ecmap = ROW_CMAPS[metric]
        n25, mean25, err25 = per_res["0p25"]
        n10, mean10, err10 = per_res["1p0"]

        # --- field columns (one shared color scale per row) ---
        vmin, vmax = _field_range(metric, [truth, mean25, mean10])
        fields = [
            (truth,  "ERA5 0.25deg (truth)",             None),
            (mean25, f"GenCast 0.25deg mean ({n25} members)",
             f"RMSE {_wrmse(mean25, _on(truth, mean25)):.2f}"),
            (mean10, f"GenCast 1.0deg mean ({n10} members)",
             f"RMSE {_wrmse(mean10, _on(truth, mean10)):.2f}"),
        ]
        im = None
        for c, (da, title, badge) in enumerate(fields):
            ax = _geo_ax(fig, gs[r, c])
            im = _draw(ax, da, vmin, vmax, fcmap)
            if r == 0:
                ax.set_title(title, fontsize=11, pad=4)
            if c == 0:
                ax.text(-0.02, 0.5, f"{sp.label} ({sp.units})", transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=11)
            if badge:
                _badge(ax, badge, dark=True)
        fig.colorbar(im, cax=fig.add_subplot(gs[r, 3]))

        # --- error columns (shared symmetric scale per row) ---
        errv = np.concatenate([np.abs(e.values.ravel()) for e in (err25, err10)])
        lim = float(np.percentile(errv[np.isfinite(errv)], 99)) or 1.0
        ime = None
        for c, (err, tag) in enumerate([(err25, "0.25deg"), (err10, "1.0deg")]):
            axe = _geo_ax(fig, gs[r, 4 + c])
            ime = _draw(axe, err, -lim, lim, ecmap)
            if r == 0:
                axe.set_title(f"error {tag}\n(fc mean - ERA5)", fontsize=11, pad=4)
            _badge(axe, f"bias {_wmean(err):+.2f}", dark=False)
        fig.colorbar(ime, cax=fig.add_subplot(gs[r, 6]))

    fig.suptitle(f"{name} — week-{weeks} lead (init = peak - {C.lead_days_for(weeks)}d): "
                 "ERA5 truth vs GenCast ensemble means", fontsize=13, y=0.995)
    fname = f"{name}_week{weeks}_maps3var.png"
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[maps3var] {name} week{weeks} ({len(rows)} row(s)) -> {fname}")
    return True


def plot(weeks_list, events):
    outdirs = [X.XFIG_DIR / "maps3var", C.ROOT / "figures" / "xres" / "maps3var"]
    made = sum(plot_event(name, weeks, outdirs)
               for weeks in weeks_list for name in events)
    print(f"[maps3var] wrote {made}/{len(weeks_list) * len(events)} figures -> "
          f"{', '.join(str(d) for d in outdirs)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weeks", type=int, nargs="+", default=list(MAP_WEEKS))
    ap.add_argument("--events", nargs="+", default=MAP_EVENTS)
    args = ap.parse_args()
    unknown = [e for e in args.events if e not in X.XRES_EVENTS]
    if unknown:
        raise SystemExit(f"unknown event(s): {unknown}")
    plot(args.weeks, args.events)


if __name__ == "__main__":
    main()
