"""Cross-resolution comparison maps: ERA5 | HRRR | 0.25deg fc+err | 1.0deg fc+err.

One figure per event (and a stacked all-events grid), each row of panels laid out as:

    Observed (ERA5) | Observed (HRRR) | 0.25deg mean | 0.25deg error | 1.0deg mean | 1.0deg error

drawn in the event's headline metric (T2m anomaly / |U850| speed / total precip). Field
panels share a color scale; error panels (forecast - ERA5) share their own symmetric scale.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import xarray as xr                    # noqa: E402

from gencast_s2s import config as C    # noqa: E402
from gencast_s2s.plotting import _to_180, _add_conus_features  # noqa: E402

from . import xconfig as X             # noqa: E402
from . import xmetrics as XM           # noqa: E402
from . import xinference as XI         # noqa: E402
from . import xdata as XD              # noqa: E402
from . import xhrrr as XH              # noqa: E402


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def era5_truth(name, metric):
    p = XD.era5_truth_path(name, metric)
    return xr.open_dataset(p)[metric] if p.exists() else None


def hrrr_truth(name, metric):
    p = XH.hrrr_truth_path(name, metric)
    return xr.open_dataset(p)[metric] if p.exists() else None


def members(res, weeks, name, metric):
    p = XI.verif_path(res, weeks, name, metric)
    return xr.open_dataset(p)[metric] if p.exists() else None


def _on(da, ref):
    """Nearest-neighbour sample ``da`` onto the grid of ``ref`` (for point-wise error)."""
    out = da.sel(lat=ref["lat"], lon=ref["lon"], method="nearest")
    return out.assign_coords(lat=ref["lat"], lon=ref["lon"])


# --------------------------------------------------------------------------- #
# Color scales
# --------------------------------------------------------------------------- #
def _field_lims(metric, fields):
    vals = np.concatenate([f.values.ravel() for f in fields if f is not None])
    vals = vals[np.isfinite(vals)]
    sp = XM.spec(metric)
    if sp.diverging:
        a = float(np.nanmax(np.abs(vals))) or 1.0
        return -a, a, sp.field_cmap
    return float(np.nanmin(vals)), (float(np.nanmax(vals)) or 1.0), sp.field_cmap


def _panel(fig, pos, da, title, cmap, vmin, vmax, ccrs, label):
    pos = pos if isinstance(pos, tuple) else (pos,)
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    ax.set_extent(C.CONUS_EXTENT, ccrs.PlateCarree())
    da = _to_180(da)
    m = ax.pcolormesh(da.lon, da.lat, da.values, transform=ccrs.PlateCarree(),
                      cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    _add_conus_features(ax)
    ax.set_title(title, fontsize=9)
    fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.04, shrink=0.9, label=label)
    return ax


def _row(name, weeks):
    """Gather the six panels' data for one event (HRRR may be None)."""
    metric = X.event_metric(name)
    era5 = era5_truth(name, metric)
    hrrr = hrrr_truth(name, metric)
    panels = []   # (data, title, kind)  kind in {field, error}
    panels.append((era5, "Observed (ERA5)", "field"))
    panels.append((hrrr, "Observed (HRRR)", "field"))
    fields = [era5, hrrr]
    errors = []
    for res in X.RES_ORDER:
        mem = members(res, weeks, name, metric)
        if mem is None:
            panels.append((None, f"{X.res_spec(res)['label']} mean (missing)", "field"))
            panels.append((None, f"{X.res_spec(res)['label']} error (missing)", "error"))
            continue
        mean = mem.mean("member")
        err = mean - _on(era5, mean) if era5 is not None else None
        panels.append((mean, f"{X.res_spec(res)['label']} mean", "field"))
        panels.append((err, f"{X.res_spec(res)['label']} error (fc-ERA5)", "error"))
        fields.append(mean)
        if err is not None:
            errors.append(err)
    return metric, panels, fields, errors


def plot_event(name, weeks, ccrs):
    metric, panels, fields, errors = _row(name, weeks)
    panels = [p for p in panels if p[0] is not None]
    sp = XM.spec(metric)
    fmin, fmax, cmap = _field_lims(metric, fields)
    emax = max((float(np.nanmax(np.abs(e.values))) for e in errors), default=1.0) or 1.0
    ncol = len(panels)
    fig = plt.figure(figsize=(4.2 * ncol, 4.0))
    for i, (da, title, kind) in enumerate(panels):
        if kind == "error":
            _panel(fig, (1, ncol, i + 1), da, title, "RdBu_r", -emax, emax, ccrs, sp.units)
        else:
            _panel(fig, (1, ncol, i + 1), da, title, cmap, fmin, fmax, ccrs, sp.units)
    fig.suptitle(f"{name} — week-{weeks} {sp.label} ({sp.units})  [init = peak - "
                 f"{C.lead_days_for(weeks)}d]", y=1.04, fontsize=12)
    fig.tight_layout()
    return fig


def make_maps(weeks=None):
    import cartopy.crs as ccrs
    weeks = X.WEEKS if weeks is None else weeks
    outdir = X.XFIG_DIR / "maps"
    outdir.mkdir(parents=True, exist_ok=True)
    names = [n for n in X.events()
             if any(members(r, weeks, n, X.event_metric(n)) is not None for r in X.RES_ORDER)]
    for name in names:
        fig = plot_event(name, weeks, ccrs)
        fig.savefig(outdir / f"{name}_compare_map.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    if names:
        print(f"[xres maps] wrote {len(names)} per-event compare maps -> {outdir}")
    return names
