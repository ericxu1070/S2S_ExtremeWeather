#!/usr/bin/env python
"""Phase 4 maps: what the pilot's tail looks like in space, against truth and baselines.

Four figures. Every panel in the first three is the SAME reduction - the 7-day-mean 2 m
temperature anomaly over the verification week ending at the event peak, i.e. exactly the
field whose cos(lat)-weighted box mean IS ``A_L`` - computed by
``xres.xmetrics.forecast_field`` through ``aires.aindex.field``. Walkers, direct ensemble
members and ERA5 truth all land on the same 105 x 237 CONUS grid, so they are directly
subtractable and can share one color scale without rescaling anything.

    compare   truth, the AI+RES tail, and every baseline on ONE shared scale
    error     each method minus ERA5, with HRRR minus ERA5 as the reference for how much
              is actually known about what happened
    walk      the most extreme lineage in space, snapshot by snapshot, Z500 over T2m
    spread    per-gridpoint mean, spread and exceedance probability, method by method

Why the maps and not just the box mean
-------------------------------------
``A_L`` is an area mean, and an area mean can be reached by the wrong field: a displaced
ridge, or a broad weak warm anomaly with no dome in it, scores the same number as the
event. Nothing in the exceedance curve or the trajectory plot can tell those apart. The
error figure is the check, and ``HRRR - ERA5`` is on it on purpose - a method whose error
is smaller than the disagreement between two observational products has not been shown
to be wrong.

    PYTHONPATH=. python -m aires.amaps --event PNW_HeatDome_2021 --tag pilot
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from aires import aindex as AI
from aires import run_aires as R
from fcn3 import fevents as F
from gencast_s2s import config as C

FIGURES = ("compare", "error", "walk", "spread")

# The one place the panel palette is decided. Anomalies are signed, so every T2m panel is
# diverging and centred on zero; only the spread and probability columns are sequential.
CMAP_ANOM = "RdBu_r"
CMAP_ERR = "BrBG_r"
# NOT magma_r: its top end is black, and the sd field saturates there over most of CONUS,
# which erases the coastlines and state borders the reader needs to place the pattern.
CMAP_SD = "Purples"
CMAP_P = "YlGnBu"
BOX_EDGE = "#111111"


# --------------------------------------------------------------------------- #
# Truth
# --------------------------------------------------------------------------- #
def truth_field(event: str, metric: str = "t2m_anom",
                source: str = "era5") -> xr.DataArray:
    """The observed 7-day-mean field for ``metric``. ``source`` is ``era5`` or ``hrrr``.

    The stem and the variable both come from ``metric`` rather than being hardcoded to
    ``t2m_anom``: ``fcn3.fevents.era5_truth_path`` and ``xres.xhrrr`` name these files
    ``<event>_verif_<metric>.nc`` / ``<event>_hrrr_verif_<metric>.nc``, and every AI+RES
    production so far has happened to be T2m. A wind- or precip-metric event would
    otherwise have loaded some other event's T2m file, or crashed on a missing key,
    rather than saying which file it wanted.
    """
    stem = (f"{event}_verif_{metric}.nc" if source == "era5"
            else f"{event}_hrrr_verif_{metric}.nc")
    p = C.OBS_DIR / stem
    if not p.exists():
        raise SystemExit(f"no observed truth at {p}")
    with xr.open_dataset(p) as ds:
        return ds[metric].load()


# --------------------------------------------------------------------------- #
# Fields from walkers
# --------------------------------------------------------------------------- #
def _window_segments(ctx: R.Ctx, segments) -> list[int]:
    """The trailing segments whose frames can cover the verification window.

    Taking only these instead of all seven is what keeps the loader under a few hundred MB:
    the window is the last 6 days of a 21-day walk, so five sixths of every trajectory is
    irrelevant to an ``A_L`` field and is never read.
    """
    peak_lead = float(ctx.horizon_days)
    need = peak_lead - 6.0
    out = [s for s in segments if s * A.GATE3_SEG_DAYS > need - 1e-9]
    if not out:
        raise SystemExit(f"no segment covers the window starting at {need} d")
    return out


def _load_t2m(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as ds:
        d = ds[["2m_temperature"]].load()
    d = d.drop_vars([c for c in ("lead_h",) if c in d.coords])
    return d.isel(batch=0, drop=True) if "batch" in d.dims else d


def res_al_fields(ctx: R.Ctx, d: dict) -> xr.DataArray:
    """``(walker, lat, lon)`` A_L fields for the 64 surviving lineages.

    Assembled across the genealogy, exactly as ``run_aires.realized_indices`` does it - a
    slot is not a lineage, so the segments are looked up through ``lineage`` and not by
    walker index.
    """
    cfg = d["config"]
    legs = A.res_legs(cfg["leads"], cfg["horizon_days"], cfg["C"])
    segs = _window_segments(ctx, sorted({s for l in legs for s in l.segments}))
    lin = d["realized"]["lineage"]
    n = int(cfg["n_walkers"])
    out = []
    for w in range(n):
        parts = [_load_t2m(ctx.diag_path(int(lin[w][str(s)]), s)) for s in segs]
        cube = xr.concat(parts, dim="time", coords="minimal",
                         compat="override").sortby("time")
        AI.check_window(cube, ctx.ev.peak, ctx.ev.metric, where=f"w{w:02d} window")
        out.append(AI.field(cube, ctx.ev.peak, ctx.ev.metric))
        for p in parts:
            p.close()
        cube.close()
    return xr.concat(out, dim="walker")


def gate3_al_fields(event: str, peak, metric: str) -> xr.DataArray:
    """``(walker, lat, lon)`` A_L fields for the free-running Gate 3 walkers.

    No genealogy: these walkers were never resampled, so segment ``k`` of walker ``w`` is
    always walker ``w``'s own. They are the fairest GenCast baseline in the set - same
    checkpoint, same init, same 12 h step, same code path as the pilot's walkers, with the
    resampling switched off.
    """
    segs = [k for k, lead, _ in A.gate3_segments() if lead > 21.0 - 6.0 - 1e-9]
    out, have = [], []
    for w in range(A.GATE3_N_WALKERS):
        paths = [A.diag_path(event, w, s) for s in segs]
        if not all(p.exists() for p in paths):
            continue
        parts = [_load_t2m(p) for p in paths]
        cube = xr.concat(parts, dim="time", coords="minimal",
                         compat="override").sortby("time")
        AI.check_window(cube, peak, metric, where=f"gate3 w{w:02d}")
        out.append(AI.field(cube, peak, metric))
        have.append(w)
        for p in parts:
            p.close()
        cube.close()
    if not out:
        raise SystemExit(f"no Gate 3 walker diagnostics under {A.walker_dir(event, 0).parent}")
    print(f"    Gate 3 walkers: {len(have)}/{A.GATE3_N_WALKERS} available")
    return xr.concat(out, dim="walker")


def cube_al_fields(path: Path, peak, metric: str, label: str) -> xr.DataArray:
    """``(member, lat, lon)`` A_L fields from a cached forecast cube.

    Works for both the 12-hourly GenCast cubes and the 6-hourly FCN3 cube: the window is
    resolved from the cube's own step, and ``check_window`` refuses a cube that does not
    span it rather than silently averaging over the frames that happen to be inside.
    """
    if not path.exists():
        raise SystemExit(f"no cube at {path}")
    with xr.open_dataset(path) as ds:
        n = AI.check_window(ds, peak, metric, where=label)
        f = AI.field(ds, peak, metric).load()
    print(f"    {label}: {f.sizes.get('member', 1)} member(s), {n} window frames")
    return f


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #
def _box_mean(da: xr.DataArray, box) -> float:
    return float(AI.area_mean(da, box))


def build_panels(ctx: R.Ctx, d: dict) -> dict:
    """Every field the four figures draw, loaded once.

    Returned as plain DataArrays keyed by a short name, plus the populations they were
    reduced from, so the spread figure can use members while the comparison figure uses
    their means without reading anything twice.
    """
    ev, peak, metric = ctx.event, ctx.ev.peak, ctx.ev.metric
    box = AI.box_for(ev)
    obs = truth_field(ev, metric, "era5")
    al = np.asarray(d["realized"]["box"], dtype=float)
    sign = float(d["config"].get("tail_sign", 1.0))
    observed = float(d["observed"])

    print("  loading fields")
    res = res_al_fields(ctx, d)
    print(f"    AI+RES: {res.sizes['walker']} surviving lineages")
    # Canonical accessors, not hand-built paths: F.gencast_cube_path is the same 0.25 deg
    # cube the FCN3-vs-GenCast head-to-head scored, and F.cube_path is that experiment's
    # FCN3 cube. Reusing them is what makes the baselines on this figure literally the
    # published baselines and not a re-derivation.
    #
    # Every baseline (and the HRRR overlay) is OPTIONAL, because an extension event
    # legitimately lacks some of them: Southwest/California run without a Gate 3 tree,
    # only the six head-to-head events have an FCN3 context cube, SCentral has no xres
    # cube unless the optional baseline job ran, and its HRRR truth needs Derecho. The
    # figures already key their panels by presence; what is NOT optional is having at
    # least one baseline, or the "comparison" would be a self-portrait.
    try:
        g3 = gate3_al_fields(ev, peak, metric)
    except SystemExit as e:
        print(f"    (no Gate 3 walker baseline: {e})")
        g3 = None
    gc_path, fc_path = F.gencast_cube_path(ctx.ev), F.cube_path(ctx.ev)
    gc = (cube_al_fields(gc_path, peak, metric, "GenCast 0.25 deg")
          if gc_path.exists() else None)
    if gc is None:
        print(f"    (no xres GenCast cube at {gc_path})")
    fc = cube_al_fields(fc_path, peak, metric, "FCN3") if fc_path.exists() else None
    if fc is None:
        print(f"    (no FCN3 context cube at {fc_path})")
    if gc is None and fc is None and g3 is None:
        raise SystemExit("no baseline available at all: need the xres cube, a Gate 3 "
                         "tree, or an FCN3 cube to compare against")
    try:
        hrrr = truth_field(ev, metric, "hrrr")
    except SystemExit:
        print("    (no HRRR truth for this event)")
        hrrr = None

    k = max(2, int(round(0.15 * al.size)))
    top = np.argsort(sign * al)[-k:]
    reached = np.flatnonzero(sign * al >= sign * observed)
    imax = int(np.argmax(sign * al))

    pops = {"res": res}
    panels = {
        "obs_era5": (obs, "ERA5 observed", "truth"),
        "res_mean": (res.mean("walker"), f"AI+RES  all {al.size} (mean)", "res"),
        "res_top": (res.isel(walker=top).mean("walker"),
                    f"AI+RES  top {k} (mean)", "res"),
        "res_max": (res.isel(walker=imax), "AI+RES  single most extreme", "res"),
    }
    if hrrr is not None:
        panels["obs_hrrr"] = (hrrr, "HRRR observed", "truth")
    if reached.size:
        # An empty selection would draw an all-NaN panel captioned "the 0 that reached
        # ERA5" - the absence is already said, louder, by res_max falling short.
        panels["res_reached"] = (res.isel(walker=reached).mean("walker"),
                                 f"AI+RES  the {reached.size} that reached ERA5 (mean)",
                                 "res")
    if gc is not None:
        pops["gencast_xres"] = gc
        panels["gc_mean"] = (gc.mean("member"),
                             f"GenCast 0.25 deg  {gc.sizes['member']}-member mean", "base")
        panels["gc_max"] = (_extreme_member(gc, "member", box, sign),
                            "GenCast 0.25 deg  most extreme member", "base")
    if fc is not None:
        pops["fcn3"] = fc
        panels["fc_mean"] = (fc.mean("member"),
                             f"FCN3  {fc.sizes['member']}-member mean", "base")
        panels["fc_max"] = (_extreme_member(fc, "member", box, sign),
                            "FCN3  most extreme member", "base")
    if g3 is not None:
        pops["gencast_walkers"] = g3
        panels["g3_mean"] = (g3.mean("walker"),
                             f"GenCast walkers  {g3.sizes['walker']} free-running (mean)",
                             "base")
        panels["g3_max"] = (_extreme_member(g3, "walker", box, sign),
                            "GenCast walkers  most extreme", "base")
    return dict(panels=panels, pops=pops, box=box, obs=obs, observed=observed,
                sign=sign, al=al, k=k, n_reached=int(reached.size))


def _extreme_member(da: xr.DataArray, dim: str, box, sign: float) -> xr.DataArray:
    """The member whose own box mean is furthest into the event's tail.

    Selected on the box mean, not on a grid-point maximum: the observable the experiment
    is about is the area mean, and the member with the hottest single grid cell is often
    not the member with the most extreme event.
    """
    vals = AI.area_mean(da, box)
    return da.isel({dim: int(np.argmax(sign * np.asarray(vals.values)))})


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _map_ax(fig, nrow, ncol, idx, ccrs):
    ax = fig.add_subplot(nrow, ncol, idx, projection=ccrs.PlateCarree())
    ax.set_extent(C.CONUS_EXTENT, ccrs.PlateCarree())
    return ax


def _draw(ax, da, cmap, vmin, vmax, ccrs):
    from gencast_s2s.plotting import _to_180, _add_conus_features
    dd = _to_180(da)
    m = ax.pcolormesh(dd.lon, dd.lat, dd.values, transform=ccrs.PlateCarree(),
                      cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    _add_conus_features(ax)
    return m


def _box_outline(ax, box, ccrs):
    """Outline the scored box. Without it a reader cannot tell which part of a panel the
    single number in its title came from, and every panel here has such a number."""
    import matplotlib.patches as mpatches
    lo_lon = ((box.lon[0] + 180) % 360) - 180
    hi_lon = ((box.lon[1] + 180) % 360) - 180
    ax.add_patch(mpatches.Rectangle(
        (lo_lon, box.lat[0]), hi_lon - lo_lon, box.lat[1] - box.lat[0],
        transform=ccrs.PlateCarree(), fill=False, edgecolor=BOX_EDGE, lw=1.4,
        zorder=10))


def _shared_cbar(fig, m, axes, label, shrink=0.42):
    """ONE colorbar for a whole shared-scale grid.

    A per-panel colorbar on a grid that shares its scale repeats the same axis a dozen
    times, and worse, it lands between two rows of panels where it reads as a caption for
    the row above. One bar under the figure says the same thing once and gives the maps
    back their space.
    """
    cb = fig.colorbar(m, ax=axes, orientation="horizontal", pad=0.035, shrink=shrink,
                      aspect=45)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb


def _cbar(fig, m, ax, label):
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.04, shrink=0.86,
                      aspect=30)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def _sym(*arrays) -> float:
    v = np.concatenate([np.asarray(a).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    return float(np.nanmax(np.abs(v))) or 1.0


def _fig_path(ctx: R.Ctx, kind: str) -> Path:
    A.fig_dir(ctx.event).mkdir(parents=True, exist_ok=True)
    return A.fig_dir(ctx.event) / f"aires_map_{kind}_{ctx.event}_{ctx.tag}.png"


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
    return path


# --------------------------------------------------------------------------- #
# Figure 1 - the comparison
# --------------------------------------------------------------------------- #
# Row-major on a 3-wide grid, and the rows are the grouping: truth, the AI+RES tail,
# every baseline's MEAN, every baseline's most extreme member. Reading down column 1 then
# compares like with like (GenCast 0.25 deg mean above GenCast 0.25 deg extreme).
ORDER_COMPARE = ("obs_era5", "obs_hrrr", "res_mean",
                 "res_top", "res_reached", "res_max",
                 "gc_mean", "g3_mean", "fc_mean",
                 "gc_max", "g3_max", "fc_max")


def plot_compare(ctx: R.Ctx, dat: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    panels, box = dat["panels"], dat["box"]
    keys = [k for k in ORDER_COMPARE if k in panels]
    vmax = _sym(*[panels[k][0].values for k in keys])
    ncol, nrow = 3, int(np.ceil(len(keys) / 3))
    fig = plt.figure(figsize=(13.2, 2.55 * nrow))
    axes, m = [], None
    for i, key in enumerate(keys, start=1):
        da, title, kind = panels[key]
        ax = _map_ax(fig, nrow, ncol, i, ccrs)
        m = _draw(ax, da, CMAP_ANOM, -vmax, vmax, ccrs)
        _box_outline(ax, box, ccrs)
        ax.set_title(f"{title}\n" + r"box $A_L$ = " + f"{_box_mean(da, box):+.2f} K",
                     fontsize=9.2, linespacing=1.4,
                     color="#b2182b" if kind == "truth" else "black")
        axes.append(ax)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.075, hspace=0.40,
                        wspace=0.04)
    _shared_cbar(fig, m, axes, "7-day-mean T2m anomaly  (K)")
    fig.suptitle(
        f"AI+RES pilot - {ctx.event} ({ctx.tag}): the observable as a FIELD, one shared "
        f"scale.\nThe number under each panel is that panel's own box mean - the same "
        f"$A_L$ every score in this experiment is taken on. Black rectangle = the scored "
        f"box.", fontsize=11, y=0.995, va="top")
    return _save(fig, _fig_path(ctx, "compare"))


# --------------------------------------------------------------------------- #
# Figure 2 - error against ERA5
# --------------------------------------------------------------------------- #
ORDER_ERROR = ("obs_hrrr", "res_mean", "res_top",
               "res_reached", "res_max", "gc_mean",
               "g3_mean", "fc_mean", "gc_max",
               "g3_max", "fc_max")


def plot_error(ctx: R.Ctx, dat: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    panels, box, obs = dat["panels"], dat["box"], dat["obs"]
    keys = [k for k in ORDER_ERROR if k in panels]
    diffs = {k: (panels[k][0] - obs) for k in keys}
    vmax = _sym(*[diffs[k].values for k in keys])
    ncol, nrow = 3, int(np.ceil((len(keys) + 1) / 3))
    fig = plt.figure(figsize=(13.2, 2.55 * nrow))
    axes, m, rank = [], None, []
    for i, key in enumerate(keys, start=1):
        _, title, kind = panels[key]
        da = diffs[key]
        ax = _map_ax(fig, nrow, ncol, i, ccrs)
        m = _draw(ax, da, CMAP_ERR, -vmax, vmax, ccrs)
        _box_outline(ax, box, ccrs)
        inbox = AI.crop(da, box).values
        rmse = float(np.sqrt(np.nanmean(inbox ** 2)))
        rank.append((rmse, title))
        ax.set_title(f"{title}  -  ERA5\nbox bias {_box_mean(da, box):+.2f} K, "
                     f"box RMSE {rmse:.2f} K", fontsize=9.2, linespacing=1.4,
                     color="#b2182b" if kind == "truth" else "black")
        axes.append(ax)
    # The spare grid slot carries the ranking rather than whitespace: with eleven panels
    # on one scale the eye cannot order them, and the ordering is the finding.
    tax = fig.add_subplot(nrow, ncol, len(keys) + 1)
    tax.axis("off")
    lines = [f"{r:5.2f} K   {t}" for r, t in sorted(rank)]
    tax.text(0.0, 0.98, "box RMSE against ERA5, best first", fontsize=9.5,
             fontweight="bold", va="top", family="monospace")
    tax.text(0.0, 0.88, "\n".join(lines), fontsize=8.2, va="top", family="monospace",
             linespacing=1.6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.075, hspace=0.40,
                        wspace=0.04)
    # Explicit cax, not ax=axes: the eleven map axes span all three columns, so a colorbar
    # that steals space from them centres itself across the full width and lands on top of
    # the ranking panel in the twelfth slot.
    cax = fig.add_axes([0.09, 0.035, 0.42, 0.013])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal")
    cb.set_label("difference from ERA5  (K)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    hrrr_note = (
        "\nThe first panel is HRRR minus ERA5 - two observational products disagreeing "
        "about the same week. A method whose error is no larger than that has not been "
        "shown to have the event in the wrong place." if "obs_hrrr" in panels else "")
    fig.suptitle(
        f"AI+RES pilot - {ctx.event} ({ctx.tag}): each field MINUS ERA5, one shared scale."
        + hrrr_note, fontsize=11, y=0.995, va="top")
    return _save(fig, _fig_path(ctx, "error"))


# --------------------------------------------------------------------------- #
# Figure 3 - the walk in space
# --------------------------------------------------------------------------- #
def _frame_at(ctx: R.Ctx, lin_of_walker: dict, segments, when, var: str,
              level: int | None = None) -> xr.DataArray:
    """One variable at one absolute time, from whichever segment of the lineage holds it."""
    for s in segments:
        p = ctx.diag_path(int(lin_of_walker[str(s)]), s)
        with xr.open_dataset(p) as ds:
            t = pd.DatetimeIndex(ds["time"].values)
            if pd.Timestamp(when) not in set(t):
                continue
            da = ds[var]
            if "batch" in da.dims:
                da = da.isel(batch=0, drop=True)
            if level is not None:
                da = da.sel(level=level)
            return da.sel(time=pd.Timestamp(when)).load()
    raise KeyError(f"no frame at {when} in this lineage")


def _anom_frame(ctx: R.Ctx, lin_of_walker: dict, segments, when) -> xr.DataArray:
    """The instantaneous T2m anomaly at ``when`` - the running index's own field."""
    t2m = _frame_at(ctx, lin_of_walker, segments, when, "2m_temperature")
    cube = t2m.expand_dims(time=[pd.Timestamp(when)]) if "time" not in t2m.dims else t2m
    ds = xr.Dataset({"2m_temperature": cube})
    return AI.instantaneous_field(ds, "t2m_anom").squeeze(drop=True)


def plot_walk(ctx: R.Ctx, dat: dict, d: dict, leads=(3, 6, 9, 12, 15, 18, 21)) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from gencast_s2s.plotting import _to_180

    cfg = d["config"]
    legs = A.res_legs(cfg["leads"], cfg["horizon_days"], cfg["C"])
    segments = sorted({s for l in legs for s in l.segments})
    al = dat["al"]
    sign = dat["sign"]
    imax = int(np.argmax(sign * al))
    lin = d["realized"]["lineage"][imax]
    box = dat["box"]

    print(f"  walk: lineage of the most extreme walker (slot {imax}, "
          f"A_L {al[imax]:+.2f} K)")
    frames, z500 = [], []
    for L in leads:
        when = ctx.valid_at(L)
        frames.append((L, _anom_frame(ctx, lin, segments, when)))
        z500.append(_frame_at(ctx, lin, segments, when, "geopotential", 500) / 9.80665)

    al_field = dat["panels"]["res_max"][0]
    vmax = _sym(*[f.values for _, f in frames], al_field.values)

    ncol, nrow = 4, 2
    fig = plt.figure(figsize=(17.0, 6.6))
    for i, ((L, f), z) in enumerate(zip(frames, z500), start=1):
        ax = _map_ax(fig, nrow, ncol, i, ccrs)
        m = _draw(ax, f, CMAP_ANOM, -vmax, vmax, ccrs)
        zz = _to_180(z)
        cs = ax.contour(zz.lon, zz.lat, zz.values, levels=8, colors="k",
                        linewidths=0.55, alpha=0.75, transform=ccrs.PlateCarree())
        ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.0f")
        _box_outline(ax, box, ccrs)
        ax.set_title(f"lead {L} d   ({ctx.valid_at(L):%Y-%m-%d %HZ})\n"
                     f"box mean {_box_mean(f, box):+.2f} K", fontsize=8.6,
                     linespacing=1.4)
    ax = _map_ax(fig, nrow, ncol, len(frames) + 1, ccrs)
    m = _draw(ax, al_field, CMAP_ANOM, -vmax, vmax, ccrs)
    _box_outline(ax, box, ccrs)
    ax.set_title(f"the 7-day mean = $A_L$\nbox mean {al[imax]:+.2f} K", fontsize=8.6,
                 linespacing=1.4, color="#1a6fa8")
    cb = fig.colorbar(m, ax=fig.axes, orientation="horizontal", pad=0.05, shrink=0.35,
                      aspect=45)
    cb.set_label("instantaneous T2m anomaly, and the 7-day mean in the last panel  (K)",
                 fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle(
        f"AI+RES pilot - {ctx.event} ({ctx.tag}): the most extreme surviving lineage in "
        f"space.\nContours are 500 hPa geopotential height (m) from the same walker - the "
        f"ridge that carries the dome. The last panel is the 7-day mean the score is "
        f"taken on.", fontsize=10.5, y=1.0, va="top")
    return _save(fig, _fig_path(ctx, "walk"))


# --------------------------------------------------------------------------- #
# Figure 4 - spread, and where the probability lives
# --------------------------------------------------------------------------- #
def plot_spread(ctx: R.Ctx, dat: dict, d: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    pops, box = dat["pops"], dat["box"]
    thr = float(dat["observed"])
    sign = dat["sign"]
    w = np.asarray(d["weights"], dtype="float64")
    Z = float(np.exp(d["log_Z"]))

    rows = []
    for key, label_fmt, dim, weighted in (
        ("res", "AI+RES  {n} resampled", "walker", True),
        ("gencast_xres", "GenCast 0.25 deg  {n} direct", "member", False),
        ("fcn3", "FCN3  {n} direct", "member", False),
        ("gencast_walkers", "GenCast walkers  {n} free-running", "walker", False),
    ):
        if key not in pops:      # optional baselines - see build_panels
            continue
        pop = pops[key]
        label = label_fmt.format(n=pop.sizes[dim])
        ind = (sign * pop >= sign * thr).astype("float64")
        if weighted:
            # The estimator itself, applied per grid point: P = Z * mean(w * 1{...}). The
            # same formula dmc.DMCResult.expectation uses for the scalar exceedance curve,
            # so this map and the curve cannot disagree.
            wa = xr.DataArray(w, dims=[dim], coords={dim: pop[dim]}) \
                if dim in pop.coords else xr.DataArray(w, dims=[dim])
            prob = Z * (ind * wa).mean(dim)
            pnote = "weighted estimate"
        else:
            prob = ind.mean(dim)
            pnote = f"raw member fraction (floor 1/{pop.sizes[dim]})"
        rows.append((label, pop.mean(dim), pop.std(dim, ddof=1), prob, pnote))

    mean_max = _sym(*[r[1].values for r in rows])
    sd_hi = float(np.nanmax([np.nanmax(r[2].values) for r in rows]))
    p_hi = float(np.nanmax([np.nanmax(r[3].values) for r in rows])) or 1.0

    ncol, nrow = 3, len(rows)
    fig = plt.figure(figsize=(13.2, 2.55 * nrow))
    col_axes = [[] for _ in range(ncol)]
    col_m = [None] * ncol
    col_unit = [None] * ncol
    for j, (label, mean, sd, prob, pnote) in enumerate(rows):
        for c, (da, cmap, vmin, vmax, unit, what) in enumerate([
            (mean, CMAP_ANOM, -mean_max, mean_max, "population mean  (K)",
             "population mean"),
            (sd, CMAP_SD, 0.0, sd_hi, "population sd  (K)", "population sd"),
            (prob, CMAP_P, 0.0, p_hi,
             f"P(local 7-day anomaly $\\geq$ {thr:+.2f} K)",
             f"P(local $\\geq$ {thr:+.2f} K)"),
        ]):
            i = j * ncol + c + 1
            ax = _map_ax(fig, nrow, ncol, i, ccrs)
            m = _draw(ax, da, cmap, vmin, vmax, ccrs)
            _box_outline(ax, box, ccrs)
            extra = (f"box {_box_mean(da, box):+.2f} K" if c < 2
                     else f"box {_box_mean(da, box):.3f}  -  {pnote}")
            ax.set_title(f"{label}\n{what}   {extra}", fontsize=8.8, linespacing=1.4)
            col_axes[c].append(ax)
            col_m[c], col_unit[c] = m, unit
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.085, hspace=0.40,
                        wspace=0.04)
    for c in range(ncol):
        _shared_cbar(fig, col_m[c], col_axes[c], col_unit[c], shrink=0.9)
    fig.suptitle(
        f"AI+RES pilot - {ctx.event} ({ctx.tag}): spread, and where the probability lives."
        f"\nColumn 3 is the same estimator as the exceedance curve, per grid point: AI+RES "
        f"uses the weights, the direct ensembles are raw member fractions and cannot go "
        f"below one member. Column 2 is why AI+RES's 64 are worth fewer than 64: a "
        f"tail-selected population of clones is TIGHTER than a free ensemble.",
        fontsize=11, y=0.995, va="top")
    return _save(fig, _fig_path(ctx, "spread"))


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default=os.environ.get("AIRES_EVENT", "PNW_HeatDome_2021"))
    ap.add_argument("--tag", default=A.RES_TAG)
    ap.add_argument("--only", default=",".join(FIGURES),
                    help=f"comma-separated subset of {FIGURES}")
    a = ap.parse_args(argv)

    ctx = R.Ctx(ev=R._event(a.event), tag=a.tag, n_walkers=A.RES_N_WALKERS,
                members=A.RES_M_MEMBERS, C=A.RES_C, leads=A.RES_LEAD_DAYS,
                horizon_days=A.RES_HORIZON_DAYS, dmc_seed=A.RES_DMC_SEED,
                base_seed=A.WALKER_BASE_SEED)
    p = A.res_dir(ctx.event, ctx.tag) / "compare.json"
    if not p.exists():
        raise SystemExit(f"no {p}; run --stage compare first")
    d = R.read_json(p)
    ctx.n_walkers = int(d["config"]["n_walkers"])

    want = [x.strip() for x in a.only.split(",") if x.strip()]
    bad = [x for x in want if x not in FIGURES]
    if bad:
        raise SystemExit(f"unknown figure(s) {bad}; want a subset of {list(FIGURES)}")

    print(f"[amaps] {ctx.event} tag={ctx.tag}: {', '.join(want)}")
    print(f"  {AI.describe(ctx.event)}")
    dat = build_panels(ctx, d)
    if "compare" in want:
        plot_compare(ctx, dat)
    if "error" in want:
        plot_error(ctx, dat)
    if "walk" in want:
        plot_walk(ctx, dat, d)
    if "spread" in want:
        plot_spread(ctx, dat, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
