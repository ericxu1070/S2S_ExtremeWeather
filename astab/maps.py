#!/usr/bin/env python
"""stage: maps - what the forecast T2m field looks like against ERA5, lead by lead.

The stability curve says *when* a chain stopped making an atmosphere. It cannot say what
the field looked like on the way there, and that is the question a reader asks next: does
the forecast decay smoothly into climatology, or does it hold a pattern and then break?
One figure per (event, model, initialization):

    row 1   the forecast's 2 m temperature
    row 2   ERA5 at the same valid time
    row 3   forecast - ERA5

    figures/astab/maps/<event>/<model>_wk<NN>.png

Three models per event per lead - the GenCast walker, FCN3 from an ERA5 IC (adapter-free),
FCN3 from the walker at 3 d (adapter-fed) - so the set is one figure per curve on
``stability.png``. They are separate files rather than panels of one sheet because the
three have different rollout lengths and different launch times, and stacking them would
force a shared column grid that two of the three do not have frames on.

Why the color scales are FIXED, and not per figure
--------------------------------------------------
The whole point is comparison: this model against that one, this lead against that one. A
scale computed per figure makes a 20 K error at 60 d look exactly like a 2 K error at 3 d,
which inverts the finding. So:

* the absolute rows share ONE scale per EVENT, taken from that event's own ERA5 series over
  every cached time (a heat dome in June and a Texas freeze in February genuinely need
  different ranges, and a single global scale would flatten both);
* the difference row is fixed at +-``DIFF_MAX_K`` for every figure in the sweep.

A panel past either scale saturates, on purpose. Saturation is never silent: every forecast
panel prints its own min/max when it leaves the shared scale, and every difference panel
prints its cos(lat)-weighted RMSE and bias, which are the numbers the eye cannot read off a
saturated map.

What the maps are NOT
--------------------
Not a skill score. Past ~8.5-12 d (Gate 2's rank-equivalence horizon, Gate 3's score-skill
curve) forecast-minus-ERA5 is measuring chaos, not error, and the difference row is
expected to grow to the amplitude of the anomaly field itself. What the maps ADD to the
stability curve is the distinction between that - a decorrelated but physical field - and a
blow-up, which looks nothing like weather. Job 1189's two silent divergences are the case
in point: the CONUS panels stayed textbook-normal while the polar stratosphere reached
1211 K, so a chain can look perfect here and still be broken. The red column frames mark
where the bounds check says that happened.

    PYTHONPATH=. python -m astab.run --stage maps
    PYTHONPATH=. python -m astab.run --stage maps --events PNW_HeatDome_2021 --weeks 10
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aires import aindex as AI
from astab import sconfig as SC
from astab.diag import MARGINAL_SEVERITY
from astab.refs import truth_fields
from astab.schedule import Chain, scoring_checkpoints
from gencast_s2s import config as C

# Fixed across every figure in the sweep - see the docstring.
DIFF_MAX_K = 12.0
CMAP_T2M = "RdYlBu_r"
CMAP_DIFF = "RdBu_r"

# Default number of lead columns. Enough that a 4-week chain loses almost nothing (it has
# nine 3-day checkpoints); longer chains are subsampled and the figure says by how much.
DEFAULT_NCOLS = 8

MODEL_LABEL = {
    "gencast": "GenCast walker (0.25 deg, 12 h step)",
    "fcn3_era5": "FCN3 from an ERA5 IC (adapter-free)",
    "fcn3_adapter": "FCN3 from the walker at 3 d (adapter-fed)",
}


# --------------------------------------------------------------------------- #
# Getting the forecast field
# --------------------------------------------------------------------------- #
def _walker_t2m(chain: Chain) -> xr.DataArray:
    """The walker's CONUS T2m, every segment on disk concatenated in time.

    ``diag.nc`` is the CONUS crop and its ``time`` is absolute, so segments concatenate
    across the ragged tail with no bookkeeping. Only ``2m_temperature`` is loaded: a full
    segment is 50 MB and a 70-day chain has 24 of them.
    """
    parts = []
    for (k, _lead, _n) in chain.schedule:
        p = chain.diag_path(k)
        if not p.exists():
            continue
        with xr.open_dataset(p) as ds:
            da = ds["2m_temperature"].load()
        parts.append(AI._squeeze(da))
    if not parts:
        return xr.DataArray(np.zeros((0, 0, 0)), dims=("time", "lat", "lon"))
    out = xr.concat(parts, dim="time").sortby("time")
    # Segments share no frames, but a re-walk that overlapped would silently double one.
    _, uniq = np.unique(out["time"].values, return_index=True)
    return out.isel(time=np.sort(uniq))


def _cube_t2m(path: Path) -> xr.DataArray:
    if not path.exists():
        return xr.DataArray(np.zeros((0, 0, 0)), dims=("time", "lat", "lon"))
    with xr.open_dataset(path) as ds:
        da = ds["2m_temperature"].load()
    da = AI._squeeze(da)
    if "member" in da.dims:
        da = da.isel(member=0, drop=True)
    return da.sortby("time")


def forecast_t2m(chain: Chain, model: str) -> xr.DataArray:
    """``(time, lat, lon)`` CONUS T2m for one chain and one model, or an empty array."""
    if model == "gencast":
        return _walker_t2m(chain)
    if model == "fcn3_era5":
        return _cube_t2m(chain.fcn3_free_cube_path())
    if model == "fcn3_adapter":
        return _cube_t2m(chain.score_cube_path())
    raise ValueError(f"unknown model {model!r}; expected one of {SC.MODELS}")


def _columns(chain: Chain, fcst: xr.DataArray, truth: xr.DataArray,
             ncols: int) -> tuple[list[pd.Timestamp], int]:
    """The valid times to draw, and how many available checkpoints were dropped.

    Only 3-day scoring checkpoints where BOTH the forecast and ERA5 have a frame. The
    first and last available are always kept - the last is the whole point of the figure,
    and a subsample that dropped it would end the story one column early.

    Off ``horizon_days``, never off ``Chain.schedule``: at an FCN3-extension lead the walk
    schedule is three days long and both FCN3 arms run the full horizon, so the columns
    come from the horizon and the ``avail`` intersection drops whatever a given model has
    no frame for.
    """
    if fcst.sizes.get("time", 0) == 0 or truth.sizes.get("time", 0) == 0:
        return [], 0
    have_f = set(pd.DatetimeIndex(fcst["time"].values))
    have_t = set(pd.DatetimeIndex(truth["time"].values))
    want = [chain.valid_at(l) for l in scoring_checkpoints(chain.horizon_days)]
    want.append(chain.peak)                      # the peak IS a column, though never a
    avail = [t for t in want if t in have_f and t in have_t]   # scoring checkpoint
    if not avail or ncols <= 0 or len(avail) <= ncols:
        return avail, 0
    idx = np.unique(np.linspace(0, len(avail) - 1, ncols).round().astype(int))
    return [avail[i] for i in idx], len(avail) - len(idx)


# --------------------------------------------------------------------------- #
# Scales
# --------------------------------------------------------------------------- #
def event_t2m_scale(event: str) -> tuple[float, float]:
    """One absolute T2m COLOR scale per event, from ERA5's own 1st-99th percentile.

    Robust rather than min/max: a single Rocky Mountain grid cell at 250 K would otherwise
    set the floor for every panel of every figure of that event.
    """
    v = _truth_values(event)
    lo, hi = np.percentile(v, [1.0, 99.0])
    return float(np.floor(lo)), float(np.ceil(hi))


def event_t2m_range(event: str) -> tuple[float, float]:
    """The FULL range ERA5 covers for this event - what "off the scale" is measured against.

    Not the color scale. The color scale clips 2% of ERA5's own points by construction, so
    a healthy forecast leaves it on nearly every frame and flagging that would put a red
    label on every panel (it did). Leaving the range ERA5 never leaves is a statement.
    """
    v = _truth_values(event)
    return float(v.min()), float(v.max())


def _truth_values(event: str) -> np.ndarray:
    v = np.asarray(truth_fields(event).values, dtype="float64")
    return v[np.isfinite(v)]


def _wrmse(diff: xr.DataArray) -> float:
    return float(np.sqrt(AI.area_mean(diff ** 2)))


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _draw(ax, da, cmap, vmin, vmax, ccrs):
    from gencast_s2s.plotting import _add_conus_features, _to_180
    dd = _to_180(da)
    m = ax.pcolormesh(dd.lon, dd.lat, dd.values, transform=ccrs.PlateCarree(),
                      cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    _add_conus_features(ax)
    return m


def _corner(ax, text, color="#111111"):
    ax.text(0.015, 0.02, text, transform=ax.transAxes, fontsize=6.6, color=color,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.72))


def _diverged_at(chain: Chain, model: str) -> float | None:
    """Lead at which this chain's bound excursion first exceeded ``MARGINAL_SEVERITY``.

    Read from the reduced CSV rather than recomputed: the figure must mark exactly what
    the headline claims, and a second implementation of that rule is a second thing to
    keep in step.
    """
    p = SC.csv_path()
    if not p.exists():
        return None
    df = pd.read_csv(p)
    c = df[(df.model == model) & (df.event == chain.event) & (df.weeks == chain.weeks)
           & (df.metric == "bounds_severity")]
    big = c[c.value.astype(float) > MARGINAL_SEVERITY]
    return float(big.checkpoint_day.min()) if len(big) else None


def draw_chain_maps(chain: Chain, model: str, *, ncols: int = DEFAULT_NCOLS,
                    force: bool = False) -> Path | None:
    """One figure: forecast / ERA5 / difference, across lead. ``None`` if there is no data.

    The canvas is sized from the CONUS panel's own aspect (26 deg of latitude over 59 of
    longitude, fixed by ``set_extent`` under PlateCarree), not guessed - and the caption is
    hard-wrapped before it is set. An unwrapped ``suptitle`` is one text object as wide as
    its string, and ``bbox_inches="tight"`` then expands the canvas to fit it: the first
    version of this figure came out 34 inches wide for 16 inches of maps.
    """
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    out = SC.map_path(chain.event, model, chain.weeks)
    if out.exists() and not force:
        print(f"[maps] {chain.label} {model}: cached {out.name}, skip")
        return out

    fcst = forecast_t2m(chain, model)
    if fcst.sizes.get("time", 0) == 0:
        print(f"[maps] {chain.label} {model}: no forecast frames on disk, skip")
        return None
    truth = truth_fields(chain.event)
    cols, dropped = _columns(chain, fcst, truth, ncols)
    if not cols:
        print(f"[maps] {chain.label} {model}: no valid time where both the forecast and "
              f"ERA5 have a frame, skip")
        return None
    if truth.sizes["lat"] != fcst.sizes["lat"] or truth.sizes["lon"] != fcst.sizes["lon"]:
        raise SystemExit(
            f"[maps] {chain.label} {model}: ERA5 truth is "
            f"{truth.sizes['lat']}x{truth.sizes['lon']} but the forecast is "
            f"{fcst.sizes['lat']}x{fcst.sizes['lon']}. These are subtracted directly and "
            f"never regridded; rebuild the truth cache with --stage refs --force.")

    vlo, vhi = event_t2m_scale(chain.event)
    olo, ohi = event_t2m_range(chain.event)
    div = _diverged_at(chain, model)
    n = len(cols)

    # -- the caption, wrapped BEFORE the canvas is sized ---------------------- #
    lo_lon, hi_lon, lo_lat, hi_lat = C.CONUS_EXTENT
    aspect = (hi_lat - lo_lat) / (hi_lon - lo_lon)
    body = (f"Absolute rows share one scale per event ([{vlo:.0f}, {vhi:.0f}] K, ERA5's "
            f"own 1st-99th percentile over every cached time); the difference row is fixed "
            f"at +-{DIFF_MAX_K:g} K for every figure in the sweep, so panels are comparable "
            f"across models and leads. A forecast panel that leaves the full range ERA5 "
            f"itself covers for this event ([{olo:.0f}, {ohi:.0f}] K) prints its own "
            f"min/max in red; every difference panel prints its cos(lat)-weighted RMSE "
            f"and bias.")
    if dropped:
        body += (f" {n} of {n + dropped} available 3-day checkpoints shown, evenly spaced, "
                 f"first and last kept.")
    if div is not None:
        body += (f" RED columns are at or past {div:g} d, where the bounds check says this "
                 f"chain diverged.")
    body += (" Past ~8.5-12 d the difference row measures chaos, not error: it is here to "
             "separate a decorrelated but physical field from a blow-up, not to score "
             "skill.")

    # -- geometry ------------------------------------------------------------ #
    PANEL_W = 2.0                                   # inches per lead column
    panel_h = PANEL_W * aspect
    cbar_gutter, left_pad, bot_pad = 1.60, 0.62, 0.30
    grid_w, grid_h = n * PANEL_W, 3 * panel_h
    wrap = max(90, int(11.5 * grid_w))              # chars that fit the map block
    caption = textwrap.fill(body, wrap)
    head_lines = 3 + caption.count("\n") + 1
    head_h = 0.20 * head_lines + 0.30               # column titles live inside the grid
    fig_w = grid_w + left_pad + cbar_gutter
    fig_h = grid_h + head_h + bot_pad
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(3, n, left=left_pad / fig_w,
                          right=(left_pad + grid_w) / fig_w,
                          bottom=bot_pad / fig_h, top=1.0 - head_h / fig_h,
                          wspace=0.03, hspace=0.05)

    mabs = mdiff = None
    stats = []
    for j, t in enumerate(cols):
        f = fcst.sel(time=t)
        o = truth.sel(time=t)
        d = f - o
        lead = (pd.Timestamp(t) - chain.init).total_seconds() / 86400.0
        past = div is not None and lead >= div - 1e-9
        stats.append((lead, _wrmse(d), float(AI.area_mean(d))))

        for row, (da, cmap, lo, hi) in enumerate((
                (f, CMAP_T2M, vlo, vhi),
                (o, CMAP_T2M, vlo, vhi),
                (d, CMAP_DIFF, -DIFF_MAX_K, DIFF_MAX_K))):
            ax = fig.add_subplot(gs[row, j], projection=ccrs.PlateCarree())
            ax.set_extent(C.CONUS_EXTENT, ccrs.PlateCarree())
            m = _draw(ax, da, cmap, lo, hi, ccrs)
            if row < 2:
                mabs = m
            else:
                mdiff = m
            if row == 0:
                ax.set_title(f"{lead:g} d   {pd.Timestamp(t):%b %d}", fontsize=9,
                             color=("#d62728" if past else "black"), pad=3)
            if j == 0:
                ax.text(-0.035, 0.5, ("forecast", "ERA5", "forecast - ERA5")[row],
                        transform=ax.transAxes, rotation=90, va="center", ha="right",
                        fontsize=9.5)
            if row == 2:
                _corner(ax, f"rmse {stats[-1][1]:.1f}   bias {stats[-1][2]:+.1f}")
            # Off-scale is never silent - but only where it MEANS something, i.e. against
            # the full range ERA5 covers, not against the clipped color scale. See
            # event_t2m_range.
            if row == 0:
                v = np.asarray(da.values, dtype="float64")
                v = v[np.isfinite(v)]
                if v.size and (v.min() < olo or v.max() > ohi):
                    _corner(ax, f"{v.min():.0f} .. {v.max():.0f} K", "#d62728")
            if past:
                ax.patch.set_edgecolor("#d62728")
                for sp in ax.spines.values():
                    sp.set(edgecolor="#d62728", linewidth=2.0, zorder=20)

    # Colorbars in the reserved right gutter, aligned to the rows they describe.
    x = (left_pad + grid_w + 0.16) / fig_w
    w = 0.16 / fig_w
    top = 1.0 - head_h / fig_h
    fig.colorbar(mabs, cax=fig.add_axes(
        [x, top - 2 * panel_h / fig_h, w, 2 * panel_h / fig_h * 0.92])
        ).set_label("2 m temperature (K)", fontsize=9)
    fig.colorbar(mdiff, cax=fig.add_axes(
        [x, bot_pad / fig_h, w, panel_h / fig_h * 0.92])
        ).set_label("forecast - ERA5 (K)", fontsize=9)

    worst = max(stats, key=lambda s: s[1])
    head = (f"{chain.event}   -   {MODEL_LABEL[model]}   -   week-{chain.weeks} init "
            f"({chain.horizon_days} d lead)")
    sub = (f"init {chain.init:%Y-%m-%d}  ->  peak {chain.peak:%Y-%m-%d}      "
           f"CONUS RMSE {stats[0][1]:.1f} K at {stats[0][0]:g} d  ->  "
           f"{stats[-1][1]:.1f} K at {stats[-1][0]:g} d "
           f"(worst {worst[1]:.1f} K at {worst[0]:g} d)")
    fig.text(0.5, 1 - 0.16 / fig_h, head, ha="center", va="top", fontsize=13)
    fig.text(0.5, 1 - 0.44 / fig_h, sub, ha="center", va="top", fontsize=10)
    fig.text(0.5, 1 - 0.74 / fig_h, caption, ha="center", va="top", fontsize=7.8,
             color="0.25", linespacing=1.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[maps] {chain.label} {model}: wrote {out} "
          f"({out.stat().st_size / 1e3:.0f} kB, {n} lead(s), "
          f"RMSE {stats[0][1]:.1f} -> {stats[-1][1]:.1f} K)")
    return out


def stage_maps(chs: list[Chain], *, models=None, ncols: int = DEFAULT_NCOLS,
               force: bool = False) -> int:
    models = tuple(models) if models else SC.MODELS
    made, skipped, not_walked = 0, 0, 0
    for ch in chs:
        for model in models:
            # The walker at an FCN3-extension lead has one 3-day segment on disk, which
            # would draw a one-column "filmstrip" whose only content is that GenCast is
            # skillful at day 3 - already the physics gate's job, and read next to the
            # other figures it implies a walker trajectory that was never rolled.
            if model == "gencast" and not ch.walks_to_peak:
                print(f"[maps] {ch.label} gencast: FCN3-extension lead, the walker rolled "
                      f"{ch.walk_days} d only - no trajectory to draw")
                not_walked += 1
                continue
            try:
                p = draw_chain_maps(ch, model, ncols=ncols, force=force)
            except SystemExit:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[maps] FAILED {ch.label} {model}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                return 1
            made += 1 if p else 0
            skipped += 0 if p else 1
    print(f"[maps] {made} figure(s) under {SC.FIGS / 'maps'}"
          + (f"; {skipped} chain/model pair(s) had no data on disk" if skipped else "")
          + (f"; {not_walked} walker figure(s) not drawn (FCN3-extension leads)"
             if not_walked else ""))
    return 0
