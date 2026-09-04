#!/usr/bin/env python
"""What the atmosphere actually did: the ERA5 running index, 6-hourly, over the whole lead.

The trajectory figure draws 64 walker curves against a single red dashed line labelled
"ERA5 observed ``A_L``". That line is a **scalar** - it is a horizontal rule, not a
trajectory. It comes from ``runs/observations/<event>_verif_t2m_anom.nc``, which
``gencast_s2s.data.true_verif_anom`` built by reading the 25 six-hourly ERA5 frames of the
verification week and then averaging the time axis away. The 21 days the walkers spend
getting there have no observed counterpart anywhere in this repo.

That is the gap this module closes. It rebuilds the SAME observable - ``aires.aindex``'s
cos(lat)-weighted box mean of the T2m anomaly - from ERA5 at every 6 h frame from the
AI+RES init to the event peak, and caches it as a time series:

    runs/aires/<event>/era5_series_6h.nc     85 frames, lead 0.00 .. 21.00 d, ~30 kB

so ``aires/aplots.py`` can draw the truth as a curve the walkers can be compared against
frame by frame, rather than as a finish-line height.

Why the walkers' own reduction, not a new one
---------------------------------------------
The series is produced by wrapping the raw ERA5 cube as a Dataset with GenCast's variable
name (``2m_temperature``) and handing it to ``aires.aindex.index_series`` - the identical
call ``aires/cfs.py`` makes for the CFSv2 baseline and ``aires/aplots.py`` makes for a
walker's ``diag.nc``. So the climatology (``runs/models/clim_1990_2019_t2m_conus.nc``,
sampled at each frame's own hour and day-of-year), the event box and the cos(lat) weights
are not re-implemented here and cannot drift from what the walkers are scored with. The
only thing this module owns is the *fetch*.

The climatology grid is checked, not assumed. A 2-degree copy of that file with the same
name has already cost this project one silently-wrong figure (see CLAUDE.md, "The
climatology trap"), so ``_check_clim_grid`` asserts ``hour=4, dayofyear=366, lat=105,
lon=237`` before any anomaly is formed.

Two stores, one cut date
------------------------
ERA5 is read from WeatherBench-2 for windows that end at or before
``gencast_s2s.data.WB2_END`` (2023-01-10 18Z) and from ARCO-ERA5 for the rest - the same
split every other reader in this repo uses, decided per event by ``data.past_wb2`` on the
whole window so a straddling event falls entirely to ARCO rather than being stitched from
two stores.

Neither is read through ``xr.open_zarr``. The WB2 surface store has 93,544 timesteps and
66 arrays; ``chunks={"time": 1}`` builds a dask graph big enough to reach 4 GB of RSS on
this 15 GB login node before the open returns (measured 2026-08-27, ``astab/refs.py``),
and ``xr.open_zarr`` on the ARCO root is the documented ~120 GB OOM. Both paths therefore
read raw zarr, one frame at a time. The WB2 half literally reuses
``astab.refs._wb2_frame``: it is imported lazily inside the reader rather than at module
scope, because ``astab`` is built ON TOP of ``aires`` everywhere else in the repo and a
top-level import here would invert that. The consequence is stated plainly: with ``astab``
removed, this module still imports and ``load()`` still works, and only ``build()`` on a
pre-2023 event fails.

The 1964 epoch guard
--------------------
``runs/xres/quarantine_1964_epoch_bug/`` is what an ARCO read with the wrong epoch
produces: plausible fields from 59 years before the event, with no error anywhere.
``data.arco_read`` returns a bare numpy array and carries no timestamp back, so this
module checks the store instead - the ``time`` array's ``units`` attribute must say
``hours since 1900-01-01``, and the value stored at the index being read must decode back
to the frame that was asked for. Both are hard failures. A frame that is inside the array
but not yet filled (ERA5 lags real time by months) comes back all-NaN and raises
``MissingTruth``, which the CLI reports as a skipped event rather than papering over: an
interpolated truth line would be a fabrication.

The self-check
--------------
A 7-day mean of the RAW (unsmoothed) series over ``[peak - 6 d, peak]`` inclusive - 25 of
the 85 frames - is by construction the scalar ``A_L`` the figure already draws. Every
build recomputes it, compares it against ``observed`` in the run's own ``compare.json``,
stores it in the cache as ``a_l``, and prints the difference. Measured over all nine
events on 2026-09-04 the largest disagreement is 1.6e-6 K; anything above ``CHECK_TOL``
means the two are not the same quantity and the curve must not be drawn next to the rule.

Smoothing
---------
The stored ``box``/``conus`` series are 24 h centred running means from
``aires.cfs.daily_mean`` - the trapezoid ``(0.5, 1, 1, 1, 0.5)/4`` at this 6 h step, an
exact null at the diurnal period, with the endpoint padding that keeps the last sample
(which sits ON the peak) both trend-exact and diurnal-free. NOT ``aires.aplots.daily``:
its ``(1, 2, 1)/4`` is the same rule at a 12 h step and spans only 12 h here, leaving most
of a several-kelvin diurnal swing on a curve drawn beside smooth 12-hourly walkers. The
unsmoothed series is stored alongside it, and the self-check uses the unsmoothed one.

    PYTHONPATH=. python -m aires.aera5 --event PNW_HeatDome_2021
    PYTHONPATH=. python -m aires.aera5 --all              # every AI+RES event
    PYTHONPATH=. python -m aires.aera5 --all --force      # ignore the caches
    PYTHONPATH=. python -m aires.aera5                    # offline: re-check the caches
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
from aires.cfs import daily_mean
from fcn3 import fevents as F
from gencast_s2s import config as C

# The series' own resolution. 6 h is ERA5's native step in both stores and twice the
# walkers' 12 h step, so the truth curve is never the coarser of the two on the figure.
STEP_H = 6
CACHE_NAME = "era5_series_6h.nc"
METRIC = "t2m_anom"

# The climatology grid this module refuses to run without. See the module docstring.
CLIM_SIZES = dict(hour=4, dayofyear=366, lat=105, lon=237)
# The CONUS crop every cube in the AI+RES stack lives on (gencast_s2s.config.LAT/LON).
CONUS_SHAPE = (105, 237)

# How far the rebuilt A_L may sit from the scalar the figure already draws. Both are the
# same 25-frame mean of the same ERA5 field through the same reduction, so the only thing
# between them is float32 accumulation order: measured over all nine events, |diff| <=
# 1.6e-6 K. The tolerance is four orders of magnitude looser than that on purpose - it is
# a tripwire for a window or store mistake, not a numerical budget.
CHECK_TOL = 0.05


class MissingTruth(RuntimeError):
    """ERA5 does not carry a frame this window needs (not published yet, or not stored)."""


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #
def cache_path(event: str) -> Path:
    """``runs/aires/<event>/era5_series_6h.nc`` - beside the run, not inside a tag.

    The truth is a property of the EVENT and its lead; every tag of an event
    (``pilot``, ``persist``, ...) shares one file, exactly as ``aconfig.cfs_dir`` does
    for the CFSv2 baseline.
    """
    return A.event_dir(event) / CACHE_NAME


def window(event: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    """``(init, peak, times)`` - the lead window, at ``STEP_H``, endpoints included.

    ``init`` is ``fcn3.fevents.Event.init``, which is what ``aires/cfs.py`` measures lead
    from and therefore what the trajectory figure's x axis is anchored to. The horizon is
    checked against ``aconfig.RES_HORIZON_DAYS`` rather than assumed: ``Event.init``
    resolves ``FCN3_WEEKS`` at IMPORT, so a stray environment variable would silently move
    lead 0 away from the walkers' own init and put the truth curve on a different axis.
    """
    ev = F.event(event)
    init, peak = pd.Timestamp(ev.init), pd.Timestamp(ev.peak)
    horizon = (peak - init) / pd.Timedelta(days=1)
    if abs(horizon - float(A.RES_HORIZON_DAYS)) > 1e-9:
        raise SystemExit(
            f"[aera5] {event}: the registry's init {init.date()} is {horizon:g} d before "
            f"the peak, but the AI+RES horizon is {A.RES_HORIZON_DAYS:g} d. The truth "
            f"curve and the walkers would be drawn on different axes. Check FCN3_WEEKS "
            f"(={F.WEEKS}) and AIRES_HORIZON_DAYS.")
    return init, peak, pd.date_range(init, peak, freq=f"{STEP_H}h")


def lead_days(event: str, times=None) -> np.ndarray:
    """Days from the init, the trajectory figure's x axis. ``0.0 .. horizon``, step 0.25."""
    init, _peak, t = window(event)
    t = pd.DatetimeIndex(times) if times is not None else t
    return np.asarray((t - init) / pd.Timedelta(days=1), dtype=float)


def verif_mask(event: str, times=None) -> np.ndarray:
    """The 25 frames of ``[peak - 6 d, peak]``, inclusive at BOTH ends.

    The same window ``aires.aindex.check_window`` demands of a forecast cube for an
    instantaneous metric, and the same one ``gencast_s2s.data.true_verif_anom`` averaged
    to produce the scalar this series is checked against. Off by one frame at either end
    and the check below compares two different 7-day means.
    """
    _init, peak, t = window(event)
    t = pd.DatetimeIndex(times) if times is not None else t
    m = np.asarray((t >= peak - pd.Timedelta(days=6)) & (t <= peak))
    want = int(round(6 * 24 / STEP_H)) + 1
    if int(m.sum()) != want:
        raise SystemExit(f"[aera5] {event}: the verification window holds {int(m.sum())} "
                         f"frames at a {STEP_H} h step, expected {want}")
    return m


# --------------------------------------------------------------------------- #
# The fetch. One frame at a time, raw zarr, never xr.open_zarr - see the docstring.
# --------------------------------------------------------------------------- #
def _check_clim_grid() -> None:
    if not C.CLIM_FILE.exists():
        raise SystemExit(f"[aera5] no climatology at {C.CLIM_FILE}; the anomaly cannot "
                         f"be formed. Run the xres/gencast prep stage first.")
    with xr.open_dataset(C.CLIM_FILE) as ds:
        got = {k: int(ds.sizes[k]) for k in CLIM_SIZES if k in ds.sizes}
    if got != CLIM_SIZES:
        raise SystemExit(
            f"[aera5] {C.CLIM_FILE} has {got}, expected {CLIM_SIZES}. A 2-degree copy of "
            f"this file with the same name has produced silently wrong anomalies before "
            f"(CLAUDE.md, 'The climatology trap'). Verify the grid, not the filename.")


def _wb2_conus_frames(times: pd.DatetimeIndex) -> xr.DataArray:
    """WeatherBench-2 ERA5 T2m, CONUS crop, one raw-zarr frame per time.

    ``astab.refs._wb2_frame`` is imported here rather than at module scope: ``astab``
    imports ``aires`` everywhere else in the repo, and this is the one call that goes the
    other way. Keeping it inside the function means the layering cost is paid only by a
    pre-2023 ``build()``, never by an import or by ``load()``.
    """
    from astab.refs import _wb2_frame

    frames = []
    for i, t in enumerate(times, 1):
        try:
            f = _wb2_frame("2m_temperature", t)
        except KeyError as e:
            raise MissingTruth(f"{t} is not in the WeatherBench-2 surface store") from e
        frames.append(f.sel(lat=C.LAT, lon=C.LON).astype("float32"))
        if i % 20 == 0 or i == times.size:
            print(f"[aera5]   {i}/{times.size} {t}", flush=True)
    return xr.concat(frames, dim=pd.Index(times, name="time"))


def _check_arco_epoch(root, t) -> None:
    """Refuse an ARCO read whose index does not decode back to the time asked for.

    ``data.arco_read`` hands back a bare numpy array: nothing downstream can tell a frame
    from 2023 apart from the same frame's slot read with a 1959 epoch, which is exactly
    what ``runs/xres/quarantine_1964_epoch_bug/`` holds. So the store is asked directly -
    its declared units, and the value it actually stores at the index being read.
    """
    from gencast_s2s import data as D

    ta = root["time"]
    units = str(ta.attrs.get("units", ""))
    if not units.startswith("hours since 1900-01-01"):
        raise SystemExit(f"[aera5] the ARCO time axis says {units!r}; data.ARCO_EPOCH is "
                         f"{D.ARCO_EPOCH.date()}. Do not read this store until they agree "
                         f"(see runs/xres/quarantine_1964_epoch_bug/).")
    ti = D._arco_tindex(t)
    if ti < 0 or ti >= int(ta.shape[0]):
        raise MissingTruth(f"{t} is outside the ARCO time axis ({ta.shape[0]} steps)")
    got = D.ARCO_EPOCH + pd.Timedelta(hours=int(np.asarray(ta[ti])))
    if got != pd.Timestamp(t):
        raise SystemExit(f"[aera5] ARCO index {ti} decodes to {got}, not {t}. The epoch "
                         f"is wrong; every frame would be off by the same offset.")


def _arco_conus_frames(times: pd.DatetimeIndex) -> xr.DataArray:
    """ARCO-ERA5 T2m, CONUS crop, one raw-zarr frame per time; epoch-checked and
    NaN-checked. ``arco_read(conus=True)`` already returns the 105 x 237 window with
    latitude ascending - the same crop ``data.true_verif_anom`` takes on this path."""
    from gencast_s2s import data as D

    root = D._arco_group()
    stop = str(root.attrs.get("valid_time_stop_era5t") or
               root.attrs.get("valid_time_stop") or "")
    lat, lon = D._arco_lat_lon_coords(conus=True, stride=1)
    frames = []
    for i, t in enumerate(times, 1):
        _check_arco_epoch(root, t)
        data = D.arco_read("2m_temperature", t, conus=True, stride=1)
        if not np.isfinite(data).all():
            raise MissingTruth(
                f"{t} is an unfilled slot in ARCO-ERA5 (store says it holds data through "
                f"{stop or 'an unstated date'}); the observed index does not exist yet")
        frames.append(xr.DataArray(np.asarray(data, dtype="float32"), dims=("lat", "lon"),
                                   coords={"lat": lat, "lon": lon}))
        if i % 20 == 0 or i == times.size:
            print(f"[aera5]   {i}/{times.size} {t}", flush=True)
    return xr.concat(frames, dim=pd.Index(times, name="time"))


def fetch(times: pd.DatetimeIndex) -> tuple[xr.DataArray, str]:
    """``(T2m[time, lat, lon] on the CONUS model grid, source label)``.

    The store is chosen by ``data.past_wb2`` on the WHOLE window, so a window that
    straddles ``WB2_END`` is read entirely from ARCO rather than stitched.
    """
    from gencast_s2s import data as D

    if D.past_wb2(times):
        da, src = _arco_conus_frames(times), "ARCO-ERA5"
    else:
        da, src = _wb2_conus_frames(times), "WeatherBench-2 ERA5"
    if (da.sizes["lat"], da.sizes["lon"]) != CONUS_SHAPE:
        raise SystemExit(f"[aera5] {src} returned lat={da.sizes['lat']} "
                         f"lon={da.sizes['lon']}, expected {CONUS_SHAPE}")
    if float(da["lat"].values[0]) > float(da["lat"].values[-1]):
        raise SystemExit("[aera5] latitude descends; aindex.crop needs it ascending")
    return da, src


# --------------------------------------------------------------------------- #
# The scalar this series has to reproduce
# --------------------------------------------------------------------------- #
def run_tags(event: str) -> list[str]:
    """Reduced AI+RES tags for this event, the default tag first."""
    tags = sorted(p.parent.name for p in
                  (A.event_dir(event) / "res").glob("*/compare.json"))
    return ([A.RES_TAG] if A.RES_TAG in tags else []) + [t for t in tags if t != A.RES_TAG]


def stored_observed(event: str, tag: str | None = None) -> tuple[float | None, str | None]:
    """``(observed A_L, tag)`` from the run's own ``compare.json``, via
    ``awalkers.run_record`` - the number the trajectory figure draws as its red rule.

    ``None`` when the event has no reduced run: the truth series is perfectly buildable
    before there is anything to check it against, exactly as the CFS cube is.
    """
    from aires.awalkers import run_record

    for t in ([tag] if tag else run_tags(event)):
        try:
            r = run_record(event, t)
        except FileNotFoundError:
            continue
        if r.get("observed") is not None:
            return float(r["observed"]), t
    return None, None


# --------------------------------------------------------------------------- #
# Build / load
# --------------------------------------------------------------------------- #
def build(event: str, *, force: bool = False) -> Path:
    """Fetch, reduce, smooth, self-check and cache. Login node; needs internet."""
    ev = F.event(event)
    if ev.metric != METRIC:
        raise SystemExit(f"[aera5] {event} is scored on {ev.metric!r}; this module builds "
                         f"the {METRIC!r} index only (the wind and precip metrics have no "
                         f"climatology in this repo to form a running anomaly against)")
    init, peak, times = window(event)
    out = cache_path(event)
    if out.exists() and not force:
        with xr.open_dataset(out) as ds:
            have = pd.DatetimeIndex(ds["time"].values)
        if have.equals(times):
            print(f"[aera5] {event}: cached ({have.size} frames) {out}")
            return out
        print(f"[aera5] {event}: cache spans {have.size} frames, want {times.size}; "
              f"rebuilding")

    _check_clim_grid()
    box = AI.box_for(event)
    print(f"[aera5] {event}: {times.size} frames, {init.date()} .. {peak.date()} "
          f"(lead 0 .. {(peak - init) / pd.Timedelta(days=1):g} d), box {box.name}",
          flush=True)

    t2m, source = fetch(times)
    cube = xr.Dataset({"2m_temperature": t2m})
    idx = AI.index_series(cube, event, METRIC)          # the walkers' own reduction
    raw = {k: np.asarray(v.values, dtype=float) for k, v in idx.items()}
    for k, v in raw.items():
        if not np.isfinite(v).all():
            raise SystemExit(f"[aera5] {event}: the {k} series has non-finite values; the "
                             f"ERA5 grid and the climatology grid did not align")

    a_l, stored, tag = check_a_l(event, raw["box"], times)
    ds = cache_dataset(event, raw, source=source, a_l=a_l, stored=stored, tag=tag)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".nc.tmp.{os.getpid()}")
    ds.to_netcdf(tmp)
    os.replace(tmp, out)
    print(f"[aera5] wrote {out} ({out.stat().st_size / 1e3:.1f} kB)")
    return out


def smooth(lead, y) -> np.ndarray:
    """The 24 h centred running mean every curve on the trajectory figure carries.

    ``aires.cfs.daily_mean``, NOT ``aires.aplots.daily``. They are the same rule at two
    steps: ``daily``'s ``(1, 2, 1)/4`` is an exact null at the diurnal period for a
    12-HOURLY walker series, and at this module's 6 h step it spans 12 h and leaves most
    of a several-kelvin diurnal swing behind. Routed through one named function so a
    future edit cannot quietly give the truth curve a different filter from the walkers'.
    """
    return daily_mean(np.asarray(lead, dtype=float), np.asarray(y, dtype=float))[1]


def check_a_l(event: str, raw_box, times=None) -> tuple[float, float | None, str | None]:
    """``(A_L from the raw series, the stored scalar, its tag)``; raises if they disagree.

    The 7-day mean of the unsmoothed box series over ``[peak - 6 d, peak]`` IS the number
    ``gencast_s2s.data.true_verif_anom`` produced and the trajectory figure already draws
    as a horizontal rule. If this series is the same observable, the two are the same
    arithmetic on the same 25 ERA5 frames and differ only by float32 accumulation order.
    Anything larger means the curve and the rule are not the same quantity, and a figure
    that draws both would be making a claim neither supports.
    """
    _init, _peak, t = window(event)
    t = pd.DatetimeIndex(times) if times is not None else t
    m = verif_mask(event, t)
    a_l = float(np.asarray(raw_box, dtype=float)[m].mean())
    stored, tag = stored_observed(event)
    diff = float("nan") if stored is None else a_l - stored
    print(f"[aera5] {event}: A_L computed {a_l:+.4f} K   stored "
          f"{'--' if stored is None else f'{stored:+.4f}'} K"
          f"{'' if stored is None else f' ({tag})   diff {diff:+.2e} K'}")
    if stored is not None and abs(diff) > CHECK_TOL:
        raise SystemExit(
            f"[aera5] {event}: the rebuilt 7-day mean is {diff:+.4f} K from the scalar the "
            f"figure already draws (tolerance {CHECK_TOL} K). These are supposed to be the "
            f"same 25-frame mean of the same field; do not cache a curve that disagrees "
            f"with its own endpoint. Check the window edges and the frame count "
            f"({int(m.sum())} used).")
    return a_l, stored, tag


def cache_dataset(event: str, raw: dict, *, source: str, a_l: float,
                  stored: float | None, tag: str | None) -> xr.Dataset:
    """The cache file's schema: raw and smoothed series for both boxes, on ``time`` with
    ``lead`` alongside it, plus everything ``load`` reports as attributes."""
    init, peak, times = window(event)
    lead = lead_days(event, times)
    box = AI.box_for(event)
    sm = {k: smooth(lead, v) for k, v in raw.items()}
    ds = xr.Dataset(
        {"box_raw": ("time", np.asarray(raw["box"], dtype=float)),
         "box": ("time", sm["box"]),
         "conus_raw": ("time", np.asarray(raw["conus"], dtype=float)),
         "conus": ("time", sm["conus"])},
        coords={"time": times, "lead": ("time", lead)})
    for v in ds.data_vars:
        ds[v].attrs.update(units="K", long_name=f"ERA5 {METRIC} area mean, {v}")
    # ``units="days since ..."`` would be CF time units and xarray would try to DECODE
    # this axis into datetimes on the way back in, which fails on read. It is an elapsed
    # time, not a date, so the reference belongs in the long name.
    ds["lead"].attrs.update(units="days", long_name="lead time from the AI+RES init")
    ds.attrs.update(
        event=event, metric=METRIC, source=source, step_h=STEP_H,
        init=str(init), peak=str(peak),
        horizon_days=float((peak - init) / pd.Timedelta(days=1)),
        box_name=box.name, box_label=box.label,
        box_lat=list(box.lat), box_lon=list(box.lon),
        a_l=float(a_l), a_l_stored=float("nan") if stored is None else float(stored),
        a_l_diff=float("nan") if stored is None else float(a_l - stored),
        a_l_tag=tag or "", n_verif_frames=int(verif_mask(event, times).sum()),
        smoothing=("24 h centred running mean, aires.cfs.daily_mean; 'box'/'conus' are "
                   "smoothed, '*_raw' are not"),
        note=("the observed running index over the AI+RES lead window, reduced by "
              "aires.aindex.index_series - the same code path that scores a walker"))
    return ds


def load(event: str) -> dict | None:
    """The cached series, or ``None`` if it has not been built.

    ``None`` on purpose rather than an exception: ``aires/aplots.py`` reads this and never
    builds it, so an event whose truth has not been fetched (or does not exist yet in
    ERA5) drops the black line and still produces a figure.

    ``box`` is the SMOOTHED series and ``raw`` the same series unsmoothed, both in K, on
    the ``lead`` axis in days from the AI+RES init. The box's NAME is ``box_name``.
    """
    p = cache_path(event)
    if not p.exists():
        return None
    with xr.open_dataset(p) as ds:
        a = ds.attrs
        return dict(
            lead=np.asarray(ds["lead"].values, dtype=float),
            box=np.asarray(ds["box"].values, dtype=float),
            raw=np.asarray(ds["box_raw"].values, dtype=float),
            conus=np.asarray(ds["conus"].values, dtype=float),
            conus_raw=np.asarray(ds["conus_raw"].values, dtype=float),
            times=np.asarray(ds["time"].values, dtype="datetime64[ns]"),
            event=str(a.get("event", event)),
            box_name=str(a.get("box_name", "")),
            box_label=str(a.get("box_label", "")),
            metric=str(a.get("metric", METRIC)),
            source=str(a.get("source", "")),
            init=str(a.get("init", "")), peak=str(a.get("peak", "")),
            horizon_days=float(a.get("horizon_days", float("nan"))),
            a_l=float(a.get("a_l", float("nan"))),
            path=p,
        )


def all_events() -> list[str]:
    """Every AI+RES event the truth curve is wanted for - the CFSv2 baseline's slate.

    Deliberately ``aires.cfs.all_events``: the two curves go on the same figure, so they
    must cover the same events or one of them is silently absent from a panel.
    """
    from aires.cfs import all_events as _slate

    return _slate()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _report_row(event: str) -> str:
    d = load(event)
    if d is None:
        return f"  {event:26s} --  no cache"
    stored, tag = stored_observed(event)
    diff = ("" if stored is None else
            f"  stored {stored:+8.3f} ({tag})  diff {d['a_l'] - stored:+.2e}")
    return (f"  {event:26s} {d['box_name']:>10s}  n={d['lead'].size:3d}  "
            f"lead {d['lead'][0]:.2f}..{d['lead'][-1]:.2f} d  A_L {d['a_l']:+8.3f}{diff}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default=None, help="build one event")
    ap.add_argument("--all", action="store_true", help="build every AI+RES event")
    ap.add_argument("--force", action="store_true", help="ignore existing caches")
    a = ap.parse_args(argv)

    if not a.event and not a.all:
        print("[aera5] cached ERA5 truth series (nothing built; --event or --all to build)")
        for ev in all_events():
            print(_report_row(ev))
        return 0

    events = [a.event] if a.event else all_events()
    built, skipped = [], []
    for ev in events:
        try:
            build(ev, force=a.force)
            built.append(ev)
        except MissingTruth as e:
            print(f"[aera5] {ev}: SKIPPED - {e}")
            skipped.append((ev, str(e)))
    print(f"\n[aera5] {len(built)} built, {len(skipped)} skipped")
    for ev in built:
        print(_report_row(ev))
    for ev, why in skipped:
        print(f"  {ev:26s} SKIPPED  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
