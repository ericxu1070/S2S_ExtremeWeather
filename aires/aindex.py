#!/usr/bin/env python
"""The scalar observable ``A_L`` - the one number AI+RES sorts walkers by.

Everything in this experiment - the score a walker receives, the outcome that score is
supposed to predict, the exceedance curve Phase 4 plots - is a reduction of a forecast
cube to a single number. Doing that reduction in more than one place is how two figures
end up disagreeing, so it happens here and nowhere else.

    A_L  =  cos(lat)-weighted area mean, over an EVENT BOX,
            of the 7-day-mean field ending at the event peak.

The field itself is ``xres.xmetrics.forecast_field`` - the same function that scores the
GenCast cubes and the FCN3 cubes in the head-to-head experiment - so a walker, an FCN3
score forecast and an ERA5 truth field are all reduced by identical code. Only the area
mean is new, and only because it is now taken over a box rather than all of CONUS.

The box
-------
``aires.md`` specifies the observable as a 7-day-mean anomaly over an "event-centered
box", with the CONUS mean kept as a free secondary, and defers the boxes themselves to
Phase 3. They are defined here (``EVENT_BOXES``) because Gate 3 is the first thing that
needs them: a CONUS mean of the 2021 PNW heat dome is mostly grid points that were not
having a heat wave, and a score function that predicts it is not being asked the question
AI+RES actually asks.

Both indices are always computed and stored. The gate verdicts are taken on the box;
Gate 2 predates these boxes and used the CONUS mean, which is why it is still reported.

Note on cropping: the cubes this module sees are already CONUS-cropped by
``xres/xinference.py`` (GenCast) or ``fcn3/run_fcn3.py::_crop_conus`` (FCN3), on the same
0.25 degree grid as the 1990-2019 climatology. A box is a further ``.sel`` inside that
crop, so no box may extend past the CONUS window - ``check_box`` says so out loud rather
than silently returning a clipped mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from gencast_s2s import config as C

# The CONUS verification window every cube in this repo is cropped to.
CONUS_LAT = (float(C.LAT.start), float(C.LAT.stop))     # 24 .. 50 N
CONUS_LON = (float(C.LON.start), float(C.LON.stop))     # 235 .. 294 E


@dataclass(frozen=True)
class Box:
    """A latitude/longitude window in the cube's own coordinates (lon in degrees EAST)."""
    name: str
    lat: tuple[float, float]
    lon: tuple[float, float]
    note: str = ""

    @property
    def label(self) -> str:
        w = lambda x: f"{360 - x:.0f}W" if x > 180 else f"{x:.0f}E"
        return (f"{self.lat[0]:.0f}-{self.lat[1]:.0f}N, "
                f"{w(self.lon[0])}-{w(self.lon[1])}")


CONUS = Box("CONUS", CONUS_LAT, CONUS_LON,
            "the whole verification window; what Gate 2 used")

# --------------------------------------------------------------------------- #
# Per-event boxes.
#
# Each is the region the event actually happened in, clipped to the CONUS crop the cubes
# carry. The clipping is a real limitation where it bites (the PNW dome's core extended
# into British Columbia, north of the 50 N crop) and is recorded in the note rather than
# left for a reader to infer from the numbers.
#
# Selection rule for a heat/cold box: the 6x6 degree window that maximises the ERA5
# week-mean anomaly, SUBJECT TO lying inside the region the event is named for. PNW obeys
# it (+7.72 K registered vs +8.07 K unconstrained CONUS-wide). Where the two halves of the
# rule conflict, the named region wins and the unconstrained maximum is recorded in the
# note - a box in Montana labelled "California Heat Wave" would score a different event
# than the one selected. ``python -m aires.aindex --scan <event>`` reproduces the numbers.
#
# The p90 cases are deliberately CONUS: p90/cases_meta.json defines them as exceedances of
# the "cos(lat)-weighted CONUS-mean daily-mean T2m anomaly", so the CONUS mean IS their
# event-defining index and any smaller box would score a different event than the one
# selected.
# --------------------------------------------------------------------------- #
EVENT_BOXES: dict[str, Box] = {
    "PNW_HeatDome_2021": Box(
        "PNW", (44.0, 50.0), (236.0, 242.0),
        "Oregon + Washington, the dome core. The observed maximum reached ~52 N into "
        "British Columbia, which is outside the 50 N CONUS crop, so this box sees the "
        "southern two thirds of the event."),
    "WinterStorm_Uri_2021": Box(
        "TX/OK", (26.0, 37.0), (254.0, 266.0),
        "Texas and Oklahoma - the freeze's footprint and the grid failure's. Cold event: "
        "the extreme tail is the NEGATIVE side of A_L."),
    "HurricaneIan_2022": Box(
        "SW Florida", (24.0, 30.0), (276.0, 284.0),
        "Cayo Costa landfall (26.7 N, 82.3 W) plus the approach and the Atlantic exit; "
        "scored on 850 hPa wind speed, not T2m."),
    # --- AI+RES extension events (fcn3.fevents.RES_EVENTS) --------------------- #
    "SCentral_HeatDome_2023": Box(
        "S Texas", (26.0, 32.0), (254.0, 260.0),
        "South/central Texas and the Rio Grande - the measured week-mean maximum "
        "(+5.73 K over this box vs a +0.52 K CONUS mean), inside the named region, so "
        "both halves of the selection rule agree. The dome's core extended well into "
        "Mexico, south of the 24 N crop, so like PNW this box sees the northern part "
        "of the event. Finalised from the ERA5 truth on 2026-08-24, before any Gate 3 "
        "reduce or production prep froze it."),
    "Southwest_HeatWave_2020": Box(
        "SW/S Plains", (29.0, 35.0), (251.0, 257.0),
        "Southern New Mexico and west Texas - where the week-mean ERA5 anomaly for the "
        "frozen 2020-08-16 peak actually maximises (+5.14 K over this box, vs a +1.54 K "
        "CONUS mean). A Desert-Southwest box over CA/AZ/NV (32-38 N, 242-248 E) sees "
        "only +2.76 K: the mid-August 2020 ridge was centred east of the name."),
    "California_HeatWave_2022": Box(
        "California", (35.0, 41.0), (236.0, 242.0),
        "Central/Northern California - the named event and the societal record "
        "(Sacramento's all-time 116 F on 2022-09-06): +4.71 K week-mean vs a +1.60 K "
        "CONUS mean. The UNCONSTRAINED week-mean maximum is elsewhere - +7.32 K over "
        "Idaho/Montana (44-50 N, 248-254 E) - and is deliberately not used: it would "
        "score a different event than the one this box names."),
    # p90_* fall through to CONUS by design - see the comment above.
}


def box_for(event: str) -> Box:
    """The event's box, or CONUS when the event's own index IS the CONUS mean."""
    return EVENT_BOXES.get(event, CONUS)


def tail_sign(event: str) -> float:
    """``+1`` if the event's extreme is the HIGH tail of ``A_L``, ``-1`` if the low one.

    Resampling always clones the walkers with the largest score, so the sign has to be
    applied before the score reaches ``aires.dmc`` - not afterwards. Winter Storm Uri is
    a cold event whose extreme is a large NEGATIVE T2m anomaly; scoring it unsigned would
    build an importance-splitting run that spends its whole GPU budget cloning the
    warmest members of a freeze. The heat, hurricane (wind speed) and p90 events are all
    high-tail. This is the only place the convention is stated.
    """
    from fcn3 import fevents as F

    ev = F.KNOWN.get(event)
    return -1.0 if (ev is not None and ev.cold) else +1.0


def boxes_for(event: str) -> dict[str, Box]:
    """Both indices, primary first. ``{'box': ..., 'conus': CONUS}``."""
    return {"box": box_for(event), "conus": CONUS}


# --------------------------------------------------------------------------- #
# Reductions
# --------------------------------------------------------------------------- #
def _squeeze(obj):
    """Drop the size-1 ``batch`` axis a GenCast rollout leaves on a walker diagnostics
    cube. FCN3 cubes do not have one; xres cubes do not either."""
    if "batch" in getattr(obj, "dims", ()):
        obj = obj.isel(batch=0, drop=True)
    return obj


def check_box(da: xr.DataArray, box: Box) -> None:
    """Refuse a box the cube cannot actually cover.

    A ``.sel`` with an out-of-range slice returns the intersection silently, so a box that
    is half outside the CONUS crop yields a mean over the half that is inside, with the
    right shape and no warning. That is exactly the kind of quiet wrongness the
    climatology-grid incident in CLAUDE.md was.
    """
    lat, lon = da["lat"].values, da["lon"].values
    if lat[0] > lat[-1]:
        raise ValueError("cube latitude must ascend before a slice-based crop")
    eps = 1e-6
    if box.lat[0] < lat[0] - eps or box.lat[1] > lat[-1] + eps:
        raise ValueError(f"box {box.name} lat {box.lat} is outside the cube's "
                         f"[{lat[0]}, {lat[-1]}]")
    if box.lon[0] < lon[0] - eps or box.lon[1] > lon[-1] + eps:
        raise ValueError(f"box {box.name} lon {box.lon} is outside the cube's "
                         f"[{lon[0]}, {lon[-1]}]")


def crop(da: xr.DataArray, box: Box) -> xr.DataArray:
    check_box(da, box)
    out = da.sel(lat=slice(*box.lat), lon=slice(*box.lon))
    if out.sizes["lat"] == 0 or out.sizes["lon"] == 0:
        raise ValueError(f"box {box.name} selected an empty region")
    return out


def area_mean(da: xr.DataArray, box: Box | None = None) -> xr.DataArray:
    """cos(lat)-weighted area mean over ``box`` (default: the cube's full extent).

    ``box=None`` reproduces ``aires/gate2.py``'s original CONUS reduction exactly, which
    is what keeps Gate 2's published numbers reproducible after this refactor.
    """
    da = _squeeze(da)
    if box is not None:
        da = crop(da, box)
    return da.weighted(np.cos(np.deg2rad(da["lat"]))).mean(("lat", "lon"))


# --------------------------------------------------------------------------- #
# Window bookkeeping
# --------------------------------------------------------------------------- #
def _inclusive_left(metric: str) -> bool:
    """Accumulated metrics drop the leading frame; instantaneous ones keep it.
    Mirrors ``xres.xmetrics.forecast_field``'s own choice per metric."""
    return metric not in ("tp_total", "tp_max12h")


def window_times(cube: xr.Dataset, peak, metric: str) -> np.ndarray:
    from xres import xmetrics as XM
    return XM.fc_times_in_window(_squeeze(cube), peak, inclusive_left=_inclusive_left(metric))


def check_window(cube: xr.Dataset, peak, metric: str, *, where: str = "cube") -> int:
    """Assert the cube actually spans the whole 7-day verification window.

    ``fc_times_in_window`` returns whatever frames happen to be inside the window, so a
    trajectory that stops early, or a score forecast whose horizon was mis-computed,
    produces a mean over two frames instead of thirteen - a number that looks perfectly
    reasonable and is not A_L. Returns the frame count.
    """
    cube = _squeeze(cube)
    t = pd.DatetimeIndex(cube["time"].values).unique().sort_values()
    if t.size < 2:
        raise ValueError(f"{where}: need >= 2 time frames to infer the step, got {t.size}")
    step_h = float(np.median(np.diff(t.values).astype("timedelta64[h]").astype(float)))
    want = int(round(6 * 24 / step_h)) + (1 if _inclusive_left(metric) else 0)
    got = len(window_times(cube, peak, metric))
    if got != want:
        raise ValueError(
            f"{where}: the {metric} verification window [peak-6d, peak] should hold {want} "
            f"frames at a {step_h:.0f} h step, but the cube supplies {got}. The cube spans "
            f"{t[0]} .. {t[-1]} and the peak is {pd.Timestamp(peak)}.")
    return got


# --------------------------------------------------------------------------- #
# The observable
# --------------------------------------------------------------------------- #
def field(cube: xr.Dataset, peak, metric: str) -> xr.DataArray:
    """The (lat, lon) verification field - identical code path to xres and fcn3."""
    from xres import xmetrics as XM
    return XM.forecast_field(_squeeze(cube), peak, metric)


def a_index(cube: xr.Dataset, peak, metric: str, box: Box | None = None, *,
            check: bool = True, where: str = "cube") -> xr.DataArray:
    """``A_L`` for every member/trajectory in ``cube``."""
    if check:
        check_window(cube, peak, metric, where=where)
    return area_mean(field(cube, peak, metric), box)


def indices(cube: xr.Dataset, event: str, peak, metric: str, *,
            check: bool = True, where: str = "cube") -> dict[str, xr.DataArray]:
    """Both indices at once: ``{'box': A_L over the event box, 'conus': A_L over CONUS}``."""
    if check:
        check_window(cube, peak, metric, where=where)
    f = field(cube, peak, metric)
    return {k: area_mean(f, b) for k, b in boxes_for(event).items()}


# --------------------------------------------------------------------------- #
# The persistence score (the paper's Standard-RES baseline)
# --------------------------------------------------------------------------- #
def instantaneous_field(cube: xr.Dataset, metric: str) -> xr.DataArray:
    """The metric's field at EVERY frame of the cube, un-averaged in time.

    ``forecast_field`` reduces over the 7-day window; this is the same quantity before
    that reduction, which is what a running index and the persistence score need. For
    ``t2m_anom`` it is the anomaly (the climatology is sampled at each frame's own
    hour/dayofyear); the wind and precip metrics are absolute by definition.
    """
    cube = _squeeze(cube)
    t = pd.DatetimeIndex(cube["time"].values)
    if metric == "t2m_anom":
        return cube["2m_temperature"] - clim_on(cube, t)
    if metric in ("u850_speed", "u850_speed_max"):
        u = cube["u_component_of_wind"].sel(level=850)
        v = cube["v_component_of_wind"].sel(level=850)
        return np.sqrt(u ** 2 + v ** 2)
    if metric in ("tp_total", "tp_max12h"):
        return cube["total_precipitation_12hr"] * 1000.0
    raise ValueError(f"unknown metric {metric!r}")


def clim_on(cube: xr.Dataset, times) -> xr.DataArray:
    """1990-2019 T2m climatology sampled at ``times``, on the cube's own time coordinate."""
    from gencast_s2s import data as D

    t = pd.DatetimeIndex(times)
    clim = D.clim_for(t)                       # dims (time, lat, lon), no time COORD
    return clim.assign_coords(time=("time", t.values))


def index_series(cube: xr.Dataset, event: str, metric: str) -> dict[str, xr.DataArray]:
    """The running value of the observable, frame by frame, for both boxes."""
    f = instantaneous_field(cube, metric)
    return {k: area_mean(f, b) for k, b in boxes_for(event).items()}


def instantaneous_index(cube: xr.Dataset, when, event: str, metric: str) -> dict[str, float]:
    """The observable's CURRENT value at time ``when`` - no forecast involved.

    This is the score function Lancelin et al. call Standard-RES: clone the walkers that
    are already extreme. It costs nothing (the walker's own diagnostics cube has the
    frame), and it is the bar an FCN3 score has to clear to be worth its GPU hours - a
    scorer that ranks no better than "who is hottest right now" has not earned the
    handoff. Gate 3 reports both.
    """
    t = pd.Timestamp(when)
    return {k: float(v.sel(time=t)) for k, v in index_series(cube, event, metric).items()}


def describe(event: str) -> str:
    b = box_for(event)
    lines = [f"{event}: primary box {b.name} ({b.label})"]
    if b.note:
        lines.append(f"    {b.note}")
    if b is not CONUS:
        lines.append(f"  secondary: {CONUS.name} ({CONUS.label})")
    return "\n".join(lines)


def scan_boxes(event: str, *, size: float = 6.0, step: float = 1.0, top: int = 8) -> None:
    """Slide a ``size`` x ``size`` degree window over the event's ERA5 truth field and
    print the ``top`` windows by cos(lat)-weighted mean, next to the registered box.

    This is the measurement behind every number in the EVENT_BOXES notes, using the exact
    reduction (``area_mean``) the experiment scores with - so "constrained vs
    unconstrained" gaps are visible, reproducible, and argued about before a box is
    frozen, not after a run.
    """
    from fcn3 import fevents as F

    ev = F.event(event)
    path = F.era5_truth_path(ev)
    if not path.exists():
        raise SystemExit(f"no ERA5 truth at {path}; run xres prep for {event} first")
    with xr.open_dataset(path) as ds:
        da = ds[ev.metric].load()
    lat0, lat1 = float(da.lat[0]), float(da.lat[-1])
    lon0, lon1 = float(da.lon[0]), float(da.lon[-1])
    rows = []
    la = lat0
    while la + size <= lat1 + 1e-6:
        lo = lon0
        while lo + size <= lon1 + 1e-6:
            box = Box("scan", (la, la + size), (lo, lo + size))
            rows.append((float(area_mean(da, box)), box))
            lo += step
        la += step
    rows.sort(key=lambda r: r[0], reverse=tail_sign(event) > 0)
    print(f"{event}: top {top} of {len(rows)} {size:g}x{size:g} deg windows "
          f"({'high' if tail_sign(event) > 0 else 'LOW (cold)'} tail), step {step:g} deg")
    for val, box in rows[:top]:
        print(f"  {val:+7.2f} K   {box.label}")
    reg = box_for(event)
    print(f"  registered: {reg.name} ({reg.label}) = "
          f"{float(area_mean(da, reg)):+.2f} K"
          + ("" if reg is not CONUS else "   [CONUS fallback - no box registered]"))
    print(f"  CONUS mean: {float(area_mean(da, CONUS)):+.2f} K")


if __name__ == "__main__":
    import sys

    from fcn3 import fevents as F
    if "--scan" in sys.argv:
        i = sys.argv.index("--scan")
        args = sys.argv[i + 1:]
        if not args or args[0].startswith("-"):
            raise SystemExit("usage: python -m aires.aindex --scan <event> "
                             "[size [step [top]]]")
        rest = [float(x) for x in args[1:4]]
        scan_boxes(args[0], size=rest[0] if rest else 6.0,
                   step=rest[1] if len(rest) > 1 else 1.0,
                   top=int(rest[2]) if len(rest) > 2 else 8)
    else:
        for n in F.KNOWN_ORDER:
            print(describe(n))
            print()
