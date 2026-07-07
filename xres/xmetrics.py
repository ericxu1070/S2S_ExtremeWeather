"""Verification-metric registry: how each event family is reduced to a (lat, lon) field.

Three headline metrics, each defined identically for the GenCast forecast cube, the ERA5
observed truth, and (in ``xhrrr``) the HRRR observed truth, over the SAME 7-day window
ending on the event peak:

    t2m_anom    weekly-MEAN 2 m-temperature minus the 1990-2019 climatology   (K)
    u850_speed  weekly-MEAN 850 hPa wind SPEED  sqrt(u^2 + v^2)               (m/s)
    tp_total    weekly-TOTAL precipitation accumulated over the window        (mm)

Two "free" secondaries derived from the same cube (weekly peak rather than mean/total):

    u850_speed_max  weekly-MAX 850 hPa wind speed                            (m/s)
    tp_max12h       weekly-MAX 12 h precip accumulation                      (mm)

Wind SPEED (not the signed u-component) is used so the metric does not self-cancel in a
mean and is invariant to HRRR's grid-relative wind orientation. T2m is an anomaly; wind
and precip are absolute (their magnitude already measures the extreme).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from gencast_s2s import config as C
from gencast_s2s import data as D

DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Metric:
    key: str
    label: str          # axis / colorbar label
    units: str
    field_cmap: str     # colormap for the obs/forecast map panels
    diverging: bool     # symmetric color scale around 0 (T2m anomaly) vs sequential
    extreme_high: int   # +1 if the extreme tail is the HIGH side, -1 if LOW (overridden
                         # per-event for T2m cold cases by xconfig.COLD_EVENTS)


METRICS = {
    "t2m_anom":   Metric("t2m_anom",   "weekly-mean T2m anomaly", "K",
                         "RdBu_r", True,  +1),
    "u850_speed": Metric("u850_speed", "weekly-mean |U850| speed", "m/s",
                         "viridis", False, +1),
    "tp_total":   Metric("tp_total",   "weekly total precip", "mm",
                         "YlGnBu", False, +1),
    # secondaries (weekly peak)
    "u850_speed_max": Metric("u850_speed_max", "weekly-max |U850| speed", "m/s",
                             "viridis", False, +1),
    "tp_max12h":      Metric("tp_max12h",      "weekly-max 12h precip", "mm",
                             "YlGnBu", False, +1),
}

# headline metric -> its "free" secondary (cached alongside; not used in the headline figs)
SECONDARY = {"u850_speed": "u850_speed_max", "tp_total": "tp_max12h"}


def spec(metric: str) -> Metric:
    return METRICS[metric]


# --------------------------------------------------------------------------- #
# Time windows (the verification week = the 7 days ending on the peak).
#   - instantaneous fields (T2m, winds): all native frames in [peak-6d, peak]
#   - accumulated precip: native accumulation frames covering (peak-6d, peak]
# --------------------------------------------------------------------------- #
def inst_window(peak, freq: str) -> pd.DatetimeIndex:
    peak = pd.Timestamp(peak)
    return pd.date_range(peak - 6 * DAY, peak, freq=freq)


def accum_window(peak, freq: str) -> pd.DatetimeIndex:
    """Accumulation frames whose period lies inside (peak-6d, peak] (drops the leading
    frame so each accumulation block is counted once and stays within the window)."""
    peak = pd.Timestamp(peak)
    return pd.date_range(peak - 6 * DAY, peak, freq=freq)[1:]


def fc_times_in_window(cube: xr.Dataset, peak, inclusive_left: bool) -> np.ndarray:
    """The forecast-cube valid timestamps (absolute datetime axis) inside the verif week."""
    peak = pd.Timestamp(peak)
    t = pd.DatetimeIndex(cube["time"].values)
    if not t.is_unique:
        t = t[~t.duplicated(keep="first")]
    lo = (t >= peak - 6 * DAY) if inclusive_left else (t > peak - 6 * DAY)
    return t[lo & (t <= peak)].values


# --------------------------------------------------------------------------- #
# GenCast forecast-cube reductions  -> (member, lat, lon) on the forecast grid.
# The cube is a Dataset of raw model output on an absolute-datetime ``time`` axis,
# already cropped to the CONUS box (see xinference).
# --------------------------------------------------------------------------- #
def _wind_speed_850(cube: xr.Dataset, times) -> xr.DataArray:
    u = _sel_times(cube["u_component_of_wind"].sel(level=850), times)
    v = _sel_times(cube["v_component_of_wind"].sel(level=850), times)
    return np.sqrt(u ** 2 + v ** 2)


def _sel_times(da: xr.DataArray, times) -> xr.DataArray:
    """Select forecast times; use isel when the cube time coord has duplicates."""
    t = pd.DatetimeIndex(da["time"].values)
    want = pd.DatetimeIndex(times)
    if t.is_unique:
        return da.sel(time=want)
    mask = t.isin(want)
    return da.isel(time=mask)


def forecast_field(cube: xr.Dataset, peak, metric: str) -> xr.DataArray:
    m = metric
    if m == "t2m_anom":
        w = fc_times_in_window(cube, peak, inclusive_left=True)
        t2m = _sel_times(cube["2m_temperature"], w).mean("time")
        clim = D.clim_for(pd.DatetimeIndex(w)).mean("time")     # 0.25deg clim grid
        anom = t2m - clim                                       # aligns on shared coords
        return anom
    if m in ("u850_speed", "u850_speed_max"):
        w = fc_times_in_window(cube, peak, inclusive_left=True)
        spd = _wind_speed_850(cube, w)
        return spd.max("time") if m.endswith("_max") else spd.mean("time")
    if m in ("tp_total", "tp_max12h"):
        w = fc_times_in_window(cube, peak, inclusive_left=False)   # accumulation blocks
        tp = _sel_times(cube["total_precipitation_12hr"], w) * 1000.0  # m -> mm (per 12h)
        return tp.max("time") if m == "tp_max12h" else tp.sum("time")
    raise ValueError(f"unknown metric {m!r}")


# --------------------------------------------------------------------------- #
# ERA5 observed-truth reductions -> (lat, lon) on the native 0.25deg CONUS grid.
# --------------------------------------------------------------------------- #
def _era5_u850_speed(peak, use_max: bool) -> xr.DataArray:
    """CONUS 850 hPa wind speed reduced over the verification week (low-RAM path)."""
    import gc

    win = inst_window(peak, "6h")
    use_arco = D.past_wb2(win)
    lat_c, lon_c = (D._arco_lat_lon_coords(conus=True, stride=1) if use_arco
                    else (None, None))
    sum_spd = max_spd = None
    n = 0
    for t in win:
        if use_arco:
            u = D.arco_read("u_component_of_wind", t, levels=[850], conus=True)[0]
            v = D.arco_read("v_component_of_wind", t, levels=[850], conus=True)[0]
        else:
            src = D._surf()
            u = src["u_component_of_wind"].sel(time=t, level=850, lat=C.LAT, lon=C.LON).load().values
            v = src["v_component_of_wind"].sel(time=t, level=850, lat=C.LAT, lon=C.LON).load().values
            if lat_c is None:
                lat_c = src["u_component_of_wind"].sel(time=t, level=850, lat=C.LAT, lon=C.LON).lat.values
                lon_c = src["u_component_of_wind"].sel(time=t, level=850, lat=C.LAT, lon=C.LON).lon.values
        spd = np.sqrt(u ** 2 + v ** 2)
        sum_spd = spd if sum_spd is None else sum_spd + spd
        max_spd = spd if max_spd is None else np.maximum(max_spd, spd)
        n += 1
        del u, v, spd
        D.release_stores()
        gc.collect()
    data = max_spd if use_max else sum_spd / n
    return xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lat_c, "lon": lon_c})


def era5_field(peak, metric: str) -> xr.DataArray:
    m = metric
    if m == "t2m_anom":
        return D.true_verif_anom(peak)                          # reuse existing recipe
    if m in ("u850_speed", "u850_speed_max"):
        return _era5_u850_speed(peak, use_max=m.endswith("_max"))
    if m in ("tp_total", "tp_max12h"):
        return _era5_precip(peak, m)
    raise ValueError(f"unknown metric {m!r}")


def _era5_precip(peak, metric: str) -> xr.DataArray:
    """Weekly total / max-12h ERA5 precip (mm) over the window, robust to the store split.

    WB2 surface store carries ready-made 6h/12h accumulations; ARCO carries hourly
    ``total_precipitation`` which we accumulate ourselves."""
    import gc

    peak = pd.Timestamp(peak)
    win6 = accum_window(peak, "6h")
    if not D.past_wb2(pd.date_range(peak - 6 * DAY, peak, freq="6h")):
        src = D._surf()
        sum_tp = None
        max_tp12 = None
        prev6 = None
        template = None
        for t in win6:
            frame = src["total_precipitation_6hr"].sel(time=t, lat=C.LAT, lon=C.LON).load()
            tp = frame.values * 1000.0
            if template is None:
                template = frame.copy(data=np.zeros_like(frame.values, dtype=np.float64))
            if metric == "tp_total":
                sum_tp = tp if sum_tp is None else sum_tp + tp
            else:
                if prev6 is not None:
                    tp12 = prev6 + tp
                    max_tp12 = tp12 if max_tp12 is None else np.maximum(max_tp12, tp12)
                prev6 = tp
            del frame, tp
            D.release_stores()
            gc.collect()
            src = D._surf()
        data = sum_tp if metric == "tp_total" else max_tp12
        out = template.copy(data=data)
        D.release_stores()
        gc.collect()
        return out
    # ARCO: hourly accumulation in metres — one CONUS hour at a time
    win1 = accum_window(peak, "1h")
    lat_c, lon_c = D._arco_lat_lon_coords(conus=True, stride=1)
    sum_tp = max_tp12 = None
    roll = None
    for t in win1:
        tp = D.arco_read("total_precipitation", t, conus=True) * 1000.0
        if metric == "tp_total":
            sum_tp = tp if sum_tp is None else sum_tp + tp
        else:
            if roll is None:
                roll = np.zeros((12,) + tp.shape, dtype=np.float64)
            roll = np.roll(roll, -1, axis=0)
            roll[-1] = tp
            block = roll.sum(axis=0)
            max_tp12 = block if max_tp12 is None else np.maximum(max_tp12, block)
        del tp
        D.release_stores()
        gc.collect()
    data = sum_tp if metric == "tp_total" else max_tp12
    out = xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lat_c, "lon": lon_c})
    return out


# --------------------------------------------------------------------------- #
# Extreme-member direction (for the "extreme tail" combined PDF). T2m uses the
# event sign (cold events -> coldest member); winds/precip -> the high tail.
# --------------------------------------------------------------------------- #
def extreme_sign(metric: str, event_name: str) -> int:
    if metric == "t2m_anom":
        from . import xconfig as X
        return -1 if event_name in X.COLD_EVENTS else +1
    return +1
