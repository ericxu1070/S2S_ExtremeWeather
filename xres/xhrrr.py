"""HRRR observed truth for the cross-resolution metrics, on the ERA5 0.25deg CONUS grid.

Extends ``gencast_s2s.hrrr`` (T2m anomaly) to the two new metrics, sourced from the same
local HRRR v3 shard archive and the verified channel layout of the ``prog_t`` / ``diag_t``
stacks:

    t2m       prog_t[37]                 -> t2m_anom    (reuses gencast_s2s.hrrr)
    u850      prog_t[18]   v850 prog_t[26]  -> u850_speed = mean/max sqrt(u^2+v^2)   (m/s)
    log_tp    diag_t[0] = ln(6h mm + 1e-5) -> tp_total = sum(exp(log_tp)-1e-5)       (mm)

Same 7-day window as ERA5. Wind speed (rotation-invariant) sidesteps HRRR's grid-relative
winds; u850/tp are absolute so they carry no HRRR-vs-ERA5 climatology bias (unlike the
T2m anomaly, which reuses the ERA5 climatology).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from gencast_s2s import config as C
from gencast_s2s import data as D
from gencast_s2s import hrrr as H

from . import xconfig as X
from . import xmetrics as XM
from . import xdata as XD

U850_CH, V850_CH, T2M_CH = 18, 26, 37     # verified prog_t channel indices


def _load_remapped(t: pd.Timestamp, idx, shape, kind):
    """Remap a derived native-HRRR field at time ``t`` onto the ERA5 CONUS grid, or None."""
    f = H._shard_path(t)
    if not f.exists():
        return None
    with np.load(f) as z:
        if kind == "speed850":
            u = z["prog_t"][U850_CH]; v = z["prog_t"][V850_CH]
            flat = np.sqrt(u * u + v * v).ravel()
        elif kind == "precip_mm":
            flat = (np.exp(z["diag_t"][0]) - 1e-5).ravel()             # ln(mm+1e-5) -> mm
        else:
            raise ValueError(kind)
    return flat[idx].reshape(shape)


def _as_da(arr) -> xr.DataArray:
    tlat, tlon = H._target_grid()
    return xr.DataArray(arr, dims=("lat", "lon"),
                        coords={"lat": tlat, "lon": tlon})


def hrrr_field(peak, metric: str):
    """Observed HRRR (lat, lon) field for ``metric`` on the ERA5 CONUS grid, or None if no
    shards cover the window."""
    if metric == "t2m_anom":
        return H.week_mean_t2m_anom(peak)                              # reuse existing recipe
    idx, shape = H._remap_index()
    if metric in ("u850_speed", "u850_speed_max"):
        win = XM.inst_window(peak, "6h")
        fields = [g for g in (_load_remapped(t, idx, shape, "speed850") for t in win)
                  if g is not None]
        if not fields:
            return None
        arr = np.stack(fields, 0)
        red = arr.max(0) if metric.endswith("_max") else arr.mean(0)
        return _as_da(red).rename(metric)
    if metric in ("tp_total", "tp_max12h"):
        win = XM.accum_window(peak, "6h")
        blocks = [g for g in (_load_remapped(t, idx, shape, "precip_mm") for t in win)
                  if g is not None]
        if not blocks:
            return None
        arr = np.stack(blocks, 0)                                      # (n6h, lat, lon) mm
        if metric == "tp_total":
            red = arr.sum(0)
        else:
            red = (arr[:-1] + arr[1:]).max(0) if arr.shape[0] >= 2 else arr.max(0)
        return _as_da(red).rename(metric)
    raise ValueError(f"unknown metric {metric!r}")


def hrrr_truth_path(name: str, metric: str):
    return C.OBS_DIR / f"{name}_hrrr_verif_{metric}.nc"


def build_hrrr_truth(overwrite: bool = False) -> None:
    """Precompute + cache HRRR truth per (event, metric) -> runs/observations/ (offline)."""
    if not C.HRRR_SHARDS.exists():
        print(f"[hrrr-truth] shard archive not found at {C.HRRR_SHARDS}; skipping HRRR")
        return
    C.ensure_dirs()
    for name, (peak, _m) in X.events().items():
        for metric in XD.metrics_for_event(name):
            f = hrrr_truth_path(name, metric)
            if f.exists() and not overwrite:
                print(f"[hrrr-truth] cached: {f.name}")
                continue
            da = hrrr_field(peak, metric)
            if da is None:
                print(f"[hrrr-truth] SKIP {name} [{metric}]: no shards in window")
                continue
            D._atomic_to_netcdf(da.to_dataset(name=metric), f)
            print(f"[hrrr-truth] {name} [{metric}]: mean={float(da.mean()):+.3g} "
                  f"max={float(da.max()):+.3g} -> {f.name}")
