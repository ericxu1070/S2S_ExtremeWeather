"""Data sources: ERA5 stores on GCS, static fields, model inputs, verification truth.

Everything is pulled anonymously from the public WeatherBench-2 / ARCO-ERA5 buckets and
written to local NetCDF under ``runs/`` so the GPU job can read from fast NFS instead of
the network. ``build_raw_inputs`` returns a clean, NetCDF-round-trippable dataset on an
*absolute* time axis; the relative-time / target-template machinery lives in
``inference.build_example_batch``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C

# --------------------------------------------------------------------------- #
# Stores. WeatherBench-2 ERA5 covers 1959-01-01 .. 2023-01-10 18:00 only; events
# whose input frames fall past that use ARCO-ERA5 (the continuously updated
# 0.25deg/37-level superset).
# --------------------------------------------------------------------------- #
SURF_STORE   = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
FULL37_STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
ARCO_STORE   = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CLIM_STORE   = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
WB2_END      = pd.Timestamp("2023-01-10T18:00:00")


def open_anon(url: str) -> xr.Dataset:
    ds = xr.open_zarr(url, storage_options={"token": "anon"}, chunks={"time": 1})
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if float(ds.lat[0]) > float(ds.lat[-1]):       # stores are 90->-90; models want -90->90
        ds = ds.isel(lat=slice(None, None, -1))
    return ds


# Lazily-opened singletons (opening a zarr store is cheap metadata only).
_surf_src = None
_atm37_src = None
_arco_src = None
_clim = None


def _surf():
    global _surf_src
    if _surf_src is None:
        _surf_src = open_anon(SURF_STORE)
    return _surf_src


def _atm37():
    global _atm37_src
    if _atm37_src is None:
        _atm37_src = open_anon(FULL37_STORE)
    return _atm37_src


def _arco():
    global _arco_src
    if _arco_src is None:
        _arco_src = open_anon(ARCO_STORE)
    return _arco_src


def _clim_src():
    """1990-2019 T2m climatology, dims (hour, dayofyear, lat, lon).

    Prefers the local CONUS cache so the infer stage needs no network; falls back to the
    global GCS store. Both yield identical anomalies because every consumer crops to CONUS.
    """
    global _clim
    if _clim is None:
        if C.CLIM_FILE.exists():
            _clim = xr.open_dataset(C.CLIM_FILE)["2m_temperature"].load()
        else:
            _clim = open_anon(CLIM_STORE)["2m_temperature"]
    return _clim


def build_clim_conus() -> xr.DataArray:
    """Open the global climatology and crop to CONUS (for the local cache)."""
    return open_anon(CLIM_STORE)["2m_temperature"].sel(lat=C.LAT, lon=C.LON).compute()


def past_wb2(times) -> bool:
    return pd.DatetimeIndex(np.atleast_1d(times)).max() > WB2_END


# --------------------------------------------------------------------------- #
# Static fields (orography + land-sea mask), global 0.25deg. Cached to disk.
# --------------------------------------------------------------------------- #
def _build_statics() -> dict[str, xr.DataArray]:
    s = _surf()
    have = {"geopotential_at_surface", "land_sea_mask"} <= set(s.data_vars)
    if not have:
        raise RuntimeError(
            "geopotential_at_surface / land_sea_mask not found in the surface store; "
            "the WB2 with_derived_variables store is expected to contain them.")
    sub = s[["geopotential_at_surface", "land_sea_mask"]]
    return {v: (sub[v].isel(time=0, drop=True) if "time" in sub[v].dims else sub[v])
            for v in sub.data_vars}


def load_statics(use_cache: bool = True) -> dict[str, xr.DataArray]:
    """Return global 0.25deg statics, reading the local cache when present."""
    if use_cache and C.STATICS_FILE.exists():
        ds = xr.open_dataset(C.STATICS_FILE).load()
        return {v: ds[v] for v in ds.data_vars}
    statics = _build_statics()
    out = xr.Dataset(statics).compute()
    C.STATICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _atomic_to_netcdf(out, C.STATICS_FILE)
    return {v: out[v] for v in out.data_vars}


# --------------------------------------------------------------------------- #
# Precip accumulation helper (build the win_h-hour accumulation ending at each frame).
# --------------------------------------------------------------------------- #
def _precip_accum(src, times, win_h, out_name) -> xr.Dataset:
    times = pd.DatetimeIndex(times)
    if "total_precipitation_6hr" in src and win_h % 6 == 0:
        var, step = "total_precipitation_6hr", 6
    elif "total_precipitation" in src:
        var, step = "total_precipitation", 1
    else:
        raise KeyError(f"no precipitation source in store to build {out_name}")
    frames = [src[var].sel(
                  time=pd.date_range(t - pd.Timedelta(hours=win_h - step), t, freq=f"{step}h")
              ).sum("time") for t in times]
    return xr.concat(frames, dim="time").assign_coords(
        time=list(times)).to_dataset(name=out_name)


# --------------------------------------------------------------------------- #
# Model inputs: the two real init frames STEP_H apart, coarsened to the model grid.
# Returns a clean dataset on an *absolute* datetime time axis (NetCDF-safe).
# --------------------------------------------------------------------------- #
def build_raw_inputs(peak, lead_days: int, model: str = "gencast",
                     statics: dict | None = None, verbose: bool = False,
                     res: float = 1.0) -> xr.Dataset:
    cfg = C.model_cfg(model, res=res)
    step_h = cfg["step_h"]
    init = pd.Timestamp(peak) - pd.Timedelta(days=lead_days)
    times = pd.to_datetime([init - pd.Timedelta(hours=step_h), init])  # 2 input frames

    if past_wb2(times):
        surf_src = atm_src = _arco()
        if verbose:
            print("   (using ARCO-ERA5: past WB2 coverage)")
    else:
        surf_src = _surf()
        atm_src = _atm37() if cfg["levels"] == 37 else _surf()

    sv = [v for v in cfg["surf"] if v in surf_src]
    parts = [atm_src[C.ATM].sel(time=times, level=cfg["levels_hpa"]),
             surf_src[sv].sel(time=times)]

    if cfg["precip_in"]:
        pname = cfg["precip_in"]
        win_h = int(pname.rsplit("_", 1)[1].removesuffix("hr"))   # ..._12hr -> 12
        if pname in surf_src:
            parts.append(surf_src[[pname]].sel(time=times))
        else:
            psrc = (surf_src if ("total_precipitation_6hr" in surf_src
                                 or "total_precipitation" in surf_src) else _arco())
            parts.append(_precip_accum(psrc, times, win_h, pname))

    ds = xr.merge(parts).expand_dims(batch=1)

    # Coarsen the GLOBAL field to the model's native grid (Mini = 1.0deg from the 0.25deg
    # stores: stride 4 -> 181x360). GenCast is a global model; we keep the whole globe and
    # only subset to CONUS *after* prediction.
    stride = int(round(cfg["res"] / 0.25))
    sub = ((lambda a: a.isel(lat=slice(None, None, stride), lon=slice(None, None, stride)))
           if stride > 1 else (lambda a: a))
    ds = sub(ds)

    if statics is None:
        statics = load_statics()
    for k, v in statics.items():
        ds[k] = sub(v)

    # Keep the *absolute* datetime time axis. The relative-time axis and NaN target
    # template are built later in build_example_batch (and would not round-trip cleanly
    # through NetCDF as a negative timedelta index coordinate).
    return ds.compute()


# --------------------------------------------------------------------------- #
# Climatology + verification truth (observed week-N weekly-mean T2m anomaly, CONUS).
# Independent of the forecast horizon: always the 7 days ending at the peak.
# --------------------------------------------------------------------------- #
def clim_for(times) -> xr.DataArray:
    clim = _clim_src()
    t = pd.DatetimeIndex(times)
    return clim.sel(hour=xr.DataArray(np.asarray(t.hour), dims="time"),
                    dayofyear=xr.DataArray(np.asarray(t.dayofyear), dims="time"))


def true_verif_anom(peak) -> xr.DataArray:
    peak = pd.Timestamp(peak)
    win = pd.date_range(peak - pd.Timedelta(days=6), peak, freq="6h")  # the verification week
    src = _arco() if past_wb2(win) else _surf()
    obs = src["2m_temperature"].sel(time=win)
    anom = obs.mean("time") - clim_for(win).mean("time")              # clim is 1990-2019 (always WB2)
    return anom.sel(lat=C.LAT, lon=C.LON).compute()                  # (lat, lon)


# --------------------------------------------------------------------------- #
# Atomic NetCDF write (safe under concurrent week2/3/4 jobs hitting shared files).
# --------------------------------------------------------------------------- #
def _atomic_to_netcdf(ds: xr.Dataset, path) -> None:
    import os
    path = str(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    ds.to_netcdf(tmp)
    os.replace(tmp, path)
