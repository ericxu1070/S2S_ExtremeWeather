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
_clim = None

# ARCO is a flat 277-variable zarr tree; xr.open_zarr(root) OOMs.  Read arrays directly.
_arco_root = None
_arco_lat = _arco_lon = _arco_lev = None
# The v3 ARCO store's time axis is "hours since 1900-01-01" (the 1959 epoch belonged
# to the retired ar/1959-2022 store; using it here shifted every read back 59 years).
ARCO_EPOCH = pd.Timestamp("1900-01-01T00:00:00")
_ARCO_LEVELED = frozenset(C.ATM)


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


def _arco_group():
    """Lightweight zarr group handle (metadata only; do not xr.open_zarr the root)."""
    global _arco_root
    if _arco_root is None:
        import gcsfs
        import zarr
        fs = gcsfs.GCSFileSystem(token="anon")
        _arco_root = zarr.open(fs.get_mapper(ARCO_STORE.replace("gs://", "")), mode="r")
    return _arco_root


def _arco_coords():
    global _arco_lat, _arco_lon, _arco_lev
    if _arco_lat is None:
        root = _arco_group()
        _arco_lat = root["latitude"][:]
        _arco_lon = root["longitude"][:]
        _arco_lev = root["level"][:]
    return _arco_lat, _arco_lon, _arco_lev


def _arco_tindex(t) -> int:
    return int((pd.Timestamp(t) - ARCO_EPOCH) / pd.Timedelta(hours=1))


def _arco_spatial_slice(*, conus: bool, stride: int) -> tuple[slice, slice]:
    lat, lon, _ = _arco_coords()
    if conus:
        li = np.where((lat >= 24) & (lat <= 50))[0]
        lo = np.where((lon >= 235) & (lon <= 294))[0]
        return slice(li[0], li[-1] + 1, stride), slice(lo[0], lo[-1] + 1, stride)
    return slice(None, None, stride), slice(None, None, stride)


def _arco_lat_lon_coords(*, conus: bool, stride: int) -> tuple[np.ndarray, np.ndarray]:
    lat, lon, _ = _arco_coords()
    lat_sl, lon_sl = _arco_spatial_slice(conus=conus, stride=stride)
    lat_c = lat[lat_sl]
    lon_c = lon[lon_sl]
    if lat_c[0] > lat_c[-1]:
        lat_c = lat_c[::-1]
    return lat_c, lon_c


def arco_read(var: str, t, *, levels=None, conus: bool = False, stride: int = 1) -> np.ndarray:
    """Read one ARCO timestep (optionally a list of pressure levels) without xarray."""
    root = _arco_group()
    ti = _arco_tindex(t)
    lat_sl, lon_sl = _arco_spatial_slice(conus=conus, stride=stride)
    arr = root[var]
    if levels is not None:
        _, _, lev = _arco_coords()
        slabs = [np.asarray(arr[ti, int(np.where(lev == p)[0][0]), lat_sl, lon_sl])
                 for p in levels]
        out = np.stack(slabs, axis=0)
    else:
        out = np.asarray(arr[ti, lat_sl, lon_sl])
    if out.ndim >= 2 and out.shape[-2] > 1:
        lat, _, _ = _arco_coords()
        if lat[lat_sl][0] > lat[lat_sl][-1]:
            out = out[..., ::-1, :]
    return out


def _arco_load_var_times(var: str, times, *, levels=None, stride: int = 1) -> xr.DataArray:
    lat_c, lon_c = _arco_lat_lon_coords(conus=False, stride=stride)
    frames = []
    for t in pd.DatetimeIndex(times):
        if levels is not None:
            data = arco_read(var, t, levels=levels, conus=False, stride=stride)
            frames.append(xr.DataArray(
                data, dims=["level", "lat", "lon"],
                coords={"level": list(levels), "lat": lat_c, "lon": lon_c}))
        else:
            data = arco_read(var, t, conus=False, stride=stride)
            frames.append(xr.DataArray(data, dims=["lat", "lon"],
                                         coords={"lat": lat_c, "lon": lon_c}))
    return xr.concat(frames, dim="time", coords="minimal").assign_coords(time=list(times))


def _arco_precip_accum(times, win_h: int, out_name: str, stride: int = 1) -> xr.Dataset:
    times = pd.DatetimeIndex(times)
    lat_c, lon_c = _arco_lat_lon_coords(conus=False, stride=stride)
    frames = []
    for t in times:
        win = pd.date_range(t - pd.Timedelta(hours=win_h - 1), t, freq="1h")
        parts = [arco_read("total_precipitation", ti, conus=False, stride=stride) for ti in win]
        frames.append(xr.DataArray(sum(parts), dims=["lat", "lon"],
                                   coords={"lat": lat_c, "lon": lon_c}))
    return xr.concat(frames, dim="time").assign_coords(
        time=list(times)).to_dataset(name=out_name)


def release_stores() -> None:
    """Drop cached zarr handles so heavy prep stages can reclaim RAM."""
    global _surf_src, _atm37_src, _clim
    global _arco_root, _arco_lat, _arco_lon, _arco_lev
    for name in ("_surf_src", "_atm37_src"):
        ds = globals()[name]
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass
        globals()[name] = None
    _arco_root = _arco_lat = _arco_lon = _arco_lev = None
    _clim = None


def _stride_isel(da: xr.DataArray, stride: int) -> xr.DataArray:
    if stride <= 1:
        return da
    return da.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))


def _load_var_at_times(src: xr.Dataset, var: str, times, *,
                       levels=None, stride: int = 1) -> xr.DataArray:
    """Load one variable for a few timesteps, coarsening before materializing each slice.

    ERA5 zarr stores chunk only on ``time``, so a multi-time ``.compute()`` on the global
    grid can pull enormous graphs into RAM.  Loading one timestep at a time keeps peak
    memory proportional to one CONUS/global slice.
    """
    frames = []
    for t in pd.DatetimeIndex(times):
        da = src[var].sel(time=t)
        if levels is not None and "level" in da.dims:
            da = da.sel(level=levels)
        frames.append(_stride_isel(da, stride).load())
        del da
    out = xr.concat(frames, dim="time")
    del frames
    return out


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
def _precip_accum(src, times, win_h, out_name, stride: int = 1) -> xr.Dataset:
    times = pd.DatetimeIndex(times)
    if "total_precipitation_6hr" in src and win_h % 6 == 0:
        var, step = "total_precipitation_6hr", 6
    elif "total_precipitation" in src:
        var, step = "total_precipitation", 1
    else:
        raise KeyError(f"no precipitation source in store to build {out_name}")
    frames = []
    for t in times:
        win = pd.date_range(t - pd.Timedelta(hours=win_h - step), t, freq=f"{step}h")
        parts = [_stride_isel(src[var].sel(time=ti), stride).load()
                 for ti in win]
        frames.append(sum(parts))
    return xr.concat(frames, dim="time").assign_coords(
        time=list(times)).to_dataset(name=out_name)


# --------------------------------------------------------------------------- #
# Model inputs: the two real init frames STEP_H apart, coarsened to the model grid.
# Returns a clean dataset on an *absolute* datetime time axis (NetCDF-safe).
# --------------------------------------------------------------------------- #
def build_raw_inputs(peak, lead_days: int, model: str = "gencast",
                     statics: dict | None = None, verbose: bool = False,
                     res: float = 1.0) -> xr.Dataset:
    import gc

    cfg = C.model_cfg(model, res=res)
    step_h = cfg["step_h"]
    init = pd.Timestamp(peak) - pd.Timedelta(days=lead_days)
    times = pd.to_datetime([init - pd.Timedelta(hours=step_h), init])  # 2 input frames
    stride = int(round(cfg["res"] / 0.25))

    if past_wb2(times):
        if verbose:
            print("   (using ARCO-ERA5: past WB2 coverage)")
        atm_vars = [v for v in C.ATM if v in _arco_group()]
        sv = [v for v in cfg["surf"] if v in _arco_group()]
        parts = []
        for var in atm_vars:
            parts.append(_arco_load_var_times(var, times, levels=cfg["levels_hpa"],
                                              stride=stride).rename(var))
            release_stores()
            gc.collect()
        for var in sv:
            parts.append(_arco_load_var_times(var, times, stride=stride).rename(var))
            release_stores()
            gc.collect()
        if cfg["precip_in"]:
            pname = cfg["precip_in"]
            win_h = int(pname.rsplit("_", 1)[1].removesuffix("hr"))
            if pname in _arco_group():
                parts.append(_arco_load_var_times(pname, times, stride=stride)
                             .to_dataset(name=pname))
            else:
                parts.append(_arco_precip_accum(times, win_h, pname, stride=stride))
    else:
        surf_src = _surf()
        atm_src = _atm37() if cfg["levels"] == 37 else _surf()
        sv = [v for v in cfg["surf"] if v in surf_src]
        atm_vars = [v for v in C.ATM if v in atm_src]
        parts = []
        for var in atm_vars:
            parts.append(_load_var_at_times(atm_src, var, times,
                                            levels=cfg["levels_hpa"], stride=stride).rename(var))
            release_stores()
            gc.collect()
        for var in sv:
            parts.append(_load_var_at_times(surf_src, var, times, stride=stride).rename(var))
            release_stores()
            gc.collect()
        if cfg["precip_in"]:
            pname = cfg["precip_in"]
            win_h = int(pname.rsplit("_", 1)[1].removesuffix("hr"))
            if pname in surf_src:
                parts.append(_load_var_at_times(surf_src, pname, times, stride=stride)
                             .to_dataset(name=pname))
            else:
                psrc = surf_src if ("total_precipitation_6hr" in surf_src
                                    or "total_precipitation" in surf_src) else _atm37()
                parts.append(_precip_accum(psrc, times, win_h, pname, stride=stride))

    ds = xr.merge(parts).expand_dims(batch=1)

    if statics is None:
        statics = load_statics()
    for k, v in statics.items():
        ds[k] = _stride_isel(v, stride)

    release_stores()
    gc.collect()
    return ds


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
    import gc

    peak = pd.Timestamp(peak)
    win = pd.date_range(peak - pd.Timedelta(days=6), peak, freq="6h")
    sum_obs = None
    lat_c = lon_c = None
    if past_wb2(win):
        lat_c, lon_c = _arco_lat_lon_coords(conus=True, stride=1)
        for t in win:
            frame = arco_read("2m_temperature", t, conus=True, stride=1)
            sum_obs = frame if sum_obs is None else sum_obs + frame
            release_stores()
            gc.collect()
    else:
        src = _surf()
        for t in win:
            da = src["2m_temperature"].sel(time=t, lat=C.LAT, lon=C.LON).load()
            if lat_c is None:
                lat_c, lon_c = da.lat.values, da.lon.values
            sum_obs = da.values if sum_obs is None else sum_obs + da.values
            del da
            release_stores()
            gc.collect()
            src = _surf()
    obs = xr.DataArray(sum_obs / len(win), dims=("lat", "lon"),
                       coords={"lat": lat_c, "lon": lon_c})
    anom = obs - clim_for(win).mean("time")
    release_stores()
    gc.collect()
    return anom.load()


# --------------------------------------------------------------------------- #
# Atomic NetCDF write (safe under concurrent week2/3/4 jobs hitting shared files).
# --------------------------------------------------------------------------- #
def _atomic_to_netcdf(ds: xr.Dataset, path) -> None:
    import os
    path = str(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    ds.to_netcdf(tmp)
    os.replace(tmp, path)
