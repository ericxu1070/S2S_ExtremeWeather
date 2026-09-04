#!/usr/bin/env python
"""ERA5, at every valid time the chains visit. Login node; needs internet.

Two products, deliberately separate files:

* **the scalar series** (``<event>_era5_ref.nc``, ~12 kB) - global-mean T2m and z500,
  CONUS-mean T2m, and the z500 high-wavenumber fraction. This is item 4's only possible
  reference (there is NO z500 climatology anywhere in this repo - ``aires/aplots.py`` says
  so in its own figure caption, and ``clim_1990_2019_t2m_conus.nc`` is T2m and CONUS only)
  and it is the day-3 physics gate's. Read by every stage.
* **the T2m field** (``<event>_era5_t2m.nc``, ~6 MB) - 2 m temperature on the CONUS model
  grid, 105 x 237, at the same times. Read ONLY by ``astab.maps``, which needs the truth
  row of every map figure. Kept apart so a stage that wants the means never pays for the
  fields, and so the ~1 GB of GLOBAL fields is never stored at all.

Both read WeatherBench-2 through raw zarr rather than xarray. ``gencast_s2s.data.open_anon``
cannot be used: it is ``xr.open_zarr(..., chunks={"time": 1})``, which builds a dask graph
with one task per timestep per variable - the WB2 surface store has 93,544 timesteps and 66
arrays, and opening it alone reached 4 GB of RSS on this 15 GB login node before returning
(measured, 2026-08-27); the 1-hourly 37-level store, at 561,264 timesteps, took the whole
box. So this reads the store the way ``data.arco_read`` reads ARCO - raw zarr, no xarray,
no dask - for exactly the reason stated at the top of ``gencast_s2s/data.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aires import aindex as AI
from astab import sconfig as SC
from astab.diag import HI_WAVENUMBER, SPECTRUM_LAT, zonal_high_wavenumber_fraction
from astab.schedule import chains
from gencast_s2s import config as C


def ref_path(event: str) -> Path:
    return SC.ref_path(event)


def ref_field_path(event: str) -> Path:
    return SC.ref_field_path(event)

def ref_times(event: str, weeks=None) -> pd.DatetimeIndex:
    """Every valid time the chains of ``event`` visit: each chain's 3-day boundaries out to
    its peak, plus its init.

    Built from ``full_schedule``, NOT ``schedule``. An FCN3-extension chain's walker stops
    at 3 d but both its FCN3 arms roll the whole horizon and report on every 3-day boundary
    of it, and ``maps`` needs an ERA5 truth frame at each - a reference built from the walk
    schedule would cover three days of a 140-day chain.

    A union rather than a single ``date_range`` because the grids do not nest: the week-4
    grid is a subset of the week-10 one (42 d is a multiple of 3 d), the week-6 and week-8
    grids are not, and each chain's ragged tail lands where no other chain's does.
    """
    ts: set[pd.Timestamp] = set()
    for ch in chains([event], weeks):
        ts.add(ch.init)
        for (_k, lead, _n) in ch.full_schedule:
            ts.add(ch.valid_at(lead))
    return pd.DatetimeIndex(sorted(ts))


# ``gencast_s2s.data.open_anon`` cannot be used for this. It is
# ``xr.open_zarr(..., chunks={"time": 1})``, which builds a dask graph with one task per
# timestep per variable: the WB2 surface store has 93,544 timesteps and 66 arrays, and
# opening it alone reached 4 GB of RSS on this 15 GB login node before returning (measured,
# 2026-08-27); the 1-hourly 37-level store, at 561,264 timesteps, took the whole box.
#
# So this reads the store the way ``data.arco_read`` reads ARCO - raw zarr, no xarray, no
# dask - for exactly the reason stated at the top of ``gencast_s2s/data.py``. Every array
# needed here lives in the 6-HOURLY store (all chain boundaries are 00Z), whose
# geopotential chunk is 13 levels rather than 37, so this is also the cheaper read.
WB2_SURF = ("weatherbench2/datasets/era5/"
            "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
WB2_EPOCH = pd.Timestamp("1959-01-01")             # both stores: "hours since 1959-01-01"
_wb2: dict = {}


def _wb2_store() -> dict:
    """``{group, time, lat, lon, level}`` - metadata only, opened once."""
    if not _wb2:
        import gcsfs
        import zarr
        fs = gcsfs.GCSFileSystem(token="anon")
        g = zarr.open(fs.get_mapper(WB2_SURF), mode="r")
        _wb2.update(group=g, time=np.asarray(g["time"][:]),
                    lat=np.asarray(g["latitude"][:]),
                    lon=np.asarray(g["longitude"][:]),
                    level=np.asarray(g["level"][:]))
    return _wb2


def _wb2_frame(var: str, when, level: int | None = None) -> xr.DataArray:
    """One global frame, lat ASCENDING (the stores are 90 -> -90; every model here is not)."""
    st = _wb2_store()
    hours = int(round((pd.Timestamp(when) - WB2_EPOCH).total_seconds() / 3600))
    ti = int(np.searchsorted(st["time"], hours))
    if ti >= st["time"].size or int(st["time"][ti]) != hours:
        raise KeyError(f"{when} is not in the WB2 surface store")
    arr = st["group"][var]
    if level is None:
        data = np.asarray(arr[ti])
    else:
        li = int(np.where(st["level"] == int(level))[0][0])
        data = np.asarray(arr[ti, li])
    lat, lon = st["lat"], st["lon"]
    if lat[0] > lat[-1]:
        data, lat = data[::-1, :], lat[::-1]
    return xr.DataArray(data, dims=("lat", "lon"),
                        coords=dict(lat=lat.astype("float32"),
                                    lon=lon.astype("float32")))


def build_reference(event: str, *, force: bool = False) -> Path:
    """Global-mean T2m and z500, CONUS-mean T2m, and the z500 high-wavenumber fraction,
    at every valid time the chains visit. Login node; needs internet.

    Every init date in the sweep precedes ``WB2_END`` - the earliest is Uri's week-20 init
    at 2020-09-28 - so this reads WeatherBench-2 and never touches the ARCO store whose
    epoch bug is quarantined under
    ``runs/xres/quarantine_1964_epoch_bug/``.
    """
    from gencast_s2s import data as D

    out = ref_path(event)
    times = ref_times(event)
    if out.exists() and not force:
        with xr.open_dataset(out) as ds:
            have = pd.DatetimeIndex(ds["time"].values)
        if set(times) <= set(have):
            print(f"[refs] {event}: cached ({have.size} times) {out}")
            return out
        print(f"[refs] {event}: cache covers {have.size} of {times.size} times; rebuilding")

    if D.past_wb2(times):
        raise SystemExit(
            f"[refs] {event}: valid times reach {times.max()}, past WB2_END "
            f"({D.WB2_END}). Every init in this sweep was verified to precede it; a time "
            f"past it would silently switch the read to the ARCO store.")

    print(f"[refs] {event}: {times.size} times, {times[0].date()} .. {times[-1].date()}",
          flush=True)
    rows = []
    for i, t in enumerate(times, 1):
        t2m = _wb2_frame("2m_temperature", t)
        z = _wb2_frame("geopotential", t, level=500)
        rows.append(dict(
            time=t,
            global_t2m=float(AI.area_mean(t2m)),
            global_z500=float(AI.area_mean(z)),
            t2m_conus_mean=float(AI.area_mean(t2m.sel(lat=C.LAT, lon=C.LON))),
            z500_hi_wavenumber_frac=zonal_high_wavenumber_fraction(z),
        ))
        print(f"[refs]   {i}/{times.size} {t}  T2m {rows[-1]['global_t2m']:.2f} K  "
              f"CONUS {rows[-1]['t2m_conus_mean']:.2f} K  "
              f"hi-k {rows[-1]['z500_hi_wavenumber_frac']:.4f}", flush=True)
        del t2m, z

    df = pd.DataFrame(rows).set_index("time")
    ds = xr.Dataset({c: ("time", df[c].values) for c in df.columns},
                    coords=dict(time=pd.DatetimeIndex(df.index)))
    ds.attrs.update(event=event, source="WeatherBench-2 ERA5",
                    note="cos(lat)-weighted means; the high-wavenumber fraction uses "
                         "astab.zonal_high_wavenumber_fraction with the same cutoff and "
                         "latitude band as the model diagnostics",
                    cutoff=HI_WAVENUMBER, lat_band=list(SPECTRUM_LAT))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".nc.tmp.{os.getpid()}")
    ds.to_netcdf(tmp)
    os.replace(tmp, out)
    print(f"[refs] wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return out


def reference_at(event: str, when) -> dict[str, float]:
    """The ERA5 reference row at ``when``; ``{}`` if the series has no such time."""
    p = ref_path(event)
    if not p.exists():
        return {}
    with xr.open_dataset(p) as ds:
        t = pd.DatetimeIndex(ds["time"].values)
        w = pd.Timestamp(when)
        if w not in set(t):
            return {}
        i = list(t).index(w)
        return {str(v): float(ds[v].values[i]) for v in ds.data_vars}



# --------------------------------------------------------------------------- #
# The T2m truth FIELD - the maps' second row
# --------------------------------------------------------------------------- #
def build_reference_fields(event: str, *, force: bool = False) -> Path:
    """ERA5 2 m temperature on the CONUS model grid at every chain checkpoint.

    ``C.LAT``/``C.LON`` are the same slices ``xres`` and ``fcn3`` crop with, and WB2 is the
    same 0.25 deg grid the 0.25 deg checkpoint runs on, so the result is 105 x 237 and is
    directly subtractable from a walker's ``diag.nc`` or an FCN3 cube with no regridding.
    That is the whole reason the truth row can share a color scale with the forecast row.

    GLOBAL fields are deliberately not stored: 128 times x 721 x 1440 x 4 B is ~530 MB per
    event for a row no figure draws. If a global map is ever wanted, ``_wb2_frame`` is
    right there and costs one read.
    """
    out = ref_field_path(event)
    times = ref_times(event)
    if out.exists() and not force:
        with xr.open_dataset(out) as ds:
            have = pd.DatetimeIndex(ds["time"].values)
        if set(times) <= set(have):
            print(f"[refs] {event}: T2m fields cached ({have.size} times) {out}")
            return out
        print(f"[refs] {event}: field cache covers {have.size} of {times.size} times; "
              f"rebuilding")

    from gencast_s2s import data as D
    if D.past_wb2(times):
        raise SystemExit(
            f"[refs] {event}: valid times reach {times.max()}, past WB2_END ({D.WB2_END}).")

    print(f"[refs] {event}: {times.size} T2m fields, "
          f"{times[0].date()} .. {times[-1].date()}", flush=True)
    frames = []
    for i, t in enumerate(times, 1):
        f = _wb2_frame("2m_temperature", t).sel(lat=C.LAT, lon=C.LON)
        frames.append(f.astype("float32"))
        if i % 10 == 0 or i == times.size:
            print(f"[refs]   {i}/{times.size} {t}  "
                  f"CONUS mean {float(AI.area_mean(f)):.2f} K", flush=True)
    da = xr.concat(frames, dim=pd.Index(pd.DatetimeIndex(times), name="time"))
    ds = xr.Dataset({"2m_temperature": da})
    ds["2m_temperature"].attrs.update(units="K", long_name="2 metre temperature")
    ds.attrs.update(event=event, source="WeatherBench-2 ERA5",
                    note="CONUS crop on the 0.25 deg model grid (config.LAT/LON); the "
                         "truth row of astab.maps",
                    grid=f"lat={ds.sizes['lat']} lon={ds.sizes['lon']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".nc.tmp.{os.getpid()}")
    ds.to_netcdf(tmp, encoding={"2m_temperature": dict(zlib=True, complevel=4)})
    os.replace(tmp, out)
    print(f"[refs] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{ds.sizes['time']} x {ds.sizes['lat']} x {ds.sizes['lon']})")
    return out


def truth_fields(event: str) -> xr.DataArray:
    """The cached ERA5 T2m field series, or a message saying how to build it."""
    p = ref_field_path(event)
    if not p.exists():
        raise SystemExit(
            f"no ERA5 truth fields at {p}.\n"
            f"  Build them on the LOGIN node (needs internet, ~1 min per event):\n"
            f"    PYTHONPATH=. python -m astab.run --stage refs --events {event}")
    with xr.open_dataset(p) as ds:
        return ds["2m_temperature"].load()
