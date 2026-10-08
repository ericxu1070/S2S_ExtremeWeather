"""HRRR analysis T2m as an alternative truth for the 42 acal cases.

The ERA5 truth (``runs/acal/index/era5_t2m_anom_12h_<year>.nc``) is the T3 index every acal
scorer reads. This module builds its HRRR analog on the same 0.25 deg grid and the same
00Z/12Z time axis, and exposes it through the truth-provider interface of ``acal.truth``:

* ``runs/acal/index_hrrr/hrrr_t2m_anom_12h_<year>.nc`` for 2021..2026, one per ERA5 index
  year file and with exactly its time axis (each file opens with a 6-day lead-in from the
  previous 26 December). Variables:

  - ``anom_raw``: HRRR f00 T2m, area-averaged from 3 km to 0.25 deg, minus the ERA5
    1990-2019 climatology ``clim(hour, dayofyear)`` (the climatology the ERA5 index uses).
  - ``anom``: ``anom_raw`` minus the leave-one-year-out HRRR-minus-ERA5 offset, per cell x
    hour (00Z/12Z) x calendar month. The headline truth ('hrrr').
  - ``mask``: the geometric HRRR coverage mask (all four 0.25 deg cell corners inside the
    HRRR Lambert grid). Both anomalies are NaN outside it.
  - ``offset_loyo(month, hour, lat, lon)``: the offset field used for frames of this
    file's calendar year (frames of the 26-31 December lead-in use the previous year's).

* Truth providers ``hrrr`` (offset-corrected, headline) and ``hrrr_raw`` (sensitivity),
  reached through ``get_truth(name)``.

* ``runs/acal/index_hrrr/cases_hrrr.csv``: per-case observed A_L under every truth.

Why the offset: there is no 30-year HRRR climatology, so ``HRRR - ERA5 clim`` folds HRRR's
analysis climate (about -0.2 K on the CONUS mean, +-1 K per cell, strongly diurnal) into the
anomaly. Removing the mean HRRR-minus-ERA5 difference, estimated from the OTHER years only,
leaves HRRR's own weather in the case window. Offset years: the HRRRv4 era only (HRRRv4 has
been operational since 2020-12-02, i.e. every frame of the index); see ``OFFSET_NOTE``.

Stages::

    python -m acal.hrrrtruth --stage fetch2026 [--workers 8]   # AWS 2026 + bad-disk frames
    python -m acal.hrrrtruth --stage build     [--workers 8]   # index year files
    python -m acal.hrrrtruth --stage cases                     # cases_hrrr.csv + daily csv

Everything is cache-aware: the AWS GRIB messages, the per-month bin-mean chunks under
``work/`` and the year files are skipped when present (``--force`` rebuilds).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from acal import ccfg

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
HRRR_NC = Path(os.environ.get("HRRR_NC", "/glade/derecho/scratch/mdarman/hrrr_work/hrrr_nc_v3"))
DISK_END = pd.Timestamp("2025-12-31T18:00")      # last valid time of the on-disk archive
ERA5_INDEX = ccfg.ACAL_ROOT / "index"
OUT = ccfg.ACAL_ROOT / "index_hrrr"
RAW = OUT / "raw"                                 # AWS GRIB messages only
WORK = OUT / "work"                               # per-month bin-mean chunks (resumable)
EPISODES = ccfg.ACAL_ROOT / "catalog" / "conus_episodes_21d_2021_2025.csv"
CASES_CSV = OUT / "cases_hrrr.csv"
DAILY_CSV = OUT / "conus_daily_hrrr_2021_2026.csv"
AWS_CHECK = OUT / "aws_vs_disk_check.json"

YEARS = tuple(range(2021, 2027))
INDEX_START = pd.Timestamp("2020-12-26T00:00")
INDEX_END = pd.Timestamp(ccfg.PEAK_END)           # 2026-08-31T00, the ERA5 index's last frame
DAILY_START, DAILY_END = "2021-01-01", "2026-08-31"
COMPARE_FRAME = pd.Timestamp("2025-07-15T00:00")  # overlapping frame for the AWS-vs-disk check

AWS = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
AWS_FIELD = ":TMP:2 m above ground:anl:"

# HRRR native grid (research/hrrr.md section 1; reproduced by pyproj to 1e-9 m).
LCC = dict(proj="lcc", lat_1=38.5, lat_2=38.5, lat_0=38.5, lon_0=262.5, R=6371229)
HRRR_NX, HRRR_NY, HRRR_DX = 1799, 1059, 3000.0

# HRRRv4 operational 2020-12-02: every frame of the index (2020-12-26 onward) is v4.
HRRRV4_START = pd.Timestamp("2020-12-02")
if INDEX_START < HRRRV4_START:
    raise SystemExit("the offset pool would mix HRRR versions; see OFFSET_NOTE")
OFFSET_NOTE = (
    "LOYO offset: per cell x hour (00Z/12Z) x calendar month mean of HRRR minus ERA5 over all "
    "index frames (2020-12-26..2026-08-31, all HRRRv4) whose calendar year differs from the "
    "frame's own year. Earlier HRRR versions (v1-v3, 2015-2020-12-01) are excluded: they are a "
    "different model/DA analysis climate, and the offset must describe the analysis the truth "
    "frames come from.")

TRUTH_NAMES = ("hrrr", "hrrr_raw")


# --------------------------------------------------------------------------- #
# Grid helpers (pure functions, tested on synthetic grids)
# --------------------------------------------------------------------------- #
def target_grid() -> tuple[np.ndarray, np.ndarray]:
    """The acal 0.25 deg grid (ascending lat 24..50, lon 235..294 E)."""
    from aires.cfs import target_grid as tg
    return tg()


def _spacing(a: np.ndarray) -> float:
    d = np.diff(a)
    if not np.allclose(d, d[0], atol=1e-6):
        raise ValueError("target axis is not uniformly spaced")
    return float(d[0])


def bin_index(hlat: np.ndarray, hlon: np.ndarray, tlat: np.ndarray, tlon: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """Map every source point to the target cell whose centre is nearest in lat/lon.

    Returns ``(flat, cnt)``: ``flat`` (source size,) int64 flat target index or -1 when the
    point lies outside the target grid, and ``cnt`` (nlat, nlon) points per cell.
    """
    dla, dlo = _spacing(tlat), _spacing(tlon)
    bi = np.rint((np.asarray(hlat, float) - tlat[0]) / dla).astype(np.int64)
    bj = np.rint((np.asarray(hlon, float) % 360.0 - tlon[0]) / dlo).astype(np.int64)
    ok = (bi >= 0) & (bi < tlat.size) & (bj >= 0) & (bj < tlon.size)
    flat = np.where(ok, bi * tlon.size + bj, -1).ravel()
    cnt = np.bincount(flat[flat >= 0], minlength=tlat.size * tlon.size)
    return flat, cnt.reshape(tlat.size, tlon.size)


def bin_mean(field: np.ndarray, flat: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """Area-average (bin-mean) of a source field onto the target grid; NaN where empty."""
    sel = flat >= 0
    s = np.bincount(flat[sel], weights=np.asarray(field, "float64").ravel()[sel],
                    minlength=cnt.size).reshape(cnt.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def corner_mask(tlat: np.ndarray, tlon: np.ndarray, *, lcc: dict | None = None,
                x0: float, y0: float, nx: int, ny: int, dx: float) -> np.ndarray:
    """True where all four corners of a target cell lie inside the source LCC grid.

    The source grid is points ``x0 + i*dx, y0 + j*dx`` (i < nx, j < ny); a point is inside
    when it lies within half a grid step of that rectangle - the grid's own cell edges.
    """
    import pyproj
    P = pyproj.Proj(**(lcc or LCC))
    dla, dlo = _spacing(tlat), _spacing(tlon)
    TLA, TLO = np.meshgrid(tlat, tlon, indexing="ij")
    full = np.ones(TLA.shape, bool)
    for a in (-0.5, 0.5):
        for b in (-0.5, 0.5):
            x, y = P(TLO + b * dlo, TLA + a * dla)
            i, j = (x - x0) / dx, (y - y0) / dx
            full &= (i >= -0.5) & (i <= nx - 0.5) & (j >= -0.5) & (j <= ny - 0.5)
    return full


def area_mean(a: np.ndarray, lat: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """cos(lat)-weighted mean over the last two axes (lat, lon), restricted to ``mask``."""
    w = np.cos(np.deg2rad(np.asarray(lat, float)))[:, None] * np.ones(a.shape[-1])
    if mask is not None:
        w = w * np.asarray(mask, bool)
    a = np.asarray(a, "float64")
    return np.nansum(a * w, axis=(-2, -1)) / np.sum(w)


# --------------------------------------------------------------------------- #
# HRRR grid, remap index and mask (built once, cached)
# --------------------------------------------------------------------------- #
GRID_CACHE = OUT / "hrrr_remap_meta.npz"


def _grid_ref_file() -> Path:
    return HRRR_NC / "2021" / "hrrr.20210101.t00z.v3.nc"


@lru_cache(maxsize=1)
def remap_meta() -> dict:
    """``flat``, ``cnt``, ``mask`` (lat, lon) and the target lat/lon, cached to disk."""
    tlat, tlon = target_grid()
    if GRID_CACHE.exists():
        z = np.load(GRID_CACHE)
        if np.array_equal(z["tlat"], tlat) and np.array_equal(z["tlon"], tlon):
            return {k: z[k] for k in z.files}
    import netCDF4
    import pyproj
    with netCDF4.Dataset(_grid_ref_file()) as ds:
        hlat = np.asarray(ds["latitude"][:], "float64")
        hlon = np.asarray(ds["longitude"][:], "float64")
    if hlat.shape != (HRRR_NY, HRRR_NX):
        raise SystemExit(f"HRRR grid shape {hlat.shape} != {(HRRR_NY, HRRR_NX)}")
    x0, y0 = pyproj.Proj(**LCC)(hlon[0, 0], hlat[0, 0])
    flat, cnt = bin_index(hlat, hlon, tlat, tlon)
    mask = corner_mask(tlat, tlon, x0=x0, y0=y0, nx=HRRR_NX, ny=HRRR_NY, dx=HRRR_DX)
    if (cnt[mask] == 0).any():
        raise SystemExit("a cell inside the corner mask received no HRRR point")
    meta = dict(flat=flat, cnt=cnt, mask=mask, tlat=tlat, tlon=tlon)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = GRID_CACHE.with_suffix(".tmp.npz")
    np.savez(tmp, **meta)
    os.replace(tmp, GRID_CACHE)
    return meta


# --------------------------------------------------------------------------- #
# Frame sources: on-disk netCDF, else AWS GRIB message
# --------------------------------------------------------------------------- #
def index_times() -> pd.DatetimeIndex:
    """Every 00Z/12Z frame of the ERA5 index (2020-12-26T00..2026-08-31T00)."""
    t = pd.date_range(INDEX_START, INDEX_END, freq="12h")
    return t[np.isin(t.hour, ccfg.HOURS)]


def disk_path(t: pd.Timestamp) -> Path:
    return HRRR_NC / f"{t.year}" / f"hrrr.{t:%Y%m%d}.t{t:%H}z.v3.nc"


def raw_path(t: pd.Timestamp) -> Path:
    return RAW / f"hrrr.{t:%Y%m%d}.t{t:%H}z.wrfsfcf00.t2m.grib2"


@lru_cache(maxsize=1)
def _bad_files() -> frozenset:
    p = HRRR_NC / "bad_files.txt"
    return frozenset(p.read_text().split()) if p.exists() else frozenset()


def disk_ok(t: pd.Timestamp) -> bool:
    p = disk_path(t)
    return t <= DISK_END and p.exists() and str(p) not in _bad_files()


def needs_aws(times) -> list[pd.Timestamp]:
    return [pd.Timestamp(t) for t in times if not disk_ok(pd.Timestamp(t))]


def read_disk(t: pd.Timestamp) -> np.ndarray:
    import netCDF4
    with netCDF4.Dataset(disk_path(t)) as ds:
        return np.asarray(ds["t2m"][:], "float64")


def read_raw(t: pd.Timestamp) -> np.ndarray:
    """Decode one AWS GRIB message (TMP 2 m analysis), checking field and valid time."""
    import pygrib
    with pygrib.open(str(raw_path(t))) as g:
        m = g[1]
        if m.shortName != "2t" or pd.Timestamp(m.validDate) != t or m.units != "K":
            raise ValueError(f"{raw_path(t).name}: {m.shortName} {m.validDate} {m.units}")
        v = np.asarray(m.values, "float64")
    if v.shape != (HRRR_NY, HRRR_NX):
        raise ValueError(f"{raw_path(t).name}: shape {v.shape}")
    return v


def read_frame(t: pd.Timestamp) -> np.ndarray:
    """HRRR f00 T2m (K) on the native 3 km grid, rows south to north."""
    t = pd.Timestamp(t)
    a = read_disk(t) if disk_ok(t) else read_raw(t)
    if not np.isfinite(a).all() or a.max() < 100.0:
        raise ValueError(f"{t}: non-finite or non-Kelvin T2m (max {np.nanmax(a):.1f})")
    return a


def _get(url: str, rng: tuple[int, int] | None = None, tries: int = 4) -> bytes:
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url)
            if rng is not None:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:                             # noqa: BLE001 - retried
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last}")


def fetch_one(t: pd.Timestamp, force: bool = False) -> Path:
    """Byte-range download of the 2 m TMP analysis message for valid time ``t``."""
    t = pd.Timestamp(t)
    out = raw_path(t)
    if out.exists() and not force:
        return out
    base = f"{AWS}/hrrr.{t:%Y%m%d}/conus/hrrr.t{t:%H}z.wrfsfcf00.grib2"
    lines = _get(base + ".idx").decode().splitlines()
    k = [i for i, ln in enumerate(lines) if AWS_FIELD in ln]
    if len(k) != 1:
        raise RuntimeError(f"{base}.idx: {len(k)} matches for {AWS_FIELD!r}")
    start = int(lines[k[0]].split(":")[1])
    end = int(lines[k[0] + 1].split(":")[1]) - 1
    data = _get(base, (start, end))
    if len(data) != end - start + 1 or data[:4] != b"GRIB":
        raise RuntimeError(f"{base}: bad range read ({len(data)} bytes)")
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, out)
    read_raw(t)                                            # decode check; raises if wrong
    return out


def stage_fetch2026(workers: int = 8, force: bool = False) -> None:
    times = needs_aws(index_times()) + [COMPARE_FRAME]
    print(f"[fetch] {len(times)} frames from AWS ({times[0]} .. {times[-2]}, + compare "
          f"{COMPARE_FRAME})", flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(lambda t: fetch_one(t, force), times):
            done += 1
            if done % 50 == 0:
                print(f"[fetch] {done}/{len(times)}  {time.time() - t0:.0f}s", flush=True)
    a, b = read_raw(COMPARE_FRAME), read_disk(COMPARE_FRAME)
    d = a - b
    rep = dict(frame=str(COMPARE_FRAME), aws=f"{AWS} wrfsfcf00 {AWS_FIELD}",
               disk=str(disk_path(COMPARE_FRAME)), max_abs_diff_K=float(np.abs(d).max()),
               mean_diff_K=float(d.mean()), rms_diff_K=float(np.sqrt((d ** 2).mean())),
               bit_identical=bool(np.array_equal(a, b)), n_aws_frames=len(times) - 1)
    AWS_CHECK.write_text(json.dumps(rep, indent=1))
    print(f"[fetch] done in {time.time() - t0:.0f}s; AWS vs disk {COMPARE_FRAME}: "
          f"max|d|={rep['max_abs_diff_K']:.3g} K, bit-identical={rep['bit_identical']}")


# --------------------------------------------------------------------------- #
# Build: per-month bin-mean chunks, then year files with the LOYO offset
# --------------------------------------------------------------------------- #
def _chunk_path(month: pd.Period) -> Path:
    return WORK / f"hrrr_t2m_bin_{month}.npz"


def build_chunk(month: str, force: bool = False) -> Path:
    """Bin-mean absolute T2m (K, float32, NaN outside the mask) for one calendar month."""
    month = pd.Period(month, "M")
    out = _chunk_path(month)
    if out.exists() and not force:
        return out
    meta = remap_meta()
    times = index_times()
    times = times[(times >= month.start_time) & (times <= month.end_time)]
    arr = np.empty((len(times),) + meta["cnt"].shape, "float32")
    src = []
    for i, t in enumerate(times):
        b = bin_mean(read_frame(t), meta["flat"], meta["cnt"])
        b[~meta["mask"]] = np.nan
        arr[i] = b
        src.append("disk" if disk_ok(t) else "aws")
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, t2m=arr, time=times.values.astype("datetime64[ns]"), src=np.array(src))
    os.replace(tmp, out)
    return out


def _months() -> list[str]:
    t = index_times()
    return [str(p) for p in pd.period_range(t[0], t[-1], freq="M")]


def load_chunks() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    ts, arrs, srcs = [], [], []
    for m in _months():
        z = np.load(_chunk_path(pd.Period(m, "M")))
        ts.append(z["time"]); arrs.append(z["t2m"]); srcs.append(z["src"])
    times = pd.DatetimeIndex(np.concatenate(ts))
    if not times.equals(index_times()):
        raise SystemExit("chunk times do not tile the index time axis")
    return times, np.concatenate(arrs), np.concatenate(srcs)


def era5_index() -> xr.DataArray:
    """The ERA5 12-hourly anomaly index, all years, de-overlapped (lazy)."""
    files = [ERA5_INDEX / f"era5_t2m_anom_12h_{y}.nc" for y in YEARS]
    da = xr.concat([xr.open_dataset(f)["t2m_anom"] for f in files], dim="time")
    _, keep = np.unique(da["time"].values, return_index=True)
    return da.isel(time=np.sort(keep))


def clim_at(times: pd.DatetimeIndex) -> np.ndarray:
    """ERA5 1990-2019 clim(hour, dayofyear) at each time, exactly as acases.build_year."""
    import gencast_s2s.data as D
    clim = D._clim_src()
    out = np.empty((len(times), clim.lat.size, clim.lon.size), "float32")
    for i, t in enumerate(times):
        out[i] = clim.sel(hour=int(t.hour), dayofyear=int(t.dayofyear)).values
    return out


def loyo_offset(diff: np.ndarray, times: pd.DatetimeIndex, pool: np.ndarray | None = None
                ) -> tuple[dict, np.ndarray, np.ndarray]:
    """Leave-one-year-out HRRR-minus-ERA5 offset per (month, hour, cell).

    ``diff`` (time, lat, lon) = HRRR minus ERA5 (NaN where undefined); ``pool`` (time,) bool
    selects the frames allowed into the estimate (default all). Returns
    ``(offsets, S, N)`` with ``offsets[year]`` an array (12, n_hours, lat, lon): the mean
    over pool frames of every OTHER calendar year, and S/N the per-(year, month, hour)
    sums and counts.
    """
    hours = list(ccfg.HOURS)
    years = sorted(set(times.year))
    pool = np.ones(len(times), bool) if pool is None else np.asarray(pool, bool)
    S = np.zeros((len(years), 12, len(hours)) + diff.shape[1:], "float64")
    N = np.zeros_like(S)
    for i, t in enumerate(times):
        if not pool[i]:
            continue
        d = diff[i]
        ok = np.isfinite(d)
        y, m, h = years.index(t.year), t.month - 1, hours.index(t.hour)
        S[y, m, h][ok] += d[ok]
        N[y, m, h][ok] += 1
    offs = {}
    for k, y in enumerate(years):
        s = S.sum(0) - S[k]
        n = N.sum(0) - N[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            offs[y] = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype("float32")
    return offs, S, N


def stage_build(workers: int = 8, force: bool = False) -> None:
    if not force and all(f.exists() for f in index_files()):
        print(f"[build] cached: {', '.join(f.name for f in index_files())}")
        return
    meta = remap_meta()
    print(f"[build] mask {int(meta['mask'].sum())} cells; {len(index_times())} frames; "
          f"{len(needs_aws(index_times()))} from AWS", flush=True)
    missing = [t for t in needs_aws(index_times()) if not raw_path(t).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} AWS frames not fetched (first {missing[0]}); "
                         f"run --stage fetch2026 first")
    months = _months()
    todo = [m for m in months if force or not _chunk_path(pd.Period(m, "M")).exists()]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, p in enumerate(ex.map(build_chunk, todo, [force] * len(todo))):
            print(f"[build] chunk {p.name} ({k + 1}/{len(todo)}, {time.time() - t0:.0f}s)",
                  flush=True)
    times, t2m, src = load_chunks()
    mask = meta["mask"]
    if np.isnan(t2m[:, mask]).any():
        raise SystemExit("unexpected NaN inside the HRRR mask")
    era = era5_index().sel(time=times).values.astype("float32")
    anom_raw = (t2m - clim_at(times)).astype("float32")
    anom_raw[:, ~mask] = np.nan
    offs, S, N = loyo_offset(anom_raw - era, times)
    anom = np.empty_like(anom_raw)
    hours = list(ccfg.HOURS)
    for i, t in enumerate(times):
        anom[i] = anom_raw[i] - offs[t.year][t.month - 1, hours.index(t.hour)]
    if np.isnan(anom[:, mask]).any():
        raise SystemExit("offset undefined for some (cell, month, hour) inside the mask")
    for y in YEARS:
        _write_year(y, times, anom_raw, anom, mask, offs, N, src)
    print(f"[build] done in {time.time() - t0:.0f}s")


def _write_year(year: int, times, anom_raw, anom, mask, offs, N, src) -> Path:
    out = OUT / f"hrrr_t2m_anom_12h_{year}.nc"
    with xr.open_dataset(ERA5_INDEX / f"era5_t2m_anom_12h_{year}.nc") as e:
        et = pd.DatetimeIndex(e.time.values)
        lat, lon = e.lat.values, e.lon.values
    k = times.get_indexer(et)
    if (k < 0).any():
        raise SystemExit(f"{year}: ERA5 index times missing from the HRRR axis")
    tlat, tlon = target_grid()
    if not (np.allclose(lat, tlat) and np.allclose(lon, tlon)):
        raise SystemExit(f"{year}: ERA5 index grid != target grid")
    years = sorted(offs)
    ds = xr.Dataset(
        {"anom": (("time", "lat", "lon"), anom[k]),
         "anom_raw": (("time", "lat", "lon"), anom_raw[k]),
         "mask": (("lat", "lon"), mask.astype("int8")),
         "offset_loyo": (("month", "hour", "lat", "lon"), offs[year]),
         "source": (("time",), src[k].astype(str))},
        coords={"time": et, "lat": lat, "lon": lon,
                "month": np.arange(1, 13), "hour": np.array(ccfg.HOURS)},
        attrs={"source": "HRRR f00 analysis T2m (TMP 2 m), 3 km -> 0.25 deg bin-mean, "
                         "minus WB2 ERA5 1990-2019 hourly climatology",
               "hours": ",".join(str(h) for h in ccfg.HOURS),
               "remap": "bin-mean: every 3 km point to the 0.25 deg cell with the nearest "
                        "centre in lat/lon; cell value = mean of its points",
               "mask": "geometric: all 4 cell corners inside the HRRR LCC grid (pyproj); "
                       "anom/anom_raw are NaN outside",
               "anom_raw": "HRRR minus ERA5 clim(hour, dayofyear)",
               "anom": "anom_raw minus offset_loyo[month, hour] (headline 'hrrr' truth)",
               "offset_note": OFFSET_NOTE,
               "offset_years_all": ",".join(str(y) for y in years),
               "offset_loyo_left_out_year": year,
               "offset_min_frames": int(np.min(N.sum(0)[:, :, mask] - N[years.index(year)]
                                               [:, :, mask])),
               "hrrr_archive": str(HRRR_NC),
               "aws": f"{AWS}/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcf00.grib2 {AWS_FIELD}",
               "note": "12-hourly CONUS T2m anomaly on the ERA5 index time axis; "
                       "A_L = 13-frame mean over [peak-6d, peak]"},
    )
    for v in ("anom", "anom_raw"):
        ds[v].attrs["units"] = "K"
    enc = {v: dict(zlib=True, complevel=4, chunksizes=(1, lat.size, lon.size))
           for v in ("anom", "anom_raw")}
    enc["offset_loyo"] = dict(zlib=True, complevel=4)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, out)
    print(f"[build] wrote {out.name}: {len(et)} frames {et[0]} .. {et[-1]}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Truth providers (interface of acal.truth: mask / frames / obs_al / daily_conus)
# --------------------------------------------------------------------------- #
def index_files() -> list[Path]:
    return [OUT / f"hrrr_t2m_anom_12h_{y}.nc" for y in YEARS]


@lru_cache(maxsize=2)
def _index(var: str) -> xr.DataArray:
    fs = index_files()
    miss = [f.name for f in fs if not f.exists()]
    if miss:
        raise SystemExit(f"HRRR index not built ({miss[0]}); run python -m acal.hrrrtruth "
                         f"--stage build")
    da = xr.concat([xr.open_dataset(f)[var] for f in fs], dim="time")
    _, keep = np.unique(da["time"].values, return_index=True)
    return da.isel(time=np.sort(keep))


@lru_cache(maxsize=1)
def _mask() -> xr.DataArray:
    _index("anom")                                         # raises if not built
    with xr.open_dataset(index_files()[0]) as ds:
        return (ds["mask"] > 0).load().rename("mask")


@lru_cache(maxsize=1)
def _peaks() -> dict:
    df = pd.read_csv(EPISODES)
    return {e: pd.Timestamp(p) for e, p in zip(df.episode_id, df.peak)}


def window_times(peak, window: str = "13f") -> pd.DatetimeIndex:
    """13f: the 13 frames at 00Z/12Z over [peak-6d 00Z, peak 00Z]; 12f: the first 12."""
    peak = pd.Timestamp(peak).normalize()
    t = pd.date_range(peak - pd.Timedelta(days=ccfg.WINDOW_DAYS), peak, freq="12h")
    if window == "13f":
        return t
    if window == "12f":
        return t[:-1]
    raise ValueError(f"window must be '13f' or '12f', not {window!r}")


class HrrrTruth:
    """HRRR truth provider. ``name`` 'hrrr' = offset-corrected, 'hrrr_raw' = raw."""

    def __init__(self, name: str):
        if name not in TRUTH_NAMES:
            raise ValueError(f"unknown HRRR truth {name!r}; expected one of {TRUTH_NAMES}")
        self.name = name
        self.var = "anom" if name == "hrrr" else "anom_raw"
        self.label = "HRRR (offset-corrected)" if name == "hrrr" else "HRRR (raw)"

    def __repr__(self) -> str:
        return f"HrrrTruth({self.name!r})"

    def mask(self) -> xr.DataArray:
        return _mask().copy()

    def frames(self, eid: str, window: str = "13f") -> xr.DataArray:
        """Anomaly (K) on the case window, (time, lat, lon), NaN outside the mask."""
        da = _index(self.var).sel(time=window_times(_peaks()[eid], window)).load()
        n = ccfg.N_WINDOW_FRAMES - (window == "12f")
        if da.sizes["time"] != n:
            raise SystemExit(f"{eid}: {da.sizes['time']} HRRR frames, expected {n}")
        da.attrs.update(truth=self.name, units="K")
        return da.rename(self.name)

    def obs_al(self, eid: str, window: str = "13f") -> float:
        f = self.frames(eid, window)
        m = self.mask().values
        return float(area_mean(f.mean("time").values, f.lat.values, m))

    def daily_conus(self) -> pd.Series:
        """Daily CONUS A_L, 2021-01-01..2026-08-31: 13-frame mean ending at each 00Z,
        cos-lat over the HRRR mask (catalog conus_daily_2021_2025.csv definition)."""
        col = "a_l_" + self.name
        d = pd.read_csv(_daily_csv(), parse_dates=["date"])
        return pd.Series(d[col].values, index=pd.DatetimeIndex(d.date), name="a_l_conus")


@lru_cache(maxsize=None)
def get_truth(name: str) -> HrrrTruth:
    return HrrrTruth(name)


def _daily_table() -> pd.DataFrame:
    """Daily 13-frame CONUS A_L: hrrr, hrrr_raw and ERA5 on the HRRR mask, plus ERA5 on the
    full box (``a_l_era5_full`` reproduces catalog/conus_daily_2021_2025.csv and extends it
    through 2026-08-31)."""
    mask = get_truth("hrrr").mask().values
    out = {}
    for name, da, m in (("hrrr", _index("anom"), mask), ("hrrr_raw", _index("anom_raw"), mask),
                        ("era5_mask", era5_index(), mask), ("era5_full", era5_index(), None)):
        da = da.sel(time=slice(INDEX_START, INDEX_END))
        cm = area_mean(da.values, da.lat.values, m)
        n = ccfg.N_WINDOW_FRAMES
        cs = np.concatenate([[0], np.cumsum(cm)])
        roll = (cs[n:] - cs[:-n]) / n
        et = pd.DatetimeIndex(da.time.values[n - 1:])
        k = (et.hour == 0) & (et >= DAILY_START) & (et <= DAILY_END)
        out["a_l_" + name] = pd.Series(np.round(roll[k], 4), index=et[k])
    df = pd.DataFrame(out)
    df.index.name = "date"
    if df.isna().any().any():
        raise SystemExit("NaN in the daily HRRR series")
    return df


def _daily_csv() -> Path:
    if not DAILY_CSV.exists():
        df = _daily_table()
        tmp = DAILY_CSV.with_name(DAILY_CSV.name + ".tmp")
        df.reset_index().assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_csv(
            tmp, index=False)
        os.replace(tmp, DAILY_CSV)
    return DAILY_CSV


# --------------------------------------------------------------------------- #
# Stage: cases
# --------------------------------------------------------------------------- #
def case_table() -> pd.DataFrame:
    ep = pd.read_csv(EPISODES)
    mask = get_truth("hrrr").mask().values
    era = era5_index()
    rows = []
    for r in ep.itertuples():
        row = dict(episode_id=r.episode_id, family=r.family, rung=r.rung, peak=r.peak,
                   a_l_conus=r.a_l_conus)
        for w in ("13f", "12f"):
            e = era.sel(time=window_times(r.peak, w)).mean("time")
            row[f"era5_full_{w}"] = float(area_mean(e.values, e.lat.values))
            row[f"era5_mask_{w}"] = float(area_mean(e.values, e.lat.values, mask))
            for name in TRUTH_NAMES:
                row[f"{name}_{w}"] = get_truth(name).obs_al(r.episode_id, w)
        rows.append(row)
    df = pd.DataFrame(rows)
    for w in ("13f", "12f"):
        for name in TRUTH_NAMES:
            df[f"d_{name}_{w}"] = df[f"{name}_{w}"] - df[f"era5_mask_{w}"]
        for name in ("era5_full", "era5_mask") + TRUTH_NAMES:
            df[f"lt2_{name}_{w}"] = df[f"{name}_{w}"].abs() < 2.0
            sgn = np.where(df.family == "heat", 1.0, -1.0)
            df[f"lt2dir_{name}_{w}"] = sgn * df[f"{name}_{w}"] < 2.0
    names = ("era5_full", "era5_mask") + TRUTH_NAMES
    cols = ["episode_id", "family", "rung", "peak", "a_l_conus"]
    for w in ("13f", "12f"):
        cols += [f"{n}_{w}" for n in names] + [f"d_{n}_{w}" for n in TRUTH_NAMES]
    for w in ("13f", "12f"):
        cols += [f"{f}_{n}_{w}" for f in ("lt2", "lt2dir") for n in names]
    return df[cols]


def summarize(df: pd.DataFrame) -> dict:
    s = {"n_cases": int(len(df))}
    for w in ("13f", "12f"):
        for name in TRUTH_NAMES:
            d = df[f"d_{name}_{w}"]
            s[f"{name}_{w}"] = dict(
                corr_vs_era5_mask=float(np.corrcoef(df[f"{name}_{w}"], df[f"era5_mask_{w}"])[0, 1]),
                corr_vs_era5_full=float(np.corrcoef(df[f"{name}_{w}"], df[f"era5_full_{w}"])[0, 1]),
                spearman_vs_era5_mask=float(df[f"{name}_{w}"].rank().corr(df[f"era5_mask_{w}"].rank())),
                mean_offset_vs_era5_mask=float(d.mean()), sd_offset=float(d.std()),
                min_offset=float(d.min()), max_offset=float(d.max()),
                mean_offset_vs_era5_full=float((df[f"{name}_{w}"] - df[f"era5_full_{w}"]).mean()),
                n_below_2K=int(df[f"lt2_{name}_{w}"].sum()),
                cases_below_2K=df.episode_id[df[f"lt2_{name}_{w}"]].tolist())
        for name in ("era5_full", "era5_mask"):
            s[f"{name}_{w}"] = dict(n_below_2K=int(df[f"lt2_{name}_{w}"].sum()),
                                    cases_below_2K=df.episode_id[df[f"lt2_{name}_{w}"]].tolist())
    s["era5_full_13f_vs_catalog_maxabs"] = float((df.era5_full_13f - df.a_l_conus).abs().max())
    return s


def stage_cases() -> None:
    df = case_table()
    if len(df) != 42:
        raise SystemExit(f"{len(df)} cases, expected 42")
    s = summarize(df)
    if s["era5_full_13f_vs_catalog_maxabs"] > 1e-3:
        raise SystemExit(f"ERA5 full-box A_L does not reproduce the catalog "
                         f"({s['era5_full_13f_vs_catalog_maxabs']:.4f} K)")
    for p, write in ((CASES_CSV, lambda tmp: df.round(4).to_csv(tmp, index=False)),
                     (CASES_CSV.with_name("cases_hrrr_summary.json"),
                      lambda tmp: tmp.write_text(json.dumps(s, indent=1)))):
        tmp = p.with_name(p.name + ".tmp")
        write(tmp)
        os.replace(tmp, p)
    _daily_csv()
    for name in TRUTH_NAMES:
        x = s[f"{name}_13f"]
        print(f"[cases] {name:8s} 13f: corr vs ERA5(mask) {x['corr_vs_era5_mask']:.4f}, "
              f"mean offset {x['mean_offset_vs_era5_mask']:+.3f} K (sd {x['sd_offset']:.3f}), "
              f"|A_L|<2K: {x['n_below_2K']}")
    print(f"[cases] wrote {CASES_CSV} ({len(df)} rows) and {DAILY_CSV.name}")


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("fetch2026", "build", "cases"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "fetch2026":
        stage_fetch2026(a.workers, a.force)
    elif a.stage == "build":
        stage_build(a.workers, a.force)
    else:
        stage_cases()


if __name__ == "__main__":
    main()
