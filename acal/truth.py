#!/usr/bin/env python
"""Verification truths for the multi-model board: ERA5 here, HRRR in `acal.hrrrtruth`.

Every forecast source on the board is scored against a `Truth`. A truth is four things:

    name                'era5' | 'hrrr' | 'hrrr_raw'
    mask()              bool (lat, lon) on the 0.25 deg acal grid, or None (= every cell).
                        HRRR does not cover the whole CONUS box, so its mask is applied to
                        the truth AND to every forecast before any area mean.
    frames(eid)         anomaly (time, lat, lon) K: the 13 frames at 00Z/12Z over
                        [peak-6d 00Z, peak 00Z], NaN outside the mask
    obs_al(eid, window) cos-lat area mean over the mask of the window mean:
                        '13f' = all 13 frames (instantaneous sources, the published A_L);
                        '12f' = peak-6d 00Z .. peak-1d 12Z (the daily-mean sources, whose
                        last UTC day is peak-1; ruling C9)
    daily_conus()       daily CONUS A_L series 2021-01-01..2026-08-31 (13-frame mean ending
                        at each 00Z, cos-lat over the mask)
    index()             (maps only) the whole 12-hourly per-cell anomaly, NaN off the mask

The P_clim pool is `pool_series(truth)` = `daily_conus()` clipped to 2021-2025, the
published catalog's span: switching the truth changes the pool's values, never its days.

ERA5 is the published truth: `obs_al(eid, '13f')` is asserted equal to the slate's
`a_l_conus` (to the slate's 3 decimals) and `pool_series(era5)` IS the catalog CSV that
`analyze` and `cfsbase` already pool, so every number scored against `get_truth('era5')`
is the number already published.

    get_truth('era5')            -> ERA5Truth
    get_truth('hrrr' | 'hrrr_raw') -> imported lazily from acal.hrrrtruth (lane B)

Where the truth is threaded (each takes `--truth`; default = the published ERA5 run):
    analyze --stage scorecard|rungs, cfsbase --stage score|paired|all, s2sbase --stage
    score|paired|maps, roc --source, reach, maps --stage fields|figures|cfs, sidebyside.
    A masked truth re-reduces the AI+RES walkers (aires_al_windows.csv al13_<tag>/al12_<tag>,
    `s2sbase --stage aires_mask`) and every source's members on the mask. Tables go to
    runs/acal/analysis/s2s/<truth>/ (`analysis_dir`), figures to `fig_dir` (figures/acal/hrrr/
    for 'hrrr', figures/acal/hrrr/raw/ for 'hrrr_raw').
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import aprep, ccfg

INDEX_DIR = ccfg.ACAL_ROOT / "index"
WINDOWS = ("13f", "12f")
GRID = (105, 237)
CATALOG_TOL = 1e-3                     # K; the slate stores a_l_conus to 3 decimals
# The P_clim pool's days: the published ERA5 catalog's span. Every truth's daily series is
# clipped to it (`pool_series`), so switching the truth changes the pool's VALUES, never
# its days - and ERA5 reproduces the published pool exactly.
POOL_START, POOL_END = pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")
# lane B's daily table: column a_l_era5_full == the catalog on 2021-2025 and extends it
# through 2026-08-31 (ERA5 full box, same 13-frame definition)
ERA5_DAILY_EXT = ccfg.ACAL_ROOT / "index_hrrr" / "conus_daily_hrrr_2021_2026.csv"


# --------------------------------------------------------------------------- #
# Window bookkeeping, shared with acal.s2sbase
# --------------------------------------------------------------------------- #
def peak_of(eid: str) -> pd.Timestamp:
    return pd.Timestamp(_slate().loc[eid, "peak"])


@functools.lru_cache(maxsize=1)
def _slate() -> pd.DataFrame:
    return aprep.episodes().set_index("episode_id")


def window_times(peak, window: str = "13f") -> pd.DatetimeIndex:
    """The 12-hourly verification frames of `window` for a peak at 00Z."""
    peak = pd.Timestamp(peak)
    if peak != peak.normalize():
        raise ValueError(f"peak {peak} is not at 00Z")
    t = pd.date_range(peak - pd.Timedelta(days=6), peak, freq="12h")      # 13 frames
    if window == "13f":
        return t
    if window == "12f":
        return t[:-1]
    raise ValueError(f"unknown window {window!r}; want one of {WINDOWS}")


def area_mean(da: xr.DataArray, mask: xr.DataArray | None = None) -> xr.DataArray:
    """cos(lat)-weighted mean over (lat, lon), NaN cells (outside `mask`) left out.

    `xarray.weighted` drops a NaN cell's weight from the denominator, so a masked field
    averages over the mask only, and an all-True mask is bit-identical to no mask.
    """
    if mask is not None:
        da = apply_mask(da, mask)
    return da.weighted(np.cos(np.deg2rad(da["lat"]))).mean(("lat", "lon"))


def apply_mask(da: xr.DataArray, mask: xr.DataArray | None) -> xr.DataArray:
    """`da` with NaN outside a bool (lat, lon) mask on the same grid (checked, not aligned)."""
    if mask is None:
        return da
    if mask.dtype != bool:
        raise TypeError(f"mask must be bool, got {mask.dtype}")
    for k in ("lat", "lon"):
        if mask.sizes[k] != da.sizes[k] or not np.allclose(mask[k].values, da[k].values,
                                                           atol=1e-6):
            raise ValueError(f"mask {k} grid != field {k} grid")
    m = xr.DataArray(mask.transpose("lat", "lon").values, dims=("lat", "lon"),
                     coords=dict(lat=da["lat"], lon=da["lon"]))
    return da.where(m)


# --------------------------------------------------------------------------- #
# Truth providers
# --------------------------------------------------------------------------- #
class Truth:
    """Interface; subclasses fill `frames` and `daily_conus` (and `index` for the maps)."""
    name: str = "?"
    label: str = "?"

    def mask(self) -> xr.DataArray | None:
        return None

    def frames(self, eid: str) -> xr.DataArray:
        raise NotImplementedError

    def obs_al(self, eid: str, window: str = "13f") -> float:
        f = self.frames(eid)
        f = f.sel(time=window_times(peak_of(eid), window)).astype("float64")
        # float64 frames AND lat weights: a float32 reduction is up to 3e-5 K off, enough
        # to move a Gaussian P(obs) by > 1e-6 (HRRR's own obs_al is float64 already)
        f = f.assign_coords(lat=f["lat"].astype("float64"))
        return float(area_mean(f.mean("time"), self.mask()))

    def daily_conus(self) -> pd.Series:
        raise NotImplementedError

    def index(self) -> xr.DataArray:
        """The whole 12-hourly per-cell anomaly series (time, lat, lon), NaN outside the
        mask: the per-cell truth and climatology of the maps."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Truth {self.name}>"


class ERA5Truth(Truth):
    """ARCO-ERA5 minus the WB2 1990-2019 climatology, from the acal 12-hourly index cubes."""
    name = "era5"
    label = "ERA5"

    def frames(self, eid: str, window: str = "13f") -> xr.DataArray:
        f = _era5_frames(eid).copy()
        return f if window == "13f" else f.sel(time=window_times(peak_of(eid), window))

    def obs_al(self, eid: str, window: str = "13f") -> float:
        v = super().obs_al(eid, window)
        if window == "13f":
            cat = float(_slate().loc[eid, "a_l_conus"])
            if abs(v - cat) > CATALOG_TOL:
                raise RuntimeError(f"[truth] {eid}: ERA5 13-frame A_L {v:+.4f} != slate "
                                   f"a_l_conus {cat:+.4f}")
        return v

    def daily_conus(self) -> pd.Series:
        """The catalog series (2021-2025, what the published P_clim pools), extended through
        2026-08-31 from lane B's ERA5 full-box column when that table exists (checked equal
        to the catalog on every overlapping day). The pool only ever sees 2021-2025."""
        from acal import analyze as AN
        cat = AN.load_daily()
        if not ERA5_DAILY_EXT.exists():
            return cat
        ext = pd.read_csv(ERA5_DAILY_EXT, parse_dates=["date"]).set_index("date")["a_l_era5_full"]
        both = ext.reindex(cat.index)
        if both.isna().any() or np.abs(both.values - cat.values).max() > 1e-4:
            raise RuntimeError(f"[truth] {ERA5_DAILY_EXT.name} a_l_era5_full != the catalog")
        tail = ext[ext.index > cat.index.max()]
        return pd.concat([cat, pd.Series(tail.values, index=tail.index, name=cat.name)])

    def index(self) -> xr.DataArray:
        from acal import maps as M
        return M.era5_index()


def pool_series(truth: Truth) -> pd.Series:
    """The P_clim pool input for `truth`: its daily CONUS A_L on POOL_START..POOL_END.
    For ERA5 this IS `analyze.load_daily()` (the published pool), value for value."""
    d = truth.daily_conus()
    return d[(d.index >= POOL_START) & (d.index <= POOL_END)]


def index_of(truth) -> xr.DataArray:
    """`truth.index()`, or for lane B's duck-typed HRRR provider its index variable."""
    if hasattr(truth, "index"):
        return truth.index()
    if str(truth.name).startswith("hrrr"):
        from acal import hrrrtruth  # noqa: PLC0415
        da = hrrrtruth._index(truth.var).load()
        return da.where(truth.mask())
    raise AttributeError(f"{truth!r} exposes no per-cell index")


def frames_on(truth, eid: str, window: str = "13f") -> xr.DataArray:
    """The truth's frames on `window` (13 frames for '13f', the first 12 for '12f')."""
    f = truth.frames(eid)
    return f.sel(time=window_times(peak_of(eid), window))


def mask_tag(truth) -> str:
    """'' for an unmasked truth, else '_' + the mask's name ('_hrrrmask')."""
    return "" if truth.mask() is None else "_" + getattr(truth, "mask_name", "hrrrmask")


def analysis_dir(truth) -> Path:
    """Where truth-switched tables go: runs/acal/analysis/s2s/<truth>/ (ERA5 included, so a
    regenerated ERA5 table never overwrites the published one)."""
    return ccfg.ACAL_ROOT / "analysis" / "s2s" / truth.name


def fig_dir(truth) -> Path:
    """figures/acal/hrrr/ ('hrrr'), figures/acal/hrrr/raw/ ('hrrr_raw'),
    figures/acal/s2s/<name>/ otherwise (ERA5 regenerations included)."""
    n = truth.name
    if n == "hrrr":
        return ccfg.FIG_ROOT / "hrrr"
    if n.startswith("hrrr_"):
        return ccfg.FIG_ROOT / "hrrr" / n[len("hrrr_"):]
    return ccfg.FIG_ROOT / "s2s" / n


def label_of(truth) -> str:
    return getattr(truth, "label", truth.name)


@functools.lru_cache(maxsize=64)
def _era5_frames(eid: str) -> xr.DataArray:
    peak = peak_of(eid)
    want = window_times(peak, "13f")
    parts = []
    for y in sorted({want[0].year, want[-1].year}):
        p = INDEX_DIR / f"era5_t2m_anom_12h_{y}.nc"
        with xr.open_dataset(p) as d:
            parts.append(d["t2m_anom"].sel(time=slice(want[0], want[-1])).load())
    w = xr.concat(parts, dim="time")
    t = pd.DatetimeIndex(w["time"].values)
    w = w.isel(time=~t.duplicated(keep="first")).sortby("time")
    if not pd.DatetimeIndex(w["time"].values).equals(want):
        raise RuntimeError(f"[truth] {eid}: ERA5 index frames != the 13 window frames")
    if (w.sizes["lat"], w.sizes["lon"]) != GRID:
        raise RuntimeError(f"[truth] ERA5 index grid {w.sizes} is not {GRID}")
    return w


def get_truth(name: str = "era5") -> Truth:
    """The truth provider called `name`. HRRR providers come from `acal.hrrrtruth`."""
    if name == "era5":
        return ERA5Truth()
    if name.startswith("hrrr"):
        from acal import hrrrtruth  # noqa: PLC0415 - lazy: lane B's module, heavy deps
        if hasattr(hrrrtruth, "get_truth"):
            return hrrrtruth.get_truth(name)
        for cls in ("HRRRTruth", "HrrrTruth"):
            if hasattr(hrrrtruth, cls):
                return getattr(hrrrtruth, cls)(name)
        raise AttributeError("acal.hrrrtruth exposes neither get_truth() nor HRRRTruth")
    raise KeyError(f"unknown truth {name!r}; want 'era5', 'hrrr' or 'hrrr_raw'")
