"""HRRR observed overlay truth: week-mean T2m anomaly on the ERA5 CONUS grid.

Mirrors :func:`gencast_s2s.data.true_verif_anom` exactly — the 7-day verification
window ending at the event peak, ``obs.mean(time) - clim.mean(time)`` — but sources 2m
temperature from the local HRRR v3 shard archive (3 km CONUS, 2015-2025, 6-hourly)
instead of ERA5. The result is regridded (nearest-neighbour) onto the same 0.25deg
CONUS grid as the ERA5 truth and written to
``runs/observations/{event}_hrrr_verif_t2m_anom.nc`` so the plotting code can overlay
an "Observed (HRRR)" curve next to the ERA5 truth.

Each shard is a per-timestamp ``.npz`` with a 41-channel ``prog_t`` stack on the native
1059x1799 Lambert grid; t2m is channel ``C.HRRR_T2M_CHANNEL`` (37). Anomalies reuse the
ERA5 1990-2019 climatology (there is no offline HRRR-native climatology), so a small
HRRR-vs-ERA5 model bias is folded into the overlay — the patterns still match ERA5
closely (spatial correlation ~0.8 on the events tested), with HRRR resolving sharper
local extremes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from . import data as D


# --------------------------------------------------------------------------- #
# Native HRRR grid -> ERA5 CONUS grid nearest-neighbour remap (cached on disk).
# --------------------------------------------------------------------------- #
def _xyz(lat, lon):
    """Lat/lon (deg) -> unit vectors on the sphere, for great-circle nearest-neighbour."""
    la, lo = np.deg2rad(np.asarray(lat)), np.deg2rad(np.asarray(lon))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _target_grid() -> tuple[np.ndarray, np.ndarray]:
    """The ERA5 CONUS (lat, lon) axes the truth/forecasts live on (from the climatology)."""
    clim = D._clim_src()
    return clim["lat"].values, clim["lon"].values


def _remap_index() -> tuple[np.ndarray, tuple[int, int]]:
    """For each ERA5 CONUS cell, the flat index of the nearest native HRRR cell.

    Depends only on the two grids, so it is computed once and cached to
    ``C.HRRR_REMAP_CACHE``. Returns ``(idx, (nlat, nlon))``.
    """
    tlat, tlon = _target_grid()
    shape = (tlat.size, tlon.size)
    if C.HRRR_REMAP_CACHE.exists():
        idx = np.load(C.HRRR_REMAP_CACHE)
        if idx.size == tlat.size * tlon.size:
            return idx, shape

    from scipy.spatial import cKDTree
    g = xr.open_dataset(C.HRRR_GRID_REF)
    src = _xyz(g["latitude"].values.ravel(), g["longitude"].values.ravel())
    TLA, TLO = np.meshgrid(tlat, tlon, indexing="ij")
    _, idx = cKDTree(src).query(_xyz(TLA.ravel(), TLO.ravel()), k=1)
    idx = idx.astype(np.int64)
    C.HRRR_REMAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(C.HRRR_REMAP_CACHE, idx)
    return idx, shape


# --------------------------------------------------------------------------- #
# Shard IO
# --------------------------------------------------------------------------- #
def _shard_path(t: pd.Timestamp):
    return C.HRRR_SHARDS / f"{t.year}" / f"{t.strftime('%Y%m%dT%H%M%S')}.npz"


def _load_t2m(t: pd.Timestamp, idx: np.ndarray, shape: tuple[int, int]):
    """Native HRRR t2m at time ``t`` remapped onto the ERA5 CONUS grid, or None if absent."""
    f = _shard_path(t)
    if not f.exists():
        return None
    with np.load(f) as z:
        flat = z["prog_t"][C.HRRR_T2M_CHANNEL].ravel()
    return flat[idx].reshape(shape)


# --------------------------------------------------------------------------- #
# Week-mean anomaly (same window/definition as data.true_verif_anom)
# --------------------------------------------------------------------------- #
def week_mean_t2m_anom(peak) -> xr.DataArray | None:
    """Observed HRRR week-mean T2m anomaly (K) over CONUS for the event peaking at ``peak``.

    Window = the 7 days ending at the peak (6-hourly), identical to the ERA5 truth.
    Returns a ``(lat, lon)`` DataArray on the ERA5 CONUS grid, or None if no shards in
    the window are available.
    """
    peak = pd.Timestamp(peak)
    win = pd.date_range(peak - pd.Timedelta(days=6), peak, freq="6h")
    idx, shape = _remap_index()

    fields, have = [], []
    for t in win:
        f = _load_t2m(t, idx, shape)
        if f is not None:
            fields.append(f)
            have.append(t)
    if not fields:
        return None

    hrrr_mean = np.mean(fields, axis=0)                       # (lat, lon) K
    # Climatology averaged over exactly the timestamps we actually used (matches the
    # ERA5 truth recipe; robust to any missing HRRR shards in the window).
    clim_mean = D.clim_for(pd.DatetimeIndex(have)).mean("time")
    tlat, tlon = _target_grid()
    anom = hrrr_mean - clim_mean.values
    return xr.DataArray(anom, dims=("lat", "lon"),
                        coords={"lat": tlat, "lon": tlon}, name="t2m_anom")


def build_hrrr_truth(events: dict | None = None, overwrite: bool = False) -> list[str]:
    """Precompute + cache HRRR week-mean anomaly per event -> runs/observations/.

    Mirrors download.prepare_truth but for HRRR and fully offline. Events with no shard
    coverage in their verification window are skipped (logged). Returns the names written.
    """
    C.ensure_dirs()
    events = C.EVENTS if events is None else events
    written = []
    for name, peak in events.items():
        f = C.OBS_DIR / f"{name}_hrrr_verif_t2m_anom.nc"
        if f.exists() and not overwrite:
            print(f"[hrrr-truth] cached: {f.name}")
            written.append(name)
            continue
        da = week_mean_t2m_anom(peak)
        if da is None:
            print(f"[hrrr-truth] SKIP {name}: no HRRR shards in {peak} verification window")
            continue
        D._atomic_to_netcdf(da.to_dataset(name="t2m_anom"), f)
        print(f"[hrrr-truth] {name}: mean={float(da.mean()):+.2f}K "
              f"max={float(da.max()):+.2f}K min={float(da.min()):+.2f}K -> {f.name}")
        written.append(name)
    return written


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Build HRRR observed verification truth.")
    p.add_argument("--overwrite", action="store_true", help="rebuild even if cached")
    p.add_argument("--events", default=None,
                   help="comma-separated event names (default: all OOS events)")
    args = p.parse_args(argv)
    ev = None
    if args.events:
        want = [s.strip() for s in args.events.split(",") if s.strip()]
        ev = {k: C.EVENTS[k] for k in want if k in C.EVENTS}
    build_hrrr_truth(events=ev, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
