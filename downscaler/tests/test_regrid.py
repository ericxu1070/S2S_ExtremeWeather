"""CachedBilinearRegridder must match the scipy path it replaced, exactly.

Every ERA5 conditioning channel of every sample goes through this regridder, so a silent
mismatch would corrupt the model's entire input. These tests pin it to
RegularGridInterpolator(method="linear", bounds_error=False, fill_value=None) — the
implementation regrid_era5 used before the weights were cached — including the linear
extrapolation outside the source grid that fill_value=None implies (the HRRR grid pokes
past the edges of a CONUS-cropped ERA5 file).
"""

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from data.era5_hrrr_dataset import CachedBilinearRegridder

RNG = np.random.default_rng(0)


def _reference(lat, lon, field, pts):
    interp = RegularGridInterpolator(
        (lat, lon), field, method="linear", bounds_error=False, fill_value=None
    )
    return interp(pts)


def _grid(ny=19, nx=37, uniform=True):
    if uniform:
        lat = np.linspace(20.0, 55.0, ny)
        lon = np.linspace(230.0, 300.0, nx)
    else:
        lat = np.sort(RNG.uniform(20.0, 55.0, ny))
        lon = np.sort(RNG.uniform(230.0, 300.0, nx))
    return lat, lon


def test_matches_scipy_interior():
    lat, lon = _grid()
    field = RNG.normal(size=(lat.size, lon.size))
    pts = np.stack(
        [RNG.uniform(lat[0], lat[-1], 5000), RNG.uniform(lon[0], lon[-1], 5000)], axis=-1
    )
    rg = CachedBilinearRegridder(lat, lon, pts)
    np.testing.assert_allclose(rg(field), _reference(lat, lon, field, pts), rtol=1e-12, atol=1e-14)


def test_matches_scipy_extrapolation():
    """Points beyond every edge — fill_value=None extrapolates from the edge cell."""
    lat, lon = _grid()
    field = RNG.normal(size=(lat.size, lon.size))
    pts = np.stack(
        [RNG.uniform(lat[0] - 5.0, lat[-1] + 5.0, 5000),
         RNG.uniform(lon[0] - 5.0, lon[-1] + 5.0, 5000)],
        axis=-1,
    )
    rg = CachedBilinearRegridder(lat, lon, pts)
    np.testing.assert_allclose(rg(field), _reference(lat, lon, field, pts), rtol=1e-12, atol=1e-14)


def test_matches_scipy_on_knots_and_corners():
    lat, lon = _grid()
    field = RNG.normal(size=(lat.size, lon.size))
    knot_pts = np.stack(np.meshgrid(lat, lon, indexing="ij"), axis=-1).reshape(-1, 2)
    rg = CachedBilinearRegridder(lat, lon, knot_pts)
    # On the knots bilinear interpolation must reproduce the field itself.
    np.testing.assert_allclose(rg(field), field.ravel(), rtol=1e-12, atol=1e-14)


def test_matches_scipy_nonuniform_axes():
    lat, lon = _grid(uniform=False)
    field = RNG.normal(size=(lat.size, lon.size))
    pts = np.stack(
        [RNG.uniform(lat[0] - 2.0, lat[-1] + 2.0, 3000),
         RNG.uniform(lon[0] - 2.0, lon[-1] + 2.0, 3000)],
        axis=-1,
    )
    rg = CachedBilinearRegridder(lat, lon, pts)
    np.testing.assert_allclose(rg(field), _reference(lat, lon, field, pts), rtol=1e-12, atol=1e-14)


def test_weights_are_reused_across_fields():
    """One regridder, many fields — the whole point of the cache."""
    lat, lon = _grid()
    pts = np.stack(
        [RNG.uniform(lat[0], lat[-1], 1000), RNG.uniform(lon[0], lon[-1], 1000)], axis=-1
    )
    rg = CachedBilinearRegridder(lat, lon, pts)
    for _ in range(3):
        field = RNG.normal(size=(lat.size, lon.size))
        np.testing.assert_allclose(rg(field), _reference(lat, lon, field, pts), rtol=1e-12, atol=1e-14)


def test_matches_detects_grid_change():
    lat, lon = _grid()
    pts = np.stack([lat[:5], lon[:5]], axis=-1)
    rg = CachedBilinearRegridder(lat, lon, pts)
    assert rg.matches(lat, lon)
    assert not rg.matches(lat + 0.5, lon)
    assert not rg.matches(lat[:-1], lon)


def test_descending_lat_rejected():
    """regrid_era5 flips descending latitude BEFORE building the cache; the regridder
    itself must refuse a descending axis rather than silently mis-index."""
    lat, lon = _grid()
    pts = np.stack([lat[:5], lon[:5]], axis=-1)
    with pytest.raises(ValueError, match="ascending"):
        CachedBilinearRegridder(lat[::-1], lon, pts)


def test_float32_field_promoted():
    """ERA5 fields arrive float32; the gather must not lose precision vs scipy."""
    lat, lon = _grid()
    field = RNG.normal(size=(lat.size, lon.size)).astype(np.float32)
    pts = np.stack(
        [RNG.uniform(lat[0], lat[-1], 2000), RNG.uniform(lon[0], lon[-1], 2000)], axis=-1
    )
    rg = CachedBilinearRegridder(lat, lon, pts)
    ref = _reference(lat, lon, field.astype(np.float64), pts)
    np.testing.assert_allclose(rg(field), ref, rtol=1e-6)
