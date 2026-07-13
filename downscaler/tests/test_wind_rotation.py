"""Tests for the HRRR grid-relative -> Earth-relative wind rotation.

The rotation is easy to get subtly wrong (a sign flip still "looks like" a rotation and
still preserves wind speed), so these tests pin down the sign against the real HRRR grid
rather than only checking self-consistency.
"""

import os

import numpy as np
import pytest

from data.wind_rotation import (
    earth_to_grid,
    grid_north_angle,
    grid_to_earth,
    lcc_convergence_angle,
    rotation_terms,
)

GRID_REF = "/home/ubuntu/Vayuh/data/moein/hrrr_work/hrrr_grid_reference.nc"


def test_zero_angle_is_identity():
    ue, vn = grid_to_earth(3.0, -4.0, np.cos(0.0), np.sin(0.0))
    assert ue == pytest.approx(3.0)
    assert vn == pytest.approx(-4.0)


def test_round_trip_is_exact():
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=(64, 64)), rng.normal(size=(64, 64))
    theta = rng.uniform(-0.4, 0.4, size=(64, 64))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    u2, v2 = earth_to_grid(*grid_to_earth(u, v, cos_t, sin_t), cos_t, sin_t)
    assert np.allclose(u, u2, atol=1e-10)
    assert np.allclose(v, v2, atol=1e-10)


def test_rotation_preserves_wind_speed():
    rng = np.random.default_rng(1)
    u, v = rng.normal(size=(32, 32)), rng.normal(size=(32, 32))
    theta = rng.uniform(-np.pi, np.pi, size=(32, 32))
    ue, vn = grid_to_earth(u, v, np.cos(theta), np.sin(theta))
    assert np.allclose(np.hypot(u, v), np.hypot(ue, vn), atol=1e-10)


def test_known_sign_convention():
    """A wind blowing along grid +y (v_g=1) at a point where grid north is rotated 30 deg
    toward the east must come out as a wind with a positive EAST component."""
    theta = np.deg2rad(30.0)
    ue, vn = grid_to_earth(0.0, 1.0, np.cos(theta), np.sin(theta))
    assert ue == pytest.approx(np.sin(theta))   # +0.5, i.e. eastward
    assert vn == pytest.approx(np.cos(theta))   # +0.866, i.e. northward
    assert ue > 0


@pytest.mark.skipif(not os.path.exists(GRID_REF), reason="HRRR grid reference not available")
def test_numeric_angle_matches_analytic_lambert_on_real_grid():
    """The angle measured from the grid's own lat/lon must reproduce the analytic Lambert
    convergence angle. These are two independent derivations; agreement pins the sign."""
    import netCDF4

    ds = netCDF4.Dataset(GRID_REF, "r")
    lat = np.asarray(ds.variables["latitude"][:], np.float64)
    lon = np.asarray(ds.variables["longitude"][:], np.float64)
    attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
    ds.close()

    numeric = grid_north_angle(lat, lon)
    analytic = lcc_convergence_angle(
        lon,
        attrs["central_longitude_deg"],
        attrs["standard_parallel_1_deg"],
        attrs["standard_parallel_2_deg"],
    )
    # Finite differencing is only 2nd-order accurate and degrades on the boundary rows,
    # so allow a hundredth of a degree.
    assert np.abs(np.rad2deg(numeric - analytic)).max() < 0.05

    # The rotation is NOT negligible: it must reach ~+-23 deg at the edges of CONUS.
    assert np.rad2deg(np.abs(numeric)).max() > 20.0

    # ...and must vanish on the central meridian, where grid north IS true north.
    on_meridian = np.abs(lon - attrs["central_longitude_deg"]) < 0.05
    assert np.abs(np.rad2deg(numeric[on_meridian])).max() < 0.05


@pytest.mark.skipif(not os.path.exists(GRID_REF), reason="HRRR grid reference not available")
def test_rotation_terms_shape_and_dtype():
    import netCDF4

    ds = netCDF4.Dataset(GRID_REF, "r")
    lat = np.asarray(ds.variables["latitude"][:], np.float64)
    lon = np.asarray(ds.variables["longitude"][:], np.float64)
    ds.close()

    cos_t, sin_t = rotation_terms(lat, lon)
    assert cos_t.shape == lat.shape and sin_t.shape == lat.shape
    assert cos_t.dtype == np.float32 and sin_t.dtype == np.float32
    assert np.allclose(cos_t**2 + sin_t**2, 1.0, atol=1e-5)
