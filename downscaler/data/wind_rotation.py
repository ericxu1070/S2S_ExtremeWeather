"""Rotate HRRR winds from grid-relative to Earth-relative.

WHY THIS EXISTS
---------------
`hrrr_grid_reference.nc` says, in its own global attributes:

    uv_relative_to_grid = 1
    "U/V wind components are relative to the grid (not Earth)."

So HRRR's `u10`/`v10` are components along the **Lambert Conformal grid axes**, not along
east/north. ERA5's winds are Earth-relative. They are different vector decompositions of the
same physical wind, and they differ by the projection's convergence angle, which is 0 on the
central meridian (262.5degE) and grows to ~+-23deg at the east/west edges of CONUS.

If you ignore this, two bad things happen:
  * verification is wrong  — you score a projection artifact as if it were model error;
  * training is wasted     — the model has to spend capacity silently learning a fixed,
    position-dependent rotation instead of learning downscaling.

We rotate HRRR **into** Earth-relative (rather than rotating ERA5 into grid space) so that
the model's *outputs* are Earth-relative too. That is the standard convention and keeps them
directly comparable to ERA5, GenCast, and anything else downstream.

THE ANGLE
---------
theta(y, x) is the bearing of the grid's +y axis measured clockwise from true north. Given
it, the grid basis expressed in (east, north) is

    e_y = ( sin th,  cos th)          # "grid north"
    e_x = ( cos th, -sin th)          # "grid east"  (perpendicular; the projection is conformal)

and a grid-relative wind V = u_g*e_x + v_g*e_y has Earth-relative components

    u_e =  u_g*cos th + v_g*sin th
    v_e = -u_g*sin th + v_g*cos th

We do NOT hardcode theta from the projection formula. Sign conventions for Lambert
convergence are a classic source of silent, plausible-looking errors, so we instead measure
theta directly from the grid's own 2-D latitude/longitude arrays by finite-differencing
along +y. That is ground truth by construction: it asks "which way on Earth does the array's
+y axis actually point?". `lcc_convergence_angle()` reproduces the analytic value and exists
only so the two can be cross-checked (see tests/test_wind_rotation.py).
"""

import numpy as np


def grid_north_angle(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bearing of the grid +y axis, clockwise from true north, in radians. Shape (H, W).

    Measured from the grid geometry itself: step one cell in +y and see which way that
    moves you on the sphere. Longitude differences are wrapped to (-180, 180] so a grid
    crossing the dateline or the 0/360 seam cannot produce a spurious ~360deg jump.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    dlat = np.gradient(lat, axis=0)                       # deg of latitude per +y cell
    dlon = (np.gradient(lon, axis=0) + 180.0) % 360.0 - 180.0   # deg of longitude per +y cell

    # Convert the (dlat, dlon) step into a local (east, north) displacement. The cos(lat)
    # factor is what makes a degree of longitude shorter as you move away from the equator.
    east = np.cos(np.deg2rad(lat)) * dlon
    north = dlat
    return np.arctan2(east, north)


def lcc_convergence_angle(lon: np.ndarray, lon_0: float, lat_1: float, lat_2: float) -> np.ndarray:
    """Analytic Lambert Conformal convergence angle, radians. Cross-check only.

    theta = n * (lon - lon_0), where n is the cone constant:
        tangent case  (lat_1 == lat_2):  n = sin(lat_1)
        secant  case  (lat_1 != lat_2):  n = ln(cos p1 / cos p2) / ln(tan(pi/4 + p2/2) / tan(pi/4 + p1/2))
    """
    p1, p2 = np.deg2rad(lat_1), np.deg2rad(lat_2)
    if abs(lat_1 - lat_2) < 1e-9:
        n = np.sin(p1)
    else:
        n = np.log(np.cos(p1) / np.cos(p2)) / np.log(
            np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2)
        )
    dlon = (np.asarray(lon, dtype=np.float64) - lon_0 + 180.0) % 360.0 - 180.0
    return n * np.deg2rad(dlon)


def rotation_terms(lat: np.ndarray, lon: np.ndarray):
    """Precompute (cos theta, sin theta) once for a grid -> float32 (H, W) each."""
    theta = grid_north_angle(lat, lon)
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


def grid_to_earth(u_grid, v_grid, cos_t, sin_t):
    """Grid-relative -> Earth-relative wind components. Works for numpy or torch."""
    u_east = u_grid * cos_t + v_grid * sin_t
    v_north = -u_grid * sin_t + v_grid * cos_t
    return u_east, v_north


def earth_to_grid(u_east, v_north, cos_t, sin_t):
    """Earth-relative -> grid-relative wind components (the exact inverse)."""
    u_grid = u_east * cos_t - v_north * sin_t
    v_grid = u_east * sin_t + v_north * cos_t
    return u_grid, v_grid
