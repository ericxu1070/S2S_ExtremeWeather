"""Contract tests for the scalar observable.

``A_L`` is the number the whole experiment sorts walkers by, and three of the ways it can
go quietly wrong have already bitten this repo in one form or another:

* a box that hangs off the edge of the cube, which ``.sel`` answers with a partial mean
  and no warning (the same failure mode as the two climatology files with one name);
* a cube that does not actually span the 7-day verification window, which
  ``fc_times_in_window`` answers with however many frames it found;
* a refactor that changes what Gate 2 measured, since ``gate2.py`` now imports its
  reduction from here.

Each gets a test. No GPU, no network, no model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aires import aindex as AI

PEAK = pd.Timestamp("2021-06-28")
INIT = PEAK - pd.Timedelta(days=21)


def _cube(step_h: int, start, end, *, members: int | None = None, value=None,
          batch: bool = False) -> xr.Dataset:
    """A synthetic CONUS cube on the real 0.25 degree verification grid."""
    lat = np.arange(24.0, 50.0 + 1e-9, 0.25, dtype="float32")
    lon = np.arange(235.0, 294.0 + 1e-9, 0.25, dtype="float32")
    t = pd.date_range(start, end, freq=f"{step_h}h")
    shape = (t.size, lat.size, lon.size)
    dims = ("time", "lat", "lon")
    arr = np.full(shape, 300.0, dtype="float32") if value is None else value
    da = xr.DataArray(arr, dims=dims, coords={"time": t, "lat": lat, "lon": lon})
    if members is not None:
        da = da.expand_dims(member=np.arange(members)).copy()
    if batch:
        da = da.expand_dims(batch=[0]).copy()
    return xr.Dataset({"2m_temperature": da})


# --------------------------------------------------------------------------- #
# Boxes
# --------------------------------------------------------------------------- #
def test_every_registered_box_fits_inside_the_conus_crop():
    """A box that hangs off the cube would be silently clipped by ``.sel``."""
    for name, box in AI.EVENT_BOXES.items():
        assert AI.CONUS_LAT[0] <= box.lat[0] < box.lat[1] <= AI.CONUS_LAT[1], name
        assert AI.CONUS_LON[0] <= box.lon[0] < box.lon[1] <= AI.CONUS_LON[1], name


def test_p90_events_use_the_conus_index():
    """p90 cases are DEFINED as exceedances of the CONUS mean; a smaller box would score
    a different event than the one selected (p90/cases_meta.json)."""
    for name in ("p90_20231107", "p90_20240802", "p90_20251224"):
        assert AI.box_for(name) is AI.CONUS


def test_out_of_range_box_raises_rather_than_clipping():
    cube = _cube(12, INIT, PEAK)
    bad = AI.Box("too-far-north", (45.0, 60.0), (236.0, 242.0))
    with pytest.raises(ValueError, match="outside"):
        AI.area_mean(cube["2m_temperature"], bad)


def test_descending_latitude_is_rejected():
    cube = _cube(12, INIT, PEAK)
    flipped = cube["2m_temperature"].isel(lat=slice(None, None, -1))
    with pytest.raises(ValueError, match="ascend"):
        AI.area_mean(flipped, AI.box_for("PNW_HeatDome_2021"))


# --------------------------------------------------------------------------- #
# Area mean
# --------------------------------------------------------------------------- #
def test_area_mean_is_cos_lat_weighted_and_box_restricted():
    """A field that is 1 inside the PNW box and 0 outside must average to the box's
    cos-lat area fraction over CONUS, and to exactly 1 over the box."""
    cube = _cube(12, INIT, INIT + pd.Timedelta(hours=12))
    da = cube["2m_temperature"].isel(time=0) * 0.0
    box = AI.box_for("PNW_HeatDome_2021")
    inside = ((da["lat"] >= box.lat[0]) & (da["lat"] <= box.lat[1])
              & (da["lon"] >= box.lon[0]) & (da["lon"] <= box.lon[1]))
    da = da.where(~inside, 1.0)

    assert float(AI.area_mean(da, box)) == pytest.approx(1.0)

    w = np.cos(np.deg2rad(da["lat"]))
    expect = float((da * w).sum() / (w * xr.ones_like(da)).sum())
    # float32 cubes, ~25k grid points: the two summation orders differ in the 6th digit.
    assert float(AI.area_mean(da, None)) == pytest.approx(expect, rel=1e-4)


def test_box_none_matches_the_full_conus_box():
    """Gate 2 reduces with ``box=None``; that must equal the explicit CONUS box, or the
    published Gate 2 numbers and the Gate 3 secondary index would mean different things."""
    rng = np.random.default_rng(0)
    lat = np.arange(24.0, 50.0 + 1e-9, 0.25, dtype="float32")
    lon = np.arange(235.0, 294.0 + 1e-9, 0.25, dtype="float32")
    da = xr.DataArray(rng.normal(size=(lat.size, lon.size)).astype("float32"),
                      dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    assert float(AI.area_mean(da, None)) == pytest.approx(float(AI.area_mean(da, AI.CONUS)),
                                                          rel=1e-6)


def test_batch_axis_is_dropped():
    """A GenCast rollout leaves a size-1 ``batch`` axis on the walker diagnostics cube."""
    cube = _cube(12, INIT, PEAK, batch=True)
    out = AI.area_mean(cube["2m_temperature"], AI.CONUS)
    assert "batch" not in out.dims


# --------------------------------------------------------------------------- #
# Window coverage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("step_h,want", [(12, 13), (6, 25)])
def test_full_window_is_accepted_at_either_model_step(step_h, want):
    """13 twelve-hourly frames (a GenCast walker) or 25 six-hourly ones (an FCN3 score
    forecast) both cover [peak-6d, peak] exactly."""
    cube = _cube(step_h, INIT, PEAK)
    assert AI.check_window(cube, PEAK, "t2m_anom") == want


def test_short_cube_is_rejected():
    """A trajectory that stops at lead 15 d supplies ONE frame of the window and would
    otherwise produce a perfectly plausible number that is not A_L."""
    cube = _cube(12, INIT, PEAK - pd.Timedelta(days=6))
    with pytest.raises(ValueError, match="should hold 13 frames"):
        AI.check_window(cube, PEAK, "t2m_anom")


def test_window_check_reports_what_it_found():
    cube = _cube(12, INIT, PEAK - pd.Timedelta(days=3))
    with pytest.raises(ValueError) as e:
        AI.check_window(cube, PEAK, "t2m_anom", where="w07 trajectory")
    assert "w07 trajectory" in str(e.value)
