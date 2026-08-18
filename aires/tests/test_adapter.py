"""Contract tests for the GenCast -> FCN3 adapter.

The important one is ``test_direct_channels_are_exact``: 69 of FCN3's 72 channels are a
pure rename plus a latitude flip, so converting a GenCast-format ERA5 frame must
reproduce the independently-built FCN3 IC for the same timestamp to the bit. If that
ever stops holding, a name map or an axis orientation has drifted.

Tests needing run data skip cleanly where it is absent.
"""
from __future__ import annotations

import glob

import numpy as np
import pytest
import xarray as xr

from aires import aconfig as A
from aires import adapter as AD

EVENT = "PNW_HeatDome_2021"
GENCAST_IC = f"runs/xres/0p25/week3/inputs/{EVENT}_inputs.nc"
FCN3_IC = f"runs/fcn3/week3/ic/{EVENT}_ic.nc"


def _need(path):
    if not glob.glob(str(A.ROOT / path)):
        pytest.skip(f"missing run data: {path}")
    return A.ROOT / path


# --------------------------------------------------------------------------- #
# Channel contract
# --------------------------------------------------------------------------- #
def test_channel_list_shape():
    assert len(A.FCN3_VARIABLES) == 72
    assert len(set(A.FCN3_VARIABLES)) == 72
    assert A.FCN3_VARIABLES[:7] == A.SURFACE_VARS
    for pfx in A.LEVEL_PREFIXES:
        assert sum(1 for v in A.FCN3_VARIABLES if v.startswith(pfx) and v[1:].isdigit()) == 13


def test_channel_list_matches_earth2studio():
    """No-op outside the fcn3 env; the point is that it cannot silently disagree."""
    A.verify_against_package()


def test_channel_list_matches_disk():
    A.verify_against_ic(_need(FCN3_IC))


@pytest.mark.parametrize("name,expect", [
    ("t2m", ("t2m", None)), ("tcwv", ("tcwv", None)), ("u10m", ("u10m", None)),
    ("u850", ("u", 850)), ("z50", ("z", 50)), ("q1000", ("q", 1000)),
])
def test_split_channel(name, expect):
    assert AD.split_channel(name) == expect


@pytest.mark.parametrize("bad", ["u851", "x500", "w500", "t2", "geopotential"])
def test_split_channel_rejects_junk(bad):
    with pytest.raises(KeyError):
        AD.split_channel(bad)


# --------------------------------------------------------------------------- #
# Derivation maths, no data needed
# --------------------------------------------------------------------------- #
def test_wind_coef_is_scale_and_rotation():
    u, v = np.array([[3.0]]), np.array([[4.0]])
    # |c| = 2, arg(c) = +90 deg  =>  speed doubles, direction rotates onto (-v, u)
    u1, v1 = AD.apply_wind_coef(u, v, 2j)
    assert np.allclose([u1[0, 0], v1[0, 0]], [-8.0, 6.0])
    assert np.hypot(u1, v1)[0, 0] == pytest.approx(2 * np.hypot(3, 4))


def test_raw_tcwv_matches_analytic_constant_profile():
    """Constant q: integral is q * (p_bot - p_top) / g, exactly."""
    q = np.full((13, 2, 2), 0.004)
    got = AD.raw_tcwv(q, A.P13)
    want = 0.004 * (1000 - 50) * 100.0 / AD.G
    assert np.allclose(got, want, rtol=1e-12)


def test_apply_tcwv_weights_is_linear():
    rng = np.random.default_rng(0)
    q = rng.random((13, 4, 5))
    w = rng.random((4, 5, 13))
    assert np.allclose(AD.apply_tcwv_weights(q, w), np.einsum("xyl,lxy->xy", w, q))


# --------------------------------------------------------------------------- #
# End-to-end against real data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def converted():
    gc_path, fcn3_path = _need(GENCAST_IC), _need(FCN3_IC)
    cal = AD.load_calibration()
    if cal is None:
        pytest.skip("no calibration; run `python -m aires.calibrate --fit`")
    with xr.open_dataset(gc_path) as gc:
        ic = AD.gencast_to_fcn3(gc, calib=cal)
    truth = xr.open_dataset(fcn3_path)["ic"]
    return ic, truth


def test_grid_and_time_match_fcn3(converted):
    ic, truth = converted
    assert list(ic["variable"].values) == list(truth["variable"].values)
    assert np.array_equal(ic["lat"].values, truth["lat"].values)
    assert np.array_equal(ic["lon"].values, truth["lon"].values)
    assert ic["time"].values[0] == truth["time"].values[0]
    assert ic["lat"].values[0] > ic["lat"].values[-1], "FCN3 needs lat descending"


def test_direct_channels_are_exact(converted):
    """The 69 renamed channels must be bit-identical to the independently built IC."""
    ic, truth = converted
    a, b = ic.isel(time=0), truth.isel(time=0)
    bad = {}
    for name in A.FCN3_VARIABLES:
        if name in A.DERIVED_VARS:
            continue
        d = np.abs(a.sel(variable=name).values - b.sel(variable=name).values)
        if d.max() != 0.0:
            bad[name] = float(d.max())
    assert not bad, f"renamed channels differ from the reference IC: {bad}"


def test_derived_channels_meet_gate1(converted):
    ic, truth = converted
    a, b = ic.isel(time=0), truth.isel(time=0)
    w = np.cos(np.deg2rad(ic["lat"].values))[:, None]

    def stats(name):
        e = a.sel(variable=name).values - b.sel(variable=name).values
        rmse = np.sqrt((w * e**2).sum() / (w.sum() * e.shape[1]))
        bias = (w * e).sum() / (w.sum() * e.shape[1])
        return float(rmse), float(bias)

    for name in ("u100m", "v100m"):
        rmse, bias = stats(name)
        assert rmse <= A.GATE1["u100m_rmse_max"], f"{name} rmse {rmse}"
        assert abs(bias) <= A.GATE1["u100m_abs_bias_max"], f"{name} bias {bias}"

    rmse, bias = stats("tcwv")
    tvals = b.sel(variable="tcwv").values
    mean = float((w * tvals).sum() / (w.sum() * tvals.shape[1]))
    assert rmse / mean <= A.GATE1["tcwv_rel_rmse_max"]
    assert abs(bias) / mean <= A.GATE1["tcwv_abs_rel_bias_max"]


def test_conus_cropped_state_is_rejected():
    """The cached xres cubes are CONUS-cropped; feeding one to FCN3 must fail loudly."""
    gc_path = _need(GENCAST_IC)
    with xr.open_dataset(gc_path) as gc:
        cropped = gc.isel(lat=slice(0, 105), lon=slice(0, 237))
        with pytest.raises(ValueError, match="global"):
            AD.gencast_to_fcn3(cropped, calib=None)


def test_missing_variable_is_rejected():
    gc_path = _need(GENCAST_IC)
    with xr.open_dataset(gc_path) as gc:
        with pytest.raises(KeyError, match="specific_humidity"):
            AD.gencast_to_fcn3(gc.drop_vars("specific_humidity"), calib=None)


def test_datasource_protocol(converted):
    """earth2studio only needs __call__(time, variable) -> DataArray."""
    ic, _ = converted
    src = AD.GenCastICSource(ic)
    got = src(ic["time"].values[0], ["t2m", "u850"])
    assert list(got["variable"].values) == ["t2m", "u850"]
    assert got.sizes["lat"] == A.FCN3_NLAT and got.sizes["lon"] == A.FCN3_NLON


def test_uncalibrated_path_still_produces_valid_channels():
    """calib=None must yield a usable (if less accurate) IC, for smoke runs."""
    gc_path = _need(GENCAST_IC)
    with xr.open_dataset(gc_path) as gc:
        ic = AD.gencast_to_fcn3(gc, calib=None)
    assert list(ic["variable"].values) == list(A.FCN3_VARIABLES)
    assert np.isfinite(ic.values).all()
    tcwv = ic.sel(variable="tcwv").values
    assert 5.0 < float(tcwv.mean()) < 50.0, "uncalibrated tcwv is not even plausible"


# --------------------------------------------------------------------------- #
# Calibration sanity
# --------------------------------------------------------------------------- #
def test_calibration_is_physical():
    cal = AD.load_calibration()
    if cal is None:
        pytest.skip("no calibration built")
    mag = np.hypot(cal["wind_coef_real"].values, cal["wind_coef_imag"].values)
    assert 1.0 < float(np.median(mag)) < 1.6, "10->100 m speed-up outside a physical range"
    ang = np.degrees(np.arctan2(cal["wind_coef_imag"].values, cal["wind_coef_real"].values))
    assert abs(float(np.median(ang))) < 5.0, "global median veering should be ~0 (NH/SH cancel)"
    assert list(cal["level"].values) == list(A.P13)
