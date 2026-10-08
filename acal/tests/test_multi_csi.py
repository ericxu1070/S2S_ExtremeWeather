"""Pure-helper tests for acal.multi_csi (tiny synthetic data, no file loading)."""
import numpy as np
import xarray as xr

from acal import multi_csi as C


def _da(vals):
    return xr.DataArray(np.array(vals, float), dims=("lat", "lon"),
                        coords=dict(lat=[30.0, 40.0], lon=[250.0, 251.0]))


def test_fig_name():
    assert C.fig_name("7d", 4.0) == "csi/csi_heat_4K.png"
    assert C.fig_name("daily", -2.0) == "csi/csi_daily_cold_2K.png"


def test_fam_of():
    assert C.fam_of(3.0) == "heat" and C.fam_of(-3.0) == "cold"


def test_wmean_ignores_nan_and_weights_by_coslat():
    da = _da([[1, np.nan], [0, 0]])
    w = np.cos(np.deg2rad([30.0, 40.0]))
    assert abs(C.wmean(da) - (1 * w[0]) / (w[0] + 2 * w[1])) < 1e-12


def test_win_share_excludes_nan():
    d = _da([[1, np.nan], [-1, 1]])
    w = np.cos(np.deg2rad([30.0, 40.0]))
    assert abs(C.win_share(d) - (w[0] + w[1]) / (w[0] + 2 * w[1])) < 1e-12
