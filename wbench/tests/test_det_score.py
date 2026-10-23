"""wbench.det_score: the sp derivation, the two-scorer predictability estimator, IC plumbing.
CPU only, imports cleanly in both envs (no earth2studio, no xres at import)."""
import numpy as np
import pandas as pd
import xarray as xr

from wbench import det_score as D


def test_sp_is_msl_over_the_ocean_and_falls_hydrostatically_with_height():
    msl, t2m = np.array([101325.0, 101325.0]), np.array([288.15, 288.15])
    zs = np.array([0.0, D.G * 1500.0])
    sp = D.derive_sp(msl, t2m, zs)
    assert sp[0] == msl[0]
    tbar = 288.15 + D.LAPSE * 1500.0 / 2
    assert np.isclose(sp[1], 101325.0 * np.exp(-D.G * 1500.0 / (D.R_D * tbar)))
    assert 84000 < sp[1] < 86000          # ~850 hPa at 1.5 km, as it must be


def test_two_independent_scorers_recover_the_predictability():
    rng = np.random.default_rng(1)
    n, v = 400_000, 0.6
    m = rng.normal(scale=np.sqrt(v), size=n)
    outcome = m + rng.normal(scale=np.sqrt(1 - v), size=n)
    a = m + rng.normal(scale=0.5, size=n)            # independent errors, different sizes
    b = m + rng.normal(scale=0.9, size=n)
    r = lambda x, y: np.corrcoef(x, y)[0, 1]
    assert abs(D.v_from_pair(r(a, outcome), r(b, outcome), r(a, b)) - v) < 0.01


def test_a_shared_error_biases_the_estimate_upward():
    # the failure mode the pairwise check exists to expose
    rng = np.random.default_rng(2)
    n, v = 400_000, 0.6
    m = rng.normal(scale=np.sqrt(v), size=n)
    outcome = m + rng.normal(scale=np.sqrt(1 - v), size=n)
    shared = rng.normal(scale=0.4, size=n)
    a = m + shared + rng.normal(scale=0.5, size=n)
    b = m + shared + rng.normal(scale=0.5, size=n)
    r = lambda x, y: np.corrcoef(x, y)[0, 1]
    assert D.v_from_pair(r(a, outcome), r(b, outcome), r(a, b)) < v   # rho_a rho_b fall, r_ab rises


def test_icfile_returns_the_requested_channels_in_order(tmp_path):
    t = np.array(["2021-06-13T00"], dtype="datetime64[ns]")
    da = xr.DataArray(np.arange(3 * 2 * 4, dtype="float32").reshape(1, 3, 2, 4),
                      dims=("time", "variable", "lat", "lon"),
                      coords={"time": t, "variable": ["t2m", "msl", "sp"],
                              "lat": [90.0, -90.0], "lon": [0.0, 90.0, 180.0, 270.0]}, name="ic")
    p = tmp_path / "ic.nc"
    da.to_netcdf(p)
    got = D.ICFile(p)(t, ["sp", "t2m"])
    assert list(got["variable"].values) == ["sp", "t2m"]
    assert np.array_equal(got.sel(variable="sp").values, da.sel(variable="sp").values)


def test_sharding_gives_every_shard_every_lead():
    cs = D.cases()
    for s in range(8):
        mine = [c for i, c in enumerate(cs) if i % 8 == s]
        assert sorted({l for _, l in mine}) == list(D.LEADS)
        assert len(mine) == 8


def test_nsteps_and_case_cost():
    assert D.nsteps("2021-06-28T00", "2021-06-13T00", 24) == 15
    assert D.nsteps("2021-06-28T00", "2021-06-13T00", 6) == 60
    assert abs(D.case_h100h(D.FCN3_S_PER_SIMDAY) - 5.642) < 0.01   # = score_noise M=1


def test_level_sp_matches_an_isothermal_atmosphere_and_keeps_msl_below_1000_hpa():
    # isothermal at T0: Z(p) = R T0 ln(msl / p), so sp = msl exp(-zs / (R T0)) exactly
    t0, msl = 270.0, 101000.0
    p = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50], float)
    zs = np.array([0.0, 50.0, D.G * 800.0, D.G * 3000.0])      # sea, below 1000 hPa, 0.8 km, 3 km
    z = (D.R_D * t0 * np.log(msl / (p * 100)))[:, None] * np.ones((1, zs.size))
    t = np.full_like(z, t0)
    got = D.derive_sp(np.full(zs.size, msl), np.full(zs.size, t0), zs, z, t, p)
    want = msl * np.exp(-zs / (D.R_D * t0))
    # the only error is the 6.5 K/km lapse assumed in Tbar where the truth is isothermal:
    # 44 Pa through the 0.56 km layer above 0.8 km, 129 Pa through 1.1 km above 3 km
    assert np.allclose(got[2:], want[2:], rtol=2.5e-3)
    msl_form = D.derive_sp(np.full(zs.size, msl), np.full(zs.size, t0), zs)
    assert np.array_equal(got[:2], msl_form[:2])                # ground below 1000 hPa
