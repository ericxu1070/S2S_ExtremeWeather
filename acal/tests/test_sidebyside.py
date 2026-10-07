import itertools

import numpy as np

from acal import sidebyside as S


def test_expected_min_matches_enumeration():
    rng = np.random.default_rng(0)
    x = rng.random(8)
    brute = np.mean([min(c) for c in itertools.combinations(x, 3)])
    assert np.isclose(S.expected_min(x, 3), brute)
    assert np.isclose(S.expected_min(x, 8), x.min())
    assert np.isclose(S.expected_min(x, 1), x.mean())


def test_land_rmse_ignores_sea_and_weights_by_coslat():
    obs = np.zeros((2, 3))
    land = np.array([[1, 1, 0], [1, 1, 0]], float)
    coslat = np.array([1.0, 0.5])
    F = np.zeros((1, 2, 3))
    F[0, :, 2] = 100.0                     # sea error must not count
    F[0, 0, :2] = 2.0                      # lat weight 1
    F[0, 1, :2] = 4.0                      # lat weight 0.5
    want = np.sqrt((2 * 4 + 0.5 * 2 * 16) / 3)
    assert np.isclose(S.land_rmse(F, obs, land, coslat)[0], want)


def test_land_corr_is_one_for_scaled_pattern():
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(4, 5))
    land = np.ones((4, 5))
    coslat = np.linspace(0.6, 1.0, 4)
    F = np.stack([2 * obs + 3, -obs])
    r = S.land_corr(F, obs, land, coslat)
    assert np.allclose(r, [1.0, -1.0])
