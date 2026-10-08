"""Light tests for acal.multi_members pure helpers (no data)."""
import numpy as np

from acal import multi_members as MM


def test_e16_equal_to_min_for_16():
    x = np.arange(16.0) + 1
    assert MM.e16(x) == 1.0


def test_e16_nan_below_16_and_ignores_nan():
    assert np.isnan(MM.e16(np.ones(10)))
    x = np.r_[np.arange(20.0) + 1, np.nan]
    assert np.isfinite(MM.e16(x))


def test_e16_between_min_and_median():
    x = np.random.default_rng(0).normal(3, 1, 51)
    assert x.min() <= MM.e16(x) <= np.median(x)


def test_closer_count_skips_nan():
    assert MM.closer_count([1, 2, np.nan, 3], [2, 1, 1, 4]) == (2, 3)
