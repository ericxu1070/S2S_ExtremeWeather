"""Tests for the 4 K reach figures (`acal/reach.py`): the tail rule, the weighted P, and
the published per-case numbers (data tests skip when `runs/` is not on this machine)."""
from __future__ import annotations

import numpy as np
import pytest

from acal import reach as RE


def test_beyond_is_ge_and_tail_signed():
    x = np.array([3.9, 4.0, 4.1, -4.0, -4.2, -3.9])
    np.testing.assert_array_equal(RE.beyond(x, 1.0), [0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(RE.beyond(x, -1.0), [0, 0, 0, 1, 1, 0])


def test_reach_stats_weighted_for_walkers_only():
    m = dict(sign=1.0, al=np.array([5.0, 3.0, 4.0, 1.0]),
             wn=np.array([0.1, 0.6, 0.2, 0.1]),
             raw=np.array([4.5, 0.0]), corr=np.array([4.5, 4.0]))
    st = RE.reach_stats(m)
    assert st["ai_res"][:2] == (2, 4) and st["ai_res"][2] == pytest.approx(0.3)
    assert st["cfs_raw"] == (1, 2, 0.5) and st["cfs_corr"] == (2, 2, 1.0)


def test_zero_goes_to_its_own_column():
    x = RE._px([0.0, 0.01])
    assert x[0] == RE.P_ZERO < RE.P_BREAK < RE.P_LO <= x[1]


needs_runs = pytest.mark.skipif(not RE.CB.BUILD_CSV.exists(), reason="runs/acal not here")


@needs_runs
def test_hits_and_published_numbers():
    t = RE.prob_table()
    h = t[t.hit].set_index("episode_id")
    assert sorted(h.index) == ["e02_c4_20210218", "e14_h4_20230104", "e33_h4_20250101",
                               "e34_c4_20250125", "e42_h4_20251228"]
    e33, e34 = h.loc["e33_h4_20250101"], h.loc["e34_c4_20250125"]
    assert e33.p_ai_res == pytest.approx(0.727322, abs=5e-7)
    assert (e33.p_cfs_corr, e33.p_cfs_raw) == (0.5625, 0.0625)
    assert e33.p_clim == pytest.approx(0.0105634, abs=5e-7)
    assert (e34.p_cfs_corr, e34.p_cfs_raw) == (0.125, 0.0625)
    # e34: 10 of 32 walkers beyond -4 K carrying ~0.9% of the weight
    cfs = RE.CB.load_cfs().set_index("episode_id")
    m = RE.event_members("e34_c4_20250125", cfs)
    k, n, p = RE.reach_stats(m)["ai_res"]
    assert (k, n) == (10, 32) and p == pytest.approx(e34.p_ai_res, abs=1e-12)
    assert p == pytest.approx(0.009075, abs=5e-6)
