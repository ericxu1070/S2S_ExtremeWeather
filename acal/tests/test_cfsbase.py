"""Tests for the CFS baseline scoring (acal/cfsbase.py): estimators, floors, bias, hind years."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from acal import cfsbase as CB

AL = np.array([-1.0, 0.5, 1.0, 2.0, 2.0, 3.0, 4.0, 0.0])


def test_empirical_is_ge_with_ties_counted():
    # two members sit exactly on 2.0 and must count (>=, as AI+RES)
    assert CB.cfs_prob(AL, 2.0, 1.0, "emp") == pytest.approx(4 / 8)
    assert CB.cfs_prob(AL, 5.0, 1.0, "emp") == 0.0


def test_cold_sign_flip():
    # cold tail: members at or below -a. -1.0 is a member, so P(A <= -1) = 1/8
    assert CB.cfs_prob(AL, -1.0, -1.0, "emp") == pytest.approx(1 / 8)
    assert CB.cfs_prob(AL, 2.0, -1.0, "emp") == pytest.approx(6 / 8)
    # mirror symmetry: negating members and the threshold swaps the sign
    assert CB.cfs_prob(-AL, -2.0, -1.0, "emp") == CB.cfs_prob(AL, 2.0, 1.0, "emp")


def test_gaussian_matches_scipy_and_uses_ddof1():
    m, sd = AL.mean(), AL.std(ddof=1)
    for a, s in ((1.5, 1.0), (-0.5, -1.0)):
        want = norm.sf((s * a - s * m) / sd)
        assert CB.cfs_prob(AL, a, s, "gauss") == pytest.approx(want)
    assert CB.cfs_pit(AL, 1.5, 1.0, "gauss") == pytest.approx(1 - CB.cfs_prob(AL, 1.5, 1.0, "gauss"))


def test_zero_spread_gaussian_degenerates_to_step():
    x = np.full(5, 1.0)
    assert CB.cfs_prob(x, 0.5, 1.0, "gauss") == 1.0
    assert CB.cfs_prob(x, 1.5, 1.0, "gauss") == 0.0


def test_empirical_pit_plus_p_is_one_without_ties_and_below_with_ties():
    # strict < for the PIT, >= for P: they sum to exactly 1 for any ensemble
    for a in (-2.0, 2.0, 2.5):
        assert CB.cfs_pit(AL, a, 1.0, "emp") + CB.cfs_prob(AL, a, 1.0, "emp") == pytest.approx(1.0)


def test_log_ratio_floors_each_side():
    # no member reached obs and AI+RES saw nothing either: both floored, ratio = log(16+1 / 32+1)
    assert CB.log_ratio(0.0, 0.0, 16) == pytest.approx(np.log(17 / 33))
    # CFS zero floors at 1/17, not infinity
    assert CB.log_ratio(0.5, 0.0, 16) == pytest.approx(np.log(0.5 * 17))
    # subset floor is 1/5
    assert CB.log_ratio(0.5, 0.0, 4) == pytest.approx(np.log(0.5 * 5))
    # AI+RES zero floors at 1/33
    assert CB.log_ratio(0.0, 0.25, 16) == pytest.approx(np.log(0.25 ** -1 / 33))


def test_members_bias_and_subset():
    al = np.arange(16.0)
    np.testing.assert_allclose(CB.members(al, 1.5, "corr_emp"), al - 1.5)
    np.testing.assert_allclose(CB.members(al, 1.5, "raw_emp"), al)
    np.testing.assert_allclose(CB.members(al, 1.5, "sub_emp"), al[-4:])
    assert CB.n_floor("sub_emp") == 4 and CB.n_floor("raw_gauss") == 16


def test_paired_stats_wins_ties_and_ci_reproducible():
    d = np.array([1.0, 2.0, -0.5, 0.0, 0.0, 3.0])
    s = CB.paired_stats(d)
    assert (s["win"], s["tie"], s["loss"]) == (3, 2, 1)
    assert s["ci_lo"] <= s["mean"] <= s["ci_hi"]
    assert CB.paired_stats(d)["ci_lo"] == s["ci_lo"]          # fixed seed
    assert np.isnan(CB.paired_stats(np.zeros(4))["wilcoxon_p"])


def test_brier():
    assert CB.brier(0.25, 1.0) == pytest.approx(0.5625)


def test_hind_jobs_exclude_own_year_and_have_enough_years():
    from acal import aprep
    jobs = CB.hind_jobs()
    peaks = {r.episode_id: pd.Timestamp(r.peak) for r in aprep.episodes().itertuples()}
    years = {}
    for eid, y, init, peak in jobs:
        assert y != peaks[eid].year
        assert peak.year == y and init < peak
        years.setdefault(eid, set()).add(y)
    assert set(years) == set(peaks)
    assert min(len(v) for v in years.values()) >= CB.MIN_YEARS
