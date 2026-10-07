"""Tests for the ROC module (`acal/roc.py`).

Pure tests pin the ROC construction (ties as one vertex, trapezoid AUC = Mann-Whitney with
ties at 0.5, the stratified paired bootstrap). The data tests reproduce numbers already
published by `analyze` and `cfsbase`, and skip when `runs/` is not on this machine.
"""
from __future__ import annotations

import numpy as np
import pytest

from acal import analyze as AN
from acal import roc as R


def test_perfect_and_inverted():
    o = np.array([1, 1, 0, 0, 0])
    assert R.auc([0.9, 0.8, 0.3, 0.2, 0.1], o) == 1.0
    assert R.auc([0.1, 0.2, 0.3, 0.8, 0.9], o) == 0.0


def test_ties_form_one_vertex_and_straight_segment():
    # all tied: one segment from (0, 0) to (1, 1), AUC 0.5 whatever the sort order
    far, hr = R.roc_curve([0.0] * 5, [1, 0, 1, 0, 0])
    np.testing.assert_array_equal(far, [0.0, 1.0])
    np.testing.assert_array_equal(hr, [0.0, 1.0])
    assert R.auc([0.0] * 5, [0, 0, 1, 0, 1]) == 0.5
    # one tie group spanning a hit and a miss gives a diagonal step
    far, hr = R.roc_curve([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0])
    np.testing.assert_allclose(far, [0, 0, 0.5, 1])
    np.testing.assert_allclose(hr, [0, 0.5, 1, 1])


@pytest.mark.parametrize("seed", range(20))
def test_trapezoid_equals_mann_whitney_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 40))
    p = rng.integers(0, 6, n) / 5.0              # coarse grid: many ties
    o = rng.random(n) < 0.35
    o[0], o[1] = True, False                     # both classes present
    assert R.auc(p, o) == pytest.approx(R.auc_mann_whitney(p, o), abs=1e-12)


def test_undefined_without_both_classes():
    with pytest.raises(ValueError):
        R.roc_curve([0.1, 0.2], [1, 1])


def test_hr_on_grid_takes_top_of_vertical_segment():
    far, hr = R.roc_curve([0.9, 0.8, 0.1], [1, 1, 0])     # (0,0) -> (0,1) -> (1,1)
    g = R.hr_on_grid(far, hr)
    assert g[0] == 1.0 and np.all(g == 1.0)


def test_bootstrap_stratified_and_paired():
    o = np.array([1, 0, 0, 1, 0, 0, 0])
    idx = R.boot_indices(o, n_boot=50, seed=1)
    # counts kept: the first 2 columns are hits, the rest misses, in every replicate
    assert np.all(o[idx[:, :2]] == 1) and np.all(o[idx[:, 2:]] == 0)
    # paired: two models with identical probabilities give identical bootstrap AUCs
    p = np.array([0.5, 0.1, 0.2, 0.7, 0.0, 0.3, 0.1])
    bt = R.paired_boot({"a": p, "b": p.copy()}, o, n_boot=50, seed=1)
    np.testing.assert_array_equal(bt["a"]["auc"], bt["b"]["auc"])
    # reproducible with the fixed seed
    bt2 = R.paired_boot({"a": p}, o, n_boot=50, seed=1)
    np.testing.assert_array_equal(bt["a"]["auc"], bt2["a"]["auc"])


# --------------------------------------------------------------------------- #
# Data: reproduce the published numbers
# --------------------------------------------------------------------------- #
needs_runs = pytest.mark.skipif(not AN.RUNGS_CASES_OUT.exists() or not R.CB.BUILD_CSV.exists(),
                                reason="runs/acal not on this machine")


@pytest.fixture(scope="module")
def rows():
    return R.case_rows()


@needs_runs
def test_published_self_normalized_and_clim(rows):
    idx = rows.set_index(["episode_id", "threshold_K"])
    _, cases = AN.load_all()
    by = {c.episode_id[:3]: c for c in cases}
    assert R.p_ai_res(by["e01"], 2.0) == pytest.approx(0.038374, abs=5e-7)
    assert R.p_ai_res(by["e02"], -3.0) == pytest.approx(0.471378, abs=5e-7)
    assert R.p_ai_res(by["e33"], 4.0) == pytest.approx(0.727322, abs=5e-7)
    assert R.p_ai_res(by["e42"], 3.0) == pytest.approx(0.481095, abs=5e-7)
    e42 = [k for k in idx.index.get_level_values(0) if k.startswith("e42")][0]
    assert idx.loc[(e42, 3.0), "p_clim"] == pytest.approx(0.044674, abs=5e-7)
    daily = AN.load_daily()
    pool = AN.clim_pool(daily, by["e01"].run["peak"])
    assert AN.p_clim(pool, 2.0, 1.0)[0] == pytest.approx(0.288732, abs=5e-7)


@needs_runs
def test_pools_and_cross_checks(rows):
    assert rows.groupby("pool").episode_id.nunique().to_dict() == {"warm": 31, "cold": 11}
    assert set(rows.threshold_K) == {3.0, 4.0, -3.0, -4.0}
    R.checks(rows)                                          # raises SystemExit on mismatch


@needs_runs
def test_auc_table_values(rows):
    tab, _ = R.analyse(rows)
    a = tab[tab.quantity == "auc"].set_index(["threshold_K", "model"]).value
    want = {3.0: (0.620, 0.693, 0.763), 4.0: (0.649, 0.774, 0.940),
            -3.0: (0.717, 0.550, 0.600), -4.0: (0.917, 1.000, 0.722)}
    for t, (raw, corr, res) in want.items():
        assert a[(t, "cfs_raw")] == pytest.approx(raw, abs=5e-4)
        assert a[(t, "cfs_corr")] == pytest.approx(corr, abs=5e-4)
        assert a[(t, "ai_res")] == pytest.approx(res, abs=5e-4)
    assert a[(4.0, "ai_res_raw")] == pytest.approx(0.905, abs=5e-4)
    n = tab.groupby("threshold_K")[["n_hits", "n_misses"]].first()
    assert n.loc[3.0].tolist() == [6, 25] and n.loc[4.0].tolist() == [3, 28]
    assert n.loc[-3.0].tolist() == [6, 5] and n.loc[-4.0].tolist() == [2, 9]
