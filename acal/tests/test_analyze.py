"""Tests for the cross-case scorecard (`acal/analyze.py`).

Pure, synthetic, and fast: a hand-built `DMCResult` with known weights stands in for a
run, so nothing here reads `runs/`. What is pinned:

* **the estimator is the runs' own.** `Case.p_raw` must equal `DMCResult.exceedance`
  away from ties (where `>` and `>=` agree) and `Z * mean(w * 1[A >= a])` at a tie.
* **self-normalization.** At the population minimum every walker is beyond the threshold,
  so `p_sn == 1` exactly and `p_raw` IS the normalization check.
* **the cold sign.** A cold case is scored on `-A`: its P must equal the heat-sign P of
  the mirrored population at the mirrored threshold, and PIT + P_sn must sum to 1.
* **the climatology pool** wraps the year boundary and excludes the case's own +/-10 d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acal import analyze as Z
from aires import dmc


def _case(al, final_v, log_z, sign=1.0, obs=None) -> Z.Case:
    n = len(al)
    res = dmc.DMCResult(n_walkers=n, C=(0.0, 1.0), final_V=np.asarray(final_v, float),
                        log_Z=float(log_z))
    return Z.Case("synthetic", sign, float(al[0] if obs is None else obs),
                  np.asarray(al, float), res, {}, {}, {})


@pytest.fixture
def heat():
    rng = np.random.default_rng(7)
    al = rng.normal(2.0, 1.0, 32)
    v = rng.normal(0.0, 0.8, 32)
    return _case(al, v, log_z=-0.6)


def test_matches_dmc_exceedance_away_from_ties(heat):
    grid = np.linspace(heat.al.min() - 0.5, heat.al.max() + 0.5, 57)
    grid = grid[~np.isin(grid, heat.al)]
    want = heat.result.exceedance(heat.al, grid)
    got = np.array([heat.p_raw(a) for a in grid])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)


def test_tie_counts_the_walker(heat):
    a = float(np.sort(heat.al)[20])
    w = np.exp(-heat.result.final_V)
    want = np.exp(heat.result.log_Z) * np.mean(w * (heat.al >= a))
    assert heat.p_raw(a) == pytest.approx(want, abs=1e-15)
    assert heat.p_raw(a) > heat.result.exceedance(heat.al, [a])[0]


def test_self_normalized_is_one_at_the_minimum(heat):
    a = float(heat.al.min())
    assert heat.p_sn(a) == pytest.approx(1.0, abs=1e-14)
    assert heat.p_raw(a) == pytest.approx(heat.result.normalization_check(), abs=1e-14)
    assert heat.n_beyond(a) == 32


def test_cold_sign_is_the_mirrored_heat_case(heat):
    cold = _case(-heat.al, heat.result.final_V, heat.result.log_Z, sign=-1.0)
    for a in np.linspace(-1, 5, 25):
        assert cold.p_raw(-a) == pytest.approx(heat.p_raw(a), abs=1e-15)
    # More negative is more extreme for a cold case: P falls as the threshold drops.
    assert cold.p_raw(-4.0) <= cold.p_raw(-2.0) <= cold.p_raw(0.0)


def test_pit_and_p_sum_to_the_normalization(heat):
    for a in (heat.al.min(), 1.5, 2.0, heat.al.max() + 1):
        assert heat.cdf_raw(a) + heat.p_raw(a) == pytest.approx(
            heat.result.normalization_check(), abs=1e-14)


def test_score_case_flags_unresolved_and_saturated(heat):
    daily = pd.Series(np.linspace(-3, 3, 400),
                      index=pd.date_range("2021-01-01", periods=400, freq="D"))
    peak = "2021-06-01"
    hi = Z.score_case(_case(heat.al, heat.result.final_V, -0.6, obs=heat.al.max() + 1),
                      2, peak, daily)
    assert hi["unresolved_obs"] and hi["p_obs_raw"] == 0.0 and hi["pit_sn"] == 1.0
    lo = Z.score_case(_case(heat.al, heat.result.final_V, -0.6, obs=heat.al.min() - 1),
                      2, peak, daily)
    assert lo["saturated_obs"] and lo["p_obs_sn"] == pytest.approx(1.0)
    assert lo["pit_sn"] == pytest.approx(0.0, abs=1e-14)
    # A conservative lift exists even where climatology never reached the value.
    assert np.isfinite(hi["lift_obs_raw_cons"])


def test_clim_pool_wraps_the_year_and_drops_own_window():
    idx = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    daily = pd.Series(np.arange(idx.size, dtype=float), index=idx)
    pool = Z.clim_pool(daily, "2023-01-05")
    d = pool.index
    assert not ((d >= "2022-12-26") & (d <= "2023-01-15")).any()       # own +/-10 d out
    assert ((d >= "2022-12-06") & (d <= "2022-12-25")).sum() == 20      # wraps into Dec
    assert (d.year == 2021).any() and (d.year == 2025).any()
    # Nothing farther than 30 d from an anniversary of Jan 5.
    assert not ((d.month >= 3) & (d.month <= 10)).any()


def test_p_clim_is_sign_aware():
    pool = pd.Series([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert Z.p_clim(pool, 1.0, +1.0) == (0.4, 2, 5)
    assert Z.p_clim(pool, -1.0, -1.0) == (0.4, 2, 5)
