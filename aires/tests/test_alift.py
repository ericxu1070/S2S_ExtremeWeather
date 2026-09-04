"""Contract tests for the lift figure's arithmetic.

The figure makes a claim - "this run said the event was N times more likely than
climatology, and nothing that produced that number knew the outcome" - so the things
worth pinning are the ones that would make the claim wrong without making it look wrong:

* the forecast side must be the SAME estimator the rest of the experiment reports, not a
  re-derivation that happens to agree on the pilot;
* the climatological side must stay empirical, i.e. it must report zero (and let the
  figure draw a bound) rather than quietly extrapolating past the pool;
* the tail sign must actually flip, or every cold event silently reports its warm tail;
* no function on the forecast path may take the observation as an argument.

No GPU, no network, no model.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from aires import alift as LF
from aires import dmc as D


# --------------------------------------------------------------------------- #
# The forecast side is the experiment's own estimator
# --------------------------------------------------------------------------- #
def _result(final_V, log_Z, values):
    r = D.DMCResult(n_walkers=len(final_V), C=(0.0,))
    r.final_V = np.asarray(final_V, dtype="float64")
    r.log_Z = float(log_Z)
    return r, np.asarray(values, dtype="float64")


def test_walker_mass_sums_to_the_normalization_check():
    """``sum_i p_i`` IS ``DMCResult.normalization_check()`` - the figure's total mass and
    the run's own self-consistency number are the same quantity, so a reader comparing
    them across figures must not see two different values."""
    final_V = np.array([-0.4, 0.0, 1.3, 2.2, -1.1])
    r, _ = _result(final_V, -0.93, np.zeros(5))
    d = {"weights": r.weights.tolist(), "log_Z": r.log_Z}
    assert LF.walker_mass(d).sum() == pytest.approx(r.normalization_check(), rel=1e-12)


def test_forecast_exceedance_matches_dmc_exceedance():
    """Subset-summing the per-walker masses must reproduce ``DMCResult.exceedance``.

    The figure sums ``p_i`` over a walker subset; the estimator multiplies ``Z`` by a
    weighted mean of an indicator. They are the same algebra, and this pins them together
    so a change to one is caught rather than silently producing two curves.
    """
    final_V = np.array([-0.8, 0.2, 1.1, 2.6, 0.5, -1.4])
    vals = np.array([1.2, 3.4, 5.1, 7.9, 4.4, 0.3])
    r, v = _result(final_V, -0.5, vals)
    thr = np.array([0.0, 2.0, 4.5, 6.0, 9.0])
    mass = LF.walker_mass({"weights": r.weights.tolist(), "log_Z": r.log_Z})
    # DMCResult uses a strict `>`; at thresholds off the sample both agree.
    assert LF.forecast_exceedance(v, mass, thr) == pytest.approx(r.exceedance(v, thr))


def test_uniform_weights_reduce_to_direct_sampling():
    """``C_k = 0`` everywhere makes ``V == 0``, ``Z == 1`` and every mass ``1/N``, so the
    forecast curve must be the plain empirical exceedance - the same end-to-end invariant
    ``tests/test_dmc.py`` pins for the estimator."""
    vals = np.array([0.0, 1.0, 2.0, 3.0])
    mass = LF.walker_mass({"weights": [1.0] * 4, "log_Z": 0.0})
    assert LF.forecast_exceedance(vals, mass, [1.0, 2.5]) == pytest.approx([0.75, 0.25])


def test_tail_sign_flips_which_tail_is_counted():
    """A cold event asks ``P(A_L <= a)``. Without the sign it would report the warm tail
    of a cold-event distribution, which is not merely wrong but plausible-looking."""
    vals = np.array([-6.0, -2.0, 0.0, 3.0])
    mass = LF.walker_mass({"weights": [1.0] * 4, "log_Z": 0.0})
    # a = 0: three walkers are at or below it, two are at or above it.
    assert LF.forecast_exceedance(vals, mass, 0.0, sign=-1.0) == pytest.approx([0.75])
    assert LF.forecast_exceedance(vals, mass, 0.0, sign=+1.0) == pytest.approx([0.50])


# --------------------------------------------------------------------------- #
# The climatological side stays empirical
# --------------------------------------------------------------------------- #
def test_clim_exceedance_is_zero_past_the_pool_and_never_extrapolates():
    """Past the pool's own extreme the honest answer is zero, which the figure turns into
    ``P_clim <= 1/n`` and a bounded lift. A fitted tail here was tried and rejected (see
    the module docstring); this test is what stops one being reintroduced silently."""
    pool = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert LF.clim_exceedance(pool, 4.5)[0] == 0.0
    assert LF.clim_exceedance(pool, 2.0)[0] == pytest.approx(0.6)


def test_clim_exceedance_at_a_quantile_is_that_quantile():
    """The rungs are defined as pool quantiles, so ``P_clim`` at rung ``lv`` must come
    back as ``1 - lv/100``; a lift is otherwise divided by the wrong base rate."""
    rng = np.random.default_rng(0)
    pool = rng.normal(size=4000)
    for lv in LF.RUNGS:
        a = float(np.quantile(pool, lv / 100.0))
        assert LF.clim_exceedance(pool, a)[0] == pytest.approx(1 - lv / 100.0, abs=2e-3)


def test_cold_tail_rung_uses_the_lower_quantile():
    pool = np.linspace(-5.0, 5.0, 2001)
    a = float(np.quantile(pool, 1 - 0.99))
    assert LF.clim_exceedance(pool, a, sign=-1.0)[0] == pytest.approx(0.01, abs=2e-3)


# --------------------------------------------------------------------------- #
# No hindsight on the forecast path
# --------------------------------------------------------------------------- #
def test_nothing_that_builds_a_curve_accepts_the_observation():
    """The point of the figure is that the threshold axis is a-priori. Every function on
    the forecast/climatology path takes values, masses and thresholds - never an
    observation - so the property is checked structurally rather than trusted."""
    banned = {"observed", "obs", "truth", "era5"}
    for fn in (LF.walker_mass, LF.forecast_exceedance, LF.clim_exceedance, LF.clim_pool):
        params = set(inspect.signature(fn).parameters)
        assert not (params & banned), f"{fn.__name__} takes {params & banned}"


# --------------------------------------------------------------------------- #
# Label de-collision
# --------------------------------------------------------------------------- #
def test_spread_labels_preserves_order_and_separates():
    y = np.array([0.0117, 0.0135, 0.0248, 3.2e-4])
    out = LF.spread_labels(y, min_ratio=1.9)
    assert np.argsort(out).tolist() == np.argsort(y).tolist()
    s = np.sort(out[out > 0])
    assert np.all(s[1:] / s[:-1] >= 1.9 - 1e-9)


def test_spread_labels_leaves_well_separated_values_alone():
    y = np.array([1e-4, 1e-2, 1.0])
    assert LF.spread_labels(y) == pytest.approx(y)


# --------------------------------------------------------------------------- #
# The climatological reference must belong to the event's own box
# --------------------------------------------------------------------------- #
def _flat_anom(cols, years=(1990, 2019)):
    import pandas as pd
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31", freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=len(idx)) for c in cols}, index=idx)


def test_clim_pool_refuses_the_conus_fallback_for_an_event_with_its_own_box():
    """The defect this figure shipped with: `WinterStorm_Elliott_2022` was added to
    EVENT_BOXES after the climatology cache was built, so its N Plains ``A_L`` (reaching
    -17 K) was scored against the CONUS column (spanning +-5 K), producing a spurious
    ``lift >= 285x``. Falling back must be an error, not a default."""
    from aires import aindex as AI

    assert "WinterStorm_Elliott_2022" in AI.EVENT_BOXES
    with pytest.raises(SystemExit, match="no column"):
        LF.clim_pool("WinterStorm_Elliott_2022", _flat_anom(["CONUS"]))


def test_clim_pool_accepts_conus_for_an_event_that_has_no_box_of_its_own():
    """The ``p90_*`` cases are DEFINED by a CONUS-mean index, so CONUS is their right
    reference rather than a fallback - the guard must not fire on them."""
    from aires import aindex as AI

    assert "p90_20240802" not in AI.EVENT_BOXES
    samp, sign = LF.clim_pool("p90_20240802", _flat_anom(["CONUS"]))
    assert sign == 1.0 and samp.size > 1000


def test_clim_pool_rejects_a_gappy_pool():
    """A box the top-up never covered comes back as a SHORT sample, not an error, and
    every quantile shifts with it. Ten years cannot pass for thirty."""
    with pytest.raises(SystemExit, match="gappy"):
        LF.clim_pool("p90_20240802", _flat_anom(["CONUS"], years=(1990, 1999)))


def test_clim_pool_reports_the_cold_tail_sign_for_a_freeze():
    """`Event` has no `tail_sign` attribute, so a getattr default silently returned +1
    here and would score a freeze's rarity off its warm tail."""
    from aires import aindex as AI

    _, sign = LF.clim_pool("WinterStorm_Elliott_2022",
                           _flat_anom(["WinterStorm_Elliott_2022"]))
    assert sign == -1.0 == AI.tail_sign("WinterStorm_Elliott_2022")


def test_clim_pool_reports_the_warm_tail_sign_for_a_heat_event():
    _, sign = LF.clim_pool("p90_20240802", _flat_anom(["CONUS"]))
    assert sign == 1.0
