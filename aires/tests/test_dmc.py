"""Tests for the RES core.

These are the whole reason `aires/dmc.py` is model-agnostic: the algorithm that will
eventually cost ~46 H100-h per event can be validated here for a few seconds of CPU,
against an Ornstein-Uhlenbeck process whose exceedance probabilities are analytic.

Three things are pinned, in increasing order of how much they cover:

* the resampler draws integer offspring counts with exact totals AND correct per-walker
  marginals - the second half of that is not implied by the first, and getting it wrong
  produces a population that is tilted toward the wrong walkers while every obvious
  invariant still holds;
* `C_k = 0` collapses the algorithm exactly onto direct sampling;
* the probability estimator is unbiased, and it beats direct sampling in the tail exactly
  when the score is predictive - which is the property Gate 3 measured on the real system.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from aires import dmc

SCHEDULE = (0.0, 1.0, 1.4, 1.8, 2.0)      # aires.md's C_k


# --------------------------------------------------------------------------- #
# Pivotal sampling
# --------------------------------------------------------------------------- #
def test_pivotal_total_is_exact_every_draw():
    rng = np.random.default_rng(0)
    p = np.array([0.15, 0.4, 0.7, 0.25, 0.5])          # sums to 2
    draws = np.array([dmc.pivotal_sample(p, rng) for _ in range(2000)])
    assert set(draws.sum(1)) == {2}


def test_pivotal_marginals_are_not_merely_permuted():
    """The bug this exists for: an implementation can hold the total exactly AND return
    the right multiset of marginals while attaching them to the WRONG units.

    Deville-Tille's first case moves all the mass onto one of two units; the one that
    keeps it must be chosen with probability proportional to its OWN share. Using the
    other unit's share instead swaps the pair's marginals - the totals stay exact, the
    sorted marginals are unchanged, and the DMC estimator comes out ~10x high because the
    population is cloned toward the wrong walkers.
    """
    rng = np.random.default_rng(1)
    p = np.array([0.15, 0.4, 0.7, 0.25, 0.5])
    got = np.array([dmc.pivotal_sample(p, rng) for _ in range(60000)]).mean(0)
    assert np.allclose(got, p, atol=0.01), f"marginals {got} != targets {p}"
    # and specifically: the two that a swap would exchange are distinguishable
    assert got[0] < got[1] - 0.15


@pytest.mark.parametrize("p", [
    np.zeros(5),
    np.ones(4),
    np.array([1.0, 0.0, 1.0]),
    np.array([0.5]),                                    # single unit, but sums to non-int
])
def test_pivotal_handles_degenerate_inputs(p):
    rng = np.random.default_rng(2)
    if abs(p.sum() - round(p.sum())) > 1e-9:
        with pytest.raises(ValueError, match="whole number"):
            dmc.pivotal_sample(p, rng)
        return
    out = dmc.pivotal_sample(p, rng)
    assert out.sum() == round(p.sum())
    assert np.array_equal(out, p.astype(bool))


def test_pivotal_rejects_a_non_integer_total():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="whole number"):
        dmc.pivotal_sample(np.array([0.3, 0.3, 0.3]), rng)


def test_pivotal_rejects_probabilities_outside_the_unit_interval():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        dmc.pivotal_sample(np.array([1.5, 0.5]), rng)


# --------------------------------------------------------------------------- #
# Multiplicities
# --------------------------------------------------------------------------- #
def test_multiplicities_are_unbiased_and_sum_exactly():
    """Cloning means expected counts above 1, so this is an allocation problem, not a
    selection one: floor() outright plus pivotal sampling on the leftover fractions."""
    rng = np.random.default_rng(4)
    e = np.array([2.3, 0.4, 0.1, 1.2])                  # sums to 4
    m = np.array([dmc.multiplicities(e, rng) for _ in range(60000)])
    assert set(m.sum(1)) == {4}
    assert np.allclose(m.mean(0), e, atol=0.02)
    assert m.min() >= 0


def test_multiplicities_preserve_population_size_for_a_real_weight_vector():
    rng = np.random.default_rng(5)
    n = 64
    w = np.exp(rng.normal(scale=1.5, size=n))
    expected = w / w.mean()                              # sums to n
    for _ in range(200):
        assert dmc.multiplicities(expected, rng).sum() == n


def test_multiplicities_reject_negative_expectations():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="non-negative"):
        dmc.multiplicities(np.array([-0.5, 2.5]), rng)


# --------------------------------------------------------------------------- #
# Degenerate schedules
# --------------------------------------------------------------------------- #
def test_all_zero_C_reduces_exactly_to_direct_sampling():
    """The strongest end-to-end invariant available: with no tilt there must be no
    resampling at all, no normalization, and no reweighting."""
    rng = np.random.default_rng(7)
    n = 64
    x0 = rng.normal(size=n)
    r = dmc.run(n, (0.0, 0.0, 0.0),
                lambda k, p, s: x0 if s is None else s[p], lambda k, s: s, rng=rng)
    assert r.log_Z == 0.0
    assert np.allclose(r.weights, 1.0)
    assert r.normalization_check() == pytest.approx(1.0)
    for step in r.steps:
        assert np.array_equal(step.counts, np.ones(n, dtype="int64"))
        assert np.array_equal(step.parents, np.arange(n))
        assert step.ess == pytest.approx(n)
    assert np.array_equal(r.states, x0)                  # population literally unchanged
    assert r.n_founders == n


def test_standardize_survives_a_degenerate_population():
    """Every walker identical: no spread to divide by. Zeros, not NaN - a NaN here would
    silently poison every downstream weight."""
    z = dmc.standardize(np.full(8, 3.5))
    assert np.all(z == 0.0) and np.all(np.isfinite(z))


def test_standardize_against_a_fixed_reference():
    z = dmc.standardize(np.array([1.0, 3.0]), ref_mean=1.0, ref_sd=2.0)
    assert z == pytest.approx([0.0, 1.0])


# --------------------------------------------------------------------------- #
# An Ornstein-Uhlenbeck walker with an analytic answer
# --------------------------------------------------------------------------- #
S = 1.0                     # stationary standard deviation
LEG_TAU = 1.0 / 6.0         # resampling interval, in correlation times
NSUB = 20                   # sub-steps per leg


def _ou_leg(x, rng):
    a = np.exp(-LEG_TAU / NSUB)
    b = S * np.sqrt(1.0 - a ** 2)
    for _ in range(NSUB):
        x = a * x + b * rng.normal(size=x.shape)
    return x


def _run_res(n, seed, C=SCHEDULE):
    """RES on the OU process, started in its stationary distribution.

    ``ref=(0, S)`` fixes the standardization to the KNOWN stationary moments instead of
    the live population's. That matters: standardizing against the population makes the
    splitting function depend on the other walkers, which turns the estimator from
    unbiased into merely consistent. The strict claim is testable only in this regime, so
    this is where it is tested.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(scale=S, size=n)
    return dmc.run(n, C,
                   lambda k, p, s: _ou_leg(x0 if s is None else s[p], rng),
                   lambda k, s: s, rng=rng, standardize_scores=True, ref=(0.0, S))


def _run_ds(n, seed, n_legs=len(SCHEDULE) + 1):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=S, size=n)
    for _ in range(n_legs):
        x = _ou_leg(x, rng)
    return x


def test_normalization_check_is_near_one_under_a_real_tilt():
    """``Z * mean(exp(-V_K))`` estimates ``E[1]``. It uses the same bookkeeping as every
    probability the experiment reports, so it catches an error in ``Z`` or in the final
    weights without needing a known answer.

    It is a NOISY diagnostic - a product of five estimated normalizations times a mean
    dominated by its smallest term. At N = 256 a single run has sd ~0.5 and can land
    anywhere from 0.6 to 3, so one run coming back at 2 is not evidence of anything. The
    tolerance here is derived from the measured spread rather than guessed.
    """
    vals = np.array([_run_res(256, 100 + t).normalization_check() for t in range(40)])
    se = vals.std(ddof=1) / np.sqrt(vals.size)
    assert abs(vals.mean() - 1.0) < 3.0 * se, (
        f"normalization {vals.mean():.4f} +- {se:.4f} "
        f"({abs(vals.mean() - 1.0) / se:.1f} standard errors from 1)")


@pytest.mark.parametrize("threshold", [2.0, 3.0])
def test_estimator_is_unbiased_on_an_ornstein_uhlenbeck_walker(threshold):
    """The single most valuable test in this file: it validates the whole estimator -
    tilting, resampling, normalization and un-tilting - against an exact answer, with no
    GPU and no model."""
    exact = float(norm.sf(threshold / S))
    n, reps = 256, 40
    est = np.array([float(r.exceedance(r.states, threshold)[0])
                    for r in (_run_res(n, 2000 + t) for t in range(reps))])
    se = est.std(ddof=1) / np.sqrt(reps)
    assert abs(est.mean() - exact) < 3.0 * se, (
        f"P(X > {threshold}) estimated {est.mean():.6f} +- {se:.6f}, exact {exact:.6f} "
        f"({abs(est.mean() - exact) / se:.1f} standard errors)")


def test_res_beats_direct_sampling_when_the_score_is_predictive():
    """The point of the algorithm. With the score correlated to the outcome (here 0.85
    one leg ahead, comparable to Gate 3's measured 0.80-0.96), importance splitting must
    estimate a tail probability with materially less spread than direct sampling at the
    same population size.
    """
    threshold, n, reps = 3.0, 256, 40
    res = np.array([float(r.exceedance(r.states, threshold)[0])
                    for r in (_run_res(n, 3000 + t) for t in range(reps))])
    ds = np.array([float((_run_ds(n, 9000 + t) > threshold).mean()) for t in range(reps)])
    assert res.std(ddof=1) < 0.7 * ds.std(ddof=1), (
        f"RES sd {res.std(ddof=1):.6f} vs DS sd {ds.std(ddof=1):.6f} - no variance "
        f"reduction, which means the tilt is not being applied where the score points")


def test_res_and_direct_sampling_agree_on_a_moderate_threshold():
    """Where direct sampling is reliable, the two must agree - the regression that would
    catch a tilt applied in the wrong direction."""
    threshold, n, reps = 1.0, 256, 40
    res = np.mean([float(r.exceedance(r.states, threshold)[0])
                   for r in (_run_res(n, 4000 + t) for t in range(reps))])
    ds = np.mean([float((_run_ds(n, 8000 + t) > threshold).mean()) for t in range(reps)])
    assert res == pytest.approx(ds, rel=0.15)


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
def test_genealogy_tracks_ancestry_and_founders_can_only_shrink():
    r = _run_res(64, 55)
    assert r.genealogy.shape == (len(SCHEDULE), 64)
    anc = r.ancestry()
    assert anc.shape == (64,) and anc.min() >= 0 and anc.max() < 64
    assert r.n_founders == np.unique(anc).size
    assert 1 <= r.n_founders <= 64
    # A resampling can only ever merge lineages, never split them across founders.
    seen = [np.unique(r.genealogy[0]).size]
    assert seen[0] <= 64


def test_ess_is_bounded_by_the_population():
    r = _run_res(64, 56)
    for step in r.steps:
        assert 0.0 < step.ess <= 64 + 1e-9
    assert r.steps[0].ess == pytest.approx(64)          # C_1 = 0 -> uniform


def test_propagate_is_called_once_past_the_last_resampling():
    """The survivors of the final resampling have not moved yet; carrying them to the
    horizon is what keeps `states`, `final_V` and `weights` describing one population."""
    calls = []

    def propagate(k, parents, states):
        calls.append(k)
        return np.zeros(8) if states is None else states[parents]

    dmc.run(8, (0.0, 1.0), propagate, lambda k, s: np.arange(8, dtype=float),
            rng=np.random.default_rng(9))
    assert calls == [1, 2, 3], f"expected a horizon leg after the last step, got {calls}"


def test_score_shape_and_finiteness_are_checked():
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError, match="must return 8 values"):
        dmc.run(8, (1.0,), lambda k, p, s: np.zeros(8), lambda k, s: np.zeros(3), rng=rng)
    with pytest.raises(ValueError, match="non-finite"):
        dmc.run(8, (1.0,), lambda k, p, s: np.zeros(8),
                lambda k, s: np.full(8, np.nan), rng=rng)


def test_exceedance_is_monotone_and_returns_one_value_per_threshold():
    r = _run_res(128, 57)
    thr = np.array([-2.0, 0.0, 1.0, 2.0, 3.0])
    p = r.exceedance(r.states, thr)
    assert p.shape == thr.shape
    assert np.all(np.diff(p) <= 1e-12), f"exceedance must be non-increasing, got {p}"
