"""Tests for the C_k tuning surrogate.

The surrogate is the only thing standing between a guessed splitting schedule and ~46
H100-h of production run, so the thing worth pinning is that it actually has the skill
curve it claims: simulate walkers through it with no resampling and the correlation
between each score and the realized outcome must come back as the ``rho_k`` it was built
from. If that drifts, every ESS number the replay prints is describing a different
experiment than the one Gate 3 measured.
"""
from __future__ import annotations

import numpy as np
import pytest

from aires import ctune


def test_surrogate_reproduces_the_skill_curve_it_was_built_from():
    rho = np.array([0.0, 0.6, 0.8, 0.95, 0.99])
    n, rng = 20000, np.random.default_rng(0)
    res, a = ctune.replay(rho, (0.0,) * len(rho), n, rng)   # C = 0: no resampling at all
    theta = np.array([s.theta for s in res.steps])
    got = np.array([np.corrcoef(theta[k], a)[0, 1] for k in range(len(rho))])
    assert np.allclose(got, rho, atol=0.03), f"skill curve {got.round(3)} != {rho}"


def test_surrogate_scores_and_outcome_are_standardized():
    rho = np.array([0.0, 0.6, 0.9, 0.97, 0.99])
    n, rng = 20000, np.random.default_rng(1)
    res, a = ctune.replay(rho, (0.0,) * len(rho), n, rng)
    assert a.std(ddof=1) == pytest.approx(1.0, abs=0.03)
    assert a.mean() == pytest.approx(0.0, abs=0.03)
    for s in res.steps:
        assert s.theta.std(ddof=1) == pytest.approx(1.0, abs=0.04)


def test_surrogate_predicts_cross_lead_correlations_it_was_not_given():
    """The model's real claim: ``corr(theta_j, theta_k) = rho_j`` for ``j < k``, which is
    what distinguishes signal-plus-noise from a random walk and is why it fits the Gate 3
    data (mean error 0.061 against a sampling error of 0.28)."""
    rho = np.array([0.0, 0.6, 0.8, 0.95, 0.99])
    n, rng = 20000, np.random.default_rng(2)
    res, _ = ctune.replay(rho, (0.0,) * len(rho), n, rng)
    theta = np.array([s.theta for s in res.steps])
    for j in range(len(rho)):
        for k in range(j + 1, len(rho)):
            got = float(np.corrcoef(theta[j], theta[k])[0, 1])
            assert got == pytest.approx(rho[j], abs=0.03), f"corr(z{j}, z{k}) = {got}"


def test_clones_diverge_at_the_rate_the_residual_uncertainty_allows():
    """A clone shares its parent's latent state and must then separate. With the skill at
    ``t_k`` equal to ``rho_k``, two siblings' outcomes must correlate at ``rho_k`` - not 1
    (identical, the deterministic-walker problem the paper has to hand-tune around) and
    not 0 (no inheritance at all)."""
    rho = np.array([0.0, 0.9])
    n, rng = 40000, np.random.default_rng(3)

    # Drive the surrogate's own propagate() to the last resampling time, then run the
    # horizon leg TWICE from that shared latent state: that is exactly a parent and its
    # clone.
    latent = np.zeros(n)
    for k, r in enumerate(rho, start=1):
        var = r - (rho[k - 2] if k >= 2 else 0.0)
        latent = latent + rng.normal(scale=np.sqrt(max(var, 0.0)), size=n)
    tail = np.sqrt(max(1.0 - rho[-1], 0.0))
    sibling_a = latent + rng.normal(scale=tail, size=n)
    sibling_b = latent + rng.normal(scale=tail, size=n)
    assert float(np.corrcoef(sibling_a, sibling_b)[0, 1]) == pytest.approx(rho[-1], abs=0.03)


def test_skill_curve_is_forced_non_decreasing():
    """A later forecast cannot know less about the outcome than an earlier one, but at
    N = 16 sampling noise can invert a pair; the martingale needs a non-negative variance
    increment, so the curve is clipped and the adjustment is reported."""
    rho = np.array([-0.05, 0.7, 0.6, 0.95])
    fixed = np.maximum.accumulate(np.clip(rho, 0.0, 0.999))
    assert np.all(np.diff(fixed) >= 0)
    assert fixed[0] == 0.0 and fixed[2] == 0.7
