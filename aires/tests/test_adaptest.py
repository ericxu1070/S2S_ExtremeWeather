"""Contract tests for the six-event adapter test.

``aires/adaptest.py`` exists to answer a question the single-event analysis could not:
does the adapter cost forecast skill? The answer is a number with an error bar, so the
things worth pinning are the properties of that number, not the plumbing:

  * the paired bootstrap must actually be PAIRED - a common member index applied to both
    ensembles. Break the pairing and the standard error inflates, which would quietly turn
    a resolved effect into an unresolved one. The test plants a known effect on top of
    shared noise and checks that the paired s.e. is the smaller of the two;
  * a planted degradation must be RECOVERED, and a true null must NOT be "resolved";
  * the sign test must be an exact binomial, including the corner cases (all one way, one
    event, an exact split);
  * pooling must be inverse-variance weighted, so a precise event outweighs a noisy one,
    and it must notice heterogeneity rather than average incompatible events silently.

All of it runs on synthetic arrays: no cubes, no GPU, no network. The one test that reads
the real registry is ``test_event_order_matches_the_slurm_array``, which pins the array
index mapping that the Slurm script and the Python module have to agree on - a silent
reordering there would score one event's adapter cube against another's native cube.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from aires import adaptest as AT

ROOT = Path(__file__).resolve().parents[2]
NLAT, NLON, M = 12, 20, 24


@pytest.fixture
def grid():
    lat = np.linspace(24.0, 50.0, NLAT)
    return lat, AT.area_weights(lat, NLON)


def synth(rng, *, offset=0.0, shared_noise=None, spread=1.0):
    """A (M, NLAT, NLON) ensemble: shared internal noise plus an IC-specific offset.

    ``shared_noise`` is what makes two ensembles a matched pair - passing the same array to
    both is the synthetic stand-in for adapter member i and native member i drawing the
    same internal-noise stream.
    """
    if shared_noise is None:
        shared_noise = rng.normal(0.0, spread, size=(M, NLAT, NLON))
    return shared_noise + offset, shared_noise


# --------------------------------------------------------------------------- #
# area weights and the scores themselves
# --------------------------------------------------------------------------- #
def test_area_weights_match_cos_lat(grid):
    lat, w = grid
    assert w.shape == (NLAT, NLON)
    np.testing.assert_allclose(w[:, 0], np.cos(np.deg2rad(lat)))
    # every column identical: the weight is a function of latitude only
    assert np.allclose(w, w[:, :1])


def test_rmse_and_crps_are_zero_for_a_perfect_deterministic_ensemble(grid):
    _, w = grid
    obs = np.full((NLAT, NLON), 3.0)
    members = np.repeat(obs[None], M, axis=0)
    s = AT.ensemble_scores(members, obs, w)
    assert s["rmse_ens_mean"] == pytest.approx(0.0)
    assert s["crps"] == pytest.approx(0.0)
    assert s["spread"] == pytest.approx(0.0)
    assert s["AL_abs_err"] == pytest.approx(0.0)
    assert s["n_members"] == M


def test_crps_is_the_fair_form_not_the_biased_one():
    """The 1/(2M^2) spread term must use the full M x M mean, including the zero diagonal.

    For a two-member ensemble {a, b} against truth y the fair CRPS is
    ``mean(|a-y|, |b-y|) - |a-b| / 4``. The biased variant divides by M(M-1) instead and
    would give ``... - |a-b| / 2`` here, so the two differ by a factor of two on the
    spread term - large enough to change a verdict and small enough to look plausible.
    """
    w = AT.area_weights(np.array([0.0]), 1)
    a, b, y = 1.0, 4.0, 2.0
    members = np.array([[[a]], [[b]]])
    obs = np.array([[y]])
    expect = (abs(a - y) + abs(b - y)) / 2 - abs(a - b) / 4
    assert AT.ensemble_scores(members, obs, w)["crps"] == pytest.approx(expect)


def test_ensemble_mean_rmse_is_not_the_mean_of_member_rmse(grid):
    """Two different, both-reported statistics. Averaging cancels; the mean is sharper."""
    _, w = grid
    rng = np.random.default_rng(0)
    obs = np.zeros((NLAT, NLON))
    members = rng.normal(0.0, 1.0, size=(M, NLAT, NLON))
    s = AT.ensemble_scores(members, obs, w)
    assert s["rmse_ens_mean"] < s["rmse_member_mean"]


# --------------------------------------------------------------------------- #
# the paired bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_pairing_beats_an_unpaired_comparison(grid):
    """The point of resampling matched indices, stated as an inequality.

    Both ensembles share an internal-noise field; the adapter side carries a constant
    +0.4 K offset on top. Resampling a common index set cancels the shared noise from the
    difference. Building the second ensemble from INDEPENDENT noise (the unpaired case)
    leaves both copies in, and the standard error must be visibly larger.
    """
    _, w = grid
    rng = np.random.default_rng(11)
    obs = np.zeros((NLAT, NLON))
    native, shared = synth(rng, spread=1.0)
    adapter, _ = synth(rng, offset=0.4, shared_noise=shared)          # paired
    unpaired, _ = synth(rng, offset=0.4)                              # independent noise

    paired = AT.paired_bootstrap(adapter, native, obs, w, n_draws=400, seed=1)
    broken = AT.paired_bootstrap(unpaired, native, obs, w, n_draws=400, seed=1)
    assert paired["rmse_ens_mean"]["se"] < broken["rmse_ens_mean"]["se"]


def test_bootstrap_recovers_a_planted_degradation(grid):
    """A large planted offset must come out positive, tight, and sign-stable."""
    _, w = grid
    rng = np.random.default_rng(3)
    obs = np.zeros((NLAT, NLON))
    native, shared = synth(rng, spread=0.5)
    adapter, _ = synth(rng, offset=2.0, shared_noise=shared)

    r = AT.paired_bootstrap(adapter, native, obs, w, n_draws=400, seed=2)["rmse_ens_mean"]
    assert r["delta"] > 0                       # adapter is worse
    assert r["ci_lo"] > 0                       # and resolvably so
    assert r["frac_sign_flips"] < 0.01


def test_bootstrap_does_not_resolve_a_true_null(grid):
    """Identical ensembles: delta exactly zero, and no CI may claim an effect."""
    _, w = grid
    rng = np.random.default_rng(5)
    obs = rng.normal(0.0, 1.0, size=(NLAT, NLON))
    native, shared = synth(rng, spread=1.0)
    adapter, _ = synth(rng, offset=0.0, shared_noise=shared)

    for stat, r in AT.paired_bootstrap(adapter, native, obs, w, n_draws=200, seed=4).items():
        assert r["delta"] == pytest.approx(0.0, abs=1e-12), stat
        assert r["ci_lo"] <= 0 <= r["ci_hi"], stat


def test_bootstrap_is_reproducible_and_refuses_mismatched_grids(grid):
    _, w = grid
    rng = np.random.default_rng(7)
    obs = np.zeros((NLAT, NLON))
    native, shared = synth(rng, spread=1.0)
    adapter, _ = synth(rng, offset=0.3, shared_noise=shared)

    a = AT.paired_bootstrap(adapter, native, obs, w, n_draws=120, seed=99)
    b = AT.paired_bootstrap(adapter, native, obs, w, n_draws=120, seed=99)
    assert a == b

    with pytest.raises(ValueError, match="member grids differ"):
        AT.paired_bootstrap(adapter[:-1], native, obs, w, n_draws=10)


# --------------------------------------------------------------------------- #
# the sign test - the statistic that actually has power
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("deltas, n, n_worse, p", [
    ([1, 1, 1, 1, 1, 1], 6, 6, 2 / 64),          # all six worse: 2 * (1/64)
    ([-1, -1, -1, -1, -1, -1], 6, 0, 2 / 64),    # all six better: same p, other direction
    ([1, 1, 1, 1, 1, -1], 6, 5, 14 / 64),        # 5 of 6
    ([1, 1, 1, -1, -1, -1], 6, 3, 1.0),          # even split: no evidence at all
    ([1], 1, 1, 1.0),                            # one event can never be significant
])
def test_sign_test_is_the_exact_binomial(deltas, n, n_worse, p):
    r = AT.sign_test(deltas)
    assert (r["n"], r["n_worse"]) == (n, n_worse)
    assert r["p_two_sided"] == pytest.approx(p, rel=1e-9)


def test_sign_test_drops_zeros_rather_than_splitting_them():
    r = AT.sign_test([1.0, 1.0, 0.0, 0.0, -1.0])
    assert r["n"] == 3 and r["n_worse"] == 2


def test_sign_test_with_no_events_does_not_raise():
    r = AT.sign_test([])
    assert r["n"] == 0 and np.isnan(r["p_two_sided"])


def test_six_events_cannot_reach_p_below_0p03():
    """The ceiling on what six events can prove, so no one over-reads a unanimous result.

    This is the honest limit of the whole design: even if all six events agree, the exact
    two-sided binomial p is 0.031. It is evidence, not proof, and a reader should know that
    before the numbers arrive.
    """
    assert AT.sign_test([1] * 6)["p_two_sided"] == pytest.approx(2 / 64)
    assert AT.sign_test([1] * 6)["p_two_sided"] > 0.03


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
def test_pooling_is_inverse_variance_weighted():
    """A precise event must dominate a noisy one, so the pooled value is not the mean."""
    r = AT.pooled_effect([0.10, 0.50], [0.01, 0.50])
    assert r["pooled"] == pytest.approx(0.1001599, rel=1e-4)
    assert r["unweighted_mean"] == pytest.approx(0.30)
    assert r["se"] < 0.01 + 1e-9


def test_pooling_shrinks_the_error_bar_with_more_events():
    one = AT.pooled_effect([0.2], [0.1])
    six = AT.pooled_effect([0.2] * 6, [0.1] * 6)
    assert six["se"] == pytest.approx(0.1 / np.sqrt(6), rel=1e-9)
    assert six["se"] < one["se"]
    assert six["detection_limit"] == pytest.approx(1.96 * six["se"])


def test_pooling_flags_heterogeneity():
    """Consistent events -> I^2 near zero; incompatible ones -> large I^2."""
    same = AT.pooled_effect([0.20, 0.21, 0.19], [0.05, 0.05, 0.05])
    wild = AT.pooled_effect([-1.0, 0.0, 1.0], [0.02, 0.02, 0.02])
    assert same["i_squared"] < 5
    assert wild["i_squared"] > 90


def test_pooling_ignores_events_with_no_usable_error_bar():
    r = AT.pooled_effect([0.2, 0.3, np.nan], [0.1, 0.0, 0.1])
    assert r["n"] == 1 and r["pooled"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# the one test that touches the real registry
# --------------------------------------------------------------------------- #
def test_event_order_matches_the_slurm_array():
    """The Slurm array index -> event mapping must equal ``fcn3.fevents.ORDER``.

    ``slurm/aires_adaptest.slurm`` hard-codes the event list so a shell script can pick one
    without importing Python. If that list is ever reordered, array task N would roll the
    adapter ensemble for one event and the score stage would pair it with another event's
    native cube - a mistake that produces perfectly finite, perfectly wrong numbers. The
    job script re-checks this at run time too; this test catches it before submission.
    """
    text = (ROOT / "slurm" / "aires_adaptest.slurm").read_text()
    m = re.search(r"EVENTS=\((.*?)\)\n", text, re.S)
    assert m, "could not find the EVENTS=( ... ) array in the Slurm script"
    names = [t for t in re.split(r"[\s\\]+", m.group(1)) if t]
    assert tuple(names) == AT.EVENTS


def test_pooled_stats_are_lower_is_better():
    """The relative delta's sign only means "worse" if every pooled score is lower-better.

    Adding e.g. a correlation to POOLED_STATS would silently flip the meaning of the sign
    test for that row, so the set is pinned.
    """
    assert AT.POOLED_STATS == ("rmse_ens_mean", "crps")
