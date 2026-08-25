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


# --------------------------------------------------------------------------- #
# the null controls
#
# The controls exist to break a confound: all six original cases are extremes, which are
# exactly the flows where a small IC error has the best chance of growing, so an effect
# measured only on them cannot be told apart from an effect the extremes manufacture.
# What has to be pinned is (a) that adding them did not disturb the six -- seeds and the
# other experiment's event list both flow through fcn3.fevents -- and (b) that the
# contrast between the two groups is a real two-sample test, not an overlap check.
# --------------------------------------------------------------------------- #
def test_controls_are_appended_so_the_six_events_keep_their_seeds():
    """Seeds are BASE_SEED + 1000 * index over the merged order.

    Inserting a control anywhere but the end would renumber an event's members, and the
    adapter ensemble would stop pairing with the native cube already on disk -- silently,
    since every downstream shape still matches. The paired bootstrap would then be
    unpaired, roughly doubling its standard error and turning the whole design back into
    the underpowered thing it was built to replace.
    """
    from fcn3 import fevents as F

    assert F.ALL_ORDER[:len(F.ORDER)] == F.ORDER
    assert set(F.CONTROL_ORDER).isdisjoint(F.ORDER)
    for i, name in enumerate(F.ORDER):
        assert F.seed_for(F.EVENTS[name]) == F.BASE_SEED + 1000 * i
    seeds = [F.seed_for(e) for e in F.all_events()]
    assert len(set(seeds)) == len(seeds), "two cases share a base seed"

    # The AI+RES extension takes indices AFTER the ten frozen cases. This prefix
    # invariant is the whole design: every seed already rolled to disk is unchanged.
    assert F.SEED_ORDER[:len(F.ALL_ORDER)] == F.ALL_ORDER
    assert set(F.RES_ORDER).isdisjoint(F.ALL_ORDER)
    frozen = {
        "PNW_HeatDome_2021": 333, "HurricaneIan_2022": 1333,
        "WinterStorm_Uri_2021": 2333, "p90_20231107": 3333,
        "p90_20240802": 4333, "p90_20251224": 5333,
        "null_20230324": 6333, "null_20240329": 7333,
        "null_20241121": 8333, "null_20250526": 9333,
    }
    for name, seed in frozen.items():
        assert F.seed_for(F.ALL_EVENTS[name]) == seed
    seeds = [F.seed_for(e) for e in F.all_events() + F.res_events()]
    assert len(set(seeds)) == len(seeds), "an extension event shares a base seed"


def test_extension_events_are_outside_the_adapter_test_universe():
    """``ALL_ORDER`` is the adapter test's universe -- the Slurm array index map.

    An extension event that leaked into it would make the array shorter than the registry
    it is pinned against, and (worse) shift a control's array index. The extension lives
    in its own tuple and only the seed space concatenates them.
    """
    from fcn3 import fevents as F

    assert len(F.ALL_ORDER) == 10
    text = (ROOT / "slurm" / "aires_adaptest.slurm").read_text()
    for name in F.RES_ORDER:
        assert name not in text, f"{name} must not appear in the adapter-test array"


def test_controls_stay_out_of_the_fcn3_vs_gencast_experiment():
    """``F.selected()`` is what run_fcn3.py and compare_fcn3_gencast.py iterate over.

    A control has no GenCast cube and never needs one, so letting it into the default
    selection would add a broken seventh panel to that experiment's figures. It must still
    be reachable when named explicitly -- the adapter test rolls its native half with the
    same driver.
    """
    import os
    from fcn3 import fevents as F

    assert [e.name for e in F.selected()] == list(F.ORDER)
    old = os.environ.get("FCN3_EVENTS")
    try:
        os.environ["FCN3_EVENTS"] = F.CONTROL_ORDER[0]
        assert [e.name for e in F.selected()] == [F.CONTROL_ORDER[0]]
        # An extension event is likewise reachable only when named explicitly (that is
        # how a native FCN3 context cube would be rolled for one), never by default.
        os.environ["FCN3_EVENTS"] = "Southwest_HeatWave_2020"
        assert [e.name for e in F.selected()] == ["Southwest_HeatWave_2020"]
    finally:
        os.environ.pop("FCN3_EVENTS", None)
        if old is not None:
            os.environ["FCN3_EVENTS"] = old


def test_the_frozen_p90_gencast_spec_is_unchanged():
    """``fcn3/fevents.py --spec`` feeds the p90 GenCast launcher, which is frozen.

    The controls are injected into xres through the same additive hook, so the function
    that builds that spec had to be generalised. Its DEFAULT output must not move.
    """
    from fcn3 import fevents as F

    assert F.xres_extra_events_spec() == (
        "p90_20231107=2023-11-07:t2m_anom,"
        "p90_20240802=2024-08-02:t2m_anom,"
        "p90_20251224=2025-12-24:t2m_anom")
    ctl = F.xres_extra_events_spec(F.controls())
    assert all(n in ctl for n in F.CONTROL_ORDER)
    assert not any(n in ctl for n in F.ORDER)
    # The extension spec declares only what xres does NOT already know: the two
    # xres-native heat events must never be re-declared to xres through the hook.
    assert F.xres_extra_events_spec(F.res_events()) == (
        "SCentral_HeatDome_2023=2023-06-27:t2m_anom")


def test_controls_are_actually_quiet_days():
    """"No anomaly" has to be literal, or the controls are just more events.

    Each is pinned to its frozen p90 control case and checked against the +2.3662 K
    threshold that defines an event in that set.
    """
    import pandas as pd
    from fcn3 import fevents as F

    cases = pd.read_csv(ROOT / "p90" / "cases.csv").set_index("case")
    for name, case in F.P90_CONTROL_CASE.items():
        row = cases.loc[case]
        assert row["kind"] == "control", f"{case} is not a control in p90/cases.csv"
        assert abs(row["peak_anom_K"]) < 0.4, f"{name} is not a quiet day"
        assert pd.Timestamp(row["peak"]).date() == pd.Timestamp(
            F.CONTROLS[name].peak).date()


def test_event_groups_expand_in_the_events_selector():
    ext = [e.name for e in AT._events("extreme")]
    null = [e.name for e in AT._events("null")]
    assert tuple(ext) == AT.GROUPS["extreme"]
    assert tuple(null) == AT.GROUPS["null"]
    assert all(AT.group_of(n) == "null" for n in null)
    assert all(AT.group_of(n) == "extreme" for n in ext)
    assert [e.name for e in AT._events(None)] == list(AT.EVENTS)


def test_contrast_is_a_two_sample_test_not_an_overlap_check():
    """Two CIs that both cross zero can still differ from each other.

    Planted here: +3% +- 1% on extremes and -3% +- 1% on controls. Each group's own 95%
    interval is [1.04, 4.96] and [-4.96, -1.04] -- but the CONTRAST is 6% +- 1.41%, which
    is the finding. A reader eyeballing two overlapping-or-not intervals gets this wrong
    in both directions, which is why it is computed rather than left to the eye.
    """
    ext = AT.pooled_effect([0.03] * 4, [0.02] * 4)      # se = 0.01
    null = AT.pooled_effect([-0.03] * 4, [0.02] * 4)
    c = AT.group_contrast(ext, null)
    assert c["diff"] == pytest.approx(0.06)
    assert c["se"] == pytest.approx(np.sqrt(2) * 0.01)
    assert c["p_two_sided"] < 1e-4 and c["regime_dependent"]

    # identical groups: no regime dependence, whatever each group's own CI says
    same = AT.group_contrast(ext, AT.pooled_effect([0.03] * 4, [0.02] * 4))
    assert same["diff"] == pytest.approx(0.0)
    assert same["p_two_sided"] == pytest.approx(1.0)
    assert not same["regime_dependent"]


def test_contrast_degrades_gracefully_when_a_group_is_empty():
    """Before the control cubes exist, the report still has to run."""
    c = AT.group_contrast(AT.pooled_effect([0.03] * 4, [0.02] * 4),
                          AT.pooled_effect([], []))
    assert not np.isfinite(c["z"]) and not c["regime_dependent"]


def test_ten_cases_can_reach_p_below_0p01():
    """What the controls buy the sign test, stated as the reason they were worth the GPU.

    Six unanimous cases bottom out at p = 0.031; ten reach 0.002. That is the difference
    between "suggestive" and "significant at any conventional level".
    """
    assert AT.sign_test([1] * 10)["p_two_sided"] == pytest.approx(2 / 1024)
    assert AT.sign_test([1] * 10)["p_two_sided"] < 0.01


@pytest.mark.parametrize("shape", [(2, 1, 1), (3, 4, 4), (7, 3, 5), (24, 12, 20)])
def test_crps_spread_matches_the_pairwise_definition(shape):
    """The O(M log M) spread term must equal the O(M^2) one it replaced, exactly.

    ``crps_spread`` uses the sorted-order identity for sum_ij |x_i - x_j| because the
    pairwise form made the paired bootstrap a 4.2-hour job. An identity that is only
    ALMOST right would shift every CRPS number in the paper by an amount no one would
    notice, so it is checked against the brute-force definition it is standing in for.
    """
    rng = np.random.default_rng(shape[0] * 31 + shape[1])
    x = rng.normal(0.0, 3.0, shape)
    brute = np.abs(x[:, None] - x[None, :]).mean(axis=(0, 1)) / 2.0
    np.testing.assert_allclose(AT.crps_spread(x), brute, rtol=1e-12, atol=1e-12)


def test_crps_spread_is_zero_for_an_identical_ensemble_and_grows_with_spread():
    same = np.ones((8, 3, 3))
    assert np.allclose(AT.crps_spread(same), 0.0)
    rng = np.random.default_rng(0)
    tight = AT.crps_spread(rng.normal(0, 1, (16, 4, 4)))
    wide = AT.crps_spread(rng.normal(0, 5, (16, 4, 4)))
    assert wide.mean() > tight.mean()
