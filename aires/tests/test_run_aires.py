"""Tests for the AI+RES production driver.

The driver's job is bookkeeping, and every one of its failure modes is silent: a leg that
reuses a segment from another lineage, a resume that draws different clone/kill decisions
than the run it is resuming, a state pruned while something still needs it. None of them
raise on their own, and all of them would only show up as a slightly wrong exceedance
curve after ~39 H100-h. So they are pinned here, on a fake run tree, with no GPU and no
model of any kind.

The end-to-end test is the important one: it runs the REAL `dmc.run` loop through the REAL
coordinator against a fake walk pool and a fake score backend, and checks the three things
that make a resumed job the same experiment - the population stays at N, the genealogy on
disk matches the replay, and a second pass launches no pool at all.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aires import aconfig as A
from aires import dmc
from aires import run_aires as R
from fcn3 import fevents as F

EVENT = "PNW_HeatDome_2021"


# --------------------------------------------------------------------------- #
# Fixtures: a run tree in tmp_path, and fake artifacts
# --------------------------------------------------------------------------- #
@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path / "aires")
    return tmp_path


def ctx_for(n_walkers=8, C=(0.0, 1.0, 1.4), leads=(3.0, 6.0, 9.0), horizon=12.0,
            **kw) -> R.Ctx:
    return R.Ctx(ev=F.EVENTS[EVENT], tag="test", n_walkers=n_walkers, members=6,
                 C=C, leads=leads, horizon_days=horizon, dmc_seed=7,
                 base_seed=A.WALKER_BASE_SEED, nshards=2, attempts=1, retry_wait=0,
                 **kw)


def fake_segment(ctx: R.Ctx, walker: int, segment: int, parent) -> None:
    """A state.nc + diag.nc pair with just enough metadata to be validated."""
    end = ctx.valid_at(segment * A.GATE3_SEG_DAYS)
    times = pd.date_range(end - pd.Timedelta(hours=A.WALKER_STEP_H), end, periods=2)
    attrs = dict(event=ctx.event, walker=walker, step=segment,
                 base_seed=ctx.base_seed, parent="init" if parent is None else str(parent),
                 params_file=A.walker_model_kwargs()["params_file"])
    sp, dp = ctx.state_path(walker, segment), ctx.diag_path(walker, segment)
    sp.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"2m_temperature": ("time", [280.0, 281.0])},
               coords={"time": times}, attrs=attrs).to_netcdf(sp)
    xr.Dataset({"2m_temperature": ("time", [280.0, 281.0])},
               coords={"time": times}, attrs=attrs).to_netcdf(dp)


def fake_walk_pool(ctx: R.Ctx, calls: list):
    """Stand in for `run_pool`: do exactly what the walk shards would have done."""
    def _run(c, label, envname, argv):
        calls.append(label)
        k = int(argv[argv.index("--leg") + 1])
        leg = c.leg(k)
        man = R.read_walk_manifest(c, k)
        for w in range(c.n_walkers):
            for s in leg.segments:
                fake_segment(c, w, s, R.parent_of(c, leg, w, s, man.get("parents")))
    return _run


# --------------------------------------------------------------------------- #
# The leg schedule
# --------------------------------------------------------------------------- #
def test_event_resolution_spans_the_extension_registry():
    """`--event` resolves the frozen ten AND the AI+RES extension events, and nothing
    else. This is the regression test for widening the resolvers from ``F.EVENTS`` to
    ``F.event`` -- before the widening, an extension event was a SystemExit at prep."""
    ev = R._event("Southwest_HeatWave_2020")
    assert (ev.metric, ev.family) == ("t2m_anom", "heat")
    assert R._event("null_20230324").family == "null"
    with pytest.raises(SystemExit, match="unknown event"):
        R._event("Chicago_HeatWave_1995")


def test_legs_tile_the_horizon_with_equal_length_segments():
    legs = A.res_legs()
    segs = [s for leg in legs for s in leg.segments]
    assert segs == list(range(1, len(segs) + 1)), "segments must be contiguous from 1"
    assert legs[-1].lead_end_days == A.RES_HORIZON_DAYS
    for leg in legs:
        assert leg.lead_end_days == leg.last_segment * A.GATE3_SEG_DAYS


def test_there_is_exactly_one_more_leg_than_resampling_times():
    """`dmc.run` calls propagate len(C)+1 times; the extra leg carries t_K -> t_f."""
    legs = A.res_legs()
    assert len(legs) == len(A.RES_C) + 1
    assert [l.C for l in legs[:-1]] == list(A.RES_C)
    assert legs[-1].C is None and not legs[-1].resampling


def test_the_carry_leg_is_split_into_standard_segments():
    """15 -> 21 d is 6 days, but it must be TWO 3-day segments, not one 12-step leg.

    GenCast is jitted per segment shape; a differently shaped final leg would cost every
    worker a second XLA compile for no scientific reason.
    """
    carry = A.res_legs()[-1]
    assert len(carry.segments) == 2
    assert all(l.lead_end_days - l.lead_start_days == A.GATE3_SEG_DAYS
               for l in A.res_legs()[:-1])


def test_a_last_resampling_on_the_horizon_is_refused():
    with pytest.raises(ValueError, match="carry"):
        A.res_legs(leads=(3.0, 21.0), horizon_days=21.0, C=(0.0, 1.0))


# --------------------------------------------------------------------------- #
# The run tree is disjoint from Gate 3's
# --------------------------------------------------------------------------- #
def test_res_paths_never_collide_with_gate3_paths(tree):
    for w, s in ((0, 1), (15, 7)):
        assert A.res_state_path(EVENT, w, s, "pilot") != A.state_path(EVENT, w, s)
        assert A.res_diag_path(EVENT, w, s, "pilot") != A.diag_path(EVENT, w, s)
    assert A.res_score_cube_path(EVENT, 3, 9, "pilot") != A.score_cube_path(EVENT, 3, 9)
    assert A.res_dir(EVENT, "a") != A.res_dir(EVENT, "b")


def test_lineage_key_is_tree_independent():
    """A state seeded from Gate 3 still validates, because only w<ii>/step<kk> is read."""
    assert (R.lineage_key(A.state_path(EVENT, 3, 2))
            == R.lineage_key(A.res_state_path(EVENT, 3, 2, "pilot")))
    assert R.lineage_key(None) == "init"


# --------------------------------------------------------------------------- #
# Parent bookkeeping
# --------------------------------------------------------------------------- #
def test_only_the_first_segment_of_a_leg_crosses_a_resampling():
    ctx = ctx_for()
    carry = A.res_legs(leads=(3.0, 6.0), horizon_days=12.0, C=(0.0, 1.0))[-1]
    parents = [7 - i for i in range(8)]
    first, second = carry.segments[0], carry.segments[1]
    assert R.parent_of(ctx, carry, 0, first, parents) == ctx.state_path(7, first - 1)
    # inside a leg no resampling happened, so a walker continues ITSELF
    assert R.parent_of(ctx, carry, 0, second, parents) == ctx.state_path(0, second - 1)


def test_segment_one_has_no_parent():
    ctx = ctx_for()
    assert R.parent_of(ctx, ctx.leg(1), 3, 1, None) is None


# --------------------------------------------------------------------------- #
# Segment validation
# --------------------------------------------------------------------------- #
def test_a_state_on_the_wrong_lineage_is_stale_even_at_the_right_time(tree):
    """The failure that timestamp checking cannot see.

    Under resampling a slot changes ancestry at every leg, so two different ancestries
    produce states at the SAME valid time. Only the recorded parent distinguishes them.
    """
    ctx = ctx_for()
    fake_segment(ctx, 0, 2, ctx.state_path(5, 1))          # continues slot 5
    ok, _ = R.segment_status(ctx, 0, 2, ctx.state_path(5, 1))
    bad, why = R.segment_status(ctx, 0, 2, ctx.state_path(4, 1))   # this leg says slot 4
    assert ok == "ok"
    assert bad == "stale" and "w04" in why


def test_a_state_from_the_wrong_checkpoint_is_stale(tree):
    ctx = ctx_for()
    fake_segment(ctx, 0, 1, None)
    with xr.open_dataset(ctx.state_path(0, 1)) as ds:
        d = ds.load()
    d.attrs["params_file"] = "gencast_mini.npz"
    ctx.state_path(0, 1).unlink()
    d.to_netcdf(ctx.state_path(0, 1))
    st, why = R.segment_status(ctx, 0, 1, None)
    assert st == "stale" and "gencast_mini" in why


def test_a_state_without_its_diagnostics_is_partial_not_ok(tree):
    ctx = ctx_for()
    fake_segment(ctx, 0, 1, None)
    ctx.diag_path(0, 1).unlink()
    assert R.segment_status(ctx, 0, 1, None)[0] == "partial"


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #
def test_the_manifest_guard_rejects_a_changed_replay(tree):
    ctx = ctx_for()
    R.write_walk_manifest(ctx, ctx.leg(2), [0, 1, 2, 3, 4, 5, 6, 7])
    with pytest.raises(SystemExit, match="does not match"):
        R.write_walk_manifest(ctx, ctx.leg(2), [1, 1, 2, 3, 4, 5, 6, 7])


def test_the_manifest_guard_accepts_an_identical_replay(tree):
    ctx = ctx_for()
    p = [0, 0, 2, 2, 4, 4, 6, 6]
    R.write_walk_manifest(ctx, ctx.leg(2), p)
    assert R.write_walk_manifest(ctx, ctx.leg(2), np.array(p))["parents"] == p


def test_freeze_config_refuses_a_different_population_under_the_same_tag(tree):
    R.freeze_config(ctx_for(n_walkers=8))
    with pytest.raises(SystemExit, match="DIFFERENT configuration"):
        R.freeze_config(ctx_for(n_walkers=16))


# --------------------------------------------------------------------------- #
# Genealogy
# --------------------------------------------------------------------------- #
def test_slot_lineage_walks_the_genealogy_backwards(tree):
    ctx = ctx_for(n_walkers=4, C=(0.0, 1.0), leads=(3.0, 6.0), horizon=12.0)
    # leg2 parents: everyone descends from slot 3; leg3 parents: everyone from slot 0
    parents = {2: [3, 3, 3, 3], 3: [0, 0, 0, 0]}
    slots = R.slot_lineage(ctx, parents, final_slot=2)
    assert slots[3] == 2 and slots[4] == 2          # the carry leg is slot 2 itself
    assert slots[2] == 0                            # ...whose parent at leg 3 was slot 0
    assert slots[1] == 3                            # ...whose parent at leg 2 was slot 3


def test_prune_states_never_deletes_a_parent(tree):
    ctx = ctx_for(n_walkers=2, keep_states=2)
    for s in (1, 2, 3):
        for w in (0, 1):
            fake_segment(ctx, w, s, None if s == 1 else ctx.state_path(w, s - 1))
    leg = ctx.leg(3)                                  # about to roll segment 3
    R.prune_states(ctx, leg)
    assert not ctx.state_path(0, 1).exists()          # segment 1 is two behind: gone
    assert ctx.state_path(0, 2).exists()              # segment 2 is the parent: kept
    assert ctx.diag_path(0, 1).exists()               # diagnostics are NEVER pruned


def test_keep_states_of_one_is_clamped_so_the_parent_survives(tree):
    ctx = ctx_for(n_walkers=1, keep_states=1)
    for s in (1, 2):
        fake_segment(ctx, 0, s, None if s == 1 else ctx.state_path(0, s - 1))
    R.prune_states(ctx, ctx.leg(2))
    assert ctx.state_path(0, 1).exists()


# --------------------------------------------------------------------------- #
# The score
# --------------------------------------------------------------------------- #
def test_a_cold_event_is_scored_on_the_low_tail(tree, monkeypatch):
    """Resampling clones the LARGEST score, so a freeze has to be negated first.

    Without this, an importance-splitting run on Winter Storm Uri would spend its whole
    GPU budget cloning the warmest members of a cold extreme.
    """
    warm = R.Ctx(ev=F.EVENTS["PNW_HeatDome_2021"], tag="t", n_walkers=3, members=6,
                 C=(1.0,), leads=(3.0,), horizon_days=6.0, dmc_seed=1,
                 base_seed=A.WALKER_BASE_SEED)
    cold = R.Ctx(ev=F.EVENTS["WinterStorm_Uri_2021"], tag="t", n_walkers=3, members=6,
                 C=(1.0,), leads=(3.0,), horizon_days=6.0, dmc_seed=1,
                 base_seed=A.WALKER_BASE_SEED)
    assert warm.sign == +1 and cold.sign == -1
    monkeypatch.setattr(R, "ensure_leg_scored", lambda c, l: None)
    monkeypatch.setattr(R, "theta_from_fcn3", lambda c, l: dict(
        box=[1.0, 2.0, 3.0], conus=[0.0] * 3, sd_box=[0.1] * 3, sd_conus=[0.1] * 3,
        members=6, backend="fcn3"))
    assert R.leg_theta(warm, warm.leg(1))["used"] == [1.0, 2.0, 3.0]
    assert R.leg_theta(cold, cold.leg(1))["used"] == [-1.0, -2.0, -3.0]


def test_a_null_C_buys_no_score_and_tilts_nothing(tree, monkeypatch):
    ctx = ctx_for(n_walkers=4)
    called = []
    monkeypatch.setattr(R, "ensure_leg_scored", lambda c, l: called.append(l.k))
    monkeypatch.setattr(R, "theta_from_persistence", lambda c, l: dict(
        box=[1.0, 2.0, 3.0, 4.0], conus=[0.0] * 4, sd_box=[np.nan] * 4,
        sd_conus=[np.nan] * 4, members=0, backend="persistence"))
    rec = R.leg_theta(ctx, ctx.leg(1))               # C_1 = 0
    assert called == [], "no FCN3 forecast may be bought where C_k = 0"
    assert rec["used"] == [0.0] * 4
    assert rec["box"] == [1.0, 2.0, 3.0, 4.0], "the persistence diagnostic is still kept"


# --------------------------------------------------------------------------- #
# End to end: the real DMC loop through the real coordinator
# --------------------------------------------------------------------------- #
def _fake_scores(ctx):
    """A score that is a deterministic function of the slot, so replays are comparable."""
    def theta(c, leg):
        v = [float((w * 37 + leg.k * 11) % 23) for w in range(c.n_walkers)]
        return dict(box=v, conus=v, sd_box=[0.5] * c.n_walkers,
                    sd_conus=[0.5] * c.n_walkers, members=6, backend="fcn3")
    return theta


def test_the_loop_runs_end_to_end_and_a_resume_launches_no_pool(tree, monkeypatch):
    ctx = ctx_for(n_walkers=8)
    calls: list = []
    monkeypatch.setattr(R, "run_pool", fake_walk_pool(ctx, calls))
    monkeypatch.setattr(R, "ensure_leg_scored", lambda c, l: None)
    monkeypatch.setattr(R, "theta_from_fcn3", _fake_scores(ctx))
    monkeypatch.setattr(R, "theta_from_persistence", _fake_scores(ctx))
    monkeypatch.setattr(R, "visible_gpus", lambda: 0)

    assert R.stage_res(ctx) == 0
    assert len(calls) == len(ctx.legs), "one walk pool launch per leg"

    res = R.read_json(A.res_dir(EVENT, "test") / "res_result.json")
    r = dmc.DMCResult.from_dict(res["dmc"])
    assert len(r.steps) == len(ctx.C)
    for s in r.steps:
        assert int(s.counts.sum()) == ctx.n_walkers, "the population must stay at N"
        assert len(s.parents) == ctx.n_walkers

    # every leg's manifest agrees with the replayed genealogy
    for leg in ctx.legs[1:]:
        man = R.read_walk_manifest(ctx, leg.k)
        assert man["parents"] == r.steps[leg.k - 2].parents.tolist()

    # ...and a second pass is free
    calls.clear()
    ctx2 = ctx_for(n_walkers=8)
    monkeypatch.setattr(R, "run_pool", fake_walk_pool(ctx2, calls))
    assert R.stage_res(ctx2) == 0
    assert calls == [], "a resume must not launch a GPU pool for a finished leg"
    r2 = dmc.DMCResult.from_dict(
        R.read_json(A.res_dir(EVENT, "test") / "res_result.json")["dmc"])
    assert r2.log_Z == pytest.approx(r.log_Z)
    assert np.array_equal(r2.ancestry(), r.ancestry())


def test_a_rewritten_score_is_caught_rather_than_silently_forking(tree, monkeypatch):
    """The guard that makes "resubmit the job" the same experiment.

    A resumed coordinator replays every resampling. If a score cube changed underneath a
    finished leg, the replay draws different parents than the walkers on disk actually
    have - and those walkers would then be spliced into the wrong genealogy.
    """
    ctx = ctx_for(n_walkers=8)
    monkeypatch.setattr(R, "run_pool", fake_walk_pool(ctx, []))
    monkeypatch.setattr(R, "ensure_leg_scored", lambda c, l: None)
    monkeypatch.setattr(R, "theta_from_fcn3", _fake_scores(ctx))
    monkeypatch.setattr(R, "theta_from_persistence", _fake_scores(ctx))
    monkeypatch.setattr(R, "visible_gpus", lambda: 0)
    assert R.stage_res(ctx) == 0

    # a different score at leg 2 -> different parents at leg 3
    p = R.theta_path(ctx, 2)
    rec = R.read_json(p)
    rec["used"] = list(reversed(rec["used"]))
    R.write_json(p, rec)
    with pytest.raises(SystemExit, match="does not match"):
        R.stage_res(ctx_for(n_walkers=8))


def test_a_zero_schedule_reproduces_direct_sampling_through_the_driver(tree, monkeypatch):
    """`C = 0` everywhere must leave the population untouched: no clones, no kills.

    `dmc` pins this on its own; running it through the driver additionally pins that the
    parent maps written to disk are the identity, which is what makes the Gate 3 seeding
    of segment 2 valid.
    """
    ctx = ctx_for(n_walkers=8, C=(0.0, 0.0), leads=(3.0, 6.0), horizon=9.0)
    monkeypatch.setattr(R, "run_pool", fake_walk_pool(ctx, []))
    monkeypatch.setattr(R, "ensure_leg_scored", lambda c, l: None)
    monkeypatch.setattr(R, "theta_from_fcn3", _fake_scores(ctx))
    monkeypatch.setattr(R, "theta_from_persistence", _fake_scores(ctx))
    monkeypatch.setattr(R, "visible_gpus", lambda: 0)
    assert R.stage_res(ctx) == 0
    r = dmc.DMCResult.from_dict(
        R.read_json(A.res_dir(EVENT, "test") / "res_result.json")["dmc"])
    for s in r.steps:
        assert s.parents.tolist() == list(range(8))
        assert np.allclose(s.counts, 1)
    assert r.log_Z == pytest.approx(0.0)
    assert np.allclose(r.weights, 1.0)


def test_gate3_seeding_only_claims_what_the_schedule_makes_identical(tree, monkeypatch):
    """Segment 1 always; segment 2 only while C_1 = 0. Nothing else is reusable."""
    ctx = ctx_for(n_walkers=4, C=(0.0, 1.0), leads=(3.0, 6.0), horizon=9.0)
    monkeypatch.setattr(A, "GATE3_N_WALKERS", 4)
    for w in range(4):
        for s in (1, 2, 3):
            (A.segment_dir(EVENT, w, s)).mkdir(parents=True, exist_ok=True)
            A.state_path(EVENT, w, s).touch()
            A.diag_path(EVENT, w, s).touch()
    assert sorted(set(s for _w, s in R.gate3_seedable(ctx))) == [1, 2]

    tilted = ctx_for(n_walkers=4, C=(1.0, 1.0), leads=(3.0, 6.0), horizon=9.0)
    assert sorted(set(s for _w, s in R.gate3_seedable(tilted))) == [1]

    reseeded = ctx_for(n_walkers=4, C=(0.0, 1.0), leads=(3.0, 6.0), horizon=9.0)
    reseeded.base_seed += 1
    assert R.gate3_seedable(reseeded) == []
