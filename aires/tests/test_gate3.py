"""Contract tests for Gate 3's scheduling, sharding and seeding.

None of this needs a GPU or a model, and all of it is the kind of bookkeeping that fails
silently: a checkpoint that does not land on the lead the score stage looks for, a shard
split that gives one GPU every long rollout, a cached state from a different run that the
resume logic adopts because the file exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aires import aconfig as A
from aires import gate3 as G3
from aires import score as S
from fcn3 import fevents as F

EV = F.EVENTS["PNW_HeatDome_2021"]


# --------------------------------------------------------------------------- #
# Segment schedule
# --------------------------------------------------------------------------- #
def test_segments_tile_the_horizon_exactly():
    segs = A.gate3_segments()
    assert sum(n for _k, _l, n in segs) * A.WALKER_STEP_H / 24 == A.GATE3_HORIZON_DAYS
    assert [k for k, _l, _n in segs] == list(range(1, len(segs) + 1))
    assert segs[-1][1] == A.GATE3_HORIZON_DAYS


def test_every_checkpoint_lead_is_a_segment_boundary():
    ends = {l for _k, l, _n in A.gate3_segments()}
    for lead in A.GATE3_LEAD_DAYS:
        assert float(lead) in ends
        assert A.gate3_step_for_lead(lead) == round(lead / A.GATE3_SEG_DAYS)


def test_all_segments_are_the_same_length():
    """Uniform segments keep GenCast's jitted step at one shape, so a worker pays the
    ~5 min XLA compile once rather than once per distinct segment length."""
    assert len({n for _k, _l, n in A.gate3_segments()}) == 1


def test_off_schedule_lead_is_rejected():
    with pytest.raises(ValueError, match="not a walker checkpoint"):
        A.gate3_step_for_lead(4.0)


# --------------------------------------------------------------------------- #
# Score task enumeration and sharding
# --------------------------------------------------------------------------- #
def test_tasks_cover_every_walker_and_lead_once():
    ts = S.tasks(EV.name, 16)
    assert len(ts) == 16 * len(A.GATE3_LEAD_DAYS)
    assert len({(t.walker, t.lead_days) for t in ts}) == len(ts)


def test_sharding_balances_rollout_length_across_gpus():
    """5 leads over 8 shards: coprime, so every GPU draws each rollout length equally.
    If it did not, one GPU would hold all the 72-step forecasts and set the wall clock."""
    ts = S.tasks(EV.name, 16)
    per = [S.shard_of(ts, 8, s) for s in range(8)]
    assert sum(len(p) for p in per) == len(ts)
    assert len({len(p) for p in per}) == 1
    for p in per:
        counts = {l: sum(1 for t in p if t.lead_days == l) for l in A.GATE3_LEAD_DAYS}
        assert len(set(counts.values())) == 1, counts


def test_task_paths_follow_the_checkpoint_schedule():
    t = S.Task(EV.name, 7, 12.0)
    assert t.step == A.gate3_step_for_lead(12.0)
    assert t.state_path == A.state_path(EV.name, 7, t.step)
    assert t.cube_path.name == "w07_lead12_cube.nc"


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
def test_score_seed_is_common_across_walkers_but_not_leads():
    """Common random numbers: every walker scored at the same t_k is rolled with the same
    M internal-noise streams, so theta differences are IC differences (see score.py)."""
    a = S.score_seed(EV.name, 9.0)
    assert a == S.score_seed(EV.name, 9.0)
    assert a != S.score_seed(EV.name, 12.0)
    assert a != S.score_seed("WinterStorm_Uri_2021", 9.0)


def test_score_seeds_are_distinct_across_every_event_and_lead():
    seeds = [S.score_seed(n, l) for n in F.ORDER for l in A.GATE3_LEAD_DAYS]
    assert len(set(seeds)) == len(seeds)


def test_walker_segment_keys_are_unique_per_walker_and_step():
    """A clone is a new walker index pointed at a parent's state; the differing index is
    the entire perturbation mechanism, so two walkers must never share a stream."""
    jax = pytest.importorskip("jax")
    from aires import walker as W
    keys = {tuple(np.asarray(jax.random.key_data(W.segment_key(w, k))).tolist())
            if hasattr(jax.random, "key_data") else tuple(np.asarray(W.segment_key(w, k)))
            for w in range(4) for k in range(1, 4)}
    assert len(keys) == 12


# --------------------------------------------------------------------------- #
# Score horizon
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lead,steps", [(3, 72), (6, 60), (9, 48), (12, 36), (15, 24)])
def test_score_forecast_runs_from_the_checkpoint_to_the_peak(lead, steps):
    assert S.nsteps_for(EV, G3.expected_valid(EV, lead)) == steps


def test_state_that_is_not_on_a_six_hour_grid_is_rejected():
    with pytest.raises(ValueError, match="whole number"):
        S.nsteps_for(EV, pd.Timestamp(EV.init) + pd.Timedelta(hours=9, minutes=30))


def test_state_at_or_past_the_peak_is_rejected():
    with pytest.raises(ValueError):
        S.nsteps_for(EV, pd.Timestamp(EV.peak))


# --------------------------------------------------------------------------- #
# Stale-cache detection
# --------------------------------------------------------------------------- #
def test_segment_status_flags_a_state_at_the_wrong_lead(tmp_path, monkeypatch):
    """``walker.run_segment`` skips any segment whose files exist. The 2-step smoke run
    (job 1154) left a w00/step01 state at lead 1 d where Gate 3 wants 3 d; existence is
    not enough, the valid time has to match the schedule."""
    import xarray as xr
    from aires import walker as W

    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path)
    src = A.gencast_inputs_path(EV.name)
    if not src.exists():
        pytest.skip(f"no init frames at {src}")
    state = W.read_state(src)                       # valid at the init, i.e. lead 0
    sp = A.state_path(EV.name, 0, 1)
    sp.parent.mkdir(parents=True, exist_ok=True)
    W.write_state(state, sp, compress=False)
    A.diag_path(EV.name, 0, 1).write_bytes(b"")

    st, detail = G3.segment_status(EV, 0, 1, 3.0)
    assert st == "stale" and "schedule wants" in detail

    assert G3.segment_status(EV, 0, 1, 0.0)[0] == "ok"          # matches -> accepted
    assert G3.segment_status(EV, 1, 1, 3.0)[0] == "missing"     # absent -> to roll
