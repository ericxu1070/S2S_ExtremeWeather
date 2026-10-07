"""Contract tests for the lead-time stability sweep (``aires/astab.py``).

The sweep's whole content is a curve of diagnostics against lead, so the things that can
quietly ruin it are not physics bugs - they are bookkeeping: a chain that lands one day
short of the peak, a tail segment of the wrong length, a pruned state that a later segment
needed, a diagnostic that cannot see the one failure mode already on record.

Two of these deserve naming, because both were live risks when the run was designed:

* **The tail table.** Only the 42-day chain is a whole number of 3-day segments. 28, 56
  *and* 70 leave a remainder, and 70 is the trap: it is a multiple of 7 and of 14, so it
  reads as if it should divide evenly by 3, and it does not. ``chain_schedule`` is pure
  arithmetic on ``horizon % 3`` with no per-chain special-casing, and the table below is
  what stops a hand-written exception from ever creeping back in.
* **The job-1155 signature.** The one *measured* failure of this machinery
  (``aires/HANDOFF.md``) was a wrong-checkpoint walker whose CONUS-mean T2m lost its
  diurnal cycle inside the first 3-day segment (00Z equal to 12Z to 0.03 K) and drifted
  26 K cold over 21 days. A diagnostic panel that cannot reproduce that on a synthetic
  cube cannot be trusted to catch its successor at 70 days.

None of this needs a GPU, a model, or any file in ``runs/``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aires import aconfig as A
from aires import score as S
from astab import diag as D
from astab import reduce as RD
from astab import schedule as SCH
from astab import sconfig as SC
from astab import stages as STG
from fcn3 import fevents as F
from gencast_s2s import config as C

# lead -> the ragged tail's length in GenCast steps, and the number of segments, for the
# FULL init-to-peak schedule. Computed during planning rather than reasoned about; only
# weeks 6, 12 and 18 divide evenly.
TAIL_STEPS = {4: 2, 6: 0, 8: 4, 10: 2, 12: 0, 14: 4, 16: 2, 18: 0, 20: 4}
N_SEGMENTS = {4: 10, 6: 14, 8: 19, 10: 24, 12: 28, 14: 33, 16: 38, 18: 42, 20: 47}

# The leads where the walker is rolled over that whole schedule. Everything else is an
# FCN3-extension lead whose walk is ONE segment; see `sconfig.STAB_WALK_WEEKS`.
EXTENSION_WEEKS = tuple(w for w in SC.STAB_WEEKS if w not in SC.STAB_WALK_WEEKS)


# --------------------------------------------------------------------------- #
# The week dicts (the only cap either model has)
# --------------------------------------------------------------------------- #
def test_the_two_week_dicts_agree():
    """``gencast_s2s.config`` and ``fcn3.fevents`` each carry a copy. A walker rolled to a
    lead the scorer resolves differently is a silent mismatch: the FCN3 arm would launch
    from a different init than the GenCast arm and nothing would say so."""
    assert C.WEEKS_TO_LEAD_DAYS == F.WEEKS_TO_LEAD_DAYS


def test_the_frozen_weeks_are_untouched():
    """Every finished experiment in this repo reads 2/3/4. Widening the dict is additive
    or it is a silent re-dating of the whole archive."""
    for w, d in {2: 14, 3: 21, 4: 28}.items():
        assert C.WEEKS_TO_LEAD_DAYS[w] == d
        assert C.lead_days_for(w) == d


def test_the_sweep_weeks_resolve_to_the_intended_leads():
    assert {w: C.lead_days_for(w) for w in SC.STAB_WEEKS} == {
        4: 28, 6: 42, 8: 56, 10: 70, 12: 84, 14: 98, 16: 112, 18: 126, 20: 140}


def test_the_walk_weeks_are_a_subset_of_the_sweep_weeks():
    """``STAB_WALK_WEEKS`` selects among the leads; it cannot introduce one. A week in it
    that is not in ``STAB_WEEKS`` would have no walker slot and no chain to attach to."""
    assert set(SC.STAB_WALK_WEEKS) <= set(SC.STAB_WEEKS)
    assert all(SC.walks_to_peak(w) for w in SC.STAB_WALK_WEEKS)
    assert not any(SC.walks_to_peak(w) for w in EXTENSION_WEEKS)


def test_a_week_outside_the_dict_still_raises():
    with pytest.raises(ValueError, match="weeks must be one of"):
        C.lead_days_for(5)


# --------------------------------------------------------------------------- #
# The segment schedule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_schedule_tiles_the_horizon_exactly(weeks):
    """Styled after ``test_gate3.py::test_segments_tile_the_horizon_exactly``. A chain
    that does not land ON the peak cannot produce A_L at all - the verification window is
    the 7 days ending there."""
    horizon = C.lead_days_for(weeks)
    segs = SCH.chain_schedule(horizon)
    assert sum(n for _k, _l, n in segs) * A.WALKER_STEP_H / 24 == horizon
    assert [k for k, _l, _n in segs] == list(range(1, len(segs) + 1))
    assert segs[-1][1] == horizon


@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_the_tail_table_is_exactly_what_planning_computed(weeks):
    """The 70-day off-by-one, pinned. 70 is a multiple of 7 and 14 and of neither 3 nor
    the 3-day segment - it needs a 1-day tail, and it is the easiest of the four to get
    wrong by eye."""
    segs = SCH.chain_schedule(C.lead_days_for(weeks))
    assert SCH.tail_steps(C.lead_days_for(weeks)) == TAIL_STEPS[weeks]
    assert len(segs) == N_SEGMENTS[weeks]
    if TAIL_STEPS[weeks]:
        assert segs[-1][2] == TAIL_STEPS[weeks]


@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_every_non_tail_segment_is_the_standard_length(weeks):
    """Mirrors ``test_gate3.py::test_all_segments_are_the_same_length``, relaxed by
    exactly one segment: uniform shape is what keeps GenCast's jitted step at one shape,
    so a ragged chain pays a SECOND ~5 min XLA compile and no more."""
    segs = SCH.chain_schedule(C.lead_days_for(weeks))
    tail = TAIL_STEPS[weeks]
    full = segs[:-1] if tail else segs
    assert {n for _k, _l, n in full} == {A.RES_SEG_STEPS}
    assert len({n for _k, _l, n in segs}) == (2 if tail else 1)


@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_the_ragged_tail_is_placed_last(weeks):
    """So every scoring checkpoint stays on a 3-day boundary, where
    ``A.segment_for_lead`` is defined. A tail placed first would shift every later
    boundary off the grid and quietly invalidate the whole scoring set."""
    segs = SCH.chain_schedule(C.lead_days_for(weeks))
    tail = TAIL_STEPS[weeks]
    for (_k, lead, n) in (segs[:-1] if tail else segs):
        assert n == A.RES_SEG_STEPS
        assert abs(lead / A.GATE3_SEG_DAYS - round(lead / A.GATE3_SEG_DAYS)) < 1e-9


def test_schedule_uses_no_per_chain_special_casing():
    """Pure arithmetic on ``horizon % 3``: it must be right for horizons nobody planned
    for, or the next lead added to the sweep gets a hand-written exception."""
    for horizon in range(3, 100):
        segs = SCH.chain_schedule(horizon)
        assert sum(n for _k, _l, n in segs) * A.WALKER_STEP_H / 24 == horizon
        assert SCH.tail_steps(horizon) == (horizon % 3) * 24 // A.WALKER_STEP_H


def test_a_horizon_shorter_than_one_segment_is_rejected():
    with pytest.raises(ValueError):
        SCH.chain_schedule(0)


# --------------------------------------------------------------------------- #
# Scoring checkpoints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_scoring_checkpoints_are_positive_three_day_multiples(weeks):
    horizon = C.lead_days_for(weeks)
    for lead in SCH.scoring_checkpoints(horizon):
        assert lead > 0 and abs(lead % A.GATE3_SEG_DAYS) < 1e-9
        assert A.segment_for_lead(lead) >= 1


@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_the_horizon_is_never_a_scoring_checkpoint(weeks):
    """The 42-day chain's last full segment lands exactly on the peak, so scoring there
    would ask FCN3 for a zero-step rollout, which ``score.nsteps_for`` rejects (n <= 0).
    Excluded for every chain, not just the one where it bites."""
    horizon = C.lead_days_for(weeks)
    cps = SCH.scoring_checkpoints(horizon)
    assert float(horizon) not in cps
    assert max(cps) < horizon


def test_the_forty_two_day_chain_is_the_one_that_would_have_tripped():
    """Named explicitly: without the exclusion this is the chain that fails, and it is
    the only one, so a test that merely checked the other three would pass."""
    ev = F.EVENTS[SC.STAB_EVENTS[0]]
    with pytest.raises(ValueError):
        S.nsteps_for(ev, pd.Timestamp(ev.peak))
    assert 42.0 not in SCH.scoring_checkpoints(42)
    assert 39.0 in SCH.scoring_checkpoints(42)


@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_lead_three_is_segment_one_for_every_chain(weeks):
    """The adapter-fed arm launches from lead 3 d on every chain regardless of horizon,
    which is what makes it one command for all four - and the longest coupled rollout
    available, since FCN3 must then carry lead-3 d."""
    assert A.segment_for_lead(SC.SCORE_LEAD_DAYS) == 1
    assert SC.SCORE_LEAD_DAYS in SCH.scoring_checkpoints(C.lead_days_for(weeks))
    assert SCH.chain_schedule(C.lead_days_for(weeks))[0][1] == SC.SCORE_LEAD_DAYS


# --------------------------------------------------------------------------- #
# Walker slots and the run tree
# --------------------------------------------------------------------------- #
def test_weeks_map_one_to_one_onto_contiguous_walker_slots():
    """``score.tasks`` enumerates ``for w in range(walkers)`` and cannot discover
    non-contiguous indices, so the four leads must occupy slots 0..3 of ONE tag rather
    than one tag each."""
    assert set(SC.WEEKS_TO_WALKER) == set(SC.STAB_WEEKS)
    assert sorted(SC.WEEKS_TO_WALKER.values()) == list(range(len(SC.STAB_WEEKS)))


def test_one_score_command_covers_every_chain_of_an_event():
    """One walker slot per lead, contiguous from 0, so ``score.tasks``'s ``range(walkers)``
    enumerates the whole coupled arm in a single command."""
    ev = SC.STAB_EVENTS[0]
    ts = S.tasks(ev, len(SC.STAB_WEEKS), [SC.SCORE_LEAD_DAYS], SC.STAB_TAG)
    assert len(ts) == len(SC.STAB_WEEKS)
    assert {t.walker for t in ts} == set(SC.WEEKS_TO_WALKER.values())
    assert all(t.step == 1 for t in ts)


def test_the_scoring_task_is_pointed_at_the_astab_tree_not_the_tag():
    """``score.Task``'s tagged layout is ``runs/aires/<event>/res/<tag>/``; this sweep's
    tree is ``runs/astab/``. ``stages._score_fed`` therefore passes explicit paths, and a
    Task that silently fell back to the tag would write eight cubes into the AI+RES run
    tree where a reducer expects one resampled population."""
    ch = SCH.Chain(SC.STAB_EVENTS[0], 4)
    t = S.Task(ch.event, ch.walker, SC.SCORE_LEAD_DAYS, SC.STAB_TAG,
               state_p=ch.state_path(1), cube_p=ch.score_cube_path(),
               zarr_p=ch.score_zarr_path())
    assert t.cube_path == ch.score_cube_path()
    assert t.state_path == ch.state_path(1)
    assert t.zarr_path == ch.score_zarr_path()
    assert SC.RUNS in t.cube_path.parents
    assert A.AIRES_ROOT not in t.cube_path.parents
    # ...and the tag is still carried, because it is stamped on the cube as `run_tag`.
    assert t.tag == SC.STAB_TAG


def test_partial_path_overrides_are_refused():
    """Set together or not at all: a Task that reads one tree and writes another is a
    cache that can never be checked."""
    ch = SCH.Chain(SC.STAB_EVENTS[0], 4)
    with pytest.raises(ValueError, match="together or not at all"):
        S.Task(ch.event, ch.walker, SC.SCORE_LEAD_DAYS, SC.STAB_TAG,
               cube_p=ch.score_cube_path())


def test_the_stab_tree_is_disjoint_from_every_other_run():
    """The sweep used to be a TAG inside the AI+RES run tree. It is now its own tree, and
    nothing it writes may land in any AI+RES population's paths - the caches are keyed by
    path, and four differently-initialised chains where a reducer expects one population
    is a silent corruption, not an error."""
    ev = SC.STAB_EVENTS[0]
    stab = {SC.state_path(ev, w, 1) for w in range(len(SC.STAB_WEEKS))}
    stab |= {SC.score_cube_path(ev, w, SC.SCORE_LEAD_DAYS)
             for w in range(len(SC.STAB_WEEKS))}
    for other in ("pilot", "persist", "smoke", "stab"):
        assert stab.isdisjoint({A.res_state_path(ev, w, 1, other) for w in range(16)})
    assert stab.isdisjoint({A.state_path(ev, w, 1) for w in range(16)})   # Gate 3, flat
    assert all(SC.RUNS in p.parents for p in stab)
    assert not any(A.AIRES_ROOT in p.parents for p in stab)


def test_every_astab_artifact_hangs_off_the_two_roots():
    """``sconfig`` is the only place the tree is decided. A path built anywhere else is a
    path ``--stage prune`` and the sync manifest do not know about."""
    ev = SC.STAB_EVENTS[0]
    for p in (SC.state_path(ev, 0, 1), SC.diag_path(ev, 0, 1), SC.record_path(ev, 0, 1),
              SC.score_cube_path(ev, 0, 3.0), SC.score_zarr_path(ev, 0, 3.0),
              SC.result_path(ev, 0), SC.log_dir(ev), SC.ref_path(ev),
              SC.ref_field_path(ev), SC.bands_path(), SC.gate3_panel_path(),
              SC.csv_path()):
        assert SC.RUNS in p.parents or p == SC.RUNS, p
    for p in (SC.fig_path(), SC.map_path(ev, "gencast", 4)):
        assert SC.FIGS in p.parents, p


@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_a_chain_knows_its_own_init_horizon_and_slot(weeks):
    ch = SCH.Chain(SC.STAB_EVENTS[0], weeks)
    assert ch.horizon_days == C.lead_days_for(weeks)
    assert ch.walker == SC.WEEKS_TO_WALKER[weeks]
    assert ch.peak - ch.init == pd.Timedelta(days=ch.horizon_days)
    # The FULL schedule always tiles init -> peak, whether or not the walker runs it.
    assert ch.init + pd.Timedelta(days=ch.full_schedule[-1][1]) == ch.peak
    # The WALK schedule tiles init -> walk_end, which is the peak only at a walk lead.
    assert ch.init + pd.Timedelta(days=ch.schedule[-1][1]) == ch.walk_end


@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_the_two_horizons_are_the_same_thing_only_at_a_walk_lead(weeks):
    """The distinction the FCN3 extension introduced, pinned at both ends.

    Conflating them costs one of two ways: ``schedule`` used where ``full_schedule``
    belongs builds ERA5 truth for three days of a 140-day chain; ``full_schedule`` where
    ``schedule`` belongs rolls 280 GenCast steps nobody asked for (~4 h of an H100).
    """
    ch = SCH.Chain(SC.STAB_EVENTS[0], weeks)
    if ch.walks_to_peak:
        assert ch.walk_days == ch.horizon_days
        assert ch.walk_end == ch.peak
        assert ch.schedule == ch.full_schedule
    else:
        assert ch.walk_days == int(SC.SCORE_LEAD_DAYS)
        assert ch.walk_end == ch.init + pd.Timedelta(days=SC.SCORE_LEAD_DAYS)
        assert ch.walk_end < ch.peak
        # ONE segment, and it is exactly the adapter-fed arm's launch state.
        assert len(ch.schedule) == 1
        assert ch.schedule == [(1, SC.SCORE_LEAD_DAYS, A.RES_SEG_STEPS)]
        assert ch.checkpoints == [SC.SCORE_LEAD_DAYS]
        assert len(ch.full_schedule) == N_SEGMENTS[weeks]


@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_a_walked_chains_checkpoints_are_the_old_ones_exactly(weeks):
    """``Chain.checkpoints`` replaced ``scoring_checkpoints(horizon)`` in the reduce
    stage. At a walk lead the two must be the same list - a refactor that changed the
    checkpoint set of the eight finished chains would re-date every verdict on disk."""
    ch = SCH.Chain(SC.STAB_EVENTS[0], weeks)
    if ch.walks_to_peak:
        assert ch.checkpoints == SCH.scoring_checkpoints(ch.horizon_days)
    assert all(0 < l < ch.horizon_days for l in ch.checkpoints)
    assert all(abs(l % A.GATE3_SEG_DAYS) < 1e-9 for l in ch.checkpoints)


@pytest.mark.parametrize("weeks", EXTENSION_WEEKS)
def test_an_extension_chain_hands_fcn3_the_whole_rest_of_the_horizon(weeks):
    """What the extension actually buys: 3 d of GenCast and lead-3 d of FCN3, so the
    coupled arm's rollout GROWS with the lead instead of the walk doing so."""
    ch = SCH.Chain(SC.STAB_EVENTS[0], weeks)
    ev = F.event(ch.event)
    n = S.nsteps_for(ev, ch.walk_end)
    assert n == (ch.horizon_days - SC.SCORE_LEAD_DAYS) * 24 // F.STEP_H
    assert n * F.STEP_H / 24 + SC.SCORE_LEAD_DAYS == ch.horizon_days
    # ...and the adapter-free arm rolls the entire thing from the ERA5 IC.
    assert ch.horizon_days * 24 // F.STEP_H == C.lead_days_for(weeks) * 4


def test_the_longest_rollout_in_the_sweep_is_twenty_weeks_of_fcn3():
    ch = SCH.Chain(SC.STAB_EVENTS[0], max(SC.STAB_WEEKS))
    assert ch.horizon_days == 140
    assert ch.horizon_days * 24 // F.STEP_H == 560            # adapter-free, ERA5 IC
    assert S.nsteps_for(F.event(ch.event), ch.walk_end) == 548  # adapter-fed, from 3 d


@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_a_chain_reads_its_own_weeks_init_frames(weeks, monkeypatch):
    """The bug the smoke run of job 1187 caught, pinned.

    ``walker.run_segment(parent_state=None)`` falls back to ``walker.initial_state``, which
    resolves the init frames through ``aconfig.WALKER_WEEKS`` - the ``AIRES_WEEKS``
    environment variable, 3 by default. This sweep spans four leads in ONE process, so
    every chain silently started from the week-3 init: the log printed
    "init 2021-05-31" and then rolled from 2021-06-07, a whole week late.

    ``Chain.inputs_path`` must therefore depend on the chain's own week and on nothing
    ambient, and ``stage_walk`` must hand it to ``run_segment`` explicitly.
    """
    ch = SCH.Chain(SC.STAB_EVENTS[0], weeks)
    assert f"week{weeks}/" in str(ch.inputs_path())
    assert ch.inputs_path() == A.gencast_inputs_path(ch.event, weeks=weeks)
    # ...and stays put whatever the ambient default says.
    monkeypatch.setattr(A, "WALKER_WEEKS", 3)
    assert f"week{weeks}/" in str(SCH.Chain(SC.STAB_EVENTS[0], weeks).inputs_path())
    if weeks != 3:
        assert ch.inputs_path() != A.gencast_inputs_path(ch.event)


def test_only_one_chain_would_have_been_right_by_accident():
    """Why the bug was worth a test rather than a fix: the default AIRES_WEEKS is 3, which
    is not one of the four leads here, so ALL EIGHT chains were wrong - and every one of
    them would have produced a clean-looking 28/42/56/70-day rollout from the wrong init."""
    assert 3 not in SC.STAB_WEEKS


def test_every_chain_is_distinct_and_the_shards_are_balanced():
    """Was "eight chains, one per GPU" - the FCN3 extension outgrew that. What still has
    to hold is that no two chains share a slot (they would overwrite each other's tree)
    and that no GPU is handed two more chains than another."""
    chs = SCH.chains()
    n = len(SC.STAB_EVENTS) * len(SC.STAB_WEEKS)
    assert len(chs) == n == 18
    assert len({(c.event, c.weeks) for c in chs}) == n
    assert len({(c.event, c.walker) for c in chs}) == n
    counts = [len(SCH.shard_of(chs, 8, s)) for s in range(8)]
    assert sum(counts) == n and max(counts) - min(counts) <= 1


def test_the_walker_slots_are_append_only():
    """Every artifact job 1189 wrote is addressed by SLOT - ``walkers/w0N/``,
    ``scores/w0N_lead03_cube.nc``, ``stab_w0N.json``. Inserting a lead ahead of 4/6/8/10
    (or sorting the tuple differently) re-points all of it at the wrong chain, silently,
    because nothing on disk records which lead a slot belonged to."""
    assert [SC.WEEKS_TO_WALKER[w] for w in (4, 6, 8, 10)] == [0, 1, 2, 3]
    assert SC.STAB_WEEKS[:4] == (4, 6, 8, 10)
    assert list(SC.STAB_WEEKS) == sorted(SC.STAB_WEEKS)


def test_the_two_events_are_opposite_regimes():
    """Not decoration: the second event is free wall-clock (the chains are serial) and it
    is what stops the answer being 'stable in June'. A cold event also exercises the
    negative tail of the observable."""
    fams = {F.event(e).family for e in SC.STAB_EVENTS}
    assert fams == {"heat", "cold"}


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_pruning_never_removes_a_state_something_still_needs(weeks):
    """Three claimants on a state: the next segment (its parent), the lead-3 d scoring
    task, and the final peak state the reduce stage reads. Deleting any of them turns a
    resumable chain into a re-walk (~2.2 h) or a missing score cube."""
    segs = SCH.chain_schedule(C.lead_days_for(weeks))
    n = len(segs)
    pinned = SCH.keep_states(segs)
    assert pinned == {1, n}
    seen: set[int] = set()
    for (k, _lead, _n) in segs:
        gone = set(SCH.prunable(segs, k))
        assert gone.isdisjoint(pinned), f"segment {k} would prune a pinned state"
        assert k not in gone and (k - 1) not in gone, "the parent chain was broken"
        assert gone.issubset(set(range(1, k - 1)))
        seen |= gone
    # Everything is released during the walk except the two pinned states and the final
    # segment's parent, which is still live when the loop ends; --stage prune takes that
    # one afterwards.
    assert seen == set(range(1, n + 1)) - pinned - {n - 1}
    assert (n - 1) not in seen


def test_pruning_is_disabled_when_keep_states_is_off():
    segs = SCH.chain_schedule(28)
    assert SCH.prunable(segs, 9, keep=0) == []


# --------------------------------------------------------------------------- #
# Diagnostics
#
# Item 1 (non-finite) and item 2 (physical bounds) are absolute and need no calibration.
# Items 3, 5 and 6 are the ones the job-1155 signature exercises.
# --------------------------------------------------------------------------- #
def _synth_diag(n_steps: int = 42, *, start="2021-06-07T12:00", amp: float = 8.0,
                drift: float = 0.0, base: float = 298.56, sd: float = 6.0,
                seed: int = 0) -> xr.Dataset:
    """A CONUS-shaped diagnostics cube with a controllable diurnal cycle and drift.

    12Z is the cold half of the CONUS day (04-07 local) and 00Z the warm half, which is
    the sign convention every diurnal quantity here inherits.
    """
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n_steps, freq="12h")
    lat = np.arange(24.0, 50.01, 0.25, dtype="float32")
    lon = np.arange(235.0, 294.01, 0.25, dtype="float32")
    warm = np.where(pd.DatetimeIndex(t).hour == 0, 0.5, -0.5)   # 00Z warm, 12Z cold
    ramp = np.arange(n_steps) / max(n_steps - 1, 1)    # 0 -> 1 over the whole series
    mean = base + amp * warm + drift * ramp
    # A deterministic spatial pattern normalised to unit sd, so the spatial-sd item is
    # exactly controllable and the diurnal difference cancels it identically.
    pat = np.linspace(-1.0, 1.0, lat.size)[:, None] * np.ones(lon.size)[None, :]
    pat = pat / pat.std()
    field = mean[:, None, None] + sd * pat[None, :, :]
    return xr.Dataset(
        {"2m_temperature": (("batch", "time", "lat", "lon"),
                            field[None].astype("float32"))},
        coords=dict(batch=[0], time=t, lat=lat, lon=lon))


def test_diurnal_amplitude_is_recovered_from_a_healthy_cube():
    d = _synth_diag(amp=8.0)
    amps = D.diurnal_amplitudes(D.conus_mean_t2m(d))
    assert len(amps) == d.sizes["time"] - 1
    assert np.allclose(amps["amp"].values, -8.0, atol=1e-3)   # 12Z minus 00Z: negative


def test_the_job_1155_diurnal_collapse_is_detected():
    """The measured signature: 00Z equal to 12Z to 0.03 K. A walker whose diurnal cycle
    has gone flat is not producing an atmosphere, whatever its mean looks like."""
    broken = _synth_diag(amp=0.03 / 2)          # peak-to-peak 0.03 K
    amps = D.diurnal_amplitudes(D.conus_mean_t2m(broken))
    assert np.abs(amps["amp"].values).max() < 0.05
    healthy = _synth_diag(amp=8.0)
    ratio = (np.abs(amps["amp"].values).mean()
             / np.abs(D.diurnal_amplitudes(D.conus_mean_t2m(healthy))["amp"].values).mean())
    assert ratio < 0.01, "a 250x amplitude collapse must not look like a small deviation"


def test_the_job_1155_cold_drift_is_detected():
    """26 K cold over 21 days. Reported as drift from the chain's own first frame, so the
    diagnostic exists with or without a climatology on the box."""
    d = _synth_diag(n_steps=42, amp=8.0, drift=-26.0)
    ser = D.conus_series(d)
    assert D.drift_from_start(ser["t2m_mean"]) == pytest.approx(-26.0, abs=0.8)
    healthy = D.conus_series(_synth_diag(n_steps=42, amp=8.0, drift=0.0))
    assert abs(D.drift_from_start(healthy["t2m_mean"])) < 0.8


def test_drift_is_phase_matched_against_the_diurnal_cycle():
    """A raw last-minus-first carries +-half the diurnal range purely from whether the
    series has an odd or an even number of frames - 8 K of pure bookkeeping on a
    diagnostic whose signal of record is 26 K. Both parities must agree."""
    even = D.conus_series(_synth_diag(n_steps=42, amp=8.0, drift=-26.0))["t2m_mean"]
    odd = D.conus_series(_synth_diag(n_steps=43, amp=8.0, drift=-26.0))["t2m_mean"]
    assert float(even.iloc[-1] - even.iloc[0]) != pytest.approx(-26.0, abs=1.0)  # the trap
    assert D.drift_from_start(even) == pytest.approx(-26.0, abs=0.8)
    assert D.drift_from_start(odd) == pytest.approx(-26.0, abs=0.8)


def test_spatial_sd_sees_a_homogenised_field():
    """Item 6. A field that has collapsed to a single value still has a perfectly ordinary
    mean; only its spatial structure is gone."""
    normal = D.conus_series(_synth_diag(sd=6.0))["t2m_sd"]
    flat = D.conus_series(_synth_diag(sd=0.0))["t2m_sd"]
    assert normal.mean() == pytest.approx(6.0, rel=0.15)
    assert flat.max() < 1e-3


def test_non_finite_fractions_are_reported_per_variable():
    d = _synth_diag(n_steps=4)
    assert D.nonfinite_fractions(d) == {"2m_temperature": 0.0}
    bad = d.copy(deep=True)
    v = bad["2m_temperature"].values
    v[0, 0, :2, :] = np.nan
    frac = D.nonfinite_fractions(bad)["2m_temperature"]
    assert 0 < frac < 1
    assert frac == pytest.approx(2 / (4 * d.sizes["lat"]), rel=1e-6)


def test_the_land_mask_is_not_a_divergence():
    """`sea_surface_temperature` is NaN over land by construction - measured at exactly
    0.3389 of the global grid on all 18 known-good Gate 3 states checked, bit-identical
    across both events, three walkers and three leads. Gating item 1 on "any non-finite
    value" marked the smoke run's perfectly healthy lead-3 d segment as diverged, and would
    have reported "GenCast stable through week (none)" for all eight chains."""
    land = {"sea_surface_temperature": 0.3389, "2m_temperature": 0.0}
    assert D.hard_nonfinite(land) == 0.0
    # ...but a mask that GROWS is a real blow-up, and growth is what is gated.
    grown = {"sea_surface_temperature": 0.51, "2m_temperature": 0.0}
    assert D.hard_nonfinite(grown, baseline=land) > 0.1
    assert D.hard_nonfinite(grown, baseline=grown) == 0.0
    # Everything else fails outright on the first non-finite value.
    assert D.hard_nonfinite({"2m_temperature": 1e-9}) > 0
    assert "sea_surface_temperature" in D.NONFINITE_STATIC


def test_the_humidity_floor_is_measured_not_assumed():
    """The plan specified ``q >= 0``. Measured over 18 known-good Gate 3 states, healthy
    GenCast reaches -1.28e-3 kg/kg with up to 0.19% of points negative from the FIRST
    segment - the ordinary small-negative-humidity artifact of an ML weather model. An
    absolute zero floor is not a divergence detector, it is an alarm that is always on."""
    lo, hi = D.BOUNDS["specific_humidity"]
    assert lo < 0, "q >= 0 fires on every healthy walker; see the measurement"
    assert lo <= -1.28e-3 * 2, "the floor must clear the worst measured value with margin"
    assert lo > -0.05, "...but must still catch a genuinely diverged humidity field"
    assert hi == 0.1


def test_physical_bounds_pass_on_a_healthy_cube_and_fail_on_a_diverged_one():
    """Item 2 is absolute and deliberately generous: it is not a skill test, it is the
    line between an atmosphere and a diverged array."""
    d = _synth_diag(n_steps=4)
    assert D.bound_violations(d) == {}
    cold = _synth_diag(n_steps=4, base=120.0)         # below the 180 K floor
    assert "2m_temperature" in D.bound_violations(cold)


def test_the_t2m_floor_clears_the_real_atmosphere():
    """The smoke run tripped a 200 K global floor at 6 d lead with a minimum of 197.8 K -
    the Antarctic plateau in austral winter, not a divergence. ERA5's OWN global minimum
    over these chains' dates is 201.0 K and healthy Gate 3 walkers reach 194.4 K, so the
    floor has to sit below both with margin. A bound that the observations violate is not
    a bound, it is a false alarm with a physical-sounding name."""
    lo, hi = D.BOUNDS["2m_temperature"]
    assert lo < 194.4, "healthy walkers reach 194.4 K globally"
    assert lo < 201.0 - 15, "ERA5 itself reaches 201.0 K; leave real margin"
    assert hi > 324.0, "healthy walkers reach 324.0 K globally"
    assert lo > 150.0 and hi < 400.0, "...but it must still be a divergence detector"


def test_a_threshold_change_is_honoured_by_records_already_on_disk():
    """Thresholds are applied at REDUCE time, to stored measurements - never baked into a
    walk-time artifact. The smoke run proved why: ``BOUNDS["specific_humidity"]`` was
    corrected after the segments were written, and a record carrying a pre-computed
    verdict kept reporting a healthy chain as diverged. A measurement is durable; a
    verdict on it is not."""
    healthy = {"specific_humidity": {"min": -1.28e-3, "max": 0.021}}
    assert D.violations_from_ranges(healthy) == {}          # under the measured floor
    diverged = {"specific_humidity": {"min": -0.4, "max": 0.021}}
    assert "specific_humidity" in D.violations_from_ranges(diverged)
    assert D.violations_from_ranges({"not_a_bounded_var": {"min": 0, "max": 1}}) == {}
    assert D.violations_from_ranges({"2m_temperature":
                                      {"min": float("nan"), "max": float("nan")}})


def test_field_ranges_ignores_the_land_mask_nans():
    """``field_ranges`` reduces over FINITE values only, so SST's land mask cannot turn
    every range into NaN and every chain into a failure."""
    d = _synth_diag(n_steps=4)
    v = d["2m_temperature"].values
    v[0, 0, :2, :] = np.nan
    r = D.field_ranges(d)["2m_temperature"]
    assert np.isfinite(r["min"]) and np.isfinite(r["max"])
    assert D.violations_from_ranges(D.field_ranges(d)) == {}


def test_the_bounds_are_the_documented_ones():
    """Written down so a later tightening is a visible edit rather than a quiet
    re-definition of what 'stable' meant in the published figure."""
    assert D.BOUNDS["2m_temperature"] == (180.0, 340.0)
    assert D.BOUNDS["mean_sea_level_pressure"] == (87000.0, 109000.0)
    assert D.BOUNDS["specific_humidity"] == (-5.0e-3, 0.1)
    assert D.BOUNDS["10m_u_component_of_wind"] == (-200.0, 200.0)


# --------------------------------------------------------------------------- #
# The zonal spectrum (item 7) - global only
# --------------------------------------------------------------------------- #
def _synth_z500(*, hi_amp: float = 0.0, wavenumber: int = 40) -> xr.DataArray:
    lat = np.linspace(-90, 90, 181, dtype="float32")
    lon = np.arange(0, 360, 0.5, dtype="float32")
    k5 = 5 * np.cos(np.deg2rad(5 * lon))[None, :] * np.ones(lat.size)[:, None]
    hi = hi_amp * np.cos(np.deg2rad(wavenumber * lon))[None, :] * np.ones(lat.size)[:, None]
    return xr.DataArray(54000.0 + k5 + hi, dims=("lat", "lon"),
                        coords=dict(lat=lat, lon=lon))


def test_a_clean_planetary_wave_has_almost_no_high_wavenumber_power():
    frac = D.zonal_high_wavenumber_fraction(_synth_z500(hi_amp=0.0))
    assert frac < 1e-6


def test_grid_scale_noise_shows_up_above_the_cutoff():
    """The classic AI-weather instability signature, and the plan's bet on which item
    fires first at long lead."""
    frac = D.zonal_high_wavenumber_fraction(_synth_z500(hi_amp=5.0, wavenumber=40))
    assert frac == pytest.approx(0.5, abs=0.05)
    assert D.zonal_high_wavenumber_fraction(_synth_z500(hi_amp=1.0, wavenumber=40)) < frac


def test_the_spectrum_cutoff_and_band_are_the_documented_ones():
    assert D.HI_WAVENUMBER == 20
    assert D.SPECTRUM_LAT == (30.0, 60.0)
    assert D.SPECTRUM_BAND_DEG == 5.0


def test_the_spectrum_refuses_a_conus_crop():
    """It MUST read state.nc, not diag.nc: the CONUS crop spans 59 degrees of longitude
    (config.LON = slice(235, 294)), far too little for a zonal wavenumber decomposition,
    and a silent partial-domain FFT would produce a plausible number that means nothing."""
    conus = _synth_z500().sel(lon=slice(235, 294))
    with pytest.raises(ValueError, match="global"):
        D.zonal_high_wavenumber_fraction(conus)


# --------------------------------------------------------------------------- #
# Reduce is idempotent across a prune
# --------------------------------------------------------------------------- #
def test_reduce_carries_forward_what_it_can_no_longer_compute(tmp_path, monkeypatch):
    """`--stage prune` deletes the pre-crop Zarrs, which are the ONLY source of the FCN3
    global items. A reduce run afterwards must not overwrite them with a panel that
    silently lacks those columns.

    This is not hypothetical: it happened during job 1189's analysis. Re-running reduce
    after prune dropped 362 rows from the CSV and erased `global_z500` and
    `z500_hi_wavenumber_frac` for both FCN3 arms - and those are exactly the diagnostics
    that the same run had just shown a CONUS-only panel cannot do without.
    """
    import json
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(SC.STAB_EVENTS[0], 4)
    p = ch.result_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"panels": {"fcn3_era5": [
        {"lead_days": 3.0, "t2m_anom_conus": 1.0, "global_z500": 55000.0,
         "z500_hi_wavenumber_frac": 0.003}]}}))

    fresh = {"fcn3_era5": [{"lead_days": 3.0, "t2m_anom_conus": 1.0}]}   # Zarr now gone
    merged = RD._merge_panels(fresh, ch)
    row = merged["fcn3_era5"][0]
    assert row["global_z500"] == 55000.0
    assert row["z500_hi_wavenumber_frac"] == 0.003
    assert row["t2m_anom_conus"] == 1.0, "a freshly computed value must still win"


def test_merge_never_overwrites_a_freshly_computed_value(tmp_path, monkeypatch):
    """Carrying forward is for values that are GONE, not for values that changed. A
    threshold correction or a bug fix has to reach the panel."""
    import json
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(SC.STAB_EVENTS[0], 4)
    ch.result_path().parent.mkdir(parents=True, exist_ok=True)
    ch.result_path().write_text(json.dumps({"panels": {"gencast": [
        {"lead_days": 3.0, "bounds": 1.0}]}}))
    merged = RD._merge_panels({"gencast": [{"lead_days": 3.0, "bounds": 0.0}]}, ch)
    assert merged["gencast"][0]["bounds"] == 0.0


def test_merge_is_a_no_op_on_a_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(SC.STAB_EVENTS[0], 4)
    fresh = {"gencast": [{"lead_days": 3.0, "bounds": 0.0}]}
    assert RD._merge_panels(fresh, ch) == fresh


def test_prune_refuses_while_a_score_cube_is_still_missing(tmp_path, monkeypatch, capsys):
    """Prune deletes the lead-3 d states the adapter-fed arm launches from. Running it
    before scoring is finished turns a 20-minute gap into a re-walk - which is exactly
    what happened during job 1189's analysis, when the scorer had to be re-run with an
    extra channel after prune had already taken the states."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    chs = SCH.chains([SC.STAB_EVENTS[0]], [4])
    assert STG.stage_prune(chs) == 1
    assert "REFUSING" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The extension's one real failure mode: a verdict for a lead nobody walked
# --------------------------------------------------------------------------- #
def _survival_frame(rows) -> pd.DataFrame:
    """A minimal ``stability.csv`` - (model, event, weeks, checkpoint_day, metric,
    status, reached_peak) is all ``_model_survival`` reads."""
    return pd.DataFrame([dict(event=e, weeks=w, lead_days=C.lead_days_for(w), model=m,
                              checkpoint_day=d, metric=met, value=v, band="",
                              status=st, extrapolated=0, reached_peak=rp, valid_time="")
                         for (m, e, w, d, met, v, st, rp) in rows])


def test_a_finished_chain_can_be_re_walked_after_a_prune(tmp_path, monkeypatch, capsys):
    """Adding a lead re-runs every shard, and a cached chain is supposed to cost seconds.

    Job 1189's chains were pruned, so their final ``state.nc`` is gone while every
    ``stab.json`` remains. Verifying the landing from the state alone would fail the whole
    pool on chains that did nothing wrong - which is what a resubmit after the FCN3
    extension would have hit on all eight of them.
    """
    import json
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(SC.STAB_EVENTS[0], 4)
    for (k, lead, n) in ch.schedule:                    # records only; states pruned
        d = SC.segment_dir(ch.event, ch.walker, k)
        d.mkdir(parents=True, exist_ok=True)
        (d / "stab.json").write_text(json.dumps(
            dict(step=k, lead_days=lead, n_steps=n,
                 valid_time=str(ch.valid_at(lead)))))
    assert not ch.state_path(ch.schedule[-1][0]).exists()
    assert STG.stage_walk([ch], 0, 1, prune_stale_states=False) == 0
    out = capsys.readouterr().out
    assert "landing verified from" in out
    assert "landed exactly on the peak" in out


def test_a_lead_the_walker_never_reached_is_not_a_gencast_verdict():
    """THE trap the FCN3-only extension introduces.

    At an extension lead the walker rolls 3 d - one clean checkpoint - and both FCN3 arms
    roll to the peak. Counted naively, that one clean checkpoint reads as a clean chain,
    ``_all_clean_through`` walks straight past the last week that was actually tested, and
    the headline reports "GenCast stable through week 20" off three days of rollout. The
    walker's real answer here is week 6.
    """
    rows = []
    for ev in SC.STAB_EVENTS:
        for w in (4, 6):                       # walked, clean
            rows.append(("gencast", ev, w, 3.0, "bounds", 0.0, "ok", 1))
        rows.append(("gencast", ev, 8, 30.0, "bounds", 1.0, "FAIL", 1))   # walked, broke
        for w in (12, 20):                     # extension: 3 d only, and clean at 3 d
            rows.append(("gencast", ev, w, 3.0, "bounds", 0.0, "ok", 0))
    df = _survival_frame(rows)

    surv = RD._model_survival(df, "gencast")
    assert sorted(surv) == [4, 6, 8], "an unwalked lead must be absent, not clean"
    assert RD._all_clean_through(surv) == 6
    assert RD.untested_weeks(df, "gencast") == [12, 20]


def test_the_scorer_is_credited_at_every_lead_it_rolled():
    """The other half: the FCN3 arms DO roll the full horizon at an extension lead, so
    their verdict must extend there. A rule that keyed off the lead rather than off what
    the model actually rolled would throw the extension's whole result away."""
    rows = [(m, ev, w, 3.0, "bounds", 0.0, "ok", 1)
            for m in ("fcn3_era5", "fcn3_adapter")
            for ev in SC.STAB_EVENTS for w in SC.STAB_WEEKS]
    df = _survival_frame(rows)
    for m in ("fcn3_era5", "fcn3_adapter"):
        assert sorted(RD._model_survival(df, m)) == list(SC.STAB_WEEKS)
        assert RD._all_clean_through(RD._model_survival(df, m)) == max(SC.STAB_WEEKS)
        assert RD.untested_weeks(df, m) == []


def test_an_fcn3_cube_that_stops_short_is_not_a_verdict_either():
    """The other way coverage fails, and the one that only appears at these lengths: the
    arm rolled, but not all the way. A week-20 free rollout is 560 FCN3 steps; a cube that
    ends at 200 would otherwise contribute a full set of clean checkpoints out to day 150
    and read as a clean 140-day rollout."""
    rows = [("fcn3_era5", ev, 20, d, "bounds", 0.0, "ok", 0)
            for ev in SC.STAB_EVENTS for d in (3.0, 30.0, 60.0)]
    rows += [("fcn3_era5", ev, 4, d, "bounds", 0.0, "ok", 1)
             for ev in SC.STAB_EVENTS for d in (3.0, 27.0)]
    df = _survival_frame(rows)
    assert RD.untested_weeks(df, "fcn3_era5") == [20]
    assert RD._all_clean_through(RD._model_survival(df, "fcn3_era5")) == 4


def test_a_csv_from_before_the_extension_still_reduces():
    """``reached_peak`` is a new column. Job 1189's CSV does not have it, and the whole
    point of ``_merge_panels`` is that old artifacts stay readable - so its absence must
    mean "everything was walked", which is exactly what was true then."""
    df = _survival_frame([("gencast", SC.STAB_EVENTS[0], 4, 3.0, "bounds", 0.0, "ok", 1)])
    df = df.drop(columns=["reached_peak"])
    assert RD.untested_weeks(df, "gencast") == []
    assert sorted(RD._model_survival(df, "gencast")) == [4]


def test_the_figure_drops_exactly_the_curves_the_headline_drops():
    """One rule, two consumers. A figure that drew the walker's lone 3-day marker beside a
    140-day FCN3 curve would say what the headline is careful not to."""
    from astab import plots as PL
    rows = [("gencast", SC.STAB_EVENTS[0], 4, 3.0, "bounds", 0.0, "ok", 1),
            ("gencast", SC.STAB_EVENTS[0], 20, 3.0, "bounds", 0.0, "ok", 0)]
    df = _survival_frame(rows)
    assert set(RD.untested_weeks(df, "gencast")) == {20}
    assert "untested_weeks" in PL.stage_figs.__globals__


@pytest.mark.parametrize("weeks", EXTENSION_WEEKS)
def test_an_extension_chain_pins_its_only_state_and_prunes_nothing(weeks):
    """A one-segment walk has one state, and it is the adapter-fed arm's launch state.
    ``keep_states`` must pin it (it is both segment 1 and the final segment) and
    ``prunable`` must return nothing - a rolling-prune rule that fired here would delete
    the only thing the walk produced."""
    ch = SCH.Chain(SC.STAB_EVENTS[0], weeks)
    segs = ch.schedule
    assert len(segs) == 1
    assert SCH.keep_states(segs) == {1}
    assert SCH.prunable(segs, 1) == []


@pytest.mark.parametrize("weeks", EXTENSION_WEEKS)
def test_the_era5_reference_covers_the_horizon_not_the_walk(weeks):
    """``refs.ref_times`` reads ``full_schedule``. Off ``schedule`` it would build three
    days of ERA5 for a 140-day chain, and every FCN3 checkpoint past day 3 would have no
    truth frame - which ``maps`` reports as "no data" rather than as an error."""
    from astab.refs import ref_times
    ch = SCH.Chain(SC.STAB_EVENTS[0], weeks)
    times = set(ref_times(ch.event))
    assert ch.init in times and ch.peak in times
    assert ch.walk_end in times
    for (_k, lead, _n) in ch.full_schedule:
        assert ch.valid_at(lead) in times


def test_the_extension_leads_write_where_no_frozen_experiment_reads():
    """The adapter-free arm's cube is written by ``fcn3/run_fcn3.py`` unmodified, into its
    own week-namespaced tree. Week 12..20 must therefore land beside week3 - the frozen
    head-to-head - and never in it."""
    ev = SC.STAB_EVENTS[0]
    frozen = SC.free_cube_path(ev, 3)
    new = {SC.free_cube_path(ev, w) for w in EXTENSION_WEEKS}
    assert frozen not in new
    assert len(new) == len(EXTENSION_WEEKS)
    for p in new:
        assert p.parent != frozen.parent


def test_coordinate_alignment_nans_are_not_a_divergence():
    """Two single-level channels in one Dataset make xarray union the `level` axis and
    fill the cross terms with NaN. Asking FCN3 for z500 and t50 gives a cube on
    level=[50,500] where geopotential@50 and temperature@500 are 100% NaN - a coordinate
    artifact that item 1 counted, reporting all 16 FCN3 rollouts as non-finite from 3 d
    and turning "FCN3 stable through week 10" into "no week clean"."""
    lat = np.arange(24.0, 26.01, 0.25, dtype="float32")
    lon = np.arange(235.0, 237.01, 0.25, dtype="float32")
    shape = (2, lat.size, lon.size)
    z = np.full(shape, np.nan); z[1] = 5.5e4          # geopotential: only level 500
    t = np.full(shape, np.nan); t[0] = 210.0          # temperature: only level 50
    ds = xr.Dataset(
        {"geopotential": (("level", "lat", "lon"), z),
         "temperature": (("level", "lat", "lon"), t)},
        coords=dict(level=[50, 500], lat=lat, lon=lon))

    assert D.nonfinite_fractions(ds) == {"geopotential": 0.0, "temperature": 0.0}
    assert D.hard_nonfinite(D.nonfinite_fractions(ds)) == 0.0
    assert D.violations_from_ranges(D.field_ranges(ds)) == {}


def test_a_partially_nan_field_is_still_a_real_signal():
    """Only ENTIRELY empty slices are dropped. A field that is half NaN is the failure
    item 1 exists to catch and must survive the cleanup untouched."""
    lat = np.arange(24.0, 26.01, 0.25, dtype="float32")
    lon = np.arange(235.0, 237.01, 0.25, dtype="float32")
    a = np.full((2, lat.size, lon.size), 300.0)
    a[0, :2, :] = np.nan                       # a real hole, not an empty level
    ds = xr.Dataset({"temperature": (("level", "lat", "lon"), a)},
                    coords=dict(level=[50, 500], lat=lat, lon=lon))
    assert D.hard_nonfinite(D.nonfinite_fractions(ds)) > 0


def test_the_scorer_is_checked_where_the_walker_actually_broke():
    """Job 1189's two SILENT GenCast divergences were `temperature` at 50 hPa. FCN3's
    production cube is CONUS-cropped and carries neither that level nor any geopotential,
    so a claim that FCN3 stayed physical to 70 d rests on a panel that could not have seen
    the failure mode GenCast actually had. `t50` exists so the scorer is tested there."""
    assert F.FCN3_TO_GENCAST["t50"] == ("temperature", 50)
    assert F.FCN3_TO_GENCAST["z500"] == ("geopotential", 500)
    assert STG._fcn3_extra_vars_env()["FCN3_EXTRA_VARS"] == "z500,t50"
    # ...and both map onto variables the bounds table actually gates.
    for v in ("temperature", "geopotential", "2m_temperature"):
        assert v in D.BOUNDS


# --------------------------------------------------------------------------- #
# Bands
# --------------------------------------------------------------------------- #
def test_a_band_beyond_the_calibrated_range_is_marked_extrapolated():
    """The Gate 3 walkers reach 21 d and no further. Past that the bands are an
    extrapolation, and the figure has to say so rather than draw them like measurements."""
    band = D.Band(metric="t2m_anom_conus", lo=-4.0, hi=4.0, n=112, max_lead_days=21.0)
    assert band.status_at(9.0)[1] is False        # measured
    assert band.status_at(45.0)[1] is True        # extrapolated
    assert band.status_at(9.0, 0.0)[0] == "ok"
    assert band.status_at(9.0, 99.0)[0] == "FAIL"


def test_a_metric_with_no_baseline_is_reported_not_gated():
    """Item 4 (global means vs the ERA5 reference) has no in-repo baseline at all. It
    ships as a number, and a number with an invented threshold is worse than no gate."""
    assert "global_t2m" in D.UNGATED
    assert D.status_for("global_t2m", 1e9, bands={}) == ("report", False)
