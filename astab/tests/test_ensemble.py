"""Contract tests for the ensemble follow-up (``astab/ensemble.py`` + the member chain).

The follow-up rolls 100 GenCast walkers at ONE lead, in the same tree, at the same lead,
with the same walker SLOT as a chain job 1189 already finished and froze. That overlap is
the whole risk profile: nothing here can produce a wrong number without first producing a
wrong PATH, and the frozen sweep's artifacts are irreplaceable (their states were pruned
and their FCN3 Zarrs cost 3.4 GPU-h).

Four failure classes, each pinned below:

* **The frozen tree.** ``stage_walk`` deletes a stale or partial segment directory with
  ``ignore_errors=True``. A week-8 member chain's ``walker`` is slot 2, so the pre-member
  code path (``SC.segment_dir(event, ch.walker, k)``) resolves to job 1189's
  ``walkers/w02/stepKK`` and would have deleted it silently. Every ``ens_*`` path, the
  raising FCN3 helpers and the routed ``result_path`` exist for the same reason.
* **The shard map.** ``i % nshards`` over a chain list that must be IDENTICAL in 32
  processes. A member selected by two shards is rolled twice; one selected by none is
  silently missing from a denominator.
* **The classification.** First-out-of-bounds and DIVERGED are different dates by design
  (job 1189's PNW@wk8: 42 d marginal, 45 d diverged), an unfinished member must never
  read as clean, and an infinite severity is a real state rather than a parse error.
* **The interval.** k = 0 and k = n are the interesting answers here, and they are exactly
  where Wald degenerates.

No GPU, no model, no file in ``runs/``.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aires import aconfig as A
from astab import diag as D
from astab import ensemble as EN
from astab import record as REC
from astab import run as RUN
from astab import schedule as SCH
from astab import sconfig as SC
from astab import stages as STG
from gencast_s2s import config as C

EV = SC.STAB_EVENTS[0]                      # PNW_HeatDome_2021
WK = 8                                      # the lead the follow-up runs at


# --------------------------------------------------------------------------- #
# The frozen tree: nothing a member chain does may reach it
# --------------------------------------------------------------------------- #
def test_a_member_chain_and_the_frozen_chain_share_a_lead_and_a_slot():
    """The premise of every other test in this file. They are the same lead, so they have
    the same walker slot and the same init - and only the paths keep them apart."""
    frozen = SCH.Chain(EV, WK)
    m = SCH.Chain(EV, WK, 2)
    assert frozen.walker == m.walker == SC.WEEKS_TO_WALKER[WK] == 2
    assert frozen.init == m.init and frozen.peak == m.peak
    assert frozen.schedule == m.schedule
    assert frozen.state_path(3) != m.state_path(3)
    assert frozen.record_path(3) != m.record_path(3)
    assert frozen.result_path() != m.result_path()


@pytest.mark.parametrize("step", [1, 3, 19])
def test_every_member_path_is_disjoint_from_the_frozen_sweeps(step):
    """Not merely different - DISJOINT, including as parents. ``walkers/`` and ``ens/``
    diverge at the first segment below the event directory, so no member path can be
    inside a frozen directory and no frozen path inside a member's."""
    frozen = {SCH.Chain(EV, w).segment_dir(step) for w in SC.STAB_WEEKS}
    frozen |= {SCH.Chain(EV, w).state_path(step) for w in SC.STAB_WEEKS}
    frozen |= {SCH.Chain(EV, w).result_path() for w in SC.STAB_WEEKS}
    for m in (0, 2, 99):
        ch = SCH.Chain(EV, WK, m)
        mine = {ch.segment_dir(step), ch.state_path(step), ch.diag_path(step),
                ch.record_path(step), ch.result_path()}
        assert mine.isdisjoint(frozen)
        for p in mine:
            assert "walkers" not in p.parts, p
            assert "ens" in p.parts, p
            for f in frozen:
                assert f not in p.parents and p not in f.parents


def test_the_stale_segment_rmtree_goes_through_the_chain(tmp_path, monkeypatch, capsys):
    """THE regression. ``stage_walk`` ``rmtree``s a stale segment with
    ``ignore_errors=True``; before ``Chain.segment_dir`` that call was
    ``SC.segment_dir(ch.event, ch.walker, k)``, and a week-8 MEMBER chain's walker is
    slot 2 - so the first stale segment of member 0 would have deleted job 1189's frozen
    ``walkers/w02/step03/`` without raising, without printing, and without a way back
    (those chains' states are pruned and their records are the only copy).

    The stale state below is a real one: a valid time the schedule does not use, which is
    exactly what a resumed chain whose ``chain_schedule`` changed would leave behind.
    """
    import aires.walker as W

    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(EV, WK, 0)

    frozen = SC.segment_dir(EV, ch.walker, 3)      # runs/astab/<ev>/walkers/w02/step03
    frozen.mkdir(parents=True)
    (frozen / "stab.json").write_text('{"job": 1189}')
    (frozen / "diag.nc").write_bytes(b"not really a cube")

    d = ch.segment_dir(3)
    d.mkdir(parents=True)
    # A state whose valid time is a week off the schedule -> `_segment_status` == "stale".
    xr.Dataset(coords=dict(time=pd.DatetimeIndex(["2021-01-01"]))).to_netcdf(
        d / "state.nc")
    (d / "diag.nc").write_bytes(b"x")

    seen = []

    def _no_model(*_a, **_k):
        return lambda: {}

    def _fake_segment(get_bundle, event, walker, step, n_steps, **kw):
        seen.append((walker, step))
        raise RuntimeError("stop before the model would load")

    monkeypatch.setattr(W, "lazy_bundle", _no_model)
    monkeypatch.setattr(W, "run_segment", _fake_segment)

    assert STG.stage_walk([ch], 0, 1, prune_stale_states=True, max_segments=3) == 1

    assert frozen.exists(), "stage_walk deleted the frozen sweep's segment"
    assert (frozen / "stab.json").read_text() == '{"job": 1189}'
    assert not (d / "state.nc").exists(), "the member's own stale segment was NOT cleared"
    # ...and it rolled with the MEMBER index, not the walker slot.
    assert seen and seen[0][0] == 0 == ch.member
    assert "STALE" in capsys.readouterr().err


def test_a_failing_member_does_not_take_its_siblings_down(tmp_path, monkeypatch):
    """One bad member is 1% of a 100-member result; killing the three siblings sharing its
    GPU turns that into 4%. The frozen sweep keeps fail-fast - 18 irreplaceable chains -
    and the shard still exits non-zero so the failure is not silent."""
    import aires.walker as W

    monkeypatch.setattr(SC, "RUNS", tmp_path)
    chs = [SCH.Chain(EV, WK, m) for m in (0, 1, 2)]
    tried = []

    def _fake_segment(get_bundle, event, walker, step, n_steps, **kw):
        tried.append(walker)
        raise RuntimeError(f"member {walker} exploded")

    monkeypatch.setattr(W, "lazy_bundle", lambda *a, **k: (lambda: {}))
    monkeypatch.setattr(W, "run_segment", _fake_segment)
    assert STG.stage_walk(chs, 0, 1, prune_stale_states=False, max_segments=1) == 1
    assert tried == [0, 1, 2], "a failed member must not stop the shard"


def test_the_fcn3_helpers_refuse_a_member_chain():
    """They are keyed on (event, walker) or (event, weeks) with NO member, so 100 members
    would name ONE cube, ONE Zarr and ONE IC - the frozen sweep's - and ``stage_prune``
    ``rmtree``s the Zarrs. Raising is what makes ``--ens``'s refusal unroutable-around."""
    ch = SCH.Chain(EV, WK, 7)
    for name in ("score_cube_path", "score_zarr_path", "fcn3_ic_path",
                 "fcn3_free_cube_path", "fcn3_free_zarr_path"):
        with pytest.raises(ValueError, match="member chain"):
            getattr(ch, name)()
    # ...and the frozen chain's are untouched.
    frozen = SCH.Chain(EV, WK)
    assert frozen.score_cube_path().name == "w02_lead03_cube.nc"


def test_result_path_routes_rather_than_raising():
    """Deliberately NOT symmetric with the FCN3 helpers. ``reduce.stage_reduce`` writes
    ``result_path()`` unconditionally, so a stray reduce over member chains must land in
    the ens tree rather than overwrite job 1189's ``stab_w02.json``."""
    ch = SCH.Chain(EV, WK, 7)
    assert ch.result_path() == SC.ens_result_path(EV, WK, 7)
    assert ch.result_path().name == "stab_m007.json"
    assert SCH.Chain(EV, WK).result_path().name == "stab_w02.json"


def test_every_ens_artifact_hangs_off_the_two_roots():
    """``sconfig`` is the only place the tree is decided. A path built anywhere else is a
    path ``--stage prune`` and the sync manifest do not know about."""
    for p in (SC.ens_dir(EV, WK), SC.ens_member_dir(EV, WK, 3),
              SC.ens_segment_dir(EV, WK, 3, 1), SC.ens_state_path(EV, WK, 3, 1),
              SC.ens_diag_path(EV, WK, 3, 1), SC.ens_record_path(EV, WK, 3, 1),
              SC.ens_result_path(EV, WK, 3), SC.ens_csv_path(EV, WK),
              SC.ens_members_csv_path(EV, WK), SC.ens_summary_path(EV, WK)):
        assert SC.RUNS in p.parents, p
    assert SC.FIGS in SC.ens_fig_path(EV, WK).parents
    # The week is zero-padded, so wk08 sorts before wk10 in any listing.
    assert "wk08" in str(SC.ens_dir(EV, 8)) and "wk10" in str(SC.ens_dir(EV, 10))
    assert "m003" in str(SC.ens_member_dir(EV, WK, 3))


def test_the_ens_roots_move_with_the_env_overrides(tmp_path, monkeypatch):
    """``AIRES_STAB_RUNS``/``AIRES_STAB_FIGS`` exist so a second sweep can be run without
    touching the one on disk. The ensemble tree has to honour them too, or the tests here
    would be writing into the real run tree."""
    monkeypatch.setattr(SC, "RUNS", tmp_path / "r")
    monkeypatch.setattr(SC, "FIGS", tmp_path / "f")
    assert (tmp_path / "r") in SC.ens_state_path(EV, WK, 0, 1).parents
    assert (tmp_path / "f") in SC.ens_fig_path(EV, WK).parents


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
def test_the_seed_is_the_member_for_a_member_and_the_slot_for_the_frozen_eighteen():
    """``walker.segment_key(index, step)`` is what makes two members two different
    GenCast draws. Handing it ``ch.walker`` instead would give all 100 members of a lead
    the SAME noise stream: 100 identical trajectories, a survival curve of 0% or 100%,
    and nothing on disk saying why."""
    for w in SC.STAB_WEEKS:
        ch = SCH.Chain(EV, w)
        assert ch.seed == ch.walker == SC.WEEKS_TO_WALKER[w]
    for m in (0, 1, 2, 41, 99):
        assert SCH.Chain(EV, WK, m).seed == m
    assert len({SCH.Chain(EV, WK, m).seed for m in range(100)}) == 100


def test_member_two_is_job_1189s_control_at_week_eight():
    """Member 2 at week 8 draws walker slot 2's stream at the same init and the same
    checkpoint, which IS job 1189's PNW@wk8 chain. That makes it a DETERMINISM CONTROL,
    not an independent draw - and the headline says so rather than quietly counting it
    twice. It is also the one member whose final state is kept."""
    frozen = SCH.Chain(EV, WK)
    m2 = SCH.Chain(EV, WK, 2)
    assert m2.seed == frozen.seed == 2
    assert m2.init == frozen.init
    assert m2.inputs_path() == frozen.inputs_path()
    assert EN.JOB_1189 == dict(event=EV, weeks=WK, member=2, first_oob=42.0,
                               diverged=45.0, mode="loud")
    assert 2 in SC.ENS_KEEP_STATE_MEMBERS


def test_a_member_chain_reads_its_own_weeks_init_frames(monkeypatch):
    """The job-1187 trap, at the member. ``run_segment(parent_state=None)`` resolves the
    init through ``aconfig.WALKER_WEEKS`` (AIRES_WEEKS, 3 by default), so a chain that did
    not name its own frames would silently roll 100 members from the WEEK-3 init and
    produce a perfectly clean-looking 56-day survival curve of the wrong experiment."""
    for w in SC.STAB_WALK_WEEKS:
        ch = SCH.Chain(EV, w, 7)
        assert f"week{w}/" in str(ch.inputs_path())
        assert ch.inputs_path() == SCH.Chain(EV, w).inputs_path()
    monkeypatch.setattr(A, "WALKER_WEEKS", 3)
    assert f"week{WK}/" in str(SCH.Chain(EV, WK, 7).inputs_path())


# --------------------------------------------------------------------------- #
# The chain list and the shard map
# --------------------------------------------------------------------------- #
def test_chains_keeps_its_two_argument_form_exactly():
    """``members`` is the THIRD positional parameter because ``refs.ref_times`` calls
    ``chains([event], weeks)`` positionally, and ``Chain(e, w) == Chain(e, w, None)``
    because ``stages._score_fed`` compares chains with ``in``."""
    assert len(SCH.chains()) == 18
    assert SCH.chains([EV], [WK]) == [SCH.Chain(EV, WK)]
    assert SCH.Chain(EV, WK) == SCH.Chain(EV, WK, None)
    assert SCH.Chain(EV, WK) != SCH.Chain(EV, WK, 0)
    assert all(c.member is None for c in SCH.chains())


def test_the_member_chain_list_is_event_then_week_then_member():
    chs = SCH.chains([EV], [WK], range(5))
    assert [c.member for c in chs] == [0, 1, 2, 3, 4]
    two = SCH.chains(SC.STAB_EVENTS, [4, WK], range(2))
    assert [(c.event.split("_")[0], c.weeks, c.member) for c in two] == [
        ("PNW", 4, 0), ("PNW", 4, 1), ("PNW", 8, 0), ("PNW", 8, 1),
        ("WinterStorm", 4, 0), ("WinterStorm", 4, 1),
        ("WinterStorm", 8, 0), ("WinterStorm", 8, 1)]


@pytest.mark.parametrize("nshards", [24, 32, 40])
def test_every_member_lands_on_exactly_one_shard(nshards):
    """The shard map is ``i % nshards`` over a list that must be identical in every one of
    32 processes. A member selected twice is rolled twice; one selected by none is missing
    from a denominator with nothing to say so."""
    chs = SCH.chains([EV], [WK], range(100))
    seen = Counter()
    for s in range(nshards):
        for ch in SCH.shard_of(chs, nshards, s):
            seen[ch] += 1
    assert set(seen.values()) == {1}
    assert len(seen) == len(chs)
    counts = [len(SCH.shard_of(chs, nshards, s)) for s in range(nshards)]
    assert sum(counts) == 100 and max(counts) - min(counts) <= 1


def test_the_labels_are_greppable_per_member():
    """4.5 h of eight interleaved shard logs. The member suffix is the only thing that
    makes one member's lines findable - and the frozen 18's labels must not change, or
    every log-scraping note in HANDOFF stops matching."""
    assert SCH.Chain(EV, WK).label == f"{EV}@wk8 "
    assert SCH.Chain(EV, WK).slug == f"{EV}_wk08"
    assert SCH.Chain(EV, WK, 7).label == f"{EV}@wk8  m007"
    assert SCH.Chain(EV, WK, 7).slug == f"{EV}_wk08_m007"
    assert SCH.Chain(EV, WK, 99).label.endswith("m099")


# --------------------------------------------------------------------------- #
# Pruning: a member pins nothing
# --------------------------------------------------------------------------- #
def test_pin_false_leaves_exactly_the_parent_and_the_current_segment():
    """A member chain claims neither pinned state: segment 1 is the adapter-fed score
    arm's launch state (``--ens`` cannot score) and the final state is released at the
    chain's end. At 100 x 19 those two pins would be 95 GB of a filesystem with 267 GB
    free, held for artifacts nothing reads."""
    segs = SCH.chain_schedule(C.lead_days_for(WK))
    n = len(segs)
    for k in range(1, n + 1):
        gone = set(SCH.prunable(segs, k, pin=False))
        resident = set(range(1, k + 1)) - gone
        assert resident == {k, k - 1} - {0}, f"segment {k} left {sorted(resident)}"


def test_the_default_still_pins_the_frozen_sweeps_two_states():
    """``pin`` defaults to True and the 18 frozen chains must be pruned exactly as job
    1189 pruned them - the score arm relaunches from segment 1."""
    segs = SCH.chain_schedule(C.lead_days_for(WK))
    n = len(segs)
    for k in range(1, n + 1):
        assert set(SCH.prunable(segs, k)).isdisjoint({1, n})
    assert SCH.keep_states(segs) == {1, n}


def test_release_keeps_only_the_control_members_final_state(tmp_path, monkeypatch):
    """The release is load-bearing: unreleased, 100 members finish holding ~95 GB. Member
    2's FINAL state survives because it is job 1189's noise stream and therefore the
    determinism control's only physical evidence - its earlier states go with the rest."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    for m, keep in ((2, True), (3, False)):
        ch = SCH.Chain(EV, WK, m)
        for (k, _l, _n) in ch.schedule:
            p = ch.state_path(k)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"0" * 16)
        STG._release_states(ch)
        last = ch.schedule[-1][0]
        assert ch.state_path(last).exists() is keep
        assert not any(ch.state_path(k).exists() for (k, _l, _n) in ch.schedule[:-1])


def test_release_never_touches_the_frozen_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    frozen = SCH.Chain(EV, WK)
    p = frozen.state_path(1)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"0" * 16)
    assert STG._release_states(frozen) == 0.0
    assert p.exists()


# --------------------------------------------------------------------------- #
# The record's new fields
# --------------------------------------------------------------------------- #
def _synth_state(*, n_time=2, levels=(50, 500, 850), t_top=None) -> xr.Dataset:
    """A tiny global-shaped state with a level axis, for the range/extreme helpers."""
    lat = np.linspace(-88.0, 88.0, 12, dtype="float32")
    lon = np.arange(0.0, 360.0, 30.0, dtype="float32")
    rng = np.random.default_rng(0)
    t = 250.0 + 20.0 * rng.standard_normal((n_time, len(levels), lat.size, lon.size))
    if t_top is not None:                     # a blow-up confined to ONE level
        t[-1, 0, 0, 0] = t_top
    return xr.Dataset(
        {"temperature": (("time", "level", "lat", "lon"), t),
         "2m_temperature": (("time", "lat", "lon"),
                            288.0 + rng.standard_normal((n_time, lat.size, lon.size)))},
        coords=dict(time=pd.date_range("2021-05-06", periods=n_time, freq="12h"),
                    level=list(levels), lat=lat, lon=lon))


def test_ranges_by_level_is_consistent_with_field_ranges():
    """An attribution of the existing number, not a second measurement of it. If the two
    reductions could disagree, ``ranges_global`` would say a variable left its bounds and
    ``ranges_global_levels`` would name no level that did."""
    ds = _synth_state()
    per = REC.ranges_by_level(ds)["temperature"]
    field = D.field_ranges(ds)["temperature"]
    assert min(v["min"] for v in per.values()) == pytest.approx(field["min"])
    assert max(v["max"] for v in per.values()) == pytest.approx(field["max"])
    assert set(per) == {"50", "500", "850"}
    assert "2m_temperature" not in REC.ranges_by_level(ds), "single-level vars have none"


def test_ranges_by_level_locates_a_single_level_blow_up():
    """Job 1189's two SILENT divergences were ``temperature`` at 50 hPa over Antarctica
    while the CONUS troposphere stayed textbook-normal. With no states retained, this is
    the only surviving answer to 'which level'."""
    ds = _synth_state(t_top=1211.0)
    per = REC.ranges_by_level(ds)["temperature"]
    assert per["50"]["max"] == pytest.approx(1211.0)
    assert per["500"]["max"] < 400.0 and per["850"]["max"] < 400.0


def test_extremes_global_is_the_last_frame_and_carries_its_coordinates():
    """Last frame ONLY, and it says so: a located extremum needs one frame's index space
    to mean anything, and the last frame is the state the next segment inherits. So it may
    sit strictly inside ``field_ranges``, which reduces over both frames."""
    ds = _synth_state()
    ds["temperature"][0, 0, 0, 0] = -999.0        # worse, but in the EARLIER frame
    ext = REC.global_extremes(ds)["temperature"]
    assert set(ext["min"]) >= {"value", "lat", "lon", "level"}
    assert ext["min"]["value"] > -999.0, "the earlier frame must not be reported here"
    assert D.field_ranges(ds)["temperature"]["min"] == pytest.approx(-999.0)
    assert ext["max"]["level"] in (50, 500, 850)


def test_extremes_global_survives_an_all_nan_field():
    ds = _synth_state()
    ds["temperature"].values[:] = np.nan
    assert REC.global_extremes(ds)["temperature"] == dict(min=None, max=None)


def test_the_record_identifies_its_member_and_its_seed(tmp_path, monkeypatch):
    """100 records under ``ens/wk08/m007/`` that do not say which member they are cannot
    be re-associated if a directory moves, and ``seed`` is what says which GenCast stream
    the member drew. The frozen sweep's records keep ``member: null``."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    m = REC.segment_record(SCH.Chain(EV, WK, 7), 1, 3.0, 6)
    assert m["member"] == 7 and m["seed"] == 7 and m["walker"] == 2
    frozen = REC.segment_record(SCH.Chain(EV, WK), 1, 3.0, 6)
    assert frozen["member"] is None and frozen["seed"] == 2
    assert json.loads(json.dumps(m, default=REC.jsonable))["member"] == 7


# --------------------------------------------------------------------------- #
# The slim
# --------------------------------------------------------------------------- #
def _synth_diag(n=6) -> xr.Dataset:
    lat = np.arange(24.0, 26.01, 0.25, dtype="float32")
    lon = np.arange(235.0, 237.01, 0.25, dtype="float32")
    t = pd.date_range("2021-05-03T12:00", periods=n, freq="12h")
    rng = np.random.default_rng(1)
    shape = (1, n, lat.size, lon.size)
    ds = xr.Dataset(
        {v: (("batch", "time", "lat", "lon"),
             (288.0 if v == "2m_temperature" else 5.0)
             + rng.standard_normal(shape).astype("float32"))
         for v in ("2m_temperature", "10m_u_component_of_wind",
                   "mean_sea_level_pressure", "specific_humidity")},
        coords=dict(batch=[0], time=t, lat=lat, lon=lon))
    ds = ds.assign_coords(lead_h=("time", np.arange(12, 12 * (n + 1), 12,
                                                    dtype="int32")))
    ds.attrs.update(event=EV, walker=7, step=1)
    return ds


def test_slim_keeps_only_the_named_vars_and_every_coordinate(tmp_path, monkeypatch):
    """The full 12-variable diag is ~36 MB per segment: 65 GB over a 100-member week-8
    run, against 267 GB free. What the slim must not cost is the CONUS series itself -
    the observable and every CONUS panel item are built from ``2m_temperature`` - so the
    series is compared bit-for-bit before and after."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(EV, WK, 7)
    ch.diag_path(1).parent.mkdir(parents=True, exist_ok=True)
    ds = _synth_diag()
    ds.to_netcdf(ch.diag_path(1))
    before = D.conus_series(ds)

    STG._slim_diag(ch, 1)

    with xr.open_dataset(ch.diag_path(1)) as slim:
        slim = slim.load()
    assert list(slim.data_vars) == ["2m_temperature"]
    assert {"time", "lat", "lon", "lead_h"} <= set(slim.coords)
    assert slim.attrs["astab_diag_vars"] == "2m_temperature"
    assert "specific_humidity" in slim.attrs["astab_diag_dropped"]
    assert slim.attrs["event"] == EV, "the walk-time attrs must survive"
    pd.testing.assert_frame_equal(before, D.conus_series(slim))


def test_slim_is_idempotent_and_never_touches_the_frozen_sweep(tmp_path, monkeypatch):
    """A resumed run meets already-slim cubes, and the 18 frozen chains must keep all 12
    variables - ``reduce`` derives their whole CONUS panel from them."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(EV, WK, 7)
    ch.diag_path(1).parent.mkdir(parents=True, exist_ok=True)
    _synth_diag().to_netcdf(ch.diag_path(1))
    STG._slim_diag(ch, 1)
    size = ch.diag_path(1).stat().st_size
    STG._slim_diag(ch, 1)
    assert ch.diag_path(1).stat().st_size == size

    frozen = SCH.Chain(EV, WK)
    frozen.diag_path(1).parent.mkdir(parents=True, exist_ok=True)
    _synth_diag().to_netcdf(frozen.diag_path(1))
    STG._slim_diag(frozen, 1)
    with xr.open_dataset(frozen.diag_path(1)) as d:
        assert len(d.data_vars) == 4


def test_the_slim_is_disableable(tmp_path, monkeypatch):
    """``AIRES_ENS_DIAG_VARS=`` keeps the full cube. The knob exists because the slim is
    the one irreversible thing the walk does to its own diagnostics."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    monkeypatch.setattr(SC, "ENS_DIAG_VARS", ())
    ch = SCH.Chain(EV, WK, 7)
    ch.diag_path(1).parent.mkdir(parents=True, exist_ok=True)
    _synth_diag().to_netcdf(ch.diag_path(1))
    STG._slim_diag(ch, 1)
    with xr.open_dataset(ch.diag_path(1)) as d:
        assert len(d.data_vars) == 4


def test_the_record_says_which_vars_its_diag_held(tmp_path, monkeypatch):
    """A record built from a slimmed cube has a narrower ``nonfinite_conus`` and
    ``ranges_conus`` than one built at walk time. Nothing else on disk would say so."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(EV, WK, 7)
    ch.diag_path(1).parent.mkdir(parents=True, exist_ok=True)
    _synth_diag().to_netcdf(ch.diag_path(1))
    full = REC.segment_record(ch, 1, 3.0, 6)
    assert full["diag_vars"] == sorted(["2m_temperature", "10m_u_component_of_wind",
                                        "mean_sea_level_pressure", "specific_humidity"])
    STG._slim_diag(ch, 1)
    assert REC.segment_record(ch, 1, 3.0, 6)["diag_vars"] == ["2m_temperature"]


# --------------------------------------------------------------------------- #
# Wilson
# --------------------------------------------------------------------------- #
def test_wilson_is_a_real_interval_at_both_ends():
    """The whole reason it is Wilson and not Wald. "0 of 100 diverged" and "100 of 100"
    are the answers this run is most likely to produce, and Wald calls both of them
    certainties."""
    z = EN.Z95
    assert EN.wilson(0, 100) == pytest.approx((0.0, z * z / (100 + z * z)))
    assert EN.wilson(100, 100) == pytest.approx((100 / (100 + z * z), 1.0))
    assert EN.wilson(0, 100)[1] > 0.0
    assert EN.wilson(100, 100)[0] < 1.0


@pytest.mark.parametrize("k,n", [(1, 10), (5, 10), (9, 10), (33, 100), (50, 100)])
def test_wilson_contains_the_point_estimate(k, n):
    lo, hi = EN.wilson(k, n)
    assert lo <= k / n <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_narrows_with_n():
    widths = [EN.wilson(n // 2, n)[1] - EN.wilson(n // 2, n)[0]
              for n in (10, 50, 100, 500)]
    assert widths == sorted(widths, reverse=True)


def test_wilson_with_no_members_at_risk_says_nothing():
    assert EN.wilson(0, 0) == (0.0, 1.0)


# --------------------------------------------------------------------------- #
# Fates, from synthetic records
# --------------------------------------------------------------------------- #
def _write_member(ch, rows, *, tmp_path):
    """Write one member's per-segment records. ``rows`` is [(step, lead, ranges), ...]."""
    for (k, lead, ranges) in rows:
        p = ch.record_path(k)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(
            event=ch.event, weeks=ch.weeks, walker=ch.walker, member=ch.member,
            seed=ch.seed, step=k, lead_days=lead, n_steps=6,
            valid_time=str(ch.valid_at(lead)),
            conus=dict(times=[str(ch.valid_at(lead))], t2m_mean=[290.0],
                       t2m_sd=[6.0], t2m_anom=[1.5], t2m_mean_last=290.0,
                       t2m_anom_last=1.5, t2m_sd_last=6.0,
                       diurnal_ratio_mean=1.0),
            nonfinite_conus={}, ranges_conus={},
            nonfinite_global={}, ranges_global=ranges,
            ranges_global_levels={"temperature": {"50": dict(min=200.0,
                                                             max=ranges.get(
                                                                 "temperature",
                                                                 {}).get("max", 300.0))}},
            extremes_global={"temperature": dict(
                min=dict(value=200.0, lat=-80.0, lon=0.0, level=50.0),
                max=dict(value=ranges.get("temperature", {}).get("max", 300.0),
                         lat=-80.0, lon=0.0, level=50.0))})))


OK_RANGES = {"temperature": dict(min=200.0, max=300.0)}


def _t_max(v):
    return {"temperature": dict(min=200.0, max=v)}


def test_first_out_of_bounds_and_diverged_are_different_dates(tmp_path, monkeypatch):
    """Job 1189's PNW@wk8, reproduced: it left the bounds at 42 d by 0.5% of the bound
    span - a humidity excursion inside the noise of where that floor was drawn - and only
    became an unrecoverable blow-up at 45 d. Reporting the first date with the worst
    magnitude would date the divergence three days early on every member."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(EV, WK, 0)
    lo, hi = D.BOUNDS["temperature"]
    span = hi - lo
    rows = []
    for (k, lead, _n) in ch.schedule:
        if lead < 42:
            rows.append((k, lead, OK_RANGES))
        elif lead < 45:
            rows.append((k, lead, _t_max(hi + 0.005 * span)))    # 0.5% past: marginal
        else:
            rows.append((k, lead, _t_max(hi + 0.33 * span)))     # 33% past: diverged
    _write_member(ch, rows, tmp_path=tmp_path)

    s = EN.member_summary(ch, {})
    assert s["complete"] is True
    assert s["first_oob"] == 42.0
    assert s["diverged"] == 45.0
    assert s["first_oob_sev"] == pytest.approx(0.005, abs=1e-6)
    assert s["worst_sev"] == pytest.approx(0.33, abs=1e-6)
    assert s["fate"] == "diverged"
    assert s["first_oob_var"] == "temperature"
    assert s["worst_level"] == 50.0
    assert "42 d" in EN.format_member_verdict(s) and "45 d" in EN.format_member_verdict(s)


def test_a_clean_member_and_a_merely_marginal_one_are_not_the_same_fate(tmp_path,
                                                                        monkeypatch):
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    lo, hi = D.BOUNDS["temperature"]
    clean = SCH.Chain(EV, WK, 1)
    _write_member(clean, [(k, l, OK_RANGES) for (k, l, _n) in clean.schedule],
                  tmp_path=tmp_path)
    assert EN.member_summary(clean, {})["fate"] == "clean"
    assert EN.member_summary(clean, {})["diverged"] is None

    marg = SCH.Chain(EV, WK, 2)
    _write_member(marg, [(k, l, OK_RANGES if l < 54 else _t_max(hi + 0.005 * (hi - lo)))
                         for (k, l, _n) in marg.schedule], tmp_path=tmp_path)
    s = EN.member_summary(marg, {})
    assert s["fate"] == "marginal" and s["diverged"] is None and s["first_oob"] == 54.0


def test_an_unfinished_member_never_reads_as_clean(tmp_path, monkeypatch):
    """The precedence that matters. A shard killed at 30 d leaves 10 clean checkpoints,
    and calling that member "clean" would put a job failure into the numerator of a
    physical result."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(EV, WK, 3)
    _write_member(ch, [(k, l, OK_RANGES) for (k, l, _n) in ch.schedule[:10]],
                  tmp_path=tmp_path)
    s = EN.member_summary(ch, {})
    assert s["complete"] is False
    assert s["fate"] == "incomplete"
    assert s["n_records"] == 10 and s["last_lead"] == 30.0
    assert "INCOMPLETE" in EN.format_member_verdict(s)


def test_a_member_that_diverged_and_then_died_is_reported_as_diverged(tmp_path,
                                                                     monkeypatch):
    """The other side of the same precedence. Divergence is a physical finding and must
    not be hidden behind a job failure - ``complete`` still carries the failure, so the
    member appears in every incompleteness count as well."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(EV, WK, 4)
    lo, hi = D.BOUNDS["temperature"]
    rows = [(k, l, OK_RANGES if l < 21 else _t_max(hi + 0.4 * (hi - lo)))
            for (k, l, _n) in ch.schedule[:10]]
    _write_member(ch, rows, tmp_path=tmp_path)
    s = EN.member_summary(ch, {})
    assert s["fate"] == "diverged" and s["complete"] is False and s["diverged"] == 21.0


def test_an_infinite_severity_is_a_state_not_a_parse_error(tmp_path, monkeypatch):
    """``bound_severity`` returns ``inf`` when a bounded field has no finite value left -
    item 1's failure. It has to survive the summary, the JSON and the figure's log axis."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(EV, WK, 5)
    rows = [(k, l, OK_RANGES if l < 30 else
             {"temperature": dict(min=float("nan"), max=float("nan"))})
            for (k, l, _n) in ch.schedule]
    _write_member(ch, rows, tmp_path=tmp_path)
    s = EN.member_summary(ch, {})
    assert np.isinf(s["worst_sev"])
    assert s["diverged"] == 30.0 and s["fate"] == "diverged"
    assert json.loads(json.dumps(s["worst_sev"], default=REC.jsonable)) == float("inf")


def test_a_member_that_came_back_is_flagged_recovered(tmp_path, monkeypatch):
    """Reported, never folded into "clean": the survival curve is cumulative, so a
    recovered member still counts as diverged from its onset date onward. The gap between
    the cumulative and instantaneous counts is exactly these members."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(EV, WK, 6)
    lo, hi = D.BOUNDS["temperature"]
    rows = [(k, l, _t_max(hi + 0.4 * (hi - lo)) if 21 <= l <= 30 else OK_RANGES)
            for (k, l, _n) in ch.schedule]
    _write_member(ch, rows, tmp_path=tmp_path)
    s = EN.member_summary(ch, {})
    assert s["diverged"] == 21.0 and s["recovered"] is True and s["fate"] == "diverged"
    curve = {r["checkpoint_day"]: r for r in EN.survival_curve([s], [21.0, 30.0, 54.0])}
    assert curve[54.0]["n_undiverged"] == 0, "cumulative: it diverged and stays counted"
    assert curve[54.0]["n_inbounds_now"] == 1, "instantaneous: it is back inside"


def test_the_loud_silent_split_uses_the_sweeps_own_thresholds(tmp_path, monkeypatch):
    """Job 1189's verdict: the calibrated CONUS panel caught 1 of 3 divergences and the
    crude GLOBAL bounds check caught 3 of 3. A member drawn red in the heatmap but sitting
    inside the band on the observable is SILENT - which is the finding that says a
    long-lead AI+RES run must gate on the global state, not on A_L."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    lo, hi = D.BOUNDS["temperature"]
    ch = SCH.Chain(EV, WK, 8)
    _write_member(ch, [(k, l, OK_RANGES if l < 30 else _t_max(hi + 0.4 * (hi - lo)))
                       for (k, l, _n) in ch.schedule], tmp_path=tmp_path)

    quiet = {"t2m_anom_conus": D.Band("t2m_anom_conus", -10.0, 10.0, 99, 21.0)}
    assert EN.member_summary(ch, quiet)["mode"] == "silent"
    loud = {"t2m_anom_conus": D.Band("t2m_anom_conus", -0.1, 0.1, 99, 21.0)}
    s = EN.member_summary(ch, loud)
    assert s["mode"] == "loud" and s["mode_metrics"] == ["t2m_anom_conus"]
    assert s["mode_extrapolated"] is True, "past 21 d the band is an extrapolation"
    assert EN.member_summary(SCH.Chain(EV, WK, 9), loud)["mode"] == "", \
        "a member that never diverged has no mode"


def test_the_loud_metrics_are_the_calibrated_ones():
    """Mirrored rather than imported so ``ensemble`` does not depend on ``stages``. If the
    calibration ever measures a different set, this is what catches the drift."""
    assert set(EN.LOUD_METRICS) == set(STG.CALIB_METRICS)
    assert "bounds" not in EN.LOUD_METRICS, "the bounds check is the GLOBAL half"


# --------------------------------------------------------------------------- #
# The survival curve
# --------------------------------------------------------------------------- #
def _summary(member, *, leads, sev, complete=True, first_oob=None, diverged=None):
    return dict(member=member, seed=member, event=EV, weeks=WK, complete=complete,
                leads=list(leads), sev=list(sev),
                bounds=[1.0 if s > 0 else 0.0 for s in sev],
                nonfinite=[0.0] * len(leads),
                status=[EN._status_at(s, 1.0 if s > 0 else 0.0, 0.0) for s in sev],
                first_oob=first_oob, diverged=diverged, n_records=len(leads),
                worst_var="temperature", worst_level=50.0, fate="clean",
                t2m_anom=[1.0] * len(leads))


def test_a_member_with_no_record_at_a_checkpoint_is_not_in_its_denominator():
    """The denominator shrinks; the numerator is never quietly wrong. A shard killed at
    30 d contributes to every checkpoint up to 30 d and to none after it."""
    full = _summary(0, leads=[3.0, 6.0, 9.0], sev=[0.0, 0.0, 0.0])
    short = _summary(1, leads=[3.0], sev=[0.0], complete=False)
    curve = {r["checkpoint_day"]: r for r in
             EN.survival_curve([full, short], [3.0, 6.0, 9.0])}
    assert curve[3.0]["n_at_risk"] == 2
    assert curve[6.0]["n_at_risk"] == 1 and curve[9.0]["n_at_risk"] == 1
    assert curve[6.0]["frac_clean"] == 1.0


def test_the_curve_is_cumulative_and_monotone():
    lo, hi = D.BOUNDS["temperature"]
    leads = [3.0, 6.0, 9.0, 12.0]
    ms = [_summary(0, leads=leads, sev=[0, 0, 0, 0]),
          _summary(1, leads=leads, sev=[0, 0, 0.5, 0.0], first_oob=9.0, diverged=9.0),
          _summary(2, leads=leads, sev=[0, 0.5, 0.6, 0.7], first_oob=6.0, diverged=6.0)]
    curve = EN.survival_curve(ms, leads)
    assert [r["n_undiverged"] for r in curve] == [3, 2, 1, 1]
    assert [r["n_clean"] for r in curve] == [3, 2, 1, 1]
    # ...while the instantaneous count is NOT monotone: member 1 came back at 12 d.
    assert [r["n_inbounds_now"] for r in curve] == [3, 2, 1, 2]
    for r in curve:
        assert r["clean_lo"] <= r["frac_clean"] <= r["clean_hi"]
        assert r["n_diverged"] == r["n_at_risk"] - r["n_undiverged"]


def test_an_empty_checkpoint_reports_no_information_rather_than_a_number():
    curve = EN.survival_curve([_summary(0, leads=[3.0], sev=[0.0])], [3.0, 60.0])
    assert curve[1]["n_at_risk"] == 0
    assert np.isnan(curve[1]["frac_clean"])
    assert (curve[1]["clean_lo"], curve[1]["clean_hi"]) == (0.0, 1.0)


# --------------------------------------------------------------------------- #
# reduce / figs, end to end on synthetic records
# --------------------------------------------------------------------------- #
def _population(tmp_path, n=6, n_div=2):
    """``n`` members of the real week-8 schedule; the first ``n_div`` diverge at 45 d."""
    lo, hi = D.BOUNDS["temperature"]
    chs = []
    for m in range(n):
        ch = SCH.Chain(EV, WK, m)
        rows = [(k, l, _t_max(hi + 0.33 * (hi - lo)) if (m < n_div and l >= 45)
                 else OK_RANGES) for (k, l, _n) in ch.schedule]
        _write_member(ch, rows, tmp_path=tmp_path)
        chs.append(ch)
    return chs


def test_reduce_writes_all_three_artifacts_and_a_headline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    chs = _population(tmp_path)
    assert EN.stage_ens_reduce(chs) == 0
    out = capsys.readouterr().out

    long = pd.read_csv(SC.ens_csv_path(EV, WK))
    assert {"member", "seed", "fate", "checkpoint_day", "metric", "status",
            "reached_peak"} <= set(long.columns)
    assert set(long.member) == set(range(6))
    assert set(long.model) == {"gencast"}

    wide = pd.read_csv(SC.ens_members_csv_path(EV, WK))
    assert len(wide) == 6
    assert set(wide.fate) == {"diverged", "clean"}
    assert "rows" not in wide.columns, "the per-checkpoint rows do not belong in the wide table"

    summary = json.loads(SC.ens_summary_path(EV, WK).read_text())
    assert summary["n_members"] == 6 and summary["n_complete"] == 6
    assert len(summary["curve"]) == len(chs[0].schedule)
    assert summary["provenance"]["base_seed"] == A.WALKER_BASE_SEED
    assert summary["provenance"]["members"] == list(range(6))
    assert len(summary["caveats"]) >= 4
    assert "2/6" in out or "33%" in out
    assert "sampling spread" in " ".join(summary["caveats"]).lower()


def test_reduce_tolerates_members_still_in_flight(tmp_path, monkeypatch, capsys):
    """It doubles as the MID-FLIGHT monitor: run it two hours into a 4.5 h job and it
    gives the curve so far. An incomplete member is named, and it is excluded from the
    headline fraction rather than mixed into it."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    chs = _population(tmp_path, n=4, n_div=1)
    half = SCH.Chain(EV, WK, 4)
    _write_member(half, [(k, l, OK_RANGES) for (k, l, _n) in half.schedule[:5]],
                  tmp_path=tmp_path)
    chs.append(half)
    assert EN.stage_ens_reduce(chs) == 0
    out = capsys.readouterr().out
    summary = json.loads(SC.ens_summary_path(EV, WK).read_text())
    assert summary["n_members"] == 5 and summary["n_complete"] == 4
    assert "INCOMPLETE" in out and "m004" in out
    curve = {r["checkpoint_day"]: r for r in summary["curve"]}
    assert curve[3.0]["n_at_risk"] == 5 and curve[54.0]["n_at_risk"] == 4


def test_reduce_names_member_two_against_job_1189(tmp_path, monkeypatch, capsys):
    """The one comparison this run can make against a result that already exists."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    EN.stage_ens_reduce(_population(tmp_path, n=4, n_div=3))
    out = capsys.readouterr().out
    assert "job 1189's noise stream" in out
    assert "DETERMINISM CONTROL" in out
    assert "45 d" in out


def test_reduce_refuses_the_frozen_sweeps_chains(tmp_path, monkeypatch, capsys):
    """It reads ``ens/`` and writes ``ensemble_*``; handed the frozen chains it must say
    so rather than produce an ensemble of one."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    assert EN.stage_ens_reduce([SCH.Chain(EV, WK)]) == 1
    assert "not a member chain" in capsys.readouterr().err


def test_the_figure_renders_from_a_synthetic_population(tmp_path, monkeypatch):
    """Draws all four panels, including the log-scale severity spaghetti with zeros, an
    infinite severity, a recovered member and an incomplete one - the cases that break a
    plotting routine written against clean data."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    monkeypatch.setattr(SC, "FIGS", tmp_path / "figs")
    lo, hi = D.BOUNDS["temperature"]
    chs = _population(tmp_path, n=5, n_div=2)
    inf = SCH.Chain(EV, WK, 5)
    _write_member(inf, [(k, l, OK_RANGES if l < 30 else
                         {"temperature": dict(min=float("nan"), max=float("nan"))})
                        for (k, l, _n) in inf.schedule], tmp_path=tmp_path)
    part = SCH.Chain(EV, WK, 6)
    _write_member(part, [(k, l, OK_RANGES) for (k, l, _n) in part.schedule[:6]],
                  tmp_path=tmp_path)
    chs += [inf, part]

    assert EN.stage_ens_figs(chs) == 0
    out = SC.ens_fig_path(EV, WK)
    assert out.exists() and out.stat().st_size > 40_000


def test_the_figure_keeps_the_sweeps_divergence_red():
    """One system, two figures. ``astab/plots.py`` marks a divergence with ``#d62728``;
    a reader moving between the sweep's figure and the ensemble's must not have to relearn
    what red means."""
    from astab import plots as PL

    assert EN.ST_DIVERGED == "#d62728"
    assert EN.FATE_COLOR["diverged"] == EN.ST_DIVERGED
    src = pathlib.Path(PL.__file__).read_text()
    assert '"#d62728"' in src, "the sweep's figure no longer marks divergence in this red"


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #
def _spy_plan(monkeypatch) -> dict:
    """Capture the chain list ``--stage plan`` is handed, without running it."""
    seen: dict = {}

    def _rec(chs, nshards):
        seen["chs"] = chs
        seen["nshards"] = nshards
        return 0

    monkeypatch.setattr(RUN, "stage_plan", _rec)
    return seen


def test_ens_refuses_the_fcn3_stages():
    """Their artifacts carry no member and would alias - and ``prune`` would ``rmtree``
    the frozen sweep's retained Zarrs. The message has to say what to run instead."""
    for st in ("score", "prune", "maps"):
        with pytest.raises(SystemExit) as e:
            RUN.main(["--stage", st, "--ens", "5", "--event", EV, "--weeks", str(WK)])
        assert "GenCast-only" in str(e.value)
        assert st in str(e.value)


def test_ens_refuses_a_lead_the_walker_does_not_walk():
    """At weeks 12..20 the walker rolls ONE 3-day segment. A 100-member run there would
    report 100/100 clean off three days of a 140-day chain - a worse outcome than no
    answer, because it looks like one."""
    for wk in (w for w in SC.STAB_WEEKS if w not in SC.STAB_WALK_WEEKS):
        with pytest.raises(SystemExit) as e:
            RUN.main(["--stage", "plan", "--ens", "5", "--event", EV,
                      "--weeks", str(wk)])
        assert "AIRES_STAB_WALK_WEEKS" in str(e.value)
        assert str(wk) in str(e.value)


def test_check_runs_once_on_the_member_free_chains(monkeypatch):
    """The segment SHAPE does not depend on the member, and ``stage_check`` isolates each
    chain in a subprocess because this login node has 15 GB of RAM. 100 identical
    subprocesses would take an hour to prove one thing once."""
    seen = {}

    def _rec(chs):
        seen["chs"] = chs
        return 0

    monkeypatch.setattr(RUN, "stage_check", _rec)
    assert RUN.main(["--check", "--ens", "100", "--event", EV, "--weeks", str(WK)]) == 0
    assert len(seen["chs"]) == 1
    assert seen["chs"][0].member is None


def test_plan_and_walk_get_the_member_chains(monkeypatch):
    seen = _spy_plan(monkeypatch)
    assert RUN.main(["--stage", "plan", "--ens", "7", "--event", EV,
                     "--weeks", str(WK)]) == 0
    assert [c.member for c in seen["chs"]] == list(range(7))


def test_reduce_and_figs_dispatch_on_the_flag(monkeypatch):
    """Never on the contents of ``chs``: ``plots.stage_figs`` ignores its argument and
    reads the frozen ``stability.csv``, so a mis-dispatch would redraw the SWEEP's figure
    and hand it back as the ensemble's."""
    called = []
    monkeypatch.setattr(RUN, "stage_ens_reduce", lambda chs: called.append("ens") or 0)
    monkeypatch.setattr(RUN, "stage_ens_figs", lambda chs: called.append("ensfig") or 0)
    monkeypatch.setattr(RUN, "stage_reduce", lambda chs: called.append("sweep") or 0)
    monkeypatch.setattr(RUN, "stage_figs", lambda chs: called.append("sweepfig") or 0)

    RUN.main(["--stage", "reduce", "--stage", "figs", "--ens", "5", "--event", EV,
              "--weeks", str(WK)])
    assert called == ["ens", "ensfig"]
    called.clear()
    RUN.main(["--stage", "reduce", "--stage", "figs", "--event", EV, "--weeks", str(WK)])
    assert called == ["sweep", "sweepfig"]


def test_ens_members_is_a_subset_of_the_population(monkeypatch):
    seen = _spy_plan(monkeypatch)
    RUN.main(["--stage", "plan", "--ens", "100", "--ens-members", "3,7,41",
              "--event", EV, "--weeks", str(WK)])
    assert [c.member for c in seen["chs"]] == [3, 7, 41]

    with pytest.raises(SystemExit, match="outside"):
        RUN.main(["--stage", "plan", "--ens", "10", "--ens-members", "3,77",
                  "--event", EV, "--weeks", str(WK)])
    with pytest.raises(SystemExit, match="--ens N"):
        RUN.main(["--stage", "plan", "--ens-members", "3", "--event", EV,
                  "--weeks", str(WK)])


def test_without_ens_nothing_changes(monkeypatch):
    """The frozen sweep's CLI is untouched: no members, and every stage still reachable."""
    seen = _spy_plan(monkeypatch)
    RUN.main(["--stage", "plan"])
    assert len(seen["chs"]) == 18
    assert all(c.member is None for c in seen["chs"])
