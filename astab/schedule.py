#!/usr/bin/env python
"""The segment schedule, and the chain that walks it.

A chain is one ``(event, lead)`` rollout: initialised at ``peak - lead`` and rolled TO THE
PEAK in 6-step (3-day) segments so a worker pays one XLA compile. Ending at the peak rather
than free-running a fixed span is what makes this the real AI+RES configuration at each
lead - the observable ``A_L`` is a 7-day mean ENDING at the peak and is undefined for a
chain that stops short.

Two horizons, not one
---------------------
Since the FCN3-only extension (weeks 12..20) a chain has **two** lengths and they are not
interchangeable:

* ``horizon_days`` - init to peak. What both FCN3 arms roll, always.
* ``walk_days``    - how far the GENCAST WALKER is rolled. Equal to ``horizon_days`` at a
  ``sconfig.STAB_WALK_WEEKS`` lead; ``SCORE_LEAD_DAYS`` (3 d, one segment) at an extension
  lead, where the walker's only job is to hand the adapter-fed arm its launch state.

A third coordinate: the MEMBER
-----------------------------
``Chain.member`` is ``None`` for the frozen 18-chain sweep and an integer for the ensemble
follow-up, where the unit of work is one (event, lead, member) rollout and 100 of them
share a lead. It changes three things and nothing else: where the chain's artifacts live
(``sconfig.ens_*``, never ``walkers/w0N/``), which diffusion noise stream it draws
(``Chain.seed``), and whether the FCN3 path helpers are defined at all (they are not).
See the ``Chain`` docstring for why each of those is a safety property.

``Chain.schedule`` is the WALK schedule - it is what ``stage_walk`` rolls, what
``stage_check`` shapes, and what ``prune`` releases. Anything that needs the full
init-to-peak grid (``refs.ref_times``, ``maps._columns``) must ask for
``full_schedule``/``horizon_days`` explicitly. Conflating them is how an extension chain
would either be rolled for 4 h of GenCast nobody asked for, or have its ERA5 truth built
only to day 3.

The two off-by-ones this module exists to prevent
------------------------------------------------
Only 42 d is a whole number of 3-day segments. 28, 56 *and* 70 leave a remainder and take a
**ragged final segment, placed last** so every scoring checkpoint stays on a 3-day boundary
where ``aconfig.segment_for_lead`` is defined:

    lead    segments                     3-day checkpoints
    28 d    9 x 6 + 2 steps  = 10        3 .. 27 d      (9)
    42 d    14 x 6           = 14        3 .. 39 d      (13 - the 14th IS the peak)
    56 d    18 x 6 + 4 steps = 19        3 .. 54 d      (18)
    70 d    23 x 6 + 2 steps = 24        3 .. 69 d      (23)
    84 d    28 x 6           = 28        3 .. 81 d      (27 - the 28th IS the peak)
    98 d    32 x 6 + 4 steps = 33        3 .. 96 d      (32)
   112 d    37 x 6 + 2 steps = 38        3 .. 111 d     (37)
   126 d    42 x 6           = 42        3 .. 123 d     (41 - the 42nd IS the peak)
   140 d    46 x 6 + 4 steps = 47        3 .. 138 d     (46)

The last five are the FCN3-extension leads: the table is what both FCN3 arms report on
(``full_schedule``), while the walker rolls only the first row of each.

* **The tail table is** ``{4: 2, 6: 0, 8: 4, 10: 2, 12: 0, 14: 4, 16: 2, 18: 0, 20: 4}``
  **steps.** 70 is a multiple of 7 and of 14 and reads as if it should divide by 3. It does
  not; neither does 140, for the same reason. ``chain_schedule`` is pure arithmetic on
  ``horizon % 3`` with no per-chain special-casing, and ``stages.stage_walk`` hard-asserts
  that the final state's valid time IS the walk end - nothing else downstream checks that,
  so a chain landing a day short would go unnoticed.
* **The horizon is never a scoring checkpoint.** The 42-day chain's last full segment lands
  exactly on the peak, so scoring there would ask FCN3 for a zero-step rollout, which
  ``aires.score.nsteps_for`` rejects. Excluded for every chain, not only the one where it
  bites. (84 and 126 d divide evenly too, but their walkers stop at 3 d, so the rule has
  nothing to exclude there - see ``walker_checkpoints``.)

Both are pinned by ``astab/tests/test_astab.py``, written before the scheduler was.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aires import aconfig as A
from astab import sconfig as SC
from fcn3 import fevents as F
from gencast_s2s import config as C

# The segment schedule
# --------------------------------------------------------------------------- #
def tail_steps(horizon_days: int) -> int:
    """GenCast steps in the ragged final segment; 0 when the horizon divides evenly.

    Pure arithmetic on ``horizon % A.GATE3_SEG_DAYS``. No per-chain table, no exceptions -
    the tail table ``{4: 2, 6: 0, 8: 4, 10: 2}`` is an OUTPUT of this function and a test,
    never an input to it.
    """
    seg_days = A.GATE3_SEG_DAYS
    rem_days = float(horizon_days) - int(float(horizon_days) / seg_days) * seg_days
    n = int(round(rem_days * 24 / A.WALKER_STEP_H))
    if abs(n * A.WALKER_STEP_H / 24 - rem_days) > 1e-9:
        raise ValueError(f"horizon {horizon_days} d leaves a remainder of {rem_days} d, "
                         f"which is not a whole number of {A.WALKER_STEP_H} h steps")
    return n


def chain_schedule(horizon_days: int) -> list[tuple[int, float, int]]:
    """``[(step, lead_days at the END of the segment, n_steps), ...]`` for one chain.

    Full ``A.RES_SEG_STEPS`` segments first, the ragged tail LAST. The tail goes last so
    every intermediate boundary stays on the 3-day grid ``A.segment_for_lead`` is defined
    on; a tail placed first would shift every later checkpoint off that grid and quietly
    invalidate the whole scoring set.
    """
    horizon = float(horizon_days)
    if horizon < A.GATE3_SEG_DAYS:
        raise ValueError(f"horizon {horizon_days} d is shorter than one "
                         f"{A.GATE3_SEG_DAYS:g} d segment")
    n_full = int(horizon / A.GATE3_SEG_DAYS)
    segs = [(k, k * A.GATE3_SEG_DAYS, A.RES_SEG_STEPS) for k in range(1, n_full + 1)]
    tail = tail_steps(horizon_days)
    if tail:
        segs.append((n_full + 1, horizon, tail))
    return segs


def scoring_checkpoints(horizon_days: int) -> list[float]:
    """Leads at which a walker state can be handed to FCN3: 3-day boundaries, horizon
    EXCLUDED.

    The exclusion is not cosmetic. The 42-day chain's last full segment lands exactly on
    the peak, and scoring there asks for a zero-step FCN3 rollout, which
    ``score.nsteps_for`` rejects with ``n <= 0``. Applied to every chain so the rule does
    not depend on which lead happens to be divisible.
    """
    horizon = float(horizon_days)
    return [k * A.GATE3_SEG_DAYS
            for k in range(1, int(horizon / A.GATE3_SEG_DAYS) + 1)
            if k * A.GATE3_SEG_DAYS < horizon - 1e-9]


def walker_checkpoints(walk_days: float, horizon_days: float) -> list[float]:
    """Leads at which THIS chain's walker state can be handed to FCN3.

    ``scoring_checkpoints`` answers the question for a chain walked to its peak. An
    FCN3-extension chain is walked to 3 d, and 3 d is not its peak, so the
    horizon-exclusion rule that protects against a zero-step FCN3 rollout does not apply
    to it - the checkpoint set is every walked boundary strictly before the peak, which
    for an extension chain is exactly ``[3.0]``.
    """
    return [lead for (_k, lead, _n) in chain_schedule(int(walk_days))
            if lead < float(horizon_days) - 1e-9]


def keep_states(schedule) -> set[int]:
    """Segments whose ``state.nc`` must survive pruning.

    Two claimants beyond the rolling parent: the lead-3 d state the coupled scoring arm
    launches from, and the final peak state the reduce stage reads.
    """
    return {1, len(schedule)}


def prunable(schedule, done_step: int, keep: int | None = None, *,
             pin: bool = True) -> list[int]:
    """States that may be deleted once ``done_step`` has been written.

    Mirrors ``run_aires.prune_states``, simplified for a chain with no legs: rolling
    segment ``s`` needs only ``s - 1``, so everything two or more segments back is dead
    weight - except the pinned states above. Unpruned, job 1189's eight chains would have
    held 134 x 477 MB = 64 GB; pruned they held ~15 GB. (An FCN3-extension chain has one
    segment, so it has nothing to prune - see ``keep_states``.) ``diag.nc`` and the per-segment
    ``stab.json`` are NEVER touched: every diagnostic in the sweep is derived from them,
    and the global items cannot be recomputed once the state is gone.

    ``pin=False`` pins NOTHING, leaving exactly ``{done_step - 1, done_step}`` resident.
    That is the ensemble follow-up's mode and it exists for one reason: the two pinned
    states are claimed by consumers a member chain does not have. Segment 1 is the
    adapter-fed score arm's launch state - and ``--ens`` refuses to score at all - while
    the final state is the reduce stage's, which for a member reads the RECORD. At 100
    members x 19 segments those two pins are 95 GB of a filesystem with 267 GB free, so
    keeping them would be paying the whole ensemble's disk budget for artifacts nothing
    reads.
    """
    keep = A.RES_KEEP_STATES if keep is None else keep
    if keep <= 0:
        return []
    # `keep` counts the states left resident AFTER `done_step` is written, so keep=2
    # leaves `done_step` and its parent - one more than rolling the next segment strictly
    # needs, which is the margin that lets a killed job resume without a re-walk.
    cut = done_step - max(keep, 2)
    pinned = keep_states(schedule) if pin else set()
    return [s for s in range(1, cut + 1) if s not in pinned]




# --------------------------------------------------------------------------- #
# A chain
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chain:
    """One (event, lead[, member]) rollout: the unit of work, one per GPU.

    Every path it exposes goes through ``astab.sconfig``; nothing here knows the layout.
    Read the module docstring on ``horizon_days`` vs ``walk_days`` before using either.

    ``member`` is the ensemble follow-up's dimension and is the LAST field on purpose:
    ``Chain(event, weeks)`` is positional in two dozen call sites and
    ``Chain(e, w) == Chain(e, w, None)`` has to keep holding (``stages._score_fed``
    compares chains with ``in``). ``member is None`` IS the frozen sweep - same paths,
    same seeds, same everything job 1189 wrote.

    A member chain differs in three ways, and each is a safety property rather than a
    convenience:

    * its paths live under ``ens/wk<NN>/m<MMM>/`` and never under ``walkers/w0N/``, which
      is what stops it from ``rmtree``-ing the frozen sweep's tree (they share a lead, and
      therefore a walker slot);
    * every FCN3 path helper RAISES for it. Those are keyed on ``(event, walker)`` or
      ``(event, weeks)`` with no member in them, so 100 members would alias one cube -
      and ``stage_prune`` would delete the frozen sweep's retained Zarrs;
    * its ``seed`` is the member index, which is what makes the population an ensemble at
      all.
    """
    event: str
    weeks: int
    member: int | None = None

    @property
    def ev(self) -> F.Event:
        return F.event(self.event)

    @property
    def walker(self) -> int:
        """The lead's SLOT in ``STAB_WEEKS``. Still defined for a member chain - the lead
        is the same lead - which is exactly why no path may be built from it here."""
        return SC.WEEKS_TO_WALKER[self.weeks]

    @property
    def seed(self) -> int:
        """The index handed to ``walker.segment_key(walker=<this>, step)``.

        ``segment_key`` is ``fold_in(fold_in(PRNGKey(WALKER_BASE_SEED), step), index)`` -
        the same family ``xres/xinference.py`` and ``gencast_s2s/inference.py`` use for
        their ensembles - so distinct indices give genuinely distinct GenCast diffusion
        draws from one deterministic init. For the frozen sweep the index is the walker
        slot, which is what job 1189 rolled; for a member chain it is the member.

        The overlap is deliberate and is a control, not an accident: member 2 at week 8
        gets slot 2's stream, which IS job 1189's PNW@wk8 chain (same init, same
        checkpoint). Whether it diverges at 45 d again is a determinism finding.
        """
        return self.walker if self.member is None else int(self.member)

    @property
    def horizon_days(self) -> int:
        return C.lead_days_for(self.weeks)

    @property
    def peak(self) -> pd.Timestamp:
        return pd.Timestamp(self.ev.peak)

    @property
    def init(self) -> pd.Timestamp:
        # NOT ``self.ev.init``: fcn3.fevents pins that to FCN3_WEEKS at import time, and
        # this process spans every lead in the sweep at once.
        return self.peak - pd.Timedelta(days=self.horizon_days)

    @property
    def walks_to_peak(self) -> bool:
        """Full walker chain, or an FCN3-extension lead where the walker rolls 3 d only."""
        return SC.walks_to_peak(self.weeks)

    @property
    def walk_days(self) -> int:
        """How far the GENCAST walker is rolled. NOT ``horizon_days`` at an extension
        lead - see the module docstring."""
        return self.horizon_days if self.walks_to_peak else int(SC.SCORE_LEAD_DAYS)

    @property
    def walk_end(self) -> pd.Timestamp:
        """What the walk stage must land on: the peak, or the 3 d launch state."""
        return self.valid_at(self.walk_days)

    @property
    def schedule(self) -> list[tuple[int, float, int]]:
        """The WALK schedule - one segment at an FCN3-extension lead."""
        return chain_schedule(self.walk_days)

    @property
    def full_schedule(self) -> list[tuple[int, float, int]]:
        """The init-to-peak schedule, whether or not the walker is rolled over it.

        The 3-day grid both FCN3 arms report on, and the grid ``refs`` builds ERA5 over.
        Equal to ``schedule`` at a walk lead.
        """
        return chain_schedule(self.horizon_days)

    @property
    def checkpoints(self) -> list[float]:
        """Leads at which this chain's WALKER state can be handed to FCN3."""
        return walker_checkpoints(self.walk_days, self.horizon_days)

    @property
    def label(self) -> str:
        """What every log line names this chain by. The member suffix is what makes 4.5 h
        of interleaved shard logs greppable per member; the frozen 18 are unchanged."""
        base = f"{self.event}@wk{self.weeks:<2d}"
        return base if self.member is None else f"{base} m{self.member:03d}"

    @property
    def slug(self) -> str:
        base = f"{self.event}_wk{self.weeks:02d}"
        return base if self.member is None else f"{base}_m{self.member:03d}"

    def valid_at(self, lead_days: float) -> pd.Timestamp:
        return self.init + pd.Timedelta(days=float(lead_days))

    # -- paths (all through sconfig; no path arithmetic anywhere else) -------- #
    #
    # Every one of these dispatches on `member`, and that dispatch is safety-critical
    # rather than cosmetic: a member chain's `walker` is the LEAD's slot, so a path built
    # from it lands in the frozen sweep's tree. `stage_walk` deletes stale and partial
    # segment directories with `ignore_errors=True`.
    def segment_dir(self, step: int) -> Path:
        """The directory holding this segment's state/diag/record.

        Exists because ``stage_walk`` ``rmtree``s it. Before this method that call was
        ``SC.segment_dir(event, ch.walker, k)``, which for a week-8 MEMBER chain resolves
        to ``walkers/w02/stepKK`` - job 1189's frozen week-8 chain - and would have
        deleted it silently on the first stale or partial segment.
        """
        if self.member is None:
            return SC.segment_dir(self.event, self.walker, step)
        return SC.ens_segment_dir(self.event, self.weeks, self.member, step)

    def state_path(self, step: int) -> Path:
        if self.member is None:
            return SC.state_path(self.event, self.walker, step)
        return SC.ens_state_path(self.event, self.weeks, self.member, step)

    def diag_path(self, step: int) -> Path:
        if self.member is None:
            return SC.diag_path(self.event, self.walker, step)
        return SC.ens_diag_path(self.event, self.weeks, self.member, step)

    def record_path(self, step: int) -> Path:
        if self.member is None:
            return SC.record_path(self.event, self.walker, step)
        return SC.ens_record_path(self.event, self.weeks, self.member, step)

    # -- the FCN3 half, which a member chain does not have ------------------- #
    #
    # These are keyed on (event, walker) or (event, weeks) and carry no member, so 100
    # members would all name ONE cube, ONE Zarr and ONE IC - i.e. the frozen sweep's.
    # `stage_prune` rmtree's the Zarrs and `stage_plan` reads the cubes, so an alias here
    # is a silent overwrite of job 1189's scoring artifacts rather than a wrong number.
    # Raising is the whole point: `--ens` refuses the score and prune stages, and this is
    # what makes that refusal impossible to route around.
    def _no_fcn3(self, what: str):
        raise ValueError(
            f"{what} is not defined for member chain {self.label}: the FCN3 artifacts are "
            f"keyed on (event, walker={self.walker}) with no member in the path, so every "
            f"member of this ensemble would alias the frozen sweep's week-{self.weeks} "
            f"artifact. The ensemble follow-up is GenCast-only - `--ens` refuses the "
            f"score, prune and maps stages for exactly this reason.")

    def score_cube_path(self, lead_days: float = SC.SCORE_LEAD_DAYS) -> Path:
        if self.member is not None:
            self._no_fcn3("score_cube_path")
        return SC.score_cube_path(self.event, self.walker, lead_days)

    def score_zarr_path(self, lead_days: float = SC.SCORE_LEAD_DAYS) -> Path:
        if self.member is not None:
            self._no_fcn3("score_zarr_path")
        return SC.score_zarr_path(self.event, self.walker, lead_days)

    def inputs_path(self) -> Path:
        """The walker's init frames. Keyed on the chain's OWN week and on nothing ambient
        (the job-1187 trap) - and shared by every member of that week, which is correct:
        the ensemble's spread is GenCast's diffusion noise from one deterministic IC."""
        return A.gencast_inputs_path(self.event, weeks=self.weeks)

    def fcn3_ic_path(self) -> Path:
        if self.member is not None:
            self._no_fcn3("fcn3_ic_path")
        return SC.fcn3_ic_path(self.event, self.weeks)

    def fcn3_free_cube_path(self) -> Path:
        if self.member is not None:
            self._no_fcn3("fcn3_free_cube_path")
        return SC.free_cube_path(self.event, self.weeks)

    def fcn3_free_zarr_path(self) -> Path:
        if self.member is not None:
            self._no_fcn3("fcn3_free_zarr_path")
        return SC.free_zarr_path(self.event, self.weeks)

    def result_path(self) -> Path:
        """Deliberately does NOT raise for a member - it routes instead.

        ``stab_w0N.json`` is the frozen sweep's per-chain verdict and ``reduce`` writes it
        unconditionally. A member chain that reached this by any path must land on
        ``ens/wk<NN>/stab_m<MMM>.json``, so that a stray reduce over member chains cannot
        overwrite job 1189's verdicts even though ``--ens`` routes reduce elsewhere.
        """
        if self.member is None:
            return SC.result_path(self.event, self.walker)
        return SC.ens_result_path(self.event, self.weeks, self.member)


def chains(events=None, weeks=None, members=None) -> list[Chain]:
    """Every chain, event-major then week-major then member-major.

    Eighteen with ``members=None``: 2 events x 9 leads, and every one of them is the
    frozen sweep's ``Chain(event, weeks)`` exactly - ``members`` is the third positional
    parameter because ``refs.ref_times`` calls ``chains([event], weeks)`` positionally.

    No longer one per GPU - the sweep outgrew a single node's 8 cards when the FCN3
    extension was added. ``shard_of`` hands each GPU more than one chain, and because the
    extension chains are ~13 min of FCN3 rather than hours of GenCast, that costs nothing
    like what the same statement would have cost at week 10.

    The ORDER is what the shard map is built on, and every array task must produce the
    same list: ``shard_of`` selects ``i % nshards``, so two processes disagreeing about
    the order would roll overlapping members and leave others unrolled.
    """
    events = SC.STAB_EVENTS if events is None else tuple(events)
    weeks = SC.STAB_WEEKS if weeks is None else tuple(weeks)
    if members is None:
        return [Chain(e, w) for e in events for w in weeks]
    members = tuple(int(m) for m in members)
    return [Chain(e, w, m) for e in events for w in weeks for m in members]


def shard_of(items, nshards: int, shard: int) -> list:
    return [t for i, t in enumerate(items) if i % nshards == shard]
