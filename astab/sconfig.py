#!/usr/bin/env python
"""What the sweep is, and where every one of its artifacts lives.

This module is the single place the stability experiment's tree is decided - the same role
``gencast_s2s/config.py`` plays for the horizon experiment and ``aires/aconfig.py`` for
AI+RES. Nothing else in ``astab/`` builds a path by hand.

The tree
--------
The sweep used to be a TAG inside the AI+RES run tree
(``runs/aires/<event>/res/stab/``), sharing ``aconfig``'s ``res_*`` helpers with the
production DMC loop. It is now its own tree, for the reason ``xres`` is its own tree next
to ``gencast_s2s``: the two experiments answer different questions and only one of them
should be able to break the other's cache.

    runs/astab/<event>/walkers/w0N/stepKK/{state.nc,diag.nc,stab.json}
    runs/astab/<event>/scores/w0N_lead03_{cube.nc,.zarr}
    runs/astab/<event>/ens/wk<NN>/m<MMM>/stepKK/{state.nc,diag.nc,stab.json}
    runs/astab/refs/{bands.json,gate3_panel.csv,<event>_era5_ref.nc,<event>_era5_t2m.nc}
    runs/astab/stability.csv
    runs/astab/ensemble_<event>_wk<NN>{,_members}.csv, ensemble_<event>_wk<NN>.json
    figures/astab/stability.png
    figures/astab/ensemble_<event>_wk<NN>.png
    figures/astab/maps/<event>/<model>_wk<NN>.png

The walker slot ``N`` is the lead's index in ``STAB_WEEKS`` and is APPEND-ONLY: weeks
4/6/8/10 keep slots 0..3, so everything job 1189 wrote stays addressable when a lead is
added.

**Two namings, and they are not interchangeable.** The frozen sweep is keyed by walker
SLOT (``w0N``, the lead's index in ``STAB_WEEKS``); the ensemble follow-up is keyed by
WEEK (``wk<NN>``) and then by MEMBER (``m<MMM>``). Slot and week are a bijection
(``WEEKS_TO_WALKER``), so the two trees address the same lead by different names on
purpose: a member chain at week 8 still has ``Chain.walker == 2``, and if it built its
paths from the slot it would write into - and ``rmtree`` - job 1189's ``walkers/w02/``.
Every ensemble path therefore goes through an ``ens_*`` helper, and nothing else.

``AIRES_STAB_RUNS`` moves the data root; ``AIRES_STAB_FIGS`` moves the figure root. Both
exist so a second sweep (a different checkpoint, an ensemble follow-up) can be run without
touching the one on disk.

Env: imports cleanly in BOTH ``moe`` and ``fcn3`` - like ``aconfig``/``adapter``/``gate2``,
and for the same reason: one driver has to span the two-env handoff. No JAX here, ever.
"""
from __future__ import annotations

import os
from pathlib import Path

from aires import aconfig as A

# --------------------------------------------------------------------------- #
# What the sweep is
# --------------------------------------------------------------------------- #
STAB_TAG = os.environ.get("AIRES_STAB_TAG", "stab")

# The leads, and the walker SLOT each occupies. Contiguous from 0 because
# ``aires.score.tasks`` enumerates ``range(walkers)`` and cannot discover non-contiguous
# indices, so the whole coupled scoring arm stays one command per event.
#
# APPEND-ONLY. 4/6/8/10 must keep slots 0..3 or every artifact job 1189 left on disk
# (``walkers/w0N/``, ``scores/w0N_lead03_cube.nc``, ``stab_w0N.json``) silently belongs to
# a different chain. New leads go on the END.
STAB_WEEKS = (4, 6, 8, 10, 12, 14, 16, 18, 20)
WEEKS_TO_WALKER = {w: i for i, w in enumerate(STAB_WEEKS)}

# The leads at which the GENCAST WALKER is rolled all the way to the peak. Everything
# outside this is an **FCN3-extension lead**: the walker rolls ONE 3-day segment there,
# purely to hand the adapter-fed arm its launch state, and both FCN3 arms then run the
# full horizon.
#
# Why the split exists. Job 1189 answered the walker's question and left the scorer's
# open: GenCast diverged on 3 of 8 chains by week 6, while FCN3 was clean through week 10
# on EVERY rollout of BOTH arms - a floor, not a limit. Finding the scorer's actual limit
# needs longer FCN3 rollouts; it does not need longer walker chains, and a 140-day walker
# chain is 280 GenCast steps (~4 h of one H100) against ~13 min for the FCN3 rollout of
# the same span. So the extension buys the open question and not the closed one.
#
# It is one tuple, not a fork: putting 12..20 in here rolls full walker chains at those
# leads too, and nothing else in the package changes. Budget it first - `--stage plan`
# prints the GenCast steps.
STAB_WALK_WEEKS = tuple(
    int(w) for w in os.environ.get("AIRES_STAB_WALK_WEEKS", "4,6,8,10").split(",") if w)


def walks_to_peak(weeks: int) -> bool:
    """Is the GenCast walker rolled to the event peak at this lead, or only to the
    adapter-fed launch state at 3 d?

    Consulted by the schedule (how many segments), by ``stage_walk`` (what the chain must
    land on) and - the one that matters for the result - by ``reduce._model_survival``,
    which may not credit the walker with surviving a lead it was never rolled to.
    """
    return int(weeks) in STAB_WALK_WEEKS


# Two events, opposite seasons and opposite tails. The chains are inherently serial, so
# with 8 GPUs the second event is free wall-clock - and it is what stops the answer being
# "stable in June". Both already have Gate 3 walkers and a same-init week-4 xres cube.
STAB_EVENTS = ("PNW_HeatDome_2021", "WinterStorm_Uri_2021")

# The one coupled (walker -> adapter -> FCN3) launch point. Lead 3 d is segment 1 for EVERY
# chain regardless of horizon, so it is one command for all of them - and it is the LONGEST
# coupled rollout available, since FCN3 must then carry lead - 3 d on its own. It is also
# what an FCN3-extension lead's single walker segment exists to produce: at week 20 this
# is 3 d of GenCast followed by 137 d of FCN3.
SCORE_LEAD_DAYS = 3.0

# The physics gate for every chain past week 4, where no same-init reference ensemble
# exists. At 3 d GenCast is genuinely skillful, so CONUS-mean T2m within this of ERA5 is a
# sound check and the 26 K wrong-checkpoint drift fails it instantly. Deliberately loose:
# it is a wrong-checkpoint detector, not a skill score - and past ~10 d it is ONLY that,
# because agreement with ERA5 is no longer expected. Measured on the first four chains of
# job 1189: +0.18 to +0.41 K at 3 d lead. It is the ONLY physics gate an FCN3-extension
# chain gets, and it is enough: those chains are gated at 3 d, where GenCast is skillful.
ERA5_GATE_TOL_K = 6.0
ERA5_GATE_SKILLFUL_DAYS = 10.0    # beyond this, the gate is a blow-up detector and says so

# Members. One of each model - this is a stability sweep, not an ensemble.
#
# It stays 1 even for the ensemble follow-up: that run gets its members from separate
# CHAINS (one per member, `Chain.member`), each rolled serially by its own worker, not
# from a batched GenCast. The 0.25 deg checkpoint takes ~64 GB of an 80 GB H100 for ONE
# member, so `walker.load_bundle(n_members=N)` does not fit at any N > 1 and this constant
# must not be raised to make an ensemble.
N_WALKER_MEMBERS = 1
N_FCN3_MEMBERS = 1

# The three curves the sweep produces, in the order every table and legend uses them.
MODELS = ("gencast", "fcn3_era5", "fcn3_adapter")

REPO = Path(__file__).resolve().parents[1]
ENV_SH = REPO / "slurm" / "aires_env.sh"


# --------------------------------------------------------------------------- #
# The tree
# --------------------------------------------------------------------------- #
RUNS = Path(os.environ.get("AIRES_STAB_RUNS", A.RUNS / "astab"))
FIGS = Path(os.environ.get("AIRES_STAB_FIGS", A.ROOT / "figures" / "astab"))


def event_dir(event: str) -> Path:
    return RUNS / event


def walker_dir(event: str, walker: int) -> Path:
    return event_dir(event) / "walkers" / f"w{walker:02d}"


def segment_dir(event: str, walker: int, segment: int) -> Path:
    return walker_dir(event, walker) / f"step{segment:02d}"


def state_path(event: str, walker: int, segment: int) -> Path:
    return segment_dir(event, walker, segment) / "state.nc"


def diag_path(event: str, walker: int, segment: int) -> Path:
    return segment_dir(event, walker, segment) / "diag.nc"


def record_path(event: str, walker: int, segment: int) -> Path:
    """Per-segment diagnostics, written at walk time from the state BEFORE it is pruned.
    The global items (4 and 7) cannot be recovered afterwards."""
    return segment_dir(event, walker, segment) / "stab.json"


def score_cube_path(event: str, walker: int, lead_days: float) -> Path:
    return event_dir(event) / "scores" / f"w{walker:02d}_lead{int(lead_days):02d}_cube.nc"


def score_zarr_path(event: str, walker: int, lead_days: float) -> Path:
    return event_dir(event) / "scores" / f"w{walker:02d}_lead{int(lead_days):02d}.zarr"


def result_path(event: str, walker: int) -> Path:
    return event_dir(event) / f"stab_w{walker:02d}.json"


# --------------------------------------------------------------------------- #
# The ensemble follow-up's tree: N members of the walker at ONE lead
#
# Job 1189 rolled n = 1 member per lead and could not separate "that lead is unstable"
# from "that member diverged". This tree holds the run that can. It is DISJOINT from the
# frozen sweep's at the first path segment (`ens/` beside `walkers/`), because the two
# answer different questions with the same walker at the same lead and only one of them is
# allowed to be re-rollable.
#
# Keyed by WEEK, not by walker slot - see the module docstring.
# --------------------------------------------------------------------------- #
def ens_dir(event: str, weeks: int) -> Path:
    return event_dir(event) / "ens" / f"wk{int(weeks):02d}"


def ens_member_dir(event: str, weeks: int, member: int) -> Path:
    return ens_dir(event, weeks) / f"m{int(member):03d}"


def ens_segment_dir(event: str, weeks: int, member: int, segment: int) -> Path:
    return ens_member_dir(event, weeks, member) / f"step{int(segment):02d}"


def ens_state_path(event: str, weeks: int, member: int, segment: int) -> Path:
    """The rolling restart state. RELEASED at the end of a member's chain (see
    ``ENS_KEEP_STATE_MEMBERS``): 100 members x 19 segments x 477 MB is 906 GB against
    267 GB free, so keeping them is not an option and the release is load-bearing."""
    return ens_segment_dir(event, weeks, member, segment) / "state.nc"


def ens_diag_path(event: str, weeks: int, member: int, segment: int) -> Path:
    """The CONUS diagnostics cube, SLIMMED to ``ENS_DIAG_VARS`` after its record is
    written. Full it is ~36 MB per segment, i.e. 65 GB over the run; T2m-only it is
    ~0.6 MB. What the slim costs is the ability to recompute a non-T2m CONUS diagnostic
    without a re-walk, which is why it happens only AFTER ``ensure_record(force=True)``."""
    return ens_segment_dir(event, weeks, member, segment) / "diag.nc"


def ens_record_path(event: str, weeks: int, member: int, segment: int) -> Path:
    """The per-segment record. NEVER pruned, and after the state release it is the only
    surviving evidence of what this member's atmosphere looked like."""
    return ens_segment_dir(event, weeks, member, segment) / "stab.json"


def ens_result_path(event: str, weeks: int, member: int) -> Path:
    """One per member, beside the members' directories - never ``stab_w0N.json``, which
    belongs to the frozen sweep's slot N and would be overwritten by member N."""
    return ens_dir(event, weeks) / f"stab_m{int(member):03d}.json"


def ens_csv_path(event: str, weeks: int) -> Path:
    """Long: one row per (member, checkpoint, metric) - ``stability.csv``'s schema plus
    ``member``/``seed``/``fate``."""
    return RUNS / f"ensemble_{event}_wk{int(weeks):02d}.csv"


def ens_members_csv_path(event: str, weeks: int) -> Path:
    """Wide: one row per member. The table a reader actually scans."""
    return RUNS / f"ensemble_{event}_wk{int(weeks):02d}_members.csv"


def ens_summary_path(event: str, weeks: int) -> Path:
    """The headline, the survival curve, the provenance and the caveats, as JSON."""
    return RUNS / f"ensemble_{event}_wk{int(weeks):02d}.json"


def ens_fig_path(event: str, weeks: int) -> Path:
    return FIGS / f"ensemble_{event}_wk{int(weeks):02d}.png"


def ensure_ens_dirs(event: str, weeks: int) -> None:
    for d in (event_dir(event), ens_dir(event, weeks), log_dir(event), refs_dir()):
        d.mkdir(parents=True, exist_ok=True)


# Which CONUS variables survive in a member's ``diag.nc`` once its record is written.
# Empty (``AIRES_ENS_DIAG_VARS=``) disables the slim and keeps the full 12-variable cube -
# 65 GB over a 100-member week-8 run, against 267 GB free on this NFS at 98% full.
# T2m alone is what every CONUS item on the panel and the A_L observable are built from.
ENS_DIAG_VARS = tuple(
    v for v in (s.strip() for s in
                os.environ.get("AIRES_ENS_DIAG_VARS", "2m_temperature").split(","))
    if v)

# Members whose FINAL state survives the release. Member 2 at week 8 shares job 1189's
# noise stream (``segment_key`` folds in the member index, and the frozen wk8 chain is
# walker slot 2 at the same init), so its final state is the one post-mortem asset worth
# 477 MB: it is the determinism control. Empty keeps nothing.
ENS_KEEP_STATE_MEMBERS = frozenset(
    int(s) for s in os.environ.get("AIRES_ENS_KEEP_STATE_MEMBERS", "2").split(",")
    if s.strip())


def log_dir(event: str) -> Path:
    return event_dir(event) / "logs"


def ensure_dirs(event: str) -> None:
    for d in (event_dir(event), event_dir(event) / "walkers",
              event_dir(event) / "scores", log_dir(event), refs_dir()):
        d.mkdir(parents=True, exist_ok=True)


# -- references and reductions, which belong to no single event --------------- #
def refs_dir() -> Path:
    return RUNS / "refs"


def ref_path(event: str) -> Path:
    """The scalar ERA5 series: global/CONUS means and the z500 spectrum, per valid time."""
    return refs_dir() / f"{event}_era5_ref.nc"


def ref_field_path(event: str) -> Path:
    """The ERA5 T2m FIELD on the CONUS model grid, per valid time - the maps' truth row.

    Separate from ``ref_path`` on purpose. The scalar series is four floats per timestep
    and is read by every stage; this is 105 x 237 per timestep and is read only by
    ``astab.maps``, so a stage that wants the means never pays for the fields.
    """
    return refs_dir() / f"{event}_era5_t2m.nc"


def bands_path() -> Path:
    return refs_dir() / "bands.json"


def gate3_panel_path() -> Path:
    return refs_dir() / "gate3_panel.csv"


def csv_path() -> Path:
    return RUNS / "stability.csv"


# -- figures ------------------------------------------------------------------ #
def fig_path() -> Path:
    return FIGS / "stability.png"


def map_path(event: str, model: str, weeks: int) -> Path:
    """One figure per (event, model, initialization) - see ``astab.maps``."""
    return FIGS / "maps" / event / f"{model}_wk{weeks:02d}.png"


def free_cube_path(event: str, weeks: int) -> Path:
    """The adapter-free arm's cube, in FCN3's own week-namespaced tree.

    NOT under ``runs/astab/``: it is produced by ``fcn3/run_fcn3.py`` unmodified, so it
    lands where that driver puts it. Moving it would mean forking the driver, which is a
    worse trade than one path that points outside this tree.
    """
    return (A.RUNS / "fcn3" / f"week{weeks}" / "shards"
            / f"{event}_shard00of01_cube.nc")


def free_zarr_path(event: str, weeks: int) -> Path:
    return (A.RUNS / "fcn3" / f"week{weeks}" / "shards"
            / f"{event}_shard00of01.zarr")


def fcn3_ic_path(event: str, weeks: int) -> Path:
    return A.RUNS / "fcn3" / f"week{weeks}" / "ic" / f"{event}_ic.nc"
