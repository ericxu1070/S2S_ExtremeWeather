#!/usr/bin/env python
"""The lead-time stability sweep's driver: one CLI over every stage.

    PYTHONPATH=. python -m astab.run --stage plan
    PYTHONPATH=. python -m astab.run --check --event PNW_HeatDome_2021 --weeks 8
    sbatch slurm/aires_stab.slurm
    PYTHONPATH=. python -m astab.run --stage reduce --stage figs --stage maps

Stages may be repeated and run in the order given. Every one is cache-aware per artifact,
so a failed job is resubmitted rather than restarted - including the eight finished
week-4/6/8/10 chains, which skip in seconds even though their states were pruned.

``--weeks`` defaults to all nine leads. The last five (12..20) are FCN3-ONLY: the walker
rolls one 3-day segment there and both FCN3 arms roll the full horizon, so ``walk`` is
cheap and ``score`` is where the time goes. ``--stage plan`` prices them separately.

The ensemble follow-up
----------------------
``--ens N`` turns the selection into N MEMBER chains of the same (event, lead) - the run
that answers the question job 1189 could not, having rolled one member per lead::

    python -m astab.run --stage plan  --ens 100 --event PNW_HeatDome_2021 --weeks 8 --nshards 64
    sbatch slurm/aires_stab_ens.slurm
    python -m astab.run --stage reduce --stage figs --ens 100 --event PNW_HeatDome_2021 --weeks 8

It is a GenCast-only mode and the CLI enforces that rather than trusting the caller:
``score``, ``prune`` and ``maps`` are refused (their artifacts carry no member in their
paths and would alias - and delete - the frozen sweep's), a lead outside
``STAB_WALK_WEEKS`` is refused (the walker rolls 3 d there, and 100 members reporting
"clean" off three days of a 140-day chain is worse than no answer), and ``check`` runs on
the member-FREE chains because the segment shape does not depend on the member and 100
isolated subprocesses to prove it would cost an hour.

``reduce`` and ``figs`` dispatch on the FLAG, never on what happens to be in ``chs``:
``plots.stage_figs`` ignores its argument entirely and reads the frozen ``stability.csv``,
so a mis-dispatch would silently redraw the sweep's figure and report it as the ensemble's.

See ``astab/__init__.py`` for what the sweep is and what it does and does not establish.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aires import aconfig as A
from astab import sconfig as SC
from astab.ensemble import stage_ens_figs, stage_ens_reduce
from astab.maps import DEFAULT_NCOLS, stage_maps
from astab.plots import stage_figs
from astab.reduce import stage_reduce
from astab.refs import build_reference, build_reference_fields
from astab.schedule import chains
from astab.stages import (stage_calibrate, stage_check, stage_plan, stage_prune,
                          stage_score, stage_walk)
from fcn3 import fevents as F
from gencast_s2s import config as C

STAGES = ["plan", "check", "refs", "calibrate", "walk", "score", "reduce", "figs",
          "maps", "prune"]

# Stages that have no meaning for a member chain, and why refusing beats "it did nothing".
#
# `score`/`prune` are keyed on (event, walker) and (event, weeks) with NO member in the
# path, so every member would name the frozen sweep's cube, Zarr and IC - and `prune`
# rmtree's the Zarrs. `maps` reads the same FCN3 cubes. The state release a member chain
# needs already happens inside `stage_walk`, so `prune` has nothing to do here even if it
# were safe.
ENS_REFUSED = {
    "score": "the FCN3 arms are keyed on (event, walker) with no member in the path, so "
             "all N members would write ONE cube - the frozen sweep's",
    "prune": "a member chain releases its own states at the end of stage_walk; running "
             "prune here would delete the frozen sweep's retained Zarrs and lead-3 d "
             "states",
    "maps":  "the maps draw the FCN3 arms against ERA5, and a member chain has no FCN3 "
             "arm",
}


def _ens_members(a, stages, weeks) -> tuple[int, ...] | None:
    """Validate ``--ens``/``--ens-members`` and return the member list, or ``None``.

    Every refusal here is a safety invariant, not a preference, and each says what to do
    instead - a message that only says "not supported" costs the reader the same hour the
    check saved them.
    """
    if not a.ens and not a.ens_members:
        return None
    if not a.ens:
        raise SystemExit(
            "--ens-members needs --ens N: N is the POPULATION size and it is the "
            "denominator every fraction in the reduce is over. Re-walking members 3 and 7 "
            "of a 100-member run is `--ens 100 --ens-members 3,7`.")
    if a.ens < 1:
        raise SystemExit(f"--ens {a.ens}: the ensemble needs at least one member")

    bad_stage = [s for s in stages if s in ENS_REFUSED]
    if bad_stage:
        why = "\n".join(f"    {s}: {ENS_REFUSED[s]}" for s in bad_stage)
        raise SystemExit(
            f"--ens refuses the stage(s) {bad_stage}. The ensemble follow-up is "
            f"GenCast-only:\n{why}\n"
            f"  Run those stages WITHOUT --ens, on the frozen sweep's chains.")

    bad_week = [w for w in weeks if w not in SC.STAB_WALK_WEEKS]
    if bad_week:
        raise SystemExit(
            f"--ens at week(s) {bad_week}: the walker rolls only {SC.SCORE_LEAD_DAYS:g} d "
            f"there (an FCN3-extension lead), so {a.ens} members would all report "
            f"'clean' off three days of rollout and the survival curve would be a "
            f"statement about nothing.\n"
            f"  Walk weeks are {list(SC.STAB_WALK_WEEKS)}. To roll FULL walker chains at "
            f"{bad_week} instead, budget it with --stage plan first and then set\n"
            f"    AIRES_STAB_WALK_WEEKS={','.join(str(w) for w in sorted(set(SC.STAB_WALK_WEEKS) | set(bad_week)))}")

    if a.ens_members:
        members = tuple(int(s) for s in a.ens_members.split(",") if s.strip())
        out = [m for m in members if not 0 <= m < a.ens]
        if out:
            raise SystemExit(
                f"--ens-members {out} lie outside 0..{a.ens - 1}. A member outside the "
                f"population would be rolled but never counted, because the reduce's "
                f"denominator is --ens.")
        return members
    return tuple(range(a.ens))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="AI+RES lead-time stability sweep (astab)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append", default=None, choices=STAGES,
                    help="may be repeated; stages run in the order given")
    ap.add_argument("--check", action="store_true", dest="check_flag",
                    help="alias for --stage check (no GPU, no rollout)")
    ap.add_argument("--event", default=None, help="one event (alias for --events)")
    ap.add_argument("--events", default=None,
                    help=f"comma-separated; default {','.join(SC.STAB_EVENTS)}")
    ap.add_argument("--weeks", default=None,
                    help=f"comma-separated leads; default "
                         f"{','.join(map(str, SC.STAB_WEEKS))}")
    ap.add_argument("--walkers", type=int, default=A.GATE3_N_WALKERS,
                    help="calibrate: how many Gate 3 walkers to measure the bands on")
    ap.add_argument("--members", type=int, default=SC.N_FCN3_MEMBERS)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("FCN3_BATCH", 1)))
    ap.add_argument("--arm", default="both", choices=["free", "fed", "both"],
                    help="score: adapter-free (ERA5 IC), adapter-fed, or both")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("AIRES_SHARD", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("AIRES_NSHARDS", 8)))
    ap.add_argument("--keep-states", type=int, default=None)
    ap.add_argument("--max-segments", type=int,
                    default=int(os.environ.get("AIRES_STAB_MAX_SEGMENTS", 0)) or None,
                    help="walk: stop each chain after N segments (smoke path only)")
    ap.add_argument("--no-keep-zarr", action="store_true",
                    help="score: drop the pre-crop global Zarr (items 4 and 7 then have "
                         "no FCN3 curve)")
    ap.add_argument("--models", default=None,
                    help=f"maps: comma-separated subset of {','.join(SC.MODELS)}")
    ap.add_argument("--ncols", type=int, default=DEFAULT_NCOLS,
                    help="maps: lead columns per figure; 0 draws every 3-day checkpoint")
    ap.add_argument("--no-fields", action="store_true",
                    help="refs: build the scalar series only, not the T2m truth fields "
                         "the maps need")
    ap.add_argument("--prune-stale", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ens", type=int,
                    default=int(os.environ.get("AIRES_STAB_ENS", 0) or 0),
                    help="the ensemble follow-up: roll N MEMBERS of the selected "
                         "(event, lead) instead of one. GenCast only - score/prune/maps "
                         "are refused, and the lead must be a STAB_WALK_WEEKS lead.")
    ap.add_argument("--ens-members", default=os.environ.get("AIRES_ENS_MEMBERS"),
                    help="comma-separated subset of 0..--ens-1, for re-walking members "
                         "a shard lost. The denominator stays --ens.")
    a = ap.parse_args(argv)

    stages = list(a.stage or [])
    if a.check_flag and "check" not in stages:
        stages.insert(0, "check")
    if not stages:
        stages = ["plan"]

    ev_sel = a.events or a.event
    events = tuple(s.strip() for s in ev_sel.split(",") if s.strip()) if ev_sel \
        else SC.STAB_EVENTS
    weeks = tuple(int(s) for s in a.weeks.split(",") if s.strip()) if a.weeks \
        else SC.STAB_WEEKS
    for e in events:
        F.event(e)                                 # resolve or die, with the known list
    bad = [w for w in weeks if w not in C.WEEKS_TO_LEAD_DAYS]
    if bad:
        raise SystemExit(f"unknown week(s) {bad}; known: {sorted(C.WEEKS_TO_LEAD_DAYS)}")
    unslotted = [w for w in weeks if w not in SC.WEEKS_TO_WALKER]
    if unslotted:
        raise SystemExit(
            f"week(s) {unslotted} have no walker slot. sconfig.STAB_WEEKS is "
            f"{SC.STAB_WEEKS}; slots must stay contiguous from 0 because "
            f"aires.score.tasks enumerates range(walkers).")
    models = tuple(s.strip() for s in a.models.split(",") if s.strip()) if a.models \
        else SC.MODELS

    members = _ens_members(a, stages, weeks)
    chs = chains(events, weeks, members)
    # The shape check does not depend on the member - it builds a segment's ARRAYS from
    # the init frames, which every member of a lead shares - and `stage_check` isolates
    # each chain in its own subprocess because this login node has 15 GB of RAM. 100
    # identical subprocesses would take an hour to prove one thing once.
    check_chs = chains(events, weeks) if members else chs

    rc = 0
    for st in stages:
        if st == "plan":
            rc |= stage_plan(chs, a.nshards)
        elif st == "check":
            rc |= stage_check(check_chs)
        elif st == "refs":
            for e in events:
                build_reference(e, force=a.force)
                if not a.no_fields:
                    build_reference_fields(e, force=a.force)
        elif st == "calibrate":
            rc |= stage_calibrate(events, a.walkers, force=a.force)
        elif st == "walk":
            rc |= stage_walk(chs, a.shard, a.nshards,
                             prune_stale_states=a.prune_stale, keep=a.keep_states,
                             max_segments=a.max_segments)
        elif st == "score":
            rc |= stage_score(chs, a.shard, a.nshards, arm=a.arm, members=a.members,
                              batch_size=a.batch_size, keep_zarr=not a.no_keep_zarr)
        elif st == "reduce":
            # Dispatch on the FLAG, never on the contents of `chs`. `plots.stage_figs`
            # ignores its argument and reads the frozen stability.csv, so a dispatch that
            # inspected `chs` and got it wrong would silently redraw the SWEEP's figure
            # and hand it back as the ensemble's.
            rc |= stage_ens_reduce(chs) if a.ens else stage_reduce(chs)
        elif st == "figs":
            rc |= stage_ens_figs(chs) if a.ens else stage_figs(chs)
        elif st == "maps":
            rc |= stage_maps(chs, models=models, ncols=a.ncols, force=a.force)
        elif st == "prune":
            rc |= stage_prune(chs, force=a.force)
        if rc:
            break
    return rc


if __name__ == "__main__":
    sys.exit(main())
