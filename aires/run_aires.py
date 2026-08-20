#!/usr/bin/env python
"""AI+RES production driver: run the DMC loop on real GenCast walkers and FCN3 scores.

Phases 1 and 2 established the pieces. The adapter is exact where it can be (Gate 1), an
adapter-built IC forecasts like a native one (Gate 2), the FCN3 score ranks walkers by
their own eventual outcome (Gate 3: rho_s 0.80 at 9 d, 0.96 at 12 d), and ``aires/dmc.py``
is the algorithm, validated against an analytic Ornstein-Uhlenbeck problem. This module is
the machinery that connects them: it makes ``dmc.run``'s two callables read and write a
run tree on disk, across two conda environments and an eight-GPU node.

    propagate(k, parents, states)   ->  N GenCast segments, moe env, 8 shards
    score(k, states)                ->  N x M FCN3 forecasts, fcn3 env, 8 shards

Nothing about the algorithm changes here; ``dmc.py`` never learns what a walker is.

Why the two pools alternate instead of draining a shared queue
--------------------------------------------------------------
``aires.md`` specifies a coordinator plus two long-lived worker pools draining a file-based
work queue. The queue is the right shape for work of uneven size; this work has neither
property, for two measured reasons:

* **The pools cannot coexist.** The 0.25 deg GenCast checkpoint occupies ~64 GB of an
  80 GB H100 and FCN3 needs most of a card too, so a resident GenCast worker leaves no
  room for a resident FCN3 worker on the same GPU. Long-lived pools would have to hold
  device memory across the barrier they are waiting on. They alternate instead: one pool
  is launched, drains, and exits before the other starts.
* **Within a step every task is the same size.** The DMC loop is a barrier at each t_k, so
  a pool launch handles exactly one leg: N segments of identical length, or N score
  forecasts of identical lead. Static ``w % nshards`` sharding is then exactly balanced,
  and the work-stealing that ``xres/xinference.py`` needs (its events differ in cost) buys
  nothing.

What it costs is one model load per phase per leg. That is real - FCN3's 4.18 GB pickle
takes ~9 min to load from NFS with 8 readers - but only on the FIRST leg: the node's page
cache holds it for the rest of the job.

Resumption is by replay, not by checkpoint
------------------------------------------
The coordinator keeps no mutable checkpoint of its own. It re-runs the whole DMC loop from
step 1 on every invocation, and every leg whose artifacts are already on disk returns in
seconds. That is exact rather than approximate because the loop is a deterministic
function of (the pivotal-sampling seed, the scores on disk): the same seed and the same
score cubes reproduce the same clone/kill decisions. A resumed job is therefore the same
experiment, and the parents recorded in each leg's ``walk.json`` are re-derived and
compared on every pass, so a changed schedule or a rewritten cube is an error rather than
a silent fork.

The run tree (disjoint from Gate 3's - see aconfig)
---------------------------------------------------
    runs/aires/<event>/res/<tag>/
        run.json                       the frozen configuration; re-runs are checked on it
        walkers/w<ii>/step<kk>/         state.nc (pruned) + diag.nc (never pruned)
        scores/w<ii>_lead<LL>_cube.nc   the FCN3 score forecasts
        steps/leg<kk>/walk.json         parents: which slot each walker continues
                     /walk.done         the verified segment audit for that leg
                     /theta.json        the reduced score per walker
        res_result.json                 the whole DMC bookkeeping

    # login node, moe (no GPU): check prerequisites, freeze the configuration
    PYTHONPATH=. python -m aires.run_aires --stage prep

    # on the node (this is what slurm/aires_res.slurm runs)
    PYTHONPATH=. python -m aires.run_aires --stage res

    # afterwards, on the login node
    PYTHONPATH=. python -m aires.run_aires --stage ds
    PYTHONPATH=. python -m aires.run_aires --stage compare
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from aires import dmc
from aires import score as S
from fcn3 import fevents as F

REPO = Path(__file__).resolve().parents[1]
ENV_SH = REPO / "slurm" / "aires_env.sh"

DEFAULT_EVENT = "PNW_HeatDome_2021"
SCORE_BACKENDS = ("fcn3", "persistence")
SECONDS_PER_FCN3_STEP = 1.40      # measured, a3mega H100, job 1161
# Measured on the pilot (job 1172, leg 3 shard 0: 48 steps in 41.4 min with the model
# already resident). NOT the ~16 s/step of Gate 3's pure rollout: a production leg writes
# a 473 MB compressed restart state per SEGMENT, and with 8 shards doing it at once the
# zlib pass plus the NFS write costs about as much as the rollout itself. Budget the leg,
# not the rollout.
SECONDS_PER_GENCAST_STEP = 52.0


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass
class Ctx:
    ev: F.Event
    tag: str
    n_walkers: int
    members: int
    C: tuple
    leads: tuple
    horizon_days: float
    dmc_seed: int
    base_seed: int
    backend: str = "fcn3"
    skip_null_scores: bool = True
    nshards: int = 8
    attempts: int = 3
    retry_wait: int = 300
    keep_states: int = A.RES_KEEP_STATES
    dry_run: bool = False
    legs: list = field(default_factory=list)

    def __post_init__(self):
        if not self.legs:
            self.legs = A.res_legs(self.leads, self.horizon_days, self.C)

    @property
    def event(self) -> str:
        return self.ev.name

    @property
    def sign(self) -> float:
        from aires import aindex as AI
        return AI.tail_sign(self.event)

    def leg(self, k: int) -> A.Leg:
        return self.legs[k - 1]

    @property
    def segments(self) -> list[int]:
        return [s for leg in self.legs for s in leg.segments]

    def valid_at(self, lead_days: float) -> pd.Timestamp:
        return pd.Timestamp(self.ev.init) + pd.Timedelta(days=float(lead_days))

    def state_path(self, walker: int, segment: int) -> Path:
        return A.res_state_path(self.event, walker, segment, self.tag)

    def diag_path(self, walker: int, segment: int) -> Path:
        return A.res_diag_path(self.event, walker, segment, self.tag)

    def leg_dir(self, k: int) -> Path:
        return A.res_leg_dir(self.event, k, self.tag)

    def config(self) -> dict:
        """The frozen configuration. Two runs that disagree on any of it are two runs."""
        return dict(
            event=self.event, tag=self.tag, n_walkers=self.n_walkers,
            members=self.members, C=list(self.C), leads=list(self.leads),
            horizon_days=self.horizon_days, seg_steps=A.RES_SEG_STEPS,
            step_h=A.WALKER_STEP_H, dmc_seed=self.dmc_seed, base_seed=self.base_seed,
            backend=self.backend, skip_null_scores=self.skip_null_scores,
            resolution=A.WALKER_RES, params_file=A.walker_model_kwargs()["params_file"],
            metric=self.ev.metric, peak=self.ev.peak, init=str(self.ev.init),
            tail_sign=self.sign,
        )


def _event(name: str) -> F.Event:
    if name not in F.EVENTS:
        raise SystemExit(f"unknown event {name!r}; known: {', '.join(F.ORDER)}")
    return F.EVENTS[name]


# --------------------------------------------------------------------------- #
# Small IO helpers - every write is atomic, because a coordinator that dies mid-write
# must leave a resumable tree rather than a half-parsed manifest.
# --------------------------------------------------------------------------- #
def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_jsonable))
    os.replace(tmp, path)
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, Path)):
        return str(o)
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# run.json - the configuration guard
# --------------------------------------------------------------------------- #
def run_json_path(ctx: Ctx) -> Path:
    return A.res_dir(ctx.event, ctx.tag) / "run.json"


def freeze_config(ctx: Ctx, *, force: bool = False) -> dict:
    """Write the configuration, or check an existing one against it.

    A run tag names a population, not a directory. Resuming with a different N, a
    different C schedule or a different seed does not continue the run - it produces a
    population whose genealogy half-belongs to another experiment, and (because the
    walker states are cached by path) it would happily reuse segments rolled under the
    other schedule. So the configuration is frozen on the first invocation and every later
    one is compared against it.
    """
    p = run_json_path(ctx)
    want = ctx.config()
    if not p.exists():
        write_json(p, want)
        return want
    have = read_json(p)
    diff = {k: (have.get(k), want[k]) for k in want if have.get(k) != want[k]}
    if diff and not force:
        lines = "\n".join(f"    {k}: on disk {a!r}, requested {b!r}"
                          for k, (a, b) in sorted(diff.items()))
        raise SystemExit(
            f"the run tag {ctx.tag!r} already exists with a DIFFERENT configuration:\n"
            f"{lines}\n"
            f"  {p}\n"
            f"  Resuming would mix two populations: the cached walker states were rolled "
            f"under the configuration on disk. Use a new --tag, or --force to overwrite "
            f"the record (which does NOT make the cached states belong to the new run).")
    if diff:
        print(f"[res] --force: overwriting run.json ({len(diff)} field(s) changed)")
        write_json(p, want)
    return want


# --------------------------------------------------------------------------- #
# Segment validation.
#
# A cached segment is trusted only if it is at the right TIME and on the right LINEAGE.
# Time alone is not enough here in a way it was not for Gate 3: every Gate 3 walker slot
# is its own lineage for the whole run, so a state at the right valid time is necessarily
# the right state. Under resampling a slot changes ancestry at every leg, and two
# different ancestries produce states at the SAME valid time - so a stale segment from a
# previous run of the same tag is indistinguishable by timestamp and would silently
# splice another experiment's walker into this genealogy.
#
# `walker.run_segment` records `parent` (the state file it continued) in the state's
# attributes, which is exactly the missing evidence. Only the trailing w<ii>/step<kk>
# components are compared, so a state seeded from the Gate 3 tree still validates.
# --------------------------------------------------------------------------- #
def lineage_key(path) -> str:
    if path is None:
        return "init"
    parts = Path(path).parts
    return "/".join(parts[-3:]) if len(parts) >= 3 else str(path)


def segment_status(ctx: Ctx, walker: int, segment: int,
                   parent: Path | None) -> tuple[str, str]:
    """``('ok'|'missing'|'stale'|'partial', detail)`` for one cached segment."""
    from aires import walker as W

    sp, dp = ctx.state_path(walker, segment), ctx.diag_path(walker, segment)
    if not sp.exists():
        return "missing", "no state.nc"
    want = ctx.valid_at(segment * A.GATE3_SEG_DAYS)
    try:
        with xr.open_dataset(sp) as ds:
            got = W.valid_time(ds)
            attrs = dict(ds.attrs)
    except Exception as e:
        return "stale", f"unreadable ({type(e).__name__}: {e})"
    if pd.Timestamp(got) != want:
        return "stale", f"valid {got}, schedule wants {want}"
    got_parent = lineage_key(attrs.get("parent", "?"))
    want_parent = lineage_key(parent)
    if got_parent != want_parent:
        return "stale", f"continues {got_parent}, this leg says {want_parent}"
    want_params = A.walker_model_kwargs()["params_file"]
    if attrs.get("params_file") not in (None, want_params):
        return "stale", (f"rolled with {attrs.get('params_file')!r}, this run uses "
                         f"{want_params!r}")
    if int(attrs.get("base_seed", ctx.base_seed)) != ctx.base_seed:
        return "stale", f"base seed {attrs.get('base_seed')} != {ctx.base_seed}"
    if not dp.exists():
        return "partial", "state.nc present but no diag.nc"
    return "ok", str(got)


def parent_of(ctx: Ctx, leg: A.Leg, walker: int, segment: int,
              parents: list | None) -> Path | None:
    """The state a given (walker, segment) continues.

    Only the FIRST segment of a leg crosses a resampling: it continues the slot named by
    ``parents``. Later segments of the same leg continue the walker's own previous
    segment, because no resampling happened in between.
    """
    if segment == 1:
        return None
    if segment == leg.first_segment:
        p = walker if parents is None else int(parents[walker])
        return ctx.state_path(p, segment - 1)
    return ctx.state_path(walker, segment - 1)


# --------------------------------------------------------------------------- #
# Leg manifests
# --------------------------------------------------------------------------- #
def walk_manifest_path(ctx: Ctx, k: int) -> Path:
    return ctx.leg_dir(k) / "walk.json"


def walk_done_path(ctx: Ctx, k: int) -> Path:
    return ctx.leg_dir(k) / "walk.done"


def theta_path(ctx: Ctx, k: int) -> Path:
    return ctx.leg_dir(k) / "theta.json"


def write_walk_manifest(ctx: Ctx, leg: A.Leg, parents) -> dict:
    """Record (and on a resume, re-check) which slot each walker continues."""
    man = dict(leg=leg.k, event=ctx.event, tag=ctx.tag,
               segments=list(leg.segments), n_steps=A.RES_SEG_STEPS,
               lead_start_days=leg.lead_start_days, lead_end_days=leg.lead_end_days,
               C=leg.C, n_walkers=ctx.n_walkers, base_seed=ctx.base_seed,
               parents=None if parents is None else [int(p) for p in parents])
    p = walk_manifest_path(ctx, leg.k)
    if p.exists():
        have = read_json(p)
        if have.get("parents") != man["parents"] or have.get("segments") != man["segments"]:
            raise SystemExit(
                f"leg {leg.k}: the resampling replay does not match what is on disk.\n"
                f"  {p}\n"
                f"  on disk : parents {have.get('parents')}\n"
                f"  replayed: parents {man['parents']}\n"
                f"  The DMC loop is a deterministic function of (dmc_seed, the score "
                f"cubes). A mismatch means one of them changed under a finished leg - a "
                f"rewritten score cube, an edited C schedule, or a different seed - so "
                f"the walkers already on disk belong to a different experiment. Start a "
                f"new --tag.")
        return have
    write_json(p, man)
    return man


def read_walk_manifest(ctx: Ctx, k: int) -> dict:
    p = walk_manifest_path(ctx, k)
    if not p.exists():
        raise SystemExit(
            f"no manifest for leg {k} at {p}\n"
            f"  The coordinator writes it: PYTHONPATH=. python -m aires.run_aires "
            f"--stage res --tag {ctx.tag}")
    return read_json(p)


# --------------------------------------------------------------------------- #
# Worker pools
# --------------------------------------------------------------------------- #
_CHILDREN: list[subprocess.Popen] = []


def _kill_children(signum, frame):
    for p in _CHILDREN:
        if p.poll() is None:
            p.terminate()
    raise SystemExit(f"terminated by signal {signum}")


def visible_gpus() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return len([x for x in vis.split(",") if x.strip() != ""])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                             timeout=60)
        return len([l for l in out.stdout.splitlines() if l.startswith("GPU ")])
    except Exception:
        return 0


def run_pool(ctx: Ctx, label: str, envname: str, argv: list[str]) -> None:
    """Launch ``ctx.nshards`` single-GPU shards of one command and wait for all of them.

    Retries the WHOLE pool rather than the failed shards: every stage is cache-aware per
    artifact, so a repeat pass costs seconds for what is already on disk and only redoes
    what is missing. This exists because the GPUs on this cluster are not reliably ours
    even under ``--exclusive`` - Gate 3's job 1155 lost its score phase to an unrelated
    vLLM sweep that took 78 GB on every card. A real bug still fails all attempts.
    """
    log_dir = A.res_log_dir(ctx.event, ctx.tag)
    log_dir.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    for attempt in range(1, ctx.attempts + 1):
        print(f"  [{label}] attempt {attempt}/{ctx.attempts}: {ctx.nshards} shard(s)",
              flush=True)
        procs, logs = [], []
        for s in range(ctx.nshards):
            log = log_dir / f"{label}-{job}-try{attempt}-shard{s}.log"
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(s)
            env["AIRES_NSHARDS"] = str(ctx.nshards)
            env.pop("LD_LIBRARY_PATH", None)     # rebuilt by aires_env.sh for the target env
            cmd = ["bash", str(ENV_SH), envname, "python", *argv,
                   "--shard", str(s), "--nshards", str(ctx.nshards)]
            fh = open(log, "w")
            procs.append(subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                          env=env, cwd=str(REPO)))
            logs.append((log, fh))
            _CHILDREN.append(procs[-1])
        rc = [p.wait() for p in procs]
        for _, fh in logs:
            fh.close()
        for p in procs:
            if p in _CHILDREN:
                _CHILDREN.remove(p)
        bad = [i for i, r in enumerate(rc) if r != 0]
        if not bad:
            return
        for i in bad:
            print(f"  [{label}] shard {i} exited {rc[i]} -- see {logs[i][0]}",
                  file=sys.stderr)
        if attempt < ctx.attempts:
            print(f"  [{label}] waiting {ctx.retry_wait}s before retrying (cached work "
                  f"is skipped, so a retry only redoes what is missing)", flush=True)
            time.sleep(ctx.retry_wait)
    raise RuntimeError(f"{label} failed {ctx.attempts} time(s)")


# --------------------------------------------------------------------------- #
# propagate: the walk phase
# --------------------------------------------------------------------------- #
def leg_walk_state(ctx: Ctx, leg: A.Leg, parents) -> dict:
    """Per-(walker, segment) status for a whole leg."""
    out = {}
    for w in range(ctx.n_walkers):
        for s in leg.segments:
            out[(w, s)] = segment_status(ctx, w, s, parent_of(ctx, leg, w, s, parents))
    return out


def prune_states(ctx: Ctx, leg: A.Leg) -> int:
    """Delete walker states more than ``keep_states`` segments behind this leg.

    States are 477 MB each and there are N per segment; keeping all of them for N=64 is
    213 GB against ~600 GB free on a 95%-full shared filesystem. Rolling a segment needs
    only its immediate parent, so two are enough. Diagnostics cubes are never touched -
    they are 6 MB and every observable in the experiment is derived from them.
    """
    if ctx.keep_states <= 0:
        return 0
    cut = leg.first_segment - max(ctx.keep_states, 2)
    if cut < 1:
        return 0
    freed = 0
    for w in range(ctx.n_walkers):
        for s in range(1, cut + 1):
            sp = ctx.state_path(w, s)
            if sp.exists():
                freed += sp.stat().st_size
                sp.unlink()
    if freed:
        print(f"  [walk] pruned walker states at segments <= {cut} "
              f"({freed / 1e9:.1f} GB freed; --keep-states {ctx.keep_states})")
    return freed


def ensure_leg_walked(ctx: Ctx, leg: A.Leg, parents) -> None:
    done = walk_done_path(ctx, leg.k)
    if done.exists():
        rec = read_json(done)
        if rec.get("parents") == (None if parents is None else [int(p) for p in parents]):
            missing = [(w, s) for w in range(ctx.n_walkers) for s in leg.segments
                       if not ctx.diag_path(w, s).exists()]
            if not missing:
                print(f"  [walk] leg {leg.k}: complete (verified {rec.get('verified')})")
                return
            print(f"  [walk] leg {leg.k}: walk.done exists but {len(missing)} diagnostics "
                  f"cube(s) are gone, e.g. {missing[0]}; re-verifying", file=sys.stderr)
            done.unlink()

    for attempt in range(2):
        st = leg_walk_state(ctx, leg, parents)
        stale = [(k, v[1]) for k, v in st.items() if v[0] == "stale"]
        if stale:
            for (w, s), why in stale[:8]:
                print(f"  [walk] STALE w{w:02d} step{s}: {why}", file=sys.stderr)
            raise RuntimeError(
                f"leg {leg.k}: {len(stale)} cached segment(s) do not belong to this run. "
                f"They are at the right valid time but on the wrong lineage or from the "
                f"wrong checkpoint. Delete them, or use a new --tag.")
        for (w, s), (state, why) in st.items():
            if state == "partial":
                print(f"  [walk] incomplete w{w:02d} step{s}: {why} - re-rolling")
                shutil.rmtree(A.res_segment_dir(ctx.event, w, s, ctx.tag),
                              ignore_errors=True)
        todo = [k for k, v in st.items() if v[0] != "ok"]
        if not todo:
            break
        if attempt:
            raise RuntimeError(
                f"leg {leg.k}: {len(todo)} segment(s) still missing after the walk pool "
                f"finished, e.g. {todo[0]}. See {A.res_log_dir(ctx.event, ctx.tag)}")
        print(f"  [walk] leg {leg.k}: {len(todo)}/{len(st)} segment(s) to roll "
              f"({leg.lead_start_days:g} -> {leg.lead_end_days:g} d)", flush=True)
        if ctx.dry_run:
            raise SystemExit(f"--dry-run: leg {leg.k} needs {len(todo)} walker segment(s)")
        t0 = time.perf_counter()
        run_pool(ctx, f"walk-leg{leg.k:02d}", "moe",
                 ["-m", "aires.run_aires", "--stage", "walk", "--event", ctx.event,
                  "--tag", ctx.tag, "--leg", str(leg.k),
                  "--walkers", str(ctx.n_walkers)])
        print(f"  [walk] leg {leg.k}: pool finished in "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    write_json(walk_done_path(ctx, leg.k), dict(
        leg=leg.k, segments=list(leg.segments),
        parents=None if parents is None else [int(p) for p in parents],
        verified=str(pd.Timestamp.utcnow()),
        states={f"w{w:02d}s{s}": lineage_key(ctx.state_path(w, s))
                for w in range(ctx.n_walkers) for s in leg.segments},
    ))


# --------------------------------------------------------------------------- #
# score: the two backends
# --------------------------------------------------------------------------- #
def theta_from_fcn3(ctx: Ctx, leg: A.Leg) -> dict:
    """Reduce this leg's N score cubes to ``theta`` = the FCN3 ensemble mean of A_L."""
    from aires import aindex as AI

    lead = leg.lead_end_days
    box, conus, sd_box, sd_conus = [], [], [], []
    for w in range(ctx.n_walkers):
        t = S.Task(ctx.event, w, lead, ctx.tag)
        with xr.open_dataset(t.cube_path) as cube:
            idx = AI.indices(cube, ctx.event, ctx.ev.peak, ctx.ev.metric,
                             where=f"score cube {t}")
        b = np.asarray(idx["box"].values, dtype=float)
        c = np.asarray(idx["conus"].values, dtype=float)
        box.append(float(b.mean()))
        conus.append(float(c.mean()))
        sd_box.append(float(b.std(ddof=1)) if b.size > 1 else float("nan"))
        sd_conus.append(float(c.std(ddof=1)) if c.size > 1 else float("nan"))
    return dict(box=box, conus=conus, sd_box=sd_box, sd_conus=sd_conus,
                members=ctx.members, backend="fcn3")


def theta_from_persistence(ctx: Ctx, leg: A.Leg) -> dict:
    """Lancelin et al.'s Standard-RES score: the observable's CURRENT value.

    Free - the walker's own diagnostics cube already holds the frame - and it is the bar
    the FCN3 score has to clear. Gate 3 measured that bar at the two leads that matter
    (rho_s 0.062 at 9 d and 0.129 at 12 d against FCN3's 0.797 and 0.959), so running the
    pilot on this backend is the control experiment, not a fallback.
    """
    from aires import aindex as AI

    when = ctx.valid_at(leg.lead_end_days)
    box, conus = [], []
    for w in range(ctx.n_walkers):
        with xr.open_dataset(ctx.diag_path(w, leg.last_segment)) as d:
            idx = AI.instantaneous_index(d, when, ctx.event, ctx.ev.metric)
        box.append(float(idx["box"]))
        conus.append(float(idx["conus"]))
    n = len(box)
    return dict(box=box, conus=conus, sd_box=[float("nan")] * n,
                sd_conus=[float("nan")] * n, members=0, backend="persistence")


def ensure_leg_scored(ctx: Ctx, leg: A.Leg) -> None:
    lead = leg.lead_end_days
    for attempt in range(2):
        todo = [t for t in (S.Task(ctx.event, w, lead, ctx.tag)
                            for w in range(ctx.n_walkers)) if not t.cube_path.exists()]
        if not todo:
            return
        if attempt:
            raise RuntimeError(
                f"leg {leg.k}: {len(todo)} score cube(s) still missing after the score "
                f"pool finished, e.g. {todo[0]}. "
                f"See {A.res_log_dir(ctx.event, ctx.tag)}")
        n = S.nsteps_for(ctx.ev, ctx.valid_at(lead))
        print(f"  [score] leg {leg.k}: {len(todo)}/{ctx.n_walkers} forecast(s) to run "
              f"({ctx.members} x {n} x {F.STEP_H} h from {lead:g} d to the peak; "
              f"{len(todo) * ctx.members * n:,} FCN3 steps)", flush=True)
        if ctx.dry_run:
            raise SystemExit(f"--dry-run: leg {leg.k} needs {len(todo)} score forecast(s)")
        t0 = time.perf_counter()
        run_pool(ctx, f"score-leg{leg.k:02d}", "fcn3",
                 ["-m", "aires.score", "--event", ctx.event, "--tag", ctx.tag,
                  "--walkers", str(ctx.n_walkers), "--members", str(ctx.members),
                  "--leads", f"{lead:g}"])
        print(f"  [score] leg {leg.k}: pool finished in "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def leg_theta(ctx: Ctx, leg: A.Leg) -> dict:
    """``theta`` for one leg, cached. Returns the record, not the signed score."""
    p = theta_path(ctx, leg.k)
    want_backend = ("skipped(C=0)" if (leg.C == 0.0 and ctx.skip_null_scores)
                    else ctx.backend)
    if p.exists():
        rec = read_json(p)
        if (rec.get("backend") == want_backend and len(rec["box"]) == ctx.n_walkers
                and rec.get("lead_days") == leg.lead_end_days):
            return rec
        print(f"  [score] leg {leg.k}: cached theta is {rec.get('backend')!r} for "
              f"{len(rec.get('box', []))} walker(s); this run wants {want_backend!r} "
              f"for {ctx.n_walkers} - recomputing")

    if leg.C == 0.0 and ctx.skip_null_scores:
        # V_k = C_k * z is identically zero here, so the score cannot affect a single
        # clone or kill; buying it would cost the longest score forecast of the whole run
        # (from 3 d, i.e. 18 days of FCN3 x M x N = ~7.7 H100-h at the pilot's size) for a
        # diagnostic Gate 3 has already measured on 16 walkers (rho_s = -0.044: no skill
        # at 3 d, which is WHY C_1 is zero). The persistence index is recorded instead
        # because it is free.
        rec = theta_from_persistence(ctx, leg)
        rec.update(backend="skipped(C=0)", note="C_k = 0; the score cannot affect "
                   "resampling, so no FCN3 forecast was bought. Values are the "
                   "persistence index, recorded as a diagnostic only.",
                   used=[0.0] * ctx.n_walkers)
    elif ctx.backend == "fcn3":
        ensure_leg_scored(ctx, leg)
        rec = theta_from_fcn3(ctx, leg)
    elif ctx.backend == "persistence":
        rec = theta_from_persistence(ctx, leg)
    else:
        raise SystemExit(f"unknown score backend {ctx.backend!r}; "
                         f"want one of {SCORE_BACKENDS}")

    if "used" not in rec:
        # The sign is applied HERE, before dmc ever sees the score: resampling clones the
        # largest values, and a cold event's extreme is the large NEGATIVE anomaly.
        rec["used"] = [ctx.sign * v for v in rec["box"]]
    rec.update(leg=leg.k, lead_days=leg.lead_end_days, C=leg.C, tail_sign=ctx.sign,
               computed=str(pd.Timestamp.utcnow()))
    write_json(p, rec)
    return rec


# --------------------------------------------------------------------------- #
# stage: prep (CPU, login node)
# --------------------------------------------------------------------------- #
def gate3_seedable(ctx: Ctx) -> list[tuple[int, int]]:
    """(walker, segment) pairs whose Gate 3 state is PROVABLY this run's state too.

    A walker segment is a deterministic function of (event, base seed, slot, segment) and
    its parent. Gate 3 rolled 16 free-running walkers from the same init with the same
    base seed, so:

      * segment 1 always matches - both continue the event's ERA5 init;
      * segment 2 matches only if C_1 = 0, because then the first resampling has V = 0,
        every expected multiplicity is exactly 1, and the parent map is the identity, so
        slot w continues its own segment 1 exactly as Gate 3's walker does.

    Nothing beyond that is reusable: from the second resampling on, the slots hold clones.
    ~1.5 of the pilot's ~39 H100-h, and it is checked rather than assumed - the states are
    validated on valid time, parent lineage, checkpoint and base seed before being linked.
    """
    if ctx.base_seed != A.WALKER_BASE_SEED:
        return []
    n = min(ctx.n_walkers, A.GATE3_N_WALKERS)
    segs = [1] + ([2] if ctx.C and ctx.C[0] == 0.0 else [])
    segs = [s for s in segs if s in set(ctx.segments)]
    out = []
    for w in range(n):
        for s in segs:
            if A.state_path(ctx.event, w, s).exists() and A.diag_path(ctx.event, w, s).exists():
                out.append((w, s))
    return out


def seed_from_gate3(ctx: Ctx, pairs) -> int:
    """Hardlink Gate 3's states/diagnostics into the run tree. Never copies, never moves."""
    from aires import walker as W

    linked = 0
    for (w, s) in pairs:
        for src, dst in ((A.state_path(ctx.event, w, s), ctx.state_path(w, s)),
                         (A.diag_path(ctx.event, w, s), ctx.diag_path(w, s))):
            if dst.exists():
                continue
            with xr.open_dataset(src) as ds:
                got, attrs = W.valid_time(ds), dict(ds.attrs)
            want = ctx.valid_at(s * A.GATE3_SEG_DAYS)
            if pd.Timestamp(got) != want:
                print(f"  [prep] skip w{w:02d} step{s}: valid {got}, want {want}")
                continue
            if attrs.get("params_file") != A.walker_model_kwargs()["params_file"]:
                print(f"  [prep] skip w{w:02d} step{s}: checkpoint "
                      f"{attrs.get('params_file')!r}")
                continue
            if lineage_key(attrs.get("parent")) != lineage_key(
                    None if s == 1 else A.state_path(ctx.event, w, s - 1)):
                print(f"  [prep] skip w{w:02d} step{s}: parent "
                      f"{attrs.get('parent')!r}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                dst.symlink_to(src)
            linked += 1
    return linked


def stage_prep(ctx: Ctx, *, seed: bool = True, force: bool = False) -> int:
    from aires import aindex as AI

    A.res_ensure_dirs(ctx.event, ctx.tag)
    print(f"AI+RES {ctx.tag} - {ctx.event}  (init {ctx.ev.init.date()}, peak "
          f"{ctx.ev.peak}, metric {ctx.ev.metric})")
    print(AI.describe(ctx.event))
    print(f"\n  N walkers        {ctx.n_walkers}")
    print(f"  score backend    {ctx.backend}   M = {ctx.members}")
    print(f"  tail             {'HIGH (clone the largest A_L)' if ctx.sign > 0 else 'LOW (cold event: clone the most negative A_L)'}")
    print(f"  C_k              {', '.join(f'{c:g}' for c in ctx.C)} at "
          f"{', '.join(f'{l:g} d' for l in ctx.leads)}")
    print(f"  legs")
    for leg in ctx.legs:
        tail = (f"C = {leg.C:g}" if leg.resampling else
                "carry to the horizon (no resampling)")
        print(f"    {leg.k}: {leg.lead_start_days:g} -> {leg.lead_end_days:g} d, "
              f"segments {list(leg.segments)}, {tail}")

    init_ok = A.gencast_inputs_path(ctx.event).exists()
    calib_ok = A.CALIB_PATH.exists()
    ref = F.gencast_cube_path(ctx.ev)
    print(f"\n  prerequisites")
    print(f"    walker init frames  {'OK ' if init_ok else 'MISSING'}  "
          f"{A.gencast_inputs_path(ctx.event)}")
    print(f"    derived calibration {'OK ' if calib_ok else 'MISSING'}  {A.CALIB_PATH}")
    print(f"    xres reference cube {'OK ' if ref.exists() else 'absent '}  {ref}"
          f"   (physical check on the first segment; not required)")

    if seed:
        pairs = gate3_seedable(ctx)
        if pairs:
            n = seed_from_gate3(ctx, pairs)
            print(f"\n  seeded {n} artifact(s) from the Gate 3 tree for "
                  f"{len(pairs)} (walker, segment) pair(s) - hardlinks, not copies.\n"
                  f"    Segment 1 is the same rollout in both runs; segment 2 is too "
                  f"because C_1 = 0 makes the first resampling the identity.")

    # what is already on disk
    have_states = sum(1 for w in range(ctx.n_walkers) for s in ctx.segments
                      if ctx.state_path(w, s).exists())
    have_diag = sum(1 for w in range(ctx.n_walkers) for s in ctx.segments
                    if ctx.diag_path(w, s).exists())
    total_seg = ctx.n_walkers * len(ctx.segments)
    # Legs that will actually buy an FCN3 forecast. The persistence backend buys none at
    # all - its score comes free from the walkers' own diagnostics - so reporting the
    # FCN3 budget for it would overstate the control run by the entire score phase.
    scored_legs = ([l for l in ctx.legs if l.resampling
                    and not (l.C == 0.0 and ctx.skip_null_scores)]
                   if ctx.backend == "fcn3" else [])
    have_cubes = sum(1 for l in scored_legs for w in range(ctx.n_walkers)
                     if S.Task(ctx.event, w, l.lead_end_days, ctx.tag).cube_path.exists())
    want_cubes = len(scored_legs) * ctx.n_walkers
    cubes = (f"{have_cubes}/{want_cubes} score cubes" if want_cubes else
             f"no FCN3 forecasts (backend {ctx.backend!r} scores from the walkers "
             f"themselves)")
    print(f"\n  cached            {have_diag}/{total_seg} walker segments, {cubes} "
          f"({have_states} states resident)")

    # budget
    gsteps = (total_seg - have_diag) * A.RES_SEG_STEPS
    fsteps = sum((ctx.n_walkers - sum(1 for w in range(ctx.n_walkers)
                                      if S.Task(ctx.event, w, l.lead_end_days,
                                                ctx.tag).cube_path.exists()))
                 * ctx.members * S.nsteps_for(ctx.ev, ctx.valid_at(l.lead_end_days))
                 for l in scored_legs)
    # Both rates are MEASURED on this cluster's H100s, not assumed - see the constants.
    walk_h = gsteps * SECONDS_PER_GENCAST_STEP / 3600
    score_h = fsteps * SECONDS_PER_FCN3_STEP / 3600
    print(f"\n  budget            walk {gsteps:,} GenCast steps ~{walk_h:.1f} GPU-h "
          f"(~{walk_h / ctx.nshards * 60:.0f} min over {ctx.nshards} GPUs)")
    print(f"                    score {fsteps:,} FCN3 steps ~{score_h:.1f} GPU-h "
          f"(~{score_h / ctx.nshards * 60:.0f} min over {ctx.nshards} GPUs)")
    print(f"                    total ~{walk_h + score_h:.1f} H100-h, "
          f"~{(walk_h + score_h) / ctx.nshards:.1f} h wall on one node")

    # disk
    live = max(ctx.keep_states, 2) + 1 if ctx.keep_states > 0 else len(ctx.segments)
    peak_gb = live * ctx.n_walkers * 0.477
    diag_gb = total_seg * 0.006
    cube_gb = want_cubes * 0.010          # 0 for a backend that buys no forecasts
    free_gb = shutil.disk_usage(A.AIRES_ROOT).free / 1e9
    need = peak_gb + diag_gb + cube_gb
    print(f"\n  disk              states {peak_gb:.0f} GB peak "
          f"({live} segments resident x {ctx.n_walkers} x 477 MB, "
          f"--keep-states {ctx.keep_states}), diagnostics {diag_gb:.0f} GB, "
          f"score cubes {cube_gb:.0f} GB")
    print(f"                    ~{need:.0f} GB needed, {free_gb:.0f} GB free")
    disk_ok = need < 0.8 * free_gb
    if not disk_ok:
        print(f"    !! {need:.0f} GB is more than 80% of the {free_gb:.0f} GB free. "
              f"Lower --keep-states, or free space first.")

    ready = init_ok and calib_ok and disk_ok
    if ready:
        cfg = freeze_config(ctx, force=force)
        print(f"\n  configuration frozen at {run_json_path(ctx)}")
    print(f"\n  READY: {'yes' if ready else 'NO - fix the items marked above'}")
    return 0 if ready else 1


# --------------------------------------------------------------------------- #
# stage: walk (GPU worker, moe env)
# --------------------------------------------------------------------------- #
def stage_walk(ctx: Ctx, leg_k: int, shard: int, nshards: int) -> int:
    from aires import walker as W

    A.res_ensure_dirs(ctx.event, ctx.tag)
    leg = ctx.leg(leg_k)
    man = read_walk_manifest(ctx, leg_k)
    parents = man.get("parents")
    mine = [w for w in range(ctx.n_walkers) if w % nshards == shard]
    print(f"[walk] host={socket.gethostname()} shard {shard}/{nshards} leg {leg_k} "
          f"({leg.lead_start_days:g} -> {leg.lead_end_days:g} d): walkers {mine}")

    get_bundle = W.lazy_bundle()
    t0, rolled = time.perf_counter(), 0
    for w in mine:
        for s in leg.segments:
            parent = parent_of(ctx, leg, w, s, parents)
            st, why = segment_status(ctx, w, s, parent)
            if st == "ok":
                continue
            if st in ("stale", "partial"):
                print(f"[walk] w{w:02d} step{s} is {st} ({why}) - re-rolling")
                shutil.rmtree(A.res_segment_dir(ctx.event, w, s, ctx.tag),
                              ignore_errors=True)
            if parent is not None and not parent.exists():
                print(f"[walk] ERROR: w{w:02d} step{s} needs parent {parent}",
                      file=sys.stderr)
                return 1
            W.run_segment(get_bundle, ctx.event, w, s, A.RES_SEG_STEPS,
                          parent_state=parent, base_seed=ctx.base_seed,
                          state_path=ctx.state_path(w, s),
                          diag_path=ctx.diag_path(w, s))
            rolled += 1
            st, why = segment_status(ctx, w, s, parent)
            if st != "ok":
                print(f"[walk] ERROR: w{w:02d} step{s} landed {st}: {why}",
                      file=sys.stderr)
                return 1
            if rolled == 1:
                # Fail in three minutes, not at the end of the leg. This is the check that
                # caught the wrong-checkpoint walkers (71 sd out); it compares against the
                # cached xres ensemble at the same valid time, and it is only meaningful
                # while the population is still untilted.
                if leg_k == 1:
                    _check_against_reference(ctx, w, s)
    print(f"[walk] shard {shard}: {rolled} segment(s) rolled, "
          f"{time.perf_counter() - t0:.0f} s")
    return 0


def _check_against_reference(ctx: Ctx, walker: int, segment: int) -> None:
    """``gate3.check_against_reference``, pointed at this run's tree."""
    from aires import aindex as AI
    from aires import gate3 as G3

    ref = F.gencast_cube_path(ctx.ev)
    if not ref.exists():
        print(f"[walk] no xres reference cube at {ref}; skipping the physical check")
        return
    with xr.open_dataset(ctx.diag_path(walker, segment)) as d, xr.open_dataset(ref) as x:
        t = pd.DatetimeIndex(d["time"].values)[-1]
        xt = list(pd.DatetimeIndex(x["time"].values))
        if t not in set(xt):
            return
        got = float(AI.area_mean(d["2m_temperature"].isel(batch=0, time=-1)))
        ens = AI.area_mean(x["2m_temperature"].isel(time=xt.index(t))).values
        mu, sd = float(ens.mean()), float(ens.std(ddof=1))
    tol = max(G3.REFERENCE_TOL_K, G3.REFERENCE_TOL_SD * sd)
    off = abs(got - mu)
    print(f"[walk] physical check w{walker:02d} step{segment} @ {t}: CONUS-mean T2m "
          f"{got:.2f} K vs xres {mu:.2f} +- {sd:.3f} K over {ens.size} members "
          f"({off / sd if sd else float('nan'):.1f} sd)")
    if off > tol:
        raise RuntimeError(
            f"w{walker:02d} step{segment}: CONUS-mean T2m is {got:.2f} K at {t}, but the "
            f"cached xres ensemble from the SAME checkpoint and init gives "
            f"{mu:.2f} +- {sd:.3f} K (tolerance {tol:.2f} K). Check which checkpoint was "
            f"loaded: {A.walker_model_kwargs()['params_file']!r} was requested.")


# --------------------------------------------------------------------------- #
# stage: res (the coordinator)
# --------------------------------------------------------------------------- #
def stage_res(ctx: Ctx, *, force: bool = False) -> int:
    A.res_ensure_dirs(ctx.event, ctx.tag)
    freeze_config(ctx, force=force)
    ngpu = visible_gpus()
    print(f"[res] {ctx.event} tag={ctx.tag} N={ctx.n_walkers} M={ctx.members} "
          f"backend={ctx.backend} C={ctx.C}")
    print(f"[res] host={socket.gethostname()} shards={ctx.nshards} "
          f"visible GPUs={ngpu} seed={ctx.dmc_seed}")
    if ngpu and ngpu < ctx.nshards:
        print(f"[res] WARNING: {ctx.nshards} shards requested but only {ngpu} GPU(s) "
              f"visible; shards will share cards and may OOM", file=sys.stderr)

    timings: dict[str, float] = {}

    def propagate(k, parents, states):
        leg = ctx.leg(k)
        t0 = time.perf_counter()
        write_walk_manifest(ctx, leg, parents)
        prune_states(ctx, leg)
        ensure_leg_walked(ctx, leg, parents)
        timings[f"walk{k}"] = time.perf_counter() - t0
        return leg

    def score(k, leg):
        t0 = time.perf_counter()
        rec = leg_theta(ctx, leg)
        timings[f"score{k}"] = time.perf_counter() - t0
        used = np.asarray(rec["used"], dtype="float64")
        print(f"  [score] leg {k} ({rec['backend']}): theta over the box "
              f"mean {np.mean(rec['box']):+.3f}, sd {np.std(rec['box'], ddof=1):.3f}")
        return used

    t_start = time.perf_counter()
    result = dmc.run(ctx.n_walkers, ctx.C, propagate, score,
                     rng=np.random.default_rng(ctx.dmc_seed), verbose=True)
    wall = time.perf_counter() - t_start

    out = dict(config=ctx.config(), dmc=result.to_dict(), wall_seconds=wall,
               timings=timings, host=socket.gethostname(),
               job=os.environ.get("SLURM_JOB_ID", "local"),
               theta=[read_json(theta_path(ctx, l.k)) for l in ctx.legs if l.resampling],
               finished=str(pd.Timestamp.utcnow()))
    p = A.res_dir(ctx.event, ctx.tag) / "res_result.json"
    write_json(p, out)

    print(f"\n{result.summary()}")
    print(f"\n[res] finished in {wall / 60:.1f} min -> {p}")
    print(f"[res] now reduce on the login node:\n"
          f"  PYTHONPATH=. python -m aires.run_aires --stage ds --event {ctx.event}\n"
          f"  PYTHONPATH=. python -m aires.run_aires --stage compare --event "
          f"{ctx.event} --tag {ctx.tag}")
    return 0


# --------------------------------------------------------------------------- #
# Trajectory assembly across the genealogy
# --------------------------------------------------------------------------- #
def slot_lineage(ctx: Ctx, parents_by_leg: dict, final_slot: int) -> dict[int, int]:
    """``{segment: slot}`` for one final walker, walking the genealogy backwards.

    A slot is not a lineage: walker 3's segment-6 state and its segment-2 state can belong
    to completely different ancestries. Every read of a trajectory has to go through this.
    """
    slots, cur = {}, int(final_slot)
    for leg in reversed(ctx.legs):
        for s in leg.segments:
            slots[s] = cur
        par = parents_by_leg.get(leg.k)
        if par is not None:
            cur = int(par[cur])
    return slots


def concat_diag(paths) -> xr.Dataset:
    """One walker's path from its per-segment diagnostics cubes.

    ``lead_h`` is dropped: it is relative to each SEGMENT's start, so keeping it would
    give a coordinate that runs 0..72 once per segment. ``time`` is absolute and is what
    everything downstream joins on.
    """
    parts = []
    for p in paths:
        d = xr.open_dataset(p).load()
        parts.append(d.drop_vars([c for c in ("lead_h",) if c in d.coords]))
    out = xr.concat(parts, dim="time", coords="minimal", compat="override")
    for d in parts:
        d.close()
    return out.sortby("time")


def parents_by_leg(ctx: Ctx) -> dict:
    out = {}
    for leg in ctx.legs:
        p = walk_manifest_path(ctx, leg.k)
        if p.exists():
            out[leg.k] = read_json(p).get("parents")
    return out


def realized_indices(ctx: Ctx) -> dict:
    """A_L and the running index for every walker in the FINAL population."""
    from aires import aindex as AI

    par = parents_by_leg(ctx)
    box, conus, series_box, series_conus, lineages = [], [], [], [], []
    lead = None
    for w in range(ctx.n_walkers):
        slots = slot_lineage(ctx, par, w)
        paths = [ctx.diag_path(slots[s], s) for s in sorted(slots)]
        traj = concat_diag(paths)
        idx = AI.indices(traj, ctx.event, ctx.ev.peak, ctx.ev.metric,
                         where=f"w{w:02d} trajectory")
        ser = AI.index_series(traj, ctx.event, ctx.ev.metric)
        t = pd.DatetimeIndex(traj["time"].values)
        if lead is None:
            lead = ((t - pd.Timestamp(ctx.ev.init)).total_seconds() / 86400.0).tolist()
        box.append(float(idx["box"]))
        conus.append(float(idx["conus"]))
        series_box.append(np.asarray(ser["box"].values, dtype=float).tolist())
        series_conus.append(np.asarray(ser["conus"].values, dtype=float).tolist())
        lineages.append({str(s): int(v) for s, v in sorted(slots.items())})
        traj.close()
    return dict(box=box, conus=conus, series_box=series_box, series_conus=series_conus,
                lead_days=lead, lineage=lineages)


# --------------------------------------------------------------------------- #
# stage: ds (the direct-sampling baseline)
# --------------------------------------------------------------------------- #
def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def ds_sources(ctx: Ctx) -> dict:
    """Every direct sample of this event's A_L that already exists on disk.

    None of it needs a GPU: the Gate 3 walkers, the 24-member xres GenCast cube and the
    24-member FCN3 cube were all produced by earlier phases. They are what AI+RES has to
    beat, and the honest comparison is against the GenCast ensembles - the FCN3 cube is a
    different model and is carried for context.
    """
    from aires import aindex as AI

    out = {}
    segs = A.gate3_segments()
    have = [w for w in range(A.GATE3_N_WALKERS)
            if all(A.diag_path(ctx.event, w, k).exists() for (k, _l, _n) in segs)]
    if have:
        vals_b, vals_c = [], []
        for w in have:
            traj = concat_diag([A.diag_path(ctx.event, w, k) for (k, _l, _n) in segs])
            idx = AI.indices(traj, ctx.event, ctx.ev.peak, ctx.ev.metric,
                             where=f"gate3 w{w:02d}")
            vals_b.append(float(idx["box"]))
            vals_c.append(float(idx["conus"]))
            traj.close()
        out["gencast_walkers"] = dict(
            box=vals_b, conus=vals_c, n=len(have),
            note="Gate 3's free-running GenCast walkers - the same model, the same "
                 "init and the same segment schedule as the AI+RES run, with no "
                 "resampling. This is the like-for-like direct sample.")

    for key, path, note in (
        ("gencast_xres", F.gencast_cube_path(ctx.ev),
         "the cached 24-member 0.25 deg GenCast ensemble from the xres experiment"),
        ("fcn3", F.cube_path(ctx.ev),
         "the 24-member FCN3 ensemble from the FCN3-vs-GenCast experiment - a different "
         "model, carried for context"),
    ):
        if not path.exists():
            continue
        with xr.open_dataset(path) as cube:
            idx = AI.indices(cube, ctx.event, ctx.ev.peak, ctx.ev.metric, where=key)
        out[key] = dict(box=np.asarray(idx["box"].values, dtype=float).tolist(),
                        conus=np.asarray(idx["conus"].values, dtype=float).tolist(),
                        n=int(np.asarray(idx["box"].values).size), note=note,
                        source=str(path))

    tp = F.era5_truth_path(ctx.ev)
    if tp.exists():
        with xr.open_dataset(tp) as tds:
            da = tds[list(tds.data_vars)[0]]
            out["observed"] = {k: float(AI.area_mean(da, b))
                               for k, b in AI.boxes_for(ctx.event).items()}
    return out


def ds_path(ctx: Ctx) -> Path:
    return A.event_dir(ctx.event) / "ds_baseline.json"


def stage_ds(ctx: Ctx) -> int:
    out = ds_sources(ctx)
    if not any(k in out for k in ("gencast_walkers", "gencast_xres")):
        print("[ds] ERROR: no direct-sampling source on disk (no Gate 3 walkers, no "
              "cached xres cube)", file=sys.stderr)
        return 1
    obs = out.get("observed", {})
    print(f"[ds] {ctx.event}: direct-sampling baseline, metric {ctx.ev.metric}, "
          f"box {ctx.sign:+.0f} tail")
    for key in ("gencast_walkers", "gencast_xres", "fcn3"):
        if key not in out:
            continue
        v = np.asarray(out[key]["box"])
        extra = ""
        if "box" in obs:
            n = int(np.sum(ctx.sign * v >= ctx.sign * obs["box"]))
            extra = f"   reaching the observed {obs['box']:+.2f}: {n}/{v.size}"
        print(f"  {key:16s} n={v.size:3d}  mean {v.mean():+7.3f}  sd "
              f"{v.std(ddof=1):.3f}  range [{v.min():+.2f}, {v.max():+.2f}]{extra}")
    out["event"] = ctx.event
    out["metric"] = ctx.ev.metric
    out["tail_sign"] = ctx.sign
    p = write_json(ds_path(ctx), out)
    print(f"[ds] -> {p}")
    return 0


# --------------------------------------------------------------------------- #
# stage: compare
# --------------------------------------------------------------------------- #
def stage_compare(ctx: Ctx) -> int:
    rp = A.res_dir(ctx.event, ctx.tag) / "res_result.json"
    if not rp.exists():
        print(f"[compare] ERROR: no {rp}; run --stage res first", file=sys.stderr)
        return 1
    peak_lead = (pd.Timestamp(ctx.ev.peak) - pd.Timestamp(ctx.ev.init)) / pd.Timedelta(days=1)
    if abs(ctx.horizon_days - peak_lead) > 1e-9:
        print(f"[compare] ERROR: this run stops at {ctx.horizon_days:g} d but A_L is the "
              f"mean over the 7 days ENDING at the peak ({peak_lead:g} d lead), so no "
              f"walker reaches the verification window. A shortened horizon is a smoke "
              f"run: --stage res exercises the loop, --stage compare cannot score it.",
              file=sys.stderr)
        return 1
    rec = read_json(rp)
    result = dmc.DMCResult.from_dict(rec["dmc"])
    sign = ctx.sign

    print(f"[compare] {ctx.event} tag={ctx.tag}: assembling {ctx.n_walkers} "
          f"trajectories across the genealogy")
    real = realized_indices(ctx)
    al = np.asarray(real["box"], dtype="float64")
    weights = result.weights

    ds = read_json(ds_path(ctx)) if ds_path(ctx).exists() else {}
    obs = ds.get("observed", {}).get("box")

    # Thresholds span every method's population, not just the RES one: a grid that stopped
    # at the RES range would leave the direct-sampling curves undefined exactly where they
    # are being compared, and vice versa.
    span = [al]
    for key in ("gencast_walkers", "gencast_xres", "fcn3"):
        if key in ds:
            span.append(np.asarray(ds[key]["box"], dtype=float))
    allv = np.concatenate(span)
    grid = list(np.linspace(float(allv.min()), float(allv.max()), 40))
    if obs is not None:
        grid.append(float(obs))
    thresholds = sorted(set(round(float(x), 4) for x in grid), reverse=sign < 0)

    def res_p(a: float) -> float:
        return float(result.expectation((sign * al >= sign * a).astype(float)))

    curve = []
    for a in thresholds:
        row = dict(threshold=a, res=res_p(a))
        for key in ("gencast_walkers", "gencast_xres", "fcn3"):
            if key in ds:
                v = np.asarray(ds[key]["box"], dtype=float)
                k = int(np.sum(sign * v >= sign * a))
                row[key] = k / v.size
                row[f"{key}_ci"] = list(_wilson(k, v.size))
        curve.append(row)

    print(f"\n  DMC diagnostics")
    print(result.summary())
    norm = result.normalization_check()
    # Z * mean(exp(-V_K)) estimates E_P[1]. It is noisy - dmc.py records a sd of ~0.5 on
    # the reference OU problem at N=256 - so this is a flag to read across runs, not a
    # threshold. Far from 1 with a probability above 1 is a bookkeeping error, not noise.
    if not (0.2 <= norm <= 5.0):
        print(f"  WARNING: normalization check is {norm:.3g}, well away from 1. That is a "
              f"statement about log Z or the final weights, and every probability below "
              f"is scaled by the same factor. Read it against the other backend's run "
              f"before trusting a number.", file=sys.stderr)

    print(f"\n  realized A_L over the {ctx.n_walkers} final walkers (weighted by "
          f"exp(-V_K), which is what makes the estimate unbiased)")
    print(f"    unweighted   mean {al.mean():+7.3f}  sd {al.std(ddof=1):.3f}  "
          f"range [{al.min():+.2f}, {al.max():+.2f}]")
    wmean = float(np.sum(weights * al) / np.sum(weights))
    print(f"    weighted     mean {wmean:+7.3f}   (weights span "
          f"{weights.min():.3g} .. {weights.max():.3g})")
    if "gencast_walkers" in ds:
        v = np.asarray(ds["gencast_walkers"]["box"], dtype=float)
        print(f"    direct (N={v.size})  mean {v.mean():+7.3f}  sd {v.std(ddof=1):.3f}  "
              f"range [{v.min():+.2f}, {v.max():+.2f}]")

    if obs is not None:
        reach = int(np.sum(sign * al >= sign * obs))
        print(f"\n  the headline question")
        print(f"    ERA5 observed A_L over the box: {obs:+.3f}")
        print(f"    AI+RES walkers reaching it:     {reach}/{ctx.n_walkers}   "
              f"P = {res_p(obs):.3g}")
        for key in ("gencast_walkers", "gencast_xres"):
            if key in ds:
                v = np.asarray(ds[key]["box"], dtype=float)
                k = int(np.sum(sign * v >= sign * obs))
                clo, chi = _wilson(k, v.size)
                print(f"    {key:16s}              {k}/{v.size}   "
                      f"P = {k / v.size:.3g}  [{clo:.3g}, {chi:.3g}]")

    out = dict(event=ctx.event, tag=ctx.tag, config=rec["config"],
               realized=real, weights=weights.tolist(), log_Z=result.log_Z,
               normalization_check=norm, curve=curve, observed=obs,
               ess_by_step=result.ess_by_step.tolist(),
               max_multiplicity_by_step=result.max_multiplicity_by_step.tolist(),
               n_founders=result.n_founders, ancestry=result.ancestry().tolist(),
               ds={k: v for k, v in ds.items() if k != "observed"})
    p = write_json(A.res_dir(ctx.event, ctx.tag) / "compare.json", out)

    csv = A.res_dir(ctx.event, ctx.tag) / "compare_curve.csv"
    cols = ["threshold", "res"] + [k for k in ("gencast_walkers", "gencast_xres", "fcn3")
                                   if k in ds]
    lines = [",".join(cols)]
    for row in curve:
        lines.append(",".join(f"{row.get(c, float('nan')):.6g}" for c in cols))
    csv.write_text("\n".join(lines) + "\n")
    print(f"\n[compare] -> {p}\n[compare] -> {csv}")
    return 0


# --------------------------------------------------------------------------- #
def build_ctx(a) -> Ctx:
    leads = tuple(float(x) for x in str(a.leads).split(",") if str(x).strip())
    C = tuple(float(x) for x in str(a.C).split(",") if str(x).strip())
    return Ctx(ev=_event(a.event), tag=a.tag, n_walkers=a.walkers, members=a.members,
               C=C, leads=leads, horizon_days=a.horizon, dmc_seed=a.dmc_seed,
               base_seed=a.base_seed, backend=a.backend,
               skip_null_scores=not a.score_null, nshards=a.nshards,
               attempts=a.attempts, retry_wait=a.retry_wait,
               keep_states=a.keep_states, dry_run=getattr(a, "dry_run", False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["prep", "walk", "score", "res", "ds", "compare"])
    ap.add_argument("--event", default=os.environ.get("AIRES_EVENT", DEFAULT_EVENT))
    ap.add_argument("--tag", default=A.RES_TAG)
    ap.add_argument("--walkers", type=int, default=A.RES_N_WALKERS)
    ap.add_argument("--members", type=int, default=A.RES_M_MEMBERS)
    ap.add_argument("--C", default=",".join(f"{c:g}" for c in A.RES_C))
    ap.add_argument("--leads", default=",".join(f"{l:g}" for l in A.RES_LEAD_DAYS))
    ap.add_argument("--horizon", type=float, default=A.RES_HORIZON_DAYS)
    ap.add_argument("--backend", default=os.environ.get("AIRES_BACKEND", "fcn3"),
                    choices=list(SCORE_BACKENDS))
    ap.add_argument("--score-null", action="store_true",
                    help="also buy the FCN3 score where C_k = 0 (a pure diagnostic; it "
                         "cannot change a single clone or kill)")
    ap.add_argument("--dmc-seed", type=int, default=A.RES_DMC_SEED)
    ap.add_argument("--base-seed", type=int, default=A.WALKER_BASE_SEED)
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("AIRES_NSHARDS", 8)))
    ap.add_argument("--shard", type=int, default=None,
                    help="walk: be ONE shard instead of launching the pool")
    ap.add_argument("--leg", type=int, default=None)
    ap.add_argument("--keep-states", type=int, default=A.RES_KEEP_STATES)
    ap.add_argument("--attempts", type=int, default=int(os.environ.get("AIRES_POOL_ATTEMPTS", 3)))
    ap.add_argument("--retry-wait", type=int, default=int(os.environ.get("AIRES_RETRY_WAIT", 300)))
    ap.add_argument("--no-seed", action="store_true",
                    help="prep: do not seed the run tree from the Gate 3 walkers")
    ap.add_argument("--dry-run", action="store_true",
                    help="res: stop rather than launch a GPU pool")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a run.json that disagrees with these arguments")
    a = ap.parse_args(argv)
    ctx = build_ctx(a)

    signal.signal(signal.SIGTERM, _kill_children)
    signal.signal(signal.SIGINT, _kill_children)

    if a.stage == "prep":
        return stage_prep(ctx, seed=not a.no_seed, force=a.force)
    if a.stage == "res":
        return stage_res(ctx, force=a.force)
    if a.stage == "ds":
        return stage_ds(ctx)
    if a.stage == "compare":
        return stage_compare(ctx)
    if a.stage == "walk":
        if a.leg is None:
            raise SystemExit("--stage walk needs --leg")
        if a.shard is None:
            leg = ctx.leg(a.leg)
            run_pool(ctx, f"walk-leg{a.leg:02d}", "moe",
                     ["-m", "aires.run_aires", "--stage", "walk", "--event", ctx.event,
                      "--tag", ctx.tag, "--leg", str(a.leg),
                      "--walkers", str(ctx.n_walkers)])
            return 0
        return stage_walk(ctx, a.leg, a.shard, a.nshards)
    if a.stage == "score":
        if a.leg is None:
            raise SystemExit("--stage score needs --leg")
        ensure_leg_scored(ctx, ctx.leg(a.leg))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
