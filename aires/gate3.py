#!/usr/bin/env python
"""Gate 3 - score skill. The go/no-go for AI+RES.

Gates 1 and 2 asked whether the GenCast -> FCN3 handoff is faithful: 69 of 72 channels are
bit-identical, the other three are derived to within 0.63 m/s / 4.6%, and an adapter-built
IC moves an FCN3 forecast less than re-seeding the model does. Both passed. Neither says
the resulting score is USEFUL.

That is this gate, and it is the one that can stop the project:

    Does an FCN3 ensemble forecast launched from a GenCast walker's state at lead t_k
    predict which walkers end up extreme?

If it does not, cloning on that score is cloning on noise and the whole importance-
splitting scheme reduces to an expensive direct sample.

The measurement
---------------
N free-running GenCast walkers are rolled from the event init to the peak (21 d), with a
GLOBAL restart state written every 3 d. Each checkpoint at t_k in {3, 6, 9, 12, 15} d is
adapted into an FCN3 initial condition and scored:

    theta(t_k)  =  mean over M FCN3 members of A_L                (aires/score.py)
    A_L(t_f)    =  the walker's OWN realized A_L                  (its diagnostics cubes)

and the gate is the Spearman rank correlation between them, across walkers, at each t_k.
Note what the ground truth is: the walker's own outcome, not ERA5. The score does not have
to be right about the atmosphere, it has to be right about the walker - that is all
resampling needs, and it is the only thing a score function can be held to.

    PASS: rho_s >= 0.5 at t_k = 9 d AND 12 d   (aires.md)

Three things are measured alongside it, because a bare correlation cannot be interpreted
without them:

* **The persistence baseline.** Lancelin et al.'s Standard-RES scores a walker by the
  observable's CURRENT value - free, no emulator. If persistence ranks as well as FCN3
  does, the AI scorer has not earned its GPU hours, and that is a finding rather than a
  failure. It is computed from the walkers' own diagnostics cubes at zero cost.
* **The noise floor.** Gate 2's M=6 lesson was that an ensemble statistic must be compared
  against its own sampling noise before a threshold on it means anything. theta is a mean
  of M=6 members, so it carries a standard error; the correlation it can achieve is capped
  at ``sqrt(1 - var_noise/var_theta)`` no matter how skilful the score is. That ceiling is
  reported beside every rho.
* **The sampling uncertainty on rho itself.** At N=16 walkers the standard error on a rank
  correlation is ~0.26, so 0.5 is roughly the 5% significance level and not much more. The
  bootstrap CI and the significance threshold are printed with the verdict rather than
  left for a reader to remember.

The observable is the per-event BOX index (``aires/aindex.py``), not the CONUS mean Gate 2
used. For this pilot the difference is not cosmetic: the observed 7-day anomaly is
+7.7 K over the PNW box and +0.6 K over CONUS, so a CONUS-mean gate would be scoring a
signal the event barely has. Both are computed and stored; the verdict is on the box.

Stages (two conda envs)
-----------------------
    plan    login node, either env : enumerate the work, verify prerequisites
    walk    GPU, ``moe``           : roll the walkers, checkpoint every 3 d
    score   GPU, ``fcn3``          : aires/score.py - M FCN3 forecasts per checkpoint
    reduce  login node, ``moe``    : A_L, theta, correlations, verdict, figure

    PYTHONPATH=. python -m aires.gate3 --stage plan          # login node
    sbatch slurm/aires_gate3.slurm                           # walk + score, one node
    PYTHONPATH=. python -m aires.gate3 --stage reduce        # login node, moe
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from aires import score as S
from fcn3 import fevents as F

DEFAULT_EVENT = "PNW_HeatDome_2021"


def _event(name: str) -> F.Event:
    if name not in F.EVENTS:
        raise SystemExit(f"unknown event {name!r}; known: {', '.join(F.ORDER)}")
    return F.EVENTS[name]


def expected_valid(ev: F.Event, lead_days: float) -> pd.Timestamp:
    return pd.Timestamp(ev.init) + pd.Timedelta(days=float(lead_days))


def result_path(ev: F.Event) -> Path:
    return A.gate3_dir(ev.name) / f"{ev.name}_gate3.json"


def members_csv_path(ev: F.Event) -> Path:
    return A.gate3_dir(ev.name) / f"{ev.name}_gate3_walkers.csv"


# --------------------------------------------------------------------------- #
# Cached-state validation.
#
# ``walker.run_segment`` skips a segment whose state.nc and diag.nc already exist. That is
# what makes a killed job resumable, and it is also how a state written by a DIFFERENT
# schedule gets silently adopted: the 2-step smoke run (job 1154) left a w00/step01 state
# at lead 1 d, where Gate 3's schedule expects lead 3 d. Existence is not enough - the
# valid time has to be checked against the schedule before a cached segment is trusted.
# --------------------------------------------------------------------------- #
def segment_status(ev: F.Event, walker: int, step: int, lead_days: float) -> tuple[str, str]:
    """``('ok'|'missing'|'stale'|'partial', detail)`` for one cached segment."""
    from aires import walker as W

    sp, dp = A.state_path(ev.name, walker, step), A.diag_path(ev.name, walker, step)
    if not sp.exists():
        return "missing", "no state.nc"
    want = expected_valid(ev, lead_days)
    try:
        with xr.open_dataset(sp) as ds:
            got = W.valid_time(ds)
    except Exception as e:
        return "stale", f"unreadable ({type(e).__name__}: {e})"
    if pd.Timestamp(got) != want:
        return "stale", f"valid {got}, schedule wants {want}"
    if not dp.exists():
        return "partial", "state.nc present but no diag.nc"
    return "ok", str(got)


def prune_stale(ev: F.Event, walker: int, step: int) -> None:
    import shutil
    d = A.segment_dir(ev.name, walker, step)
    if d.exists():
        shutil.rmtree(d)
        print(f"  [walk] removed stale segment {d}")


# --------------------------------------------------------------------------- #
# plan (CPU, either env)
# --------------------------------------------------------------------------- #
def stage_plan(ev: F.Event, walkers: int, members: int, leads, nshards: int) -> int:
    from aires import aindex as AI

    segs = A.gate3_segments()
    print(f"Gate 3 - {ev.name}  (init {ev.init.date()}, peak {ev.peak}, "
          f"{F.LEAD_DAYS} d lead, metric {ev.metric})")
    print(AI.describe(ev.name))
    print(f"\n  walkers          {walkers}")
    print(f"  segments/walker  {len(segs)} x {A.GATE3_SEG_STEPS} x {A.WALKER_STEP_H} h "
          f"= {len(segs) * A.GATE3_SEG_STEPS} GenCast steps to the peak")
    print(f"  checkpoints      {', '.join(f'{l:g} d' for l in leads)}")
    print(f"  score members    M = {members}")

    init_ok = A.gencast_inputs_path(ev.name).exists()
    calib_ok = A.CALIB_PATH.exists()
    print(f"\n  prerequisites")
    print(f"    walker init frames  {'OK ' if init_ok else 'MISSING'}  "
          f"{A.gencast_inputs_path(ev.name)}")
    print(f"    derived calibration {'OK ' if calib_ok else 'MISSING'}  {A.CALIB_PATH}")
    truth = F.era5_truth_path(ev)
    print(f"    ERA5 truth          {'OK ' if truth.exists() else 'absent '}  {truth}"
          f"   (context only, not required)")

    # walker segments
    counts = {"ok": 0, "missing": 0, "stale": 0, "partial": 0}
    stale = []
    for w in range(walkers):
        for (k, lead, _n) in segs:
            st, detail = segment_status(ev, w, k, lead)
            counts[st] += 1
            if st == "stale":
                stale.append((w, k, lead, detail))
    print(f"\n  walker segments   {counts['ok']} cached, {counts['missing']} to roll, "
          f"{counts['stale']} STALE, {counts['partial']} partial")
    for w, k, lead, detail in stale:
        print(f"    !! w{w:02d} step{k} (lead {lead:g} d): {detail}")
    if counts["partial"]:
        print("    -> partial segments (state.nc but no diag.nc) are a crashed write and "
              "are re-rolled automatically.")
    if counts["stale"]:
        print("    -> STALE segments sit at a lead this schedule does not use, so they may "
              "belong to another run; `--stage walk --prune-stale` removes and rebuilds "
              "them, and without it the walk stage refuses to start.")

    # score tasks
    all_tasks = S.tasks(ev.name, walkers, leads)
    have = sum(1 for t in all_tasks if t.cube_path.exists())
    print(f"\n  score forecasts   {have} cached, {len(all_tasks) - have} to run "
          f"({len(all_tasks)} total = {walkers} walkers x {len(leads)} leads)")
    fcn3_steps = sum(S.nsteps_for(ev, expected_valid(ev, t.lead_days))
                     for t in all_tasks if not t.cube_path.exists()) * members
    print(f"                    {fcn3_steps:,} FCN3 steps x {members} members already "
          f"included; ~{fcn3_steps / 3600:.1f} GPU-h at 1 s/step, "
          f"~{fcn3_steps / 3600 / nshards * 60:.0f} min over {nshards} GPUs")

    gsteps = counts["missing"] * A.GATE3_SEG_STEPS
    print(f"  walker rollout    {gsteps:,} GenCast steps; ~{gsteps * 28 / 3600:.1f} GPU-h "
          f"at 28 s/step, ~{gsteps * 28 / 3600 / nshards * 60:.0f} min over {nshards} GPUs")

    print(f"\n  disk              states {counts['missing'] * 0.477:.1f} GB new "
          f"(477 MB each), diag {counts['missing'] * 0.036:.2f} GB, "
          f"score cubes ~{(len(all_tasks) - have) * 0.01:.2f} GB")
    ready = init_ok and calib_ok and counts["stale"] == 0
    print(f"\n  READY: {'yes' if ready else 'NO - fix the items marked above'}")
    return 0 if ready else 1


# --------------------------------------------------------------------------- #
# walk (GPU, moe env)
# --------------------------------------------------------------------------- #
def stage_walk(ev: F.Event, walkers: int, shard: int, nshards: int, *,
               prune_stale_states: bool) -> int:
    from aires import walker as W

    A.ensure_dirs(ev.name)
    segs = A.gate3_segments()
    mine = [w for w in range(walkers) if w % nshards == shard]
    print(f"[walk] host={socket.gethostname()} shard {shard}/{nshards}: walkers {mine}")
    print(f"[walk] {len(segs)} segments x {A.GATE3_SEG_STEPS} steps to lead "
          f"{A.GATE3_HORIZON_DAYS} d, base seed {A.WALKER_BASE_SEED}")

    # Validate every cached segment BEFORE loading the model, so a stale cache costs a
    # second rather than a GPU allocation plus a five-minute XLA compile.
    # "partial" and "stale" are not the same failure and must not be treated alike.
    # A partial segment (state.nc written, diag.nc not) is a crashed write - the state
    # alone cannot serve the reduce stage and re-rolling is the only repair, so it is
    # removed automatically; a walk killed by a walltime or an OOM must not need a flag
    # to resume. A stale one sits at a lead this schedule does not use, which means it may
    # belong to a different run (the 2-step smoke job's w00/step01 did), so it takes a
    # human decision.
    stale, partial = [], []
    for w in mine:
        for (k, lead, _n) in segs:
            st, detail = segment_status(ev, w, k, lead)
            (stale if st == "stale" else partial if st == "partial" else []).append(
                (w, k, lead, detail))
    for w, k, lead, detail in partial:
        print(f"[walk] incomplete w{w:02d} step{k} (lead {lead:g} d): {detail} - re-rolling")
        prune_stale(ev, w, k)
    if stale and not prune_stale_states:
        for w, k, lead, detail in stale:
            print(f"[walk] STALE w{w:02d} step{k} (lead {lead:g} d): {detail}",
                  file=sys.stderr)
        print("[walk] ERROR: cached segments sit at a lead Gate 3's schedule does not use, "
              "so they may belong to another run. Re-run with --prune-stale to delete and "
              "rebuild them.", file=sys.stderr)
        return 1
    for w, k, _lead, _detail in stale:
        prune_stale(ev, w, k)

    from gencast_s2s import model as M
    bundle: dict = {}

    def get_bundle():
        if not bundle:
            t0 = time.perf_counter()
            bundle.update(M.load_gencast(
                n_members=1, res=0.25 if A.WALKER_RES == "0p25" else 1.0))
            print(f"[walk] model loaded in {time.perf_counter() - t0:.0f} s", flush=True)
        return bundle

    t_start = time.perf_counter()
    rolled = 0
    for w in mine:
        parent = None
        for (k, lead, nsteps) in segs:
            st, _ = segment_status(ev, w, k, lead)
            if st == "ok":
                parent = A.state_path(ev.name, w, k)
                continue
            if st in ("stale", "partial"):
                prune_stale(ev, w, k)
            if parent is not None and not parent.exists():
                print(f"[walk] ERROR: w{w:02d} step{k} needs parent {parent}",
                      file=sys.stderr)
                return 1
            W.run_segment(get_bundle, ev.name, w, k, nsteps, parent_state=parent)
            rolled += 1
            parent = A.state_path(ev.name, w, k)
            got, _ = segment_status(ev, w, k, lead)
            if got != "ok":
                print(f"[walk] ERROR: w{w:02d} step{k} did not land on lead {lead:g} d "
                      f"({got})", file=sys.stderr)
                return 1
        print(f"[walk] w{w:02d}: complete to lead {A.GATE3_HORIZON_DAYS} d "
              f"({time.perf_counter() - t_start:.0f} s elapsed on this shard)", flush=True)
    print(f"[walk] shard {shard}: {rolled} segment(s) rolled, "
          f"{time.perf_counter() - t_start:.0f} s total")
    return 0


# --------------------------------------------------------------------------- #
# reduce (CPU, moe env)
# --------------------------------------------------------------------------- #
def trajectory(ev: F.Event, walker: int, segs) -> xr.Dataset:
    """One walker's whole 21-day path, from its per-segment diagnostics cubes.

    ``lead_h`` is dropped before the concat: it is relative to each SEGMENT's own start
    (walker.py says so explicitly), so keeping it would produce a coordinate that runs
    0..72 seven times over. ``time`` is absolute and is what everything downstream joins on.
    """
    parts = []
    for (k, _lead, _n) in segs:
        p = A.diag_path(ev.name, walker, k)
        if not p.exists():
            raise FileNotFoundError(f"w{walker:02d}: missing {p}")
        d = xr.open_dataset(p).load()
        parts.append(d.drop_vars([c for c in ("lead_h",) if c in d.coords]))
    traj = xr.concat(parts, dim="time")
    if "batch" in traj.dims:
        traj = traj.isel(batch=0, drop=True)
    t = pd.DatetimeIndex(traj["time"].values)
    traj = traj.isel(time=np.argsort(t.values))
    t = pd.DatetimeIndex(traj["time"].values)
    if not t.is_unique:
        traj = traj.isel(time=~t.duplicated(keep="first"))
    return traj


def _spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(x), np.asarray(y))
    return float(r.statistic), float(r.pvalue)


def _bootstrap_rho(x, y, n_boot: int = 5000, seed: int = 12345):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = x.size
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(x[idx]) == 0 or np.ptp(y[idx]) == 0:
            out[b] = np.nan
            continue
        out[b] = spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def _rho_significance(n: int, alpha: float = 0.05) -> float:
    """Critical Spearman rho for a one-sided test at ``alpha`` with ``n`` walkers."""
    from scipy.stats import t as tdist
    if n < 4:
        return float("nan")
    tc = tdist.ppf(1 - alpha, n - 2)
    return float(tc / np.sqrt(n - 2 + tc ** 2))


def stage_reduce(ev: F.Event, walkers: int, members: int, leads, *, plot: bool = True,
                 allow_partial: bool = False) -> int:
    from aires import aindex as AI

    A.ensure_dirs(ev.name)
    A.gate3_dir(ev.name).mkdir(parents=True, exist_ok=True)
    segs = A.gate3_segments()
    keys = ("box", "conus")
    box = AI.box_for(ev.name)

    # ---- realized outcome + running index, per walker ------------------------ #
    have, missing = [], []
    for w in range(walkers):
        if all(A.diag_path(ev.name, w, k).exists() for (k, _l, _n) in segs):
            have.append(w)
        else:
            missing.append(w)
    if missing and not allow_partial:
        print(f"[reduce] ERROR: {len(missing)} walker(s) have incomplete trajectories: "
              f"{missing}\n         Finish the walk stage, or pass --allow-partial to "
              f"score the {len(have)} that are complete.", file=sys.stderr)
        return 1
    if not have:
        print("[reduce] ERROR: no complete walker trajectories", file=sys.stderr)
        return 1
    if missing:
        print(f"[reduce] WARNING: scoring {len(have)} of {walkers} walkers; "
              f"{missing} are incomplete")

    realized = {k: [] for k in keys}
    series = {k: [] for k in keys}
    persistence = {k: {float(l): [] for l in leads} for k in keys}
    series_lead = None
    print(f"[reduce] {ev.name}: {len(have)} walker(s), metric {ev.metric}, "
          f"box {box.name} ({box.label})")
    for w in have:
        traj = trajectory(ev, w, segs)
        idx = AI.indices(traj, ev.name, ev.peak, ev.metric, where=f"w{w:02d} trajectory")
        for k in keys:
            realized[k].append(float(idx[k]))
        ser = AI.index_series(traj, ev.name, ev.metric)
        t = pd.DatetimeIndex(traj["time"].values)
        if series_lead is None:
            series_lead = ((t - pd.Timestamp(ev.init)).total_seconds() / 86400.0).tolist()
        for k in keys:
            series[k].append(np.asarray(ser[k].values, dtype=float).tolist())
            for l in leads:
                persistence[k][float(l)].append(
                    float(ser[k].sel(time=expected_valid(ev, l))))
        traj.close()

    realized = {k: np.asarray(v) for k, v in realized.items()}

    # ---- theta from the FCN3 score cubes ------------------------------------ #
    theta = {k: {} for k in keys}
    theta_sd = {k: {} for k in keys}
    absent = []
    for l in leads:
        per = {k: [] for k in keys}
        sd = {k: [] for k in keys}
        for w in have:
            t = S.Task(ev.name, w, float(l))
            if not t.cube_path.exists():
                absent.append(str(t))
                for k in keys:
                    per[k].append(np.nan)
                    sd[k].append(np.nan)
                continue
            with xr.open_dataset(t.cube_path) as cube:
                idx = AI.indices(cube, ev.name, ev.peak, ev.metric,
                                 where=f"score cube {t}")
                for k in keys:
                    v = np.asarray(idx[k].values, dtype=float)
                    per[k].append(float(v.mean()))
                    sd[k].append(float(v.std(ddof=1)) if v.size > 1 else np.nan)
        for k in keys:
            theta[k][float(l)] = np.asarray(per[k])
            theta_sd[k][float(l)] = np.asarray(sd[k])
    if absent:
        print(f"[reduce] ERROR: {len(absent)} score cube(s) missing, e.g. {absent[:3]}\n"
              f"         Run the score stage: sbatch slurm/aires_gate3.slurm",
              file=sys.stderr)
        return 1

    # ---- statistics --------------------------------------------------------- #
    n = len(have)
    rho_crit = _rho_significance(n)
    stats = {k: {} for k in keys}
    for k in keys:
        for l in leads:
            l = float(l)
            th, sd = theta[k][l], theta_sd[k][l]
            r, p = _spearman(th, realized[k])
            lo, hi = _bootstrap_rho(th, realized[k])
            pr, pp = _spearman(persistence[k][l], realized[k])
            var_total = float(np.var(th, ddof=1))
            var_noise = float(np.mean(sd ** 2) / members)
            ceiling = float(np.sqrt(max(var_total - var_noise, 0.0) / var_total)) \
                if var_total > 0 else float("nan")
            stats[k][l] = dict(
                spearman=r, p_value=p, ci95=[lo, hi],
                persistence_spearman=pr, persistence_p=pp,
                theta_mean=float(np.mean(th)), theta_sd_between=float(np.sqrt(var_total)),
                member_sd_within=float(np.mean(sd)),
                theta_se=float(np.sqrt(var_noise)), noise_ceiling=ceiling,
                pearson=float(np.corrcoef(th, realized[k])[0, 1]),
            )

    required = [float(l) for l in A.GATE3["required_leads"]]
    thr = A.GATE3["spearman_min"]
    verdict_lead = {l: stats["box"][l]["spearman"] >= thr for l in required
                    if l in stats["box"]}
    ok = bool(verdict_lead) and all(verdict_lead.values())

    # ---- observed truth, for context --------------------------------------- #
    observed = {}
    tp = F.era5_truth_path(ev)
    if tp.exists():
        with xr.open_dataset(tp) as tds:
            da = tds[list(tds.data_vars)[0]]
            observed = {k: float(AI.area_mean(da, b))
                        for k, b in AI.boxes_for(ev.name).items()}

    # ---- report ------------------------------------------------------------- #
    print(f"\n  realized A_L over {n} free-running walkers ({ev.metric})")
    for k in keys:
        r = realized[k]
        extra = (f"   observed {observed[k]:+.2f}" if k in observed else "")
        print(f"    {k:5s}  mean {r.mean():+7.3f}   sd {r.std(ddof=1):.3f}   "
              f"range [{r.min():+.2f}, {r.max():+.2f}]{extra}")
    if observed:
        for k in keys:
            n_over = int((realized[k] >= observed[k]).sum())
            print(f"    {k:5s}  walkers reaching the observed value: {n_over}/{n}")

    print(f"\n  GATE 3  (box index {box.name}; N = {n} walkers, M = {members} score members)")
    print(f"    {'t_k':>6s}  {'rho_s':>7s}  {'95% CI':>16s}  {'p':>7s}  "
          f"{'ceiling':>7s}  {'persist':>7s}   verdict")
    for l in leads:
        l = float(l)
        s = stats["box"][l]
        mark = ""
        if l in required:
            mark = "  PASS" if s["spearman"] >= thr else "  FAIL"
        print(f"    {l:5.0f}d  {s['spearman']:+7.3f}  "
              f"[{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]".rjust(18) +
              f"  {s['p_value']:7.4f}  {s['noise_ceiling']:7.3f}  "
              f"{s['persistence_spearman']:+7.3f}{mark}")
    print(f"\n    threshold rho_s >= {thr:.2f} at t_k = "
          f"{', '.join(f'{l:g} d' for l in required)}")
    print(f"    at N = {n} walkers, rho_s > {rho_crit:.3f} is the 5% one-sided "
          f"significance level, so the")
    print(f"    threshold is roughly 'significantly better than chance' and no stronger.")
    print(f"\n  GATE 3: {'PASS' if ok else 'FAIL'}")

    print(f"\n  secondary index (CONUS mean, what Gate 2 used)")
    for l in leads:
        s = stats["conus"][float(l)]
        print(f"    {float(l):5.0f}d  rho_s {s['spearman']:+.3f}  "
              f"persistence {s['persistence_spearman']:+.3f}")

    print(f"\n  noise floor (Gate 2's M=6 lesson, applied here)")
    for l in leads:
        s = stats["box"][float(l)]
        print(f"    {float(l):5.0f}d  within-ensemble sd {s['member_sd_within']:.3f} K -> "
              f"s.e.(theta) {s['theta_se']:.3f} K; between-walker sd "
              f"{s['theta_sd_between']:.3f} K")
    print(f"    'ceiling' above is the largest correlation theta could reach with ANY "
          f"target given that noise.")

    # ---- persist ------------------------------------------------------------ #
    res = dict(
        event=ev.name, metric=ev.metric, init=ev.init.isoformat(), peak=ev.peak,
        n_walkers=n, walkers=have, incomplete=missing, m_members=members,
        leads=[float(l) for l in leads], box=dict(name=box.name, lat=box.lat,
                                                  lon=box.lon, note=box.note),
        threshold=thr, required_leads=required, rho_significance_005=rho_crit,
        realized={k: realized[k].tolist() for k in keys},
        observed=observed,
        theta={k: {str(l): theta[k][float(l)].tolist() for l in leads} for k in keys},
        theta_member_sd={k: {str(l): theta_sd[k][float(l)].tolist() for l in leads}
                         for k in keys},
        persistence={k: {str(l): persistence[k][float(l)] for l in leads} for k in keys},
        stats={k: {str(l): stats[k][float(l)] for l in leads} for k in keys},
        index_series={k: series[k] for k in keys}, series_lead_days=series_lead,
        gate3_pass=bool(ok),
        segments=[[int(k), float(l), int(nn)] for (k, l, nn) in segs],
    )
    rp = result_path(ev)
    rp.write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {rp}")

    rows = []
    for i, w in enumerate(have):
        for l in leads:
            l = float(l)
            rows.append(dict(walker=w, lead_days=l,
                             theta_box=theta["box"][l][i], theta_conus=theta["conus"][l][i],
                             theta_member_sd_box=theta_sd["box"][l][i],
                             persistence_box=persistence["box"][l][i],
                             persistence_conus=persistence["conus"][l][i],
                             realized_box=realized["box"][i],
                             realized_conus=realized["conus"][i]))
    csv = members_csv_path(ev)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"  wrote {csv}")

    if plot:
        _plot(ev, res)
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def _plot(ev: F.Event, res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    leads = res["leads"]
    required = res["required_leads"]
    realized = np.array(res["realized"]["box"])
    thr = res["threshold"]
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.6))

    # --- (0,0) & (0,1): theta vs realized at the two required leads ----------- #
    for j in range(len(required), 2):
        ax[0, j].axis("off")
    for j, l in enumerate(required[:2]):
        a = ax[0, j]
        th = np.array(res["theta"]["box"][str(l)])
        se = np.array(res["theta_member_sd"]["box"][str(l)]) / np.sqrt(res["m_members"])
        s = res["stats"]["box"][str(l)]
        a.errorbar(th, realized, xerr=se, fmt="o", ms=6, color="#1f77b4",
                   ecolor="0.7", elinewidth=1, capsize=2, zorder=3)
        obs = res.get("observed", {}).get("box")
        # The observed value is kept inside the axes on purpose: whether the walker
        # ensemble reaches it at all is the question Phase 4 exists to answer, and a
        # panel that crops it out hides the answer.
        ends = [th.min() - se.max(), th.max() + se.max(), realized.min(), realized.max()]
        if obs is not None:
            ends.append(obs)
        lo, hi = min(ends), max(ends)
        pad = 0.08 * (hi - lo)
        a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", lw=1, zorder=1)
        if obs is not None:
            a.axhline(obs, color="#d62728", lw=1.2, ls="--")
            a.text(hi + pad, obs, "ERA5 observed  ", fontsize=8, color="#d62728",
                   va="bottom", ha="right")
        a.set_xlim(lo - pad, hi + pad); a.set_ylim(lo - pad, hi + pad)
        a.set_xlabel(rf"FCN3 score $\theta$ at $t_k$ = {l:g} d   (K)")
        a.set_ylabel(r"walker's realized $A_L$   (K)")
        a.set_title(rf"$t_k$ = {l:g} d:  $\rho_s$ = {s['spearman']:+.3f} "
                    rf"[{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"
                    "\n" rf"persistence $\rho_s$ = {s['persistence_spearman']:+.3f}, "
                    rf"noise ceiling {s['noise_ceiling']:.2f}")
        a.grid(alpha=.3)

    # --- (1,0): rho vs lead --------------------------------------------------- #
    a = ax[1, 0]
    rho = [res["stats"]["box"][str(l)]["spearman"] for l in leads]
    per = [res["stats"]["box"][str(l)]["persistence_spearman"] for l in leads]
    ceil = [res["stats"]["box"][str(l)]["noise_ceiling"] for l in leads]
    lo95 = [res["stats"]["box"][str(l)]["ci95"][0] for l in leads]
    hi95 = [res["stats"]["box"][str(l)]["ci95"][1] for l in leads]
    a.fill_between(leads, lo95, hi95, color="#1f77b4", alpha=.15, lw=0,
                   label="95% CI (bootstrap over walkers)")
    a.plot(leads, ceil, color="0.55", lw=1.2, ls=":",
           label=rf"noise ceiling at M={res['m_members']}")
    a.plot(leads, rho, "o-", color="#1f77b4", lw=2, label=r"FCN3 score $\theta$")
    a.plot(leads, per, "s--", color="#ff7f0e", lw=1.8,
           label="persistence (Standard-RES)")
    a.axhline(thr, color="#2ca02c", lw=1.4, ls="--")
    a.text(leads[0], thr, f"  gate: {thr:.2f}", color="#2ca02c", fontsize=9, va="bottom")
    a.axhline(res["rho_significance_005"], color="0.6", lw=1, ls="-.")
    a.text(leads[-1], res["rho_significance_005"],
           f"5% signif. (N={res['n_walkers']})  ", color="0.4", fontsize=8,
           va="bottom", ha="right")
    a.axhline(0, color="0.85", lw=1)
    for l in required:
        a.axvline(l, color="0.9", lw=6, zorder=0)
    a.set_xlabel(r"resampling time $t_k$ (days lead)")
    a.set_ylabel(r"Spearman $\rho_s$ vs realized $A_L$")
    a.set_ylim(-1.05, 1.05)
    a.set_title("score skill vs lead\n(shaded columns = the leads the gate is taken at)")
    a.legend(fontsize=8, loc="lower left")
    a.grid(alpha=.3)

    # --- (1,1): walker spaghetti, coloured by outcome ------------------------- #
    a = ax[1, 1]
    lead_d = np.array(res["series_lead_days"])
    ser = np.array(res["index_series"]["box"])
    norm = colors.Normalize(vmin=realized.min(), vmax=realized.max())
    cmap = plt.get_cmap("coolwarm")
    for i in range(ser.shape[0]):
        a.plot(lead_d, ser[i], lw=1.1, color=cmap(norm(realized[i])), alpha=.85)
    a.axvspan(res["segments"][-1][1] - 6, res["segments"][-1][1], color="0.9", zorder=0)
    a.annotate(r"$A_L$ window", xy=(res["segments"][-1][1] - 3, 0.97),
               xycoords=("data", "axes fraction"), fontsize=8, color="0.35",
               ha="center", va="top")
    for l in leads:
        a.axvline(l, color="0.75", lw=0.8, ls=":")
    if res.get("observed", {}).get("box") is not None:
        a.axhline(res["observed"]["box"], color="#d62728", lw=1.2, ls="--")
        a.text(0, res["observed"]["box"], " ERA5 observed 7-day mean", fontsize=8,
               color="#d62728", va="bottom")
    a.set_xlabel("lead (days from init)")
    a.set_ylabel(rf"instantaneous {res['box']['name']} index   (K)")
    a.set_title(f"the {res['n_walkers']} free-running GenCast walkers\n"
                r"(coloured by their realized $A_L$; dotted lines = $t_k$)")
    a.grid(alpha=.3)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=a,
                 label=r"realized $A_L$ (K)")

    verdict = "PASS" if res["gate3_pass"] else "FAIL"
    fig.suptitle(f"Gate 3 - score skill, {ev.name} (init {res['init'][:10]}, "
                 f"box {res['box']['name']}, N={res['n_walkers']}, "
                 f"M={res['m_members']}): {verdict}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = A.fig_dir() / f"gate3_{ev.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["plan", "walk", "reduce"])
    ap.add_argument("--event", default=os.environ.get("AIRES_EVENT", DEFAULT_EVENT))
    ap.add_argument("--walkers", type=int, default=A.GATE3_N_WALKERS)
    ap.add_argument("--members", type=int, default=A.GATE3_M_MEMBERS)
    ap.add_argument("--leads", default=",".join(str(l) for l in A.GATE3_LEAD_DAYS))
    ap.add_argument("--shard", type=int, default=int(os.environ.get("AIRES_SHARD", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("AIRES_NSHARDS", 8)))
    ap.add_argument("--prune-stale", action="store_true",
                    help="walk: delete cached segments whose valid time is off-schedule")
    ap.add_argument("--allow-partial", action="store_true",
                    help="reduce: score only the walkers whose trajectories are complete")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    ev = _event(a.event)
    leads = [float(x) for x in a.leads.split(",") if x.strip()]

    if a.stage == "plan":
        return stage_plan(ev, a.walkers, a.members, leads, a.nshards)
    if a.stage == "walk":
        return stage_walk(ev, a.walkers, a.shard, a.nshards,
                          prune_stale_states=a.prune_stale)
    return stage_reduce(ev, a.walkers, a.members, leads, plot=not a.no_plot,
                        allow_partial=a.allow_partial)


if __name__ == "__main__":
    sys.exit(main())
