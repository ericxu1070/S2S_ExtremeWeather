#!/usr/bin/env python
"""The stages that produce the sweep, in the order they run.

    plan       login node : the schedule, the budget, the prerequisites
    check      login node : build every segment SHAPE, incl. both ragged tails. No GPU.
    refs       login node : the ERA5 reference series and truth fields (astab.refs)
    calibrate  login node : measure the diagnostic bands on the 16 Gate 3 walkers per
                            event that are ALREADY on disk. CPU, free, and the single
                            highest-value step here - it turns guessed thresholds into
                            measured distributions out to 21 d before any GPU is booked.
    walk       GPU, moe   : roll the chains, one per GPU
    score      GPU, fcn3  : --arm free (ERA5 IC -> peak) and --arm fed (walker lead-3 d
                            state -> peak, through the adapter)
    prune      login node : release the pinned states and the retained global Zarrs

``reduce``/``figs``/``maps`` are in ``astab.reduce``, ``astab.plots`` and ``astab.maps``;
the CLI that drives all of them is ``astab.run``.

Env: ``plan``/``check``/``calibrate``/``prune`` and ``walk`` are ``moe``; ``score`` is
``fcn3``. Both halves import this module, so nothing at module level may touch JAX or
torch - every such import is inside the stage that needs it.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from aires import aindex as AI
from aires import score as S
from astab import sconfig as SC
from astab.diag import (BAND_WIDEN, band_from_samples, conus_series, diurnal_ratio_series,
                        zonal_high_wavenumber_fraction, _phase_matched_first)
from astab.record import ensure_record, jsonable
from astab.refs import ref_path, reference_at
from astab.schedule import Chain, prunable, scoring_checkpoints, shard_of, tail_steps
from fcn3 import fevents as F
from gencast_s2s import config as C

REPO = SC.REPO

# What one GenCast step and one shard's startup actually cost, MEASURED on job 1189's
# week-8 chain (logs/Vayuh-s2s-1189-walk-try1-shard2.log): 28.6 s per step with 8
# concurrent serial bf16 workers on one node, plus ~26 s of per-segment overhead. The 52 s
# used by the frozen sweep's budget line is `run_aires.SECONDS_PER_GENCAST_STEP`, the
# conservative ceiling; the ensemble block prints both because a 100-member run is priced
# in GPU-days and the factor of 1.8 between them decides how many nodes to ask for.
ENS_SECONDS_PER_STEP = 28.6
ENS_SECONDS_PER_STEP_CEILING = 52.0
# ~13 min: 17 s to read the 4.18 GB checkpoint off warm page cache, ~5 min of XLA compile,
# and the first segment. Paid ONCE per worker process, and the JAX compile cache at
# runs/models/jax_cache_0p25 is 928 kB - effectively empty - so every worker compiles.
ENS_SHARD_STARTUP_H = 13 / 60.0
# A slimmed (T2m-only) 6-step diag.nc, and a segment record carrying the per-level ranges.
# The full 12-variable diag is ~36 MB, i.e. 65 GB over a 100-member week-8 run.
ENS_SLIM_DIAG_MB = 0.6
ENS_RECORD_KB = 8.0
ENS_STATE_GB = 0.477


# stage: plan  (login node, either env)
# --------------------------------------------------------------------------- #
def stage_plan(chs: list[Chain], nshards: int) -> int:
    walk_wk = sorted({c.weeks for c in chs if c.walks_to_peak})
    ext_wk = sorted({c.weeks for c in chs if not c.walks_to_peak})
    ens = [c for c in chs if c.member is not None]
    # 100 identical table rows and 200 identical prerequisite lines are not a plan, they
    # are a wall. The per-chain table stays for the frozen sweep and for a small re-walk;
    # past that the ensemble block below carries the same information as totals.
    per_chain_rows = not ens or len(ens) <= 12
    print(f"AI+RES lead-time stability sweep  (tag {SC.STAB_TAG!r})")
    print(f"  {len(chs)} chain(s): {len(set(c.event for c in chs))} event(s) x "
          f"{len(set(c.weeks for c in chs))} lead(s)"
          + (f" x {len({c.member for c in ens})} member(s)" if ens else "")
          + f", {SC.N_WALKER_MEMBERS} walker member, {SC.N_FCN3_MEMBERS} FCN3 member each")
    print(f"  walker to the peak at week(s) {walk_wk or '-'}; "
          f"FCN3-only extension at week(s) {ext_wk or '-'} "
          f"(walker rolls {SC.SCORE_LEAD_DAYS:g} d there, to launch the adapter-fed arm)")
    if ens:
        print(f"  ENSEMBLE MODE: GenCast only. No FCN3 arm is planned, priced or checked "
              f"for a member chain -\n"
              f"  its score/prune/maps stages are refused and its FCN3 path helpers "
              f"raise.")
    print()

    print(f"  {'chain':>28s}  {'init':>10s} {'peak':>10s}  {'walk':>5s} {'segs':>4s} "
          f"{'steps':>5s} {'tail':>4s}  {'fcn3':>5s}  cached")
    gsteps = fsteps = 0
    todo_steps: list[int] = []
    done_chains = 0
    ready = True
    for ch in chs:
        segs = ch.schedule
        tail = tail_steps(ch.walk_days)
        done = sum(1 for (k, _l, _n) in segs if ch.state_path(k).exists()
                   or ch.record_path(k).exists())
        todo = [(k, l, n) for (k, l, n) in segs if not ch.record_path(k).exists()]
        gsteps += sum(n for _k, _l, n in todo)
        todo_steps.append(sum(n for _k, _l, n in todo))
        done_chains += 1 if not todo else 0
        # adapter-free: init -> peak; adapter-fed: lead 3 d -> peak. BOTH are the full
        # horizon whatever the walker does, which is the whole point of an extension lead.
        # A MEMBER chain has neither: those cubes carry no member in their path, so asking
        # for them would price - and later overwrite - the frozen sweep's artifacts.
        n_free = ch.horizon_days * 24 // F.STEP_H
        if ch.member is None:
            if not ch.fcn3_free_cube_path().exists():
                fsteps += n_free
            if not ch.score_cube_path().exists():
                fsteps += int((ch.horizon_days - SC.SCORE_LEAD_DAYS) * 24 // F.STEP_H)
        if per_chain_rows:
            fcol = "-" if ch.member is not None else str(n_free)
            print(f"  {ch.label:>28s}  {ch.init.date()} {ch.peak.date()}  "
                  f"{str(ch.walk_days) + 'd':>5s} {len(segs):4d} "
                  f"{sum(n for _k, _l, n in segs):5d} {tail:4d}  {fcol:>5s}  "
                  f"{done}/{len(segs)}")
    if not per_chain_rows:
        print(f"  {'(' + str(len(ens)) + ' member chains, rows suppressed)':>28s}  "
              f"{chs[0].init.date()} {chs[0].peak.date()}  "
              f"{str(chs[0].walk_days) + 'd':>5s} {len(chs[0].schedule):4d} "
              f"{sum(n for _k, _l, n in chs[0].schedule):5d} "
              f"{tail_steps(chs[0].walk_days):4d}      -  "
              f"{done_chains}/{len(chs)} member(s) complete")

    print(f"\n  prerequisites")
    seen_prereq: set = set()
    for ch in chs:
        wants = [("walker init frames", ch.inputs_path())]
        if ch.member is None:
            wants.append(("FCN3 ERA5 IC     ", ch.fcn3_ic_path()))
        for what, p in wants:
            ok = p.exists()
            ready &= ok
            # Every member of a week shares that week's init frames - one deterministic
            # IC is what makes this a diffusion ensemble - so the line is printed once.
            if (what, p) in seen_prereq:
                continue
            seen_prereq.add((what, p))
            print(f"    {ch.label}  {what}  {'OK ' if ok else 'MISSING'}  {p}")
    for what, p in (("derived calibration", A.CALIB_PATH),
                    ("measured bands     ", SC.bands_path())):
        ok = p.exists()
        print(f"    {'':>28s}  {what}  {'OK ' if ok else 'MISSING'}  {p}")
        ready &= ok
    for ev in sorted({c.event for c in chs}):
        p = ref_path(ev)
        ok = p.exists()
        ready &= ok
        print(f"    {ev:>28s}  ERA5 reference     {'OK ' if ok else 'MISSING'}  {p}")
        anchor = _xres_cube_path(ev, 4)
        print(f"    {ev:>28s}  week-4 anchor cube {'OK ' if anchor.exists() else 'absent '}"
              f"  {anchor}   (the same-init physics gate)")

    clim = C.CLIM_FILE
    if clim.exists():
        with xr.open_dataset(clim) as cd:
            sizes = dict(cd.sizes)
        good = sizes.get("lat") == 105 and sizes.get("lon") == 237
        print(f"    {'':>28s}  climatology grid   {'OK ' if good else 'WRONG  '}  "
              f"{sizes}   (expect lat=105 lon=237; CLAUDE.md's stale-2deg trap)")
        ready &= good

    walk_h = gsteps * 52.0 / 3600.0
    score_h = fsteps * 1.40 / 3600.0
    print(f"\n  budget")
    print(f"    walker      {gsteps:6,d} GenCast steps  ~{walk_h:5.1f} GPU-h  "
          f"(52 s/step, run_aires.SECONDS_PER_GENCAST_STEP)")
    print(f"    scorer      {fsteps:6,d} FCN3 steps     ~{score_h:5.1f} GPU-h  "
          f"(1.40 s/step, job 1161)")
    # The longest chain STILL TO ROLL, not the longest in the sweep: a resubmit that skips
    # every cached chain does not wait on the week-10 walk it is not redoing.
    longest = max(todo_steps, default=0)
    longest_f = max((c.horizon_days * 24 // F.STEP_H for c in chs
                     if c.member is None and not c.fcn3_free_cube_path().exists()),
                    default=0)
    print(f"    wall clock  walk: the longest chain left to roll, {longest} steps at "
          f"28-52 s = {longest * 28 / 3600:.1f}-{longest * 52 / 3600:.1f} h, plus ~15 min "
          f"model load and 1-2 XLA compiles")
    print(f"                score: the longest rollout, {longest_f} FCN3 steps at 1.40 s "
          f"= {longest_f * 1.40 / 3600:.1f} h, plus a ~9 min model load per pool")
    # The Zarr estimate is per FCN3 STEP (~5.8 MB of global z500/t50/t2m/wind at 0.25 deg,
    # measured over job 1189's 3,040 steps), never per chain: a week-20 rollout is twice a
    # week-10 one and a per-chain figure would understate the extension by half.
    # 3 states per full chain (2 rolling + the pinned lead-3 d one); an FCN3-extension
    # chain has ONE segment and therefore one state.
    n_states = sum(3 if c.walks_to_peak else 1 for c in (c for c in chs
                                                         if c.member is None))
    print(f"    disk        states ~{n_states * ENS_STATE_GB:.1f} GB retained "
          f"({n_states} state(s)), diag ~{gsteps / 6 * 0.036:.1f} GB, "
          f"FCN3 Zarrs ~{fsteps * 0.0058:.1f} GB until --stage prune")
    if ens:
        print(f"                ^ these are the SWEEP's rules (states retained, diag "
              f"unslimmed). A member chain releases its states and slims its diag - "
              f"see the ensemble block below for what this run actually costs.")
    _print_free_space()

    if ens:
        _print_ensemble_budget(ens, chs, todo_steps, nshards)

    print(f"\n  shards      {len(chs)} chain(s) over {nshards} GPU(s): "
          f"{[len(shard_of(chs, nshards, s)) for s in range(nshards)]}")
    print(f"\n  READY: {'yes' if ready else 'NO - fix the items marked above'}")
    return 0 if ready else 1


def _print_ensemble_budget(ens: list[Chain], chs: list[Chain], todo_steps: list[int],
                           nshards: int) -> None:
    """What a 100-member walker ensemble at one lead actually costs, in GPU-h and in GB.

    Printed instead of the per-chain table because the answer a reader needs before
    submitting is not "what does member 57 do", it is "how many nodes, how long, and does
    it fit on the filesystem". Every constant here is measured (job 1189) and named at the
    top of this module.

    The disk line is the one to read twice. Unslimmed the diagnostics would be 65 GB and
    unreleased the states would be 906 GB, against 267 GB free on a 98%-full shared NFS:
    the slim and the release are not tidiness, they are what makes the run possible.
    """
    per_ev = sorted({(c.event, c.weeks) for c in ens})
    members = sorted({c.member for c in ens})
    seg0 = ens[0].schedule
    steps_each = sum(n for _k, _l, n in seg0)
    todo = sum(todo_steps)
    # Wall clock is the WORST shard, not the mean: the job ends when the last one does.
    per_shard = [sum(t for i, t in enumerate(todo_steps) if i % nshards == s)
                 for s in range(nshards)]
    worst = max(per_shard, default=0)
    n_seg_total = sum(len(c.schedule) for c in ens)

    print(f"\n  ensemble")
    for ev, wk in per_ev:
        ch0 = next(c for c in ens if c.event == ev and c.weeks == wk)
        print(f"    {ev} @ week {wk} ({ch0.horizon_days} d): init {ch0.init.date()} "
              f"-> peak {ch0.peak.date()}")
    print(f"    members     {len(members)}  (m{min(members):03d}..m{max(members):03d}); "
          f"seed = the member index, so each is a distinct GenCast diffusion draw from "
          f"ONE deterministic ERA5 init")
    print(f"    per member  {len(seg0)} segment(s) x {A.RES_SEG_STEPS} steps "
          f"(tail {tail_steps(ens[0].walk_days)}) = {steps_each} GenCast steps to "
          f"{ens[0].walk_days} d")
    busy = sum(1 for c, t in zip(chs, todo_steps) if c.member is not None and t)
    print(f"    to roll     {todo:,d} step(s) of {steps_each * len(ens):,d} "
          f"({busy} member-chain(s) still have work)")
    print(f"    GPU-h       ~{todo * ENS_SECONDS_PER_STEP / 3600:.0f} at "
          f"{ENS_SECONDS_PER_STEP:g} s/step (measured, job 1189, 8 concurrent serial bf16 "
          f"workers)")
    print(f"                ~{todo * ENS_SECONDS_PER_STEP_CEILING / 3600:.0f} at "
          f"{ENS_SECONDS_PER_STEP_CEILING:g} s/step (the conservative ceiling)")
    print(f"    wall        the worst shard: {worst:,d} step(s) = "
          f"{worst * ENS_SECONDS_PER_STEP / 3600:.1f}-"
          f"{worst * ENS_SECONDS_PER_STEP_CEILING / 3600:.1f} h, plus "
          f"~{ENS_SHARD_STARTUP_H * 60:.0f} min of startup per worker")
    print(f"    compile     the JAX cache {A.RUNS / 'models' / 'jax_cache_0p25'} is "
          f"effectively empty - budget one ~5 min XLA compile per worker, not per job")
    print(f"    disk        slim diag {SC.ENS_DIAG_VARS or '(SLIM DISABLED)'} "
          f"~{n_seg_total * ENS_SLIM_DIAG_MB / 1e3:.1f} GB over {n_seg_total:,d} "
          f"segment(s); records ~{n_seg_total * ENS_RECORD_KB / 1e6:.2f} GB")
    print(f"                states are TRANSIENT: <= 2 x {ENS_STATE_GB:.3f} GB per live "
          f"worker = ~{2 * ENS_STATE_GB * nshards:.0f} GB across {nshards} shard(s), "
          f"released at each member's chain end")
    keep = sorted(m for m in SC.ENS_KEEP_STATE_MEMBERS if m in set(members))
    print(f"                kept afterwards: final state of member(s) "
          f"{keep or 'none'} = {len(keep) * ENS_STATE_GB:.2f} GB "
          f"(AIRES_ENS_KEEP_STATE_MEMBERS)")
    if not SC.ENS_DIAG_VARS:
        print(f"                WARNING: the slim is DISABLED (AIRES_ENS_DIAG_VARS is "
              f"empty). Full diagnostics are ~{n_seg_total * 36 / 1e3:.0f} GB.")
    print(f"    shards      {[len(shard_of(chs, nshards, s)) for s in range(nshards)]} "
          f"chain(s) each; step load {min(per_shard, default=0):,d}..{worst:,d}")


def _print_free_space() -> None:
    try:
        st = os.statvfs(SC.RUNS)
        print(f"    free        {st.f_bavail * st.f_frsize / 1e9:.0f} GB on "
              f"{SC.RUNS}")
    except OSError:
        pass


def _xres_cube_path(event: str, weeks: int) -> Path:
    """``F.gencast_cube_path`` pinned to an explicit week.

    ``fcn3.fevents`` resolves ``WEEKS`` from ``FCN3_WEEKS`` at MODULE IMPORT, and this
    process spans every lead in the sweep at once, so the week has to be passed rather
    than inherited.
    """
    return F.gencast_cube_path(F.event(event), weeks=weeks)


# --------------------------------------------------------------------------- #
# stage: check  (login node, moe, no GPU)
# --------------------------------------------------------------------------- #
def stage_check(chs: list[Chain], *, isolate: bool | None = None) -> int:
    """Build every distinct segment SHAPE in the sweep and verify the batch structure.

    ``n_steps`` of 2 and 4 - the two ragged tails - have never been built against the real
    0.25 deg checkpoint, only against synthetic task specs. Discovering a problem with
    them here costs a minute on the login node; discovering it in the walk stage costs an
    8-GPU allocation and a five-minute XLA compile first.

    **Isolated per chain by default**, the same way ``run_xres.py``'s prep stage is and for
    the same reason: this login node has 15 GB of RAM, a global 2-frame state is 0.71 GB,
    and a 6-step batch materialises another ~0.5 GB of NaN targets plus the concat's
    copies. Eight chains in one process reach the OOM killer (measured, this run). One
    subprocess per chain hands every byte back between chains. ``AIRES_CHECK_ISOLATE=0``
    turns it off.
    """
    import gc

    from aires import walker as W

    if isolate is None:
        isolate = os.environ.get("AIRES_CHECK_ISOLATE", "1") != "0"
    if isolate and len(chs) > 1:
        rc = 0
        for ch in chs:
            r = subprocess.run(
                [sys.executable, "-m", "astab.run", "--stage", "check",
                 "--events", ch.event, "--weeks", str(ch.weeks)],
                cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO),
                                    "AIRES_CHECK_ISOLATE": "0"})
            rc |= 1 if r.returncode else 0
        print(f"[check] {'OK' if rc == 0 else 'FAILED'} over {len(chs)} chain(s)")
        return rc

    # The checkpoint's task config, loaded once. `_task_config_only` is walker.py's own
    # no-GPU path (it is what `python -m aires.walker --check` uses) and reading it here
    # rather than re-deriving it keeps this in step with whatever the loader does.
    cfg = W._task_config_only()
    rc = 0
    for ch in chs:
        p = ch.inputs_path()
        if not p.exists():
            print(f"[check] {ch.label}: MISSING init frames {p}", file=sys.stderr)
            rc = 1
            continue
        state = W.read_state(p)
        got = W.valid_time(state)
        print(f"[check] {ch.label}: state valid {got}, "
              f"{state.sizes['lat']}x{state.sizes['lon']} global, "
              f"{len(A.STATE_PROGNOSTIC)} prognostic + {len(A.STATE_STATIC)} static")
        if pd.Timestamp(got) != ch.init:
            print(f"[check] {ch.label}: ERROR init frames are valid {got}, but a "
                  f"{ch.horizon_days} d chain must start at {ch.init}", file=sys.stderr)
            rc = 1
        for n in sorted({n for _k, _l, n in ch.schedule}):
            arrays = W.eval_arrays({"task_config": cfg}, state, n)
            print(f"[check] {ch.label}: a {n}-step segment builds cleanly "
                  f"({'tail' if n != A.RES_SEG_STEPS else 'standard'})")
            del arrays
            gc.collect()
        state.close()
        del state
        gc.collect()
        segs = ch.schedule
        assert sum(n for _k, _l, n in segs) * A.WALKER_STEP_H / 24 == ch.walk_days
        print(f"[check] {ch.label}: {len(segs)} segment(s) tile the {ch.walk_days} d WALK "
              f"exactly, landing on {ch.walk_end}"
              + ("" if ch.walks_to_peak else
                 f"   (FCN3-extension lead: the walker stops there; both FCN3 arms still "
                 f"roll the full {ch.horizon_days} d to {ch.peak})"))
    print(f"[check] {'OK' if rc == 0 else 'FAILED'}")
    return rc


# --------------------------------------------------------------------------- #
# stage: calibrate  (login node, CPU, free)
#
# CLAUDE.md's hard-won rule is: measure the noise floor before setting a threshold. The
# Gate 3 tree already holds 16 free-running, physically-validated 21-day walkers PER
# EVENT, with a global state and a CONUS diagnostics cube every 3 d. Running the whole
# diagnostic panel over those known-good trajectories converts items 3, 5, 6 and 7 from
# guessed thresholds into measured distributions out to 21 d - using data that already
# exists, before any GPU is booked. Beyond 21 d the bands are an extrapolation and every
# consumer says so.
# --------------------------------------------------------------------------- #
CALIB_METRICS = ("t2m_anom_conus", "t2m_conus_drift", "diurnal_ratio", "conus_t2m_sd",
                 "z500_hi_wavenumber_frac")


def _gate3_samples(event: str, walkers: int, *, verbose: bool = True) -> pd.DataFrame:
    """The panel, evaluated on every Gate 3 segment of ``event`` that is on disk."""
    segs = A.gate3_segments()
    rows = []
    for w in range(walkers):
        first_mean = None
        for (k, lead, _n) in segs:
            dp, sp = A.diag_path(event, w, k), A.state_path(event, w, k)
            if not dp.exists():
                continue
            with xr.open_dataset(dp) as d:
                d = d.load()
            ser = conus_series(d)
            dur = diurnal_ratio_series(d)
            if first_mean is None:
                first_mean = _phase_matched_first(ser.index, ser["t2m_mean"].values,
                                                  hour=int(pd.Timestamp(
                                                      ser.index[-1]).hour))
            row = dict(event=event, walker=w, step=k, lead_days=float(lead),
                       t2m_anom_conus=float(ser["t2m_anom"].iloc[-1]),
                       t2m_conus_drift=float(ser["t2m_mean"].iloc[-1] - first_mean),
                       conus_t2m_sd=float(ser["t2m_sd"].iloc[-1]),
                       diurnal_ratio=(float(np.nanmean(dur["ratio"].values))
                                      if not dur.empty else np.nan))
            del d
            if sp.exists():
                # Only the one level is materialised: 16 walkers x 7 segments x 477 MB is
                # 53 GB per event, and the spectrum needs z500 alone.
                with xr.open_dataset(sp) as s:
                    z = s["geopotential"].sel(level=500).isel(time=-1).load()
                    row["global_z500"] = float(AI.area_mean(z))
                    row["global_t2m"] = float(AI.area_mean(
                        s["2m_temperature"].isel(time=-1).load()))
                row["z500_hi_wavenumber_frac"] = zonal_high_wavenumber_fraction(z)
                del z
            rows.append(row)
        if verbose:
            print(f"  [calibrate] {event} w{w:02d}: {len([r for r in rows if r['walker'] == w])} "
                  f"segment(s)", flush=True)
    return pd.DataFrame(rows)


def stage_calibrate(events, walkers: int, *, force: bool = False) -> int:
    out = SC.bands_path()
    samples_csv = SC.gate3_panel_path()
    if samples_csv.exists() and not force:
        df = pd.read_csv(samples_csv)
        df = df[df["event"].isin(list(events))]
        print(f"[calibrate] reusing {samples_csv} ({len(df)} rows); "
              f"--force to re-measure")
    else:
        parts = []
        for ev in events:
            print(f"[calibrate] {ev}: scanning {walkers} Gate 3 walkers", flush=True)
            parts.append(_gate3_samples(ev, walkers))
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if df.empty:
            print("[calibrate] ERROR: no Gate 3 segments found on disk. This stage "
                  "measures the bands from walkers that ALREADY exist "
                  f"({A.event_dir(list(events)[0]) / 'walkers'}).", file=sys.stderr)
            return 1
        samples_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(samples_csv, index=False)
        print(f"[calibrate] wrote {samples_csv} ({len(df)} rows)")

    max_lead = float(df["lead_days"].max())
    per_event, pooled = {}, {}
    for m in CALIB_METRICS:
        if m not in df:
            continue
        pooled[m] = band_from_samples(m, df[m].values, max_lead).as_dict()
        for ev, sub in df.groupby("event"):
            per_event.setdefault(str(ev), {})[m] = band_from_samples(
                m, sub[m].values, max_lead).as_dict()

    payload = dict(
        generated=str(pd.Timestamp.utcnow()),
        n_samples=int(len(df)), walkers=int(walkers),
        max_lead_days=max_lead, band_widen=BAND_WIDEN,
        source="Gate 3 free-running walkers already on disk "
               "(runs/aires/<event>/walkers/w*/step*)",
        caveat=f"Measured to {max_lead:g} d only. Beyond that every band is an "
               f"EXTRAPOLATION and is labelled as such wherever it is drawn.",
        pooled=pooled, events=per_event)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=jsonable))
    os.replace(tmp, out)

    print(f"\n  measured bands ({len(df)} samples from {df['walker'].nunique()} walkers "
          f"x {df['event'].nunique()} event(s), out to {max_lead:g} d)")
    print(f"    {'metric':>26s}  {'p05':>10s} {'p50':>10s} {'p95':>10s}   band")
    for ev in sorted(per_event):
        print(f"    {ev}")
        for m, b in per_event[ev].items():
            print(f"    {m:>26s}  {b['p05']:10.3f} {b['p50']:10.3f} {b['p95']:10.3f}   "
                  f"[{b['lo']:.3f}, {b['hi']:.3f}]  n={b['n']}")
    print(f"\n  wrote {out}")
    print(f"  NOTE: {payload['caveat']}")
    return 0


# --------------------------------------------------------------------------- #
# stage: walk  (GPU, moe)
# --------------------------------------------------------------------------- #
def _segment_status(chain: Chain, step: int, lead_days: float) -> tuple[str, str]:
    """``('ok'|'missing'|'stale'|'partial', detail)``, in ``gate3.segment_status``'s shape.

    ``walker.run_segment`` skips any segment whose files exist and does NOT re-verify the
    valid time - which is how a state written under a different schedule gets silently
    adopted. That matters more here than anywhere else in the repo: if
    ``chain_schedule()`` is ever edited between a failed run and a resubmit, a wrong-length
    tail already on disk would be inherited without a word.
    """
    from aires import walker as W

    sp, dp = chain.state_path(step), chain.diag_path(step)
    if not sp.exists():
        return ("ok", "state pruned, record present") if chain.record_path(step).exists() \
            else ("missing", "no state.nc")
    want = chain.valid_at(lead_days)
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


def check_physics(chain: Chain, step: int, lead_days: float) -> None:
    """Fail three minutes into a chain, not two hours into it.

    Two forms, because the two halves of the sweep have different references:

    * **week-4 chains** get ``gate3.check_against_reference``'s comparison as-is, pointed
      at the tag's tree: a 24-member 0.25 deg week-4 cube exists for BOTH events, same
      init and same checkpoint, and that is the strongest anchor available anywhere.
    * **every longer chain** has no same-init ensemble, and reusing the week-3 cube would
      FALSE-FAIL - a walker 21+ days diverged legitimately sits many sd from a fresh
      week-3 ensemble mean. They are gated against ERA5 at the day-3 valid time instead:
      at 3 d GenCast is genuinely skillful, so CONUS-mean T2m within a few K of ERA5 is a
      sound statement, and the 26 K wrong-checkpoint drift fails it instantly. This is the
      ONLY gate an FCN3-extension chain gets, and it is enough - its walk IS the day-3
      segment, so the gate covers the whole of what the walker produced there.

    Every chain draws its bundle from the single ``walker.load_bundle`` call site, so one
    passing week-4 check already covers the wrong-checkpoint class for the whole job; the
    ERA5 gate is the per-chain backstop.
    """
    dp = chain.diag_path(step)
    if not dp.exists():
        return
    with xr.open_dataset(dp) as d:
        t = pd.DatetimeIndex(d["time"].values)[-1]
        got = float(AI.area_mean(d["2m_temperature"].isel(batch=0, time=-1)))

    ref = _xres_cube_path(chain.event, chain.weeks)
    if ref.exists():
        from aires import gate3 as G3
        with xr.open_dataset(ref) as x:
            xt = list(pd.DatetimeIndex(x["time"].values))
            if t in set(xt):
                ens = AI.area_mean(x["2m_temperature"].isel(time=xt.index(t))).values
                mu, sd = float(ens.mean()), float(ens.std(ddof=1))
                tol = max(G3.REFERENCE_TOL_K, G3.REFERENCE_TOL_SD * sd)
                off = abs(got - mu)
                print(f"[walk] physics {chain.label} step{step} @ {t}: CONUS-mean T2m "
                      f"{got:.2f} K vs the same-init xres ensemble {mu:.2f} +- {sd:.3f} K "
                      f"over {ens.size} members ({off / sd if sd else float('nan'):.1f} sd)")
                if off > tol:
                    raise RuntimeError(
                        f"{chain.label} step{step}: CONUS-mean T2m is {got:.2f} K at {t}, "
                        f"but the cached xres ensemble from the SAME checkpoint and init "
                        f"gives {mu:.2f} +- {sd:.3f} K (tolerance {tol:.2f} K). The first "
                        f"thing to check is which checkpoint was loaded: "
                        f"{A.walker_model_kwargs()['params_file']!r} was requested.")
                return

    era5 = reference_at(chain.event, pd.Timestamp(t))
    if not era5 or not np.isfinite(era5.get("t2m_conus_mean", np.nan)):
        print(f"[walk] {chain.label} step{step}: no same-init cube and no ERA5 reference "
              f"at {t}; the physics gate could not run. Build it with "
              f"`--stage refs`.", file=sys.stderr)
        return
    obs = era5["t2m_conus_mean"]
    off = abs(got - obs)
    kind = ("skill-bearing" if lead_days <= SC.ERA5_GATE_SKILLFUL_DAYS
            else "blow-up detector only - ERA5 agreement is NOT expected at this lead")
    print(f"[walk] physics {chain.label} step{step} @ {t}: CONUS-mean T2m {got:.2f} K vs "
          f"ERA5 {obs:.2f} K ({got - obs:+.2f} K; gate {SC.ERA5_GATE_TOL_K:.0f} K at "
          f"{lead_days:g} d lead, {kind})")
    if off > SC.ERA5_GATE_TOL_K:
        raise RuntimeError(
            f"{chain.label} step{step}: CONUS-mean T2m is {got:.2f} K at {t}, but ERA5 "
            f"gives {obs:.2f} K - {off:.2f} K away, past the {SC.ERA5_GATE_TOL_K:.0f} K gate. "
            f"At {lead_days:g} d lead GenCast is genuinely skillful, so this is not a "
            f"predictability limit. Check which checkpoint was loaded: "
            f"{A.walker_model_kwargs()['params_file']!r} was requested.")


def stage_walk(chs: list[Chain], shard: int, nshards: int, *,
               prune_stale_states: bool, keep: int | None = None,
               max_segments: int | None = None) -> int:
    """Roll this shard's chains to their walk end, cache-aware, one segment at a time.

    The walk end is the PEAK at a ``STAB_WALK_WEEKS`` lead and the 3 d launch state at an
    FCN3-extension lead - ``Chain.walk_days``, never ``horizon_days``. The landing
    assertion is the same either way and is the only thing in the package that checks the
    schedule tiled what it claimed to.

    ``max_segments`` truncates every chain, for the smoke path only: two segments of the
    week-4 PNW chain costs one GPU and ~10 min and exercises the whole loop - the loader,
    the compile, the physics gate, the record, the prune - before an 8-GPU allocation is
    booked on it. A truncated chain does NOT land on its walk end, so the landing assertion
    is skipped and the fact is printed rather than left to be inferred from a missing
    verdict.

    **ONE member per worker, always.** ``W.lazy_bundle(SC.N_WALKER_MEMBERS)`` is
    ``n_members = 1`` and must stay that way even when this shard holds four member chains
    of the ensemble follow-up: the 0.25 deg checkpoint occupies ~64 GB of an 80 GB H100
    for a SINGLE member (measured, a3mega job 1154), so ``load_bundle(n_members=4)`` does
    not fit and there is no batched path to fall back on. An ensemble here is serial
    chains, one after another on the same card, sharing the one loaded bundle - which is
    also why the bundle is lazy and loaded once per process rather than once per chain.

    **Member chains fail one at a time.** A ``check_physics`` RuntimeError or a bad
    segment kills the chain it happened on and the shard continues to its next member;
    the shard exits non-zero at the end with the list. The frozen sweep keeps fail-fast
    (18 chains, each irreplaceable). With 100 members the arithmetic is the other way
    round: one bad member is 1% of a result, and taking its three siblings on that GPU
    down with it turns a 1% loss into 4%.
    """
    from aires import walker as W

    mine = shard_of(chs, nshards, shard)
    print(f"[walk] host={socket.gethostname()} shard {shard}/{nshards}: "
          f"{[c.label for c in mine]}")
    if max_segments:
        print(f"[walk] SMOKE: chains truncated to {max_segments} segment(s). This does "
              f"NOT reach the peak and produces no verdict - it exercises the loop.")
    if not mine:
        return 0

    # Validate every cached segment BEFORE loading the model, so a stale cache costs a
    # second rather than an allocation plus a five-minute XLA compile.
    for ch in mine:
        if ch.member is None:
            SC.ensure_dirs(ch.event)
        else:
            SC.ensure_ens_dirs(ch.event, ch.weeks)
        for (k, lead, _n) in (ch.schedule[:max_segments] if max_segments
                              else ch.schedule):
            st, detail = _segment_status(ch, k, lead)
            # `ch.segment_dir(k)`, NEVER `SC.segment_dir(ch.event, ch.walker, k)`. A
            # member chain's `walker` is the LEAD's slot: at week 8 that is slot 2, so the
            # slot-built path is job 1189's frozen `walkers/w02/stepKK/` and this rmtree
            # (ignore_errors=True) would delete it without a word on the first stale or
            # partial segment of member 0.
            if st == "partial":
                print(f"[walk] incomplete {ch.label} step{k}: {detail} - re-rolling")
                shutil.rmtree(ch.segment_dir(k), ignore_errors=True)
            elif st == "stale":
                print(f"[walk] STALE {ch.label} step{k} (lead {lead:g} d): {detail}",
                      file=sys.stderr)
                if not prune_stale_states:
                    print("[walk] ERROR: a cached segment sits at a lead this schedule "
                          "does not use, so it may have been rolled under a different "
                          "chain_schedule(). Re-run with --prune-stale to delete and "
                          "rebuild it; NEVER resume a chain whose schedule changed.",
                          file=sys.stderr)
                    return 1
                shutil.rmtree(ch.segment_dir(k), ignore_errors=True)

    get_bundle = W.lazy_bundle(SC.N_WALKER_MEMBERS)
    t_start = time.perf_counter()
    rolled = 0
    failed: list[tuple[str, str]] = []

    def _walk_one(ch: Chain) -> int:
        """Roll ONE chain to its walk end. Returns 0, or 1 with the reason printed."""
        nonlocal rolled
        segs = ch.schedule[:max_segments] if max_segments else ch.schedule
        dest = (f"peak {ch.peak}" if ch.walks_to_peak else
                f"the {ch.walk_days} d launch state {ch.walk_end} "
                f"(FCN3-extension lead; the peak {ch.peak} is FCN3's to reach)")
        print(f"[walk] {ch.label}: {len(segs)} segment(s), "
              f"{sum(n for _k, _l, n in segs)} steps, init {ch.init} -> {dest}",
              flush=True)
        parent = None
        for (k, lead, n_steps) in segs:
            st, _ = _segment_status(ch, k, lead)
            if st == "ok":
                ensure_record(ch, k, lead, n_steps)
                parent = ch.state_path(k)
                continue
            if parent is not None and not parent.exists():
                print(f"[walk] ERROR: {ch.label} step{k} needs parent {parent}, which has "
                      f"been pruned. Delete this chain's tree and re-walk it.",
                      file=sys.stderr)
                if ch.member is not None:
                    print(f"[walk]        A finished member cannot be re-walked "
                          f"incrementally - its states were released. Delete "
                          f"{SC.ens_member_dir(ch.event, ch.weeks, ch.member)} and re-walk "
                          f"with --ens-members {ch.member}.", file=sys.stderr)
                return 1
            # Segment 1's parent is named EXPLICITLY, never left to default. With
            # `parent_state=None`, `run_segment` falls back to `walker.initial_state`,
            # which resolves the init frames through `aconfig.WALKER_WEEKS` -- i.e. the
            # AIRES_WEEKS env var, 3 by default. This process spans every lead at once, so
            # every chain would have silently started from the WEEK-3 init: the smoke run
            # of job 1187 printed "init 2021-05-31" and then rolled from 2021-06-07.
            # `_segment_status` catches it one segment later, but a guard that fires after
            # a 5-minute XLA compile is not where this belongs.
            src = parent if parent is not None else ch.inputs_path()
            if not src.exists():
                print(f"[walk] ERROR: {ch.label}: no init frames at {src}",
                      file=sys.stderr)
                return 1
            # `ch.seed`, NOT `ch.walker`: it is the walker slot for the frozen sweep (so
            # job 1189's noise streams are reproduced exactly) and the MEMBER index for an
            # ensemble chain, which is what makes 100 members 100 different GenCast draws
            # rather than 100 copies of one.
            W.run_segment(get_bundle, ch.event, ch.seed, k, n_steps,
                          parent_state=src,
                          state_path=ch.state_path(k), diag_path=ch.diag_path(k))
            rolled += 1
            got, detail = _segment_status(ch, k, lead)
            if got != "ok":
                print(f"[walk] ERROR: {ch.label} step{k} did not land on lead {lead:g} d "
                      f"({got}: {detail})", file=sys.stderr)
                return 1
            # The record is written from the state BEFORE it can be pruned: the global
            # items cannot be recovered from the CONUS crop afterwards.
            ensure_record(ch, k, lead, n_steps, force=True)
            if rolled == 1 or k == 1:
                check_physics(ch, k, lead)
            # ...and ONLY THEN is the diag narrowed. Both the record (whose
            # `nonfinite_conus` and `ranges_conus` cover all 12 variables) and the physics
            # gate read the full cube; slimming first would silently narrow the CONUS half
            # of the panel to whatever ENS_DIAG_VARS happens to name.
            _slim_diag(ch, k)
            parent = ch.state_path(k)
            # A member chain pins nothing: segment 1 is the score arm's launch state and
            # `--ens` cannot score, and the final state is released below anyway.
            for dead in prunable(segs, k, keep, pin=ch.member is None):
                p = ch.state_path(dead)
                if p.exists():
                    freed = p.stat().st_size
                    p.unlink()
                    print(f"  [walk] pruned {ch.label} step{dead} state "
                          f"({freed / 1e9:.2f} GB)")

        if max_segments:
            print(f"[walk] {ch.label}: SMOKE complete ({len(segs)} segment(s), lead "
                  f"{segs[-1][1]:g} d of {ch.walk_days} d). Not checked against the walk "
                  f"end - a truncated chain is not supposed to reach it.", flush=True)
            return 0

        # The chain MUST land on its walk end. Nothing downstream checks this, so a chain
        # one day short - the ragged-tail failure mode - would otherwise go unnoticed until
        # A_L came out of a window that does not exist.
        #
        # The state may be GONE: `--stage prune` deletes it, and re-running the walk pool
        # over a finished chain is now routine (adding a lead re-runs every shard, and
        # cached chains are supposed to cost seconds). The per-segment record carries the
        # same valid time, written at walk time from the state itself, so a pruned chain is
        # verified from the record rather than failing the pool it did nothing wrong in.
        final = ch.state_path(segs[-1][0])
        rec = ch.record_path(segs[-1][0])
        if final.exists():
            with xr.open_dataset(final) as s:
                landed = pd.Timestamp(W.valid_time(s))
        elif rec.exists():
            landed = pd.Timestamp(json.loads(rec.read_text())["valid_time"])
            print(f"[walk] {ch.label}: final state pruned; landing verified from "
                  f"{rec.name}")
        else:
            print(f"[walk] ERROR: {ch.label}: no final state at {final} and no record at "
                  f"{rec}", file=sys.stderr)
            return 1
        if landed != ch.walk_end:
            what = ("the event peak" if ch.walks_to_peak else
                    f"its {ch.walk_days} d launch state")
            print(f"[walk] ERROR: {ch.label} finished at {landed}, but {what} is "
                  f"{ch.walk_end}. The segment schedule does not tile the walk.",
                  file=sys.stderr)
            return 1
        where = ("exactly on the peak" if ch.walks_to_peak else
                 f"on its {ch.walk_days} d launch state; FCN3 carries the remaining "
                 f"{ch.horizon_days - ch.walk_days} d to the peak")
        print(f"[walk] {ch.label}: complete, landed {where} {landed} "
              f"({time.perf_counter() - t_start:.0f} s elapsed on this shard)", flush=True)
        # AFTER the landing assertion, which may have read the final state.
        _release_states(ch)
        _print_member_verdict(ch)
        return 0

    for ch in mine:
        if ch.member is None:
            rc = _walk_one(ch)
            if rc:
                return rc                       # the frozen sweep keeps fail-fast
            continue
        try:
            if _walk_one(ch):
                failed.append((ch.label, "the chain reported an error above"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[walk] FAILED {ch.label}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append((ch.label, f"{type(e).__name__}: {e}"))

    print(f"[walk] shard {shard}: {rolled} segment(s) rolled, "
          f"{time.perf_counter() - t_start:.0f} s total")
    if failed:
        print(f"[walk] shard {shard}: {len(failed)} member chain(s) FAILED and "
              f"{len(mine) - len(failed)} finished:", file=sys.stderr)
        for label, why in failed:
            print(f"[walk]   {label}: {why}", file=sys.stderr)
        print(f"[walk] The finished members are on disk and complete. Re-walk the failed "
              f"ones with --ens-members <indices>; the reduce stage names any member it "
              f"has no record for rather than mixing it into the denominator.",
              file=sys.stderr)
        return 1
    return 0


def _slim_diag(chain: Chain, step: int) -> None:
    """Narrow a member's ``diag.nc`` to ``sconfig.ENS_DIAG_VARS``. Members only.

    100 members x 19 segments x 36 MB is 65 GB of CONUS diagnostics against 267 GB free on
    a 98%-full shared NFS. Keeping T2m alone costs ~0.6 MB per segment and loses nothing
    the ensemble's question needs: every CONUS item on the panel, and the ``A_L``
    observable itself, is built from ``2m_temperature``.

    What it DOES lose is the ability to recompute a non-T2m CONUS diagnostic without a
    re-walk. That is why it runs after ``ensure_record(force=True)`` - the record's
    ``nonfinite_conus`` and ``ranges_conus`` are computed from the full cube first - and
    why the surviving variable list is written into the file as ``astab_diag_vars`` and
    into the record as ``diag_vars``. A cube that has been slimmed says so.

    Idempotent: a resumed run that meets an already-slim cube leaves it alone.
    """
    keep = tuple(SC.ENS_DIAG_VARS)
    if chain.member is None or not keep:
        return
    dp = chain.diag_path(step)
    if not dp.exists():
        return
    have = [v for v in keep if v is not None]
    with xr.open_dataset(dp) as d:
        present = [str(v) for v in d.data_vars]
        want = [v for v in have if v in present]
        if not want:
            print(f"  [walk] {chain.label} step{step}: diag has none of "
                  f"{list(keep)} ({present}); NOT slimming", file=sys.stderr)
            return
        if set(present) == set(want):
            return                                    # already slim
        slim = d[want].load()
        slim.attrs.update(d.attrs)
    before = dp.stat().st_size
    slim.attrs["astab_diag_vars"] = ",".join(sorted(want))
    slim.attrs["astab_diag_dropped"] = ",".join(sorted(set(present) - set(want)))
    tmp = dp.with_suffix(f".nc.tmp.{os.getpid()}")
    slim.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4} for v in want})
    os.replace(tmp, dp)
    print(f"  [walk] slimmed {chain.label} step{step} diag to {want} "
          f"({before / 1e6:.0f} -> {dp.stat().st_size / 1e6:.1f} MB)")


def _release_states(chain: Chain) -> float:
    """Delete a finished member's remaining ``state.nc`` files. Members only. Returns GB.

    The rolling prune leaves two states resident at the end of every chain; 100 members
    would therefore finish holding ~95 GB, and unpruned they would have held 906 GB. The
    frozen sweep keeps its states until ``--stage prune`` because the adapter-fed score
    arm launches from segment 1 - a member chain is never scored, so nothing claims them.

    ``sconfig.ENS_KEEP_STATE_MEMBERS`` is the exception, and it is one member: the one
    sharing job 1189's noise stream, whose final state is the determinism control's only
    physical evidence. Its EARLIER states still go - only the last is kept.
    """
    if chain.member is None:
        return 0.0
    keep_final = chain.member in SC.ENS_KEEP_STATE_MEMBERS
    last = chain.schedule[-1][0]
    freed = 0
    for (k, _l, _n) in chain.schedule:
        if keep_final and k == last:
            continue
        p = chain.state_path(k)
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
    if freed or keep_final:
        print(f"  [walk] released {chain.label} states ({freed / 1e9:.2f} GB)"
              + (f"; kept step{last} (AIRES_ENS_KEEP_STATE_MEMBERS)" if keep_final
                 else ""))
    return freed / 1e9


def _print_member_verdict(chain: Chain) -> None:
    """One line per member at its chain's end, from the records it just wrote.

    The point is that a 4.5 h shard log should answer "did this member blow up?" without
    waiting for the reduce stage - and a diverged member is the interesting one, so it
    must not be discoverable only in aggregate hours later. Uses the same
    ``ensemble.member_summary`` the reduce does, so the log and the CSV cannot disagree.
    """
    if chain.member is None:
        return
    try:
        from astab.diag import load_bands
        from astab.ensemble import format_member_verdict, member_summary

        print(f"[walk] {format_member_verdict(member_summary(chain, load_bands(chain.event)))}",
              flush=True)
    except Exception as e:                     # a verdict is a courtesy, never a gate
        print(f"[walk] {chain.label}: verdict unavailable ({type(e).__name__}: {e})")


# --------------------------------------------------------------------------- #
# stage: score  (GPU, fcn3)
#
# Two arms, and they answer different questions:
#
#   --arm free  ERA5 IC at the chain's init -> the peak. Isolates the SCORER's stability
#               from the walker's drift: whatever FCN3 does here, it did on its own.
#   --arm fed   the chain's lead-3 d walker state -> the peak, through the adapter.
#               Confirms the coupled walker -> adapter -> FCN3 path survives, and it is
#               the LONGEST coupled rollout available at each lead.
#
# The free arm is `fcn3/run_fcn3.py` unmodified, in a subprocess: FCN3_WEEKS is read at
# module import, so one process cannot span several leads. The fed arm is `score.run_task`,
# the same unit Gate 3 and the production DMC loop use. No new rollout code in either.
# --------------------------------------------------------------------------- #
def _fcn3_extra_vars_env() -> dict:
    """Ask both FCN3 drivers for global z500 alongside their usual outputs.

    Items 4 and 7 need z500, and FCN3's production cube is CONUS-cropped before it reaches
    disk and carries no z500 at all. ``fevents.fcn3_vars`` honours ``FCN3_EXTRA_VARS``, so
    this is opt-in for the diagnostic path and leaves every frozen cube untouched.
    """
    return dict(FCN3_EXTRA_VARS="z500,t50")


def stage_score(chs: list[Chain], shard: int, nshards: int, *, arm: str,
                members: int, batch_size: int, keep_zarr: bool) -> int:
    mine = shard_of(chs, nshards, shard)
    print(f"[score] host={socket.gethostname()} shard {shard}/{nshards} arm={arm}: "
          f"{[c.label for c in mine]}")
    if not mine:
        return 0
    rc = 0
    if arm in ("free", "both"):
        rc |= _score_free(mine, members=members, batch_size=batch_size,
                          keep_zarr=keep_zarr)
    if arm in ("fed", "both"):
        rc |= _score_fed(mine, members=members, batch_size=batch_size,
                         keep_zarr=keep_zarr)
    return rc


def _score_free(mine: list[Chain], *, members: int, batch_size: int,
                keep_zarr: bool) -> int:
    """The adapter-free arm: ``fcn3/run_fcn3.py --stage infer``, one process per lead."""
    rc = 0
    for ch in mine:
        out = ch.fcn3_free_cube_path()
        if out.exists():
            print(f"[score] {ch.label} free: cached {out.name}, skip")
            continue
        if not ch.fcn3_ic_path().exists():
            print(f"[score] {ch.label} free: ERROR no ERA5 IC at {ch.fcn3_ic_path()}\n"
                  f"        Build it on the login node: FCN3_WEEKS={ch.weeks} "
                  f"FCN3_EVENTS={ch.event} python fcn3/run_fcn3.py --stage prep",
                  file=sys.stderr)
            rc = 1
            continue
        env = dict(os.environ)
        env.update(_fcn3_extra_vars_env())
        env["FCN3_WEEKS"] = str(ch.weeks)          # read at MODULE IMPORT: one per week
        env["FCN3_EVENTS"] = ch.event
        cmd = [sys.executable, str(REPO / "fcn3" / "run_fcn3.py"), "--stage", "infer",
               "--members", str(members), "--shard", "0", "--nshards", "1",
               "--batch-size", str(batch_size)]
        if keep_zarr:
            cmd.append("--keep-zarr")
        nsteps = ch.horizon_days * 24 // F.STEP_H
        print(f"[score] {ch.label} free: {members} member(s) x {nsteps} x {F.STEP_H} h "
              f"from the ERA5 IC at {ch.init} -> peak {ch.peak}", flush=True)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, env=env, cwd=str(REPO))
        print(f"[score] {ch.label} free: rc={p.returncode} in "
              f"{time.perf_counter() - t0:.0f} s")
        rc |= 1 if p.returncode else 0
    return rc


def _score_fed(mine: list[Chain], *, members: int, batch_size: int,
               keep_zarr: bool) -> int:
    """The adapter-fed arm: ``score.run_task`` from each chain's lead-3 d state."""
    import torch

    from aires import adapter as AD
    from fcn3.run_fcn3 import _get_model, check_disco_kernel

    os.environ.update(_fcn3_extra_vars_env())
    todo = [ch for ch in mine if not ch.score_cube_path().exists()]
    for ch in mine:
        if ch not in todo:
            print(f"[score] {ch.label} fed: cached {ch.score_cube_path().name}, skip")
    if not todo:
        return 0
    missing = [ch for ch in todo if not ch.state_path(1).exists()]
    if missing:
        print(f"[score] ERROR: walker state(s) absent, e.g. {missing[0].state_path(1)}\n"
              f"        Run the walk phase first.", file=sys.stderr)
        return 1

    check_disco_kernel(strict=os.environ.get("FCN3_ALLOW_FP32") != "1")
    A.verify_against_package()
    if not torch.cuda.is_available():
        print("[score] ERROR: no CUDA device visible; scoring is a GPU stage",
              file=sys.stderr)
        return 1
    _omp = os.environ.get("OMP_NUM_THREADS")
    if _omp:
        torch.set_num_threads(int(_omp))
    calib = AD.require_calibration()
    print(f"[score] calibration = {calib.attrs.get('provenance')}")
    model = _get_model()

    failed = []
    for ch in todo:
        # Explicit paths, not the tag: `score.Task`'s tagged layout is
        # runs/aires/<event>/res/<tag>/, and this sweep's tree is runs/astab/. The
        # tag is still carried so the cube's `run_tag` attribute stays 'stab'.
        task = S.Task(ch.event, ch.walker, SC.SCORE_LEAD_DAYS, SC.STAB_TAG,
                      state_p=ch.state_path(1), cube_p=ch.score_cube_path(),
                      zarr_p=ch.score_zarr_path())
        try:
            # `F.event()` resolves `Event.init` through FCN3_WEEKS, which this process
            # cannot make agree with four different leads at once. It does not have to:
            # `run_task` reads the launch time from the STATE's own `valid_time` and the
            # horizon from `ev.peak`, and touches `ev.init` nowhere. The registry event is
            # therefore correct here whatever FCN3_WEEKS happens to say - which is worth
            # stating, because the same assumption in `stage_walk` was a bug (job 1187).
            S.run_task(model, ch.ev, task, members=members, batch_size=batch_size,
                       calib=calib, keep_zarr=keep_zarr, device="cuda")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[score] FAILED {ch.label} fed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            failed.append(ch.label)
    if failed:
        print(f"[score] {len(failed)} fed task(s) failed: {failed}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# stage: prune
# --------------------------------------------------------------------------- #
def stage_prune(chs: list[Chain], *, force: bool = False) -> int:
    """Release the retained states and Zarrs. **This is the LAST step, and it is one-way.**

    Two things stop working afterwards, both learned by doing them during job 1189:

    * the adapter-fed score arm cannot be re-run - it launches from each chain's lead-3 d
      ``state.nc``, and prune deletes it. Re-scoring then needs those segments re-walked
      (``--stage walk --max-segments 1``, ~15 min on 8 GPUs), which is cheap but is not
      nothing and is not obvious from the error;
    * the FCN3 GLOBAL items have no other source than the pre-crop Zarrs. ``reduce`` now
      carries previously computed values forward (see ``_merge_panels``) so re-running it
      is safe, but it can never RE-derive them.

    So the score cubes are checked first: if any is missing, the run is not finished and
    pruning would turn a 20-minute gap into a re-walk.
    """
    missing_scores = [ch.label for ch in chs
                      if not ch.score_cube_path().exists()
                      or not ch.fcn3_free_cube_path().exists()]
    if missing_scores and not force:
        print(f"[prune] REFUSING: {len(missing_scores)} chain(s) have no score cube yet, "
              f"e.g. {missing_scores[0]}.\n"
              f"        Pruning now deletes the lead-3 d states the adapter-fed arm "
              f"launches from, so the score phase would need those segments re-walked "
              f"first. Finish scoring, or pass --force.", file=sys.stderr)
        return 1
    freed = 0
    for ch in chs:
        if not ch.result_path().exists() and not force:
            print(f"[prune] {ch.label}: no {ch.result_path().name} yet - refusing to "
                  f"release its states. Run --stage reduce first, or --force.")
            continue
        segs = ch.schedule
        for (k, _l, _n) in segs:
            if not ch.record_path(k).exists() and not force:
                continue
            p = ch.state_path(k)
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
        for z in (ch.fcn3_free_zarr_path(), ch.score_zarr_path()):
            if z.exists():
                freed += sum(f.stat().st_size for f in z.rglob("*") if f.is_file())
                shutil.rmtree(z)
                print(f"[prune] removed {z}")
    print(f"[prune] released {freed / 1e9:.1f} GB. diag.nc and stab.json are never "
          f"pruned - every diagnostic in the sweep is derived from them.")
    _print_free_space()
    return 0


