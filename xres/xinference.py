"""Inference: roll GenCast out at one resolution, CACHE THE ENTIRE output, derive metrics.

For each event we run the diffusion ensemble to the 2-week horizon and, unlike the original
pipeline (which kept only the verification-week T2m field), we stream EVERY target variable
for EVERY rollout step and EVERY member, cropped to the CONUS box, into a compressed cube:

    runs/xres/<res>/week2/cache/<event>_cube.nc
        dims (member, time, level, lat, lon); ``time`` = absolute valid datetime.

From that cube we derive the event's headline verification metric (and its free secondary)
per member:

    runs/xres/<res>/week2/verif/<event>_<metric>_members.nc   dims (member, lat, lon)

The cube is the durable artifact: any other metric (MSLP, 10 m wind, ...) can be recomputed
offline from it without re-running the model.

Serial mode (``XRES_SERIAL_INFER=1``, the 0.25deg-on-A100-40GB path) is member-resumable
and multi-worker:

- each finished member is written immediately to ``cache/<event>_memberNN.nc`` so a
  walltime-killed job resumes from the completed members instead of restarting;
- concurrent worker processes (one per GPU, launched by the PBS script) coordinate through
  claim files in ``cache/claims/`` — a member is rolled by whichever worker atomically
  creates its claim first (claims of dead PIDs are stolen, so a crashed worker's members
  are re-rolled by the survivors on their next pass);
- once all ``n_members`` member files exist, the last worker to finish assembles the cube,
  derives the verification metrics, and deletes the member files.

Claim files are only meaningful within one node/job; the PBS script clears
``cache/claims/`` at job start. If you run serial infer manually after a crash, do the same.

The pmap path (1.0deg) is unchanged: all-members rollout, cube written at the end.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pandas as pd
import xarray as xr
import jax

from graphcast import data_utils, rollout

from gencast_s2s import config as C
from gencast_s2s import model as M
from gencast_s2s.inference import build_example_batch

from . import xconfig as X
from . import xmetrics as XM
from . import xdata as XD


_WORKER = os.environ.get("XRES_WORKER", "0")

# One-member-per-job mode (PBS array): this process rolls ONLY member N of every
# event, cross-node partition is static so no claim files are used/needed.
_MEMBER_SEL = os.environ.get("XRES_MEMBER_SEL")


# --------------------------------------------------------------------------- #
# Compressed atomic NetCDF write (cubes are large; zlib roughly halves them).
# --------------------------------------------------------------------------- #
def _atomic_to_netcdf(ds: xr.Dataset, path, compress: bool = False) -> None:
    path = str(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    enc = None
    if compress:
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)


def cube_path(res: str, weeks: int, name: str):
    return X.cache_dir(res, weeks) / f"{name}_cube.nc"


def member_path(res: str, weeks: int, name: str, m: int):
    return X.cache_dir(res, weeks) / f"{name}_member{m:02d}.nc"


def verif_path(res: str, weeks: int, name: str, metric: str):
    return X.verif_dir(res, weeks) / f"{name}_{metric}_members.nc"


def claims_dir(res: str, weeks: int):
    d = X.cache_dir(res, weeks) / "claims"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Claim files: same-node work stealing between concurrent worker processes.
# --------------------------------------------------------------------------- #
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _try_claim(path) -> bool:
    """Atomically claim a unit of work. Claims held by dead PIDs are stolen.

    Only valid between processes on ONE node (PID liveness check); stale claims
    from a previous job must be cleared externally (the PBS script does).
    """
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(path.read_text().strip() or "0")
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                return False
            try:                                       # dead owner -> steal and retry
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    return False


# --------------------------------------------------------------------------- #
# Event inputs (shared by both paths).
# --------------------------------------------------------------------------- #
def _event_eval(bundle: dict, res: str, weeks: int, name: str, peak: str):
    step_h = C.model_cfg("gencast")["step_h"]
    lead_days = C.lead_days_for(weeks)
    raw = XD.load_or_build_inputs(res, weeks, name, peak)
    full, init = build_example_batch(raw, peak, lead_days, step_h)
    task = bundle["task_config"]
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        full, target_lead_times=slice(f"{step_h}h", f"{lead_days * 24}h"),
        **dataclasses.asdict(task))
    return eval_inputs, eval_targets * np.nan, eval_forcings, init


def _to_valid_time(ds: xr.Dataset, init) -> xr.Dataset:
    """Rollout lead-time axis -> absolute valid datetimes (+ lead_h coord)."""
    lead_td = ds["time"].values.astype("timedelta64[ns]")
    valid = np.datetime64(init) + lead_td
    return ds.assign_coords(
        time=("time", valid),
        lead_h=("time", (lead_td / np.timedelta64(1, "h")).astype("int32")))


# --------------------------------------------------------------------------- #
# Serial path: one member at a time, per-member caching + claims.
# --------------------------------------------------------------------------- #
def _roll_member(bundle: dict, res: str, weeks: int, name: str, peak: str,
                 m: int, nm: int, ev: tuple) -> None:
    eval_inputs, targets_nan, eval_forcings, init = ev
    n_steps = C.n_rollout_steps(weeks)
    rng = jax.random.PRNGKey(0)
    rngs = np.stack([jax.random.fold_in(rng, m)], axis=0)   # member rng == pmap path's
    gen = rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=bundle["forward"], rngs=rngs,
        inputs=eval_inputs, targets_template=targets_nan,
        forcings=eval_forcings, num_steps_per_chunk=1,
        num_samples=1, pmap_devices=None, verbose=True)
    buf = []
    for step_i, chunk in enumerate(gen, start=1):
        print(f"  [w{_WORKER} {res}] {name}: member {m + 1}/{nm}  "
              f"step {step_i}/{n_steps}", flush=True)
        buf.append(chunk.sel(lat=C.LAT, lon=C.LON).compute())
    if len(buf) != n_steps:
        raise RuntimeError(f"{name} member {m}: got {len(buf)}/{n_steps} steps")
    ds = xr.concat(buf, dim="time", coords="minimal", compat="override")
    if "sample" in ds.dims:
        ds = ds.isel(sample=0, drop=True)
    if "batch" in ds.dims:
        ds = ds.isel(batch=0, drop=True)
    ds = _to_valid_time(ds, init).expand_dims(member=[m]).transpose("member", "time", ...)
    ds.attrs.update(event=name, peak=str(pd.Timestamp(peak)), init=str(init),
                    resolution=res, weeks=int(weeks))
    _atomic_to_netcdf(ds, member_path(res, weeks, name, m), compress=True)
    print(f"  [w{_WORKER} {res}] {name}: member {m + 1}/{nm} saved", flush=True)


def _assemble_cube(res: str, weeks: int, name: str, peak: str, nm: int) -> None:
    """Member files -> cube. Tolerates losing an assembly race to another job
    (possibly on another node): if the member files vanish mid-read, the winner's
    cube is already (being) written and we simply return."""
    files = [member_path(res, weeks, name, m) for m in range(nm)]
    try:
        parts = [xr.load_dataset(f) for f in files]
    except (FileNotFoundError, OSError):
        if cube_path(res, weeks, name).exists():
            print(f"  {name}: lost assembly race (cube already written elsewhere)")
            return
        raise
    cube = xr.concat(parts, dim="member", coords="minimal", compat="override")
    cube = cube.transpose("member", "time", ...)
    cube.attrs.update(event=name, peak=str(pd.Timestamp(peak)),
                      init=parts[0].attrs.get("init", ""),
                      resolution=res, weeks=int(weeks))
    cube_f = cube_path(res, weeks, name)
    _atomic_to_netcdf(cube, cube_f, compress=True)
    print(f"  saved cube {cube_f.name}  members={cube.sizes['member']} "
          f"steps={cube.sizes['time']} vars={len(cube.data_vars)} "
          f"size={cube_f.stat().st_size / 1e6:.0f}MB", flush=True)
    _derive_metrics(res, weeks, name, peak, cube=cube)
    for f in files:                                    # cube is durable; drop member files
        f.unlink(missing_ok=True)


def _run_event_serial(get_bundle, res: str, weeks: int, name: str, peak: str,
                      nm: int, members: list[int], use_claims: bool) -> dict:
    """One pass over one event, rolling only ``members`` (all of 0..nm-1 in the
    same-node worker-pool mode; a single index in XRES_MEMBER_SEL array mode).
    Returns dict(actions=<work done>, pending=<work owned by other jobs>)."""
    cube_f = cube_path(res, weeks, name)
    if cube_f.exists():
        try:
            have = int(xr.open_dataset(cube_f).sizes.get("member", 0))
        except Exception:
            have = -1
        if have == nm:
            _derive_metrics(res, weeks, name, peak)    # ensure verif files exist
            return dict(actions=0, pending=0)
        print(f"  stale cube ({have} members != {nm}) -> re-rolling {name}")

    claims = claims_dir(res, weeks) if use_claims else None
    actions = 0
    ev = None                                          # inputs built on first rolled member
    for m in members:
        if member_path(res, weeks, name, m).exists():
            continue
        if use_claims and not _try_claim(claims / f"{name}_m{m:02d}.claim"):
            continue
        bundle = get_bundle()
        if ev is None:
            ev = _event_eval(bundle, res, weeks, name, peak)
        _roll_member(bundle, res, weeks, name, peak, m, nm, ev)
        actions += 1

    missing = [m for m in range(nm) if not member_path(res, weeks, name, m).exists()]
    if missing:                                        # other jobs/workers still rolling
        return dict(actions=actions, pending=len(missing))

    if use_claims:                                     # same-node pool: claim arbitrates
        if _try_claim(claims / f"{name}_cube.claim"):
            _assemble_cube(res, weeks, name, peak, nm)
            return dict(actions=actions + 1, pending=0)
        return dict(actions=actions, pending=0 if cube_f.exists() else 1)

    # Cross-node array mode: no reliable claims. Assemble if we just rolled a member
    # (we may be the last finisher; ties are safe — _assemble_cube tolerates races and
    # writes are atomic). Member-0's job also assembles on a resweep, so a cube missed
    # because the last finisher crashed gets picked up by resubmitting the array.
    if actions > 0 or (members and members[0] == 0):
        _assemble_cube(res, weeks, name, peak, nm)
        return dict(actions=actions + 1, pending=0)
    return dict(actions=actions, pending=0 if cube_f.exists() else 1)


# --------------------------------------------------------------------------- #
# pmap path (1.0deg): unchanged all-members rollout, cube written at the end.
# --------------------------------------------------------------------------- #
def _run_event_pmap(bundle: dict, res: str, weeks: int, name: str, peak: str) -> None:
    nm = bundle["n_members"]
    n_dev = len(bundle["devices"])
    cube_f = cube_path(res, weeks, name)
    if cube_f.exists():
        try:
            have = int(xr.open_dataset(cube_f).sizes.get("member", 0))
        except Exception:
            have = -1
        if have == nm:
            print(f"  cached cube: {cube_f.name} ({have} members)")
            _derive_metrics(res, weeks, name, peak)    # ensure verif files exist
            return
        print(f"  stale cube ({have} members != {nm}) -> re-running")

    eval_inputs, targets_nan, eval_forcings, init = _event_eval(
        bundle, res, weeks, name, peak)
    rng = jax.random.PRNGKey(0)
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(nm)], axis=0)
    gen = rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=bundle["forward"], rngs=rngs,
        inputs=eval_inputs, targets_template=targets_nan,
        forcings=eval_forcings, num_steps_per_chunk=1,
        num_samples=nm, pmap_devices=bundle["devices"], verbose=True)

    # The multi-run generator yields ``n_waves * n_steps`` chunks (one chunk per
    # ensemble wave per rollout step).  Concatenate each wave along ``time``,
    # then stack waves on ``sample`` — NOT all chunks along ``time``.
    n_steps = C.n_rollout_steps(weeks)
    n_waves = nm // n_dev
    total_chunks = n_steps * n_waves
    print(f"  rollout: {n_steps} steps x {n_waves} waves of {n_dev} members "
          f"({nm} total, pmap x{n_dev}); JAX compile on first step often takes 15-40 min",
          flush=True)
    waves, buf = [], []
    for chunk_i, chunk in enumerate(gen, start=1):
        wave = (chunk_i - 1) // n_steps
        step = (chunk_i - 1) % n_steps
        m0 = wave * n_dev
        m1 = m0 + int(chunk.sizes.get("sample", n_dev)) - 1
        print(f"  [{res}] {name}: step {step + 1}/{n_steps}  "
              f"members {m0}-{m1}  (chunk {chunk_i}/{total_chunks})", flush=True)
        buf.append(chunk.sel(lat=C.LAT, lon=C.LON).compute())
        if len(buf) == n_steps:
            waves.append(xr.concat(buf, dim="time", coords="minimal", compat="override"))
            print(f"  [{res}] {name}: wave {len(waves)}/{n_waves} complete "
                  f"(members {m0}-{m1})", flush=True)
            buf = []
    if buf:
        raise RuntimeError(
            f"rollout ended with {len(buf)}/{n_steps} steps buffered for {name}")
    cube = xr.concat(waves, dim="sample", coords="minimal", compat="override")

    # member axis + drop the singleton batch
    if "sample" in cube.dims:
        cube = cube.rename({"sample": "member"})
    if "batch" in cube.dims:
        cube = cube.isel(batch=0, drop=True)
    if "member" not in cube.dims:
        cube = cube.expand_dims(member=[0])

    cube = _to_valid_time(cube, init).transpose("member", "time", ...)
    cube.attrs.update(event=name, peak=str(pd.Timestamp(peak)), init=str(init),
                      resolution=res, weeks=int(weeks))

    _atomic_to_netcdf(cube, cube_f, compress=True)
    print(f"  saved cube {cube_f.name}  members={cube.sizes['member']} "
          f"steps={cube.sizes['time']} vars={len(cube.data_vars)} "
          f"size={cube_f.stat().st_size / 1e6:.0f}MB")

    _derive_metrics(res, weeks, name, peak, cube=cube)


def _derive_metrics(res: str, weeks: int, name: str, peak: str,
                    cube: xr.Dataset | None = None) -> None:
    """Compute + cache the per-member headline metric (and its secondary) from the cube."""
    for metric in XD.metrics_for_event(name):
        out_f = verif_path(res, weeks, name, metric)
        if out_f.exists() and cube is None:
            continue
        if cube is None:
            cube = xr.open_dataset(cube_path(res, weeks, name)).load()
        field = XM.forecast_field(cube, peak, metric)        # (member, lat, lon)
        field = field.transpose("member", "lat", "lon")
        _atomic_to_netcdf(field.to_dataset(name=metric), out_f)
        print(f"    verif [{metric}]: members={field.sizes['member']} "
              f"mean={float(field.mean()):+.3g} -> {out_f.name}")


def run_all(res: str, weeks: int, n_members: int | None = None) -> None:
    # Surface graphcast rollout's absl logging (Chunk N/M, Samples X/Y) in PBS logs.
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.INFO)
        absl_logging.use_absl_handler()
    except Exception:
        pass

    rs = X.res_spec(res)
    requested = X.N_MEMBERS[res] if n_members is None else n_members
    serial = os.environ.get("XRES_SERIAL_INFER") == "1"
    X.ensure_dirs(res, weeks)

    # Lazy model load: a worker that finds no claimable work (everything cached or
    # owned by other workers) exits without spending ~20 min on load + compile.
    box: dict = {}
    def get_bundle():
        if "b" not in box:
            box["b"] = M.load_gencast(n_members=requested,
                                      params_file=rs["params"], res=rs["res"])
        return box["b"]

    events = list(X.events().items())
    mode = "pmap"
    if serial:
        mode = (f"serial member {_MEMBER_SEL} only" if _MEMBER_SEL is not None
                else f"serial worker {_WORKER}")
    print(f"=== xres {res} ({rs['label']}): week{weeks}, lead {C.lead_days_for(weeks)}d, "
          f"{C.n_rollout_steps(weeks)} steps, {requested} members, {mode} ===", flush=True)

    if not serial:
        bundle = get_bundle()
        for ev_i, (name, (peak, _metric)) in enumerate(events, start=1):
            print(f"{name} [{res} week{weeks}] init=peak-{C.lead_days_for(weeks)}d ... "
                  f"(event {ev_i}/{len(events)})", flush=True)
            _run_event_pmap(bundle, res, weeks, name, peak)
        return

    if _MEMBER_SEL is not None:
        # PBS-array mode: this job owns exactly one member index of every event.
        # Static cross-node partition -> no claim files; single pass suffices.
        m_sel = int(_MEMBER_SEL)
        if not 0 <= m_sel < requested:
            raise ValueError(f"XRES_MEMBER_SEL={m_sel} outside 0..{requested - 1}")
        pending = 0
        for ev_i, (name, (peak, _metric)) in enumerate(events, start=1):
            print(f"{name} [m{m_sel} {res} week{weeks}] (event {ev_i}/{len(events)})",
                  flush=True)
            st = _run_event_serial(get_bundle, res, weeks, name, peak, requested,
                                   members=[m_sel], use_claims=False)
            pending += st["pending"]
        print(f"[m{m_sel}] member {m_sel} done for all events"
              + (f"; {pending} unit(s) still owned by other member jobs" if pending
                 else "; all cubes assembled"), flush=True)
        return

    # Same-node worker-pool mode: keep passing over all events until everything this
    # worker can do is done. ``pending`` = members/cubes owned by other live workers;
    # if a pass does no work but some are pending, the owners will finish (or die, in
    # which case their claims become stealable — but by then this worker has exited;
    # the next chained job re-rolls anything they left behind).
    n_pass = 0
    while True:
        n_pass += 1
        actions = pending = 0
        for ev_i, (name, (peak, _metric)) in enumerate(events, start=1):
            print(f"{name} [w{_WORKER} {res} week{weeks}] pass {n_pass} "
                  f"(event {ev_i}/{len(events)})", flush=True)
            st = _run_event_serial(get_bundle, res, weeks, name, peak, requested,
                                   members=list(range(requested)), use_claims=True)
            actions += st["actions"]
            pending += st["pending"]
        if pending == 0:
            print(f"[w{_WORKER}] all events complete after {n_pass} pass(es)", flush=True)
            return
        if actions == 0:
            print(f"[w{_WORKER}] no claimable work left ({pending} unit(s) owned by "
                  f"other live workers) -> exiting", flush=True)
            return
