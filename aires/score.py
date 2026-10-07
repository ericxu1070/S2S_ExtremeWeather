#!/usr/bin/env python
"""AI+RES's score function: ``theta(X)`` - what an FCN3 ensemble says a walker will become.

In the paper the score is the whole point of the algorithm: at each resampling time every
walker is handed to a cheap emulator, which runs a short ensemble forecast of the target
observable, and the ensemble MEAN of that observable becomes the walker's score. Walkers
with high scores are cloned; low ones are killed. Here the walker is GenCast and the
emulator is FourCastNet 3.

    walker state at t_k  ->  aires.adapter  ->  M FCN3 forecasts to the event peak
                                            ->  A_L per member  ->  theta = mean

This module is the middle two arrows, and only the middle two: it produces a cube per
(walker, t_k) and stops. Reducing that cube to ``theta`` needs the 1990-2019 climatology
and ``xres.xmetrics``, which live in the GenCast env, so it happens in ``aires/gate3.py``
(and, in production, in the DMC driver). The split is the same two-env handoff the rest of
the stack uses, for the same reason: JAX and PyTorch/earth2studio cannot share a process.

Env: ``fcn3``. The path/seed helpers at the top are pure stdlib+numpy and import in both
envs, which is what lets a GenCast-side driver enumerate and check the same artifacts.

Seeding: COMMON RANDOM NUMBERS across walkers
---------------------------------------------
The seed is a function of ``(event, lead)`` and deliberately NOT of the walker, so every
walker scored at the same t_k is rolled with the same M internal-noise streams. Gate 3
measures how well ``theta`` RANKS walkers; making the noise common across walkers removes
it as a source of ranking error, exactly as paired sampling does in Gate 2. It is not a
thumb on the scale - production scoring should do the same thing - and if earth2studio's
generator does not in fact reproduce a member's stream across calls, the only consequence
is that theta is noisier, never biased.

This holds only while ``batch_size`` is fixed (member i's draws begin at offset
i * nsteps in the stream), which is why the default is 1 and the value is recorded in the
cube's attributes.

    # GPU, fcn3 env, one process per GPU
    PYTHONPATH=. python -m aires.score --event PNW_HeatDome_2021 --shard 0 --nshards 8
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from fcn3 import fevents as F

# earth2studio / huggingface_hub read these at import time -> set before any e2s import.
os.environ.setdefault("EARTH2STUDIO_CACHE", str(F.FCN3_CACHE / "e2s"))
os.environ.setdefault("HF_HOME", str(F.FCN3_CACHE / "hf"))


# --------------------------------------------------------------------------- #
# Task enumeration - shared by the GPU driver and the CPU reducer, so a missing
# artifact is always described by the same (walker, lead) pair.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    """One score forecast. ``tag`` picks the tree it reads and writes.

    ``tag=None`` is Gate 3's flat layout. Any other value is an AI+RES run tree
    (``runs/aires/<event>/res/<tag>/``), which is separate because a resampled walker slot
    is not a lineage - see ``aconfig``'s run-tree comment. The segment a lead corresponds
    to is the same arithmetic either way (segments are 3 d), so nothing else changes.

    ``state_p``/``cube_p``/``zarr_p`` override those two layouts entirely, for a driver
    whose tree is neither - ``astab`` writes to ``runs/astab/<event>/``. They are set
    together or not at all: a Task that reads one tree and writes another is a cache that
    can never be checked. ``tag`` is still carried when they are used, because it is
    stamped on the cube as ``run_tag`` and is how an artifact says which run made it.
    """
    event: str
    walker: int
    lead_days: float
    tag: str | None = None
    state_p: Path | None = None
    cube_p: Path | None = None
    zarr_p: Path | None = None

    def __post_init__(self):
        given = [p is not None for p in (self.state_p, self.cube_p, self.zarr_p)]
        if any(given) and not all(given):
            raise ValueError(
                "state_p, cube_p and zarr_p are set together or not at all; got "
                f"state_p={self.state_p}, cube_p={self.cube_p}, zarr_p={self.zarr_p}")

    @property
    def step(self) -> int:
        return A.segment_for_lead(self.lead_days)

    @property
    def state_path(self) -> Path:
        if self.state_p is not None:
            return Path(self.state_p)
        if self.tag is None:
            return A.state_path(self.event, self.walker, self.step)
        return A.res_state_path(self.event, self.walker, self.step, self.tag)

    @property
    def cube_path(self) -> Path:
        if self.cube_p is not None:
            return Path(self.cube_p)
        if self.tag is None:
            return A.score_cube_path(self.event, self.walker, self.lead_days)
        return A.res_score_cube_path(self.event, self.walker, self.lead_days, self.tag)

    @property
    def zarr_path(self) -> Path:
        if self.zarr_p is not None:
            return Path(self.zarr_p)
        if self.tag is None:
            return A.score_zarr_path(self.event, self.walker, self.lead_days)
        return A.res_score_zarr_path(self.event, self.walker, self.lead_days, self.tag)

    def __str__(self) -> str:
        return f"w{self.walker:02d}@{self.lead_days:g}d"


def tasks(event: str, walkers: int, leads=None, tag: str | None = None) -> list[Task]:
    """Every (walker, checkpoint) pair, walker-major.

    Walker-major ordering plus ``i % nshards`` sharding balances the work by construction
    when ``len(leads)`` and ``nshards`` are coprime (5 and 8 here): each shard then draws
    each lead - i.e. each rollout length - the same number of times, so no GPU is left
    holding all the long forecasts. AI+RES scores ONE lead per pool launch (the DMC loop
    is a barrier), where every task is the same rollout length and the balance is exact
    for any shard count.
    """
    leads = A.GATE3_LEAD_DAYS if leads is None else leads
    return [Task(event, w, float(l), tag) for w in range(walkers) for l in leads]


def shard_of(all_tasks: list[Task], nshards: int, shard: int) -> list[Task]:
    return [t for i, t in enumerate(all_tasks) if i % nshards == shard]


def score_seed(event: str, lead_days: float) -> int:
    """Deterministic per-(event, lead) seed; see the module docstring on common random
    numbers. CRC32 of a labelled string rather than arithmetic on ``F.seed_for`` - the
    latter spaces events only 1000 apart and any walker/lead offset large enough to be
    collision-free would run into the next event."""
    key = f"aires-score|{event}|lead{lead_days:g}"
    return zlib.crc32(key.encode()) % (2 ** 31 - 1)


def nsteps_for(ev: F.Event, valid_time) -> int:
    """FCN3 steps from a walker state's valid time to the event peak.

    Computed from the state's OWN timestamp rather than from the nominal lead, so a state
    that is not where the schedule says it is fails loudly on the assertion below instead
    of producing a forecast that stops short of the verification window.
    """
    delta = pd.Timestamp(ev.peak) - pd.Timestamp(valid_time)
    n, rem = divmod(int(delta.total_seconds()), F.STEP_H * 3600)
    if rem or n <= 0:
        raise ValueError(f"state valid {valid_time} is not a whole number of "
                         f"{F.STEP_H} h steps before the peak {ev.peak} (got {delta})")
    return n


# --------------------------------------------------------------------------- #
# Cube assembly (GenCast schema, CONUS crop) - the same shape xres/fcn3 cubes have,
# so aires.aindex scores all three through one code path.
# --------------------------------------------------------------------------- #
def assemble_cube(ev: F.Event, task: Task, zarr_path: Path, out_path: Path, *,
                  init, members: int, seed: int, batch_size: int) -> Path:
    from fcn3.run_fcn3 import _crop_conus

    ds = xr.open_zarr(zarr_path)
    init = np.datetime64(pd.Timestamp(init))

    data_vars: dict[str, xr.DataArray] = {}
    by_level: dict[str, dict[int, xr.DataArray]] = {}
    for v in F.fcn3_vars(ev):
        gname, lev = F.FCN3_TO_GENCAST[v]
        da = ds[v]                                    # (ensemble, time, lead_time, lat, lon)
        valid = init + da["lead_time"].values
        da = da.isel(time=0).drop_vars("time", errors="ignore")
        da = da.rename({"ensemble": "member", "lead_time": "time"})
        da = da.assign_coords(time=("time", valid), member=("member", np.arange(members)))
        da = _crop_conus(da)
        if lev is None:
            data_vars[gname] = da
        else:
            by_level.setdefault(gname, {})[lev] = da
    for gname, per_lev in by_level.items():
        levs = sorted(per_lev)
        data_vars[gname] = xr.concat([per_lev[l] for l in levs], dim="level") \
                             .assign_coords(level=("level", np.array(levs)))

    cube = xr.Dataset(data_vars)
    cube.attrs.update(
        event=ev.name, family=ev.family, metric=ev.metric, model="FourCastNet3",
        role="aires_score_forecast", walker=int(task.walker),
        score_lead_days=float(task.lead_days), init=str(pd.Timestamp(init)),
        peak=ev.peak, members=int(members), step_h=F.STEP_H,
        nsteps=int(cube.sizes["time"]), seed=int(seed), batch_size=int(batch_size),
        walker_state=str(task.state_path), resolution=F.RES,
        run_tag=task.tag or "gate3",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".nc.tmp")
    cube.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4} for v in cube.data_vars})
    os.replace(tmp, out_path)
    ds.close()
    return out_path


# --------------------------------------------------------------------------- #
# infer (GPU, fcn3 env)
# --------------------------------------------------------------------------- #
def run_task(model, ev: F.Event, task: Task, *, members: int, batch_size: int,
             calib, keep_zarr: bool, device: str) -> dict:
    """One score forecast: walker state -> adapter -> M FCN3 members -> cube."""
    from earth2studio.io import ZarrBackend
    from earth2studio.perturbation import Zero
    from earth2studio.run import ensemble

    from aires import adapter as AD
    # Only the state-contract helpers are used here; aires/walker.py keeps every JAX
    # import inside its rolling functions, so importing it in the fcn3 env is safe.
    from aires import walker as W

    state = W.read_state(task.state_path)
    valid = W.valid_time(state)
    n = nsteps_for(ev, valid)

    ic = AD.gencast_to_fcn3(state, calib=calib)
    if not np.isfinite(ic.values).all():
        raise ValueError(f"{task}: adapter produced non-finite channels from "
                         f"{task.state_path}")
    data = AD.GenCastICSource(ic)
    state.close()

    seed = score_seed(ev.name, task.lead_days)
    model.set_rng(seed=seed, reset=True)

    if task.zarr_path.exists():
        shutil.rmtree(task.zarr_path)
    task.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    io = ZarrBackend(file_name=str(task.zarr_path),
                     chunks={"ensemble": 1, "time": 1, "lead_time": 1},
                     backend_kwargs={"overwrite": True})
    out_vars = F.fcn3_vars(ev)

    print(f"[score] {task}: {members} member(s) x {n} x {F.STEP_H} h from {valid} "
          f"-> peak {ev.peak}  (seed {seed}, vars {out_vars})", flush=True)
    t0 = time.perf_counter()
    io = ensemble([pd.Timestamp(valid).strftime("%Y-%m-%dT%H:%M:%S")], n, members,
                  model, data, io, Zero(), batch_size=batch_size,
                  output_coords={"variable": np.array(out_vars)}, device=device,
                  verbose=False)
    elapsed = time.perf_counter() - t0

    assemble_cube(ev, task, task.zarr_path, task.cube_path, init=valid, members=members,
                  seed=seed, batch_size=batch_size)
    if not keep_zarr and task.zarr_path.exists():
        shutil.rmtree(task.zarr_path)
    print(f"[score] {task}: {elapsed:.1f} s ({elapsed / max(n, 1):.2f} s/step) -> "
          f"{task.cube_path.name} ({task.cube_path.stat().st_size / 1e6:.0f} MB)", flush=True)
    return dict(task=str(task), seconds=elapsed, nsteps=n)


def stage_infer(ev: F.Event, *, walkers: int, leads, members: int, shard: int,
                nshards: int, batch_size: int, keep_zarr: bool,
                tag: str | None = None) -> int:
    import torch

    from aires import adapter as AD
    from fcn3.run_fcn3 import _get_model, check_disco_kernel

    A.ensure_dirs(ev.name)
    if tag is not None:
        A.res_ensure_dirs(ev.name, tag)
    check_disco_kernel(strict=os.environ.get("FCN3_ALLOW_FP32") != "1")
    A.verify_against_package()

    mine = shard_of(tasks(ev.name, walkers, leads, tag), nshards, shard)
    todo = [t for t in mine if not t.cube_path.exists()]
    print(f"[score] host={socket.gethostname()} shard {shard}/{nshards}: "
          f"{len(mine)} task(s), {len(mine) - len(todo)} cached, {len(todo)} to run")
    if not todo:
        return 0

    missing = [t for t in todo if not t.state_path.exists()]
    if missing:
        where = ("PYTHONPATH=. python -m aires.gate3 --stage walk" if tag is None else
                 f"PYTHONPATH=. python -m aires.run_aires --stage walk --tag {tag}")
        print(f"[score] ERROR: {len(missing)} walker state(s) do not exist, e.g. "
              f"{missing[0].state_path}\n"
              f"        Run the walk phase first: {where}", file=sys.stderr)
        return 1

    _omp = os.environ.get("OMP_NUM_THREADS")
    if _omp:
        torch.set_num_threads(int(_omp))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[score] ERROR: no CUDA device visible; scoring is a GPU stage",
              file=sys.stderr)
        return 1

    calib = AD.require_calibration()
    print(f"[score] calibration = {calib.attrs.get('provenance')}")
    model = _get_model()

    failed = []
    for t in todo:
        try:
            run_task(model, ev, t, members=members, batch_size=batch_size, calib=calib,
                     keep_zarr=keep_zarr, device=device)
        except Exception as e:              # one bad task must not lose the other nine
            import traceback
            traceback.print_exc()
            print(f"[score] FAILED {t}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(str(t))
    if failed:
        print(f"[score] shard {shard}: {len(failed)} task(s) failed: {failed}",
              file=sys.stderr)
        return 1
    print(f"[score] shard {shard}: all {len(todo)} task(s) done")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default=os.environ.get("AIRES_EVENT", "PNW_HeatDome_2021"))
    ap.add_argument("--walkers", type=int, default=A.GATE3_N_WALKERS)
    ap.add_argument("--members", type=int, default=A.GATE3_M_MEMBERS)
    ap.add_argument("--leads", default=",".join(str(l) for l in A.GATE3_LEAD_DAYS))
    ap.add_argument("--shard", type=int, default=int(os.environ.get("FCN3_SHARD", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("FCN3_NSHARDS", 8)))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("FCN3_BATCH", 1)))
    ap.add_argument("--keep-zarr", action="store_true")
    ap.add_argument("--tag", default=None,
                    help="AI+RES run tag; omit for Gate 3's flat tree")
    ap.add_argument("--list", action="store_true", help="print this shard's tasks and exit")
    a = ap.parse_args(argv)

    ev = F.event(a.event)
    leads = [float(x) for x in a.leads.split(",") if x.strip()]

    if a.list:
        for t in shard_of(tasks(ev.name, a.walkers, leads, a.tag), a.nshards, a.shard):
            print(f"{t}  state={'OK ' if t.state_path.exists() else 'MISSING'}  "
                  f"cube={'cached' if t.cube_path.exists() else 'todo'}")
        return 0

    return stage_infer(ev, walkers=a.walkers, leads=leads, members=a.members,
                       shard=a.shard, nshards=a.nshards, batch_size=a.batch_size,
                       keep_zarr=a.keep_zarr, tag=a.tag)


if __name__ == "__main__":
    sys.exit(main())
