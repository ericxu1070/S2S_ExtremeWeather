#!/usr/bin/env python
"""Driver for the p90 T2m-anomaly experiment (FourCastNet3 now, GenCast later).

Scores forecast skill on 90 CONUS cases (45 p90 T2m-anomaly events 2022-2025 + 45
matched no-event controls) at a 3-week lead, 50 ensemble members, verifying the
week-mean T2m anomaly over [peak-6d, peak] on the 0.25deg CONUS box vs ERA5 truth.
See ``p90/pconfig.py`` for the frozen contract (paths, cube schema, env vars).

Four stages, split by conda env so each only imports what it has (all heavy imports
are LAZY inside stage functions):

  prep-truth              (my-env/moe, internet):  ERA5 week-mean T2m-anomaly truth for
        the verification window (peak) and the persistence baseline (init), per case.
  prep-ic  --model fcn3   (fcn3 env, internet, login node):  fetch the FCN3 global IC at
        each case's init from ARCO-ERA5, cache to NetCDF, then pre-download the FCN3
        checkpoint files WITHOUT building the model (login-node RAM).
  infer    --model fcn3   (fcn3 env, GPU):  roll the ensemble for this shard's cases and
        write a cube matching the CUBE SCHEMA; PID-claim work-stealing lets many workers
        share the case list on one node; per-case cached + atomic.
  compare  --model fcn3   (my-env/moe, offline):  metrics + figures (delegated to
        p90.pmetrics.compute_all / p90.pcompare.make_all).

--model gencast for prep-ic/infer is a hard stop: GenCast members are produced by a
separate adapter that must write cubes matching the CUBE SCHEMA into cube_path("gencast",
case); ``compare --model gencast`` then works unchanged.

Env vars (see pconfig header): P90_KIND, P90_CASES_SEL, P90_MAX_CASES, P90_N_MEMBERS,
P90_SHARD, P90_WORKER (log tag), P90_FCN3_BATCH (default 4), P90_KEEP_ZARR=1, GENCAST_RUNS.

Run from the repo root, e.g.:
  /glade/derecho/scratch/exu/conda-envs/fcn3/bin/python p90/run_p90.py --stage prep-ic --model fcn3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

# --------------------------------------------------------------------------- #
# sys.path bootstrap so `import p90.pconfig` works when run as a script from any cwd.
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

import p90.pconfig as P

# earth2studio / huggingface_hub read these at import time, so set them BEFORE any
# earth2studio import (which only happens inside the fcn3 stages). Reuse the fcn3-test
# cache: the checkpoint is already fetched there.
os.environ.setdefault("EARTH2STUDIO_CACHE", str(P.FCN3_CACHE / "e2s"))
os.environ.setdefault("HF_HOME", str(P.FCN3_CACHE / "hf"))


# --------------------------------------------------------------------------- #
# prep-truth: ERA5 week-mean T2m-anomaly truth + persistence baseline (my-env/moe).
# --------------------------------------------------------------------------- #
def stage_prep_truth():
    from gencast_s2s import data as D

    P.truth_dir().mkdir(parents=True, exist_ok=True)
    df = P.cases()
    n = len(df)
    print(f"[prep-truth] {n} cases (P90_KIND={os.environ.get('P90_KIND', '-')})")
    n_written = 0
    for k, (_, row) in enumerate(df.iterrows(), 1):
        name = row["case"]
        peak_iso = pd.Timestamp(row["peak"]).isoformat()
        init_iso = pd.Timestamp(row["init"]).isoformat()
        tpath = P.truth_path(name)
        ppath = P.persist_path(name)
        if tpath.exists() and ppath.exists():
            print(f"[prep-truth] {k}/{n} {name}: cached, skip")
            continue
        print(f"[prep-truth] {k}/{n} {name}: peak={peak_iso} init={init_iso}")
        if not tpath.exists():
            da = D.true_verif_anom(row["peak"])
            ds = da.to_dataset(name="t2m_anom")
            ds.attrs.update(case=name, peak=peak_iso, init=init_iso)
            D._atomic_to_netcdf(ds, tpath)
            n_written += 1
            print(f"[prep-truth]   wrote {tpath}")
            D.release_stores(); gc.collect()
        if not ppath.exists():
            # persistence baseline: week-mean anomaly over [init-6d, init]
            da = D.true_verif_anom(row["init"])
            ds = da.to_dataset(name="t2m_anom")
            ds.attrs.update(case=name, peak=peak_iso, init=init_iso)
            D._atomic_to_netcdf(ds, ppath)
            n_written += 1
            print(f"[prep-truth]   wrote {ppath}")
            D.release_stores(); gc.collect()
    print(f"[prep-truth] DONE: wrote {n_written} file(s) for {n} case(s)")


# --------------------------------------------------------------------------- #
# prep-ic (fcn3): fetch the FCN3 global IC per case, then pre-download the checkpoint.
# --------------------------------------------------------------------------- #
def _fcn3_variables():
    from earth2studio.models.px.fcn3 import VARIABLES
    return list(VARIABLES)


def stage_prep_ic():
    from earth2studio.data import ARCO

    P.ic_dir().mkdir(parents=True, exist_ok=True)
    df = P.cases()
    n = len(df)
    variables = _fcn3_variables()
    print(f"[prep-ic] {n} cases; {len(variables)} FCN3 variables per IC")
    data = ARCO(cache=True, verbose=True)
    n_written = 0
    for k, (_, row) in enumerate(df.iterrows(), 1):
        name = row["case"]
        icp = P.ic_path(name)
        if icp.exists():
            print(f"[prep-ic] {k}/{n} {name}: cached, skip")
            continue
        init = pd.Timestamp(row["init"])
        print(f"[prep-ic] {k}/{n} {name}: fetching IC at {init.isoformat()}")
        da = data(time=[np.datetime64(init)], variable=variables)  # (time, variable, lat, lon)
        da = da.astype("float32")
        tmp = icp.with_suffix(".nc.tmp")
        da.to_dataset(name="ic").to_netcdf(tmp)
        os.replace(tmp, icp)
        n_written += 1
        print(f"[prep-ic]   wrote {icp}  {dict(da.sizes)}")

    # Pre-download the FCN3 checkpoint files to the shared earth2studio cache (login node,
    # has internet) WITHOUT building the model -- the 710M-param CPU build OOM-kills the
    # login node. package.get() fetches each file; the model is built later on the GPU node.
    print("[prep-ic] pre-downloading FCN3 checkpoint files (no model build) ...")
    from earth2studio.models.px import FCN3
    package = FCN3.load_default_package()
    files = ["config.json", "metadata.json", "best_ckpt_mp0.tar", "orography.nc",
             "land_mask.nc", "global_means.npy", "global_stds.npy", "mins.npy", "maxs.npy"]
    for f in files:
        try:
            p = package.get(f)
            print(f"[prep-ic]   cached {f} -> {p}")
        except Exception as e:
            print(f"[prep-ic]   skip {f} ({type(e).__name__}: {e})")
    print(f"[prep-ic] DONE: wrote {n_written} IC(s); checkpoint cache "
          f"{os.environ['EARTH2STUDIO_CACHE']}")


# --------------------------------------------------------------------------- #
# PID-claim work-stealing helpers (copied from xres/xinference.py; NOT imported from
# there so the JAX stack never enters the fcn3 env). Valid only between processes on ONE
# node; stale claims from a previous job must be cleared externally (--clean-claims).
# --------------------------------------------------------------------------- #
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _try_claim(path: Path) -> bool:
    """Atomically claim a unit of work. Claims held by dead PIDs are stolen."""
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
# infer (fcn3): local IC data source + cube assembly.
# --------------------------------------------------------------------------- #
class LocalIC:
    """earth2studio DataSource backed by the cached IC NetCDF (no network in infer)."""

    def __init__(self, path):
        self.da = xr.open_dataset(path)["ic"].load()

    def __call__(self, time, variable):
        t = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        v = list(np.atleast_1d(variable))
        return self.da.sel(time=t, variable=v)


def _assemble_cube(name: str, row: pd.Series, members: int, seed: int):
    """Read the t2m Zarr, crop to the CONUS box, write a cube matching CUBE SCHEMA."""
    zpath = P.zarr_path("fcn3", name)
    ds = xr.open_zarr(zpath)
    t2m = ds["t2m"]  # dims (ensemble, time, lead_time, lat, lon)
    init = np.datetime64(pd.Timestamp(row["init"]))
    lead = t2m["lead_time"].values                     # timedelta64
    valid = init + lead                                # absolute valid times
    t2m = t2m.isel(time=0)                             # single init time
    t2m = t2m.rename({"ensemble": "member", "lead_time": "time"})
    t2m = t2m.assign_coords(time=("time", valid), member=("member", np.arange(members)))
    # FCN3 lat is descending (90..-90) -> select the CONUS box then sort ascending.
    t2m = t2m.sel(lat=slice(50.0, 24.0), lon=slice(235.0, 294.0))
    t2m = t2m.sortby("lat")
    cube = t2m.to_dataset(name="2m_temperature")
    cube.attrs.update(
        case=name, kind=str(row["kind"]),
        init=pd.Timestamp(row["init"]).isoformat(),
        peak=pd.Timestamp(row["peak"]).isoformat(),
        model="fcn3", members=int(members), nsteps=int(P.NSTEPS), seed=int(seed),
    )
    cpath = P.cube_path("fcn3", name)
    tmp = cpath.with_suffix(".nc.tmp")
    cube.to_netcdf(tmp)
    os.replace(tmp, cpath)
    print(f"[infer]   wrote cube {cpath}  {dict(cube.sizes)} "
          f"lat[{float(cube.lat[0])}..{float(cube.lat[-1])}] "
          f"lon[{float(cube.lon[0])}..{float(cube.lon[-1])}]")
    return cpath


def stage_infer(case_sel, batch_size: int, clean_claims: bool):
    import torch
    from earth2studio.io import ZarrBackend
    from earth2studio.perturbation import Zero
    from earth2studio.run import ensemble
    from fcn3.model_cache import get_model              # shared build-once-load cache

    worker = os.environ.get("P90_WORKER", "0")
    host = socket.gethostname()
    # Explicit truthiness: bool("0") is True, so P90_KEEP_ZARR=0/false/no must disable it.
    keep_zarr = os.environ.get("P90_KEEP_ZARR", "").lower() not in ("", "0", "false", "no")

    # The DISCO CUDA kernel gates FCN3's bf16 path (earth2studio fcn3.py:382); without it
    # the model silently runs fp32. Fail loudly rather than produce off-precision cubes.
    try:
        from disco_helpers import optimized_kernels_is_available
        if not optimized_kernels_is_available() and os.environ.get("FCN3_ALLOW_FP32") != "1":
            raise SystemExit(
                "[infer] torch-harmonics DISCO CUDA extension MISSING -> FCN3 would run "
                "fp32.\n        Rebuild it (see HANDOFF.md) or set FCN3_ALLOW_FP32=1.")
    except ImportError:
        pass

    # Pin torch's intra-op threads to the launcher's per-worker core slice. Unpinned, each
    # of the 8 pool workers spawns one thread per NODE core and they thrash during model
    # setup (a3mega jobs 700/723). torch does not always read OMP_NUM_THREADS itself.
    _omp = os.environ.get("OMP_NUM_THREADS")
    if _omp:
        torch.set_num_threads(int(_omp))

    df = P.cases()
    if case_sel:
        df = df[df["case"] == case_sel]
    i, n = P.shard()
    # This shard's cases: POSITION in the filtered frame (0-based, stable) modulo n == i.
    shard_rows = [row for pos, (_, row) in enumerate(df.iterrows()) if pos % n == i]
    print(f"[infer] worker={worker} host={host} shard={i}/{n}: "
          f"{len(shard_rows)} of {len(df)} case(s), batch_size={batch_size}")

    claims = P.claims_dir("fcn3")
    if clean_claims:
        removed = 0
        for row in shard_rows:
            cf = claims / f"{row['case']}.claim"
            try:
                cf.unlink()
                removed += 1
            except FileNotFoundError:
                pass
        print(f"[infer] --clean-claims: removed {removed} stale claim(s) for this shard")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[infer] device={dev}; torch {torch.__version__} (cuda build {torch.version.cuda})")
    if dev == "cuda":
        print(f"[infer] GPU: {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

    model = None
    done = skipped = failed = 0
    failures = []
    for row in shard_rows:
        name = row["case"]
        idx = int(row["idx"])
        cpath = P.cube_path("fcn3", name)
        if cpath.exists():
            print(f"[infer] {name}: cube cached, skip")
            skipped += 1
            continue
        cf = claims / f"{name}.claim"
        if not _try_claim(cf):
            print(f"[infer] {name}: claimed by another worker, skip")
            skipped += 1
            continue
        try:
            icp = P.ic_path(name)
            if not icp.exists():
                raise SystemExit(f"IC not found: {icp} -- run --stage prep-ic first")
            if model is None:                          # once per worker: load the shared
                model = get_model(P.FCN3_CACHE / "model")   # pickle (built by fcn3 run 724)

            seed = P.seed_for(idx)
            model.set_rng(seed=seed, reset=True)
            init_iso = pd.Timestamp(row["init"]).strftime("%Y-%m-%dT%H:%M:%S")
            data = LocalIC(icp)

            zpath = P.zarr_path("fcn3", name)
            if zpath.exists():
                shutil.rmtree(zpath)
            io = ZarrBackend(
                file_name=str(zpath),
                chunks={"ensemble": 1, "time": 1, "lead_time": 1},
                backend_kwargs={"overwrite": True},
            )
            out_coords = {"variable": np.array(["t2m"])}  # store only t2m; crop CONUS after

            print(f"[infer] {name}: rolling {P.N_MEMBERS} members x {P.NSTEPS} steps "
                  f"(seed={seed}) ...")
            if dev == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            io = ensemble(
                [init_iso], P.NSTEPS, P.N_MEMBERS, model, data, io, Zero(),
                batch_size=batch_size, output_coords=out_coords, device=dev, verbose=True,
            )
            if dev == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else float("nan")
            print(f"[infer] {name}: rollout {elapsed:.1f} s ({elapsed / 60:.2f} min); "
                  f"peak GPU mem {peak_gb:.1f} GB")

            _assemble_cube(name, row, P.N_MEMBERS, seed)

            timing = {
                "case": name, "kind": str(row["kind"]), "model": "fcn3",
                "members": int(P.N_MEMBERS), "nsteps": int(P.NSTEPS),
                "batch_size": int(batch_size), "device": dev,
                "gpu": (torch.cuda.get_device_name(0) if dev == "cuda" else None),
                "seconds_total": round(elapsed, 2),
                "minutes_total": round(elapsed / 60, 3),
                "seconds_per_member": round(elapsed / P.N_MEMBERS, 2),
                "peak_gpu_mem_gb": (round(peak_gb, 2) if dev == "cuda" else None),
                "worker": worker, "hostname": host,
            }
            tp = P.timing_path("fcn3", name)
            tmp = tp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(timing, indent=2))
            os.replace(tmp, tp)
            print(f"[infer]   wrote {tp}")

            if not keep_zarr and zpath.exists():
                shutil.rmtree(zpath)
            done += 1
        except Exception as e:
            print(f"[infer] {name}: FAILED ({type(e).__name__}: {e})")
            traceback.print_exc()
            failures.append(name)
            failed += 1
            try:
                cf.unlink()                            # release claim so a retry can grab it
            except FileNotFoundError:
                pass

    print(f"[infer] worker={worker} summary: done={done} skipped={skipped} failed={failed}")
    if failures:
        print(f"[infer] failed cases: {', '.join(failures)}")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# compare (my-env/moe): metrics + figures (delegated).
# --------------------------------------------------------------------------- #
def stage_compare(model: str):
    from p90 import pmetrics, pcompare

    print(f"[compare] model={model}: computing metrics ...")
    pmetrics.compute_all(model)
    print(f"[compare] model={model}: building figures ...")
    pcompare.make_all(model)
    print(f"[compare] DONE model={model}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="p90 T2m-anomaly experiment driver")
    ap.add_argument("--stage", required=True,
                    choices=["prep-truth", "prep-ic", "infer", "compare"])
    ap.add_argument("--model", choices=list(P.MODELS), default="fcn3")
    ap.add_argument("--case", default=None, help="restrict to a single case name")
    ap.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("P90_FCN3_BATCH", 1)))  # FCN3 ~53 GiB/mem
                    # -> batch>1 OOMs an 80 GB H100 (hard capacity, not fragmentation)
    ap.add_argument("--clean-claims", action="store_true",
                    help="unlink claim files for THIS shard's cases before infer")
    args = ap.parse_args()

    if args.stage in ("prep-ic", "infer") and args.model == "gencast":
        raise SystemExit(
            "[p90] --model gencast has no prep-ic/infer here: GenCast members come from a\n"
            "separate adapter that must write cubes matching the CUBE SCHEMA into\n"
            f"cube_path('gencast', <case>) (see p90/pconfig.py). Once those cubes exist,\n"
            "`run_p90.py --stage compare --model gencast` scores them unchanged."
        )

    if args.stage == "prep-truth":
        P.truth_dir().mkdir(parents=True, exist_ok=True)
        P.ic_dir().mkdir(parents=True, exist_ok=True)
        stage_prep_truth()
        return

    P.ensure_dirs(args.model)

    if args.stage == "prep-ic":
        stage_prep_ic()
    elif args.stage == "infer":
        sys.exit(stage_infer(args.case, args.batch_size, args.clean_claims))
    elif args.stage == "compare":
        stage_compare(args.model)


if __name__ == "__main__":
    main()
