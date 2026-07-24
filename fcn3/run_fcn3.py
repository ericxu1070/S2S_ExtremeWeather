#!/usr/bin/env python
"""FourCastNet 3 ensemble forecasts for the FCN3-vs-GenCast comparison.

Covers all six events in ``fcn3/fevents.py`` at week-3 (21 d lead), 0.25 deg, matched to
the cached GenCast xres week3 cubes (same init, same verification week, same member count).

Three stages, kept separate so the GPU job is pure compute (project rule: "GPU nodes are
for GPU compute ONLY"):

  prep      (login node, internet):  fetch the ERA5 0.25 deg GLOBAL initial condition for
            the full 72-variable FCN3 input set at each event's init, cache to NetCDF, and
            pre-download the FCN3 checkpoint files without building the model.
  infer     (GPU, offline):  for each event, replicate the cached IC across this shard's
            ensemble slots (FCN3's stochasticity is internal -> Zero() perturbation), roll
            84 x 6 h steps, and write a per-shard cube. Records wall time per (event, shard).
  assemble  (CPU):  merge the per-shard cubes into one canonical cube per event plus a
            node-wall timing sidecar.

The cubes use GENCAST'S variable names (``2m_temperature``, ``u_component_of_wind`` /
``v_component_of_wind`` with a ``level`` dim) so that ``xres.xmetrics.forecast_field``
scores FCN3 and GenCast through the exact same code path -- the only difference in the
figures is the forecast, not the post-processing. The anomaly/PDF comparison itself runs
in moe/my-env via ``fcn3/compare_fcn3_gencast.py``; this file never imports gencast_s2s.

Env: the `fcn3` conda env (earth2studio[fcn3]); NOT moe.
  python fcn3/run_fcn3.py --stage prep                      # login node
  FCN3_EVENTS=HurricaneIan_2022 python fcn3/run_fcn3.py --stage infer   # single event
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from fcn3 import fevents as F

# earth2studio / huggingface_hub read these at import time -> set before any e2s import.
os.environ.setdefault("EARTH2STUDIO_CACHE", str(F.FCN3_CACHE / "e2s"))
os.environ.setdefault("HF_HOME", str(F.FCN3_CACHE / "hf"))


# --------------------------------------------------------------------------- #
# Member sharding: shard s of n owns a contiguous slice of the ensemble for EVERY event.
# --------------------------------------------------------------------------- #
def _shard_members(total: int, nshards: int, shard: int) -> tuple[int, int]:
    """(count, offset) of ensemble members assigned to ``shard`` of ``nshards``.

    Split as evenly as possible (24 over 8 -> 3 each; 12 over 8 -> [2,2,2,2,1,1,1,1]).
    Returns (0, 0) when this shard gets no members."""
    this = np.array_split(np.arange(total), nshards)[shard]
    if len(this) == 0:
        return 0, 0
    return int(len(this)), int(this[0])


# --------------------------------------------------------------------------- #
# prep (login node, internet)
# --------------------------------------------------------------------------- #
def _fcn3_variables() -> list[str]:
    from earth2studio.models.px.fcn3 import VARIABLES
    return list(VARIABLES)


def check_disco_kernel(strict: bool = True) -> bool:
    """Is the torch-harmonics DISCO CUDA extension built into this env?

    This is NOT the cosmetic "will be slower" warning earth2studio prints. Its FCN3
    forward runs ``torch.autocast(bfloat16 if _cuda_extension_available else float32)``
    (`earth2studio/models/px/fcn3.py`), so a missing kernel silently downgrades the model
    to **fp32 on an unoptimized path**. That breaks two things this experiment depends on:
    the bf16 precision match to GenCast's 0.25 deg run, and the runtime comparison, which
    is the headline result. Newer torch-harmonics falls back instead of crashing, so
    nothing else would flag it -- hence this explicit check at prep time, on the login
    node, rather than discovering it after a GPU allocation has started.

    Rebuild with (login node, ~15 min):
        export CUDA_HOME=/home/ubuntu/miniconda3/envs/fcn3
        export PATH=$CUDA_HOME/bin:$PATH FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST=9.0
        pip install --no-build-isolation --no-cache-dir --force-reinstall --no-deps \
            torch-harmonics==0.9.1
    """
    try:
        from disco_helpers import optimized_kernels_is_available
        ok = bool(optimized_kernels_is_available())
    except Exception as e:
        print(f"[check] could not query torch-harmonics kernels ({type(e).__name__}: {e})")
        ok = False
    if ok:
        print("[check] torch-harmonics DISCO CUDA extension: PRESENT -> FCN3 runs bf16")
    else:
        msg = ("[check] torch-harmonics DISCO CUDA extension: MISSING -> FCN3 would run "
               "fp32 on a slow path,\n"
               "        breaking both the bf16 match to GenCast and the runtime comparison.")
        if strict:
            raise SystemExit(msg + "\n        See check_disco_kernel() for the rebuild recipe.")
        print(msg)
    return ok


def stage_prep() -> int:
    """Fetch each event's global FCN3 IC from ARCO-ERA5, then cache the checkpoint."""
    from earth2studio.data import ARCO

    F.ensure_dirs()
    events = F.selected()
    variables = _fcn3_variables()
    print(f"[prep] {len(events)} event(s), {len(variables)} FCN3 variables per IC, "
          f"week{F.WEEKS} ({F.LEAD_DAYS} d lead)")

    data = ARCO(cache=True, verbose=True)
    n_written = 0
    for k, ev in enumerate(events, 1):
        icp = F.ic_path(ev)
        if icp.exists():
            print(f"[prep] {k}/{len(events)} {ev.name}: IC cached, skip")
            continue
        print(f"[prep] {k}/{len(events)} {ev.name}: fetching IC at {ev.init.isoformat()} "
              f"(peak {ev.peak})")
        da = data(time=[np.datetime64(ev.init)], variable=variables)
        da = da.astype("float32")
        print(f"[prep]   IC {dict(da.sizes)} lat[{float(da.lat[0])}..{float(da.lat[-1])}] "
              f"lon[{float(da.lon[0])}..{float(da.lon[-1])}]")
        tmp = icp.with_suffix(".nc.tmp")
        da.to_dataset(name="ic").to_netcdf(tmp)
        os.replace(tmp, icp)
        n_written += 1
        print(f"[prep]   wrote {icp}")

    # Pre-download the checkpoint files WITHOUT building the model -- the 710M-param CPU
    # build OOM-kills the login node. package.get() fetches each file to the shared cache;
    # the model itself is built later on the GPU node.
    #
    # NOTE the weights live at `training_checkpoints/best_ckpt_mp0.tar` in the HF repo, not
    # at the repo root: asking for the bare name raises FileNotFoundError and would leave a
    # clean machine with NO weights, so the GPU job would either try to download them
    # (forbidden -- see CLAUDE.md) or die. Alternatives are tried in order and the first
    # hit wins; missing weights are a hard failure, not a warning.
    print("[prep] pre-downloading FCN3 checkpoint files (no model build) ...")
    from earth2studio.models.px import FCN3
    package = FCN3.load_default_package()

    def _get_first(*candidates: str) -> str | None:
        for c in candidates:
            try:
                return package.get(c)
            except Exception:
                continue
        return None

    for f in ("config.json", "metadata.json", "orography.nc", "land_mask.nc",
              "global_means.npy", "global_stds.npy", "mins.npy", "maxs.npy"):
        got = _get_first(f)
        print(f"[prep]   cached {f} -> {got}" if got else f"[prep]   MISSING {f}")

    ckpt = _get_first("training_checkpoints/best_ckpt_mp0.tar", "best_ckpt_mp0.tar")
    if ckpt is None:
        print("[prep] ERROR: could not fetch the FCN3 weights (best_ckpt_mp0.tar).\n"
              "       The GPU job would have to download them, which is forbidden.")
        return 1
    print(f"[prep]   cached weights -> {ckpt} ({os.path.getsize(ckpt)/1e9:.2f} GB)")

    # Fail here, on the login node, rather than after a GPU allocation has started.
    check_disco_kernel(strict=os.environ.get("FCN3_ALLOW_FP32") != "1")
    print(f"[prep] DONE: wrote {n_written} IC(s); cache {os.environ['EARTH2STUDIO_CACHE']}")
    return 0


# --------------------------------------------------------------------------- #
# infer (GPU, offline)
# --------------------------------------------------------------------------- #
class LocalIC:
    """earth2studio DataSource backed by the cached IC NetCDF (no network in infer)."""

    def __init__(self, path: Path):
        self.da = xr.open_dataset(path)["ic"].load()

    def __call__(self, time, variable):
        t = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        v = list(np.atleast_1d(variable))
        return self.da.sel(time=t, variable=v)


# --------------------------------------------------------------------------- #
# Build-once-load: constructing FCN3 recomputes the DISCO geometry tensors on CPU
# (~2 min in isolation on Derecho). Running one build per shard CONCURRENTLY contends on
# memory bandwidth -- a3mega job 723: 4 shards, still building at 17 min, GPUs idle. So
# the FIRST shard to grab a node lock builds the model and pickles it to NFS; every other
# shard (this job and later ones) just torch.loads it (~seconds). The whole built module
# is saved, not state_dict(), because the geometry tensors are registered persistent=False
# and would otherwise be dropped (verified: buffers + forward output identical after a
# save/load round-trip). Everything is best-effort with graceful fallback: if the wrapper
# does not pickle, or the env changed, a shard simply builds its own -- but still serially,
# under the same lock, so builds never overlap.
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
    """Atomically create a lock file. A lock held by a dead PID is stolen. Valid only
    between processes on ONE node (PID-liveness check)."""
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
            try:
                path.unlink()                          # dead owner -> steal and retry
            except FileNotFoundError:
                pass
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    return False


def _model_cache_tag() -> str:
    """Version key: any change here must invalidate a stored pickle. Covers the libraries
    whose internals a pickled nn.Module depends on, plus the FCN3 variable set."""
    import hashlib
    import earth2studio
    import torch
    import torch_harmonics
    parts = [torch.__version__, torch_harmonics.__version__,
             getattr(earth2studio, "__version__", "?"),
             ",".join(_fcn3_variables())]
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
    return f"{torch.__version__}_th{torch_harmonics.__version__}_{h}"


def _build_model():
    """Construct FCN3 from the cached package (CPU-bound). Timed -- it was the whole cost
    of the failed jobs 700/723, so it must be visible, not inferred."""
    from earth2studio.models.px import FCN3
    print("[infer] building FCN3 (CPU-bound DISCO precompute) ...", flush=True)
    t = time.perf_counter()
    model = FCN3.load_model(FCN3.load_default_package())
    print(f"[infer] model built in {(time.perf_counter()-t)/60:.2f} min", flush=True)
    return model


def _try_load_pickle(path: Path):
    import torch
    if not path.exists():
        return None
    try:
        t = time.perf_counter()
        # weights_only=False: this is our own file (built + written by _get_model on this
        # cluster), never an untrusted download. map to CPU; ensemble() moves it to the GPU.
        model = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[infer] loaded cached model {path.name} in {time.perf_counter()-t:.1f} s",
              flush=True)
        return model
    except Exception as e:
        print(f"[infer] cached model {path.name} unusable ({type(e).__name__}: {e}); "
              f"will rebuild", flush=True)
        return None


def _save_pickle(model, path: Path) -> None:
    import torch
    tmp = path.parent / f"{path.name}.tmp.{os.getpid()}"   # name has dots -> avoid with_suffix
    try:
        torch.save(model, tmp)
        os.replace(tmp, path)                          # atomic publish on NFS
        print(f"[infer] saved built model -> {path.name} "
              f"({path.stat().st_size/1e9:.2f} GB)", flush=True)
    except Exception as e:
        print(f"[infer] could not pickle model ({type(e).__name__}: {e}); "
              f"other shards will build their own", flush=True)
        try:
            tmp.unlink()
        except OSError:
            pass


def _get_model():
    """Return an FCN3 model, building it AT MOST once per node via a lock. Fast path is a
    pickle load; the lock only bites on a cold cache so builds never run concurrently."""
    cache = F.model_cache_path(_model_cache_tag())
    lock = F.model_build_lock()
    F.model_cache_dir().mkdir(parents=True, exist_ok=True)

    m = _try_load_pickle(cache)                        # 1. fast path, no lock
    if m is not None:
        return m

    disabled = os.environ.get("FCN3_MODEL_CACHE", "1") == "0"
    waited = 0.0
    while True:
        if _try_claim(lock):                           # 2. we build; others wait
            try:
                m = _try_load_pickle(cache)            # someone may have built while we waited
                if m is None:
                    m = _build_model()
                    if not disabled:
                        _save_pickle(m, cache)
                return m
            finally:
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
        # 3. another shard holds the lock. Poll for the pickle it is about to write; if
        #    saving is impossible the holder frees the lock after ~1 build (~2 min) and the
        #    next loop iteration lets THIS shard claim it -> builds stay serialized. The
        #    timeout is generous (covers N serial builds if pickling never works) so it
        #    fires only for a genuinely dead/stuck holder, not the normal wait.
        time.sleep(10)
        waited += 10
        m = _try_load_pickle(cache)
        if m is not None:
            return m
        if waited >= 2400:                             # ~40 min: builder truly stuck/dead
            print("[infer] waited 40 min for a peer build; building locally", flush=True)
            return _build_model()                      # last resort -- lock frees on exit


def _crop_conus(da: xr.DataArray) -> xr.DataArray:
    """Crop to the CONUS box and match the climatology grid exactly.

    FCN3 lat runs 90 -> -90, so the slice is (HI, LO) and the result is re-sorted
    ascending. Coordinates are cast to float32 because the 1990-2019 climatology and the
    GenCast cubes are float32 -- xarray aligns anomalies on coordinate VALUES, and a dtype
    mismatch there would silently produce an all-NaN anomaly field. Every 0.25 deg grid
    value is exactly representable, so the cast is lossless.
    """
    out = da.sel(lat=slice(F.LAT_HI, F.LAT_LO), lon=slice(F.LON_LO, F.LON_HI)).sortby("lat")
    return out.assign_coords(lat=out["lat"].values.astype("float32"),
                             lon=out["lon"].values.astype("float32"))


def _assemble_shard_cube(ev: F.Event, zarr_path: Path, out_path: Path,
                         count: int, offset: int, seed: int) -> Path:
    """Read this shard's Zarr, crop to CONUS, write a GenCast-schema cube."""
    ds = xr.open_zarr(zarr_path)
    init = np.datetime64(ev.init)
    members = np.arange(offset, offset + count)

    data_vars: dict[str, xr.DataArray] = {}
    by_level: dict[str, dict[int, xr.DataArray]] = {}
    for v in F.fcn3_vars(ev):
        gname, lev = F.FCN3_TO_GENCAST[v]
        da = ds[v]                                    # (ensemble, time, lead_time, lat, lon)
        valid = init + da["lead_time"].values         # absolute valid times
        da = da.isel(time=0).drop_vars("time", errors="ignore")
        da = da.rename({"ensemble": "member", "lead_time": "time"})
        da = da.assign_coords(time=("time", valid), member=("member", members))
        da = _crop_conus(da)
        if lev is None:
            data_vars[gname] = da
        else:
            by_level.setdefault(gname, {})[lev] = da

    for gname, per_lev in by_level.items():
        levs = sorted(per_lev)
        stacked = xr.concat([per_lev[l] for l in levs], dim="level")
        data_vars[gname] = stacked.assign_coords(level=("level", np.array(levs)))

    cube = xr.Dataset(data_vars)
    cube.attrs.update(event=ev.name, family=ev.family, metric=ev.metric,
                      init=ev.init.isoformat(), peak=ev.peak, model="FourCastNet3",
                      resolution=F.RES, weeks=F.WEEKS, lead_days=F.LEAD_DAYS,
                      members=int(count), member_offset=int(offset),
                      nsteps=int(F.NSTEPS), step_h=F.STEP_H, seed=int(seed))
    tmp = out_path.with_suffix(".nc.tmp")
    cube.to_netcdf(tmp)
    os.replace(tmp, out_path)
    ds.close()
    print(f"[infer]   wrote {out_path.name}  {dict(cube.sizes)} "
          f"vars={list(cube.data_vars)} "
          f"lat[{float(cube.lat[0])}..{float(cube.lat[-1])}] "
          f"lon[{float(cube.lon[0])}..{float(cube.lon[-1])}]")
    return out_path


def stage_infer(shard: int, nshards: int, members: int, batch_size: int,
                keep_zarr: bool) -> int:
    """Roll this shard's members for every selected event on ONE visible GPU."""
    import torch
    from earth2studio.io import ZarrBackend
    from earth2studio.perturbation import Zero
    from earth2studio.run import ensemble

    F.ensure_dirs()
    check_disco_kernel(strict=os.environ.get("FCN3_ALLOW_FP32") != "1")
    events = F.selected()
    count, offset = _shard_members(members, nshards, shard)
    host = socket.gethostname()
    if count == 0:
        print(f"[infer] shard {shard}/{nshards}: no members for total={members}; nothing to do")
        return 0

    # Honour the launcher's per-shard core slice. Without this each of the N shard
    # processes spawns one OpenMP thread per CORE ON THE NODE, so N shards oversubscribe
    # the box N-fold and FCN3's CPU-bound model construction thrashes (job 700: 8 shards,
    # ~1856 threads on 208 cores, GPUs idle 80+ min). torch reads OMP_NUM_THREADS only at
    # some layers, so set its intra-op pool explicitly too.
    _omp = os.environ.get("OMP_NUM_THREADS")
    if _omp:
        torch.set_num_threads(int(_omp))
    print(f"[infer] CPU threads: torch={torch.get_num_threads()} "
          f"OMP_NUM_THREADS={_omp or '<unset>'}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[infer] host={host} shard={shard}/{nshards}: members {offset}..{offset+count-1} "
          f"({count} of {members}) over {len(events)} event(s)")
    print(f"[infer] device={dev}; torch {torch.__version__} (cuda build {torch.version.cuda})")
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"[infer] GPU: {torch.cuda.get_device_name(0)}  {p.total_memory/1e9:.0f} GB")

    model = None                                      # lazy: skip load if all cached
    done = skipped = failed = 0
    failures: list[str] = []

    for ev in events:
        cpath = F.shard_cube_path(ev, shard, nshards)
        if cpath.exists():
            print(f"[infer] {ev.name}: shard cube cached, skip")
            skipped += 1
            continue
        icp = F.ic_path(ev)
        if not icp.exists():
            print(f"[infer] {ev.name}: FAILED -- IC not found: {icp}\n"
                  f"         run `python fcn3/run_fcn3.py --stage prep` on the login node")
            failures.append(ev.name)
            failed += 1
            continue
        try:
            if model is None:
                model = _get_model()                  # load-once-per-node (see _get_model)

            seed = F.seed_for(ev) + shard             # distinct RNG stream per (event, shard)
            model.set_rng(seed=seed, reset=True)
            data = LocalIC(icp)

            zpath = F.shard_zarr_path(ev, shard, nshards)
            if zpath.exists():
                shutil.rmtree(zpath)
            io = ZarrBackend(file_name=str(zpath),
                             chunks={"ensemble": 1, "time": 1, "lead_time": 1},
                             backend_kwargs={"overwrite": True})
            out_vars = F.fcn3_vars(ev)
            out_coords = {"variable": np.array(out_vars)}

            print(f"[infer] {ev.name}: rolling {count} member(s) x {F.NSTEPS} steps "
                  f"(init {ev.init.isoformat()}, vars={out_vars}, seed={seed}) ...")
            if dev == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            io = ensemble([ev.init.strftime("%Y-%m-%dT%H:%M:%S")], F.NSTEPS, count,
                          model, data, io, Zero(), batch_size=batch_size,
                          output_coords=out_coords, device=dev, verbose=True)
            if dev == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else float("nan")
            print(f"[infer] {ev.name}: rollout {elapsed:.1f} s ({elapsed/60:.2f} min); "
                  f"peak GPU mem {peak_gb:.1f} GB")

            _assemble_shard_cube(ev, zpath, cpath, count, offset, seed)

            timing = {
                "event": ev.name, "family": ev.family, "model": "fcn3",
                "members": int(count), "member_offset": int(offset),
                "total_members": int(members), "shard": int(shard), "nshards": int(nshards),
                "nsteps": int(F.NSTEPS), "weeks": int(F.WEEKS), "batch_size": int(batch_size),
                "device": dev,
                "gpu": (torch.cuda.get_device_name(0) if dev == "cuda" else None),
                "seconds_total": round(elapsed, 2),
                "minutes_total": round(elapsed / 60, 3),
                "seconds_per_member": round(elapsed / count, 2),
                "peak_gpu_mem_gb": (round(peak_gb, 2) if dev == "cuda" else None),
                "hostname": host,
            }
            tp = F.shard_timing_path(ev, shard, nshards)
            tmp = tp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(timing, indent=2))
            os.replace(tmp, tp)

            if not keep_zarr and zpath.exists():
                shutil.rmtree(zpath)                  # ~1 GB/var/shard-event otherwise
            done += 1
        except Exception as e:
            print(f"[infer] {ev.name}: FAILED ({type(e).__name__}: {e})")
            traceback.print_exc()
            failures.append(ev.name)
            failed += 1

    print(f"[infer] shard {shard}/{nshards} summary: done={done} skipped={skipped} "
          f"failed={failed}")
    if failures:
        print(f"[infer] failed events: {', '.join(failures)}")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# assemble (CPU)
# --------------------------------------------------------------------------- #
def stage_assemble(members: int, nshards: int) -> int:
    """Merge per-shard cubes into one cube per event + a node-wall timing sidecar."""
    F.ensure_dirs()
    events = F.selected()
    missing_any = False

    for ev in events:
        cpath = F.cube_path(ev)
        if cpath.exists():
            print(f"[assemble] {ev.name}: cube cached, skip")
            continue
        parts, secs, gpu, peak = [], [], None, []
        missing = []
        for s in range(nshards):
            count, _ = _shard_members(members, nshards, s)
            if count == 0:
                continue
            sc = F.shard_cube_path(ev, s, nshards)
            if not sc.exists():
                missing.append(s)
                continue
            parts.append(xr.open_dataset(sc))
            st = F.shard_timing_path(ev, s, nshards)
            if st.exists():
                t = json.loads(st.read_text())
                secs.append(t.get("seconds_total", float("nan")))
                gpu = gpu or t.get("gpu")
                if t.get("peak_gpu_mem_gb") is not None:
                    peak.append(t["peak_gpu_mem_gb"])
        if missing:
            print(f"[assemble] {ev.name}: SKIP -- shard cube(s) missing: {missing}")
            missing_any = True
            for p in parts:
                p.close()
            continue

        full = xr.concat(parts, dim="member").sortby("member")
        full = full.assign_coords(member=("member", np.arange(full.sizes["member"])))
        full.attrs.update(event=ev.name, family=ev.family, metric=ev.metric,
                          init=ev.init.isoformat(), peak=ev.peak, model="FourCastNet3",
                          resolution=F.RES, weeks=F.WEEKS, lead_days=F.LEAD_DAYS,
                          members=int(full.sizes["member"]), nsteps=int(F.NSTEPS),
                          step_h=F.STEP_H, parallel_shards=int(nshards))
        tmp = cpath.with_suffix(".nc.tmp")
        full.to_netcdf(tmp)
        os.replace(tmp, cpath)
        for p in parts:
            p.close()
        print(f"[assemble] {ev.name}: wrote {cpath}  {dict(full.sizes)}")

        node_wall = max(secs) if secs else float("nan")
        timing = {
            "event": ev.name, "family": ev.family, "model": "fcn3",
            "members": int(full.sizes["member"]), "nshards": int(nshards), "parallel": True,
            "weeks": int(F.WEEKS), "nsteps": int(F.NSTEPS), "gpu": gpu,
            "peak_gpu_mem_gb": (round(max(peak), 2) if peak else None),
            "seconds_total": round(node_wall, 2),
            "minutes_total": round(node_wall / 60, 3),
            "seconds_per_shard": [round(x, 2) for x in secs],
            "gpu_seconds": round(float(np.nansum(secs)), 2) if secs else None,
            "note": "seconds_total is the node wall time for this event (max over the "
                    "parallel shards), i.e. time to generate the full ensemble on one "
                    "8-GPU node; gpu_seconds is the summed single-GPU cost.",
        }
        tp = F.timing_path(ev)
        tmp = tp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(timing, indent=2))
        os.replace(tmp, tp)
        print(f"[assemble] {ev.name}: {timing['minutes_total']:.2f} min node wall "
              f"({timing['gpu_seconds']} GPU-s over {nshards} shards)")

    return 1 if missing_any else 0


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["prep", "infer", "assemble", "plan"], required=True)
    ap.add_argument("--members", type=int, default=F.N_MEMBERS)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("FCN3_BATCH", 1)))
    ap.add_argument("--shard", type=int, default=int(os.environ.get("FCN3_SHARD", 0)),
                    help="this process's shard index (0..nshards-1)")
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("FCN3_NSHARDS", 1)),
                    help="number of GPU processes splitting the ensemble (1 = single-GPU)")
    ap.add_argument("--keep-zarr", action="store_true",
                    help="keep the intermediate global Zarr (~1 GB per var per shard-event)")
    args = ap.parse_args()

    if args.stage == "plan":
        print(F.describe())
        return
    if args.stage == "prep":
        sys.exit(stage_prep())
    if args.stage == "assemble":
        sys.exit(stage_assemble(args.members, args.nshards))
    sys.exit(stage_infer(args.shard, args.nshards, args.members, args.batch_size,
                         args.keep_zarr))


if __name__ == "__main__":
    main()
