#!/usr/bin/env python
"""FourCastNet 3 ensemble forecast for the PNW Heat Dome 2021, matched to the
GenCast xres week-2 (0.25 deg) run for a head-to-head comparison.

Two stages, kept separate so the GPU job is pure compute (project rule:
"GPU nodes are for GPU compute ONLY"):

  prep  (login node, internet):  fetch the ERA5 0.25 deg global initial condition for
        the FCN3 variable set at the init time from ARCO-ERA5 and cache it to NetCDF.
  infer (GPU, offline):          load FCN3, replicate the cached IC across `--members`
        ensemble slots (FCN3's stochasticity is internal -> Zero() perturbation), roll
        `--nsteps` 6-hourly steps, and write a GenCast-style cube
        `2m_temperature(member, time, lat, lon)` cropped to the CONUS box. Records the
        wall time to generate the ensemble to a JSON sidecar.

The anomaly / PDF comparison is intentionally NOT done here: it runs in `my-env` via
`fcn3/compare_fcn3_gencast.py`, which reuses `xres.xmetrics.forecast_field` so the
week-mean T2m anomaly is computed by the exact same recipe as GenCast.

Env: the `fcn3` conda env (earth2studio[fcn3]); NOT `my-env`.
  /glade/derecho/scratch/exu/conda-envs/fcn3/bin/python fcn3/run_fcn3.py --stage prep
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# --- matched hyperparameters (from GenCast xres week-2 0p25 PNW cube attrs) ----------
EVENT = "PNW_HeatDome_2021"
INIT = "2021-06-14T00:00:00"     # 14-day (week-2) lead
PEAK = "2021-06-28T00:00:00"     # event peak; verification week = [peak-6d, peak]
NSTEPS = 56                      # 56 x 6h = 14 days
N_MEMBERS = 12

# CONUS verification box (gencast_s2s.config: LAT=slice(24,50), LON=slice(235,294)).
LAT_LO, LAT_HI = 24.0, 50.0
LON_LO, LON_HI = 235.0, 294.0

# Repo root derived from this file's location so the script is portable across machines
# (Derecho /glade... vs a3mega /home/ubuntu/...). fcn3/run_fcn3.py -> parents[1] is the root.
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "runs" / "fcn3" / EVENT
IC_PATH = OUTDIR / f"{EVENT}_ic.nc"
CUBE_PATH = OUTDIR / f"{EVENT}_fcn3_cube.nc"
TIMING_PATH = OUTDIR / f"{EVENT}_fcn3_timing.json"
ZARR_PATH = OUTDIR / f"{EVENT}_fcn3.zarr"


def _shard_members(total: int, nshards: int, shard: int):
    """(count, offset) of ensemble members assigned to `shard` of `nshards`.

    Members are split as evenly as possible (e.g. 24 over 8 -> 3 each; 12 over 8 ->
    [2,2,2,2,1,1,1,1]). Returns (0, 0) if this shard gets no members.
    """
    this = np.array_split(np.arange(total), nshards)[shard]
    if len(this) == 0:
        return 0, 0
    return int(len(this)), int(this[0])


def _shard_paths(shard: int, nshards: int):
    """(zarr, cube, timing) paths. Canonical paths for a single shard (nshards<=1);
    a per-shard set under shards/ otherwise, so the 8 GPU processes never collide."""
    if nshards <= 1:
        return ZARR_PATH, CUBE_PATH, TIMING_PATH
    sdir = OUTDIR / "shards"
    sdir.mkdir(parents=True, exist_ok=True)
    tag = f"shard{shard:02d}of{nshards:02d}"
    return (sdir / f"{EVENT}_{tag}.zarr",
            sdir / f"{EVENT}_{tag}_cube.nc",
            sdir / f"{EVENT}_{tag}_timing.json")

# Persistent caches on scratch so the login-node prep and the GPU infer share downloaded
# data + model weights (set before importing earth2studio / huggingface_hub).
_CACHE = ROOT / "runs" / "fcn3" / ".cache"
os.environ.setdefault("EARTH2STUDIO_CACHE", str(_CACHE / "e2s"))
os.environ.setdefault("HF_HOME", str(_CACHE / "hf"))


def _fcn3_variables():
    from earth2studio.models.px.fcn3 import VARIABLES
    return list(VARIABLES)


def stage_prep():
    """Fetch the FCN3 global IC at INIT from ARCO-ERA5 and cache to NetCDF."""
    from earth2studio.data import ARCO

    OUTDIR.mkdir(parents=True, exist_ok=True)
    variables = _fcn3_variables()
    print(f"[prep] fetching {len(variables)} FCN3 variables at {INIT} from ARCO-ERA5")
    data = ARCO(cache=True, verbose=True)
    da = data(time=[np.datetime64(INIT)], variable=variables)  # (time, variable, lat, lon)
    da = da.astype("float32")
    print(f"[prep] IC array {dict(da.sizes)}  lat[{float(da.lat[0])}..{float(da.lat[-1])}]"
          f" lon[{float(da.lon[0])}..{float(da.lon[-1])}]")
    tmp = IC_PATH.with_suffix(".nc.tmp")
    da.to_dataset(name="ic").to_netcdf(tmp)
    os.replace(tmp, IC_PATH)
    print(f"[prep] wrote {IC_PATH}")

    # Pre-download the FCN3 checkpoint files to the shared earth2studio cache (login node,
    # has internet) WITHOUT building the model -- the 710M-param CPU build OOM-kills the
    # login node. package.get() fetches each file to a local cache path; the actual model
    # is built later on the GPU node (which has the RAM + the GPU).
    print("[prep] pre-downloading FCN3 checkpoint files (no model build) ...")
    from earth2studio.models.px import FCN3
    package = FCN3.load_default_package()
    files = ["config.json", "metadata.json", "best_ckpt_mp0.tar", "orography.nc",
             "land_mask.nc", "global_means.npy", "global_stds.npy", "mins.npy", "maxs.npy"]
    for f in files:
        try:
            p = package.get(f)
            print(f"[prep]   cached {f} -> {p}")
        except Exception as e:
            print(f"[prep]   skip {f} ({type(e).__name__}: {e})")
    print(f"[prep] checkpoint cache: {os.environ['EARTH2STUDIO_CACHE']}")


class LocalIC:
    """earth2studio DataSource backed by the cached IC NetCDF (no network in infer)."""

    def __init__(self, path: Path):
        self.da = xr.open_dataset(path)["ic"].load()

    def __call__(self, time, variable):
        t = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        v = list(np.atleast_1d(variable))
        return self.da.sel(time=t, variable=v)


def stage_infer(members: int, nsteps: int, batch_size: int, seed: int,
                shard: int = 0, nshards: int = 1):
    """Roll this shard's slice of the ensemble on ONE visible GPU.

    With nshards>1 the caller pins each process to a distinct GPU (CUDA_VISIBLE_DEVICES)
    and passes shard=0..nshards-1; each shard rolls its own members with a distinct seed
    (independent draws) and writes a per-shard cube. `--stage assemble` merges them.
    """
    import torch
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px import FCN3
    from earth2studio.perturbation import Zero
    from earth2studio.run import ensemble

    if not IC_PATH.exists():
        raise SystemExit(f"IC not found: {IC_PATH} -- run --stage prep on the login node first")

    count, offset = _shard_members(members, nshards, shard)
    if count == 0:
        print(f"[infer] shard {shard}/{nshards}: no members for total={members} -- nothing to do")
        return
    zarr_path, cube_path, timing_path = _shard_paths(shard, nshards)
    seed_use = seed + shard  # distinct RNG stream per shard -> members don't duplicate

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[infer] shard {shard}/{nshards}: members {offset}..{offset+count-1} "
          f"({count} of {members}) seed={seed_use}")
    print(f"[infer] device={dev}; torch {torch.__version__} (cuda build {torch.version.cuda})")
    if dev == "cuda":
        print(f"[infer] GPU: {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

    print("[infer] loading FCN3 checkpoint ...")
    package = FCN3.load_default_package()
    model = FCN3.load_model(package)
    model.set_rng(seed=seed_use, reset=True)

    data = LocalIC(IC_PATH)
    if zarr_path.exists():
        import shutil
        shutil.rmtree(zarr_path)
    io = ZarrBackend(
        file_name=str(zarr_path),
        chunks={"ensemble": 1, "time": 1, "lead_time": 1},
        backend_kwargs={"overwrite": True},
    )
    # Store only t2m globally (72x smaller); crop CONUS in post-processing.
    out_coords = {"variable": np.array(["t2m"])}

    print(f"[infer] rolling {count} members x {nsteps} steps (batch_size={batch_size}) ...")
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    io = ensemble(
        [INIT], nsteps, count, model, data, io, Zero(),
        batch_size=batch_size, output_coords=out_coords, device=dev, verbose=True,
    )
    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else float("nan")
    print(f"[infer] DONE ensemble rollout in {elapsed:.1f} s "
          f"({elapsed/60:.2f} min); peak GPU mem {peak_gb:.1f} GB")

    _assemble_cube(count, cube_path, zarr_path, member_offset=offset)
    timing = {
        "event": EVENT, "model": "FourCastNet3", "members": count, "member_offset": offset,
        "total_members": members, "shard": shard, "nshards": nshards, "nsteps": nsteps,
        "batch_size": batch_size, "device": dev,
        "gpu": (torch.cuda.get_device_name(0) if dev == "cuda" else None),
        "seconds_total": round(elapsed, 2), "minutes_total": round(elapsed / 60, 3),
        "seconds_per_member": round(elapsed / count, 2),
        "peak_gpu_mem_gb": (round(peak_gb, 2) if dev == "cuda" else None),
    }
    timing_path.write_text(json.dumps(timing, indent=2))
    print(f"[infer] wrote {timing_path}\n{json.dumps(timing, indent=2)}")


def _assemble_cube(members: int, cube_path: Path, zarr_path: Path, member_offset: int = 0):
    """Read the t2m Zarr, crop to CONUS, build a GenCast-style cube NetCDF."""
    ds = xr.open_zarr(zarr_path)
    t2m = ds["t2m"]  # dims (ensemble, time, lead_time, lat, lon)
    # absolute valid time = init + lead_time
    init = np.datetime64(INIT)
    lead = t2m["lead_time"].values  # timedelta64
    valid = init + lead
    t2m = t2m.isel(time=0)  # single init time
    t2m = t2m.rename({"ensemble": "member", "lead_time": "time"})
    members_idx = np.arange(member_offset, member_offset + members)
    t2m = t2m.assign_coords(time=("time", valid), member=("member", members_idx))
    # crop CONUS; FCN3 lat is descending (90..-90) -> select then sort ascending
    t2m = t2m.sel(lat=slice(LAT_HI, LAT_LO), lon=slice(LON_LO, LON_HI))
    t2m = t2m.sortby("lat")
    cube = t2m.to_dataset(name="2m_temperature")
    cube.attrs.update(event=EVENT, init=INIT, peak=PEAK, model="FourCastNet3",
                      resolution="0p25", members=int(members))
    tmp = cube_path.with_suffix(".nc.tmp")
    cube.to_netcdf(tmp)
    os.replace(tmp, cube_path)
    print(f"[infer] wrote cube {cube_path}  {dict(cube.sizes)} "
          f"lat[{float(cube.lat[0])}..{float(cube.lat[-1])}] "
          f"lon[{float(cube.lon[0])}..{float(cube.lon[-1])}]")


def stage_assemble(members: int, nshards: int):
    """Merge per-shard cubes/timings (from the parallel GPU processes) into the canonical
    cube + timing sidecar. Wall time reported is the NODE wall (max over parallel shards)."""
    if nshards <= 1:
        print("[assemble] nshards<=1: nothing to merge (infer already wrote the canonical cube)")
        return
    cubes, secs, gpu, peak = [], [], None, []
    for s in range(nshards):
        count, _ = _shard_members(members, nshards, s)
        if count == 0:
            continue
        _, cube_path, timing_path = _shard_paths(s, nshards)
        if not cube_path.exists():
            raise SystemExit(f"[assemble] missing shard cube {cube_path} -- shard {s} did not finish")
        cubes.append(xr.open_dataset(cube_path))
        if timing_path.exists():
            t = json.loads(timing_path.read_text())
            secs.append(t.get("seconds_total", float("nan")))
            gpu = gpu or t.get("gpu")
            if t.get("peak_gpu_mem_gb") is not None:
                peak.append(t["peak_gpu_mem_gb"])

    full = xr.concat(cubes, dim="member").sortby("member")
    full = full.assign_coords(member=("member", np.arange(full.sizes["member"])))
    full.attrs.update(event=EVENT, init=INIT, peak=PEAK, model="FourCastNet3",
                      resolution="0p25", members=int(full.sizes["member"]),
                      parallel_shards=int(nshards))
    tmp = CUBE_PATH.with_suffix(".nc.tmp")
    full.to_netcdf(tmp)
    os.replace(tmp, CUBE_PATH)
    print(f"[assemble] wrote merged cube {CUBE_PATH}  {dict(full.sizes)}")

    node_wall = max(secs) if secs else float("nan")
    timing = {
        "event": EVENT, "model": "FourCastNet3", "members": int(full.sizes["member"]),
        "nshards": nshards, "parallel": True,
        "gpu": gpu, "peak_gpu_mem_gb": (round(max(peak), 2) if peak else None),
        "seconds_total": round(node_wall, 2), "minutes_total": round(node_wall / 60, 3),
        "seconds_per_shard": [round(x, 2) for x in secs],
        "note": "seconds_total is the node wall time (max over parallel shards), "
                "i.e. time to generate the full ensemble on one 8-GPU node.",
    }
    TIMING_PATH.write_text(json.dumps(timing, indent=2))
    print(f"[assemble] wrote {TIMING_PATH}\n{json.dumps(timing, indent=2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "infer", "assemble"], required=True)
    ap.add_argument("--members", type=int, default=int(os.environ.get("FCN3_MEMBERS", N_MEMBERS)))
    ap.add_argument("--nsteps", type=int, default=int(os.environ.get("FCN3_NSTEPS", NSTEPS)))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("FCN3_BATCH", 1)))
    ap.add_argument("--seed", type=int, default=333)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("FCN3_SHARD", 0)),
                    help="this process's shard index (0..nshards-1)")
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("FCN3_NSHARDS", 1)),
                    help="number of GPU processes splitting the ensemble (1 = single-GPU)")
    args = ap.parse_args()
    if args.stage == "prep":
        stage_prep()
    elif args.stage == "assemble":
        stage_assemble(args.members, args.nshards)
    else:
        stage_infer(args.members, args.nsteps, args.batch_size, args.seed,
                    shard=args.shard, nshards=args.nshards)


if __name__ == "__main__":
    main()
