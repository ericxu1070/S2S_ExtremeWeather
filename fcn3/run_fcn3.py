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

ROOT = Path("/glade/derecho/scratch/exu/S2S_ExtremeWeather")
OUTDIR = ROOT / "runs" / "fcn3" / EVENT
IC_PATH = OUTDIR / f"{EVENT}_ic.nc"
CUBE_PATH = OUTDIR / f"{EVENT}_fcn3_cube.nc"
TIMING_PATH = OUTDIR / f"{EVENT}_fcn3_timing.json"
ZARR_PATH = OUTDIR / f"{EVENT}_fcn3.zarr"

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


def stage_infer(members: int, nsteps: int, batch_size: int, seed: int):
    import torch
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px import FCN3
    from earth2studio.perturbation import Zero
    from earth2studio.run import ensemble

    if not IC_PATH.exists():
        raise SystemExit(f"IC not found: {IC_PATH} -- run --stage prep on the login node first")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[infer] device={dev}; torch {torch.__version__} (cuda build {torch.version.cuda})")
    if dev == "cuda":
        print(f"[infer] GPU: {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

    print("[infer] loading FCN3 checkpoint ...")
    package = FCN3.load_default_package()
    model = FCN3.load_model(package)
    model.set_rng(seed=seed, reset=True)

    data = LocalIC(IC_PATH)
    if ZARR_PATH.exists():
        import shutil
        shutil.rmtree(ZARR_PATH)
    io = ZarrBackend(
        file_name=str(ZARR_PATH),
        chunks={"ensemble": 1, "time": 1, "lead_time": 1},
        backend_kwargs={"overwrite": True},
    )
    # Store only t2m globally (72x smaller); crop CONUS in post-processing.
    out_coords = {"variable": np.array(["t2m"])}

    print(f"[infer] rolling {members} members x {nsteps} steps (batch_size={batch_size}) ...")
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    io = ensemble(
        [INIT], nsteps, members, model, data, io, Zero(),
        batch_size=batch_size, output_coords=out_coords, device=dev, verbose=True,
    )
    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else float("nan")
    print(f"[infer] DONE ensemble rollout in {elapsed:.1f} s "
          f"({elapsed/60:.2f} min); peak GPU mem {peak_gb:.1f} GB")

    _assemble_cube(members)
    timing = {
        "event": EVENT, "model": "FourCastNet3", "members": members, "nsteps": nsteps,
        "batch_size": batch_size, "device": dev,
        "gpu": (torch.cuda.get_device_name(0) if dev == "cuda" else None),
        "seconds_total": round(elapsed, 2), "minutes_total": round(elapsed / 60, 3),
        "seconds_per_member": round(elapsed / members, 2),
        "peak_gpu_mem_gb": (round(peak_gb, 2) if dev == "cuda" else None),
    }
    TIMING_PATH.write_text(json.dumps(timing, indent=2))
    print(f"[infer] wrote {TIMING_PATH}\n{json.dumps(timing, indent=2)}")


def _assemble_cube(members: int):
    """Read the t2m Zarr, crop to CONUS, build a GenCast-style cube NetCDF."""
    ds = xr.open_zarr(ZARR_PATH)
    t2m = ds["t2m"]  # dims (ensemble, time, lead_time, lat, lon)
    # absolute valid time = init + lead_time
    init = np.datetime64(INIT)
    lead = t2m["lead_time"].values  # timedelta64
    valid = init + lead
    t2m = t2m.isel(time=0)  # single init time
    t2m = t2m.rename({"ensemble": "member", "lead_time": "time"})
    t2m = t2m.assign_coords(time=("time", valid), member=("member", np.arange(members)))
    # crop CONUS; FCN3 lat is descending (90..-90) -> select then sort ascending
    t2m = t2m.sel(lat=slice(LAT_HI, LAT_LO), lon=slice(LON_LO, LON_HI))
    t2m = t2m.sortby("lat")
    cube = t2m.to_dataset(name="2m_temperature")
    cube.attrs.update(event=EVENT, init=INIT, peak=PEAK, model="FourCastNet3",
                      resolution="0p25", members=int(members))
    tmp = CUBE_PATH.with_suffix(".nc.tmp")
    cube.to_netcdf(tmp)
    os.replace(tmp, CUBE_PATH)
    print(f"[infer] wrote cube {CUBE_PATH}  {dict(cube.sizes)} "
          f"lat[{float(cube.lat[0])}..{float(cube.lat[-1])}] "
          f"lon[{float(cube.lon[0])}..{float(cube.lon[-1])}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "infer"], required=True)
    ap.add_argument("--members", type=int, default=int(os.environ.get("FCN3_MEMBERS", N_MEMBERS)))
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("FCN3_BATCH", 1)))
    ap.add_argument("--seed", type=int, default=333)
    args = ap.parse_args()
    if args.stage == "prep":
        stage_prep()
    else:
        stage_infer(args.members, args.nsteps, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
