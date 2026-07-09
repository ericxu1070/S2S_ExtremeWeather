"""Prep stage: pull everything the GPU job needs onto local NFS, organized into folders.

    runs/models/params/<ckpt>.npz          GenCast 1.0deg Mini checkpoint
    runs/models/stats/*.nc                  normalization statistics
    runs/models/statics.nc                  orography + land-sea mask (global 0.25deg)
    runs/observations/<event>_verif_t2m_anom.nc   observed verification-week truth (shared)
    runs/week{N}/inputs/<event>_inputs.nc          init frames for that horizon

Run this once on a node with internet (the login node works); the a3mega job then reads
only local files. Re-running skips work that is already present. Writes are atomic, so the
three week2/3/4 jobs can also run this concurrently without corrupting shared files.
"""
from __future__ import annotations

import argparse
import os

import gcsfs

from . import config as C
from . import data as D

_GCS = None


def _gcs():
    global _GCS
    if _GCS is None:
        _GCS = gcsfs.GCSFileSystem(token="anon")
    return _GCS


def _copy_from_gcs(src_url: str, dst_path) -> None:
    """Atomically copy a GCS blob to a local path (skip if already present)."""
    dst_path = str(dst_path)
    if os.path.exists(dst_path):
        return
    tmp = f"{dst_path}.tmp.{os.getpid()}"
    fs = _gcs()
    with fs.open(src_url, "rb") as fsrc, open(tmp, "wb") as fdst:
        while True:
            chunk = fsrc.read(64 << 20)   # 64 MiB
            if not chunk:
                break
            fdst.write(chunk)
    os.replace(tmp, dst_path)


def download_model() -> None:
    """Checkpoint + normalization stats + static fields -> runs/models/."""
    C.ensure_dirs()
    ckpt = C.GENCAST_PARAMS
    print(f"[model] checkpoint: {ckpt}")
    _copy_from_gcs(C.PARAMS_DIR_GCS.replace("gs://", "") + ckpt, C.PARAMS_DIR / ckpt)
    for nm in C.STATS_FILES:
        print(f"[model] stats: {nm}")
        _copy_from_gcs(C.STATS_DIR_GCS.replace("gs://", "") + nm, C.STATS_DIR / nm)
    print("[model] statics (orography + land-sea mask)")
    D.load_statics()   # builds + caches statics.nc if missing
    download_climatology()
    print("[model] done")


def download_climatology() -> None:
    """1990-2019 T2m climatology cropped to CONUS -> runs/models/ (so infer is offline)."""
    C.ensure_dirs()
    if C.CLIM_FILE.exists():
        print(f"[clim] cached: {C.CLIM_FILE.name}")
        return
    clim = D.build_clim_conus()
    D._atomic_to_netcdf(clim.to_dataset(name="2m_temperature"), C.CLIM_FILE)
    print(f"[clim] saved CONUS 1990-2019 climatology -> {C.CLIM_FILE.name}")


def prepare_truth() -> None:
    """Observed verification-week T2m anomaly per event -> runs/observations/ (shared)."""
    C.ensure_dirs()
    for name, peak in C.EVENTS.items():
        f = C.OBS_DIR / f"{name}_verif_t2m_anom.nc"
        if f.exists():
            print(f"[truth] cached: {f.name}")
            continue
        da = D.true_verif_anom(peak)
        D._atomic_to_netcdf(da.to_dataset(name="t2m_anom"), f)
        print(f"[truth] {name}: mean={float(da.mean()):+.2f}K max={float(da.max()):+.2f}K -> {f.name}")


def prepare_hrrr_truth() -> None:
    """Observed HRRR week-mean T2m anomaly per event -> runs/observations/ (shared, offline).

    Best-effort: needs the local HRRR netCDF archive (C.HRRR_NC). Skipped with a notice
    if it is absent (e.g. a node without that mount) so prep still completes for ERA5.
    """
    if not C.HRRR_NC.exists():
        print(f"[hrrr-truth] HRRR archive not found at {C.HRRR_NC}; skipping HRRR overlay")
        return
    from . import hrrr
    hrrr.build_hrrr_truth()


def prepare_inputs(weeks: int, model: str = "gencast") -> None:
    """Init frames for this horizon -> runs/week{N}/inputs/ (per event)."""
    C.ensure_dirs(weeks)
    lead = C.lead_days_for(weeks)
    statics = D.load_statics()
    for name, peak in C.EVENTS.items():
        f = C.inputs_dir(weeks) / f"{name}_{model}_inputs.nc"
        if f.exists():
            print(f"[inputs week{weeks}] cached: {f.name}")
            continue
        ds = D.build_raw_inputs(peak, lead_days=lead, model=model, statics=statics, verbose=True)
        D._atomic_to_netcdf(ds, f)
        print(f"[inputs week{weeks}] {name}: init=peak-{lead}d -> {f.name}")


def prepare_all(weeks: int, model: str = "gencast") -> None:
    download_model()
    prepare_truth()
    prepare_hrrr_truth()
    prepare_inputs(weeks, model=model)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Download/prepare GenCast S2S data into runs/.")
    p.add_argument("--weeks", type=int, default=None, choices=C.VALID_WEEKS,
                   help="prepare init frames for this horizon (omit to prep all horizons)")
    p.add_argument("--what", choices=["all", "model", "clim", "truth", "hrrr", "inputs"],
                   default="all")
    args = p.parse_args(argv)

    if args.what in ("all", "model"):
        download_model()       # includes the climatology cache
    elif args.what == "clim":
        download_climatology()
    if args.what in ("all", "truth"):
        prepare_truth()
    if args.what in ("all", "hrrr"):
        prepare_hrrr_truth()
    if args.what in ("all", "inputs"):
        weeks_list = [args.weeks] if args.weeks else list(C.VALID_WEEKS)
        for w in weeks_list:
            prepare_inputs(w)


if __name__ == "__main__":
    main()
