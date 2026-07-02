"""Prep helpers: checkpoints, ERA5 observed truth (per metric), per-resolution inputs.

Everything lands on the shared NFS caches so the GPU ``infer`` stage reads only local
files. ERA5 truth lives in ``runs/observations/<event>_verif_<metric>.nc`` (resolution
independent, shared with the original pipeline for ``t2m_anom``).
"""
from __future__ import annotations

import os

import gcsfs
import xarray as xr

from gencast_s2s import config as C
from gencast_s2s import data as D

from . import xconfig as X
from . import xmetrics as XM


# --------------------------------------------------------------------------- #
# Checkpoints (both full <2019 models) + shared stats/statics/clim.
# --------------------------------------------------------------------------- #
_GCS = None


def _gcs():
    global _GCS
    if _GCS is None:
        _GCS = gcsfs.GCSFileSystem(token="anon")
    return _GCS


def _copy_from_gcs(src_url: str, dst_path) -> None:
    dst_path = str(dst_path)
    if os.path.exists(dst_path):
        return
    tmp = f"{dst_path}.tmp.{os.getpid()}"
    with _gcs().open(src_url, "rb") as fsrc, open(tmp, "wb") as fdst:
        while True:
            chunk = fsrc.read(64 << 20)
            if not chunk:
                break
            fdst.write(chunk)
    os.replace(tmp, dst_path)


def download_models() -> None:
    """Both checkpoints (0p25 + full 1p0) + normalization stats + statics + T2m clim."""
    C.ensure_dirs()
    for res in X.RES_ORDER:
        ckpt = X.params_for(res)
        print(f"[model] checkpoint ({res}): {ckpt}")
        _copy_from_gcs(C.PARAMS_DIR_GCS.replace("gs://", "") + ckpt, C.PARAMS_DIR / ckpt)
    for nm in C.STATS_FILES:
        print(f"[model] stats: {nm}")
        _copy_from_gcs(C.STATS_DIR_GCS.replace("gs://", "") + nm, C.STATS_DIR / nm)
    print("[model] statics (orography + land-sea mask)")
    D.load_statics()
    if not C.CLIM_FILE.exists():
        print("[clim] building CONUS 1990-2019 T2m climatology")
        clim = D.build_clim_conus()
        D._atomic_to_netcdf(clim.to_dataset(name="2m_temperature"), C.CLIM_FILE)
    else:
        print(f"[clim] cached: {C.CLIM_FILE.name}")
    print("[model] done")


# --------------------------------------------------------------------------- #
# ERA5 observed truth, one file per (event, metric). Headline metric per event plus
# its "free" secondary (u850_speed_max / tp_max12h) when applicable.
# --------------------------------------------------------------------------- #
def era5_truth_path(name: str, metric: str):
    return C.OBS_DIR / f"{name}_verif_{metric}.nc"


def metrics_for_event(name: str) -> list[str]:
    m = X.event_metric(name)
    out = [m]
    if m in XM.SECONDARY:
        out.append(XM.SECONDARY[m])
    return out


def build_era5_truth(overwrite: bool = False) -> None:
    C.ensure_dirs()
    for name, (peak, _metric) in X.events().items():
        for metric in metrics_for_event(name):
            f = era5_truth_path(name, metric)
            if f.exists() and not overwrite:
                print(f"[era5-truth] cached: {f.name}")
                continue
            da = XM.era5_field(peak, metric)
            D._atomic_to_netcdf(da.to_dataset(name=metric), f)
            print(f"[era5-truth] {name} [{metric}]: "
                  f"mean={float(da.mean()):+.3g} max={float(da.max()):+.3g} -> {f.name}")


# --------------------------------------------------------------------------- #
# Per-resolution model inputs (2 init frames at the model's native grid).
# --------------------------------------------------------------------------- #
def input_path(res: str, weeks: int, name: str):
    return X.inputs_dir(res, weeks) / f"{name}_inputs.nc"


def load_or_build_inputs(res: str, weeks: int, name: str, peak,
                         statics=None) -> xr.Dataset:
    f = input_path(res, weeks, name)
    if f.exists():
        return xr.open_dataset(f).load()
    X.ensure_dirs(res, weeks)
    rs = X.res_spec(res)
    ds = D.build_raw_inputs(peak, lead_days=C.lead_days_for(weeks), model="gencast",
                            statics=statics, verbose=True, res=rs["res"])
    D._atomic_to_netcdf(ds, f)
    return ds


def build_inputs(res: str, weeks: int) -> None:
    X.ensure_dirs(res, weeks)
    statics = D.load_statics()
    for name, (peak, _m) in X.events().items():
        f = input_path(res, weeks, name)
        if f.exists():
            print(f"[inputs {res} week{weeks}] cached: {f.name}")
            continue
        load_or_build_inputs(res, weeks, name, peak, statics=statics)
        print(f"[inputs {res} week{weeks}] {name}: init=peak-{C.lead_days_for(weeks)}d -> {f.name}")
