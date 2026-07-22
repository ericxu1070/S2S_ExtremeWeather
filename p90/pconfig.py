"""Static configuration for the p90 T2m-anomaly experiment.

Scores forecast skill of FourCastNet3 (and later GenCast, via the same cube
contract) on 90 CONUS cases: 45 p90 T2m-anomaly events (2022-2025) + 45 matched
no-event controls, at a 3-week lead (init = peak - 21 days), 50 ensemble members.
Verification is the week-mean T2m anomaly over [peak-6d, peak] on the 0.25deg CONUS
box vs ERA5 truth, anomalies vs the 1990-2019 climatology - the exact xres recipe.

Reuses ``gencast_s2s.config`` for the portable ROOT / env-overridable RUNS root, and
adds a separate output tree under ``runs/p90_t2m/`` so nothing else is touched. All
paths flow from here. Directories are created only by ``ensure_dirs`` (called by stage
code), never at import.

Committed inputs (do not modify): ``cases.csv`` (90 rows), ``cases_meta.json``
(thr_K, seed, lead_days), ``events_summary.csv``, ``make_cases.py``.

Env vars (all optional):
  P90_KIND        "event" | "control" -> keep only that kind
  P90_CASES_SEL   comma-separated case names -> keep only those
  P90_MAX_CASES   int -> keep first N (after the above filters)
  P90_N_MEMBERS   ensemble size (default 50)
  P90_SHARD       "i/n" work split; else SLURM_ARRAY_TASK_ID/COUNT; else 0/1
  P90_WORKER      free-form log tag (read by stage code, not here)
  P90_FCN3_BATCH  FCN3 member batch size (default 4; read by infer stage)
  P90_KEEP_ZARR   keep intermediate zarr (read by infer stage)
  GENCAST_RUNS    override the runs/ root (via gencast_s2s.config)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from gencast_s2s import config as C

# --------------------------------------------------------------------------- #
# Package location + shared roots
# --------------------------------------------------------------------------- #
PKG_DIR = Path(__file__).resolve().parent
ROOT = C.ROOT                                  # .../S2S_ExtremeWeather (portable)
P90_ROOT = C.RUNS / "p90_t2m"                  # this experiment's output tree

CASES_CSV = PKG_DIR / "cases.csv"
META = json.loads((PKG_DIR / "cases_meta.json").read_text())
THR_K = float(META["thr_K"])

# --------------------------------------------------------------------------- #
# Experiment constants (fixed by the contract)
# --------------------------------------------------------------------------- #
LEAD_DAYS = 21
STEP_H = 6
# 87*6h: step 84 = peak 00Z; 85/86/87 = peak 06/12/18Z so the peak DAY has all 4
# synoptic hours available for the daily-mean verification.
NSTEPS = 87
N_MEMBERS = int(os.environ.get("P90_N_MEMBERS", 50))
BASE_SEED = 333
HOURS = (0, 6, 12, 18)                         # synoptic hours for daily means
MODELS = ("fcn3", "gencast")

# Reuse the fcn3-test earth2studio/HF cache (checkpoint already fetched).
FCN3_CACHE = C.RUNS / "fcn3" / ".cache"


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def cases() -> pd.DataFrame:
    """Load cases.csv; parse peak/init as datetimes; tag original row order as ``idx``
    BEFORE any filtering, then filter (in order) by P90_KIND, P90_CASES_SEL,
    P90_MAX_CASES. The returned frame keeps the original DataFrame index."""
    df = pd.read_csv(CASES_CSV, parse_dates=["peak", "init"])
    df["idx"] = range(len(df))                 # original row order, stable across filters

    kind = os.environ.get("P90_KIND")
    if kind:
        df = df[df["kind"] == kind]

    sel = os.environ.get("P90_CASES_SEL")
    if sel:
        want = [s.strip() for s in sel.split(",") if s.strip()]
        df = df[df["case"].isin(want)]

    maxn = os.environ.get("P90_MAX_CASES")
    if maxn:
        df = df.iloc[:int(maxn)]

    return df


def all_cases() -> pd.DataFrame:
    """Full 90-row case frame with the ``idx`` column, IGNORING the P90_KIND /
    P90_CASES_SEL / P90_MAX_CASES env filters. The compare stage must score the whole
    45-event/45-control design, so it reads cases through here - a filter leaked from a
    smoke/shard run (a persistent export) would otherwise silently bias the skill report."""
    df = pd.read_csv(CASES_CSV, parse_dates=["peak", "init"])
    df["idx"] = range(len(df))
    return df


def case_row(name: str) -> pd.Series:
    """Full row (unfiltered lookup) for a single case by name."""
    df = pd.read_csv(CASES_CSV, parse_dates=["peak", "init"])
    df["idx"] = range(len(df))
    hit = df[df["case"] == name]
    if len(hit) == 0:
        raise KeyError(f"no case named {name!r}")
    return hit.iloc[0]


def seed_for(idx: int) -> int:
    """Deterministic per-case ensemble seed (BASE_SEED offset by original row order)."""
    return BASE_SEED + int(idx)


# --------------------------------------------------------------------------- #
# Output layout. Truth + IC are model-agnostic; everything else is per model.
# --------------------------------------------------------------------------- #
def ic_dir() -> Path:
    return P90_ROOT / "ic"


def ic_path(name: str) -> Path:
    return ic_dir() / f"{name}_ic.nc"


def truth_dir() -> Path:
    return P90_ROOT / "truth"


def truth_path(name: str) -> Path:
    return truth_dir() / f"{name}_verif_t2m_anom.nc"


def persist_path(name: str) -> Path:
    return truth_dir() / f"{name}_persist_t2m_anom.nc"


def model_dir(m: str) -> Path:
    return P90_ROOT / m


def cache_dir(m: str) -> Path:
    return model_dir(m) / "cache"


def cube_path(m: str, name: str) -> Path:
    return cache_dir(m) / f"{name}_cube.nc"


def zarr_dir(m: str) -> Path:
    return model_dir(m) / "zarr"


def zarr_path(m: str, name: str) -> Path:
    return zarr_dir(m) / f"{name}.zarr"


def claims_dir(m: str) -> Path:
    return cache_dir(m) / "claims"


def timing_dir(m: str) -> Path:
    return model_dir(m) / "timing"


def timing_path(m: str, name: str) -> Path:
    return timing_dir(m) / f"{name}_timing.json"


def scores_dir(m: str) -> Path:
    return model_dir(m) / "scores"


def fig_dir(m: str) -> Path:
    return C.ROOT / "figures" / "p90_t2m" / m


def ensure_dirs(m: str) -> None:
    """Create the full output tree for model ``m`` plus the shared truth/IC dirs.
    Only stage code calls this - never at import time."""
    for d in (ic_dir(), truth_dir(),
              model_dir(m), cache_dir(m), zarr_dir(m), claims_dir(m),
              timing_dir(m), scores_dir(m), fig_dir(m)):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Work sharding. Case c belongs to shard i iff its POSITION in the filtered frame
# (0-based, stable across workers) modulo n == i.
# --------------------------------------------------------------------------- #
def shard() -> tuple[int, int]:
    """(i, n): P90_SHARD="i/n" wins; else (SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT)
    if both are set; else (0, 1)."""
    env = os.environ.get("P90_SHARD")
    if env:
        i, n = env.split("/")
        return int(i), int(n)
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    cnt = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if tid is not None and cnt is not None:
        return int(tid), int(cnt)
    return 0, 1
