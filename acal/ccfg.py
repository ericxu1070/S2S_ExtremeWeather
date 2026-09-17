"""Static configuration for the AI+RES calibration campaign: dates, boxes, thresholds.

Everything the case slate depends on lives here, so `cases.csv` can be regenerated from
this module alone and any change to the population is a diff in one file.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from gencast_s2s import config as C
from aires import aindex as AI

# --------------------------------------------------------------------------- #
# Paths. Own tree, disjoint from runs/aires/ - a calibration case is not a frozen
# production run and must never land in the same directory as one.
# --------------------------------------------------------------------------- #
ROOT = C.ROOT
RUNS = Path(os.environ.get("GENCAST_RUNS", ROOT / "runs"))
ACAL_ROOT = RUNS / "acal"
FIG_ROOT = ROOT / "figures" / "acal"

INDEX_CUBE = ACAL_ROOT / "era5_index_12h.nc"     # 12-hourly CONUS T2m anomaly, 0.25 deg
CASES_CSV = ACAL_ROOT / "cases.csv"
CASES_META = ACAL_ROOT / "cases_meta.json"

# --------------------------------------------------------------------------- #
# The scan window.
#
# 2021-01-01 is the first PEAK date. The walker is initialised 21 days earlier, so the
# ERA5 read actually starts 2020-12-05 (peak - 21 d - 6 d of verification window).
# The end is bounded by ARCO-ERA5's own coverage; `acases.py` clips to what is readable
# rather than trusting this constant.
# --------------------------------------------------------------------------- #
PEAK_START = os.environ.get("ACAL_PEAK_START", "2021-01-01")
PEAK_END = os.environ.get("ACAL_PEAK_END", "2026-08-31")

# A_L's own sampling: [peak-6d, peak] inclusive at a 12 h step = 13 frames.
# These are NOT free parameters - they mirror aires.aindex.check_window, and
# tests/test_acases.py pins them against it.
HOURS = (0, 12)
WINDOW_DAYS = 6
N_WINDOW_FRAMES = WINDOW_DAYS * 24 // 12 + 1      # 13

# --------------------------------------------------------------------------- #
# The box grid.
#
# AI+RES scores a 6x6 degree window (aires/aindex.py). A calibration slate cannot use
# the per-event hand-picked boxes - those were chosen to sit on the event, which is the
# selection-on-outcome the whole campaign is trying to avoid - so the scan tiles the
# CONUS crop on a fixed lattice and lets the rule pick which window an event lands in.
#
# BOX_STEP is the lattice spacing, not the box size. 2 degrees oversamples on purpose:
# a 6 degree box on a 6 degree lattice can split a dome across four cells and register
# none of them, and the declustering below removes the redundancy the overlap creates.
# --------------------------------------------------------------------------- #
BOX_SIZE = float(os.environ.get("ACAL_BOX_SIZE", 6.0))
BOX_STEP = float(os.environ.get("ACAL_BOX_STEP", 2.0))

CONUS_LAT = AI.CONUS_LAT          # (24.0, 50.0)
CONUS_LON = AI.CONUS_LON          # (235.0, 294.0)


def box_lattice() -> list[tuple[float, float, float, float]]:
    """`[(lat_lo, lat_hi, lon_lo, lon_hi), ...]` - every BOX_SIZE window inside CONUS."""
    lat0 = np.arange(CONUS_LAT[0], CONUS_LAT[1] - BOX_SIZE + 1e-9, BOX_STEP)
    lon0 = np.arange(CONUS_LON[0], CONUS_LON[1] - BOX_SIZE + 1e-9, BOX_STEP)
    return [(float(a), float(a + BOX_SIZE), float(o), float(o + BOX_SIZE))
            for a in lat0 for o in lon0]


def box_of(lat_lo: float, lon_lo: float) -> AI.Box:
    """The `aires.aindex.Box` for a lattice cell, named the way cases.csv names it."""
    return AI.Box(box_name(lat_lo, lon_lo),
                  (lat_lo, lat_lo + BOX_SIZE), (lon_lo, lon_lo + BOX_SIZE),
                  "acal lattice cell")


def box_name(lat_lo: float, lon_lo: float) -> str:
    """Stable, filesystem-safe box id: `b32N_248E` is lat 32-38 N, lon 248-254 E."""
    return f"b{int(round(lat_lo)):02d}N_{int(round(lon_lo)):03d}E"


# --------------------------------------------------------------------------- #
# The selection rule.
#
# The user's rule: every +/-2, +/-3 and +/-4 K anomaly, hot and cold, 2021-2026.
# A crossing of +4 K is also a crossing of +3 and +2, so a case is filed under the
# DEEPEST rung it reaches and `rung` is that number. Nothing is double counted.
# --------------------------------------------------------------------------- #
RUNGS = (2.0, 3.0, 4.0)

# Declustering. A heat dome is one event, not the 40 (box, day) pairs that cross a
# threshold inside it. Two crossings belong to the same event when their peaks are
# within DECLUSTER_DAYS of each other AND their boxes overlap by more than
# DECLUSTER_OVERLAP of a box area; the representative is the (box, day) with the
# largest |A_L|, which is exactly aires/aindex.py's own box-selection rule applied
# without the "inside the region it is named for" clause (there is no name here).
DECLUSTER_DAYS = int(os.environ.get("ACAL_DECLUSTER_DAYS", 10))
DECLUSTER_OVERLAP = float(os.environ.get("ACAL_DECLUSTER_OVERLAP", 0.25))

# --------------------------------------------------------------------------- #
# The AI+RES run each case gets. N=32 halves the frozen production's walker count:
# for a calibration campaign twice the cases at half the walkers is the better trade,
# and `acal/HANDOFF.md` records the variance cost.
# --------------------------------------------------------------------------- #
RES_N_WALKERS = int(os.environ.get("ACAL_N_WALKERS", 32))
RES_M_MEMBERS = int(os.environ.get("ACAL_M_MEMBERS", 6))
LEAD_DAYS = int(os.environ.get("ACAL_LEAD_DAYS", 21))     # week-3 init, as in aires/


def case_dir(case_id: str) -> Path:
    return ACAL_ROOT / "cases" / case_id


def ensure_dirs() -> None:
    for d in (ACAL_ROOT, ACAL_ROOT / "cases", FIG_ROOT):
        d.mkdir(parents=True, exist_ok=True)
