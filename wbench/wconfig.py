"""What the walker benchmark runs, where its artifacts live, and what it is compared to.

Stdlib only at import time (plus ``aires.aconfig``/``fcn3.fevents``, which are written to
import in both envs), so every module - GenCast side, FCN3 side, the report - reads the
same numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from aires import aconfig as A
from fcn3 import fevents as F

# --------------------------------------------------------------------------- #
# The one atmosphere every config starts from: PNW heat dome, week-3 init
# (2021-06-07 00Z). GenCast reads the xres init frames (0.25 deg and the stride-4 1.0
# deg copy xres builds), FCN3 reads its own ERA5 IC at the same instant. Timing does not
# depend on the weather, but starting from a real state keeps the rollouts physical, so
# a NaN or a blow-up in a benchmark is a finding rather than an artifact of noise input.
# --------------------------------------------------------------------------- #
EVENT = os.environ.get("WB_EVENT", "PNW_HeatDome_2021")

# One AI+RES segment is 3 days, the resampling interval of every production run.
SEG_DAYS = A.RES_SEG_STEPS * A.WALKER_STEP_H / 24          # 3.0
BASE_SEED = int(os.environ.get("WB_BASE_SEED", 20261015))


@dataclass(frozen=True)
class ModelConfig:
    name: str
    env: str               # which conda env runs it: "gencast" or "fcn3"
    step_h: int
    precision: str
    label: str

    @property
    def seg_steps(self) -> int:
        n = SEG_DAYS * 24 / self.step_h
        if n != int(n):
            raise ValueError(f"{self.name}: {SEG_DAYS} d is not a whole number of "
                             f"{self.step_h} h steps")
        return int(n)

    @property
    def steps_per_day(self) -> int:
        return 24 // self.step_h


CONFIGS = {
    "gencast_0p25": ModelConfig("gencast_0p25", "gencast", 12, "bf16",
                                "GenCast 0.25 deg (production walker)"),
    "gencast_1p0":  ModelConfig("gencast_1p0", "gencast", 12, "fp32",
                                "GenCast 1.0 deg"),
    "fcn3_e2s":     ModelConfig("fcn3_e2s", "fcn3", F.STEP_H, "bf16-autocast",
                                "FCN3 via earth2studio"),
    "fcn3_native":  ModelConfig("fcn3_native", "fcn3", F.STEP_H, "bf16-autocast",
                                "FCN3 native batched"),
}
ORDER = ("gencast_0p25", "gencast_1p0", "fcn3_e2s", "fcn3_native")
BASELINE = "gencast_0p25"

# GenCast res tag per config, through the SAME registry the walker uses
# (`aires.aconfig.walker_model_kwargs`) - never a hand-built checkpoint name.
GENCAST_RES = {"gencast_0p25": "0p25", "gencast_1p0": "1p0"}

# The 1.0 deg grid. `aires.walker.check_state` asserts the 0.25 deg grid on every read;
# the 1.0 deg benchmark widens that one assertion for its own process and nothing else.
GRID_1P0 = (181, 360)

# --------------------------------------------------------------------------- #
# Benchmark sizing. ~1.5 h on one 8xH100 node with the defaults (vm/README.md has the
# phase-by-phase budget). WB_QUICK=1 drops to one timed segment and a short sweep.
# --------------------------------------------------------------------------- #
QUICK = os.environ.get("WB_QUICK") == "1"
TIMED_SEGMENTS = int(os.environ.get("WB_SEGMENTS", 1 if QUICK else 3))
SWEEP_BATCHES = tuple(int(b) for b in os.environ.get(
    "WB_SWEEP", "1,2,4" if QUICK else "1,2,3,4,6,8").split(","))
SWEEP_WARM_STEPS = 2
SWEEP_TIMED_STEPS = int(os.environ.get("WB_SWEEP_STEPS", 4 if QUICK else 8))
# A batch size is worth running at node scale only if it beats B=1 by this much in
# walker-steps per second; otherwise phase 2 skips it rather than spend GPU time
# confirming a null.
BATCH_GAIN_MIN = 1.10

# --------------------------------------------------------------------------- #
# Paths. Everything lands under runs/wbench/<run_id>/; nothing else is written.
# --------------------------------------------------------------------------- #
RUNS = A.RUNS
WB_ROOT = Path(os.environ.get("WB_OUT", RUNS / "wbench"))


def run_dir(run_id: str) -> Path:
    return WB_ROOT / run_id


def results_dir(run: Path) -> Path:
    return Path(run) / "results"


def scratch_dir(run: Path) -> Path:
    """Walker states written by the segment benchmarks; pruned as they go."""
    return Path(run) / "scratch"


def gencast_init_path(res: str) -> Path:
    return A.gencast_inputs_path(EVENT, res=res, weeks=3)


def fcn3_ic_path() -> Path:
    return F.week_dir() / "ic" / f"{EVENT}_ic.nc"


# --------------------------------------------------------------------------- #
# a3mega reference numbers - measured, not assumed. Restated here so the report can put
# a Nebius number next to the a3mega one; `tests/test_wconfig.py` pins each against its
# source so the two cannot drift.
# --------------------------------------------------------------------------- #
REF_A3MEGA = {
    # aires/HANDOFF.md "The ragged tail costs nothing": job 1189, a free-running walker
    # pool, 28.6 s per GenCast step + 26.3 s fixed per segment.
    "gencast_0p25_free_s_per_step": 28.6,
    "gencast_0p25_free_seg_overhead_s": 26.3,
    # aires/run_aires.py SECONDS_PER_GENCAST_STEP: job 1172, a production DMC leg with 8
    # shards writing N=64 restart states at once.
    "gencast_0p25_prod_s_per_step": 52.0,
    # aires/run_aires.py SECONDS_PER_FCN3_STEP: job 1161, FCN3 score member-step.
    "fcn3_s_per_member_step": 1.40,
    # aires/HANDOFF.md: ~5 min one-time XLA compile on one GPU (job 1154); ~15 min with
    # 8 concurrent workers (Gate 3). The 0.25 deg executable is 4.17 GB, over the JAX
    # persistent cache's 2 GiB limit, so every process pays it.
    "gencast_0p25_compile_s_solo": 300.0,
    "gencast_0p25_compile_s_8way": 900.0,
}

# --------------------------------------------------------------------------- #
# Layer 3: the AI+RES case the projection prices. Read from the production config, so a
# change to the schedule there changes the projection here.
# --------------------------------------------------------------------------- #
CASE_N_WALKERS = tuple(int(n) for n in os.environ.get("WB_CASE_N", "32,64").split(","))
CASE_HORIZON_DAYS = A.RES_HORIZON_DAYS                    # 21 d
CASE_LEAD_DAYS = A.RES_LEAD_DAYS                          # resampling times
CASE_C = A.RES_C                                          # C_k; 0 => no score bought
CASE_M_MEMBERS = A.RES_M_MEMBERS                          # FCN3 members per score
CASE_LEGS = len(A.RES_LEAD_DAYS) + 1                      # one more leg than resamplings
GPUS_PER_NODE = 8
