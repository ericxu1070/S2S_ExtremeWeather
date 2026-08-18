#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# First GPU exercise of aires/walker.py: roll ONE short segment and prove the round trip.
#
# Everything CPU-checkable about the walker already passes (aires/tests/test_walker.py,
# `--check`), so what is unproven here is only what needs the model: that a segment rolled
# from a global 2-frame state produces a state that is itself a valid restart point, and
# that the adapter can turn it into an FCN3 initial condition. That is the whole
# walker -> scorer handoff, and it is worth 2 steps of GPU time before committing 8-16
# free-running walkers to Gate 3.
#
# Two steps only (~28 s/step at 0.25 deg on an H100, plus a one-time ~5 min XLA compile).
# One GPU: a walker segment is serial by construction -- the 0.25 deg checkpoint needs
# ~50 GiB VRAM for a single member, so the pmap path does not apply.
#
# Host RAM: one serial 0.25 deg GenCast process peaks around 237 GiB, hence mem=320G.
#
# Prereqs (login node): the event's init frames must exist --
#   ls runs/xres/0p25/week3/inputs/PNW_HeatDome_2021_inputs.nc
# and `python -m aires.walker --check` should pass. This job builds nothing.
#
# Submit:
#   sbatch slurm/aires_walker_smoke.slurm
#   sbatch --export=ALL,AIRES_SMOKE_STEPS=4 slurm/aires_walker_smoke.slurm
set -eo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate "${GENCAST_CONDA_ENV:-moe}"

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1

# The 0.25 deg checkpoint does not fit an 80 GB H100 in the pmap path; serial + bf16 is
# the same configuration the xres 0.25 deg pool runs (slurm/xres_pool_0p25_week3.slurm).
export XRES_SERIAL_INFER=1
export XRES_BF16=1
# --gres=gpu:1 does not necessarily narrow CUDA_VISIBLE_DEVICES here, so JAX still
# enumerates all 8 H100s and load_gencast rounds the requested ensemble up to a multiple
# of the device count. Harmless in serial mode (roll_segment passes exactly one rng and
# ignores bundle["n_members"]), but it makes the log claim an 8-member ensemble that is
# never rolled. Pin it so the log says what actually happens.
export XRES_INFER_GPUS=1

EVENT=${AIRES_SMOKE_EVENT:-PNW_HeatDome_2021}
STEPS=${AIRES_SMOKE_STEPS:-2}
WALKER=${AIRES_SMOKE_WALKER:-0}

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
echo "Walker smoke: event=$EVENT walker=$WALKER steps=$STEPS (${STEPS}x12 h)"

INIT="runs/xres/0p25/week3/inputs/${EVENT}_inputs.nc"
if [[ ! -f "$INIT" ]]; then
  echo "ERROR: missing init frames $INIT" >&2
  exit 1
fi

# 1. roll the segment
python -m aires.walker --event "$EVENT" --walker "$WALKER" --step 1 --steps "$STEPS"

# 2. prove the product is a valid restart point AND a valid FCN3 initial condition.
#    This is the walker -> adapter -> scorer handoff, end to end.
python - <<'PY'
import os, sys
import numpy as np
sys.path.insert(0, os.environ["PYTHONPATH"])
from aires import aconfig as A, adapter as AD, walker as W

ev = os.environ.get("AIRES_SMOKE_EVENT", "PNW_HeatDome_2021")
wk = int(os.environ.get("AIRES_SMOKE_WALKER", "0"))
sp = A.state_path(ev, wk, 1)

state = W.read_state(sp)                       # re-asserts the full state contract
print(f"[verify] state {sp.name}: valid {W.valid_time(state)}, "
      f"{state.sizes['lat']}x{state.sizes['lon']} global, 2 frames")

init = W.initial_state(ev)
adv = (W.valid_time(state) - W.valid_time(init)).total_seconds() / 3600.0
print(f"[verify] advanced {adv:.0f} h from the event init")

# The state must be usable as the START of another segment - that is what resampling does.
full, i2 = W.build_batch(state, 2)
print(f"[verify] restarts cleanly: a further 2-step batch builds from {i2}")

# ... and as an FCN3 initial condition.
ic = AD.gencast_to_fcn3(state, calib=AD.require_calibration())
assert tuple(str(s) for s in ic["variable"].values) == A.FCN3_VARIABLES
assert np.isfinite(ic.values).all(), "adapter produced non-finite channels"
t2m = float(ic.sel(variable="t2m").mean())
tcwv = float(ic.sel(variable="tcwv").mean())
print(f"[verify] adapter -> FCN3 IC: 72 channels, all finite; "
      f"global mean t2m {t2m:.2f} K, tcwv {tcwv:.2f} kg m-2")
assert 250.0 < t2m < 300.0, t2m
assert 5.0 < tcwv < 50.0, tcwv
print("[verify] OK - walker state is both a restart point and an FCN3 IC")
PY

echo "End: $(date)"
