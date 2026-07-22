#!/bin/bash
# FCN3 vs GenCast head-to-head test driver (PNW Heat Dome 2021, week-2, 0.25 deg).
#
# a3mega cluster (Slurm / H100). CPU work runs on the login node; the ensemble infer runs
# on ONE whole 8-GPU H100 node via sbatch, with the 24-member ensemble split across the 8
# GPUs (see slurm/fcn3_infer.slurm). The compare step is CPU-only and runs on the login
# node in `moe` (needs the synced xres cube + ERA5 truth).
#
# Stages:
#   1. prep      login node: fetch/verify the ERA5 IC + FCN3 checkpoint cache
#   2. infer     sbatch slurm/fcn3_infer.slurm: 8 GPUs -> 8 shard cubes -> merged cube + timing
#   3. compare   login node, after infer completes: FCN3-vs-GenCast PDF + runtime figures
#
# Usage (run from the repo root or from fcn3/):
#   bash fcn3/run_test.sh                 # full: prep -> sbatch infer(24 members / 8 GPUs)
#   bash fcn3/run_test.sh smoke           # smoke: prep -> sbatch infer(8 members x 4 steps)
#   bash fcn3/run_test.sh prep            # prep only (login node)
#   bash fcn3/run_test.sh submit          # skip prep, just sbatch infer
#   bash fcn3/run_test.sh compare         # run the compare figures on the login node
set -eo pipefail

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO"

EVENT=PNW_HeatDome_2021
IC=runs/fcn3/$EVENT/${EVENT}_ic.nc

MODE=${1:-full}

run_prep() {
  if [[ -f "$IC" ]]; then
    echo "[prep] IC already cached: $IC (delete it to force a refetch) -- skipping prep"
  else
    echo "[prep] fetching IC + FCN3 checkpoint on the login node ..."
    python fcn3/run_fcn3.py --stage prep
  fi
  [[ -f "$IC" ]] || { echo "[prep] FAILED: $IC not created"; exit 1; }
}

submit_infer() {
  local export_arg="$1"
  echo "[infer] sbatch slurm/fcn3_infer.slurm ${export_arg:+($export_arg)}"
  local JID
  if [[ -n "$export_arg" ]]; then
    JID=$(sbatch --parsable --export="ALL,$export_arg" slurm/fcn3_infer.slurm)
  else
    JID=$(sbatch --parsable slurm/fcn3_infer.slurm)
  fi
  echo "[infer] submitted job $JID"
  echo
  echo "Monitor:      squeue -u \$USER -n fcn3_infer"
  echo "Live log:     tail -f logs/fcn3_infer-${JID}.out"
  echo "After it OKs: bash fcn3/run_test.sh compare"
}

run_compare() {
  echo "[compare] FCN3 vs GenCast figures (login node, moe) ..."
  python fcn3/compare_fcn3_gencast.py
}

case "$MODE" in
  prep)    run_prep ;;
  submit)  submit_infer "" ;;
  smoke)   run_prep; submit_infer "FCN3_MEMBERS=8,FCN3_NSTEPS=4" ;;
  full)    run_prep; submit_infer "" ;;
  compare) run_compare ;;
  *)       echo "unknown mode: $MODE (use: full | smoke | prep | submit | compare)"; exit 2 ;;
esac
