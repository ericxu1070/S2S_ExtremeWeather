#!/bin/bash
#SBATCH --job-name=fcn3_infer
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# FourCastNet 3 ensemble infer for PNW Heat Dome 2021, week-2 (0.25 deg), on ONE whole
# a3mega H100 node (8x H100-80GB). FCN3 is single-GPU PyTorch, so the ensemble is split
# across the 8 GPUs: NSHARDS=8 processes, each pinned to one GPU via CUDA_VISIBLE_DEVICES,
# each rolling MEMBERS/NSHARDS members with a distinct seed. The last step merges the 8
# per-shard cubes into the canonical cube + a node-wall timing sidecar.
#
# GCP a3mega has no sub-node GPU sharing, so a whole node is the smallest GPU unit anyway;
# splitting the ensemble across its 8 GPUs is what keeps all 8 busy (project rule: a GPU
# allocation is for GPU compute only).
#
# Prereqs (pre-stage on the LOGIN node, which has internet -- never in this GPU job):
#   the IC + FCN3 checkpoint cache must already exist under runs/fcn3/. Build/sync with:
#     python fcn3/run_fcn3.py --stage prep      # login node
#   (or rsync runs/fcn3/PNW_HeatDome_2021/*_ic.nc + runs/fcn3/.cache/ from Derecho).
#
# Submit:
#   sbatch slurm/fcn3_infer.slurm                                 # full: 24 members / 8 GPUs
#   sbatch --export=ALL,FCN3_MEMBERS=8,FCN3_NSTEPS=4 slurm/fcn3_infer.slurm   # smoke
set -eo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
# FCN3 needs earth2studio + torch 2.11.0+cu128; moe carries torch 2.8.0 for the downscaler,
# so infer runs in a dedicated env by default. Override with FCN3_CONDA_ENV if you built it
# into a different env. (compare/ still runs in moe -- it needs gencast_s2s/xres, not this.)
conda activate "${FCN3_CONDA_ENV:-/home/ubuntu/Vayuh/data/eric/conda-envs/fcn3}"

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO"

# Shared caches on scratch (must match run_fcn3.py defaults on this machine).
export EARTH2STUDIO_CACHE="$REPO/runs/fcn3/.cache/e2s"
export HF_HOME="$REPO/runs/fcn3/.cache/hf"

MEMBERS=${FCN3_MEMBERS:-24}
NSTEPS=${FCN3_NSTEPS:-56}
BATCH=${FCN3_BATCH:-1}
NSHARDS=${FCN3_NSHARDS:-8}   # one shard per GPU on a whole a3mega node

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
echo "FCN3 infer: members=$MEMBERS nsteps=$NSTEPS batch_size=$BATCH nshards=$NSHARDS"

# Launch one process per GPU, each pinned to a single H100. Backgrounded; wait for all.
pids=()
for ((s = 0; s < NSHARDS; s++)); do
  CUDA_VISIBLE_DEVICES=$s \
    python fcn3/run_fcn3.py --stage infer \
      --members "$MEMBERS" --nsteps "$NSTEPS" --batch-size "$BATCH" \
      --nshards "$NSHARDS" --shard "$s" \
      > "logs/fcn3_infer-${SLURM_JOB_ID}-shard${s}.log" 2>&1 &
  pids+=($!)
  echo "  launched shard $s on GPU $s (pid ${pids[-1]})"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: shard $i (pid ${pids[$i]}) failed -- see logs/fcn3_infer-${SLURM_JOB_ID}-shard${i}.log"
    fail=1
  fi
done
[[ $fail -eq 0 ]] || { echo "One or more shards failed; not assembling."; exit 1; }

echo "All shards done; merging into the canonical cube ..."
python fcn3/run_fcn3.py --stage assemble --members "$MEMBERS" --nshards "$NSHARDS"

echo "End: $(date)"
