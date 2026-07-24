#!/bin/bash
#SBATCH --job-name=fcn3_infer
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# FourCastNet 3 ensemble infer for the FCN3-vs-GenCast comparison: all six events in
# fcn3/fevents.py at week-3 (21 d lead), 0.25 deg, 24 members, on ONE whole a3mega H100
# node (8x H100-80GB).
#
# FCN3 is single-GPU PyTorch, so the ENSEMBLE is split across the 8 GPUs: NSHARDS=8
# processes, each pinned to one GPU via CUDA_VISIBLE_DEVICES, each rolling 24/8 = 3
# members of EVERY event with a distinct seed. Every shard walks the full event list, so
# all 8 GPUs stay busy for the whole job. The final step merges the per-shard cubes into
# one canonical cube per event plus a node-wall timing sidecar.
#
# Expected cost: FCN3 rolls ~1 s per 6-hourly step, so 84 steps ~ 1.5 min/member =>
# 3 members x 6 events ~ 30 min per GPU. The 4 h walltime is generous headroom.
#
# Prereqs (pre-stage on the LOGIN node, which has internet -- never in this GPU job):
#   python fcn3/run_fcn3.py --stage prep
# must have written runs/fcn3/week3/ic/<event>_ic.nc for every event and cached the
# checkpoint under runs/fcn3/.cache/. This job verifies that and refuses to start
# otherwise -- a GPU job must never download or build data (see CLAUDE.md).
#
# Submit:
#   sbatch slurm/fcn3_infer.slurm                                        # all 6 events
#   sbatch --export=ALL,FCN3_EVENTS=HurricaneIan_2022 slurm/fcn3_infer.slurm
#   sbatch --export=ALL,FCN3_MEMBERS=8,FCN3_NSTEPS=4 slurm/fcn3_infer.slurm   # smoke
set -eo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
# FCN3 needs earth2studio + torch 2.11.0+cu128; moe carries torch 2.8.0 for the downscaler,
# so infer runs in a dedicated env. (compare/ still runs in moe -- it needs xres, not this.)
conda activate "${FCN3_CONDA_ENV:-fcn3}"

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO"

# Shared caches on NFS (must match fcn3/fevents.py defaults on this machine).
export EARTH2STUDIO_CACHE="$REPO/runs/fcn3/.cache/e2s"
export HF_HOME="$REPO/runs/fcn3/.cache/hf"
export PYTHONUNBUFFERED=1

MEMBERS=${FCN3_MEMBERS:-24}
BATCH=${FCN3_BATCH:-1}
export FCN3_WEEKS=${FCN3_WEEKS:-3}

# --- shard count and CPU threads -------------------------------------------------- #
# FCN3 model CONSTRUCTION (torch-harmonics precomputing the DISCO convolution tensors) is
# a heavy CPU job that runs before the GPU is touched. Two incidents on 2026-07-23:
#   - job 700: 8 shards, OMP_NUM_THREADS unset -> ~1856 threads on 208 cores (load 763),
#     8 H100s idle 80+ min, no rollout.
#   - job 723: 4 shards x 52 pinned threads fixed the oversubscription (load ~204), but
#     4 model builds running CONCURRENTLY still contended (memory bandwidth): 17+ min in,
#     no shard had finished building, GPUs still idle.
#
# Real fix (run_fcn3.py::_get_model): the model is built AT MOST ONCE per node under a
# lock and pickled to NFS; every other shard -- and every later job -- loads that file in
# seconds. So concurrent builds no longer happen and all 8 GPUs can be used for the
# rollout. Threads are still pinned to a disjoint per-shard slice as cheap insurance
# (matters only for the lone builder + any brief load overlap).
NSHARDS=${FCN3_NSHARDS:-8}
CORES=$(nproc)
THREADS=${FCN3_THREADS:-$(( CORES / NSHARDS ))}
[ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
python fcn3/run_fcn3.py --stage plan
echo "FCN3 infer: members=$MEMBERS batch_size=$BATCH nshards=$NSHARDS weeks=$FCN3_WEEKS"
echo "CPU: $CORES cores / $NSHARDS shards -> OMP_NUM_THREADS=$THREADS per shard"

# GPU nodes are inference-only: every IC must already be on disk (pure cache hit).
missing=$(python - <<'PY'
import sys
sys.path.insert(0, ".")
from fcn3 import fevents as F
print(",".join(e.name for e in F.selected() if not F.ic_path(e).exists()))
PY
)
if [[ -n "$missing" ]]; then
  echo "ERROR: missing IC(s): $missing" >&2
  echo "On the LOGIN node run:  python fcn3/run_fcn3.py --stage prep" >&2
  exit 1
fi

# Launch one process per GPU, each pinned to a single H100. Backgrounded; wait for all.
pids=()
for ((s = 0; s < NSHARDS; s++)); do
  CUDA_VISIBLE_DEVICES=$s \
    python fcn3/run_fcn3.py --stage infer \
      --members "$MEMBERS" --batch-size "$BATCH" \
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
[[ $fail -eq 0 ]] || echo "WARNING: a shard failed; assembling whatever is complete."

echo "Merging per-shard cubes ..."
python fcn3/run_fcn3.py --stage assemble --members "$MEMBERS" --nshards "$NSHARDS" || true

echo "Cubes: $(ls runs/fcn3/week${FCN3_WEEKS}/cache/*_cube.nc 2>/dev/null | wc -l)"
echo "End: $(date)"
[[ $fail -eq 0 ]]
