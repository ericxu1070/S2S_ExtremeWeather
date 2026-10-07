#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# AI+RES Gate 3 -- score skill (aires/gate3.py + aires/score.py). The Phase 1 go/no-go.
#
# Two phases on ONE allocation, in TWO conda envs. They cannot share a process (JAX vs
# PyTorch/earth2studio), but they can share a node, and doing so means one queue wait and
# a page cache still holding the 477 MB walker states the second phase reads:
#
#   walk   moe  : N free-running GenCast walkers, init -> peak, global state every 3 d.
#                 8 workers, one per GPU, walker w on shard (w % 8). 0.25 deg needs ~72 GiB
#                 VRAM for ONE member, so each worker is serial + bf16 -- the same
#                 configuration as slurm/xres_pool_0p25_week3.slurm.
#   score  fcn3 : M FCN3 forecasts from every checkpoint to the peak. 8 workers, one per
#                 GPU, tasks round-robined so each GPU draws each rollout length equally.
#
# Each pool is retried up to AIRES_POOL_ATTEMPTS (3) times. Both stages are cache-aware
# per artifact, so a retry only redoes what is missing -- see run_pool.
#
# Cost at the defaults (N=16 walkers, M=6 members, checkpoints 3/6/9/12/15 d):
#   walk   672 GenCast steps @ ~28 s  = ~5.2 GPU-h -> ~40 min over 8 GPUs (+ ~5 min compile)
#   score  23,040 FCN3 steps  @ ~1 s  = ~6.4 GPU-h -> ~50 min over 8 GPUs (+ model load)
# The 4 h walltime is headroom; both phases are cache-aware, so a timeout is resumed by
# resubmitting, not restarted.
#
# Host RAM: one serial 0.25 deg GenCast worker peaks ~120 GiB, so 8 concurrent is ~1 TB of
# the node's 1.86 TB. --exclusive because that is most of the box.
#
# Disk: ~53 GB of walker states (477 MB x 16 x 7) + ~4 GB diagnostics + ~1 GB score cubes.
#
# Prereqs (LOGIN node -- GPU jobs are pure cache-hit-then-compute, CLAUDE.md):
#   PYTHONPATH=. python -m aires.gate3 --stage plan      # must print READY: yes
# It verifies the init frames, the derived-channel calibration, and that no cached walker
# segment sits at the wrong lead. This job re-runs that check and refuses to start if it
# does not pass.
#
# Submit:
#   sbatch slurm/aires_gate3.slurm
#   sbatch --export=ALL,AIRES_GATE3_PHASE=score slurm/aires_gate3.slurm   # score only
#   sbatch --export=ALL,AIRES_GATE3_WALKERS=24 slurm/aires_gate3.slurm    # extend N
#
# Then, on the login node in moe:
#   PYTHONPATH=. python -m aires.gate3 --stage reduce
set -eo pipefail

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1

EVENT=${AIRES_EVENT:-PNW_HeatDome_2021}
WALKERS=${AIRES_GATE3_WALKERS:-16}
MEMBERS=${AIRES_GATE3_MEMBERS:-6}
NGPU=${AIRES_NSHARDS:-8}
PHASE=${AIRES_GATE3_PHASE:-all}
TAG="${SLURM_JOB_NAME:-gate3}-${SLURM_JOB_ID:-local}"

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
echo "Gate 3: event=$EVENT walkers=$WALKERS members=$MEMBERS gpus=$NGPU phase=$PHASE"

run_pool_once() {                  # run_pool_once <label> <attempt> <cmd...>
  local label=$1 attempt=$2; shift 2
  local pids=() fail=0
  for ((s = 0; s < NGPU; s++)); do
    CUDA_VISIBLE_DEVICES=$s "$@" --shard "$s" --nshards "$NGPU" \
      > "logs/${TAG}-${label}-try${attempt}-shard${s}.log" 2>&1 &
    pids+=($!)
    echo "  launched $label shard $s on GPU $s (pid ${pids[-1]})"
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "ERROR: $label shard $i failed -- see logs/${TAG}-${label}-try${attempt}-shard${i}.log" >&2
      fail=1
    fi
  done
  return $fail
}

# Retry the whole pool, not the failed shards: both stages are cache-aware per artifact,
# so a repeat pass costs seconds for everything already on disk and only redoes what is
# missing. This exists because the GPUs on this cluster are NOT reliably ours even under
# `--exclusive` -- job 1155's score phase died with "GPU 0 ... Process 2214660 has
# 78.05 GiB memory in use" when an unrelated vLLM sweep, launched outside Slurm, took
# every card on the node. That contention is transient; a retry rides it out, and a real
# bug still fails all ATTEMPTS passes and stops the job.
ATTEMPTS=${AIRES_POOL_ATTEMPTS:-3}
RETRY_WAIT=${AIRES_RETRY_WAIT:-300}

run_pool() {
  local label=$1; shift
  for ((a = 1; a <= ATTEMPTS; a++)); do
    echo "  $label: attempt $a/$ATTEMPTS"
    if run_pool_once "$label" "$a" "$@"; then
      return 0
    fi
    if (( a < ATTEMPTS )); then
      echo "  $label: attempt $a failed; waiting ${RETRY_WAIT}s before retrying "
      echo "  (cached work is skipped, so a retry only redoes what is missing)"
      sleep "$RETRY_WAIT"
    fi
  done
  echo "ERROR: $label failed $ATTEMPTS time(s)" >&2
  return 1
}

# --------------------------------------------------------------------------- #
# Phase 1: the walkers (GenCast, moe)
# --------------------------------------------------------------------------- #
if [[ "$PHASE" == "all" || "$PHASE" == "walk" ]]; then
(
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  conda activate "${GENCAST_CONDA_ENV:-moe}"

  SITE=$(python -c "import site; print(site.getsitepackages()[0])")
  export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  # The 0.25 deg checkpoint occupies ~64 GB of an 80 GB H100, so the pmap path does not
  # apply: one serial member per GPU, bf16, exactly as the xres 0.25 deg pool runs.
  export XRES_SERIAL_INFER=1
  export XRES_BF16=1
  export XRES_INFER_GPUS=1
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
  export JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1
  export JAX_COMPILATION_CACHE_DIR="$REPO/runs/models/jax_cache_0p25"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"

  # Refuse to burn an allocation on a cache that does not match the schedule.
  python -m aires.gate3 --stage plan --event "$EVENT" --walkers "$WALKERS" \
    --members "$MEMBERS" --nshards "$NGPU"

  echo "--- walk phase: $(date) ---"
  run_pool walk python -m aires.gate3 --stage walk --event "$EVENT" --walkers "$WALKERS"
)
fi

# --------------------------------------------------------------------------- #
# Phase 2: the scores (FCN3, fcn3)
# --------------------------------------------------------------------------- #
if [[ "$PHASE" == "all" || "$PHASE" == "score" ]]; then
(
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  conda activate "${FCN3_CONDA_ENV:-fcn3}"

  SITE=$(python -c "import site; print(site.getsitepackages()[0])")
  export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  export EARTH2STUDIO_CACHE="$REPO/runs/fcn3/.cache/e2s"
  export HF_HOME="$REPO/runs/fcn3/.cache/hf"
  export FCN3_WEEKS=${FCN3_WEEKS:-3}

  # Pin each shard to a disjoint core slice: 8 unpinned processes each spawn one OpenMP
  # thread per NODE core and thrash during FCN3's model construction (job 700).
  CORES=$(nproc)
  THREADS=${FCN3_THREADS:-$(( CORES / NGPU ))}
  [ "$THREADS" -lt 1 ] && THREADS=1
  export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
  export OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
  export OMP_PROC_BIND=close OMP_PLACES=cores
  echo "CPU: $CORES cores / $NGPU shards -> OMP_NUM_THREADS=$THREADS per shard"

  echo "--- score phase: $(date) ---"
  run_pool score python -m aires.score --event "$EVENT" --walkers "$WALKERS" \
    --members "$MEMBERS"
)
fi

echo "End: $(date)"
echo "Now reduce on the login node in moe:"
echo "  PYTHONPATH=. python -m aires.gate3 --stage reduce --event $EVENT --walkers $WALKERS"
