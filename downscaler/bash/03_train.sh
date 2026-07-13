#!/usr/bin/env bash
# =============================================================================
# 03_train.sh — start (or resume) real training. Stop anytime with 04_stop.sh.
#
#   bash bash/03_train.sh                          # start / resume
#   bash bash/03_train.sh training.per_gpu_batch=2 # Hydra overrides pass through
#   DRY_RUN=1 bash bash/03_train.sh                # show what would run, run nothing
#   LOCAL=1  bash bash/03_train.sh                 # force local torchrun (skip Slurm)
#
# Where it runs:
#   - If Slurm is available (login node): submits slurm/train.sbatch — one
#     8-GPU a3mega node, 24 h walltime, resubmit to continue.
#   - Otherwise (a GPU box): nohup torchrun on all visible GPUs (GPUS=N to
#     limit), log in logs/train_<timestamp>.log, PID in logs/train.pid.
#
# Either way the run auto-resumes from checkpoints/latest.pt, so "start" and
# "resume" are the same command. A stop loses at most save_every_n_steps (100)
# optimizer steps. Refuses to start if the prerequisites (bash/02) are missing
# or if a run is already active.
# =============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN="${DRY_RUN:-0}"
LOCAL="${LOCAL:-0}"
cd "$PROJ" || exit 1

run() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN — would run: $*"; else "$@"; fi; }

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

# ---------------------------------------------------------------- preconditions
readarray -t CFG < <(python - <<'PY'
from omegaconf import OmegaConf
c = OmegaConf.load("configs/data/era5_hrrr.yaml")
print(c.index_path); print(c.stats_path)
PY
)
INDEX="${CFG[0]:-}"; STATS="${CFG[1]:-}"
[ -n "$INDEX" ] && [ -n "$STATS" ] || {
    echo "FAIL: could not read index_path/stats_path from configs/data/era5_hrrr.yaml"; exit 1; }
MISSING=0
[ -s "$INDEX" ] || { echo "missing timestamp index: $INDEX"; MISSING=1; }
[ -f "$STATS" ] || { echo "missing norm stats: $STATS"; MISSING=1; }
if [ "$MISSING" = 1 ]; then
    echo "run: bash bash/02_build_index_and_stats.sh   (one-time, ~35 min)"
    exit 1
fi

# ---------------------------------------------------------------- Slurm path
if [ "$LOCAL" != 1 ] && command -v sbatch >/dev/null; then
    ACTIVE=$(squeue -h -u "$USER" -n downscaler -o "%i %T" 2>/dev/null || true)
    if [ -n "$ACTIVE" ]; then
        echo "a 'downscaler' job is already in the queue:"
        echo "$ACTIVE" | sed 's/^/  job /'
        echo "watch it (bash/05_status.sh) or stop it first (bash/04_stop.sh)."
        exit 1
    fi
    echo "submitting slurm/train.sbatch (overrides: ${*:-none})"
    run sbatch slurm/train.sbatch "$@"
    echo
    echo "watch:  bash bash/05_status.sh     (or: tail -f logs/slurm-downscaler-<jobid>.out)"
    echo "stop:   bash bash/04_stop.sh       (graceful; loses <= 100 steps)"
    echo "resume: bash bash/03_train.sh      (same command, auto-resumes)"
    exit 0
fi

# ---------------------------------------------------------------- local path
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
    || { echo "no Slurm and no usable GPU on this host — nowhere to train."; exit 1; }

PIDFILE="$PROJ/logs/train.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "training already running locally (pid $(cat "$PIDFILE")) — stop it first: bash bash/04_stop.sh"
    exit 1
fi

NGPU="${GPUS:-$(nvidia-smi -L | wc -l)}"
LOG="$PROJ/logs/train_$(date +%Y%m%d_%H%M%S).log"

# Env lore inherited from slurm/worker.sh — each line prevents a real failure mode:
export HDF5_USE_FILE_LOCKING=FALSE                       # netCDF over NFS deadlocks workers
export TMPDIR=/tmp                                       # NFS tmp breaks DataLoader teardown
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "starting local training: $NGPU GPU(s), log $LOG (overrides: ${*:-none})"
if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN — would run: nohup torchrun --standalone --nproc_per_node=$NGPU train.py $*"
    exit 0
fi
mkdir -p "$PROJ/logs"
nohup torchrun --standalone --nproc_per_node="$NGPU" train.py "$@" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "pid $(cat "$PIDFILE")"
echo
echo "watch:  tail -f $LOG          (progress lines look like: [hb] step=...)"
echo "stop:   bash bash/04_stop.sh  (graceful; loses <= 100 steps)"
echo "resume: bash bash/03_train.sh"
