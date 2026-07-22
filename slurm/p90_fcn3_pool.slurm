#!/bin/bash
#SBATCH --job-name=p90_fcn3_pool
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# FourCastNet3 p90 T2m infer: whole-node worker POOL, ALL 8 H100s busy (billed per
# node). Eight workers, one pinned per GPU (CUDA_VISIBLE_DEVICES=0..7), each pulls
# cases via run_p90.py's claim files and writes one cube per case (CUBE SCHEMA) to
# runs/p90_t2m/fcn3/cache/<case>_cube.nc. Cubes + claims are cache-aware, so a
# resubmit skips finished cases - the pool is idempotent.
#
# Usage (run from the a3mega checkout; prep MUST be done first on the LOGIN node):
#   bash p90/bash/00_prep.sh                       # login node: truth + IC frames
#   sbatch slurm/p90_fcn3_pool.slurm               # 1 node, all 90 cases
#   sbatch --array=0-1 slurm/p90_fcn3_pool.slurm   # 2 nodes; each task takes a shard
# Resubmit-safe: cached cubes/claims mean re-running only fills the gaps.
#
# Rough cost: 90 cases x 50 members x 87 steps. Per case per GPU is roughly
# 10-40 min depending on batch size; a single 8-GPU node clears all 90 well inside
# the 12 h wall. GPU nodes are inference-only (CLAUDE.md) - prep is pre-staged.
set -eo pipefail
shopt -s nullglob

ROOT="${P90_ROOT_DIR:-/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather}"
RUNS="${GENCAST_RUNS:-$ROOT/runs}"
NGPU=8

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate "${FCN3_ENV:-fcn3}"

# torch ships its own CUDA libs under site-packages/nvidia/*/lib (same trick as
# pbs/fcn3_infer.pbs) - put them on the loader path so the checkpoint's kernels load.
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Reuse the fcn3-test earth2studio / HuggingFace caches (checkpoint already fetched).
export EARTH2STUDIO_CACHE="$RUNS/fcn3/.cache/e2s"
export HF_HOME="$RUNS/fcn3/.cache/hf"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Multi-node arrays: turn the array coordinates into a P90_SHARD so each node's
# pool takes a disjoint slice of the 90 cases (run_p90.py does the position%n split).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  export P90_SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
else
  # Single-node submit: sbatch's default --export=ALL carries an inherited P90_SHARD
  # (left over from a smoke/shard test in the submitting shell) into the job, which would
  # silently shrink this node's pool to a fraction of the 90 cases. Clear it.
  unset P90_SHARD
fi
# The pool ALWAYS runs the full 90-case design. These case filters are also carried in by
# --export=ALL; a leaked value silently shrinks the run (e.g. P90_MAX_CASES=1 -> 1 cube),
# burning a whole 8xH100 node on a fraction of the work. Refuse rather than run silent
# (use run_p90.py --stage infer directly for a deliberate subset).
for v in P90_KIND P90_CASES_SEL P90_MAX_CASES; do
  if [ -n "${!v:-}" ]; then
    echo "ERROR: $v=${!v} is set in the pool job environment; the pool must run all 90" >&2
    echo "cases. Unset it before submitting (it was likely exported during a smoke test)." >&2
    exit 1
  fi
done
SH_I=0; SH_N=1
if [ -n "${P90_SHARD:-}" ]; then SH_I="${P90_SHARD%/*}"; SH_N="${P90_SHARD#*/}"; fi

cd "$ROOT"
mkdir -p "$ROOT/logs"

echo "Host: $(hostname)  Start: $(date)  ROOT=$ROOT  shard=$SH_I/$SH_N  NGPU=$NGPU"
nvidia-smi -L || true

# ------------------------------------------------------------------ preflight
CASES_CSV="$ROOT/p90/cases.csv"
IC_DIR="$RUNS/p90_t2m/ic"
CKPT_CACHE="$RUNS/fcn3/.cache/e2s"

fail() { echo "ERROR: $*" >&2; echo "run p90/bash/00_prep.sh on the login node first" >&2; exit 1; }

[ -f "$CASES_CSV" ] || fail "missing $CASES_CSV"
NCASES=$(($(wc -l < "$CASES_CSV") - 1))
IC_N=$(ls "$IC_DIR"/*_ic.nc 2>/dev/null | wc -l)
[ "$IC_N" -eq "$NCASES" ] || fail "IC count $IC_N != case count $NCASES ($IC_DIR)"
CKPT=$(find "$CKPT_CACHE" -name best_ckpt_mp0.tar 2>/dev/null | head -1)
[ -n "$CKPT" ] || fail "FCN3 checkpoint best_ckpt_mp0.tar not found under $CKPT_CACHE"
echo "[preflight] cases=$NCASES ic=$IC_N ckpt=$CKPT"

# expected cubes for THIS node's shard (position%n == i over the 90 cases)
EXP=0
for ((p=0; p<NCASES; p++)); do (( p % SH_N == SH_I )) && EXP=$((EXP+1)); done

# ------------------------------------------------------------------ launch pool
# Worker 0 goes first with --clean-claims (drop any stale claims from a dead job),
# then a short pause before the rest so exactly one worker resets the claim dir.
PIDS=()
for i in $(seq 0 $((NGPU-1))); do
  wlog="$ROOT/logs/p90_fcn3_${SLURM_JOB_ID}_w$i.log"
  echo "[launcher] worker $i -> GPU $i -> $wlog"
  if [ "$i" -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=$i P90_WORKER=$i \
      python "$ROOT/p90/run_p90.py" --stage infer --model fcn3 \
        --batch-size "${P90_FCN3_BATCH:-4}" --clean-claims > "$wlog" 2>&1 &
    PIDS+=($!)
    sleep 5
  else
    CUDA_VISIBLE_DEVICES=$i P90_WORKER=$i \
      python "$ROOT/p90/run_p90.py" --stage infer --model fcn3 \
        --batch-size "${P90_FCN3_BATCH:-4}" > "$wlog" 2>&1 &
    PIDS+=($!)
  fi
done

FAIL=0
for k in "${!PIDS[@]}"; do
  if wait "${PIDS[$k]}"; then echo "[launcher] worker $k ok"; else echo "[launcher] worker $k FAILED"; FAIL=1; fi
done

CUBES=$(ls "$RUNS/p90_t2m/fcn3/cache"/*_cube.nc 2>/dev/null | wc -l)
echo "[status] cubes=$CUBES (this shard expects $EXP; 90 total across all nodes)  workers-failed=$FAIL"
echo "End: $(date)"
[ "$FAIL" -eq 0 ] || exit 1
