#!/bin/bash
#SBATCH --job-name=xres_0p25pool_wk4
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# 0.25deg WEEK-4 infer (56 rollout steps): whole-node worker POOL, ALL 8 H100s busy.
#
# Same design as slurm/xres_pool_0p25_week3.slurm: 8 serial workers, one pinned per GPU
# (CUDA_VISIBLE_DEVICES=0..7), each rolling members via claim files (bf16). One member fills
# a whole H100 (~72 GiB); the node's 1.86 TB RAM lets all 8 run at once. 24 members = 3 waves.
# Launch this AFTER week3 has populated JAX_COMPILATION_CACHE_DIR so these workers skip the
# compile entirely and go straight to rolling.
#
# GPU nodes are inference-only (CLAUDE.md): prep must be pre-staged on the login node.
set -eo pipefail
shopt -s nullglob

WEEKS=4
NGPU=8

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export XRES_SERIAL_INFER=1
export XRES_BF16=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export PYTHONUNBUFFERED=1
export JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1
export JAX_COMPILATION_CACHE_DIR=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/runs/models/jax_cache_0p25
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

cd /home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
CACHE=runs/xres/0p25/week${WEEKS}/cache

echo "Host: $(hostname)  Start: $(date)  WEEKS=$WEEKS  NGPU=$NGPU"
nvidia-smi -L || true

inp=(runs/xres/0p25/week${WEEKS}/inputs/*_inputs.nc)
if [ "${#inp[@]}" -lt 12 ] || [ ! -f "runs/models/params/GenCast 0p25deg <2019.npz" ]; then
  echo "ERROR: prep incomplete (${#inp[@]}/12). On the LOGIN node run:" >&2
  echo "  python run_xres.py --stage prep --res 0p25 --weeks ${WEEKS}" >&2
  exit 1
fi

rm -f "$CACHE"/claims/* 2>/dev/null || true

JOBTAG=${SLURM_JOB_ID}
PIDS=()
for i in $(seq 0 $((NGPU-1))); do
  wlog="logs/xres_0p25pool_wk${WEEKS}_${JOBTAG}_w$i.log"
  echo "[launcher] worker $i -> GPU $i -> $wlog"
  CUDA_VISIBLE_DEVICES=$i XRES_WORKER=$i \
    python run_xres.py --stage infer --res 0p25 --weeks "$WEEKS" > "$wlog" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for k in "${!PIDS[@]}"; do
  if wait "${PIDS[$k]}"; then echo "[launcher] worker $k ok"; else echo "[launcher] worker $k FAILED"; FAIL=1; fi
done

cubes=$(ls "$CACHE"/*_cube.nc 2>/dev/null | wc -l)
echo "[status] cubes=$cubes/12  workers-failed=$FAIL"
echo "End: $(date)"
