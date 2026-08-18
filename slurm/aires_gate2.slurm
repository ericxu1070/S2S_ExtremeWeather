#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# AI+RES Gate 2 -- forecast equivalence (aires/gate2.py).
#
# Rolls the ADAPTER-IC ensemble for one event at week-3 lead. The native-IC ensemble it is
# compared against already exists (runs/fcn3/week3/cache/<event>_cube.nc, 24 members from
# the FCN3-vs-GenCast experiment), so this job produces the second half of a matched pair
# and costs nothing for the first.
#
# Members are seeded EXACTLY as the native run was -- same 8-shard split, same
# `seed_for(ev) + shard` -- so adapter member i and native member i share an internal-noise
# stream and their difference isolates the IC. That matching is why this job must keep
# NSHARDS=8 and MEMBERS=24: change either and the seeds no longer line up.
#
# Cost: 24 members x 84 six-hour steps at ~1 s/step = ~4 min per GPU, 8 GPUs in parallel.
# The 1 h walltime is almost entirely headroom for the model load.
#
# Prereqs (LOGIN node, never here -- GPU jobs are pure cache-hit-then-compute):
#   PYTHONPATH=. python -m aires.gate2 --stage prep
# writes runs/aires/gate2/ic/<event>_adapter_ic.nc. This job verifies it and refuses to
# start otherwise.
#
# Submit:
#   sbatch slurm/aires_gate2.slurm
#   sbatch --export=ALL,AIRES_GATE2_EVENT=WinterStorm_Uri_2021 slurm/aires_gate2.slurm
#
# Then, on the login node in `moe`:
#   PYTHONPATH=. python -m aires.gate2 --stage score
set -eo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate "${FCN3_CONDA_ENV:-fcn3}"

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO"

export EARTH2STUDIO_CACHE="$REPO/runs/fcn3/.cache/e2s"
export HF_HOME="$REPO/runs/fcn3/.cache/hf"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"

EVENT=${AIRES_GATE2_EVENT:-PNW_HeatDome_2021}
MEMBERS=${AIRES_GATE2_MEMBERS:-24}
NSHARDS=${FCN3_NSHARDS:-8}
BATCH=${FCN3_BATCH:-1}
export FCN3_WEEKS=${FCN3_WEEKS:-3}

# Pin each shard to a disjoint core slice: 8 unpinned processes spawn one OpenMP thread
# per node core each and thrash during model construction (fcn3_infer.slurm, job 700).
CORES=$(nproc)
THREADS=${FCN3_THREADS:-$(( CORES / NSHARDS ))}
[ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close OMP_PLACES=cores

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
echo "Gate 2: event=$EVENT members=$MEMBERS nshards=$NSHARDS batch=$BATCH weeks=$FCN3_WEEKS"
echo "CPU: $CORES cores / $NSHARDS shards -> OMP_NUM_THREADS=$THREADS per shard"

IC="runs/aires/gate2/ic/${EVENT}_adapter_ic.nc"
if [[ ! -f "$IC" ]]; then
  echo "ERROR: missing adapter IC $IC" >&2
  echo "On the LOGIN node run:  PYTHONPATH=. python -m aires.gate2 --stage prep --event $EVENT" >&2
  exit 1
fi
NATIVE="runs/fcn3/week3/cache/${EVENT}_cube.nc"
if [[ ! -f "$NATIVE" ]]; then
  echo "ERROR: missing native cube $NATIVE -- nothing to compare against." >&2
  exit 1
fi

pids=()
for ((s = 0; s < NSHARDS; s++)); do
  CUDA_VISIBLE_DEVICES=$s \
    python -m aires.gate2 --stage infer --event "$EVENT" \
      --members "$MEMBERS" --batch-size "$BATCH" \
      --nshards "$NSHARDS" --shard "$s" \
      > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gate2-shard${s}.log" 2>&1 &
  pids+=($!)
  echo "  launched shard $s on GPU $s (pid ${pids[-1]})"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: shard $i failed -- see logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gate2-shard${i}.log"
    fail=1
  fi
done

echo "Merging shard cubes ..."
python -m aires.gate2 --stage assemble --event "$EVENT" \
  --members "$MEMBERS" --nshards "$NSHARDS"

echo "End: $(date)"
echo "Now score on the login node in moe:  PYTHONPATH=. python -m aires.gate2 --stage score --event $EVENT"
[[ $fail -eq 0 ]]
