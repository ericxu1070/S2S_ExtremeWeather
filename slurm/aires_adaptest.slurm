#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=01:30:00
#SBATCH --array=0-5%2
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%A_%a.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%A_%a.err

# AI+RES adapter test -- the GPU half (aires/adaptest.py).
#
# One array task per event. Each task rolls the ADAPTER-IC ensemble for its event at
# week-3 lead across 8 GPUs, then assembles the cube. The native-IC ensemble it will be
# compared against already exists (runs/fcn3/week3/cache/<event>_cube.nc, 24 members from
# the FCN3-vs-GenCast experiment), so this job produces the second half of six matched
# pairs and costs nothing for the first halves.
#
# WHY AN ARRAY AND NOT A LOOP. A loop on one node would pay the ~9 min shared model load
# once per event anyway (each event relaunches 8 fresh processes), so a loop buys nothing
# and gives up per-event resumability. `%2` throttles to two concurrent nodes because
# a3mega has only 8 and every event is ~15 min of work.
#
# WHY THIS REUSES aires.gate2. The adapter IC builder and the FCN3 driver live in
# aires/gate2.py and there is deliberately no second copy of either. `adaptest` adds the
# six-event ANALYSIS (paired bootstrap, across-event sign test), not a second pipeline.
#
# SEED MATCHING IS THE EXPERIMENT. Members are seeded exactly as the native run was --
# same 8-shard split, same `seed_for(ev) + shard` -- so adapter member i and native
# member i share an internal-noise stream and their difference isolates the IC. Change
# MEMBERS or NSHARDS and the seeds no longer line up, which silently turns the paired
# bootstrap into an unpaired one and roughly doubles its standard error. Do not.
#
# Cost: 24 members x 84 six-hour steps at ~1 s/step = ~4 min per GPU, 8 GPUs in parallel,
# plus ~9 min for 8 processes reading the shared 4.18 GB model pickle off NFS. Short FCN3
# jobs are dominated by that load, not by compute -- the 1.5 h walltime is headroom for it
# and for one retry.
#
# Prereqs (LOGIN node, never here -- a GPU job must be pure cache-hit-then-compute):
#   PYTHONPATH=. python -m aires.adaptest --stage plan     # what is missing
#   PYTHONPATH=. python -m aires.adaptest --stage prep     # builds every adapter IC
# This job verifies its IC and its native cube and refuses to start otherwise.
#
# Submit:
#   sbatch slurm/aires_adaptest.slurm                      # all six events
#   sbatch --array=1,2,3,4,5%2 slurm/aires_adaptest.slurm  # skip PNW (index 0, done)
#   sbatch --array=2 slurm/aires_adaptest.slurm            # redo one; cached shards skip
#
# Then, on the login node in `moe`:
#   PYTHONPATH=. python -m aires.adaptest --stage score
#   PYTHONPATH=. python -m aires.adaptest --stage report
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
export FCN3_WEEKS=${FCN3_WEEKS:-3}

# The event for this array task. The order is fcn3.fevents.ORDER, which is frozen -- it is
# also what `--stage plan` prints as the --array indices, so the two cannot drift.
EVENTS=(PNW_HeatDome_2021 HurricaneIan_2022 WinterStorm_Uri_2021 \
        p90_20231107 p90_20240802 p90_20251224)
IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= ${#EVENTS[@]} )); then
  echo "ERROR: array index $IDX is past the ${#EVENTS[@]} known events" >&2
  exit 1
fi
EVENT=${EVENTS[$IDX]}

# Pin the event list against the Python registry rather than trusting this array to have
# been kept in sync by hand. A silent reordering would score event A's adapter cube against
# event B's native cube, which no downstream check would catch.
python - "$IDX" "$EVENT" <<'PY'
import sys
from fcn3 import fevents as F
idx, name = int(sys.argv[1]), sys.argv[2]
if F.ORDER[idx] != name:
    raise SystemExit(f"array index {idx} is {name!r} here but {F.ORDER[idx]!r} in "
                     f"fcn3.fevents.ORDER -- fix slurm/aires_adaptest.slurm")
print(f"[check] array index {idx} == {name} (matches fcn3.fevents.ORDER)")
PY

MEMBERS=${AIRES_GATE2_MEMBERS:-24}
NSHARDS=${FCN3_NSHARDS:-8}
BATCH=${FCN3_BATCH:-1}
ATTEMPTS=${AIRES_POOL_ATTEMPTS:-3}
RETRY_WAIT=${AIRES_POOL_WAIT:-120}

# Pin each shard to a disjoint core slice: 8 unpinned processes each spawn one OpenMP
# thread per node core and thrash during model construction (fcn3_infer.slurm, job 700).
CORES=$(nproc)
THREADS=${FCN3_THREADS:-$(( CORES / NSHARDS ))}
[ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close OMP_PLACES=cores

TAG="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${IDX}"
echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
echo "adaptest: event=$EVENT (array $IDX) members=$MEMBERS nshards=$NSHARDS weeks=$FCN3_WEEKS"
echo "CPU: $CORES cores / $NSHARDS shards -> OMP_NUM_THREADS=$THREADS per shard"

IC="runs/aires/gate2/ic/${EVENT}_adapter_ic.nc"
if [[ ! -f "$IC" ]]; then
  echo "ERROR: missing adapter IC $IC" >&2
  echo "On the LOGIN node run:  PYTHONPATH=. python -m aires.adaptest --stage prep" >&2
  exit 1
fi
NATIVE="runs/fcn3/week3/cache/${EVENT}_cube.nc"
if [[ ! -f "$NATIVE" ]]; then
  echo "ERROR: missing native cube $NATIVE -- nothing to compare against." >&2
  exit 1
fi
CUBE="runs/aires/gate2/cache/${EVENT}_adapter_cube.nc"
if [[ -f "$CUBE" ]]; then
  echo "adapter cube already built -> $CUBE   (nothing to do)"
  echo "End: $(date)"
  exit 0
fi

# Foreign processes have taken this node's GPUs before even under --exclusive (Gate 3 job
# 1155: an out-of-Slurm vLLM server holding 78 GiB on all 8 H100s). Both stages below are
# cache-aware per artifact, so a retry costs seconds for whatever already finished and
# only redoes what is missing; a real bug still fails every attempt.
attempt=1
while (( attempt <= ATTEMPTS )); do
  echo "--- attempt $attempt/$ATTEMPTS ($(date))"
  pids=()
  for ((s = 0; s < NSHARDS; s++)); do
    CUDA_VISIBLE_DEVICES=$s \
      python -m aires.gate2 --stage infer --event "$EVENT" \
        --members "$MEMBERS" --batch-size "$BATCH" \
        --nshards "$NSHARDS" --shard "$s" \
        > "logs/${TAG}-try${attempt}-shard${s}.log" 2>&1 &
    pids+=($!)
    echo "  launched shard $s on GPU $s (pid ${pids[-1]})"
  done

  fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "  shard $i failed -- see logs/${TAG}-try${attempt}-shard${i}.log"
      fail=1
    fi
  done

  if (( fail == 0 )); then
    echo "all shards done; merging ..."
    python -m aires.gate2 --stage assemble --event "$EVENT" \
      --members "$MEMBERS" --nshards "$NSHARDS"
    break
  fi

  if (( attempt == ATTEMPTS )); then
    echo "ERROR: $EVENT failed on every attempt -- check for foreign GPU processes with" >&2
    echo "       nvidia-smi, then read the shard logs before blaming the code." >&2
    echo "End: $(date)"
    exit 1
  fi
  echo "  retrying in ${RETRY_WAIT}s"
  sleep "$RETRY_WAIT"
  (( attempt++ ))
done

if [[ ! -f "$CUBE" ]]; then
  echo "ERROR: assemble reported success but $CUBE is missing" >&2
  exit 1
fi
ls -l "$CUBE"
echo "End: $(date)"
echo "When every array task is done, on the login node in moe:"
echo "  PYTHONPATH=. python -m aires.adaptest --stage score"
echo "  PYTHONPATH=. python -m aires.adaptest --stage report"
