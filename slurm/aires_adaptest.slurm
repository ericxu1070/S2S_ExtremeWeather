#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%A_%a.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%A_%a.err

# AI+RES adapter test -- the GPU half (aires/adaptest.py).
#
# One array task per case. Each task produces the matched PAIR of 24-member FCN3 ensembles
# for its case at week-3 lead, across 8 GPUs, then assembles the cubes:
#
#   native cube   FCN3 initialised straight from ERA5   (fcn3/run_fcn3.py)
#   adapter cube  FCN3 initialised through the adapter  (aires/gate2.py)
#
# For the SIX EXTREME EVENTS (indices 0-5) the native cube already exists -- 24 members
# from the FCN3-vs-GenCast experiment -- so the task builds only the adapter half and the
# first half is free. For the FOUR NULL CONTROLS (indices 6-9) neither half exists, so the
# task builds both and costs roughly twice as much. Both stages are skipped outright if
# their cube is already on disk, so a resubmit is cheap.
#
# WHY CONTROLS. Every one of the six events is an extreme by construction, which makes them
# a biased sample for "does the adapter cost skill?" -- they are exactly the cases where a
# small IC perturbation has the best chance of growing. The controls are quiet days (CONUS
# mean T2m anomaly within +-0.2 K) selected from the frozen p90 control set, and they say
# whether the adapter penalty is a generic property of the added IC error or something that
# only appears when the atmosphere is already unstable. See fcn3/fevents.py::CONTROLS.
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
# jobs are dominated by that load, not by compute, and a control pays that load TWICE
# because the native and adapter rollouts are separate process launches -- hence ~15 min
# for an event and ~30 min for a control. The 3 h walltime is headroom for that plus retries.
#
# Prereqs (LOGIN node, never here -- a GPU job must be pure cache-hit-then-compute):
#   bash scripts/adaptest_prep.sh                          # does all of the below
#   PYTHONPATH=. python -m aires.adaptest --stage plan     # what is missing
#   PYTHONPATH=. python -m aires.adaptest --stage prep     # builds every adapter IC
# A control additionally needs its xres init frames + ERA5 truth (run_xres.py --stage prep)
# and its native FCN3 IC (fcn3/run_fcn3.py --stage prep); adaptest_prep.sh builds both.
# This job verifies every input it needs and refuses to start otherwise.
#
# Submit:
#   sbatch slurm/aires_adaptest.slurm                      # all ten cases
#   sbatch --array=1-5%2 slurm/aires_adaptest.slurm        # the extremes only (PNW is done)
#   sbatch --array=6-9%2 slurm/aires_adaptest.slurm        # the null controls only
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

# The case for this array task. The order is fcn3.fevents.ALL_ORDER -- the six events
# frozen first, the four null controls appended -- which is also what `--stage plan` prints
# as the --array indices, so the two cannot drift. Appending is not cosmetic: seeds are
# BASE_SEED + 1000 * ALL_ORDER.index(name), so inserting a control ahead of an event would
# renumber its members and silently unpair them from the native cube already on disk.
EVENTS=(PNW_HeatDome_2021 HurricaneIan_2022 WinterStorm_Uri_2021 \
        p90_20231107 p90_20240802 p90_20251224 \
        null_20230324 null_20240329 null_20241121 null_20250526)
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
if F.ALL_ORDER[idx] != name:
    raise SystemExit(f"array index {idx} is {name!r} here but {F.ALL_ORDER[idx]!r} in "
                     f"fcn3.fevents.ALL_ORDER -- fix slurm/aires_adaptest.slurm")
ev = F.ALL_EVENTS[name]
print(f"[check] array index {idx} == {name} ({ev.family}, seed {F.seed_for(ev)}) "
      f"-- matches fcn3.fevents.ALL_ORDER")
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
  echo "On the LOGIN node run:  bash scripts/adaptest_prep.sh" >&2
  exit 1
fi
NATIVE="runs/fcn3/week${FCN3_WEEKS}/cache/${EVENT}_cube.nc"
NATIVE_IC="runs/fcn3/week${FCN3_WEEKS}/ic/${EVENT}_ic.nc"
CUBE="runs/aires/gate2/cache/${EVENT}_adapter_cube.nc"

# --------------------------------------------------------------------------------- #
# One 8-shard rollout, with retries. Called once for the native half and once for the
# adapter half; the two differ only in which driver rolls the members, so they share the
# retry policy rather than having two copies of it that can drift apart.
#
# Foreign processes have taken this node's GPUs before even under --exclusive (Gate 3 job
# 1155: an out-of-Slurm vLLM server holding 78 GiB on all 8 H100s). Every stage below is
# cache-aware per artifact, so a retry costs seconds for whatever already finished and
# only redoes what is missing; a real bug still fails every attempt.
# --------------------------------------------------------------------------------- #
roll() {
  local kind=$1 out=$2
  if [[ -f "$out" ]]; then
    echo "[$kind] cube already built -> $out   (nothing to do)"
    return 0
  fi
  echo "=== [$kind] rolling $MEMBERS members of $EVENT across $NSHARDS GPUs ($(date))"

  local attempt=1 s fail i
  while (( attempt <= ATTEMPTS )); do
    echo "--- [$kind] attempt $attempt/$ATTEMPTS ($(date))"
    local pids=()
    for ((s = 0; s < NSHARDS; s++)); do
      if [[ "$kind" == native ]]; then
        CUDA_VISIBLE_DEVICES=$s FCN3_EVENTS="$EVENT" \
          python fcn3/run_fcn3.py --stage infer \
            --members "$MEMBERS" --batch-size "$BATCH" \
            --nshards "$NSHARDS" --shard "$s" \
            > "logs/${TAG}-${kind}-try${attempt}-shard${s}.log" 2>&1 &
      else
        CUDA_VISIBLE_DEVICES=$s \
          python -m aires.gate2 --stage infer --event "$EVENT" \
            --members "$MEMBERS" --batch-size "$BATCH" \
            --nshards "$NSHARDS" --shard "$s" \
            > "logs/${TAG}-${kind}-try${attempt}-shard${s}.log" 2>&1 &
      fi
      pids+=($!)
      echo "  [$kind] launched shard $s on GPU $s (pid ${pids[-1]})"
    done

    fail=0
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        echo "  [$kind] shard $i failed -- see logs/${TAG}-${kind}-try${attempt}-shard${i}.log"
        fail=1
      fi
    done

    if (( fail == 0 )); then
      echo "  [$kind] all shards done; merging ..."
      if [[ "$kind" == native ]]; then
        FCN3_EVENTS="$EVENT" python fcn3/run_fcn3.py --stage assemble \
          --members "$MEMBERS" --nshards "$NSHARDS"
      else
        python -m aires.gate2 --stage assemble --event "$EVENT" \
          --members "$MEMBERS" --nshards "$NSHARDS"
      fi
      break
    fi

    if (( attempt == ATTEMPTS )); then
      echo "ERROR: [$kind] $EVENT failed on every attempt -- check for foreign GPU" >&2
      echo "       processes with nvidia-smi, then read the shard logs before blaming" >&2
      echo "       the code." >&2
      return 1
    fi
    echo "  [$kind] retrying in ${RETRY_WAIT}s"
    sleep "$RETRY_WAIT"
    (( attempt++ ))
  done

  if [[ ! -f "$out" ]]; then
    echo "ERROR: [$kind] assemble reported success but $out is missing" >&2
    return 1
  fi
  ls -l "$out"
  return 0
}

# --- half one: the ERA5-initialised (native) ensemble ------------------------------ #
# Present already for the six extreme events; the null controls have never been rolled, so
# this is where their first half gets built. Both halves must come from the same code path
# and the same seeds as the six, which is why this calls the ordinary FCN3 driver rather
# than anything adaptest-specific.
if [[ ! -f "$NATIVE" ]]; then
  if [[ ! -f "$NATIVE_IC" ]]; then
    echo "ERROR: no native cube AND no native IC for $EVENT" >&2
    echo "  missing: $NATIVE_IC" >&2
    echo "  On the LOGIN node run:  bash scripts/adaptest_prep.sh" >&2
    exit 1
  fi
  echo "native cube absent -- this case needs BOTH halves rolled."
fi
roll native "$NATIVE" || exit 1

# --- half two: the adapter-initialised ensemble ------------------------------------ #
roll adapter "$CUBE" || exit 1

echo "End: $(date)"
echo "When every array task is done, on the login node in moe:"
echo "  PYTHONPATH=. python -m aires.adaptest --stage score"
echo "  PYTHONPATH=. python -m aires.adaptest --stage report"
