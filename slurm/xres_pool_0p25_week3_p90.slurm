#!/bin/bash
#SBATCH --job-name=xres_0p25pool_wk3_p90
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# GenCast 0.25deg WEEK-3 infer for the THREE p90 cases added to the FCN3-vs-GenCast
# comparison. Identical machinery to slurm/xres_pool_0p25_week3.slurm (whole-node worker
# pool, 8 serial bf16 workers, one pinned per H100, members claimed via claim files) --
# the ONLY differences are:
#
#   1. XRES_EXTRA_EVENTS injects the three p90 events into the xres event registry (an
#      additive hook in xres/xconfig.py; unset -> the original 12 events, unchanged), and
#      XRES_EVENTS_SEL restricts this job to exactly those three. The 12 original cubes
#      are never touched.
#   2. The launcher records a node-wall timing JSON, because xres/xinference.py is not
#      instrumented and the comparison needs a hardware-matched GenCast runtime. The
#      already-cached xres events use job 223's measured wall instead (same node type,
#      same lead, same member count) -- see fcn3/fevents.py.
#
# Expected cost: job 223 did 12 events x 24 members at this lead in 12 h 41 min on one
# node (~63 min/event, ~28 s per 12-hourly step). Three events => ~3.2 h; 8 h walltime
# is headroom. Resubmitting is cheap: finished members/cubes are cached and skipped.
#
# GPU nodes are inference-only (CLAUDE.md): prep must be pre-staged on the login node with
#   XRES_EXTRA_EVENTS="$(python fcn3/fevents.py --spec)" \
#   XRES_EVENTS_SEL=p90_20231107,p90_20240802,p90_20251224 \
#   python run_xres.py --stage prep --res 0p25 --weeks 3
set -eo pipefail
shopt -s nullglob

WEEKS=3
NGPU=8

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd /home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather

# The three p90 events, injected and selected. Derived from fcn3/fevents.py so the event
# list has exactly one definition.
export XRES_EXTRA_EVENTS="$(python fcn3/fevents.py --spec)"
export XRES_EVENTS_SEL="${XRES_EVENTS_SEL:-$(python -c "
import sys; sys.path.insert(0,'.')
from fcn3 import fevents as F
print(','.join(e.name for e in F.EVENTS.values() if e.source=='p90'))")}"

export XRES_SERIAL_INFER=1
export XRES_BF16=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export PYTHONUNBUFFERED=1
export JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1
export JAX_COMPILATION_CACHE_DIR=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/runs/models/jax_cache_0p25
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

CACHE=runs/xres/0p25/week${WEEKS}/cache
IFS=',' read -ra EVENTS <<< "$XRES_EVENTS_SEL"
NEV=${#EVENTS[@]}

echo "Host: $(hostname)  Start: $(date)  WEEKS=$WEEKS  NGPU=$NGPU"
echo "Events ($NEV): $XRES_EVENTS_SEL"
nvidia-smi -L || true

# GPU nodes are inference-only: prep pre-staged on the login node.
missing=0
for ev in "${EVENTS[@]}"; do
  [[ -f "runs/xres/0p25/week${WEEKS}/inputs/${ev}_inputs.nc" ]] || { echo "  missing inputs: $ev"; missing=1; }
done
if [[ $missing -eq 1 || ! -f "runs/models/params/GenCast 0p25deg <2019.npz" ]]; then
  echo "ERROR: prep incomplete. On the LOGIN node run:" >&2
  echo "  XRES_EXTRA_EVENTS=\"\$(python fcn3/fevents.py --spec)\" XRES_EVENTS_SEL=$XRES_EVENTS_SEL \\" >&2
  echo "    python run_xres.py --stage prep --res 0p25 --weeks $WEEKS" >&2
  exit 1
fi

# Claims are PID-based and only valid within one job; clear leftovers for THESE events only
# (never a blanket wipe -- that would steal live claims if another xres job were running).
mkdir -p "$CACHE/claims"
for ev in "${EVENTS[@]}"; do rm -f "$CACHE"/claims/"${ev}"* 2>/dev/null || true; done

T0=$(date +%s)
JOBTAG=${SLURM_JOB_ID}
PIDS=()
for i in $(seq 0 $((NGPU-1))); do
  wlog="logs/xres_0p25pool_wk${WEEKS}_p90_${JOBTAG}_w$i.log"
  echo "[launcher] worker $i -> GPU $i -> $wlog"
  CUDA_VISIBLE_DEVICES=$i XRES_WORKER=$i \
    python run_xres.py --stage infer --res 0p25 --weeks "$WEEKS" > "$wlog" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for k in "${!PIDS[@]}"; do
  if wait "${PIDS[$k]}"; then echo "[launcher] worker $k ok"; else echo "[launcher] worker $k FAILED"; FAIL=1; fi
done
T1=$(date +%s)

built=0
for ev in "${EVENTS[@]}"; do [[ -f "$CACHE/${ev}_cube.nc" ]] && built=$((built+1)); done
echo "[status] cubes=$built/$NEV  workers-failed=$FAIL  wall=$(( (T1-T0)/60 )) min"

# Node-wall timing sidecar for the runtime comparison (only meaningful if ALL events were
# actually rolled this job -- a resubmit that skips cached cubes would report a bogus wall).
python - "$T0" "$T1" "$built" "$NEV" "$JOBTAG" <<'PY'
import json, sys, os
sys.path.insert(0, ".")
from fcn3 import fevents as F
t0, t1, built, nev, job = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
names = os.environ["XRES_EVENTS_SEL"].split(",")
wall_min = (t1 - t0) / 60.0
out = {
    "model": "gencast", "job_id": job, "weeks": F.WEEKS, "resolution": F.RES,
    "members": F.N_MEMBERS, "nsteps": F.GENCAST_NSTEPS, "gpus": 8,
    "events": names, "n_events": nev, "cubes_built_this_job": built,
    "seconds_total": t1 - t0, "minutes_total": round(wall_min, 2),
    "minutes_per_event": round(wall_min / nev, 2) if nev else None,
    "hardware": "a3mega 1 node x 8 H100-80GB",
    "complete": built == nev,
    "note": "Node wall time for the whole pool over n_events. Valid as a runtime "
            "measurement only when cubes_built_this_job == n_events; a resubmit that "
            "skipped cached cubes under-reports the cost.",
}
p = F.gencast_pool_timing_path()
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=2))
print(f"[timing] wrote {p}\n{json.dumps(out, indent=2)}")
PY

echo "End: $(date)"
[[ $FAIL -eq 0 ]]
