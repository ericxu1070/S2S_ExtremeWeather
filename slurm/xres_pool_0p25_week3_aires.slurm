#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# GenCast 0.25deg WEEK-3 infer for the AI+RES EXTENSION events that have no cached cube
# (fcn3/fevents.py::RES_EVENTS, source == "aires"; today: SCentral_HeatDome_2023). The
# cube is the OPTIONAL 24-member direct-sampling baseline for `run_aires --stage ds` --
# Gate 3's 16 walkers already satisfy that stage, so submit this only if the figures
# should carry the same three-row DS comparison the PNW pilot has.
#
# Identical worker-pool machinery to slurm/xres_pool_0p25_week3_p90.slurm, with two
# deliberate differences:
#
#   1. XRES_EXTRA_EVENTS comes from `fcn3/fevents.py --res-spec` (the additive hook;
#      the frozen 12 xres events and the p90 launcher's spec are untouched).
#   2. NO timing sidecar. The p90 launcher unconditionally overwrites
#      runs/fcn3/week3/timing/gencast_pool_timing.json -- the frozen runtime record of
#      the FCN3-vs-GenCast comparison -- which is exactly why it must not be reused for
#      other events. Wall time is echoed to the log and that is all.
#
# Expected cost: ~63 min/event for 24 members at this lead (job 223 measured); 4 h
# walltime is headroom. Resubmitting is cheap: cached members/cubes are skipped.
#
# GPU nodes are inference-only (CLAUDE.md): prep must be pre-staged on the login node with
#   XRES_EXTRA_EVENTS="$(python fcn3/fevents.py --res-spec)" \
#   XRES_EVENTS_SEL=SCentral_HeatDome_2023 \
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

# The extension events, injected and selected. Derived from fcn3/fevents.py so the event
# list has exactly one definition; xres-native extension events (source == "xres") need
# no injection and already have cubes.
export XRES_EXTRA_EVENTS="${XRES_EXTRA_EVENTS:-$(python fcn3/fevents.py --res-spec)}"
export XRES_EVENTS_SEL="${XRES_EVENTS_SEL:-$(python -c "
import sys; sys.path.insert(0,'.')
from fcn3 import fevents as F
print(','.join(e.name for e in F.res_events() if e.source=='aires'))")}"

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
  echo "  XRES_EXTRA_EVENTS=\"\$(python fcn3/fevents.py --res-spec)\" XRES_EVENTS_SEL=$XRES_EVENTS_SEL \\" >&2
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
  wlog="logs/xres_0p25pool_wk${WEEKS}_aires_${JOBTAG}_w$i.log"
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

echo "End: $(date)"
[[ $FAIL -eq 0 ]]
