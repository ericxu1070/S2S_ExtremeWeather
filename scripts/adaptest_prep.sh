#!/usr/bin/env bash
# =============================================================================
# adaptest_prep.sh - the CPU half of the ten-case adapter test.
#
# RUN THIS ON A LOGIN NODE, never inside a GPU allocation. Everything here is
# CPU, network and cache work: fetching ERA5, writing init frames and observed
# truth, and writing the adapter initial conditions. A whole 8xH100 node doing
# this is pure waste, and by the time slurm/aires_adaptest.slurm starts, every
# input it needs must already exist so the job is a pure cache-hit-then-compute.
#
# Three things get built, in order, because each needs the last:
#
#   1. xres init frames + ERA5 truth   for the null controls (moe env, internet)
#      -- the six extreme events already have theirs from earlier experiments.
#   2. native FCN3 ICs                 for the null controls (fcn3 env, internet)
#      -- the adapter IC is verified channel-by-channel against this one, so it
#         is a hard prerequisite, not just an input to the rollout.
#   3. adapter ICs                     for every case (moe env, no network)
#
# Steps 1 and 2 are no-ops once their files exist, and step 2 runs in the OTHER
# conda env -- FCN3 needs earth2studio, which moe does not carry.
#
#   bash scripts/adaptest_prep.sh              # plan, build everything, plan again
#   bash scripts/adaptest_prep.sh --check      # read-only readiness matrix
#   bash scripts/adaptest_prep.sh --force      # rebuild cached adapter ICs
#   AIRES_ADAPTEST_EVENTS=WinterStorm_Uri_2021 bash scripts/adaptest_prep.sh
#   AIRES_ADAPTEST_EVENTS=null bash scripts/adaptest_prep.sh   # the controls only
#
# Works on either box. The FCN3 rollout does NOT: it needs ~64 GB of VRAM and
# therefore an 80 GB H100, so Derecho's A100-40GB cannot produce an adapter
# cube (confirmed, job 6857923). On Derecho this script still does something
# useful - it builds the ICs, which then travel with
# scripts/sync_a3mega_to_derecho.sh in reverse, or it re-scores cubes that came
# the other way.
#
# Overridable:
#   CONDA_ENV   conda env to activate   (default: moe on a3mega, my-env on Derecho)
#   REPO        repo root               (default: two levels up from this script)
# =============================================================================
set -euo pipefail

REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$REPO"

MODE=run
FORCE=""
for a in "$@"; do
  case "$a" in
    --check)  MODE=check ;;
    --force)  FORCE="--force" ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag $a (try --help)" >&2; exit 2 ;;
  esac
done

# --- environment ------------------------------------------------------------
# The two clusters name the same env differently and reach conda differently.
if [[ -z "${CONDA_ENV:-}" ]]; then
  if [[ -d /glade ]]; then CONDA_ENV=my-env; else CONDA_ENV=moe; fi
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  set +u                      # conda's activate.d hooks are not `set -u` clean
  if [[ -f /home/ubuntu/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  elif command -v module >/dev/null 2>&1; then
    module load conda >/dev/null 2>&1 || true
  fi
  conda activate "$CONDA_ENV"
  set -u
fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "host    $(hostname)"
echo "repo    $REPO"
echo "env     ${CONDA_DEFAULT_ENV:-unknown}"
echo "events  ${AIRES_ADAPTEST_EVENTS:-all ten cases}"
echo

# --- the calibration is the one input prep cannot build itself --------------
CALIB="runs/aires/calib/derived_calib.nc"
if [[ ! -f "$CALIB" ]]; then
  echo "ERROR: missing $CALIB" >&2
  echo "  The adapter derives u100m/v100m/tcwv from it. Build it first (~2 min, CPU):" >&2
  echo "    PYTHONPATH=. python -m aires.calibrate --fit" >&2
  exit 1
fi

# --- the climatology grid trap ---------------------------------------------
# Two files with this name have existed, on the same 0.25 degree filename but on
# different grids (105x237 here, 14x30 at ~2 degrees on the other box). Scoring
# subtracts a climatology sampled on that grid, so the wrong file yields either a
# shape hard-fail or silently wrong anomalies. Verify the grid, not the filename.
# Read the dims with xarray, NOT ncdump: ncdump is not installed in the `moe` env on
# a3mega, and under `set -euo pipefail` the old `ncdump | tr | grep | tr` pipeline exited
# non-zero with EMPTY output, killing this script silently before it built anything. A
# guard that dies without saying why is worse than no guard.
CLIM="runs/models/clim_1990_2019_t2m_conus.nc"
if [[ -f "$CLIM" ]]; then
  if ! python "$REPO/scripts/check_clim_grid.py" "$CLIM"; then
    exit 1
  fi
else
  echo "WARNING: $CLIM absent; the score stage will need it." >&2
fi
echo

# --- readiness --------------------------------------------------------------
echo "=== readiness before prep ==="
python -m aires.adaptest --stage plan || true

if [[ "$MODE" == check ]]; then
  echo
  echo "--check: nothing written."
  exit 0
fi

# --- 1. the null controls' xres init frames + ERA5 truth --------------------
# The six extreme events inherit theirs from the xres / p90 experiments. The
# controls are new cases, so xres has to be TOLD about them: XRES_EXTRA_EVENTS
# is the additive injection hook (it never overwrites an existing name) and
# XRES_EVENTS_SEL restricts this run to exactly the controls, leaving the 12
# original xres events untouched. Both are derived from fcn3/fevents.py so the
# case list has one definition.
CTL=$(python fcn3/fevents.py --controls)
CTL_SPEC=$(python fcn3/fevents.py --controls-spec)
NEED_XRES=""
for e in ${CTL//,/ }; do
  [[ -f "runs/xres/0p25/week3/inputs/${e}_inputs.nc" && \
     -f "runs/observations/${e}_verif_t2m_anom.nc" ]] || NEED_XRES="${NEED_XRES}${e},"
done
if [[ -n "$NEED_XRES" ]]; then
  echo
  echo "=== 1/3 xres init frames + ERA5 truth for ${NEED_XRES%,} (CPU, internet) ==="
  echo "    this reads ARCO ERA5 and takes ~10 min per case"
  # ONE EVENT PER PROCESS, deliberately. run_xres.py has its own per-event subprocess
  # isolation, but it is gated on `isolate and not skip_models` -- so passing
  # XRES_PREP_SKIP_MODELS=1 (which we want: the checkpoints are already on disk) silently
  # turns the isolation OFF and builds every event in one process. On this 15 GB login
  # node that OOMs. Looping here gets the isolation back and a readable per-case log.
  # --res 0p25 because the adapter reads the 0.25 deg init frames and nothing here needs
  # the 1.0 deg ones.
  for e in ${NEED_XRES//,/ }; do
    echo "--- $e"
    XRES_EXTRA_EVENTS="$CTL_SPEC" XRES_EVENTS_SEL="$e" \
      XRES_PREP_SKIP_MODELS=1 XRES_PREP_ISOLATE=0 PYTHONUNBUFFERED=1 \
      python run_xres.py --stage prep --weeks 3 --res 0p25
  done
else
  echo
  echo "=== 1/3 xres init frames + ERA5 truth: all present ==="
fi

# --- 2. the null controls' native FCN3 ICs ----------------------------------
# Different conda env (earth2studio lives in fcn3, not moe), so this runs in a
# subshell rather than switching the env under the rest of the script.
NEED_FCN3=""
for e in ${CTL//,/ }; do
  [[ -f "runs/fcn3/week3/ic/${e}_ic.nc" ]] || NEED_FCN3="${NEED_FCN3}${e},"
done
if [[ -n "$NEED_FCN3" ]]; then
  echo
  echo "=== 2/3 native FCN3 ICs for ${NEED_FCN3%,} (CPU, internet, fcn3 env) ==="
  (
    set -e
    # `set -u` OFF across the conda activation: the fcn3 env ships an activate.d hook
    # (~cuda-nvcc_activate.sh) that reads NVCC_PREPEND_FLAGS before setting it, which is
    # fatal under the `set -euo pipefail` this script runs with. Conda's activation
    # scripts are not -u clean in general; only OUR code is held to that.
    set +u
    # shellcheck disable=SC1091
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
    conda activate "${FCN3_CONDA_ENV:-fcn3}"
    set -u
    export EARTH2STUDIO_CACHE="$REPO/runs/fcn3/.cache/e2s"
    export HF_HOME="$REPO/runs/fcn3/.cache/hf"
    export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
    FCN3_EVENTS="${NEED_FCN3%,}" FCN3_WEEKS=3 python fcn3/run_fcn3.py --stage prep
  )
else
  echo
  echo "=== 2/3 native FCN3 ICs: all present ==="
fi

# --- 3. build the adapter ICs -----------------------------------------------
echo
echo "=== 3/3 building adapter ICs (CPU, no network) ==="
python -m aires.adaptest --stage prep $FORCE

echo
echo "=== readiness after prep ==="
python -m aires.adaptest --stage plan || true

cat <<'EOF'

Next, on a3mega (the FCN3 rollout needs an 80 GB H100):
  sbatch slurm/aires_adaptest.slurm            # all ten; already-built cubes exit early
  sbatch --array=6-9%2 slurm/aires_adaptest.slurm   # the null controls only
  squeue -u $USER -n Vayuh-s2s

Then back on a login node in the GenCast env:
  PYTHONPATH=. python -m aires.adaptest --stage score
  PYTHONPATH=. python -m aires.adaptest --stage report
  PYTHONPATH=. python -m aires.adaptest --stage maps
EOF
