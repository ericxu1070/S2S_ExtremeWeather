#!/usr/bin/env bash
# =============================================================================
# adaptest_prep.sh - the CPU half of the six-event adapter test.
#
# RUN THIS ON A LOGIN NODE, never inside a GPU allocation. Everything here is
# CPU and cache work: reading ERA5 init frames off NFS and writing the six
# adapter initial conditions. A whole 8xH100 node doing this is pure waste, and
# by the time slurm/aires_adaptest.slurm starts, every input it needs must
# already exist so the job is a pure cache-hit-then-compute.
#
#   bash scripts/adaptest_prep.sh              # plan, prep, plan again
#   bash scripts/adaptest_prep.sh --check      # read-only readiness matrix
#   bash scripts/adaptest_prep.sh --force      # rebuild cached adapter ICs
#   AIRES_ADAPTEST_EVENTS=WinterStorm_Uri_2021 bash scripts/adaptest_prep.sh
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
  if [[ -f /home/ubuntu/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  elif command -v module >/dev/null 2>&1; then
    module load conda >/dev/null 2>&1 || true
  fi
  conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "host    $(hostname)"
echo "repo    $REPO"
echo "env     ${CONDA_DEFAULT_ENV:-unknown}"
echo "events  ${AIRES_ADAPTEST_EVENTS:-all six}"
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
CLIM="runs/models/clim_1990_2019_t2m_conus.nc"
if [[ -f "$CLIM" ]]; then
  DIMS=$(ncdump -h "$CLIM" 2>/dev/null | tr -d '\t ' | grep -E '^(lat|lon)=' | tr '\n' ' ')
  echo "climatology grid: $DIMS"
  if ! grep -q 'lat=105' <<<"$DIMS" || ! grep -q 'lon=237' <<<"$DIMS"; then
    echo "ERROR: $CLIM is not on the 0.25 degree grid (expected lat=105 lon=237)." >&2
    echo "  Every anomaly downstream would be silently wrong. Fix this first." >&2
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

# --- build the adapter ICs --------------------------------------------------
echo
echo "=== building adapter ICs (CPU) ==="
python -m aires.adaptest --stage prep $FORCE

echo
echo "=== readiness after prep ==="
python -m aires.adaptest --stage plan || true

cat <<'EOF'

Next, on a3mega (the FCN3 rollout needs an 80 GB H100):
  sbatch slurm/aires_adaptest.slurm            # all six; already-built cubes exit early
  squeue -u $USER -n Vayuh-s2s

Then back on a login node in the GenCast env:
  PYTHONPATH=. python -m aires.adaptest --stage score
  PYTHONPATH=. python -m aires.adaptest --stage report
EOF
