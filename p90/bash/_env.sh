#!/usr/bin/env bash
# =============================================================================
# _env.sh - sourced helper for the p90 operator scripts. Detects the machine
# and exposes the two Python environments this experiment needs:
#
#   MOE env  - my-env (Derecho) / moe (a3mega): truth prep, compare, plotting.
#   FCN3 env - the earth2studio/FourCastNet3 interpreter: prep-ic and infer.
#
# Source it, do not execute it:  source "$(dirname "$0")/_env.sh"
# Sets PROJ (repo root) and exports PYTHONPATH="$PROJ". Provides:
#   activate_moe   - activate the MOE conda env in the current shell.
#   fcn3_python    - echo the FCN3 interpreter path (no activation, no side effects).
#
# Overridable: MOE_ENV, FCN3_ENV (a3mega env name), FCN3_PY (explicit interpreter).
# =============================================================================

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

if [ -d /glade ]; then
    # Derecho (NCAR, PBS, module-provided conda).
    P90_MACHINE="derecho"
    MOE_ENV="${MOE_ENV:-my-env}"
    FCN3_PY="${FCN3_PY:-/glade/derecho/scratch/exu/conda-envs/fcn3/bin/python}"
else
    # a3mega (GCP Slurm, miniconda).
    P90_MACHINE="a3mega"
    MOE_ENV="${MOE_ENV:-moe}"
    FCN3_ENV="${FCN3_ENV:-fcn3}"
fi
export P90_MACHINE MOE_ENV

activate_moe() {
    if [ "$P90_MACHINE" = "derecho" ]; then
        module load conda >/dev/null 2>&1
    else
        source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
    fi
    conda activate "$MOE_ENV"
}

fcn3_python() {
    # Echo the FCN3 interpreter path. An explicit FCN3_PY always wins.
    if [ -n "${FCN3_PY:-}" ]; then
        echo "$FCN3_PY"
        return 0
    fi
    # a3mega: resolve via the conda base + env name.
    local base
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh 2>/dev/null
    base="$(conda info --base 2>/dev/null)"
    echo "${base}/envs/${FCN3_ENV}/bin/python"
}
