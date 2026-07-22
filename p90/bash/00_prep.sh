#!/usr/bin/env bash
# =============================================================================
# 00_prep.sh - login-node prep for the p90 T2m experiment. Run this ONCE before
# submitting the GPU pool; it fills the two model-agnostic caches:
#
#   +-----------------------------------------------------------------------+
#   | (a) TRUTH  -> MOE env (my-env / moe), internet (ARCO/WB2)             |
#   |     ERA5 verification + persistence T2m anomalies, 2 files/case.      |
#   | (b) IC     -> FCN3 env (earth2studio interpreter), internet          |
#   |     FourCastNet3 initial-condition frames, 1 file/case.              |
#   +-----------------------------------------------------------------------+
#
# Both stages are cache-aware and skipped when already complete.
#   bash p90/bash/00_prep.sh          # run prep
#   DRY_RUN=1 bash p90/bash/00_prep.sh  # print the commands, run nothing
#   P90_PREP_PBS=1 bash p90/bash/00_prep.sh  # (Derecho) print the qsub hint, exit
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DRY_RUN="${DRY_RUN:-0}"

run() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN - would run: $*"; else "$@"; fi; }

# ------------------------------------------------------------- resolve paths
activate_moe
eval "$(python - <<'PY'
import pandas as pd
from p90 import pconfig as P
print(f"TRUTH_DIR={P.truth_dir()}")
print(f"IC_DIR={P.ic_dir()}")
print(f"NCASES={len(pd.read_csv(P.CASES_CSV))}")
PY
)"
[ -n "${NCASES:-}" ] || { echo "FAIL: could not read cases via pconfig"; exit 1; }
TRUTH_EXP=$((NCASES * 2))
echo "cases=$NCASES  expect truth=$TRUTH_EXP (verif+persist)  ic=$NCASES"

# ----------------------------------------------------- Derecho ARCO warning
if [ "$P90_MACHINE" = "derecho" ]; then
    echo
    echo "NOTE (Derecho): truth prep is ARCO-heavy-ish - per-case ERA5 reads with"
    echo "release_stores, roughly 1-3 min/case over ~$NCASES cases. Login-node OK,"
    echo "but the PBS 'develop' (cpudev) queue is the fallback if it gets throttled."
    if [ "${P90_PREP_PBS:-0}" = 1 ]; then
        echo
        echo "P90_PREP_PBS=1 set - submit prep to the develop queue instead, e.g.:"
        echo "  qsub -q develop -A UCSC0009 -l select=1:ncpus=8:mem=80gb -l walltime=04:00:00 \\"
        echo "       -- \$(command -v bash) $PROJ/p90/bash/00_prep.sh"
        exit 0
    fi
fi

# --------------------------------------------------------- (a) truth (MOE)
echo
TRUTH_N=$(ls "$TRUTH_DIR"/*.nc 2>/dev/null | wc -l)
if [ "$TRUTH_N" -ge "$TRUTH_EXP" ]; then
    echo "[truth] complete ($TRUTH_N/$TRUTH_EXP .nc) - skip"
else
    echo "[truth] $TRUTH_N/$TRUTH_EXP present - building (MOE env)"
    run python "$PROJ/p90/run_p90.py" --stage prep-truth
    TRUTH_N=$(ls "$TRUTH_DIR"/*.nc 2>/dev/null | wc -l)
fi

# --------------------------------------------------------- (b) IC (FCN3)
echo
IC_N=$(ls "$IC_DIR"/*_ic.nc 2>/dev/null | wc -l)
if [ "$IC_N" -ge "$NCASES" ]; then
    echo "[ic] complete ($IC_N/$NCASES) - skip"
else
    FCN3_PY="$(fcn3_python)"
    echo "[ic] $IC_N/$NCASES present - building (FCN3 env: $FCN3_PY)"
    run "$FCN3_PY" "$PROJ/p90/run_p90.py" --stage prep-ic --model fcn3
    IC_N=$(ls "$IC_DIR"/*_ic.nc 2>/dev/null | wc -l)
fi

echo
echo "==== prep summary ===="
echo "  truth: $TRUTH_N/$TRUTH_EXP .nc"
echo "  ic:    $IC_N/$NCASES"
echo "next: p90/bash/01_submit.sh (a3mega)"
