#!/usr/bin/env bash
# =============================================================================
# 01_submit.sh - submit the FourCastNet3 GPU pool (a3mega). Preflights the
# caches, refuses to double-submit, then sbatches slurm/p90_fcn3_pool.slurm.
#
#   bash p90/bash/01_submit.sh            # 1 node (all 90 cases)
#   NODES=2 bash p90/bash/01_submit.sh    # 2 nodes via a job array (0-1)
#   DRY_RUN=1 bash p90/bash/01_submit.sh  # print the sbatch line, submit nothing
#
# Run p90/bash/00_prep.sh on the login node first - the pool needs truth + IC.
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DRY_RUN="${DRY_RUN:-0}"
NODES="${NODES:-1}"

run() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN - would run: $*"; else "$@"; fi; }

command -v sbatch >/dev/null || { echo "FAIL: no sbatch here - run this on the a3mega login node"; exit 1; }

# ------------------------------------------------------------- resolve paths
activate_moe
eval "$(python - <<'PY'
import pandas as pd
from p90 import pconfig as P
print(f"IC_DIR={P.ic_dir()}")
print(f"CKPT_CACHE={P.FCN3_CACHE / 'e2s'}")
print(f"NCASES={len(pd.read_csv(P.CASES_CSV))}")
PY
)"
[ -n "${NCASES:-}" ] || { echo "FAIL: could not read cases via pconfig"; exit 1; }

# ------------------------------------------------------------- preflight
[ -f "$PROJ/p90/cases.csv" ] || { echo "FAIL: missing $PROJ/p90/cases.csv"; exit 1; }
IC_N=$(ls "$IC_DIR"/*_ic.nc 2>/dev/null | wc -l)
[ "$IC_N" -eq "$NCASES" ] || { echo "FAIL: IC count $IC_N != case count $NCASES - run p90/bash/00_prep.sh"; exit 1; }
CKPT=$(find "$CKPT_CACHE" -name best_ckpt_mp0.tar 2>/dev/null | head -1)
[ -n "$CKPT" ] || { echo "FAIL: FCN3 checkpoint best_ckpt_mp0.tar not found under $CKPT_CACHE"; exit 1; }
echo "[preflight] cases=$NCASES ic=$IC_N ckpt=$CKPT"

# ------------------------------------------------------------- no leaked filters
# sbatch --export=ALL carries these into the 8-GPU job; a value exported during a
# smoke/shard test would silently shrink the pool below the full 90 cases. Refuse HERE,
# before the (scarce) allocation. P90_SHARD is set by the array path inside the slurm.
for v in P90_SHARD P90_KIND P90_CASES_SEL P90_MAX_CASES; do
    if [ -n "${!v:-}" ]; then
        echo "FAIL: $v=${!v} is set in this shell; sbatch --export=ALL would carry it into"
        echo "the pool and shrink it below the full 90 cases. 'unset $v' and resubmit."
        exit 1
    fi
done

# ------------------------------------------------------------- no double-submit
ACTIVE=$(squeue -h -u "$USER" -n p90_fcn3_pool -o "%i %T" 2>/dev/null || true)
if [ -n "$ACTIVE" ]; then
    echo "a 'p90_fcn3_pool' job is already queued/running:"
    echo "$ACTIVE" | sed 's/^/  job /'
    echo "watch it (p90/bash/02_status.sh) or cancel it before resubmitting."
    exit 1
fi

# ------------------------------------------------------------- submit
SLURM="$PROJ/slurm/p90_fcn3_pool.slurm"
if [ "$NODES" -le 1 ]; then
    echo "submitting 1 node: $SLURM"
    run sbatch "$SLURM"
else
    echo "submitting $NODES nodes (array 0-$((NODES-1))): $SLURM"
    run sbatch --array=0-$((NODES-1)) "$SLURM"
fi

echo
echo "watch:  bash p90/bash/02_status.sh"
echo "logs:   tail -f $PROJ/logs/p90_fcn3_<jobid>_w0.log"
