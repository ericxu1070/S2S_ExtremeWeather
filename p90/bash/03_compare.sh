#!/usr/bin/env bash
# =============================================================================
# 03_compare.sh - score the cubes and build figures (MOE env, offline). Runs
# after the pool finishes; cache-aware, so partial cubes still score.
#
#   bash p90/bash/03_compare.sh              # model fcn3 (default)
#   MODEL=gencast bash p90/bash/03_compare.sh  # once gencast cubes exist
#   DRY_RUN=1 bash p90/bash/03_compare.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DRY_RUN="${DRY_RUN:-0}"
MODEL="${MODEL:-fcn3}"

run() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN - would run: $*"; else "$@"; fi; }

activate_moe

echo "[compare] model=$MODEL"
run python "$PROJ/p90/run_p90.py" --stage compare --model "$MODEL"

# ------------------------------------------------------------- report outputs
eval "$(python - "$MODEL" <<'PY'
import sys
from p90 import pconfig as P
m = sys.argv[1]
print(f"FIG_DIR={P.fig_dir(m)}")
print(f"SCORES_DIR={P.scores_dir(m)}")
PY
)"

echo
echo "=============== figures ($FIG_DIR) ==============="
ls -1 "$FIG_DIR" 2>/dev/null | sed 's/^/  /' || echo "  (none)"

SUMMARY=$(find "$FIG_DIR" "$SCORES_DIR" -name summary.md 2>/dev/null | head -1)
echo
if [ -n "$SUMMARY" ]; then
    echo "summary: $SUMMARY"
else
    echo "summary: (not found; expected under $FIG_DIR or $SCORES_DIR)"
fi
