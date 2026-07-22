#!/usr/bin/env bash
# =============================================================================
# 02_status.sh - one-glance progress for the p90 FCN3 pool. Read-only, safe
# anywhere, anytime; never exits nonzero.
#
#   bash p90/bash/02_status.sh
#
# Shows: queued/running pool jobs, cube/timing/claim counts vs 90, truth + IC
# counts, the newest worker log tail, and total GPU-time from the timing jsons.
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

echo "=============== run state ==============="
if command -v squeue >/dev/null; then
    JOBS=$(squeue -h -u "$USER" -n p90_fcn3_pool -o "%i %T %M %R" 2>/dev/null || true)
    if [ -n "$JOBS" ]; then echo "$JOBS" | sed 's/^/  slurm job /'; else echo "  no p90_fcn3_pool job in the queue"; fi
else
    echo "  (no squeue here - not the a3mega login node)"
fi

# resolve paths (best effort; never fatal). `|| true` cannot rescue a `set -u`
# unbound-variable abort raised inside conda's activation functions, so drop -u around
# it (conda's own recommendation) to keep this script's never-nonzero contract.
set +u; activate_moe 2>/dev/null || true; set -u
eval "$(python - 2>/dev/null <<'PY'
from p90 import pconfig as P
print(f"TRUTH_DIR={P.truth_dir()}")
print(f"IC_DIR={P.ic_dir()}")
print(f"CACHE={P.cache_dir('fcn3')}")
print(f"CLAIMS={P.claims_dir('fcn3')}")
print(f"TIMING={P.timing_dir('fcn3')}")
PY
)"

echo
echo "=============== progress (target 90) ==============="
CUBES=$(ls "${CACHE:-/nonexistent}"/*_cube.nc 2>/dev/null | wc -l)
TIMES=$(ls "${TIMING:-/nonexistent}"/*_timing.json 2>/dev/null | wc -l)
CLAIMS=$(ls "${CLAIMS:-/nonexistent}"/* 2>/dev/null | wc -l)
TRUTH=$(ls "${TRUTH_DIR:-/nonexistent}"/*.nc 2>/dev/null | wc -l)
IC=$(ls "${IC_DIR:-/nonexistent}"/*_ic.nc 2>/dev/null | wc -l)
echo "  cubes:  $CUBES/90"
echo "  timing: $TIMES/90"
echo "  claims: $CLAIMS (in-flight)"
echo "  truth:  $TRUTH/180 (verif+persist)   ic: $IC/90"

echo
echo "=============== total GPU-time (timing jsons) ==============="
if [ "$TIMES" -gt 0 ]; then
    python - "$TIMING" <<'PY' 2>/dev/null || echo "  (could not parse timing jsons)"
import glob, json, sys
tot, n = 0.0, 0
for f in glob.glob(sys.argv[1] + "/*_timing.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    # sum any top-level numeric duration-ish field
    for k, v in (d.items() if isinstance(d, dict) else []):
        if isinstance(v, (int, float)) and any(t in k.lower() for t in ("sec", "elapsed", "wall", "dur")):
            tot += float(v); n += 1; break
print(f"  {n} cases timed, total {tot/60:.1f} GPU-min ({tot/3600:.2f} GPU-h)")
PY
else
    echo "  none yet"
fi

echo
echo "=============== newest worker log ==============="
LOG=$(ls -t "$PROJ"/logs/p90_fcn3_*_w*.log 2>/dev/null | head -1)
if [ -n "${LOG:-}" ]; then
    echo "  $LOG"
    tail -3 "$LOG" 2>/dev/null | sed 's/^/  /'
else
    echo "  no worker logs yet"
fi
exit 0
