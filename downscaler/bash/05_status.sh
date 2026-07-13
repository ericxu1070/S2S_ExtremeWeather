#!/usr/bin/env bash
# =============================================================================
# 05_status.sh — one-glance health check. Read-only, safe anywhere, anytime.
#
#   bash bash/05_status.sh
#
# Shows: whether a run is active (Slurm or local), the latest checkpoint and
# its step, recent heartbeat lines from the newest log, and whether the data
# prerequisites exist.
# =============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-$PROJ/checkpoints}"
PIDFILE="$PROJ/logs/train.pid"

echo "=============== run state ==============="
ACTIVE=0
if command -v squeue >/dev/null; then
    JOBS=$(squeue -h -u "$USER" -n downscaler -o "%i %T %M %R" 2>/dev/null || true)
    if [ -n "$JOBS" ]; then
        echo "$JOBS" | sed 's/^/  slurm job /'
        ACTIVE=1
    fi
fi
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "  local run: pid $(cat "$PIDFILE")"
    ACTIVE=1
fi
[ "$ACTIVE" = 0 ] && echo "  not running (start/resume: bash bash/03_train.sh)"

echo
echo "=============== checkpoints ($CKPT_DIR) ==============="
NEWEST=$(ls "$CKPT_DIR"/checkpoint_step*.pt 2>/dev/null | sort | tail -1)
if [ -n "${NEWEST:-}" ]; then
    STEP=$(basename "$NEWEST" | sed 's/checkpoint_step0*\([0-9]\)/\1/; s/\.pt//')
    echo "  step $STEP — $(basename "$NEWEST")"
    [ -f "$CKPT_DIR/latest.pt" ] && \
        echo "  latest.pt updated $(date -r "$CKPT_DIR/latest.pt" '+%Y-%m-%d %H:%M:%S')"
    du -sh "$CKPT_DIR" 2>/dev/null | awk '{print "  disk: "$1}'
else
    echo "  none yet — training has not saved a checkpoint"
fi

echo
echo "=============== recent progress ==============="
LOG=$(ls -t "$PROJ"/logs/train_*.log "$PROJ"/logs/slurm-downscaler-*.out 2>/dev/null | head -1)
if [ -n "${LOG:-}" ]; then
    echo "  newest log: $LOG"
    HB=$(grep "\[hb\] step=" "$LOG" 2>/dev/null | tail -3)
    if [ -n "$HB" ]; then
        echo "$HB" | sed 's/^/  /'
    else
        tail -3 "$LOG" | sed 's/^/  /'
    fi
else
    echo "  no training logs yet"
fi

echo
echo "=============== prerequisites ==============="
readarray -t CFG < <(cd "$PROJ" && source /home/ubuntu/miniconda3/etc/profile.d/conda.sh \
    && conda activate moe 2>/dev/null && python - <<'PY'
from omegaconf import OmegaConf
c = OmegaConf.load("configs/data/era5_hrrr.yaml")
print(c.index_path); print(c.stats_path)
PY
)
INDEX="${CFG[0]:-}"; STATS="${CFG[1]:-}"
if [ -s "${INDEX:-/nonexistent}" ]; then
    echo "  index: $(wc -l < "$INDEX") timestamps"
else
    echo "  index: MISSING — run bash/02_build_index_and_stats.sh"
fi
if [ -f "${STATS:-/nonexistent}" ]; then
    echo "  stats: $STATS"
else
    echo "  stats: MISSING — run bash/02_build_index_and_stats.sh"
fi
