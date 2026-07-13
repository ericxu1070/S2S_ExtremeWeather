#!/usr/bin/env bash
# =============================================================================
# 04_stop.sh — gracefully stop training and give the GPUs back.
#
#   bash bash/04_stop.sh
#
# Drops a STOP sentinel in the checkpoint dir; the trainer notices it at the
# next step boundary, writes a full checkpoint (model + optimizer + scheduler +
# EMA), consumes the sentinel, and exits 0. There is NO deadline on this path —
# nothing is killed — and at most save_every_n_steps (100) optimizer steps of
# work are redone on resume. Resume with: bash bash/03_train.sh
#
# Handles both run modes: a Slurm job (delegates to slurm/stop.sh, which also
# cancels PENDING jobs) or a local torchrun (waits on logs/train.pid).
# =============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-$PROJ/checkpoints}"
PIDFILE="$PROJ/logs/train.pid"

# ---------------------------------------------------------------- Slurm run?
if command -v squeue >/dev/null; then
    if [ -n "$(squeue -h -u "$USER" -n downscaler -o "%i" 2>/dev/null || true)" ]; then
        exec bash "$PROJ/slurm/stop.sh"
    fi
fi

# ---------------------------------------------------------------- local run?
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    PID=$(cat "$PIDFILE")
    touch "$CKPT_DIR/STOP"
    echo "dropped stop sentinel: $CKPT_DIR/STOP"
    echo -n "waiting for pid $PID to checkpoint and exit"
    for _ in $(seq 1 120); do   # up to ~10 min; a save is seconds, this is slack
        if ! kill -0 "$PID" 2>/dev/null; then
            echo " done."
            rm -f "$PIDFILE"
            [ -f "$CKPT_DIR/latest.pt" ] && \
                echo "checkpoint: $CKPT_DIR/latest.pt ($(date -r "$CKPT_DIR/latest.pt" '+%Y-%m-%d %H:%M:%S'))"
            echo "resume with: bash bash/03_train.sh"
            exit 0
        fi
        echo -n "."
        sleep 5
    done
    echo
    echo "warning: still running after 10 min — check the newest logs/train_*.log." >&2
    echo "         last resort: kill $PID   (the trainer catches SIGTERM and checkpoints)" >&2
    exit 1
fi

echo "nothing to stop — no Slurm 'downscaler' job and no local run (logs/train.pid)."
# A sentinel with no consumer would kill the NEXT run at startup; the trainer
# clears stale ones itself, but tidy up anyway if we left one behind.
if [ -f "$CKPT_DIR/STOP" ]; then
    rm -f "$CKPT_DIR/STOP"
    echo "removed a stale STOP sentinel from $CKPT_DIR."
fi
exit 0
