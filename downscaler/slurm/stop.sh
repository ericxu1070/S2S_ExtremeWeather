#!/usr/bin/env bash
# Gracefully stop a running downscaler training job.
#
#   bash slurm/stop.sh            # ask the trainer to checkpoint and exit, then wait
#   bash slurm/stop.sh --no-wait  # just drop the sentinel and return
#
# This drops a STOP sentinel file in the checkpoint dir. The trainer polls for it at
# micro-step boundaries, saves a full checkpoint (model + optimizer + scheduler + EMA),
# deletes the sentinel, and exits 0 — the Slurm job then completes normally and the node
# goes back to the queue.
#
# There is NO deadline on this path, which is why it is preferred over `scancel`. Plain
# `scancel` sends SIGTERM and SIGKILLs 30 s later (KillWait); the trainer handles SIGTERM
# and will try to checkpoint, but a large save can lose that race. `scancel -s USR1 <jobid>`
# is an equally safe alternative — it signals without the kill timer.
#
# Restarting is just: sbatch slurm/train.sbatch   (it auto-resumes from latest.pt)
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/downscaler/checkpoints}"
JOB_NAME="${JOB_NAME:-downscaler}"
WAIT=1
[ "${1:-}" = "--no-wait" ] && WAIT=0

if [ ! -d "$CKPT_DIR" ]; then
    echo "error: checkpoint dir does not exist: $CKPT_DIR" >&2
    echo "       set CKPT_DIR=... if the run uses a different one." >&2
    exit 1
fi

JOBS=$(squeue -h -u "$USER" -n "$JOB_NAME" -o "%i %T" || true)
if [ -z "$JOBS" ]; then
    echo "No '$JOB_NAME' job in the queue — nothing to stop."
    exit 0
fi
echo "Found:"
echo "$JOBS" | sed 's/^/  job /'

touch "$CKPT_DIR/STOP"
echo "Dropped stop sentinel: $CKPT_DIR/STOP"
echo "The trainer will checkpoint at its next step boundary and exit."

# A PENDING job never runs the trainer, so it would never consume the sentinel — and the
# sentinel would then be sitting there waiting to kill the NEXT run. Cancel those outright.
PENDING=$(echo "$JOBS" | awk '$2=="PENDING"{print $1}')
if [ -n "$PENDING" ]; then
    echo "Cancelling PENDING jobs (they never started, so they cannot consume the sentinel):"
    for j in $PENDING; do echo "  scancel $j"; scancel "$j"; done
    # Nothing is running to eat the sentinel; remove it so the next sbatch is not killed.
    if [ -z "$(echo "$JOBS" | awk '$2=="RUNNING"{print $1}')" ]; then
        rm -f "$CKPT_DIR/STOP"
        echo "No RUNNING job — removed the sentinel again."
        exit 0
    fi
fi

if [ "$WAIT" = "0" ]; then
    exit 0
fi

echo -n "Waiting for the job to exit"
for _ in $(seq 1 120); do   # up to ~10 min; a save is seconds, this is slack
    if [ -z "$(squeue -h -u "$USER" -n "$JOB_NAME" -o "%i" || true)" ]; then
        echo " done."
        LATEST="$CKPT_DIR/latest.pt"
        [ -f "$LATEST" ] && echo "Checkpoint: $LATEST ($(date -r "$LATEST" '+%Y-%m-%d %H:%M:%S'))"
        echo "Resume with: sbatch slurm/train.sbatch"
        exit 0
    fi
    echo -n "."
    sleep 5
done

echo
echo "warning: job still in the queue after 10 min. Check the log, then if needed:" >&2
echo "  scancel -s USR1 <jobid>    # signal-based stop, no kill timer" >&2
echo "  scancel <jobid>            # hard stop (SIGTERM, 30 s to save)" >&2
exit 1
