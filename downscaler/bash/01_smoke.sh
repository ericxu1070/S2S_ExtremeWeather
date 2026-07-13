#!/usr/bin/env bash
# =============================================================================
# 01_smoke.sh — verify the code and the stop/resume harness. No GPU used.
#
# Plain-bash port of slurm/smoke.sbatch, so it runs anywhere the moe env exists
# (login node included, ~2-3 min). Deliberately CPU-only: it will never touch a
# GPU that is busy with someone else's job.
#
#   bash bash/01_smoke.sh
#
# What it proves, on a tiny dummy model:
#   1. environment  — moe env + torch + project code
#   2. pytest       — the unit/smoke suite (incl. the transformer bottleneck)
#   3. train        — a fresh run checkpoints mid-epoch; latest.pt is a hardlink
#   4. resume       — a second run picks up at the exact step the first stopped at
#   5. STOP         — the stop sentinel makes it checkpoint, clean up, and exit 0
#
# Steps 3-5 are the start/stop/resume cycle real training relies on. Passing
# means the harness is sound; it says nothing about the science (random tensors).
# Scratch checkpoints go to checkpoints/_smoke — never the real latest.pt.
# =============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$PROJ/checkpoints/_smoke"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/downscaler_smoke.XXXXXX")"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""      # CPU-only on purpose: never disturb a busy GPU
export HDF5_USE_FILE_LOCKING=FALSE  # NFS
export TMPDIR=/tmp                  # node-local: avoids NFS .nfsXXXX silly-rename churn

# Tiny model + tiny grid: this is a plumbing test, so make it cheap.
SMOKE=(
  data.use_dummy=true
  data.grid_height=24 data.grid_width=32 data.num_workers=0
  model.C_base=16 'model.n_res_blocks=[1,1]' model.num_groups=8
  logging.enabled=false
  "ckpt_dir=$CKPT"
)

FAILS=0
ok()   { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; FAILS=$((FAILS + 1)); }
chk()  { if [ "$1" = 0 ]; then ok "$2"; else fail "$2"; fi; }
hdr()  { echo; echo "=============== $1 ==============="; }

cd "$PROJ" || exit 1
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

# ---------------------------------------------------------------- 1. environment
hdr "1. environment"
echo "  host=$(hostname)  cpus=$(nproc)"
python - <<'PY'
import torch, sys
print(f"  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"  python={sys.version.split()[0]}")
PY
chk $? "project + moe env + torch import"

rm -rf "$CKPT"

# ---------------------------------------------------------------- 2. unit tests
hdr "2. pytest"
python -m pytest tests/ -q 2>&1 | tail -3
chk "${PIPESTATUS[0]}" "pytest tests/"

# ---------------------------------------------------------------- 3. fresh train
hdr "3. fresh train (4 steps, checkpoint every 2)"
python train.py "${SMOKE[@]}" training.max_steps=4 training.n_epochs=1 \
    training.save_every_n_steps=2 > "$TMP/train.log" 2>&1
chk $? "train.py exits 0"
grep -E "Saved checkpoint|Reached max_steps" "$TMP/train.log" | sed 's/^.*trainer\]//' | sed 's/^/    /'

[ -f "$CKPT/latest.pt" ]; chk $? "latest.pt written"
[ -f "$CKPT/checkpoint_step00000004.pt" ]; chk $? "step-4 snapshot written"
LINKS=$(stat -c %h "$CKPT/latest.pt" 2>/dev/null || echo 0)
[ "$LINKS" -ge 2 ]; chk $? "latest.pt is a hardlink, not a copy (nlink=$LINKS)"

# ---------------------------------------------------------------- 4. resume
hdr "4. resume (should pick up at step 4, run to 8)"
python train.py "${SMOKE[@]}" training.max_steps=8 training.n_epochs=1 \
    training.save_every_n_steps=2 > "$TMP/resume.log" 2>&1
chk $? "train.py exits 0"
grep -E "Loaded checkpoint|Auto-resumed" "$TMP/resume.log" | sed 's/^.*trainer\]//' | sed 's/^/    /'

grep -q "Auto-resumed" "$TMP/resume.log"; chk $? "auto-resumed from latest.pt"
grep -q "(step 4)" "$TMP/resume.log";     chk $? "resumed at the exact step it stopped on (4)"
[ -f "$CKPT/checkpoint_step00000008.pt" ]; chk $? "advanced to step 8"
KEPT=$(ls "$CKPT"/checkpoint_step*.pt 2>/dev/null | wc -l)
[ "$KEPT" -le 3 ]; chk $? "pruning kept last 3 snapshots (found $KEPT)"

# ---------------------------------------------------------------- 5. STOP sentinel
hdr "5. STOP sentinel (the manual-stop path)"
python train.py "${SMOKE[@]}" training.max_steps=100000 training.n_epochs=100 \
    training.save_every_n_steps=50 > "$TMP/stop.log" 2>&1 &
TRAIN_PID=$!

# Wait until it is genuinely mid-training before asking it to stop. The trainer
# heartbeats every 10 steps; this run resumes at 8, so "[hb] step=10" lands fast.
for _ in $(seq 1 120); do
    grep -q "\[hb\] step=" "$TMP/stop.log" 2>/dev/null && break
    kill -0 "$TRAIN_PID" 2>/dev/null || break
    sleep 1
done
sleep 2

# Read the step out of latest.pt rather than comparing mtimes (NFS attr cache).
ckpt_step() { python - "$1" <<'PY'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["global_step"])
PY
}

echo "  dropping stop sentinel: $CKPT/STOP"
BEFORE_STEP=$(ckpt_step "$CKPT/latest.pt")
touch "$CKPT/STOP"
T0=$SECONDS
wait "$TRAIN_PID"; STOP_RC=$?
echo "  trainer noticed, checkpointed and exited $((SECONDS - T0))s later, rc=$STOP_RC"

[ "$STOP_RC" = 0 ]; chk $? "exits 0 on stop (a clean completion, not a failure)"
[ ! -f "$CKPT/STOP" ]; chk $? "sentinel consumed (next run will not be killed by it)"
AFTER_STEP=$(ckpt_step "$CKPT/latest.pt")
chk $? "latest.pt still loads after the interrupt (not truncated)"
[ "${AFTER_STEP:-0}" -gt "${BEFORE_STEP:-0}" ]
chk $? "checkpointed on the way out (latest.pt advanced: step $BEFORE_STEP -> $AFTER_STEP)"
grep -q "epoch_completed=False" "$TMP/stop.log"
chk $? "interrupted epoch recorded as incomplete (re-run, not skipped)"

# ---------------------------------------------------------------- summary
hdr "summary"
if [ "$FAILS" = 0 ]; then
    rm -rf "$CKPT" "$TMP"
    echo "  ALL CHECKS PASSED — the checkpoint/stop/resume harness works here."
    echo "  Next: bash bash/02_build_index_and_stats.sh   (one-time, ~5 min)"
    exit 0
fi
cp "$TMP"/*.log "$CKPT/" 2>/dev/null
echo "  $FAILS CHECK(S) FAILED — evidence kept in $CKPT/ (incl. the run logs)"
exit 1
