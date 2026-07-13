#!/usr/bin/env bash
# =============================================================================
# Downscaler smoke test — CPU only, `debug` partition.
#
#   sbatch slurm/smoke.sbatch
#   tail -f logs/slurm-downscaler-smoke-<jobid>.out
#
# Verifies the code and the interruptible-training harness end to end, on a tiny
# dummy model, WITHOUT burning an a3mega GPU node:
#
#   1. environment  — the node can see the project, the `moe` env, and torch
#   2. pytest       — the unit/smoke suite
#   3. train        — fresh run checkpoints mid-epoch, latest.pt is a hardlink
#   4. resume       — a second run picks up at the exact step it left off at
#   5. STOP         — the stop sentinel makes it checkpoint and exit 0
#
# Steps 3-5 are the whole point: they are the start/stop/resume cycle that the real
# job on a3mega relies on. Passing here means the harness is sound; it says nothing
# about the science (this trains on random tensors).
#
# The debug nodes are 1 vCPU / 7.5 GB and POWERED_DOWN when idle, so expect a few
# minutes of GCP boot before the job starts. The work itself is ~2 minutes.
# =============================================================================
#SBATCH --job-name=downscaler-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/downscaler/logs/slurm-%x-%j.out
set -uo pipefail

PROJ=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/downscaler
CKPT="$PROJ/checkpoints/_smoke"     # scratch; never touches the real run's latest.pt

export HOME=/home/ubuntu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""      # debug nodes have no GPU; keep torch honest about it
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
echo "  node=$(hostname)  cpus=$(nproc)  mem=$(free -g | awk '/^Mem:/{print $2}')G"
echo "  python=$(which python)"
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
    training.save_every_n_steps=2 > /tmp/smoke_train.log 2>&1
chk $? "train.py exits 0"
grep -E "Saved checkpoint|Reached max_steps" /tmp/smoke_train.log | sed 's/^.*trainer\]//' | sed 's/^/    /'

[ -f "$CKPT/latest.pt" ]; chk $? "latest.pt written"
[ -f "$CKPT/checkpoint_step00000004.pt" ]; chk $? "step-4 snapshot written"
LINKS=$(stat -c %h "$CKPT/latest.pt" 2>/dev/null || echo 0)
[ "$LINKS" -ge 2 ]; chk $? "latest.pt is a hardlink, not a copy (nlink=$LINKS)"

# ---------------------------------------------------------------- 4. resume
hdr "4. resume (should pick up at step 4, run to 8)"
python train.py "${SMOKE[@]}" training.max_steps=8 training.n_epochs=1 \
    training.save_every_n_steps=2 > /tmp/smoke_resume.log 2>&1
chk $? "train.py exits 0"
grep -E "Loaded checkpoint|Auto-resumed" /tmp/smoke_resume.log | sed 's/^.*trainer\]//' | sed 's/^/    /'

grep -q "Auto-resumed" /tmp/smoke_resume.log; chk $? "auto-resumed from latest.pt"
grep -q "(step 4)" /tmp/smoke_resume.log;     chk $? "resumed at the exact step it stopped on (4)"
[ -f "$CKPT/checkpoint_step00000008.pt" ];    chk $? "advanced to step 8"
KEPT=$(ls "$CKPT"/checkpoint_step*.pt 2>/dev/null | wc -l)
[ "$KEPT" -le 3 ]; chk $? "pruning kept last 3 snapshots (found $KEPT)"

# ---------------------------------------------------------------- 5. STOP sentinel
hdr "5. STOP sentinel (the manual-stop path)"
python train.py "${SMOKE[@]}" training.max_steps=100000 training.n_epochs=100 \
    training.save_every_n_steps=50 > /tmp/smoke_stop.log 2>&1 &
TRAIN_PID=$!

# Wait until it is genuinely mid-training before asking it to stop, else we would be
# racing startup rather than testing an interrupt. The trainer heartbeats every 10
# optimizer steps; this run resumes at step 8, so "[hb] step=10" lands almost at once.
for _ in $(seq 1 120); do
    grep -q "\[hb\] step=" /tmp/smoke_stop.log 2>/dev/null && break
    kill -0 "$TRAIN_PID" 2>/dev/null || break
    sleep 1
done
sleep 2

# Read the step out of latest.pt rather than comparing mtimes: mtime has 1 s granularity
# and is served from the NFS attribute cache, so a save that lands within a second of the
# previous one can look unchanged even though it is on disk. Loading the checkpoint also
# proves it is intact and resumable, which is the property we actually care about.
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

[ "$STOP_RC" = 0 ]; chk $? "exits 0 on stop (a clean Slurm completion, not a failure)"
[ ! -f "$CKPT/STOP" ]; chk $? "sentinel consumed (next job will not be killed by it)"
AFTER_STEP=$(ckpt_step "$CKPT/latest.pt")
chk $? "latest.pt still loads after the interrupt (not truncated)"
[ "${AFTER_STEP:-0}" -gt "${BEFORE_STEP:-0}" ]
chk $? "checkpointed on the way out (latest.pt advanced: step $BEFORE_STEP -> $AFTER_STEP)"
grep -q "epoch_completed=False" /tmp/smoke_stop.log
chk $? "interrupted epoch recorded as incomplete (it will be re-run, not skipped)"
grep -E "Stop file|Saved checkpoint" /tmp/smoke_stop.log | tail -2 | sed 's/^.*trainer\]//' | sed 's/^/    /'

# ---------------------------------------------------------------- summary
hdr "summary"
if [ "$FAILS" = 0 ]; then
    rm -rf "$CKPT"
    echo "  ALL CHECKS PASSED — the checkpoint/stop/resume harness works on this cluster."
    echo "  Next: sbatch slurm/train.sbatch   (a3mega, 8 GPUs, still dummy data)"
    exit 0
fi
# Keep the scratch checkpoints on failure — they are the evidence. /tmp is node-local and
# the debug node powers down when idle, so copy the run logs somewhere durable first.
cp /tmp/smoke_train.log /tmp/smoke_resume.log /tmp/smoke_stop.log "$CKPT/" 2>/dev/null
echo "  $FAILS CHECK(S) FAILED"
echo "  evidence kept in $CKPT/ (incl. smoke_{train,resume,stop}.log)"
exit 1
