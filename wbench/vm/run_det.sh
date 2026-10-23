#!/bin/bash
# Deterministic AI+RES scorers on the 16 Gate 3 walkers (wbench/det_score.py): Pangu24,
# then SFNO, one shard per GPU, then the CPU reduce. Needs a bundle built with
# `make_bundle.sh --det` (the 64 ICs) and `setup_vm.sh ... det`.
#
#   bash wbench/vm/run_det.sh                 # every visible GPU
#   WB_NGPU=1 bash wbench/vm/run_det.sh       # a 1-GPU VM
#   bash wbench/vm/run_det.sh sfno            # one model (cached cubes are skipped)
#
# Cache-aware: a finished (walker, lead) cube is skipped, so after a preemption the same
# command picks up where it stopped.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
EVENT=${WB_EVENT:-PNW_HeatDome_2021}
NGPU=${WB_NGPU:-$(nvidia-smi -L | wc -l)}
MODELS=("$@")
[ "${#MODELS[@]}" -eq 0 ] && MODELS=(pangu24 sfno)
LOG=$REPO/runs/aires/$EVENT/det/logs
mkdir -p "$LOG"
export WB_THREADS=${WB_THREADS:-$(( $(nproc) / NGPU ))}

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | while IFS=, read -r i mem; do
  [ "${mem//[^0-9]/}" -lt 2048 ] || { echo "GPU $i already holds$mem - not yours? stop"; exit 1; }
done

for model in "${MODELS[@]}"; do
  bash "$HERE/env.sh" fcn3 python -m wbench.det_score --stage infer --model "$model" --dry \
    --nshards 1 | tee "$LOG/${model}_plan.log"
  echo "[run_det] $model on $NGPU GPU(s), logs in $LOG"
  t0=$SECONDS
  pids=()
  for ((i = 0; i < NGPU; i++)); do
    CUDA_VISIBLE_DEVICES=$i bash "$HERE/env.sh" fcn3 python -m wbench.det_score \
      --stage infer --model "$model" --shard "$i" --nshards "$NGPU" \
      > "$LOG/${model}_shard$i.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || { echo "[run_det] a $model shard FAILED - tail $LOG/${model}_shard*.log"; exit 1; }
  echo "[run_det] $model done in $(( SECONDS - t0 )) s"
done

# CPU; the fcn3 env carries everything the reduce imports (no JAX on that path)
bash "$HERE/env.sh" fcn3 python -m wbench.det_score --stage reduce | tee "$LOG/reduce.log"
echo "[run_det] results: runs/aires/$EVENT/det/det_reduce.json"
