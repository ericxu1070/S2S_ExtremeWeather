#!/bin/bash
# The whole walker benchmark on one GPU VM: preflight -> solo -> node8 -> report.
# ~1.5 h on 8xH100 with the defaults; WB_QUICK=1 is a ~35 min smoke run.
#
#   bash wbench/vm/run_all.sh                       # new run, id = UTC timestamp
#   WB_RUN_ID=<id> bash wbench/vm/run_all.sh        # resume: finished steps are skipped
#   bash wbench/vm/run_all.sh preflight solo        # only these phases, in this order
#   WB_QUICK=1 bash wbench/vm/run_all.sh            # 1 timed segment, short sweep
#
# Phases
#   preflight  GPUs idle, disk, envs, DISCO kernel, FCN3 pickle; machine snapshot to logs/
#   solo       one worker per config on its own GPU (all configs at once, other GPUs busy
#              with the other configs): layer 1 + the FCN3 batch sweep + walker checks +
#              solo segments, GenCast 0.25 twice for run-to-run spread
#   node8      8 identical workers per config, one wave per config - the production
#              layout, where per-step cost under contention is the number that matters
#   report     REPORT.md + results.json + results.csv (layer 3 projection included)
#
# A step is skipped when its result JSON already exists, so a killed run resumes by
# re-running the same command with the same WB_RUN_ID. One failed worker never stops
# its phase: failures are listed at the end and in status.tsv.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export WB_RUN_ID=${WB_RUN_ID:-$(date -u +%Y%m%dT%H%MZ)}
RUN=$REPO/runs/wbench/$WB_RUN_ID
RES=$RUN/results
LOGS=$RUN/logs
STATUS=$RUN/status.tsv
mkdir -p "$RES" "$LOGS"
NGPU=${WB_NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
PHASES=("$@")
[ "${#PHASES[@]}" -eq 0 ] && PHASES=(preflight solo node8 report)

log() { echo "[run_all $(date -u +%H:%M:%S)] $*" | tee -a "$RUN/run_all.log"; }
w2() { printf '%02d' "$1"; }

declare -A CPUS
while read -r g c; do CPUS[$g]=$c; done < <(python3 "$HERE/cpu_slices.py" "$NGPU")
ncpus() { python3 -c "import sys; sys.path.insert(0, '$HERE'); import cpu_slices as c; print(len(c.expand('$1')))"; }

# --------------------------------------------------------------------------- #
# GPU slot scheduler: one process per GPU, launched when a slot frees up.
# --------------------------------------------------------------------------- #
declare -a SLOT_PID SLOT_NAME SLOT_T0
FAILS=()
FREE_GPU=-1

reap() {
  local g pid rc dt
  for ((g = 0; g < NGPU; g++)); do
    pid=${SLOT_PID[$g]:-}
    [ -z "$pid" ] && continue
    kill -0 "$pid" 2>/dev/null && continue
    wait "$pid"; rc=$?
    dt=$(( SECONDS - SLOT_T0[g] ))
    printf '%s\t%s\t%s\t%s\t%s\n' "$WB_PHASE" "${SLOT_NAME[$g]}" "$g" "$rc" "$dt" >> "$STATUS"
    if [ "$rc" -eq 0 ]; then
      log "  done   ${SLOT_NAME[$g]} (gpu$g, ${dt} s)"
    else
      log "  FAILED ${SLOT_NAME[$g]} (gpu$g, rc=$rc, ${dt} s): $LOGS/${WB_PHASE}__${SLOT_NAME[$g]}__gpu$g.log"
      FAILS+=("$WB_PHASE/${SLOT_NAME[$g]}")
    fi
    SLOT_PID[$g]=""
  done
}

next_free() {
  local g
  while true; do
    reap
    for ((g = 0; g < NGPU; g++)); do
      [ -z "${SLOT_PID[$g]:-}" ] && { FREE_GPU=$g; return; }
    done
    sleep 5
  done
}

drain() {
  local g busy
  while true; do
    reap
    busy=0
    for ((g = 0; g < NGPU; g++)); do [ -n "${SLOT_PID[$g]:-}" ] && busy=1; done
    [ "$busy" -eq 0 ] && return
    sleep 5
  done
}

# sub <name> <result json or ""> <env> <cmd ...>
sub() {
  local name=$1 res=$2 env=$3
  shift 3
  if [ -n "$res" ] && [ -f "$RES/$res" ]; then
    log "  skip   $name (have $res)"
    return
  fi
  next_free
  local g=$FREE_GPU
  local logf=$LOGS/${WB_PHASE}__${name}__gpu$g.log
  if [ "${WB_DRY:-0}" = 1 ]; then
    # Scheduler test: no GPU work, just the command line, a short sleep and a fake
    # result so resume logic can be exercised too.
    ( echo "DRY gpu$g cpus ${CPUS[$g]}: $env $*"; sleep $(( RANDOM % 3 + 1 ))
      [ -n "$res" ] && echo '{}' > "$RES/$res" ) > "$logf" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$g WB_THREADS=$(ncpus "${CPUS[$g]}") \
      taskset -c "${CPUS[$g]}" bash "$HERE/env.sh" "$env" "$@" > "$logf" 2>&1 &
  fi
  SLOT_PID[$g]=$!
  SLOT_NAME[$g]=$name
  SLOT_T0[$g]=$SECONDS
  log "  start  $name on gpu$g (cpus ${CPUS[$g]})"
}

gpus_idle() {  # wait up to 2 min for leftover compute processes to exit
  local i n
  [ "${WB_DRY:-0}" = 1 ] && return 0
  for i in $(seq 24); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "$n" -eq 0 ] && return 0
    sleep 5
  done
  log "WARNING: $n compute process(es) still on the GPUs - timings in this phase may be contaminated"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$RUN/run_all.log"
}

cleanup() {
  local g
  for ((g = 0; g < NGPU; g++)); do
    [ -n "${SLOT_PID[$g]:-}" ] && kill "${SLOT_PID[$g]}" 2>/dev/null
  done
  [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null
}
trap 'log "interrupted - killing workers"; cleanup; exit 130' INT TERM
trap cleanup EXIT

# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #
phase_preflight() {
  export WB_PHASE=preflight
  local ok=1
  [ "${WB_DRY:-0}" = 1 ] && { log "preflight: dry run, skipped"; return 0; }
  log "preflight: $NGPU GPU(s), $(nproc) CPUs, run $WB_RUN_ID"
  [ "$NGPU" -ge 1 ] || { log "no GPUs"; return 1; }
  [ "$NGPU" -eq 8 ] || log "WARNING: $NGPU GPUs, not 8 - node8 becomes node$NGPU"
  local busy
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', *' '$2 > 2000 {print "gpu" $1 "=" $2 "MiB"}')
  if [ -n "$busy" ]; then
    log "GPUs already in use: $busy"
    [ "${WB_IGNORE_BUSY:-0}" = 1 ] || { log "refusing (set WB_IGNORE_BUSY=1 to override)"; return 1; }
  fi
  local free_gb
  free_gb=$(df -BG --output=avail "$RUN" | tail -1 | tr -dc 0-9)
  log "free disk under runs/: ${free_gb} GB"
  [ "$free_gb" -ge 40 ] || { log "need >= 40 GB free"; ok=0; }
  CUDA_VISIBLE_DEVICES=0 bash "$HERE/env.sh" gencast python -c \
    "import jax; d=jax.devices(); assert d[0].platform=='gpu', d; print('jax', jax.__version__, d[0].device_kind)" \
    >> "$RUN/run_all.log" 2>&1 || { log "gencast env cannot see a GPU"; ok=0; }
  CUDA_VISIBLE_DEVICES=0 bash "$HERE/env.sh" fcn3 python -c \
    "import torch; from wbench import fcn3_walker as FW; from fcn3.model_cache import model_cache_path; from fcn3 import fevents as F; assert torch.cuda.is_available(); assert FW.disco_kernel_ok(), 'DISCO kernel missing'; p=model_cache_path(F.model_cache_dir()); assert p.exists(), f'no FCN3 pickle {p}'; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'pickle', p.name)" \
    >> "$RUN/run_all.log" 2>&1 || { log "fcn3 env not ready (GPU, DISCO kernel or FCN3 pickle)"; ok=0; }
  # Snapshot of the machine, next to the numbers it produced.
  nvidia-smi -q > "$LOGS/nvidia-smi-q.txt" 2>&1
  nvidia-smi topo -m > "$LOGS/nvidia-smi-topo.txt" 2>&1
  lscpu > "$LOGS/lscpu.txt" 2>&1
  { free -g; df -h "$RUN"; uname -a; } > "$LOGS/host.txt" 2>&1
  for e in gencast fcn3; do
    bash "$HERE/env.sh" "$e" python -m pip freeze > "$LOGS/pip-freeze-$e.txt" 2>&1
  done
  python3 "$HERE/cpu_slices.py" "$NGPU" > "$LOGS/cpu_slices.txt"
  [ "$ok" -eq 1 ] && log "preflight OK" || { log "preflight FAILED - see $RUN/run_all.log"; return 1; }
}

phase_solo() {
  export WB_PHASE=solo WB_CONCURRENCY=1
  gpus_idle
  log "solo: one worker per config (~25 min; bounded by GenCast 0.25 deg)"
  local R=(--run "$RUN")
  # Longest first, so the GenCast runs start on the first free GPUs.
  sub gencast_0p25_w0 "gencast_0p25__solo__w00__segments.json" gencast \
    python -m wbench.gencast_bench --config gencast_0p25 "${R[@]}" --worker 0
  sub gencast_0p25_w1 "gencast_0p25__solo__w01__segments.json" gencast \
    python -m wbench.gencast_bench --config gencast_0p25 "${R[@]}" --worker 1
  sub gencast_1p0_w0 "gencast_1p0__solo__w00__segments.json" gencast \
    python -m wbench.gencast_bench --config gencast_1p0 "${R[@]}" --worker 0
  sub checks "fcn3_native__solo__w00__checks.json" fcn3 \
    python -m wbench.checks "${R[@]}"
  sub native_sweep_w0 "fcn3_native__solo__w00__sweep.json" fcn3 \
    python -m wbench.fcn3_bench --path native --mode sweep "${R[@]}" --worker 0
  sub native_sweep_w1 "fcn3_native__solo__w01__sweep.json" fcn3 \
    python -m wbench.fcn3_bench --path native --mode sweep "${R[@]}" --worker 1
  sub e2s_sweep_w0 "fcn3_e2s__solo__w00__sweep.json" fcn3 \
    python -m wbench.fcn3_bench --path e2s --mode sweep --batches 1,2 "${R[@]}" --worker 0
  sub native_seg_nc "fcn3_native__solo__w00__segments_b1_nc.json" fcn3 \
    python -m wbench.fcn3_bench --path native --mode segments --batch 1 --io nc "${R[@]}" --worker 0
  sub native_seg_npz "fcn3_native__solo__w00__segments_b1_npz.json" fcn3 \
    python -m wbench.fcn3_bench --path native --mode segments --batch 1 --io npz "${R[@]}" --worker 0
  sub e2s_seg_nc "fcn3_e2s__solo__w00__segments_b1_nc.json" fcn3 \
    python -m wbench.fcn3_bench --path e2s --mode segments --batch 1 --io nc "${R[@]}" --worker 0
  drain
}

wave() {  # wave <label> <env> <result-json pattern with WW for the worker> <cmd ... with WORKER>
  local label=$1 env=$2 pat=$3
  shift 3
  gpus_idle
  log "node8 wave $label: $NGPU workers"
  local w args a
  for ((w = 0; w < NGPU; w++)); do
    args=()
    for a in "$@"; do args+=("${a//WORKER/$w}"); done
    sub "${label}_w$(w2 $w)" "${pat//WW/$(w2 $w)}" "$env" "${args[@]}"
  done
  drain
}

phase_node8() {
  export WB_PHASE=node8 WB_CONCURRENCY=$NGPU
  local R=(--run "$RUN")
  log "node8: $NGPU identical workers per config (~45 min)"
  wave native_b1 fcn3 "fcn3_native__node8__wWW__segments_b1_nc.json" \
    python -m wbench.fcn3_bench --path native --mode segments --batch 1 "${R[@]}" --worker WORKER
  local B=${WB_DRY_BEST_BATCH:-1}
  [ "${WB_DRY:-0}" = 1 ] || \
    B=$(bash "$HERE/env.sh" gencast python -m wbench.report --run "$RUN" --best-batch 2>/dev/null | tail -1)
  if [[ "$B" =~ ^[0-9]+$ ]] && [ "$B" -gt 1 ]; then
    wave "native_b$B" fcn3 "fcn3_native__node8__wWW__segments_b${B}_nc.json" \
      python -m wbench.fcn3_bench --path native --mode segments --batch "$B" "${R[@]}" --worker WORKER
  else
    log "node8: batching did not beat B=1 by the required margin in the sweep - no batched wave"
  fi
  wave e2s_b1 fcn3 "fcn3_e2s__node8__wWW__segments_b1_nc.json" \
    python -m wbench.fcn3_bench --path e2s --mode segments --batch 1 "${R[@]}" --worker WORKER
  wave gencast_0p25 gencast "gencast_0p25__node8__wWW__segments.json" \
    python -m wbench.gencast_bench --config gencast_0p25 "${R[@]}" --worker WORKER
  wave gencast_1p0 gencast "gencast_1p0__node8__wWW__segments.json" \
    python -m wbench.gencast_bench --config gencast_1p0 "${R[@]}" --worker WORKER
}

phase_report() {
  export WB_PHASE=report
  [ "${WB_DRY:-0}" = 1 ] && { log "report: dry run, skipped"; return 0; }
  bash "$HERE/env.sh" gencast python -m wbench.report --run "$RUN" > "$LOGS/report.log" 2>&1 \
    || { log "report FAILED: $LOGS/report.log"; return 1; }
  log "report: $RUN/REPORT.md"
}

# --------------------------------------------------------------------------- #
log "run $WB_RUN_ID: phases ${PHASES[*]} -> $RUN"
if command -v nvidia-smi >/dev/null && [ "${WB_DRY:-0}" != 1 ]; then
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
    --format=csv -l 5 >> "$LOGS/gpu_samples.csv" 2>/dev/null &
  SAMPLER=$!
fi
T0=$SECONDS
for p in "${PHASES[@]}"; do
  declare -F "phase_$p" >/dev/null || { log "unknown phase '$p'"; exit 2; }
  t=$SECONDS
  if ! "phase_$p"; then
    log "phase $p failed - stopping"
    exit 1
  fi
  log "phase $p finished in $(( (SECONDS - t) / 60 )) min"
done
log "all phases done in $(( (SECONDS - T0) / 60 )) min"
if [ "${#FAILS[@]}" -gt 0 ]; then
  log "${#FAILS[@]} step(s) failed: ${FAILS[*]}"
fi
log "copy home:  rsync -av <vm>:$RUN/ ./wbench_$WB_RUN_ID/   (REPORT.md, results/, logs/)"
[ "${#FAILS[@]}" -eq 0 ]
