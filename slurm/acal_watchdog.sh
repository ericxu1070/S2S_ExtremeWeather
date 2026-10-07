#!/bin/bash
# slurm/acal_watchdog.sh -- detached login-node babysitter for the acal calibration array.
#
#   nohup setsid bash slurm/acal_watchdog.sh > /dev/null 2>&1 &
#
# Every POLL seconds it:
#   * surveys the 8 a3mega nodes over SSH (sinfo "idle" is not free: foreign non-Slurm
#     tenants hold 60-81 GB/card) and classes each node OURS / FREE / FOREIGN; a foreign
#     sighting is sticky for FOREIGN_STICKY seconds because the transient multi-GPU
#     tenants of 2026-09-18 came and went inside 10 minutes;
#   * keeps every pending Vayuh-s2s task's ExcNodeList equal to the FOREIGN set, and the
#     array throttle at running + min(free nodes, disk slots), so a task starts only when a
#     genuinely free node and >= DISK_RESERVE GB of headroom both exist; the total number
#     of nodes we hold never exceeds logs/acal_watchdog_state/max_nodes (default 8, set
#     with `slurm/acal_ctl.sh nodes N`, read every poll);
#   * runs `--stage compare` then `--stage prune` on each case as it lands (CPU, here),
#     which is what keeps the disk flat (~45 GB of walker states per case);
#   * on a task failure classes it: contention (CUDA OOM / RESOURCE_EXHAUSTED / node fail /
#     timeout) -> resubmit that one case once, excluding the node it died on; any other
#     failure, a second failure of the same case, or a CANCELLED task -> CRITICAL;
#   * declares CRITICAL on disk < DISK_CRIT GB, a running case silent > SILENT_MAX, or
#     nothing running for > NORUN_MAX while tasks are pending;
#   * while logs/acal_watchdog_state/paused exists (slurm/acal_ctl.sh pause) it only
#     compares + prunes landed cases and leaves the queue alone.
# It appends events to logs/acal_watch_<STAMP>.md, a per-poll line to
# logs/acal_watchdog_<STAMP>.log, and writes exactly one "DONE ..." or "CRITICAL ..." line
# to logs/acal_trigger_<STAMP>.txt before exiting. Nothing else is meant to wake anyone.
# It never touches a process that is not ours.
set -uo pipefail

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO" || exit 1
STAMP=${ACAL_WD_STAMP:-20260918}
SD=logs/acal_watchdog_state
RUNG=${ACAL_RUNG:-$( [ -f "$SD/rung" ] && tr -dc '0-9.' < "$SD/rung" || echo 3)}   # the campaign's rung (slurm/acal_ctl.sh resume)
TAG=${ACAL_RES_TAG:-acal}
POLL=${ACAL_WD_POLL:-300}
SACCT_SINCE=${ACAL_WD_SINCE:-2026-09-17T18:00}
DISK_CRIT=80          # GB free -> CRITICAL
DISK_RESERVE=100      # GB kept free before admitting another case
CASE_GB=60            # live footprint of one in-flight case
SILENT_MAX=90         # minutes a RUNNING case may go without writing anything
NORUN_MAX=$((3*3600)) # seconds with nothing running while tasks pend
FOREIGN_STICKY=1800   # seconds a foreign sighting keeps a node excluded
NODEPFX=nucla3m-a3meganodeset-
WATCH=logs/acal_watch_$STAMP.md
TRIG=logs/acal_trigger_$STAMP.txt
LOG=logs/acal_watchdog_$STAMP.log
mkdir -p "$SD" logs

source slurm/aires_env.sh moe
export AIRES_N_WALKERS=${AIRES_N_WALKERS:-32} AIRES_M_MEMBERS=${AIRES_M_MEMBERS:-6} AIRES_NSHARDS=${AIRES_NSHARDS:-8}

ts()    { date -u +%Y-%m-%dT%H:%MZ; }
log()   { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }
watch() { echo "$*" >> "$WATCH"; }
free_gb(){ df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }
critical() {
  log "CRITICAL: $*"
  watch ""; watch "## $(ts) CRITICAL -- watchdog stopped: $*"
  echo "CRITICAL $(ts) $*" >> "$TRIG"
  exit 2
}

# Rung-aware slate: a queued task's %K indexes the slate ITS ARRAY was submitted at
# (recorded in $SD/rung_<array>), not necessarily this one -- see slurm/acal_slate.sh.
source slurm/acal_slate.sh
[ "$NCASE" -gt 0 ] || { echo "empty slate" >&2; exit 1; }
casedir() { echo "runs/aires/${SLATE[$1]}/res/$TAG"; }

echo $$ > "$SD/pid"
trap 'log "watchdog exiting (pid $$)"' EXIT
log "watchdog start pid $$ slate=$NCASE poll=${POLL}s"
watch ""; watch "## $(ts) watchdog (slurm/acal_watchdog.sh, pid $$) took over the watch; it writes DONE/CRITICAL to $TRIG"

declare -A LASTFOREIGN=() ARR_RUN=() ARR_PEND=()
OURS_NODES=""; RUN_K=""; RUN_DESC=""; PEND_IDS=""; NRUN=0; NPEND=0
FREE_NODES=""; FOREIGN_NODES=""; NORUN_SINCE=""; HEARTBEAT=0

read_queue() {
  local lines id st node k arr
  lines=$(squeue -u "$USER" -n Vayuh-s2s -r -h -o "%i|%t|%N|%K|%F" 2>/dev/null) || lines=""
  OURS_NODES=""; RUN_K=""; RUN_DESC=""; PEND_IDS=""; NRUN=0; NPEND=0
  ARR_RUN=(); ARR_PEND=()
  while IFS='|' read -r id st node k arr; do
    [ -z "$id" ] && continue
    k=$(task_k "$arr" "$k")     # "" when the case lies outside this slate; node still ours
    if [ "$st" = "R" ] || [ "$st" = "CG" ]; then
      OURS_NODES+="${node##*-} "; [ -n "$k" ] && RUN_K+="$k "; NRUN=$((NRUN+1))
      ARR_RUN[$arr]=$(( ${ARR_RUN[$arr]:-0} + 1 )); RUN_DESC+="$id@n${node##*-} "
    elif [ "$st" = "PD" ]; then
      PEND_IDS+="$id "; NPEND=$((NPEND+1)); ARR_PEND[$arr]=$(( ${ARR_PEND[$arr]:-0} + 1 ))
    fi
  done <<< "$lines"
}

survey() {
  local now n out nv bad
  now=$(date +%s); FREE_NODES=""; FOREIGN_NODES=""; SURVEY_DESC=""
  for n in 0 1 2 3 4 5 6 7; do
    if [[ " $OURS_NODES " == *" $n "* ]]; then SURVEY_DESC+="n$n:ours "; continue; fi
    out=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODEPFX$n" \
          nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd, -)
    nv=$(echo "$out" | tr ',' '\n' | grep -c '^[0-9]\+$')
    bad=$(echo "$out" | tr ',' '\n' | awk '$1+0>=1000' | wc -l)
    if [ "$nv" -ne 8 ] || [ "$bad" -gt 0 ]; then LASTFOREIGN[$n]=$now; fi
    if [ -n "${LASTFOREIGN[$n]:-}" ] && [ $((now - LASTFOREIGN[$n])) -lt $FOREIGN_STICKY ]; then
      FOREIGN_NODES+="$n "; SURVEY_DESC+="n$n:FOREIGN(${out:-unreachable}) "
    else
      FREE_NODES+="$n "; SURVEY_DESC+="n$n:free "
    fi
  done
  FREE_NODES=${FREE_NODES% }; FOREIGN_NODES=${FOREIGN_NODES% }
}

manage_pending() {
  local fg disk_slots nfree slots want id cur curx wantx arr want_t cur_t remaining give
  fg=$(free_gb)
  disk_slots=$(( (fg - DISK_RESERVE) / CASE_GB )); [ "$disk_slots" -lt 0 ] && disk_slots=0
  nfree=$(echo $FREE_NODES | wc -w)
  slots=$(( nfree < disk_slots ? nfree : disk_slots ))
  maxn=$(cat "$SD/max_nodes" 2>/dev/null || echo 8); [[ "$maxn" =~ ^[0-9]+$ ]] || maxn=8
  [ "$slots" -gt $((maxn - NRUN)) ] && slots=$((maxn - NRUN)); [ "$slots" -lt 0 ] && slots=0
  want=""; for n in $FOREIGN_NODES; do want+="$NODEPFX$n,"; done; want=${want%,}
  wantx=$( [ -n "$want" ] && echo "$want" | tr ',' '\n' | sort | paste -sd, - )
  for id in $PEND_IDS; do
    cur=$(scontrol show job "$id" 2>/dev/null | grep -o 'ExcNodeList=[^ ]*' | head -1 | cut -d= -f2)
    [ "$cur" = "(null)" ] && cur=""
    curx=$( [ -n "$cur" ] && scontrol show hostnames "$cur" 2>/dev/null | sort | paste -sd, - )
    if [ "$curx" != "$wantx" ]; then
      if scontrol update JobId="$id" ExcNodeList="$want" 2>>"$LOG"; then
        log "pin $id ExcNodeList=[${want:-none}]"
        watch "- $(ts) $id ExcNodeList -> foreign set [${FOREIGN_NODES:-none}]; free [${FREE_NODES:-none}]; df ${fg}G"
      fi
    fi
  done
  # The free slots are shared out across the pending arrays (oldest first) so that two
  # arrays in the queue at once cannot both start `slots` tasks on the same poll.
  remaining=$slots
  for arr in $(printf '%s\n' "${!ARR_PEND[@]}" | sort -n); do
    [ "${ARR_PEND[$arr]}" -gt 0 ] || continue
    give=$(( ${ARR_PEND[$arr]} < remaining ? ${ARR_PEND[$arr]} : remaining )); remaining=$((remaining - give))
    want_t=$(( ${ARR_RUN[$arr]:-0} + give )); [ "$want_t" -lt 1 ] && want_t=1
    cur_t=$(scontrol show job "$arr" 2>/dev/null | grep -o 'ArrayTaskThrottle=[0-9]*' | head -1 | cut -d= -f2)
    if [ -n "$cur_t" ] && [ "$cur_t" != "$want_t" ]; then
      if scontrol update JobId="$arr" ArrayTaskThrottle="$want_t" 2>>"$LOG"; then
        log "throttle $arr $cur_t->$want_t (run=${ARR_RUN[$arr]:-0} free=$nfree disk_slots=$disk_slots max_nodes=$maxn)"
        watch "- $(ts) throttle $arr $cur_t -> $want_t (running ${ARR_RUN[$arr]:-0}, free nodes [${FREE_NODES:-none}], df ${fg}G)"
      fi
    fi
  done
}

handle_failures() {
  local rows jid raw state ec node k ka arr ev d n sig cnt exc new
  rows=$(sacct -u "$USER" --name=Vayuh-s2s -S "$SACCT_SINCE" -X -n -P -o JobID,JobIDRaw,State,ExitCode,NodeList 2>/dev/null)
  while IFS='|' read -r jid raw state ec node; do
    [ -z "$jid" ] && continue
    case "$state" in RUNNING*|PENDING*|COMPLETED*|COMPLETING*) continue;; esac
    [ -f "$SD/handled_$raw" ] && continue
    arr=${jid%_*}; ka=${jid##*_}
    if [ "$arr" = "$jid" ] || ! [[ "$ka" =~ ^[0-9]+$ ]]; then touch "$SD/handled_$raw"; log "ignoring $jid ($raw) $state: not an array task"; continue; fi
    ev=$(task_case "$arr" "$ka"); k=$(task_k "$arr" "$ka"); n=${node##*-}
    touch "$SD/handled_$raw"
    if [ -z "$ev" ]; then log "ignoring $jid ($raw) $state: task $ka is past the end of array $arr's slate"; continue; fi
    if [ -z "$k" ]; then
      log "task $jid ($raw) $state exit $ec node $n: case $ev is OUTSIDE the rung-$RUNG slate; not resubmitting"
      watch ""; watch "## $(ts) INCIDENT (watchdog): $jid (job $raw, case $ev) $state exit $ec on node $n -- outside this slate, resubmit by hand if wanted"
      continue
    fi
    d=$(casedir "$k")
    if [ -f "$d/res_result.json" ]; then log "task $jid ($raw) $state but case $k $ev has res_result -> ignore"; continue; fi
    sig=$(grep -lE "CUDA out of memory|RESOURCE_EXHAUSTED|CUDA error|out of memory|CUDA_ERROR" \
          "logs/Vayuh-s2s-$raw.err" "$d"/logs/*-"$raw"-*.log 2>/dev/null | wc -l)
    cnt=$(cat "$SD/resub_$k" 2>/dev/null || echo 0)
    case "$state" in CANCELLED*) critical "task $jid (job $raw, case $k $ev) was CANCELLED on node $n; not resubmitting";; esac
    [ "$cnt" -ge 1 ] && critical "case $k $ev failed twice: $jid (job $raw) $state exit $ec on node $n after one resubmission"
    if [[ "$state" == FAILED* ]] && [ "$sig" -eq 0 ]; then
      critical "task $jid (job $raw, case $k $ev) $state exit $ec on node $n with no contention signature; see logs/Vayuh-s2s-$raw.err"
    fi
    LASTFOREIGN[$n]=$(date +%s)
    exc="$NODEPFX$n"; for f in $FOREIGN_NODES; do [ "$f" != "$n" ] && exc+=",$NODEPFX$f"; done
    new=$(ACAL_RUNG=$RUNG sbatch --parsable --array="$k" --exclude="$exc" slurm/acal_res.slurm 2>>"$LOG"); new=${new%%;*}
    [ -n "$new" ] || critical "resubmit of case $k $ev after $jid $state failed (sbatch error, see $LOG)"
    record_rung "$new"
    echo $((cnt+1)) > "$SD/resub_$k"
    log "INCIDENT $jid ($raw) $state exit $ec node $n sig=$sig -> resubmitted case $k $ev as $new excluding [$exc]"
    watch ""; watch "## $(ts) INCIDENT (watchdog): $jid (job $raw, case $k $ev) $state exit $ec on node $n"
    watch "- contention signature in $sig log file(s); resubmitted as job $new (--array=$k, --exclude=$exc). Foreign process untouched."
  done <<< "$rows"
}

case_summary() {  # $1 = case index
  python - "$(casedir "$1")" <<'PY' 2>/dev/null
import json, sys
d = sys.argv[1]
r = json.load(open(f"{d}/res_result.json")); c = json.load(open(f"{d}/compare.json"))
ess = ",".join(f"{x:.1f}" for x in c.get("ess_by_step", []))
print(f"job {r.get('job')} on {r.get('host')}, wall {r.get('wall_seconds',0)/60:.1f} min; "
      f"log_Z {c.get('log_Z',float('nan')):.3f}, founders {c.get('n_founders')}, "
      f"normalization_check {c.get('normalization_check',float('nan')):.3f}, ESS by step [{ess}]")
PY
}

reduce_cases() {
  local k ev d nst before after
  for ((k=0; k<NCASE; k++)); do
    ev=${SLATE[$k]}; d=$(casedir "$k")
    [ -f "$d/res_result.json" ] || continue
    [[ " $RUN_K " == *" $k "* ]] && continue
    [ -f "$SD/reduce_failed_$k" ] && continue
    if [ ! -f "$d/compare.json" ]; then
      log "compare case $k $ev"
      if ! timeout 3600 python -m aires.run_aires --stage compare --event "$ev" --tag "$TAG" > "logs/acal_compare_${ev}.log" 2>&1; then
        touch "$SD/reduce_failed_$k"; critical "compare failed for case $k $ev (logs/acal_compare_${ev}.log)"
      fi
      log "compare OK case $k $ev"
    fi
    nst=$(find "$d" -name '*state*.nc' 2>/dev/null | wc -l)
    if [ "$nst" -gt 0 ]; then
      before=$(free_gb)
      if ! timeout 1800 python -m aires.run_aires --stage prune --event "$ev" --tag "$TAG" > "logs/acal_prune_${ev}.log" 2>&1; then
        touch "$SD/reduce_failed_$k"; critical "prune failed for case $k $ev (logs/acal_prune_${ev}.log)"
      fi
      after=$(free_gb)
      log "prune OK case $k $ev df ${before}G->${after}G"
      watch ""; watch "## $(ts) case $k $ev landed (watchdog): compare + prune done, df ${before}G -> ${after}G"
      watch "- $(case_summary "$k")"
    fi
  done
}

check_silence() {
  local k d
  for k in $RUN_K; do
    d=$(casedir "$k"); [ -d "$d" ] || continue
    if [ -z "$(find "$d" -type f -mmin -$SILENT_MAX -print -quit 2>/dev/null)" ]; then
      critical "case $k ${SLATE[$k]} is RUNNING but has written nothing for > $SILENT_MAX min"
    fi
  done
}

all_done() {
  local k d
  for ((k=0; k<NCASE; k++)); do
    d=$(casedir "$k")
    [ -f "$d/res_result.json" ] && [ -f "$d/compare.json" ] || return 1
    [ "$(find "$d" -name '*state*.nc' 2>/dev/null | wc -l)" -eq 0 ] || return 1
  done
  return 0
}

finish() {
  local k
  watch ""; watch "## $(ts) CAMPAIGN DONE (watchdog): all $NCASE cases have res_result.json + compare.json and are pruned"
  watch ""; watch "| k | case | summary |"; watch "|---|------|---------|"
  for ((k=0; k<NCASE; k++)); do watch "| $k | ${SLATE[$k]} | $(case_summary "$k") |"; done
  watch ""; watch "- sacct: $(sacct -u "$USER" --name=Vayuh-s2s -S "$SACCT_SINCE" -X -n -P -o JobID,State,Elapsed,NodeList | grep -v PENDING | tr '\n' ' ')"
  watch "- df /home: $(free_gb)G free"
  log "DONE"
  echo "DONE $(ts) all $NCASE cases finished, compared, pruned; df $(free_gb)G" >> "$TRIG"
  exit 0
}

# --- main loop ---------------------------------------------------------------------
while true; do
  if [ -f "$SD/paused" ]; then     # slurm/acal_ctl.sh pause: reduce landed cases, touch nothing else
    read_queue; reduce_cases; NORUN_SINCE=""
    log "paused since $(cat "$SD/paused"): run=$NRUN pend=$NPEND df=$(free_gb)G"
    sleep "$POLL"; continue
  fi
  read_queue
  handle_failures
  read_queue
  survey
  manage_pending
  reduce_cases
  fg=$(free_gb)
  [ "$fg" -lt "$DISK_CRIT" ] && critical "df /home ${fg}G free < ${DISK_CRIT}G"
  if [ "$NRUN" -eq 0 ] && [ "$NPEND" -eq 0 ]; then
    all_done && finish
    critical "queue is empty but the campaign is not complete (see case table in $LOG)"
  fi
  if [ "$NRUN" -eq 0 ]; then
    : "${NORUN_SINCE:=$(date +%s)}"
    [ $(( $(date +%s) - NORUN_SINCE )) -gt "$NORUN_MAX" ] && critical "nothing running for > $((NORUN_MAX/3600)) h with $NPEND task(s) pending; foreign [${FOREIGN_NODES:-none}]"
  else
    NORUN_SINCE=""
  fi
  check_silence
  log "poll run=$NRUN [${RUN_DESC}] pend=$NPEND [${PEND_IDS}] free=[${FREE_NODES}] foreign=[${FOREIGN_NODES}] df=${fg}G :: $SURVEY_DESC"
  HEARTBEAT=$((HEARTBEAT+1))
  if [ $((HEARTBEAT % 12)) -eq 1 ]; then
    watch "- $(ts) watchdog: running ${RUN_DESC:-none}; pending ${PEND_IDS:-none}; free [${FREE_NODES:-none}]; foreign [${FOREIGN_NODES:-none}]; df ${fg}G"
  fi
  sleep "$POLL"
done
