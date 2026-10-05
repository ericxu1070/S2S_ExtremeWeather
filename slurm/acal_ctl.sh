#!/bin/bash
# slurm/acal_ctl.sh -- pause / resume / status for the acal calibration campaign.
#
#   bash slurm/acal_ctl.sh status          what is done, running, pending; nodes; disk; watchdog
#   bash slurm/acal_ctl.sh pause           free the GPU nodes NOW: cancel every acal task. Each
#                                          cancelled case resumes later from its last finished
#                                          leg (walker states and score cubes are cached), so at
#                                          most one in-flight pool (~20-40 min) is lost per case.
#   bash slurm/acal_ctl.sh pause --drain   admit no new case; let the running ones finish (up to
#                                          ~4 h) and then release their nodes.
#   bash slurm/acal_ctl.sh resume          (alias: play) survey the nodes, resubmit every
#                                          unfinished case as one array, route around nodes that
#                                          other people are using, restart the watchdog.
#   bash slurm/acal_ctl.sh nodes N         hold at most N nodes at once from now on (each case
#                                          takes one whole node = 8 H100 for ~4 h). Takes effect
#                                          within 5 min, no restart needed; `nodes` alone shows it.
#   ACAL_DRY=1 bash slurm/acal_ctl.sh pause|resume     print what would be done, do nothing.
#   ACAL_RUNG=2 bash slurm/acal_ctl.sh status|resume   the 42-case slate (rung 2; rung 3 is the
#                                          12-case |A_L| >= 3 slate). Every case of both slates
#                                          already has its init frame under runs/acal/inputs/.
#                                          `resume` REMEMBERS the rung (logs/acal_watchdog_state/
#                                          rung), so later calls and the watchdog default to it;
#                                          since 2026-09-18 08:04Z that is 2.
#
# Anyone with a shell on the login node can run this. It only ever cancels jobs submitted from
# slurm/acal_res.slurm (job name Vayuh-s2s); it never touches anybody else's processes.
# The campaign babysitter, slurm/acal_watchdog.sh, honours logs/acal_watchdog_state/paused:
# while that flag exists it neither resubmits nor re-pins anything, it only compares+prunes
# cases that already landed. `resume` clears the flag and (re)starts it if it is not running.
set -uo pipefail

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO" || exit 1
STAMP=${ACAL_WD_STAMP:-20260918}
SD=logs/acal_watchdog_state
RUNG=${ACAL_RUNG:-$( [ -f "$SD/rung" ] && tr -dc '0-9.' < "$SD/rung" || echo 3)}   # the campaign's rung, set by `resume`
TAG=${ACAL_RES_TAG:-acal}
LAUNCHER=slurm/acal_res.slurm
WATCH=logs/acal_watch_$STAMP.md
NODEPFX=nucla3m-a3meganodeset-
DISK_RESERVE=100; CASE_GB=60
DRY=${ACAL_DRY:-0}
mkdir -p "$SD" logs

run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; else "$@"; fi; }
note() { echo "$*"; [ "$DRY" = 1 ] || echo "- $(date -u +%Y-%m-%dT%H:%MZ) acal_ctl ($USER): $*" >> "$WATCH"; }
free_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }

source slurm/aires_env.sh moe >/dev/null 2>&1 || { echo "cannot activate the moe env" >&2; exit 1; }

# The slate, rung-aware. A queued task's %K is an index into the slate ITS ARRAY was
# submitted at, which need not be this one (array 1217 is rung 3; the 42-case array is
# rung 2), so every task goes through task_k/task_case -- see slurm/acal_slate.sh.
source slurm/acal_slate.sh
casedir() { echo "runs/aires/${SLATE[$1]}/res/$TAG"; }
initfile() { echo "runs/acal/inputs/${SLATE[$1]}_inputs.nc"; }

# --- our acal jobs in the queue: "id|state|node|k|raw|episode" ---------------------------
# k is the case's index in the CURRENT slate ("" if the task's case lies outside it);
# episode is always filled in.
acal_tasks() {
  local id st node k raw
  squeue -u "$USER" -n Vayuh-s2s -r -h -o "%i|%t|%N|%K|%A" 2>/dev/null | while IFS='|' read -r id st node k raw; do
    [ -n "$id" ] || continue
    scontrol show job "$id" 2>/dev/null | grep -q "Command=.*$(basename "$LAUNCHER")" || continue
    echo "$id|$st|$node|$(task_k "${id%_*}" "$k")|$raw|$(task_case "${id%_*}" "$k")"
  done
}

# --- node survey (sinfo idle is NOT free: other people's non-Slurm processes) ------------
survey() {
  local ours=" $1 " n out nv bad
  FREE=""; FOREIGN=""; SURVEY=""
  for n in 0 1 2 3 4 5 6 7; do
    if [[ "$ours" == *" $n "* ]]; then SURVEY+="  node$n: ours (acal task running)"$'\n'; continue; fi
    out=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODEPFX$n" \
          nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd, -)
    nv=$(echo "$out" | tr ',' '\n' | grep -c '^[0-9]\+$')
    bad=$(echo "$out" | tr ',' '\n' | awk '$1+0>=1000' | wc -l)
    if [ "$nv" -ne 8 ] || [ "$bad" -gt 0 ]; then
      FOREIGN+="$n "; SURVEY+="  node$n: IN USE by someone else (MB/card: ${out:-unreachable})"$'\n'
    else
      FREE+="$n "; SURVEY+="  node$n: free"$'\n'
    fi
  done
  FREE=${FREE% }; FOREIGN=${FOREIGN% }
}

max_nodes() { local m; m=$(cat "$SD/max_nodes" 2>/dev/null); [[ "$m" =~ ^[0-9]+$ ]] && echo "$m" || echo 8; }
watchdog_alive() { local p; p=$(cat "$SD/pid" 2>/dev/null) && [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
watchdog_start() {
  if watchdog_alive; then echo "watchdog already running (pid $(cat "$SD/pid"))"; return; fi
  if [ "$DRY" = 1 ]; then echo "DRY: start slurm/acal_watchdog.sh"; return; fi
  ACAL_RUNG=$RUNG nohup setsid bash slurm/acal_watchdog.sh > "logs/acal_watchdog_$STAMP.nohup" 2>&1 < /dev/null &
  sleep 3; watchdog_alive && echo "watchdog started (pid $(cat "$SD/pid"))" || echo "watchdog did not start; see logs/acal_watchdog_$STAMP.nohup" >&2
}

# --- status ----------------------------------------------------------------------------
cmd_status() {
  local tasks k ev d q id st node tk raw ep done_n red_n run_n pend_n todo_n line ours
  tasks=$(acal_tasks)
  ours=$(echo "$tasks" | awk -F'|' '$2=="R"||$2=="CG"{n=$3; sub(/.*-/,"",n); printf "%s ", n}')
  echo "acal campaign, rung >= $RUNG ($NCASE cases), tag $TAG    $(date -u +%Y-%m-%dT%H:%MZ)"
  [ -f "$SD/paused" ] && echo "STATE: PAUSED since $(cat "$SD/paused")" || echo "STATE: active"
  watchdog_alive && echo "watchdog: running (pid $(cat "$SD/pid"))" || echo "watchdog: NOT running"
  echo "node cap: at most $(max_nodes) node(s) held at once (each case = 1 node = 8 H100; change with: $0 nodes N)"
  echo "disk: $(free_gb) GB free on /home"
  echo
  printf "%3s  %-18s %-6s %-9s %-9s %-7s  %s\n" k case init res_result compared pruned queue
  done_n=0; red_n=0; run_n=0; pend_n=0; todo_n=0
  for ((k=0; k<NCASE; k++)); do
    ev=${SLATE[$k]}; d=$(casedir "$k"); q="-"
    line=$(echo "$tasks" | awk -F'|' -v k="$k" '$4!="" && $4==k')
    if [ -n "$line" ]; then
      IFS='|' read -r id st node tk raw ep <<< "$line"
      case "$st" in R|CG) q="RUNNING $id on ${node##*-} ($(squeue -h -j "$id" -o %M 2>/dev/null))"; run_n=$((run_n+1));;
                    PD)   q="pending $id"; pend_n=$((pend_n+1));;
                    *)    q="$st $id";; esac
    fi
    local res=n cmp=n pr=-
    [ -f "$d/res_result.json" ] && { res=y; done_n=$((done_n+1)); }
    [ -f "$d/compare.json" ] && cmp=y
    if [ "$res" = y ]; then pr=$( [ "$(find "$d" -name '*state*.nc' 2>/dev/null | wc -l)" -eq 0 ] && echo y || echo n ); fi
    [ "$res" = y ] && [ "$cmp" = y ] && [ "$pr" = y ] && red_n=$((red_n+1))
    [ "$res" = n ] && [ -z "$line" ] && todo_n=$((todo_n+1))
    printf "%3d  %-18s %-6s %-9s %-9s %-7s  %s\n" "$k" "$ev" "$( [ -f "$(initfile "$k")" ] && echo y || echo MISSING)" "$res" "$cmp" "$pr" "$q"
  done
  echo
  echo "finished $done_n/$NCASE (fully reduced $red_n); running $run_n; pending $pend_n; not queued $todo_n"
  line=$(echo "$tasks" | awk -F'|' '$4==""{printf "%s %s (%s) ", $1, $6, $2}')
  [ -n "$line" ] && echo "also queued, cases OUTSIDE the rung-$RUNG slate (submitted at another rung): $line"
  survey "$ours"
  echo "nodes:"; printf "%s" "$SURVEY"
  echo "watch log: $WATCH"
}

# --- pause -----------------------------------------------------------------------------
cmd_pause() {
  local drain=0 tasks id st node k raw ep n=0
  [ "${1:-}" = "--drain" ] && drain=1
  tasks=$(acal_tasks)
  if [ -z "$tasks" ]; then echo "no acal task in the queue; marking paused so the watchdog submits nothing"; fi
  # Tell the watchdog first: cancelled tasks are ours, not failures, and no re-pinning now.
  while IFS='|' read -r id st node k raw ep; do [ -n "$raw" ] && run touch "$SD/handled_$raw"; done <<< "$tasks"
  [ "$DRY" = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%MZ) by $USER$( [ $drain = 1 ] && echo ' (drain)')" > "$SD/paused"
  while IFS='|' read -r id st node k raw ep; do
    [ -n "$id" ] || continue
    if [ "$st" = "PD" ]; then
      if [ $drain = 1 ]; then run scontrol hold "$id"; note "held pending $id (case ${k:-?} $ep)"
      else run scancel "$id"; note "cancelled pending $id (case ${k:-?} $ep)"; fi
    else
      if [ $drain = 1 ]; then echo "leaving $id running on ${node##*-} (case ${k:-?} $ep); it will finish on its own"
      else
        local last; last=$(ls -t "runs/aires/$ep/res/$TAG/logs/" 2>/dev/null | head -1 | sed -E 's/-[0-9]+-try[0-9]+-shard[0-9]+\.log$//')
        run scancel "$id"; note "cancelled RUNNING $id on ${node##*-} (case ${k:-?} $ep, was in ${last:-?}); it resumes from its last finished leg"
      fi
    fi
    n=$((n+1))
  done <<< "$tasks"
  note "PAUSED$( [ $drain = 1 ] && echo ' (drain)'): $n task(s) handled; resume with: bash slurm/acal_ctl.sh resume"
  [ $drain = 1 ] || { sleep 5; echo; echo "queue now:"; squeue -u "$USER" -n Vayuh-s2s -o "%.10i %.2t %.10M %.20R"; }
}

# --- resume ----------------------------------------------------------------------------
cmd_resume() {
  local tasks id st node k raw ep queued="" todo="" ours="" nfree fg slots thr exc want new
  tasks=$(acal_tasks)
  while IFS='|' read -r id st node k raw ep; do
    [ -n "$id" ] || continue; [ -n "$k" ] && queued+="$k "
    { [ "$st" = R ] || [ "$st" = CG ]; } && ours+="${node##*-} "
  done <<< "$tasks"
  survey "$ours"
  echo "nodes:"; printf "%s" "$SURVEY"
  want=""; for n in $FOREIGN; do want+="$NODEPFX$n,"; done; want=${want%,}
  # held tasks from a --drain pause: re-pin away from busy nodes, then release
  while IFS='|' read -r id st node k raw ep; do
    [ -n "$id" ] || continue
    if scontrol show job "$id" 2>/dev/null | grep -q "JobState=PENDING.*Reason=JobHeldUser"; then
      run scontrol update JobId="$id" ExcNodeList="$want"; run scontrol release "$id"; note "released $id (case ${k:-?} $ep)"
    fi
  done <<< "$tasks"
  for ((k=0; k<NCASE; k++)); do
    [ -f "$(casedir "$k")/res_result.json" ] && continue
    [[ " $queued " == *" $k "* ]] && continue
    [ -f "$(initfile "$k")" ] || { echo "case $k ${SLATE[$k]} has no init frame; run: python -m acal.aprep --rung $RUNG" >&2; continue; }
    todo+="$k,"
  done
  todo=${todo%,}
  if [ -n "$todo" ]; then
    fg=$(free_gb); nfree=$(echo $FREE | wc -w)
    slots=$(( (fg - DISK_RESERVE) / CASE_GB )); [ "$slots" -lt 0 ] && slots=0
    thr=$(( nfree < slots ? nfree : slots ))
    [ "$thr" -gt $(( $(max_nodes) - $(echo $ours | wc -w) )) ] && thr=$(( $(max_nodes) - $(echo $ours | wc -w) ))
    [ "$thr" -lt 1 ] && thr=1
    echo "resubmitting cases [$todo] as one array, throttle $thr (free nodes [${FREE:-none}], disk $fg GB, node cap $(max_nodes)), excluding [${FOREIGN:-none}]"
    if [ "$DRY" = 1 ]; then
      echo "DRY: sbatch --parsable --array=$todo%$thr ${want:+--exclude=$want} $LAUNCHER"
    else
      new=$(ACAL_RUNG=$RUNG sbatch --parsable --array="$todo%$thr" ${want:+--exclude="$want"} "$LAUNCHER") || { echo "sbatch failed" >&2; exit 1; }
      record_rung "${new%%;*}"
      note "RESUMED: submitted job ${new%%;*} (rung $RUNG) --array=$todo%$thr excluding [${FOREIGN:-none}]"
    fi
  else
    echo "nothing to resubmit: every unfinished case is already queued"
  fi
  run rm -f "$SD/paused"
  [ "$DRY" = 1 ] || echo "$RUNG" > "$SD/rung"     # from now on status/pause/resume and the watchdog default to this rung
  watchdog_start
  echo; cmd_status
}

case "${1:-status}" in
  status) cmd_status;;
  pause) cmd_pause "${2:-}";;
  resume|play) cmd_resume;;
  nodes) if [ -n "${2:-}" ]; then [[ "$2" =~ ^[1-8]$ ]] || { echo "nodes: give a number 1-8" >&2; exit 1; }
           run bash -c "echo $2 > '$SD/max_nodes'"; note "node cap set to $2 node(s)"; fi
         echo "node cap: at most $(max_nodes) node(s) held at once";;
  watchdog) case "${2:-}" in start) watchdog_start;; stop) watchdog_alive && run kill "$(cat "$SD/pid")" && echo "watchdog stopped";; *) watchdog_alive && echo "running (pid $(cat "$SD/pid"))" || echo "not running";; esac;;
  *) sed -n '2,20p' "$0"; exit 1;;
esac
