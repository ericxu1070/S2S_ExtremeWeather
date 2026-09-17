#!/bin/bash
# Wait for a genuinely free a3mega node, then sbatch onto it.
#
#   slurm/submit_when_free.sh [--survey-only] <sbatch args ...>
#   nohup slurm/submit_when_free.sh slurm/acal_walktest.slurm > logs/walktest_watch.log 2>&1 &
#
# WHY THIS EXISTS. Slurm's own view of this partition is not usable for placement.
# a3mega is OverSubscribe=EXCLUSIVE, so a job takes a whole 8xH100 node -- but teammates
# run NON-Slurm processes on those nodes (vLLM servers, inkling/continuum probes) that
# hold 60-81 GB per card while sinfo still reports the node "idle". The 0.25 deg GenCast
# checkpoint needs ~64 GB of an 80 GB card, so a job landing on such a node dies in
# seconds. The only honest test is nvidia-smi on the node itself, over SSH.
#
# FREE means: the node answers, it reports exactly 8 cards, and every one of them is under
# 1000 MB. Anything else -- a partial answer, a timeout, a refused connection, 7 values,
# one card at 63 GB -- counts as BUSY. There is no partial credit; a single held card
# spoils the allocation.
#
# It submits ONCE and exits. It refuses to submit if this user already has a Vayuh-s2s job
# in the queue, so a watcher left running from an earlier session cannot double-book a node.
# Never kill what the survey finds; it is not ours.
#
# Exit: 0 submitted | 1 sbatch failed | 2 usage | 4 refused, Vayuh-s2s already queued
#       5 timeout, no free node in 14 h
#
# The survey is sequential over 8 nodes at up to 25 s each, so a round costs 45 s plus the
# survey time -- a few seconds when the nodes answer, up to ~4 min when they hang. That is
# the intended cost: a hung node is a busy node.
set -o pipefail

NODES="0 1 2 3 4 5 6 7"
NODE_PREFIX=nucla3m-a3meganodeset-
THRESH_MB=1000          # a free card; the checkpoint needs ~64 GB, so anything held loses
NCARDS=8
INTERVAL=45             # seconds between rounds
MAX_HOURS=14
QUIET_ROUNDS=10         # print a survey line on round 1, then every 10th
JOBNAME=Vayuh-s2s

utc() { date -u +%FT%TZ; }

SURVEY_ONLY=0
if [ "${1:-}" = "--survey-only" ]; then SURVEY_ONLY=1; shift; fi
if [ "$SURVEY_ONLY" -eq 0 ] && [ "$#" -lt 1 ]; then
  echo "usage: $(basename "$0") [--survey-only] <sbatch args ...>" >&2
  echo "   e.g. $(basename "$0") slurm/acal_walktest.slurm" >&2
  exit 2
fi

# ---------------------------------------------------------------------------- #
# probe_node <n> -- sets NODE_STATE (free|busy|partial|noanswer), NODE_CSV, NODE_DESC.
# Read-only: it runs nvidia-smi on the node and nothing else.
# ---------------------------------------------------------------------------- #
probe_node() {
  local n=$1 csv v nb=0 mn mx x
  csv=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 "${NODE_PREFIX}${n}" \
          nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '\r' | paste -sd, -)
  NODE_CSV=$csv
  if [ -z "$csv" ]; then
    NODE_STATE=noanswer; NODE_DESC=noanswer; return
  fi
  local IFS=,
  read -r -a v <<< "$csv"
  unset IFS
  if [ "${#v[@]}" -ne "$NCARDS" ]; then
    NODE_STATE=noanswer; NODE_DESC="noanswer(${#v[@]}vals)"; return
  fi
  mn=999999; mx=-1
  for x in "${v[@]}"; do
    case "$x" in ''|*[!0-9]*) NODE_STATE=noanswer; NODE_DESC=noanswer; return;; esac
    [ "$x" -ge "$THRESH_MB" ] && nb=$((nb + 1))
    [ "$x" -lt "$mn" ] && mn=$x
    [ "$x" -gt "$mx" ] && mx=$x
  done
  if [ "$nb" -eq 0 ]; then
    NODE_STATE=free;    NODE_DESC="FREE(max${mx}M)"
  elif [ "$nb" -eq "$NCARDS" ]; then
    NODE_STATE=busy;    NODE_DESC="busy($((mn / 1024))-$((mx / 1024))G)"
  else
    NODE_STATE=partial; NODE_DESC="partial($nb/$NCARDS,$((mn / 1024))-$((mx / 1024))G)"
  fi
}

# ---------------------------------------------------------------------------- #
if [ "$SURVEY_ONLY" -eq 1 ]; then
  echo "$(utc) survey-only (read-only; nothing will be submitted)"
  for n in $NODES; do
    probe_node "$n"
    printf 'node%s  %-22s state=%-8s raw=%s\n' "$n" "$NODE_DESC" "$NODE_STATE" "${NODE_CSV:-<none>}"
  done
  exit 0
fi

DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
echo "$(utc) watching $NODE_PREFIX{$(echo "$NODES" | tr ' ' ',')} for a node with $NCARDS cards all < ${THRESH_MB} MB"
echo "$(utc) will submit: sbatch --nodelist=<node> $*"

round=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  round=$((round + 1))
  line=""; chosen=""; chosen_csv=""
  for n in $NODES; do
    probe_node "$n"
    line="$line node$n=$NODE_DESC"
    if [ "$NODE_STATE" = free ]; then chosen=$n; chosen_csv=$NODE_CSV; break; fi
  done

  if [ -n "$chosen" ]; then
    # Never double-book. An earlier watcher, or the production array, owns the name too.
    queued=$(squeue -h -u "$USER" -n "$JOBNAME" 2>/dev/null)
    if [ -n "$queued" ]; then
      echo "$(utc) REFUSED: node $chosen is free but $USER already has $JOBNAME in the queue:"
      echo "$queued"
      echo "Cancel or wait for it, then rerun this watcher."
      exit 4
    fi
    out=$(sbatch --nodelist="${NODE_PREFIX}${chosen}" "$@" 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "ERROR sbatch failed: $out"
      exit 1
    fi
    jid=$(printf '%s\n' "$out" | awk '/[0-9]$/ {print $NF}' | tail -1)
    echo "SUBMITTED job=${jid:-unknown} node=$chosen at $(utc) survey=$chosen_csv"
    exit 0
  fi

  if [ "$round" -eq 1 ] || [ $((round % QUIET_ROUNDS)) -eq 0 ]; then
    echo "$(utc) round $round:$line"
  fi
  sleep "$INTERVAL"
done

echo "TIMEOUT no free node in ${MAX_HOURS}h"
exit 5
