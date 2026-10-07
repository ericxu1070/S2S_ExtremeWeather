# slurm/acal_slate.sh -- the acal slate, rung-aware. Sourced by slurm/acal_ctl.sh and
# slurm/acal_watchdog.sh (never run on its own). Needs $RUNG and $SD already set.
#
# The 42-episode catalog nests: rung 2 = 42 cases, rung 3 = 12, rung 4 = 5, each ordered by
# peak, so ARRAY TASK k MEANS A DIFFERENT CASE AT EACH RUNG (rung-3 task 5 is e24_c3, rung-2
# task 5 is e06_h2). Arrays submitted at different rungs share the queue -- the 12-case array
# 1217 was still running when the 42-case slate was opened on 2026-09-18 -- so nothing may
# read a queued task's %K as an index into the current slate. Everything goes through
# task_k / task_case here, which look up the rung the task's ARRAY was submitted at.
#
# Where an array's rung is recorded: $SD/rung_<array job id>, one integer. Written by the
# ctl/watchdog at sbatch time and by the launcher itself when a task starts (so a hand
# `sbatch` is covered too). An array with no record is assumed to be at $RUNG.
#
# Fills:  SLATE[k]        episode of case k of the CURRENT slate (rung >= $RUNG)
#         NCASE
#         K_OF[episode]   k in the current slate
#         CASE_AT["r:k"]  episode of task k of a slate at rung r, for every rung on record
# Defines: arr_rung ARR   -> the rung array ARR was submitted at
#          task_case ARR K -> episode of that array task ("" if past the end of its slate)
#          task_k ARR K    -> that case's index in the CURRENT slate ("" if outside it)

declare -A K_OF=() CASE_AT=()
SLATE=(); NCASE=0

_acal_known_rungs() {
  local f r
  { echo "2"; echo "3"; echo "4"; echo "$RUNG"
    for f in "$SD"/rung_*; do [ -f "$f" ] && { r=$(tr -dc '0-9.' < "$f"); [ -n "$r" ] && echo "$r"; }; done
  } | sort -u
}

_acal_load_slates() {
  local r k ep
  while read -r r k ep; do
    [ -n "$ep" ] || continue
    CASE_AT["$r:$k"]=$ep
    if [ "$r" = "$RUNG" ]; then SLATE[$k]=$ep; K_OF[$ep]=$k; fi
  done < <(python - $(_acal_known_rungs) <<'PY'
import sys, pandas as pd
df = pd.read_csv("runs/acal/catalog/conus_episodes_21d_2021_2025.csv")
for r in sys.argv[1:]:
    s = df[df.a_l_conus.abs() >= float(r)].sort_values("peak").reset_index(drop=True)
    for k, ep in enumerate(s.episode_id):
        print(r, k, ep)
PY
)
  NCASE=${#SLATE[@]}
}

arr_rung()  { local r=""; [ -f "$SD/rung_$1" ] && r=$(tr -dc '0-9.' < "$SD/rung_$1"); [ -n "$r" ] && echo "$r" || echo "$RUNG"; }
task_case() { echo "${CASE_AT["$(arr_rung "$1"):$2"]:-}"; }
task_k()    { local ep; ep=$(task_case "$1" "$2"); [ -n "$ep" ] && echo "${K_OF[$ep]:-}"; }
record_rung() { [ -n "${1:-}" ] && mkdir -p "$SD" && echo "$RUNG" > "$SD/rung_$1"; }

_acal_load_slates
