#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --array=0-7
#SBATCH --time=12:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/ens/%x-%A_%a.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/ens/%x-%A_%a.err

# The ENSEMBLE follow-up to the lead-time stability sweep: N GenCast walkers at ONE lead.
#
# Job 1189 rolled ONE member per lead. It found that divergence is not monotonic in lead -
# PNW@wk8 blew up at 45 d while PNW@wk10 never left bounds - and with n = 1 it could not
# separate "that lead is unstable" from "that member diverged". Its own conclusion was
# that the follow-up needs an ensemble for the STABILITY question too. This is that run:
#
#   100 members x PNW_HeatDome_2021 @ week 8 (56 d, init 2021-05-03 -> peak 2021-06-28)
#   GenCast only. No FCN3 arm - `--ens` refuses the score, prune and maps stages.
#
# Deliverable: the fraction of members inside the physical bounds at each of 19 three-day
# checkpoints, with a 95% Wilson interval, plus the loud/silent and worst-variable
# breakdown. Written to runs/astab/ensemble_<event>_wk<NN>{,_members}.csv + .json and
# figures/astab/ensemble_<event>_wk<NN>.png.
#
# The same mechanism runs at another lead with ONLY --weeks changed. The 56-day curve
# already gives a first-order S(L) for every L <= 56 d; a shorter-lead run then confirms
# it from that lead's OWN init, which is not the same experiment.
#
# WHAT THIS JOB DOES NOT TOUCH. The frozen 18-chain sweep lives in
# runs/astab/<event>/walkers/w0N/ and its stability.csv / stability.png. A member chain
# writes to runs/astab/<event>/ens/wk<NN>/m<MMM>/ and NOTHING else: `Chain.segment_dir`
# routes the stale/partial rmtree, the FCN3 path helpers raise rather than alias, and
# `result_path` routes to the ens tree. A week-8 member's walker SLOT is still 2, so
# every one of those is load-bearing.
#
# ---------------------------------------------------------------------------
# Budget (measured, job 1189, logs/Vayuh-s2s-1189-walk-try1-shard2.log)
#
#   per member    19 segments = 18 x 6 + a 4-step tail = 112 GenCast steps
#                 ~200 s/segment x 18 + 141 s tail ~ 62 min (28.6 s/step + ~26 s/segment
#                 overhead, 8 concurrent serial bf16 workers on one node)
#   per shard     ~13 min startup: 17 s model load off warm page cache + ~5 min XLA
#                 compile + the first segment. The JAX cache runs/models/jax_cache_0p25
#                 is 928 kB - effectively empty - so EVERY worker compiles.
#   total         ~103 GPU-h of rollout + ~0.2 h x NSHARDS of startup ~ 115 GPU-h
#   8 nodes       64 shards, <= 2 members each -> ~2.5 h wall (3.5 h at the 52 s/step
#                 ceiling). The startup is paid PER WORKER, so doubling the nodes does not
#                 quite halve the wall: 64 x ~13 min of model load and XLA compile is
#                 ~14 GPU-h of the total, against ~7 GPU-h on 4 nodes.
#   host RAM      ~120 GiB per worker; 8 per 1.86 TB node (proven, jobs 1160/1189)
#   disk          records ~15 MB + T2m-only slim diag ~1 GB. Transient states <= 2 x
#                 477 MB per LIVE worker (~61 GB on 8 nodes), released at each member's
#                 chain end. Unslimmed the diag would be 65 GB and unreleased the states
#                 906 GB, against 267 GB free on a 98%-full NFS: the slim and the release
#                 are what make the run possible, not tidiness.
#
# ---------------------------------------------------------------------------
# Prereqs, ALL on the LOGIN node (a GPU job is a pure cache-hit-then-compute; CLAUDE.md).
# The job re-runs the plan and refuses to start if it does not print READY: yes.
#
#   runs/xres/0p25/week8/inputs/PNW_HeatDome_2021_inputs.nc     the walker's init frames
#   runs/astab/refs/PNW_HeatDome_2021_era5_ref.nc               the day-3 physics gate
#   runs/astab/refs/bands.json                                  the measured CONUS bands
#   runs/models/clim_1990_2019_t2m_conus.nc                     105 x 237, NOT the 2 deg one
#
#   PYTHONPATH=. python -m astab.run --stage plan --ens 100 --event PNW_HeatDome_2021 \
#       --weeks 8 --nshards 64                      # must print READY: yes
#   PYTHONPATH=. python -m astab.run --check --ens 100 --event PNW_HeatDome_2021 --weeks 8
#   mkdir -p logs/ens                               # Slurm will NOT create it
#
# ---------------------------------------------------------------------------
# Submit
#
#   SMOKE, one node, ~15 min - exercises the loader, the compile, the physics gate, the
#   record, the slim and the release-skip on ONE member and ONE segment:
#     sbatch --array=0 --export=ALL,AIRES_STAB_ENS=1,AIRES_STAB_MAX_SEGMENTS=1,AIRES_ENS_NODES=1 \
#            slurm/aires_stab_ens.slurm
#
#   PRODUCTION, 8 nodes / 64 shards, ~2.5 h - the whole partition:
#     sbatch slurm/aires_stab_ens.slurm
#
#   Fewer nodes - the --array WIDTH and AIRES_ENS_NODES must AGREE (see the guard):
#     sbatch --array=0-3 --export=ALL,AIRES_ENS_NODES=4 slurm/aires_stab_ens.slurm  # ~4.5 h
#     sbatch --array=0-1 --export=ALL,AIRES_ENS_NODES=2 slurm/aires_stab_ens.slurm  # ~7.5 h
#
#   Re-walk the members one node lost (delete their directories first - a finished member
#   has no states left to resume from):
#     sbatch --array=0 --export=ALL,AIRES_ENS_NODES=1,AIRES_ENS_MEMBERS=17,41 \
#            slurm/aires_stab_ens.slurm
#
#   Monitor, and get the partial curve at any time (the reduce tolerates members still
#   in flight - that is what makes it the mid-flight monitor):
#     squeue -u $USER -n Vayuh-s2s
#     ls runs/astab/PNW_HeatDome_2021/ens/wk08/m*/step19/stab.json | wc -l
#     PYTHONPATH=. python -m astab.run --stage reduce --ens 100 \
#         --event PNW_HeatDome_2021 --weeks 8
#
# ---------------------------------------------------------------------------
# Two things about the shard map that WILL bite if changed carelessly
#
# 1. Every array task must receive an IDENTICAL --events/--weeks/--ens selection. The map
#    is `i % nshards` over the chain list, so two tasks that disagree about the list roll
#    overlapping members and leave others unrolled. That is why the selection is baked in
#    here rather than derived per task.
# 2. NSHARDS is AIRES_ENS_NODES x NGPU and is NEVER derived from SLURM_ARRAY_TASK_COUNT.
#    On `sbatch --array=2` (redo one node) that count is 1, so nshards would be 8, every
#    shard id would be >= 8, and all eight workers would select ZERO members and exit 0 -
#    a job that looks like a success and did nothing.
set -eo pipefail

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1

# Baked in, identical across every array task - see note 1 above.
EVENTS=${AIRES_STAB_EVENTS:-PNW_HeatDome_2021}
WEEKS=${AIRES_STAB_WEEKS:-8}
ENS=${AIRES_STAB_ENS:-100}
NNODES=${AIRES_ENS_NODES:-8}
NGPU=${AIRES_ENS_NGPU:-8}
NSHARDS=$(( NNODES * NGPU ))
T=${SLURM_ARRAY_TASK_ID:-0}
TAG="${SLURM_JOB_NAME:-astab-ens}-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
STAGGER=${AIRES_ENS_STAGGER:-20}

SEL=(--events "$EVENTS" --weeks "$WEEKS" --ens "$ENS")
[ -n "${AIRES_ENS_MEMBERS:-}" ] && SEL+=(--ens-members "$AIRES_ENS_MEMBERS")
WALK_EXTRA=""
[ -n "${AIRES_STAB_MAX_SEGMENTS:-}" ] && WALK_EXTRA="--max-segments $AIRES_STAB_MAX_SEGMENTS"

# HARD GUARD. A task id at or past NNODES has no shard range: its global shard ids run
# past NSHARDS, every worker selects zero members, and the job exits 0 having done
# nothing. Fail loudly instead, and say exactly what to resubmit.
if (( T >= NNODES )); then
  echo "ERROR: array task $T but AIRES_ENS_NODES=$NNODES (shards 0..$((NSHARDS - 1)))." >&2
  echo "       This task's global shard ids would be $((T * NGPU))..$((T * NGPU + NGPU - 1))," >&2
  echo "       past the shard count, so every worker would select ZERO members and exit 0." >&2
  echo "       The --array WIDTH and AIRES_ENS_NODES must agree. Resubmit with:" >&2
  echo "         sbatch --array=0-$((NNODES - 1)) --export=ALL,AIRES_ENS_NODES=$NNODES $0" >&2
  echo "       or, to use $((T + 1)) nodes:" >&2
  echo "         sbatch --array=0-$T --export=ALL,AIRES_ENS_NODES=$((T + 1)) $0" >&2
  exit 1
fi
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "$SLURM_ARRAY_TASK_COUNT" -ne "$NNODES" ]; then
  echo "WARNING: SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT but AIRES_ENS_NODES=$NNODES." >&2
  echo "         NSHARDS stays $NSHARDS (nodes x gpus), NEVER the array count - see note 2." >&2
  echo "         If this is a partial resubmit that is correct and intended; if it is a" >&2
  echo "         full run, the member set is being split over the wrong number of shards." >&2
fi

mkdir -p logs/ens
echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
echo "astab-ens: events=$EVENTS weeks=$WEEKS members=$ENS${AIRES_ENS_MEMBERS:+ (subset $AIRES_ENS_MEMBERS)}"
echo "           array task $T of $NNODES node(s) x $NGPU GPU(s) = $NSHARDS shard(s);" \
     "this node runs global shards $((T * NGPU))..$((T * NGPU + NGPU - 1))"

run_pool_once() {                  # run_pool_once <label> <attempt> <cmd...>
  local label=$1 attempt=$2; shift 2
  local pids=() fail=0 gs
  for ((s = 0; s < NGPU; s++)); do
    gs=$(( T * NGPU + s ))
    # STAGGERED. 32 processes each reading the 4.18 GB checkpoint and a 706 MB init
    # frame off the shared NFS at the same instant is job 1153's failure mode: eight
    # shards sat in D state for ~9 minutes with the GPUs idle. The sleep is inside the
    # background subshell so the launches themselves are not serialised.
    ( sleep $(( s * STAGGER ))
      CUDA_VISIBLE_DEVICES=$s AIRES_NSHARDS=$NSHARDS "$@" \
        --shard "$gs" --nshards "$NSHARDS" ) \
      > "logs/ens/${TAG}-${label}-try${attempt}-shard${gs}.log" 2>&1 &
    pids+=($!)
    echo "  launched $label global shard $gs on GPU $s (pid ${pids[-1]}, +$((s * STAGGER))s)"
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "ERROR: $label shard $((T * NGPU + i)) failed --" \
           "logs/ens/${TAG}-${label}-try${attempt}-shard$((T * NGPU + i)).log" >&2
      fail=1
    fi
  done
  return $fail
}

# Retry the WHOLE pool, not the failed shards: every stage is cache-aware per artifact, so
# a repeat pass costs seconds for what is already on disk and only redoes what is missing.
# This exists because the GPUs on this cluster are NOT reliably ours even under
# `--exclusive` -- Gate 3's job 1155 lost its score phase to an unrelated vLLM sweep
# holding 78 GB on every card. That contention is transient; a retry rides it out, and a
# real bug still fails all ATTEMPTS passes and stops the job. A shard that finished its
# members re-verifies them from their records in seconds.
ATTEMPTS=${AIRES_POOL_ATTEMPTS:-3}
RETRY_WAIT=${AIRES_RETRY_WAIT:-300}

run_pool() {
  local label=$1; shift
  for ((a = 1; a <= ATTEMPTS; a++)); do
    echo "  $label: attempt $a/$ATTEMPTS"
    if run_pool_once "$label" "$a" "$@"; then
      return 0
    fi
    if (( a < ATTEMPTS )); then
      echo "  $label: attempt $a failed; waiting ${RETRY_WAIT}s before retrying"
      echo "  (cached members are skipped, so a retry only redoes what is missing)"
      sleep "$RETRY_WAIT"
    fi
  done
  echo "ERROR: $label failed $ATTEMPTS time(s)" >&2
  return 1
}

# --------------------------------------------------------------------------- #
# The walk. There is no second phase: this run is GenCast-only.
# --------------------------------------------------------------------------- #
(
  source slurm/aires_env.sh moe

  # The READY gate. Refuse to burn 4 nodes on a missing init frame or a stale 2-degree
  # climatology; it also prints the ensemble budget and the shard histogram into this
  # task's .out, which is where a post-mortem starts. `set -e` stops the job on rc != 0.
  python -m astab.run --stage plan "${SEL[@]}" --nshards "$NSHARDS"

  echo "--- walk phase: $(date) ---"
  run_pool walk python -m astab.run --stage walk "${SEL[@]}" $WALK_EXTRA
)

echo "End: $(date)"
echo "Now reduce on the login node in moe (safe while other tasks are still running -"
echo "the reduce reports partial members rather than mixing them into the denominator):"
echo "  PYTHONPATH=. python -m astab.run --stage reduce --stage figs --ens $ENS --events $EVENTS --weeks $WEEKS"
echo "  ls -la figures/astab/ensemble_${EVENTS}_wk0${WEEKS}.png runs/astab/ensemble_${EVENTS}_wk0${WEEKS}*.{csv,json}"
