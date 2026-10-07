#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# AI+RES lead-time stability sweep (the astab/ package) -- how far can the walker and the
# scorer be pushed before they stop producing a physical atmosphere?
#
# EIGHTEEN chains over 8 GPUs: 2 events x 9 leads, ONE GenCast member and ONE FCN3 member
# each. The chains are inherently serial, so the second event is free wall-clock and tests
# the opposite season and the opposite tail.
#
#   weeks 4/6/8/10  (28/42/56/70 d)   FULL chains - the walker rolls to the peak
#   weeks 12..20    (84..140 d)       FCN3-ONLY extension - the walker rolls 3 d, purely
#                                     to launch the adapter-fed arm, and both FCN3 arms
#                                     roll the whole horizon
#
# The split is `astab.sconfig.STAB_WALK_WEEKS`, and it exists because job 1189 answered the
# walker's question (diverged on 3 of 8 chains by week 6) and left the scorer's open (clean
# at week 10 on every rollout of both arms - a floor, not a limit). Weeks 12..20 buy the
# open question: 8,840 FCN3 steps against 60 GenCast steps.
#
# `reduce` will not report a GenCast verdict at an extension lead - every CSV row carries
# `reached_peak` and `_model_survival` drops the (model, week) pairs where it is 0. Moving
# 12..20 into STAB_WALK_WEEKS (or AIRES_STAB_WALK_WEEKS=4,6,...,20) rolls full walker
# chains there instead; budget it with `--stage plan` first, a 140-day chain is 280 GenCast
# steps (~4 h of one H100).
#
# Three pools on ONE allocation, in TWO conda envs -- they cannot share a process (JAX vs
# PyTorch/earth2studio) but they can share a node, and the second pool then reads walker
# states the first left in page cache:
#
#   walk        moe  : chain i on GPU i, 6-step (3 d) segments to the peak, global restart
#                      state each segment. 0.25 deg needs ~64 GB of an 80 GB H100 for ONE
#                      member, so each worker is serial + bf16 -- the same configuration
#                      as slurm/xres_pool_0p25_week3.slurm and slurm/aires_gate3.slurm.
#   score free  fcn3 : the ADAPTER-FREE arm. FCN3 from the ERA5 IC at each chain's own
#                      init, rolled to the peak. Isolates the scorer's stability from the
#                      walker's drift.
#   score fed   fcn3 : the ADAPTER-FED arm. FCN3 from each chain's lead-3 d walker state,
#                      through aires/adapter.py. Confirms the coupled path survives, and
#                      it is the LONGEST coupled rollout available at each lead.
#
# The two score arms are separate pools rather than one, because `fcn3.fevents` resolves
# FCN3_WEEKS at MODULE IMPORT: the free arm needs one process per lead, and folding it
# into the fed arm's process would mean two FCN3 models resident on one card for no gain.
#
# Cost (SECONDS_PER_GENCAST_STEP = 52.0, SECONDS_PER_FCN3_STEP = 1.40, both measured).
# The original 8 chains cost 784 GenCast + 3,040 FCN3 steps and are on disk; ADDING weeks
# 12..20 to them costs:
#   walk        60 GenCast steps ~0.9 GPU-h -> 10 chains x ONE 6-step segment. Dominated
#               by the ~15 min model load and the ~5 min XLA compile, not by the rollout.
#   score       8,840 FCN3 steps ~3.4 GPU-h -> set by the longest rollout, 560 steps
#               (week 20, adapter-free) = ~13 min. Each free-arm chain is its OWN
#               subprocess (FCN3_WEEKS is read at import), so a shard holding 3 chains
#               pays 3 model loads; the ~9 min load is what dominates a short shard
#               (8 shards reading the shared 4.18 GB pickle off NFS sit in D state;
#               a3mega job 1153).
# The 6 h walltime is headroom. Every stage is cache-aware per artifact, so a timeout is
# resumed by resubmitting, not restarted - and the finished 4/6/8/10 chains skip in
# seconds even though they were pruned (the landing check falls back to their records).
#
# Host RAM: one serial 0.25 deg GenCast worker peaks ~120 GiB on this cluster (measured;
# CLAUDE.md's 237 GiB figure is Derecho's 487 GB nodes), so 8 concurrent is ~960 GiB of the
# node's 1.86 TB. Job 1160 proved 8-way concurrency works there. WATCH THE FIRST MINUTES
# anyway -- if RSS tracks toward 237 GiB, cut to one event with
#   sbatch --export=ALL,AIRES_STAB_EVENTS=PNW_HeatDome_2021 slurm/aires_stab.slurm
# rather than risk an OOM at hour two.
#
# Disk: everything lands under runs/astab/ -- its own tree since 2026-08-27, where it
# used to be the `stab` tag inside runs/aires/<event>/res/. For the extension: ~4.8 GB of
# retained walker states (one per extension chain), ~0.4 GB of diagnostics, and ~51 GB of
# retained pre-crop FCN3 Zarrs (--keep-zarr; items 4 and 7 need GLOBAL z500 and the
# production cube is CONUS-cropped). That Zarr figure scales with FCN3 STEPS, not chains --
# a week-20 rollout is twice a week-10 one -- and this filesystem had 303 GB free on
# 2026-08-27, so run `--stage reduce` then `--stage prune` before adding more leads.
# `--stage prune` releases the states and the Zarrs once the per-chain JSONs exist. The
# adapter-free arm's cubes are the one exception: they are written by fcn3/run_fcn3.py
# unmodified and stay in runs/fcn3/week<N>/shards/.
#
# Prereqs -- ALL of them on the LOGIN node (GPU jobs are pure cache-hit-then-compute,
# CLAUDE.md), and this job re-runs the check and refuses to start if it does not pass:
#   for W in 6 8 10 12 14 16 18 20; do XRES_EVENTS_SEL=PNW_HeatDome_2021,WinterStorm_Uri_2021 \
#       XRES_PREP_SKIP_MODELS=1 python run_xres.py --stage prep --res 0p25 --weeks $W; done
#   for W in 4 6 8 10 12 14 16 18 20; do FCN3_WEEKS=$W \
#       FCN3_EVENTS=PNW_HeatDome_2021,WinterStorm_Uri_2021 \
#       bash slurm/aires_env.sh fcn3 python fcn3/run_fcn3.py --stage prep; done
#                       # one PROCESS per week (FCN3_WEEKS is read at import), and the
#                       # `fcn3` ENV: stage_prep imports earth2studio, which moe does not
#                       # have. The xres prep above and every astab stage are moe.
#   PYTHONPATH=. python -m astab.run --stage refs        # ERA5 scalar series + T2m truth fields
#                                                        #   128 times/event once the extension
#                                                        #   leads are in; it REBUILDS when the
#                                                        #   cache no longer covers them
#   PYTHONPATH=. python -m astab.run --stage calibrate   # measured bands, CPU, free
#   PYTHONPATH=. python -m astab.run --check             # every segment shape, no GPU
#   PYTHONPATH=. python -m astab.run --stage plan        # must print READY: yes
#
# Submit:
#   sbatch slurm/aires_stab.slurm
#   sbatch --export=ALL,AIRES_STAB_PHASE=score slurm/aires_stab.slurm      # score only
#   sbatch --export=ALL,AIRES_STAB_WEEKS=4 slurm/aires_stab.slurm          # one lead
#   sbatch --export=ALL,AIRES_STAB_WEEKS=12,14,16,18,20 slurm/aires_stab.slurm   # the
#                                                        #   extension only - identical
#                                                        #   result to the full submit,
#                                                        #   which skips the cached 8
#
# Then, on the login node in moe:
#   PYTHONPATH=. python -m astab.run --stage reduce --stage figs --stage maps
#   PYTHONPATH=. python -m astab.run --stage prune
set -eo pipefail

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1

EVENTS=${AIRES_STAB_EVENTS:-PNW_HeatDome_2021,WinterStorm_Uri_2021}
WEEKS=${AIRES_STAB_WEEKS:-4,6,8,10,12,14,16,18,20}
NGPU=${AIRES_NSHARDS:-8}
PHASE=${AIRES_STAB_PHASE:-all}
MEMBERS=${AIRES_STAB_MEMBERS:-1}
TAG="${SLURM_JOB_NAME:-astab}-${SLURM_JOB_ID:-local}"
SEL=(--events "$EVENTS" --weeks "$WEEKS")
# The smoke path: AIRES_STAB_MAX_SEGMENTS=2 AIRES_NSHARDS=1 AIRES_STAB_PHASE=walk
# AIRES_STAB_WEEKS=4 AIRES_STAB_EVENTS=PNW_HeatDome_2021 -- one GPU, ~10 min, and it
# exercises the loader, the XLA compile, the physics gate, the record and the prune.
WALK_EXTRA=""
[ -n "${AIRES_STAB_MAX_SEGMENTS:-}" ] && WALK_EXTRA="--max-segments $AIRES_STAB_MAX_SEGMENTS"

mkdir -p logs
echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
echo "astab: events=$EVENTS weeks=$WEEKS gpus=$NGPU phase=$PHASE members=$MEMBERS"

run_pool_once() {                  # run_pool_once <label> <attempt> <cmd...>
  local label=$1 attempt=$2; shift 2
  local pids=() fail=0
  for ((s = 0; s < NGPU; s++)); do
    CUDA_VISIBLE_DEVICES=$s AIRES_NSHARDS=$NGPU "$@" --shard "$s" --nshards "$NGPU" \
      > "logs/${TAG}-${label}-try${attempt}-shard${s}.log" 2>&1 &
    pids+=($!)
    echo "  launched $label shard $s on GPU $s (pid ${pids[-1]})"
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "ERROR: $label shard $i failed -- logs/${TAG}-${label}-try${attempt}-shard${i}.log" >&2
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
# real bug still fails all ATTEMPTS passes and stops the job.
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
      echo "  (cached work is skipped, so a retry only redoes what is missing)"
      sleep "$RETRY_WAIT"
    fi
  done
  echo "ERROR: $label failed $ATTEMPTS time(s)" >&2
  return 1
}

# --------------------------------------------------------------------------- #
# Phase 1: the chains (GenCast, moe)
# --------------------------------------------------------------------------- #
if [[ "$PHASE" == "all" || "$PHASE" == "walk" ]]; then
(
  source slurm/aires_env.sh moe

  # Refuse to burn an allocation on a cache or a prerequisite set that is not ready.
  python -m astab.run --stage plan "${SEL[@]}" --nshards "$NGPU"

  echo "--- walk phase: $(date) ---"
  run_pool walk python -m astab.run --stage walk "${SEL[@]}" $WALK_EXTRA
)
fi

# --------------------------------------------------------------------------- #
# Phase 2: the scorer, adapter-FREE (FCN3 from the ERA5 IC, fcn3)
# --------------------------------------------------------------------------- #
if [[ "$PHASE" == "all" || "$PHASE" == "score" || "$PHASE" == "free" ]]; then
(
  source slurm/aires_env.sh fcn3
  # Global z500 alongside the usual outputs: items 4 and 7 need it and the production cube
  # is CONUS-cropped with no geopotential at all. Opt-in, so no frozen cube is affected.
  export FCN3_EXTRA_VARS=z500,t50
  echo "CPU: $(nproc) cores / $NGPU shards -> OMP_NUM_THREADS=$OMP_NUM_THREADS per shard"

  echo "--- score phase, adapter-free arm: $(date) ---"
  run_pool score-free python -m astab.run --stage score --arm free "${SEL[@]}" \
    --members "$MEMBERS"
)
fi

# --------------------------------------------------------------------------- #
# Phase 3: the scorer, adapter-FED (FCN3 from the walker at 3 d, fcn3)
# --------------------------------------------------------------------------- #
if [[ "$PHASE" == "all" || "$PHASE" == "score" || "$PHASE" == "fed" ]]; then
(
  source slurm/aires_env.sh fcn3
  export FCN3_EXTRA_VARS=z500,t50

  echo "--- score phase, adapter-fed arm: $(date) ---"
  run_pool score-fed python -m astab.run --stage score --arm fed "${SEL[@]}" \
    --members "$MEMBERS"
)
fi

echo "End: $(date)"
echo "Now reduce on the login node in moe:"
echo "  PYTHONPATH=. python -m astab.run --stage reduce --stage figs --stage maps --events $EVENTS --weeks $WEEKS"
echo "  PYTHONPATH=. python -m astab.run --stage prune  --events $EVENTS --weeks $WEEKS"
