#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=02:15:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err
#
# GPU test of the walker OOM fix: XLA_PYTHON_CLIENT_PREALLOCATE=true (the BFC arena
# reserved once at startup) against the =false that array 1209 ran under.
#
# WHAT FAILED. Array 1209 task 4 (job 1214), e23_h3_20231228, leg 1: all 8 shards, all 3
# attempts -- 24 of 24 shard-attempts -- died with
#   RESOURCE_EXHAUSTED: Out of memory while trying to allocate 19.59GiB
#   [executable_name='jit_concatenate']
# surfacing in write_state -> to_netcdf -> np.asarray.
#
# THE DIAGNOSIS, CORRECTED. That 19.59 GiB is NOT a device-side concatenate at
# serialisation time. Three things in the shard logs say so: the BFC give-up prints BEFORE
# rollout.py:437's FutureWarning, which precedes the first rollout concat; the six
# "step k/6" lines print in under 3 s against a measured ~31 s/step, i.e. JAX async
# dispatch had only QUEUED them and the error surfaced at the first blocking host fetch;
# and the largest jnp.concatenate anywhere on the walker/write path is 162 MB. The request
# is made during the FIRST predictor call. 'jit_concatenate' is the first CONSUMER of
# jit_apply_fn's outputs (_get_next_inputs), so the name is ATTRIBUTION, not the source.
#
# What ran out is the BFC ARENA EXTENSION: the driver refused a new region on the
# 32.00 -> 21.00 GiB ladder (cuda_executor.cc:1182, 20 lines) while BFC's own map showed a
# ~59 GiB arena a third free. With PREALLOCATE=false the arena grows piecemeal, so its
# size at the first execution is HISTORY-DEPENDENT -- which is why e02_c4_20210218 passed
# 7 of 8 shards in the same array: its arena had already grown to fill the card (shard 0
# logged the driver refusing even a 128 MiB -> 2 MiB BFC growth) and the run completed
# anyway, proving the whole computation fits inside the 72 GB MEM_FRACTION cap.
# PREALLOCATE=true takes that 72 GB once at startup, so the outcome turns on PEAK use, not
# on growth history -- and a foreign tenant on a card fails the shard in seconds at
# startup instead of minutes later.
#
# (Commit a678a66's _host_materialize fixes a real but SEPARATE defect -- Dataset.compute()
# is a no-op on xarray_jax's JaxArrayWrapper, so states stayed device-resident into
# to_netcdf -- and does not touch the 19.59 GiB request. Both are in effect here.)
#
# WHY TWO POOLS, ONE JOB.
#   A  e23_h3_20231228, preallocate=true  (AIRES_XLA_PREALLOCATE unset => the new default).
#      The case that failed 24/24. This is the campaign question.
#   B  e12_c3_20221121, preallocate=false (AIRES_XLA_PREALLOCATE=false). The control: the
#      OLD config, on the SAME node, minutes later.
# Reading them: B failing with the 19.59 GiB signature while A passed is the causal
# evidence. B ALSO passing does not refute A -- it means the old config is not
# deterministic on this node (its arena history differed, as e02's did), and the campaign
# still runs with A's config either way. B is informational; the job exits with A's code.
# B is not a throwaway run: e12 is task 1 of the rung-3 slate and never reached the rollout
# on 09-16, so its leg 1 banks 32 real states the campaign reuses.
#
# WHY THE WHOLE NODE, 8 SHARDS. a3mega is OverSubscribe=EXCLUSIVE -- one card costs the
# same node-time as eight. Eight concurrent JAX processes each holding the ~64 GB 0.25 deg
# checkpoint IS the failure condition; a one-shard test would not reproduce it.
#
# --attempts 1 on purpose: a test must fail fast, not retry the whole pool 3x with 300 s
# waits between attempts (that is the production setting, for transient GPU theft).
#
# --nodelist is deliberately NOT set here. slurm/submit_when_free.sh passes it on the
# sbatch command line once it finds a node whose 8 cards are all under 1000 MB (sinfo
# "idle" is meaningless here; teammates hold 63-81 GB/card outside Slurm). Submit with:
#   nohup slurm/submit_when_free.sh slurm/acal_walktest.slurm > logs/walktest_watch.log 2>&1 &
#
# Exit codes: rcA from pool A (0 = the fix holds); 2 prep failed; 3 a foreign tenant held a
# card at startup; 4 slurm/aires_env.sh does not honour AIRES_XLA_PREALLOCATE, which would
# make the A/B comparison void.
set -o pipefail   # NOT -e: both pools' exit codes are captured by hand.

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
source "$REPO/slurm/aires_env.sh" moe
cd "$REPO" || exit 1

# The calibration config, exported here so run_pool's env=dict(os.environ) carries it into
# every shard (aconfig reads these at import).
export AIRES_N_WALKERS=32
export AIRES_M_MEMBERS=6
export AIRES_NSHARDS=8
export AIRES_MEMSTATS=1        # walker.py's allocator probe: device 0 memory_stats() at
                               # the end of each roll_segment and both sides of the
                               # host materialisation, so a PASS still shows the headroom.
TAG=acal
EVENT_A=e23_h3_20231228        # failed 24/24 shard-attempts on 2026-09-16
EVENT_B=e12_c3_20221121        # rung-3 task 1, leg 1 unrolled

echo "Host:  $(hostname)"
echo "Start: $(date -u)"
echo "HEAD:  $(git rev-parse --short HEAD) dirty=$(git status --short | wc -l) files"
echo "--- walker env in effect ---"
env | grep -E 'XLA_PYTHON|AIRES_|XRES_|JAX_' | sort
echo "----------------------------"

# The A/B comparison is only meaningful if the env script reads the override.
if ! grep -q 'AIRES_XLA_PREALLOCATE' "$REPO/slurm/aires_env.sh"; then
  echo "PRECONDITION FAILED: slurm/aires_env.sh does not read AIRES_XLA_PREALLOCATE."
  echo "  Pool B could not switch the old behaviour back on; the test would be void."
  exit 4
fi
if [ "${XLA_PYTHON_CLIENT_PREALLOCATE:-<unset>}" != "true" ]; then
  echo "PRECONDITION FAILED: XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-<unset>}"
  echo "  with AIRES_XLA_PREALLOCATE unset; pool A must run with 'true'."
  exit 4
fi

# --------------------------------------------------------------------------- #
# Startup guard. A shard placed on a card a teammate is already holding dies in seconds,
# and that must never be read as the fix failing. sinfo "idle" does not mean free.
# --------------------------------------------------------------------------- #
guard() {  # $1 = label. Exit 3 if any card is held by someone else.
  local SMI NCARD BUSY
  SMI=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  echo "--- cards at $1 (index, MB used) ---"
  echo "$SMI"
  NCARD=$(printf '%s\n' "$SMI" | grep -c .)
  BUSY=$(printf '%s\n' "$SMI" | awk -F'[,[:space:]]+' '$2+0 > 2000 {printf "gpu%s=%sMB ", $1, $2}')
  if [ "$NCARD" -ne 8 ] || [ -n "$BUSY" ]; then
    echo "CONTENTION at $1: ${BUSY:-only $NCARD card(s) visible, expected 8}"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo "A foreign tenant held a card. Nothing was tested; resubmit elsewhere."
    echo "Do not kill what you find -- it is not ours."
    exit 3
  fi
}
# Job 1215 (2026-09-17) passed this guard at 11:16:32 and lost every card to a teammate's
# eval launched at 11:16:38, while prep and the checkpoint load were still on the host.
# So the guard runs again immediately before each walk pool, and the shards themselves
# now reserve the card as their first act (run_aires.stage_walk).
guard startup

# --------------------------------------------------------------------------- #
# Per-card memory sampler, 5 s cadence, for the whole job.
# --------------------------------------------------------------------------- #
GPUCSV=$REPO/logs/acal_walktest_gpumem_${SLURM_JOB_ID:-local}.csv
: > "$GPUCSV"
(
  while true; do
    {
      echo "$(date -u +%FT%TZ),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
        | sed 's/^/  app,/'
    } >> "$GPUCSV"
    sleep 5
  done
) >/dev/null 2>&1 &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT
echo "gpu sampler pid=$SAMPLER -> $GPUCSV"

# --------------------------------------------------------------------------- #
# POOL A -- the candidate production config (preallocate=true).
# --------------------------------------------------------------------------- #
echo
echo "=== POOL A  event=$EVENT_A  preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE  $(date -u) ==="
# Cache VERIFY only. The init frame is the login node's job; anything that downloads or
# builds here is a bug, not a slow start. --no-seed: seeding is never a GPU job's side effect.
python -m aires.run_aires --stage prep --event "$EVENT_A" --tag "$TAG" --no-seed \
  || { echo "PREP FAILED (pool A): missing init, missing calibration, or no disk."; exit 2; }

echo "--- pool A walk leg 1: $(date -u)"
tA0=$SECONDS
guard "pool A launch"
python -m aires.run_aires --stage walk --event "$EVENT_A" --tag "$TAG" --leg 1 --attempts 1
rcA=$?
tA=$((SECONDS - tA0))
nA=$(ls runs/aires/$EVENT_A/res/acal/walkers/w*/step01/state.nc 2>/dev/null | wc -l)
echo "--- pool A rc=$rcA  ${tA}s  states=$nA/$AIRES_N_WALKERS  $(date -u)"

# --------------------------------------------------------------------------- #
# POOL B -- the control: the old behaviour, same node, minutes later.
# --------------------------------------------------------------------------- #
export AIRES_XLA_PREALLOCATE=false
# Re-source so this shell's printed env matches what the shards will compute for
# themselves: run_pool re-runs aires_env.sh per shard with env=dict(os.environ), so
# AIRES_XLA_PREALLOCATE is the only lever that actually reaches the JAX process.
source "$REPO/slurm/aires_env.sh" moe
cd "$REPO" || exit 1
echo
echo "=== POOL B  event=$EVENT_B  preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE  $(date -u) ==="
if [ "${XLA_PYTHON_CLIENT_PREALLOCATE:-}" != "false" ]; then
  echo "WARNING: AIRES_XLA_PREALLOCATE=false did not take; pool B is NOT the old config."
fi

rcB=0; nB=0; tB=0
python -m aires.run_aires --stage prep --event "$EVENT_B" --tag "$TAG" --no-seed
if [ $? -ne 0 ]; then
  echo "PREP FAILED (pool B) -- informational only, pool A above is the campaign answer."
  rcB=2
else
  echo "--- pool B walk leg 1: $(date -u)"
  tB0=$SECONDS
  python -m aires.run_aires --stage walk --event "$EVENT_B" --tag "$TAG" --leg 1 --attempts 1
  rcB=$?
  tB=$((SECONDS - tB0))
fi
nB=$(ls runs/aires/$EVENT_B/res/acal/walkers/w*/step01/state.nc 2>/dev/null | wc -l)
echo "--- pool B rc=$rcB  ${tB}s  states=$nB/$AIRES_N_WALKERS  $(date -u)"

# --------------------------------------------------------------------------- #
# Epilogue.
# --------------------------------------------------------------------------- #
kill "$SAMPLER" 2>/dev/null
echo
echo "--- cards at exit (index, MB used) ---"
nvidia-smi --query-gpu=index,memory.used --format=csv
echo "gpu memory trace: $GPUCSV"
echo "shard logs:       runs/aires/{$EVENT_A,$EVENT_B}/res/$TAG/logs/walk-leg01-${SLURM_JOB_ID:-local}-try1-shard*.log"

if [ "$rcA" -eq 0 ]; then SA=PASS; else SA=FAIL; fi
if [ "$rcB" -eq 0 ]; then SB=PASS; else SB=FAIL; fi
echo
echo "RESULT A (preallocate=true,  $EVENT_A): $SA rc=$rcA states=$nA/$AIRES_N_WALKERS ${tA}s"
echo "RESULT B (preallocate=false, $EVENT_B): $SB rc=$rcB states=$nB/$AIRES_N_WALKERS ${tB}s"
echo "A is the campaign question; B is informational. B=FAIL with the 19.59 GiB signature"
echo "while A=PASS is the causal evidence; B=PASS means the old config is merely"
echo "nondeterministic on this node, and the campaign still runs with A's config."
echo "End: $(date -u)"
exit "$rcA"
