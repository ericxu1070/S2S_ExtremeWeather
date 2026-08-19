#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# AI+RES production run -- the DMC loop on real GenCast walkers and FCN3 scores.
#
# This job is ONE process: aires/run_aires.py --stage res, the coordinator. It launches
# the two 8-shard GPU pools itself, one leg at a time, through slurm/aires_env.sh:
#
#   leg k   walk   moe  : N GenCast segments, 3 d each, one per walker slot
#           score  fcn3 : N FCN3 forecasts, M members, from t_k to the event peak
#           resample     : pivotal sampling on the standardized score  (CPU, seconds)
#
# The pools ALTERNATE rather than coexist because they cannot share a card: the 0.25 deg
# GenCast checkpoint occupies ~64 GB of an 80 GB H100 and FCN3 needs most of one too. That
# is also why the coordinator, not a queue, sequences them -- see the module docstring.
#
# Cost at the pilot's defaults (N=64, M=6, C=(0,1,1.4,1.8,2) at 3/6/9/12/15 d, 21 d
# horizon), from `--stage prep`:
#   walk    2,496 GenCast steps @ ~28 s = ~19.4 GPU-h  -> ~2.4 h over 8 GPUs
#   score  64,512 FCN3 steps    @ ~1 s  = ~17.9 GPU-h  -> ~2.2 h over 8 GPUs
#   plus one model load per pool launch: FCN3's 4.18 GB pickle costs ~9 min on the FIRST
#   leg with 8 concurrent readers and is page-cached for the rest of the job.
# The 12 h walltime is headroom. A timeout is RESUMED by resubmitting, not restarted:
# every leg already on disk replays in seconds and no GPU pool is launched for it.
#
# Disk: ~97 GB (states are pruned to the 2 most recent segments; diagnostics never are).
#
# Prereqs -- on the LOGIN node, in moe (GPU jobs must be pure cache-hit-then-compute):
#   PYTHONPATH=. python -m aires.run_aires --stage prep      # must print READY: yes
# It checks the init frames, the derived-channel calibration and the free disk, seeds the
# tree from Gate 3's walkers where that is provably the same rollout, and freezes the
# configuration. This job re-runs the check and refuses to start if it does not pass.
#
# Submit:
#   sbatch slurm/aires_res.slurm
#   sbatch --export=ALL,AIRES_RES_TAG=persist,AIRES_BACKEND=persistence slurm/aires_res.slurm
#   sbatch --export=ALL,AIRES_N_WALKERS=16,AIRES_RES_TAG=smoke slurm/aires_res.slurm
#
# Then, on the login node in moe:
#   PYTHONPATH=. python -m aires.run_aires --stage ds
#   PYTHONPATH=. python -m aires.run_aires --stage compare
set -eo pipefail

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
source "$REPO/slurm/aires_env.sh" moe

EVENT=${AIRES_EVENT:-PNW_HeatDome_2021}
TAG=${AIRES_RES_TAG:-pilot}
export AIRES_NSHARDS=${AIRES_NSHARDS:-8}

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
echo "AI+RES: event=$EVENT tag=$TAG backend=${AIRES_BACKEND:-fcn3} shards=$AIRES_NSHARDS"

# Refuse to burn an allocation on a cache that does not match the frozen configuration.
# --no-seed because seeding is prep's job on the login node and must not be a surprise
# side effect of a GPU job.
python -m aires.run_aires --stage prep --event "$EVENT" --tag "$TAG" --no-seed

echo "--- res: $(date) ---"
python -m aires.run_aires --stage res --event "$EVENT" --tag "$TAG"

echo "End: $(date)"
echo "Now, on the login node in moe:"
echo "  PYTHONPATH=. python -m aires.run_aires --stage ds --event $EVENT"
echo "  PYTHONPATH=. python -m aires.run_aires --stage compare --event $EVENT --tag $TAG"
