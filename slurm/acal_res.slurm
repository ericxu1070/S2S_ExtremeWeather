#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --array=0-11%2
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err
#
# AI+RES calibration campaign: one array task per calibration case.
#
# Array task k runs case k of the slate in `runs/acal/catalog/
# conus_episodes_21d_2021_2025.csv`, filtered to |A_L| >= $ACAL_RUNG and ordered by peak.
# Each task is one whole node and runs the same coordinator as slurm/aires_res.slurm --
# only the event name and the walker count differ.
#
# THE ARRAY WIDTH AND THE THROTTLE ARE BOTH LOAD-BEARING.
#
#   --array=0-11%2   = the 12-case (>=3 K) slate, at most 2 running at once.
#
# The range must equal the slate size, and the %N throttle is bounded by TWO things:
#
#   nodes  Only 3 of the 8 a3mega nodes were actually free at last survey (2026-09-14):
#          0,1,2,6,7 carried 63-81 GB/card of non-Slurm work while sinfo showed "idle".
#          The 0.25 deg checkpoint needs ~64 GB of an 80 GB H100, so a card at 63 GB has
#          no room. Re-survey before submitting and update --exclude:
#            for n in 0 1 2 3 4 5 6 7; do echo -n "node$n: "; \
#              ssh nucla3m-a3meganodeset-$n nvidia-smi --query-gpu=memory.used \
#              --format=csv,noheader,nounits | paste -sd, -; done
#
#   disk   Each in-flight case holds ~46 GB of live walker states (3 resident segments x
#          32 walkers x 477 MB) and leaves ~9.7 GB of residue behind. With 550 GB free the
#          12-case slate tolerates up to 9 concurrent, but the full 42-case slate tolerates
#          only 2 once residue accumulates -- run_aires refuses to start a case when its
#          ~56 GB need exceeds 80% of free, so an over-wide array WEDGES partway through
#          rather than failing fast. For the 42-case slate use --array=0-41%2.
#
# Submit:
#   sbatch --exclude=nucla3m-a3meganodeset-[0,1,2,3,6,7] slurm/acal_res.slurm
#   sbatch --array=0-41%2 --export=ALL,ACAL_RUNG=2 ... slurm/acal_res.slurm   # all 42
#
# Prereq, on the LOGIN node in moe (GPU jobs must be pure cache-hit-then-compute):
#   python -m acal.aprep --rung 3        # builds the init frames; ~3 min each
# Each task re-checks its own init and refuses to start if it is missing.
set -eo pipefail

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
source "$REPO/slurm/aires_env.sh" moe
cd "$REPO"

RUNG=${ACAL_RUNG:-3}
TAG=${ACAL_RES_TAG:-acal}
K=${SLURM_ARRAY_TASK_ID:-0}

# The calibration config. ccfg.RES_N_WALKERS is acal's INTENT (32); the value the run
# actually reads is aconfig.RES_N_WALKERS, which defaults to the pilot's 64. They are
# different environment variables, so this export is what makes the campaign N=32 --
# without it every case silently runs at double the walkers, double the GPU-h and
# ~92 GB of live states instead of 46.
export AIRES_N_WALKERS=${AIRES_N_WALKERS:-32}
export AIRES_M_MEMBERS=${AIRES_M_MEMBERS:-6}
export AIRES_NSHARDS=${AIRES_NSHARDS:-8}

EVENT=$(python - "$RUNG" "$K" <<'PY'
import sys, pandas as pd
rung, k = float(sys.argv[1]), int(sys.argv[2])
df = pd.read_csv("runs/acal/catalog/conus_episodes_21d_2021_2025.csv")
df = df[df.a_l_conus.abs() >= rung].sort_values("peak").reset_index(drop=True)
if k >= len(df):
    sys.exit(f"array index {k} past the end of the {len(df)}-case slate at rung {rung}")
print(df.iloc[k].episode_id)
PY
)

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
echo "acal case $K (rung>=$RUNG): event=$EVENT tag=$TAG N=$AIRES_N_WALKERS M=$AIRES_M_MEMBERS"

# Refuse to burn an allocation on a missing init or a stale cache. --no-seed: seeding is
# prep's job on the login node, never a side effect of a GPU job.
python -m aires.run_aires --stage prep --event "$EVENT" --tag "$TAG" --no-seed

echo "--- res: $(date) ---"
python -m aires.run_aires --stage res --event "$EVENT" --tag "$TAG"

echo "End: $(date)"
