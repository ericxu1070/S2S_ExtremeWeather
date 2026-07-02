#!/bin/bash
#SBATCH --job-name=xres_0p25_wk2
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# 0.25deg arm of the cross-resolution experiment: full GenCast 0p25deg <2019 checkpoint
# (mesh 6, native 721x1440), 24-member ensemble, 14-day rollout. Caches the ENTIRE output
# (all vars / steps / members, CONUS box) per event, then derives the per-member metric.
#
# NOTE: 0.25deg is ~16x the grid + mesh of the 1.0deg Mini and the GPU dense-attention swap
# in model.py was validated only at 1.0deg. Events run sequentially with per-event caching,
# so the first event's success/timing is visible within ~30-60 min and any failure preserves
# all completed events. 24h wall budget; reduce XRES_N_MEMBERS_0P25 if time is tight.
set -eo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

cd /home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
python -c "import jax; print('JAX devices:', jax.devices())"

# prep is cache-aware: instant if already run on the login node. infer caches the cubes.
python run_xres.py --stage prep  --res 0p25
python run_xres.py --stage infer --res 0p25

echo "End: $(date)"
