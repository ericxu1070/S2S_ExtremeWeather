#!/bin/bash
#SBATCH --job-name=gencast_smoke
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# Quick end-to-end validation: ONE event, 8 members (a single 8-GPU pmap wave), full
# week-2 horizon, then PDF + cartopy map + scoring. Exercises exactly the same code path
# as the real runs. Run this first to confirm the environment before the full sweep.
set -eo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

# Subset to one event and one 8-member pmap wave for speed.
export GENCAST_MAX_EVENTS=1
export GENCAST_N_MEMBERS=8

cd /home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
python -c "import jax; print('JAX devices:', jax.devices())"

python run_experiment.py --weeks 2 --stage all

echo "End: $(date)"
