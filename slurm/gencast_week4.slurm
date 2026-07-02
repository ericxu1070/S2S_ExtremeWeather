#!/bin/bash
#SBATCH --job-name=gencast_week4
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# Week-4 experiment: 28-day forecast (init = peak - 28d). Same verification week (the
# 7 days ending on the peak), reached at a 4-week lead. 56 rollout steps x 24 members
# across all 8 H100s, then PDFs + cartopy CONUS maps (obs|mean|error) + scoring.
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

python run_experiment.py --weeks 4 --stage all

echo "End: $(date)"
