#!/bin/bash
#SBATCH --job-name=gencast_week2
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# Week-2 experiment: 14-day forecast (init = peak - 14d), the original notebook setup.
# 24-member GenCast ensemble across all 8 H100s (3 pmap waves of 8), then PDFs +
# cartopy CONUS maps (obs|mean|error) + CRPS / rank histograms.
set -eo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

# Make pip-installed CUDA libs discoverable for the jax CUDA plugin (same trick fuxi
# uses for onnxruntime). Harmless if jax already finds them.
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Grow GPU memory on demand (avoid the 75% pre-grab OOM-ing the diffusion rollout).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

cd /home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi -L || true
python -c "import jax; print('JAX devices:', jax.devices())"

# prep is cache-aware: instant if you already ran it on the login node (recommended),
# otherwise it downloads here (needs internet on the compute node).
python run_experiment.py --weeks 2 --stage all

echo "End: $(date)"
