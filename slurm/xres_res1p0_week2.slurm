#!/bin/bash
#SBATCH --job-name=xres_1p0_wk2
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# 1.0deg arm of the cross-resolution experiment: full GenCast 1p0deg <2019 checkpoint
# (mesh 5, 181x360) -- the matched-resolution counterpart to the 0.25deg model (NOT the
# lightweight Mini used by the original week2/3/4 runs). 24-member ensemble, 14-day rollout,
# same full-output caching + per-member metrics as the 0.25deg arm. Much cheaper than 0.25deg.
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

python run_xres.py --stage prep  --res 1p0
python run_xres.py --stage infer --res 1p0

echo "End: $(date)"
