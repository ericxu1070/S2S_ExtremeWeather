#!/bin/bash
#SBATCH --job-name=xres_1p0_wk4
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=18:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# WEEK-4 (28-day lead) arm of the cross-resolution experiment, 1.0deg (full <2019
# checkpoint, NOT the Mini). init = peak - 28d, same verification week, 56 rollout steps
# x 24 members on 8 H100s (pmap). ~2x the week-2 wall time; still much cheaper than 0.25deg.
set -eo pipefail

WEEKS=4

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

cd /home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather

echo "Host: $(hostname)"; echo "Start: $(date)"; echo "WEEKS=$WEEKS"
nvidia-smi -L || true
python -c "import jax; print('JAX devices:', jax.devices())"

python run_xres.py --stage prep  --res 1p0 --weeks "$WEEKS"
python run_xres.py --stage infer --res 1p0 --weeks "$WEEKS"

echo "End: $(date)"
