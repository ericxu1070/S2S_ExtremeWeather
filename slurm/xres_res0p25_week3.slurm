#!/bin/bash
#SBATCH --job-name=xres_0p25_wk3
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=36:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# WEEK-3 (21-day lead) arm of the cross-resolution experiment, 0.25deg.
# Same as slurm/xres_res0p25_week2.slurm but a 3-week horizon: init = peak - 21d, the
# SAME verification week (7 days ending on the peak), reached at a longer lead. That is
# 42 rollout steps (vs 28 at week 2) x 24 members, so ~1.5x the week-2 wall time.
#
# 0.25deg (~50 GiB VRAM/member) fits the pmap path on H100-80GB (one member per GPU,
# 24 members = 3 waves of 8). Events run sequentially with per-event cube caching, so a
# walltime kill preserves completed events and a re-submit resumes from the next event.
# a3mega has no walltime cap; 36 h is a generous budget with resume as the safety net.
set -eo pipefail

WEEKS=3

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

# prep is cache-aware: instant if the login-node prep already built the week-3 init frames
# (which it must, if the compute node has no internet). infer caches the cubes.
python run_xres.py --stage prep  --res 0p25 --weeks "$WEEKS"
python run_xres.py --stage infer --res 0p25 --weeks "$WEEKS"

echo "End: $(date)"
