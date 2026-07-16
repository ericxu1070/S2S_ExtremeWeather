#!/usr/bin/env bash
# =============================================================================
# GenCast(xres 1deg) -> downscaler coupling: GPU inference for the demo figures.
#
# Pure GPU compute. The conditioning + truth caches are built FIRST on the login node
# (python scripts/xres_downscale_demo.py --stage prep), so this job only loads the trained
# EDM downscaler and draws one conditional HRRR sample per case -- no data building on the GPU
# (the STRICT rule in CLAUDE.md). Uses a single H100; the a3mega nodes are whole-node, so it
# still reserves a node, but the work is a handful of sampler runs (~minutes).
#
#   PREP    python scripts/xres_downscale_demo.py --stage prep     # login node, CPU
#   RUN     sbatch slurm/xres_downscale_infer.sbatch
#   WATCH   tail -f logs/slurm-xres-downscale-<jobid>.out
#   PLOT    python scripts/xres_downscale_demo.py --stage plot     # login node, CPU
# =============================================================================
#SBATCH --job-name=xres-downscale
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:40:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/downscaler/logs/slurm-%x-%j.out

# NOT -u: conda's activation scripts reference unset variables.
set -o pipefail

PROJ=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/downscaler
export HOME=/home/ubuntu
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# HDF5 file locking over NFS deadlocks concurrent netCDF opens; the infer stage opens only a
# couple of files, but keep it off to match the training path.
export HDF5_USE_FILE_LOCKING=FALSE

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
cd "$PROJ"

echo "================ xres_downscale_infer ================"
echo "job=$SLURM_JOB_ID host=$(hostname -s) nodelist=$SLURM_JOB_NODELIST"
echo "gpus visible: ${CUDA_VISIBLE_DEVICES:-<all>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
echo "======================================================"

# Fail loudly if prep was skipped -- this job must NEVER build data.
for f in outputs/xres_demo_cache/testset_prep.npz outputs/xres_demo_cache/idalia_prep.npz; do
  if [ ! -f "$f" ]; then
    echo "MISSING $f -- run 'python scripts/xres_downscale_demo.py --stage prep' on the login node first" >&2
    exit 2
  fi
done

python scripts/xres_downscale_demo.py --stage infer \
  --ckpt "$PROJ/checkpoints/latest.pt" --n-steps 18 --seed 0

echo "infer exit=$?  ->  now run: python scripts/xres_downscale_demo.py --stage plot"
