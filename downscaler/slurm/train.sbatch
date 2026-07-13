#!/usr/bin/env bash
# =============================================================================
# ERA5 -> HRRR downscaler: interruptible DDP training on the a3mega cluster.
#
# This job is designed to be started and stopped BY HAND, in the gaps between the
# serve-* inference jobs that normally hold the nodes. Stopping it never loses more
# than `training.save_every_n_steps` optimizer steps, and restarting it is just
# resubmitting: the trainer auto-resumes from <ckpt_dir>/latest.pt.
#
#   START    sbatch slurm/train.sbatch                       # queues; runs when a node frees
#   WATCH    tail -f logs/slurm-downscaler-<jobid>.out
#   STOP     bash slurm/stop.sh                              # graceful, no deadline
#   RESUME   sbatch slurm/train.sbatch                       # picks up where it left off
#
# Hydra overrides pass straight through:
#   sbatch slurm/train.sbatch training.per_gpu_batch=2 model.C_base=128
#
# The a3mega nodes are whole-node (OverSubscribe=NO), so this cannot share a node with a
# serve-* job — it will sit in PENDING until one exits. That is expected; submit anyway.
# =============================================================================
#SBATCH --job-name=downscaler
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/downscaler/logs/slurm-%x-%j.out
set -uo pipefail

PROJ=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/downscaler

export HOME=/home/ubuntu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Rendezvous for DDP: first node of the allocation is the master. The port is derived from
# the job id so two concurrent runs cannot collide on it.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

echo "================ downscaler train.sbatch ================"
echo "job=$SLURM_JOB_ID  nodes=$SLURM_NNODES  ntasks=$SLURM_NTASKS"
echo "nodelist=$SLURM_JOB_NODELIST"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "overrides: $*"
echo "stop with: bash $PROJ/slurm/stop.sh   (or: scancel -s USR1 $SLURM_JOB_ID)"
echo "========================================================"

# --signal=USR1@300 above delivers SIGUSR1 to these tasks 300 s before the time limit;
# the trainer catches it, checkpoints, and exits 0. Do NOT use --kill-on-bad-exit here:
# a rank exiting cleanly on a stop request must not be treated as a failure.
srun "$PROJ/slurm/worker.sh" "$@"
