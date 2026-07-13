#!/usr/bin/env bash
# Per-rank wrapper: one instance per GPU, launched by srun from train.sbatch.
# Sets up the `moe` conda env + DDP rank mapping, then execs train.py.
# All arguments are forwarded to train.py as Hydra overrides.
#
# NOT -u: conda's activation scripts reference unset variables.
set -o pipefail

PROJ=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/downscaler

# --- DDP rank mapping straight from Slurm (one task per GPU, so no torchrun needed).
# train.py's setup_ddp() keys off RANK being present in the environment.
export RANK=${SLURM_PROCID:-0}
export LOCAL_RANK=${SLURM_LOCALID:-0}
export WORLD_SIZE=${SLURM_NTASKS:-1}
export HOME=/home/ubuntu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# All 8 GPUs are visible to every task; train.py selects its own via LOCAL_RANK.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- NCCL.
# Single node: NCCL uses NVLink and needs nothing here. Loading the GPUDirect-TCPXO
# (FasTrak) plugin on one node can abort the rank, so it is deliberately not sourced.
# Multi-node: plain NCCL over TCP on the primary NIC — slower all-reduce than FasTrak but
# stable (FasTrak intermittently hangs sustained DDP all-reduce at high rank counts; see
# moein/regional/hrrr_edm/scripts/a3mega_env.sh).
NNODES="${SLURM_JOB_NUM_NODES:-1}"
if [ "${NNODES}" -gt 1 ]; then
  export NCCL_SOCKET_IFNAME=enp0s12
  export NCCL_IB_DISABLE=1
  export NCCL_CROSS_NIC=0
fi
# Abort on a collective timeout rather than hanging for days — a dead job can be
# resubmitted and will resume from latest.pt; a hung one silently burns the allocation.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# --- CRITICAL for the real-data path: HDF5 file locking over NFS deadlocks when many
# DataLoader workers open netCDF files concurrently, which drops a rank and hangs the job.
export HDF5_USE_FILE_LOCKING=FALSE

# --- Ephemeral scratch must be node-local, never NFS: Python's multiprocessing rmtree's
# its temp dir at DataLoader-worker teardown, and on NFS the silly-rename (.nfsXXXX) makes
# that unlink fail with EBUSY and core-dumps the rank.
export TMPDIR=/tmp

# --- Env: the downscaler runs in the `moe` conda env (NOT the GenCast `my-env`).
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
cd "$PROJ"

echo "[worker] host=$(hostname -s) RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE \
MASTER=$MASTER_ADDR:$MASTER_PORT overrides=$*"

exec python train.py "$@"
