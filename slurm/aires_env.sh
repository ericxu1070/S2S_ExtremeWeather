#!/bin/bash
# Activate one of the two AI+RES conda environments, from scratch.
#
#   source slurm/aires_env.sh moe            # set this shell up for GenCast/JAX
#   bash   slurm/aires_env.sh fcn3 python -m aires.score ...   # set up, then exec
#
# Both forms exist because both callers exist: `slurm/aires_res.slurm` sources it to set
# a phase up, and `aires/run_aires.py` (the DMC coordinator, which itself runs in `moe`)
# spawns the FCN3 pool through it. That coordinator is the reason the activation starts
# by DEACTIVATING: a child launched from an active `moe` inherits CONDA_PREFIX, PATH and
# LD_LIBRARY_PATH, and `conda activate fcn3` on top of that STACKS rather than replaces,
# leaving moe's site-packages and moe's bundled NVIDIA libraries ahead of fcn3's on the
# search paths. Unwinding to shlvl 0 first, and rebuilding LD_LIBRARY_PATH from the
# environment that actually ends up active, is what keeps the two stacks apart.
_aires_want=${1:?usage: aires_env.sh moe-or-fcn3 [cmd ...] }
shift

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO" || return 1 2>/dev/null || exit 1
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate || break; done

case "$_aires_want" in
  moe|gencast|walk)
    conda activate "${GENCAST_CONDA_ENV:-moe}"
    # The 0.25 deg checkpoint occupies ~64 GB of an 80 GB H100, so the pmap path does not
    # apply: one serial member per GPU in bf16, exactly as the xres 0.25 deg pool runs.
    export XRES_SERIAL_INFER=1
    export XRES_BF16=1
    export XRES_INFER_GPUS=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
    export JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1
    export JAX_COMPILATION_CACHE_DIR="$REPO/runs/models/jax_cache_0p25"
    mkdir -p "$JAX_COMPILATION_CACHE_DIR"
    ;;
  fcn3|score)
    conda activate "${FCN3_CONDA_ENV:-fcn3}"
    export EARTH2STUDIO_CACHE="$REPO/runs/fcn3/.cache/e2s"
    export HF_HOME="$REPO/runs/fcn3/.cache/hf"
    export FCN3_WEEKS=${FCN3_WEEKS:-3}
    # Pin each shard to a slice of the cores: 8 unpinned processes each spawn one OpenMP
    # thread per NODE core and thrash through FCN3's model construction (job 700).
    _cores=$(nproc)
    _threads=${FCN3_THREADS:-$(( _cores / ${AIRES_NSHARDS:-8} ))}
    [ "${_threads:-0}" -lt 1 ] && _threads=1
    export OMP_NUM_THREADS=$_threads MKL_NUM_THREADS=$_threads
    export OPENBLAS_NUM_THREADS=$_threads NUMEXPR_NUM_THREADS=$_threads
    export OMP_PROC_BIND=close OMP_PLACES=cores
    ;;
  *)
    echo "aires_env.sh: unknown environment '$_aires_want' (want moe|fcn3)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

# Rebuilt from the environment that is NOW active, never inherited.
unset LD_LIBRARY_PATH
_aires_site=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$(echo "$_aires_site"/nvidia/*/lib | tr ' ' ':')"
unset _aires_site _aires_want _cores _threads

if [ "$#" -gt 0 ]; then
  exec "$@"
fi
