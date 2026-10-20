#!/bin/bash
# Put one of the two benchmark envs on PATH, from scratch - the VM's aires_env.sh.
#
#   source wbench/vm/env.sh gencast                     # this shell, GenCast/JAX
#   bash   wbench/vm/env.sh fcn3 python -m wbench.checks --run ...   # set up, then exec
#
# The envs are micromamba prefixes, used by PATH rather than `activate`, so nothing from
# the caller's environment stacks underneath: PATH, PYTHONPATH and LD_LIBRARY_PATH are
# rebuilt from the env that ends up active (the moe-under-fcn3 stacking aires_env.sh
# guards against on a3mega cannot happen here, but the rebuild costs nothing).
_wb_want=${1:?usage: env.sh gencast|fcn3 [cmd ...]}
shift

WB_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WB_HOME=${WB_HOME:-$(dirname "$WB_REPO")}
WB_ENVS=${WB_ENVS:-$WB_HOME/envs}
cd "$WB_REPO" || { return 1 2>/dev/null || exit 1; }
export PYTHONPATH="$WB_REPO" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1

case "$_wb_want" in
  gencast)
    _wb_env=$WB_ENVS/wb-gencast
    # aires_env.sh's walker settings, for the same reasons: preallocate once at startup
    # (acal array 1209), and no persistent compile cache - the 0.25 deg executable is
    # over its 2 GiB limit anyway, and a cache hit on 1.0 deg would make a cold-start
    # number depend on which worker happened to compile first.
    export XLA_PYTHON_CLIENT_PREALLOCATE=${WB_XLA_PREALLOCATE:-true}
    export XLA_PYTHON_CLIENT_MEM_FRACTION=${WB_XLA_MEM_FRACTION:-.90}
    export JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1
    unset JAX_COMPILATION_CACHE_DIR
    ;;
  fcn3)
    _wb_env=$WB_ENVS/wb-fcn3
    export EARTH2STUDIO_CACHE="$WB_REPO/runs/fcn3/.cache/e2s"
    export HF_HOME="$WB_REPO/runs/fcn3/.cache/hf"
    export FCN3_WEEKS=3
    ;;
  *)
    echo "env.sh: unknown env '$_wb_want' (want gencast|fcn3)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

if [ ! -x "$_wb_env/bin/python" ]; then
  echo "env.sh: $_wb_env is not built - run wbench/vm/setup_vm.sh first" >&2
  return 1 2>/dev/null || exit 1
fi

# One worker per GPU, each on its own core slice (run_all.sh sets WB_THREADS).
_wb_threads=${WB_THREADS:-$(( $(nproc) / 8 ))}
[ "${_wb_threads:-0}" -lt 1 ] && _wb_threads=1
export OMP_NUM_THREADS=$_wb_threads MKL_NUM_THREADS=$_wb_threads
export OPENBLAS_NUM_THREADS=$_wb_threads NUMEXPR_NUM_THREADS=$_wb_threads

export CONDA_PREFIX=$_wb_env
export PATH="$_wb_env/bin:/usr/local/bin:/usr/bin:/bin"
unset LD_LIBRARY_PATH
_wb_site=$("$_wb_env/bin/python" -c 'import site; print(site.getsitepackages()[0])')
_wb_ld=$(ls -d "$_wb_site"/nvidia/*/lib 2>/dev/null | tr '\n' ':')
[ -n "$_wb_ld" ] && export LD_LIBRARY_PATH="${_wb_ld%:}"
unset _wb_site _wb_ld _wb_env _wb_want _wb_threads

if [ "$#" -gt 0 ]; then
  exec "$@"
fi
