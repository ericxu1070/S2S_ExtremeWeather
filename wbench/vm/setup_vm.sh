#!/bin/bash
# One-time setup of a fresh 8xH100 VM for the walker benchmark. ~45 min, most of it
# downloads (two envs, a CUDA 12.8 toolkit, the 2.85 GB FCN3 checkpoint) and the ~10 min
# torch-harmonics kernel compile. Every step leaves a marker and is skipped on a re-run,
# so a failure is fixed and the script simply run again.
#
#   bash wbench/vm/setup_vm.sh              # all steps
#   bash wbench/vm/setup_vm.sh fcn3 th      # just these steps (markers ignored)
#
# Steps, in order:
#   sys        driver, compilers, git/curl, disk; apt-get installs what is missing (sudo)
#   verify     the bundle's data files against MANIFEST.sha256
#   mamba      micromamba binary -> $WB_HOME/bin
#   gencast    env wb-gencast: python 3.12 + requirements-gencast.txt + bundled graphcast
#   fcn3       env wb-fcn3:   python 3.11 + CUDA 12.8 toolkit + torch cu128 + makani +
#              earth2studio[fcn3], all at the commits the a3mega/Derecho env runs
#   th         torch-harmonics v0.8.1 built from source WITH its DISCO CUDA kernel.
#              Without the kernel earth2studio silently runs FCN3 in fp32 on a slow
#              path (HANDOFF.md "The DISCO CUDA kernel"), so every FCN3 number would be
#              wrong. The step fails unless the kernel is importable afterwards.
#   weights    FCN3 checkpoint from HuggingFace into runs/fcn3/.cache/e2s
#   pickle     build the FCN3 module once and pickle it (fcn3/model_cache.py), so the
#              benchmark times a load, as production does, not a ~2 min CPU build
#   check      tests in both envs, GenCast batch checks, GPU visible from both envs
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
WB_HOME=${WB_HOME:-$(dirname "$REPO")}
WB_ENVS=${WB_ENVS:-$WB_HOME/envs}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$WB_HOME/micromamba}
# env.sh runs with this set; without it pip counts ~/.local packages as installed and skips them
export PYTHONNOUSERSITE=1
MM=$WB_HOME/bin/micromamba
MARK=$WB_HOME/.setup
mkdir -p "$MARK" "$WB_ENVS"
LOG=$WB_HOME/setup.log
exec > >(tee -a "$LOG") 2>&1

# Pins (see constraints-fcn3.txt and HANDOFF.md "FCN3 environment").
TORCH_INDEX=https://download.pytorch.org/whl/cu128
MAKANI_REF=b38fcb2799d7dbc146fa60459f3f9823394a8bf1
E2S_REF=ccc31ddf77eb84f47d9032e5d4749a5ce2df2ad4
TH_REF=a632ca748a12bd9f74dbc1e00653317810991f74          # torch-harmonics v0.8.1
CUDA_LABEL=${WB_CUDA_LABEL:-nvidia/label/cuda-12.8.1}

log() { echo "[setup $(date -u +%H:%M:%S)] $*"; }
die() { echo "[setup] FAILED: $*" >&2; exit 1; }
py_gc() { "$WB_ENVS/wb-gencast/bin/python" "$@"; }
py_f3() { "$WB_ENVS/wb-fcn3/bin/python" "$@"; }

step_sys() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found - this VM has no NVIDIA driver"
  local n drv
  n=$(nvidia-smi -L | wc -l)
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  log "GPUs: $n x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), driver $drv"
  [ "$n" -ge 1 ] || die "no GPUs visible"
  [ "${drv%%.*}" -ge 535 ] || log "WARNING: driver $drv < 535; CUDA 12.8 wheels may not load"
  local need=()
  for t in gcc g++ git curl bzip2 tar sha256sum taskset; do
    command -v "$t" >/dev/null || need+=("$t")
  done
  if [ "${#need[@]}" -gt 0 ]; then
    log "missing: ${need[*]} -> apt-get install (needs sudo)"
    sudo -n true 2>/dev/null || die "need ${need[*]} and passwordless sudo is unavailable"
    sudo apt-get update -y && sudo apt-get install -y build-essential git curl bzip2 util-linux
  fi
  local free_gb
  free_gb=$(df -BG --output=avail "$WB_HOME" | tail -1 | tr -dc 0-9)
  log "free disk at $WB_HOME: ${free_gb} GB"
  [ "$free_gb" -ge 80 ] || die "need >= 80 GB free at $WB_HOME (envs ~25 GB, runs ~40 GB)"
}

step_verify() {
  [ -f "$HERE/MANIFEST.sha256" ] || die "no MANIFEST.sha256 - was this bundle built by make_bundle.sh?"
  (cd "$REPO" && sha256sum -c --quiet "$HERE/MANIFEST.sha256") || die "bundle data corrupt"
  log "bundle data verified ($(wc -l < "$HERE/MANIFEST.sha256") files)"
}

step_mamba() {
  mkdir -p "$WB_HOME/bin"
  curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/${WB_MAMBA_VERSION:-latest}" \
    | tar -xj -C "$WB_HOME" bin/micromamba
  "$MM" --version
}

step_gencast() {
  local env=$WB_ENVS/wb-gencast
  [ -d "$env" ] && rm -rf "$env"
  "$MM" create -y -p "$env" -c conda-forge python=3.12 pip
  py_gc -m pip install --no-cache-dir -r "$HERE/requirements-gencast.txt"
  [ -f "$REPO/third_party/graphcast/setup.py" ] || die "bundled third_party/graphcast missing"
  grep -q "from jax.sharding import PartitionSpec as P" \
    "$REPO/third_party/graphcast/graphcast/rollout.py" \
    || die "bundled graphcast/rollout.py is not patched (see setup_env.sh)"
  py_gc -m pip install --no-deps -e "$REPO/third_party/graphcast"
  bash "$HERE/env.sh" gencast python -c \
    "from graphcast import gencast, rollout, xarray_jax; import jax; print('jax', jax.__version__)"
}

step_fcn3() {
  local env=$WB_ENVS/wb-fcn3
  [ -d "$env" ] && rm -rf "$env"
  "$MM" create -y -p "$env" -c conda-forge python=3.11 pip ninja
  # nvcc that matches torch's CUDA. The pip nvcc wheel ships ptxas only, not nvcc, and a
  # system CUDA of another version will not build against a cu128 torch (HANDOFF trap 1).
  "$MM" install -y -p "$env" -c "$CUDA_LABEL" -c conda-forge cuda-toolkit
  "$env/bin/nvcc" --version | tail -2
  py_f3 -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX"
  local C=(-c "$HERE/constraints-fcn3.txt" --extra-index-url "$TORCH_INDEX")
  # makani first: pip cannot resolve it from earth2studio's uv sources (HANDOFF note).
  py_f3 -m pip install --no-cache-dir "${C[@]}" "makani @ git+https://github.com/NVIDIA/makani.git@$MAKANI_REF"
  py_f3 -m pip install --no-cache-dir "${C[@]}" \
    "earth2studio[fcn3] @ git+https://github.com/NVIDIA/earth2studio.git@$E2S_REF"
  py_f3 -m pip install --no-cache-dir "${C[@]}" pytest
  py_f3 -c "import torch; assert torch.__version__.startswith('2.11.0'), torch.__version__; print('torch', torch.__version__)"
}

step_th() {
  local env=$WB_ENVS/wb-fcn3 src=$WB_HOME/src/torch-harmonics
  rm -rf "$src"; mkdir -p "$(dirname "$src")"
  # The repo uses git-lfs, which a fresh VM does not have: bypass it (HANDOFF trap 4).
  local nolfs=(-c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false)
  git "${nolfs[@]}" clone https://github.com/NVIDIA/torch-harmonics.git "$src"
  git -C "$src" "${nolfs[@]}" checkout -f "$TH_REF"
  local site
  site=$(py_f3 -c "import site; print(site.getsitepackages()[0])")
  (
    export CUDA_HOME=$env PATH="$env/bin:$PATH"
    export LD_LIBRARY_PATH="$(ls -d "$site"/nvidia/*/lib | tr '\n' ':')$env/lib"
    export FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST=9.0 CC=gcc CXX=g++
    export MAX_JOBS=$(( $(nproc) / 2 ))
    # A source build, forced: `pip install torch-harmonics==...` takes a prebuilt
    # pure-python wheel and never compiles the kernel (HANDOFF traps 2-3).
    py_f3 -m pip install --no-build-isolation --no-cache-dir --force-reinstall --no-deps "$src"
  )
  bash "$HERE/env.sh" fcn3 python -c \
    "import earth2studio.models.px.fcn3 as m; assert m._cuda_extension_available, 'DISCO CUDA kernel NOT available'; print('DISCO CUDA kernel: available')"
}

step_weights() {
  bash "$HERE/env.sh" fcn3 python - <<'PY'
# The file list fcn3/run_fcn3.py::stage_prep fetches - checkpoint only, no model build.
from earth2studio.models.px import FCN3
pkg = FCN3.load_default_package()
for f in ("config.json", "metadata.json", "orography.nc", "land_mask.nc",
          "global_means.npy", "global_stds.npy", "mins.npy", "maxs.npy"):
    try:
        print("cached", f, "->", pkg.get(f))
    except Exception as e:
        print("optional file not in package:", f, type(e).__name__)
got = None
for c in ("training_checkpoints/best_ckpt_mp0.tar", "best_ckpt_mp0.tar"):
    try:
        got = pkg.get(c)
        break
    except Exception:
        continue
if got is None:
    raise SystemExit("could not fetch the FCN3 weights (best_ckpt_mp0.tar)")
import os
print("weights ->", got, f"{os.path.getsize(got) / 1e9:.2f} GB")
PY
}

step_pickle() {
  bash "$HERE/env.sh" fcn3 python -c \
    "from wbench import fcn3_walker as FW; px = FW.load_px('cpu'); print('FCN3 ready:', len(px.variables), 'variables')"
  ls -la "$REPO/runs/fcn3/.cache/model/"
}

step_check() {
  bash "$HERE/env.sh" gencast python -c \
    "import jax; d = jax.devices(); assert d[0].platform == 'gpu', d; print('jax sees', len(d), d[0].device_kind)"
  bash "$HERE/env.sh" fcn3 python -c \
    "import torch; assert torch.cuda.is_available(); print('torch sees', torch.cuda.device_count(), torch.cuda.get_device_name(0))"
  bash "$HERE/env.sh" gencast python -m pytest -q -p no:cacheprovider wbench/tests
  bash "$HERE/env.sh" fcn3 python -m pytest -q -p no:cacheprovider wbench/tests
  for c in gencast_0p25 gencast_1p0; do
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu bash "$HERE/env.sh" gencast \
      python -m wbench.gencast_bench --config "$c" --check
  done
  log "setup complete - next: bash wbench/vm/run_all.sh"
}

ALL=(sys verify mamba gencast fcn3 th weights pickle check)
STEPS=("$@")
FORCE=1
if [ "${#STEPS[@]}" -eq 0 ]; then STEPS=("${ALL[@]}"); FORCE=0; fi
for s in "${STEPS[@]}"; do
  declare -F "step_$s" >/dev/null || die "unknown step '$s' (have: ${ALL[*]})"
  if [ "$FORCE" -eq 0 ] && [ -f "$MARK/$s.done" ]; then
    log "step $s: done earlier, skip"
    continue
  fi
  log "step $s: start"
  t0=$SECONDS
  "step_$s"
  touch "$MARK/$s.done"
  log "step $s: ok ($(( SECONDS - t0 )) s)"
done
