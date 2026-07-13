#!/usr/bin/env bash
# =============================================================================
# 00_preflight.sh — read-only environment & data checks. Run this first.
#
# Verifies everything training needs WITHOUT changing any state: the conda env,
# the code, the two datasets, the grid reference, disk space, and (informational)
# GPU / Slurm availability on this host. Safe on a login node.
#
#   bash bash/00_preflight.sh
#
# Exit 0 = ready (possibly with warnings). Exit 1 = something training needs is
# missing; the FAIL lines say what.
# =============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FAILS=0; WARNS=0
ok()   { echo "  PASS  $1"; }
warn() { echo "  warn  $1"; WARNS=$((WARNS + 1)); }
fail() { echo "  FAIL  $1"; FAILS=$((FAILS + 1)); }
hdr()  { echo; echo "=============== $1 ==============="; }

cd "$PROJ" || { echo "cannot cd to $PROJ"; exit 1; }

# ---------------------------------------------------------------- environment
hdr "environment"
echo "  host=$(hostname)  cpus=$(nproc)  mem=$(free -g | awk '/^Mem:/{print $2}')G"
if source /home/ubuntu/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate moe 2>/dev/null; then
    ok "conda env 'moe' activates"
else
    fail "conda env 'moe' not found (source project env; NOT the GenCast 'my-env')"
fi
if python - <<'PY' 2>/dev/null
import torch, sys
print(f"  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}  python={sys.version.split()[0]}")
PY
then ok "torch imports"; else fail "torch import failed in the moe env"; fi

if PYTHONPATH="$PROJ" python -c "from model.unet import HRRRUNet; from model.attention import TransformerBlock; from training.trainer import Trainer" 2>/dev/null; then
    ok "project code imports (model + trainer)"
else
    fail "project code does not import — wrong env or broken checkout"
fi

# ---------------------------------------------------------------- config + data
hdr "data (paths come from configs/data/era5_hrrr.yaml)"
readarray -t CFG < <(python - <<'PY'
from omegaconf import OmegaConf
c = OmegaConf.load("configs/data/era5_hrrr.yaml")
print(c.era5_dir); print(c.hrrr_dir); print(c.grid_meta_path)
print(c.index_path); print(c.stats_path)
PY
)
ERA5_DIR="${CFG[0]:-}"; HRRR_DIR="${CFG[1]:-}"; GRID_META="${CFG[2]:-}"
INDEX="${CFG[3]:-}"; STATS="${CFG[4]:-}"

if [ -d "$ERA5_DIR" ]; then
    N=$(find "$ERA5_DIR" -name '*.nc' 2>/dev/null | wc -l)
    if [ "$N" -gt 10000 ]; then ok "ERA5 dir: $N files ($ERA5_DIR)"; else fail "ERA5 dir has only $N .nc files ($ERA5_DIR)"; fi
else fail "ERA5 dir missing: $ERA5_DIR"; fi

if [ -d "$HRRR_DIR" ]; then
    N=$(find "$HRRR_DIR" -name '*.nc' 2>/dev/null | wc -l)
    if [ "$N" -gt 10000 ]; then ok "HRRR dir: $N files ($HRRR_DIR)"; else fail "HRRR dir has only $N .nc files ($HRRR_DIR)"; fi
else fail "HRRR dir missing: $HRRR_DIR"; fi

[ -f "$GRID_META" ] && ok "HRRR grid reference: $GRID_META" || fail "grid reference missing: $GRID_META"

# ------------------------------------------------- derived prerequisites (02)
hdr "derived prerequisites (bash/02_build_index_and_stats.sh creates these)"
if [ -s "$INDEX" ]; then
    ok "timestamp index: $(wc -l < "$INDEX") timestamps ($INDEX)"
else
    warn "timestamp index not built yet — run bash/02_build_index_and_stats.sh"
fi
if [ -f "$STATS" ]; then
    ok "normalization stats: $STATS"
else
    warn "norm stats not built yet — run bash/02_build_index_and_stats.sh"
fi

# ---------------------------------------------------------------- compute
hdr "compute on this host (informational)"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi -L | sed 's/^/  /'
    ok "GPUs visible — bash/03_train.sh can run locally here (LOCAL=1)"
else
    warn "no usable GPU on this host — training must go through Slurm"
fi
if command -v sbatch >/dev/null; then
    ok "Slurm available — bash/03_train.sh will submit slurm/train.sbatch"
else
    warn "no sbatch on this host — bash/03_train.sh needs local GPUs"
fi

DISK=$(df -BG --output=avail "$PROJ" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$DISK" ] && [ "$DISK" -ge 20 ]; then
    ok "disk: ${DISK}G free at $PROJ (checkpoints need ~5G)"
else
    warn "disk: only ${DISK:-?}G free at $PROJ"
fi

# ---------------------------------------------------------------- summary
hdr "summary"
if [ "$FAILS" = 0 ]; then
    echo "  READY ($WARNS warning(s))."
    echo "  Next: bash bash/01_smoke.sh"
    exit 0
fi
echo "  $FAILS FAIL(S), $WARNS warning(s) — fix the FAIL lines before training."
exit 1
