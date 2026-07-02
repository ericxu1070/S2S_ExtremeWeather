#!/bin/bash
# One-time setup for the GenCast S2S pipeline, into the existing `moe` conda env.
# Run on the login node (it has internet). Everything lands on NFS, so the a3mega
# compute nodes see it too.
#
#   bash setup_env.sh
#
set -eo pipefail

PROJ="/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather"
cd "$PROJ"

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe
echo "python: $(which python)"

# --- graphcast (provides gencast/rollout/normalization/...) -------------------
# moe already has all real runtime deps (jax[cuda], haiku, jraph, dm-tree, trimesh,
# rtree, chex, xarray, dask, ...). Install with --no-deps so pip does NOT try to pull
# graphcast's colab-only / unused extras (colabtools, dinosaur-dycore, xarray_tensorstore)
# or downgrade moe's jax 0.10.2.
if [ ! -d third_party/graphcast/graphcast ]; then
  echo "cloning graphcast ..."
  git clone --depth 1 https://github.com/google-deepmind/graphcast.git third_party/graphcast
fi

# Patch rollout.py for modern xarray: Dataset(Dataset) copy-construct was removed.
# Idempotent: after the first rewrite the original pattern is gone.
ROLL="third_party/graphcast/graphcast/rollout.py"
python - "$ROLL" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
for v in ("inputs", "targets_template", "forcings"):
    s = s.replace(f"{v} = xarray.Dataset({v})", f"{v} = {v}.copy()")
open(p, "w").write(s)
print("patched", p)
PY

pip install --no-deps -e third_party/graphcast
pip install --no-deps properscoring

# --- cartopy Natural Earth features (so the CONUS maps render offline on a3mega) ---
# Cached under ~/.local/share/cartopy on NFS -> visible to compute nodes.
echo "pre-fetching cartopy Natural Earth features ..."
python - <<'PY'
import cartopy.io.shapereader as shp
sets = [
    ("110m", "physical", "coastline"),
    ("110m", "cultural", "admin_0_boundary_lines_land"),
    ("110m", "cultural", "admin_1_states_provinces_lakes"),
    ("50m",  "cultural", "admin_1_states_provinces_lakes"),
]
for res, cat, name in sets:
    try:
        shp.natural_earth(resolution=res, category=cat, name=name)
        print("  ok", res, cat, name)
    except Exception as e:
        print("  WARN could not fetch", res, cat, name, "->", e)
PY

# --- sanity: imports + GenCast submodule resolve --------------------------------
python - <<'PY'
from graphcast import gencast, rollout, normalization, nan_cleaning, data_utils, xarray_jax
import properscoring, cartopy
print("imports OK: gencast/rollout/properscoring/cartopy all import")
print("multiple_runs:", hasattr(rollout, "chunked_prediction_generator_multiple_runs"))
PY

echo "setup_env.sh: done."
