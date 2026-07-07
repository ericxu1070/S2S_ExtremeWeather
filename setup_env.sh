#!/bin/bash
# One-time setup for the GenCast S2S pipeline, into the existing `my-env` conda env.
# Run on the login node (it has internet). Outputs land on GLADE scratch.
#
#   bash setup_env.sh
#
set -eo pipefail

PROJ="/glade/derecho/scratch/exu/S2S_ExtremeWeather"
cd "$PROJ"

module load conda
conda activate my-env
echo "python: $(which python)"

# --- runtime deps (graphcast installed with --no-deps below) -------------------
pip install -U \
  "jax[cuda12]" \
  dm-haiku jraph dm-tree trimesh rtree chex \
  dask gcsfs zarr netCDF4 cartopy matplotlib pandas properscoring \
  dinosaur-dycore

# --- graphcast (provides gencast/rollout/normalization/...) -------------------
if [ ! -d third_party/graphcast/graphcast ]; then
  echo "cloning graphcast ..."
  git clone --depth 1 https://github.com/google-deepmind/graphcast.git third_party/graphcast
fi

# Patch rollout.py for modern xarray: Dataset(Dataset) copy-construct was removed.
ROLL="third_party/graphcast/graphcast/rollout.py"
python - "$ROLL" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
for v in ("inputs", "targets_template", "forcings"):
    s = s.replace(f"{v} = xarray.Dataset({v})", f"{v} = {v}.copy()")
# jax.P was removed from the jax namespace; use PartitionSpec instead.
if "from jax.sharding import PartitionSpec as P" not in s:
    s = s.replace(
        "import jax.numpy as jnp\n",
        "import jax.numpy as jnp\nfrom jax.sharding import PartitionSpec as P\n",
    )
s = s.replace("jax.P(", "P(")
open(p, "w").write(s)
print("patched", p)
PY

pip install --no-deps -e third_party/graphcast

# --- cartopy Natural Earth features (offline maps on compute nodes) ----------
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
import properscoring, cartopy, jax
print("imports OK: gencast/rollout/properscoring/cartopy all import")
print("multiple_runs:", hasattr(rollout, "chunked_prediction_generator_multiple_runs"))
print("jax version:", jax.__version__)
PY

echo "setup_env.sh: done."
