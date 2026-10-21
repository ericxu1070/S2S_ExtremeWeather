#!/bin/bash
# Build the walker-benchmark bundle for the Nebius VM. Derecho login node: disk only,
# no GPU, no network, ~3 min. Produces one ~1.5 GB tar plus its checksum:
#
#   bash wbench/vm/make_bundle.sh                      # -> $OUT/wbench_bundle.tar
#   bash wbench/vm/make_bundle.sh --with-fcn3-weights  # +2.85 GB, if the VM cannot reach HF
#
# What goes in, and why each piece cannot be fetched on the VM instead:
#   code         gencast_s2s xres aires fcn3 wbench, from THIS working tree (so uncommitted
#                wbench work travels too)
#   graphcast    third_party/graphcast with setup_env.sh's rollout.py patch ALREADY
#                applied - a fresh clone would need re-patching
#   checkpoints  GenCast 0.25 and 1.0 deg <2019 + the normalisation stats (220 MB each)
#   inputs       ONE event's week-3 init: GenCast 0.25 deg frames, the stride-4 1.0 deg
#                copy, and FCN3's ERA5 IC at the same instant. Building these needs ARCO
#                ERA5 and the xres prep stage; shipping them is simpler and makes every
#                machine start from the identical atmosphere
#   MANIFEST     sha256 of every data file, checked by setup_vm.sh step `verify`
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${WB_BUNDLE_OUT:-/glade/derecho/scratch/$USER/wbench_bundle}
EVENT=${WB_EVENT:-PNW_HeatDome_2021}
WITH_WEIGHTS=0
[ "${1:-}" = "--with-fcn3-weights" ] && WITH_WEIGHTS=1

STAGE=$OUT/stage
DST=$STAGE/S2S_ExtremeWeather
rm -rf "$STAGE"
mkdir -p "$DST"
cd "$REPO"

# --- code ---------------------------------------------------------------------------- #
for d in gencast_s2s xres aires fcn3 wbench; do
  rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude '.pytest_cache' "$d" "$DST/"
done
cp CLAUDE.md "$DST/"
mkdir -p "$DST/third_party"
rsync -a --exclude .git --exclude __pycache__ --exclude '*.egg-info' --exclude '*.ipynb' \
  third_party/graphcast "$DST/third_party/"
grep -q "from jax.sharding import PartitionSpec as P" \
  "$DST/third_party/graphcast/graphcast/rollout.py" \
  || { echo "third_party/graphcast/graphcast/rollout.py is not patched - run setup_env.sh"; exit 1; }

# --- data (cp -L: some inputs are symlinks into other trees) --------------------------- #
put() {  # put <repo-relative path>
  [ -e "$1" ] || { echo "missing: $1"; exit 1; }
  mkdir -p "$DST/$(dirname "$1")"
  cp -L "$1" "$DST/$1"
}
put "runs/models/params/GenCast 0p25deg <2019.npz"
put "runs/models/params/GenCast 1p0deg <2019.npz"
for f in diffs_stddev_by_level.nc mean_by_level.nc stddev_by_level.nc min_by_level.nc; do
  put "runs/models/stats/$f"
done
put "runs/xres/0p25/week3/inputs/${EVENT}_inputs.nc"
put "runs/xres/1p0/week3/inputs/${EVENT}_inputs.nc"
put "runs/fcn3/week3/ic/${EVENT}_ic.nc"
if [ "$WITH_WEIGHTS" -eq 1 ]; then
  for f in runs/fcn3/.cache/e2s/fcn3/*; do [ -f "$f" ] && put "$f"; done  # skip e2s's empty cache/ dir
fi

# --- manifest + tar ---------------------------------------------------------------------- #
(cd "$DST" && find runs -type f -print0 | sort -z | xargs -0 sha256sum) \
  > "$DST/wbench/vm/MANIFEST.sha256"
tar -cf "$OUT/wbench_bundle.tar" -C "$STAGE" S2S_ExtremeWeather
(cd "$OUT" && sha256sum wbench_bundle.tar > wbench_bundle.tar.sha256)
rm -rf "$STAGE"
du -h "$OUT/wbench_bundle.tar"
echo "bundle: $OUT/wbench_bundle.tar (+ .sha256) - next: wbench/vm/README.md step 2"
