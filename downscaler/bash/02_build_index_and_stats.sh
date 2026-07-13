#!/usr/bin/env bash
# =============================================================================
# 02_build_index_and_stats.sh — one-time data prerequisites. CPU only, ~35 min
# (the stats pass streams 500 samples through the real read+regrid path at ~4 s each).
#
# Builds the two derived files real training needs, skipping whatever already
# exists (delete a file, or FORCE_REBUILD=1, to redo it):
#
#   1. the timestamp index (data.index_path) — every 6-hourly timestamp between
#      train_start and test_end for which BOTH an ERA5 file and an HRRR file
#      exist on disk;
#   2. the normalization stats (data.stats_path) — per-channel z-score mean/std
#      streamed over STATS_SAMPLES (default 500) evenly-strided TRAIN timestamps
#      via scripts/precompute_stats.py.
#
#   bash bash/02_build_index_and_stats.sh
#   STATS_SAMPLES=1000 bash bash/02_build_index_and_stats.sh
#
# Both writes are atomic (tmp + rename), so an interrupted run cannot leave a
# truncated file behind. Stats are computed from train-years only — never from
# val/test — and on the rotated (Earth-relative) winds the model actually sees.
# =============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATS_SAMPLES="${STATS_SAMPLES:-500}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

export HDF5_USE_FILE_LOCKING=FALSE
cd "$PROJ" || exit 1
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe

readarray -t CFG < <(python - <<'PY'
from omegaconf import OmegaConf
c = OmegaConf.load("configs/data/era5_hrrr.yaml")
print(c.index_path); print(c.stats_path)
PY
)
INDEX="${CFG[0]:-}"; STATS="${CFG[1]:-}"
[ -n "$INDEX" ] && [ -n "$STATS" ] || {
    echo "FAIL: could not read index_path/stats_path from configs/data/era5_hrrr.yaml"; exit 1; }

# Single-instance lock: two concurrent builds race each other on the final stats write
# (this actually happened on Jul 13 — two instances, ~30 min apart). mkdir is atomic.
LOCK="$(dirname "$STATS")/.build_lock"
mkdir -p "$(dirname "$STATS")"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "Another 02_build_index_and_stats.sh appears to be running (lock: $LOCK)."
    echo "If you are sure it is not, remove the lock and rerun:  rmdir '$LOCK'"
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# ---------------------------------------------------------------- 1. index
echo "=============== 1. timestamp index ==============="
if [ -s "$INDEX" ] && [ "$FORCE_REBUILD" != 1 ]; then
    echo "  exists: $INDEX ($(wc -l < "$INDEX") timestamps) — skipping (FORCE_REBUILD=1 to redo)"
else
    python - <<'PY'
import os
from datetime import datetime, timedelta
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/data/era5_hrrr.yaml")

def tree(root):
    """Set of '<year>/<filename>' relative paths — one scandir per year dir."""
    have = set()
    with os.scandir(root) as it:
        for e in it:
            if e.is_dir():
                with os.scandir(e.path) as jt:
                    have.update(f"{e.name}/{f.name}" for f in jt if f.is_file())
    return have

era5, hrrr = tree(cfg.era5_dir), tree(cfg.hrrr_dir)
print(f"  on disk: {len(era5)} ERA5 files, {len(hrrr)} HRRR files")

t = datetime.fromisoformat(str(cfg.train_start))
end = datetime.fromisoformat(str(cfg.test_end)) + timedelta(hours=18)
kept, missing = [], 0
while t <= end:
    if t.strftime(cfg.era5_path_template) in era5 and t.strftime(cfg.hrrr_path_template) in hrrr:
        kept.append(t.strftime("%Y-%m-%dT%H:00:00"))
    else:
        missing += 1
    t += timedelta(hours=6)

tmp = cfg.index_path + ".tmp"
with open(tmp, "w") as f:
    f.write("\n".join(kept) + "\n")
os.replace(tmp, cfg.index_path)

def count(a, b):
    return sum(1 for ts in kept if int(a) <= int(ts[:4]) <= int(b))

print(f"  wrote {cfg.index_path}: {len(kept)} timestamps ({missing} slots missing a file)")
print(f"    train {cfg.train_start[:4]}-{cfg.train_end[:4]}: {count(cfg.train_start[:4], cfg.train_end[:4])}")
print(f"    val   {cfg.val_start[:4]}-{cfg.val_end[:4]}:   {count(cfg.val_start[:4], cfg.val_end[:4])}")
print(f"    test  {cfg.test_start[:4]}-{cfg.test_end[:4]}: {count(cfg.test_start[:4], cfg.test_end[:4])}")
PY
    [ -s "$INDEX" ] || { echo "  FAIL: index was not written"; exit 1; }
fi

# ---------------------------------------------------------------- 2. norm stats
echo
echo "=============== 2. normalization stats ==============="
if [ -f "$STATS" ] && [ "$FORCE_REBUILD" != 1 ]; then
    echo "  exists: $STATS — skipping (FORCE_REBUILD=1 to redo)"
else
    echo "  streaming $STATS_SAMPLES train samples through the real read+regrid path..."
    python -u scripts/precompute_stats.py "+stats.n_samples=$STATS_SAMPLES" 2>&1 \
        | sed 's/^.*\[INFO\] - /  /' || true
    [ -f "$STATS" ] || { echo "  FAIL: stats were not written — check the output above"; exit 1; }
fi

if ! python - <<PY
import numpy as np
s = np.load("$STATS")
print("  stats check:")
for k in ("era5_mu", "hrrr_mu", "diag_mu"):
    print(f"    {k:8s} = {np.array2string(s[k], precision=2, suppress_small=True)}")
bad = [k for k in ("era5_sigma", "hrrr_sigma", "diag_sigma") if not np.all(np.isfinite(s[k])) or np.any(s[k] <= 0)]
assert not bad, f"non-finite or non-positive sigma in {bad}"
print("    all sigmas finite and positive")
PY
then
    echo "  FAIL: $STATS exists but failed the sanity check — delete it (or FORCE_REBUILD=1) and rerun"
    exit 1
fi

echo
echo "Done. Both prerequisites are in place — start training with: bash bash/03_train.sh"
