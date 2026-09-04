#!/usr/bin/env bash
# =============================================================================
# sync_a3mega_to_derecho.sh - push the completed FCN3 week-3 (6-event) results,
# and the GenCast/truth files Derecho is missing, from the a3mega H100 cluster
# to Derecho.
#
# RUN THIS ON a3mega. It is a PUSH: rsync outbound over SSH while logged into
# the H100 box. `runs/` is gitignored, so git does not move any of this data;
# only `figures/` and `logs/` travel with the branch.
#
#   bash scripts/sync_a3mega_to_derecho.sh --census-only # what is here? read-only, runs on EITHER box
#   bash scripts/sync_a3mega_to_derecho.sh --dry-run     # census + rsync -n, moves nothing
#   bash scripts/sync_a3mega_to_derecho.sh               # transfer, then verify counts
#   bash scripts/sync_a3mega_to_derecho.sh --with-p90    # also the 90-case p90_t2m experiment
#   bash scripts/sync_a3mega_to_derecho.sh --with-shards # also the per-shard zarr intermediates
#   bash scripts/sync_a3mega_to_derecho.sh --with-walkers # also the AI+RES walker states (~0.5 GB each)
#
# The AI+RES Gate 2 artifacts are in the DEFAULT manifest (~530 MB): the adapter
# ensemble cube cannot be rebuilt on Derecho (no GPU that fits FCN3), so without
# it a Derecho session can read the Gate 2 verdict but cannot re-derive it.
#
# Overridable:
#   DEST_HOST   ssh target for Derecho          (default: derecho.hpc.ucar.edu)
#   DEST_ROOT   repo root on Derecho            (default: /glade/derecho/scratch/exu/S2S_ExtremeWeather)
#   SRC_ROOT    repo root here                  (default: two levels up from this script)
#
# Every manifest group is verified after transfer by counting files on the
# remote side and comparing against the local count. Nothing is ever deleted:
# there is no --delete anywhere in this script.
# =============================================================================
set -uo pipefail

DEST_HOST="${DEST_HOST:-derecho.hpc.ucar.edu}"
DEST_ROOT="${DEST_ROOT:-/glade/derecho/scratch/exu/S2S_ExtremeWeather}"
SRC_ROOT="${SRC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

DRY_RUN=0
CENSUS_ONLY=0
WITH_P90=0
WITH_SHARDS=0
WITH_WALKERS=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=1 ;;
        --census-only) CENSUS_ONLY=1 ;;
        --with-p90)    WITH_P90=1 ;;
        --with-shards) WITH_SHARDS=1 ;;
        --with-walkers) WITH_WALKERS=1 ;;
        -h|--help)     sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)             echo "unknown flag: $arg (see --help)"; exit 2 ;;
    esac
done

# ------------------------------------------------------------------ direction
# This script only makes sense in the a3mega -> Derecho direction. /glade only
# exists on Derecho, so its presence means someone ran this on the wrong box.
# --census-only is read-only, so it is allowed on either box (and is the way to
# see the receiving side's gaps before transferring).
if [ -d /glade ] && [ "$CENSUS_ONLY" = 0 ]; then
    cat <<'MSG'
FAIL: you are on Derecho (/glade exists). This script PUSHES from a3mega to
Derecho and must be run from an a3mega login node:

    ssh <a3mega>
    cd .../S2S_ExtremeWeather && git fetch && git checkout aires
    bash scripts/sync_a3mega_to_derecho.sh --dry-run
MSG
    exit 1
fi

cd "$SRC_ROOT" || { echo "FAIL: cannot cd to SRC_ROOT=$SRC_ROOT"; exit 1; }

# ------------------------------------------------------------------- manifest
# One entry per line: "<glob relative to repo root>|<expected count>|<label>"
# Expected count of 0 means "no assertion, transfer whatever is there".
MANIFEST=(
    # --- the completed FCN3 6-event week-3 comparison -------------------------
    "runs/fcn3/week3/cache/*_cube.nc|6|FCN3 forecast cubes (24 x 85 x 105 x 237)"
    "runs/fcn3/week3/ic/*_ic.nc|6|FCN3 initial conditions"
    "runs/fcn3/week3/timing/*.json|0|FCN3 + GenCast pool timing"
    "runs/fcn3/week3/scores/*|0|fcn3_vs_gencast_scores.csv"
    # --- GenCast side for the 3 p90 events: absent on Derecho -----------------
    "runs/xres/0p25/week3/cache/p90_*_cube.nc|3|GenCast 0.25 deg week-3 cubes, p90 events"
    "runs/xres/0p25/week3/inputs/p90_*_inputs.nc|3|GenCast init frames, p90 events"
    "runs/observations/p90_*|0|ERA5 truth/persistence, p90 events"
    # --- AI+RES (branch `aires`) ---------------------------------------------
    # The adapter ensemble is the one artifact here that Derecho CANNOT rebuild:
    # it takes FCN3, which does not fit an A100-40GB. Everything else in this
    # group is reproducible there on CPU and is carried only for convenience.
    "runs/aires/gate2/cache/*_adapter_cube.nc|0|Gate 2 adapter-IC FCN3 ensemble (NOT rebuildable on Derecho)"
    "runs/aires/gate2/ic/*_adapter_ic.nc|0|Gate 2 adapter initial conditions"
    "runs/aires/gate2/scores/*|0|Gate 2 verdict JSON + per-member CSV"
    "runs/aires/calib/*.nc|0|derived-channel calibration (also rebuildable in ~2 min)"
    # The per-run reductions: every AI+RES figure is drawn from these four files and
    # nothing else. ~0.3 MB per run, and without them a Derecho session can read a
    # verdict out of HANDOFF.md but cannot redraw or re-reduce a single production run
    # -- the walker states they came from are 91 GB and stay on this box.
    "runs/aires/*/res/*/compare.json|0|AI+RES reduction: weights, realized A_L, curve, DS baselines"
    "runs/aires/*/res/*/res_result.json|0|AI+RES run result: DMC steps, ESS, ancestry"
    "runs/aires/*/res/*/run.json|0|AI+RES run configuration (event, box, C_k, seeds)"
    "runs/aires/*/res/*/compare_curve.csv|0|AI+RES exceedance curve, tabular"
    "runs/aires/clim/*.nc|0|climatological A_L box means, 1959-2023 (rebuildable, ~25 min)"
)
if [ "$WITH_SHARDS" = 1 ]; then
    MANIFEST+=( "runs/fcn3/week3/shards|0|per-shard zarr intermediates (large, regenerable)" )
fi
if [ "$WITH_WALKERS" = 1 ]; then
    # Global 2-frame walker states, ~477 MB each. Only worth moving if Derecho is
    # going to restart walkers, which it cannot score -- normally leave these here.
    MANIFEST+=( "runs/aires/*/walkers|0|AI+RES walker states + diagnostics (large)" )
fi
if [ "$WITH_P90" = 1 ]; then
    MANIFEST+=(
        "runs/p90_t2m/ic|0|p90 90-case initial conditions"
        "runs/p90_t2m/truth|0|p90 90-case ERA5 truth + persistence"
        "runs/p90_t2m/fcn3|0|p90 90-case FCN3 cubes/timing/scores"
        "figures/p90_t2m/fcn3|0|p90 90-case figures"
    )
fi

# --------------------------------------------------------------- local census
# Count what actually exists here BEFORE transferring, so a group that never
# finished on a3mega is visible now rather than as a puzzling gap on Derecho.
count_local() {  # $1 = glob or directory
    if [ -d "$1" ]; then
        find "$1" -type f 2>/dev/null | wc -l
    else
        # shellcheck disable=SC2086  # deliberate glob expansion
        ls -1d $1 2>/dev/null | wc -l
    fi
}

if   [ "$CENSUS_ONLY" = 1 ]; then MODE="CENSUS ONLY (read-only)"
elif [ "$DRY_RUN" = 1 ];     then MODE="DRY RUN (nothing moves)"
else                              MODE="TRANSFER"
fi
echo "=============================================================="
echo " source : $(hostname):$SRC_ROOT"
echo " dest   : $DEST_HOST:$DEST_ROOT"
echo " mode   : $MODE"
echo "=============================================================="
printf '%-52s %6s %8s  %s\n' "PATH" "FOUND" "SIZE" "LABEL"

CENSUS_FAIL=0
declare -a LOCAL_COUNT
for i in "${!MANIFEST[@]}"; do
    IFS='|' read -r glob want label <<< "${MANIFEST[$i]}"
    n="$(count_local "$glob")"
    LOCAL_COUNT[i]="$n"
    # shellcheck disable=SC2086
    sz="$(du -shc $glob 2>/dev/null | tail -1 | cut -f1)"
    flag=""
    if [ "$want" != 0 ] && [ "$n" != "$want" ]; then
        flag="  <== EXPECTED $want"
        CENSUS_FAIL=1
    elif [ "$n" = 0 ]; then
        flag="  <== nothing here"
    fi
    printf '%-52s %6s %8s  %s%s\n' "$glob" "$n" "${sz:--}" "$label" "$flag"
done
echo

if [ "$CENSUS_ONLY" = 1 ]; then
    echo "CENSUS ONLY - nothing transferred."
    [ "$CENSUS_FAIL" = 1 ] && exit 1
    exit 0
fi

if [ "$CENSUS_FAIL" = 1 ]; then
    echo "WARNING: at least one group does not match its expected count on the"
    echo "         SOURCE side. Fix that on a3mega before transferring, or the"
    echo "         gap just moves to Derecho. Continuing anyway in 5 s..."
    [ "$DRY_RUN" = 1 ] || sleep 5
fi

# ------------------------------------------------------------------- transfer
# -R keeps the path relative to the repo root, so runs/... lands as runs/... .
# No -z: netCDF/zarr here is already compressed. No --delete, ever.
RSYNC_OPTS=(-rlt --partial -h --info=progress2)
[ "$DRY_RUN" = 1 ] && RSYNC_OPTS+=(-n --itemize-changes)

FAILED=0
for i in "${!MANIFEST[@]}"; do
    IFS='|' read -r glob want label <<< "${MANIFEST[$i]}"
    [ "${LOCAL_COUNT[$i]}" = 0 ] && { echo "-- skip (empty): $glob"; continue; }
    echo "-- rsync: $glob"
    # shellcheck disable=SC2086
    rsync -R "${RSYNC_OPTS[@]}" $glob "${DEST_HOST}:${DEST_ROOT}/" || {
        echo "   FAIL: rsync exit $? for $glob"; FAILED=1; }
done
echo

if [ "$DRY_RUN" = 1 ]; then
    echo "DRY RUN complete - nothing was transferred. Re-run without --dry-run."
    exit 0
fi

# --------------------------------------------------------------- remote verify
# Single ssh round trip: count files per group on Derecho and print them so the
# two censuses can be compared side by side.
echo "-- verifying on $DEST_HOST"
REMOTE_SCRIPT="cd '$DEST_ROOT' || exit 1"$'\n'
for entry in "${MANIFEST[@]}"; do
    IFS='|' read -r glob want label <<< "$entry"
    REMOTE_SCRIPT+="if [ -d '$glob' ]; then n=\$(find '$glob' -type f 2>/dev/null | wc -l); "
    REMOTE_SCRIPT+="else n=\$(ls -1d $glob 2>/dev/null | wc -l); fi; "
    REMOTE_SCRIPT+="printf '%-52s %6s\\n' '$glob' \"\$n\""$'\n'
done

remote_out="$(ssh "$DEST_HOST" "bash -s" <<< "$REMOTE_SCRIPT")" || {
    echo "FAIL: could not verify over ssh. Transfer may still be fine; check by hand."
    exit 1
}

printf '%-52s %6s %6s  %s\n' "PATH" "LOCAL" "REMOTE" "STATUS"
while IFS= read -r line; do
    rglob="$(awk '{print $1}' <<< "$line")"
    rn="$(awk '{print $2}' <<< "$line")"
    for i in "${!MANIFEST[@]}"; do
        IFS='|' read -r glob want label <<< "${MANIFEST[$i]}"
        [ "$glob" = "$rglob" ] || continue
        ln="${LOCAL_COUNT[$i]}"
        if [ "$rn" -ge "$ln" ] 2>/dev/null; then st="ok"; else st="MISMATCH"; FAILED=1; fi
        printf '%-52s %6s %6s  %s\n' "$glob" "$ln" "$rn" "$st"
    done
done <<< "$remote_out"
echo

if [ "$FAILED" = 1 ]; then
    echo "RESULT: FAILED - at least one group did not arrive intact. Re-run the"
    echo "        script; rsync is incremental so completed files are skipped."
    exit 1
fi
cat <<MSG
RESULT: ok - all groups present on Derecho.

Next, on Derecho:
    module load conda && conda activate my-env
    cd $DEST_ROOT && PYTHONPATH=. python fcn3/compare_fcn3_gencast.py
This should regenerate figures/fcn3/week3/ from the transferred cubes.
MSG
