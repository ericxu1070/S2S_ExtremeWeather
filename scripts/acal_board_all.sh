#!/usr/bin/env bash
# =============================================================================
# acal_board_all.sh - the multi-model S2S baseline board, end to end.
#
# LOGIN NODE, CPU only (Derecho login or a3mega login; env my-env / moe). Nothing
# here needs a GPU, and nothing here downloads: every source's cubes, hind cubes
# and json must already be built by its own module (acal/s2s_<name>.py).
#
# For every source that HAS DATA (s2sbase --stage with_data: cfs13, gefs, geps, and
# ec46 as soon as its cubes exist - no edit needed to pick it up) and every truth
# (era5, hrrr = headline HRRR, hrrr_raw = sensitivity) it runs:
#
#   json -> bias (LOYO sources only; EC46 writes its own reforecast bias)
#        -> score -> paired -> maps -> roc -> reach -> sidebyside (closest members)
#        -> per-source map / ROC / reach figures (era5 + hrrr only; hrrr_raw = tables)
#
# then the board (acal.board --stage all) and the figure modules acal/errmaps.py,
# acal/overall.py, acal/bbsubs.py - each only if its file exists.
#
# Idempotent and cache-aware: a step whose output is newer than all of its inputs
# (source json/cubes, bias, truth index, the AI+RES window table, the code) is
# skipped. FORCE=1 reruns everything. Steps keep going past a failure; the exit
# code is the number of failed steps (0 = clean).
#
# Outputs: runs/acal/analysis/s2s/<truth>/ (tables), figures/acal/s2s/<truth>/<source>/
# (per-source figures), plus whatever board/errmaps/overall/bbsubs write. Nothing under
# the published runs/acal/{cfs,analysis/*.csv} or figures/acal/*.png is written.
#
# Usage (from anywhere):
#   bash scripts/acal_board_all.sh                        # everything
#   TRUTHS="era5" SOURCES="gefs" bash scripts/acal_board_all.sh   # a subset
#   setsid nohup bash scripts/acal_board_all.sh > runs/acal/analysis/s2s/logs/board_all.log 2>&1 &
# The truths run in parallel (one log each: runs/acal/analysis/s2s/logs/board_all_<truth>.log;
# SERIAL=1 runs them one after another in this log). Wall time ~25 min from scratch
# (per source and truth: score + paired ~2 min, maps ~2 min, figures ~2 min), a few
# minutes when everything is cached.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 99
if ! python -c "import xarray, acal" >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    module load conda >/dev/null 2>&1 && conda activate my-env 2>/dev/null \
        || conda activate moe 2>/dev/null || true
fi

TRUTHS="${TRUTHS:-era5 hrrr hrrr_raw}"
FIG_TRUTHS="${FIG_TRUTHS:-era5 hrrr}"          # hrrr_raw is a table-only sensitivity
FORCE="${FORCE:-}"
AN=runs/acal/analysis/s2s
FAILED=()

step() {        # run one step, keep going on failure
    echo "=== $(date +%T) $*"
    if ! "$@"; then
        echo "!!! FAILED: $*"
        FAILED+=("$*")
    fi
}

fresh() {       # fresh OUT DEP...: OUT exists and no DEP (file, or dir at depth 1) is newer
    local out=$1; shift
    [ -n "$FORCE" ] && return 1
    [ -e "$out" ] || return 1
    local d
    for d in "$@"; do
        [ -e "$d" ] || continue
        [ -n "$(find -L "$d" -maxdepth 1 -newer "$out" -print -quit 2>/dev/null)" ] && return 1
    done
    return 0
}

skip() { echo "--- $(date +%T) cached: $*"; }

has_flag() {    # has_flag MODULE FLAG: the module's --help mentions FLAG
    python -m "$1" --help 2>/dev/null | grep -q -- "$2"
}

CODE="acal/s2sbase.py acal/cfsbase.py acal/analyze.py acal/truth.py acal/hrrrtruth.py"

# --------------------------------------------------------------------------- #
# 0. Shared AI+RES inputs (cache-aware inside the module)
# --------------------------------------------------------------------------- #
step python -m acal.s2sbase --stage aires_fields    # AI+RES 13f/12f walker fields (cached)
case " $TRUTHS " in *" hrrr"*)
    # masked walker A_L columns (hrrr and hrrr_raw share the HRRR mask); cached per case
    if ! head -1 "$AN/aires_al_windows.csv" 2>/dev/null | grep -q al12_hrrrmask || [ -n "$FORCE" ]; then
        step python -m acal.s2sbase --stage aires_mask --truth hrrr
    else
        skip "aires_al_windows.csv masked columns"
    fi ;;
esac

# --------------------------------------------------------------------------- #
# 1. Per source x truth
# --------------------------------------------------------------------------- #
mapfile -t ROWS < <(python -m acal.s2sbase --stage with_data)
if [ "${#ROWS[@]}" -eq 0 ]; then
    echo "!!! no source has data (s2sbase --stage with_data printed nothing)"
    exit 98
fi
echo "sources with data: $(printf '%s ' "${ROWS[@]%% *}")"

# 1a. per source: json + bias (serial; shared by every truth)
SEL=()
for row in "${ROWS[@]}"; do
    read -r S BIAS ROOT BIAS_CSV <<< "$row"
    if [ -n "${SOURCES:-}" ] && [[ " $SOURCES " != *" $S "* ]]; then
        continue
    fi
    SEL+=("$row")
    # json: one record per case (the module skips records that exist; FORCE rebuilds)
    if [ -n "$FORCE" ]; then
        step python -m acal.s2sbase --source "$S" --stage json --force
    else
        step python -m acal.s2sbase --source "$S" --stage json
    fi
    # LOYO bias from the hind cubes (cfs13 reads the published CFS bias; EC46 = reforecast)
    if [ "$BIAS" = loyo ] && [ "$S" != cfs13 ]; then
        if fresh "$BIAS_CSV" "$ROOT/hind" acal/s2sbase.py; then
            skip "$S bias"
        else
            step python -m acal.s2sbase --source "$S" --stage bias
        fi
    fi
done

# 1b. per truth (truths in parallel: every output, s2s_summary.json included, is per
# truth; sources stay serial within a truth)
run_truth() {
    local T=$1 row S BIAS ROOT BIAS_CSV OD TRUTH_DEPS DEPS SRC_CODE FD FIG
    local -a FAILED=()     # dynamic scope: step() appends here
    OD="$AN/$T"
    for row in "${SEL[@]}"; do
        read -r S BIAS ROOT BIAS_CSV <<< "$row"
        SRC_CODE="$CODE"
        [ -f "acal/s2s_$S.py" ] && SRC_CODE="$SRC_CODE acal/s2s_$S.py"
        TRUTH_DEPS="$AN/aires_al_windows.csv"
        [[ "$T" == hrrr* ]] && TRUTH_DEPS="$TRUTH_DEPS runs/acal/index_hrrr"
        DEPS="$ROOT $BIAS_CSV $TRUTH_DEPS $SRC_CODE"
        # shellcheck disable=SC2086
        {
        fresh "$OD/${S}_scorecard.csv" $DEPS && skip "$S $T score" \
            || step python -m acal.s2sbase --source "$S" --stage score --truth "$T"
        fresh "$OD/${S}_paired.csv" $DEPS && skip "$S $T paired" \
            || step python -m acal.s2sbase --source "$S" --stage paired --truth "$T"
        fresh "$OD/maps_daily_${S}.nc" $DEPS acal/maps.py && skip "$S $T maps" \
            || step python -m acal.s2sbase --source "$S" --stage maps --truth "$T"
        FIG=0
        [[ " $FIG_TRUTHS " == *" $T "* ]] && FIG=1
        FD="figures/acal/s2s/$T/$S"
        if [ $FIG = 1 ]; then
            fresh "$FD/roc/roc_overlay.png" $DEPS acal/roc.py && skip "$S $T roc" \
                || step python -m acal.roc --source "$S" --truth "$T" --figures
            fresh "$FD/reach/prob_4K.png" $DEPS acal/roc.py acal/reach.py \
                && skip "$S $T reach" \
                || step python -m acal.reach --source "$S" --truth "$T"
            fresh "$OD/maps_land_means_${S}.json" "$OD/maps_daily_${S}.nc" \
                "$OD/maps_fields_${S}.nc" acal/maps.py && skip "$S $T map figures" \
                || step python -m acal.maps --stage source --source "$S" --truth "$T"
        else
            fresh "$OD/roc_${S}_auc.csv" $DEPS acal/roc.py && skip "$S $T roc" \
                || step python -m acal.roc --source "$S" --truth "$T"
            fresh "$OD/${S}_prob_4K.csv" $DEPS acal/roc.py acal/reach.py \
                && skip "$S $T reach" \
                || step python -m acal.reach --source "$S" --truth "$T" --no-figures
        fi
        fresh "$OD/closest_members_${S}.csv" $DEPS acal/maps.py acal/sidebyside.py \
            && skip "$S $T closest members" \
            || step python -m acal.sidebyside --source "$S" --truth "$T"
        }
    done
    echo "=== $(date +%T) truth $T: ${#FAILED[@]} failed step(s)"
    for f in "${FAILED[@]}"; do echo "    FAILED: $f"; done
    return "${#FAILED[@]}"
}

LOGDIR="${LOGDIR:-$AN/logs}"
mkdir -p "$LOGDIR"
declare -A PID
for T in $TRUTHS; do
    if [ -n "${SERIAL:-}" ]; then
        run_truth "$T" || FAILED+=("truth $T: $? step(s)")
    else
        run_truth "$T" > "$LOGDIR/board_all_$T.log" 2>&1 &
        PID[$T]=$!
        echo "--- truth $T running (pid ${PID[$T]}), log $LOGDIR/board_all_$T.log"
    fi
done
for T in "${!PID[@]}"; do
    wait "${PID[$T]}"; rc=$?
    tail -n 3 "$LOGDIR/board_all_$T.log" | sed "s/^/[$T] /"
    [ "$rc" -ne 0 ] && FAILED+=("truth $T: $rc step(s), see $LOGDIR/board_all_$T.log")
done

# --------------------------------------------------------------------------- #
# 2. The board and the board figures (each only if its module exists)
# --------------------------------------------------------------------------- #
if [ -f acal/board.py ]; then
    step python -m acal.board --stage all
else
    echo "--- acal/board.py not written yet: board skipped"
fi
for M in errmaps overall bbsubs; do
    if [ ! -f "acal/$M.py" ]; then
        echo "--- acal/$M.py not written yet: skipped"
        continue
    fi
    if has_flag "acal.$M" "--truth"; then
        for T in $TRUTHS; do
            step python -m "acal.$M" --truth "$T"
        done
    else
        step python -m "acal.$M"
    fi
done

echo "=== $(date +%T) done; ${#FAILED[@]} failed step(s)"
for f in "${FAILED[@]}"; do echo "    FAILED: $f"; done
exit "${#FAILED[@]}"
