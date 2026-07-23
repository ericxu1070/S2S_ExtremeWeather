#!/bin/bash
# FCN3 vs GenCast head-to-head driver -- six events, week-3 (21 d lead), 0.25 deg.
#
# Events (fcn3/fevents.py): PNW Heat Dome 2021 (heat), Hurricane Ian 2022 (u850 wind
# speed), Winter Storm Uri 2021 (cold), plus three p90 CONUS-mean-T2m cases spanning
# year/season/magnitude. Week 3 is both the xres week3 lead and the frozen p90 lead, so
# GenCast is already cached for the first three and only the three p90 cases need a run.
#
# a3mega cluster (Slurm / H100). ALL CPU + internet work runs on the login node; both
# infer jobs run on ONE whole 8-GPU H100 node via sbatch. Compare is CPU-only, in `moe`.
#
# Stages:
#   prep-gencast  login node, moe:  ERA5 truth + GenCast init frames for the 3 p90 events
#   prep-fcn3     login node, fcn3: global ERA5 ICs for all 6 events + checkpoint cache
#   submit-fcn3   sbatch slurm/fcn3_infer.slurm            (~30 min, all 6 events)
#   submit-gencast sbatch slurm/xres_pool_0p25_week3_p90.slurm  (~3.2 h, 3 p90 events)
#   compare       login node, moe:  PDFs + skill + runtime figures and the scores CSV
#
# Usage (from the repo root or from fcn3/):
#   bash fcn3/run_test.sh plan       # print the event table + what is already built
#   bash fcn3/run_test.sh prep       # both prep stages (login node; safe to re-run)
#   bash fcn3/run_test.sh submit     # both infer jobs
#   bash fcn3/run_test.sh compare    # figures + scores
#   bash fcn3/run_test.sh all        # prep -> submit (compare after the jobs finish)
set -eo pipefail

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
cd "$REPO"

WEEKS=${FCN3_WEEKS:-3}
export FCN3_WEEKS=$WEEKS
CONDA_SH=/home/ubuntu/miniconda3/etc/profile.d/conda.sh

P90_EVENTS=$(python -c "
import sys; sys.path.insert(0,'.')
from fcn3 import fevents as F
print(','.join(e.name for e in F.EVENTS.values() if e.source=='p90'))")

plan() {
  python fcn3/run_fcn3.py --stage plan
  echo
  echo "FCN3 ICs:        $(ls runs/fcn3/week$WEEKS/ic/*_ic.nc 2>/dev/null | wc -l)/6"
  echo "FCN3 cubes:      $(ls runs/fcn3/week$WEEKS/cache/*_cube.nc 2>/dev/null | wc -l)/6"
  echo "GenCast inputs (p90): $(ls runs/xres/0p25/week$WEEKS/inputs/p90_*_inputs.nc 2>/dev/null | wc -l)/3"
  echo "GenCast cubes (p90):  $(ls runs/xres/0p25/week$WEEKS/cache/p90_*_cube.nc 2>/dev/null | wc -l)/3"
  echo "ERA5 truth (p90):     $(ls runs/observations/p90_*_verif_t2m_anom.nc 2>/dev/null | wc -l)/3"
}

prep_gencast() {
  echo "[prep-gencast] ERA5 truth + init frames for: $P90_EVENTS"
  # shellcheck disable=SC1090
  source "$CONDA_SH"; conda activate moe
  XRES_EXTRA_EVENTS="$(python fcn3/fevents.py --spec)" \
  XRES_EVENTS_SEL="$P90_EVENTS" \
    python run_xres.py --stage prep --res 0p25 --weeks "$WEEKS"
}

prep_fcn3() {
  echo "[prep-fcn3] global ERA5 ICs for all 6 events + FCN3 checkpoint cache"
  # shellcheck disable=SC1090
  source "$CONDA_SH"; conda activate "${FCN3_CONDA_ENV:-fcn3}"
  SITE=$(python -c "import site; print(site.getsitepackages()[0])")
  export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  python fcn3/run_fcn3.py --stage prep
}

submit_fcn3() {
  local JID
  JID=$(sbatch --parsable slurm/fcn3_infer.slurm)
  echo "[submit] FCN3 infer -> job $JID   (tail -f logs/fcn3_infer-$JID.out)"
}

submit_gencast() {
  local n
  n=$(ls runs/xres/0p25/week"$WEEKS"/cache/p90_*_cube.nc 2>/dev/null | wc -l)
  if [[ "$n" -eq 3 ]]; then
    echo "[submit] GenCast p90 cubes already built (3/3); skipping"
    return
  fi
  local JID
  JID=$(sbatch --parsable slurm/xres_pool_0p25_week3_p90.slurm)
  echo "[submit] GenCast p90 infer -> job $JID   (tail -f logs/xres_0p25pool_wk3_p90-$JID.out)"
}

run_compare() {
  echo "[compare] FCN3 vs GenCast figures + scores (login node, moe)"
  # shellcheck disable=SC1090
  source "$CONDA_SH"; conda activate moe
  python fcn3/compare_fcn3_gencast.py
}

case "${1:-plan}" in
  plan)           plan ;;
  prep)           prep_gencast; prep_fcn3 ;;
  prep-gencast)   prep_gencast ;;
  prep-fcn3)      prep_fcn3 ;;
  submit)         submit_fcn3; submit_gencast ;;
  submit-fcn3)    submit_fcn3 ;;
  submit-gencast) submit_gencast ;;
  compare)        run_compare ;;
  all)            prep_gencast; prep_fcn3; submit_fcn3; submit_gencast ;;
  *) echo "unknown mode: $1"; echo "use: plan|prep|prep-gencast|prep-fcn3|submit|submit-fcn3|submit-gencast|compare|all"; exit 2 ;;
esac
