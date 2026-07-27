#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# p90 truth backfill: build the missing ERA5 truth/persistence files, ONE ISOLATED
# PROCESS PER CASE in parallel (the sequential login-node build needs ~6 GB and
# ~18 min per case; the 15 GB login node can only run it serially). CPU + internet
# only — no GPU is touched; run this only when a node would otherwise sit idle.
# Cache-aware: cases whose truth already exists are skipped, so resubmits are cheap.
set -uo pipefail

ROOT="/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather"
cd "$ROOT"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate moe
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mapfile -t MISSING < <(
  awk -F, 'NR>1{print $1}' p90/cases.csv | while read -r c; do
    [ -f "runs/p90_t2m/truth/${c}_verif_t2m_anom.nc" ] && \
    [ -f "runs/p90_t2m/truth/${c}_persist_t2m_anom.nc" ] || echo "$c"
  done
)
echo "missing truth for ${#MISSING[@]} case(s): ${MISSING[*]}"
[ "${#MISSING[@]}" -eq 0 ] && exit 0

PIDS=(); FAIL=0
for c in "${MISSING[@]}"; do
  P90_CASES_SEL="$c" python p90/run_p90.py --stage prep-truth \
    > "logs/p90_truth_${SLURM_JOB_ID}_${c}.log" 2>&1 &
  PIDS+=($!)
done
for k in "${!PIDS[@]}"; do
  wait "${PIDS[$k]}" || { echo "case ${MISSING[$k]} FAILED"; FAIL=1; }
done

N=$(ls runs/p90_t2m/truth/*_verif_t2m_anom.nc 2>/dev/null | wc -l)
echo "verif truth files: $N/90  failed=$FAIL"
exit $FAIL
