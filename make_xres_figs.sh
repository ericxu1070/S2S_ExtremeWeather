#!/usr/bin/env bash
# Build the requested xres figure sets, entirely from CACHED data. Login-node CPU work
# only (netCDF reads + matplotlib) — no GPU, no sbatch, no internet.
#
#   set 1  figures/xres/week{2,3,4}/xres_combined_pooled.png   per-lead PDFs, 4 curves
#          (plus the standard mean/best/extreme + _vs_era5_0p25 companions xcombined
#           always emits)
#   set 2  figures/xres/leads/xres_pdf_leads_{0p25,1p0}.png    per-res PDFs, 3 leads + truth
#   set 3  figures/xres/maps3var/<event>_week{2,3,4}_maps3var.png
#          4 representative events x 3 leads; rows = each event's cached metrics
#          (headline + free secondary), 5 map columns
#
# Idempotent: figures are simply rewritten; nothing else is touched. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/home/ubuntu/miniconda3/envs/moe/bin/python}

echo "== [1/3] set 1: per-week combined PDFs (weeks 2/3/4) =="
for W in 2 3 4; do
  "$PY" -c "from xres import xcombined; xcombined.make_all($W)"
done

echo "== [2/3] set 2: cross-lead PDFs (one figure per resolution) =="
"$PY" -m xres.xleadpdfs

echo "== [3/3] set 3: CONUS map grids (4 events x 3 weeks, cached metrics only) =="
"$PY" -m xres.xmapgrid

echo "done — figures under figures/xres/{week2,week3,week4,leads,maps3var}/"
