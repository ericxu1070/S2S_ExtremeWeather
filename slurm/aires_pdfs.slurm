#!/bin/bash
#SBATCH --job-name=Vayuh-s2s
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7552M
#SBATCH --time=00:20:00
#SBATCH --output=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.out
#SBATCH --error=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/logs/%x-%j.err

# AI+RES binned PDFs -- aires/apdfs.py, both stages.
#
#   build   rebuilds all 64 walkers' verification-week T2m anomaly FIELDS from the
#           genealogy (segments 5/6/7, `2m_temperature` only, ~200 MB read), reduces the
#           ERA5/HRRR truth and the three direct ensembles, checks gates G1/G4/G13 and
#           writes runs/aires/<event>/res/<tag>/pdf/pdf_fields.nc (~13 MB).
#   figs    draws figures/aires/<event>/aires_pdf_{conus,box}_<event>_<tag>.png and
#           aires_anom_maps_<event>_<tag>.png from that intermediate alone.
#
# NO GPU. Measured cost on the login node: ~17 s for build, ~15 s for the three figures.
# The 20 min walltime is almost entirely headroom for a cold NFS cache and the cartopy
# feature load.
#
# Resource note: the `debug` partition's nodes are ONE CPU and 7552 MB each
# (`scontrol show partition debug`: TotalCPUs=4 over 4 nodes, DefMemPerCPU=7552). A
# request for 4 CPUs or 16 G on this partition is unsatisfiable and the job would sit
# PENDING forever rather than fail loudly, so this asks for exactly one node's worth. The
# work is single-threaded numpy/xarray and does not want more.
#
# Prereqs (already on disk for the pilot; the build says so and exits nonzero if not):
#   runs/aires/<event>/res/<tag>/compare.json      <- run_aires --stage compare
#   runs/aires/<event>/res/<tag>/res_result.json   <- run_aires --stage res
#   runs/aires/<event>/ds_baseline.json            <- run_aires --stage ds
#   the walker diagnostics cubes under res/<tag>/walkers/w*/step0{5,6,7}/diag.nc
#
# Submit:
#   sbatch slurm/aires_pdfs.slurm
#   sbatch --export=ALL,AIRES_EVENT=WinterStorm_Uri_2021 slurm/aires_pdfs.slurm
#   sbatch --export=ALL,AIRES_PDF_ARGS=--z-scaled slurm/aires_pdfs.slurm
#
# Fallback if `debug` is busy (this is CPU work the login node can do directly):
#   source slurm/aires_env.sh moe
#   python -m aires.apdfs --stage all --force
set -eo pipefail

REPO=/home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
source "$REPO/slurm/aires_env.sh" moe

EVENT=${AIRES_EVENT:-PNW_HeatDome_2021}
TAG=${AIRES_RES_TAG:-pilot}
EXTRA=${AIRES_PDF_ARGS:-}

# Single-threaded on purpose: one debug CPU, and BLAS spawning a thread per node core is
# the failure mode slurm/aires_env.sh already pins against for the FCN3 pool.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

echo "Host: $(hostname)"; echo "Start: $(date)"
echo "AI+RES PDFs: event=$EVENT tag=$TAG extra='${EXTRA}'"
echo "python: $(which python)"

echo
echo "=== stage build ==="
python -m aires.apdfs --stage build --event "$EVENT" --tag "$TAG" --force

echo
echo "=== stage figs ==="
# shellcheck disable=SC2086
python -m aires.apdfs --stage figs --event "$EVENT" --tag "$TAG" $EXTRA

echo
echo "End: $(date)"
