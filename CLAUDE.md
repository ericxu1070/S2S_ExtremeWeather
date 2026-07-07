# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GenCast S2S extreme-weather ensemble forecasting. For each out-of-sample (post-2019) CONUS
extreme event, GenCast is initialized weeks before the event peak and the forecast ensemble
distribution over the verification week (the 7 days ending at the event peak) is scored
against ERA5 (and optionally HRRR) truth. Two experiments share one codebase:

1. **Original horizon experiment** (`gencast_s2s/` + `run_experiment.py`): 1.0° Mini
   checkpoint, weeks 2/3/4 leads, week-mean T2m anomaly only.
2. **Cross-resolution experiment** (`xres/` + `run_xres.py`): full 0.25° vs 1.0° `<2019`
   checkpoints, week-2 only, 12 events (heat/cold + hurricanes + extreme rain), metrics
   T2m anomaly / u850 wind speed / total precip. Caches the *full* output cube per event so
   new variables can be re-plotted offline without re-running GenCast.

`xres` imports from and reuses `gencast_s2s` (shared checkpoints, stats, statics,
climatology, ERA5 truth under `runs/observations/`), but writes to a separate tree
(`runs/xres/`) so the original outputs are untouched. `xres/xconfig.py` builds on
`gencast_s2s/config.py`.

**Read `HANDOFF.md` first** when resuming work — it tracks live PBS job IDs, what data is
already built, known failure modes, and per-job triage steps. Update it after resubmitting
jobs.

## Running (Derecho / PBS)

This is a login node: `prep` gets OOM-killed (exit 137) if run here — submit everything via
PBS. Project code for `#PBS -A` is in the `pbs/*.pbs` scripts. Env: `module load conda &&
conda activate my-env` (the README's `moe` env refers to the original 8×H100 Slurm cluster;
on Derecho it is always `my-env`). `slurm/` scripts are for that original cluster — on
Derecho use `pbs/` only.

```bash
# xres pipeline: full dependency chain (see HANDOFF.md "Resubmit")
PREP_ID=$(qsub pbs/xres_prep.pbs)                                   # CPU, develop/cpudev
JOB025=$(qsub -W depend=afterok:$PREP_ID pbs/xres_res0p25_week2.pbs) # GPU
JOB10=$(qsub -W depend=afterok:$PREP_ID pbs/xres_res1p0_week2.pbs)   # GPU
qsub -W depend=afterok:$JOB025:$JOB10 pbs/xres_compare.pbs           # CPU

# Manual stages (inside a PBS job, not the login node):
python run_xres.py --stage prep                    # CPU + internet
python run_xres.py --stage infer --res 0p25        # GPU; --res required for infer
python run_xres.py --stage compare                 # CPU, after BOTH infer jobs

# Original experiment:
python run_experiment.py --weeks 2 --stage {prep,infer,plot,combined,all}

# Smoke test one event without redownloading checkpoints:
XRES_EVENTS_SEL=HurricaneIdalia_2023 XRES_PREP_SKIP_MODELS=1 python run_xres.py --stage prep

# Monitoring:
qstat -u exu
qstat -fx JOBID.desched1 | grep -E "job_state|Exit_status"
tail -80 logs/<jobname>.log
```

There is no test suite or linter; validation is smoke runs (one event via
`XRES_EVENTS_SEL=...` / `GENCAST_EVENTS=...` / `*_MAX_EVENTS=1`) plus checking expected
output files (counts listed in HANDOFF.md).

## Architecture

Stage pipeline, each stage cache-aware (existing output files are skipped, so failed jobs
are resubmitted, not restarted):

- **prep** (CPU, needs internet): download checkpoints/stats/statics from GCS, build ERA5
  truth + per-event initialization frames. In `run_xres.py`, prep runs each event in an
  isolated subprocess by default (`XRES_PREP_ISOLATE=1`) to contain memory.
- **infer** (GPU, fully offline): load checkpoint (`model.py` swaps sparse attention for a
  GPU dense-attention implementation), roll the ensemble, save per-member verification
  fields (and, for xres, the full cube under `cache/`). Events run sequentially with
  per-event caching, so a mid-run failure preserves completed events.
- **plot / compare** (CPU): PDFs, cartopy CONUS maps, CRPS/rank histograms
  (`plotting.py`/`xplotting.py`, `combined_pdfs.py`/`xcombined.py`).

Key modules: `config.py`/`xconfig.py` define events, checkpoints, the CONUS box, and the
entire `runs/` folder layout (all paths flow from here); `data.py`/`xdata.py` handle ERA5
access; `xmetrics.py` derives per-member metrics from cubes; `hrrr.py`/`xhrrr.py` build the
HRRR overlay truth (skipped on Derecho — no shard archive, figures are ERA5-only).

`third_party/graphcast/` is a gitignored editable install created by `setup_env.sh`, which
also patches `rollout.py` for modern xarray/jax — never reclone without re-applying the
patch (i.e., rerun `setup_env.sh`). `runs/` (all data/outputs) is also gitignored.

## Constraints that bite

- **ARCO ERA5**: never `xr.open_zarr()` the ARCO root store (~120 GB OOM). Use
  `arco_read()` in `gencast_s2s/data.py` for post-2023 events.
- **0.25° checkpoint needs ~50 GiB VRAM per member** — does not fit Derecho's A100-40GB in
  the normal `pmap` path. `pbs/xres_res0p25_week2.pbs` runs it single-GPU, serial
  (`XRES_SERIAL_INFER=1`), bfloat16 (`XRES_BF16=1`). The 1.0° model runs the standard
  multi-GPU path.
- **Ensemble size must be a multiple of the visible GPU count** (pmap requirement).
  Defaults to 24; override with `XRES_N_MEMBERS_0P25`/`XRES_N_MEMBERS_1P0` or
  `GENCAST_N_MEMBERS`.
- **Queues**: direct `-q gpu` is denied — use `#PBS -q main` with `ngpus` and it routes to
  the gpu queue. Prep uses `develop` (cpudev); do not use gpudev for infer. Account
  walltime cap is 12 h.
- Only post-2019 events are valid (the `<2019` checkpoints would otherwise be scored on
  their training period).
