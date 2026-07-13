# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Two independent stacks live in this repo.** They share a scientific goal (skillful,
high-resolution forecasts of CONUS extreme events) but share *no code, no environment, and
no cluster*. Figure out which one you are working on before touching anything:

| | **GenCast S2S** (original) | **Downscaler** (extension) |
|---|---|---|
| Dirs | `gencast_s2s/`, `xres/`, `pbs/`, `runs/` | `downscaler/` (self-contained) |
| Framework | JAX / GraphCast | PyTorch (DDP) |
| Runs on | Derecho, via PBS (`my-env`) | this box / GPU cluster (`moe` env) |
| Does | ensemble forecasts at 1.0° & 0.25°, weeks 2–4 lead | super-resolves 1° → 3 km |
| Docs | this file + `HANDOFF.md` | `downscaler/README.md` |

### 1. GenCast S2S ensemble forecasting (`gencast_s2s/`, `xres/`)

For each out-of-sample (post-2019) CONUS extreme event, GenCast is initialized weeks before
the event peak and the forecast ensemble distribution over the verification week (the 7 days
ending at the event peak) is scored against ERA5 (and optionally HRRR) truth. Two experiments
share this codebase:

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

### 2. ERA5→HRRR downscaler (`downscaler/`) — the extension

Conditional-diffusion (EDM) model that maps **coarse ERA5 (1°) → fine HRRR (3 km CONUS)**.

**Why it lives here.** The `xres` experiment showed what resolution buys you at S2S leads,
but 0.25° GenCast costs ~50 GiB VRAM/member and ~15 h/event — it does not scale, and it
still stops well short of convection-permitting scale. The downscaler is the cheaper other
half of that story: leave the forecast coarse (1.0°, the resolution that actually fits the
ensemble budget) and learn the 1° → 3 km step as a separate generative model. Trained on
ERA5/HRRR *analysis* pairs, it is forecast-model-agnostic, so at inference it can be applied
to a coarse GenCast member to yield a 3 km realization — a **downscaled S2S ensemble**
without ever running GenCast at 0.25°.

That coupling is the intended endpoint, **not yet built**: today the downscaler is trained
and evaluated purely on ERA5→HRRR analysis pairs, and nothing in `downscaler/` imports from
`gencast_s2s/` or `xres/` (nor should it — keep the stacks decoupled; a future coupling
belongs in a thin adapter that writes GenCast members into the dataset's expected format).

**Read `downscaler/README.md`** before working on it — it documents the model, the data
contract, and the channel-count invariants. See "Downscaler" section below for how to run it.

## Running GenCast (Derecho / PBS)

Applies to `gencast_s2s/` + `xres/` only — the downscaler runs on a GPU box, not Derecho.

This is a login node: `prep` gets OOM-killed (exit 137) if run here — submit everything via
PBS. Project code for `#PBS -A` is in the `pbs/*.pbs` scripts. Env: `module load conda &&
conda activate my-env` (the README's `moe` env refers to the original 8×H100 Slurm cluster;
on Derecho it is always `my-env`). `slurm/` scripts are for that original cluster — on
Derecho use `pbs/` only.

```bash
# xres infer (current method): 24-subjob single-GPU PBS ARRAYS — subjob N rolls
# ensemble member N for all events (XRES_MEMBER_SEL mode), cached per member,
# cube assembled by the last finisher. Resubmitting an array is cheap (cached
# members skip in ~2 min without loading the model).
A025=$(qsub pbs/xres_member_0p25.pbs)
A10=$(qsub pbs/xres_member_1p0.pbs)
qsub -W depend=afterok:$A025:$A10 pbs/xres_compare_dev.pbs

# Other stages / fallbacks (independent, cache-aware):
qsub pbs/xres_prep.pbs             # CPU, develop/cpudev
qsub pbs/xres_compare_dev.pbs      # CPU figures (xres_compare.pbs = slow cpu queue)
qsub pbs/xres_res0p25_week2.pbs    # fallback: full-node self-chaining worker pool
qsub pbs/xres_res1p0_week2.pbs     # fallback: full-node pmap path

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

## GenCast architecture

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

## GenCast constraints that bite

- **ARCO ERA5**: never `xr.open_zarr()` the ARCO root store (~120 GB OOM). Use
  `arco_read()` in `gencast_s2s/data.py` for post-2023 events.
- **0.25° checkpoint needs ~50 GiB VRAM per member** — does not fit Derecho's A100-40GB in
  the normal `pmap` path; it must run serial (`XRES_SERIAL_INFER=1`, bfloat16) at ~80 s/step
  ⇒ ~15 h/event for 24 members. Current method: `pbs/xres_member_*.pbs` PBS arrays where
  subjob N rolls only member N of every event (`XRES_MEMBER_SEL`) on 1 GPU (~8 h/subjob for
  0.25°), cached to `cache/<event>_memberNN.nc`; the last finisher assembles each cube.
  One serial process peaks at ~237 GiB host RAM ⇒ 0.25° subjobs request `mem=243gb`
  (2 per 487 GB node). Fallback: `pbs/xres_res0p25_week2.pbs`, a full-node self-chaining
  worker pool using the same member cache plus `cache/claims/` work-stealing.
- **Ensemble size must be a multiple of the visible GPU count** in the pmap path (1.0°);
  the serial 0.25° path has no such constraint. Defaults to 24; override with
  `XRES_N_MEMBERS_0P25`/`XRES_N_MEMBERS_1P0` or `GENCAST_N_MEMBERS`.
- **Queues**: direct `-q gpu` is denied — use `#PBS -q main` with `ngpus` and it routes to
  the gpu queue. Prep uses `develop` (cpudev); do not use gpudev for infer. Account
  walltime cap is 12 h.
- Only post-2019 events are valid (the `<2019` checkpoints would otherwise be scored on
  their training period).

## Downscaler (`downscaler/`)

Self-contained PyTorch project — **do not** load the GenCast env or submit it to PBS. It was
moved in from `eric/downscaler` as an extension of this work (see "What this is" above).

The model is a convolutional **U-Net** (`model/unet.py`) wrapped in **EDM preconditioning**
(`model/preconditioning.py`, Karras et al. 2022), i.e. the diffusion denoiser `D_θ`.
Downscaling is conditional generation: coarse ERA5 is bilinearly regridded onto the HRRR
grid and concatenated onto the noisy target as extra input channels, and the U-Net denoises
the HRRR field conditioned on it. Inference is a **single-shot** conditional sample (EDM Heun
ODE, `evaluation/`), not an autoregressive rollout.

The architecture (UNet + EDM + EMA + DDP trainer) is a **verbatim port** of the HRRR
diffusion *emulator* at `/home/ubuntu/Vayuh/data/moein/regional/hrrr_edm`, re-tasked from
6-hour forecasting to cross-dataset super-resolution. The port is deliberate: everything
generic is unchanged, and the real work is localized to `data/` + `configs/`. If you are
comparing against the source project, the deltas are: conditioning is coarse ERA5 (not the
HRRR state at T), the target is the HRRR **full field** (not the T→T+6h tendency), and
normalization is a full z-score (not zero-mean tendency).

```bash
# env is `moe` (PyTorch), NOT the GenCast `my-env`
cd downscaler

# CPU-safe smoke run — verified working from this location
python train.py data.use_dummy=true training.max_steps=3 training.n_epochs=1 \
  data.grid_height=24 data.grid_width=32 data.num_workers=0 \
  model.C_base=16 'model.n_res_blocks=[1,1]' model.num_groups=8 logging.enabled=false

# full-res dummy (needs GPU) / real training
torchrun --nproc_per_node=<N> train.py data.use_dummy=true training.max_steps=10
torchrun --nproc_per_node=<N> train.py data.use_dummy=false

pytest tests/     # test_smoke.py (dummy dataset + one denoise step), test_model.py (shapes/formulas)
```

Config is a Hydra tree; `configs/data/era5_hrrr.yaml` is the main knob.

### Downscaler constraints that bite

- **Real data is on disk and `use_dummy: false` is now the default.** ERA5 (16,072 files)
  and HRRR (16,019 files, `hrrr_nc_v3_rebuilt/`) both cover 2015–2025 6-hourly, and the
  config's paths/coord names/variable names are verified against them. `use_dummy=true`
  still swaps in random tensors for smoke runs — do not report loss curves from that path
  as if they mean anything. **No trained checkpoint exists yet**; still missing are the
  timestamp index and `stats.npz` (see `downscaler/README.md`).
- **HRRR winds are grid-relative; ERA5's are Earth-relative.** `Era5HrrrDataset` rotates the
  HRRR wind targets to Earth-relative (`data/wind_rotation.py`) so model outputs are
  Earth-relative. The Lambert convergence angle reaches ±22.8° at the edges of CONUS —
  ignoring it inflates 10 m u-wind RMSE by ~16% and makes the model learn the projection.
  Any new wind channel must be declared in `data.hrrr_wind_pairs` or it will NOT be rotated.
- **Sanity-check the data contract on one timestep** before a long run:
  `python scripts/compare_timestep.py --timestamp 2023-07-15T18` — prints the
  bilinear-interpolation baseline (the number the model must beat) and writes a figure.
- **Channel counts are asserted at startup** by `train.py` and will hard-fail on mismatch:
  `n_era5_cond == len(era5_variables)`, `n_prognostic`/`n_diagnostic` likewise,
  `model.n_output_ch == n_prognostic + n_diagnostic`, and
  `model.n_input_ch == n_output_ch + n_era5_cond + n_static`. Variable lists live in
  `configs/data/era5_hrrr.yaml` and the counts in `configs/model/unet.yaml` — **grow them
  together**, it is a config edit, not a code change. Currently 9 ERA5 drivers
  (`n_input_ch=16`, `n_output_ch=5`).
- **ERA5 precip needs its transform.** ERA5 `total_precipitation_12hr` is a 12 h accumulation
  in **metres**; the HRRR `log_tp` target is a 6 h accumulation of `ln(mm + 1e-5)`. The
  `log_precip_m_to_mm` transform (`ERA5_TRANSFORMS` in `data/era5_hrrr_dataset.py`) puts the
  driver on the target's units/scale — without it the channel is 1000x off and 77% zeros.
  The 12h→6h window mismatch remains; `specific_humidity:850/700` is the clean precip
  predictor and is also in the conditioning.
- **The loss is a per-channel mean**, not `loss_prog + loss_diag` (which gave the single
  precip channel 50% of the gradient). `training.diagnostic_loss_weight` (default 1.0) is the
  knob if you want to prioritize precip on purpose.
- **Absolute paths in configs** (`ckpt_dir`, `stats_path`, `index_path`, `era5_dir`,
  `hrrr_dir`, `grid_meta_path`) are machine-specific and were rewritten for this location —
  re-check them if the project moves again.
- **The transformer bottleneck is config-gated.** `model/attention.py` implements
  Swin-style windowed self-attention (shifted windows, relative position bias, adaLN-Zero
  noise conditioning — each block is an exact identity at init), stacked at the
  8×-downsampled bottleneck and on by default (`model.use_attention=true`; knobs:
  `attn_depth`, `attn_window_size`, `attn_num_heads`, `attn_mlp_ratio`). `attn_num_heads`
  must divide the bottleneck channel count `C_base * 2^len(n_res_blocks)` — remember this
  when shrinking `C_base` for smoke runs. `model.use_attention=false` recovers the source
  emulator's pure-conv U-Net.
- `Era5HrrrDataset` regrids ERA5 **per `__getitem__`**, which is fine for smoke runs and
  wasteful for production — regrid once offline and point the dataset at cached fields.
- Cluster-ops scripts from the source project (a3mega launchers, shard builders, rollout
  movies, watchdogs) were **not** copied; grab them from `moein/regional/hrrr_edm/scripts/`
  if needed.
