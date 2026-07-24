# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Subagent model tier (ultracode / multi-agent runs)

When orchestrating subagents (Agent tool or Workflow `agent()` calls), run them **one model
tier below the main session model**: Fable main → `opus` subagents; Opus main → `sonnet`
subagents. Pass the override explicitly (e.g. `model: 'opus'`) — do not let subagents
inherit the top-tier session model.

## Job name — always `Vayuh-s2s` (STRICT)

**Every** cluster job submitted from this repo — Slurm and PBS, GPU and CPU, arrays,
smoke tests, and ad-hoc `sbatch`/`qsub` — MUST use the job name `Vayuh-s2s`. This is a
fixed, unique identifier for all of this project's jobs on shared schedulers; do not use
per-task descriptive names.

- Slurm scripts: `#SBATCH --job-name=Vayuh-s2s`
- PBS scripts: `#PBS -N Vayuh-s2s`
- Ad-hoc submissions: `sbatch --job-name=Vayuh-s2s ...`, `qsub -N Vayuh-s2s ...`
- When creating a new job script or editing an existing one, set/replace the job name to
  `Vayuh-s2s` — even if the surrounding script still has an old descriptive name.
- Monitor this project's jobs with `squeue -u $USER -n Vayuh-s2s` (Slurm) or
  `qstat -u exu` then filter on the name (PBS).

Log files carry the job name via `%x-%j`, i.e. `logs/Vayuh-s2s-<jobid>.{out,err}`; per-run
distinctiveness comes from the unique job id.

## Which machine you are on (read this first)

This checkout lives on the **a3mega Slurm GPU cluster** (GCP; hostnames start with
`nucla3m`). You are on its **login node, which has no usable GPU** (`nvidia-smi` fails);
GPUs live on 8 whole-node `a3mega` nodes (H100s) plus small CPU `debug` nodes, reachable
**only via `sbatch`**. The conda env here is **`moe`**. Derecho — where the GenCast
experiments ran — is a **different machine** (NCAR, PBS Pro,
`/glade/derecho/scratch/exu/S2S_ExtremeWeather`, env `my-env`) and is **not reachable from
this box**.

Traps this split sets:

- This box has `qstat`/`qsub` binaries, but they are **Slurm's PBS-compat wrappers for the
  local cluster**. `qstat -u exu` exits 0 with EMPTY output — that means "wrong machine",
  not "no jobs". Every PBS/qstat command in this file and in `HANDOFF.md` is Derecho-only.
- **GenCast now runs on this cluster too** (since Jul 15 2026, weeks 3/4), but only with
  the right resource split: GPU `infer` goes through `sbatch` (`slurm/xres_*` — the 0.25°
  pmap path OOMs on H100-80GB; use the 8-worker serial pool scripts
  `slurm/xres_pool_0p25_week*.slurm`), CPU `prep`/`compare`/figures run on the login node.
  Never launch infer directly on the login node (no GPU), and note the HRRR overlay build
  (`xhrrr`, `C.HRRR_NC`) is still Derecho-only (`/glade` path) — the synced
  `runs/observations/*_hrrr_verif_*.nc` files cover it here.
- What this box holds of GenCast: full source, model checkpoints (`runs/models/`), the
  original horizon-experiment outputs (`runs/week{2,3,4}/`), and — since Jul 15–16 2026 —
  the **complete xres data tree** (`runs/xres/`, ~234 GB: week2 cubes/verif synced from
  Derecho, week3/4 produced locally by Slurm jobs 218/219/223/224) plus all compare
  figures (`figures/xres/`). Derecho is no longer required for xres analysis.
- The **downscaler and xres analysis are the live projects on this machine**; only new
  Derecho PBS runs (or HRRR truth rebuilds) need a Derecho session.

## GPU nodes are for GPU compute ONLY (STRICT)

A GPU allocation — an `sbatch` job on the a3mega H100 nodes, or a Derecho GPU/PBS job — is
reserved **exclusively for the work that actually needs the GPU**: the inference rollout
(GenCast / `xres`) or the training / eval steps (downscaler). H100 nodes are scarce and
hard to get; a whole 8×H100 node doing anything a CPU could do is pure waste (a node idling
on ERA5 downloads for an hour burns ~dozens of GPU-hours).

**Everything that does NOT need the GPU must be done first, on a cheaper resource** — the
**login node** (has internet) or a CPU/`debug` node (a3mega) / the `develop` CPU queue
(Derecho). That includes:

- **Data gathering / I/O:** checkpoint/stats/statics downloads, ERA5/HRRR reads, building
  per-event init frames and verification truth (`run_xres.py --stage prep`), the
  downscaler's timestamp index + norm stats (`downscaler/bash/02_*`). Pre-stage all of it to
  the shared NFS `runs/` caches.
- **Environment / software builds & compiles:** conda/pip installs, editable installs,
  `setup_env.sh`, any CUDA-extension or package compilation.
- **Preprocessing / figures:** the `compare` stage, plotting, metrics, CSVs.

By the time a GPU job starts, **all inputs must already exist so the job is a pure
cache-hit → GPU compute**. For `xres`: run `python run_xres.py --stage prep --weeks N` on the
**login node**, confirm `ls runs/xres/<res>/weekN/inputs/*_inputs.nc | wc -l` == 12, and only
*then* `sbatch` the infer job. The infer scripts still call `prep` in-job, but that must be an
instant **cache verify**, never a builder — if a GPU job is seen downloading or building data,
stop it and pre-stage on the login node.

The one unavoidable on-GPU startup cost is the **JAX/XLA first-step compile** (it targets the
device and cannot be moved off-GPU). That is acceptable, but it must be the *only* startup the
GPU pays — never compound it with prep/downloads. (Incident 2026-07-15: `xres` week3/4 were
`sbatch`ed straight to the H100s and spent ~75 min per node building ERA5 init frames with all
8 GPUs idle before compiling — exactly what this rule forbids. Do prep on the login node.)

## What this is

**Two independent stacks live in this repo.** They share a scientific goal (skillful,
high-resolution forecasts of CONUS extreme events) but share **no code**, and their live
homes are different clusters. (They are not fully separate *environments*: the `moe` env
here imports both stacks, which is exactly why the machine check above matters.) Figure out
which one you are working on before touching anything:

| | **GenCast S2S** (original) | **Downscaler** (extension) |
|---|---|---|
| Dirs | `gencast_s2s/`, `xres/`, `pbs/`, `runs/` | `downscaler/` (self-contained) |
| Framework | JAX / GraphCast | PyTorch (DDP) |
| Runs on | **Derecho** (PBS, `my-env`) — not reachable from here | **this cluster** (a3mega Slurm, `moe` env) |
| Does | ensemble forecasts at 1.0° & 0.25°, weeks 2–4 lead | super-resolves 1° → 3 km |
| Docs | this file + `HANDOFF.md` | `downscaler/README.md` + `downscaler/bash/README.md` |

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

## Running GenCast (Derecho sessions ONLY)

Applies to `gencast_s2s/` + `xres/` only, **and only in a session on Derecho**
(`/glade/derecho/scratch/exu/S2S_ExtremeWeather`). None of the commands in this section
work from the a3mega box — see "Which machine you are on" above.

Derecho's login node OOM-kills `prep` (exit 137) — submit everything via PBS. Project code
for `#PBS -A` is in the `pbs/*.pbs` scripts. Env: `module load conda && conda activate
my-env` (the root README's `moe` env refers to this repo's original a3mega home; on Derecho
it is always `my-env`). The root `slurm/` scripts are for the original a3mega runs — on
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

The GenCast stack has no test suite or linter (the downscaler has one — `pytest
downscaler/tests/`); validation is smoke runs (one event via
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
HRRR overlay truth from the netCDF archive at `C.HRRR_NC` (a `/glade` path, Derecho-only;
the old npz-shard vars `HRRR_SHARDS`/`HRRR_GRID_REF`/`HRRR_T2M_CHANNEL` in the root README
are gone, and the xres figures DO include the red HRRR curve/panel since Jul 7).

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

## Downscaler (`downscaler/`) — the live project on this machine

Self-contained PyTorch project; runs on THIS cluster (env `moe`, GPUs via `sbatch` — never
PBS). It was moved in from `eric/downscaler` as an extension of this work (see "What this
is" above).

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
# env is `moe`. ALL downscaler commands assume cwd = downscaler/ (relative config paths
# break from the repo root).
cd downscaler

# Preferred path: the operator runbook in bash/ (six idempotent scripts; bash/README.md):
bash bash/00_preflight.sh              # read-only env/data/config checks (~1 min)
bash bash/01_smoke.sh                  # harness self-test incl. stop/resume, no GPU (~3 min)
bash bash/02_build_index_and_stats.sh  # timestamp index + norm stats. ~35 min (stats stream
                                       #   500 samples at ~4 s each) — it is NOT hung. Never
                                       #   run two at once (the stats write is not atomic).
bash bash/03_train.sh                  # start OR resume (submits slurm/train.sbatch here;
                                       #   falls back to torchrun only where GPUs are visible)
bash bash/05_status.sh                 # running? which step? healthy?
bash bash/04_stop.sh                   # graceful stop — checkpoints, then exits
                                       #   (CKPT_DIR=... if the run used a non-default ckpt_dir)

# State checks — trust these, not status claims in docs (doc claims rot):
wc -l valid_timestamps.txt             # 16019 => timestamp index built
ls norm_stats/stats.npz                # norm stats built? (FORCE_REBUILD=1 bash/02 to redo)
ls checkpoints/                        # trained checkpoints, if any
squeue -u $USER -n downscaler          # training job live?

# Manual CPU-safe smoke run. The ckpt_dir override is MANDATORY: without it the tiny smoke
# checkpoint lands in the real checkpoints/, and the next real run auto-resumes from a toy
# model and crashes on a state-dict mismatch.
python train.py data.use_dummy=true training.max_steps=3 training.n_epochs=1 \
  data.grid_height=24 data.grid_width=32 data.num_workers=0 \
  model.C_base=16 'model.n_res_blocks=[1,1]' model.num_groups=8 logging.enabled=false \
  ckpt_dir=checkpoints/_smoke

# torchrun works ONLY where GPUs are visible (inside an sbatch job or a real GPU box —
# NOT on this login node, where nvidia-smi itself fails):
torchrun --nproc_per_node=<N> train.py data.use_dummy=false

pytest tests/   # 6 files: test_smoke (dummy data + one denoise step), test_model
                # (UNet/EDM/EMA shapes & formulas, ckpt weight loading), test_attention
                # (Swin invariants, identity-at-init), test_var_spec (spec parsing +
                # precip transform, DictConfig regression), test_wind_rotation (Lambert
                # angle vs analytic), test_variable_weighting (Ocampo schedule/cap)
```

Config is a Hydra tree; `configs/data/era5_hrrr.yaml` is the main knob. Hydra gotcha: keys
not already in the config need a `+` prefix — e.g. `+stats.n_samples=500` for
`scripts/precompute_stats.py` (without the `+`, Hydra rejects the override).

### Downscaler constraints that bite

- **Real data is on disk and `use_dummy: false` is now the default.** ERA5 (16,072 files)
  and HRRR (16,019 files, `hrrr_nc_v3_rebuilt/`) both cover 2015–2025 6-hourly, and the
  config's paths/coord names/variable names are verified against them. `use_dummy=true`
  still swaps in random tensors for smoke runs — do not report loss curves from that path
  as if they mean anything. Check training prerequisites (index / stats / checkpoints) with
  the state commands above, never with a doc's status line.
- **A train/val/test split exists and matters**: 2015–2022 / 2023 / 2024–25 by year
  (`configs/data/era5_hrrr.yaml`; `train.py` filters the index; stats use train years only).
  **Warning:** 9 of the 12 xres extreme events fall inside 2015–2022, so scoring downscaled
  GenCast members on pre-2023 events with this split evaluates on training data. Before the
  definitive training run either set `train_end: 2018-12-31` (mirrors GenCast's `<2019`
  checkpoints — all 12 events become out-of-sample) or plan to score 2023+ events only.
- **HRRR winds are grid-relative; ERA5's are Earth-relative.** `Era5HrrrDataset` rotates the
  HRRR wind targets to Earth-relative (`data/wind_rotation.py`) so model outputs are
  Earth-relative. The Lambert convergence angle reaches ±22.8° at the edges of CONUS —
  ignoring it inflates 10 m u-wind RMSE by ~16% and makes the model learn the projection.
  Any new wind channel must be declared in `data.hrrr_wind_pairs` or it will NOT be rotated.
- **Sanity-check the data contract on one timestep** before a long run:
  `python scripts/compare_timestep.py --timestamp 2023-07-15T18` — prints the
  bilinear-interpolation baseline and writes a figure. Beating that baseline on
  single-sample RMSE is NOT the success criterion — see "Evaluation caveats" in
  `downscaler/README.md`.
- **Channel counts are asserted at startup** by `train.py` and will hard-fail on mismatch:
  `n_era5_cond == len(era5_variables)`, `n_prognostic`/`n_diagnostic` likewise,
  `model.n_output_ch == n_prognostic + n_diagnostic`, and
  `model.n_input_ch == n_output_ch + n_era5_cond + n_static`. Variable lists live in
  `configs/data/era5_hrrr.yaml` and the counts in `configs/model/unet.yaml` — **grow them
  together**, it is a config edit, not a code change. Currently 9 ERA5 drivers
  (`n_input_ch=16`, `n_output_ch=5`). Three things those asserts do NOT catch: (1) a stale
  `stats.npz` — any variable-list change requires
  `FORCE_REBUILD=1 bash bash/02_build_index_and_stats.sh`, else training crashes on a shape
  mismatch at the FIRST BATCH, after the Slurm allocation; (2) a variable name that isn't in
  the ERA5 files — only 14 variables exist, inventory in `downscaler/docs/variables.md`;
  (3) nothing about attention — `attn_num_heads` is NOT affected by input-channel growth
  (the stem conv absorbs the input count before the bottleneck).
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
- `Era5HrrrDataset` regrids ERA5 per `__getitem__` via **cached bilinear weights**
  (built once; scipy-equivalence pinned by `tests/test_regrid.py`). A warm sample still
  costs ~0.9 s of netCDF reads, so `data.num_workers` (12) is what keeps the GPUs fed —
  2 workers starved an 8-GPU run to ~21 s/step (Jul 13, job 172).
- Cluster-ops scripts from the source project (a3mega launchers, shard builders, rollout
  movies, watchdogs) were **not** copied; grab them from `moein/regional/hrrr_edm/scripts/`
  if needed.
