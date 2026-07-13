# ERA5 → HRRR downscaler

Conditional-diffusion (EDM) **downscaler**: coarse **ERA5 (1°)** → fine **HRRR (3 km CONUS)**.

The model architecture — UNet + EDM preconditioning + EMA + the DDP training loop —
is **ported verbatim** from the HRRR diffusion *emulator* at
`/home/ubuntu/Vayuh/data/moein/regional/hrrr_edm`. This project re-tasks that
architecture for **cross-dataset super-resolution** instead of 6-hour forecasting.

## What is the model?

A **U-Net with a transformer bottleneck** (`model/unet.py`: `Conv2d` stem/head,
`ResBlock`/`DownBlock`/`UpBlock`, GroupNorm+SiLU, skip connections) wrapped in **EDM
preconditioning** (`model/preconditioning.py`, Karras et al. 2022). It is the diffusion
**denoiser** `D_θ`. Downscaling = conditional generation: the coarse ERA5 field
(upsampled to the HRRR grid) is concatenated onto the noisy target as extra input
channels, and the U-Net denoises the HRRR field conditioned on it.

> The transformer lives only at the bottleneck (`model/attention.py`, gated by
> `model.use_attention`, default **true**): `model.attn_depth` Swin-style blocks of
> windowed multi-head self-attention (16×16 windows, alternating unshifted/shifted,
> learned relative position bias) + MLP, at the 8×-downsampled resolution (~133×225 at
> full HRRR res) where global mixing is cheap. Blocks are noise-conditioned via
> **adaLN-Zero**, so each is an exact identity at init and enabling attention does not
> perturb early training. `attn_num_heads` must divide the bottleneck channel count
> (`C_base * 2^n_levels`). Set `model.use_attention=false` to recover the pure-conv
> U-Net (the source emulator's architecture).

## What changed vs. the HRRR emulator

| | HRRR emulator (source) | This downscaler |
|---|---|---|
| Input / conditioning | HRRR state at T (44 ch) | **coarse ERA5 regridded to HRRR grid** + statics |
| Target | HRRR **tendency** T→T+6h | HRRR **full-field state** at T |
| Sample = | (T, T+6h) pair | single timestamp T (ERA5→HRRR) |
| Target normalization | zero-mean tendency | full z-score (mean-subtracted) |
| Wind convention | grid-relative throughout | HRRR winds **rotated to Earth-relative** (`data/wind_rotation.py`) |
| Inference | autoregressive **rollout** | **single-shot** conditional sample (`evaluation/downscale.py`) |

### Wind rotation (easy to miss, and wrong if you do)

HRRR's `u10`/`v10` are **grid-relative** — components along the Lambert Conformal grid axes
(`hrrr_grid_reference.nc`: `uv_relative_to_grid=1`). ERA5's winds are **Earth-relative**.
They differ by the projection's convergence angle: zero on the 262.5°E central meridian,
**±22.8° at the edges of CONUS**.

The emulator could ignore this (HRRR→HRRR, both grid-relative). A *downscaler* cannot: it
maps ERA5→HRRR, so the two conventions meet. `Era5HrrrDataset` therefore rotates the HRRR
wind targets into Earth-relative before normalizing, which means the model's outputs are
Earth-relative and directly comparable to ERA5/GenCast. Turn it off with
`data.rotate_hrrr_winds=false`; declare which channels are winds via `data.hrrr_wind_pairs`.

Skipping the rotation inflates the 10 m u-wind baseline RMSE by **16%** on a sample
timestep — error that is pure projection artifact, and that the model would otherwise waste
capacity learning to reproduce.

Everything generic (UNet, EDM, EMA, loss, sigma sampler, Heun sampler, trainer) is
unchanged. The work is localized to `data/` + `configs/`.

## Data flow (`data/era5_hrrr_dataset.py`)

```
ERA5 vars @1° (lat/lon)  ──bilinear regrid──▶  HRRR 1059×1799 Lambert grid
    (scipy RegularGridInterpolator)                 │
                                                    ▼
                          z-score + [orog, lsm]  ──▶  x_cond   (n_era5 + n_static, H, W)
HRRR analysis @3 km (state) ── z-score ──────────▶  target    (n_prognostic + n_diagnostic, H, W)
                                                    │
                             EDM denoiser  D_θ(noisy_target, σ, x_cond)
```

## Running a training job

Training is submitted to Slurm and is **safe to stop and restart at any time** — it
checkpoints inside the epoch and auto-resumes. **[`bash/README.md`](bash/README.md)** is the
operator runbook (six numbered scripts: preflight → smoke → data prep → train/stop/status);
**[`slurm/README.md`](slurm/README.md)** documents the underlying Slurm harness. The short
version:

```bash
bash bash/00_preflight.sh     # read-only checks: env, data trees, index, stats, configs
sbatch slurm/smoke.sbatch     # verify the harness first — CPU `debug` node, ~3 min, no GPU
bash bash/03_train.sh         # start (or resume) — wraps sbatch slurm/train.sbatch
bash bash/04_stop.sh          # stop  (checkpoints, then exits; no deadline)
bash bash/03_train.sh         # resume from where it left off
```

Metrics go to **`metrics/train_metrics.jsonl`** — in-repo JSONL, no external service and
no W&B account (`logging=local` is the default backend). Plot curves with
`python scripts/plot_metrics.py`; opt back into W&B with `logging=wandb` (needs
`wandb login`). The file appends across stop/resume cycles; consumers keep the last
occurrence of a step.

Local smoke runs, no cluster needed:

```bash
# tiny CPU run (autocast('cuda') no-ops without a GPU). Keep the ckpt_dir override —
# without it the toy smoke checkpoint lands in the real checkpoints/ and the next real
# run auto-resumes from it and crashes on a state-dict mismatch.
python train.py data.use_dummy=true training.max_steps=3 training.n_epochs=1 \
  data.grid_height=24 data.grid_width=32 data.num_workers=0 \
  model.C_base=16 'model.n_res_blocks=[1,1]' model.num_groups=8 logging.enabled=false \
  ckpt_dir=checkpoints/_smoke

# full-resolution dummy (needs a GPU): drop the grid/model overrides
torchrun --nproc_per_node=<N> train.py data.use_dummy=true training.max_steps=10
```

Unit/smoke tests: `pytest tests/` — six files: `test_smoke.py` (dummy dataset + one
denoise/backward step), `test_model.py` (UNet/EDM/EMA shapes & formulas, checkpoint
weight loading), `test_attention.py` (window locality, shift, masking, identity-at-init),
`test_var_spec.py` (variable-spec parsing, precip transform, DictConfig regression),
`test_wind_rotation.py` (round-trip, Lambert angle numeric-vs-analytic),
`test_variable_weighting.py` (Ocampo schedule/cap/normalization).

## Status: real data is on disk; the model is not trained yet

Both datasets are present and the config is verified against them:

| | files | span | path |
|---|---|---|---|
| ERA5 1° | 16,072 | 2015–2025, 6-hourly | `eric/era5_1deg/` |
| HRRR 3 km | 16,019 | 2015–2025, 6-hourly | `moein/hrrr_work/hrrr_nc_v3_rebuilt/` |

Sanity-check one pair end to end (reads both files, regrids, scores, plots):

```bash
python scripts/compare_timestep.py --timestamp 2023-07-15T18
```

That prints the **bilinear-interpolation baseline** — the no-skill reference the downscaler
has to beat. For 2023-07-15T18: t2m RMSE 1.98 K, 10 m wind 1.41/1.43 m/s, and
`std(interp) < std(HRRR)` in every field, i.e. interpolation is too smooth. Closing that
fine-scale variance gap is the model's entire job.

**No usefully-trained checkpoint exists yet** (`ls checkpoints/` to verify — as of Jul 13 it
holds only step 90/98 from the input-pipeline verification run, ~1 epoch, and `bash/03_train.sh`
will RESUME from it; clear the directory to start the definitive run fresh).
`scripts/compare_timestep.py --ckpt <path>` adds a model column (it loads the EMA weights and
scores winds against rotated truth, same as the baseline rows), but a barely-trained checkpoint
scores far *worse* than the baseline — a near-random EDM sample is noise.

### Steps to train (verify state with commands, not prose)

`bash bash/02_build_index_and_stats.sh` builds both prerequisites idempotently. Any
"already built" sentence in a doc can rot — check disk:

1. **Timestamp index** (`data.index_path`): `wc -l valid_timestamps.txt` → 16,019 = built
   (ISO timestamps, one per line, where *both* an ERA5 and an HRRR file exist — a verified
   two-sided intersection of the trees; gitignored as derived).
2. **Normalization stats** (`data.stats_path`): `ls norm_stats/stats.npz`. The build
   streams 500 train-year samples through the real read+regrid path at ~4 s each —
   **~35 min total; it is not hung**. Rebuild with `FORCE_REBUILD=1` — mandatory after ANY
   change to the variable lists, since `train.py`'s startup asserts don't check the stats
   file and a stale one crashes at the first batch (after the Slurm allocation). Direct
   call needs Hydra's `+` prefix: `python scripts/precompute_stats.py +stats.n_samples=500`.
   Never run two builds concurrently (the final write is not atomic).
3. `bash bash/03_train.sh` (wraps `sbatch slurm/train.sbatch`).

### Train/val/test split & evaluation caveats

The split is **by year**: train 2015–2022, val 2023, test 2024–2025
(`configs/data/era5_hrrr.yaml`; `train.py` filters the index; stats use train years only).
Neighbouring 6-h samples are strongly autocorrelated, so a year-block split is the right
shape — but the effective sample count is far below the raw 11.6k train stamps.

- **Contamination warning:** 9 of the 12 xres extreme events (2019–2022) fall inside the
  train years. Scoring downscaled GenCast members on those events with this split means
  evaluating on HRRR fields the model trained on. Either move `train_end` to `2018-12-31`
  (mirrors the GenCast `<2019` checkpoints — all 12 events become out-of-sample) before
  the definitive run, or restrict downstream scoring to 2023+ events.
- **The bilinear baseline is not the bar for a single sample.** A calibrated generative
  model pays ~√2 in RMSE versus the conditional mean, so one sharp sample can legitimately
  lose to smooth interpolation on RMSE while being a far better realization. Judge with
  **ensemble-mean RMSE, CRPS, and power spectra** (`Downscaler.ensemble()` exists in
  `evaluation/downscale.py`; nothing calls it yet).
- **Perfect-prognosis caveat:** trained on ERA5 *analysis* conditioning, but the intended
  deployment conditions on week-2+ GenCast *forecast* members — smoother, biased, and
  12-hourly vs the 6-hourly training distribution. Standard practice, but say so next to
  any downstream skill number.
- **log_tp is bimodal:** 77% of HRRR precip pixels sit exactly at the log floor
  (ln(1e-5) ≈ −11.51; the 1e-5 is baked into the HRRR files at build time). Precip MSE is
  dominated by the wet/dry boundary, not by amounts — read precip skill accordingly.

### Conditioning: every target has a driver

`era5_variables` is now 9 channels, chosen so that nothing in the target set is being guessed:

| target | driver(s) |
|---|---|
| `t2m` | `2m_temperature` |
| `u10`, `v10` | `10m_u/v_component_of_wind` (same height) + `u/v_component_of_wind:850` (steering flow) |
| `log_sp` | `mean_sea_level_pressure` |
| `log_tp` | `total_precipitation_12hr` + `specific_humidity:850`/`:700` |

**Precipitation needed care.** ERA5 tp is a **12-hour** accumulation in **metres**; the HRRR
`log_tp` target is a **6-hour** accumulation of `ln(mm + 1e-5)` (see
`moein/hrrr_work/download_hrrr_v3.py:53`). So the driver is passed through the
`log_precip_m_to_mm` transform (`data/era5_hrrr_dataset.py::ERA5_TRANSFORMS`), which converts
m→mm and applies the *same* log, so a dry pixel maps to the identical floor (−11.51) on both
the input and output sides. Without it, the raw channel would sit on a scale 1000× off from
its target with a distribution that is 77% zeros — z-scoring that is near-useless.

The residual 12h→6h **window mismatch is real and is not fixed**: the HRRR 6 h window is
contained in the ERA5 12 h window, so the model must learn the disaggregation. Specific
humidity is the clean predictor here (no window problem), which is why both are included.

Transforms are applied to the **coarse** field before regridding, so the interpolant does not
smear a heavy-precip core across neighbouring dry cells.

Growing these lists means bumping `n_era5_cond` and `model.n_input_ch` together (see the
channel-count contract below); `train.py` asserts it at startup.

### Loss

MSE at heart — `(D_out - target)²` on the **denoised prediction** in normalized z-score space
— with λ(σ) noise weighting (Karras et al. 2022), spatial area weighting on the prognostic
channels, and adaptive per-variable (Ocampo) weights, normalized to mean 1 and updated on
validation epochs after a warmup. The spatial weights are **uniform**: the HRRR Lambert grid
is near-equal-area (true cell areas vary ~±5%), so the emulator's cos(lat) weighting — right
on a lat/lon grid — would have down-weighted the northern US by ~a third here. The stats key
is still named `cos_lat_weights` for compatibility.

The final reduction is a **per-channel mean**. It used to be `loss_prog.mean() +
loss_diag.mean()`, which handed each *group* half the gradient regardless of size — so the
lone precip channel took 50% of the loss and each of `t2m`/`u10`/`v10`/`log_sp` got 12.5%.
Now every channel gets 1/5. Set `training.diagnostic_loss_weight > 1` to prioritize precip
deliberately (`4.0` reproduces the old behaviour).

### Channel-count contract (asserted at startup by `train.py`)

```
n_era5_cond  == len(era5_variables)
n_prognostic == len(hrrr_prognostic_variables)
n_diagnostic == len(hrrr_diagnostic_variables)
model.n_output_ch == n_prognostic + n_diagnostic
model.n_input_ch  == n_output_ch + n_era5_cond + n_static
```

Defaults ship the **core-4 + precip** target set driven by 9 ERA5 channels
(`n_input_ch=16`, `n_output_ch=5`); grow the variable lists and these counts together.

### Performance note

`Era5HrrrDataset` regrids ERA5 per `__getitem__` through **cached bilinear weights**
(`CachedBilinearRegridder`: corner indices/weights are computed once — neither grid ever
changes — with equivalence to the old scipy path pinned by `tests/test_regrid.py`).
A warm sample costs ~0.9 s, dominated by the netCDF reads, so `data.num_workers` (12)
is what hides the input pipeline: 2 workers starved the GPUs to ~21 s/optimizer step
(measured Jul 13). If loading is ever the bottleneck again, the next lever is regridding
once **offline** and pointing the dataset at the cached fields.

## Layout

```
model/         UNet (+ Swin attention bottleneck) + EDM preconditioning + EMA
training/      trainer (DDP, interruptible), EDM loss, sigma sampler, variable weighting
data/          era5_hrrr_dataset (regrid+pair), normalization, wind_rotation, dummy, dataloader
evaluation/    sampler (EDM Heun ODE), downscale (single-shot inference + ensemble())
configs/       Hydra config tree (data/era5_hrrr.yaml is the main knob)
scripts/       precompute_stats.py, compare_timestep.py
bash/          operator runbook — six numbered scripts (bash/README.md)
slurm/         sbatch harness: train/smoke/stop (slurm/README.md)
docs/          variables.md + era5_variables.html (the 14 ERA5 vars actually in the files),
               model_reference.html + architecture.html (model spec / explainer)
tests/         test_smoke, test_model, test_attention, test_var_spec, test_wind_rotation
```

Cluster-ops scripts from the source (a3mega launchers, shard builders, rollout
movies, watchdogs) were **not** copied — they are HRRR-emulator/Derecho-specific.
Grab them from `moein/regional/hrrr_edm/scripts/` if needed.
