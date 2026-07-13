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
checkpoints inside the epoch and auto-resumes. See **[`slurm/README.md`](slurm/README.md)**
for the runbook; the whole of it is:

```bash
sbatch slurm/smoke.sbatch     # verify the harness first — CPU `debug` node, ~3 min, no GPU
sbatch slurm/train.sbatch     # start (queues until a node frees)
bash   slurm/stop.sh          # stop  (checkpoints, then exits; no deadline)
sbatch slurm/train.sbatch     # resume from where it left off
```

Local smoke runs, no cluster needed:

```bash
# tiny CPU run (autocast('cuda') no-ops without a GPU)
python train.py data.use_dummy=true training.max_steps=3 training.n_epochs=1 \
  data.grid_height=24 data.grid_width=32 data.num_workers=0 \
  model.C_base=16 'model.n_res_blocks=[1,1]' model.num_groups=8 logging.enabled=false

# full-resolution dummy (needs a GPU): drop the grid/model overrides
torchrun --nproc_per_node=<N> train.py data.use_dummy=true training.max_steps=10
```

Unit/smoke tests: `pytest tests/` (`test_smoke.py` = dummy dataset + one denoise
step; `test_model.py` = UNet/EDM/EMA shapes & formulas).

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

**No trained checkpoint exists yet.** `scripts/compare_timestep.py --ckpt <path>` will add a
model column, but an untrained checkpoint scores far *worse* than the baseline — a
random-weight EDM sample is noise.

### Remaining steps to train

1. **Build the timestamp index** (`data.index_path`): ISO timestamps, one per line, for
   which *both* an ERA5 file and an HRRR file exist. Intersect the two trees on disk.
2. **Compute normalization stats**: `python scripts/precompute_stats.py`
   → writes `data.stats_path` (nothing is there yet). ~1 min for 200 samples.
3. `sbatch slurm/train.sbatch`

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
— with λ(σ) noise weighting (Karras et al. 2022), cos(lat) area weighting on the prognostic
channels, and adaptive per-variable (Ocampo) weights.

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

`Era5HrrrDataset` regrids ERA5 **per `__getitem__`**. For production, regrid once
**offline** (mirrors how the source project precomputes HRRR shards) and point the
dataset at the cached fields — the coarse→fine interpolation is the same either way.

## Layout

```
model/         UNet + EDM preconditioning + EMA        (verbatim port)
training/      trainer (DDP), EDM loss, sigma sampler, variable weighting
data/          era5_hrrr_dataset (regrid+pair), normalization, dummy, dataloader
evaluation/    sampler (EDM Heun ODE), downscale (single-shot inference)
configs/       Hydra config tree (data/era5_hrrr.yaml is the main knob)
scripts/       precompute_stats.py
tests/         test_smoke.py, test_model.py
```

Cluster-ops scripts from the source (a3mega launchers, shard builders, rollout
movies, watchdogs) were **not** copied — they are HRRR-emulator/Derecho-specific.
Grab them from `moein/regional/hrrr_edm/scripts/` if needed.
