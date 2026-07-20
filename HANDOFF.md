# FourCastNet 3 vs GenCast — PNW Heat Dome, 2-week lead — Agent Handoff

**Goal:** Compare **FourCastNet 3 (FCN3)** against **GenCast** on the **PNW Heat Dome 2021**
event at **2-week (14-day) lead**, at **0.25°**. Produce (1) T2m-anomaly PDF plots
(FCN3 vs GenCast vs ERA5 truth) and (2) a comparison of the **wall time to generate 12
ensemble members** for each model. GenCast results are reused from cache; FCN3 is new work.

> **Where to run FCN3:** the **H100 (80 GB) nodes**. FCN3's checkpoint badge says **80 GB
> GPU** — Derecho has **only A100-40GB** (`pbsnodes` → `a100_40gb`, no 80GB / no H100), so
> the full FCN3 run is being moved to the **a3mega H100 80GB** cluster (Slurm). GenCast
> stays as-is (already run on Derecho A100-40GB; cache is synced to a3mega per prior notes).

---

## STATUS (Jul 20 2026, later) — FCN3 env BUILT + prep DONE on Derecho; run moved OFF Derecho

Per user: **do not run FCN3 on Derecho** (A100-40GB). Move the run to the a3mega **H100
80GB** cluster. Derecho was used only to build the env and pre-stage data; all Derecho jobs
are cancelled and none are queued.

**What is already done and reusable (mostly machine-portable):**
- `fcn3/run_fcn3.py` (prep/infer driver), `fcn3/compare_fcn3_gencast.py` (my-env plots),
  `pbs/fcn3_infer.pbs` (Derecho; rewrite as an sbatch on a3mega).
- **prep DONE:** global ERA5 IC (`runs/fcn3/PNW_HeatDome_2021/PNW_HeatDome_2021_ic.nc`,
  299 MB, 72 vars 721×1440) + FCN3 checkpoint (2.7 GB `best_ckpt_mp0.tar` + aux) cached in
  `runs/fcn3/.cache/e2s/fcn3/`. NOTE: these are on Derecho scratch — re-fetch on a3mega
  (login node, has internet) via `run_fcn3.py --stage prep`, or copy the cache over.
- Conda env `/glade/derecho/scratch/exu/conda-envs/fcn3` is Derecho-specific — **rebuild on
  a3mega** (see recipe below; the `[[fcn3-env-recipe]]` memory has the full gotcha list).

**Verified findings (carry to H100):**
- Env recipe: Python 3.11; makani + torch-harmonics fork are **git-pinned in
  `[tool.uv.sources]`** (plain pip can't resolve `makani` → install both from git FIRST, then
  `earth2studio[fcn3]`). On Derecho A100 torch had to be **cu12**: `torch==2.11.0+cu128` from
  `--index-url https://download.pytorch.org/whl/cu128` (default PyPI torch 2.12.1 is cu13,
  won't load; cu128 index tops out at 2.11.0). `torchvision==0.26.0+cu128` to match.
  `LD_LIBRARY_PATH` must include the env's `site-packages/nvidia/*/lib`. On H100 you can
  likely use the default (cu13) torch 2.12.1 — the cu12 downgrade was only for Derecho's
  driver.
- **The torch-harmonics CUDA DISCO kernel is MANDATORY on GPU** — not the "slower fp32
  fallback" the loader warning implies. Without it FCN3 crashes mid-forward:
  `NotImplementedError: Could not run 'disco_kernels::forward' with the 'CUDA' backend`.
  Build with `FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST=... module load cuda; pip install
  --no-build-isolation --no-cache-dir --force-reinstall --no-deps "torch-harmonics @
  git+...@a632ca7"`. **Gotcha:** nvcc rejects Derecho's default Intel `icpx` host compiler
  (`icpx 0.0.0 < 6.0.0`); set `CC=gcc CXX=g++` (module `gcc/12.5.0`). On a3mega, use that
  cluster's gcc + `TORCH_CUDA_ARCH_LIST=9.0` (H100).
- **Precision:** FCN3 runs **bf16** iff the CUDA kernel is present (`torch.autocast(bfloat16
  if _cuda_extension_available else float32)`), else fp32 — so bf16 is both the native path
  and the fair match to GenCast's bf16 0.25° run.
- **Memory:** the Derecho smoke test (job 6824712, 1 member/4 steps, batch_size=1) loaded the
  model and *started* the forward on A100-40GB with **no OOM** (it crashed only on the missing
  CUDA kernel). So a single member's memory looked fine on 40GB — model parallelism likely
  unnecessary; on H100-80GB there is ample headroom to raise `batch_size` (data-parallel over
  members). earth2studio's FCN3/`run.ensemble` is single-device only; makani has the
  model-parallel machinery (`--h_parallel_size`/`--w_parallel_size`) but it's not exposed
  through earth2studio.

**Remaining on H100:** rebuild env → prep (or copy cache) → build CUDA kernel (arch 9.0) →
run `infer` (12 members, 56 steps; raise batch_size on 80GB) → `compare_fcn3_gencast.py`.

---

## (original plan) GenCast side DONE (cached); FCN3 side

### ✅ Done — GenCast (reuse from cache, do NOT rerun)

The 0.25° week-2 GenCast forecast for PNW Heat Dome already exists:

- Cube: `runs/xres/0p25/week2/cache/PNW_HeatDome_2021_cube.nc`
  (24 members, 28×12-hourly steps, all vars/levels; init 2021-06-14 00:00, peak 2021-06-28)
- Per-member week-mean T2m anomaly: `runs/xres/0p25/week2/verif/PNW_HeatDome_2021_t2m_anom_members.nc`
  (24 members, 105×237 on the CONUS box)
- ERA5 0.25° truth: `runs/observations/PNW_HeatDome_2021_verif_t2m_anom.nc`
- **Subsample GenCast 24 → 12 members** to match FCN3 for the comparison.

### GenCast runtime (pulled from logs — no need to re-time)

Source: `logs/6665322[0].desched1.OU` (Derecho A100-40GB, node deg0050, Jul 8; job
`6665322` = 24-subjob member array, subjob N = member N of all 12 events). Timestamped
rollout chunks for PNW member 0:

- **~80 s / 12-hourly step**, 28 steps/member ⇒ **~37 min/member** at steady state.
- One-time XLA compile **~5 min** (first member only). PNW member 0 incl. compile:
  00:56:59 → ~01:44 ≈ **47 min**.
- **12 members on one A100-40GB ≈ 7.5 h** (5 min compile + 12 × 37 min). This is the
  head-to-head "time to generate 12 members" number for GenCast. (Production parallelized
  members across the 24-subjob array, so per-subjob wall clock was ~37–47 min, but the
  single-GPU cost for 12 members is ~7.5 h.)

### ⬜ Not started — FCN3

Nothing built yet. New conda env required (existing `my-env` is Python 3.10; earth2studio
needs ≥3.11). See plan below.

---

## FCN3 facts (verified against NVIDIA source / HF, Jul 20 2026)

- **Code:** inference in **`NVIDIA/earth2studio`** (`earth2studio.models.px.FCN3`,
  `models/px/fcn3.py`); **`NVIDIA/makani`** is a hard runtime dep (FCN3 imports
  `makani.models.model_package.load_model_package`). Paper: arXiv **2507.12144**.
- **Checkpoint:** `hf://nvidia/fourcastnet3@76ef0c60237e458b33196ba027134e27f3fc4538`
  — **NOT gated**, Apache-2.0, ~2.85 GB. (NGC mirror `nvidia/earth-2/fourcastnet3` needs
  an NGC key; use HF to avoid that.) 710,867,670 params.
- **Install:** `pip install "earth2studio[fcn3] @ git+https://github.com/NVIDIA/earth2studio.git"`.
  Pulls makani (pinned commit `b38fcb27…`), `torch-harmonics>=0.8.0` (pinned NVIDIA fork
  commit `a632ca74…`), ruamel.yaml, scipy, moviepy. Needs **Python ≥3.11, torch ≥2.5**
  (examples pin torch==2.11.0). Tested GPUs: A100 / H100 / L40S, Linux only.
- **torch-harmonics CUDA gotcha:** the DISCO kernel must be compiled
  (`export FORCE_CUDA_EXTENSION=1; pip install --no-build-isolation torch-harmonics`), else
  inference falls back to a slow fp32 path (code emits a warning and drops bf16).
- **Model:** global **0.25°, 721×1440** equirectangular (lat 90→-90, lon 0→360 endpoint
  excluded), **6-hourly** steps, 72 variables
  (surface: u10m,v10m,u100m,v100m,t2m,msl,tcwv; + 13 levels ×{u,v,z,t,q}). Levels:
  50,100,150,200,250,300,400,500,600,700,850,925,1000 hPa.
  Input tensor `(batch, time, lead_time=1, 72, 721, 1440)` — **single init frame** (unlike
  GenCast's 2-frame init).
- **Ensemble mechanism:** FCN3 is **probabilistic with INTERNAL stochastic noise** (latent
  noise diffusion on the sphere, per batch/ensemble index, reset at rollout start). Do NOT
  perturb the IC — replicate the same IC across the ensemble dim and pass
  **`earth2studio.perturbation.Zero()`**. Seed via `FCN3(core_model, seed=…)` /
  `model.set_rng(seed=…, reset=True)`. Paper evaluates with 50 members.
- **Perf claim:** 60-day 0.25° 6-hourly global rollout in **<4 min on one H100** (NVIDIA).
  So 12 members × 14 days should be minutes, not hours — the expected punchline vs
  GenCast's ~7.5 h.
- **S2S recipe caveat:** `earth2studio/recipes/s2s/` only wires DLESyM + HENS-SFNO, NOT
  FCN3. Build the FCN3 rollout on the generic `earth2studio.run.ensemble(...)` API.

### Minimal FCN3 ensemble call (self-assembled — validate against `fcn3.py`/`run.py`)

```python
from earth2studio.data import ARCO          # or ERA5-backed source; 0.25° global IC
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN3
from earth2studio.perturbation import Zero
from earth2studio.run import ensemble

package = FCN3.load_default_package()        # hf://nvidia/fourcastnet3@76ef0c6…
model   = FCN3.load_model(package)
data    = ARCO()
io = ZarrBackend(file_name="outputs/fcn3_pnw.zarr",
                 chunks={"ensemble": 1, "time": 1, "lead_time": 1},
                 backend_kwargs={"overwrite": True})
io = ensemble(["2021-06-14T00:00:00"], nsteps=56, nensemble=12,
              model, data, io, Zero(),
              batch_size=12)   # batch_size: 12 on H100-80GB; drop to 1 if OOM
```

`nsteps=56` = 14 days @ 6h; `nensemble=12`; `Zero()` because stochasticity is internal.

---

## Matched hyperparameters (FCN3 ← GenCast week-2 cache)

| | value |
|---|---|
| Init | 2021-06-14 00:00 UTC (ERA5 0.25° **global** IC — FCN3 is a global model) |
| Lead / rollout | 14 days (FCN3 `nsteps=56` @ 6h; GenCast 28 @ 12h — same window) |
| Verification week | 7 days ending at peak **2021-06-28 00:00** |
| Ensemble | **12 members** (subsample GenCast 24→12 to match) |
| Grid / region | native **0.25°**, scored on CONUS box **24–50°N, 235–294°E** |
| Metric | week-mean **T2m anomaly** vs same ERA5 climatology |
| Truth | ERA5 0.25° (`runs/observations/PNW_HeatDome_2021_verif_t2m_anom.nc`) |

> GenCast reuses its cropped/cached CONUS inputs, but **FCN3 needs the full global ERA5 IC**
> (721×1440) — fetch it fresh via earth2studio's data source, do NOT reuse GenCast's
> CONUS-cropped `runs/xres/0p25/week2/inputs/*.nc`.

---

## Plan (remaining work — all FCN3 side)

1. **Env** (login node, has internet): new conda env (Py 3.11),
   `pip install "earth2studio[fcn3] @ git+…"`, compile torch-harmonics CUDA ext
   (`FORCE_CUDA_EXTENSION=1`), pre-download the HF checkpoint.
2. **Smoke test** (short GPU job): load FCN3, run 1–2 steps, confirm it fits and produces
   sane output.
3. **Driver** `fcn3/run_fcn3.py`: fetch 2021-06-14 00:00 global ERA5 IC, roll 12 members ×
   56 steps, save T2m over the verification week into a cube/verif matching GenCast's layout
   (CONUS box, week-mean T2m anomaly vs the same ERA5 climatology); **record wall time** for
   the 12-member generation.
4. **Submit** the FCN3 job (Slurm on a3mega H100).
5. **Comparison + plots** (CPU): matched week-mean T2m-anomaly PDFs (pooled members, log-y —
   mirror `xres/xcombined.py` `PDF_histogram`) for FCN3 vs GenCast-12 vs ERA5 truth, plus a
   runtime bar chart (FCN3 vs GenCast, 12 members).

## Key paths

- GenCast cube / verif / truth: see "Done" section above.
- PDF convention to mirror: `xres/xcombined.py` (`PDF_histogram` — pooled all-members,
  100 bins over mean±4σ, log-y), and `xres/xleadpdfs.py`.
- GenCast runtime source log: `logs/6665322[0].desched1.OU`.
- New FCN3 code should live in a **new `fcn3/` dir** — do not touch `gencast_s2s/` or `xres/`.

---

*Last updated: Mon Jul 20 2026. This handoff was reset to track the FCN3-vs-GenCast PNW
Heat Dome comparison. The prior GenCast xres (0.25° vs 1.0°, weeks 2/3/4) handoff content
was cleared — that experiment is complete (all cubes/figures built) and its history lives in
git and in `CLAUDE.md`.*
