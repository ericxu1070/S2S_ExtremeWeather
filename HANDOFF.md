# FourCastNet 3 vs GenCast — 6 events, week-3 (21 d lead), 0.25° — Agent Handoff

**Goal:** compare **FourCastNet 3 (FCN3)** against **GenCast** across six CONUS events at a
**3-week (21-day) lead**, at **0.25°**, 24 members each. Deliverables: (1) per-event and
combined verification-week PDFs (FCN3 vs GenCast vs ERA5 truth), (2) per-event skill (CRPS,
ensemble-mean RMSE, bias, ACC, spread-error), (3) a **hardware-matched runtime comparison**
of the cost to generate a 24-member ensemble.

Everything runs on **this box** (a3mega Slurm / H100). Derecho is not involved.

---

## The six events (`fcn3/fevents.py` — single source of truth)

| event | family | peak | init | metric | GenCast cube |
|---|---|---|---|---|---|
| `PNW_HeatDome_2021` | heat | 2021-06-28 | 2021-06-07 | `t2m_anom` | **cached** (xres week3) |
| `HurricaneIan_2022` | hurricane | 2022-09-28 | 2022-09-07 | `u850_speed` | **cached** (xres week3) |
| `WinterStorm_Uri_2021` | cold | 2021-02-15 | 2021-01-25 | `t2m_anom` | **cached** (xres week3) |
| `p90_20231107` | p90 | 2023-11-07 | 2023-10-17 | `t2m_anom` | must run |
| `p90_20240802` | p90 | 2024-08-02 | 2024-07-12 | `t2m_anom` | must run |
| `p90_20251224` | p90 | 2025-12-24 | 2025-12-03 | `t2m_anom` | must run |

**Why week 3.** It is simultaneously xres `week3` (`init = peak − 21 d`, so all 12 xres
GenCast cubes already exist at this lead) **and** the frozen p90 lead
(`p90/cases_meta.json: lead_days = 21`, so the p90 cases keep their committed inits). One
lead across all six events means figure differences are event-type differences, not lead
differences — and the GenCast side of three events costs nothing.

**Why these p90 cases.** Chosen to span year, season and magnitude so the moderate-extreme
panel is not three samples of one regime: `p90_20240802` is marginal (+2.39 K, just over the
+2.3662 K p90 threshold) and the only warm-season case; `p90_20231107` is median (+3.10 K);
`p90_20251224` is the strongest in the whole 45-case set (+4.96 K, a 12-day event). All are
post-2019 → out of sample for GenCast's `<2019` checkpoints and for FCN3 (trained through
2017). Provenance back to `p90/cases.csv` is in `fevents.P90_SOURCE_CASE`.

**Hurricane note.** Ian is scored on `u850_speed` (the xres convention for hurricanes), not
T2m, so the FCN3 cube for that event also carries `u850`/`v850`. Uri is in
`xres.xconfig.COLD_EVENTS`, so its extreme tail flips to the cold side automatically.

---

## Runbook

```bash
bash fcn3/run_test.sh plan            # event table + what is already built
bash fcn3/run_test.sh prep            # BOTH prep stages, login node (safe to re-run)
bash fcn3/run_test.sh submit          # both infer jobs
bash fcn3/run_test.sh compare         # figures + scores, login node, moe
```

Stage by stage:

| stage | where | env | what |
|---|---|---|---|
| `prep-gencast` | login node | `moe` | ERA5 truth + GenCast init frames for the 3 p90 events |
| `prep-fcn3` | login node | `fcn3` | global 72-var ERA5 ICs for all 6 events + checkpoint cache |
| `submit-fcn3` | `sbatch slurm/fcn3_infer.slurm` | `fcn3` | 8 shards × 3 members × 6 events, ~30 min |
| `submit-gencast` | `sbatch slurm/xres_pool_0p25_week3_p90.slurm` | `moe` | 3 p90 events × 24 members, ~3.2 h |
| `compare` | login node | `moe` | `fcn3/compare_fcn3_gencast.py` |

**Project rule (CLAUDE.md): GPU nodes are for GPU compute only.** Both infer scripts verify
their inputs exist and *refuse to start* otherwise — they must never download or build data.

## How GenCast gets run on non-xres events

`xres/xconfig.py` gained one additive hook, `XRES_EXTRA_EVENTS`, that merges extra
`name=peak:metric` entries into `XRES_EVENTS` before the usual `XRES_EVENTS_SEL` /
`XRES_MAX_EVENTS` filters. Unset → exactly the original 12 events, so every prior run is
bit-identical. The p90 spec comes from one place:

```bash
XRES_EXTRA_EVENTS="$(python fcn3/fevents.py --spec)" \
XRES_EVENTS_SEL=p90_20231107,p90_20240802,p90_20251224 \
  python run_xres.py --stage prep --res 0p25 --weeks 3
```

After injection the ordinary xres pipeline builds their truth, init frames and cubes into
the same tree as the other nine. The 12 original cubes are never touched.

## Cube contract (why the two models are comparable)

`fcn3/run_fcn3.py` writes cubes using **GenCast's** variable names — `2m_temperature`, and
`u_component_of_wind`/`v_component_of_wind` with a `level` dim — on the **same CONUS grid**
(lat 24–50 ascending × lon 235–294, 105×237 at 0.25°, coords cast to float32 to match the
climatology exactly). So `xres.xmetrics.forecast_field` scores both models through one code
path; the only difference in a figure is the forecast, not the post-processing.

Verified offline before any GPU time: a synthetic FCN3 zarr assembles into a cube whose
grid is `array_equal` to the GenCast cube's, yields 25 six-hourly frames in the
verification week, and produces finite `t2m_anom` and `u850_speed` fields that align with
the ERA5 truth files.

## Runtime measurement (both models on the same hardware)

`xres/xinference.py` is not instrumented and must not be modified, so GenCast timing comes
from launcher-level node wall clocks — both from **one a3mega 8×H100 node**, same lead,
same 24 members, same code path, hence directly comparable to FCN3:

- **Cached xres events:** Slurm job **223** (`logs/xres_0p25pool_wk3-223.out`), 06:57:36 →
  19:38:23 = **760.8 min for 12 events ⇒ ~63.4 min/event**; worker logs show **~28 s per
  12-hourly step** (~19.6 min/member) plus a one-time ~5 min XLA compile.
- **The 3 p90 events:** `slurm/xres_pool_0p25_week3_p90.slurm` writes
  `runs/fcn3/week3/timing/gencast_pool_timing.json` (`complete: true` only when the job
  actually rolled every event — a resubmit that skips cached cubes under-reports the cost).
- **FCN3:** self-timed per (event, shard); `--stage assemble` reduces to a per-event node
  wall plus summed GPU-seconds.

The Derecho A100-40GB number (~80 s/step, ~2.9× slower per step than H100) is kept only as
a labelled cross-machine reference bar.

---

## Derecho smoke test (validate the FCN3 path before spending an 8xH100 node)

**Why:** as of 2026-07-23 FCN3 has **never completed a model build or a single forecast
step** anywhere. Three attempts died for unrelated reasons (697 cancelled: no CUDA kernel;
700: CPU thread thrash, GPUs idle 80+ min; 707: cancelled while queued). Everything about
the driver below is therefore *unproven at runtime*, and a3mega nodes are scarce. One
A100 smoke on Derecho de-risks the machine-independent half cheaply.

**What a Derecho smoke DOES prove** (all machine-independent, none of it ever executed):
the event registry and init arithmetic, that FCN3 constructs at all with the pinned
torch-harmonics, the rollout, the multi-variable path (`u850`/`v850` for the hurricane),
cube assembly, and the cube schema. It also yields the first real **model-build time**,
which decides whether the build cache below is worth writing.

**What it does NOT prove:** the a3mega-specific failures. The DISCO kernel is compiled per
GPU arch (Derecho A100 = **8.0**, a3mega H100 = **9.0**; not portable), and the thread
thrash was a 208-core/8-shard artefact that cannot reproduce on a 1-GPU PBS job. An
a3mega smoke is still required before the full run.

### Steps (run in a Derecho session)

```bash
cd /glade/derecho/scratch/exu/S2S_ExtremeWeather
git pull                                        # brings fcn3/fevents.py + the new driver

# 1. Rebuild torch-harmonics WITH the CUDA kernel for A100 (arch 8.0).
#    Use the recipe in "FCN3 environment" below but set TORCH_CUDA_ARCH_LIST=8.0, and
#    point CUDA_HOME at the Derecho fcn3 env. All four traps apply (nvcc version match,
#    PyPI wheel silently overriding FORCE_CUDA_EXTENSION, no 0.9.1 sdist, git-lfs).
#    Derecho extra gotcha: nvcc rejects the default Intel icpx host compiler
#    ("icpx 0.0.0 < 6.0.0") -> module load gcc/12.5.0 and set CC=gcc CXX=g++.
#    Verify before submitting anything:
/glade/derecho/scratch/exu/conda-envs/fcn3/bin/python -c \
  "import earth2studio.models.px.fcn3 as m; print('bf16:', m._cuda_extension_available)"

# 2. Stage the week-3 IC (CPU+internet; login node OOM-kills prep, so use the queue).
#    NOTE the cached Derecho IC is the OLD week-2 init (2021-06-14); week-3 needs
#    2021-06-07. Nothing is reused.
qsub -v FCN3_EVENTS=PNW_HeatDome_2021 pbs/fcn3_prep.pbs

# 3. Smoke: 4 members x 4 steps, one event, single A100.
qsub -v FCN3_EVENTS=PNW_HeatDome_2021,FCN3_MEMBERS=4,FCN3_NSTEPS=4 pbs/fcn3_infer.pbs

# 4. Optional second smoke -- the ONLY one that exercises u850/v850 + the level dim:
qsub -v FCN3_EVENTS=HurricaneIan_2022,FCN3_MEMBERS=2,FCN3_NSTEPS=4 pbs/fcn3_prep.pbs
qsub -v FCN3_EVENTS=HurricaneIan_2022,FCN3_MEMBERS=2,FCN3_NSTEPS=4 pbs/fcn3_infer.pbs
```

### What to report back

1. `[check] ... DISCO CUDA extension: PRESENT -> FCN3 runs bf16` (else the run is invalid).
2. **`model ready in N min`** — the build number. If large, implement the build cache
   (see "Build-once cache" below) and restore 8 shards on a3mega.
3. Seconds per rollout step, and `peak_gpu_mem_gb` from the timing JSON — the 40 GB
   headroom figure decides whether `batch_size>1` is safe on an 80 GB H100.
4. Whether the cube passes the schema contract:

```python
import xarray as xr, pandas as pd, sys; sys.path.insert(0, ".")
from fcn3 import fevents as F
from xres import xmetrics as XM          # needs my-env, not the fcn3 env
ev = F.EVENTS["PNW_HeatDome_2021"]
c = xr.open_dataset(F.cube_path(ev))
print(dict(c.sizes), list(c.data_vars))   # expect 2m_temperature(member,time,lat,lon), 105x237
print(float(c["2m_temperature"].mean()))  # sane T2m in Kelvin (~280-300)
```

> A **smoke cube is truncated** (`FCN3_NSTEPS=4` stops ~20 d before the peak), so
> `XM.forecast_field` will find no frames in the verification week. That is expected --
> do not read it as a failure. Only a full-length run (84 steps) produces a scorable cube.

> **`compare` will not work on Derecho.** The GenCast week-3 cubes live on a3mega (jobs
> 218/219/223/224 produced week3/4 locally); Derecho has week-2 only. Run
> `fcn3/compare_fcn3_gencast.py` on a3mega.

### Build-once cache (only if the build time justifies it)

The build computes **geometry tables, not weights** — DISCO sparse convolution tensors,
Legendre polynomials, quadrature weights — identical for every event, member and run.
They are registered `persistent=False`, so the 2.84 GB checkpoint does **not** carry them
and they are recomputed on every construction; the `@lru_cache` in torch-harmonics is
in-memory and dies with the process. Verified 2026-07-23 that `torch.save(model)` on the
whole module (not `state_dict`) *does* carry them: buffers identical after reload and the
forward output bit-identical. So: build once, pickle to NFS, have workers `torch.load`.
Needs a version key in the filename + rebuild fallback (pickles break across torch /
torch-harmonics upgrades), and `weights_only=False` on load, so only ever load our own file.
GenCast already gets this for free via `JAX_COMPILATION_CACHE_DIR` — adding it to FCN3
removes a handicap rather than granting an advantage.

## Outputs

```
runs/fcn3/week3/ic/<event>_ic.nc              global 72-var ERA5 IC (299 MB each)
runs/fcn3/week3/cache/<event>_cube.nc         merged FCN3 cube, GenCast schema
runs/fcn3/week3/shards/                       per-shard cubes/timings (intermediate)
runs/fcn3/week3/timing/<event>_timing.json    per-event node wall + GPU-seconds
runs/fcn3/week3/scores/fcn3_vs_gencast_scores.csv
runs/xres/0p25/week3/cache/p90_*_cube.nc      GenCast cubes for the 3 p90 events
runs/observations/p90_*_verif_t2m_anom.nc     ERA5 truth for the 3 p90 events
figures/fcn3/week3/fcn3_vs_gencast_pdfs.png   6-panel combined PDFs
figures/fcn3/week3/<event>_pdf.png            per-event PDFs
figures/fcn3/week3/fcn3_vs_gencast_skill.png  CRPS + ens-mean RMSE per event
figures/fcn3/week3/fcn3_vs_gencast_runtime.png
```

`compare_fcn3_gencast.py` is incremental: events whose cube or truth is missing are skipped
with a warning and their panel reads "not built yet", so it is useful before everything is
finished.

---

## FCN3 environment (`fcn3` conda env on THIS box)

`earth2studio 0.17.0a0`, `torch 2.11.0+cu128`, `torch-harmonics 0.8.1` (source-built with
CUDA kernels), CUDA 12.8 toolkit in-env, Python 3.11.

### The DISCO CUDA kernel — read this before touching the env

**Symptom if missing:** earth2studio logs only a mild `torch-harmonics disco CUDA extension
is not available / FCN3 run on GPU/CUDA will be slower` warning and *keeps going*. Recent
torch-harmonics falls back rather than crashing (the older pinned fork did crash with
`NotImplementedError: Could not run 'disco_kernels::forward'`), so **nothing fails loudly**.

**Why that is fatal here anyway:** `earth2studio/models/px/fcn3.py:382` runs
`torch.autocast(..., bfloat16 if _cuda_extension_available else float32)`. No kernel ⇒ FCN3
silently runs **fp32 on an unoptimized path**, which destroys (a) the bf16 precision match
to GenCast's 0.25° run and (b) the runtime comparison, which is the headline result.

**Guard:** `fcn3/run_fcn3.py::check_disco_kernel()` runs in BOTH `--stage prep` (login node)
and `--stage infer`, and hard-fails with the rebuild recipe. Override deliberately with
`FCN3_ALLOW_FP32=1`. Incident 2026-07-23: job 697 was cancelled 4 min in for exactly this.

**Rebuild recipe (login node, ~8 min compile).** Four traps, all hit on 2026-07-23:

1. `nvcc` must match torch's CUDA (12.8). The system only has CUDA **13.0**, which will not
   build against a cu128 torch, and the pip package `nvidia-cuda-nvcc-cu12` ships only
   `ptxas`, **not** `nvcc`. Fix: `conda install -n fcn3 -c nvidia cuda-toolkit=12.8`.
2. `pip install torch-harmonics==0.9.1` silently installs a **prebuilt manylinux wheel**, so
   `FORCE_CUDA_EXTENSION=1` never applies. You must force a source build.
3. There is **no sdist for 0.9.1** on PyPI (sdists stop at 0.8.0), so `--no-binary` fails.
4. Installing straight from git fails: the repo uses **git-lfs**, which is not installed
   (`git-lfs filter-process: 1: git-lfs: not found` → `Clone succeeded, but checkout failed`).

Working sequence — clone the commit earth2studio pins, bypassing LFS, and build locally:

```bash
git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false \
    clone https://github.com/NVIDIA/torch-harmonics.git /tmp/th
git -C /tmp/th -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false \
    checkout -f a632ca748a12bd9f74dbc1e00653317810991f74      # == v0.8.1

export CUDA_HOME=/home/ubuntu/miniconda3/envs/fcn3
export PATH="$CUDA_HOME/bin:$PATH"
SITE=$($CUDA_HOME/bin/python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(echo "$SITE"/nvidia/*/lib | tr ' ' ':'):$CUDA_HOME/lib"
export FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST=9.0 CC=gcc CXX=g++ MAX_JOBS=8
$CUDA_HOME/bin/pip install --no-build-isolation --no-cache-dir --force-reinstall --no-deps /tmp/th
```

Sanity checks: the produced wheel is **~4.8 MB** (the pure-python one is 540 kB), and

```bash
python -c "from disco_helpers import cuda_kernels_is_available; print(cuda_kernels_is_available())"
python -c "import earth2studio.models.px.fcn3 as m; print(m._cuda_extension_available)"
```

must both print `True`. Note this pins torch-harmonics to **0.8.1**, which is what
earth2studio's `[tool.uv.sources]` pins (`>=0.8.0` + that commit) — not a regression.

### Other env notes

- makani + the torch-harmonics fork are git-pinned in `[tool.uv.sources]`; plain pip cannot
  resolve `makani` → install both from git FIRST, then `earth2studio[fcn3]`.
- `LD_LIBRARY_PATH` must include the env's `site-packages/nvidia/*/lib` (both slurm scripts
  and `run_test.sh` do this).
- Never build the model on the login node: the 710M-param CPU build OOM-kills it (15 GB
  total). `--stage prep` only calls `package.get()` to fetch files.
- The weights live at **`training_checkpoints/best_ckpt_mp0.tar`** in the HF repo, not at the
  repo root. Asking for the bare name raises `FileNotFoundError`; the old prep caught that
  and printed `skip`, exiting 0 having fetched no weights. `stage_prep` now tries both paths
  and treats missing weights as a hard failure.

## FCN3 model facts

- Global **0.25°, 721×1440**, **6-hourly**, 72 variables (7 surface + 13 levels × {u,v,z,t,q}).
  Input is a **single init frame** (unlike GenCast's 2-frame init).
- **Probabilistic with INTERNAL stochastic noise** — do NOT perturb the IC; replicate it
  across the ensemble dim and pass `earth2studio.perturbation.Zero()`. Seed via
  `model.set_rng(seed=…, reset=True)`; this driver uses a distinct seed per (event, shard).
- Checkpoint `hf://nvidia/fourcastnet3@76ef0c60…`, Apache-2.0, not gated, ~2.85 GB.
- `earth2studio/recipes/s2s/` does NOT wire FCN3; the rollout is built on the generic
  `earth2studio.run.ensemble(...)` API. `output_coords={"variable": [...]}` subsets the
  saved variables (one zarr array per variable name) — verified against `run.py`.

## Known state / gotchas

- `runs/fcn3/PNW_HeatDome_2021/` is the **abandoned week-2 smoke run** (1 member × 4 steps).
  A week-2 IC is the wrong init for week 3; the new tree is `runs/fcn3/week3/`. Harmless.
- The login node has only **15 GB RAM**. Do not run the GenCast prep (706 MB init files,
  multi-GB peak) and the FCN3 prep concurrently.
- FCN3 shard zarrs are global (721×1440) and ~1 GB per variable per shard-event; they are
  deleted after each shard cube is written unless `--keep-zarr`.
- Ensemble size is 24 to match the cached GenCast cubes. `compare` subsamples both models to
  the smaller available member count so CRPS and the PDF tails stay comparable.

---

*Last updated: Thu Jul 23 2026 — scope extended from the single PNW Heat Dome week-2 test to
six events at week-3 (added Hurricane Ian, Winter Storm Uri and three p90 cases).*
