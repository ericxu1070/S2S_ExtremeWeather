# FourCastNet 3 vs GenCast — 6 events, week-3 (21 d lead), 0.25° — Agent Handoff

**Goal:** compare **FourCastNet 3 (FCN3)** against **GenCast** across six CONUS events at a
**3-week (21-day) lead**, at **0.25°**, 24 members each. Deliverables: (1) per-event and
combined verification-week PDFs (FCN3 vs GenCast vs ERA5 truth), (2) per-event skill (CRPS,
ensemble-mean RMSE, bias, ACC, spread-error), (3) a **hardware-matched runtime comparison**
of the cost to generate a 24-member ensemble.

Everything **runs** on **this box** (a3mega Slurm / H100). Derecho runs none of it, but as
of 2026-08-17 it holds a **full copy of the results** for offline analysis — see "Artifact
sync to Derecho" below.

> **This experiment is complete.** Live work has moved to the **`aires` branch** (AI+RES:
> GenCast walkers, FCN3 score), which reuses this stack's cubes, ICs and event registry.
> Its state lives in **`aires/HANDOFF.md`** and its design in **`aires.md`** — read those,
> not this file, for anything AI+RES. The section "AI+RES — what a Derecho session can do"
> below is the only part of it that belongs here, because it concerns this file's sync.

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

> **SUPERSEDED (2026-08-17).** This section is kept for history. FCN3 has since
> **completed the full 6-event week-3 run on a3mega** — 6 cubes in
> `runs/fcn3/week3/cache/`, per-event timing, and `fcn3_vs_gencast_scores.csv`. The
> de-risking smoke below is no longer a prerequisite for anything.

**Why (as written 2026-07-23):** at that time FCN3 had **never completed a model build or a
single forecast step** anywhere. Three attempts died for unrelated reasons (697 cancelled:
no CUDA kernel; 700: CPU thread thrash, GPUs idle 80+ min; 707: cancelled while queued).
Everything about the driver below was therefore *unproven at runtime*, and a3mega nodes are
scarce. One A100 smoke on Derecho de-risked the machine-independent half cheaply.

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

### Smoke result (2026-07-23, job 6857923 on Derecho A100-40GB)

- **DISCO CUDA kernel:** rebuilt for A100 arch 8.0 and confirmed `PRESENT -> FCN3 runs
  bf16` in both prep and infer. Recipe worked with the Derecho `cuda/12.9.0` module for
  nvcc (minor skew vs cu128 torch is fine; no conda cuda-toolkit needed) +
  `module load gcc/12.5.0`, `CC=gcc CXX=g++`, `TORCH_CUDA_ARCH_LIST=8.0`. Wheel 4.6 MB,
  `disco/_C*.so` 7.1 MB, `cuda_kernels_is_available()` and `_cuda_extension_available`
  both True. ~12 min compile.
- **Model build:** `model ready in 2.1 min`. Small -> the build-once cache below is NOT
  worth implementing.
- **Rollout:** started cleanly (fetched LocalIC, began batch 0), so the registry, init
  arithmetic, construction, and the ensemble/perturbation wiring all work.
- **BLOCKER - OOM, and it is a real hardware limit, not a bug:** the first forward died
  in the decoder DISCO conv `torch.einsum(...).contiguous()` (torch_harmonics
  `convolution.py:518`) trying to allocate **20.37 GiB** on top of **29.51 GiB** already
  resident => **~50 GiB peak**, over the A100's 40. That einsum runs whether or not the
  CUDA kernel is present (the kernel only does the sparse contraction at :507), so the
  bf16 kernel does not shrink it. `members=1` does not help (OOM is inside one member's
  forward); `expandable_segments` does not help (hard capacity gap, not fragmentation).
  **The FCN3 80 GB badge is real.** A scorable cube, s/step and peak-mem must come from
  the a3mega H100-80GB, where ~50 GiB peak fits => plan **batch_size=1 only** (batch>1
  would roughly double activations past 80 GB). The old note here that "40 GB is fine,
  job 6824712 started the forward with no OOM" was wrong: that run died before reaching
  the decoder.

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

## Artifact sync to Derecho (2026-08-17)

Derecho now holds a **complete copy of the finished experiment data** so the analysis can be
re-run there. It computed none of it; a3mega remains where the work happens. Access, the
`ControlMaster`/Duo recipe, and the byte-verification snippet are documented under "Reaching
Derecho from a3mega" in `CLAUDE.md`.

Moved with `scripts/sync_a3mega_to_derecho.sh` (manifest-driven, never deletes) plus a
second pass for what the manifest did not cover. **~89.5 GB total.** After both passes,
**93 directories are byte-identical (348.20 GB)**.

**Since 2026-08-18 the default manifest also carries the AI+RES Gate 2 artifacts**
(~530 MB: the adapter ensemble cube, the adapter IC, the verdict JSON/CSV, the calibration)
plus an optional `--with-walkers` group. The adapter cube is the only file in the manifest
Derecho cannot rebuild for itself. See "AI+RES — what a Derecho session can do" below.

| Pass | Contents | Size |
|---|---|---|
| 1 (manifest) | 6 FCN3 cubes, 6 ICs, 7 timing JSONs, scores CSV, 3 GenCast 0.25° p90 cubes, 3 p90 init frames, 3 ERA5 p90 truth | 22.95 GB / 29 files |
| 2 (gap fill) | `runs/p90_t2m/**` (90 cubes, 90 ICs, 180 truth, 90 timing, 90 maps, 3 scores), the 0.25° climatology, 3 p90 verif members, `figures/p90_t2m/`, `runs/xres/figures/` | 66.51 GB |

**Deliberately NOT synced** (and should stay that way):

| Path | Why |
|---|---|
| `runs/models/jax_cache_0p25` | machine-specific JAX compile cache — copying it is harmful |
| `runs/fcn3/.cache/model` (4.18 GB) | FCN3 weights, re-downloadable |
| `runs/fcn3/week3/shards` (1.63 GB) | regenerable zarr intermediates (`--with-shards`) |
| `*/cache/claims/` | work-stealing lock files, 0 bytes |

**Two gotchas this exposed**, both worth remembering before trusting a cross-machine figure:

1. **The sync script verifies file COUNTS, not bytes.** It reported `ok` on every group
   while Derecho still held a wrong-resolution climatology. Diff byte totals after any sync.
2. **`runs/models/clim_1990_2019_t2m_conus.nc` was same-name/different-grid** — 105x237
   (0.25°) here vs 14x30 (~2°) there. Since cubes are `lat=105, lon=237` and anomalies are
   formed against this file, the stale copy would have crashed or silently corrupted every
   anomaly on Derecho. Fixed; the old file is kept as `...nc.bak-2deg`. Check the grid with
   `ncdump -h` rather than assuming the filename means the same thing on both boxes.

Note `figures/p90_t2m/` is **untracked in git** — it reached Derecho only via this sync, not
via the branch.

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

## AI+RES — what a Derecho session can do (2026-08-18)

Branch `aires`. Phase 1 is complete except Gate 3. Full state: `aires/HANDOFF.md`.

**Where the work happened.** Gate 1 (channel fidelity) was built on Derecho and
**re-verified on a3mega with identical numbers**. Gate 2 (forecast equivalence) ran here —
Slurm job 1153, ~11 min on one 8×H100 node — because it needs FCN3, which does not fit an
A100-40GB. `aires/walker.py` (the global, restartable GenCast walker Gate 3 depends on) is
written, tested and exercised on an H100 (job 1154).

**Gate 2's result, in one paragraph.** Two FCN3 ensembles from the same ERA5 analysis at
the same init, differing only in how the 72 channels were assembled, with seeds matched
shard-for-shard against the cached 24-member `PNW_HeatDome_2021` cube (so the native side
cost no GPU time). `|Δ mean A_L|` = 0.083 K (**PASS**, threshold 0.25 K); paired Spearman
at the 21 d window = 0.107 (**FAIL**, threshold 0.95); max ratio of the adapter's
divergence to FCN3's own internal-noise divergence = 0.990 (**PASS**, added criterion).
The gate's purpose is met — the score a walker receives is an ensemble mean and it is
reproduced — but the rank-correlation criterion as written is not a valid discriminator at
21 d lead, where any IC difference has saturated. Resolved by lead, the two ensembles are
rank-equivalent out to 8.5 d. **Whether to amend that criterion in `aires.md` is an open
decision**, recorded in `aires/HANDOFF.md`.

**To re-derive Gate 2 on Derecho.** It is CPU-only — no GPU, no network — but it needs one
file Derecho cannot produce:

```bash
# 1. ON a3mega, push the aires group (now in the DEFAULT manifest, ~530 MB).
#    Needs a live ControlMaster (see "Reaching Derecho from a3mega" in CLAUDE.md).
bash scripts/sync_a3mega_to_derecho.sh --census-only     # what will move
bash scripts/sync_a3mega_to_derecho.sh                   # then verify BYTES, not counts

# 2. ON Derecho
module load conda && conda activate my-env
cd /glade/derecho/scratch/exu/S2S_ExtremeWeather
git fetch && git checkout aires && git pull

ncdump -h runs/models/clim_1990_2019_t2m_conus.nc | sed -n '2,8p'   # expect lat=105 lon=237
PYTHONPATH=. python -m pytest aires/tests/ -q            # 48 pass, 1 skipped, ~100 s
PYTHONPATH=. python -m aires.calibrate --fit --gate1     # ~2 min, must reproduce Gate 1
PYTHONPATH=. python -m aires.gate2 --stage score         # re-derives the Gate 2 verdict
```

`--stage score` reads two cubes and writes the verdict JSON, the per-member CSV and
`figures/aires/PNW_HeatDome_2021/gate2_PNW_HeatDome_2021.png`. It needs `scipy` and `matplotlib` in `my-env`.

| file | on Derecho? | note |
|---|---|---|
| `runs/fcn3/week3/cache/PNW_HeatDome_2021_cube.nc` | **yes** | native side, from the 2026-08-17 sync |
| `runs/aires/gate2/cache/PNW_HeatDome_2021_adapter_cube.nc` | **only after the sync** | **cannot be rebuilt there** — needs FCN3 on an 80 GB H100 |
| `runs/aires/calib/derived_calib.nc` | rebuildable | `--fit` regenerates it in ~2 min, deterministically |
| `runs/aires/gate2/ic/*_adapter_ic.nc` | rebuildable | CPU only, from the GenCast init frames |
| `runs/aires/*/walkers/**` | no | `--with-walkers` if ever needed; Derecho cannot score them |

**What Derecho still cannot do for AI+RES:** anything producing new FCN3 or 0.25° GenCast
output — Gate 3, the walker rolls, the production RES run. Those stay on a3mega.

---

*Last updated: Tue Aug 18 2026 — this experiment is complete; live work is on branch
`aires`. Added the AI+RES section above: Gate 2 ran on a3mega (job 1153) with a split
verdict, `aires/walker.py` rolled on an H100 (job 1154), and the sync manifest now carries
the Gate 2 artifacts so Derecho can re-derive the analysis. The Jul 23 Derecho smoke result
(job 6857923: FCN3 OOMs on an A100-40GB, ~50 GiB peak, needs the 80 GB H100) still stands
and is why AI+RES scoring is a3mega-only — see "Smoke result" above.*
