# GenCast xres (0.25° vs 1.0° week-2) — Agent Handoff

**Goal:** Run the cross-resolution GenCast S2S experiment comparing **0.25°** and
**1.0°** full checkpoints on **12 out-of-sample extreme events** at **week-2**
(14-day lead), then produce cross-resolution comparison figures.

**System:** NCAR **Derecho** (PBS Pro). Conda env: **`my-env`**
(`/glade/work/exu/conda-envs/my-env`, Python 3.10). Project dir:
`/glade/derecho/scratch/exu/S2S_ExtremeWeather/`.

**Use this file when:** a job disappears from `qstat -u exu` — either it finished
(success or failure) or was killed. Spin up a new agent, point it at this file,
and follow the **"When a job leaves qstat"** section below.

---

## Pipeline (4 PBS jobs, dependency chain)

```
xres_prep (CPU)  ──afterok──►  xres_0p25_wk2 (GPU)  ──┐
                     │                                  ├──afterok──►  xres_compare (CPU)
                     └──afterok──►  xres_1p0_wk2 (GPU) ──┘
```

| Job ID | Name | PBS script | Queue | Walltime | Log |
|---|---|---|---|---|---|
| `6652566` | `xres_prep` | `pbs/xres_prep.pbs` | develop → cpudev | 4 h | `logs/xres_prep.log` |
| `6652567` | `xres_0p25_wk2` | `pbs/xres_res0p25_week2.pbs` | main → **gpu** | 12 h | `logs/xres_0p25_wk2.log` |
| `6652568` | `xres_1p0_wk2` | `pbs/xres_res1p0_week2.pbs` | main → **gpu** | 6 h | `logs/xres_1p0_wk2.log` |
| `6652569` | `xres_compare` | `pbs/xres_compare.pbs` | main → cpu | 2 h | `logs/xres_compare.log` |

Prep `6652566` **running** on cpudev (ARCO memory fix + per-event subprocess isolation).
Jobs 6652567–6652569 are **Held** until prep succeeds. GPU infer jobs re-run
`prep --res <r>` first (cache-aware; instant if prep already built that resolution's
inputs).

**Expected prep runtime (post-fix):** ~2–4 h on cpudev for remaining work (6 ERA5 precip
files + 11 input pairs per resolution). Prep is cache-aware — completed files are skipped.

| Prior prep attempts | Queue | Outcome |
|---|---|---|
| 6651105, 6652181, 6652312 | cpudev | OOM exit 137 on ARCO events (pre-fix) |
| 6651299 | cpu (200 GB) | Never ran (~1200 jobs ahead) |
| 6652121 | pcpu | Cancelled when switching back to cpudev |

---

## What is already done

### Environment (`my-env`)
- `setup_env.sh` was adapted for Derecho and run successfully.
- Installed: `jax[cuda12]`, `dm-haiku`, `jraph`, `dm-tree`, `trimesh`, `rtree`,
  `chex`, `dask`, `gcsfs`, `zarr`, `netCDF4`, `cartopy`, `matplotlib`, `pandas`,
  `properscoring`, `dinosaur-dycore`.
- GraphCast cloned to `third_party/graphcast/` (editable install, xarray patch applied).
- Imports verified: `from graphcast import gencast` works.

### Data (`runs/`) — prep in progress (`6652566`)

**Complete (models + shared):**
- `runs/models/params/` — both GenCast checkpoints (0p25 + 1p0)
- `runs/models/stats/` — normalization stats (4 files)
- `runs/models/statics.nc`, `runs/models/clim_1990_2019_t2m_conus.nc`

**ERA5 truth: 12 / 18 files** (`runs/observations/*_verif_*.nc`)

| Status | Events |
|---|---|
| Done (12) | 6 heat/cold T2m + Ida + Ian + **Idalia** wind metrics |
| **TODO (6)** | `California_AR_Floods_2023`, `Kentucky_Floods_2022`, `Vermont_Floods_2023` (each needs `tp_total` + `tp_max12h`) |

**Model inputs: 1 / 12 per resolution** (`runs/xres/*/week2/inputs/`)

| Resolution | Done | Remaining |
|---|---|---|
| 0.25° | `HurricaneIdalia_2023_inputs.nc` | 11 events |
| 1.0° | `HurricaneIdalia_2023_inputs.nc` | 11 events |

HRRR truth skipped (no shard archive on Derecho; figures are ERA5-only).

### Code fixes applied (Jul 6 evening)
- **ARCO OOM root cause:** `xr.open_zarr()` on the full ARCO store (277 vars) OOM'd at
  ~120 GB. Fixed in `gencast_s2s/data.py` — direct zarr array reads via `arco_read()`.
- **Per-event subprocess isolation** in `run_xres.py` (`XRES_PREP_ISOLATE=1` default).
- **Timestep-wise ERA5 loads** for WB2 stores + incremental numpy reductions in
  `xres/xmetrics.py`.
- `pbs/xres_prep.pbs`: `PYTHONUNBUFFERED=1` for live log output.

### PBS scripts
All in `pbs/`:
- `xres_prep.pbs` — CPU prep on develop/cpudev (8 cores, 120 GB)
- `xres_res0p25_week2.pbs` — GPU prep+infer 0.25° via `main` queue
- `xres_res1p0_week2.pbs` — GPU prep+infer 1.0° via `main` queue
- `xres_compare.pbs` — CPU compare after both infer jobs

---

## When a job leaves qstat

Run these first:

```bash
cd /glade/derecho/scratch/exu/S2S_ExtremeWeather

# Which jobs are still active?
qstat -u exu

# Exit status of a finished job (0 = success)
qstat -fx JOBID.desched1 | grep -E "job_state|Exit_status|resources_used.walltime"

# Tail the log
tail -80 logs/<jobname>.log
```

Then match the finished job to the section below.

---

### If `xres_prep` finished

**Check success:**
```bash
grep -E "End:|Traceback|Error|Killed" logs/xres_prep.log
ls runs/observations/*_verif_*.nc | wc -l   # expect 18
ls runs/xres/0p25/week2/inputs/*_inputs.nc | wc -l   # expect 12
ls runs/xres/1p0/week2/inputs/*_inputs.nc | wc -l    # expect 12
```

| Outcome | What happens next | Your action |
|---|---|---|
| **Success** (Exit_status=0, all inputs exist) | GPU jobs auto-release from Hold | Monitor GPU jobs; nothing to resubmit |
| **Failed** (nonzero exit, Traceback, or Killed) | GPU jobs stay Held forever | Diagnose log, resubmit prep, re-chain GPU jobs (see **Resubmit** below) |
| **Partial** (some inputs missing) | GPU jobs may start but infer fails mid-run | Resubmit prep; cache skips completed files |

Common prep failures:
- **OOM (historical)** — pre-ARCO-fix jobs OOM'd on Idalia + precip events. Fixed Jul 6.
  If OOM recurs, check log for which event; smoke-test with
  `XRES_EVENTS_SEL=<event> XRES_PREP_SKIP_MODELS=1 python run_xres.py --stage prep`.
- **Never run on login node** — use PBS cpudev/develop.
- **GCS timeout** — rerun; downloads are cache-aware.
- **Walltime** — increase `#PBS -l walltime` in `pbs/xres_prep.pbs` (currently 4 h).
- **Google FutureWarning** — harmless (Python 3.10 EOL notice from gcsfs); ignore.

---

### If `xres_0p25_wk2` or `xres_1p0_wk2` finished

**Check success:**
```bash
# 0.25° outputs (~60 GB total across 12 events)
ls runs/xres/0p25/week2/cache/     # expect <event>_cube.nc per event
ls runs/xres/0p25/week2/verif/     # expect <event>_<metric>_members.nc

# 1.0° outputs (~5 GB total)
ls runs/xres/1p0/week2/cache/
ls runs/xres/1p0/week2/verif/

grep -E "End:|Traceback|CUDA|OOM|Killed" logs/xres_0p25_wk2.log
grep -E "End:|Traceback|CUDA|OOM|Killed" logs/xres_1p0_wk2.log
```

| Outcome | What happens next | Your action |
|---|---|---|
| **Success** | If *both* GPU jobs done, compare auto-starts | Wait for compare; verify figures |
| **Failed — partial cubes** | Compare stays Held | Resubmit *only the failed resolution*; infer is per-event cached so completed events are preserved |
| **Failed — walltime** | Job killed mid-event | Increase walltime (0.25° max is **12 h** per account limit), resubmit failed arm |
| **Failed — GPU OOM** | JAX CUDA OOM | Reduce `XRES_N_MEMBERS_0P25` (e.g. 16 or 8); must be multiple of 4 (GPU count) |

**Resume a failed GPU infer** (cache preserves completed events):
```bash
cd /glade/derecho/scratch/exu/S2S_ExtremeWeather
# Only resubmit the failed resolution; no need to redo the other if it succeeded.
qsub pbs/xres_res0p25_week2.pbs   # or xres_res1p0_week2.pbs

# After BOTH infer jobs succeed, run compare:
qsub pbs/xres_compare.pbs
# Or on login/cpudev:
module load conda && conda activate my-env
python run_xres.py --stage compare
```

---

### If `xres_compare` finished

**Check success:**
```bash
grep -E "End:|Traceback|Error" logs/xres_compare.log
ls -la xres_combined_*.png
ls -la runs/xres/figures/
```

| Outcome | Your action |
|---|---|
| **Success** | Done. Figures at project root: `xres_combined_mean.png`, `xres_combined_best.png`, `xres_combined_extreme.png`. Cross-resolution maps in `runs/xres/figures/`. |
| **Failed** | Both infer jobs must have finished. Check verif files exist, then rerun: `python run_xres.py --stage compare` or `qsub pbs/xres_compare.pbs`. |

---

## Resubmit (full chain or partial)

```bash
cd /glade/derecho/scratch/exu/S2S_ExtremeWeather

# Full chain from scratch:
PREP_ID=$(qsub pbs/xres_prep.pbs)
JOB025=$(qsub -W depend=afterok:$PREP_ID pbs/xres_res0p25_week2.pbs)
JOB10=$(qsub -W depend=afterok:$PREP_ID pbs/xres_res1p0_week2.pbs)
COMPARE_ID=$(qsub -W depend=afterok:$JOB025:$JOB10 pbs/xres_compare.pbs)
echo "PREP=$PREP_ID  0p25=$JOB025  1p0=$JOB10  COMPARE=$COMPARE_ID"
```

**Update the Job ID table at the top of this file** after resubmitting.

---

## Key commands (manual stages)

```bash
module load conda && conda activate my-env
cd /glade/derecho/scratch/exu/S2S_ExtremeWeather

# Prep (CPU, needs internet — use PBS not login node)
python run_xres.py --stage prep

# Infer (GPU only)
python run_xres.py --stage prep  --res 0p25   # cache-aware
python run_xres.py --stage infer --res 0p25
python run_xres.py --stage prep  --res 1p0
python run_xres.py --stage infer --res 1p0

# Compare (CPU, after BOTH infer jobs)
python run_xres.py --stage compare

# Smoke test: one event (useful after OOM fix validation)
XRES_EVENTS_SEL=HurricaneIdalia_2023 XRES_PREP_SKIP_MODELS=1 \
  python run_xres.py --stage prep

# Disable per-event subprocess isolation (default is on)
XRES_PREP_ISOLATE=0 python run_xres.py --stage prep
```

---

## Known issues / constraints

| Issue | Detail |
|---|---|
| **Direct `gpu` queue denied** | `qsub -q gpu` returns "Access to queue is denied". Use `#PBS -q main` with `select=1:ncpus=64:ngpus=4` — routes to production `gpu` queue. |
| **Do NOT use `gpudev`** | User requested production GPU nodes. `develop` queue routes to gpudev/cpudev — only used for **prep**, not infer. |
| **Walltime limit** | Account max is **12 h**. 0.25° infer was budgeted 12 h (original SLURM used 24 h on 8×H100). May be tight on 4×A100-40GB. |
| **GPU hardware** | Production nodes via `main` are **4× A100-40GB** (not H100). Code works but slower than original 8×H100 Ubuntu cluster. |
| **HRRR overlay missing** | Default `HRRR_SHARDS` path doesn't exist on Derecho. Prep skips HRRR; compare figures are ERA5-only (no red HRRR curve). |
| **Login node OOM** | Never run `prep` on login node — killed with exit 137. |
| **ARCO ERA5 OOM (fixed)** | Post-2023 events use ARCO zarr. Do **not** `xr.open_zarr` the ARCO root; use `arco_read()` in `gencast_s2s/data.py`. |
| **cpudev vs cpu queue** | cpudev (120 GB, shared) starts fast. cpu queue (200 GB) needs exclusive node — long queue (~1200 jobs). Use cpudev for prep. |
| **Google FutureWarning** | Python 3.10 EOL notice from gcsfs on import. Harmless until Oct 2026. |
| **Ensemble size** | Default 24 members. Must be multiple of GPU count (4). Override: `XRES_N_MEMBERS_0P25=16`. |

---

## Output layout

```
runs/
├── models/params/          # GenCast 0p25deg + 1p0deg checkpoints (downloaded)
├── models/stats/           # normalization stats
├── models/statics.nc       # orography + LSM
├── models/clim_*.nc        # T2m climatology (built during prep)
├── observations/           # ERA5 truth per (event, metric)
└── xres/
    ├── 0p25/week2/{inputs,cache,verif,figures}
    ├── 1p0/week2/{inputs,cache,verif,figures}
    └── figures/            # cross-resolution comparison maps

# After compare:
xres_combined_mean.png      # project root
xres_combined_best.png
xres_combined_extreme.png
```

---

## Files reference

| File | Purpose |
|---|---|
| `run_xres.py` | Driver: `--stage prep\|infer\|compare` |
| `xres/README.md` | Experiment design docs |
| `setup_env.sh` | One-time env setup (already run) |
| `pbs/*.pbs` | PBS job scripts |
| `logs/*.log` | Job stdout/stderr |
| `gencast_s2s/` | Core package (config, data, model, inference) |
| `xres/` | Cross-resolution extensions |
| `third_party/graphcast/` | GraphCast source (editable install) |

---

## Agent checklist (copy-paste for new agent)

When user says "a job left qstat, read HANDOFF.md":

1. `qstat -u exu` — note which jobs remain
2. Identify which job finished (prep / 0p25 / 1p0 / compare)
3. `qstat -fx JOBID.desched1 | grep Exit_status` — 0 = success
4. `tail -80 logs/<job>.log` — look for Traceback, Killed, "End:"
5. Check expected output files for that stage (see sections above)
6. **Success →** wait for next dependent job or verify final figures
7. **Failure →** diagnose log, fix, resubmit failed stage only (infer is resumable)
8. **Update this file** with new job IDs if you resubmit

---

*Last updated: Mon Jul 6 2026 ~20:08 MDT. Prep `6652566` running on cpudev (subprocess
isolation, ARCO fix). Disk: ERA5 12/18, inputs 1/12 per res. GPU jobs 6652567–6652569
Held. Prior jobs 6651105–6652312 OOM'd before ARCO fix; Idalia smoke-tested OK post-fix.*
