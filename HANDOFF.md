# GenCast xres (0.25° vs 1.0° week-2) — Agent Handoff

**Goal:** Run the cross-resolution GenCast S2S experiment comparing **0.25°** and
**1.0°** full checkpoints on **12 out-of-sample extreme events** at **week-2**
(14-day lead), then produce cross-resolution comparison figures.

> **Scope of this file.** This handoff tracks the **GenCast experiments only** (`gencast_s2s/`,
> `xres/`) — live PBS job IDs, what data is built, failure modes, triage. Everything below
> the next section assumes Derecho/PBS/JAX.
>
> The repo also contains **`downscaler/`** (ERA5 1° → HRRR 3 km diffusion model, PyTorch, runs
> off-Derecho). It has **no cluster jobs, no queue state, and nothing to triage**, so it is
> deliberately not tracked in the STATUS stack below — see the Jul 11 entry for where it came
> from, `CLAUDE.md` for how the two stacks relate, and `downscaler/README.md` for the project
> itself. If you were sent here to work on the downscaler, this file has almost nothing for you.

---

## STATUS (Jul 11 2026) — DOWNSCALER MOVED INTO THIS REPO (no jobs; GenCast unaffected)

The ERA5→HRRR diffusion downscaler was moved from `eric/downscaler` to **`downscaler/`**
inside this repo, because it is an **extension of this project** rather than a separate one:
`xres` measured what 0.25° buys at week-2 and how brutally it costs (~50 GiB VRAM/member,
~15 h/event, ~185 GPU-h for one 12-event pass) — the downscaler is the cheap alternative
path to fine scale. Keep the ensemble coarse at 1.0°, and learn 1° → 3 km as a separate
generative model. Because it trains on ERA5/HRRR *analysis* pairs it is forecast-agnostic,
so it can eventually be applied to a coarse GenCast member to produce a **3 km downscaled S2S
ensemble** with no 0.25° GenCast run at all.

**Nothing about the GenCast experiments changed** — no code, config, path, `runs/` output, or
figure was touched, and the completed xres results below still stand.

What the move involved:

- `mv` into `downscaler/` (it had no git history of its own; it is now untracked content in
  this repo — `git add downscaler/` when you want it versioned).
- Rewrote the 3 absolute paths that pointed at the old location (`ckpt_dir` in
  `downscaler/configs/config.yaml`; `stats_path` + `index_path` in
  `downscaler/configs/data/era5_hrrr.yaml`). The other absolute paths in that config
  (`era5_dir`, `hrrr_dir`, `grid_meta_path`) point outside the project and were left alone.
- Re-ran the CPU smoke test from the new location — passes end-to-end, and the checkpoint
  writes to the new `ckpt_dir`, confirming the rewrite. Its artifacts (`checkpoints/`,
  hydra `outputs/`) were deleted afterwards and are now gitignored.

**The two stacks are decoupled and must stay that way** — `downscaler/` does not import from
`gencast_s2s/` or `xres/`, uses PyTorch (not JAX), and runs in the `moe` env on a GPU box (not
`my-env` on Derecho). Do not submit it to PBS. The GenCast↔downscaler coupling described
above is **not built yet**; when it is, it belongs in a thin adapter that writes GenCast
members into the format `Era5HrrrDataset` expects, not in a cross-import.

**Status of the downscaler itself: plumbing works, no science yet.** The 1° ERA5 is still on
Derecho (scratch was down), so `data.use_dummy: true` is the default and the model has only
ever been run on *random tensors*. `downscaler/README.md` lists the 6 steps to switch to real
data — the first is transferring the 1° ERA5 off Derecho, which is the actual blocker.

## PREVIOUS STATUS (Jul 9 2026, later) — SECOND VERIFICATION BATCH + ROOT DIR SORTED (job `6684088`)

- Every PDF/Brier/CRPS/rank figure now comes in TWO batches: own-grid truth (no
  suffix, as before) and a **common-baseline batch** (`*_vs_era5_0p25.png`) where
  BOTH resolutions are verified against native 0.25deg ERA5 (1.0deg members
  nearest-upsampled; PDF variant drops the ERA5 1.0deg curve).
  `xres_scores.csv` gained a `truth` column (`own_grid` / `era5_0p25`).
- **Project root sorted**: all figure PNGs now live under `figures/xres/` (xres) and
  `figures/original/` (original-experiment combined PDFs + allbestmaps); compare
  writes to `figures/xres/` + `runs/xres/figures/`, NOT the root anymore. Stray
  `true.e/o*` PBS logs moved into `logs/`.
- Forecast-curve styling made more visible (dense dots, white halo, dark marker
  edges, slightly deeper light colors).

## PREVIOUS STATUS (Jul 9 2026) — FIGURE OVERHAUL (compare job `6677479`)

Per user request the compare-stage figures were reworked (all code in
`xres/xcombined.py`, new `xres/xscores.py`, colors in `xres/xconfig.py`):

- **Combined PDFs** (`xres_combined_*.png`): HRRR curve REMOVED; ERA5 now drawn at
  BOTH grids (0.25deg native + sampled onto the 1.0deg forecast grid); y floor
  lowered to 1e-6; PDFs switched from gaussian_kde to the binned-histogram
  `PDF_histogram` method of github.com/moeindarman77/TransferLearning-QG
  (100 bins over mean±4sigma, hist/N, shared bin range per panel, physical x-axis).
  Style: truth = dark + solid, forecast = light + dotted; reds = 0.25deg,
  blues = 1.0deg; distinct markers per curve (colorblind-redundant).
- **New skill figures** (`xres/xscores.py`, same 4x3 grids, each res scored vs ERA5
  on ITS OWN grid): `xres_brier.png` (tail-exceedance BS at 75/90/95/99th truth
  percentiles, extreme-direction aware, f(1-f) reference), `xres_crps.png`
  (lat-weighted CRPS bars per event, paneled by metric family, % = 0.25 vs 1.0),
  `xres_rank_hist.png` + `xres_rank_hist_pooled.png`. Numbers also in
  `runs/xres/figures/xres_scores.csv`. Reader's guide with the math:
  `docs/xres_score_figures.html`.
- HRRR truth files and map panels are untouched (only the PDF curves dropped HRRR).

## PREVIOUS STATUS (Jul 8 2026 morning) — ✅ EXPERIMENT COMPLETE (both resolutions + figures)

The member arrays finished overnight, **all 48 subjobs exit 0**:

- **0.25°: DONE.** 12/12 cubes (verified 24 members each), 18/18 verif files.
  Subjobs queued 20:16 Jul 7 → started 00:55-01:34 (≈5 h queue wait, all 24
  concurrent, one node each) → last done 09:46. ~7.7 h/subjob, ≈185 GPU-h total.
- **1.0°: DONE.** Idalia + Vermont re-rolled post-epoch-fix; 12/12 cubes, 18/18
  verif. Subjobs took only **~9 min each** (2 events, packed up to 4/node).
- **Compare `6665324` ran automatically** (end 09:49 Jul 8): 12 maps +
  `xres_combined_{mean,pooled,best,extreme}.png` now include ERA5 + HRRR + both
  resolutions. Quarantine dir `runs/xres/quarantine_1964_epoch_bug/` is now safe
  to delete after a figure sanity-check.

Scheduling/charging facts confirmed (docs + qhist): Derecho GPU nodes CAN be
shared by `ngpus=1` jobs (1.0° subjobs packed 4/node); accounting records
NGPUs=1 per subjob, and per NCAR charging docs shared jobs are charged for the
requested `ngpus` only. Single-GPU subjobs also backfilled ~5 h after submission
where the earlier full-node request had waited ~24 h without starting.

## PREVIOUS STATUS (Jul 7 2026, night) — INFER RESTRUCTURED AS 24-MEMBER PBS ARRAYS (both resolutions)

Per user request, infer now runs as **single-GPU PBS array jobs — one subjob = one
ensemble member across all 12 events** (cheaper than full nodes, schedules/backfills
faster, and members cache independently). The full-node jobs `6664794` (0.25°
self-chain) and `6665042` (1.0° re-infer) and held compare `6665043` were **qdel'd**
and replaced Jul 7 ~19:00 by:

| Job | What | Per subjob | Status |
|---|---|---|---|
| `6665322[]` | 0.25° infer, `pbs/xres_member_0p25.pbs`, 24 subjobs | 1 GPU, 16 cpu, **243 GB** (2/node), 12 h; ~8 h expected | Q |
| `6665323[]` | 1.0° infer, `pbs/xres_member_1p0.pbs`, 24 subjobs | 1 GPU, 16 cpu, 100 GB (4/node), 3 h; only re-rolls Idalia+Vermont (10 cubes cached) | Q |
| `6665324` | compare, `pbs/xres_compare_dev.pbs` | cpudev | H, `afterok:6665322[]:6665323[]` |

Mechanics (`XRES_MEMBER_SEL` mode in `xres/xinference.py`): subjob N rolls ONLY
member N of each event into `cache/<event>_memberNN.nc` (skipped if cached, and
events with a complete cube are skipped outright — that's why the 1.0° array is
cheap). The last subjob to finish an event assembles the cube + verif metrics and
deletes the member files; assembly races across nodes are safe (atomic writes +
tolerant reader), and member-0's subjob picks up any cube whose assembler died.
No claim files, no self-chaining in this mode. **Recovery = resubmit the array**
(`qsub pbs/xres_member_0p25.pbs`): cached subjobs exit in ~2 min without loading
the model. Compare afterok requires ALL subjobs to exit 0 — if a few failed,
fix/resubmit the array, then `qsub pbs/xres_compare_dev.pbs` manually.
Per-subjob logs: `logs/xres_0p25_m.o<id>.<member>` / `logs/xres_1p0_m.o<id>.<member>`.
Smoke-tested on cached 1.0° events (skip path + assembly wiring, exit 0).

The full-node worker-pool/self-chain path (previous status below) still exists in
`pbs/xres_res0p25_week2.pbs` as a fallback but is **superseded** by the arrays.

## PREVIOUS STATUS (Jul 7 2026, late evening) — ARCO_EPOCH BUG FOUND; Idalia + Vermont REBUILDING

- **ARCO_EPOCH bug (found Jul 7 ~18:30, fixed in `gencast_s2s/data.py`).** The v3 ARCO
  store's time axis is `hours since 1900-01-01`, but `ARCO_EPOCH` was 1959-01-01 — every
  ARCO read was shifted **59 years into the past**. Any event whose truth window or init
  frames fall past `WB2_END` (2023-01-10T18) was built from 1964 ERA5 data:
  **HurricaneIdalia_2023** (u850 truth + inputs) and **Vermont_Floods_2023** (tp truth +
  inputs). California_AR_Floods_2023 (peak 2023-01-10) squeaks under the cutoff — clean.
  All other events and the HRRR truths are unaffected. Symptom that exposed it: ERA5
  weekly-mean |u850| for Idalia had its max over the Great Lakes and no hurricane, while
  HRRR showed 64 m/s in the SE; Ian/Ida ERA5-vs-HRRR correlate 0.99, Idalia only 0.54.
- **Contaminated files quarantined** (moved, not deleted) to
  `runs/xres/quarantine_1964_epoch_bug/`: 4 ERA5 truth files, 2×2 inputs, 2 1p0 cubes,
  4 1p0 verif files. Safe to delete once the rebuild is verified.
- **Rebuild chain submitted:** `6665041` `xres_prep_fix1964` (develop; rebuilds the 4
  truth files + 4 inputs via `XRES_EVENTS_SEL=HurricaneIdalia_2023,Vermont_Floods_2023`)
  → afterok → `6665042` 1.0° re-infer (cache-aware: only the 2 missing events re-roll)
  → afterok → `6665043` compare (regenerates all figures incl. new pooled mode).
- **Prep-fix DONE & VERIFIED (Jul 7 18:28, exit 0).** All 4 truth + 4 input files
  rebuilt from real 2023 data: Idalia inputs carry 2023-08-15T12/16T00 frames; Idalia
  ERA5-vs-HRRR u850 corr 0.54 → **0.96** with hurricane-strength max (45.9 m/s) in the
  W Atlantic; Vermont tp corr 0.50 ≈ Kentucky's 0.55 (normal for summer convection;
  Vermont-box means agree, 24 vs 22 mm). GPU jobs `6664794` (0.25° chain) + `6665042`
  (1.0° re-infer) auto-released to Q. Quarantine dir is safe to delete after figures
  are confirmed.
- **0.25° chain `6664794`: race with the prep-fix CLOSED (Jul 7 ~18:20)** via
  `qalter -W depend=afterok:6665041 6664794` — it now waits for the prep-fix and
  auto-releases, keeping its queue position. Verified while checking the race: all
  truth/input writes are atomic (`tmp.<pid>` + `os.replace`, `gencast_s2s/data.py` /
  `xres/xdata.py`), so concurrent prep could not have corrupted files; the real risk
  was different — in `pbs/xres_res0p25_week2.pbs` prep runs BEFORE the successor is
  submitted, so a prep crash (missing inputs + no GCS from a GPU node) would have
  killed the job before chaining and the self-chain would silently never start.
  If `6665041` fails, `6664794` stays Held forever: fix prep, then
  `qalter -W depend=None 6664794` (or qdel + fresh qsub).
- **New figure mode (Jul 7 ~17:45):** `xres/xcombined.py` gained an ALL-MEMBERS-POOLED
  figure (`xres_combined_pooled.png`, matches the original experiment's pooled
  `combinedpdf.png` statistic); `pbs/xres_combined_figs.pbs` regenerates just the 4
  combined-PDF figures on develop.

## PREVIOUS STATUS (Jul 7 2026, evening) — 1.0° + HRRR DONE, 0.25° RESTARTED (self-chaining job `6664794`)

- **Prep: complete.** 18/18 ERA5 truth files, 12/12 inputs per resolution.
- **1.0° infer: complete & verified.** All 12 cubes (24 members × 28 steps) in
  `runs/xres/1p0/week2/cache/`, all 18 verif files in `runs/xres/1p0/week2/verif/`.
  Log shows clean end Jul 7 03:40 (third attempt in `logs/xres_1p0_wk2.log`; the
  first two runs in that log failed on a jax `P` AttributeError and a pandas
  InvalidIndexError, both fixed before the final run).
- **0.25° infer: RESTARTED Jul 7 evening with a reworked pipeline.** The abandoned
  Jul 6-7 attempt (job `6654166`) hit the 12 h walltime with zero output: serial
  single-GPU rollout at ~80 s/step needs ~15 h for ONE event (24 members × 28 steps
  = 672 chunks), and cubes were only written after ALL members finished, so 19
  completed members were discarded. Fixed with per-member caching + resume,
  memory-gated multi-worker rollout, and self-chaining PBS jobs (see
  **"0.25° pipeline design"** below). First job of the chain: **`6664794`**.
  The chain runs unattended — each hop resumes from cached members and the finishing
  job submits compare itself. Expect several 12 h hops (≈2-5 days wall-clock
  depending on how many workers fit in host RAM).
- **Compare/figures: DONE (Jul 7 17:19).** Ran via new `pbs/xres_compare_dev.pbs`
  (develop queue — the original `pbs/xres_compare.pbs` routes to the `cpu` queue which
  had ~1800 queued jobs; submission `6664453` was qdel'd). Job `6664462` produced all
  figures; job `6664470` regenerated them after `xres/xplotting.py::plot_event` was
  changed to drop missing panels (HRRR + 0.25°) instead of leaving blank axes, so
  per-event maps are now 3 panels: ERA5 | 1.0° mean | 1.0° error.
  Outputs: `xres_combined_{mean,best,extreme}.png` at project root (all 12 events,
  ERA5 black vs GenCast 1.0° orange) and 12 maps in `runs/xres/figures/maps/`.
- **HRRR overlay: DONE (Jul 7 17:34, job `6664608`).** The npz shard archive never
  existed on Derecho, but a HRRR v3 netCDF archive does:
  `/glade/derecho/scratch/mdarman/hrrr_work/hrrr_nc_v3` (6-hourly, 2015–2025, one
  file per timestamp with `t2m`, `u`/`v` on pressure levels, `log_tp`; full coverage
  of all 12 event windows). `gencast_s2s/config.py` now points there (`HRRR_NC`,
  env-overridable) and `gencast_s2s/hrrr.py` + `xres/xhrrr.py` read nc instead of npz
  (`HRRR_SHARDS`/`HRRR_GRID_REF`/`HRRR_T2M_CHANNEL` are gone). All 18
  `runs/observations/*_hrrr_verif_*.nc` files built via `pbs/xres_hrrr_compare.pbs`
  (develop queue), and figures were regenerated with the red "Observed (HRRR)"
  curve/panel. Legends now also state the ensemble size, e.g.
  "GenCast 1.0deg wk-2 mean (24 members)".
  When the 0.25° chain finishes, compare re-runs automatically and all figures
  regenerate with the 0.25° panels/curves added.

### 0.25° pipeline design (added Jul 7, in `xres/xinference.py` + `pbs/xres_res0p25_week2.pbs`)

Why: 0.25° needs ~50 GiB VRAM/member → no pmap on A100-40GB → serial one-member-at-
a-time rollout at ~80 s/step → ~15 h/event → longer than the 12 h walltime cap. Also
one serial process peaked at **~237 GiB host RAM** (job `6654166`,
`resources_used.mem`), so 4 naive per-GPU processes (~950 GiB) may not fit the
node's 487 GB either. Design:

1. **Per-member caching (resume).** In serial mode every finished member is written
   immediately to `cache/<event>_memberNN.nc`. When all 24 exist, whichever worker
   wrote the last one assembles the cube, derives the verif metrics, and deletes the
   member files. A killed job loses at most the member in flight (~37 min).
2. **Work-stealing workers.** Members are claimed via atomic claim files in
   `cache/claims/` (PID-stamped; claims of dead PIDs are stolen). Any number of
   concurrent worker processes cooperate with no static partition. A worker that
   finds no claimable work exits WITHOUT loading the model (lazy load), so extra
   workers and post-completion hops are cheap.
3. **Adaptive worker pool.** The PBS job starts worker 0 on GPU 0, then every 15 min
   (`XRES_WORKER_STAGGER=900`) starts the next worker (GPU 1-3) only if
   `MemAvailable` ≥ `XRES_WORKER_MIN_FREE_GB` (250). Worst case it degrades to the
   old 1-worker throughput — but now resumable.
4. **Self-chaining jobs.** Right after prep, each job submits an `afterany`
   successor of itself (`XRES_CHAIN_REMAIN` hops left, default 11, decremented per
   hop). Walltime kills are the *expected* end of most hops. The hop that finds 12
   cubes + 18 verif files qdels its successor and submits `pbs/xres_compare_dev.pbs`;
   a hop that makes zero progress qdels its successor and stops the chain.

**To STOP everything:** `qstat -u exu`, then `qdel` BOTH the running `xres_0p25_wk2`
job AND its held successor (there is always at most one successor pending).

**Throughput math:** 288 member-rollouts total, ~37 min each ⇒ ~180 GPU-h ⇒ ~16 hops
at 1 worker, ~8 at 2, ~4-5 at 4. If the 11-hop budget exhausts before completion,
just `qsub pbs/xres_res0p25_week2.pbs` — it resumes from cache with a fresh budget.

**Logs per hop:** orchestration in `logs/xres_0p25_wk2.o<jobid>` (written at job
end); live worker progress in `logs/xres_0p25_wk2_<jobid>_w<0-3>.log` (tail these).

**Env knobs** (set via `qsub -v`): `XRES_MAX_WORKERS` (4), `XRES_WORKER_STAGGER`
(900 s), `XRES_WORKER_MIN_FREE_GB` (250), `XRES_CHAIN_REMAIN` (11),
`XRES_XLA_ALLOC` (`platform`; try `default` for the faster BFC allocator — if it
OOMs the chain just resumes).

**System:** NCAR **Derecho** (PBS Pro). Conda env: **`my-env`**
(`/glade/work/exu/conda-envs/my-env`, Python 3.10). Project dir:
`/glade/derecho/scratch/exu/S2S_ExtremeWeather/`.

**Use this file when:** a job disappears from `qstat -u exu` — either it finished
(success or failure) or was killed. Spin up a new agent, point it at this file,
and follow the **"When a job leaves qstat"** section below.

---

## Pipeline (current: member arrays)

```
prep (done)  ─►  xres_0p25_m[0-23]  (24 × 1-GPU subjobs) ──┐
                                                            ├─afterok─►  xres_compare (cpudev)
                 xres_1p0_m[0-23]   (24 × 1-GPU subjobs) ──┘
```

| Job / array | Name | PBS script | Per subjob | Log | Status |
|---|---|---|---|---|---|
| `6652566` | `xres_prep` | `pbs/xres_prep.pbs` | cpudev | `logs/xres_prep.log` | done (+ fix1964 `6665041`) |
| `6665322[]` | `xres_0p25_m` | `pbs/xres_member_0p25.pbs` | 1 GPU, 243 GB, 12 h | `logs/xres_0p25_m.o<id>.<m>` | **queued** |
| `6665323[]` | `xres_1p0_m` | `pbs/xres_member_1p0.pbs` | 1 GPU, 100 GB, 3 h | `logs/xres_1p0_m.o<id>.<m>` | **queued** |
| `6665324` | `xres_compare` | `pbs/xres_compare_dev.pbs` | cpudev, 2 h | `logs/xres_compare.log` | Held on both arrays |

Subjob N = ensemble member N for all events. `qstat -t "6665322[]"` lists per-subjob
states. Superseded scripts kept as fallback: `pbs/xres_res0p25_week2.pbs` (full-node
self-chaining pool) and `pbs/xres_res1p0_week2.pbs` (full-node pmap).

Historical prep attempts (all resolved Jul 6): 6651105/6652181/6652312 OOM'd on ARCO
events pre-fix; 6651299 sat unstarted in the `cpu` queue; 6652121 cancelled.
Historical 0.25° attempts: 6652567/6653517/6653688/6653768/6653941/6653948/6653990
(config/code errors), 6654166 (ran 12 h at ~80 s/step, walltime-killed at member
19/24 of event 1 with nothing saved — the failure that motivated the redesign).

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

### If member-array subjobs (`xres_0p25_m` / `xres_1p0_m`) finished

**Check progress / success:**
```bash
# Durable outputs per resolution (0p25 shown; same for 1p0)
ls runs/xres/0p25/week2/cache/*_cube.nc | wc -l          # 12 when done
ls runs/xres/0p25/week2/verif/*_members.nc | wc -l       # 18 when done
ls runs/xres/0p25/week2/cache/*_member*.nc | wc -l       # members of not-yet-assembled events

# Per-subjob states (X = finished):
qstat -t "6665322[]"          # 0.25°
qstat -t "6665323[]"          # 1.0°

# A subjob's log (member m):
tail -20 logs/xres_0p25_m.o<arrayid>.<m>

# Any subjob failures?
for m in $(seq 0 23); do qstat -fx "6665322[$m]" | grep -q "Exit_status = 0" || echo "member $m NOT ok"; done
```

| Outcome | What happens next | Your action |
|---|---|---|
| **All subjobs exit 0** | 12 cubes + 18 verif per res; compare auto-releases | Verify figures after compare |
| **A few subjobs failed** | Their members missing; affected cubes unassembled; compare stays Held (afterok) | Resubmit that array (`qsub pbs/xres_member_0p25.pbs`) — cached members skip in ~2 min; then `qsub pbs/xres_compare_dev.pbs` manually (the Held one dies with the failed dependency) |
| **Subjob walltime kill (0.25°)** | Shouldn't happen (~8 h of ~12 h used) | Check its log for abnormal step times; resubmit array |
| **Subjob host-mem kill** | mem=243gb vs ~237 GiB observed peak | Bump `mem=` in `pbs/xres_member_0p25.pbs` (2/node needs ≤243gb; 1/node up to 487gb), resubmit array |

Cube assembly: last finisher per event assembles; if an assembler died mid-way, the
member files are still there and **member-0's subjob assembles it on the next array
resubmit** (or run assembly manually on cpudev: `XRES_SERIAL_INFER=1 XRES_MEMBER_SEL=0
python run_xres.py --stage infer --res 0p25` — no GPU needed if all members exist).

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

## Resubmit

```bash
cd /glade/derecho/scratch/exu/S2S_ExtremeWeather

# Member arrays (current method; fully cache-aware, resubmitting is cheap):
A025=$(qsub pbs/xres_member_0p25.pbs)
A10=$(qsub pbs/xres_member_1p0.pbs)
qsub -W depend=afterok:$A025:$A10 pbs/xres_compare_dev.pbs

# Only if ever needed:
qsub pbs/xres_prep.pbs             # prep (cache-aware)
qsub pbs/xres_compare_dev.pbs      # figures alone (develop queue; xres_compare.pbs = slow cpu queue)
qsub pbs/xres_res0p25_week2.pbs    # fallback: full-node self-chaining worker pool
qsub pbs/xres_res1p0_week2.pbs     # fallback: full-node pmap
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
| **Walltime limit** | Account max is **12 h**. 0.25° infer needs ~180 GPU-h total, so it runs as a self-chain of 12 h hops with per-member resume (see "0.25° pipeline design"). |
| **Host RAM per worker** | One serial 0.25° process peaked at ~237 GiB (job 6654166). The PBS launcher only starts extra per-GPU workers while `MemAvailable` ≥ 250 GB — expect 1-2 workers usually, not 4. |
| **Claim files** | `runs/xres/0p25/week2/cache/claims/` coordinates same-node workers; PID-based, cleared at each hop start. If you run serial infer manually, `rm -f` that dir's contents first. |
| **GPU hardware** | Production nodes via `main` are **4× A100-40GB** (not H100). Code works but slower than original 8×H100 Ubuntu cluster. |
| **HRRR overlay (fixed Jul 7)** | Now sourced from the netCDF archive `/glade/derecho/scratch/mdarman/hrrr_work/hrrr_nc_v3` via `C.HRRR_NC`. Figures include the red HRRR curve/panel. |
| **Login node OOM** | Never run `prep` on login node — killed with exit 137. |
| **ARCO ERA5 OOM (fixed)** | Post-2023 events use ARCO zarr. Do **not** `xr.open_zarr` the ARCO root; use `arco_read()` in `gencast_s2s/data.py`. |
| **cpudev vs cpu queue** | cpudev (120 GB, shared) starts fast. cpu queue (200 GB) needs exclusive node — long queue (~1200 jobs). Use cpudev for prep. |
| **Google FutureWarning** | Python 3.10 EOL notice from gcsfs on import. Harmless until Oct 2026. |
| **Ensemble size** | Default 24 members. pmap path (1.0°) needs a multiple of the GPU count (4); the serial 0.25° path has no such constraint. Override: `XRES_N_MEMBERS_0P25=16`. Changing it invalidates existing cubes (member-count mismatch → re-roll). |

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
    │     └── cache/ also holds transient <event>_memberNN.nc files + claims/
    │         while the 0.25° chain is running (deleted once a cube is assembled)
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
| `downscaler/` | **Separate stack:** ERA5 1° → HRRR 3 km diffusion model (PyTorch, no PBS). See `downscaler/README.md` |

---

## Agent checklist (copy-paste for new agent)

When user says "a job left qstat, read HANDOFF.md":

1. `qstat -t "6665322[]"` / `qstat -t "6665323[]"` — per-subjob states of the two
   member arrays (subjob N = ensemble member N; `X` = finished).
2. `ls runs/xres/<res>/week2/cache/*_cube.nc | wc -l` (12 = done per res) and
   `... *_member*.nc | wc -l` — members of not-yet-assembled events.
3. Subjob failures: `qstat -fx "6665322[<m>]" | grep Exit_status` (0 = ok);
   tracebacks in `logs/xres_0p25_m.o<arrayid>.<m>` / `logs/xres_1p0_m.o<arrayid>.<m>`.
4. **All subjobs ok →** compare `6665324` auto-releases; afterwards verify
   `xres_combined_{mean,best,extreme,pooled}.png` + `runs/xres/figures/maps/`
   include the 0.25° curves/panels.
5. **Some subjobs failed →** resubmit that array (`qsub pbs/xres_member_0p25.pbs`
   and/or `_1p0.pbs`; cached members skip in ~2 min), then submit compare manually
   (`qsub pbs/xres_compare_dev.pbs` — the Held one dies with its failed dependency).
6. **All members exist but a cube is missing →** the assembler died; resubmit the
   array (member-0's subjob assembles) or assemble on CPU:
   `XRES_SERIAL_INFER=1 XRES_MEMBER_SEL=0 python run_xres.py --stage infer --res <res>`.
7. **Update this file** (STATUS + job table) whenever the picture changes.

---

*Last updated: Sat Jul 11 2026. The `downscaler/` project (ERA5 1° → HRRR 3 km diffusion)
was moved into this repo as an extension — no GenCast code, config, or output was touched,
and it has no cluster jobs; see the Jul 11 STATUS entry at the top. The GenCast xres
experiment itself remains **complete** (both resolutions, 12/12 cubes each, figures built) as
of the Jul 8–9 entries — no jobs are expected in `qstat`.*

*(Historical note: the previous footer, dated Jul 7 ~19:10 MDT, described the then-live member
arrays `6665322[]` / `6665323[]` / compare `6665324` and disk state "0p25 cubes 0/12". That
footer was already stale — those arrays finished Jul 8 with all 48 subjobs exit 0.)*
