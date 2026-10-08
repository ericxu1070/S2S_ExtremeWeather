# acal — live state

Read `acal/__init__.py` for what this package is for and `ccfg.py` for the knobs. This
file is what is **actually built**, not what is planned.

Last updated 2026-10-08.

The 42-case campaign finished 2026-09-30. The cross-case result is in "Campaign analysis (2026-10-01)" and the head-to-head against NCEP CFSv2 in "CFSv2 operational baseline (2026-10-05/06)". The multi-model board (CFSv2, GEFSv12 and ECCC GEPS scored; ECMWF IFS (EC46) and the BB-SUBS estimate pending the user's ECDS token) against ERA5 and HRRR truth is in "Multi-model board: CFSv2, GEFSv12, ECCC GEPS, ECMWF IFS (EC46, pending), BB-SUBS estimate (pending), ERA5 and HRRR truth (2026-10-07/08)", the last section of this file, with a one-page reader summary in `runs/acal/analysis/s2s/board/SUMMARY.md`. Read that section's steering caveat (tilt check) before quoting any AI+RES-vs-baseline log ratio.

## Built

**Stage `cube` — DONE, all six years.**
`runs/acal/index/era5_t2m_anom_12h_{2021..2026}.nc`, 400 MB total. 0.25 deg CONUS crop
(105x237), 12-hourly (00Z/12Z), ARCO-ERA5 2m_temperature minus the WB2 1990-2019 hourly
climatology. 2021-2025 are full years (742 frames each, incl. the 6-day December lead-in);
2026 stops at `PEAK_END` = 2026-08-31 (497 frames).

The climatology grid was verified against the trap in the root `CLAUDE.md` — this box's
`runs/models/clim_1990_2019_t2m_conus.nc` is the correct 0.25 deg file (105x237, 24-50 N /
235-294 E, d=0.25), not the stale 2 deg one.

Trap: `logs/acal/cube_*.log` all end in a `RuntimeError` traceback from
`gcsfs`/`aiohttp` (`Task ... attached to a different loop`). **It is benign** — it fires in
a `weakref` atexit finalizer at interpreter shutdown, *after* `wrote era5_t2m_anom_12h_*.nc`
is printed. The cubes are complete. Do not rebuild on the strength of that traceback.

**Stage `scan` — NOT frozen.** `runs/acal/cases.csv` and `cases_meta.json` do not exist.
The 2021-2026 slate has never been written.

## Also built: the 2021-2025 day catalog

`runs/acal/catalog/` — a frozen 2021-2025 catalog answering "which days cross +/-2, 3, 4 K",
with its own `README.md` (definitions, counts, caveats) and `build_catalog.py` (rerunnable,
CPU, offline). Four CSVs: the CONUS-wide daily index, every (box, day) crossing, the
acal-rule declustered cases, and one-per-10-day-window CONUS events.

This is **not** `cases.csv` and must not be renamed into it: it covers 2021-2025 only,
where the campaign's slate is 2021-2026.

## Known problem: the declustering rule over-counts

`find_cases` suppresses a candidate only when a stronger one is within `DECLUSTER_DAYS`
**and** overlaps it by more than `DECLUSTER_OVERLAP` of a box area. Both conditions must
hold, so spatially distinct boxes inside one continental outbreak all survive.

Measured on 2021-2025: 5,215 cases on 1,387 distinct dates (3.8 per date).
**February 2021 (Uri) alone produces 38 separate "cold" cases** across 10 dates and 38
boxes; four of them take top-5 cold by |A_L|.

The visible symptom is that the exclusive rung bins come out **inverted** — 1,680 cases at
2-3 K against 2,279 at >= 4 K. Rarity must fall with depth; here it rises, because weak
crossings next to a strong one get absorbed while the strong one never is. If one case per
event is what is wanted, the overlap test needs to become a disjunction or gain a
CONUS-wide suppression pass.

Cost consequence: at `RES_N_WALKERS`=32 the report stage prices the slate at ~31 H100-h per
case. 5,215 cases is ~162,000 H100-h = ~20,200 node-h — about 7 months of wall time on five
8-GPU nodes. The campaign is not runnable at this case count; fixing the declustering is a
prerequisite, not a cleanup.

## Selection-bias note for whoever writes the paper

The rungs are absolute K against a 1990-2019 baseline over a 2021-2025 period, so warming
trend is inside the "anomaly". Hot outruns cold ~2.5:1 per box at every rung. The ordering
flips CONUS-wide at the deep end (+/-3 K: 32 hot vs 33 cold days; +/-4 K: 5 vs 8) — the
warm bias shifts the bulk while the deepest cold outliers remain the more extreme. A
percentile or detrended rung would not carry this.

## The 21-day-lead episode slate (2026-09-14)

`runs/acal/catalog/conus_episodes_21d_2021_2025.csv` + `EPISODES_21d.md`: 42 CONUS-wide
episodes 2021-2025 (31 heat, 11 cold), nesting exactly to 12 at `|A_L| >= 3 K` and 5 at
`>= 4 K`. Each row carries `init = peak - 21 d` — the ERA5 date to download — and the
strongest 6 deg lattice cell on the peak day.

Storage at 21 d lead: **10.16 GB per case** (0.66 GB global init + 9.50 GB run output at
`RES_N_WALKERS=32`, 7 segments). 427 GB for 42 cases, 122 GB for 12, 51 GB for 5. ERA5
truth is free — the index cubes already cover every peak window.

**The init cannot be CONUS-cropped.** GenCast is global and `walker.check_state` requires
721x1440; `CLAUDE.md` records that CONUS-cropped cubes cannot restart a walker or
initialise FCN3. 673 MB per init is the floor, and it is already only 2 frames.

**Known limitation of this slate: it is not a heat-wave population.** Season split DJF 22,
MAM 9, SON 10, **JJA 1**. A regional dome cannot move a 26x59 deg box mean by 2 K. The
2021 PNW Heat Dome — an `aires/` production event — peaks at +0.83 K CONUS-wide while its
own 6 deg box reads +11.08 K, and is absent. Heat-wave calibration needs the box-level scan
with the declustering fix above, not this.

## Session 2026-09-14/15: staged to the edge of a run, blocked on nodes

Everything below `res` is built and verified. **Nothing has been submitted.**

### Built this session

**`acal/aprep.py` — the missing prep stage.** `aires.walker.initial_state()` resolves inits
through `aconfig.gencast_inputs_path(event)`, keyed by a NAMED event; a calibration case is
an arbitrary (box, peak). aprep builds each case's init with the same primitive xres uses
(`gencast_s2s.data.build_raw_inputs`, which already takes an arbitrary peak/lead), writes
the bytes to `runs/acal/inputs/` and leaves a symlink in
`runs/xres/0p25/week3/inputs/<case>_inputs.nc`. One subprocess per case -- this login node
has 15 GB RAM and one 0.25 deg init costs ~5 GB to assemble; a single-process build dies
around case 16 (the harness culled a background watcher mid-run for memory pressure).

**All 42 inits: built, 28 GB, verified.** 673 MB each, none undersized, 42 symlinks and 0
broken, xres's own 20 input files untouched. One was loaded back through
`walker.initial_state` + `check_state`: PASS, global 721x1440, 2 frames 12 h apart,
`valid_time` == peak - 21 d. Took 144 min, 0 failures.

**`fcn3/fevents.py` — acal case registration.** THE reason the first submission failed:
`run_aires` resolves events through `F.KNOWN`, and 42 init files are not 42 events. The
loader reads the catalog CSV and appends; `KNOWN` 14 -> 56. Verified: `SEED_ORDER[:10] ==
ALL_ORDER`, `[:14] == ALL_ORDER + RES_ORDER`, `seed_for(PNW)` still 333, `selected()` still
6, cold cases give `tail_sign` -1.0. `aires/aindex.py` untouched -- `box_for` falls back to
CONUS, correct for a CONUS-selected slate. Full suite: 680 passed, 1 skipped.

**`slurm/acal_res.slurm`** -- array, task k = case k of the slate filtered to
`|A_L| >= $ACAL_RUNG`. Exports `AIRES_N_WALKERS=32`: `ccfg.RES_N_WALKERS` is acal's INTENT
but the run reads `aconfig.RES_N_WALKERS`, which defaults to the pilot's 64 -- different
env vars, and without the export every case silently runs at double the walkers and ~92 GB
of live states instead of 46.

### Disk

HRRR 2019-2021 deleted (383 GB) -- see `downscaler/docs/HRRR_GAP.md`; the downscaler index
and norm stats are STALE and training crashes at the first batch until rebuilt. Free went
167 -> 550 GB, now **520 GB** after the inits. 12-case slate needs 177 GB, 42-case needs
467 GB (only ~6 GB of slack at the final case -- it wedges rather than fails if someone
else fills /home).

### Why nothing is running

Submitted array 1201 (12-case slate) -> all tasks failed in <25 s on `unknown event`, which
is what produced the fevents work above. Cancelled; nothing left behind. After the fix, the
node survey showed **all 8 nodes occupied** (60-80 GB/card of non-Slurm work) while `sinfo`
reported all 8 `idle` with an empty queue. Nodes 4 and 5 had been free 90 min earlier.
Submitting into that places jobs on full cards and they die in seconds.

**Next step: re-survey, and submit only when a node genuinely has 8 free cards.**

    for n in 0 1 2 3 4 5 6 7; do echo -n "node$n: "; \
      ssh nucla3m-a3meganodeset-$n nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader,nounits | paste -sd, -; done
    sbatch --exclude=<the busy ones> slurm/acal_res.slurm     # 12-case, %2 throttle

## Session 2026-09-16: the 12-case slate is RUNNING

### Node survey (the thing that blocked the last session)

`sinfo` again reported all 8 a3mega nodes `idle` with an empty queue, and again it was
wrong. Real `nvidia-smi` memory, 2026-09-16 00:08 UTC:

    node0: 39783,79115,79115,79115,79115,79115,79115,39409   busy
    node1: 66593,64543,62959,64495,64519,64519,64501,66077   busy
    node2: 33729,79869,79869,79869,79869,79869,79869,31729   busy
    node3: 34489,79949,79949,79949,79949,79949,79949,32509   busy
    node4: 33529,79769,79769,79769,79769,79769,79769,31549   busy
    node5: 34609,80709,80709,80709,80709,80709,80709,32609   busy
    node6: 0,0,0,0,0,0,0,0                                   FREE
    node7: 74525,74525,679,679,4271,4271,6527,909            partial - unusable

**Exactly one node was free.** node7 is the case worth remembering: six of its eight cards
are nearly empty, but two hold 74 GB, and a case needs all 8 (8 walker shards, ~64 GB of
checkpoint per card). Partial nodes are not usable at this resolution -- the arithmetic is
per-card, not per-node.

### Submitted

    sbatch --array=0-11%1 --exclude=nucla3m-a3meganodeset-[0,1,2,3,4,5,7] slurm/acal_res.slurm
    -> array 1209, task 0 = e02_c4_20210218 on node6, log logs/Vayuh-s2s-1210.out

`%1`, not the script's default `%2`: the throttle is bounded by free nodes, and there was
one. If nodes free up, `scontrol update jobid=1209 arraytaskthrottle=2` raises it without
resubmitting -- but re-survey first, and widen `--exclude` accordingly.

Pre-flight before submitting (all passed): 12/12 inits present at 706 MB with 12/12 live
symlinks; `pytest aires/tests/ -q` = 416 passed, 1 skipped (the fevents prefix invariant
holds with `KNOWN` at 56); `df -h /home` = 510 GB free against 177 GB for the sequential
12-case slate.

### What a healthy first task looks like

Task 0 cleared prep and printed `READY: yes`, then:

    [res] e02_c4_20210218 tag=acal N=32 M=6 backend=fcn3 C=(0.0, 1.0, 1.4, 1.8, 2.0)
    [res] host=nucla3m-a3meganodeset-6 shards=8 visible GPUs=8 seed=20260819
      [walk] leg 1: 32/32 segment(s) to roll (0 -> 3 d)

`N=32` on that line is the proof that `AIRES_N_WALKERS` took effect -- it is the one thing
to check on every new task, because the failure mode is silent (64 walkers, double the
GPU-h, ~92 GB of live states). `tail LOW` on a cold event is the other: e02 is
`A_L = -5.176 K`, and a cold case must clone the most negative A_L.

Run's own budget, printed per case: walk 1,344 GenCast steps ~19.4 GPU-h + score 32,256
FCN3 steps ~12.5 GPU-h = **~32 H100-h, ~4 h wall on one node**. At `%1` the 12-case slate
is strictly sequential: **~48 h wall, ~384 H100-h**. The 12 h per-task walltime has ~3x
headroom over the 4 h estimate.

### Note for the next session

`logs/Vayuh-s2s-120{2..8}.{out,err}` are the **previous** session's cancelled array (1201,
the `unknown event` failure). They are dead. The live array's logs start at 1210. Anything
grepping `logs/Vayuh-s2s-12*` will pick up both -- date-stamp or job-id filter before
believing a failure line.

### THE CAMPAIGN IS BLOCKED: a 19.59 GiB device allocation in `write_state`

Array 1209 ran five cases and **all five failed**. This is the blocker; do not resubmit
the slate until it is resolved.

    task 0  e02_c4_20210218   29:13   FAILED 1:0    28 of 32 leg-1 walkers banked
    task 1  e12_c3_20221121    0:31   FAILED 1:0    nothing
    task 2  e13_c3_20221225    0:54   FAILED 1:0    nothing
    task 3  e14_h4_20230104    0:14   FAILED 9:0    nothing (SIGKILL)
    task 4  e23_h3_20231228   34:29   FAILED 1:0    nothing -- exhausted all 3 attempts

Tasks 5-11 are `scontrol hold`-ed (`JobHeldUser`), not cancelled. Release with
`scontrol release 1209_[5-11]` once there is a fix.

**The error, identical on every failing shard:**

    jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to
    allocate 19.59GiB. [executable_name='jit_concatenate']
      walker.py write_state -> ds.to_netcdf -> xarray_jax __array__ -> np.asarray

**It is not the rollout and not host RAM.** Every shard completed all 6 GenCast steps
first; the failure is in materialising the state to write it. Host memory was measured
throughout task 4 and was never near the limit -- peak 859 GB of 1842 GB, **938 GB still
available at the worst moment**, no kernel OOM. An earlier reading of this as a host-RAM
ceiling (8 x ~237 GiB) was wrong: measured per-shard RSS plateaus at ~113 GB, not 237.

**The arithmetic says it should fit, which is why this reads as fragmentation.**
`aires_env.sh` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` with `MEM_FRACTION=.90`, i.e. a
72 GB cap on an 80 GB H100. Steady state is ~42 GB resident, so 42 + 19.59 = ~62 GB
against 72 GB. A contiguous 19.59 GiB block is what is missing, not 19.59 GiB of total
free space -- and `PREALLOCATE=false` is precisely the setting that grows the arena
piecemeal and fragments it.

**The event-dependence is the part that is NOT yet explained, and it matters:**

    e02_c4_20210218  attempt 1:  7 of 8 shards rolled 4 segments each; only shard 1 died
    e23_h3_20231228  attempts 1-3:  8 of 8 shards died, every time, after 6/6 steps

Same code, same node, same 42 GB resident, minutes apart. A pure static-size argument
cannot produce 7/8 on one event and 0/24 on another, so something state- or
timing-dependent is involved. Do not assume a fix works because one case passes.

**Cheap next experiment -- do this before spending another 12 h allocation.** One case,
one shard, one GPU, ~10 min, on any free node:

    XLA_PYTHON_CLIENT_PREALLOCATE=true   # one contiguous arena instead of a grown one
    # and/or MEM_FRACTION=.95 (76 GB), and/or XLA_PYTHON_CLIENT_ALLOCATOR=platform

Run `--stage walk --leg 1 --shard 0 --nshards 8` on e23_h3_20231228 (the reliable
reproducer -- it fails 24 times out of 24) and see whether `write_state` completes. e02 is
a bad test case: it passes 7 times in 8 for reasons not understood.

If the allocator knobs do not fix it, the structural fix is in `walker.py::write_state`:
pull the arrays to host per-variable (`jax.device_get`) before building the Dataset, so
the concatenate happens in numpy rather than as a single 19.59 GiB device allocation.

**Unrelated but real, and it taxes every case:** the JAX persistent compile cache is dead
for this executable --

    Error writing persistent compilation cache entry for 'jit_apply_fn':
    GpuExecutableProto ... must be smaller than 2GiB: 4171376048

The 0.25 deg executable is 4.17 GB against a 2 GiB entry cap, so
`runs/models/jax_cache_0p25` is never populated for it and every shard of every case
re-pays the XLA compile. Not a correctness problem; it is why the compile shows up in
every walk leg instead of once per campaign.

### State at the end of the session (2026-09-16 ~01:20 UTC)

Nothing of ours is running. `runs/aires/e02_c4_20210218/` holds **14 GB of real banked
work** -- 28 of 32 leg-1 walker states, which a resubmit resumes from rather than
recomputing. The other four cases have nothing (40 KB-584 KB of scaffolding). 495 GB free.

**node6 was taken within minutes of our last task clearing it** -- 62-65 GB/card of
non-Slurm work, the same pattern as every other node. So the fix experiment above needs
its own node survey first; at this instant there is no free node on the cluster.

### The fix (2026-09-17) -- COMMITTED, and NOT yet proven on a GPU

**Superseded in part: the 19.59 GiB request is NOT made by `write_state` -- see the
2026-09-17 session below. What follows is the historical record of that reading.**

`aires/walker.py` + `aires/tests/test_walker.py`, committed in **a678a66** (2026-09-17
09:00:36 UTC, 130 insertions over the two files). The heading and this paragraph used to
read "uncommitted ... in the working tree": a28c3f3 froze that text *before* the code
commit landed. The only thing still uncommitted in `aires/walker.py` is the 23-line
`_memstats` probe added later the same morning (next session below).
`git log -S_host_materialize -- aires/walker.py` -> a678a66.

**The real finding is a silent no-op.** `roll_segment` line ~320 reads
`g = _tidy(_to_valid_time(g, init)).compute()   # the one global materialisation`.
It materialises nothing. `xarray_jax.JaxArrayWrapper` implements the NEP-18 duck-array
protocol (`__array_function__`/`__array_ufunc__`), and xarray's `to_duck_array()`
special-cases only chunked dask/cubed arrays -- any other duck array is returned
**unchanged**. So `.compute()` hands back the same device-resident wrapper, every frame
stays on the GPU through the ring buffer and into `write_state`, and `ds.to_netcdf()` is
the first thing that ever touches the values. That is why the traceback always points at
`write_state` even though nothing about writing is wrong.

It also means `roll_segment`'s docstring claim -- "the ring buffer holds two GLOBAL frames
(~0.70 GB total) and nothing else global survives a loop iteration" -- **is not true as
written** for device memory.

**The fix**: `_host_materialize(ds)` (`walker.py:113`), called from `write_state`
(`walker.py:161`) right after `check_state`. It does
`np.asarray(jax.device_get(xarray_jax.unwrap_data(da)))` **one variable at a time** and
rebuilds a plain-numpy Dataset, so only one variable's buffer is touched per transfer and
any concatenation happens in host numpy. `unwrap_data` is required: `.data` on a
duck-typed variable returns the still-wrapped `JaxArrayWrapper`, which `jax.device_get`
cannot unwrap.

**Verified on CPU:** `pytest aires/tests/ -q` = **419 passed, 1 skipped** (was 416+1; three
new tests). Separately, a round trip of a REAL 706 MB init
(`runs/acal/inputs/e23_h3_20231228_inputs.nc`) through `read_state -> write_state ->
read_state` compared every data var, coord, dtype and attr: **CLEAN**, `valid_time`
preserved. That check mattered because `_host_materialize` rebuilds the Dataset and so
drops per-variable `.encoding` (the source file's `time` carries a `calendar` key); the
round trip shows it does not change what lands on disk. Output is 425 MB vs 706 MB in --
compression settings, not lost data.

**What is NOT established.** Nobody has run this on a GPU. Worse, there is an unresolved
inconsistency in the mechanism: JAX dispatches eagerly but surfaces allocation errors at
the first blocking fetch, so a failure *could* originate in the `xr.concat`/`xr.merge` at
`walker.py:335-336` and only surface later -- in which case `_host_materialize`, which runs
afterwards, would merely move where the same error appears. The argument against that:
`ring` holds two global frames totalling ~0.70 GB, so concatenating them cannot plausibly
request **19.59 GiB** (28x the whole state). That points at the allocation being triggered
by the materialisation itself, which is what the fix changes. But it is an argument, not a
measurement.

**So the first GPU run is a test, not production.** One shard, one card:
`--stage walk --leg 1 --shard 0 --nshards 8` on `e23_h3_20231228` (fails 24/24 unfixed).
If it still OOMs, the next move is to make the line-320 materialisation real -- unwrap and
`device_get` inside the rollout loop -- so device residency is capped during the rollout
rather than at the end.

**Still unexplained:** why `e02_c4_20210218` failed 1 shard in 8 while `e23_h3_20231228`
failed 8 in 8, three times over. Do not treat a single passing case as proof of a fix.

## Session 2026-09-17: the diagnosis was wrong, the fix is in the allocator, and it ran

The sections above are the historical record of a misreading: `write_state` never made the
19.59 GiB request -- it is made during the FIRST predictor execution and only surfaces at
the next blocking fetch. The fix that followed was an **allocator setting**, not code, and
it is now measured. **Job 1216 ran both pools on 8 H100s and both PASSED** (COMPLETED 0:0,
39:55, node1, 11:19:43-11:59:38 UTC, empty `.err`). Read "what this does and does not
prove" before calling the blocker closed.

### Job 1215: the node was lost in 6 seconds

All 8 nodes were held by non-Slurm work from 09:09 through 11:16 UTC. The watcher
`slurm/submit_when_free.sh` (started 10:51:27 UTC as pid **2983729**) found node1 with 8
empty cards at 11:16:20 and submitted **job 1215**. It was dead before it compiled:

    11:16:32  the job's own startup guard reads all 8 cards at 0 MB -> proceed
    11:16:38  pid 2543959 (/home/ubuntu/continuum/inkling/venv-hf/bin/python) takes
              80,391 MB on EVERY card -- our shards are still loading the checkpoint
              on the HOST and have claimed nothing
    11:17:17  all 8 shards' arena reservations are refused (cuda_executor ladder
              71.26 GiB -> 16 GiB and below); 7 shards die
    11:17:30  the foreign process itself vanishes
    11:19     we cancel 1215 (sacct: CANCELLED, 2:51)

**A contention guard that runs once at job start is worth about 6 seconds on this
cluster**, and with `PREALLOCATE=true` the arena is reserved at the first ALLOCATION, not
at backend init -- so the ~30 s checkpoint load is a window in which the job holds a node
and no cards. Two fixes went in on the spot: (a) `aires/run_aires.py::stage_walk` does
`jax.device_put(np.zeros(1)).block_until_ready()` as its **first act**, before
`lazy_bundle()`, which moves the reservation to a few seconds after process start and
fails fast on a held card instead of after a ~6 min compile (in 1216 the sampler dated the
claim at **+18 s from the pool launch**); (b) `slurm/acal_walktest.slurm`'s guard became a
function, run at startup **and** immediately before each walk pool. Resubmitted as **job
1216** at 11:19:43 on node1, still empty.

### Job 1216: both pools PASS

- **Pool A** = `e23_h3_20231228` leg 1, `PREALLOCATE=true`, `MEM_FRACTION=.90` -- the
  candidate production config, on the case that failed **24 of 24** shard-attempts on
  09-16: **PASS, 32/32 states, 1,190 s**.
- **Pool B** = `e12_c3_20221121` leg 1, `AIRES_XLA_PREALLOCATE=false` -- the control, old
  behaviour, same node, minutes later: **PASS, 32/32 states, 1,183 s**.

Across all 8 shard logs of both pools: **zero** `cuda_executor` lines, zero
`bfc_allocator` lines, zero `RESOURCE_EXHAUSTED`/OOM lines, zero tracebacks, zero
`Failed to load in-memory CUBIN`. The 5 s `nvidia-smi` sampler saw **no foreign process on
the node for the whole 40 min** (16 python pids, all ours).

Banked: **32 e23 + 32 e12 leg-1 states** (472 MB each) plus 64 `diag.nc`. Disk went
**475 -> 444 GB** free: **16 GB per case** for one leg.

### Memory, measured

JAX `memory_stats` in the shards plus the 5 s per-card sampler:

- **The arena limit is 71.26 GiB = 72,970 MiB in BOTH modes.** `MEM_FRACTION` sets
  `bytes_limit`; `PREALLOCATE` only decides whether it is taken up front. **The fraction is
  applied to what CUDA reports free at init (81,078 MiB), not to `nvidia-smi`'s 81,559
  MiB** -- .90 x 81,078 = 72,970.
- **Pool A**: cards at 73,529 MiB from the claim at 11:20:12, flat through the ~380 s
  compile, then **80,629 MiB from the first execution on -- 930 MiB of the card free at
  steady state (1.1%)**. Non-arena usage is 7,659 MiB: 559 MiB of context/handles plus
  ~7,096 MiB of executable image and workspaces at first execution, of which the
  uncacheable `GpuExecutableProto` is 3,978 MiB.
- **Pool B**: 563 MiB during the compile, then the arena grew to 42,471 MiB at the first
  execution = **34,812 MiB of BFC regions**.
- **Peak in use: 21.33 GiB (A) / 21.49 GiB (B)**, the first walker's peak **bit-identical
  across all 8 shards** (21.17 / 21.36 GiB), growth across 4 walkers **<= 0.16 GiB**
  (allocator ordering, not a leak), `num_allocs` **exactly 2,203 per walker**.
- `write_state:before` == `write_state:after` == `roll_segment:after-loop` on **all 64
  walkers**: **the write allocates nothing on the device**, closing the 09-16 reading.

**The 19.59 GiB request of 09-16 is 91% of the measured 21.33 GiB peak -- it is the
dominant buffer of every normal first execution**, not an anomaly. The "~42 GB resident" /
"~64 GB per card" figures in the older sections were BFC's GROWN REGIONS, not live tensors:
pool B overshot its own peak by **1.58x** (34.8 GiB of regions for 21.5 GiB in use), and
09-16's ~59 GiB arena is the same overshoot carried further.

**Trap now fixed:** the probe printed `in_use=0.00 GiB reserved=0.00 GiB
largest_free_block=0.00 GiB` on all 208 lines of 1216 -- those keys are **absent** on this
JAX build (0.10.2 populates only `peak_bytes_in_use`, `bytes_limit`, `num_allocs`) and
`st.get(key, 0)` rendered absent as zero, which reads as a full arena with nothing
allocatable and misled the first pass over the log. `_memstats` now prints **only present
keys**: `[memstats where] peak=21.33 GiB limit=71.26 GiB num_allocs=2204`.

### What this does and does not prove

**Does:** the case that failed 24/24 passed with **zero driver requests**, and the
computation fits with **3.3x arena headroom** (21.33 GiB peak against a 71.26 GiB limit).
The leg-1 states are banked and schema-valid.

**Does not:** three things changed at once against 09-16 -- preallocation, the walker
hardening (a678a66 + today's), and a clean node1 instead of node6 -- and **pool B did not
reproduce the failure**: it ran `e12`, not `e23`, after A had warmed the caches, and its
arena grew only to 34 GiB, never reaching the ~59 GiB state the driver refused on 09-16.
**So there is no negative control.** "Preallocation is why it passed" is the
best-supported reading -- the old failure was a driver-refused arena EXTENSION and
preallocation makes no extension requests at all -- but it is a reading, not a
measurement. The decisive missing run is **e23 leg 1 under `AIRES_XLA_PREALLOCATE=false` on
a clean node**; e23's leg 1 is now banked, so it needs a fresh tag (`AIRES_RES_TAG=ctrl`):
~20 min of one node, +16 GB.

Best current explanation of 09-16: **environmental** -- a foreign tenant on node6
constraining what the driver would grant, exactly what job 1215 showed happens within
seconds. The startup device claim and the double guard now detect that in seconds.

### Timing, measured

Model load **7-8 s warm** (36 s cold). **XLA compile ~380 s per shard per leg**,
uncacheable (4.17 GB executable against the 2 GiB entry cap) => 6 x 380 s x 8 GPUs =
**5.1 H100-h per case, 29% of the walk**. **196 s per walker** = **32.7 s per 12 h step**
including the 473 MB state write and 34 MB diag; the new per-step host materialisation
costs **<= 0.6 s per walker (<= 0.3%)** against 09-16. Per leg (1 segment) **~1,183 s =
19.7 min**, leg 6 (2 segments) **~32.8 min**, per-case walk **2.19 h wall = 17.5 H100-h**.
The run's printed budget of 19.4 GPU-h is close **for the wrong reason**: steps are 37%
cheaper than budgeted and the recompiles, which the budget omits, eat the difference. The
score stage is still **~94 min estimated** and unexercised.

**The 3-min physical canary did not run for either pool.** `_check_against_reference` needs
an xres reference cube and none exists for any acal case, so **PASS means the process
exited 0 with valid files, not that the atmosphere is right.** Both banks were checked
offline instead: area-weighted global T2m **286.2 / 286.6 K**, MSLP **93-106 kPa**, Z500
and Z50 in range, **32 distinct fields per bank** with CONUS-mean spread **0.20 / 0.23 K**,
NaN only in SST under an identical inherited mask. Specific-humidity minima of
**-8.4e-4 .. -2.5e-4** are the normal GenCast floor (`astab/diag.py` flags at -5e-3), not
divergence.

### MEM_FRACTION: how much of the card to give back

Arena = f x 81,078 MiB; card free = 81,559 - arena - 7,659 MiB of non-arena:

    f     arena MiB   card free MiB   headroom over the 21.5 GiB peak
    .90     72,970            930     3.32x
    .80     64,862          9,038     2.95x
    .75     60,809         13,092     2.76x
    .70     56,755         17,146     2.58x
    .60     48,647         25,253     2.21x
    .50     40,539         33,361     1.84x   (only 4.8 GiB above pool B's grown 34 GiB)

**Recommendation .70 (`AIRES_XLA_MEM_FRACTION=.70`), but only after a qualifying re-run
of pool A** -- e23 needs a fresh tag now, so the cheaper qualifier is the next campaign
case's leg 1: watch `limit=` print **55.42 GiB** and `peak` stay **<= 21.5 GiB**. **.90 is
what passed; leave it as the default until then.** The 930 MiB of card slack at .90 is set
by the executable image, which the campaign does not control.

### Changes in the working tree (all uncommitted)

`git diff --stat`: `acal/HANDOFF.md`, `aires/walker.py`, `aires/run_aires.py`,
`aires/tests/test_walker.py`, `slurm/aires_env.sh`; untracked `slurm/acal_walktest.slurm`,
`slurm/submit_when_free.sh`, `logs/submit_when_free_20260917T105127Z.log`.

- **`slurm/aires_env.sh`** -- preallocate default plus the two override knobs:
  `PREALLOCATE=${AIRES_XLA_PREALLOCATE:-true}`, `MEM_FRACTION=${AIRES_XLA_MEM_FRACTION:-.90}`.
- **`aires/walker.py`** -- `_host_materialize` upcasts float16/bfloat16 to float32 (netCDF4
  has no bfloat16; `ml_dtypes` reports kind `V`) with a corrected docstring; `_write_nc` =
  mkdir + `.nc.tmp.<pid>` + `os.replace` + `except BaseException: unlink; raise`; the diag
  cube is materialised like the state; `roll_segment`'s per-step `.compute()` (a no-op on
  `JaxArrayWrapper`) is a real materialisation; `_memstats` prints only the keys the
  backend reports, and an `AIRES_MEMSTATS`-gated UTC stamp prefixes the `step k/n` and both
  `wrote ...` lines (default log format unchanged).
- **`aires/run_aires.py`** -- the `stage_walk` device claim above.
- **`aires/tests/test_walker.py`** -- +5 tests: **424 passed, 1 skipped** full suite.
  **Trap: the walker suite takes ~5 min on this 15 GB login node.**
- **`slurm/acal_walktest.slurm`** the two-pool rig that produced 1216;
  **`slurm/submit_when_free.sh`** the login-node watcher (submits ONCE and exits; never run
  two at a time).

**A `git checkout` of any of these mid-campaign silently reverts the tested
configuration.** The other `PREALLOCATE=false` copies -- `slurm/aires_gate3.slurm`,
`slurm/xres_pool_*.slurm`, `slurm/gencast_week*.slurm` -- were deliberately left alone.

### Resume state, disk, and the tmp sweep

Leg 1 banked: **e02 28/32** (w01, w09, w17, w25 missing = shard 1's list), **e23 32/32**,
**e12 32/32**; the other 9 cases have nothing. **444 GB free.**

**The 25 orphan `.nc.tmp.<pid>` files from the 09-16 OOM were swept today** -- 24 at
exactly 9,786 B under `e23_h3_20231228/.../w0?/step01/`, 1 at 9,741 B under
`e02_c4_20210218/w01/step01/`, all header-only, all dated 09-16. `find runs/aires -name
'*.nc.tmp.*'` is now empty and the 32 e23 + 28 e02 `state.nc`/`diag.nc` are untouched.
`_write_nc`'s `try/except` is what stops them recurring.

**The prune trap from the previous section still applies**: a finished case keeps ~55 GB
(45.8 GB states + 7.8 GB diag + ~1.9 GB score cubes) because `prune_states` cuts only to
`first_segment - 2`, and `stage_prep`'s `need < 0.8 x free` guard fails around the 8th-9th
case. **Prune as cases land, or submit in halves. Do not submit 0-11 and walk away.**

### Next command

**Nothing of ours is queued and nothing ran after 1216.** The slate needs a fresh `sbatch`
(1209 is gone). Survey and submit in one breath -- K = nodes showing 8 empty cards:

    for n in 0 1 2 3 4 5 6 7; do echo -n "node$n: "; \
      ssh nucla3m-a3meganodeset-$n nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader,nounits | paste -sd, -; done
    sbatch --array=0-11%K --exclude=nucla3m-a3meganodeset-[<the busy ones>] \
           --export=ALL,AIRES_MEMSTATS=1 slurm/acal_res.slurm

Or let the watcher place it:

    LOG=logs/submit_when_free_$(date -u +%Y%m%dT%H%M%SZ).log
    nohup setsid bash slurm/submit_when_free.sh --array=0-11%1 \
      --export=ALL,AIRES_MEMSTATS=1 slurm/acal_res.slurm > "$LOG" 2>&1 < /dev/null &

**Arg order verified today** against the script: usage is
`submit_when_free.sh [--survey-only] <sbatch args ...>` and it runs
`sbatch --nodelist="<node>" "$@"`, so extra args pass through **after** its own
`--nodelist` and `--survey-only`, if used, must come first. Keep `AIRES_MEMSTATS=1` on --
it is now the only clock in the walker log.

Two optional follow-ups, priced: **the negative control** (`AIRES_RES_TAG=ctrl`, e23 leg 1,
`AIRES_XLA_PREALLOCATE=false`, one node, `--attempts 1`) at ~20 min and +16 GB, the only
run that can turn "preallocation is why it passed" into a measurement; and **the .70
qualifier**, free if it rides the next campaign case's leg 1 -- set
`AIRES_XLA_MEM_FRACTION=.70`, confirm `limit=55.42 GiB` and `peak <= 21.5 GiB` in the first
shard log, and the campaign gains ~17 GB of card slack per GPU.

## Campaign analysis (2026-10-01)

**All 42 cases done (finished 2026-09-30), analysed on the Derecho login node, CPU only.**
Module `acal/analyze.py` (docstring = the method), tests `acal/tests/test_analyze.py`.

    module load conda && conda activate my-env
    python -m acal.analyze --stage all      # collect -> scorecard -> figures, ~minutes

Outputs: `runs/acal/analysis/cases.csv` (42 rows: catalog + log_Z, norm check, ESS,
founders, walker means, wall/host/job), `runs/acal/analysis/scorecard.csv` (42 rows, per-case
probabilities and lifts), `runs/acal/analysis/summary.json` (pooled medians/IQRs);
`figures/acal/acal_{scorecard,lift_by_rung,pit,health,curves_grid}.png`.

**Definitions.** `P_RES(obs) = Z * mean(w * 1[s A >= s obs])`, `>=` as in `stage_compare`
(`DMCResult.exceedance` is strict `>`); `s = run.json tail_sign`, so cold cases are scored on
-A. Self-normalized `P_sn = P_raw / normalization_check`. Climatology = 2021-2025 daily CONUS
A_L (`runs/acal/catalog/conus_daily_2021_2025.csv`), days within +/-30 d of the peak
anniversary minus the case's own +/-10 d (~284 days). Checks: 42/42 recomputed peak A_L match
the catalog within 1e-3 K; estimator matches `compare_curve.csv` to 5e-6.

**Numbers** (median [IQR], from `summary.json`):

| quantity | value |
|---|---|
| P_RES(obs) raw | 0.101 [0.042, 0.177], n=42 |
| P_RES(obs) self-normalized | 0.116 [0.043, 0.338] |
| P_clim(obs) | 0.035 [0.018, 0.077] |
| lift raw = P_RES/P_clim | **1.95 [0.83, 3.86]**, n=36; >1 in **25/36** |
| lift self-normalized | 2.42 [0.56, 6.63], n=36; >1 in 26/36 |
| lift raw, conservative (P_clim=0 -> 1/284) | 2.28 [0.91, 4.64], n=42; >1 in 30/42 |
| lift at the case's rung threshold | 1.60 [1.04, 2.62], n=40; >1 in 31/40 |
| lift by rung 2/3/4 K | 1.95 (n=27) / 2.38 (n=6) / 1.94 (n=3) |
| lift heat / cold | **2.59 (n=27) / 1.28 (n=9)** |
| PIT self-normalized | 0.88 [0.66, 0.96]; 19/42 >= 0.9 |
| PIT raw | 0.49 [0.28, 1.00]; 11/42 > 1 (raw is not a probability) |

Lift undefined (climatology never reached obs): e02, e03, e04, e12, e30, e42.
Unresolved, P_RES(obs) = 0 (no walker reached obs): e11, e27. Saturated (all 32 walkers
beyond obs): e03, e07, e21, e32. normalization_check outside [0.25, 2]: e05 2.29, e07 0.18,
e21 0.22, e23 3.16, e27 0.16, e28 2.91, e30 0.19, e42 0.17.

**Reading.** AI+RES at 21 d puts ~2x more mass than 2021-2025 climatology on the tail that
was observed, in about 2 of 3 cases, more for heat than for cold. The self-normalized PIT
piling up near 1 is mostly the selection, but it also fits a RES-weighted forecast that is
underdispersed or pulled toward climatology at 21 d.

**Caveats - state these wherever the numbers go.**
1. **Selected on outcome** (every case |A_L| >= 2 K): no reliability diagram is possible.
   Lift > 1 means more mass on the observed tail than climatology, NOT calibrated
   probabilities.
2. **Climatology is 2021-2025 only**, and the anomaly is against 1990-2019, so the warming
   trend sits inside it: heat days are commoner in the pool, which lowers heat lift.
3. **The slate is continental winter swings, not heat domes**: DJF 22, SON 10, MAM 9, JJA 1.
4. **N=32** resolves probabilities only down to ~1/32 times the weights (hence e11/e27 at 0).
5. **Estimator variance**: the normalization_check spread (0.16-3.16) is the size of it;
   raw vs self-normalized lifts differ by up to 3x per case.

**Anomaly verdicts (all benign).**
- **e27-e42 "2x disk"**: did not reproduce. Every case is ~9.0-9.1 GB, same float32/zlib4
  encoding. The 18 GB `du` reading coincided with the e27-e42 sync (16:27-17:05 today). No
  aires/fcn3 commits 2026-09-18 to 10-01.
- **e33 scores/ ~15 GB**: 8 orphaned FCN3 zarr scratch stores
  `runs/aires/e33_h4_20250101/res/acal/scores/w0{0..7}_lead06.zarr` (~5.5 GB, mtime
  2026-09-18 02:05-02:09), from job 1246 cancelled mid score-leg02. The result is from clean
  job 25512. Deletable; NOT deleted.
- **e24/e25 ~31 min walls** (jobs 2074/2075): resumes. Jobs 1224/1225 had banked 182/224 and
  176/224 walker segments and 128/128 score cubes before cancellation 2026-09-18T08:09:36;
  only walk leg 6 ran. Configs unchanged (N=32, base_seed 20260101, C 0/1/1.4/1.8/2).
- **e14 2.37 h wall** (job 1222, after 1221 failed): resume, confirmed from
  `res_result.json` timings - walk1-3 and score1-2 took <0.02 s each (cache hits); walk4-6
  and score3-5 ran.

**Side fix.** `aires/awalkers.py::discover()` now excludes tag `acal` (the 42 acal runs had
broken 4 `test_ainsights` tests). aires + acal tests: 432 passed, 1 skipped.

**Next steps.**
1. **Non-event controls** (same lead, |A_L| < 2 K peaks): the only route to a true
   reliability test. a3mega GPU, ~17 H100-h per case.
2. **A per-case box-level slate for heat domes** (regional box A_L, JJA), since the CONUS
   slate is winter swings.
3. Optional: delete the e33 orphan zarr stores (~5.5 GB) - user decision.

### Per-rung calibration (2026-10-01)

`python -m acal.analyze --stage rungs` (also in `all`; ~20 s CPU). Outputs:
`runs/acal/analysis/rungs_cases.csv` (96 case x rung rows), `rungs_sharpness_2k.csv`,
`rungs_summary.json`; `figures/acal/acal_rungs_{reliability,counts,sharpness}.png`.

**Method.** Conditional reliability: selection is on s*obs >= 2 K, so test
q_i(a) = F_i(a)/F_i(2) = sum(w 1[sA>=a]) / sum(w 1[sA>=2]) against o_i = 1[s*obs >= a].
Unbiased if F_i is calibrated and selection depends only on the 2 K outcome (deviations:
10-day declustering; the valid date is the observed peak). Z cancels, so q is the same for
raw and self-normalized. Count test = exact Poisson-binomial, two-sided p = 2 x min tail.
q intervals: 2000-walker bootstraps per case (5-95%). q_clim from the same +/-30 d pool;
all-season fallback when the pool has < 5 days at the conditioning rung (2 cases: e04, e12).

| rung | n | expected [boot 90%] | observed | 90% range | p | BSS vs clim | clim expected |
|---|---|---|---|---|---|---|---|
| 3\|2 all | 42 | 14.45 [14.2, 17.3] | 12 | 10-19 | 0.44 | -0.26 | 10.7 |
| 3\|2 heat | 31 | 9.31 | 6 | 6-13 | 0.16 | -0.15 | 4.6 |
| 3\|2 cold | 11 | 5.14 | 6 | 3-7 | 0.79 | -0.42 | 6.1 |
| 4\|2 all | 42 | 3.37 [3.1, 4.7] | 5 | 1-6 | 0.44 | +0.38 | 2.6 |
| 4\|2 heat | 31 | 2.08 | 3 | 0-4 | 0.66 | +0.38 | 0.75 |
| 4\|2 cold | 11 | 1.29 | 2 | 0-3 | 0.76 | +0.38 | 1.8 |
| 4\|3 all | 12 | 4.46 [3.9, 5.6] | 5 | 2-7 | 0.96 | +0.36 | 2.7 |
| 4\|3 heat | 6 | 2.28 | 3 | 1-4 | 0.82 | +0.22 | 1.0 |
| 4\|3 cold | 6 | 2.18 | 2 | 1-4 | 1.00 | +0.52 | 1.7 |

**Reading.** No rung rejects calibration (all p >= 0.16). At 3|2 AI+RES over-forecasts
heat (9.3 expected vs 6) and has WORSE Brier than climatology (BSS -0.26), driven by two
confident misses with q = 1.0 (e07_h2: all 32 walkers >= 3 K, obs 2.81; e20_c2: all 7
walkers past 2 K also past 3 K) plus e31/e32/e35/e16 at q 0.56-0.73. At 4 K it beats
climatology (BSS +0.38 conditional on 2, +0.36 on 3): e33_h4 got q = 0.73, e02_c4 0.34,
while climatology gave 0.04-0.08. 12 q == 0 at 4|2 (no walker reached 4 K), all misses.

**Undefined q: 0.** Every case has >= 1 walker at 2 K (e11_h2 has exactly 1, F(2) = 4e-4).
**Rung 2 (sharpness only):** median F(2) raw 0.18, self-normalized 0.35, P_clim 0.12;
lift_sn > 1 in 30/42 (heat 22/31, cold 8/11). 16/42 have F_sn(2) >= 0.5.

**Caveats.** n is small (12 hits at 3 K, 5 at 4 K): a 2x miscalibration would not be
detected. N=32 sets q resolution (~1/32 times weights; 3 q == 1 at 3|2). The 3|2 and 4|2
pairs share cases, so the pooled reliability bins are not independent. The bootstrap of
the expected count is right-skewed (dropping a heavy walker between 2 and 3 K raises q).
Climatology is 2021-2025 with the warming trend inside it; the slate is mostly winter.
`pytest` must be run as `python -m pytest` in `my-env` (bare `pytest` cannot import
`acal`). acal + aires tests: 436 passed, 1 skipped.

## CFSv2 operational baseline (2026-10-05/06)

**Built and run on the Derecho login node, CPU + internet, env `my-env`.** Question: on the
same 42 cases, same 21 d lead, same CONUS `A_L`, does AI+RES put more probability on the
observed tail than NCEP CFSv2? Plan: `acal/CFS_PLAN.md`. Code: `acal/cfsbase.py` (docstring =
the method), `--stage cfs` of `acal/maps.py`, tests `acal/tests/test_cfsbase.py`
(acal + aires: 449 passed, 1 skipped).

    module load conda && conda activate my-env
    python -m acal.cfsbase --stage build             # 42 cubes, ~50 min (download), 2.8 GB
    python -m acal.cfsbase --stage hind --workers 4  # 191 (case, year) jobs, ~1 h, 148 MB
    python -m acal.cfsbase --stage bias              # seconds
    python -m acal.cfsbase --stage all               # score + paired, seconds
    python -m acal.maps --stage cfs                  # CFS fields + figures, ~5 min

`runs/acal/cfs/` is 3.0 GB in total (cubes, `hind/`, `bias.nc`).

**Ensemble.** 16 trailing 6-hourly cycles ending on the AI+RES init (leads 21.0-24.75 d),
equal weights, 42/42 cases with 16 members and no skipped cycle. Anomaly = CFS minus the ERA5
1990-2019 climatology, raw, as for the walkers. The last 4 members are the aires-convention
subset (`sub_emp`).

**Outputs.**
- `runs/acal/cfs/`: `<case>_cfs16.{nc,json}`, `build.csv` (members, `al_mean`, `n_reach`,
  archives), `hind/` (per case x year), `bias.nc` (CONUS scalar and 105x237 field bias, 7-day
  and daily), `bias.csv` (per case: years, `bias_conus`, `bias_sd`).
- `runs/acal/analysis/`: `cfs_scorecard.csv`, `cfs_paired.csv`, `cfs_summary.json` (variants +
  `paired`), `maps_fields_cfs.nc`, `maps_daily_cfs.nc`.
- `figures/acal/`: `acal_cfs_scorecard.png`, `acal_cfs_paired.png`,
  `acal_map_{accuracy,bss}_cfs.png`, `acal_map_{pod,csi}_{7d,daily}_cfs.png`,
  `acal_map_bss_cfscorr.png`, `acal_map_csi_{7d,daily}_cfscorr.png`,
  `acal_map_{bss,csi}_{7d,daily}_diff_cfs.png` (AI+RES minus CFS raw, red = AI+RES better).

**Environment.** `cfgrib` and `eccodes` were pip-installed into `my-env` on 2026-10-05; the
env had no GRIB reader. `aires/cfs.py::build` gained an optional `out=` path, because
`aconfig.cfs_cube_path` does not encode the cycle count and a 16-member cube there would have
replaced the 4-member aires cube. No other aires or existing acal output was modified.

**Validation.**
- The own-year hindcast reproduces the build cube's member `A_L` bit-identically.
- ERA5 `A_L` of e02_c4 matches the catalog (-5.176 K).
- Maps: CFS 7-day CONUS field means match the cube `al` to 1e-3 K (asserted in
  `cfs_member_fields`).
- The AI+RES maps reproduce byte-identically after the maps refactor.

**CFS bias (leave-one-year-out, `bias.csv`).** Mean CFS minus ERA5 CONUS bias -0.67 K over
42 cases (heat -0.57 K, n=31; cold -0.96 K, n=11); negative in 37/42 cases (counted from
`bias.csv`). 4-5 other years per case. Year-to-year sd of the bias is ~1.3 K (mean of
`bias_sd`), so ~0.6 K per-case uncertainty on a 4-5 year mean. Part of the negative bias is
the warming trend: the anomaly is against 1990-2019 and the hindcast years are 2021-2026.
Hindcast member spread averages 1.25 K.

**Scorecard** (median, `cfs_summary.json`; AI+RES self-normalized p_obs 0.116, raw lift 1.95,
>1 in 25/36). Lift against the same `P_clim` (median 0.035); "lift>1" counts the 36 cases
with defined lift, "cons" counts the conservative lift (P_clim = 0 -> 1/284) over 42.

| variant | p_obs | lift (n=36) | lift>1 | lift cons | cons>1 (heat/cold) | PIT | zero-obs cases |
|---|---|---|---|---|---|---|---|
| raw_emp (headline) | 0.0625 | 0.58 | 15/36 | 0.41 | 17/42 (10/7) | 0.938 | 20 |
| raw_gauss | 0.043 | 1.08 | 18/36 | 1.64 | 23/42 (15/8) | 0.957 | 0 |
| corr_emp | 0.0625 | 1.87 | 22/36 | 1.78 | 24/42 (20/4) | 0.938 | 15 |
| corr_gauss | 0.087 | 1.58 | 21/36 | 2.12 | 26/42 (22/4) | 0.913 | 0 |
| sub_emp (4 members) | 0.0 | 0.0 | 6/36 | 0.0 | 8/42 (5/3) | 1.0 | 34 |

Raw empirical lift by family (median): heat 0.32 (n=27), cold 1.78 (n=9). Corrected empirical:
heat 2.22, cold 0.59. Correction lifts heat and lowers cold, consistent with the cold-biased
raw CFS (it already puts mass on cold tails and loses it when the bias is removed).

**Paired head-to-head (AI+RES self-normalized vs CFS; log ratio log(P_RES/P_CFS), positive =
AI+RES better; Wilcoxon p; bootstrap 90% CI of the mean).**

| comparison | subset | mean log ratio [90% CI] | W/L | p |
|---|---|---|---|---|
| vs raw_emp | all (42) | +0.49 [0.19, 0.79] | 24/18 | 0.026 |
| vs raw_emp | heat (31) | +0.78 [0.43, 1.12] | 21/10 | 0.003 |
| vs raw_emp | cold (11) | -0.33 [-0.71, 0.06] | 3/8 | 0.24 |
| vs corr_emp | all | +0.18 [-0.09, 0.44] | 24/18 | 0.33 |
| vs corr_emp | heat / cold | +0.20 / +0.12 | 18/13, 6/5 | 0.30 / 0.50 |
| vs raw_gauss | all | +0.47 [0.17, 0.76] | 25/17 | 0.026 |
| vs sub_emp | all | -0.53 [-0.81, -0.24] | 14/27 | 0.007 |

Brier (CFS minus AI+RES, positive = AI+RES better), vs raw_emp, all cases:
- 2 K: +0.270 (30 wins of 42, p = 0.001), heat +0.366, cold +0.001. Every case has o = 1 at
  2 K by selection, so this is only mass on the observed tail. vs corr_emp it is +0.203.
- 3 K: +0.019 [-0.047, 0.083], p = 0.69, W/T/L 14/3/25. Mean Brier 0.224 (AI+RES) vs 0.243.
- 4 K: +0.026 [-0.001, 0.064], p = 0.11, W/T/L 6/12/24. Mean Brier 0.077 vs 0.103.
- vs corr_emp: 3 K +0.006 (p = 0.95), 4 K +0.010 (p = 0.23).
The median Brier difference at 3 and 4 K is ~0 (most cases are ties or tiny); the means are
carried by a few confident misses, as in the rung calibration above. By rung, AI+RES is
worse at 3 K and 4 K for the rung-2 cases (mean -0.061, p = 0.0005, and -0.003, p = 0.016).

**Gridpoint maps** (`maps_fields_cfs.nc`, 7-day mean; thresholds +K = heat, -K = cold;
cos-latitude land mean, median in brackets for BSS). CFS is ~0.94 deg regridded to 0.25 deg.

| 7-day BSS | +2 K | +3 K | -2 K | -3 K |
|---|---|---|---|---|
| AI+RES | 0.211 [0.226] | 0.035 [0.175] | 0.058 [0.221] | -0.694 [0.159] |
| CFS raw | -0.140 [-0.082] | -0.666 [-0.063] | -0.014 [0.261] | -5.533 [0.225] |
| CFS corrected | 0.042 [0.053] | 0.032 [0.065] | 0.000 [0.087] | -1.742 [0.083] |

7-day CSI land mean at +2 K / -2 K: AI+RES 0.559 / 0.321, CFS raw 0.304 / 0.311, CFS
corrected 0.420 / 0.171. At +4 K / -4 K BSS means are dominated by rare cells (AI+RES -0.37 /
-2.03, CFS raw -8.89 / -27.94): use the medians (AI+RES 0.126 / 0.114, CFS raw -0.026 /
0.170). Daily means (BSS median, +2 K / -2 K): AI+RES 0.030 / 0.052, CFS raw -0.062 / 0.109,
corrected -0.074 / -0.044; daily CSI at +2 K: 0.481 vs 0.325 (raw) vs 0.406 (corrected).

**Reading.** AI+RES beats raw CFSv2 on mass-on-the-observed-tail (paired log ratio +0.49,
p = 0.026), and the gap is driven by heat (+0.78, 21 of 31 wins). Most of the gap is CFS cold
drift: against the bias-corrected CFS the log ratio falls to +0.18 (p = 0.33, CI spans zero)
and the 2 K Brier gap falls from +0.270 to +0.203. There is no Brier difference at 3 K or
4 K (p = 0.69, 0.11). Against the 4-member subset CFS is better (-0.53), which shows the 16
member lag ensemble is the fair baseline, not the 4-cycle one. At the gridpoint level AI+RES
is better for heat broadly (7-day BSS median +0.226 vs -0.082 at +2 K), while raw CFS is
slightly better for cold 7-day BSS (median 0.261 vs 0.221 at -2 K; land mean -0.014 vs 0.058
goes the other way, so this is a tail-cell effect).

**Caveats - state these wherever the numbers go.**
1. **Selected on outcome** (all |A_L| >= 2 K): the comparison says which forecast put more
   mass on what happened, not which is calibrated; it is silent on false alarms.
2. **Not the same ensemble**: 16 equal-weight lagged members (0-3.75 d staler than the init)
   vs 32 importance-weighted walkers, with a different resolution of probabilities (1/16 vs
   1/32 times the weights). Raw empirical CFS has 20 of 42 cases at P(obs) = 0.
3. **Resolution**: CFS is ~0.94 deg. Negligible for the CONUS index (<0.17 K), not for the
   gridpoint maps, where a coarse model cannot verify sharp local anomalies. Printed on the
   figures.
4. **The bias correction is noisy**: 4-5 years per case, ~0.6 K per-case uncertainty, and it
   absorbs the warming trend against 1990-2019 and the 6 h vs 12 h frame sampling. Treat the
   corrected rows as a bound on drift, not as the better forecast. Cases share seasons (DJF
   22), so the bootstrap CIs are optimistic.
5. **Archive**: NCEI lacks 2024 and Dec 2025, so the AWS mirror was used (from 2023-04-22).
   Archives per case: NCEI 15, AWS 15, both 12. 7 cycles lacked an inventory file and were
   fetched whole.

**Next steps.**
1. Optional tighter bias: add +/-7 d and +/-14 d inits in the other years (~5x samples,
   ~3 h download).
2. Non-event controls remain the route to real reliability for both forecasts.
3. Before the next GPU campaign: every walk leg recompiles GenCast (~8.6 min/leg, ~7
   H100-h per case) because the 4.17 GB executable exceeds the JAX cache's 2 GiB limit.
   Persistent walk workers would recover ~40 min wall per case. NOT done; details and fix
   plan in `aires/HANDOFF.md`, "Open problems" item 8.

## AI+RES vs CFSv2 side by side (2026-10-07)

Code: `acal/sidebyside.py` (docstring = the method), tests `acal/tests/test_sidebyside.py`.
CPU only, reads what `acal.maps` and `acal.cfsbase` already built.

    python -m acal.sidebyside --stage csi       # 6 figures, ~1 min
    python -m acal.sidebyside --stage members   # ~40 min (reads every walker diag.nc)
    python -m acal.sidebyside --stage figures   # redraw member maps from the .nc, ~2 min

**Outputs.**
- `figures/acal/sidebyside/csi_{heat,cold}_{2,3,4}K.png`: per threshold, rows 7-day / daily,
  columns AI+RES | CFSv2 raw | difference. Same `maps.scores` as the existing CSI maps (the
  land means reproduce them: +2 K 7-day 0.56 vs 0.30). Fixed scales ([0, 1], +/-0.6).
- `figures/acal/sidebyside/members/<case>.png` (42) and `closest_member_summary.png`:
  ERA5 | closest AI+RES walker | closest CFSv2 member, plus both ensemble means.
- `runs/acal/analysis/closest_members.{csv,nc}`: per case the closest walker / member,
  RMSE, pattern correlation, A_L, walker weight and rank, CFS member lead.

**Method.** Closest = smallest cos(lat)-weighted RMSE of the 7-day-mean T2m anomaly vs ERA5
over CONUS land (`maps.land_mask`). CFS raw. `res_min16_exp` = expected minimum walker RMSE
over random 16-walker subsets (exact order statistic), the like-for-like comparison with
16 CFS members.

**Result (42 cases).**

| | AI+RES | CFSv2 raw | AI+RES closer |
|---|---|---|---|
| closest-member RMSE, median | 2.28 K (E16 2.42 K) | 2.92 K | 38/42 (16 vs 16: 35/42) |
| closest-member pattern r, median | 0.72 | 0.58 | 36/42 |
| ensemble-mean RMSE | | | 34/42 |

Heat 26/31 and cold 9/11 like-for-like. The closest walker is often a low-weight one (in the
top 8 of 32 by weight in only 22/42 cases), so "the ensemble contains a physically close
trajectory" is a weaker statement than "the forecast favoured it". CFS caveats as above
(cold drift inside the raw score, ~0.94 deg resolution).

## Multi-model board: CFSv2, GEFSv12, ECCC GEPS, ECMWF IFS (EC46, pending), BB-SUBS estimate (pending), ERA5 and HRRR truth (2026-10-07/08)

**Built and run on the Derecho login node, CPU + internet, env `my-env`.** The question is whether, on the same
42 cases, at the same 21 d lead and on the same CONUS `A_L`, AI+RES still puts more
probability on the observed tail than the operational S2S ensembles, and whether the verdict
holds when the truth is the HRRR analysis instead of ERA5. This section extends "CFSv2
operational baseline (2026-10-05/06)" from one baseline to a source registry, and from one
truth to three. The reader-facing one-page version is
`runs/acal/analysis/s2s/board/SUMMARY.md`.

Board rows with data are AI+RES, CFSv2 (`cfs13`, the 16 published CFS members re-reduced on
13 frames; the published 25-frame `cfs` rows are kept as a reference), GEFSv12 and ECCC
GEPS. ECMWF IFS (EC46) and the BB-SUBS estimate are PENDING the user's ECDS token. Their
code is written and tested, and no data exists yet. Truths are `era5`, `hrrr` (the headline
HRRR truth, offset-corrected) and `hrrr_raw` (sensitivity). The headline variant is raw
(`raw_emp`); bias-corrected (`corr_emp`) is the sensitivity, as in the CFS section.

**Code.** New modules (each docstring = the method):
- `acal/s2sbase.py` source registry, window reducers, per-source json / LOYO bias / score /
  paired / maps stages; `acal/truth.py` + `acal/hrrrtruth.py` truth providers.
- `acal/s2s_gefs.py`, `acal/s2s_geps.py`, `acal/s2s_ec46.py` per-source fetch, canonical
  cubes and hind cubes (EC46 also its reforecast bias).
- `acal/board.py` (the one table), `acal/overall.py` (headline figures), `acal/errmaps.py`
  (error and skill maps), `acal/bbsubs.py` (published BB-SUBS numbers + estimator),
  `acal/tilt.py` (steering check); driver `scripts/acal_board_all.sh`.
- `analyze`, `cfsbase`, `maps`, `reach`, `roc`, `sidebyside` gained `truth` / `source`
  arguments. With ERA5 and source `cfs` they reproduce the published tables byte-identically
  (scorecard, rungs, cfs_scorecard, cfs_paired, prob_4K, closest_members) and the published
  ROC/reach figures pixel-identically. Nothing under `runs/acal/cfs/`,
  `runs/acal/analysis/*.csv` or `figures/acal/*.png` was rewritten.
- Tests are `acal/tests/test_{s2sbase,truth,hrrrtruth,s2s_gefs,s2s_geps,s2s_ec46,score_sources,board,overall,errmaps,bbsubs,tilt,daily_pairing}.py`
  plus the existing files, 19 files in all. 259 passed on 2026-10-08, run in two groups of files
  (124 + 135, ~25 s and ~38 s). Run per file or in groups; one pytest process over all of
  `acal/tests` was OOM-killed on the login node once.

The commands below run from the repo root in `my-env`, inputs first, then the driver.

    module load conda && conda activate my-env
    cd /glade/derecho/scratch/exu/S2S_ExtremeWeather

    # inputs, once (all cache-aware; --force rebuilds)
    python -m acal.hrrrtruth --stage fetch2026 --workers 8   # HRRR 2026-01..08 from AWS, ~1 min
    python -m acal.hrrrtruth --stage build --workers 8       # HRRR index 2021-2026, ~4 min
    python -m acal.hrrrtruth --stage cases                   # cases_hrrr.csv, ~10 s
    python -m acal.s2s_gefs --stage build                    # 42 cubes from AWS, ~25 min from scratch
    python -m acal.s2s_gefs --stage hind                     # 191 LOYO hind cubes (11 members)
    python -m acal.s2s_geps --stage fetch --workers 4        # SubX raw download
    python -m acal.s2s_geps --stage build                    # ~20 s
    python -m acal.s2s_geps --stage hind --workers 8         # ~20 s
    python -m acal.s2sbase --stage aires_mask --truth hrrr --workers 8   # walker A_L on the HRRR mask, ~3 min

    # everything downstream: json, bias, score, paired, maps, roc, reach, sidebyside,
    # per-source figures, then board, errmaps, overall, bbsubs (~25 min from scratch,
    # minutes when cached; exit code = number of failed steps)
    setsid nohup bash scripts/acal_board_all.sh > runs/acal/analysis/s2s/logs/board_all.log 2>&1 &
    python -m acal.tilt --stage all                          # steering check, NOT in the driver

    # single steps
    python -m acal.s2sbase --source gefs --stage score --truth hrrr    # likewise paired, maps
    python -m acal.roc --source gefs --truth hrrr --figures
    python -m acal.board --stage all                         # ~2 min, truths in parallel
    python -m acal.overall                                   # forest, scoreboard, hrrr_vs_era5
    python -m acal.errmaps --truth era5 --stage all          # also --truth hrrr, hrrr_raw
    python -m acal.bbsubs                                    # context chart, estimate, method note

**EC46, one command once `~/.ecdsapirc` exists.** The file holds two lines,
`url: https://ecds.ecmwf.int/api` and `key: <token>` (chmod 600), after the S2S licence is
accepted on the Download tab of BOTH `s2s-forecasts` and `s2s-reforecasts`. It was absent on
2026-10-08.

    setsid nohup python -m acal.s2s_ec46 --stage fetch --workers 4 \
        > runs/acal/s2s/ec46/logs/fetch.log 2>&1 < /dev/null &
    #   168 ECDS requests, then 42 cubes, reforecast hind cubes, bias and verify;
    #   resumable (rerun the same command); exit 3 = still no token
    python -m acal.s2s_ec46 --stage verify
    bash scripts/acal_board_all.sh       # picks ec46 up via s2sbase --stage with_data; the
                                         # BB-SUBS estimate rows follow from acal.board
    # then add "ec46" to acal.tilt.MODELS and rerun: python -m acal.tilt --stage all

The request plan (`runs/acal/s2s/ec46/requests.csv`) was validated for e01 and e42 against
the anonymous ECDS costing endpoint, and ERA5 at all 840 hindcast dates is already built
(`runs/acal/s2s/ec46/era5_hdates/`), so only the download is missing.

**Coverage** (`board_summary.json` "coverage"; EC46 from `runs/acal/s2s/ec46/requests.csv`;
BB-SUBS from `board/bbsubs_method.md`).

| row | data | cases | members | lead to peak | scored window (truth window) | native grid | corrected variant |
|---|---|---|---|---|---|---|---|
| AI+RES | GenCast walkers, FCN3 score | 42 | 32 importance-weighted walkers | 21 d | 13f; 12f when paired with a daily-mean source | 0.25 deg | none |
| CFSv2 (`cfs13`) | NCEP operational 6-hourly tmp2m (NCEI/AWS) | 42 | 16 trailing 6-hourly cycles | 21.0-24.75 d | 13f (13f) | ~0.94 deg | published 25-frame LOYO bias |
| GEFSv12 | AWS `noaa-gefs-pds`, 00Z, 0.5 deg pgrb2ap5 | 42 | 31 | 21 d (lag 0) | 13f (13f) | 0.5 deg | LOYO, 11-member hind, 4-5 other years |
| ECCC GEPS 6/7/8 | IRIDL SubX daily-mean `tas` | 42 | 21 | 21-27 d | d6 = UTC days peak-6..peak-1 (12f) | 1 deg | LOYO, 4-5 other years |
| ECMWF IFS (EC46) | ECDS `s2s-forecasts` | 0 of 42 (planned 42) | 51 (17 cases) or 101 (25 cases) | 21 d (33 cases), 22-24 d (9) | d6 (12f) | 1.5 deg | reforecast model climate, 11 members x 20 years |
| BB-SUBS estimate | none (not public); k x EC46 corrected | 0 of 42 (planned 42) | product has 64 | n/a | n/a | 1.5 deg | error metrics only |

13f = the 13 frames 00Z/12Z from peak-6d 00Z to peak 00Z; 12f = the same without the peak
frame. A daily-mean source is scored on UTC days peak-6..peak-1 against the interval-mean
climatology, and its truth and the AI+RES row it is paired with are both re-reduced on 12f.

**HRRR truth** (`acal/hrrrtruth.py`, `runs/acal/index_hrrr/`). HRRR f00 analysis 2 m
temperature, from the 3 km netCDF archive on Derecho for 2021-2025 and from AWS
(`wrfsfcf00`, `TMP:2 m above ground:anl`) for 2026-01-01..2026-08-31 and two bad on-disk
frames. AWS and disk are bit-identical on 2025-07-15T00 (`aws_vs_disk_check.json`). Each 3 km
field is bin-mean remapped to the 0.25 deg ERA5 index grid; a geometric 4-corner mask keeps
23,902 cells and removes 4.3% of the box area, mostly ocean. The anomaly is taken against
the same ERA5 1990-2019 climatology, giving `hrrr_t2m_anom_12h_{2021..2026}.nc` on the ERA5
index time axis (vars `anom`, `anom_raw`, `mask`, `offset_loyo`). Every forecast and the
walkers are area-meaned on the same mask before they are compared with an HRRR truth.

The offset is the mean HRRR-minus-ERA5 anomaly per cell x hour (00Z/12Z) x calendar month,
leave-one-year-out by the frame's calendar year, from HRRRv4 frames only (2020-12-26 to
2026-08-31, at least 120 frames per cell, month and hour). `anom` = `anom_raw` minus that
offset. HRRRv3 was excluded because its analysis climate differs from v4 at 12Z by more than
the v4 year-to-year noise (`offset_version_check.json`). The v3-minus-v4 land mean at 12Z
is -0.150 K and negative in 12 of 12 months, against -0.032 K for 2021-22 minus 2023-25
within v4. The land-mean offset is -0.474 K for v3 and -0.378 K for v4.

| 42 cases, 13f, vs ERA5 on the HRRR mask (`cases_hrrr_summary.json`) | r | mean HRRR - ERA5 | sd | range | cases below 2 K |
|---|---|---|---|---|---|
| `hrrr` (offset-corrected, headline) | 0.9995 | -0.023 K | 0.092 | -0.21..+0.17 | 0 |
| `hrrr_raw` | 0.9991 | -0.180 K | 0.120 | -0.45..+0.07 | 9 (e04 e11 e18 e22 e29 e35 e38 e39 e40) |

On 12f the corrected truth has 1 case below 2 K (e22) and the raw truth 6. The offset sets the mean HRRR-minus-ERA5 difference near zero by construction (-0.023 K, sd
0.092 K). The correlation is not produced by the offset, since it is 0.9991 for the raw truth
too. The LOYO offset is fitted against ERA5, so the corrected truth inherits the ERA5
climatology. It tests whether the verdict depends on the analysis system's anomalies, not on
its climatology, and it cannot test whether the ERA5-trained AI+RES gains from being verified
in the ERA5 climate.

**Method.** Scoring is unchanged from the CFS section. It uses tail-signed P(obs) (members at or beyond the
observation), AI+RES with self-normalized importance weights, log ratio with each side
floored at 1/(N+1), Brier at 2/3/4 K, CRPS of the forecast distribution, 90% case-bootstrap
CI (n = 5000), Wilcoxon p, and positive = AI+RES better. LOYO bias for GEFSv12 and GEPS uses the
same (case, year) jobs as CFS (`cfsbase.hind_jobs`). Mean LOYO bias (`bias.csv`) is
-0.63 K for GEFSv12 (range -1.70..+0.38, negative in 35/42), -1.20 K for GEPS
(-2.98..+0.58, 40/42) and -0.67 K for CFSv2. The daily maps of daily-mean sources pair UTC
day D with the ERA5 frames (00Z D, 12Z D) (`s2sbase.DAILY_PAIR`, fixed 2026-10-08); the 7-day
board files were byte-identical before and after that fix.

**Outputs.**
- `runs/acal/s2s/{cfs,gefs,geps,ec46}/` cubes, json, hind, `bias.{nc,csv}`;
  `runs/acal/index_hrrr/` HRRR truth.
- `runs/acal/analysis/s2s/<truth>/` per-source scorecard / paired / maps / ROC / reach /
  closest-member tables; `runs/acal/analysis/s2s/board/` `board_cases.csv` (5040 rows),
  `board_paired.csv`, `board_paired_cases.csv`, `board_summary.json`, `overall_numbers.csv`,
  `bbsubs_method.md`, `SUMMARY.md`; `tilt_check{,_paired}.csv`, `tilt_check.json`.
- `figures/acal/overall/` forest, scoreboard, hrrr_vs_era5, `errmap_<truth>`,
  `skillmaps_{bss,csi}_<truth>`, bbsubs_published, tilt_check;
  `figures/acal/s2s/<truth>/<source>/{maps,roc,reach}/`, `figures/acal/s2s/errmaps/<truth>/`
  (42 per truth), `figures/acal/hrrr/` (the published figure set against HRRR).

**Headline, the paired log ratio of AI+RES vs each model** (`board_paired.csv`, AI+RES variant
`sn`, model `raw_emp` headline and `corr_emp` sensitivity; mean [90% CI], W/T/L, Wilcoxon p).

| truth | model (window) | cases | raw (headline) | W/T/L, p | corrected | W/T/L, p |
|---|---|---|---|---|---|---|
| ERA5 | CFSv2 (13f) | all 42 | +0.49 [+0.20, +0.79] | 24/0/18, 0.023 | +0.20 [-0.07, +0.47] | 26/0/16, 0.30 |
| | | heat 31 | +0.78 [+0.42, +1.12] | 21/0/10, 0.0027 | +0.24 [-0.10, +0.57] | 20/0/11, 0.26 |
| | | cold 11 | -0.30 [-0.66, +0.08] | 3/0/8, 0.27 | +0.12 [-0.28, +0.53] | 6/0/5, 0.50 |
| | GEFSv12 (13f) | all | +0.69 [+0.37, +0.99] | 28/0/14, 0.0014 | +0.38 [+0.08, +0.67] | 25/0/17, 0.040 |
| | | heat | +1.02 [+0.68, +1.35] | 24/0/7, 0.00018 | +0.47 [+0.09, +0.83] | 20/0/11, 0.044 |
| | | cold | -0.26 [-0.66, +0.14] | 4/0/7, 0.40 | +0.12 [-0.22, +0.45] | 5/0/6, 0.83 |
| | ECCC GEPS (12f) | all | +0.66 [+0.32, +1.00] | 27/0/15, 0.0056 | +0.20 [-0.05, +0.43] | 25/0/17, 0.17 |
| | | heat | +1.08 [+0.72, +1.44] | 25/0/6, 0.00011 | +0.19 [-0.12, +0.47] | 20/0/11, 0.21 |
| | | cold | -0.52 [-0.93, -0.10] | 2/0/9, 0.078 | +0.23 [-0.15, +0.62] | 5/0/6, 0.35 |
| HRRR | CFSv2 (13f) | all 42 | +0.49 [+0.19, +0.79] | 24/0/18, 0.031 | +0.19 [-0.08, +0.44] | 25/0/17, 0.32 |
| | | heat 31 | +0.77 [+0.40, +1.13] | 21/0/10, 0.0033 | +0.23 [-0.11, +0.54] | 19/0/12, 0.25 |
| | | cold 11 | -0.29 [-0.58, +0.01] | 3/0/8, 0.14 | +0.08 [-0.29, +0.47] | 6/0/5, 0.56 |
| | GEFSv12 (13f) | all | +0.63 [+0.32, +0.92] | 27/0/15, 0.0028 | +0.36 [+0.06, +0.65] | 25/0/17, 0.048 |
| | | heat | +0.95 [+0.61, +1.27] | 24/0/7, 0.00031 | +0.45 [+0.06, +0.82] | 20/0/11, 0.052 |
| | | cold | -0.28 [-0.66, +0.08] | 3/0/8, 0.23 | +0.08 [-0.25, +0.38] | 5/0/6, 0.90 |
| | ECCC GEPS (12f) | all | +0.64 [+0.31, +0.97] | 25/0/17, 0.0038 | +0.26 [+0.01, +0.49] | 26/0/16, 0.079 |
| | | heat | +1.06 [+0.72, +1.39] | 23/0/8, 7.5e-5 | +0.28 [-0.03, +0.56] | 21/0/10, 0.11 |
| | | cold | -0.56 [-1.00, -0.12] | 2/0/9, 0.078 | +0.20 [-0.19, +0.61] | 5/0/6, 0.40 |

Against HRRR raw (sensitivity, all 42, raw / corrected) the log ratios are CFSv2 +0.62 [+0.32, +0.92] / +0.26
[-0.00, +0.52]; GEFSv12 +0.66 [+0.37, +0.95] / +0.42 [+0.12, +0.70]; GEPS +0.77 [+0.43, +1.11]
/ +0.32 [+0.07, +0.56]. The published 25-frame CFS reference row gives +0.488 (ERA5),
+0.474 (HRRR) and +0.598 (HRRR raw), against +0.494 / +0.490 / +0.618 for `cfs13`.

**CRPS and Brier at 3/4 K** (`board_paired.csv`, all 42 cases, model minus AI+RES, positive
= AI+RES better; mean CRPS of each side from `mean_aires` / `mean_model`).

| truth | model | mean CRPS AI+RES / model (K) | dCRPS raw | W/T/L | dCRPS corr | dBrier 3 K raw (W/T/L, p) | dBrier 4 K raw (W/T/L, p) | dBrier 3 / 4 K corr |
|---|---|---|---|---|---|---|---|---|
| ERA5 | CFSv2 | 0.939 / 1.592 | +0.65 [+0.40, +0.90] | 33/0/9 | +0.58 [+0.29, +0.86] | +0.021 [-0.043, +0.085] (14/3/25, 0.69) | +0.028 [-0.000, +0.067] (6/12/24, 0.11) | +0.005 / +0.011 |
| | GEFSv12 | 0.939 / 1.456 | +0.52 [+0.29, +0.75] | 31/0/11 | +0.50 [+0.26, +0.75] | +0.005 [-0.057, +0.065] (15/3/24, 0.62) | +0.031 [+0.003, +0.067] (10/12/20, 0.75) | +0.002 / +0.025 |
| | ECCC GEPS | 0.962 / 1.861 | +0.90 [+0.62, +1.19] | 33/0/9 | +0.67 [+0.43, +0.93] | +0.004 [-0.066, +0.074] (15/3/24, 0.47) | +0.043 [+0.006, +0.088] (10/12/20, 0.97) | -0.002 / +0.026 |
| HRRR | CFSv2 | 0.975 / 1.635 | +0.66 [+0.39, +0.92] | 32/0/10 | +0.59 [+0.30, +0.88] | +0.018 [-0.054, +0.090] (15/3/24, 0.75) | +0.026 [-0.006, +0.066] (7/11/24, 0.125) | +0.002 / +0.014 |
| | GEFSv12 | 0.975 / 1.487 | +0.51 [+0.28, +0.75] | 30/0/12 | +0.50 [+0.25, +0.75] | -0.005 [-0.075, +0.064] (15/2/25, 0.51) | +0.032 [+0.001, +0.071] (10/11/21, 0.50) | -0.007 / +0.023 |
| | ECCC GEPS | 1.000 / 1.920 | +0.92 [+0.63, +1.22] | 33/0/9 | +0.69 [+0.44, +0.95] | -0.005 [-0.084, +0.076] (11/3/28, 0.18) | +0.042 [+0.005, +0.087] (13/11/18, 0.49) | -0.004 / +0.027 |

Split by family (raw, ERA5), dCRPS for heat is CFSv2 +0.86 [+0.63, +1.07] (28/0/3), GEFSv12 +0.72
[+0.49, +0.95] (27/0/4), GEPS +1.31 [+1.07, +1.56] (29/0/2). For cold
it is +0.08 [-0.55, +0.76]
(5/0/6), -0.05 [-0.53, +0.47] (4/0/7), -0.25 [-0.72, +0.28] (4/0/7). HRRR gives heat
+0.87 / +0.71 / +1.34 and cold +0.07 / -0.05 / -0.28.

The 4 K Brier means are positive for every model and the CIs for GEFSv12 and GEPS exclude
zero, but the Wilcoxon p is 0.49 to 0.97. Eleven or twelve cases are ties and the mean is set
by a few confident misses, so this is not a robust 4 K difference.

**Where the forecasts sit** (`board_cases.csv`, ERA5, 42 cases; signed error = tail sign x
(ensemble mean - obs), negative = short of the extreme).

| ERA5 | AI+RES 13f | CFSv2 raw / corr | GEFSv12 raw / corr | GEPS raw / corr |
|---|---|---|---|---|
| mean signed error, all (K) | -0.97 | -2.21 / -2.04 | -2.08 / -1.99 | -2.58 / -2.25 |
| heat / cold (K) | -0.56 / -2.12 | -2.06 / -2.62 (raw) | -1.99 / -2.34 (raw) | -2.73 / -2.15 (raw) |
| median P(obs) | 0.116 | 0.062 / 0.062 | 0.048 / 0.097 | 0.048 / 0.095 |
| cases with P(obs) = 0 | 2 | 20 / 15 | 10 / 10 | 17 / 11 |
| mean spread (K) | 0.81 | 1.30 | 1.34 | 1.57 |

**16-member sensitivity** (`board_paired.csv`, all 42 cases, log ratio, ERA5 / HRRR).
`e16_raw_emp` is the expected score over random 16-member subsets on both sides (exact
hypergeometric for the models, 1000 seeded 16-walker resamples for AI+RES); `s16_emp` is a
fixed 16-member subset of the model against the native 32-walker AI+RES.

| model | native (headline) | e16_raw_emp | s16_emp | e16_corr_emp |
|---|---|---|---|---|
| CFSv2 (N = 16) | +0.49 / +0.49 | +0.80 [+0.55, +1.04] 32/3/7 / +0.80 [+0.56, +1.05] 32/3/7 | n/a | +0.51 / +0.50 |
| GEFSv12 (N = 31) | +0.69 / +0.63 | +0.68 [+0.43, +0.92] 32/1/9 / +0.66 [+0.42, +0.90] 32/2/8 | +0.45 [+0.11, +0.78] (p 0.088) / +0.39 [+0.06, +0.71] (p 0.105) | +0.44 / +0.43 |
| ECCC GEPS (N = 21) | +0.66 / +0.64 | +0.77 [+0.50, +1.05] 31/3/8 / +0.77 [+0.51, +1.04] 30/3/9 | +0.50 [+0.18, +0.83] (p 0.043) / +0.48 [+0.15, +0.81] (p 0.042) | +0.42 / +0.48 |

dCRPS at e16 (raw, ERA5) is +0.69 / +0.58 / +0.96 for CFSv2 / GEFSv12 / GEPS. The CFS last-4
subset (`sub_emp`) scores -0.52 [-0.80, -0.24]. Reducing a model to 16 members LOWERS the
log ratio (GEFSv12 +0.69 to +0.45), because the floor 1/(N+1) rises from 1/32 to 1/17 and a
case where no member reaches the observation costs the model less. The e16 AI+RES side
re-normalizes the weights within each subset, which moves its mean toward the extreme (era5
13f mean signed error -0.97 K native, -0.76 K e16, `board_summary.json` conventions.e16), so
e16 rows are a sensitivity and not a 16-walker run.

**Error maps and skill maps** (7-day ensemble-mean T2m anomaly, CONUS land, cos-lat).
Composite error (`errmaps_composite_<truth>.csv`; bias = composite mean error, r = pattern
correlation of the composite with the truth composite; MAE over all 42 cases):

| truth | model | heat bias (K), r | cold bias (K), r | MAE all 42 (K) |
|---|---|---|---|---|
| ERA5 | AI+RES | -0.76, 0.90 | +3.27, 0.74 | 2.84 |
| | CFSv2 | -2.94, 0.21 | +3.75, 0.07 | 3.93 |
| | GEFSv12 | -2.67, 0.36 | +3.48, 0.55 | 3.53 |
| | ECCC GEPS | -4.01, 0.32 | +3.04, 0.13 | 4.55 |
| HRRR | AI+RES | -0.72, 0.90 | +3.24, 0.74 | 2.84 |
| | CFSv2 | -2.91, 0.18 | +3.73, 0.07 | 3.93 |
| | GEFSv12 | -2.64, 0.34 | +3.45, 0.54 | 3.52 |
| | ECCC GEPS | -3.97, 0.33 | +3.01, 0.11 | 4.54 |

The difference in per-case field RMSE of the ensemble mean (`board_paired.csv` dfield_rmse, ERA5, model minus
AI+RES) is CFSv2 +1.23 [+0.86, +1.61] (34/0/8; 3.48 vs 4.71 K), GEFSv12 +0.78 [+0.43, +1.16]
(26/0/16; 3.48 vs 4.26 K), GEPS +1.75 [+1.29, +2.21] (35/0/7, 12f; 3.58 vs 5.33 K). HRRR gives
+1.23 / +0.77 / +1.74.

Skill maps (`overall_numbers.csv`, figure scoreboard; BSS = land median, CSI = land mean, from
`<truth>/maps_land_means_<source>.json`; AI+RES 13f, in brackets the 12f AI+RES that pairs
with GEPS):

| truth | score | AI+RES | CFSv2 raw | GEFSv12 raw | GEPS raw |
|---|---|---|---|---|---|
| ERA5 | BSS +2 K | 0.226 [0.243] | -0.094 | 0.015 | -0.243 |
| | BSS -2 K | 0.221 [0.243] | 0.257 | 0.317 | 0.341 |
| | CSI +2 K | 0.559 [0.568] | 0.302 | 0.315 | 0.185 |
| | CSI -2 K | 0.321 [0.329] | 0.311 | 0.275 | 0.393 |
| HRRR | BSS +2 K | 0.229 [0.237] | -0.106 | 0.024 | -0.238 |
| | BSS -2 K | 0.222 [0.243] | 0.251 | 0.318 | 0.315 |
| | CSI +2 K | 0.559 [0.566] | 0.298 | 0.316 | 0.184 |
| | CSI -2 K | 0.320 [0.330] | 0.313 | 0.275 | 0.392 |

**Is AI+RES still better than CFSv2 against HRRR?** Yes, on the same measures and by the same
amount as against ERA5, and with the same limits. Against the offset-corrected HRRR truth the
log ratio vs raw CFSv2 is +0.49 [+0.19, +0.79] (24/0/18, p 0.031), against +0.49 [+0.20,
+0.79] (p 0.023) on ERA5. dCRPS is +0.66 [+0.39, +0.92] against +0.65. The gap is carried by
heat (+0.77) while cold is negative (-0.29). Against bias-corrected CFSv2 the log ratio is
+0.19 [-0.08, +0.44] (p 0.32), so most of the log-ratio gap is CFS cold drift, as on ERA5. The
3 K and 4 K Brier differences are not significant (p 0.75 and 0.125). Against the raw HRRR
truth the log ratio grows to +0.62, because 9 cases fall below 2 K and the HRRR-ERA5 offset
enters the anomaly. The steering caveat below applies to all of these numbers.

**Steering caveat (tilt check)** (`acal/tilt.py`; `runs/acal/analysis/s2s/tilt_check.json`,
`tilt_check_paired.csv`, `figures/acal/overall/tilt_check.png`). Every acal run cloned its
walkers toward the OBSERVED tail sign, and the baselines are blind to it. Three AI+RES
variants are scored against the same baselines. `sn` is the published forecast. `uniform`
gives the same 32 final walkers equal weights, which is the tilted population and an upper
bound on the steering benefit. `untilted` is FCN3 launched at lead 6 d from the 32
pre-selection walker states (192 members). These are the only blind forecasts that reach the
window, and they are FCN3 rather than GenCast over lead 6-21 d.

| paired vs (ERA5 / HRRR) | CFSv2 | GEFSv12 | ECCC GEPS |
|---|---|---|---|
| log ratio, sn (published) | +0.49 / +0.49 | +0.69 / +0.63 | +0.66 / +0.64 |
| log ratio, uniform | +1.97 / +1.98 | +2.16 / +2.11 | +2.13 / +2.12 |
| log ratio, untilted | -0.13 [-0.43, +0.16] / -0.11 | +0.07 [-0.25, +0.37] / +0.02 | +0.01 [-0.33, +0.35] / -0.00 |
| dCRPS, sn | +0.65 / +0.66 | +0.52 / +0.51 | +0.90 / +0.92 |
| dCRPS, uniform | +0.92 / +0.92 | +0.78 / +0.77 | +1.19 / +1.21 |
| dCRPS, untilted | +0.33 [+0.14, +0.50] / +0.33 | +0.19 [+0.01, +0.37] / +0.18 [-0.01, +0.37] | +0.56 [+0.30, +0.81] / +0.58 |

With equal weights the final walkers sit +0.55 K beyond the observed `A_L` on the tail side
(ERA5, 42 cases). The importance weights pull the mean back by 1.52 K [1.32, 1.73] to -0.97 K;
the untilted FCN3 forecasts are at -1.82 K. The correction is limited at N = 32. The Kish
effective sample size is 5.3 on average (median 4.0) of 32, the largest normalized weight is
0.41 on average, and on average 29.6 of the 32 walkers (minimum 26) lie on the tail side of
the weighted mean. The published log ratio keeps 25-32% of the uniform one and the published
dCRPS 66-76%.

The blind FCN3 forecasts (`untilted`, floor 1/193) have log ratios whose CIs all span zero
against the raw baselines (table above). With the 32-member floor 1/33 (`untilted_f32`) they
are +0.09 [-0.12, +0.32] / +0.29 [+0.03, +0.54] / +0.24 [-0.05, +0.51] on ERA5, so the GEFSv12
one excludes zero. Against the bias-corrected baselines (`corr_emp`) the blind log ratio is
-0.42 [-0.69, -0.15] / -0.24 [-0.48, -0.01] / -0.45 [-0.69, -0.22] on ERA5 (Wilcoxon p 0.026 /
0.11 / 0.0085) and -0.42 / -0.25 / -0.39 on HRRR, every CI below zero, and -0.19 / -0.02 /
-0.23 with the 1/33 floor. The corrected log-ratio advantage of the published forecast (+0.20 /
+0.38 / +0.20) therefore cannot be attributed to forecast quality with the existing runs.

Against the raw baselines the blind dCRPS keeps 37-62% of the published margin. It is
significant against CFSv2 and GEPS (Wilcoxon p 0.008 / 0.001 on ERA5, 0.010 / 0.001 on HRRR)
and not against GEFSv12 (+0.19 [+0.01, +0.37], 26/0/16, p 0.064 on ERA5; +0.18 [-0.01, +0.37],
p 0.076 on HRRR). The raw margin is confounded by family. It is +0.61 / +0.47 / +1.04 on the 31
heat cases and -0.47 [-0.73, -0.21] / -0.60 [-0.87, -0.34] / -0.80 [-1.06, -0.53] on the 11
cold cases. The blind chain's mean signed error is -1.35 K on heat and -3.13 K on cold (13f),
against -2.06 / -2.62 K (CFSv2), -1.99 / -2.34 K (GEFSv12) and -2.73 / -2.15 K (GEPS, 12f).
That relative warm shift wins on a slate that is 31:11 heat to cold. Against the bias-corrected
baselines the blind dCRPS is +0.25 [+0.07, +0.43] (p 0.033) / +0.17 [+0.01, +0.34] (p 0.21) /
+0.33 [+0.18, +0.50] (p 0.003), positive in both families. The CRPS advantage therefore
survives a blind forecast from the same GenCast-FCN3 chain against CFSv2 and GEPS, and it is
not significant against GEFSv12. The blind control also removes much of the ensemble-mean
advantage. It falls short by 1.82 K (13f; 1.87 K on 12f), so 69% / 76% / 55% of the published
mean-error gap to CFSv2 / GEFSv12 / GEPS is absent without steering. The log-ratio advantage,
which measures the probability placed on the observed extreme, cannot be separated from
residual steering with the existing runs. The tilt check scores only the CONUS index (P(obs),
CRPS, Brier, mean error). The maps, the field RMSE and the composites have no blind control.

**Reading.** On the 42 selected cases AI+RES puts more probability on the observed tail than
raw CFSv2, GEFSv12 and ECCC GEPS (log ratio +0.49 to +0.69 on ERA5, every CI above zero), and
has lower CRPS (by 0.52 to 0.90 K). The result is the same against HRRR truth. The gap comes
from the heat cases. For cold cases the log ratio is negative against all three (-0.26 to
-0.52; the GEPS CI excludes zero, Wilcoxon p 0.078, n = 11). Bias correction of the baselines
removes 58% / 45% / 70% of the log-ratio gap for CFSv2 / GEFSv12 / GEPS on ERA5. CFSv2 and GEPS
fall to +0.20 with CIs straddling zero, and GEFSv12 to +0.38 [+0.08, +0.67] (p 0.040). The CRPS
gap stays (+0.50 to +0.67 K). No AI+RES hindcast exists, so the correction removes the
baselines' drift only. Against the corrected baselines the blind control scores below zero on
log ratio (steering caveat above). No 3 K Brier difference was detected for any model (CIs span
zero, p 0.18 to 0.75 over both truths). The ensemble mean of AI+RES is closer to the
observation (signed error -0.97 K vs -2.08 to -2.58 K), but the blind control is at -1.82 K, so
55-76% of that gap is absent without steering. The 7-day field of AI+RES has a lower RMSE and a
higher heat-composite pattern correlation (0.90 vs 0.21 to 0.36). The maps and composites are
family-matched on outcome-selected cases (+2 K is scored on the 31 heat cases only and -2 K on
the 11 cold cases only, the cases the walkers were cloned toward), so they are conditional
scores, not unconditional skill, and none has a blind control. At -2 K the baselines have the
higher land-median BSS (0.257 to 0.341 vs 0.221). Of the baselines GEFSv12 has the lowest CRPS
(1.456 K raw, 1.439 K corrected) and GEPS the highest (1.861 K raw). After correction GEFSv12
is the only baseline AI+RES still beats on log ratio with a CI above zero, but at equal N
(`e16_corr_emp`) the margins are +0.51 / +0.44 / +0.42, so no baseline ranks consistently on
log ratio once N is equalized. The tilt check bounds the CONUS-index scores only. The CRPS
margin survives the blind control against CFSv2 and GEPS and is not significant against
GEFSv12. The log-ratio margin has no blind support.

**Caveats - state these wherever the numbers go.**
1. **Selected on outcome** (all |A_L| >= 2 K). The board says which forecast put more mass on
   what happened, not which is calibrated; false alarms are not scored.
2. **AI+RES was steered toward the observed tail** (tilt check above). The baselines were not.
   The log-ratio advantage is not separable from residual steering, and against the
   bias-corrected baselines the blind control scores below zero. The CRPS advantage is kept by
   a blind FCN3 control against CFSv2 and GEPS, not significantly against GEFSv12.
3. **Ensemble size, resolution, window and lead differ.** AI+RES is 32 importance-weighted
   walkers (Kish ESS ~5) at 0.25 deg; CFSv2 16 lagged members at ~0.94 deg with leads
   21.0-24.75 d; GEFSv12 31 members at 0.5 deg, lag 0; GEPS 21 members at 1 deg, daily means,
   leads 21-27 d. The log-ratio floor depends on N (16-member table above). Coarse grids
   limit the gridpoint maps, not the CONUS index.
4. **Bias corrections are noisy.** LOYO uses 4-5 other years per case; CFSv2 uses the
   25-frame bias on 13-frame members; under HRRR truths the `corr` variants use the full-box
   ERA5-referenced bias on masked members. No AI+RES hindcast exists, so the corrected rows
   remove the baselines' drift only, not that of AI+RES.
5. **42 cases, 11 cold, sharing seasons** (DJF 22), so the bootstrap CIs are optimistic and
   the cold rows are weak.
6. **HRRR is not an independent verdict on the index.** The offset makes the mean HRRR-minus-ERA5 difference near zero by construction (-0.023 K, sd
   0.092 K; raw -0.180 K). The correlation is not produced by the offset (0.9995 corrected,
   0.9991 raw). The LOYO offset is fitted against ERA5, so the corrected truth inherits the
   ERA5 climatology. It tests the analysis anomalies, not the climatology.

**What could not be done, and why.**
1. **ECMWF IFS (EC46).** Code, tests, the 168-request plan and the ERA5 at the hindcast dates
   are done; the download needs the user's ECDS token (`~/.ecdsapirc` absent on 2026-10-08).
2. **No separate "EC21".** At a 21 d lead the operational IFS forecast is EC46 (the IFS ENS
   extended range). IFS ENS medium range, HRES, AIFS Single, AIFS ENS and TIGGE stop at 15 d.
   SEAS5 starts on the 1st of the month, so its lead at the peak is 21-51 d (median 35 d),
   which is not a 3-week row; not built.
3. **BB-SUBS is not public.** Its row is an estimate (k x bias-corrected EC46 error, k = 0.877
   MSE, 0.936 CRPS, 0.947 Brier at week 3), so it waits on EC46. It covers error metrics, plus case P(obs) only if Tier 2 (a Gaussian
   signal-noise model) reproduces EC46's own P(obs); never maps. The published BB-SUBS skill is vendor-reported and winter-only
   (MSESS over the Oct-Mar winters 2023-26, RPSS over DJF 2025/26), while the slate is DJF 22,
   SON 10, MAM 9, JJA 1. Both published scores are unconditional, so k is an average-weather
   ratio applied to tail events selected on the outcome (method note, caveat 4). 15 cases fall in winters almost certainly inside BB-SUBS training and 5 have Apr-Sep
   inits. Real forecasts need Brightband's 2-week pilot (the user). See `board/bbsubs_method.md`.
4. **AI S2S models.** None has a public 2022-2025 forecast archive at a 21 d lead. FuXi-S2S
   hindcasts end 2021-12-29 (8 of 42 cases); AI Weather Quest submissions (incl. AIFS-SUBS)
   are quintile probabilities for 3 of 42 cases; NeuralGCM, DLESyM and ACE2 are open weights
   only (GPU runs, out of scope on this box); GenCast, GraphCast, Pangu and Aurora public
   forecasts stop at 10-15 d.
5. **S2S-database models** (UKMO GloSea6, JMA, KMA, CMA, CNRM and others) need the same ECDS
   token, or an IRIDL login with the S2S terms.
6. Three anonymous sources were not built this round, the ERA5-initialised IFS reforecast (Planette,
   39 of 42 cases), NASA GEOS-S2S (SubX) and CESM2 (on Derecho disk, 20-21 of 42 cases).

**Environment.** The packages pip-installed into `my-env` on 2026-10-07 for EC46 are
`cdsapi` 0.7.7,
`ecmwf-datastores-client` 0.5.3, `multiurl` 0.3.9, `tqdm` 4.70.1 (versions re-read with
`importlib.metadata` on 2026-10-08). `cdsapi.Client(url=ECDS)` resolves to the
`ecmwf.datastores` legacy client.

**Disk.** `runs/acal/s2s/` 27 GB (of which `gefs/raw/` 19 GB of GRIB, deletable once the board
is final; cubes rebuild from AWS in ~25 min), `runs/acal/index_hrrr/` 1.6 GB (incl. `work/`
395 MB of build chunks, deletable), `runs/acal/analysis/s2s/` 7.6 GB.

**Spot-checks** (12 numbers in this section re-read from the files on 2026-10-08 by a
session script, `spotcheck.py` in the session scratchpad; all match at the printed precision).
Board files are under `runs/acal/analysis/s2s/board/`, the other two under `runs/acal/`.
The same script also confirmed `board_cases.csv` has 5040 rows, the e16 GEPS dCRPS (0.9555), the
AI+RES heat signed error (-0.5555 K), the AI+RES e16 signed error (-0.7635 K) and the untilted
share of the published CRPS margin (0.500 / 0.369 / 0.621 for CFSv2 / GEFSv12 / GEPS).

| # | file | quantity | value read | value written | check |
|---|---|---|---|---|---|
| 1 | `board_paired.csv` | ERA5 log ratio vs CFSv2 raw, all | +0.49 [+0.20, +0.79] 24/0/18 | +0.49 [+0.20, +0.79] 24/0/18 | OK |
| 2 | `board_paired.csv` | HRRR log ratio vs GEFSv12 raw, all | +0.63 [+0.32, +0.92] | +0.63 [+0.32, +0.92] | OK |
| 3 | `board_paired.csv` | ERA5 dCRPS vs GEPS raw; CRPS AI+RES/model | +0.90; 0.962 / 1.861 | +0.90; 0.962 / 1.861 | OK |
| 4 | `board_paired.csv` | HRRR log ratio vs CFSv2 corrected, p | +0.19 [-0.08, +0.44] p 0.32 | +0.19 [-0.08, +0.44] p 0.32 | OK |
| 5 | `board_paired.csv` | ERA5 dBrier 4 K vs GEFSv12 raw | +0.031 [+0.003, +0.067] 10/12/20 p 0.75 | +0.031 [+0.003, +0.067] 10/12/20 p 0.75 | OK |
| 6 | `board_cases.csv` | ERA5 median P(obs) AI+RES; P(obs)=0 CFSv2 raw | 0.116; 20 | 0.116; 20 | OK |
| 7 | `overall_numbers.csv` | ERA5 land-median BSS +2 K AI+RES 13f / CFSv2 | 0.226 / -0.094 | 0.226 / -0.094 | OK |
| 8 | `errmaps_composite_era5.csv` | heat composite bias AI+RES / CFSv2; MAE all AI+RES | -0.76 / -2.94; 2.84 | -0.76 / -2.94; 2.84 | OK |
| 9 | `tilt_check.json` | ERA5 log ratio vs CFSv2: sn / uniform / untilted | +0.49 / +1.97 / -0.13 | +0.49 / +1.97 / -0.13 | OK |
| 10 | `index_hrrr/cases_hrrr_summary.json` | HRRR-ERA5 13f mean offset corrected / raw; raw cases < 2 K | -0.023 / -0.180; 9 | -0.023 / -0.180; 9 | OK |
| 11 | `board_summary.json` | members cfs13 / gefs / geps; GEPS lead | 16 / 31 / 21; 21-27 d | 16 / 31 / 21; 21-27 d | OK |
| 12 | `board_paired.csv` | ERA5 e16 log ratio vs CFSv2 | +0.80 32/3/7 | +0.80 32/3/7 | OK |

**Next steps.**
1. EC46 (user, ~10 min). Create `~/.ecdsapirc`. Then the one command above (ECDS queue
   1-3 h), the driver (~30 min) and `acal.tilt` with `ec46` in `MODELS`. The BB-SUBS estimate
   rows and the EC46 rows in every overall figure follow without code changes.
2. Settle the steering question (GPU, a3mega). Rerun a subset of cases with the tail direction
   unknown (two-sided, one run per tail scored as one forecast, or the direction fixed before
   the outcome), add an all-C = 0 GenCast control to the peak, and apply the same protocol to
   non-event control dates.
3. Non-event controls remain the route to reliability for every row.
4. Optional anonymous rows are the IFS reforecast (Planette) and NASA GEOS-S2S.
