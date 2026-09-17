# acal — live state

Read `acal/__init__.py` for what this package is for and `ccfg.py` for the knobs. This
file is what is **actually built**, not what is planned.

Last updated 2026-09-11.

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

### The fix (2026-09-17) -- uncommitted, and NOT yet proven on a GPU

`aires/walker.py` + `aires/tests/test_walker.py`, in the working tree, not committed.

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
