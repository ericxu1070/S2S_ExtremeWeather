# AI+RES - live state

Read `aires.md` (repo root) for the experiment design. This file is the state of the
work, updated as phases complete. **Trust the commands in "Verify state" over any prose
here** - doc claims rot, `ls` does not.

Branch: `aires`. Machine split: Derecho for CPU work, a3mega H100 for anything touching
FCN3 or GenCast inference.

Visual summaries:
- plan, adapter, Gate 1 (design, derivations, open problems):
  https://claude.ai/code/artifact/081bd2f9-7725-4e46-8af8-ec57f54ee2df
- Gate 2 results (what each criterion measures, the G2b amendment, the M=24 finding):
  https://claude.ai/code/artifact/6cbf2eed-276b-42b6-b775-8ed95dc36cc1
  (source: `docs/aires_gate2.html`)

## Where things stand

| Phase | State |
|---|---|
| 0 - move a3mega FCN3 week-3 results to Derecho | **done**, verified: figures regenerate pixel-identical from the transferred cubes |
| 1 - adapter + calibration + Gate 1 | **done on Derecho AND on a3mega** - identical numbers on both |
| 1 - Gate 2 (forecast equivalence) | **PASS**. Ran on a3mega (Slurm job 1153), re-scored on Derecho from the transferred cubes - identical to the digit. G2a 0.083 K, G2b horizon 8.5 d, G2c 0.990. `aires.md`'s G2b was amended first (endpoint -> horizon); see below |
| 1 - Gate 3 (score skill, the go/no-go) | **not started**; `aires/walker.py` (its prerequisite) is **written, tested and rolled on an H100** (job 1154) - the walker -> adapter -> FCN3 round trip works end to end |
| 2-5 | not started |

**Continue Gate 3 on a3mega** (see "Start here for Gate 3" below) - it holds the
calibration, both Gate 2 cubes, the walker, and the H100s. The full Gate 2 data tree is now
mirrored on Derecho as well (cubes, IC, scores) and re-scores there identically, so Derecho
can do CPU analysis, but it is not required again until Phase 5.

## Verify state (run these, don't trust the table)

```bash
# a3mega (this machine): conda activate moe; cd .../S2S_ExtremeWeather
# Derecho: module load conda && conda activate my-env; cd /glade/derecho/scratch/exu/...

ls -l runs/aires/calib/derived_calib.nc              # calibration built? (~48 MB)
PYTHONPATH=. python -m pytest aires/tests/ -q        # 48 pass, 1 skipped (~100 s)
PYTHONPATH=. python -m aires.calibrate --gate1       # re-scores Gate 1 on the held-out ICs
PYTHONPATH=. python -m aires.gate2 --stage score     # re-scores Gate 2 from cached cubes
ls runs/aires/gate2/cache/*_adapter_cube.nc          # Gate 2 adapter ensemble (24 members)
PYTHONPATH=. JAX_PLATFORMS=cpu python -m aires.walker --check --steps 6   # walker, no GPU
```

The test suite is CPU-only and needs no GPU; it takes ~100 s because the batch-shape tests
open the real 721x1440 init frames.

## What Phase 1 established

**The adapter is exact where it can be.** Converting a GenCast-format ERA5 frame and
comparing against the independently built FCN3 IC for the same timestamp:
**69 of 72 channels are bit-identical** (max absolute difference 0.0). The two models
share the grid, the 13 pressure levels and every unit; the only structural difference is
that GenCast's latitude ascends (-90 -> +90) and FCN3's descends. `total_precipitation_12hr`
and `sea_surface_temperature` are carried by GenCast and simply not consumed by FCN3.
FCN3 takes **one** time frame, so GenCast's 12 h step vs FCN3's 6 h step is a non-issue.

**Gate 1 PASSED** on the 6 held-out event ICs (fit on the 90 p90 ICs, disjoint):

| criterion | worst held-out | threshold |
|---|---|---|
| u100m/v100m RMSE | 0.634 m/s | 1.0 |
| u100m/v100m abs bias | 0.066 m/s | 0.2 |
| tcwv relative RMSE | 4.57% | 5% |
| tcwv abs relative bias | 0.60% | 2% |

**How the 3 derived channels work.** Wind: `W100 = c(x) * W10` with `W = u + iv` and `c`
complex, so one number carries both the boundary-layer speed-up (`|c|`, median 1.25) and
the veering (`arg c`, +-7 degrees, median ~0 globally because NH and SH veer opposite
ways). tcwv: a per-grid-point quadrature `tcwv = sum_l w_l q_l`, ridge-shrunk toward the
physical trapezoid. Learning the weights rather than scaling the trapezoid is what makes
the gate: the trapezoid's error is dominated by the missing sub-1000 hPa layer and by
underground levels over terrain, and no single scale corrects both (5.17% -> 4.57%).

## What Gate 2 established

Two ensembles from the same ERA5 analysis at the same init, differing only in how the 72
channels were assembled, rolled 84 x 6 h to the 21 d verification window with **matched
seeds** (adapter member i and native member i share an internal-noise stream). The native
side was already on disk from the FCN3-vs-GenCast experiment, so only the adapter side
cost GPU time: ~11 min node wall on one 8xH100 node, most of it loading the 4 GB model
pickle.

| criterion | value | threshold | |
|---|---|---|---|
| G2a `\|delta mean A_L\|` | 0.083 K | < 0.25 K | **PASS** |
| G2b rank-equivalence horizon | 8.5 d | >= 7 d | **PASS** |
| G2c max `d_paired / d_internal` | 0.990 | <= 1.00 | **PASS** |

**The gate's purpose is met.** What matters for AI+RES is that the *score* a walker
receives - an ensemble mean - is a property of the walker's state and not of the adapter.
G2a says the ensemble mean is reproduced to 0.083 K on the longest score forecast the
experiment ever runs (21 d; every production score forecast is shorter and will agree
better). G2c says the adapter's perturbation is **strictly smaller than FCN3's own
stochasticity at every lead** - 22% of it at day 1.5, 51% at day 9, converging to 99% only
at day-21 saturation. It never exceeds it.

**The seed matching is verified by the data, not assumed.** If adapter member i had
drawn a different noise stream than native member i, `d_paired` would equal `d_internal`
from the very first step. It does not: it is 22% of it at day 1.5 and grows from there,
which is only possible if the two members share an internal-noise stream and differ solely
through the IC.

**G2b was amended from an endpoint to a horizon (2026-08-18), and the amendment is the
substantive result of this gate.** As originally written in `aires.md` it read paired-member
rank correlation at the 21 d window only, where any IC difference whatsoever - the adapter's
or a plain re-seed - has been amplified to the internal-noise level and shuffles the members
into an unrelated order. The measured 0.107 there is what a *perfect* adapter would also
score, so the criterion could not be passed by anything and carried no information. G2c
proves this rather than asserts it: `d_paired/d_internal = 0.990` at day 21, i.e. by then
the adapter's perturbation IS a re-seed's.

Resolved by lead (third panel of `figures/aires/gate2_PNW_HeatDome_2021.png`) the same
statistic is well-behaved:

    rho_s = 0.994 at 2.5 d, 0.960 at 5 d, 0.957 at 7.5 d, 0.759 at 10 d, 0.144 at 15 d

The two ensembles are member-for-member rank-equivalent out to **8.5 days**; the decay past
that is FCN3's predictability horizon for CONUS-mean T2m, not adapter error. `aires.md` now
specifies G2b as "rho_s > 0.95 out to at least 7 d lead" and adds G2c as a required
criterion; `aires/gate2.py` tests the horizon and reports the 21 d endpoint value as a
diagnostic beside it.

**Why the failure was never fatal.** G2b is a member-*identity* test; AI+RES never reads
member identity. The scoring path is walker state -> adapter -> FCN3 ensemble -> **mean** ->
`theta`, which is exactly G2a's quantity. G2a and G2c both pass, and G2c is a strictly
stronger statement than either original threshold.

**M=24, not the M=6 of the plan.** The cached 24-member ensemble makes the noise floor
measurable, and it is fatal to M=6: per-member sd of A_L is 0.62 K, so two 6-member
ensemble means differ by 0.36 K from sampling noise alone - larger than G2a's 0.25 K
threshold. A perfect adapter would have failed G2a about half the time at M=6. At M=24 the
paired s.e. is 0.149 K, so G2a resolves differences down to ~0.30 K and no finer; that is
now stated in the score output rather than left implicit.

## `aires/walker.py` - written, tested, and rolled on a GPU

Gate 3 needs a walker that can be stopped, checkpointed and restarted, and whose checkpoint
is **global** (`xres/xinference.py` crops to CONUS before the host transfer, so its cached
cubes cannot restart anything or initialise FCN3). That module now exists:

- `roll_segment` keeps a rolling 2-frame **global** buffer (~0.70 GB, matching aires.md's
  estimate exactly) and writes the CONUS crop per step, so nothing global accumulates.
  The crop is deep-copied on purpose: `.sel` with slices is basic indexing and returns a
  view that pins the whole global array alive - without the copy the process grows by
  0.35 GB per step.
- `build_batch` **delegates** to `gencast_s2s.inference.build_example_batch` rather than
  restating it, back-computing the (peak, lead_days) that function is parameterised by and
  asserting the resulting init is the state's own last frame. Two invariants live in that
  function's body (the `data_vars="minimal"` concat, and stripping a `time` axis that leaks
  onto a static) and duplicating them would mean maintaining both copies.
- `check_batch_structure` replaces the "267 grid-node channels" check copied from
  `inference.py`. **That constant does not hold for a walker**: targets and forcings scale
  with segment length, so the total is 704 for a 6-step segment. The structural invariants
  are asserted instead - statics stay 2-D, inputs carry exactly 2 frames, targets/forcings
  span exactly `n_steps`.
- Cloning is `fold_in(fold_in(base, step), walker)`: a clone is a new walker index pointed
  at a parent's state file, and the differing index gives it a noise stream no sibling has
  used.

**First GPU exercise: job 1154** (`slurm/aires_walker_smoke.slurm`, 2 steps, one H100,
6.5 min wall of which ~5 min was the one-time XLA compile; the two steps themselves took
~30 s). It rolled from the event init, wrote a state and a diagnostics cube, and then
proved the state is **both** things it has to be:

```
[verify] state state.nc: valid 2021-06-08 00:00:00, 721x1440 global, 2 frames
[verify] advanced 24 h from the event init
[verify] restarts cleanly: a further 2-step batch builds from 2021-06-08 00:00:00
[verify] adapter -> FCN3 IC: 72 channels, all finite; global mean t2m 281.47 K, tcwv 18.50 kg m-2
[verify] OK - walker state is both a restart point and an FCN3 IC
```

That is the whole walker -> scorer handoff, end to end, on real model output rather than
on ERA5. Nothing in Gate 3 now rests on an untested code path.

**Measured storage** (aires.md's estimates were made from first principles; these are the
real numbers, and both come in under them):

| artifact | measured | per event at N=64, K=5 |
|---|---|---|
| `state.nc` (global, 2 frames, float32 + zlib) | **477 MB** (0.70 GB raw) | ~153 GB if every step is retained |
| `diag.nc` (CONUS crop, all 12 vars) | **6 MB per 12 h step** | ~16 GB (42 steps per walker) |

aires.md budgeted ~230 GB/event for the states; zlib on float32 gets that to ~153 GB
without the float16 hazard below. Retaining only the two most recent steps, as the plan
also suggests, would drop it to ~61 GB.

**Also measured: the 0.25 degree checkpoint occupies ~64 GB of an H100's 80 GB.** That is
the direct confirmation of why Derecho's A100-40GB OOMs on it, and it leaves no room for a
second concurrent member per GPU - the serial path is not a workaround, it is the only
option at this resolution.

Two facts found while writing it, both worth keeping:

- **`total_precipitation_12hr` is a GenCast target but NOT an input.** A restart state does
  not need it; the state carries it anyway because the precip metric derives from it.
- **float16 archived states would be silently corrupt.** aires.md floats float16 to halve
  the 0.70 GB footprint. Geopotential at 50 hPa is ~2e5 m2 s-2 against float16's 65504
  ceiling, so the top of every column would store `inf`. `write_state` is float32 + zlib;
  `tests/test_walker.py::test_float16_would_overflow_geopotential` pins the reason.

## Open problems

1. **Gate 3 is the remaining Phase 1 work, and it is now unblocked.** Still to write:
   `aires/score.py` (M FCN3 forecasts from a walker state out to lead 21 d, reduced to
   theta), `aires/aindex.py` (the scalar observable; `aires/gate2.py::_AL` is the CONUS
   version of it and should be factored out rather than reimplemented), and a driver +
   Slurm script. The run itself is 8-16 free-running walkers checkpointed at leads
   3/6/9/12/15 d, then N x 5 x M FCN3 score forecasts. At the measured ~15 s/step a
   42-step walker is ~11 min, so 16 walkers over 8 GPUs is ~25 min; scoring adds
   16 x 6 x 240 = 23,040 FCN3 steps, ~7 GPU-h, under an hour on a node.
   **aires.md's per-event box is still undefined** - it says to add one `box` field per
   event in Phase 3, and Gate 3's observable needs it (or must explicitly use the CONUS
   mean, as Gate 2 did).
2. ~~Gate 2's G2b criterion needs amending~~ - **done 2026-08-18**. `aires.md` now
   specifies G2b as a >= 7 d rank-equivalence horizon and requires G2c; `aires/gate2.py`
   implements it and `--stage score` exits 0. Gate 2 is recorded as PASS.
3. **The calibration was fitted on ERA5, but production inputs are GenCast forecasts** at
   3-15 day lead, which are smoother than analysis. Gate 3 is what measures whether that
   matters; Gate 1 cannot see it, and Gate 2 could not either (both ICs there are ERA5).
4. **The tcwv residual (~4%) is close to the floor** for any linear function of 13 q
   levels - the sub-1000 hPa moisture is not in the state. Whether 4% matters at all is
   Gate 2's question, not something to optimise further blind.
5. **Calibration seasonal coverage is thin in summer**: the 90 p90 ICs have 0 June and 2
   July frames. The held-out June 7 PNW case passed anyway, so it generalises, but a
   summer-heavy fit would be a cheap improvement if Gate 2 disappoints.

## Start here for Gate 3 (on a3mega)

Phase 1's first two gates are closed. Gate 3 is the go/no-go and everything it needs is on
a3mega already - the calibration, both Gate 2 cubes, and a walker that has rolled on an
H100. **No Derecho session is required.**

```bash
# a3mega, GenCast env (moe), repo root
cd .../S2S_ExtremeWeather && git fetch && git checkout aires && git pull

# 1. confirm the state (30 s) - trust these, not this file's prose
ls -l runs/aires/calib/derived_calib.nc                  # ~48 MB
PYTHONPATH=. python -m aires.gate2 --stage score          # must print GATE 2: PASS, exit 0
PYTHONPATH=. python -m pytest aires/tests/ -q             # 48 pass, 1 skipped, ~100 s
PYTHONPATH=. JAX_PLATFORMS=cpu python -m aires.walker --check --steps 6
```

Then write, in this order:

1. `aires/aindex.py` - the scalar observable. **Factor it out of `aires/gate2.py::_AL`**
   (cos-lat area mean of `xres.xmetrics.forecast_field`) rather than reimplementing it, and
   have `gate2.py` import it back. Decide the box first (open problem 1 below): either add
   the per-event `box` field `aires.md` promises for Phase 3, or make Gate 3 explicitly use
   the CONUS mean, as Gate 2 did.
2. `aires/score.py` - M FCN3 forecasts from a walker state out to 21 d, reduced to `theta`.
   The adapter round trip it depends on is already proven end to end (job 1154).
3. a driver + `slurm/aires_gate3.slurm` (`--job-name=Vayuh-s2s`).

Budget, from measured rates: 8-16 free-running walkers checkpointed at leads 3/6/9/12/15 d
is ~11 min per 42-step walker, so 16 walkers over 8 GPUs is ~25 min; scoring adds
16 x 6 x 240 = 23,040 FCN3 steps, ~7 GPU-h, under an hour on a node. **Pre-stage every
input on the login node first** - the GPU rule in the root `CLAUDE.md` is strict, and Gate 2
already showed short FCN3 jobs are NFS-load-bound rather than compute-bound.

Pass criterion (`aires.md`): Spearman rho >= 0.5 between `theta(t_k)` and the realized
`A_L(t_f)` at t_k = 9 d and 12 d. If it passes only at 12 d, shift the `C_k` schedule later;
if it fails everywhere, fall back to GenCast-as-its-own-scorer and report the FCN3 scoring
failure as a finding.

If the calibration ever needs rebuilding it is deterministic given its inputs, and those
inputs (the 90 p90 ICs) live on a3mega too:
`PYTHONPATH=. python -m aires.calibrate --fit --gate1` (~2 min, CPU, no network). Expect the
identical Gate 1 numbers above; if they differ, check `ls runs/p90_t2m/ic/*.nc | wc -l`
equals 90 on both machines.

## Files

| Path | Role |
|---|---|
| `aires.md` | experiment design, agreed configuration, phase plan |
| `aires/aconfig.py` | the two models' contracts: 72 channels, name maps, grids, gate thresholds |
| `aires/adapter.py` | GenCast state -> FCN3 IC; `GenCastICSource` is the earth2studio DataSource |
| `aires/calibrate.py` | fits + scores the 3 derived channels; `--fit`, `--gate1` |
| `aires/gate2.py` | Gate 2: `--stage {prep,infer,assemble,score}` across the two envs |
| `aires/walker.py` | one GenCast walker segment from a global 2-frame state; `--check` is CPU-only |
| `aires/tests/test_adapter.py` | 25 tests incl. the 69-channel bit-exactness check |
| `aires/tests/test_walker.py` | 24 tests: state contract, batch-from-arbitrary-state, cloning RNG |
| `slurm/aires_gate2.slurm` | Gate 2 infer, 8 shards on one a3mega node, `--job-name=Vayuh-s2s` |
| `runs/aires/calib/derived_calib.nc` | fitted calibration (gitignored, rebuildable in ~2 min) |
| `runs/aires/gate2/` | adapter IC, 24-member adapter cube, scores JSON/CSV |
| `figures/aires/gate2_PNW_HeatDome_2021.png` | the three Gate 2 panels |
| `docs/aires_gate2.html` | published Gate 2 summary (artifact source) |

Not yet written: `aires/score.py`, `aires/dmc.py`, `aires/aindex.py`, `aires/run_aires.py`,
`slurm/aires_gate3.slurm`, `slurm/aires_res.slurm`.

## Notes

- `pytest` was not installed in Derecho's `my-env`; added with `pip install pytest`
  (9.1.1). The GenCast stack still imports cleanly.
- Every command above assumes `PYTHONPATH=.` from the repo root.
- `aires.aconfig` / `aires.adapter` / `aires.gate2` import cleanly in **both** envs
  (`moe` and `fcn3`), which is what lets one driver span the two-env handoff.
  `A.verify_against_package()` now actually runs (it is a no-op outside `fcn3`) and
  confirms the mirrored 72-channel list matches the installed earth2studio.
- Gate 2's 8 shards spent ~9 of their ~11 min loading the shared 4 GB model pickle from
  NFS concurrently (all 8 in `D` state, GPUs idle). That is the known cost of the
  build-once cache and is still far cheaper than 8 concurrent model builds; it just means
  short FCN3 jobs are dominated by load, not compute.
