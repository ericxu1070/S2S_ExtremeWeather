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
- Gate 3 results (score skill vs the persistence baseline, the DS gap, the checkpoint bug):
  https://claude.ai/code/artifact/faac92dd-c753-4cd3-8209-3117e7d998f4
  (source: `docs/aires_gate3.html`)

## Where things stand

| Phase | State |
|---|---|
| 0 - move a3mega FCN3 week-3 results to Derecho | **done**, verified: figures regenerate pixel-identical from the transferred cubes |
| 1 - adapter + calibration + Gate 1 | **done on Derecho AND on a3mega** - identical numbers on both |
| 1 - Gate 2 (forecast equivalence) | **PASS**. Ran on a3mega (Slurm job 1153), re-scored on Derecho from the transferred cubes - identical to the digit. G2a 0.083 K, G2b horizon 8.5 d, G2c 0.990. `aires.md`'s G2b was amended first (endpoint -> horizon); see below |
| 1 - Gate 3 (score skill, the go/no-go) | **PASS**, a3mega job **1160** (2 h 08 m, one node). rho_s = 0.797 at t_k = 9 d and 0.959 at 12 d, against a 0.5 threshold; persistence scores 0.062 and 0.129 at the same leads. See "What Gate 3 established" |
| 2 - RES core + C_k tuning | **not started, and now unblocked** - Phase 1 is closed |
| 3-5 | not started |

**Phase 1 is complete: all three gates pass.** Continue with Phase 2 on a3mega (see
"Start here for Phase 2" below) - it holds the calibration, the Gate 2 cubes, the 16 walker
trajectories and the H100s. The full Gate 2 data tree is mirrored on Derecho as well and
re-scores there identically, so Derecho can do CPU analysis, but it is not required again
until Phase 5.

## Verify state (run these, don't trust the table)

```bash
# a3mega (this machine): conda activate moe; cd .../S2S_ExtremeWeather
# Derecho: module load conda && conda activate my-env; cd /glade/derecho/scratch/exu/...

ls -l runs/aires/calib/derived_calib.nc              # calibration built? (~48 MB)
PYTHONPATH=. python -m pytest aires/tests/ -q        # 81 pass, 1 skipped (~70 s)
PYTHONPATH=. python -m aires.calibrate --gate1       # re-scores Gate 1 on the held-out ICs
PYTHONPATH=. python -m aires.gate2 --stage score     # re-scores Gate 2 from cached cubes
ls runs/aires/gate2/cache/*_adapter_cube.nc          # Gate 2 adapter ensemble (24 members)
PYTHONPATH=. JAX_PLATFORMS=cpu python -m aires.walker --check --steps 6   # walker, no GPU
PYTHONPATH=. python -m aires.gate3 --stage plan            # Gate 3 readiness + budget
PYTHONPATH=. python -m aires.gate3 --stage reduce          # Gate 3 verdict, once it has run
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

## What Gate 3 established

**PASS, and by a wide margin at the leads that matter.** 16 free-running GenCast walkers
from the event init to the peak, a global restart state every 3 d, each checkpoint adapted
into an FCN3 IC and scored by M=6 FCN3 members run to the peak. The statistic is the
Spearman correlation, across walkers, between `theta(t_k)` and the walker's OWN realized
`A_L` - the score does not have to be right about the atmosphere, only about the walker.

| t_k | rho_s | 95% CI | p | noise ceiling | persistence |
|---|---|---|---|---|---|
| 3 d | -0.044 | [-0.60, +0.49] | 0.87 | 0.663 | -0.382 |
| 6 d | +0.606 | [+0.15, +0.87] | 0.013 | 0.783 | +0.391 |
| **9 d** | **+0.797** | [+0.51, +0.92] | 0.0002 | 0.952 | +0.062 |
| **12 d** | **+0.959** | [+0.81, +0.99] | <1e-4 | 0.990 | +0.129 |
| 15 d | +0.988 | [+0.91, +1.00] | <1e-4 | 0.998 | +0.556 |

Threshold is `rho_s >= 0.5` at 9 d and 12 d. Both clear it with p <= 2e-4, and the CIs
exclude the threshold at both leads.

**The AI scorer earns its GPU hours - this is the substantive result.** Lancelin et al.'s
Standard-RES scores a walker by the observable's current value, which costs nothing. At the
two gate leads that baseline has **no skill at all** (0.062 and 0.129, both inside the
noise), while FCN3 reaches 0.797 and 0.959. Persistence only becomes useful at 15 d
(0.556), by which point the verification window has already begun. Whatever FCN3 is doing,
it is not reading off the current state.

**Skill switches on between 3 d and 6 d.** At t_k = 3 d the score is worthless
(rho_s = -0.044): FCN3 is being asked for an 18-day forecast, well past its horizon. It
crosses the gate threshold at ~5.5 d and is near-perfect by 12 d. `aires.md`'s `C_k`
schedule already sets `C_1 = 0`, so no score forecast is bought at the first resampling
time - that choice is now measured rather than assumed, and the schedule should not be
shifted earlier.

**M=6 is not the binding constraint where it counts.** The noise ceiling - the largest
correlation `theta` could reach with any target given its own M=6 sampling error - is 0.952
at 9 d and 0.990 at 12 d, so FCN3 is running at 84% and 97% of its own ceiling. It IS
binding at 3-6 d (ceiling 0.66/0.78), but the score has no skill there for unrelated
reasons. Raising M would buy almost nothing at the leads AI+RES actually resamples at.

**The direct-sampling baseline for Phase 4, and the gap AI+RES has to close.**

| index | mean | sd | range | ERA5 observed | walkers reaching it |
|---|---|---|---|---|---|
| PNW box | +2.54 K | 3.02 | [-3.16, +7.49] | **+7.72 K** | **0 / 16** |
| CONUS | +0.97 K | 0.44 | [-0.14, +1.67] | +0.64 K | 13 / 16 |

16 direct samples get to +7.49 K and stop just short of the observed +7.72 K. That is
exactly the headline Phase 4 is set up to answer: does resampling reach a tail that direct
sampling misses? It also settles the box question empirically - on the CONUS mean the
"extreme" is unremarkable (13 of 16 walkers already exceed it), so a CONUS-mean experiment
would have had no tail to find.

Artifacts: `runs/aires/PNW_HeatDome_2021/gate3/*.json|csv`,
`figures/aires/gate3_PNW_HeatDome_2021.png`, 16 walker trajectories (57 GB) and 80 score
cubes under `runs/aires/PNW_HeatDome_2021/`.

## The checkpoint bug that cost two runs (read before touching the walker)

Gate 3's first two attempts produced physically wrong walkers, and the reason is a trap
that is still live for anything else building on `gencast_s2s`:

**`gencast_s2s.config.model_cfg` takes a `res` argument but does NOT use it to choose the
checkpoint.** `res` sets the ERA5 coarsening stride; the checkpoint is
`params_file or GENCAST_PARAMS`, and `GENCAST_PARAMS` is the 1.0 degree **Mini** model. So

```python
M.load_gencast(n_members=1, res=0.25)      # silently loads Mini, runs it on 0.25 deg data
```

Nothing raises - GraphCast builds its mesh from whatever grid the inputs carry - so the run
looks healthy. The output is not: CONUS-mean T2m lost its diurnal cycle inside the first
3-day segment (00Z equal to 12Z to 0.03 K), drifted 26 K cold over 21 days, z500 ended
1780 m2 s-2 low, and realized `A_L` came out at **-22 K** against an observed +7.7 K.
`xres` is unaffected because it passes `params_file=rs["params"]` explicitly from
`xres/xconfig.py::RES_SPECS`.

This also means **job 1154's smoke test never verified the walker's physics.** Its
`250 < t2m < 300` assertion passed on a global mean of 281.47 K that was already ~6 K cold.
The round trip it proved is real; the physics was not checked.

Three things now prevent a recurrence, in order of how much each would have helped:

1. **One loader.** `aires/walker.py::load_bundle` is the only place in `aires/` that
   chooses a checkpoint, and it reads `xres/xconfig.py::RES_SPECS` through
   `aconfig.walker_model_kwargs()` - the same registry xres uses, so the walker and the
   cubes it is compared against cannot name different models. The first fix patched
   `walker.py` and missed the duplicate loader in `gate3.stage_walk`, which is the code the
   walk phase actually runs; that cost the second run.
   `tests/test_walker.py::test_only_one_place_in_aires_loads_a_gencast_checkpoint` parses
   `aires/*.py` with `ast` and pins that there is exactly one call site.
2. **A physical check after the FIRST segment.** `gate3.check_against_reference` compares a
   freshly rolled segment's CONUS-mean T2m against the cached xres ensemble at the same
   valid time - same checkpoint, same init, so a walker must sit inside its spread.
   Measured 298.56 +- 0.162 K at 3 d lead. The broken walkers were **71 sd** out; the
   correct ones came in at 0.2-3.8 sd. This is what turns a 38-minute 8-GPU run plus a
   score phase into a 3-minute failure.
3. **Provenance.** Every state now records the `params_file` it was rolled with. Nothing on
   disk from the first two runs said which model made it.

The rollout itself was never at fault, and that was worth establishing rather than assuming
- a one-GPU controlled comparison (init 2021-06-07, 6 steps, against cached xres member 0):

```
                    step1   step2   step3   step4   step5   step6
xres cube m0       292.424 298.672 292.751 298.554 292.386 298.733
xres 21-day batch  292.422 298.672 292.753 298.550 292.382 298.744
walker 6-step batch 292.420 298.673 292.750 298.554 292.373 298.724
```

Agreement to ~0.01 K, including with the exact `segment_key(1,1)` the walk pool uses. The
walker's batch construction, forcings, ring buffer, restart handoff and RNG are all correct.

## GPU contention: the node is not reliably ours, even with `--exclusive`

Job 1155's score phase died with `CUDA out of memory ... Process 2214660 has 78.05 GiB
memory in use`. That process was a vLLM server from an unrelated project
(`/home/ubuntu/continuum/inklingA/ink_bk_ceiling.sh`), launched **outside Slurm** on the
node Slurm had allocated `--exclusive`, holding ~76 GB on all 8 H100s. It also stalled the
walk phase's XLA compile for ~10 min earlier in the same job.

`slurm/aires_gate3.slurm` now retries each pool up to `AIRES_POOL_ATTEMPTS` (3) times with
a wait. Both stages are cache-aware per artifact, so a retry costs seconds for everything
already on disk and only redoes what is missing; a real bug still fails every attempt. If a
long sweep is running on the cluster this is a workaround, not a fix - check for foreign
processes before blaming the code.

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

1. ~~Gate 3 is the remaining Phase 1 work~~ - **done 2026-08-19, PASS** (job 1160).
   `aires/aindex.py`, `aires/score.py`, `aires/gate3.py` and `slurm/aires_gate3.slurm`
   exist; the per-event boxes `aires.md` deferred to Phase 3 are defined in
   `aires/aindex.py::EVENT_BOXES`. **Phase 2 is the next work.** Measured cost for the
   whole gate on one 8xH100 node: **2 h 08 m** (walk 38 min incl. a ~15 min XLA compile
   with 8 concurrent workers, score ~85 min), 57 GB of walker states + 80 score cubes.
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

## Start here for Phase 2 (on a3mega)

Phase 1 is closed: the adapter is exact where it can be, an adapter IC forecasts like a
native one, and the FCN3 score ranks walkers with rho_s = 0.80 at 9 d and 0.96 at 12 d.
Nothing about the DMC loop is built yet.

```bash
# a3mega, GenCast env (moe), repo root
cd .../S2S_ExtremeWeather && git fetch && git checkout aires && git pull

# confirm Phase 1 still stands (all CPU, ~4 min total)
PYTHONPATH=. python -m pytest aires/tests/ -q            # 81 pass, 1 skipped
PYTHONPATH=. python -m aires.gate2 --stage score         # GATE 2: PASS, exit 0
PYTHONPATH=. python -m aires.gate3 --stage reduce        # GATE 3: PASS, exit 0
```

`--stage reduce` re-derives everything from the cached trajectories and score cubes, so it
is the cheap regression test for the whole gate.

Then Phase 2 (`aires.md`), in order:

1. `aires/dmc.py` - the model-agnostic DMC core: score standardization, the splitting
   function `V_k = C_k * theta_hat`, the weight update, **pivotal sampling**
   (Deville-Tille) for clone/kill counts, genealogy bookkeeping, and the unbiased
   estimator. CPU only. The two tests worth writing before any GPU time are in `aires.md`:
   `sum N_k^i == N` exactly with `E[N_k^i] = w_k^i / wbar_k`, and the OU-process
   unbiasedness check against brute force.
2. The **C_k tuning replay**. This is now cheap in a way the plan did not anticipate: the
   16 Gate 3 trajectories and all 80 score cubes are on disk, so the DMC loop can be
   replayed over real `theta(t_k)` values - not just the persistence proxy `aires.md`
   suggested - to check that `C_k = (0, 1.0, 1.4, 1.8, 2.0)` keeps ESS above N/4 and clone
   multiplicities bounded. No GPU at all.

Two Gate 3 findings that should shape Phase 2:

- **Do not shift `C_k` earlier.** The score has no skill at t_k = 3 d (rho_s = -0.044) and
  only crosses the gate threshold at ~5.5 d. `C_1 = 0` is correct; if anything the case for
  `C_2 = 0` is worth testing in the replay.
- **Do not raise M above 6.** FCN3 already runs at 84% (9 d) and 97% (12 d) of the
  correlation ceiling its own M=6 sampling error imposes, so extra members buy almost
  nothing where resampling actually happens.

When Phase 3 needs GPUs again, `slurm/aires_gate3.slurm` is the working two-env,
two-phase-on-one-allocation pattern to copy, retries included.

## Files

| Path | Role |
|---|---|
| `aires.md` | experiment design, agreed configuration, phase plan |
| `aires/aconfig.py` | the two models' contracts: 72 channels, name maps, grids, gate thresholds |
| `aires/adapter.py` | GenCast state -> FCN3 IC; `GenCastICSource` is the earth2studio DataSource |
| `aires/calibrate.py` | fits + scores the 3 derived channels; `--fit`, `--gate1` |
| `aires/gate2.py` | Gate 2: `--stage {prep,infer,assemble,score}` across the two envs |
| `aires/walker.py` | one GenCast walker segment from a global 2-frame state; `load_bundle` is the ONLY checkpoint chooser; `--check` is CPU-only |
| `aires/aindex.py` | the observable `A_L`, the per-event boxes, the running index and the persistence score |
| `aires/score.py` | the FCN3 score function: M forecasts from a walker checkpoint to the peak |
| `aires/gate3.py` | Gate 3: `--stage {plan,walk,reduce}`, plus the physical check against the xres ensemble |
| `aires/tests/test_adapter.py` | 25 tests incl. the 69-channel bit-exactness check |
| `aires/tests/test_walker.py` | 24 tests: state contract, batch-from-arbitrary-state, cloning RNG |
| `slurm/aires_gate2.slurm` | Gate 2 infer, 8 shards on one a3mega node, `--job-name=Vayuh-s2s` |
| `slurm/aires_gate3.slurm` | Gate 3 walk (moe) + score (fcn3) on ONE allocation, 8 GPUs, retried |
| `runs/aires/calib/derived_calib.nc` | fitted calibration (gitignored, rebuildable in ~2 min) |
| `runs/aires/gate2/` | adapter IC, 24-member adapter cube, scores JSON/CSV |
| `figures/aires/gate2_PNW_HeatDome_2021.png` | the three Gate 2 panels |
| `figures/aires/gate3_PNW_HeatDome_2021.png` | the four Gate 3 panels |
| `runs/aires/<event>/walkers/`, `/scores/`, `/gate3/` | 16 trajectories, 80 score cubes, the verdict |
| `docs/aires_gate2.html` | published Gate 2 summary (artifact source) |
| `docs/aires_gate3.html` | published Gate 3 summary (artifact source) |

Not yet written: `aires/dmc.py`, `aires/aplots.py`, `aires/run_aires.py`,
`slurm/aires_res.slurm`.

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
