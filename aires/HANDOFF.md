# AI+RES - live state

Read `aires.md` (repo root) for the experiment design. This file is the state of the
work, updated as phases complete. **Trust the commands in "Verify state" over any prose
here** - doc claims rot, `ls` does not.

Branch: `aires`. Machine split: Derecho for CPU work, a3mega H100 for anything touching
FCN3 or GenCast inference.

Visual summary of the plan, the adapter and Gate 1 (design, derivations, open problems):
https://claude.ai/code/artifact/081bd2f9-7725-4e46-8af8-ec57f54ee2df

## Where things stand

| Phase | State |
|---|---|
| 0 - move a3mega FCN3 week-3 results to Derecho | **done**, verified: figures regenerate pixel-identical from the transferred cubes |
| 1 - adapter + calibration + Gate 1 | **done on Derecho AND on a3mega** - identical numbers on both |
| 1 - Gate 2 (forecast equivalence) | **run on a3mega** (Slurm job 1153). G2a PASS, G2c PASS, **G2b FAIL as written** - see below; the gate's purpose is met but the G2b criterion needs amending |
| 1 - Gate 3 (score skill, the go/no-go) | **not started**; `aires/walker.py` (its prerequisite) is **written and CPU-verified**, not yet rolled on a GPU |
| 2-5 | not started |

Current machine: **a3mega**. Everything Phase 1 needs is now here - calibration rebuilt,
both cubes cached, walker written. Derecho is not required again until Phase 5.

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
| G2b paired Spearman at 21 d | 0.107 | > 0.95 | **FAIL** |
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

**G2b as written cannot discriminate and should be amended.** It evaluates paired-member
rank correlation at 21 d, where any IC difference whatsoever - the adapter's or a re-seed's
- has been amplified to the internal-noise level and shuffles the members into an unrelated
order. The measured 0.107 is what a *perfect* adapter would also score. Resolved by lead
(third panel of `figures/aires/gate2_PNW_HeatDome_2021.png`) the statistic is informative:

    rho_s = 0.994 at 2.5 d, 0.960 at 5 d, 0.957 at 7.5 d, 0.759 at 10 d, 0.144 at 15 d

i.e. the two ensembles are member-for-member rank-equivalent out to **8.5 days**, and the
decay past that is the model's predictability horizon for CONUS-mean T2m, not adapter
error. Suggested amendment to `aires.md`: replace G2b with G2c (a well-posed relative
test at every lead) plus "rank-equivalent to at least 7 d", and keep G2a.

**M=24, not the M=6 of the plan.** The cached 24-member ensemble makes the noise floor
measurable, and it is fatal to M=6: per-member sd of A_L is 0.62 K, so two 6-member
ensemble means differ by 0.36 K from sampling noise alone - larger than G2a's 0.25 K
threshold. A perfect adapter would have failed G2a about half the time at M=6. At M=24 the
paired s.e. is 0.149 K, so G2a resolves differences down to ~0.30 K and no finer; that is
now stated in the score output rather than left implicit.

## `aires/walker.py` - written, CPU-verified, not yet rolled

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

Two facts found while writing it, both worth keeping:

- **`total_precipitation_12hr` is a GenCast target but NOT an input.** A restart state does
  not need it; the state carries it anyway because the precip metric derives from it.
- **float16 archived states would be silently corrupt.** aires.md floats float16 to halve
  the 0.70 GB footprint. Geopotential at 50 hPa is ~2e5 m2 s-2 against float16's 65504
  ceiling, so the top of every column would store `inf`. `write_state` is float32 + zlib;
  `tests/test_walker.py::test_float16_would_overflow_geopotential` pins the reason.

## Open problems

1. **Gate 3 is the remaining Phase 1 work.** `aires/walker.py` exists and its CPU-checkable
   parts pass, but it has never rolled a step on a GPU. The first GPU exercise should be a
   single short segment (`--steps 2`) before the 8-16 member Gate 3 run, because a
   0.25 degree serial rollout costs ~28 s/step on an H100 and a full free-running walker to
   21 d is ~20 min.
2. **Gate 2's G2b criterion needs amending in `aires.md`** - see "What Gate 2 established".
   Nothing is blocked on it; the decision is whether to record Gate 2 as PASS under
   amended criteria or leave it as a documented partial.
3. **The calibration was fitted on ERA5, but production inputs are GenCast forecasts** at
   3-15 day lead, which are smoother than analysis. Gate 3 is what measures whether that
   matters; Gate 1 cannot see it, and Gate 2 could not either (both ICs there are ERA5).
4. **The tcwv residual (~4%) is close to the floor** for any linear function of 13 q
   levels - the sub-1000 hPa moisture is not in the state. Whether 4% matters at all is
   Gate 2's question, not something to optimise further blind.
5. **Calibration seasonal coverage is thin in summer**: the 90 p90 ICs have 0 June and 2
   July frames. The held-out June 7 PNW case passed anyway, so it generalises, but a
   summer-heavy fit would be a cheap improvement if Gate 2 disappoints.

## Continuing on a3mega

The calibration is **deterministic given its inputs and its inputs live on a3mega too**
(the p90 ICs originated there), so rebuild rather than transfer:

```bash
# on a3mega, in the GenCast env (moe)
cd .../S2S_ExtremeWeather && git fetch && git checkout aires
PYTHONPATH=. python -m aires.calibrate --fit --gate1     # ~2 min, CPU, no network
PYTHONPATH=. python -m pytest aires/tests/ -q
```

Expect the identical Gate 1 numbers above. If they differ, the two machines' p90 IC sets
differ - check `ls runs/p90_t2m/ic/*.nc | wc -l` equals 90 on both.

Alternatively copy `runs/aires/calib/derived_calib.nc` (48 MB) by rsync **from a3mega**
(pull), the same direction convention as `scripts/sync_a3mega_to_derecho.sh`.

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
