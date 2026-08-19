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
- Adapter skill vs truth (does the adapter cost skill? the noise floor, the dome miss):
  https://claude.ai/code/artifact/853f6f70-7026-40c0-945f-b84711a2bdae
  (source: `docs/aires_adapter_skill.html`)

## Where things stand

| Phase | State |
|---|---|
| 0 - move a3mega FCN3 week-3 results to Derecho | **done**, verified: figures regenerate pixel-identical from the transferred cubes |
| 1 - adapter + calibration + Gate 1 | **done on Derecho AND on a3mega** - identical numbers on both |
| 1 - Gate 2 (forecast equivalence) | **PASS**. Ran on a3mega (Slurm job 1153), re-scored on Derecho from the transferred cubes - identical to the digit. G2a 0.083 K, G2b horizon 8.5 d, G2c 0.990. `aires.md`'s G2b was amended first (endpoint -> horizon); see below |
| 1 - Gate 3 (score skill, the go/no-go) | **PASS**, a3mega job **1160** (2 h 08 m, one node). rho_s = 0.797 at t_k = 9 d and 0.959 at 12 d, against a 0.5 threshold; persistence scores 0.062 and 0.129 at the same leads. See "What Gate 3 established" |
| 2 - RES core + C_k tuning | **done**, CPU only. `aires/dmc.py` validated against an analytic Ornstein-Uhlenbeck problem; `aires/ctune.py` replays the schedule on the measured Gate 3 skill curve. `aires.md`'s C_k is **validated as-is**. See "What Phase 2 established" |
| follow-up - adapter skill vs truth, 6 events (branch `adapter-test`) | **code written and CPU-verified; GPU half not run.** `aires/adaptest.py` + `slurm/aires_adaptest.slurm` + `scripts/adaptest_prep.sh`. 1 of 6 adapter cubes exists; 5 need ~75 node-min on a3mega. See "The adapter test" below |
| 3-5 | not started |

**Phases 1 and 2 are complete.** All three gates pass and the RES core is written and
validated without a GPU. Phase 3 (the workers and the Slurm driver) is next; see
"Start here for Phase 3" below. a3mega holds the calibration, the Gate 2 cubes, the 16
walker trajectories and the H100s. The full Gate 2 data tree is mirrored on Derecho as well and
re-scores there identically, so Derecho can do CPU analysis, but it is not required again
until Phase 5.

## The adapter test (branch `adapter-test`) - the open question and how to close it

Gate 2 established that an adapter IC forecasts *like* a native one. Scoring both ensembles
against **observed truth** afterwards (`runs/aires/gate2/skill/`, artifact above) left a
residue that one event cannot settle:

- On most ensemble-mean scores the adapter run is the marginally **worse** of the two:
  CONUS ens-mean RMSE 1.739 vs 1.640 K, PNW box 6.312 vs 5.523 K, PNW CRPS 4.733 vs 3.684 K.
  The sign holds against HRRR truth as well as ERA5.
- **But nothing is resolved.** Every paired-bootstrap CI crosses zero; the strongest case
  (PNW CRPS) still flips sign in 10% of resamples, i.e. two-sided p ~ 0.2. And the two
  tightest reductions **disagree in sign**: seed-matched per-member RMSE over CONUS makes
  the adapter marginally *better* (-0.050 +- 0.154 K), as does its CONUS A_L bias.
- **A tally across metrics is not evidence.** The single-event analysis leaned "adapter
  worse" on ten of fourteen signed comparisons, but all fourteen come from the same 24
  member pairs at one init, so the tally carries roughly one event's worth of information.
- **Direction was never a coin flip.** The adapter IC carries real added error (Gate 1:
  0.63 m/s on u100m/v100m, 4.6% on tcwv) and adding IC error cannot help in expectation, so
  a small degradation is the **prior**, not a finding. What is worth reporting is its size
  and its upper bound.
- **What Gate 2's data does establish** is a ceiling: below ~0.3 K of CONUS RMSE and ~1.4 K
  over the event box (2 paired s.e.), under the ensemble's own sampling noise and under the
  0.83 K by which ERA5 and HRRR disagree about what happened. For AI+RES that bound is the
  thing that matters - the walker score is an ensemble mean, and a bias smaller than the
  sampling noise cannot reorder walkers.

**The fix is more events, not more members.** Pushing the per-event s.e. below the effect
would need ~M = 100 on one event and would still be one draw of one atmosphere. Across
*independent* events each sign is a fair coin under the null, so six events is a real test.
All six already have native FCN3 cubes from the FCN3-vs-GenCast experiment, so only the
adapter halves cost GPU time.

| | metric | adapter IC | adapter cube |
|---|---|---|---|
| `PNW_HeatDome_2021` | t2m_anom | built | **built** (Gate 2) |
| `HurricaneIan_2022` | u850_speed | needs prep | needs infer |
| `WinterStorm_Uri_2021` | t2m_anom | **built** (Derecho, 2026-08-19) | needs infer |
| `p90_20231107` | t2m_anom | needs prep | needs infer |
| `p90_20240802` | t2m_anom | needs prep | needs infer |
| `p90_20251224` | t2m_anom | needs prep | needs infer |

### Run it

```bash
# 1. LOGIN NODE (either box; env moe on a3mega, my-env on Derecho). CPU only.
bash scripts/adaptest_prep.sh --check          # readiness matrix, writes nothing
bash scripts/adaptest_prep.sh                  # builds the 5 missing adapter ICs (~300 MB each)

# 2. a3mega ONLY - FCN3 needs ~64 GB of VRAM, so an 80 GB H100. A Derecho A100-40GB OOMs.
sbatch slurm/aires_adaptest.slurm              # array 0-5%2; built cubes exit early
squeue -u $USER -n Vayuh-s2s

# 3. LOGIN NODE again, GenCast env.
PYTHONPATH=. python -m aires.adaptest --stage score
PYTHONPATH=. python -m aires.adaptest --stage report
```

Cost: **~15 min of one 8xH100 node per event, ~75 node-min for the five** - and ~9 min of
each 15 is eight processes reading the shared 4.18 GB model pickle off NFS, not compute.

### What the code does and does not do

- `aires/adaptest.py` adds the six-event **analysis**, not a second pipeline: the adapter-IC
  builder and the FCN3 driver are `aires/gate2.py`'s, called through it. There is exactly one
  of each in this repo and that stays true.
- Three statistics, in increasing order of what they can say: a **paired bootstrap** over the
  24 matched member pairs (resampling a *common* index set - break the pairing and the s.e.
  roughly doubles, which `tests/test_adaptest.py` pins as an inequality); an **exact binomial
  sign test** across events, which is the test with real power; and an **inverse-variance
  pooled** relative effect `(adapter - native) / native`, dimensionless so a wind event and a
  temperature event can pool, with an I^2 heterogeneity check.
- Each event is scored on **its own metric**, so Ian is `u850_speed`. That metric has no
  zero-skill reference (a zero wind field is not a climatology), so no skill score is reported
  for it - only raw RMSE/CRPS and the relative delta, which is what pools anyway.
- Cold events need no sign handling: RMSE and CRPS are sign-blind and the A_L comparison uses
  `|A_L - observed|`, so Uri's negative tail is automatic. Stated because
  `xres.xmetrics.extreme_sign` exists and it is tempting to reach for.

### Ceilings to know before reading the output

- **Six events cannot beat p = 0.031.** That is the exact two-sided binomial p when all six
  agree. `test_six_events_cannot_reach_p_below_0p03` pins it so a unanimous result is not
  over-read.
- At the PNW event's precision, six events give a detection limit of roughly **+-11%** of the
  native score (it was +-27% at one event). If the true effect is a few percent, this design
  will bound it, not measure it - and that is still the useful answer.
- The array index -> event mapping is duplicated in the Slurm script so a shell can pick an
  event without importing Python. A silent reordering would pair one event's adapter cube with
  another's native cube and produce finite, wrong numbers. Two guards: the job re-checks
  against `fcn3.fevents.ORDER` at run time, and
  `test_event_order_matches_the_slurm_array` parses the script before submission.

## Verify state (run these, don't trust the table)

```bash
# a3mega (this machine): conda activate moe; cd .../S2S_ExtremeWeather
# Derecho: module load conda && conda activate my-env; cd /glade/derecho/scratch/exu/...

ls -l runs/aires/calib/derived_calib.nc              # calibration built? (~48 MB)
PYTHONPATH=. python -m pytest aires/tests/ -q        # 110 pass, 1 skipped (~2 min)
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

## What Phase 2 established

**`aires.md`'s C_k schedule is validated as written, and one of my own follow-up
hypotheses is refuted.** No GPU was used.

`aires/dmc.py` is the DMC core - standardization, the splitting function, pivotal
sampling, genealogy, and the estimator `E_P[phi] = Z * mean(phi * exp(-V_K))`. It is
model-agnostic, so the thing that will cost ~46 H100-h per event was validated in eight
seconds of CPU against an Ornstein-Uhlenbeck walker whose exceedance probabilities are
analytic: unbiased at 2 and 3 sigma within Monte Carlo error, and at N=256 it estimates
P(X > 3 sigma) with 3x less spread than direct sampling.

**Two bugs the tests found that the code review did not:**

- **Pivotal sampling had the wrong branch probability** in Deville-Tille's first case
  (`p_j/(p_i+p_j)` where it must be `p_i/(p_i+p_j)`). It kept the total exactly right and
  returned the correct *multiset* of marginals while attaching them to the **wrong units**,
  so the population was cloned toward the wrong walkers and the estimator came out ~10x
  high. Sum-is-exact and marginals-are-right-on-average both passed.
- **`run()` returned the pre-resampling population while the weights described the
  post-resampling one.** `propagate()` is now called once past the last resampling to carry
  survivors to the horizon, so `states`, `final_V` and `weights` always describe one
  population.

### The tuning replay

`aires/ctune.py` replays the loop on a surrogate whose only input is measured: the Gate 3
skill curve `rho_k = (0.00, 0.61, 0.82, 0.95, 1.00)` at leads 3/6/9/12/15 d. The five
scores of a walker are five forecasts of the *same* outcome, not five steps of a random
walk - a Markov chain predicts `corr(z_6d, z_15d) = 0.28` where the data says 0.61 - so the
model is signal-plus-noise with a martingale signal. It predicts `corr(theta_j, theta_k) =
rho_j`, which it was **not** fitted to, with a mean error of 0.061 against a sampling error
of 0.28 at N=16.

At N=64, 2000 repeats (ESS/N per resampling time; target is > 0.25 at the last one):

| schedule | 3 d | 6 d | 9 d | 12 d | 15 d | final p10 | max clone mult (mean/p90) |
|---|---|---|---|---|---|---|---|
| **aires.md** (0,1.0,1.4,1.8,2.0) | 1.00 | 0.43 | 0.36 | 0.38 | **0.59** | 0.41 | 8.5 / 13 |
| skip 6d (0,0,1.4,1.8,2.0) | 1.00 | 1.00 | 0.25 | 0.38 | 0.57 | 0.38 | 11.6 / 19 |
| paper-like (0,0,1.6,1.8,2.0) | 1.00 | 1.00 | 0.20 | 0.37 | 0.56 | 0.36 | 14.2 / 24 |
| gentle (0,0.5,0.8,1.1,1.4) | 1.00 | 0.79 | 0.68 | 0.67 | 0.77 | 0.68 | 3.8 / 5 |
| strong (0,1.5,2.0,2.5,3.0) | 1.00 | 0.22 | 0.19 | 0.20 | **0.28** | **0.10** | 15.4 / 26 |

Tail estimate, spread divided by the true probability (lower is better; the surrogate's
answer is analytic):

| schedule | 2 sigma | 3 sigma | 3.5 sigma | 4 sigma |
|---|---|---|---|---|
| **aires.md** | 0.60 | **1.51** | **2.90** | **5.33** |
| skip 6d | 0.66 | 2.15 | 5.14 | 14.30 |
| paper-like | 0.65 | 2.09 | 5.12 | 15.34 |
| gentle | 0.61 | 1.88 | 3.47 | 5.82 |
| strong | 0.84 | 1.47 | 2.68 | 5.63 |
| direct sampling | 0.82 | 3.43 | 8.03 | 24.64 |

**Three findings:**

1. **Keep `C_k = (0, 1.0, 1.4, 1.8, 2.0)`.** It clears the ESS target with room (0.59 final,
   0.41 at the 10th percentile), keeps clone multiplicities bounded (8.5 mean, 13 at p90),
   stays unbiased, and has the best deep-tail spread of every candidate tried.
2. **`C_2 = 0` is refuted** - and this HANDOFF proposed it after Gate 3, so the note is a
   correction. The reasoning was that Gate 3 found no score skill at 3 d; but `C_1` is
   *already* the 3 d coefficient and it is already zero. `C_2` acts at **6 d**, where the
   score does have skill (rho = 0.61), and switching it off makes the 4 sigma estimate
   **2.7x noisier** (5.33 -> 14.30). Turning off a resampling time where the score works
   costs real accuracy.
3. **The schedule cannot be pushed much harder.** `strong` is the only candidate that fails
   the ESS target (0.28 mean, 0.10 at p10, up to 26 children from one walker) while buying
   nothing in the tail.

**What the pilot can honestly claim.** The variance reduction over direct sampling at
N=64, K=5 is 1.4x at 2 sigma, 2.3x at 3 sigma, 2.8x at 3.5 sigma and 4.6x at 4 sigma - real,
growing with depth exactly as importance splitting should, and nothing like the ~100x the
paper reports. That speedup comes from N=400, K=6 and a target far deeper in the tail. The
claim this experiment supports is "RES reaches a tail that direct sampling misses", which
is what `aires.md` already scoped, not a large-factor speedup.

**Caveat, stated plainly.** The surrogate is a Gaussian caricature calibrated to one
event's skill curve, and that curve came from 16 walkers, so each `rho_k` carries a
standard error of roughly 0.1-0.25. The ESS and multiplicity columns are what it is for;
the exceedance column is a schedule comparison, not a prediction of the pilot's answer.

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

## Start here for Phase 3 (on a3mega)

Phases 1 and 2 are closed. The adapter is exact where it can be, an adapter IC forecasts
like a native one, the FCN3 score ranks walkers (rho_s 0.80 at 9 d, 0.96 at 12 d), and the
DMC core is written and validated against an analytic problem. What does not exist yet is
the machinery to run the loop on real walkers.

```bash
# a3mega, GenCast env (moe), repo root
cd .../S2S_ExtremeWeather && git fetch && git checkout aires && git pull

# confirm Phases 1-2 still stand (all CPU, ~4 min)
PYTHONPATH=. python -m pytest aires/tests/ -q            # 110 pass, 1 skipped
PYTHONPATH=. python -m aires.gate2 --stage score         # GATE 2: PASS, exit 0
PYTHONPATH=. python -m aires.gate3 --stage reduce        # GATE 3: PASS, exit 0
PYTHONPATH=. python -m aires.ctune --check               # the surrogate's fit to Gate 3
```

`gate3 --stage reduce` re-derives the whole gate from cached trajectories and score cubes,
so it is the cheap regression test for Phase 1; the pytest run covers Phase 2 entirely.

Then Phase 3 (`aires.md`), which is plumbing rather than new science:

1. `aires/run_aires.py --stage {prep,walk,score,res,ds,compare}`, cache-aware per
   (walker, step), matching the stage idiom the rest of the repo uses.
2. `slurm/aires_res.slurm`. **Copy `slurm/aires_gate3.slurm`** - it is the working
   pattern: two conda envs and two GPU phases on ONE allocation, 8 shards per phase,
   retries, and a plan check that refuses to start on a bad cache.
3. The one genuinely new piece: the DMC loop is a **barrier** at each `t_k`, so the walk
   and score phases have to interleave per step rather than run to completion. A rank-0
   coordinator advancing `k` only when all N walker segments and all N x M score artifacts
   for step `k` exist on disk is what `aires.md` specifies; the `cache/claims/`
   work-stealing pattern in `xres/xinference.py` is the model for the workers.

`aires/dmc.py` is driven by exactly two callables, `propagate(step, parents, states)` and
`score(step, states)`, so Phase 3's job is to make those two read and write the run tree -
nothing about the algorithm needs to change. Note that `propagate` is called once past the
last resampling, to carry survivors from `t_K` to `t_f`.

Settled by Phase 2, do not re-litigate: `C_k = (0, 1.0, 1.4, 1.8, 2.0)` as written, M = 6,
N = 64.

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
| `aires/dmc.py` | the RES core: splitting, pivotal sampling, genealogy, the unbiased estimator |
| `aires/ctune.py` | replays the C_k schedule on a surrogate calibrated to the Gate 3 skill curve |
| `aires/tests/test_adapter.py` | 25 tests incl. the 69-channel bit-exactness check |
| `aires/tests/test_walker.py` | state contract, batch-from-arbitrary-state, cloning RNG, checkpoint choice |
| `aires/tests/test_dmc.py` | pivotal sampling, the C=0 identity, OU unbiasedness and variance reduction |
| `aires/tests/test_ctune.py` | the surrogate reproduces the skill curve and the correlations it was not given |
| `slurm/aires_gate2.slurm` | Gate 2 infer, 8 shards on one a3mega node, `--job-name=Vayuh-s2s` |
| `slurm/aires_gate3.slurm` | Gate 3 walk (moe) + score (fcn3) on ONE allocation, 8 GPUs, retried |
| `runs/aires/calib/derived_calib.nc` | fitted calibration (gitignored, rebuildable in ~2 min) |
| `runs/aires/gate2/` | adapter IC, 24-member adapter cube, scores JSON/CSV |
| `figures/aires/gate2_PNW_HeatDome_2021.png` | the three Gate 2 panels |
| `figures/aires/gate3_PNW_HeatDome_2021.png` | the four Gate 3 panels |
| `figures/aires/ctune_PNW_HeatDome_2021.png` | the C_k schedule replay |
| `runs/aires/<event>/walkers/`, `/scores/`, `/gate3/` | 16 trajectories, 80 score cubes, the verdict |
| `docs/aires_gate2.html` | published Gate 2 summary (artifact source) |
| `docs/aires_gate3.html` | published Gate 3 summary (artifact source) |

Not yet written: `aires/aplots.py`, `aires/run_aires.py`, `slurm/aires_res.slurm`.

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
