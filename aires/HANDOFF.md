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
| 3 - workers + Slurm driver | **done**, smoke-tested on the hardware (job **1161**, 30.7 min). `aires/run_aires.py`, `slurm/aires_res.slurm`, `slurm/aires_env.sh`. The loop rolled clones, resampled on real FCN3 scores, pruned states, and replayed bit-identically in 2 s. See "What Phase 3 established" |
| 4 - run the pilot + figures | **in progress.** `aires/aplots.py` is written and smoke-tested on a synthetic run tree (4 figures, 6 of `aires.md`'s items). The DS baseline is built. The pilot itself is job **1172**, submitted 2026-08-20 |
| 5 | not started |
| follow-up - adapter skill vs truth, 6 events + 4 null controls (branch `adapter-test`) | **DONE.** 10/10 cases rolled and scored (jobs 1162, 1167, 1171). Pooled effect **+0.23% [-5.20, +5.65]**, sign 6/10, extreme-vs-null contrast p 0.505 - no effect detectable at +-5.4%, no regime dependence. See "The adapter test" below |

**Phases 1, 2 and 3 are complete.** All three gates pass, the RES core is validated, and
the driver that runs it on real walkers exists and has been exercised end to end on an
8xH100 node. What is left is Phase 4: submit the pilot and write the figures. See
"Start here for Phase 4" below. a3mega holds the calibration, the Gate 2 cubes, the 16
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

**The fix is more cases, not more members.** Pushing the per-event s.e. below the effect
would need ~M = 100 on one event and would still be one draw of one atmosphere. Across
*independent* cases each sign is a fair coin under the null, so a case count is what buys
power. Six unanimous cases bottom out at p = 0.031; ten reach p = 0.002.

**And the six could not have answered it alone, because all six are extremes.** They are
exactly the flows where a small IC perturbation has the best chance of growing, so an
effect measured only on them cannot be told apart from an effect the extremes manufacture.
Four **null controls** - quiet days, CONUS-mean T2m anomaly inside +-0.33 K - were added
for that reason (`fcn3.fevents.CONTROLS`, drawn from the frozen `p90/cases.csv` control
set). They give the sign test four more independent draws, tighten the pooled CI, and
produce the **extreme-vs-null contrast**, the only statistic here that can say whether the
effect is regime-dependent.

| case | group | metric | native cube | adapter IC | adapter cube |
|---|---|---|---|---|---|
| `PNW_HeatDome_2021` | extreme | t2m_anom | had it | built | **built** (Gate 2) |
| `HurricaneIan_2022` | extreme | u850_speed | had it | built | **built** (job 1162) |
| `WinterStorm_Uri_2021` | extreme | t2m_anom | had it | built | **built** (job 1162) |
| `p90_20231107` | extreme | t2m_anom | had it | built | **built** (job 1162) |
| `p90_20240802` | extreme | t2m_anom | had it | built | **built** (job 1162) |
| `p90_20251224` | extreme | t2m_anom | had it | built | **built** (job 1162) |
| `null_20230324` | null | t2m_anom | **built** (job 1167) | built | **built** (job 1167) |
| `null_20240329` | null | t2m_anom | **built** (job 1167) | built | **built** (job 1167) |
| `null_20241121` | null | t2m_anom | **built** (job 1167) | built | **built** (job 1171) |
| `null_20250526` | null | t2m_anom | **built** (job 1167) | built | **built** (job 1167) |

A control costs **twice** an event: it has no native FCN3 cube from the earlier
experiment, so `slurm/aires_adaptest.slurm` rolls both halves for it.

### Run it

```bash
# 1. LOGIN NODE (either box; env moe on a3mega, my-env on Derecho). CPU only.
bash scripts/adaptest_prep.sh --check          # readiness matrix, writes nothing
bash scripts/adaptest_prep.sh                  # controls' xres frames + ERA5 truth,
                                               # controls' native FCN3 ICs, then every
                                               # adapter IC (~300 MB each)

# 2. a3mega ONLY - FCN3 needs ~64 GB of VRAM, so an 80 GB H100. A Derecho A100-40GB OOMs.
sbatch slurm/aires_adaptest.slurm              # array 0-9%2; built cubes exit early
sbatch --array=6-9%2 slurm/aires_adaptest.slurm    # controls only
squeue -u $USER -n Vayuh-s2s

# 3. LOGIN NODE again, GenCast env.
PYTHONPATH=. python -m aires.adaptest --stage score
PYTHONPATH=. python -m aires.adaptest --stage report    # per-group pooling + contrast
PYTHONPATH=. python -m aires.adaptest --stage maps      # the comparison, drawn
```

Cost: **~15 min of one 8xH100 node per event, ~30 min per control** (two rollouts), and
~9 min of every rollout is eight processes reading the shared 4.18 GB model pickle off
NFS, not compute.

### Result (2026-08-20, all ten cases scored)

**Not resolved, and now with a much tighter bound.** ERA5 truth, event box, ensemble-mean
RMSE, all ten cases pooled:

| group | pooled delta_rel | 95% CI | sign test | I2 |
|---|---|---|---|---|
| extreme (6) | **+1.86%** | [-5.38, +9.09] | 4/6 worse, p 0.688 | 15% |
| null (4) | **-1.87%** | [-10.08, +6.34] | 2/4 worse, p 1.000 | 0% |
| **all (10)** | **+0.23%** | **[-5.20, +5.65]** | 6/10 worse, p 0.754 | 0% |
| contrast | +3.73% | z +0.67, p 0.505 | - | - |

Read it as three separate statements:

1. **No effect is detectable at ±5.4%.** The pooled estimate is +0.23% - an eighth of its
   own error bar. Six events alone gave +1.86% with a ±7.2% detection limit; adding four
   controls both moved the point estimate toward zero and tightened the limit by ~25%.
2. **No regime dependence.** The extreme-vs-null contrast is +3.73% with z = 0.67
   (p = 0.505). The controls do not behave differently from the extremes, which is the
   specific confound they were added to rule out - and it means the six extremes were not,
   in fact, a biased sample for this question. `figures/aires/adaptest_diff_maps.png` shows
   the same thing by eye: the quiet days' adapter-minus-native fields have the same
   character and the same magnitude (1.3-2.0 K) as the events' (1.4-3.6 K).
3. **The per-case scatter is chaos, not adapter error.** Individual cases run from -14.1%
   (p90_20231107) to +14.3% (PNW dome) with s.e. of the same size, and I2 is 0-15% - the
   cases are consistent with ONE common effect plus sampling noise. At a 21 d lead, past the
   ~8.5 d predictability horizon Gate 2 measured, a paired difference is dominated by which
   member happened to land where.

The direction was never a coin flip a priori (Gate 1: the adapter IC carries 0.55-0.63 m/s
on u100m/v100m and ~1.0 kg/m2 on tcwv, and added IC error cannot help in expectation), so
the finding worth quoting is the **upper bound, not the sign**: whatever the adapter costs
in forecast skill at week 3, it is under ~5% of the native run's RMSE, on quiet days as
much as on extremes. For AI+RES that is the number that matters - the walker score is an
ensemble mean, and a bias this far under the ensemble's own sampling noise cannot reorder
walkers.

Caveats that belong with the number:

- **"No anomaly" means no CONUS-MEAN anomaly, not no weather.** The p90 index is the
  cos(lat)-weighted CONUS mean, so a control is a day that mean says nothing happened on.
  `null_20240329` has a CONUS mean of +0.033 K and a strong regional dipole underneath it -
  about -8 K over the northern plains (`figures/aires/adaptest_maps_null_20240329.png`).
  That is still the right control - a day nobody would have picked as an extreme, initialised
  and scored exactly like one - but the null group is not a quiescent atmosphere and should
  not be described as one.
- **No summer control.** Every summer non-event day in the frozen p90 control set predates
  the WB2 cutoff (see the traps below), so the controls are spring/autumn/late-spring while
  two of the six events are warm-season. The contrast is not season-matched.
- **HRRR truth covers only the three xres events**, so its rows are 3 cases and its CIs are
  correspondingly wide (+4.6% to +8.8%, all crossing zero). It leans the same way as ERA5.
- **`u850_speed` (Ian) has no zero-skill reference**, so no skill score is reported for it -
  only the raw scores and the relative delta, which is what pools.

Artifacts: `runs/aires/adaptest/adaptest.json` (verdict + per-group pooling + contrast),
`runs/aires/adaptest/adaptest_events.csv` (52 rows: case x truth x region x stat),
`runs/aires/adaptest/scores/<case>_skill.json`, `figures/aires/adaptest_effect.png` (forest
plot, extremes blue / controls orange), `figures/aires/adaptest_maps_<case>.png` (ten
six-panel comparisons: observed / native / adapter, both error fields, and
adapter-minus-native on its own scale), `figures/aires/adaptest_diff_maps.png` (all ten
difference fields side by side).

### Traps hit while wiring the controls up (2026-08-19/20)

Four of these cost real time and all four are the kind that look like "it hung":

- **The a3mega login node has 15 GB of RAM and the xres prep OOMs on it** for any case
  BEFORE `gencast_s2s.data.WB2_END` (2023-01-10). Pre-cutoff ERA5 comes from the
  WeatherBench2 zarr, and one `.sel(time=t)` on that store makes dask build a task graph
  that reached 7.2 GB and was killed twice, with an EMPTY log both times. After the cutoff
  the reader falls through to `arco_read` (plain numpy per frame, a few hundred MB) - which
  is why the three p90 cases prepped here fine and why **all four controls were chosen
  post-cutoff**. The debug partition is no escape: those nodes are 1 CPU / 7.5 GB.
- **`XRES_PREP_SKIP_MODELS=1` silently disables per-event subprocess isolation** in
  `run_xres.py` (`isolate and not skip_models`), so "skip the checkpoint download" also
  means "build every event in one process". `scripts/adaptest_prep.sh` now loops one event
  per process itself.
- **`ncdump` is not in the `moe` env on a3mega.** The climatology-grid guard in
  `adaptest_prep.sh` was an `ncdump | tr | grep | tr` pipeline; under `set -euo pipefail`
  it exited non-zero with empty output and killed the script before it built anything. It
  is now `scripts/check_clim_grid.py` (xarray), which still fails loudly on a wrong grid.
- **conda's `activate.d` hooks are not `set -u` clean.** Activating `fcn3` under
  `set -euo pipefail` dies on `NVCC_PREPEND_FLAGS: unbound variable`. `set +u` around every
  activation.
- **A foreign, out-of-Slurm process took a node's GPUs mid-job again** (job 1167_8,
  2026-08-20). `/home/ubuntu/continuum/inkling/inkling_cued_eval.py` - same Unix user,
  different project, not submitted through Slurm - held ~79 GiB on six of the eight H100s
  on `nucla3m-a3meganodeset-4` while that node was allocated to this job, and every adapter
  shard died with `torch.OutOfMemoryError`. Same shape as the Gate 3 job 1155 incident. The
  fix is NOT to kill it (it is someone else's live run): cancel the affected array task and
  resubmit it with `--exclude=<node>`. Nothing is lost - both halves are cache-aware, so the
  native cube already on disk was skipped and only the adapter half re-ran. Check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` on the node before blaming
  the code, and `ps -o user,cmd -p <pid>` before blaming another Slurm job.
- **`aires/gate2.py::_event` resolved names against `F.EVENTS`, not `F.ALL_EVENTS`**, so
  every control shard exited "unknown event" *after* the GPU allocation started (job 1167,
  attempt 1). Fixed; the job's own retry loop picked the fix up without a resubmit.
  `aires/score.py` and `aires/gate3.py` still scope to the six on purpose - controls have
  no walker trajectories.

### One real speedup

`ensemble_scores` built the full M x M pairwise-difference array for the CRPS spread term:
a 57 MB temporary per call, 2 x 2000 calls per region per truth source, **4.2 hours for ten
cases** (measured). `crps_spread` now uses the sorted-order identity
`sum_{i<j}|x_i - x_j| = sum_k (2k - M + 1) x_(k)`, which is the same number to 6e-15 and
11x faster - 24 min for the set. `test_crps_spread_matches_the_pairwise_definition` pins it
against the brute-force form.

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
PYTHONPATH=. python -m aires.run_aires --stage prep        # Phase 3: pilot readiness + budget
PYTHONPATH=. python -m aires.run_aires --stage ds          # the direct-sampling baseline
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

## What Phase 3 established

**The driver exists and has been run on the hardware.** `aires/run_aires.py --stage
{prep,walk,score,res,ds,compare}` plus `slurm/aires_res.slurm` and `slurm/aires_env.sh`.
The smoke run (job **1161**, N=8, C=(0, 1.0) at 3/6 d, horizon 12 d, 30.7 min on one node)
exercised every path the pilot needs, including the two that had never run: a leg whose
walkers continue a DIFFERENT slot's state, and a leg made of two segments.

```
  [score] leg 1 (skipped(C=0)): theta over the box mean -3.444, sd 1.310
  [dmc] step 1: C=0.00 R=1.0000 ESS=8.0/8 killed=0 max_mult=1
  [score] leg 2: pool finished in 12.2 min
  [score] leg 2 (fcn3): theta over the box mean +2.458, sd 1.652
  [dmc] step 2: C=1.00 R=1.4842 ESS=4.8/8 killed=2 max_mult=2
  [walk] pruned walker states at segments <= 1 (3.8 GB freed; --keep-states 2)
  [walk] leg 3: 16/16 segment(s) to roll (6 -> 12 d)
```

leg 3's parent map came out `[0, 1, 1, 4, 5, 5, 6, 7]` and each state on disk records the
slot it actually continued (`w02/step03` continues `w01/step02`, `w03/step03` continues
`w04/step02`), so the clone bookkeeping is verified against the files, not just the log.

**Resumption is replay, and it is exact.** The coordinator holds no mutable checkpoint: it
re-runs the whole DMC loop every invocation, and every finished leg returns in seconds.
Re-running the smoke on the LOGIN node reproduced `log Z`, `final_V` and every parent map
**bit-identically in 2.0 s with no GPU**. This works because the loop is a deterministic
function of (the pivotal-sampling seed, the score cubes on disk); the parents are
re-derived and compared against each leg's `walk.json` on every pass, so a rewritten cube
or an edited schedule is a hard error rather than a silent fork.

**Deleting every walker state still leaves a replayable run.** Verified by removing all of
them from the smoke tree: `walk.done` plus the diagnostics cubes are sufficient. That is
what makes `--keep-states` safe.

**Two rates, measured, and one of them was wrong in the docs.** FCN3 runs at **1.40 s per
member-step** (job 1161: 8.41 s for a 6-member lead step), not the 1 s/step
`slurm/aires_gate3.slurm` budgeted - which is exactly why Gate 3's score phase took 85 min
against a predicted 50. GenCast 0.25 deg serial is 27-28 s/step. `--stage prep` now uses
both. **The pilot is ~44 H100-h** (19.4 walk + 25.1 score), ~5.6 h on one 8-GPU node.

**Two savings, both checked rather than assumed:**

- **Legs 1 and 2 cost nothing.** Gate 3's 16 free-running walkers ARE the pilot's first two
  segments: segment 1 trivially (same event, same base seed, same init), segment 2 because
  `C_1 = 0` makes the first resampling the exact identity. That last part is now measured,
  not argued - job 1161's step 1 came back `R = 1.0000`, ESS 8/8, `max_mult = 1`, parents
  the identity. `--stage prep` hardlinks them (link count 2, no bytes copied) after
  validating valid time, parent lineage, checkpoint and base seed. ~1.5 H100-h.
- **No score is bought where `C_k = 0`.** `V_k = C_k * z` is identically zero, so the score
  cannot move a clone or a kill; buying it would cost ~7.7 H100-h at N=64 for the longest
  score forecast in the run. The free persistence index is recorded in its place and
  labelled `skipped(C=0)`. `--score-null` buys it anyway.

**The design deviates from `aires.md` in one structural way, for a hardware reason.** The
plan specifies two long-lived worker pools draining a file-based work queue. They cannot
coexist: the 0.25 deg GenCast checkpoint holds ~64 GB of an 80 GB H100 and FCN3 needs most
of a card too, so a resident walker pool leaves no room for a resident scorer. The pools
alternate instead - the coordinator launches one, waits for it to exit, launches the
other. Nothing is lost: the DMC loop is a barrier, so a pool launch handles exactly one
leg, where every task is the same size and static `w % nshards` sharding is already
exactly balanced. (`cache/claims/` work-stealing exists in `xres` because its events
differ in cost. These do not.) What it costs is one model load per pool per leg - real for
FCN3's 4.18 GB pickle, but only on the first leg, since the node page-caches it.

**The run tree is disjoint from Gate 3's, and this is not tidiness.**
`runs/aires/<event>/res/<tag>/`. Both trees index walkers by slot, but only Gate 3's slots
are lineages; under resampling a slot is a container different ancestries pass through, so
writing the pilot into the gate's paths would overwrite artifacts that are supposed to be
free-running and `gate3 --stage reduce` would start reporting a tilted population.

The same fact creates a validation trap worth stating on its own: **a cached segment
cannot be validated on its timestamp.** Two ancestries produce states at the same valid
time, so an off-lineage state from an earlier run of the same tag is invisible to the
check Gate 3 uses. Every cached segment is therefore checked against the `parent` its own
attributes record (trailing `w<ii>/step<kk>` only, so a Gate-3-seeded state still
validates), plus the checkpoint and the base seed.

**A cold event must be scored on the low tail.** `aindex.tail_sign` negates `A_L` for Uri
before the score reaches `dmc`, because resampling clones the LARGEST score. Without it an
importance-splitting run on a freeze spends its whole GPU budget cloning the warmest
members of a cold extreme. The pilot event is heat, so the sign is +1 and this is latent -
which is exactly why it is pinned by a test.

**Disk is a binding constraint.** Every state retained is 213 GB per event at N=64 against
~600 GB free on a 95%-full shared NFS. `--keep-states` (default 2) prunes states more than
two segments back - rolling a segment needs only its parent - and never touches the
diagnostics cubes, which are 6 MB each and are what every observable is derived from. Peak
92 GB.

**The direct-sampling baseline is built and costs no GPU** (`--stage ds`). It reduces Gate
3's 16 walkers, the cached 24-member xres GenCast cube and the 24-member FCN3 cube through
one code path, and it reproduces Gate 3's published walker numbers to the digit (+2.543,
sd 3.022, range [-3.16, +7.49]) - which is the cross-check that the driver's own trajectory
assembly across the genealogy agrees with `gate3.stage_reduce`.

| direct sample | n | mean A_L | sd | max | reaching the observed +7.72 K |
|---|---|---|---|---|---|
| GenCast walkers (Gate 3) | 16 | +2.54 | 3.02 | +7.49 | **0** |
| GenCast 24-member (xres) | 24 | +2.46 | 2.66 | +7.08 | **0** |
| FCN3 24-member | 24 | +2.29 | 3.10 | +7.87 | 1 |

40 direct GenCast samples stop 0.23 K short of the observed dome. The FCN3 ensemble does
reach it once - worth reporting, but it is a different model, not a baseline for a GenCast
walker.

**Two score backends behind one interface**: `fcn3` (production) and `persistence` (the
paper's Standard-RES, free from the walker's own diagnostics cube). Running the pilot twice
is the control experiment for "the AI scorer earns its GPU hours" on production data rather
than on Gate 3's 16 walkers. The `gencast` PFS backend is **not** implemented - it needs a
nested GenCast ensemble per walker per resampling time, i.e. the cost of the whole
experiment again.

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
   `aires/aindex.py::EVENT_BOXES`. Measured cost for the whole gate on one 8xH100 node:
   **2 h 08 m** (walk 38 min incl. a ~15 min XLA compile with 8 concurrent workers, score
   ~85 min), 57 GB of walker states + 80 score cubes. ~~Phase 2 is the next work.~~
   Phases 2 and 3 are also done; **Phase 4 (submit the pilot, write `aires/aplots.py`) is
   the next work.**
2. ~~Gate 2's G2b criterion needs amending~~ - **done 2026-08-18**. `aires.md` now
   specifies G2b as a >= 7 d rank-equivalence horizon and requires G2c; `aires/gate2.py`
   implements it and `--stage score` exits 0. Gate 2 is recorded as PASS.
3. **The calibration was fitted on ERA5, but production inputs are GenCast forecasts** at
   3-15 day lead, which are smoother than analysis. Gate 3 is what measures whether that
   matters; Gate 1 cannot see it, and Gate 2 could not either (both ICs there are ERA5).
4. **The tcwv residual (~4%) is close to the floor** for any linear function of 13 q
   levels - the sub-1000 hPa moisture is not in the state. Whether 4% matters at all is
   Gate 2's question, not something to optimise further blind.
5. **`normalization_check` is uninformative at small N and must not be read from one
   run.** The smoke run (N=8, one resampling) returned 1.61 where the estimator wants ~1.
   That is not evidence of a bug: it is a product of K estimated normalizations times a
   mean dominated by its smallest term, and `dmc.py`'s own docstring records a standard
   deviation of ~0.5 on the reference OU problem at N=256. At the pilot's N=64 and K=5 it
   is worth watching across the two backend runs, not diagnosing from one number.
6. **The pilot has one event.** Everything from Gate 3 onward - the skill curve the C_k
   schedule was tuned on, the box, the DS gap - is PNW_HeatDome_2021. Uri and Ian have
   boxes defined and would run unchanged (Uri exercises the cold-tail sign, which is
   currently only pinned by a test), but neither has walkers.
7. **Calibration seasonal coverage is thin in summer**: the 90 p90 ICs have 0 June and 2
   July frames. The held-out June 7 PNW case passed anyway, so it generalises, but a
   summer-heavy fit would be a cheap improvement if Gate 2 disappoints.

## Start here for Phase 4 (on a3mega)

Phases 1-3 are closed. The adapter is exact where it can be, an adapter IC forecasts like
a native one, the FCN3 score ranks walkers, the DMC core is validated against an analytic
problem, and the driver that runs it on real walkers has been exercised end to end on an
8xH100 node. What is left is to submit the pilot and draw the figures.

```bash
# a3mega, GenCast env (moe), repo root
cd .../S2S_ExtremeWeather && git fetch && git checkout aires && git pull

# confirm Phases 1-3 still stand (all CPU, ~6 min)
PYTHONPATH=. python -m pytest aires/tests/ -q            # 133 pass, 1 skipped
PYTHONPATH=. python -m aires.gate2 --stage score         # GATE 2: PASS, exit 0
PYTHONPATH=. python -m aires.gate3 --stage reduce        # GATE 3: PASS, exit 0
PYTHONPATH=. python -m aires.ctune --check               # the surrogate's fit to Gate 3

# the pilot: readiness, budget, disk, and the Gate 3 seeding -- must print READY: yes
PYTHONPATH=. python -m aires.run_aires --stage prep
```

`prep` is the gate. It validates the init frames and the calibration, hardlinks the legs
Gate 3 already rolled, prints the budget from MEASURED rates, checks the projected disk
against what is actually free, and freezes the configuration in `run.json`. Run it on the
LOGIN node - it is CPU-only, and a GPU job must be a pure cache-hit-then-compute.

Then the pilot itself, ~44 H100-h / ~5.6 h on one node:

```bash
sbatch slurm/aires_res.slurm                             # N=64, fcn3 backend
# and the control, which costs only its walkers (no FCN3 at all):
AIRES_RES_TAG=persist AIRES_BACKEND=persistence sbatch --export=ALL slurm/aires_res.slurm

# afterwards, login node, moe:
PYTHONPATH=. python -m aires.run_aires --stage ds        # already built; re-run is free
PYTHONPATH=. python -m aires.run_aires --stage compare   # -> compare.json + compare_curve.csv
```

A walltime kill is resumed by **resubmitting the same script**: every finished leg replays
in seconds and no GPU pool is launched for it. Do not start a new tag to resume.

Then Phase 4's actual new work, `aires/aplots.py` (six figures, listed in `aires.md`).
`--stage compare` already writes everything they need: the exceedance curve for AI+RES and
each direct-sampling source, the per-step ESS and multiplicities, the genealogy, the
per-walker running index, and the ancestry of every final walker.

Settled by Phases 2-3, do not re-litigate: `C_k = (0, 1.0, 1.4, 1.8, 2.0)` as written,
M = 6, N = 64, states pruned to the last 2 segments, no score bought where `C_k = 0`.

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
| `aires/aplots.py` | the Phase 4 figures: exceedance, diagnostics, genealogy, composite |
| `aires/run_aires.py` | the production driver: `--stage {prep,walk,score,res,ds,compare}`. `res` IS the coordinator and launches both GPU pools itself |
| `aires/tests/test_adapter.py` | 25 tests incl. the 69-channel bit-exactness check |
| `aires/tests/test_walker.py` | state contract, batch-from-arbitrary-state, cloning RNG, checkpoint choice |
| `aires/tests/test_dmc.py` | pivotal sampling, the C=0 identity, OU unbiasedness and variance reduction |
| `aires/tests/test_ctune.py` | the surrogate reproduces the skill curve and the correlations it was not given |
| `aires/tests/test_aplots.py` | sibling pairing, the exceedance standard error, weighted quantiles, and that each figure renders |
| `aires/tests/test_run_aires.py` | leg schedule, lineage validation, the replay guard, pruning, the cold-event sign; runs the real DMC loop through the real coordinator on a fake tree |
| `slurm/aires_gate2.slurm` | Gate 2 infer, 8 shards on one a3mega node, `--job-name=Vayuh-s2s` |
| `slurm/aires_gate3.slurm` | Gate 3 walk (moe) + score (fcn3) on ONE allocation, 8 GPUs, retried |
| `slurm/aires_res.slurm` | the pilot: ONE process (the coordinator), 12 h, `--job-name=Vayuh-s2s` |
| `slurm/aires_env.sh` | the ONE place either conda env is activated; deactivates first so `moe` cannot leak into `fcn3` |
| `runs/aires/calib/derived_calib.nc` | fitted calibration (gitignored, rebuildable in ~2 min) |
| `runs/aires/gate2/` | adapter IC, 24-member adapter cube, scores JSON/CSV |
| `figures/aires/gate2_PNW_HeatDome_2021.png` | the three Gate 2 panels |
| `figures/aires/gate3_PNW_HeatDome_2021.png` | the four Gate 3 panels |
| `figures/aires/ctune_PNW_HeatDome_2021.png` | the C_k schedule replay |
| `runs/aires/<event>/walkers/`, `/scores/`, `/gate3/` | 16 trajectories, 80 score cubes, the verdict |
| `runs/aires/<event>/res/<tag>/` | the pilot's own tree - walkers, scores, per-leg manifests, `res_result.json`. DISJOINT from the gate's |
| `runs/aires/<event>/ds_baseline.json` | the direct-sampling baseline (no GPU, tag-independent) |
| `docs/aires_gate2.html` | published Gate 2 summary (artifact source) |
| `docs/aires_gate3.html` | published Gate 3 summary (artifact source) |

Everything `aires.md` names is written. What is outstanding is the pilot's own
output: job 1172's `compare.json` and the four figures drawn from it.

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
