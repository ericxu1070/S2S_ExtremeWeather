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
- Phase 4 pilot results (the trajectory plot, the maps, the estimator caveats):
  https://claude.ai/code/artifact/d4addb42-3ab7-42a0-baec-fd1b3cf47708
  (source: `docs/aires_pilot.html` + `scripts/embed_figs.py`, which inlines the PNGs as
  data URIs - the Artifact CSP blocks every external host, so the page cannot link them)

## Where things stand

| Phase | State |
|---|---|
| 0 - move a3mega FCN3 week-3 results to Derecho | **done**, verified: figures regenerate pixel-identical from the transferred cubes |
| 1 - adapter + calibration + Gate 1 | **done on Derecho AND on a3mega** - identical numbers on both |
| 1 - Gate 2 (forecast equivalence) | **PASS**. Ran on a3mega (Slurm job 1153), re-scored on Derecho from the transferred cubes - identical to the digit. G2a 0.083 K, G2b horizon 8.5 d, G2c 0.990. `aires.md`'s G2b was amended first (endpoint -> horizon); see below |
| 1 - Gate 3 (score skill, the go/no-go) | **PASS**, a3mega job **1160** (2 h 08 m, one node). rho_s = 0.797 at t_k = 9 d and 0.959 at 12 d, against a 0.5 threshold; persistence scores 0.062 and 0.129 at the same leads. See "What Gate 3 established" |
| 2 - RES core + C_k tuning | **done**, CPU only. `aires/dmc.py` validated against an analytic Ornstein-Uhlenbeck problem; `aires/ctune.py` replays the schedule on the measured Gate 3 skill curve. `aires.md`'s C_k is **validated as-is**. See "What Phase 2 established" |
| 3 - workers + Slurm driver | **done**, smoke-tested on the hardware (job **1161**, 30.7 min). `aires/run_aires.py`, `slurm/aires_res.slurm`, `slurm/aires_env.sh`. The loop rolled clones, resampled on real FCN3 scores, pruned states, and replayed bit-identically in 2 s. See "What Phase 3 established" |
| 4 - run the pilot + figures | **done**, job **1172** (7 h 45 m). AI+RES reaches the observed dome (42/64 walkers) where 40 direct GenCast members do not (0/40); P = 0.054. See "What Phase 4 established" |
| 4b - rebuild the pilot's probabilities as binned PDFs vs ERA5 | **done**, job **1174** (45 s, debug CPU node). The weighted spatial PDF reaches ERA5's tail: box P(point >= +9 K) = 0.0505 vs ERA5's 0.0848, where 24 direct members give 0.0011. See "The pilot's probabilities as binned PDFs" |
| 5 | not started |
| follow-up - adapter skill vs truth, 6 events + 4 null controls (branch `adapter-test`) | **DONE.** 10/10 cases rolled and scored (jobs 1162, 1167, 1171). Pooled effect **+0.23% [-5.20, +5.65]**, sign 6/10, extreme-vs-null contrast p 0.505 - no effect detectable at +-5.4%, no regime dependence. See "The adapter test" below |
| 5-scoping - lead-time stability sweep (weeks 4/6/8/10) | **DONE**, job **1189** (1 h 47 m, one node). **GenCast stable through week 6 (42 d) on every chain; FCN3 through week 10 (70 d) on every rollout, both arms.** 3 of 8 GenCast chains diverged, not monotonically in lead - and TWO of the three did so in the polar stratosphere (1211 K at 50 hPa) while CONUS stayed normal, invisible to every calibrated CONUS diagnostic. See "The lead-time stability sweep" below |
| 5-scoping - FCN3-only extension to 20 weeks (140 d) | **DONE**, job **1194** (2 h 16 m, one node, 2026-08-28). **FCN3 stable through week 14 (98 d) on every rollout of both arms.** At week 16 and beyond the PNW rollouts diverge (ERA5-IC arm at 75 d of the 112-d chain and 129 d of the 140-d one; adapter-fed arm at 105-111 d on all three) while every Uri rollout is clean to 140 d. No GenCast verdict at these leads, by design. See "The FCN3-only extension" below |
| 5-scoping - 100-member GenCast ensemble at week 8 (PNW) | **DONE**, job **1197** (4 nodes, 32 shards, 4 h 34 m, 113 GPU-h). **9 of 100 members diverged by 56 d (9%, Wilson 95% 5-16%); 100% clean through 36 d.** All nine loud, all starting on negative 850 hPa humidity at 39-54 d and growing into the stratospheric winds. Member 2 did NOT reproduce job 1189's 45 d blow-up on the same noise stream. See "The ensemble follow-up" below |
| 4c - multi-event extension: 3 heat events + controls | **IN FLIGHT** (2026-08-24). Registry/boxes/tests landed; productions running for Southwest_HeatWave_2020 (job **1176**) and California_HeatWave_2022 (job **1177**); Gate 3 for SCentral_HeatDome_2023 (job **1178**). See "The multi-event extension (wave log)" below |

**Phases 1-4 are complete.** All three gates pass, the RES core is validated, the
driver ran the pilot end to end on an 8xH100 node (job 1172) with its figures written
(see "What Phase 4 established"), and the pilot's probabilities are rebuilt as binned
PDFs against ERA5 (job 1174). What is left is Phase 5 (gated, exploratory) and the
un-run persistence-backend control. a3mega holds the calibration, the Gate 2 cubes, the 16
walker trajectories and the H100s. The full Gate 2 data tree is mirrored on Derecho as well and
re-scores there identically, so Derecho can do CPU analysis, but it is not required again
until Phase 5.

## The multi-event extension (wave log) - 2026-08-24

Open problem 6 ("the pilot has one event") is being closed with a **severity ladder**,
not more heat domes: one more genuinely extreme event in a different regime, one moderate
event, and (optionally) a mild control - because the deliverable is a weighted exceedance
curve, and hits alone cannot show the weights are honest. A moderate event tests the
mid-curve probabilities (the only place a handful of hindcasts can detect miscalibration)
and maps where RES stops paying for its FCN3 hours; a mild/null case is the false-alarm
control for a sampler that is built to push walkers hot. Full plan:
`/home/ubuntu/.claude/plans/come-up-with-a-snoopy-sunrise.md`.

The slate (all week-3, frozen C_k = (0, 1.0, 1.4, 1.8, 2.0) - deliberately NOT re-tuned
per event, so "one schedule generalises" stays testable):

| event | peak | box (measured, pinned by test) | role | Gate 3? |
|---|---|---|---|---|
| SCentral_HeatDome_2023 | 2023-06-27 | S Texas 26-32N 106-100W, **+5.73 K** (CONUS +0.52) | second extreme hit, new regime; dome core clipped at the Mexico border like PNW's at BC | **yes** (job 1178) - new region, no baseline data; its 16 walkers double as the DS baseline |
| Southwest_HeatWave_2020 | 2020-08-16 | SW/S Plains 29-35N 109-103W, **+5.14 K** (CONUS +1.54) | moderate rung / mid-curve calibration | no - xres cube IS the DS baseline; skill risk covered post-hoc (see below) |
| California_HeatWave_2022 | 2022-09-06 | California 35-41N 124-118W, **+4.71 K** (CONUS +1.60) | third rung; the UNCONSTRAINED max (+7.32 K) is Idaho/Montana and was deliberately not used - see the box note | no - same |

Gate 3 is skipped for the two staged events on purpose: its DS-baseline role is redundant
(24-member xres cubes exist), seeding saves only ~2.8 GPU-h, and the skill go/no-go is
insurance PNW passed 0.797/0.959 vs 0.5. The residual-risk checks are free: the first
resampling (3 d, C=0) is the identity, so production walkers are still i.i.d. at 6 d - a
post-hoc score-vs-realized rho_s at 3/6 d falls out of the run's own score cubes - and a
skill collapse shows up as per-leg ESS/N crashing without recovery.

What landed in code (all tests green, 202 passed):

- `fcn3/fevents.py`: `RES_EVENTS`/`RES_ORDER`, and `SEED_ORDER = ALL_ORDER + RES_ORDER` -
  the ten frozen cases keep their exact seeds (pinned literally in `test_adaptest.py`);
  extension seeds are 10333/11333/12333. `F.event(name)` resolves anything registered;
  `--res-spec`/`--res-events` CLI. ALL_ORDER is untouched, so the adapter-test Slurm
  array pin still passes unchanged.
- `aires/aindex.py`: the three boxes (selection rule now stated in the module comment) and
  `python -m aires.aindex --scan <event>` - the box-picker that measured every number
  above from the ERA5 truth with the experiment's own reduction.
- `aires/run_aires.py`: `run.json` now records the box (theta is a box mean - two boxes
  are two experiments; pre-existing PNW tags will diff once and need `--force`, do this
  for `persist` before launching the control); `--stage prune` (guarded: refuses without
  BOTH res_result.json and compare.json; `--dry-run` supported); the disk estimate now
  uses the measured 35 MB/diag, not 6.
- `slurm/xres_pool_0p25_week3_aires.slurm`: optional 24-member baseline-cube launcher for
  extension events; unlike the p90 launcher it does NOT overwrite the frozen
  `gencast_pool_timing.json` sidecar. Only needed if SCentral should get the same
  three-row DS figure as the pilot.

Wave state (cap: <= 4 GPU nodes for this experiment at any time; never > 2 concurrent
productions):

- [x] SCentral xres prep on the login node (ARCO path, ~25 min; truth grid max +7.53 K)
- [x] box scan -> S Texas box finalised and pinned BEFORE any reduce/prep froze it
- [x] job **1176** Southwest production: **done** (7 h 34 m), ds + compare reduced.
  **Two findings.** (1) CALIBRATION PASSES: through the range 24 direct members can
  resolve, the weighted RES exceedance tracks direct sampling within binomial noise
  (P(>=+2.23 K) 0.077 vs 0.083; P(>=+3.03) 0.026 vs 0.042), then extends ~2 decades
  deeper (P ~ 3e-4 at +4.31 K) - the weights deflate honestly (weighted mean +1.24 vs
  unweighted +2.47). (2) THE OBSERVED EVENT IS OUT OF REACH: ERA5 box A_L +5.14 K,
  direct 0/24 (max +3.82), AI+RES **0/64** (max +4.31). GenCast's week-3 distribution
  (mean +0.71, sd 1.33) puts the observation at +3.3 sigma - deeper relative to model
  spread than PNW was - and a C<=2, K=5 tilt moves the mean ~+1.75 K, not +4.4. Run
  health: log Z -0.8660, normalization check 1.0623, 16/64 founders; ESS/N crashed to
  0.11 at step 3 (C=1.4, max mult 23) then recovered 0.25/0.37 - the clone-correlation
  caveat (open problem: log-ps perturbation) is doing real damage to reach here.
  Full figure set written (apdfs/aplots/amaps). `aires/amaps.py` needed a fix first:
  it hard-required the Gate 3 tree, the FCN3 context cube and the HRRR truth, none of
  which an extension event necessarily has - all baselines are now optional (at least
  one required), the empty "0 that reached ERA5" panel is dropped, and the HRRR caption
  only appears when the HRRR panel does.
- [x] job **1177** California production: **done** (8 h 03 m), ds + compare reduced,
  states pruned (+91 GB). **A HIT at moderate severity**: 25/64 walkers reached the
  observed +4.71 K (max +6.33) -> weighted **P = 0.0186**, where direct sampling scored
  0/24 with its best member grazing at +4.70. Calibration clean where direct resolves
  (P(>=+2.99) 0.114 vs 0.125; P(>=+3.99) 0.045 vs 0.042), curve extends to 7.7e-5 at
  +6.33 K. Run health: log Z -1.2488, normalization check 0.7771 (the low side of the
  replay distribution - compare across runs per open problem 5), 16/64 founders; ESS/N
  0.22/0.18/0.25/0.47 (earliest collapse of the three, max mult 15/17).
  **The severity ladder now reads**: reach follows how far the observation sits outside
  the MODEL's forecast spread - PNW ~2.0 sigma: reached, P 0.054; CA ~2.9 sigma:
  reached, P 0.019; SW ~3.3 sigma: NOT reached, P < 3e-4 (calibrated mid-curve).
- [x] job **1178** SCentral Gate 3: **PASS** (2 h 28 m). rho_s **0.700** at 9 d
  [CI 0.26, 0.93] and **0.765** at 12 d [0.33, 0.94] vs the 0.50 threshold; 0.985 at
  15 d. CAVEAT worth carrying into the writeup: persistence is far stronger here at 12 d
  (+0.582) than it was for PNW (+0.129) - FCN3 still beats it, but the margin over "who
  is hottest now" is thin at that lead (persistence at 9 d is -0.006, so the 9 d
  checkpoint is where FCN3 earns its hours). `ctune --check`: mean |assumption error|
  0.153 vs ~0.28 noise at N=16 - the frozen C_k schedule is consistent with SCentral's
  measured skill curve.
- [x] SCentral production prep: frozen (tag `pilot`), 32/448 segments seeded from Gate 3
  (hardlinks), ~61.1 H100-h, READY: yes. **Submission held** until a wave-1 production
  finishes: with 333 GB free, a third concurrent production's 111 GB peak would land on
  the disk guard, and the plan caps concurrent productions at 2 regardless.
- [x] job **1180** PNW persistence control (Standard-RES): **done** (3 h 50 m), compared,
  states pruned. **THE ATTRIBUTION QUESTION IS CLOSED**: same walkers, same DMC, same
  C_k, same seeds, only the score swapped - persistence-RES reached the observed +7.72 K
  with **0/64** walkers (max +7.33), where FCN3-scored AI+RES reached with 42/64 (max
  +10.29, P 0.054). The persistence selection is noise at this lead: ESS/N 0.04/0.02 at
  steps 3-4 (max mult 39/49), 3/64 founders survive, weights span 0.013..869,
  normalization check **8.75**, weighted mean +0.29 (the weights cancel the tilt).
  Cloning on a skillful forecast of the event beats cloning on "who is hottest now" -
  the FCN3 score is the ingredient, exactly what Gate 3's persistence column predicted.
- [x] job **1179** SCentral production: **done** (7 h 13 m), ds + compare reduced,
  states pruned. **THE GENERALITY CLAIM LANDS**: 6/64 walkers reached the observed
  +5.73 K (max +6.19) -> weighted **P = 0.00273**, where the 16-walker Gate 3 direct
  baseline had nothing (max +4.19, a ~2.2 sigma target). The frozen C_k schedule,
  ported unchanged to a subtropical-ridge regime, reaches and prices the second extreme
  event. Weights honest: weighted mean +2.35 vs direct mean +2.04. Run health: log Z
  -0.7771, normalization check 1.2672, 17/64 founders, final ESS/N 0.65.
  Full figure set written (13 figures, all internal gates 0.00e+00). `aires/apdfs.py`
  needed the same optional-baseline treatment as amaps: with no xres cube it now
  PROMOTES the Gate 3 walkers to the direct-GenCast slot (they are the like-for-like
  ensemble by construction), records `gencast_xres_source="gate3"` in pdf_fields.nc,
  relabels the curve, and validates G13 against the ds key the ensemble was recorded
  under. Suite green after: 202 passed, 1 skipped.
- [ ] wave 3 (optional, user's call): p90_20240802 at N=32 tag `pilot32` (~29 GPU-h).
  With SW's curve-agreement calibration + the nulls' role in the adapter test, the
  amplitude-scaling question may already be answered well enough - the four-run set
  (PNW hit / CA hit / SCentral hit / SW honest miss + persistence control) is a
  complete story.

**Wave summary (the paper's table, 2026-08-25).** All at week-3 lead, N=64, frozen C_k:

| event | obs box A_L | direct sampling | AI+RES | weighted P(>= obs) |
|---|---|---|---|---|
| PNW_HeatDome_2021 | +7.72 K (~2.0 sigma of direct) | 0/40 (max +7.49) | **42/64** (max +10.29) | **0.054** |
| SCentral_HeatDome_2023 | +5.73 K (~2.2 sigma) | 0/16 (max +4.19) | **6/64** (max +6.19) | **0.0027** |
| California_HeatWave_2022 | +4.71 K (~2.9 sigma) | 0/24 (max +4.70) | **25/64** (max +6.33) | **0.019** |
| Southwest_HeatWave_2020 | +5.14 K (~3.3 sigma) | 0/24 (max +3.82) | 0/64 (max +4.31) | < 3e-4 (not reached) |
| PNW persistence control | +7.72 K | 0/16 | **0/64** (max +7.33) | - (score is the ingredient) |

Reach follows the observation's depth in the MODEL's forecast distribution, not the
absolute anomaly; the honest miss (SW) plus the mid-curve calibration agreement on all
three curves is what makes the three hits believable; the persistence control attributes
the entire gain to the FCN3 score. Total spent: ~232 H100-h across 5 runs + 1 Gate 3.

## Wave 2 - cold regimes and the low end of the ladder - 2026-08-25 (IN FLIGHT)

Wave 1 answered one question in one regime at the top of the severity scale: every
production was a warm, high-tail T2m heat event. Two things were untested as a result.
(1) **Regime generality** - `aires.aindex.tail_sign()` returns `-1` for a cold event and
`run_aires.leg_theta` applies it before `dmc` ever sees a score, but **no production run
had ever exercised that path**, so a sign bug was invisible. (2) **No-harm at the bottom
of the ladder** - every wave-1 row is a *reach* result with direct sampling at 0/24 or
1/24, so none of them can show the weights are honest at a target the model already
resolves. Plan: `/home/ubuntu/.claude/plans/can-you-run-some-staged-unicorn.md`.

**The measurement that set the slate.** The `p90` cases were selected on the PEAK-DAY
CONUS mean anomaly, but AI+RES scores the 7-DAY mean over `[peak-6d, peak]`. On the
observable that matters the cases are scrambled - the `null_*` controls are not the
near-zero cases and the `p90_*` events are not all extreme. Re-derived here from the ERA5
truth and each event's own cached 24-member week-3 GenCast cube, with the experiment's
own reduction (`aires.aindex`), and every number below reproduced the plan's:

| candidate | box | obs `A_L` | ens mean +- sd | sigma-depth | direct reach |
|---|---|---|---|---|---|
| `WinterStorm_Elliott_2022` | N Plains (new) | -14.42 K | -0.76 +- 4.87 | **+2.81** | 0/24 |
| `WinterStorm_Uri_2021` | TX/OK | -7.40 K | +0.06 +- 3.27 | **+2.28** | 0/24 |
| `p90_20251224` | CONUS | +3.26 K | +0.25 +- 1.58 | **+1.91** | **1/24** |
| `p90_20240802` | CONUS | +1.06 K | +1.06 +- 0.60 | **0.00** | 14/24 |
| `p90_20231107` | CONUS | +0.21 K | +1.28 +- 1.09 | -0.97 | 21/24 |
| *(wave-1 reference)* `PNW_HeatDome_2021` | PNW | +7.72 K | +2.46 +- 2.65 | +1.98 | 0/24 |

Elliott yields a **free second rung**: `aindex.indices()` records both the event box and
CONUS on every run and `stage_compare` stores the whole dict, so its CONUS index is a
second calibration anchor at **-2.73 K, +1.89 sigma, direct 1/24** - the cold-side match
to `p90_20251224`'s role on the warm side.

**Hurricanes were evaluated and rejected.** On its registered `u850_speed` metric
`HurricaneIan_2022` puts the observation at **-0.60 sigma**, BELOW the ensemble mean,
with 17/24 direct members already exceeding it (+0.19 on `u850_speed_max`); Idalia is
-0.44/+0.47, Ida the best at +1.43 but needs a new metric wired end to end. A 7-day-mean,
box-mean 850 hPa wind washes out a one-day landfall, so at week-3 lead a hurricane is not
a rare-event observable in this reduction. Not a tuning problem; out of scope.

The slate - 5 productions, week-3, N=64, M=6, frozen `C_k = (0, 1.0, 1.4, 1.8, 2.0)` at
3/6/9/12/15 d, horizon 21 d, tag `pilot`. `C_k` is again deliberately NOT re-tuned per
event, so "one schedule generalises" stays testable:

| # | event | sign | role |
|---|---|---|---|
| 1 | `WinterStorm_Uri_2021` | **-1** | first cold extreme; first end-to-end exercise of the negative-tail path |
| 2 | `WinterStorm_Elliott_2022` | **-1** | second cold rung, deeper (+2.81 sigma) - where the cold-side reach boundary sits |
| 3 | `p90_20251224` | +1 | +1.91 sigma, **direct 1/24** - the calibration anchor: the one rung where direct sampling returns a NON-ZERO probability to check the weighted curve against |
| 4 | `p90_20240802` | +1 | **0.00 sigma** - the observation sits exactly at the ensemble median. No-harm / false-alarm control |
| 5 | `p90_20231107` | +1 | -0.97 sigma, a literal ~0 K anomaly BELOW the median |

**What runs 4 and 5 actually deliver is NOT "walkers reaching the observation."** With the
target at or below the median that will be ~half the population and the number means
nothing; say so in the writeup. The deliverables are (a) **curve agreement** - weighted
`P(A_L >= a)` overlaying the direct curve across the range direct resolves, within
binomial noise; (b) **weight honesty** - the weighted mean returning to the direct mean
where the *unweighted* RES mean is well above it; (c) **the median check** - weighted
`P(>= obs)` landing near the direct empirical quantile, ~0.58 for `p90_20240802` (14/24)
and ~0.88 for `p90_20231107` (21/24). AI+RES reporting 0.05 for a median target would
mean the estimator is broken, and that is a result worth having.

What landed in code (suite green, **220 passed, 1 skipped**; `test_adaptest.py`'s seed
literals passed UNCHANGED, which is what proves the frozen index prefix survived):

- `fcn3/fevents.py`: `WinterStorm_Elliott_2022` APPENDED to `RES_EVENTS`/`RES_ORDER`
  (seed index 13 -> 13333). `family="cold"` is the functional bit - it makes `Event.cold`
  true and flips `aindex.tail_sign` to -1. `source="xres"` keeps it out of
  `xres_extra_events_spec`, so the frozen p90 GenCast spec is untouched. `ALL_ORDER` is
  unchanged, so all ten adapter-test seeds are unchanged. **Uri needed no registry work** -
  it has been in `EVENTS` at index 2 since the head-to-head experiment.
- `aires/aindex.py`: the `N Plains` box, 43-49 N / 106-100 W, **-14.42 K** vs a -2.73 K
  CONUS mean, pinned by `test_aindex.py`. Registered in `fevents` BEFORE the scan was run:
  `tail_sign()` returns `+1` for an unregistered event, so scanning first would have
  sorted warmest-first and reported the wrong window. Elliott is a named STORM, not a
  named region, so the "inside the region the event is named for" half of the selection
  rule does not bind and the window is chosen unconstrained. The scan's outright minimum
  is -14.64 K over 44-50 N, which is 0.22 K deeper but sits FLUSH against the 50 N CONUS
  crop edge - the outbreak continued into the Canadian Prairies, so that window reports a
  clipped mean; the registered box clears the edge by a full degree.
- `aires/apdfs.py`: `TAIL_THRESHOLD = 9.0` was a module constant and is now per-event.
  It was wrong-SIGNED for a cold event (a freeze has no grid points above +9 K, so every
  row would read 0.00000) and wrong-SCALE for a near-median one. Now
  `EVENT_TAIL_THRESHOLD` **pins 9.0 for the four wave-1 events** - all four published
  `P(point >= 9 K)` tables were drawn with the constant - and anything else derives the
  sign-aware `TAIL_PERCENTILE`th (95th) percentile of the OBSERVED ERA5 field over the
  event box: event-intrinsic, correctly scaled, and readable as "what fraction of the
  population's grid points got as extreme as the most extreme 5% of what actually
  happened". `tail_probability()` takes `sign`; the axis inverts for a cold event (the
  house convention `aplots` already used: further right is always more extreme); labels
  become `P(point <= a)`; units come from the metric. The value AND its provenance are
  written to `pdf_fields.nc` attrs (`tail_threshold`, `tail_threshold_source`,
  `tail_sign`) - nothing recorded them before, so a regenerated figure could not be told
  from an archived one. **Verified**: PNW regenerates bit-identical numbers (box weighted
  0.05053, ERA5 0.08480, direct 0.00107).
- `aires/apdfs.py` prose: the footnote asserted the heaviest weight belongs to "the
  population's coldest member". That inverts on a freeze - resampling tilts toward the
  cold tail there, so the least-selected walker is the WARMEST. It is now MEASURED (the
  rank of `argmax(weight)` in the tail direction) rather than restating the mechanism.
- `aires/amaps.py`: `truth_field()` hardcoded the stem `<event>_verif_t2m_anom.nc` and the
  variable `t2m_anom`, ignoring `ev.metric`. Correct for this slate (all five are T2m) but
  a latent trap - a wind event would have silently loaded the same event's T2m field.
  Now metric-driven, matching `fevents.era5_truth_path`. New `aires/tests/test_amaps.py`
  (there was none) pins it, plus `_sym`'s behaviour on a near-median field.
- Two sign bugs that would have misreported on the cold runs, both diagnostic-only:
  `aires/gate3.py`'s "walkers reaching the observed value" counted `realized >= observed`
  unsigned (on a freeze that is the walkers WARMER than the event, a number near N that
  reads as "no trouble getting there"); `aires/aplots.py::_marginal`'s "k/N" annotation
  had the same bug, where the exceedance panel 20 lines away already did it correctly.
  Gate 3's PASS/FAIL verdict is a Spearman correlation and is legitimately sign-symmetric.

Prep verification, all five, BEFORE any GPU time (the check this whole plan guards):
`READY: yes` and the tail line reads **LOW (cold event: clone the most negative A_L)** for
Uri and Elliott and **HIGH** for the three p90 cases. ~63.9 H100-h and ~8.0 h wall each.

Wave-2 state (cap: <= 4 GPU nodes; never > 2 concurrent productions; prune each run the
moment `compare.json` exists):

- [x] job **1181** Uri Gate 3: **PASS** (2 h 07 m). rho_s **+0.909** at 9 d
  [CI +0.71, +0.98] and **+0.968** at 12 d [+0.85, +1.00] against the 0.50 threshold -
  the STRONGEST of the three Gate 3 runs (PNW 0.797/0.959, SCentral 0.700/0.765), and it
  establishes FCN3 score skill on a COLD target, which no earlier gate did. Persistence
  is +0.432 at 9 d and +0.062 at 12 d, so the margin over "who is coldest now" is wide at
  both required leads - unlike SCentral, where persistence reached +0.582 at 12 d. (3 d
  is -0.035, but its noise ceiling is 0.000: at that lead the 6-member score sd exceeds
  the between-walker spread, so nothing is measurable there. C_1 = 0 anyway.) Secondary
  CONUS index agrees: +0.906 / +0.941.
  **THE COLD SIGN CHAIN IS NOW VERIFIED END TO END** - `tail_sign` -1 -> the box scan
  sorted coldest-first -> prep printed `LOW (cold event...)` -> the score ranks the cold
  tail, and the reduce's own reach line reads `walkers reaching the observed value
  (A_L <= -7.40): 0/16`. That line is also the proof the sign fix works: unsigned it
  would have read 16/16, because every free-running walker is warmer than the freeze.
  The 16 free-running walkers double as a second DS baseline (box mean +0.406 +- 3.227,
  range [-5.16, +6.43] vs the observed -7.40 - 0/16, consistent with the xres cube's
  +0.06 +- 3.27 and 0/24).
- [x] job **1182** `p90_20251224` production: **done** (8 h 42 m), ds + compare reduced,
  full figure set written, states pruned (91 GB freed; the run settles at **19 GB**,
  exactly the plan's estimate). Started CONCURRENTLY with the Gate 3 rather than after
  it: a warm high-tail case shares no part of the cold sign chain the gate was testing,
  and 2 GPU nodes / 1 production is inside both caps.
  **THE CALIBRATION ANCHOR HOLDS.** ERA5 observed +3.259 K; AI+RES reaches it with
  **46/64** walkers -> weighted **P = 0.0661**, against direct sampling's **1/24 =
  0.0417 [0.00739, 0.202]**. The AI+RES estimate sits INSIDE the direct estimate's 95%
  CI. This is the thing no wave-1 row could do: every one of them had direct at 0/24, so
  the weighted probability had nothing to be checked against. Here it does, and the two
  methods agree on a probability direct sampling can only just resolve. (FCN3 direct is
  0/24; its curve stops at +2.66.)
  Run health, the best of any production so far: log Z -1.1846, normalization check
  0.6890, 16/64 founders, ESS/N **1.00 / 0.32 / 0.56 / 0.54 / 0.69** - it RECOVERS after
  the first barrier instead of degrading, and max multiplicity peaks at 9 and falls to 3.
  Wave 1 by contrast bottomed out at 0.11 (SW, mult 23) and 0.18 (CA). The clone-
  correlation damage is much milder at a +1.91 sigma target than at a +3.3 sigma one.
  Three things worth carrying into the writeup:
  1. **Mid-curve agreement is good but not perfect.** Through the range direct resolves,
     AI+RES runs consistently ~2x the direct estimate (P(>=+1.28) 0.481 vs 0.250;
     P(>=+1.49) 0.331 vs 0.167), which is 2.2-2.6 binomial s.e. at n=24 - a mild high
     bias, not clearly significant on 24 members, but it leans the same way at every
     threshold. In the deep tail they agree (0.066 vs 0.042, well inside the CI).
  2. **The weighted mean does NOT return to the direct mean** (+1.907 vs +0.248, against
     an unweighted +3.829) - and it CANNOT, because the resampled population's support
     starts at **+1.25 K**. Every low-A_L lineage was killed, so no reweighting of what
     survives can reach a mean of +0.248. This is a SUPPORT limitation, not a weighting
     error, and it is why "the weighted mean returns to the direct mean" is only a
     meaningful test on the near-median runs (4 and 5), where the tilt is gentle enough
     that the population still covers the bulk.
  3. **The normalization check is visible directly on the exceedance figure**: the RES
     curve saturates at 0.689, not 1.0, for every threshold below the population minimum
     - because at those thresholds the estimator returns exactly Z*mean(w). That is open
     problem 5 made legible, and it means the curve's absolute level carries a ~31% low
     bias here. Self-normalizing would put P(>= obs) at 0.096, also inside the direct CI.
  The DERIVED tail threshold validated itself on first use: p95 of the observed ERA5
  field over the box gives **+9.043 K**, and ERA5's own binned curve returns **0.0534**
  at it - 5% by construction, recovered to binning precision. AI+RES weighted 0.0103,
  direct 0.0042, unweighted 0.0907. `pdf_box` was skipped as a deliberate duplicate:
  a p90 case's scored region IS CONUS, so the two panels would be one picture under two
  filenames.
- [ ] job **1183** Uri production (submitted 2026-08-25 23:44, after the gate passed).
  **THE NEGATIVE-TAIL PATH IS NOW EXERCISED END TO END IN A PRODUCTION RUN** - the gap
  this whole wave exists to close. Checked on leg 2's live `theta.json` (the run's FIRST
  real FCN3-scored resampling; leg 1 is `skipped(C=0)` and sets `used` to all zeros, so
  it cannot demonstrate anything): `used == -box` exactly, and the walker `dmc` ranks
  FIRST has box A_L **-6.09 K, which IS the population minimum** (the leg's spread runs
  -6.09 to +5.62 K). The sampler is cloning the coldest walker of a freeze, not the
  warmest. `run_aires.leg_theta` is where this happens - `rec["box"]` stays the raw
  PHYSICAL A_L (so a cold event's printed theta is correctly negative: -3.054 at leg 1,
  -0.234 +- 2.305 at leg 2) and `rec["used"] = sign * box` is the only thing dmc sees.
  Prep seeded **64 artifacts for 32 (walker, segment) pairs** from the Gate 3 tree as
  hardlinks - segment 1 is the same rollout in both runs and segment 2 is too, because
  C_1 = 0 makes the first resampling the identity - which drops the budget from 63.9 to
  **61.1 H100-h** (~7.6 h wall). Seeding is done in the manual login-node prep on
  purpose: `aires_res.slurm` passes `--no-seed` in-job so a GPU allocation can never
  silently become a builder.
- [x] job **1183** Uri production: **done** (7 h 12 m), ds + compare reduced, full figure
  set written, states pruned (91 GB).
  **THE FIRST COLD HIT, AND THE FIRST CLEAN DEMONSTRATION OF WEIGHT HONESTY.**
  ERA5 observed **-7.397 K**; AI+RES reaches it with **32/64** walkers (most extreme
  **-10.20 K**, i.e. it overshoots the observation by 2.8 K) -> weighted
  **P = 0.00895**. Direct sampling scores **0/16** free-running Gate 3 walkers, **0/24**
  xres GenCast and **0/24** FCN3 - three independent baselines, all empty, best member
  only -6.51 K. The frozen `C_k` schedule, ported unchanged from four warm heat events to
  a Texas freeze, reaches and prices it.
  **The tilt is FULLY UNDONE here**, which `p90_20251224` could not show: unweighted mean
  **-6.780** -> weighted **+0.210**, against a direct mean of +0.406 (walkers) / +0.064
  (xres). The population range is [-10.20, **+1.57**], so unlike the p90 anchor it still
  covers the bulk, and the weights walk the mean all the way back. That is the "weighted
  mean returns to the direct mean" criterion, satisfied on a real run.
  **Curve agreement is the best of the wave.** Through the whole range direct resolves,
  the weighted curve tracks BOTH direct ensembles within binomial noise, leaning slightly
  LOW if anything: P(<=-1.67) 0.247 vs 0.250/0.292; P(<=-2.95) 0.165 vs 0.188/0.208;
  P(<=-3.38) 0.095 vs 0.125/0.125; P(<=-5.09) 0.054 vs 0.063/0.083. Then it extends ~2
  decades past where every direct curve stops.
  Run health: log Z -0.6735, **normalization check 1.7688**, 16/64 founders, ESS/N
  1.00 / 0.40 / 0.43 / 0.41 / 0.30.
  **Two caveats that belong with the numbers.**
  1. The normalization check is **1.7688 here against 0.6890 for `p90_20251224`** - the
     two bracket 1 from opposite sides, and the direction of the mid-curve bias tracks
     it (the run with check < 1 read ~2x HIGH vs direct; this one, with check > 1, reads
     slightly LOW). No single rescaling fixes both, which is exactly why open problem 5
     says to read this statistic ACROSS runs and not to "correct" one curve by it.
  2. **The weight ESS is only 2.24 of 64** (weights span 0.0108 .. 144), against PNW's
     5.68 and the p90 anchor's 7.24. The A_L exceedance is a 64-walker weighted estimate
     and survives that, but the SPATIAL PDF is effectively two walkers wide and should be
     read as shape, not level. On the per-grid-point tail the box ERA5 truth has
     P(point <= -15.46 K) = 0.0630 where all three direct ensembles give exactly 0.0000
     and AI+RES gives 0.00003 weighted / 0.0030 unweighted - AI+RES is the only method
     that puts ANY mass there, but it is still far short of what ERA5 did.
  The cold figure layer works end to end: axis inverted so further right is more extreme,
  `P(A_L <= a)` / `P(point <= -15.46 K)` labels, threshold derived as the p5 of the
  observed field, and the measured footnote correctly names the heaviest-weighted walker
  as the one ranked **64/64 towards the extreme at +1.57 K** - the population's WARMEST.
  The old hardcoded caption said "the population's coldest member", which is precisely
  backwards on a freeze.
- [x] job **1184** Elliott production: **done** (7 h 26 m), ds + compare reduced, full
  figure set written, states pruned (91 GB). No Gate 3 and no seeding, so the full
  63.9 H100-h.
  **THE SECOND COLD HIT, AND THE DEEPEST REACH OF THE WHOLE EXPERIMENT.** ERA5 observed
  **-14.422 K** at **+2.81 sigma** - deeper in the model's own distribution than anything
  wave 1 reached, and 0.5 sigma deeper than Southwest, which AI+RES MISSED. AI+RES
  reaches it with **20/64** walkers (most extreme **-17.04 K**, overshooting by 2.6 K)
  -> weighted **P = 0.0079**. Direct sampling: **0/24**, best member only -9.83 K, i.e.
  4.6 K short.
  **The free second rung pays off.** `aindex.indices()` records both boxes on every run,
  so the CONUS index comes out at no extra GPU cost and is a SECOND CALIBRATION ANCHOR
  on the cold side: observed **-2.73 K** at +1.89 sigma, direct **1/24 = 0.0417
  [0.0074, 0.202]**, AI+RES **32/64 -> weighted P = 0.136** - inside the direct CI,
  matching `p90_20251224`'s role on the warm side.
  Run health: log Z -1.1612, **normalization check 0.8108** (the closest to 1 of the
  wave), 14/64 founders, ESS/N 1.00 / 0.39 / **0.29** / 0.33 / **0.59** - it takes its
  worst hit at step 3 and then RECOVERS, the same shape `p90_20251224` showed.
  Weight ESS 8.61, the healthiest of the wave.
  **The tilt is only PARTLY undone here**: unweighted -11.863 -> weighted **-3.638**,
  against a direct mean of -0.760. The population range is [-17.04, **-0.41**], so unlike
  the p90 anchor the support does still bracket the direct mean - the weights simply do
  not carry it all the way. Read across the three finished runs there is a clear pattern:
  **the deeper the tilt, the less of it the reweighting recovers** - Uri (+2.28 sigma)
  returns essentially exactly (+0.210 vs +0.406/+0.064), Elliott (+2.81 sigma) gets
  two thirds of the way back, and `p90_20251224` cannot return at all because resampling
  killed every walker below +1.25 K. That is a statement about the ESTIMATOR's small-
  sample behaviour, not about the physics, and it is the thing runs 4 and 5 were put on
  the slate to pin down at the gentle end.
  Only ONE direct baseline exists for this event (the 24-member xres cube - no Gate 3
  tree, and Elliott is not one of the six head-to-head events so there is no FCN3 cube).
  Both `apdfs` and `amaps` handled that through the optional-baseline path the wave-1
  extension added, with no code change needed.
- [ ] Elliott and the two remaining p90 cases skip Gate 3, on the wave-1 argument: their
  cached 24-member xres cubes ARE the DS baseline, and once Uri passes, Elliott inherits
  the cold-T2m skill argument. Residual-risk checks are free - the first resampling
  (3 d, C=0) is the identity, so walkers are still i.i.d. at 6 d and a post-hoc
  score-vs-realized `rho_s` falls out of the run's own score cubes; a skill collapse shows
  as per-leg ESS/N crashing without recovery.
- [x] job **1185** `p90_20240802` production: **done** (7 h 48 m), reduced, figures
  written, states pruned (91.6 GB). Run 4, the no-harm / false-alarm control at
  **0.00 sigma** - the observation sits exactly at the ensemble median.
  **THE MEDIAN CHECK PASSES, WHICH IS THE WHOLE POINT OF THIS RUN.** ERA5 observed
  **+1.057 K**; AI+RES weighted **P = 0.649** against direct sampling's **14/24 = 0.583
  [0.388, 0.755]** (FCN3 independently also 14/24). The plan predicted ~0.58 and the
  estimator returned 0.649 - inside the direct CI, and nowhere near the failure mode it
  was built to detect. **If AI+RES had reported 0.05 for a median target the estimator
  would be broken; it reports 0.65.**
  Do NOT quote the reach count as a headline: 63/64 walkers "reach" a median target,
  which is a statement about where the tilt pushed the population and carries no
  information. The deliverables are the three below.
  **Spatial PDF - the cleanest no-harm evidence in the wave.** On the per-grid-point tail
  at the derived threshold (+3.952 K, p95 of the observed field), AI+RES weighted gives
  **0.0727** against direct GenCast's **0.0702** - agreement to 4% - while the UNWEIGHTED
  RES population gives 0.1627, i.e. 2.3x too heavy. The weights do their job on the
  spatial distribution, not just on the scalar. (ERA5 truth 0.0468, FCN3 0.0543.)
  **Curve agreement** is exact at the observation (0.649 vs 0.583/0.583) and again leans
  HIGH in the mid-range (0.630 vs 0.375-0.417 around +1.3 to +1.4 K) before converging by
  +1.5 K. The weighted curve is visibly COARSE there - it steps 0.630 -> 0.255 between
  +1.433 and +1.512 - because at ESS ~10 a handful of walkers carry the whole estimate.
  **Weight honesty, partial**: unweighted +1.932 -> weighted +1.508 against a direct mean
  of +1.058. The unweighted mean IS well above direct as the plan predicted, and the
  weights recover most of the gap, but a +0.45 K residual (0.74 direct-sd) remains even
  at 0.00 sigma. Read with the other three runs, the residual in ensemble-sd units is
  Uri **0.045**, Elliott 0.59, this run 0.74, `p90_20251224` 1.05 - so it does NOT track
  sigma-depth cleanly, and Uri is the outlier that returns almost exactly. The honest
  statement is that the weighted mean retains a residual bias TOWARDS the tail of roughly
  0.6-1.0 ensemble sd in three of four runs.
  Run health: log Z -0.6063, normalization check 0.7039, 15/64 founders, ESS/N
  1.00 / 0.53 / 0.37 / 0.30 / 0.40.
  **A diagnostic lead worth chasing (n = 4, so a lead and not a finding):** the
  normalization check and the DIRECTION of the mid-curve discrepancy are anti-correlated
  across every run of this wave. Check < 1 -> the weighted curve reads HIGH vs direct
  (`p90_20251224` 0.689, Elliott 0.811, this run 0.704); check > 1 -> it reads LOW (Uri
  1.769). If that holds up it is a usable per-run correction indicator, which open
  problem 5 currently lacks.
- [x] job **1186** `p90_20231107` production: **done** (7 h 25 m), reduced, figures
  written, states pruned (91 GB). Run 5, the last of the slate: -0.97 sigma, a literal
  ~0 K anomaly BELOW the ensemble median.
  **THE ESTIMATOR DEGENERATES AT A BELOW-MEDIAN TARGET. This is the most useful thing the
  bottom of the ladder produced, and it is a NEGATIVE result.** ERA5 observed +0.211 K;
  direct sampling puts it at **21/24 = 0.875 [0.676, 0.973]**; AI+RES reports
  **P = 0.5827**, BELOW the direct CI. The plan predicted ~0.88; the run missed it.
  The cause is structural and was verified numerically, not inferred. Resampling tilted
  the whole population warm, so its MINIMUM realized A_L is **+1.498 K**, well above the
  +0.211 K target: the indicator `1{A_L >= obs}` is identically 1 over all 64 walkers.
  The DMC estimate is `Z * mean(w * 1{...})`, so with the indicator identically 1 it
  collapses to `Z * mean(w)` - **exactly the normalization check: 0.582662 vs 0.582662,
  to every digit printed**. The estimator cannot report a probability above its own
  normalization check, and here the truth (0.875) is above that ceiling. Self-normalizing
  is no better: it returns exactly **1.000**, because "every walker reached it" is all the
  information the population retains. **Neither number is the answer, and nothing inside
  the run reveals that** - the failure is silent, and a reader of `compare.json` alone
  would take 0.5827 at face value.
  The rule this establishes: **AI+RES has no resolution at a target the tilt has already
  carried the entire population past.** The usable operating range is targets the
  resampled population still STRADDLES. `p90_20240802` at 0.00 sigma straddles it
  (population min +0.39 vs target +1.06) and answers correctly; this run at -0.97 sigma
  does not and returns a pinned value. That boundary is now measured rather than assumed,
  and it is the honest lower limit of the method.
  Run health is otherwise the BEST of the wave - log Z -1.0156, 15/64 founders, ESS/N
  1.00 / 0.41 / **0.62** / 0.53 / 0.53, min 0.41 - which is exactly the point: a healthy
  sampler can still return a meaningless probability if the target is on the wrong side
  of the tilt. **Health metrics do not detect this failure.**
  The spatial PDF shows the over-tilt from the other side: at the derived +3.697 K
  threshold ERA5 gives 0.0496 and direct GenCast 0.1629, while AI+RES weighted gives
  **0.1976** and unweighted 0.4131 - weighted is close to direct, but BOTH models are
  ~3x heavier than the observed field in that tail, which is a forecast-bias statement
  about the event rather than an AI+RES one.
- [x] all five productions reduced, drawn and pruned; each run settles at ~19 GB.

**Wave-2 summary (the paper's second table, 2026-08-26).** All week-3, N=64, M=6, frozen
`C_k = (0, 1.0, 1.4, 1.8, 2.0)`, tag `pilot`. `direct` is the 24-member 0.25 deg xres
GenCast cube; sigma-depth is `sign * (obs - ens mean) / ens sd` on that cube.

| event | obs box A_L | sigma-depth | direct sampling | AI+RES | weighted P | direct P [95% CI] | norm check | min ESS/N |
|---|---|---|---|---|---|---|---|---|
| Winter Storm Elliott 2022 | -14.42 K | **+2.81** | 0/24 (max -9.83) | **20/64** (max -17.04) | **0.0079** | 0.000 [0.000, 0.142] | 0.8108 | 0.29 |
| Winter Storm Uri 2021 | -7.40 K | **+2.28** | 0/24 (max -6.51) | **32/64** (max -10.20) | **0.0090** | 0.000 [0.000, 0.142] | 1.7688 | 0.30 |
| p90 2025-12-24 | +3.26 K | **+1.91** | 1/24 (max +3.46) | **46/64** (max +5.12) | **0.0661** | 0.042 [0.001, 0.211] | 0.6890 | 0.32 |
| p90 2024-08-02 | +1.06 K | **-0.00** | 14/24 (max +1.96) | **63/64** (max +2.37) | **0.649** | 0.583 [0.366, 0.779] | 0.7039 | 0.30 |
| p90 2023-11-07 | +0.21 K | **-0.97** | 21/24 (max +3.14) | 64/64 (max +4.16) | 0.583 **(pinned)** | 0.875 [0.676, 0.973] | 0.5827 | 0.41 |

What wave 2 establishes, in the order the plan asked the questions:

1. **Regime generality: YES.** The negative-tail path is exercised end to end for the
   first time and it works. Two cold extremes, both reached and priced, with the frozen
   `C_k` ported unchanged from four warm heat domes: Uri +2.28 sigma (32/64, P 0.0090)
   and Elliott +2.81 sigma (20/64, P 0.0079), against 0/24 direct in both.
   **Elliott at +2.81 sigma is the deepest reach of the entire experiment** - deeper than
   anything wave 1 managed - and Southwest at +3.3 sigma remains the only miss, so the
   reach boundary sits between +2.8 and +3.3 sigma and is NOT sign-dependent.
2. **Calibration against a non-zero direct probability: YES, twice.** `p90_20251224`
   (+1.91 sigma): AI+RES 0.0661 vs direct 0.042 [0.001, 0.211] - inside. Elliott's FREE
   CONUS rung (+1.89 sigma, from `aindex.indices()` at no extra GPU cost): AI+RES 0.136
   vs direct 0.042 [0.007, 0.202] - inside. Both lean high; both agree within the (wide)
   uncertainty of a 1-in-24 estimate.
3. **No-harm at the median: YES at 0.00 sigma, NO below it.** `p90_20240802` returns
   0.649 vs direct 0.583 [0.366, 0.779] - inside - and its spatial PDF matches direct to
   4% (0.0727 vs 0.0702) where the unweighted population is 2.3x too heavy. But
   `p90_20231107` at -0.97 sigma returns a PINNED 0.583 against a true 0.875. **The
   operating range is targets the resampled population still straddles.**
4. **Sampler health is regime-independent but does NOT certify the answer.** Minimum
   ESS/N is 0.29-0.41 across all five runs, spanning -0.97 to +2.81 sigma and both signs,
   against wave 1's 0.11-0.18 on its deeper targets - the frozen schedule self-regulates
   over this range. Yet the run with the HEALTHIEST sampler (`p90_20231107`, min ESS/N
   0.41) is the one that returned a meaningless probability.
5. **Weight honesty is partial and does not track sigma-depth.** Residual of the weighted
   mean from the direct mean, in ensemble-sd units: Uri **0.045**, Elliott 0.59,
   `p90_20240802` 0.74, `p90_20251224` 1.05. Uri returns almost exactly; the rest keep a
   bias TOWARDS the tail of ~0.6-1.0 sd. In `p90_20251224` it is a SUPPORT limitation and
   not a weighting error at all - resampling killed every walker below +1.25 K, so no
   reweighting of the survivors could reach the direct mean of +0.248.
6. **A diagnostic lead for open problem 5 (n = 5, so a lead, not a finding).** The
   normalization check and the DIRECTION of the mid-curve discrepancy are anti-correlated
   in every run of this wave: check < 1 -> the weighted curve reads HIGH vs direct
   (0.689, 0.811, 0.704, 0.583); check > 1 -> it reads LOW (Uri, 1.769). If that holds up
   it is a usable per-run correction indicator, which open problem 5 currently lacks.

Cost: **~318 H100-h** - 5 x ~62 productions plus ~18 for Uri's Gate 3 - over ~24 h
elapsed at 2 concurrent nodes (jobs 1181-1186). Uri's Gate 3 seeding saved ~2.8 H100-h.
- [ ] `python -m aires.alift` has NOT been run over the wave-2 slate. It is another
  session's in-flight work (`aires/alift.py` + `aires/aclim.py`, written 2026-08-26
  21:15) and it redraws the whole slate from every `compare.json` it finds, so it will
  pick up all five new runs whenever its author runs it.
- [ ] wave 3 (optional): p90_20240802 at N=32 tag `pilot32` as the mild control
- [ ] after each finished+compared run: `--stage ds`, `--stage compare`, apdfs/aplots/amaps,
  then `PYTHONPATH=. python -m aires.alift` and `PYTHONPATH=. python -m aires.awalkers`
  (both cross-event, both redraw the whole slate from every `compare.json` they find -
  no per-run argument), then `--stage prune`
- [ ] **register a new event's box BEFORE its first `alift`, and rebuild the climatology
  cache with it.** `aires/aclim.py` reduces whatever `aindex.EVENT_BOXES` held when the
  cache was WRITTEN, so an event added later has a box but no climatology column. That
  happened to `WinterStorm_Elliott_2022`: its N Plains `A_L` (reaching -17 K) was scored
  against the CONUS column (spanning +-5 K), reporting a spurious `lift >= 285x` where the
  honest number is 0.35x. Rebuild is ~10 min on the login node, internet, and reproduces
  every pre-existing box bit-for-bit:
  `PYTHONPATH=. python -m aires.aclim --build`. Now pinned by
  `test_aclim.py::test_the_cache_on_disk_is_not_stale_relative_to_the_registered_boxes`
  and refused at runtime by `alift.clim_pool`.
- [x] pruned via the guarded `--stage prune`: PNW pilot states (192 files, 91 GB) and
  Southwest pilot states (192 files, 91 GB) - both only after res_result.json AND
  compare.json existed. 339 GB free after.
- [ ] **two deletions still pending the user** (raw `rm`/`find -delete` is blocked by
  the permission classifier; only the guarded python prune passes): PNW Gate 3
  step03-07 states, 80 files / 38 GB (`find runs/aires/PNW_HeatDome_2021/walkers -path
  '*step0[3-7]/state.nc' -delete` - KEEPS step01/02, which the persistence control
  seeds from), and `rm -rf runs/aires/PNW_HeatDome_2021/res/smoke` (1.3 GB). Not on the
  critical path.

## The population figure - every walker, its anomaly, its `p_i` (2026-08-26)

`aires/awalkers.py` (new, + `aires/tests/test_awalkers.py`, 16 tests). Draws the raw
material every other figure reduces: all 64 surviving walkers of each run, where each
landed on `A_L`, and the probability

    p_i = Z * w_i / N ,   so   P(A_L >= a) = sum of p_i over the walkers past a
                          and  sum_i p_i   = the run's own normalization check

`walker_mass` is IMPORTED from `aires/alift.py` rather than re-derived, so the two
figures cannot drift apart; `test_awalkers.py` pins the identity against
`dmc.DMCResult.expectation` and that the import is the same object. Verified against the
published tables: every headline reproduces (PNW 0.054, CA 0.019, SCentral 0.00273,
Elliott 0.00790, Uri 0.00895, `p90_20251224` 0.0661, `p90_20240802` 0.649,
`p90_20231107` 0.583 PINNED), as do the weight ESS values (PNW 5.68, Uri 2.24).

    PYTHONPATH=. python -m aires.awalkers               # slate figure + the CSV
    PYTHONPATH=. python -m aires.awalkers --per-event   # + one detail figure per run

Writes `figures/aires/aires_walkers.png` (10 panels, hardest target first),
`figures/aires/<event>/aires_walkers_<event>_<tag>.png` per run, and
`runs/aires/aires_walkers.csv` - 640 rows, one per walker: `A_L`, `A_L_conus`,
`weight`, `p_i`, `p_over_direct`, `rank`, `cum_p`, `founder`, `family`, `reached`.

Three things the panels make legible that the tables cannot:

- **The `1/N` line is the price of reach.** A direct-sampling member is worth exactly
  `1/N`; PNW's hottest walker carries `2.5%` of that (`p_over_direct` 0.025). Reach is
  bought and then charged for, and the panel shows the exchange rate per walker.
- **The masses come in tied blocks.** Walkers cloned at the LAST resampling inherit their
  parent's `V_K` and carry identical mass, so Uri's 64 walkers hold only 34 distinct
  values (largest family 9) and the persistence control only 15. That is the number of
  independent pieces of information behind the estimate, and it is why the weight ESS
  (2.24 for Uri, 1.01 for `persist`) is the right thing to quote next to a `P`.
- **The pinned failure is visible.** For `p90_20231107` the red line sits below every
  walker, so the shaded "counts towards P" band covers the whole panel - the estimator
  is summing everything and returning its own normalization check. `compare.json` alone
  shows a plausible 0.583.

Southwest and the `persist` control have nothing in the band; their panels report `P = 0`
plus the deepest level the run resolved (3.2e-04 and 2.6e-04), which is the honest
statement and matches the wave-1 table's "< 3e-4".

## The population as a density - the KDE of `A_L` (2026-09-02)

`aires/akde.py` (new, + `aires/tests/test_akde.py`, 12 tests). The same 64 walkers per
run as `aires_walkers.png`, drawn as a Gaussian kernel density over `A_L` instead of as
a scatter: three curves per panel, log y, one panel per run in the same cell order
(`awalkers.order_runs`, extracted from `plot_walkers` so the two slates line up).

    PYTHONPATH=. python -m aires.akde               # figures/aires/aires_kde.png
    PYTHONPATH=. python -m aires.akde --per-event   # + figures/aires/<event>/aires_kde_<event>_<tag>.png
    PYTHONPATH=. python -m aires.akde --z-scaled --bw 0.5   # sum p_i scaling; one width for all

- **weighted by `p_i`** (solid blue) - the estimator's own PDF; `p_i` comes through
  `awalkers.run_record`, i.e. `alift.walker_mass`, nothing re-derived. Self-normalised by
  default; `--z-scaled` multiplies it by `sum p_i` (8.75 for the persistence control).
- **unweighted** (dotted blue) - where the sampler put its walkers. Its mass past the
  observation is 0.4-0.7 on the reached events and is NOT a probability; the legend says so.
- **direct ensemble** (dashed brown) - the 24-member xres cube at `1/n` each, the
  like-for-like baseline. It and the weighted curve estimate the same law.
- rug ticks at the foot of each panel: the discrete positions the curves were smoothed from.

Two decisions a KDE forces, both pinned by tests:

- **One bandwidth for both walker curves**: Scott's rule on the 64 unweighted positions
  (0.13 K for `p90_20240802` up to 1.75 K for Elliott). scipy's weighted default sizes the
  kernel from the Kish ESS of the weights and would give 2.0 K for Uri and 3.7 K for the
  persistence control - a blob wider than half the population. The unweighted estimator
  is pinned to `scipy.stats.gaussian_kde` to 1e-10.
- **Every curve ends at its own most extreme member.** A Gaussian kernel is never zero:
  unclipped, the direct ensemble's KDE puts 0.035 (persist) and 0.0013 (Southwest) past
  the observation with 0/24 members there, which would draw the reach the experiment
  denies. Curves are NaN outside `[min, max]` of their own sample.

The panel title prints the EXACT subset-sum `P` (as `aires_walkers.png` does), never a KDE
integral. `main` echoes the unclipped KDE tail beside it so the smoothing is on record -
it overstates `P / sum p_i` by 1.15x (CA, PNW), 1.3x (Uri), 1.4x (Elliott) and 1.75x
(SCentral), and `tail_mass` (an exact normal-CDF sum, no grid) collapses onto the subset
sum as `--bw` -> 0. Not yet registered in `scripts/embed_slate_figs.py` /
`docs/aires_event_slate.html`; add an `ALT_TO_FIG` entry and a slot if it should go into
the slate PDF.

## The lead-time stability sweep - weeks 4/6/8/10 (the `astab/` package) - 2026-08-27

*(Everything below is job 1189, the eight-chain run. The scorer was extended to week
20 the same day - see "The FCN3-only extension to 20 weeks" further down; that
section supersedes this one wherever the two disagree about chain counts, the week
dicts, or `ref_times`.)*

Every AI+RES run so far - the pilot, wave 1, wave 2, all three gates - sits at **week-3
lead (21 d)**. Before committing GPU hours to a longer-lead experiment, this asks the one
question that has to come first:

> At what rollout length does the GenCast walker or the FCN3 score function stop producing
> a physical atmosphere?

**Numerical stability only.** It does not, and cannot, say whether the score still *ranks*
usefully out there - that needs a rank correlation, which needs an ensemble, which one
member cannot produce. See "Stable is not useful" below; sizing that follow-up is the whole
point of running this first. Nothing about the frozen week-3 experiment changes: new tree,
new tag (`stab`), new module.

### What was actually in the way

**Neither model caps rollout length in code.** GenCast's
`rollout.chunked_prediction_generator*` just divides `num_target_steps` by a chunk size;
FCN3's `_default_generator` is a bare `while True`. The only cap in the entire stack was a
three-entry dict in two mirrored copies - `gencast_s2s/config.py:WEEKS_TO_LEAD_DAYS` and
`fcn3/fevents.py:WEEKS_TO_LEAD_DAYS` - which this run widened to
`{2: 14, 3: 21, 4: 28, 6: 42, 8: 56, 10: 70}` and the FCN3 extension widened again, to 20
weeks (140 d). Purely additive both times; 2/3/4 keep the values they have always had, and
`test_astab.py::test_the_frozen_weeks_are_untouched` pins that.

**One blast radius to know about:** `gencast_s2s/download.py:140` falls back to
`list(C.VALID_WEEKS)` when `--weeks` is absent, so a bare `python -m gencast_s2s.download`
now preps **eleven** weeks of init frames for every event in `ALL_EVENTS` instead of three.
Nothing in this sweep invokes it.

### The schedule, and the two off-by-ones in it

Each chain is initialised at `peak - lead` and rolled **to the peak** in 6-step (3-day)
segments, so a worker pays one XLA compile. Ending at the peak rather than free-running a
fixed span is what makes this the real AI+RES configuration at each lead: `A_L` is a 7-day
mean *ending* at the peak and is undefined for a chain that stops short.

| lead | segments | tail | scoring checkpoints |
|---|---|---|---|
| 28 d (wk 4) | 9x6 + **2** = 10 | 2 steps | 3 .. 27 d (9) |
| 42 d (wk 6) | 14x6 = 14 | **0** | 3 .. 39 d (13) |
| 56 d (wk 8) | 18x6 + **4** = 19 | 4 steps | 3 .. 54 d (18) |
| 70 d (wk 10) | 23x6 + **2** = 24 | 2 steps | 3 .. 69 d (23) |

- **The tail table is `{4: 2, 6: 0, 8: 4, 10: 2}` steps.** Only 42 d divides evenly. 70 is
  the trap: a multiple of 7 *and* of 14, so it reads as though it should divide by 3, and
  it does not. `chain_schedule()` is pure arithmetic on `horizon % 3` with **no per-chain
  special-casing**, and `stage_walk` hard-asserts that the final state's `valid_time` IS
  the event peak - nothing else downstream checks that, so a chain landing a day short
  would go unnoticed until `A_L` came out of a window that does not exist.
- **The horizon is never a scoring checkpoint.** The 42-day chain's last full segment lands
  exactly on the peak, so scoring there would ask FCN3 for a zero-step rollout, which
  `score.nsteps_for` rejects (`n <= 0`). Excluded for every chain, not just the one where
  it bites.

The ragged tail is placed **last** so every intermediate boundary stays on the 3-day grid
`aconfig.segment_for_lead` is defined on.

### One tag, four walker slots

`WEEKS_TO_WALKER = {4: 0, 6: 1, 8: 2, 10: 3}` - **not four tags**, because `score.tasks()`
enumerates `for w in range(walkers)` and cannot discover non-contiguous indices. One tag
means the coupled scoring arm is a single command per event, and no new path helper was
needed: everything goes through `A.res_state_path(event, w, step, "stab")` and
`A.res_score_cube_path(...)`.

### The two scorer arms

- **adapter-free** - FCN3 from the ERA5 IC at each chain's own init, rolled to the peak
  (112/168/224/280 6-h steps). Isolates the scorer's stability from the walker's drift.
  `fcn3/run_fcn3.py --stage infer` unmodified, one process per lead (`FCN3_WEEKS` is read
  at MODULE IMPORT).
- **adapter-fed** - FCN3 from each chain's **lead-3 d** state, through `aires/adapter.py`.
  Lead 3 is `segment_for_lead(3.0) == 1` for every chain regardless of horizon, so it is
  one command for all four *and* the longest coupled rollout available at each lead.
  `score.run_task`, the same unit Gate 3 and the DMC loop use.

### The diagnostic panel, and the bands it is judged against

Anchored on the one **measured** failure on record (the wrong-checkpoint incident, jobs
1155/1156): CONUS-mean T2m lost its diurnal cycle inside the first 3-day segment (00Z equal
to 12Z to 0.03 K), drifted 26 K cold over 21 days, and z500 finished 1780 m2 s-2 low.

| # | diagnostic | domain | detects |
|---|---|---|---|
| 1 | non-finite fraction, per variable | global + CONUS | terminal blow-up. **Hard fail, no tuning** |
| 2 | physical bounds (T2m 180-340 K, MSLP 870-1090 hPa, q >= -5e-3, wind < 200 m/s, finite geopotential; every one measured, see below) | global + CONUS | gross divergence before it reaches NaN |
| 3 | CONUS T2m anomaly vs climatology, and drift from the chain's own start | CONUS, 12 h | the -26 K signature |
| 4 | global-mean T2m and z500 vs the ERA5 reference series | global, 3 d | a runaway outside CONUS that item 3 cannot see. **Reported, not gated** - no in-repo baseline exists |
| 5 | diurnal amplitude / the climatology's own 12Z-00Z difference | CONUS | the 0.03 K collapse. The best-grounded item: a ratio against a real seasonal reference, so it self-calibrates across April-June and December-February |
| 6 | spatial sd of CONUS T2m | CONUS, 12 h | homogenisation or grid-scale checkerboarding |
| 7 | zonal z500 power above wavenumber 20, 30-60 N in 5 deg bands | **global only** | high-wavenumber pile-up, the classic AI-weather instability signature |

Item 7 **must** read `state.nc`, not `diag.nc`: the CONUS crop spans 59 degrees of
longitude, far too little for a zonal wavenumber decomposition.
`zonal_high_wavenumber_fraction` raises rather than returning the plausible-looking number
a partial-domain FFT would give.

**The bands are measured, not guessed** (`--stage calibrate`, CPU, free, ~12 min). The Gate
3 tree already holds 16 free-running, physically-validated 21-day walkers per event, so the
whole panel was run over **224 known-good segments** before any GPU was booked:

| metric | PNW p05 / p50 / p95 | PNW band | Uri p05 / p50 / p95 | Uri band |
|---|---|---|---|---|
| `t2m_anom_conus` (K) | 0.077 / 1.892 / 2.956 | [-2.67, 4.68] | -2.584 / 0.624 / 2.963 | [-6.46, 6.69] |
| `t2m_conus_drift` (K) | -0.470 / 0.636 / 1.711 | [-3.31, 3.42] | -1.032 / 1.703 / 4.569 | [-5.06, 8.84] |
| `diurnal_ratio` | 0.935 / 1.061 / 1.216 | [0.704, 1.401] | 0.773 / 0.964 / 1.280 | [0.558, 1.518] |
| `conus_t2m_sd` (K) | 5.687 / 6.409 / 7.095 | [4.20, 7.99] | 8.918 / 10.719 / 13.141 | [6.53, 15.99] |
| `z500_hi_wavenumber_frac` | 0.002 / 0.003 / 0.005 | [0.001, 0.007] | 0.001 / 0.002 / 0.003 | [0.000, 0.005] |

Two things that fall straight out of these numbers:

- **The bands actually separate the known failure.** A healthy walker reproduces the
  climatological diurnal amplitude to within about +-30% (ratio 0.70-1.40). The job-1155
  collapse sits at a ratio of ~0.004 - roughly 200x outside. Its -26 K drift is ~8x outside
  the widest drift band measured here.
- **ERA5 independently corroborates item 7.** The ERA5 reference series, built with the
  *same* reduction, gives a high-wavenumber fraction of **0.0019-0.0056** for PNW's dates
  and **0.0011-0.0046** for Uri's - sitting on top of the walkers' own measured bands. The
  walkers' spectra are ERA5's spectra. That is a real reference for item 7, not an
  extrapolation.

**Measured to 21 d only.** Beyond that every band is an EXTRAPOLATION, and the CSV carries
an `extrapolated` column and the figure says so in the panel annotation and the suptitle.

### The physics gate, in two forms

- **week-4 chains**: `gate3.check_against_reference`'s comparison as-is, pointed at the
  tag's tree. A 24-member 0.25 deg week-4 cube exists for **both** events - same init, same
  checkpoint - and that is the strongest anchor available anywhere in the repo.
- **week-6/8/10 chains**: no same-init ensemble exists, and reusing the week-3 cube would
  **false-fail** (a walker 21+ days diverged legitimately sits many sd from a fresh week-3
  ensemble mean). Gated against **ERA5 at the day-3 valid time** instead: at 3 d lead
  GenCast is genuinely skillful, so CONUS-mean T2m within 6 K of ERA5 is a sound statement,
  and the 26 K wrong-checkpoint drift fails it instantly.

All eight chains draw their bundle from the single `walker.load_bundle` call site, so one
passing week-4 check already covers the wrong-checkpoint class for the whole job; the ERA5
gate is the per-chain backstop.

**All eight fired within ~15 min of job 1189 starting, and all eight passed:**

| chain | reference | walker | reference value | offset |
|---|---|---|---|---|
| PNW @ wk4, 3 d | same-init xres, 24 members | 297.42 K | 297.24 +- 0.123 K | **1.5 sd** |
| PNW @ wk4, 9 d | same-init xres, 24 members | 296.88 K | 297.60 +- 0.428 K | **1.7 sd** |
| Uri @ wk4, 3 d | same-init xres, 24 members | 281.31 K | 281.18 +- 0.174 K | **0.8 sd** |
| PNW @ wk6, 3 d | ERA5 | 294.68 K | 294.36 K | **+0.32 K** |
| PNW @ wk8, 3 d | ERA5 | 292.14 K | 292.36 K | **-0.22 K** |
| PNW @ wk10, 3 d | ERA5 | 287.64 K | 287.36 K | **+0.29 K** |
| Uri @ wk6, 3 d | ERA5 | 282.16 K | 281.75 K | **+0.41 K** |
| Uri @ wk8, 3 d | ERA5 | 281.15 K | 281.34 K | **-0.18 K** |
| Uri @ wk10, 3 d | ERA5 | 285.01 K | 284.68 K | **+0.33 K** |

Every long-lead chain lands within 0.41 K of ERA5 at 3 d, against a 6 K gate, and the two
week-4 chains sit inside their own same-init ensemble's spread. That is a much stronger
result than the gate needed, and it doubles as an independent check on the ERA5 reference
series itself: the series was built with a hand-rolled zarr reader (see below), and it
agrees with an independent GenCast rollout to a few tenths of a Kelvin.

### Two thresholds that were wrong until they were measured

Both were specified in the plan, both looked unarguable, and both fired on a **perfectly
healthy** walker. They are recorded because the pattern is the point - this is
CLAUDE.md's "measure the noise floor before setting a threshold" applied to bounds nobody
thought needed measuring.

- **`q >= 0` is not a bound.** Measured over 18 known-good Gate 3 global states (both
  events, walkers 0/5/15, leads 3/12/21 d), healthy GenCast reaches **-1.28e-3 kg/kg**
  with up to 0.19% of points negative, from the FIRST segment, not growing with lead - the
  ordinary small-negative-humidity artifact of an ML weather model. An absolute zero floor
  is an alarm that is always on. Now `-5e-3`, about 4x the worst measured value.
- **T2m in [200, 330] K is a CONUS range applied to a GLOBAL field.** ERA5's *own* global
  minimum over these chains' dates is **201.0 K** (2021-06-28, Antarctic plateau in austral
  winter) and healthy walkers reach **194.4 K**. The smoke run tripped the 200 K floor at
  6 d lead with a global minimum of 197.8 K. Now `[180, 340]`, which clears the worst
  measured walker by 14 K and ERA5 by 21 K.

Measured global ranges over those 18 known-good states, for the record:

| variable | measured | bound |
|---|---|---|
| `2m_temperature` | [194.4, 324.0] K | [180, 340] |
| `temperature` | [181.1, 324.9] K | [150, 350] |
| `mean_sea_level_pressure` | [93080, 106400] Pa | [87000, 109000] |
| `specific_humidity` | [-1.28e-3, 2.84e-2] | [-5e-3, 0.1] |
| `u`/`v`, 10 m and upper | [-84.9, 109.5] m/s | [-200, 200] |
| `geopotential` | [-5501, **2.08e+5**] m2 s-2 | [-1e4, 5e5] |

That geopotential maximum is where CLAUDE.md's float16 trap lives: float16's ceiling is
65504, so a narrowed archive stores `inf` for the top of every column, silently and on
write.

Item 2 is deliberately **not** what catches the job-1155 failure - a 26 K cold CONUS mean
leaves the field minima far inside these bounds. Items 3 and 5 catch that, by 8x and 200x
respectively. Item 2 is the coarse net for genuine blow-up.

### Item 1 had the same problem, for a different reason

`sea_surface_temperature` is NaN over land by construction - measured at **exactly 0.3389**
of the global grid on every one of those 18 states, bit-identical across both events, three
walkers and three leads, plus a further 0.004225 of the OCEAN points (coastal and ice-edge
cells the source ERA5 mask leaves empty). Gating item 1 on "any non-finite value" marked the
smoke run's healthy lead-3 d segment as diverged and would have reported *GenCast stable
through week (none)* for all eight chains. `NONFINITE_STATIC` gates those variables on
**growth against the chain's own first segment** instead, which still catches a real NaN
blow-up; everything else fails outright on the first non-finite value.

### Thresholds are applied at REDUCE time, never baked into an artifact

The per-segment record stores `ranges_global` / `ranges_conus` - the measured min/max per
variable - and `--stage reduce` applies `BOUNDS` to them. The smoke run made the reason
concrete: `BOUNDS["specific_humidity"]` was corrected *after* the segments were written,
and records that had stored a pre-computed verdict kept reporting a healthy chain as
diverged. **A measurement is durable; a verdict on it is not.**

### The ragged tail costs nothing, and the reason matters

The plan budgeted "a second XLA compile per ragged chain, ~5 min each" for three of the
four leads. Measured on job 1189, from the never-pruned `diag.nc` mtimes:

| segment | steps | wall |
|---|---|---|
| a normal segment (any chain, any lead) | 6 | **197-199 s** |
| the 56-day chain's tail | 4 | **141 s** |
| the 28-day chains' tails | 2 | **83-84 s** |

Two points fit it exactly: **28.6 s per GenCast step plus 26.3 s of fixed per-segment
overhead** (read the parent, build the batch, write a 475 MB compressed state and its
diag). That predicts 140.7 s for the 4-step tail against 141 s measured. There is **no
second compile at all** - the ragged tail is simply a shorter loop.

The reason is `roll_segment`'s `num_steps_per_chunk=1`: GenCast's jitted function is
compiled for ONE step, so `n_steps` never enters the compiled shape and only changes how
many times the generator iterates. This is worth stating because
`aconfig.GATE3_SEG_STEPS`'s own comment gives a different reason - "keeping the segment
shape constant guarantees one XLA compile per worker rather than one per distinct segment
length". The **conclusion** is right and the constant is still worth keeping for other
reasons; the stated **mechanism** is not what produces it, and anyone sizing a run with a
non-uniform schedule should not pay a compile penalty they will not be charged.

For sizing the follow-up: 28.6 s/step here is close to `fevents.GENCAST_A3MEGA_S_PER_STEP`
(28.0) and nowhere near `run_aires.SECONDS_PER_GENCAST_STEP` (52.0). The 52 figure is a
PRODUCTION DMC leg, where 8 shards write N=64 states at once and the zlib pass plus the NFS
write costs as much as the rollout; this sweep writes one state per chain per ~3 min, so
that contention never arises. **Budget 28-33 s/step for a free-running walker pool and 52
for a resampling one.**

### Two more traps, unrelated to thresholds

1. **`data.open_anon` cannot be used on this login node.** It is
   `xr.open_zarr(..., chunks={"time": 1})`, which builds a dask graph with one task per
   timestep per variable. The WB2 surface store has 93,544 timesteps and 66 arrays and
   reached **4 GB of RSS just to open**; the 1-hourly 37-level store (561,264 timesteps)
   took the whole 15 GB box, three times. `astab._wb2_frame` reads it the way
   `data.arco_read` reads ARCO - raw zarr, no xarray, no dask - and the same 58-timestep
   loop then runs in ~90 s at **0.32 GB**. Everything needed lives in the 6-hourly store
   (all chain boundaries are 00Z), whose geopotential chunk is 13 levels rather than 37.
   **This login node has 15 GB of RAM**, which is worth knowing before any prep is written.
2. **Drift must be phase-matched.** Frames alternate 00Z and 12Z and the CONUS diurnal
   range is ~8 K, so a raw last-minus-first carries a +-4 K offset decided by nothing but
   whether the series has an odd or an even number of frames - on a diagnostic whose signal
   of record is 26 K. `drift_from_start` compares against the first frame sharing the last
   frame's hour, and `test_drift_is_phase_matched_against_the_diurnal_cycle` pins it.

### The bug the smoke run caught (job 1187)

`walker.run_segment(parent_state=None)` falls back to `walker.initial_state`, which
resolves the init frames through `aconfig.WALKER_WEEKS` - i.e. the `AIRES_WEEKS` env var,
**3 by default**. This sweep spans four leads in ONE process, so every chain silently
started from the week-3 init: the log printed `init 2021-05-31` and then rolled
`from 2021-06-07`, a whole week late.

The default is 3, which is not one of the four leads here, so **all eight chains were
wrong**, and every one of them would have produced a clean-looking 28/42/56/70-day rollout
from the wrong initial condition. `_segment_status` does catch it one segment later (the
valid time misses the schedule), but a guard that fires after a five-minute XLA compile is
not where this belongs. `stage_walk` now names segment 1's parent explicitly, and
`test_a_chain_reads_its_own_weeks_init_frames` pins it for all four leads.

This is the same family as the checkpoint bug: structurally valid, physically wrong, and
invisible to every existing check.

### State (2026-08-27)

Prep is complete and gated on **artifacts, not exit codes**:

| artifact | state |
|---|---|
| `runs/xres/0p25/week{4,6,8,10}/inputs/*_inputs.nc` | **8/8**, 706 MB each |
| `runs/fcn3/week{4,6,8,10}/ic/*_ic.nc` | **8/8**, 299 MB each |
| `runs/astab/refs/{PNW_HeatDome_2021,WinterStorm_Uri_2021}_era5_ref.nc` | **built**, 58 times each, covering every chain valid time |
| `runs/astab/refs/*_era5_t2m.nc` | **built**, 58 x 105 x 237 each (3.2 MB), the truth row of every map |
| `runs/astab/refs/bands.json` + `gate3_panel.csv` | **built**, 224 samples, 32 walkers, 2 events |
| `--check` over all 8 chains, incl. both ragged tails (2 and 4 steps) | **OK**, no GPU |
| `--stage plan` | **READY: yes** |
| `pytest aires/tests/ astab/tests/` | **281 + 101 passed** (`astab/tests/`: 83 in `test_astab.py`, 18 in `test_maps.py`) |

Budget confirmed by `--stage plan`: 784 GenCast steps (~11.3 GPU-h) + 3,040 FCN3 steps
(~1.2 GPU-h), 8 chains one per GPU on one node, ~11 GB of retained states + ~4.7 GB of
diagnostics + ~18 GB of retained pre-crop Zarrs (released by `--stage prune`), against
309 GB free.

### The result - job 1189, COMPLETED in 1 h 47 m on one 8xH100 node
(plus 1190/1192/1193, the scorer re-run with `t50` after the prune incident below;
1187/1188 were the smoke. Total ~2 h 20 m of one node.)

Walk: 784 GenCast steps, 8 chains, **every one landed exactly on its event peak** (the
hard assertion in `stage_walk`), longest chain 87 min. Score: 8 adapter-free rollouts
(112/168/224/280 x 6 h) and 8 adapter-fed (100/156/212/268 x 6 h), all rc=0, 52.9 GB peak
GPU. Zero retries, zero failed shards.

> **GenCast stable through week 6 (42 d) - every chain.
> FCN3 stable through week 10 (70 d) - every rollout, both arms.**

FCN3 is the clean half, and it is checked **where GenCast actually broke** - the arms carry
`t50` and `z500` as extra global channels precisely so the claim does not rest on a
CONUS-only panel:

| FCN3, all 16 rollouts, 248 checkpoints to 69 d | measured | for comparison |
|---|---|---|
| global-mean 50 hPa temperature | **210.1 - 212.1 K** (a 2 K range over 70 d) | GenCast reached **1211 K** at this level |
| global z500 power above k = 20 | **0.0008 - 0.0065** | ERA5 itself: 0.0011 - 0.0056 |
| bound violations | **zero** | |
| non-finite values | **zero** | |

FCN3 stays on ERA5's own spectral distribution for seventy days. Item 1 never fired for
either model, anywhere.

GenCast diverged on **3 of 8 chains**:

| chain | first out of bounds | via | diverged | worst |
|---|---|---|---|---|
| PNW @ wk4, wk6, wk10 | never | - | - | - |
| Uri @ wk4, wk6 | never | - | - | - |
| PNW @ wk8 | 42 d (0.5% past, **marginal**) | `specific_humidity` | 45 d | 64% past, `u_component_of_wind` |
| Uri @ wk8 | 51 d (4.1% past) | `temperature` | 51 d | 124% past, `temperature` |
| Uri @ wk10 | 30 d (33.1% past) | `temperature` | 30 d | 431% past, `temperature` |

**Divergence is not monotonic in lead.** PNW's 70-day chain never leaves bounds, while
Uri's leaves at 30 d and *both* 56-day chains fail. With n = 1 member per lead this run
cannot separate "that lead is unstable" from "that member diverged" - and it was never able
to. **The follow-up needs an ensemble for the STABILITY question too, not only for the
skill one.** That is a finding about the experiment's design, not a caveat on its result,
and it was not anticipated in the plan.

### Two failure modes, and only one of them is the one everybody expects

This is the part worth carrying forward.

**LOUD - PNW @ wk8.** Every diagnostic fires together from 42 d, exactly as designed:

| lead | T2m anom | diurnal ratio | hi-k fraction |
|---|---|---|---|
| 39 d | -0.1 K | 0.85 | 0.003 |
| 42 d | **-6.3 K** | **0.32** | **0.035** |
| 45 d | **-12.3 K** | **-0.01** | **0.150** |
| 48 d | **-16.9 K** | 0.03 | **0.312** |
| 56 d | -23.7 K | 0.00 | **0.60** |

The plan bet that item 7 (high-wavenumber pile-up) would be "likely the first thing to
fire". For this mode it was right: the zonal power fraction above k = 20 goes from ERA5's
own 0.003 to **0.60** - over half the z500 power at grid scale. z500 drift crosses the
-181 m broken-run line at ~50 d and reaches -410 m. It is the job-1155 signature, reproduced
at long lead by a *correct* checkpoint.

**SILENT - Uri @ wk8 and @ wk10.** The bounds check climbs monotonically (0.33 -> 4.31 of
the bound span from 30 d to 70 d) and **every other diagnostic stays normal the whole way**:
T2m anomaly within +-3.6 K, diurnal ratio 0.90-1.49, hi-k fraction 0.001-0.003, spatial sd
8.6-16.6 K. Opening the final state says why:

```
Uri @ wk10, lead 70 d, temperature by level
    50 hPa    95.3 .. 1211.2 K       <- 1211 K at lat -85.5, lon 113.5 (Antarctica)
   100 hPa    82.3 ..  289.9 K
   150 hPa  -141.6 ..  234.2 K
   ...
  1000 hPa   136.5 ..  315.1 K
same field cropped to CONUS:  50 hPa [203.3, 218.1] K,  T2m [252.0, 301.5] K   -- normal
```

The walker is producing a textbook CONUS troposphere on top of a polar stratosphere at
**1211 K**. Items 3, 5 and 6 are CONUS; item 7 is z500 over 30-60 N. None of them can see
it. The only diagnostic that caught it is item 2 - the crude global bounds check the plan
called "the coarse net" and that I noted "is deliberately NOT what catches the job-1155
failure".

**So the sophisticated, carefully calibrated panel caught 1 of 3 divergences and the crude
one caught 3 of 3.** The difference is not sophistication, it is DOMAIN: item 2 is the only
item evaluated globally and on every variable and level.

Three consequences:

1. Any longer-lead panel needs at least one **global, all-level** diagnostic. A CONUS-only
   panel, however well calibrated, is blind to two thirds of what actually went wrong here.
2. **AI+RES's own observable would not have noticed.** `A_L` is a CONUS box T2m mean. A
   resampling run at 70-day lead would have scored, cloned and reported Uri @ wk10 as a
   perfectly good walker while its stratosphere was at 1211 K. Any long-lead production
   MUST gate on the global state, not on the observable.
3. The figure marks each divergence with a red X **on every panel**, including the panels
   whose own metric never moved - without it, panel 4 reads as "all eight chains healthy to
   70 d", which is false.

### `--stage prune` is one-way, and it now says so

Two things stop working after a prune, both discovered by doing them during this run's
analysis:

* **The adapter-fed arm cannot be re-run.** It launches from each chain's lead-3 d
  `state.nc`, and prune deletes even the pinned states. Re-scoring then needs those
  segments re-walked - `--stage walk --max-segments 1`, ~13 min on 8 GPUs. Cheap, but not
  obvious from the "walker state(s) absent" error. `stage_prune` now REFUSES while any
  score cube is missing.
  (A useful thing fell out of the repair: the re-walked segments reproduced the originals
  to the digit - 297.43 K vs 297.42 K CONUS-mean - confirming `segment_key(walker, step)`
  makes a segment deterministic and re-derivable, which is what makes a lost state a
  15-minute problem rather than a lost experiment.)
* **The FCN3 global items have no other source than the pre-crop Zarrs.** Re-running
  `reduce` after prune silently dropped 362 CSV rows and erased `global_z500` and
  `z500_hi_wavenumber_frac` for both scorer arms. `reduce` now MERGES with the previous
  panel (`_merge_panels`): a value it can no longer compute is carried forward, a value it
  can compute always wins. Measured once, kept.

A second thing came out of that: the FCN3 arms' first pass carried only `t2m` and `z500`,
so **every hard-fail check on the scorer was CONUS-only or z500-only** - precisely the
blind spot this run had just documented for GenCast. "FCN3 stable through week 10" could
not have been supported by that panel. The arms were re-run with `t50` added
(`FCN3_EXTRA_VARS=z500,t50`), so the scorer is now checked at 50 hPa, where the walker
actually broke. A claim about one model has to be tested where the other one failed.

### What this settles for the follow-up

The mini-Gate-3 sizing in the plan (16 walkers x ~6,300 GenCast steps ~ 91 H100-h) is
unchanged in shape but its per-step cost is now measured, not assumed: **28.6 s/step for a
free-running pool**, not the 52 s of a resampling leg (see "The ragged tail costs nothing").
That is ~50 H100-h for the walker half, not 91.

What changed is what the follow-up has to measure. It cannot just ask "does the score still
rank?" at 4/6/8/10 weeks - it must also establish, with an ensemble, **what fraction of
walkers survive each lead**, because this run showed that fraction is neither 0 nor 1 and
is not ordered by lead. A resampling scheme that clones a diverged walker is worse than
useless: divergence is exactly the kind of extreme the score function is built to select
for.

### Stable is not useful

Gate 2 measured FCN3's rank-equivalence horizon at **~8.5 d** and Gate 3 measured score
skill rho_s = 0.797 at 9 d and 0.959 at 12 d against a 0.5 threshold. If both models stay
stable to 70 d, that is **permission** to run AI+RES out there, not a reason to: past the
predictability horizon the score function ranks chaos, resampling adds no information, and
the frozen schedule (C_k at 3/6/9/12/15 d) would leave a 55-day free-running carry leg that
washes out any tilt it did apply.

Establishing the **useful** horizon needs a rank correlation, which needs an ensemble. The
natural follow-up is a mini-Gate-3 at each lead: 16 walkers x (56+84+112+140) steps
~ 6,300 GenCast steps ~ **91 H100-h**, plus FCN3 scoring - 2-3 nodes over ~12 h. Sizing
that is what this run is for.

Nothing here is compared to ERA5 as if agreement were expected past ~12 d; beyond that,
ERA5 enters only as a magnitude/sign sanity descriptor, never as "error" or "skill".

## The FCN3-only extension to 20 weeks (140 d) - built 2026-08-27, RUN as job 1194 on 2026-08-28

Job 1189 closed one of the sweep's two questions and left the other open:

* **GenCast**: diverged on 3 of 8 chains by week 6. A measured limit.
* **FCN3**: clean at week 10 on **every** rollout of **both** arms. That is a FLOOR, not a
  limit - the sweep stopped before the scorer did, so "stable through week 10" is a
  statement about how far it was asked to go.

Weeks **12/14/16/18/20** (84/98/112/126/140 d) extend the sweep on the **scorer only**. At
those leads the GenCast walker rolls ONE 3-day segment - the adapter-fed arm's launch state
- and both FCN3 arms roll the full horizon. Eighteen chains now: 2 events x 9 leads.

Why FCN3-only: the open question is the scorer's, and the closed one costs 46x more.
Extending the walker to 140 d is 280 GenCast steps per chain (~4 h of one H100); the FCN3
rollout of the same span is ~13 min. `--stage plan` prices the whole extension at **60
GenCast steps (~0.9 GPU-h) + 8,840 FCN3 steps (~3.4 GPU-h)**.

### The one thing that could have gone wrong, and what stops it

An extension chain's walker panel is ONE clean checkpoint at 3 d. Counted naively that
reads as a clean chain, `_all_clean_through` walks straight past week 10, and the headline
reports **"GenCast stable through week 20" off three days of rollout**.

Every CSV row now carries **`reached_peak`**; `reduce._model_survival` drops the
(model, week) pairs where it is 0 - neither clean nor failed, simply not tested - and
`reduce.untested_weeks` is the single rule the printed headline, `stability.png` and the
map stage all consume. The headline prints what each verdict covers ("tested to the peak:
GenCast week(s) [...], FCN3 week(s) [...]"), the figure does not draw the walker at an
extension lead, and no walker map is written there.
`test_a_lead_the_walker_never_reached_is_not_a_gencast_verdict` pins it; a CSV from before
the extension (no such column) still reduces, and means "everything was walked", which is
what was true then.

### What changed in the code

| | |
|---|---|
| `gencast_s2s/config.py`, `fcn3/fevents.py` | `WEEKS_TO_LEAD_DAYS` now 2..20 (`{... 12: 84, 14: 98, 16: 112, 18: 126, 20: 140}`), still mirrored, still additive - 2/3/4 untouched |
| `astab/sconfig.py` | `STAB_WEEKS` = 9 leads (**append-only** - 4/6/8/10 keep walker slots 0..3, so every artifact job 1189 wrote stays addressable); new `STAB_WALK_WEEKS` + `walks_to_peak()` |
| `astab/schedule.py` | a chain now has TWO horizons: `horizon_days` (init->peak, what FCN3 rolls) and `walk_days` (what the walker rolls). `Chain.schedule` is the WALK schedule; `full_schedule`/`checkpoints`/`walk_end` are new |
| `astab/stages.py` | plan prints both horizons and prices them separately; walk lands on `walk_end`, not the peak; **a pruned chain's landing is verified from its `stab.json`** so a resubmit over finished chains costs seconds instead of failing the pool |
| `astab/reduce.py` | `reached_peak` per row, `untested_weeks()`, survival gated on it |
| `astab/plots.py`, `astab/maps.py` | the walker is not drawn where it was not rolled |
| `astab/refs.py` | `ref_times` reads `full_schedule` - **128 times per event**, not 58; off `schedule` it would build 3 days of ERA5 truth for a 140-day chain |
| tests | 190 in `astab/tests/` (was 101); `aires/tests/` unchanged at 281 |

`AIRES_STAB_WALK_WEEKS=4,6,8,10,12,14,16,18,20` turns the extension leads into full walker
chains and nothing else in the package changes. Price it with `--stage plan` first.

### Prerequisites (login node, CPU + internet)

```bash
for W in 12 14 16 18 20; do XRES_EVENTS_SEL=PNW_HeatDome_2021,WinterStorm_Uri_2021 \
    XRES_PREP_SKIP_MODELS=1 python run_xres.py --stage prep --res 0p25 --weeks $W; done
for W in 12 14 16 18 20; do FCN3_WEEKS=$W \
    FCN3_EVENTS=PNW_HeatDome_2021,WinterStorm_Uri_2021 \
    bash slurm/aires_env.sh fcn3 python fcn3/run_fcn3.py --stage prep; done
PYTHONPATH=. python -m astab.run --stage refs          # REBUILDS: 58 -> 128 times/event
PYTHONPATH=. python -m astab.run --check --weeks 12,14,16,18,20
PYTHONPATH=. python -m astab.run --stage plan          # must print READY: yes
```

One PROCESS per week (`FCN3_WEEKS` is read at import) and the **`fcn3` env** for that
loop only - `stage_prep` imports earth2studio, which `moe` does not have; the xres prep and
every `astab` stage are `moe`.

10 x 706 MB of init frames + 10 x 299 MB of FCN3 ICs ~ 10 GB. The earliest init in the
sweep is now **Uri week-20 at 2020-09-28**, still well inside WB2.

Then `sbatch slurm/aires_stab.slurm` - its default `WEEKS` is all nine leads, and the eight
finished chains skip in seconds.

### Disk - the constraint to watch

The retained pre-crop Zarrs scale with FCN3 **steps**, not chains: **~51 GB** for the
extension (a week-20 rollout is twice a week-10 one) against **303 GB free** on this
98%-full filesystem. Run `--stage reduce` and then `--stage prune` before adding leads.

### Result - job 1194, COMPLETED in 2 h 16 m on one 8xH100 node (2026-08-28)

`sbatch slurm/aires_stab.slurm` over all nine leads; the eight finished chains skipped in
seconds. Walk 31 min, adapter-free scoring 32 min, adapter-fed scoring 73 min. Three walk
shards lost attempt 1 to CUDA out-of-memory during their first compile - three cards on
that node were not fully ours at that moment, the pattern documented under "GPU
contention" below - and attempt 2 rolled everything that was missing, which is what the
retry exists for. Reduce, figures and maps ran on the login node
(`logs/astab_ext_reduce.log`); the retained Zarrs were pruned afterwards
(`logs/astab_ext_prune.log`, 115 GB released).

**FCN3 stable through week 14 (98 d) - every rollout, both arms.** The scorer's limit is
now measured rather than floored, and it is event-dependent:

| chain | ERA5-IC arm | adapter-fed arm |
|---|---|---|
| PNW week 16 (112 d) | diverged at 75 d (non-finite field, `inf` severity) | diverged at 108 d |
| PNW week 18 (126 d) | clean to 126 d | diverged at 111 d |
| PNW week 20 (140 d) | diverged at 129 d | diverged at 105 d |
| Uri weeks 12-20 (84-140 d) | clean, every lead | clean, every lead |

Two things to read off it. First, FCN3's divergence is not monotonic in lead either: the
ERA5-IC arm survives 126 d at week 18 and fails at 75 d of the week-16 chain - the same
n = 1 ambiguity the walker showed at weeks 8/10. Second, the adapter-fed arm fails at a
consistent 105-111 d on all three PNW chains while the ERA5-IC onsets scatter
(75 / never / 129 d), which is at least consistent with the adapter-fed launch state
carrying something the ERA5 IC does not; three chains cannot establish it. `stability.csv`
(12,274 rows) and `stability.png` now carry all 18 chains.

The caveat is unchanged and sharper out here: Gate 2 put FCN3's rank-equivalence horizon
at ~8.5 d, so a 140-day rollout is ~16x past the lead at which the score was last shown to
rank. This measures whether the scorer still makes an atmosphere, and nothing else.

## The ensemble follow-up: 100 GenCast members at week 8 (PNW) - 2026-09-03/04, job 1197, DONE

Job 1189 rolled ONE member per lead and found divergence non-monotonic in lead - PNW@wk8
blew up at 45 d while PNW@wk10 never left bounds - which with n = 1 cannot separate "that
lead is unstable" from "that member diverged". This run puts an ensemble on the STABILITY
question: **100 members of the GenCast 0.25 deg walker, PNW_HeatDome_2021, init
2021-05-03 -> peak 2021-06-28 (56 d), GenCast only, no FCN3 arm.** Members differ only in
their diffusion noise stream (`walker.segment_key(member, step)`, the same family `xres`
and `gencast_s2s` use for their ensembles); member 2 shares job 1189's stream at the same
init and is a determinism control, counted once. Deliverable: the survival curve - the
fraction inside the physical bounds at each of the 19 three-day checkpoints, with a 95%
Wilson interval - plus the loud/silent and worst-variable/level breakdown, as
`figures/astab/ensemble_PNW_HeatDome_2021_wk08.png` and
`runs/astab/ensemble_PNW_HeatDome_2021_wk08{,_members}.csv` + `.json`. The same mechanism
runs at another lead with only `--weeks` changed, which is how the longest STABLE lead
gets found before an AI+RES production is booked there. Plan:
`/home/ubuntu/.claude/plans/can-you-run-gencast-lovely-pillow.md`; design in
`astab/__init__.py` and `astab/ensemble.py`.

### What landed in code (2026-09-03; 264 tests in `astab/tests/`, 293 in `aires/tests/`, green)

| | |
|---|---|
| `astab/sconfig.py` | the `ens_*` tree - `runs/astab/<event>/ens/wk<NN>/m<MMM>/stepKK/`, keyed by WEEK then MEMBER, never by walker slot; `ENS_DIAG_VARS` (T2m-only slim); `ENS_KEEP_STATE_MEMBERS` (member 2's final state survives the release) |
| `astab/schedule.py` | `Chain.member` (LAST field, default None - the frozen 18 are `Chain(e, w)` unchanged), `Chain.seed`, `Chain.segment_dir`; every path dispatches on `member`, and the FCN3 path helpers RAISE for a member chain rather than alias the frozen cubes |
| `astab/record.py` | `member`, `seed`, `diag_vars`, `ranges_global_levels`, `extremes_global` (last frame) - with no states retained these are the only surviving attribution of WHERE a member left its bounds |
| `astab/stages.py` | walk passes `ch.seed` to `run_segment` and `ch.segment_dir` to the stale/partial rmtree (a week-8 member's walker SLOT is 2, so the slot-built path was job 1189's `walkers/w02/`); `_slim_diag` runs after `ensure_record(force=True)` and the physics gate, `_release_states` after the landing assertion; member chains fail one at a time; `stage_plan` prints the ensemble budget |
| `astab/ensemble.py` | `wilson`, `member_summary` (fate precedence diverged > marginal > incomplete > clean; first-out-of-bounds and diverged are different dates), `survival_curve` (members AT RISK per checkpoint, never the run size), `stage_ens_reduce` (tolerates in-flight members - it is the mid-flight monitor), `stage_ens_figs` |
| `astab/run.py` | `--ens N`, `--ens-members`; refuses `score`/`prune`/`maps` and any non-`STAB_WALK_WEEKS` lead; `check` runs once on the member-free chain; reduce/figs dispatch on the FLAG |
| `slurm/aires_stab_ens.slurm` | a Slurm ARRAY, one exclusive node per task; `NSHARDS = AIRES_ENS_NODES x 8`, never the array count (a one-task redo would otherwise select zero members and exit 0); launches staggered 20 s; the 3-attempt cache-aware retry |

### Run log

- **Prereqs** (login node, `moe`): both suites green;
  `python -m astab.run --stage plan --ens 100 --event PNW_HeatDome_2021 --weeks 8 --nshards 64`
  printed READY: yes - 11,200 GenCast steps, ~89 GPU-h at the measured 28.6 s/step and
  ~162 at the 52 s ceiling; slim diag ~1.1 GB, records 0.02 GB, transient states
  <= 2 x 477 MB per live worker; `--check --ens 100 ...` OK (19 segments tile the 56 d
  walk exactly, landing on 2021-06-28).
- **Job 1195** (smoke, 1 node): landed on node 4 and died in 14 s - `Unable to initialize
  backend 'cuda'` - because every card on that node was held by a teammate's non-Slurm
  `inkling` probe at 80.6 of 81.5 GB. Cancelled. A survey over SSH found FIVE of the eight
  nodes in that state while `sinfo` showed all eight idle; see "GPU contention" below.
- **Job 1196** (smoke, pinned to clean node 1): COMPLETED in 22 min. One member, one
  segment: physics gate -0.07 K vs ERA5 at 3 d; diag slimmed 34 -> 0.4 MB; the record
  carries member/seed/diag_vars/ranges_global_levels/extremes_global; the frozen sweep's
  317 files byte-identical before and after. **Startup measured 21.5 min** for model load +
  XLA compile + the first 6-step segment, against the 13 min budgeted from job 1189.
- **Job 1197** (production): `sbatch --array=0-3 --exclude=nucla3m-a3meganodeset-[0,4,5,7]
  --export=ALL,AIRES_ENS_NODES=4 slurm/aires_stab_ens.slurm` at 21:46 UTC - nodes 1, 2, 3,
  6 (node 3 had freed up between the survey and the submit), 32 shards, 3-4 members each,
  every task READY: yes. Eight nodes were asked for; four was every node the cluster had
  free of foreign processes. The reduce and figure stages were exercised on the smoke's
  one segment while the run was in flight (RC 0, figure and CSVs written).

### RESULT - job 1197, COMPLETED 2026-09-04 02:21 UTC (tasks 3 h 18 m to 4 h 34 m, no retries)

**9 of 100 members diverged by 56 d (9%, Wilson 95% 5-16%); 91 never left the bounds.
Every member was clean through 36 d.** The survival curve (never past 2% of a bound span):
100% to 36 d, 99% at 39 d, 97% at 42-45 d, 95% at 48 d, 93% at 51 d, 92% at 54 d, 91% at
56 d. Onsets 39-56 d, median 48 d; no member recovered.

| member | left bounds (first variable, % of span past) | diverged | worst at 56 d |
|---|---|---|---|
| m034 | 39 d (q, 5.0%) | 39 d | u-wind 100 hPa, 101% past |
| m004 | 39 d (q, 0.9%) | 42 d | u-wind 100 hPa, 95% |
| m046 | 39 d (q, 0.9%) | 42 d | u-wind 50 hPa, 89% |
| m008 | 48 d (q, 3.7%) | 48 d | temperature 100 hPa, 32% |
| m069 | 45 d (q, 0.8%) | 48 d | u-wind 50 hPa, 43% |
| m022 | 51 d (q, 4.7%) | 51 d | u-wind 50 hPa, 18% |
| m075 | 48 d (q, 1.8%) | 51 d | u-wind 50 hPa, 29% |
| m083 | 54 d (q, 3.6%) | 54 d | specific humidity 850 hPa, 6% |
| m013 | 54 d (q, 1.1%) | 56 d | specific humidity 850 hPa, 4% |

**One failure mode, nine times.** Every one of the nine first left the bounds on specific
humidity - NEGATIVE q at 850 hPa, -0.006 to -0.010 kg/kg at a mid-latitude grid point,
against a floor of -0.005 that `astab/diag.py` set 4x below the healthy walkers' -0.0013 -
at 39-54 d in; and every one then grew into the stratospheric winds (u at 50 or 100 hPa,
6 of 9) or temperature at 100 hPa (1), with the two latest onsets (54 d) still on humidity
at 56 d because they had no time to grow. Once past 2% an excursion never came back: the
severity climbs monotonically to 18-101% of a bound span by 56 d (panel B).

**All nine were LOUD, none silent.** Every CONUS diagnostic fired on every one, and the
CONUS-mean T2m anomaly of a diverging member collapses by 10-25 K within ~9 d of its onset
(panel C) - so AI+RES's own observable `A_L` would have seen each of these. That is the
opposite of job 1189's Uri chains (1211 K polar stratosphere over a normal CONUS). It says
the silent mode is not the only one, not that it is absent: one event and one lead cannot
show that, and the CLAUDE.md rule - gate a long-lead production on the GLOBAL state, not
on the observable - stands.

**Member 2 did not reproduce job 1189.** Same noise stream, same init, same checkpoint:
1189 left the bounds at 42 d and diverged at 45 d; here member 2 never left them.
bf16/XLA non-reproducibility (the 1189 re-walk already gave 297.43 vs 297.42 K) is enough
to move a member across the divergence boundary at 6 weeks - a single chain's fate at this
lead is not a property of its seed. That is the strongest form of the sweep's own finding
that n = 1 cannot separate the lead from the member: the same member cannot separate them
either.

**Cost, measured.** 1,900 segments = 11,200 GenCast steps in **113 GPU-h** of worker time
(startup included) on 32 workers over 4 nodes; wall **4 h 34 m** for the four shards that
held 4 members, 3 h 18 m for a 3-member shard. Per member **62.4 min median** (61.8-70.2)
= **33.4 s/step** with 8 concurrent bf16 workers per node, against the 28.6 s/step budgeted
from job 1189; the first member on each worker took 72 min median (67-116) including its
startup. No pool retry fired (4 attempts, one per task). Disk: `ens/wk08/` is **1.1 GB**
at the end - 1,900 records, 1,900 T2m-only diagnostics, member 2's final state (474 MB) -
and free space returned to 266 GB; the peak was 67 transient states (~32 GB). The frozen
sweep's 317 files are byte-identical before and after.

Artifacts: `figures/astab/ensemble_PNW_HeatDome_2021_wk08.png`;
`runs/astab/ensemble_PNW_HeatDome_2021_wk08.csv` (long, per checkpoint),
`..._wk08_members.csv` (one row per member), `..._wk08.json` (curve, headline, provenance,
caveats); shard logs `logs/ens/Vayuh-s2s-1197-walk-try1-shard*.log`.

**The figure was redrawn on 2026-09-04** (same path, `astab/ensemble.py::_fig_one`) to say
the same thing in fewer words and in kelvin as well as percent. Panel A is the survival
curve as before. B is the excursion past a bound in % of the bound's span (log axis, "0" on
the floor) and C sits directly under it on the same lead axis with the CONUS-mean T2m
anomaly in K, a dot on every diverged trajectory at its first checkpoint past 2% in both,
so the crossing and the collapse are read as one lead. D plots kelvin against percent for
every checkpoint that was past a bound at all (hollow below 2%, filled at or past it):
every filled point sits below the coldest checkpoint of any clean member (-2.3 K) and every
path runs down and to the right - at the first checkpoint past 2% the CONUS mean is already
-4 to -10 K, which is what makes 2% a threshold and not a nuisance. The reference for
"cold" is now the run's own 91 clean members at every checkpoint, not the Gate 3 band
(which was measured to 21 d and extrapolated beyond; it still decides loud/silent in the
CSV and JSON). The 2%-in-units note (3.2 K T2m, 4 K T aloft, 8 m/s wind, 2.1 g/kg q) is
computed from `BOUNDS`, and job 1189's single chain is drawn dashed through B-D, summarised
by the same `member_summary` as the members. The heatmap panel is gone; its "which
variable" content is the table above and `_members.csv`.

**What it means for the next step.** At this event and lead the 0.25 deg walker survives
56 d with probability ~0.91 [0.84, 0.95], and every failure declared itself through the
CONUS observable within a segment or two of onset. The 56-d curve is also a first-order
S(L) for every shorter lead from THIS init: 100% through 36 d (Wilson lower bound 96.3%),
so week 4 (28 d) sits well inside the clean window and week 6 (42 d) sits on the first
onsets (97% [91.5, 99.0] never diverged at 42 d) - to be confirmed from those leads' own
inits, which is `--weeks 6` / `--weeks 4` with no code change. For an AI+RES production at
week 6 or 8 the resampler must be able to KILL a diverging walker rather than clone it,
and with loud failures at a 9% rate a global bounds check at each resampling time is
sufficient here. Whether the lead is USEFUL is the other question (see "Stable is not
useful"). The companion runs that would make this general: Uri at week 8 (the silent
mode's event) and PNW at week 10 from its own init (`--weeks 10`, 24 segments,
~129 GPU-h).

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
  about -8 K over the northern plains (`figures/aires/null_20240329/adaptest_maps_null_20240329.png`).
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
plot, extremes blue / controls orange), `figures/aires/<case>/adaptest_maps_<case>.png` (ten
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

# the lead-time stability sweep (the astab/ package) -- all CPU except `walk`/`score`
PYTHONPATH=. python -m astab.run --stage plan               # readiness + budget; must say READY: yes
PYTHONPATH=. python -m astab.run --check                    # every segment SHAPE, incl. both
                                                            #   ragged tails. Isolated per chain:
                                                            #   this login node has 15 GB of RAM
PYTHONPATH=. python -m astab.run --stage refs               # ERA5 scalar series + the T2m truth
                                                            #   fields the maps need. Needs internet
PYTHONPATH=. python -m astab.run --stage calibrate          # re-measure the bands on the Gate 3
                                                            #   walkers already on disk (~12 min)
PYTHONPATH=. python -m astab.run --stage reduce --stage figs --stage maps
column -s, -t runs/astab/stability.csv | head -30           # first FAIL row per chain, one query
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

Resolved by lead (third panel of `figures/aires/PNW_HeatDome_2021/gate2_PNW_HeatDome_2021.png`) the same
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
`figures/aires/PNW_HeatDome_2021/gate3_PNW_HeatDome_2021.png`, 16 walker trajectories (57 GB) and 80 score
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

**Two rates, measured, and both were wrong in the docs - in the same direction.**

- FCN3 runs at **1.40 s per member-step** (job 1161: 8.41 s for a 6-member lead step), not
  the 1 s/step `slurm/aires_gate3.slurm` budgeted - which is exactly why Gate 3's score
  phase took 85 min against a predicted 50.
- GenCast 0.25 deg runs at **~52 s/step in a production leg** (job 1172, leg 3: 48 steps
  per shard in 41.4 min, model already resident), about 3x Gate 3's ~16 s/step of pure
  rollout. **The extra is not the model, it is the checkpoint**: a leg writes a 473 MB
  compressed restart state per segment, and 8 shards doing that at once make the zlib pass
  plus the NFS write cost about as much as the rollout. That is the standing price of a
  walker that can be stopped and cloned, and it should be budgeted per LEG, never from a
  rollout benchmark.

Both are constants at the top of `run_aires.py` and `--stage prep` uses them. **The pilot
is ~61 H100-h** (36 walk + 25 score), ~7.6 h on one 8-GPU node; the Standard-RES control
is ~36 H100-h, all walkers.

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

## What Phase 4 established - the pilot (job 1172, N=64, FCN3 score)

**AI+RES reached the observed heat dome; direct sampling does not.** 7 h 45 m on one
8xH100 node, 465 min of coordinator wall - exactly the ~7.6 h the corrected rates predicted.

| | n | mean A_L | sd | max | reaching the observed +7.72 K |
|---|---|---|---|---|---|
| **AI+RES** | 64 | +8.24 | 1.49 | **+10.29** | **42** |
| GenCast direct (Gate 3 walkers) | 16 | +2.54 | 3.02 | +7.49 | 0 |
| GenCast direct (xres) | 24 | +2.46 | 2.66 | +7.08 | 0 |
| FCN3 direct | 24 | +2.29 | 3.10 | +7.87 | 1 |

**The headline claim needs no estimator at all.** 42 of 64 resampled walkers reach or
exceed the ERA5 value and the population extends to +10.29 K, while 40 direct GenCast
members stop at +7.49 K and reach it zero times. That is a statement about where walkers
landed - it does not depend on a single weight - and it is what `aires.md` scoped the pilot
to answer.

**The probability, which does depend on the weights: `P(A_L >= +7.72) = 0.054`,** about 1
in 19 members. The estimated curve runs out to `P = 3.9e-4` (1 in 2600) at +10.29 K, where
40 direct draws can only report a 95% upper bound of 0.14-0.19. Read it with the
calibration below, not as three significant figures.

### Is the estimator trustworthy here?

Two checks, and they say different things:

- **Where RES and direct sampling can both be measured, they agree.** Over +3.7 to +6.8 K
  the ratio of the two exceedance curves has a geometric mean of 1.27 and a range of
  0.97-2.13 - inside the sampling error of 16-24 direct draws, whose own Wilson intervals
  span factors of two to three there. The RES curve is not systematically displaced.
- **The total-mass check is 0.568, not 1** - and that is ordinary, not a bug. Replaying the
  schedule 2000 times at N=64 puts 0.568 at the **33rd percentile**; the statistic is
  strongly right-skewed (mean 0.99, **median 0.73**, sd 1.17) and 36% of runs land below
  0.6. `dmc.py`'s "should be ~1" is the mean, not what a single run at this N typically
  reads.

The mechanism behind both is the same, and it is visible in the weight histogram: 58 of the
64 final weights sit near 1 and the total mass is carried by about three walkers at 10, 12
and 34.5. **RES deliberately abandons the bulk** - only 11 of 64 founders still have
descendants and no walker finished below +3.54 K - so the mass that belongs at low `A_L` is
represented by a handful of heavily-weighted survivors and `P(everything)` is noisy. Below
+3.54 K the RES curve is flat at 0.568 because it has no members there; that segment is the
total-mass check drawn sideways, **not** a bulk estimate, and must not be read as one.
Absolute probabilities carry roughly a factor of two at this N. The ratio comparisons and
the qualitative reach do not.

### The schedule held up

| step | 3 d | 6 d | 9 d | 12 d | 15 d |
|---|---|---|---|---|---|
| pilot ESS/N | 1.00 | 0.277 | 0.261 | 0.461 | **0.602** |
| `ctune` replay | 1.00 | 0.431 | 0.358 | 0.378 | 0.583 |
| pilot max children | 1 | 11 | 11 | 8 | 3 |

It ran hot in the middle - steps 2 and 3 sat at about the 5th percentile of the replay -
and finished essentially on the predicted value, **0.602 against the N/4 = 0.25 target**.
The mid-run diagnosis was made before the answer was known, and held: the step-2 scores
were consistent with Gaussian (Shapiro-Wilk p = 0.51), exactly-Gaussian scores at N=64 give
0.268 at their own 5th percentile, and replays conditioned on a step-2 ESS that low all
finished above target, because consolidating early onto good ancestors leaves the later
steps less degenerate. `C_k = (0, 1.0, 1.4, 1.8, 2.0)` is confirmed at N=64 on real scores.

The score drove the population into the tail monotonically: theta = **+1.98, +4.07, +5.57,
+6.62 K** at 6/9/12/15 d.

### Score skill on the resampled population, and why it reads lower than Gate 3

`rho_s(theta(t_k), realized A_L)` came out **+0.23, +0.73, +0.85, +0.93** at 6/9/12/15 d,
against Gate 3's +0.61, +0.80, +0.96 on free-running walkers. **That is range restriction,
not a worse scorer.** Gate 3's 16 walkers spanned -3.2 to +7.5 K; the resampled population
spans +3.5 to +10.3 K and is concentrated above +7. Correlation is attenuated by exactly
that compression, and these walkers are descendants rather than independent draws, so the
number is not comparable to the gate's - the figure says so on its face.

### Clones separate on their own - the design bet paid off

The paper perturbs log surface pressure with spherical harmonics because PlaSim is
deterministic and its clones would otherwise be identical. GenCast's diffusion noise was
supposed to make that unnecessary. It does: sibling clones born at 15 d differ by **1.0 K
RMS 2 m temperature over CONUS after a single 12 h step**, growing to ~2.5 K by the 21 d
horizon, against ~3.0-4.5 K for unrelated walkers.

The honest half: at the horizon siblings are still separated by only about 60% of the
unrelated-pair distance, so **clones remain correlated at verification** and the effective
number of independent tail samples is smaller than 64. This is the first quantitative
handle on that, and it is the argument for adding the paper's perturbation if a later run
resamples harder or later.

### Cost, measured

| leg | walk | score |
|---|---|---|
| 1 (0-3 d) | 40.3 min (48 rolled, 16 seeded from Gate 3) | skipped, `C_1 = 0` |
| 2 (3-6 d) | 34.4 min | 71.8 min |
| 3 (6-9 d) | 41.4 min | 55.7 min |
| 4 (9-12 d) | 42.0 min | 43.2 min |
| 5 (12-15 d) | 39.0 min | 30.6 min |
| 6 (15-21 d) | 64.9 min (128 segments) | - |

Peak disk 37 GB against the 92 GB projected: with `--keep-states 2` only one or two
segments of states are ever resident. All 448 diagnostics cubes retained.

Artifacts: `runs/aires/PNW_HeatDome_2021/res/pilot/{res_result,compare}.json`,
`compare_curve.csv`, and nine figures `figures/aires/PNW_HeatDome_2021/aires_*_PNW_HeatDome_2021_pilot.png` -
five line/scatter figures from `aires/aplots.py` (`exceedance`, `diagnostics`, `genealogy`,
`composite`, `trajectory`) and four maps from `aires/amaps.py` (`map_compare`, `map_error`,
`map_walk`, `map_spread`).

The `trajectory` figure needs **all 448 diagnostics cubes**, not just the surviving
lineages': the killed branches are not in any artifact, because nothing downstream needed
them, so it rebuilds the whole forest from disk and cross-checks the rebuilt survivor path
against `compare.json`'s `series_box`. It hard-fails rather than drawing a plausible but
wrong tree if the parent lookup is off.

### What the maps established (2026-08-21)

The box mean cannot tell a real dome from a displaced ridge that scores the same number.
`aires/amaps.py` reduces every field through the same `aindex.field` call on the same
105x237 grid, so walkers, direct members and truth are directly subtractable.

| field | box bias vs ERA5 | box RMSE |
|---|---|---|
| **AI+RES, all 64 (mean)** | +0.52 K | **0.86 K** |
| FCN3, most extreme member | +0.16 K | 1.14 K |
| GenCast 0.25 deg, most extreme member | -0.64 K | 1.38 K |
| GenCast walkers, most extreme | -0.23 K | 1.44 K |
| AI+RES, the 42 that reached ERA5 (mean) | +1.36 K | 1.55 K |
| *HRRR observed* | *-0.83 K* | *2.34 K* |
| AI+RES, top 10 (mean) | +2.31 K | 2.49 K |
| AI+RES, single most extreme | +2.58 K | 2.84 K |
| GenCast walkers, 16 free-running (mean) | -5.17 K | 5.32 K |
| GenCast 0.25 deg, 24-member mean | -5.26 K | 5.41 K |
| FCN3, 24-member mean | -5.43 K | 5.52 K |

Three things, in decreasing order of what they can claim:

- **The baseline ensemble MEANS have no dome at all** at 21 d lead: -5.2 to -5.4 K of box
  bias. Their most extreme members do, in the right place, at +7.08 / +7.49 / +7.87 K. The
  pattern is available to a 24-member ensemble; the amplitude is only barely.
- **The AI+RES population mean is the closest field to what happened (0.86 K), and that is
  NOT a skill claim.** That population was deliberately aimed at this event, so its
  unweighted mean is not an estimate of the forecast mean. What it does say is that the
  observed event sits *inside* the resampled distribution rather than at its edge - which is
  what a rare-event sampler is for. Quote it against the 2.34 K that ERA5 and HRRR disagree
  by, never against a baseline's forecast skill.
- **The far tail overshoots the event.** Top-10 composite +2.31 K of box bias, single most
  extreme +2.58 K. Correct behaviour for an algorithm told to maximise `A_L`, and a caution:
  the extreme end of this population is *more* extreme than 2021, not a reconstruction.

Two more, from the other two map figures:

- **At the 9 d barrier the lineage that ends as slot 60 (`A_L` +10.29 K) - then occupying
  slot 63 - had the 4th-coolest persistence index of the 64 walkers alive**: -2.69 K
  against a population mean of +1.46 K (z ~ -1.3, i.e. well under one expected child from
  a persistence score), while FCN3 ranked it 10th HIGHEST (theta +7.18 K) and it drew 11
  children, the step's maximum multiplicity. (Corrected 2026-08-21: this bullet originally
  read "slot 60 was the COOLEST of all 64 lineages, -2.69 K against a mean of -0.17 K" -
  those numbers describe the 14 back-traced SURVIVOR ancestries, weighted by descendants,
  not the population at the barrier; recomputed from the step03 diag cubes plus
  `res_result.json`'s theta/parents. Same conclusion, right reference set.) The Z500
  contours in `map_walk` show why: the ridge was already building offshore while the
  surface under it had not warmed. One walker, so an illustration - the evidence stays
  Gate 3's aggregate (persistence 0.06 vs FCN3 0.80 at that lead) - but it shows what
  those numbers mean.
- **The resampled population's own box spread is 1.77 K against 2.85 / 3.17 / 3.25 K for the
  three free ensembles.** A tail-selected population of clones is *tighter* than a free one:
  the sibling-divergence deficit, seen as a map, and the reason 64 resampled walkers are
  worth fewer than 64 independent draws.

Traps hit drawing them, all cosmetic but all real:

- `coolwarm`'s midpoint is near-white, so on the trajectory figure a mid-`A_L` survivor was
  indistinguishable from a grey killed branch - the one distinction the figure exists to
  make. Survivors now use a truncated `YlOrRd`; `colors` never gets a neutral point on a
  panel that also carries grey.
- `magma_r` saturates to black over most of CONUS on the spread field and erases the
  coastlines. `Purples` instead.
- A per-panel colorbar on a grid that SHARES its scale repeats the same axis a dozen times
  and lands between two rows, where it reads as a caption for the row above. One shared bar,
  and on the error figure an explicit `cax` - eleven map axes span all three columns, so a
  colorbar that steals space from them centres across the full width and covers the ranking
  panel in the twelfth slot.
- The raw 12-hourly index has a real diurnal swing of several kelvin (a dome's daytime
  anomaly far exceeds its nighttime one, and the climatology is sampled per hour so the
  anomaly does not remove it). The trajectory figure plots a 24 h running mean, which is
  continuous across a barrier because every branch polyline carries its parent's last frames.
- **A running mean of consecutive PAIRS lands on the midpoints, not on the samples.** The
  first version of `aplots.daily` returned `0.5*(x[:-1]+x[1:])`, which slid every curve on
  the trajectory figure 6 h to the left: the forest appeared to start at lead 0.75 d, every
  branch appeared to die a quarter-day BEFORE the barrier that killed it, and no polyline
  reached the peak. It now uses the centred trapezoid `(1, 2, 1)/4` - the mean of the two
  pair means straddling each frame, which is the same exact null at the 24 h period but
  stays on the sample's own timestamp. Branch polylines therefore carry TWO lead-in frames
  from the parent (`daily(..., drop_first=1)`) so the point sitting ON the barrier still
  gets a full centred window.
- **The x axis of the trajectory figure counts DOWN.** It is lead time remaining to the
  peak (21 d at the init, 0 at the peak), converted from the upstream lead-from-init at the
  drawing boundary only. Ticks are at `GATE3_SEG_DAYS` so they land on the barriers. Note
  the pilot page quotes barriers as lead FROM the init (3, 6, 9, 12, 15 d), which on this
  axis is 18, 15, 12, 9, 6 - `docs/aires_pilot.html` says so explicitly. The genealogy
  figure's tree panel still uses lead-from-init.
- Every trajectory is drawn from the event's ERA5 init (`A.gencast_inputs_path`), not from
  the first forecast frame at +12 h, so all 64 leave one shared point at 21 d. That file is
  built by the xres prep stage and lives outside `runs/aires`; `_init_index_series` returns
  `None` and the figure falls back to starting at 0.5 d if it is absent.

## The CFSv2 operational baseline on the trajectory figures - 2026-08-25

**What a real forecast centre had at the same 21-day lead.** Every other curve on the
trajectory figure is a machine-learning model, which made the figure an argument between
AI systems and left the obvious question - *would the operational dynamical model have
seen this event either?* - unanswered. `aires/cfs.py` answers it with the same reduction:

    A_L^CFS = aires.aindex.a_index(CFSv2 forecast, peak, metric, the event's own box)

NCEP's operational CFSv2 9-month run, initialised at the AI+RES init, verified over the
identical 7-day window ending at the peak. Four members, from the four 6-hourly cycles
**ENDING** at the init (`init-18h ... init`), so leads are 21.0-21.75 d: the 00Z member is
exactly lead-matched to the walkers and no member is at a shorter lead than the thing it
is a baseline for.

**All nine AI+RES events now carry it** (extended 2026-08-26; Elliott's AI+RES column is
filled in and Uri + the three p90 rungs are new).

| event | CFS mean A_L | CFS best member | AI+RES mean | observed | CFS reaching | AI+RES reaching |
|---|---|---|---|---|---|---|
| PNW_HeatDome_2021 | +0.30 | +3.42 | +8.24 | +7.72 | 0/4 | 42/64 |
| SCentral_HeatDome_2023 | -0.14 | +1.19 | +4.10 | +5.73 | 0/4 | 6/64 |
| Southwest_HeatWave_2020 | +1.05 | +1.60 | +2.47 | +5.14 | 0/4 | 0/64 |
| California_HeatWave_2022 | +2.41 | +3.68 | +4.11 | +4.71 | 0/4 | 25/64 |
| WinterStorm_Elliott_2022 | -2.81 | -10.74 | -11.86 | -14.42 | 0/4 | 20/64 |
| WinterStorm_Uri_2021 | +0.54 | -3.86 | -6.78 | -7.40 | 0/4 | 32/64 |
| p90_20231107 | +0.99 | +1.75 | +3.23 | +0.21 | **3/4** | 64/64 |
| p90_20240802 | +1.05 | +1.35 | +1.93 | +1.06 | **2/4** | 63/64 |
| p90_20251224 | -1.35 | +0.22 | +3.83 | +3.26 | 0/4 | 46/64 |

"Reaching" is sign-aware, so for the two cold events it means far enough BELOW. The three
p90 rungs are the useful new information: they are the only cases where CFS reaches the
observation at all, and they are exactly the mild ones (observed A_L +0.21 and +1.06 K
over CONUS). That is the severity ladder working as intended from the other end - the
operational baseline is competitive precisely where the event is not extreme, and
p90_20240802 is the one case where CFS is effectively perfect (+1.05 vs +1.06 observed).
p90_20251224 breaks the pattern: a +3.26 K CONUS-mean event that CFS misses by 4.6 K,
with the ensemble mean on the WRONG SIDE of climatology (-1.35 K).

Read that table with the box in mind - each row's `A_L` is over that event's own box (see
`aires/aindex.py::EVENT_BOXES`), so the columns compare methods within a row, not events
across rows. **CFS at 21 d lead is nowhere near any of these events**: 0/4 members reach
the observed value on any of the four with a run, and on three of four the ensemble mean
is within ~1 K of climatology. It is a *lower* bar than GenCast direct sampling on PNW
(+0.30 vs +2.46 K), and a comparable one on California (+2.41 vs +1.37 K).

The spread between four cycles 6 h apart is the striking part - PNW runs -2.03 to +3.42 K,
Elliott -10.74 to +3.93 K. That IS the lagged ensemble's own spread at S2S lead, and it is
the same "past the predictability horizon you are measuring chaos" fact Gate 2's noise
floor made concrete. Do not read a single cycle as "what CFS said".

**On the figure.** The four members are drawn as black dashed running-index curves on the
main panel, and `A_L` as a black bar spanning exactly the shaded 7-day window at exactly
the ensemble-mean height (a curve's ENDPOINT is not `A_L`, and drawing the number anywhere
else invites that misreading). The marginal panel gains a `CFSv2 operational` column with
the same sign-aware `k/n` count as the other methods.

### Is the comparison fair? CFS is on a ~0.94 deg grid (measured 2026-08-26)

The obvious objection to every row of that table: CFS runs on a T126 Gaussian grid
(190x384, ~0.94 deg) and is bilinearly interpolated up to the 0.25 deg CONUS grid before
`aindex` scores it. Does the coarse grid damp the event and make `A_L^CFS` too small?

**Measured, on the truth, where the answer is known.** Take each event's ERA5 verification
field at 0.25 deg, push it through CFS's own grid (0.25 -> T126 -> 0.25, both steps
bilinear, the identical path `grib_to_cube` puts CFS through) and re-score it:

| event | ERA5 at 0.25 deg | via the CFS grid | delta | native cells | gap to CFS |
|---|---|---|---|---|---|
| PNW_HeatDome_2021 | +7.716 | +7.707 | -0.010 | +7.837 | +7.42 |
| SCentral_HeatDome_2023 | +5.730 | +5.689 | -0.041 | +5.676 | +5.87 |
| Southwest_HeatWave_2020 | +5.142 | +5.099 | -0.044 | +5.173 | +4.09 |
| California_HeatWave_2022 | +4.710 | +4.717 | +0.007 | +4.692 | +2.30 |
| WinterStorm_Elliott_2022 | -14.422 | -14.364 | +0.058 | -14.589 | -11.61 |
| WinterStorm_Uri_2021 | -7.397 | -7.409 | -0.012 | -7.478 | -7.93 |
| p90_20231107 | +0.211 | +0.228 | +0.017 | +0.219 | -0.78 |
| p90_20240802 | +1.057 | +1.039 | -0.018 | +1.046 | +0.01 |
| p90_20251224 | +3.259 | +3.393 | +0.134 | +3.255 | +4.61 |

Round-trip cost: mean **+0.010 K**, max **|0.134| K**, sd 0.056 K. The "native cells"
column is the other estimator (cos(lat) mean over the CFS cells whose centres fall in the
box, no regridding at all); it differs from the 0.25 deg mean by at most 0.167 K. So the
**largest resolution term anywhere is ~0.17 K**, against gaps of 2.3-11.6 K on the six
genuinely extreme events. Resolution does not explain those misses; it is 1-3% of them.

**Why it is so small, and when it would not be.** `A_L` is a 7-day mean over a 6x6 degree
box - the reduction is itself a brutal spatial smoother, and coarsening a field that is
about to be averaged over ~36 deg^2 barely moves the average. Resolution WOULD matter, a
lot, for a point maximum, a spatial percentile or a precipitation extreme. It does not
matter for this index, and that is a property of the index, not a general fact about CFS.

**Where the caveat still bites: the two mild rungs.** `p90_20240802`'s gap is +0.01 K, so
the ~0.02-0.05 K resolution term is the SAME ORDER as the thing being measured. Do not
quote "CFS was exactly right on p90_20240802" as a precise result - it is right to within
a discretisation uncertainty comparable to its own error. The extreme rows are safe; the
mild ones carry a ~0.1 K error bar that the extreme ones can ignore.

**What this test does NOT cover**, and must not be read as covering: the *climatology*
mismatch (below). Coarse-grid representation is fair; anomalies taken against the wrong
climatology are the open term. And a coarse MODEL genuinely failing to build a sharp ridge
is not unfairness at all - that is forecast skill, which is the thing being measured.

Cross-check on the climatology term: CFS's day-0/1 box anomaly minus the ERA5 init frames'
is within +-0.56 K on eight of nine events (mean |.| 0.30 K excluding Elliott), consistent
with the 0.65 K matched-time PNW figure below. Elliott reads -4.27 K, which is NOT a bias:
`init_anom` covers init+6h..+24h while the ERA5 init frames are at init-12h and init, and
a front crossed the N Plains box in that half-day gap. The windows do not overlap, so this
proxy is only good where the pattern is slow - exactly the caveat the `init_anom` docstring
already makes.

### Things to know before quoting a number

- **The anomaly is against the ERA5 1990-2019 climatology, not CFSv2's own reforecast
  climatology.** So `A_L^CFS` carries CFS's model drift and representativity offset. This
  is the *consistent* choice - the walkers and the FCN3 cubes get the identical treatment
  - but it is NOT a bias-corrected forecast. Measured, not assumed: at 12 h into the run,
  before anything has drifted, CFS's PNW box-mean T2m is **0.65 K** below a GenCast
  walker's (0.28 K over CONUS), on a ~0.94 deg Gaussian grid against 0.25 deg in
  mountainous terrain. Small against the +7.7 K the event registers; not zero, so do not
  argue a 1 K-level result from this cube alone. Same cause, smaller term: a 6 deg box is
  only ~6.4 CFS cells across, so the box mean is resolution-dependent - regridding to
  0.25 deg and averaging there differs from a cos(lat) mean over the native cells whose
  centres fall in the box by -0.09 K over the PNW forecast (-0.26 K over the verification
  window). That spread is the discretisation uncertainty on `A_L^CFS`.
- **`init_anom` is weather, not bias.** The printed day-0/1 box anomaly is where the
  forecast starts. PNW's is -4.98 K because the Pacific Northwest genuinely was ~5 K below
  climatology three weeks before the dome; the ERA5-initialised walkers start at -3.95 K
  over the same box. The figure draws both, so a real offset would be visible.
- **The archive, and its two traps.** NCEI's object store
  (`https://www.ncei.noaa.gov/oa/prod-cfs-operational-forecast`, `time-series/` product),
  one ~95 MB GRIB2 per cycle per variable. Each ships a wgrib `.inv` giving per-record
  byte offsets and the store honours range requests, so a 21-day window is a single ~7 MB
  prefix read. (1) The inventory URL **replaces** `.grb2` with `.inv` - appending gives a
  clean 404 on a cycle that is perfectly present. (2) The `.inv` is not always shipped
  (`tmp2m` at 2020-07-25 06Z has the GRIB and no inventory); the module falls back to the
  whole file. A genuinely absent cycle is skipped with a warning and recorded in the cube
  attrs, not silently dropped.
- **NCEI is incomplete, and the AWS mirror is the fallback** (amended 2026-08-26; the
  earlier note here said the mirror was useless, which was true of the first five events
  and false of the experiment). Measured: NCEI 404s **every cycle of 2024 and of December
  2025**, which is every cycle `p90_20240802` and `p90_20251224` need - both read as
  "ABSENT from the archive" and got no baseline at all. NOAA's Open-Data mirror
  `noaa-cfs-pds` carries them. It is a rolling archive starting 2023-04-22, so it cannot
  serve the four oldest events - but NCEI can, and between them every event has a source.
  `aires/cfs.py` now tries NCEI first and falls through per cycle; the cube records which
  archive served it in the `archives` attr.
- **The two archives are the same bytes where they overlap** - verified, not assumed, on
  2023-10-16 06Z `tmp2m`: identical `Content-Length` (91,845,361 B), identical md5 over
  the 6.9 MB window prefix the fetch actually reads, and byte-identical inventories. The
  fallback is a second ADDRESS for one product, not a second data source.
- **The two use OPPOSITE inventory conventions.** NCEI REPLACES `.grb2` with `.inv`; AWS
  APPENDS `.idx` to the whole name. Apply either convention to the other archive and you
  get a clean 404 against an object that is present - the same trap the `.inv` note
  above records, one archive over. Both are wgrib inventories that `parse_inv` reads
  unchanged, and the layout differs too (`cfs.<YYYYMMDD>/<HH>/time_grib_01/` vs NCEI's
  four nested date levels).
- **Precip metrics are refused, not guessed.** CFS ships `prate`, a 6 h mean RATE; the
  12 h accumulation `xres.xmetrics` wants is a conversion with nothing here to validate
  it against. `t2m_anom` (`tmp2m`) and `u850_speed*` (`wnd850`) are supported.
- **`cfgrib` was pip-installed into `moe`** (with `eccodes`/`eccodeslib`/`eckitlib`/
  `findlibs`; additive, no numpy/xarray upgrade). `aires/aplots.py` imports it lazily and
  drops the CFS curve with a message if it is unavailable, so the figure stage keeps
  working without it.
- **The 6-hourly diurnal filter is not `aplots.daily`.** CFS is 6-hourly where the walkers
  are 12-hourly, so the 24 h trapezoid is the 5-point `(0.5,1,1,1,0.5)/4`, not the 3-point
  `(1,2,1)/4` - which at a 6 h step spans 12 h and leaves most of the diurnal swing on the
  curve. The end padding generalises `daily`'s `y[0]+y[1]-y[2]` to `y[-j] = y[P-j] -
  (y[P]-y[0])` (one 24 h period back, shifted by one period of trend): simultaneously
  exact on a linear trend and an exact periodic extension, so the LAST point - the one on
  the event peak where `A_L` is read - is neither pulled down its own trend nor left
  carrying a quarter of the diurnal amplitude. `test_cfs.py` pins that it reduces to
  `aplots.daily` exactly at a 12 h step.

```bash
# login node (internet, no GPU), moe env, repo root. ~10 s and ~27 MB per event.
PYTHONPATH=. python -m aires.cfs --all
PYTHONPATH=. python -m aires.cfs --all --summary-only     # no network; A_L from cache
PYTHONPATH=. python -m aires.aplots --event PNW_HeatDome_2021 --tag pilot --only trajectory
```

`aires/aplots.py` only READS the cache - the figure stage stays offline, and a missing
cube drops the CFS curve with a message telling you how to build it rather than failing.

## The pilot's probabilities as binned PDFs (job 1174) - the weighted tail reaches ERA5's

**Reweighted by its DMC importance weights, the pilot's population places tail mass where
ERA5 put it; direct sampling does not.** `aires/apdfs.py` (new) rebuilds the probability
distribution as house-style binned PDFs (the `xres/xcombined.py::PDF_histogram` recipe:
100 shared bins, probability per bin, log y) of the verification-week T2m anomaly at every
grid point, pooled over the ensemble, each walker's 105x237 (or 25x25 in-box) points
carrying its weight `exp(-V_K)`. The plotted curve is **self-normalized**
(`p_b = sum_i w_i n_ib / (G sum_i w_i)`, total mass 1, bin-for-bin comparable to ERA5 and
the direct ensembles); the Z-scaled estimator differs by exactly the constant 0.5684 - the
total-mass check above, a uniform 0.245-decade shift - and is stated on the figures rather
than applied.

| P(grid point >= +9 K) | PNW box | CONUS |
|---|---|---|
| ERA5 truth | 0.0848 | 0.0032 |
| **AI+RES weighted** | **0.0505** (0.60x ERA5) | **0.0026** (0.82x) |
| AI+RES resampled population, unweighted | 0.391 (4.6x over) | 0.0154 (4.8x) |
| GenCast direct (xres, 24) | 0.0011 (~80x short) | 0.0001 |
| GenCast direct (Gate 3 walkers, 16) | 0.0176 | 0.0008 |
| FCN3 direct (24) | 0.0079 | 0.0003 |

The weighted curve carries mass out to +14 K in the box; every direct ensemble is done by
~+10 K. (An early draft quoted the direct shortfall as ~46x; the measured value is ~80x.)

Read the box PDF's **left flank** with the same discipline as the exceedance curve's flat
segment: weight ESS is 5.68 of 64 (Kish), the three heaviest walkers carry 60% of the
mass, and the heaviest (w07, w = 34.46, 37%) is the population's *coldest* member
(A_L = +3.54 K), so the weighted mode (+4.8 K vs ERA5's +8.4 K) is a small-sample
artifact. The right tail pools many small-weight walkers and is the part to trust; the
figures carry both caveats on their face.

Mechanics, verified by an independent pass (16 gates, fresh process, nothing trusted from
the build's own printout): per-walker fields are stitched through the genealogy
(`compare.json` `realized.lineage`, cross-checked against `run_aires.slot_lineage`;
only segments 5/6/7 cover the window, only `2m_temperature` is read),
`check_window == 13` is hard-asserted per walker, and the recomputed per-walker `A_L`
reproduce `compare.json` **bit-exactly** (all 64, box and CONUS) - as do the three direct
baselines vs `compare.json["ds"]` and the weighted exceedance at all 41 curve thresholds
(max diff 2.8e-17). One number in older prose corrected on the way: the observed CONUS
`A_L` is **+0.64 K** (cos-lat weighted, `ds_baseline.json` `observed.conus`); +0.72 is the
unweighted plain mean.

Artifacts: `runs/aires/PNW_HeatDome_2021/res/pilot/pdf/pdf_fields.nc` (13 MB, all fields +
weights + attrs; gitignored, `--stage build` rebuilds it in ~45 s),
`figures/aires/PNW_HeatDome_2021/aires_pdf_{conus,box}_PNW_HeatDome_2021_pilot.png`,
`figures/aires/PNW_HeatDome_2021/aires_anom_maps_PNW_HeatDome_2021_pilot.png`. Entry:
`PYTHONPATH=. python -m aires.apdfs --stage all --event PNW_HeatDome_2021 --tag pilot`
(login node or `sbatch slurm/aires_pdfs.slurm`); figures redraw in seconds from the
intermediate. 19 unit tests in `aires/tests/test_apdfs.py`.

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

**Survey the cards before booking nodes; `sinfo` cannot see this.** Slurm reported all
eight a3mega nodes idle on 2026-09-03 while FIVE of them were unusable: teammates'
non-Slurm processes held 63-81 GB on every card of nodes 0, 3, 4 and 5 (`inkling` probes,
`continuum/`) and two cards of node 7 (vLLM workers, plus whisper/pyannote servers on the
rest). The ensemble smoke (job 1195) landed on node 4 and died in 14 s with JAX unable to
initialise CUDA. The nodes accept SSH from the login node, so the check is one loop:

```bash
for n in 0 1 2 3 4 5 6 7; do h=nucla3m-a3meganodeset-$n; printf "%s: " $h
  ssh -o BatchMode=yes -o ConnectTimeout=5 $h 'nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr "\n" " "; echo'; done
# then: sbatch --exclude=nucla3m-a3meganodeset-[0,3,4,5,7] ...   (or --nodelist=... for one node)
```

A card is usable for the 0.25 deg walker only if it has ~70 GB free (the checkpoint takes
~64 GB). Never kill what you find - it is not yours - route around it with `--exclude`.

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
| `aires/aplots.py` | the Phase 4 figures: trajectory, exceedance, diagnostics, genealogy, composite |
| `aires/cfs.py` | the CFSv2 operational baseline: range-fetches NCEI's archive, regrids to the 0.25 deg CONUS grid, reports `A_L` at the 21 d lead. `--event NAME` / `--all`. Needs internet, no GPU |
| `aires/apdfs.py` | the pilot's PDFs: `--stage {build,figs,all}` - lineage-stitched walker fields, weighted binned PDFs (CONUS + box), the 3-panel map |
| `aires/awalkers.py` | the population figure: every surviving walker's `A_L` and its mass `p_i = Z w_i/N`, per run and across the slate, + `runs/aires/aires_walkers.csv` |
| `astab/` | the lead-time stability sweep, **its own package and its own tree since 2026-08-27** (it was `aires/astab.py` writing into the `stab` tag of `runs/aires/<event>/res/`). Driver `python -m astab.run --stage {plan,check,refs,calibrate,walk,score,reduce,figs,maps,prune}`. See `astab/__init__.py` |
| `astab/sconfig.py` | the only place the tree is decided; `AIRES_STAB_RUNS`/`AIRES_STAB_FIGS` move either root |
| `astab/schedule.py` | `chain_schedule` is pure arithmetic on `horizon % 3` - no per-chain tail table |
| `astab/refs.py` | ERA5: the scalar series AND the CONUS T2m truth fields. `_wb2_frame` reads WeatherBench-2 through raw zarr because `data.open_anon` does not fit in this login node's 15 GB |
| `astab/maps.py` | the lead-by-lead T2m maps: forecast / ERA5 / difference, one figure per (event, model, init). Fixed scales - +-12 K difference everywhere, one absolute scale per event - so figures are comparable |
| `astab/tests/` | 101 tests: the tail table, the horizon exclusion, the prune rule, the job-1155 signature on a synthetic cube, the measured bounds, that a chain reads its OWN week's init frames, and that the maps refuse a grid mismatch rather than aligning it |
| `slurm/aires_stab.slurm` | the sweep: walk (moe) + score free/fed (fcn3) on ONE allocation, 8 GPUs, retried; `AIRES_STAB_MAX_SEGMENTS` is the smoke path |
| `runs/astab/refs/` | `bands.json` (measured operating ranges), `gate3_panel.csv` (the 224 samples behind them), `<event>_era5_ref.nc` (scalar series), `<event>_era5_t2m.nc` (the truth fields) |
| `runs/astab/stability.csv`, `figures/astab/stability.png` | the sweep's long table and its four panels |
| `figures/astab/maps/<event>/<model>_wk<NN>.png` | 24 figures: 2 events x 3 models x 4 leads |
| `aires/run_aires.py` | the production driver: `--stage {prep,walk,score,res,ds,compare}`. `res` IS the coordinator and launches both GPU pools itself |
| `aires/tests/test_adapter.py` | 25 tests incl. the 69-channel bit-exactness check |
| `aires/tests/test_walker.py` | state contract, batch-from-arbitrary-state, cloning RNG, checkpoint choice |
| `aires/tests/test_dmc.py` | pivotal sampling, the C=0 identity, OU unbiasedness and variance reduction |
| `aires/tests/test_ctune.py` | the surrogate reproduces the skill curve and the correlations it was not given |
| `aires/tests/test_aplots.py` | sibling pairing, the exceedance standard error, weighted quantiles, and that each figure renders |
| `aires/tests/test_awalkers.py` | that `p_i` is the estimator's own (imported, not re-derived), sums to the normalization check, ranks by the tail sign, and flags a pinned target |
| `aires/tests/test_run_aires.py` | leg schedule, lineage validation, the replay guard, pruning, the cold-event sign; runs the real DMC loop through the real coordinator on a fake tree |
| `aires/tests/test_cfs.py` | CFS URL/inventory contracts, the trailing lag ensemble, and the 6-hourly 24 h filter (incl. that it reduces to `aplots.daily` at a 12 h step) |
| `aires/tests/test_apdfs.py` | weighted_pdf identities: weights=None == `xcombined.PDF_histogram`, ones == None, the Z-scaling constant, mass 1 |
| `slurm/aires_gate2.slurm` | Gate 2 infer, 8 shards on one a3mega node, `--job-name=Vayuh-s2s` |
| `slurm/aires_gate3.slurm` | Gate 3 walk (moe) + score (fcn3) on ONE allocation, 8 GPUs, retried |
| `slurm/aires_res.slurm` | the pilot: ONE process (the coordinator), 12 h, `--job-name=Vayuh-s2s` |
| `slurm/aires_env.sh` | the ONE place either conda env is activated; deactivates first so `moe` cannot leak into `fcn3` |
| `slurm/aires_pdfs.slurm` | the PDF rebuild on a `debug` CPU node (45 s), `--job-name=Vayuh-s2s` |
| `runs/aires/calib/derived_calib.nc` | fitted calibration (gitignored, rebuildable in ~2 min) |
| `runs/aires/gate2/` | adapter IC, 24-member adapter cube, scores JSON/CSV |
| `figures/aires/PNW_HeatDome_2021/gate2_PNW_HeatDome_2021.png` | the three Gate 2 panels |
| `figures/aires/PNW_HeatDome_2021/gate3_PNW_HeatDome_2021.png` | the four Gate 3 panels |
| `figures/aires/PNW_HeatDome_2021/ctune_PNW_HeatDome_2021.png` | the C_k schedule replay |
| `runs/aires/<event>/walkers/`, `/scores/`, `/gate3/` | 16 trajectories, 80 score cubes, the verdict |
| `runs/aires/<event>/res/<tag>/` | the pilot's own tree - walkers, scores, per-leg manifests, `res_result.json`. DISJOINT from the gate's |
| `figures/aires/aires_walkers.png`, `runs/aires/aires_walkers.csv` | all 64 walkers per run: anomaly, `p_i`, rank, running total, founder, clone family |
| `runs/aires/<event>/res/<tag>/pdf/pdf_fields.nc` | the PDF intermediate: 64 walker fields + weights + all baselines (gitignored, ~45 s rebuild) |
| `figures/aires/PNW_HeatDome_2021/aires_pdf_*_PNW_HeatDome_2021_pilot.png`, `aires_anom_maps_*` | the two binned PDFs (CONUS, box) + the three-map comparison |
| `runs/aires/<event>/ds_baseline.json` | the direct-sampling baseline (no GPU, tag-independent) |
| `runs/aires/<event>/cfs/` | the CFSv2 baseline cube + its `A_L` JSON (tag-independent; ~17 MB/event, gitignored, ~10 s rebuild) |
| `docs/aires_gate2.html` | published Gate 2 summary (artifact source) |
| `docs/aires_gate3.html` | published Gate 3 summary (artifact source) |

Everything `aires.md` names is written, including the pilot's own output (job 1172:
`compare.json` and the four figures drawn from it) and the PDF rebuild on top of it
(job 1174: `aires/apdfs.py` and its three figures).

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
