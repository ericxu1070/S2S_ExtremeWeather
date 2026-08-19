# AI+RES with GenCast walkers and FCN3 scoring

## Context

Lancelin et al., *AI-Boosted Rare Event Sampling to Characterize Extreme Weather*
(arXiv:2510.27066v3, PRL 2026) introduce **AI+RES**: a Diffusion Monte Carlo (DMC)
importance-splitting algorithm where a physics GCM (PlaSim) propagates N=400 "walkers"
and a cheap AI emulator supplies the **score function** - at each resampling time each
walker is scored by the ensemble mean of an M=100-member emulator forecast of the target
observable. Walkers with high scores are cloned, low-score walkers are killed, and a
weight bookkeeping scheme keeps probability estimates unbiased. They report accurate
return periods out to 50,000 years with only 400 walkers - a ~100x speedup - and validate
against direct sampling (DS) with N=50,000.

We want the same machinery with **GenCast as the walker** and **FourCastNet3 as the
scorer**, applied to the CONUS extreme events this repo already studies at week-3 lead.

Two things make this a genuine contribution rather than a port:

1. **GenCast is stochastic.** The paper's walker (PlaSim) is deterministic, so cloned
   walkers are identical and must be split apart by hand-tuned perturbations to the
   spherical harmonics of log surface pressure (amplitude 3e-3). GenCast clones diverge
   naturally from fresh diffusion noise. The paper's own Discussion names a probabilistic
   emulator as the key next step and cites GenCast (ref. [30]) as the example.
2. **The paper's Discussion explicitly names S2S forecast tails** as a target
   application: *"The framework could also be well suited to characterize the tails of
   distributions in subseasonal to seasonal ensemble forecasts, where large ensembles are
   needed."* That is exactly this repo's problem.

**Honest statement of what changes by making the walker an AI model.** In the paper the
walker is the high-fidelity reference and the AI is the cheap helper. Inverting that means
the sampled distribution is GenCast's attractor, not the atmosphere's - the paper's own
AI-DS baseline showed AI emulators have biased tails. So the claim this experiment can
support is: *RES efficiently reaches the extreme tail of the GenCast forecast
distribution, recovering observed extremes that a 24-member ensemble misses.* It cannot
support an unbiasedness claim without a large same-model DS reference, which was
deliberately descoped (see "Descoped, with cost, if you change your mind").

## Agreed configuration

| Decision | Value |
|---|---|
| Machine | a3mega Slurm H100 (FCN3 needs ~50 GiB; Derecho A100-40GB OOMs - confirmed, job 6857923) |
| Walker | GenCast 0.25° `<2019` checkpoint, 12 h step, `XRES_SERIAL_INFER=1`, bf16 |
| Scorer | FourCastNet3 0.25°, 6 h step, via `earth2studio.run.ensemble` + `Zero()` perturbation |
| Observable | 7-day-mean T2m anomaly, cos-lat-weighted, over an **event-centered box** (CONUS mean stored as a free secondary) |
| N walkers | 64 |
| M score members | 6 |
| K resampling steps | 5, tau = 3 days |
| C_k schedule | (0, 1.0, 1.4, 1.8, 2.0) - tunable, first step free-running |
| Pilot event | `PNW_HeatDome_2021` (peak 2021-06-28, init 2021-06-07, 21-day lead) |
| Baselines | existing 24-member GenCast + FCN3 ensembles, ERA5 observed value |
| Stage B | seasonal-scale return periods, gated on a GenCast drift diagnostic |

### Pilot timeline (PNW_HeatDome_2021)

```
init 2021-06-07 00Z  (+ history frame 2021-06-06 12Z; GenCast needs 2 frames 12 h apart)
horizon 42 x 12 h steps -> 2021-06-28 00Z
t_f  = lead 15 d = 2021-06-22 00Z ;  L = 7 d  -> A_L window = leads 15..21 d
t_k  = leads 3, 6, 9, 12, 15 d  (GenCast steps 6, 12, 18, 24, 30)
score forecast at t_k runs FCN3 from lead t_k to lead 21 d: 72, 60, 48, 36, 24 six-hour steps
```

Because `C_1 = 0`, no score forecasts are needed at k=1 - the weights stay uniform and
nothing is resampled. Scoring cost is therefore `64 x 6 x (60+48+36+24) = 64,512` FCN3
steps ~ 25 H100-h, plus 21 H100-h of walker segments: **~46 H100-h/event, ~6 h wall on one
8xH100 node.**

## The GenCast -> FCN3 adapter (the piece you were unsure about)

Resolved by inspecting `earth2studio/models/px/fcn3.py::VARIABLES`. FCN3 takes **72
channels and exactly one time frame** - the 12 h vs 6 h timestep mismatch is a non-issue,
and `total_precipitation_12hr` / `sea_surface_temperature` are GenCast outputs FCN3 never
consumes.

**65 + 4 channels map 1:1 from GenCast's target state:**

| FCN3 | GenCast | note |
|---|---|---|
| `u{50..1000}`, `v{...}`, `z{...}`, `t{...}`, `q{...}` | `u_component_of_wind`, `v_component_of_wind`, `geopotential`, `temperature`, `specific_humidity` | identical 13 WeatherBench levels (`config.P13`) - direct rename |
| `t2m`, `msl`, `u10m`, `v10m` | `2m_temperature`, `mean_sea_level_pressure`, `10m_u/v_component_of_wind` | exact |

**Only 3 channels must be derived** - FCN3 uses `q` (not relative humidity) and `msl`
(not surface pressure), both of which GenCast carries:

- **`u100m`, `v100m`**: log-law extrapolation from `u10m/v10m` and the 1000/925 hPa wind,
  using `geopotential` to get the level height, with a per-gridpoint calibration fitted
  once against true ERA5 `u100m/v100m` over a month of samples.
- **`tcwv`**: `(1/g)` * pressure-trapezoid of `q` over the 13 levels, times a fitted
  per-gridpoint scale factor absorbing the missing sub-1000 hPa layer.

Grid handling: flip latitude to `linspace(90,-90,721)`, keep `lon = linspace(0,360,1440,
endpoint=False)`. **No regridding at all**, since both models run at 0.25° - this is why
the 0.25° walker was chosen over a 1.0° one.

Injection point needs **no change to the FCN3 driver**: `fcn3/run_fcn3.py::LocalIC` is
already a duck-typed `earth2studio` `DataSource` (`__call__(time, variable) -> DataArray`).
The adapter emits the same object from a GenCast state instead of a cached NetCDF.

## Blocking constraints found during exploration

1. **Two conda envs.** GenCast is JAX (`moe` on a3mega / `my-env` on Derecho), FCN3 is
   PyTorch/earth2studio (`fcn3`). They cannot share a process, so walker and scorer
   workers must hand off through disk. Reuse the two-env orchestration idiom already in
   `fcn3/run_test.sh`.
2. **Cached GenCast cubes are CONUS-cropped** (`xres/xinference.py`, `.sel(lat=C.LAT,
   lon=C.LON)` before host transfer). You cannot restart a walker or build an FCN3 IC from
   them. AI+RES must write **global** 2-frame checkpoints. Design that avoids the original
   OOM the crop was protecting against: stream global predictions, keep only a rolling
   2-frame buffer in RAM, write the CONUS crop per step to a diagnostics cube, and write
   the global 2-frame state to disk **only at resampling times**.
3. **Storage**: a global 2-frame 0.25° state is ~690 MB float32 (83 fields x 2 frames x
   1.04M points). 64 walkers x 5 steps = **~230 GB/event** if all steps are retained for
   resumability. Halve it with float16 for archived (non-restart) states, or retain only
   the two most recent steps.
4. **Host RAM**: one serial 0.25° GenCast process peaks at ~237 GiB. 8 concurrent walker
   workers would need ~1.9 TB. The a3mega 8-worker pool scripts (`slurm/xres_pool_0p25_
   week*.slurm`) already do this, so it is presumably fine there - **confirm before scaling**,
   otherwise run 4 walker workers + 4 FCN3 workers.
5. **`C_k` tuning is empirical** in the paper. Tune it with the cheap persistence score
   before spending FCN3 hours (see Phase 2 gate).

## Working agreement

- New branch **`aires`**, cut from `fcn3-6event-week3`. That branch's experiment (FCN3 vs
  GenCast, 6 events, week-3) is already complete on the a3mega H100 cluster, so it stays
  closed; all AI+RES work lands on `aires`.
- **One phase per session.** Each phase below is a stopping point - nothing from Phase N+1
  starts until Phase N is verified. Phase 0 is the whole of the current step.

## Plan

### Phase 0 - Move the a3mega FCN3 week-3 results to Derecho (~1 h, mostly transfer)

Flow: write the sync script here on Derecho, commit and push to `aires`, then pull it on
a3mega and run it there (rsync outbound over SSH from the H100 box, the established
route). Destination host is parameterized (`DEST_HOST`) since only the a3mega side knows
which Derecho hostname it can reach.

Nothing here is Derecho-resident today: `runs/fcn3/week3/cache/` is empty, only 1 of 6 ICs
exists, and the 3 p90 events have no GenCast cubes or truth. The 16 comparison figures and
all logs *are* committed (`runs/` is gitignored, `figures/`+`logs/` are not).

Write `scripts/sync_a3mega_to_derecho.sh`, **to be run from an a3mega login node** (the
usual route: rsync outbound while SSHed into the H100 box), with an explicit manifest and
post-transfer file-count assertions:

| Path | Expect |
|---|---|
| `runs/fcn3/week3/cache/*_cube.nc` | 6 files (24 x 85 x 105 x 237) |
| `runs/fcn3/week3/ic/*_ic.nc` | 6 files, ~300 MB each |
| `runs/fcn3/week3/{scores,timing}/` | incl. `fcn3_vs_gencast_scores.csv` |
| `runs/xres/0p25/week3/cache/p90_*_cube.nc` | 3 files, ~5.8 GB each |
| `runs/observations/p90_*` | truth for the 3 p90 events |
| `runs/p90_t2m/fcn3/**` | optional, the separate 90-case experiment |

~20-25 GB total. Then run `python fcn3/compare_fcn3_gencast.py` on Derecho to confirm the
figures reproduce from transferred cubes, and fix the stale machine framing in
`HANDOFF.md` / `CLAUDE.md` (both currently describe Derecho as unreachable, read from a
Derecho checkout).

### Phase 1 - Adapter + three validation gates (~1 day, ~4 H100-h)

Create `aires/` as a new stack that imports from `gencast_s2s`, `xres`, and `fcn3` without
modifying them (mirroring how `xres/` builds on `gencast_s2s/`).

`aires/adapter.py` - the GenCast-state -> FCN3 72-channel `DataSource` described above,
plus `aires/calibrate.py` to fit the `u100m/v100m/tcwv` corrections from ERA5 via
`gencast_s2s/data.py::arco_read`.

Three gates, each with a numeric pass criterion. **Do not build the DMC loop until gate 3
passes.**

- **Gate 1, channel fidelity.** Derive the 3 channels from ERA5's GenCast-variable subset
  and compare against true ERA5. Pass: `u100m` RMSE < 1.0 m/s and |bias| < 0.2 m/s;
  `tcwv` relative RMSE < 5%, |bias| < 2%.
- **Gate 2, forecast equivalence.** Run FCN3 with matched seeds from (a) the existing
  native IC `runs/fcn3/week3/ic/PNW_HeatDome_2021_ic.nc` and (b) an adapter-built IC from
  the same ERA5 fields. Pass, all three:

      G2a  |delta mean A_L| < 0.25 K
      G2b  paired-member Spearman rank correlation > 0.95 out to at least 7 d lead
      G2c  d_paired(t) <= d_internal(t) at EVERY lead, where
             d_paired(t)   = RMS over members of (adapter_i(t) - native_i(t))
             d_internal(t) = sqrt(2) * ensemble sd of the native run   (FCN3's own noise)

  **M must be >= 24, not 6.** The per-member sd of `A_L` is ~0.62 K for this event, so two
  6-member ensemble means differ by 0.36 K from sampling noise alone - larger than G2a's
  own threshold, i.e. a *perfect* adapter would fail at M=6 about half the time. At M=24
  the paired s.e. is 0.149 K and G2a resolves differences down to ~0.30 K and no finer;
  the score stage prints that limit with the verdict.

  **G2b is a horizon, not an endpoint** (amended 2026-08-18 after the first run). Read at
  the 21 d verification window only, it cannot discriminate: any IC difference whatsoever -
  the adapter's or a plain re-seed - has by then grown to the internal-noise level and
  shuffled the members into an unrelated order, so a perfect adapter also scores ~0 there.
  Evaluate it as a function of lead and require rank-equivalence out to >= 7 d. G2c is the
  well-posed statement of "forecast equivalence" and holds at every lead: if the adapter
  moves the forecast less than re-seeding the model does, it is indistinguishable from
  another draw of the same ensemble.

  **Result (job 1153, PNW_HeatDome_2021, M=24, 21 d): PASS.**
  G2a 0.083 K; G2b rho_s > 0.95 out to 8.5 d (0.996 at 1.5 d, 0.957 at 7.5 d, and the decay
  past 8.5 d is FCN3's predictability horizon for CONUS-mean T2m, not adapter error);
  G2c max ratio 0.990, reached only at day-21 saturation (0.222 at 1.5 d, 0.409 at 7.5 d,
  0.593 at 10 d). Re-scored on Derecho from the transferred cubes: identical to the digit.
- **Gate 3, score skill - the real go/no-go.** Roll 8-16 GenCast 0.25° members *with global
  2-frame checkpoints* at leads 3/6/9/12/15 d (~3-5 H100-h), score each with FCN3, and
  correlate `theta(t_k)` against the realized `A_L(t_f)`. Pass: Spearman rho >= 0.5 at
  t_k = 9 d and 12 d. If it only passes at 12 d, shift the `C_k` schedule later. If it
  fails everywhere, fall back to GenCast-as-its-own-scorer (the paper's PFS+RES upper
  bound) and report the FCN3 scoring failure as a finding.

  **The ground truth is the walker's own outcome, not ERA5.** The score does not have to
  be right about the atmosphere; it has to be right about the walker. That is all
  resampling reads, and it is the only thing a score function can fairly be held to.

  **The observable is the per-event box, not the CONUS mean** (amended 2026-08-19, when
  Gate 3 forced the Phase 3 box decision - see below). For the pilot the difference is
  not cosmetic: the observed 7-day anomaly is **+7.72 K over the PNW box and +0.64 K over
  CONUS**, so a CONUS-mean gate would be scoring a signal the event barely has. The CONUS
  index is computed and stored as the secondary, which is also what keeps Gate 2's
  published numbers comparable.

  **Three things are reported beside the correlation**, because it cannot be interpreted
  without them and Gate 2 already taught this lesson once:

    * the **persistence baseline** - the paper's Standard-RES score, the observable's
      current value. It is free (the walkers' own diagnostics cubes hold it) and it is
      the bar FCN3 has to clear to be worth its GPU hours. If persistence ranks as well,
      that is a finding, not a failure;
    * the **noise ceiling** - `theta` is a mean of M members, so its own sampling error
      caps the correlation it can reach with any target at `sqrt(1 - var_noise/var_theta)`,
      no matter how skilful the score is. This is the M=6 lesson from Gate 2 applied to a
      correlation instead of a difference;
    * the **bootstrap CI on rho** - at N=16 walkers the 5% one-sided significance level is
      rho ~ 0.43, so the 0.5 threshold means "significantly better than chance" and little
      more. Everything is cache-aware per (walker, segment) and per (walker, t_k), so N is
      raised by rerunning with a larger `--walkers`; nothing already computed is redone.

  Score members use **common random numbers across walkers** at a given `t_k` (the seed is
  a function of event and lead, not of the walker). Gate 3 measures ranking, so making the
  internal noise common removes it as a source of ranking error - the same argument as
  Gate 2's matched seeds, and what production scoring should do too.

  **Result (job 1160, PNW_HeatDome_2021, N=16 walkers, M=6, PNW box): PASS.**
  rho_s = **0.797 at t_k = 9 d** (p = 2e-4, 95% CI [0.51, 0.92]) and **0.959 at 12 d**
  (p < 1e-4, CI [0.81, 0.99]); also 0.606 at 6 d and 0.988 at 15 d. Three findings that
  change later phases:

  * **The AI scorer is the reason this works.** The persistence baseline - Standard-RES,
    the observable's current value - scores 0.062 at 9 d and 0.129 at 12 d, i.e. nothing.
    FCN3 is not reading off the current state.
  * **Skill switches on between 3 d and 6 d** (rho_s = -0.044 at 3 d, crossing 0.5 at
    ~5.5 d). `C_1 = 0` is therefore correct and the schedule must not be shifted earlier;
    `C_2 = 0` is worth testing in the Phase 2 replay.
  * **M = 6 is enough.** The correlation ceiling imposed by M=6 sampling error is 0.952 at
    9 d and 0.990 at 12 d, so FCN3 runs at 84% and 97% of its own ceiling. Extra members
    buy almost nothing where resampling happens. (It IS binding at 3-6 d, where the score
    has no skill anyway.)

  The 16 walkers are also the **direct-sampling baseline**: realized `A_L` over the PNW box
  is +2.54 +- 3.02 K, range [-3.16, +7.49], against an observed **+7.72 K** that **0 of 16
  reach**. That is the gap Phase 4 asks RES to close. On the CONUS mean the same event is
  unremarkable (13 of 16 walkers exceed the observed +0.64 K), which is the empirical
  argument for the box.

Gate 3's global-checkpoint members are not throwaway - they are the first real exercise of
`aires/walker.py` and seed the DS baseline.

### Phase 2 - RES core and the C_k tuning pass (~1-2 days, ~2 H100-h)

`aires/dmc.py` - the model-agnostic DMC core, structured after the paper's own reusable
module (`AI-RES-public/RES/resampling/`, which the authors state is model-agnostic):

- score standardization across the walker population at each `t_k`
- splitting function `V_k = C_k * theta_hat(X_k)`
- weight update `w_k^i = wbar_{k-1} * exp(V_k(X_k^i) - V_{k-1}(Xhat_{k-1}^i))`
- **pivotal sampling** (Deville-Tille) for the clone/kill counts, `sum N_k^i = N`,
  `E[N_k^i] = w_k^i / wbar_k`
- genealogy bookkeeping (parent id per walker per step)
- the unbiased estimator (paper Eq. 2) reducing to `P(A_L > a)` and exceedance curves

Score backends behind one interface: `fcn3` (production), `persistence` (current index
value - the paper's Standard-RES), `gencast` (PFS-style fallback).

Then a **cheap tuning pass**: replay the DMC loop at N=64 to sanity-check that
`C_k = (0, 1.0, 1.4, 1.8, 2.0)` gives an effective sample size that does not collapse
(target: ESS > N/4 after the final step) and that clone multiplicities stay bounded. Costs
almost nothing and protects the ~46 H100-h production run from a mis-tuned schedule.

**Done (2026-08-19, `aires/ctune.py`), and the schedule is validated as written.** The
replay does better than the persistence proxy this plan assumed: Gate 3 left real FCN3
scores for 16 walkers at all five leads plus each walker's realized outcome, so the
surrogate is calibrated to the **measured** skill curve
`rho_k = (0.00, 0.61, 0.82, 0.95, 1.00)` rather than to a stand-in. (The five scores are
five forecasts of the same outcome, not five steps of a random walk, so the model is
signal-plus-noise with a martingale signal; it predicts the cross-lead correlations it was
not fitted to with a mean error of 0.061 against a sampling error of 0.28.)

| schedule | final ESS/N | p10 | max clone mult | spread/P at 4 sigma |
|---|---|---|---|---|
| **(0, 1.0, 1.4, 1.8, 2.0)** | **0.59** | 0.41 | 8.5 | **5.3** |
| (0, 0, 1.4, 1.8, 2.0) | 0.57 | 0.38 | 11.6 | 14.3 |
| (0, 0.5, 0.8, 1.1, 1.4) | 0.77 | 0.68 | 3.8 | 5.8 |
| (0, 1.5, 2.0, 2.5, 3.0) | 0.28 | 0.10 | 15.4 | 5.6 |
| direct sampling | 1.00 | 1.00 | 1 | 24.6 |

Keep the schedule as written. Two things it settles:

- **`C_2 = 0` is a mistake**, even though Gate 3 found no score skill at 3 d: `C_1` is the
  3 d coefficient and is already zero, while `C_2` acts at **6 d** where the score does
  work (rho = 0.61). Switching it off makes the 4 sigma estimate 2.7x noisier.
- **The schedule cannot be pushed harder.** `(0, 1.5, 2.0, 2.5, 3.0)` is the only candidate
  that fails the ESS target, and it buys nothing in the tail.

**What the pilot can honestly claim.** At N=64, K=5 the variance reduction over direct
sampling is 1.4x at 2 sigma, 2.3x at 3 sigma and 4.6x at 4 sigma - real, growing with depth
as importance splitting should, and nothing like the paper's ~100x, which comes from N=400,
K=6 and a far deeper target. That is consistent with the claim already scoped above ("RES
efficiently reaches the extreme tail of the GenCast forecast distribution"), and it is not
a speedup claim.

### Phase 3 - Walker/score workers and the Slurm driver (~2 days)

- `aires/aconfig.py` - RES hyperparameters, `runs/aires/<event>/...` layout. Reuses the
  event registry in `fcn3/fevents.py` (already has the 6 events, inits, metrics).
  **The per-event boxes are done** (moved forward into Phase 1 because Gate 3's observable
  needs them): they live in `aires/aindex.py::EVENT_BOXES` rather than in the event
  registry, so `fcn3/fevents.py` stays dependency-light and the box travels with the
  reduction that uses it. PNW 44-50N/124-118W, Uri 26-37N/106-94W, Ian 24-30N/84-76W; the
  p90 cases stay on CONUS because `p90/cases_meta.json` DEFINES them as exceedances of the
  cos-lat CONUS mean, so any smaller box would score a different event than the one
  selected.
- `aires/walker.py` - roll one walker segment from a global 2-frame state. Built on
  `gencast_s2s/model.py::load_gencast` (`bundle["forward"]`), a generalized
  `gencast_s2s/inference.py::build_example_batch` that accepts an arbitrary 2-frame state
  instead of ERA5, `graphcast/data_utils.extract_inputs_targets_forcings`, and
  `rollout.chunked_prediction_generator` with `num_steps_per_chunk=1`. Forcings for
  arbitrary future timestamps come free from `data_utils.add_derived_vars` (GenCast's
  forcings are just year/day-progress sin/cos - no TISR). Fresh rng per cloned branch is
  the clone-perturbation mechanism. Adopt the build-once model cache pattern from
  `fcn3/model_cache.py`.
- `aires/score.py` - M FCN3 forecasts from an adapted IC out to lead 21 d. **Written in
  Phase 1 for Gate 3.** Reuses `earth2studio.run.ensemble` with `Zero()` perturbation and
  `model.set_rng`, exactly as `fcn3/run_fcn3.py::stage_infer` does, with `output_coords`
  trimmed to the needed variables. It stops at the cube; the reduction to `theta` needs
  the climatology and so happens on the GenCast side of the two-env handoff.
- `aires/aindex.py` - the scalar observable. **Written in Phase 1 for Gate 3.** Reuses
  `xres/xmetrics.py::forecast_field` (so a walker trajectory, an FCN3 score cube and ERA5
  truth are reduced by identical code) with a cos-lat-weighted area mean over the event
  box, and emits the CONUS mean alongside. `aires/gate2.py` imports its reduction back
  from here. Also carries the running index and the persistence score.
- `aires/run_aires.py --stage {prep,walk,score,res,ds,compare}` - cache-aware per
  (walker, step), matching the repo's existing stage idiom so a killed job resumes.
- `slurm/aires_res.slurm` - one a3mega node, `#SBATCH --job-name=Vayuh-s2s`.

**Done (2026-08-19), and smoke-tested on the hardware** (job 1161, N=8, one resampling,
30.7 min on one node: it rolled clones, resampled on real FCN3 scores, pruned states, and
replayed bit-identically in 2 s on the login node afterwards). `aires/run_aires.py` and
`slurm/aires_res.slurm` exist, plus `slurm/aires_env.sh`, which is the one place either
conda environment is activated. Four things came out differently from the sketch above,
all for reasons that were measured rather than assumed:

- **The two worker pools alternate; there is no work queue.** They cannot coexist: the
  0.25 deg GenCast checkpoint holds ~64 GB of an 80 GB H100 and FCN3 needs most of a card
  too, so a resident walker pool leaves no room for a resident scorer on the same GPU.
  And the queue would buy nothing anyway - the DMC loop is a barrier, so one pool launch
  handles exactly one leg, where every task is the same size and static `w % nshards`
  sharding is already exactly balanced. (`cache/claims/` work-stealing exists in `xres`
  because its events differ in cost. These do not.) The coordinator launches a pool, waits
  for it to exit, and launches the other.
- **Resumption is replay, not a checkpoint.** The coordinator keeps no mutable state. It
  re-runs the whole loop on every invocation and every finished leg returns in seconds,
  which is exact because the loop is a deterministic function of (the pivotal-sampling
  seed, the score cubes on disk). Verified: a second pass reproduced `log Z`, `final_V`
  and every parent map bit-identically in 2 s with no GPU. The parents are re-derived and
  compared against each leg's `walk.json` on every pass, so a rewritten cube or an edited
  schedule is an error rather than a silent fork.
- **The run tree is disjoint from Gate 3's** (`runs/aires/<event>/res/<tag>/`). Both index
  walkers by slot, but only Gate 3's slots are lineages; under resampling a slot is a
  container that different ancestries pass through, so writing this population into the
  gate's paths would overwrite artifacts that are supposed to be free-running. That also
  makes a cached segment impossible to validate on its timestamp alone - two ancestries
  produce states at the same valid time - so every cached segment is checked against the
  `parent` its state records, not just its clock.
- **The horizon leg is split into two standard 3 d segments**, not run as one 12-step leg,
  so a worker still pays exactly one XLA compile.

Two savings worth naming, both checked rather than assumed:

- **Legs 1 and 2 are free.** A segment is a deterministic function of (event, base seed,
  slot, segment) and its parent, so Gate 3's 16 free-running walkers ARE this run's first
  two segments - segment 1 trivially, segment 2 because `C_1 = 0` makes the first
  resampling the exact identity (confirmed on the hardware: `R = 1.0000`, ESS 8/8,
  `max_mult = 1`). `--stage prep` hardlinks them after validating valid time, parent
  lineage, checkpoint and base seed. ~1.5 H100-h.
- **No score is bought where `C_k = 0`.** `V_k = C_k * z` is identically zero there, so
  the score cannot move a single clone or kill; buying it would cost the longest score
  forecast in the run (~7.7 H100-h at N=64) for a diagnostic Gate 3 already measured
  (rho_s = -0.044 at 3 d - which is *why* `C_1` is zero). The free persistence index is
  recorded in its place. `--score-null` buys it anyway.

**Two score backends, behind one interface**: `fcn3` (production) and `persistence`
(the paper's Standard-RES - the walker's current index value, free from its own
diagnostics cube). Running the pilot twice, once per backend, is the control experiment
for "the AI scorer earns its GPU hours" on production data rather than on Gate 3's 16
walkers. The `gencast` PFS backend is **not** implemented: it would need a nested GenCast
ensemble per walker per resampling time, which is the cost of the entire experiment again.

**Disk is a binding constraint, not a footnote.** Retaining every state is 213 GB per
event at N=64 against ~600 GB free on a 95%-full shared NFS, so `--keep-states` (default
2) prunes states more than two segments back; rolling a segment needs only its parent, and
the diagnostics cubes every observable is derived from are never pruned. Peak is 92 GB.
Verified by deleting every state of the smoke run: it still replays, because `walk.done`
plus the diagnostics are sufficient.

**A cold event must be scored on the low tail.** Resampling clones the largest score, so
`aindex.tail_sign` negates `A_L` for Uri; without it an importance-splitting run on a
freeze would spend its whole budget cloning the warmest members.

**Measured rates, replacing the estimates this plan was written with.** GenCast 0.25 deg
serial is 27-28 s/step (Gate 3, job 1160) and FCN3 is **1.40 s per member-step** (job
1161: 8.41 s for a 6-member lead step), not the 1 s/step `slurm/aires_gate3.slurm`
budgeted - which is why that gate's score phase overran its predicted 50 min by 35. At
N=64 the pilot is **~44 H100-h** (19.4 walk + 25.1 score), ~5.6 h on one 8-GPU node plus
one model load per pool launch; `--stage prep` prints the number for the actual cache
state. That is within the ~46 H100-h this plan budgeted.

### Phase 4 - Run the pilot and produce figures (~1 day, ~44 H100-h)

The run itself is `sbatch slurm/aires_res.slurm` (Phase 3); what Phase 4 adds is
`aires/aplots.py`, reusing `xres/xplotting.py`'s cartopy helpers. The direct-sampling
baseline the first figure needs is already built and costs no GPU -
`--stage ds` reduces Gate 3's 16 free-running walkers, the cached 24-member xres
GenCast ensemble and the 24-member FCN3 ensemble through one code path:

| direct sample | n | mean A_L | sd | max | reaching the observed +7.72 K |
|---|---|---|---|---|---|
| GenCast walkers (Gate 3) | 16 | +2.54 | 3.02 | +7.49 | **0** |
| GenCast 24-member (xres) | 24 | +2.46 | 2.66 | +7.08 | **0** |
| FCN3 24-member | 24 | +2.29 | 3.10 | +7.87 | 1 |

So 40 direct GenCast samples do not reach the observed heat dome and stop 0.23 K
short. That is the gap the pilot asks RES to close. (The FCN3 ensemble does reach
it once, which is worth reporting but is a different model, not a baseline for a
GenCast walker.)


1. **Exceedance / return-period curve** for `A_L`: AI+RES N=64 vs the existing 24-member
   GenCast DS vs the 24-member FCN3 DS, with the ERA5 observed value marked. Headline
   question: does the RES tail contain the observed heat dome when the 24-member DS does not?
2. **Genealogy tree** - walker ancestry over the 5 resampling steps, coloured by `A_L`.
3. **Score-skill scatter** - `theta(t_k)` vs realized `A_L(t_f)` per resampling time, the
   quantitative version of Gate 3 on production data.
4. **ESS and weight diagnostics** per step - the honest report on whether N=64 held up.
5. **Composite maps** of the RES-selected extreme members (Z500 and T2m anomaly at
   lags -3 d, 0 d), the paper's Fig. 4 analogue.
6. **Clone-divergence diagnostic** - pairwise separation of sibling walkers vs lead, to
   confirm diffusion noise alone splits clones. If separation is too slow, add the paper's
   3e-3 log-ps spherical-harmonic perturbation.

### Phase 5 - Stage B, seasonal return periods (gated, exploratory)

Both framings were requested, staged. Stage B is where the paper's actual claim (return
periods in *years*) lives, and it depends on GenCast running stably far past its training
horizon.

**Gate first, cheaply.** Roll 8 GenCast 0.25° members 40 days from a climatological
2021-07-01 IC and compare the walker climatology at leads 20/30/40 d against ERA5
June-August climatology. Abort criteria: CONUS-mean T2m drift > 1 K, or a >20% loss of
Z500 eddy variance (spectral blurring). ~4 H100-h.

If it passes, run AI+RES with `t_f` set to the climatological peak of the seasonal index
and estimate return periods across several years' July-1 ICs. If it fails, report the
drift diagnostic as the reason the climatological framing needs a stable climate emulator
(ACE2/cBottle) as the walker instead of GenCast - a legitimate negative result, and the
same conclusion the paper reaches when it chose PlaSim over an emulator for the walker.

## Verification

Per-phase, all runnable:

```bash
# Phase 0 (from a3mega)
bash scripts/sync_a3mega_to_derecho.sh --dry-run     # manifest + counts, no transfer
bash scripts/sync_a3mega_to_derecho.sh               # then assert counts on arrival
python fcn3/compare_fcn3_gencast.py                  # figures reproduce from transferred cubes

# Phase 1 gates (a3mega; gate 1 is CPU-only, login node)
python -m aires.calibrate --fit                      # fit u100m/v100m/tcwv corrections
pytest aires/tests/                                  # gate 1 assertions as unit tests
sbatch --job-name=Vayuh-s2s slurm/aires_gate2.slurm  # forecast equivalence
sbatch --job-name=Vayuh-s2s slurm/aires_gate3.slurm  # score skill - the go/no-go

# Phase 2 (CPU only, no GPU)
pytest aires/tests/test_dmc.py                       # pivotal sampling + estimator unit tests
python -m aires.run_aires --stage res --score persistence --dry-run   # C_k tuning replay

# Phase 4
python -m aires.run_aires --stage prep --event PNW_HeatDome_2021      # login node, cache verify
sbatch --job-name=Vayuh-s2s slurm/aires_res.slurm
squeue -u $USER -n Vayuh-s2s
python -m aires.run_aires --stage compare --event PNW_HeatDome_2021
```

Unit tests worth writing (`aires/tests/`, following `p90/tests/test_pmetrics.py`'s style
of pinning against a reference implementation):

- **pivotal sampling**: `sum N_k^i == N` exactly, and `E[N_k^i] -> w_k^i / wbar_k` over
  many draws.
- **estimator unbiasedness on a toy system**: run the DMC core on a 1-D Ornstein-Uhlenbeck
  walker where the exceedance probability is analytic; assert the RES estimate matches
  brute force within Monte Carlo error. This is the single most valuable test - it
  validates `aires/dmc.py` with zero GPU time.
- **adapter round trip**: GenCast state -> FCN3 channels -> assert all 72 present, correct
  order, correct grid orientation, physical ranges.
- **`C_k = 0` no-op**: with all `C_k = 0` the algorithm must reduce exactly to direct
  sampling (uniform weights, no resampling) - a strong end-to-end invariant.

Prep discipline per `CLAUDE.md`: run `--stage prep` on the **login node** and confirm the
caches exist before any `sbatch`, so GPU jobs are pure cache-hit-then-compute.

## Descoped, with cost, if you change your mind

Only the existing-24-member + ERA5 baseline was selected. Two cheap additions would
upgrade the claim from "RES reaches further into the tail" to "RES is statistically
unbiased", which is the paper's central result:

- **Extend the DS reference.** GenCast member seeds are `fold_in(PRNGKey(0), i)`, so the
  existing 24-member cube extends by simply rolling members 24..N - no re-run of what
  exists. N=96 costs ~24 H100-h/event and lets the RES exceedance curve be checked against
  DS where they overlap.
- **Standard-RES with the persistence score** is ~10 lines once the score backend is
  pluggable, and needs no FCN3 hours. It is the paper's headline comparison and the entire
  justification for using an AI scorer.

## References

- Paper: arXiv:2510.27066v3, *Phys. Rev. Lett.* (2026), DOI 10.1103/b1gc-9c2q
- Reference implementation: https://github.com/amaurylancelin/AI-RES-public
  (`RES/resampling/` is the model-agnostic DMC core; `RES/pseudocode.md` is the algorithm spec)
- Paper's RES hyperparameters (Table I): N=400, M=100, K=6, tau=5 d,
  C_k=(0,0,0,1.6,1.8,2.0), L=7 d
