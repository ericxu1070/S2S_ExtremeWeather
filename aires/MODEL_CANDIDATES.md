# Candidate models for AI+RES - scan of 2026-10-23

A literature/web scan for newer models that could replace or cross-check the GenCast walker
or the FCN3 scorer. Nothing here is built or run. Each claim is marked with how it was
checked; re-check availability before acting on it, it changes fast.

## The trap that applies to every new model: 2021 is in-sample

The AI+RES events are 2021 (Uri, PNW heat dome). GenCast's `<2019` checkpoint keeps them
out-of-sample. Every new open model below trained on 2021:

| Model | Training data | 2021 events |
|---|---|---|
| WeatherNext 2 / Cyclones | earliest checkpoint is `<2023` (trained through 2022) | in-sample |
| AIFS-ENS v2 | ERA5 1979-2022 pre-training, then 2018-2024 analyses | in-sample |

Either rescore on 2023+ events or train a model that holds the event years out (AIFS-SUBS
holds out a mid-record block, 2007-2011, instead of using a cutoff year).

## Open weights - usable now

### WeatherNext 2 (FGN, Google DeepMind) - best walker candidate

- **Availability (verified by listing the bucket, 2026-10-23):** anonymous download from
  `gs://dm_graphcast/weathernext2/params/`, uploaded 2026-08-06:
  `WeatherNext2_<2025_model{1..4}.npz` and `WeatherNextCyclones_<2023/<2024/<2025_model{1..4}.npz`
  (0.25 deg, 701 MB each), `WeatherNextCyclones_Mini_<2023/<2024.npz` (1 deg, 216 MB).
  Code Apache 2.0, other materials CC BY 4.0. Repo: github.com/google-deepmind/weathernext.
  Sample data at `gs://dm_graphcast/weathernext2/dataset/` (HRES init, 13 levels).
- **Fit:** JAX and graphcast tooling, so the swap is mostly `bundle["forward"]` in
  `aires/walker.py::roll_segment`. Noise is RNG-key driven, so `fold_in(step, walker)`
  cloning carries over. One network pass per member (no diffusion solve), so a walker step
  should be much cheaper than GenCast's (unverified, not benchmarked).
- **Changes needed (inferred, not run):** 6 h step (release notes) vs GenCast's 12 h, so
  `WALKER_STEP_H` and `GATE3_SEG_STEPS` change; `STATE_PROGNOSTIC` differs; WN2 predicts
  100 m wind, so the adapter would derive only `tcwv`.
- **Caveats:** `<2025` is fine-tuned on HRES, so ERA5 inits are a distribution shift
  (`Cyclones_<2023` is the least bad). It is a 15-day model; nobody has published 30+ day
  rollouts. Non-Mini needs an H100; on GPU the attention implementation must be switched
  (per the demo notebook).
- **Before any AI+RES use:** run the `astab` stability sweep on it.

### AIFS-ENS v2.0 (ECMWF, Hugging Face `ecmwf/aifs-ens-2.0`, May 2026) - poor fit

- CRPS-lineage ensemble, N320 (~0.25 deg) reduced Gaussian grid, 6 h step, 14 levels to
  10 hPa, PyTorch via `anemoi-inference`, weights CC BY 4.0 (verified on the model card).
- **Walker:** every restart would need a regrid N320 <-> the 0.25 deg lat-lon state the
  adapter uses; each regrid adds error.
- **Scorer:** a GenCast -> AIFS adapter likely needs land/surface channels GenCast lacks
  (soil, skin temperature, ...; input list not checked). Much more derivation than FCN3's 3.
- The AIFS-SUBS paper's control shows AIFS-ENS v2 run past day 15 develops "strong biases
  under autoregressive rollout beyond the medium range". Realistic use: a pure AIFS-ENS
  chain as an independent cross-check, not a component.

### Stochastically Perturbed Weights (MeteoSwiss, arXiv 2609.08412)

Ensembles from deterministic checkpoints by noising weight tensors at inference, tested on
SFNO, Aurora, GraphCast, AIFS. Code: `MeteoSwiss/ai-models-ensembles` v1.0.0. Relevant to
`aires/score_noise.py` and the wbench deterministic scorers: a tunable noise level on SFNO.
Per-architecture tuning; failure mode is a coherent whole-field offset.

## Not usable

- **AIFS-SUBS (ECMWF, arXiv 2607.05100):** no weights (not on ECMWF's Hugging Face,
  checked 2026-10-23). Only public output is its AI Weather Quest submissions (quintile
  probabilities, below): no members or states, so it cannot restart a walker.
- **FGT (freely evolving seasonal model, Research Square preprint):** no code found.
- **Microsoft Weather S2S:** private preview. **WeatherNext 3:** forecasts only, no weights.

## Long-rollout evidence (relevant to scorer choice)

- arXiv 2605.30184, 2-year rollouts of 9 models: SFNO, Aurora, DLESyM stable; GenCast blows
  up at 71-91 d and loses seasonality from ~79 d (consistent with `astab`: GenCast members
  diverged from 39 d, job 1197). FCN3 and AIFS-ENS were not in the 9.
- FCN3 paper: stable spectra to 60 d. Our sweep: clean to 98 d on both arms (job 1194).
- FCN3 vs 0.25 deg GenCast at 21 d on 6 CONUS events
  (`figures/fcn3/week3/fcn3_vs_gencast_skill.png`): FCN3 CRPS lower on 5/6 (tie on Ian),
  ens-mean RMSE better on 4/6. Skill against climatology (RPSS) at weeks 3-4 was not
  measured.

## What AIFS-SUBS did right (design lessons for a future walker)

From the paper's methods (appendix not read):

1. **24 h step:** ~42 steps to week 6 instead of ~240. Main stability lever. A 24 h walker
   also halves AI+RES segment step counts vs GenCast.
2. **Model top at 2 hPa** (18 levels) plus ERA5.1 for 2000-2006 upper levels; no humidity
   at/above 100 hPa. Our GenCast divergences start in the polar stratosphere (the 1211 K
   chains) under a 50 hPa top; the paper does not claim the higher top cures instability.
3. **afCRPS training with noise via conditional layer norm:** one-pass stochastic members,
   seed-driven, the same clone mechanism we use. ~230M params, O96 (~0.9 deg).
4. **Cheap recipe:** single-step pre-train (~4 d on 16 A100), 3-step rollout fine-tune
   (longer did not help), EMA weights "important for the stability of long autoregressive
   rollouts". Mixed ERA5 + operational-analysis fine-tune for init-source robustness.
5. **Anomalies vs the model's own reforecast climatology** (20 prior years, same date).
   Applying this to `A_L` is an untested extrapolation.

Claim that anemoi-training ships an AIFS-SUBS config: from a search snippet, unverified.

## AI Weather Quest (ECMWF S2S competition) - if we ever enter

- Submit a global 1.5 deg grid, shape `(5, 121, 240)`, lat 90 -> -90, lon 0 -> 358.5;
  quintile probabilities summing to 1 for weekly-mean `tas`, `mslp`, `pr`; days 19-25 and
  26-32. Optional MJO phase and tropical-storm-day targets. Thursday init, 4-day window,
  only data available before the window. Ranked only with 13 consecutive weekly
  submissions. Docs: ecmwf-ai-weather-quest.readthedocs.io. 2026-27 period dates not checked.
- **AI+RES is the wrong tool for it:** quintiles are 20% bins (not rare events), the RES
  weights target one CONUS observable while scoring is global, and real time needs
  operational inits (ERA5 lags ~5 d). A plain FCN3 ensemble from IFS open-data inits is
  the sensible entry; FCN3 is clean at these leads, GenCast is not reliably.
- Public archive of all submissions (incl. AIFS-SUBS-ERA5, FuXi-S2S as "Fengshun"):
  data.ecmwf.int/ai-weatherquest/, CC BY 4.0, inits Sep 2025 - Aug 2026.

## Suggested next steps (none started)

1. Login-node shape check: download `WeatherNextCyclones_<2023`, run the sample batch
   through the shapes, print its variable list against `aconfig.STATE_PROGNOSTIC` (~30 min).
2. `astab`-style stability sweep of WN2 to 30+ d before any walker work.
3. Score the existing FCN3 week-3/4 cubes as CONUS quintile RPSS vs a 20-year ERA5
   climatology (~1 d, mostly CPU) - the Weather Quest metric on data we already have.

Fuller write-up of AIFS-SUBS: the `AIFS-SUBS - ECMWF Data-Driven Sub-Seasonal Forecasting`
note in Eric's Obsidian vault (not in this repo).
