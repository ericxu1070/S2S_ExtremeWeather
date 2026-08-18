# p90 T2m experiment - forecast skill on 90th-percentile CONUS heat anomalies

How well does a fast ML ensemble predict *moderately* extreme events at S2S lead? The
earlier xres work scored 99th-percentile-class events; this experiment scores the much
larger population of **p90 CONUS-mean T2m-anomaly events** so the skill statistics have a
real sample size.

## Design (frozen in `cases.csv` / `cases_meta.json` - committed, do not regenerate)

- **Cases:** 45 events (2022-2025, daily cos-lat-weighted CONUS-mean T2m anomaly >=
  +2.3662 K = the p90 of the 2022-2025 daily series) + **45 matched controls**
  (same-calendar-month non-event days, >= 7 d from any event, deterministic seed 20260722).
  Controls exist because every event is observed "yes" by construction - Brier, reliability
  and ROC are undefined without observed "no" cases.
- **Forecast:** init = peak - 21 d (3-week lead), 50 members, 87 x 6 h steps (the peak day
  keeps all 4 synoptic hours). Model #1 is **FourCastNet3** (internal stochasticity,
  `Zero()` perturbation); the whole pipeline is model-agnostic via the cube contract below,
  so the fcn3-test winner (FCN3 vs GenCast on skill + runtime) can be swapped in.
- **Verification:** week-mean T2m anomaly over [peak-6d, peak], 0.25 deg CONUS box, vs ERA5,
  anomalies vs the 1990-2019 climatology - the exact xres recipe
  (`xres.xmetrics.forecast_field`, `gencast_s2s.data.true_verif_anom`), so p90 results are
  directly comparable to the p99 xres results.

## Where things run

| stage | machine | env | needs |
|---|---|---|---|
| `prep-truth` | a3mega **login node** (or Derecho) | `moe` / `my-env` | internet (ARCO/WB2) |
| `prep-ic` | a3mega **login node** | `fcn3` | internet (ARCO + HF cache) |
| `infer` | a3mega **H100 nodes via sbatch** | `fcn3` | GPU only - pure cache-hit start |
| `compare` | anywhere the cubes live | `moe` / `my-env` | offline |

The committed case list was generated on Derecho (`make_cases.py`, needs the daily anomaly
cube under `runs/p90_t2m_events_2022_2025/`); nothing else needs that cube.

## Runbook (a3mega)

```bash
# 1. login node: truth + IC frames (cache-aware, ~2-4 h first time)
bash p90/bash/00_prep.sh

# 2. submit the 8xH100 worker pool (1 node; NODES=2 for a 2-node array)
bash p90/bash/01_submit.sh

# 3. watch
bash p90/bash/02_status.sh

# 4. metrics + figures + report
bash p90/bash/03_compare.sh
```

Resubmitting the pool is cheap: cubes and claims are cache-aware, so a rerun only fills
gaps. Smoke a single case on one GPU with
`P90_CASES_SEL=event_01_20220113 python p90/run_p90.py --stage infer --model fcn3`.

## Cube contract (how GenCast plugs in)

One NetCDF per case at `runs/p90_t2m/<model>/cache/<case>_cube.nc`:
`2m_temperature(member, time, lat, lon)` float32 Kelvin, absolute 6-hourly times
init..init+87*6h (88 frames), CONUS crop lat 24..50 ascending / lon 235..294 at 0.25 deg,
attrs `case, kind, init, peak, model, members, nsteps, seed`. Any adapter that writes
these cubes (e.g. GenCast members regridded/renamed) gets the full compare stack via
`--stage compare --model gencast` - no metric code changes.

## Metrics (`pmetrics.py` -> `scores/case_scores.csv`, `pooled.npz`; figures in
`figures/p90_t2m/<model>/`, report in `scores/summary.md`)

- **CRPS** of the week-mean anomaly (grid CRPS, cos-lat-weighted mean; standard ensemble
  estimator, properscoring-equivalent) + **CRPSS** vs two references: climatology
  (zero-anomaly forecast, CRPS_ref = wmean|obs anom|) and **persistence** (week ending at
  init, from `truth/<case>_persist_t2m_anom.nc`).
- **Event probability skill:** p_event = fraction of members with any verification-week day
  whose CONUS-mean daily anomaly (00/06/12/18Z mean, clim-matched by hour+dayofyear) >=
  +2.3662 K - the exact event definition. Scored over all 90 cases with **Brier score, BSS,
  Murphy decomposition (reliability/resolution/uncertainty), reliability diagram, ROC/AUC**.
- **Rank histogram** (strict-below convention, 51 bins) pooled over events and controls
  separately + **spread-error ratio** (sqrt((M+1)/M) fair factor).
- **Ensemble-mean RMSE / bias / ACC** (cos-lat-weighted, uncentered anomaly correlation)
  per case, plotted against event magnitude.
- **PDFs** (pooled members vs ERA5 truth, xres `PDF_histogram` convention, log-y) per event.
- **Runtime**: per-case timing JSONs -> total GPU-hours + per-case distribution.

## Maps (`pmaps.py` -> `figures/p90_t2m/<model>/maps/`)

The spatial fields behind those scores, events only, in the `allbestmaps.png` /
`xres.xmapgrid` style. Three ensemble collapses per case: **pooled mean** (ensemble mean),
**p80 extreme member** (the member at the `--extreme-q` percentile of cos-lat-weighted
CONUS-mean anomaly - the nearest-rank quantile member, rank 40/50 at the default q=0.80,
*not* `xres`'s argmax "extreme"), and **best member** (lowest cos-lat-weighted RMSE vs
ERA5, the `xres.xcombined` criterion). The tail is the warm side for every case by
construction, so there is no cold-event sign flip. Note the member quantile is unrelated to
the "p90" in the experiment name: **p90 = the event population** (90th-percentile CONUS
heat days), **`--extreme-q` = which member within a case's ensemble**.

Both member columns badge their **cold->warm heat rank** (`rk N/50`) on the same scale: the
extreme member's is fixed at 40/50 by construction, the best member's says where the
truth-closest realization actually sat in the ensemble's temperature spread (its RMSE rank
would be 1 by definition and carries no information). The composite badges the 45-event
mean of that rank.

Outputs `p90_maps_composite_q80.png` (45-event average of each field + error maps) and
`p90_maps_events_q80_p{1..5}.png` (9 events per page). Reduced `(lat, lon)` fields are
cached to `runs/p90_t2m/<model>/maps/<case>_mapfields_q80.nc` so re-plotting never re-reads
the 418 MB cubes. Cache and figure names carry the quantile tag, so runs at different q
coexist instead of overwriting each other (the original q=0.90 set is on disk as `*_q90.*`).

```bash
python -m p90.pmaps                     # composite + all 5 per-event pages (~4 min cold)
python -m p90.pmaps --only composite    # cache is warm -> seconds
python -m p90.pmaps --extreme-q 0.90    # the older 90th-percentile member (rank 45/50)
python -m p90.pmaps --force             # re-derive the fields from the cubes
```

## Knobs (env)

`P90_N_MEMBERS` (50), `P90_FCN3_BATCH` (4; raise on H100-80GB if memory allows),
`P90_CASES_SEL`/`P90_MAX_CASES`/`P90_KIND` (subsetting - infer/prep only; scoring always
uses the full design), `P90_SHARD=i/n` (multi-node split, auto from Slurm arrays),
`P90_KEEP_ZARR=1` (keep the ~18 GB/case intermediate zarr), `GENCAST_RUNS` (relocate all
data, used by the test sandbox).

## Validation

- `python p90/tests/test_pmetrics.py` (or pytest): 14 unit tests incl.
  properscoring-equivalence for CRPS, Murphy identity, ROC/AUC properties.
- `GENCAST_RUNS=<SB>/runs PYTHONPATH=. python p90/tests/make_smoke_sandbox.py` then
  `... p90/run_p90.py --stage compare --model fcn3`: full offline compare chain on
  synthetic data (the script refuses to run against the real runs/ tree).

## Caveats

- FCN3's `<2019`-free training: FCN3 is trained through 2017 (paper) - all 2022-2025 cases
  are out of sample, same as GenCast's `<2019` checkpoints.
- The verification week ends at peak 00Z (xres convention) while the event definition uses
  full calendar days; the daily p_event series covers all 4 synoptic hours of every day
  including the peak day (hence 87 steps, not 84).
- 2022 truth uses the WB2 store, later dates use ARCO (handled inside
  `gencast_s2s.data.true_verif_anom`).
