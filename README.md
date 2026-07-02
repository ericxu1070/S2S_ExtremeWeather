# GenCast S2S extreme-weather ensemble — batch pipeline (8×H100 / Slurm)

Batch port of `gencast_extreme_events_ensemble.ipynb`. For each out-of-sample extreme
event we initialize GenCast (1.0° "Mini" diffusion checkpoint, `<2019`) some weeks before
the event's peak and ask whether the forecast **distribution** over the verification week
captures the observed extreme. The deterministic Google-Drive/Colab path is gone:
everything is pulled to local NFS and organized into folders, and the ensemble runs across
all 8 H100s of one `a3mega` node.

## Three experiments (forecast horizon)

| script                     | weeks | lead    | init       | rollout steps (12 h) |
|----------------------------|:-----:|:-------:|------------|:--------------------:|
| `slurm/gencast_week2.slurm`|   2   | 14 days | peak − 14d | 28                   |
| `slurm/gencast_week3.slurm`|   3   | 21 days | peak − 21d | 42                   |
| `slurm/gencast_week4.slurm`|   4   | 28 days | peak − 28d | 56                   |

The **verification target is identical in all three**: the weekly-mean 2 m-temperature
anomaly over CONUS for the 7 days ending on the event peak. Only the lead time to reach it
changes, so the three runs isolate how week-N predictability of the extreme degrades as
GenCast is initialized earlier.

## Layout

```
S2S_ExtremeWeather/
├── gencast_s2s/           # the pipeline package
│   ├── config.py          # events, horizons, CONUS box, model spec, folder layout
│   ├── data.py            # GCS ERA5 access, statics, input builder, truth
│   ├── download.py        # prep stage: pull params/stats/inputs/truth into runs/
│   ├── model.py           # load checkpoint, GPU attention swap, pmapped forward
│   ├── inference.py       # rollout -> keep verification week -> save members
│   ├── plotting.py        # PDFs, cartopy CONUS maps (obs|mean|error), CRPS/rank hist
│   ├── hrrr.py            # HRRR observed overlay truth (week-mean T2m anomaly, offline)
│   └── combined_pdfs.py   # the three weeks-2/3/4 overlay PDFs at the project root
├── run_experiment.py      # CLI: --weeks {2,3,4} --stage {prep,infer,plot,combined,all}
├── setup_env.sh           # one-time install into the `moe` conda env
├── slurm/                 # gencast_week{2,3,4}.slurm + gencast_smoke.slurm
├── third_party/graphcast/ # graphcast source (cloned by setup_env.sh)
├── logs/                  # slurm stdout/stderr
└── runs/                  # all outputs (created at runtime)
    ├── models/                             # shared, all cached for offline infer:
    │   ├── params/  stats/  statics.nc     #   checkpoint + normalization stats + orography
    │   └── clim_1990_2019_t2m_conus.nc     #   climatology for the anomaly conversion
    ├── observations/<event>_verif_t2m_anom.nc        # ERA5 observed truth (shared)
    │   └── <event>_hrrr_verif_t2m_anom.nc            # HRRR observed overlay (shared)
    └── week{2,3,4}/
        ├── inputs/<event>_gencast_inputs.nc     # initialization frames
        ├── forecasts/<event>_gencast_verif_t2m_anom_members.nc
        └── figures/{pdf,maps,scores}/
```

## Run it

```bash
# 0) one-time, on the login node (has internet):
bash setup_env.sh

# 1) download + organize all data into runs/ (login node; ~minutes):
conda activate moe
python run_experiment.py --weeks 2 --stage prep
python run_experiment.py --weeks 3 --stage prep
python run_experiment.py --weeks 4 --stage prep

# 2) smoke test on the GPU node (1 event, 8 members, ~minutes once compiled):
sbatch slurm/gencast_smoke.slurm

# 3) the three experiments:
sbatch slurm/gencast_week2.slurm
sbatch slurm/gencast_week3.slurm
sbatch slurm/gencast_week4.slurm
```

Each `*.slurm` runs `--stage all` (prep → infer → plot). `prep` is cache-aware, so if you
already ran step 1 on the login node it is instant; if the compute node has internet it
self-heals. The three jobs are independent and can run on the 8 idle `a3mega` nodes at once;
writes to shared files (checkpoint, truth) are atomic.

## Outputs

Per horizon, under `runs/week{N}/figures/`:
- `pdf/` — observed vs GenCast ensemble PDF of the week-N CONUS T2m anomaly, per event + combined.
- `maps/` — cartopy CONUS maps: **observed | ensemble-mean | error (forecast − obs)**, per event + combined.
- `scores/` — `scores.csv` (CRPS, spread), rank histograms (per event + pooled), CRPS bar summary.

At the project root, the three weeks-2/3/4 **overlay** PDFs (one subplot per event,
all horizons on one axis):
- `combinedpdf.png` — full 24-member ensemble PDF per week.
- `combinedbestpdf.png` — single best member (lowest weighted RMSE vs obs) per week.
- `combinedextremepdf.png` — single most-extreme member (warmest/coldest) per week.

Regenerate them (reads every week's forecasts, so run after the per-week plots exist):

```bash
python run_experiment.py --weeks 4 --stage combined   # or: python -m gencast_s2s.combined_pdfs
```

## HRRR observed overlay

Every PDF (per-event, per-week combined, and the three root figures) overlays a second
observed truth, **`Observed (HRRR)`** (red), next to ERA5 (black). It is the HRRR
week-mean 2 m-temperature anomaly over the *same* 7-day verification window, sourced from
the local HRRR v3 shard archive (3 km CONUS, 2015–2025, 6-hourly) and regridded
nearest-neighbour onto the ERA5 0.25° CONUS grid — so it overlays apples-to-apples with
the ERA5 curve and the forecasts. HRRR resolves sharper local extremes (fatter tails)
while tracking ERA5's pattern closely (spatial correlation ~0.8).

Built offline by `gencast_s2s/hrrr.py` (part of `--stage prep`, or standalone):

```bash
python -m gencast_s2s.hrrr                 # all events with shard coverage
python -m gencast_s2s.hrrr --overwrite     # rebuild
```

Anomalies reuse the ERA5 1990–2019 climatology (there is no offline HRRR-native
climatology), so a small HRRR-vs-ERA5 model bias is folded in. Paths/channel are
configurable via `HRRR_SHARDS`, `HRRR_GRID_REF`, `HRRR_T2M_CHANNEL` in `config.py`.

## Notes / knobs

- **Ensemble size** defaults to 24 (`GENCAST_N_MEMBERS`); it is rounded up to a multiple of
  the visible GPU count because `chunked_prediction_generator_multiple_runs` requires
  `num_samples % n_devices == 0`. On 8 GPUs that is 3 pmap waves of 8.
- **Event subset** for quick tests: `GENCAST_EVENTS="PNW_HeatDome_2021,..."` or
  `GENCAST_MAX_EVENTS=1`.
- **Env / deps:** uses the existing `moe` conda env. `setup_env.sh` adds only graphcast
  (editable, `--no-deps`) and `properscoring`, and pre-fetches cartopy's Natural Earth
  features so the maps render offline on the compute nodes (cache lives on NFS).
- Only post-2019 out-of-sample events are run (the `<2019` checkpoint would otherwise be
  scored on its own training period). The original 6 (heat/cold) are joined by 5
  out-of-distribution CONUS events added for the HRRR overlay — hurricanes
  **Ida 2021, Ian 2022, Idalia 2023** and extreme-precipitation **California atmospheric
  rivers Jan 2023, Eastern Kentucky floods Jul 2022**. Their HRRR truth is already built;
  their GenCast forecasts (and ERA5 truth/inputs) appear in the figures once you run
  `prep`/`infer` for them on the GPU node. Note the verification metric is still the
  week-mean **T2m** anomaly — for these precip-driven events that is a secondary field,
  not the headline (precip-anomaly scoring would be a further extension).
