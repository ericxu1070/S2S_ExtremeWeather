# Cross-resolution GenCast S2S experiment (0.25° vs 1.0°)

A follow-up to the `gencast_s2s` week2/3/4 pipeline. It runs the **same out-of-sample
extreme events at two resolutions** and compares the forecast distributions over the
verification week. The original pipeline is untouched (this writes to `runs/xres/`).

## What changed vs the original

| | original `gencast_s2s` | this `xres` experiment |
|---|---|---|
| Model | GenCast 1p0deg **Mini** | full **0p25deg `<2019`** and **1p0deg `<2019`** (matched pair) |
| Horizons | weeks 2/3/4 | **week 2 only** (3/4 held) |
| Saved output | week-mean **T2m** over CONUS (~165 KB/event) | **entire output cube** — all target vars, every rollout step, every member, CONUS box |
| Events | 6 heat/cold | 6 heat/cold + **3 hurricanes** + **3 extreme-rain** (12 total) |
| Metrics | T2m anomaly | T2m anomaly **+ u850 wind speed + total precip** |
| Figures | per-resolution obs\|mean\|error | **6-panel cross-resolution maps** + **3 combined PDFs** (ERA5\|HRRR\|0.25°\|1.0°) |

## Metrics (one headline per event family)

| Family | Metric key | Definition | Secondary (free, cached) |
|---|---|---|---|
| Heat / Cold | `t2m_anom` | weekly-mean 2 m-T **anomaly** vs 1990–2019 clim (K) | — |
| Hurricanes | `u850_speed` | weekly-mean 850 hPa wind **speed** √(u²+v²) (m/s) | `u850_speed_max` |
| Extreme rain | `tp_total` | weekly **total** precip (mm) | `tp_max12h` |

Wind **speed** (not signed u) is used so the metric does not self-cancel in a mean and is
invariant to HRRR's grid-relative wind orientation. u850/tp are absolute (their magnitude is
the extreme); only T2m is an anomaly. **Because the full cube is cached, any other variable
(MSLP, 10 m wind, …) can be re-plotted offline without re-running GenCast.**

## Layout

```
runs/
├── models/                              # shared; both checkpoints land here in prep
├── observations/                        # shared ERA5 + HRRR truth, one file per (event, metric)
│   ├── <event>_verif_<metric>.nc                ERA5
│   └── <event>_hrrr_verif_<metric>.nc           HRRR
└── xres/
    ├── 0p25/week2/{inputs,cache,verif,figures}
    │   ├── cache/<event>_cube.nc                 FULL output (member,time,level,lat,lon)
    │   └── verif/<event>_<metric>_members.nc     per-member verification field
    ├── 1p0/week2/{...}
    └── figures/                                  cross-resolution maps + the 3 combined PDFs
```

The 3 combined PDFs are also copied to the project root:
`xres_combined_mean.png`, `xres_combined_best.png`, `xres_combined_extreme.png`.

## Run it (2 GPU nodes, one per resolution)

```bash
# 0) login node (internet, CPU): pull both checkpoints + build ERA5/HRRR truth + inputs
conda activate moe
python run_xres.py --stage prep            # both resolutions

# 1) one 8×H100 node per resolution
sbatch slurm/xres_res0p25_week2.slurm      # ~8–12 h, ~60 GB of cubes
sbatch slurm/xres_res1p0_week2.slurm       # ~1–2 h, ~5 GB of cubes

# 2) after BOTH finish (login node, CPU): the comparison figures
python run_xres.py --stage compare
```

Each slurm job re-runs `prep --res <r>` (cache-aware, instant if step 0 ran) then `infer`.
Events run sequentially with per-event cube caching, so a late failure preserves all
completed events and a re-submit resumes.

**On Derecho (PBS)** use `pbs/` instead of `slurm/`. The 0.25° model does not fit the
pmap path on A100-40GB, so both resolutions run as **24-subjob single-GPU member arrays**
(`pbs/xres_member_0p25.pbs` / `pbs/xres_member_1p0.pbs`): subjob N rolls ensemble member N
of every event (`XRES_MEMBER_SEL`), cached to `cache/<event>_memberNN.nc`; the last subjob
per event assembles the cube. Submit compare with `depend=afterok:` on both arrays.
Resubmitting an array is cheap (cached members skip without loading the model). Fallback:
`pbs/xres_res0p25_week2.pbs`, a full-node claim-file worker pool with self-chaining 12 h
jobs. See `HANDOFF.md` for operations.

## Knobs (env)

- `XRES_N_MEMBERS_0P25`, `XRES_N_MEMBERS_1P0` (default 24 each) — ensemble size per resolution.
- `XRES_EVENTS_SEL="HurricaneIda_2021,Kentucky_Floods_2022"` — subset events (smoke tests).
- `XRES_MAX_EVENTS=1` — first N events only.

## Notes / caveats

- **0.25° is the cost driver.** Same network as the 1.0° models but mesh-6 (~16× the mesh
  nodes) on the native 721×1440 grid. The GPU dense-attention swap in `model.py` was only
  validated at 1.0°; the 0.25° node has a 24 h budget and logs per-event progress so an early
  problem is visible within the first event.
- The CONUS cache box verifies hurricane **landfall over land** but does not cache the
  open-ocean track (the chosen scope).
- HRRR T2m anomalies reuse the ERA5 climatology (small model bias); u850/tp are absolute so
  they carry **no** HRRR-vs-ERA5 climatology bias.
