# Plan: CFSv2 operational baseline for the acal campaign

Status: PLAN, nothing built. Drafted 2026-10-05.

Question: on the same 42 cases (21 d lead, CONUS `A_L`, 31 heat / 11 cold), does AI+RES put
more probability on the observed tail than NCEP's operational CFSv2?

## Agreed choices

| knob | choice |
|---|---|
| ensemble | 16 trailing 6-hourly cycles ending at the AI+RES init (leads 21.0-24.75 d); 4-cycle subset reported too |
| bias | both: **raw** ERA5-1990-2019 anomaly is the headline (same treatment as AI+RES); lead-dependent mean bias correction from CFS's own forecasts in the other years is the sensitivity |
| estimator | empirical member fraction with the same `>=` rule (primary); Gaussian fit to members (sensitivity) |
| scope | scorecard side-by-side, paired head-to-head, gridpoint maps |

Where: Derecho login node, CPU + internet, env `my-env`. No GPU anywhere in this plan.
New module `acal/cfsbase.py` (stages below); it reuses `aires/cfs.py` for fetch/regrid and
`acal/analyze.py` / `acal/maps.py` for climatology and scoring. It modifies neither
`aires/` nor the existing acal outputs.

## Steps

1. **Build the 16-member cubes** (~45 min wall, mostly download, ~6 GB).
   `python -m acal.cfsbase --stage build` loops `aires.cfs.build(case, n_cycles=16)` over
   the 42 cases. Fix first: `aconfig.cfs_cube_path` does not encode the cycle count, so
   write acal cubes to `runs/acal/cfs/<case>_cfs16.nc` instead of the shared aires path.
   Gate: 42 cubes, `n_members` 16 (or `cycles_skipped` recorded), every `check_window`
   passes, CONUS `A_L` of the cube matches `aires.aindex` to 1e-3 K.

2. **Bias hindcasts** (~3 h background, ~25 GB, deduped by cycle).
   For each case, the same 16-cycle lag at the same calendar init in every OTHER year
   2021-2026 that the ERA5 index cubes cover (`runs/acal/index/`, 2026 stops Aug 31), so
   it is leave-one-year-out by construction. Bias = mean over those forecasts of
   (CFS - ERA5), as a CONUS scalar and as a 105x237 field over the 7-day window.
   Output `runs/acal/cfs/bias.nc`. Gate: at least 4 years per case; print the bias
   distribution (heat vs cold season) before using it.

3. **Scorecard** (~1 h code, seconds to run). `--stage score` writes
   `runs/acal/analysis/cfs_scorecard.csv` + `cfs_summary.json`: per case
   `P_CFS(obs)` in 4 variants {raw, corrected} x {empirical, Gaussian}, plus the
   4-cycle subset; lift against the SAME `P_clim` (`AN.clim_pool`); PIT. Same table
   layout as `summary.json` so the AI+RES row and CFS rows sit next to each other.

4. **Paired head-to-head** (~1.5 h). `--stage paired`: per case
   - Brier at 3 K and 4 K (both outcomes occur there; at 2 K every case is o=1 by
     selection, so 2 K Brier only measures mass on the observed tail - report it as that);
   - log ratio `log(P_RES(obs) / P_CFS(obs))` with a 1/(N+1) floor on each side;
   - win counts, Wilcoxon signed-rank, case-bootstrap 90% CI; split heat / cold / rung.
   Output `cfs_paired.csv`, `figures/acal/acal_cfs_{scorecard,paired}.png`.

5. **Gridpoint maps** (~2 h). Refactor `acal/maps.py` so `fields()` / `scores()` take a
   forecast source; add the CFS source (16 equal weights, raw and corrected). Figures
   `acal_map_{accuracy,bss,pod,csi}_*_cfs.png` and AI+RES-minus-CFS difference maps for
   BSS and CSI.

6. **Tests + write-up** (~1 h). `acal/tests/test_cfsbase.py` (lag cycles, estimator on
   a synthetic ensemble, bias LOO excludes the case year, map source parity). Results and
   caveats go into `acal/HANDOFF.md` under a new dated section.

Total: ~6-7 h of work plus ~4 h of unattended download.

## Caveats to carry into every number

- **Selected on outcome** (all |A_L| >= 2 K): head-to-head says which forecast put more
  mass on what happened, not which is calibrated.
- **Not the same ensemble**: 16 equal-weight lagged members vs 32 importance-weighted
  walkers. Lag gives CFS 0-3.75 d of extra staleness, never less lead than AI+RES.
- **Resolution**: CFS is ~0.94 deg regridded to 0.25. Measured negligible for the CONUS
  index (<0.17 K, `aires/cfs.py` docstring), but NOT for gridpoint maps - a coarse model
  cannot verify sharp local anomalies. State this on the map figures.
- **Archive holes**: NCEI lacks 2024 and Dec 2025; AWS mirror fills from 2023-04-22.
  Record served archive per cycle; a case with < 12 members is flagged, not dropped.
- **Raw headline means CFS drift is inside the score**, exactly as GenCast's is. The
  corrected variant bounds how much of any gap is drift.
