# The AI+RES insights deck - plan

**Status: BUILT, 2026-09-11.** Written as a plan on 2026-09-10, implemented the next day.
Branch `aires`, one phase per session per the working agreement in `aires.md`. No job was
launched and no GPU was used at any point; the whole deck is CPU work on cached output.

    aires/ainsights.py          panels 1, 2, 6      aires/tests/test_ainsights.py  (47 tests)
    aires/aceiling.py           follow-up, NOT a    aires/tests/test_aceiling.py   (25 tests)
                                deck panel
    scripts/embed_insights.py   PNG -> WebP -> data URI
    docs/aires_insights_template.html  ->  docs/aires_insights.html
                                14 panels, 60 figures, 12.30 MB of the 16 MB ceiling

Suite at completion: **680 passed, 1 skipped** (`aires/tests/ astab/tests/`, 178 s),
against a 602-passed baseline before this work.

**Numbers in this document that were found wrong during implementation have been corrected
in place, each with a dated note saying what it was and why it was wrong.** The corrections
are worth reading as a group, because they share two root causes: quoting
`aires/HANDOFF.md` prose instead of deriving from files, and reading a **wave-scoped**
superlative as if it were slate-wide. See the sigma-depth note under Panel 2, the weight
ESS note under "Uncertainty", the baseline-row count under Panel 1, and section 3.5's
table header.

**What this is.** Vayuh's "Customer Insights (Non-Smooth)" deck
(`docs/vayuh_customer_insights/`, 1 parent page + 7 subpages + 149 figures) is a hindcast
verification climatology: many initialization dates crossed with every CONUS gridcell,
scored as a 2x2 contingency table at fixed anomaly thresholds. This document plans the
AI+RES equivalent from what is already on disk in `runs/aires/` and `runs/astab/`.

**The one-sentence version of the scoping call.** The deck estimates broad skill across
many dates; AI+RES estimates the tail probability of one event. Nine events cannot produce
an ROC curve, a by-year panel, a by-season panel, or a per-market panel, and this plan does
not propose any. What AI+RES can do instead is price events the deck's method cannot score
at all, because the whole ensemble misses them and every contingency-table cell is zero.

Every number below is followed by the file it came from. Where a number is an estimate it
is labelled as one.

---

## Step 1. What the deck asks for

### 1.1 Provenance of this inventory

The markdown carries almost no captions (2 of 149 figures). The panel inventory below was
built by reading the figures themselves: the title band of all 149 PNGs was cropped and
stacked into 15 contact sheets, then 8 figures were read at full resolution to confirm
axis labels, colorbar ranges and legend contents. Figure references use the export's own
filenames relative to `docs/vayuh_customer_insights/Customer Insights (Non-Smooth)/`.

### 1.2 The panel types

| # | Panel type | Metric on the axis / colorbar | Stratifications present | Thresholds | Baseline | Count |
|---|---|---|---|---|---|---|
| P1 | Per-gridcell CONUS map, "Total Accuracy of Daily Extremes" | undefined scalar, colorbar 0 to 0.12-0.5 depending on panel | CONUS, season, year, lead day, none | -2/-3/-4, +2/+3/+4 C | none | 67 |
| P2 | Per-gridcell CONUS map, "True Positive Rate of Daily Extremes" | TPR, colorbar 0 to 1.0 | CONUS, year | -2/-3/-4 C | none | 14 |
| P3 | Per-gridcell CONUS map, "Positive Accuracy of Daily Extremes" | as P1, warm tail | season | +2/+3/+4 C | none | 12 |
| P4 | Per-gridcell CONUS map, "Percentage of Dates with N C Anomaly" | observed base rate, colorbar 0 to 0.5 | none | -2/-3 C | n/a (this is the climatology) | 2 |
| P5 | Reliability curve, precision vs probability threshold p, against y=p | Precision (y-axis, read at full res in `download 1.png`) | CONUS, year, market, season, sign | -2/-3/-4, +2/+3/+4 C | the 45-degree line | 8 |
| P6 | ROC curve, TPR vs FPR, with AUC | AUC | single threshold, season, market | +/-2/3/4 C | diagonal | 6 |
| P7 | ROC curve, Vayuh vs CFSv2, per threshold | AUC | none | +/-2/3/4 C | CFSv2 | 6 |
| P8 | ROC curve, Vayuh vs CFSv2, per market | AUC | 7 markets | +/-3 C | CFSv2 | 14 |
| P9 | Per-location AUC map | AUC, colorbar to 0.75 | none | -3 C | none | 1 |
| P10 | Per-location AUC-minus-CFS delta map | delta AUC, colorbar to 0.2 | none | -3 C | CFSv2 | 1 |
| P11 | Anomaly distribution histogram | frequency | season, market | n/a | none | 5 |
| P12 | "Total Accuracy by Season" bar chart | as P1, y-axis "% Accuracy", 0 to 0.11 | season | 2 C | none | 1 |
| P13 | Market definition map, 7 colored polygons | n/a | n/a | n/a | n/a | 1 |

Stratifications actually present in the export: **season** (spring/summer/fall/winter),
**market** (7, named below), **year** (2016 through 2021), **magnitude** (2/3/4 C),
**lead time** (days 14 through 27, every integer day), **sign** (cold vs warm).

> **Scoping decision (user, this session): no per-market breakdown. The deck is
> CONUS-scope.** Panel types P8 (per-market ROC, 14 figures) and P13 (the market
> definition map) are therefore inventoried here for completeness and carry no verdict in
> Step 3, and no market panel is proposed in Step 4. The market stratification is dropped
> as an axis, not deferred. **This decision is about geographic slicing only and does not
> change the observable**; see section 3.5, which measures what happens if it is read the
> other way.

**The prediction-count-versus-threshold panel named in the brief is not in the export.**
The parent page's text "Lastly we look at the effect of choice of probability threshold p
on number of predictions made by the model" is followed by `download 27.png` and
`download 28.png`, which are the two P4 base-rate maps ("Percentage of Dates with -2 C
Anomaly", "Percentage of Dates with -3 C Anomaly"), not curves. Either the intended figure
was never pasted or it was replaced. Flagged for Vayuh; the plan does not assume it exists.

### 1.3 Four things in the deck that are wrong or undefined, plus five more found while reading

The brief named four. All four confirmed. Five further defects turned up in the figures.
None of the nine is reproduced in the proposed deck.

**D1. "Total Accuracy of Daily Extremes" is never defined, and its colorbar is not an
accuracy.** Confirmed at full resolution: `By Region/download.png`, titled "Total Accuracy
of Daily Extremes (-2.0 C): CONUS", has a colorbar running 0.00 to 0.22 with its top
labelled tick at 0.20. A plain accuracy (fraction of dates classified correctly) on a
23.7% base rate would sit near 0.7 to 0.9, not 0.2.

Three definitions are consistent with the observed range. Evidence for each:

1. **Joint hit frequency, TP / N**: the fraction of all date-gridcell samples that were
   both predicted extreme (at some probability threshold p) and observed extreme. Bounded
   above by the base rate, which `download 27.png` shows peaking near 0.35 to 0.40 in the
   northern Plains. The "Total Accuracy" map peaks in exactly that region at ~0.20.
   Equivalently base rate x TPR, which reproduces the bar chart: `download.png` gives
   winter 0.113, spring 0.027, fall 0.025, summer 0.008, and a winter base rate near 0.30
   times a TPR near 0.38 is 0.113.
2. **Critical success index, TP / (TP + FP + FN)**: also small for rare events, but harder
   to reconcile with the summer value of 0.008 in the same bar chart.
3. **Precision x recall or F1**: no supporting evidence in any panel.

**Recommendation: definition 1, TP / N, the joint frequency of a correctly predicted
extreme over all samples.** It is the only one of the three that reproduces both the
colorbar ceiling and the seasonal bar chart, and it explains the word "Total": the
denominator is the total sample count rather than the observed-positive count, which is
what distinguishes it from the separately mapped TPR (P2, colorbar 0 to 1.0). Two further
supports: one panel is titled "Total Accuracy of Daily Extremes 30% Threshold (-4.0 C):
CONUS" (`By Region/download 3.png`), so a probability threshold p = 0.30 is applied before
the count, which only makes sense for a metric with a predicted-positive cell; and the
warm-tail twin is named "Positive Accuracy", consistent with a signed version of the same
count. **This is an inference from figure evidence, not a recovered definition. Confirm
with Vayuh before publishing any number that uses it.**

**D2. The colorbar maximum is not fixed across panels, so the maps cannot be compared by
eye.** Observed maxima on "Total Accuracy" panels: 0.20 (`by Lead Time/download.png`, cold
leads), 0.30 (season panels), 0.5 (`by Lead Time/download 1.png`, warm leads), and on the
by-year panels 0.12, 0.14, 0.16, 0.175, 0.25 and 0.30 on different years. The deck asks
the reader to read a lead-time decay and a year-to-year trend off these panels, which a
per-panel autoscale makes invalid. Not in the brief; found in the contact sheets.

**D3. `Untitled 2.png` labels all four season ROC curves "ROC for fall".** Confirmed at
full resolution: title "ROC Curve For Different Energy Seasons (+3 C)", four legend
entries all reading "ROC for fall", AUC 0.61, 0.61, 0.63, 0.66 in matplotlib's default
blue/orange/green/red order.

**The mapping is recoverable, contrary to the brief, by matching against the sibling
panel.** `download 18.png` is titled "ROC Curve By Season (+3 C)" and carries the same four
curves in the same color order with correct labels: fall 0.60 (blue), spring 0.61
(orange), summer 0.62 (green), winter 0.66 (red). The AUCs differ by 0.01 to 0.02, so
these are two different runs, but the legend order is alphabetical in both and the color
cycle is positional. **Therefore blue = fall, orange = spring, green = summer, red =
winter, and the outlier at AUC 0.66 is winter, not fall.** The same alphabetical-plus-color-cycle
pattern holds for the market panels (`Untitled.png` lists ERCOT, MISO, NPCC-NE, NPCC-NY,
PJM, WECC-CA, WECC-NWPP in that order). Marked as a high-confidence inference, still worth
one line of confirmation from Vayuh.

**D4. The lead-time page's prose and its figures disagree.** `by Lead Time (Days 14-27)`
says "Here we compute the false positive rate per location at different lead times". All 28
figures on that page are titled "Total Accuracy of Daily Extremes (+/-2.0 C) Lead NN". No
false-positive-rate panel exists anywhere in the export. Either the prose is stale or the
wrong metric was plotted; unresolvable from the export.

**D5. The deck's CFS baseline formula is degenerate as written.** The parent page defines,
for anomaly threshold `a` and predicted temperature `y_hat`:

    P_CFS(y_t >= a) = 1 - Phi(y_hat + a | mu = y_hat, sigma_c)

Since the evaluation point `y_hat + a` and the mean `y_hat` move together, this reduces to
`1 - Phi(a / sigma_c)`, which does **not depend on the forecast at all**. As printed, every
date and gridcell gets the same probability, the baseline has zero skill by construction,
and its ROC would be the diagonal. The plotted CFS curves are not diagonals: `download
11.png` shows the CFSv2 curve well above the diagonal, and the per-market panels give CFS
AUC from 0.51 (`Per Market Plots +3 C/download 5.png`, WECC-NWPP) to 0.64 (`Per Market
Plots +3 C/download.png`, ERCOT). So the implemented code must define the anomaly against
climatology, as every other page does, and the printed formula is a documentation error.

**Convention this plan uses: anomaly relative to climatology, everywhere, with no
exceptions.** Reason: it is what AI+RES already does end to end. `A_L` is a 7-day-mean T2m
anomaly against the ERA5 1990-2019 climatology, and `aires/HANDOFF.md` line 2338 records
that the CFSv2 baseline gets the identical treatment ("The anomaly is against the ERA5
1990-2019 climatology, not CFSv2's own reforecast climatology... This is the *consistent*
choice"). Adopting the deck's forecast-relative convention would require re-deriving every
cached number in `runs/aires/` and would make the AI+RES and CFS columns incomparable.

**D6. Two generations of the same panel coexist with different numbers and no version
labels.** `Untitled.png` and `download 19.png` carry the identical title "ROC Curve For
Different Energy Markets (-3 C)" and the identical seven markets, but disagree:

| market | `Untitled.png` ("score=") | `download 19.png` ("AUC:") |
|---|---|---|
| ERCOT | 0.63 | 0.58 |
| MISO | 0.66 | 0.65 |
| NPCC-NE | 0.63 | 0.63 |
| NPCC-NY | 0.62 | 0.64 |
| PJM | 0.61 | 0.62 |
| WECC-CA | 0.63 | 0.60 |
| WECC-NWPP | 0.63 | 0.64 |

ERCOT moves by 0.05, which is larger than the market-to-market spread the panel is meant
to show. A reader has no way to tell which is current. The `Untitled*.png` family uses the
"ROC for X / score=" label convention and the `download*.png` family uses "X AUC:", so the
convention is a reliable version marker, but nothing says which came first.

**D7. Three CFS comparison panels carry the wrong sign in the legend.** Confirmed at full
resolution: `download 11.png` is titled "ROC Curves for +2 C Threshold", and both legend
entries read "ROC for -2 C" (CFSv2 and Vayuh), with n=698858. That n matches the +2 C panel
`download 9.png` (n=698858) and not the -2 C panel `download 10.png` (n=443374), so the
title and the data are right and the legend label is wrong. The same mismatch appears on
`download 12.png` (title +3 C, legend -3 C, n=522377) and `download 13.png` (title +4 C,
legend -4 C, n=377181). The three cold panels `download 14/15/16.png` are internally
consistent.

**D8. The prose calls precision "true positive rate".** The parent page's caption says
"Here we plot the true positive rate at various thresholds p and compare it to the line
y=p", and its ROC paragraph defines TPR as "percentage of times a predicted anomaly is
correct". That parenthetical is the definition of **precision**, not TPR. The figures are
correct: `download 1.png` is titled "Precision of Negative Anomaly Predictions Over Conus
as a Function of p" with y-axis "Precision". This matters for the conclusion drawn from
them: precision against p compared to y=p is a **reliability diagram**, and "we are
underestimating" is a valid reading of it. The same sentence read as TPR against p would
mean something else entirely and would not support that conclusion.

**D9. "Since 2014" is claimed but the by-year panels start in 2016.** Section A asks "What
is our success rate in predicting cold anomalies spatially since 2014?" and the specific
events named are 2014, 2021 and 2022. The `by Year` page contains exactly 36 figures,
which is 6 years (2016, 2017, 2018, 2019, 2020, 2021) x 3 thresholds (-2/-3/-4 C) x 2
metrics (Total Accuracy, TPR). No 2014, 2015, 2022, 2023, 2024 or 2025 panel exists.

### 1.4 What the export does not contain

Confirmed absent:

- **No numeric tables and no CSVs of any kind.** `find docs/vayuh_customer_insights -type f
  -not -name '*.png' -not -name '*.md'` returns nothing.
- **No verification date range** stated anywhere in text. The only evidence is the by-year
  panels, which imply 2016 to 2021 (D9), and the ROC sample counts below.
- **No confidence intervals** on any AUC, any reliability curve, or any map cell.
- **No initialization-date count, no gridcell count, no events-per-cell count.** The ROC
  legends give a pooled n only.
- **No definition of "Total Accuracy"** (D1), and no definition of the probability
  threshold p used for the map panels except where a title states it ("30% Threshold").
- **No smoothing definition.** The deck is titled "Non-Smooth" and the CFS section refers
  to a forecast "smoothed over several days (a running average)", but the window is never
  given.

Two things the brief listed as absent that are **present**, and both change the scoping:

- **Sample counts exist**, in the ROC legends only. Read from `download 9.png` through
  `download 16.png`: +2 C n=698858, -2 C n=443374, +3 C n=522377, -3 C n=323096,
  +4 C n=377181, -4 C n=232149. These are pooled over all locations and dates, so they do
  not separate into dates x cells without Vayuh's grid definition, but they set the order
  of magnitude: **the deck's cheapest panel rests on ~2.3e5 samples and its richest on
  ~7.0e5.** That is the number the nine-event comparison has to be made against.
- **All seven markets are named**, in figure legends and per-market ROC titles rather than
  in prose: **ERCOT, MISO, NPCC-NE, NPCC-NY, PJM, WECC-CA, WECC-NWPP** (`Untitled.png`,
  `download 19.png`, and the 14 per-market ROC panels). The unlabeled seven-color map
  `By Region/Untitled.png` is decodable against them by geography: Washington plus Oregon =
  WECC-NWPP, California = WECC-CA, Texas = ERCOT, Minnesota/Wisconsin/Illinois/Indiana/
  Michigan = MISO, Pennsylvania/Ohio/Maryland/Virginia/New Jersey/Delaware = PJM, New York =
  NPCC-NY, New England = NPCC-NE. **The polygons are state-aggregates, not real ISO service
  territories** (real ERCOT excludes El Paso and parts of East Texas; real MISO extends into
  the Dakotas and the Gulf South).

  *Moot for this plan given the CONUS-only scoping decision, and recorded only so that the
  deck's "the regions are power markets" reading is not lost. It also stops being a
  question for Vayuh: the two market items are dropped from the open-questions list.*

---

## Step 2. What AI+RES actually has

Read for this section: `aires.md` (553 lines), `aires/HANDOFF.md` (2772 lines),
`astab/__init__.py`, and the data files named below.

### 2.1 The design in one paragraph

AI+RES is a Diffusion Monte Carlo importance-splitting sampler (Lancelin et al.,
arXiv:2510.27066, PRL 2026) with the paper's roles inverted: GenCast 0.25 degree is the
**walker** and FourCastNet 3 is the **score function** that decides which walkers are
cloned and which are killed (`aires.md` lines 1 to 38). Configuration, frozen across every
production run (`aires.md` "Agreed configuration"): N = 64 walkers, M = 6 score members,
K = 5 resampling steps at leads 3/6/9/12/15 d, `C_k = (0, 1.0, 1.4, 1.8, 2.0)`, horizon
21 d, observable `A_L` = 7-day-mean T2m anomaly, cos-latitude weighted, over an
event-centered box, with a CONUS mean carried as a free secondary index.

The claim the design says this can support, stated in `aires.md` lines 33 to 38: *"RES
efficiently reaches the extreme tail of the GenCast forecast distribution, recovering
observed extremes that a 24-member ensemble misses."* It explicitly **cannot** support an
unbiasedness claim, because a large same-model direct-sampling reference was descoped.
Any deck built from this must not overclaim past that sentence.

### 2.2 The event slate

Nine events, all week-3 lead, all N=64, all on the frozen `C_k`, plus one control.
Directory listing of `runs/aires/` and the 10 records in `runs/aires/aires_lift.json`:

| event | box (`aires_lift.json`) | tail sign | tag |
|---|---|---|---|
| PNW_HeatDome_2021 | PNW | +1 | `pilot`, and `persist` (the control) |
| California_HeatWave_2022 | California | +1 | `pilot` |
| SCentral_HeatDome_2023 | S Texas | +1 | `pilot` |
| Southwest_HeatWave_2020 | SW/S Plains | +1 | `pilot` |
| WinterStorm_Uri_2021 | TX/OK | -1 | `pilot` |
| WinterStorm_Elliott_2022 | N Plains | -1 | `pilot` |
| p90_20231107 | CONUS | +1 | `pilot` |
| p90_20240802 | CONUS | +1 | `pilot` |
| p90_20251224 | CONUS | +1 | `pilot` |

`runs/aires/aires_walkers.csv` has 640 data rows, exactly 64 per (event, tag) pair across
all 10 pairs, verified by grouping the file.

### 2.3 The headline numbers, recomputed from the CSV

Computed directly from `runs/aires/aires_walkers.csv`, not copied from prose:

| event / tag | reached | max `A_L` | min `A_L` | observed | sum `p_i` |
|---|---|---|---|---|---|
| PNW_HeatDome_2021 / pilot | 42/64 | +10.292 | +3.536 | +7.716 | 0.5684 |
| PNW_HeatDome_2021 / persist | 0/64 | +7.334 | +0.257 | +7.716 | 8.7497 |
| California_HeatWave_2022 | 25/64 | +6.330 | +0.445 | +4.710 | 0.7771 |
| SCentral_HeatDome_2023 | 6/64 | +6.187 | +1.257 | +5.730 | 1.2672 |
| Southwest_HeatWave_2020 | 0/64 | +4.308 | -0.382 | +5.142 | 1.0623 |
| WinterStorm_Uri_2021 | 32/64 | -10.204 (most extreme) | +1.569 | -7.397 | 1.7688 |
| WinterStorm_Elliott_2022 | 20/64 | -17.043 (most extreme) | -0.407 | -14.422 | 0.8108 |
| p90_20231107 | 64/64 | +4.163 | +1.498 | +0.211 | 0.5827 |
| p90_20240802 | 63/64 | +2.374 | +0.385 | +1.057 | 0.7039 |
| p90_20251224 | 46/64 | +5.121 | +1.251 | +3.259 | 0.6890 |

The `sum p_i` column is the run's own normalization check; it should be 1 and is not
(`aires/HANDOFF.md` line 613 defines the identity, and open problem 5 is that it drifts).
It ranges 0.568 to 1.769 across the nine productions and reaches **8.75** on the
persistence control. This is a caveat the deck has to carry on every probability, not a
footnote.

`runs/aires/aires_lift.json` additionally carries, per event: `p_clim` and `p_fc` at the
95th and 99th climatological rungs and their ratio `lift`, plus `obs_pct`, `ess`, `log_Z`,
`total_mass`, `pool_edge`, `pool_n`, `beyond_pool_mass` and `lift_bound`. Example, the row
this plan quotes later: `California_HeatWave_2022` has `lift` 3.97 at the 95 rung and
10.87 at the 99 rung, `ess` 11.03, `log_Z` -1.2488.

### 2.4 Baselines, and the fact that they are not uniform

The brief describes `ds_baseline.json` as "the 24-member direct-sampling FCN3 and GenCast
baselines". Read across all nine files, it is **not uniformly populated**:

| event | `fcn3` | `gencast_walkers` | `gencast_xres` |
|---|---|---|---|
| PNW_HeatDome_2021 | 24 | 16 | 24 |
| WinterStorm_Uri_2021 | 24 | 16 | 24 |
| SCentral_HeatDome_2023 | absent | 16 | **absent** |
| California_HeatWave_2022 | absent | absent | 24 |
| Southwest_HeatWave_2020 | absent | absent | 24 |
| WinterStorm_Elliott_2022 | absent | absent | 24 |
| p90_20231107 | 24 | absent | 24 |
| p90_20240802 | 24 | absent | 24 |
| p90_20251224 | 24 | absent | 24 |

So: **8 of 9 events have a 24-member GenCast direct sample** (SCentral does not; its 16
Gate 3 walkers are promoted into that slot, and `aires/HANDOFF.md` line 163 records that
`aires/apdfs.py` does this explicitly and records `gencast_xres_source="gate3"`). **5 of 9
have the 24-member FCN3 ensemble.** Only PNW and Uri have all three. Any panel that draws
three baseline rows must be built to drop rows, which `aires/amaps.py` and `aires/apdfs.py`
were already fixed to do (`aires/HANDOFF.md` lines 120 and 163).

**CFSv2 is the one baseline that is complete.** All nine events have
`runs/aires/<event>/cfs/<event>_cfs_lead21_t2m_anom.{json,nc}`, verified by directory
listing. Each holds 4 members from the four 6-hourly cycles ending at the AI+RES init, so
leads run 21.0 to 21.75 d (`PNW_HeatDome_2021_cfs_lead21_t2m_anom.json` has
`n_members: 4`, `member_lead_days: [21.75, 21.5, 21.25, 21.0]`, `lead_days: 21.0`).
The published cross-event table is `aires/HANDOFF.md` lines 2240 to 2251.

### 2.5 The lead-time stability sweep

`runs/astab/stability.csv`: 12,274 rows, columns `event, weeks, lead_days, model,
checkpoint_day, metric, value, band, status, extrapolated, reached_peak, valid_time`.
Two events (PNW_HeatDome_2021, WinterStorm_Uri_2021), three models (`gencast`,
`fcn3_adapter`, `fcn3_era5`), nine leads (weeks 4, 6, 8, 10, 12, 14, 16, 18, 20), and 44
(event, model, week) chains with `reached_peak = 1`. Status values: 6,791 `ok`, 4,460
`report`, 1,023 `FAIL`.

**Trap, and it is important for anyone building a panel from this file.** The thresholds
that produce the published verdict are applied at reduce time, not baked into the CSV
(`aires/HANDOFF.md` line 907, "Thresholds are applied at REDUCE time, never baked into an
artifact"). Taking the first `status == FAIL` row per chain does **not** reproduce the
verdict: doing so gives, for example, a GenCast failure at day 24 of the week-4 PNW chain,
against a published verdict of "GenCast stable through week 6 (42 d) on every chain"
(`aires/HANDOFF.md` line 41). The verdict comes from `astab.reduce._model_survival`, which
also drops the (model, week) pairs where `reached_peak` is 0 (110 of 12,274 rows), so that
three days of walker rollout cannot be read as a clean 140-day chain. **Any stability panel
must call `astab.reduce`, never read the `status` column directly.**

Published verdicts, from `aires/HANDOFF.md` lines 41 to 43 and `astab/__init__.py`:

- **GenCast**: stable through week 6 (42 d) on every chain; 3 of 8 chains diverged, not
  monotonically in lead. Two of the three diverged in the polar stratosphere (1211 K at
  50 hPa) with CONUS normal, invisible to every calibrated CONUS diagnostic (job 1189).
- **FCN3**: clean through week 14 (98 d) on every rollout of both arms; diverges at week 16
  and beyond on the PNW chains only, while every Uri rollout is clean to 140 d (job 1194).
- **GenCast ensemble at week 8 (PNW)**: 9 of 100 members diverged by 56 d (Wilson 95% CI 5
  to 16%); 100% clean through 36 d; none before 39 d (job 1197). Files:
  `runs/astab/ensemble_PNW_HeatDome_2021_wk08.{csv,json}` and `..._members.csv`.

Consequence for this deck: **days 14 to 27, the deck's entire lead-time axis, sits inside
GenCast's measured stable range** (42 d), so a lead-time sweep is a budget question and not
a physics question. That is a genuinely useful thing to be able to say.

### 2.6 Measured cost, for the estimates in Step 3

From `aires/HANDOFF.md`:

- One full production, no Gate 3 seeding: **~63.9 H100-h**, ~8.0 h wall on one 8xH100 node
  (line 302; Elliott at line 423 confirms "the full 63.9 H100-h").
- With Gate 3 seeding available: **~61.1 H100-h** (lines 144 and 378).
- A Gate 3 for a new regime: **~18 H100-h** (line 580).
- Standard-RES persistence control (walkers only, no FCN3): **~36 H100-h** (line 1946).
- Wave 1 total: ~232 H100-h over 5 runs plus 1 Gate 3 (line 188).
- Wave 2 total: ~318 H100-h over 5 productions plus 1 Gate 3 (line 579).
- Measured wall clocks, one 8xH100 node each: 7 h 12 m to 8 h 42 m per production
  (jobs 1172, 1176, 1177, 1179, 1182, 1183, 1184, 1185, 1186).

**Planning figure used below: 64 H100-h and ~8 h wall per new (event, lead) pair.** This is
a measured per-run cost, carried forward unchanged; it is an estimate only in that it
assumes new events behave like the nine already run.

---

## Step 3. The scoping call, panel by panel

### 3.1 The structural mismatch, stated once

The deck's unit of analysis is the **(date, gridcell) sample**. It has between 2.3e5 and
7.0e5 of them (section 1.4). Skill is a contingency table summed over that population, and
every stratification is a partition of it.

AI+RES's unit of analysis is the **event**. There are nine, each one box and one
initialization, each with 64 walkers. The 64 walkers are not 64 samples of the deck's kind:
they are one importance-weighted estimate of one tail probability, and their effective
sample size is far below 64. `aires/HANDOFF.md` line 621 gives the measured weight ESS as
**5.68 for PNW, 2.24 for Uri, 7.24 for the p90 anchor, and 1.01 for the persistence
control**, and records that Uri's 64 walkers hold only 34 distinct mass values because
walkers cloned at the last resampling inherit their parent's mass.

So the honest count is not "nine events, 640 walkers". It is **nine independent
probability estimates**, several of which rest on the equivalent of two to seven
independent pieces of information.

Everything in 3.2 follows from that.

### 3.2 Verdicts

**Reproducible now** means from files already in `runs/aires/` or `runs/astab/` with no GPU
time. **Reproducible with compute** gives the additional runs needed and the H100-hours at
64 per run. **Not reproducible** means no amount of compute inside this experiment's design
produces it.

| Deck panel | Verdict | Detail |
|---|---|---|
| **P1/P3 Per-gridcell "Total Accuracy" map** | **Not reproducible** | Two independent blockers. (a) The metric is undefined (D1); building it would mean inventing a definition, which this plan refuses to do. (b) Even granting definition 1, it is a frequency over many dates per cell, and AI+RES has one initialization per event. A per-cell count over 9 events is 0, 1/9 or 2/9 and carries no information. Not a sample-size problem that compute fixes at reasonable cost: matching the deck's ~10 to 100 dates per cell needs O(1e2) initializations, i.e. O(1e4) H100-h. |
| **P2 Per-gridcell TPR map** | **Not reproducible** | Same denominator problem. TPR per cell needs many observed extremes per cell. |
| **P4 Base-rate map ("Percentage of Dates with N C Anomaly")** | **Reproducible now**, with a caveat | This panel is pure ERA5 climatology and needs no forecast at all. `runs/aires/clim/` holds the climatology cache built by `aires/aclim.py`, and `runs/aires/<event>/era5_series_6h.nc` holds the ERA5 series. **Caveat: this would be a Vayuh-style panel computed from this repo's ERA5, not a reproduction of Vayuh's number**, since the grid, the climatology period and the date range all differ. Useful only as context for where the nine boxes sit. Low value; not proposed in Step 4. |
| **P5 Reliability curve (precision vs p)** | **Not reproducible** at the deck's resolution; a degenerate 9-point version is possible and should not be drawn | The deck's curve has ~15 threshold bins each backed by ~1e4 to 1e5 samples. AI+RES has 9 probability estimates and 9 binary outcomes. A 3-bin reliability plot at n=9 has bins of 2 to 4 events, and its binomial CI spans nearly [0,1]. **The AI+RES answer to the calibration question is a different panel** and it already exists: the mid-curve agreement check against direct sampling, per event (see Step 4, panel 5). |
| **P6/P7 ROC curves and AUC (CONUS and per threshold)** | **Not reproducible** | An ROC needs a population of forecast-outcome pairs. Nine events give a 9-point ROC whose AUC has a standard error of roughly 0.2, i.e. it cannot distinguish 0.5 from 0.9. To reach the deck's AUC precision (differences of 0.01 to 0.05 are read as meaningful) needs O(1e3) events at minimum, which at 64 H100-h each is **O(6e4) H100-h**, roughly 30,000 node-hours on an 8xH100 node. This is not a funding-level gap, it is a design mismatch: AI+RES is expensive precisely because it does something a bulk-skill metric does not need. |
| **P8 Per-market ROC** | **Out of scope** | Dropped by the CONUS-only scoping decision (section 1.2). It was also not reproducible, for the P6/P7 reason plus a boxing mismatch, but the axis is gone and no verdict is needed. |
| **P9 Per-location AUC map** | **Not reproducible** | Per-location AUC needs an ROC per gridcell. Compounds both blockers above. |
| **P10 Per-location AUC-minus-CFS delta map** | **Not reproducible** | Same. Note the CFS **comparison** is fully reproducible at the event level (all 9 events have CFS, section 2.4); it is the per-location AUC form that is not. |
| **P11 Anomaly distribution histogram** | **Not reproducible** as stratified | Requires season partitions of a large sample (the market partition is out of scope). The event-level analogue, the distribution of `A_L` across the walker population, exists and is better (Step 4, panel 3). |
| **P12 "Total Accuracy by Season" bar chart** | **Not reproducible** | 9 events across 4 seasons is 1 to 4 events per season, and the metric is undefined (D1). |
| **P13 Market definition map** | **Out of scope** | Dropped by the CONUS-only scoping decision. |
| **Season stratification (any metric)** | **Reproducible with compute**, but not to a useful precision | The 9 events span summer (PNW Jun, SCentral Jun, Southwest Aug, p90_20240802 Aug), autumn (California Sep, p90_20231107 Nov) and winter (Uri Feb, Elliott Dec, p90_20251224 Dec). Spring has **zero** events. To get even 5 events per season is 20 events: **~1,280 H100-h** plus Gate 3s for new regimes. That buys 5 numbers per season with binomial CIs still spanning ~0.2 to 0.8. Recommend against. |
| **Year stratification** | **Reproducible with compute**, not recommended | The 9 events span 2020, 2021 (x2), 2022 (x2), 2023 (x2), 2024, 2025. The deck's 6 years x 5 events is 30 events: **~1,920 H100-h**. Same precision objection. |
| **Magnitude stratification (2/3/4 C)** | **Reproducible now**, and it is already the experiment's spine | AI+RES does not stratify by fixed C thresholds; it stratifies by **sigma-depth into the model's own forecast distribution**, which is the better axis and is already measured. From `aires/HANDOFF.md` lines 181 to 187 and 205 to 212: PNW +1.98 sigma, SCentral +2.2, California +2.9, Southwest +3.3, Elliott +2.81, Uri +2.28, p90_20251224 +1.91, p90_20240802 0.00, p90_20231107 -0.97. **This is the severity ladder and it is the deck's magnitude axis done properly.** See Step 4, panel 2. |
| **Market stratification (any metric)** | **Out of scope** | Dropped by the CONUS-only scoping decision (section 1.2). For the record, had it been kept: AI+RES boxes are event-centered 6x6 degree windows chosen by `aires/aindex.py`'s selection rule, not ISO territories, and `A_L` is a box mean, so re-boxing is a full re-run of every event (`aires/HANDOFF.md` line 100, "two boxes are two experiments"). |
| **Cold vs warm** | **Reproducible now** | 6 warm events (PNW, California, SCentral, Southwest, p90 x3 as warm-tail) and 2 cold (Uri, Elliott), with `sign` recorded per event in `runs/aires/aires_lift.json` and `tail_sign` in `runs/aires/aires_walkers.csv`. `aires/HANDOFF.md` lines 307 to 320 record that the cold sign chain was verified end to end. **Caveat: n=2 on the cold side.** Both cold events are hits (Uri 32/64, Elliott 20/64), which is worth stating, but two events do not establish a cold-side skill claim. |
| **Lead time (days 14 to 27)** | **Reproducible with compute**, and this is the one stratification worth buying | AI+RES is fixed at 21 d. Each additional lead is a full re-run of the event: 64 H100-h. A useful version is **3 events x 4 leads (14, 18, 24, 27 d) = 12 runs = ~768 H100-h**, ~96 h wall at one node or ~24 h at four. A full replication of the deck's 14-day axis on all 9 events is 9 x 14 = 126 runs = **~8,064 H100-h**, which is not proposed. **Physics is not the blocker**: section 2.5 establishes that GenCast is stable to 42 d, so every lead in the deck's window is inside the measured stable range. |
| **CFSv2 comparison at the event level** | **Reproducible now** | All 9 events, 4 members each, already cached (section 2.4). See Step 4, panel 8. |
| **24-member direct-sampling comparison** | **Reproducible now**, with the coverage table of section 2.4 respected | 8 of 9 events have GenCast direct, 5 of 9 have FCN3 direct. |

### 3.3 What AI+RES answers better than the deck does

Stated plainly, and limited to what the files support.

1. **It assigns a probability to events the deck's method scores as identically zero.** On
   six of the nine events, every member of the 24-member direct GenCast ensemble misses the
   observation (section 2.3, `reached` column against the baselines in section 2.4). A
   contingency-table method sees TP = 0, FP = 0, TPR = 0, and its ROC point is the origin.
   It cannot distinguish "we missed this by 0.2 K" from "we missed it by 4.6 K". AI+RES
   returns a number in both cases: PNW P = 0.054 with 42/64 walkers reaching, and Southwest
   P < 3e-4 with 0/64 (`aires/HANDOFF.md` lines 107 to 123 and 182 to 186).
2. **It measures where the miss comes from.** The severity ladder shows reach follows the
   observation's depth in **the model's** distribution, not the absolute anomaly
   (`aires/HANDOFF.md` line 187). Elliott at -14.42 K is reached; Southwest at +5.14 K is
   not, and Southwest is the shallower event in absolute terms. The deck's fixed 2/3/4 C
   thresholds cannot express this, because they are thresholds on the observation and the
   quantity that predicts reach is a property of the forecast distribution.
3. **It has a clean attribution control the deck has no analogue for.** Job 1180 re-ran PNW
   with the same walkers, the same DMC, the same `C_k` and the same seeds, swapping only
   the score function for persistence: 0/64 reached, against 42/64 with the FCN3 score
   (`aires/HANDOFF.md` lines 147 to 155). That isolates the gain to the scorer.
4. **It reports its own failure boundary as a measured quantity.** `p90_20231107` is a
   negative result the deck's structure would have hidden: at a below-median target the
   resampled population no longer straddles the observation, the indicator is identically
   1, and the estimator collapses to its own normalization check, returning 0.5827 where
   direct sampling gives 21/24 = 0.875 (`aires/HANDOFF.md` lines 496 to 519). **The failure
   is silent in `compare.json`.** Publishing this is the deck's best single piece of
   evidence that the other numbers were checked rather than assumed.
5. **It knows how far the models can be rolled.** Section 2.5. The deck's lead axis stops
   at 27 d with no statement about why; AI+RES has a measured stability limit at 42 d for
   the walker and 98 d for the scorer.

### 3.4 What AI+RES cannot answer at all

1. **Anything about broad skill.** "What is our success rate since 2014" has no AI+RES
   answer. Nine curated extreme events are not a sample from which a success rate can be
   estimated, and they were selected on being extreme, so even treating them as a sample
   would be selection-biased in the direction that flatters the method.
2. **Any ROC, AUC, or per-location skill map.** Section 3.2.
3. **Any by-year trend, by-season panel, or per-market panel.** Section 3.2.
4. **Calibration in the bulk of the distribution.** This is not merely a sample-size limit;
   it is a **measured property of the estimator**. Point 4 of section 3.3: AI+RES has no
   resolution at a target the tilt has already carried the whole population past. The
   usable operating range is targets the resampled population still straddles
   (`aires/HANDOFF.md` line 513). The deck's central region, where the base rate is 23.7%,
   is exactly the region AI+RES is worst at.
5. **Anything about precipitation or wind at these leads.** `aires/HANDOFF.md` lines 217 to
   222 record that hurricanes were evaluated and rejected: on `u850_speed` the observation
   for `HurricaneIan_2022` sits at **-0.60 sigma**, below the ensemble mean, with 17 of 24
   direct members already exceeding it. A 7-day-mean box-mean wind is not a rare-event
   observable. CFS precip metrics are refused outright (`aires/HANDOFF.md` line 2392).
6. **An unbiasedness claim.** `aires.md` lines 33 to 38, quoted in section 2.1.

### 3.5 "CONUS" means the deck's geographic scope, not the observable

The CONUS-only scoping decision has two possible readings, and only one of them is safe.
This section records the measurement that separates them, because the difference is not a
matter of taste.

- **Reading A, adopted.** Drop the market axis. The deck covers CONUS as a whole and is not
  sliced geographically. **The observable stays the event-centered box `A_L`**, which is
  what every number in `runs/aires/` was computed on.
- **Reading B, rejected on the evidence below.** Change the observable to the CONUS-mean
  index `A_L_conus`, the free secondary index that `aindex.indices()` records on every run
  and that appears as a column in `runs/aires/aires_walkers.csv`.

Reading B is superficially attractive: the column already exists, it costs no GPU time, and
it makes all nine events share one spatial reduction. It would also destroy the experiment.

Measured from `runs/aires/<event>/ds_baseline.json`, comparing the published box
sigma-depths (`aires/HANDOFF.md` lines 181 to 187 and 205 to 212) against the CONUS index
scored the same way, sign-aware, against each event's own direct GenCast ensemble:

**Both columns recomputed at `ddof=1` on 2026-09-11**, so the two are directly comparable.
An earlier draft of this table mixed the published box values (effectively `ddof=1`) with
CONUS values computed at `ddof=0`, which put a spurious +0.04 shift on `p90_20251224`, an
event whose box **is** CONUS and whose shift is therefore exactly zero. Every conclusion
below survived the correction unchanged.

| event | box sigma | CONUS sigma | shift | direct reach on CONUS |
|---|---|---|---|---|
| Southwest_HeatWave_2020 | +3.32 | +1.37 | -1.95 | 2/24 |
| California_HeatWave_2022 | +2.85 | +1.47 | -1.38 | 1/24 |
| WinterStorm_Elliott_2022 | +2.81 | +1.89 | -0.92 | 1/24 |
| WinterStorm_Uri_2021 | +2.28 | +1.85 | -0.43 | 2/24 |
| SCentral_HeatDome_2023 | +2.24 | **+0.14** | -2.10 | **8/16** |
| PNW_HeatDome_2021 | +1.98 | **-0.91** | -2.89 | **19/24** |
| p90_20251224 | +1.91 | +1.91 | 0.00 | 1/24 |
| p90_20240802 | -0.00 | -0.00 | 0.00 | 14/24 |
| p90_20231107 | -0.97 | -0.97 | 0.00 | 21/24 |

The three `p90_*` events are unchanged, because their registered box already **is** CONUS
(`runs/aires/aires_lift.json`, `"box": "CONUS"`), and at a consistent `ddof` their shift is
identically zero, which is the check that the table is being computed correctly. Every
event with a real box moves down, and two move catastrophically:

- **The PNW heat dome, the experiment's flagship result, becomes a below-median target.**
  It is +7.716 K over its own box and **+0.643 K over CONUS**
  (`runs/aires/PNW_HeatDome_2021/ds_baseline.json`, `observed`), where the direct ensemble's
  CONUS mean is +1.140 K. Nineteen of 24 direct members already exceed it. The headline
  "42/64 walkers reach where 0/40 direct members do, P = 0.054" has no CONUS analogue,
  because on CONUS there is nothing to reach.
- **SCentral lands at +0.15 sigma with 8 of 16 direct members past it**, i.e. almost exactly
  at the median.

**Four of the nine events would sit at or below the median on the CONUS index**
(PNW -0.91, SCentral +0.14, p90_20240802 -0.00, p90_20231107 -0.97). Section 3.3 point 4
established, from `p90_20231107`, that this is precisely the regime where the estimator has
**no resolution and fails silently**: the resampled population no longer straddles the
target, the indicator is identically 1, and `P` collapses to the run's own normalization
check with nothing inside the run revealing it (`aires/HANDOFF.md` lines 496 to 519). So
Reading B would not merely weaken the deck. It would reproduce the one documented failure
mode of the method on four of nine events, and the resulting numbers would look plausible.

This is a property of the reduction, not a limitation to be engineered around. `A_L` is a
7-day mean, and taking it over the whole of CONUS averages a regional extreme against the
rest of the continent. A +7.7 K dome over a 6x6 degree box is a small positive number
against CONUS, which is the correct physical answer and the wrong observable for a
rare-event sampler.

**Therefore: every panel in Step 4 keeps the event-centered box as the observable.** The
CONUS index is retained where it already earns its place, as the free second calibration
anchor on Elliott (Step 4, panel 6), where it happens to sit at +1.89 sigma with direct
sampling at 1/24, i.e. in the range where it is a usable target. If Reading B was actually
intended, this section is the reason to say so explicitly before any figure is drawn.

---

## Step 4. The deck AI+RES can stand behind

Ordered as a narrative, not as a mirror of the source deck. Each panel names the file it
comes from and what it shows. Figures marked **exists** are already on disk; figures marked
**new** need plotting code, which is a later session's work, not this one's.

**Scope, fixed by section 1.2 and section 3.5: CONUS, no market breakdown, and the
observable is the event-centered box `A_L` throughout.** No panel below slices by market,
and none substitutes the CONUS-mean index for the box index. The one place a CONUS-mean
number appears is panel 6, where Elliott's CONUS index is a genuine second calibration
anchor at +1.89 sigma; it is labelled as such rather than presented as the event's score.

### Deliverable (decided 2026-09-10)

A self-contained HTML deck, following the precedent already in the repo
(`docs/aires_pilot.html` + `scripts/embed_figs.py`, `docs/aires_event_slate.html` +
`scripts/embed_slate_figs.py`). The Artifact CSP blocks every external host, so figures
must be inlined as data URIs rather than linked.

    aires/ainsights.py         panels 1, 2 and 6 (the only new figures)
    scripts/embed_insights.py  inlines every PNG as a data URI
    docs/aires_insights.html   the deck itself

### Uncertainty: measured quantities only, no new statistics (decided 2026-09-10)

**A new bootstrap was considered and deliberately descoped.** The tempting design is to
resample founder lineages (`runs/aires/aires_walkers.csv` carries `founder`, `family` and
`family_size`, and only 11 to 17 of 64 founders survive per production, 3 on the
persistence control), because the weights are
not exchangeable and a naive walker bootstrap is wrong. But the estimator under resampling
is not obvious: whether `Z` is recomputed per draw, and how to treat the tied mass blocks
(Uri's 64 walkers hold only 34 distinct values, because walkers cloned at the last
resampling inherit their parent's `V_K`). Inventing that is the single highest-risk act
available in this deck, and section 3.3 point 4 is the standing proof that this experiment
can fail silently and plausibly.

**So every probability is reported beside three already-measured numbers, and no new
interval is computed:**

1. the **direct-sampling comparison with its binomial CI**, where direct resolves at all
   (the two anchors of panel 6, published in `aires/HANDOFF.md`);
2. the **weight ESS** out of 64. **Corrected 2026-09-11: the slate range is 1.01 to
   18.81**, not the 1.01 to 8.61 an earlier draft gave. 8.61 is Elliott's, which
   `aires/HANDOFF.md` calls "the healthiest of the wave" meaning **wave 2 only**; over all
   ten runs Southwest reaches 18.81, SCentral 11.40 and California 11.03
   (`runs/aires/aires_lift.json`, `ess`). Do not quote 8.61 as a slate maximum. Two other
   wave-scoped superlatives in the same document fail the same way and were corrected in
   the deck;
3. the **run's own normalization check**, `sum p_i` (0.568 to 1.769 across productions,
   8.75 on the control, section 2.3).

Every one of those is traceable to `compare.json`, `runs/aires/aires_walkers.csv` or
`aires/HANDOFF.md`. None is derived by a method this plan invents. If cross-event intervals
are wanted later, the lineage bootstrap gets its own session, its own spec and a unit test
pinning it against `dmc.DMCResult.expectation`, and it does not ride along with the deck.

### Section A: what was run

**Panel 1. The slate.** *(new, table not figure)* Sources: `runs/aires/aires_lift.json`,
`runs/aires/aires_walkers.csv`, `aires/aindex.py::EVENT_BOXES`. Nine events plus the
persistence control: event, box, peak date, init date, tail sign, observed `A_L`,
sigma-depth, N walkers, which baselines exist. The baseline-coverage table of section 2.4
belongs here rather than in a footnote, so a reader knows before the first figure that
**four** events have only one baseline row: SCentral (`gencast_walkers` only), California,
Southwest and Elliott (`gencast_xres` only). An earlier draft of this sentence said three,
contradicting section 2.4's own table; the files say four.

### Section B: the result

**Panel 2. The severity ladder.** *(new)* Source: `runs/aires/aires_walkers.csv` and
`runs/aires/<event>/ds_baseline.json`.

> **Derive the sigma-depths, do not scrape them from prose.** They are quoted in
> `aires/HANDOFF.md` lines 181 to 187 and 205 to 212, but they are reproducible from files
> and must be computed. Verified 2026-09-10 across all nine events:
>
>     sigma = tail_sign * (observed.box - mean(direct.box)) / std(direct.box, ddof=1)
>
> where `direct` is the first available of `gencast_xres`, `gencast_walkers`, `fcn3` in
> `ds_baseline.json` (the coverage table of section 2.4 says which event gets which, and
> SCentral has only `gencast_walkers`).
>
> **Corrected 2026-09-11.** An earlier draft of this note claimed the formula "reproduces
> the published ladder to within 0.01 on every event". That is true of six events and
> false of three, because those three are published in `aires/HANDOFF.md` as
> one-significant-figure prose: SCentral reads "~2.2" against a derived **+2.24**,
> California "~2.9" against **+2.85**, Southwest "~3.3" against **+3.32**. The formula is
> right; the tolerance claim was not. **Derive these numbers, and tolerance the three
> rounded events at 0.06, the other six at 0.01.** Full derived ladder, `ddof=1`:
> PNW +1.98, SCentral +2.24, California +2.85, Elliott +2.81, Uri +2.28,
> Southwest +3.32, p90_20251224 +1.91, p90_20240802 -0.00, p90_20231107 -0.97.
>
> **`ddof=1` is load-bearing**: the population sd shifts PNW from +1.98 to +2.02 and
> Southwest from +3.32 to +3.39. `aires/ainsights.py::check_sigma_ddof1` now asserts this
> at call time against a fabricated sample where the two differ by 15%.

One scatter: x is
sigma-depth of the observation in the direct ensemble, y is weighted `P(A_L >= obs)`, one
point per event, log y, marker filled if AI+RES reached and hollow if not, with the direct
sampling estimate beside it where direct resolves. **This is the deck's magnitude axis done
correctly** (section 3.2) and it is the single most important panel.

**The reach boundary is +2.85 to +3.32, and it must be derived, not quoted.** An earlier
draft of this plan named Elliott (+2.81) as the deepest reached event; it is not.
**California at +2.85 is deeper and was also reached** (25/64), so the boundary runs from
California +2.85 (reached) to Southwest +3.32 (missed). The same stale pair appears in
`aires/HANDOFF.md` lines 550 to 551. `aires/ainsights.py::reach_boundary` computes it from
the data rather than carrying a literal, which is why the figure is right where both prose
sources are wrong.

**Panel 3. Every walker, its anomaly, its probability mass.** *(exists:
`figures/aires/aires_walkers.png`, 10 panels)* Source: `runs/aires/aires_walkers.csv` via
`aires/awalkers.py`. The raw material every other figure reduces. Carries the `1/N` line
that makes the price of reach legible: PNW's hottest walker carries 2.5% of what a direct
member is worth (`p_over_direct` 0.025, `aires/HANDOFF.md` line 630).

**Panel 4. The population as a density.** *(exists: `figures/aires/aires_kde.png`)* Source:
`aires/akde.py`. Three curves per panel: weighted by `p_i`, unweighted, and the direct
ensemble. Keep the existing decision that every curve ends at its own most extreme member,
and keep printing the exact subset-sum `P` in the title rather than a KDE integral
(`aires/HANDOFF.md` lines 680 to 690); the unclipped KDE overstates `P` by 1.15x to 1.75x
depending on the event.

**Panel 5. Per-event exceedance curve with all available baselines.** *(exists:
`figures/aires/<event>/aires_exceedance_<event>_pilot.png`, 9 events + the control)* The
per-event detail behind panel 2. **The mid-curve region is the calibration evidence and
should be annotated as such**: through the range direct sampling resolves, the weighted
curve tracks it within binomial noise, and only then extends about two decades deeper.

### Section C: is it calibrated

**Panel 6. The two calibration anchors.** *(new, from existing data)* These are the only
two places in the whole experiment where direct sampling returns a non-zero probability to
check the weighted curve against, and they are the deck's reliability question answered at
the precision the data supports:

- `p90_20251224`: AI+RES weighted **P = 0.0661** against direct sampling's **1/24 = 0.0417
  [0.00739, 0.202]**, inside the direct CI (`aires/HANDOFF.md` lines 324 to 331).
- `WinterStorm_Elliott_2022` CONUS secondary index: observed -2.73 K at +1.89 sigma, direct
  **1/24 = 0.0417 [0.0074, 0.202]**, AI+RES **32/64, weighted P = 0.136**, inside the CI
  (`aires/HANDOFF.md` lines 431 to 435). This one costs no extra GPU time because
  `aindex.indices()` records both boxes on every run.

**Panel 7. The no-harm control and the negative result, side by side.** *(exists:
`figures/aires/p90_20240802/` and `figures/aires/p90_20231107/`)*

- `p90_20240802` at 0.00 sigma: the median check **passes**. AI+RES weighted **P = 0.649**
  against direct **14/24 = 0.583 [0.388, 0.755]** (`aires/HANDOFF.md` lines 459 to 469).
  The spatial PDF is the cleanest evidence in the wave: weighted **0.0727** against direct
  GenCast's **0.0702**, agreement to 4%, where the unweighted population gives 0.1627.
  **Do not quote the 63/64 reach count as a headline**; the HANDOFF says so explicitly and
  it carries no information at a median target.
- `p90_20231107` at -0.97 sigma: the estimator **degenerates**, returning 0.5827, which is
  exactly its own normalization check to every printed digit, against a true 0.875. Section
  3.3 point 4.

Putting these two on one page is the deck's credibility. One is the control passing and
one is the method failing, both reported at the same size.

**Panel 8. Run health.** *(exists: `figures/aires/<event>/aires_diagnostics_*.png`)*
Per-leg ESS/N, max multiplicity, founder count, `log_Z`, and the normalization check.
**The normalization check must appear on the same page as every probability**, given the
0.568 to 1.769 range across productions and 8.75 on the control (section 2.3). The
anti-correlation noted at `aires/HANDOFF.md` line 574 (check below 1 means the weighted
curve reads high against direct, check above 1 means it reads low) should be shown as the
n=5 lead it is, labelled a lead and not a finding.

### Section D: against the alternatives

**Panel 9. The CFSv2 operational baseline.** *(exists:
`figures/aires/<event>/aires_trajectory_*.png`, all 9 events)* Source:
`runs/aires/<event>/cfs/*_cfs_lead21_t2m_anom.json`, published table at
`aires/HANDOFF.md` lines 2240 to 2251. This is the deck's own comparison, and it is the one
AI+RES can make on every event. **CFS reaches the observation on 0 of 4 members for all six
genuinely extreme events, and reaches on 3/4 and 2/4 for the two mildest** (p90_20231107
and p90_20240802). That pattern, the operational baseline being competitive exactly where
the event is not extreme, is the cleanest statement of what the method is for.

Two caveats to carry on the panel, both already measured:

- **Resolution is not the explanation.** Round-tripping ERA5 truth through CFS's own T126
  grid costs mean +0.010 K, max 0.134 K, against gaps of 2.3 to 11.6 K on the six extreme
  events (`aires/HANDOFF.md` lines 2295 to 2310).
- **The two mild rungs carry a ~0.1 K error bar comparable to their own gap.** Do not quote
  "CFS was exactly right on p90_20240802" as precise (`aires/HANDOFF.md` line 2315).

**Panel 10. The attribution control.** *(exists:
`figures/aires/PNW_HeatDome_2021/aires_exceedance_PNW_HeatDome_2021_persist.png`)*
Persistence-RES against FCN3-scored AI+RES on identical walkers, seeds and schedule: 0/64
versus 42/64. Section 3.3 point 3. Note beside it that the control's normalization check is
**8.75** and its weight ESS **1.01**, i.e. the control is not merely worse, it is
degenerate, which is itself the point.

### Section E: space, and how far this can be pushed

**Panel 11. Maps.** *(exists: `aires_map_compare`, `map_error`, `map_walk`, `map_spread`,
`aires_anom_maps` per event)* Source: `aires/amaps.py`, everything reduced through the same
`aindex.field` call on the same 105x237 grid so walkers, direct members and truth are
directly subtractable (`aires/HANDOFF.md` line 2137). These answer the question the box mean
cannot: whether a walker that scores the right number has the right **pattern**, or is a
displaced ridge that happens to average the same.

**Panel 12. Spatial PDFs.** *(exists: `aires_pdf_box`, `aires_pdf_conus` per event)*
Source: `aires/apdfs.py`. Per-grid-point tail probabilities against ERA5's own. Carry the
Uri caveat: its weight ESS is 2.24, so its spatial PDF is effectively two walkers wide and
**should be read as shape, not level** (`aires/HANDOFF.md` line 405).

**Panel 13. Lead-time stability.** *(exists: `figures/astab/stability.png`; source
`runs/astab/stability.csv` via `astab.reduce`, never the raw `status` column, per section
2.5)* The honest answer to the deck's lead-time page. GenCast stable to 42 d, FCN3 to 98 d,
and the 9-of-100-member ensemble divergence rate at week 8. **The one result to feature is
the diagnostic finding, not the limit**: a CONUS-only diagnostic panel caught 1 of GenCast's
3 divergences while a crude global bounds check caught 3 of 3, because two chains held a
textbook CONUS troposphere on top of a 1211 K polar stratosphere. Any long-lead production
must gate on the global state.

### Section F: what this does not say

**Panel 14. The limits, as a page, not a footnote.** Written from section 3.4. Specifically:
no ROC and why (n=9, and O(1e3) events at ~6e4 H100-h to fix); no by-year, by-season or
per-market panel; no bulk calibration, because the estimator is measured to have no
resolution below the median; no unbiasedness claim, per `aires.md`; no precipitation or
wind observable at this lead, with the Hurricane Ian -0.60 sigma number as the evidence;
and the selection point, that the nine events were chosen for being extreme and are not a
sample of anything.

### Optional, if compute is authorized

**Panel 15. Lead-time sweep on three events.** *(needs compute)* 3 events x 4 leads (14,
18, 24, 27 d) = 12 runs at 64 H100-h = **~768 H100-h**, ~24 h wall at 4 concurrent nodes.
This is the only deck stratification worth buying (section 3.2), it is inside GenCast's
measured stable range, and it would turn panel 13 from a stability statement into a skill
statement. Recommend PNW (the deepest characterized event), Uri (cold, and the strongest
Gate 3 at rho_s +0.909 / +0.968) and p90_20251224 (the calibration anchor, the one event
where the answer can be checked against direct sampling at every lead).

**Panel 16. Extend the direct-sampling reference.** *(needs compute)* `aires.md` lines 537
to 541 record that GenCast member seeds are `fold_in(PRNGKey(0), i)`, so the existing
24-member cube extends by rolling members 24..N with no re-run of what exists. N=96 costs
**~24 H100-h per event**, so ~216 H100-h for all nine. This is the cheapest single upgrade
in the whole plan: it widens the range over which the weighted curve can be checked against
direct sampling, which is the evidence panels 5 and 6 rest on, and section 3.1's ESS
numbers are the reason that evidence needs widening.

---

## Open questions for Vayuh

Answers needed before anything in Section 1 is quoted back to them. None of these blocks
Step 4, which uses no Vayuh number.

1. **What is "Total Accuracy of Daily Extremes"?** (D1) Best inference is TP / N, the joint
   frequency of a correctly predicted extreme over all samples, equivalently base rate x
   TPR. Confirm or correct.
2. **What probability threshold p do the map panels use?** One panel says 30%; the rest say
   nothing.
3. **What is the verification date range?** The by-year panels imply 2016 to 2021; the text
   says "since 2014" (D9).
4. **Confirm the season-to-curve mapping** inferred in D3: blue fall, orange spring, green
   summer, red winter, so the AUC 0.66 outlier is winter.
5. **Is the CFS baseline implemented against climatology?** The printed formula is
   degenerate (D5) and the plotted curves prove the code differs from it.
6. **What smoothing window does "Non-Smooth" refer to?** Never stated.
7. **How many initialization dates and gridcells are behind the pooled n?** The ROC legends
   give totals from 2.3e5 to 7.0e5 but no factorization.

Two market questions (whether `By Region/Untitled.png` is real masking or a schematic, and
which generation of the market ROC panel is current) are **withdrawn** under the CONUS-only
scoping decision. D6's two-generations defect still stands as a finding about the deck's
version hygiene; it just no longer needs an answer for this work.

---

## Notes on scope

- This session wrote this file and one line in `.gitignore` (the vendored Notion export,
  149 PNGs at ~11 MB, is reference material and is now excluded from repo history). Nothing
  else was changed, no analysis code was written, and no job was submitted.
- **Readiness checked 2026-09-10**, all on the login node, no GPU: `pytest aires/tests/
  astab/tests/` gives **602 passed, 1 skipped** in 238 s, which includes
  `test_aclim.py::test_the_cache_on_disk_is_not_stale_relative_to_the_registered_boxes`,
  so the climatology cache is current against all 7 registered boxes and Elliott's lift is
  not the spurious 285x. `runs/aires/aires_lift.json` (2026-08-26 21:33) postdates
  `runs/aires/clim/box_t2m_daily_1959_2023.nc` (21:26) by 7 minutes, consistent.
- **The implementation is CPU-only and submits nothing.** Panels 15 and 16 stay optional
  and unbuilt; they are the only items needing GPU time (~768 and ~216 H100-h). Nothing in
  Sections A to F requires an allocation, so no `sbatch`, and the `Vayuh-s2s` job-name rule
  never comes up.
- Next session's work: **done 2026-09-11**, see the status block at the top.

## Found during implementation, not yet acted on

Three things surfaced while building the deck. None is a deck panel; all three are
follow-ups, recorded here so they are not lost.

1. **`aires/aclim.py:471` has a silent CONUS fallback.** The line is
   `col = name if name in anom.columns else "CONUS"`. An event registered after the
   climatology cache was built has a box but no column, so its regional `A_L` is scored
   against the continental CONUS column with no error and no warning. This is the exact
   failure that once reported a spurious `lift >= 285x` for Elliott, whose N Plains index
   reaches -17 K against a CONUS climatology spanning +-5 K. `alift.clim_pool` guards it
   and documents it at length; `aclim.report` does not, and `aires/aceiling.py` was reading
   the unguarded path and caching the result one further remove from the `.nc` staleness
   test. `aceiling.check_report_boxes` now applies the same rule to the report, but **the
   fallback in `aclim.py` itself is still there** and any new caller inherits it. It should
   raise rather than fall back.
2. **Two verified events are priced BELOW their own climatological base rate**: Uri at
   0.30x and Elliott at 0.54x on the 1959-2022 pool (`aires/aceiling.py`, table T1). Both
   are cold extremes that AI+RES otherwise reaches and prices. Since the nine events were
   selected because they happened, a system with positive resolution should price them
   above climatology, so this is the signature worth chasing. It is a hindcast-reliability
   question and needs many more events, not nine.
3. **The deck's tests can no longer write into `runs/`, but the module still can.**
   `aceiling.climatology_report()` writes `runs/aires/clim/report.json` on a cache miss,
   and five tests reached it through `collect()` with their skip guards placed after the
   call rather than before. Guards fixed 2026-09-11 and verified by pointing `CLIM_REPORT`
   at a nonexistent path (6 skips, 3 s, cache untouched). The module's own write is
   deliberate and matches where `aires_walkers.csv` lives, but it is the one place the
   no-writes-to-`runs/` rule is touched.
