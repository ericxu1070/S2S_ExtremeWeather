Read `docs/vayuh_customer_insights/` in this repo. It is a complete Notion export of Vayuh's "Customer Insights (Non-Smooth)" deck: one parent page, seven subpages, 149 figures. Start with `Customer Insights (Non-Smooth) 29198aff651846738315e88b16c7dabe.md`, then the seven subpages under `Customer Insights (Non-Smooth)/`. The figures carry their own matplotlib titles, so read the images, not just the markdown; only 2 of 149 have captions.

Goal: produce the AI+RES equivalent of that deck, on branch `aires`, from the GenCast-walker plus FCN3-scorer results already in `runs/aires/` and `runs/astab/`.

This session writes a plan only, to `aires/INSIGHTS_PLAN.md`. Do not write analysis code and do not launch any job.

## Step 1: inventory what the deck asks for

Table every distinct panel type: metric, stratification, threshold, baseline. Expect per-gridcell "Total Accuracy of Daily Extremes" maps, reliability curves (true positive rate at threshold p against the line y=p), ROC curves with AUC, per-location AUC maps and AUC-minus-CFS delta maps, and prediction-count-versus-threshold curves. Stratifications: season, region/market, year, magnitude (2/3/4 °C), lead time (days 14 to 27), cold versus warm.

Four things in the source deck are wrong or undefined. Flag them in the plan, do not reproduce them:

- "Total Accuracy of Daily Extremes" is never defined anywhere and its colorbar runs 0 to 0.20, which is not an accuracy in the usual sense. Write down the two or three definitions consistent with that range, say which you would use, and mark it as needing confirmation from Vayuh.
- `Untitled 2.png` on the parent page plots four season ROC curves and labels all four "ROC for fall" (AUC 0.61, 0.61, 0.63, 0.66). The season-to-curve mapping is unrecoverable from the export.
- The "by Lead Time (Days 14-27)" subpage says in prose that it computes false positive rate per location; every figure on that page is titled "Total Accuracy".
- The deck's CFS baseline defines the anomaly threshold `a` relative to the forecast mean, while every other page uses anomaly relative to climatology. Say which convention you will use and why.

Also record what the export does not contain: no numeric tables or CSVs of any kind, no market or region definitions beyond one unlabeled seven-color map in `By Region/Untitled.png`, no sample counts, no verification date range, no confidence intervals. Note that ROC titles name ERCOT, so the regions are power markets, but the other six are never named in text.

## Step 2: inventory what AI+RES actually has

Read these, do not assume their contents:

- `aires.md` and `aires/HANDOFF.md` (long; at minimum the "Verify state", "What Phase 4 established", and CFSv2 baseline sections)
- `runs/aires/aires_walkers.csv`, 640 rows: per-walker `A_L`, `A_L_conus`, `weight`, `p_i`, `p_over_direct`, `rank`, `cum_p`, `reached`, `observed`, `tail_sign`
- `runs/aires/aires_lift.json`: per-event `p_clim`, `p_fc`, `lift` at the 95 and 99 rungs, plus `obs_pct`, `ess`, `log_Z`
- `runs/aires/<event>/ds_baseline.json`: the 24-member direct-sampling FCN3 and GenCast baselines
- `runs/aires/<event>/cfs/*_cfs_lead21_t2m_anom.json`: CFSv2, 4 members per event
- `runs/astab/`: the weeks 4/6/8/10 lead-time stability sweep

The event slate is nine: `PNW_HeatDome_2021`, `California_HeatWave_2022`, `SCentral_HeatDome_2023`, `Southwest_HeatWave_2020`, `WinterStorm_Uri_2021`, `WinterStorm_Elliott_2022`, `p90_20231107`, `p90_20240802`, `p90_20251224`.

## Step 3: the honest scoping call, which is the part that matters most

The deck is a hindcast climatology: many initialization dates crossed with every CONUS gridcell from 2014 onward. AI+RES is nine events, one box and one initialization each, 64 walkers. You cannot compute an ROC curve, a by-year panel, a by-season panel, or a per-market panel from nine events, and you must not produce something that looks as though you can.

For every panel type from step 1, give one of three verdicts:

- **Reproducible now** from files on disk. Name the exact inputs and the caveat, including that nine events demands bootstrap confidence intervals on anything that looks like a skill score.
- **Reproducible with compute.** State how many additional events or initialization dates are needed and give an H100-hour estimate derived from the per-event cost already recorded in `aires.md` and `aires/HANDOFF.md`.
- **Not reproducible.** State why.

Then state plainly which of the deck's questions the AI+RES framing answers better than the deck does, and which it cannot answer at all. AI+RES estimates the tail probability for one event; the deck estimates broad skill across many. Do not paper over that difference, and do not let the deck's structure force panels that the method cannot support.

## Step 4: propose the deck AI+RES can actually stand behind

Sketch the panel list in order, for a deck that stands on its own rather than imitating the original panel for panel. For each panel say which file it comes from and what the figure shows. Include the CFSv2 comparison and the 24-member direct-sampling baseline, since both already exist for all nine events.

## Constraints

- Branch `aires`. One phase per session, per the working agreement in `aires.md`.
- Every number in the plan must come from a file you actually read, with its path. No estimates presented as measurements.
- Flag uncertainty explicitly rather than filling it in. Never invent a metric definition, a market boundary, or a sample count.
- No em dashes.
