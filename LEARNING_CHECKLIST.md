# S2S Extreme Weather — Learning Checklist

A running map of what you should deeply understand. We work through it **stage by stage**.
Legend: ⬜ not yet · 🟡 partial · ✅ mastered (confirmed by quiz/restatement)

---

## Stage 1 — The Problem & Motivation (the "why")  ✅ MASTERED
- ✅ 1.1 What S2S (subseasonal-to-seasonal) forecasting is, and why week-2 extremes matter
- ✅ 1.2 The core scientific question of this repo (init 14 days before peak → can the forecast capture the observed extreme?)
- ✅ 1.3 Why we care about a *distribution / PDF*, not a single point forecast
- ✅ 1.4 Why **GenCast** (diffusion, generative ensemble) and **not GraphCast** (deterministic → degenerate PDF)

## Stage 2 — The Experiment Design (the "what" & "how")
- ✅ 2.1 Out-of-sample (OOS): training cutoff `<2019`, why in-sample events risk memorization
- ✅ 2.2 Protocol: init = peak − 14d; verify week-2 (lead days 8–14); weekly-mean T2m **anomaly** over CONUS
- ✅ 2.3 What an *anomaly* is and the role of the 1990–2019 climatology
- ⬜ 2.4 Model specs: 1.0° "Mini", 12h step → 28 rollout steps, 13 levels, the input variables
- ⬜ 2.5 CONUS box definition
- ⬜ 2.6 Pipeline shape: cells 0→A→B→C→D→E→F→G and the caching design

## Stage 3 — Scoring & Interpreting Results  (mostly ✅)
- ✅ 3.1 The PDF figure: what's pooled (truth grid pts vs members × grid pts), log-y floor, normalized KDE per curve
- ✅ 3.2 CRPS: definition, units (K), lower=better, the two-term tradeoff
- ✅ 3.3 Why CRPS reduces to MAE for the deterministic control
- ✅ 3.4 Rank histograms: flat = calibrated, U = under-dispersed, dome = over-dispersed, slope = bias (which-end-piles)
- ✅ 3.5 The observed result: symmetric under-prediction of extremes (U-shape) = regression to climatology
- 🟡 3.6 Diagnosing the cause: conditional bias vs under-dispersion vs too-few-members (discussed; not yet quizzed)

## Stage 4 — Edge Cases & Engineering Details
- ⬜ 4.1 WB2 vs ARCO-ERA5 fallback, and why HeatDome 2023 downloads slowly
- ⬜ 4.2 Why Cell E is slow (diffusion rollout × members × events + JIT + live data)
- ⬜ 4.3 The TPU→GPU attention swap (splash_mha → triblockdiag_mha)
- ⬜ 4.4 The 267-channel guard (static vars must stay 2-D, no leaked time axis)
- ⬜ 4.5 Streaming rollout: keep only week-2, discard the rest

## Stage 5 — Broader Context & Impact
- ⬜ 5.1 Fundamental week-2 predictability limit (affects all models)
- ⬜ 5.2 ML models damp extremes; how GenCast tries to fix it; why Mini is the weak config
- ⬜ 5.3 What "better" looks like: full-res, more members, calibration, slow drivers (soil moisture, SST, MJO, polar vortex)
- ⬜ 5.4 Why this matters in the real world

---

### Session log
- Stage 1 ✅ — quiz 3/3 (determinism, week-2 lead, why-distribution-for-tails). Strong conceptual grasp.
- Stage 2 bite 1 ✅ — quiz 4/4 (anomaly, week-2 window, why-anomaly, OOS). Corrected a "only day 14" misconception.
- Stage 2 bite 2 — STILL OPEN: CONUS cropping (2.5) + cell pipeline & caching (2.6) not yet restated/quizzed.
- Stage 3 ✅ (3.1–3.5) — quiz 4/4 (U-shape, bias direction, CRPS<MAE, why-MAE). 3.6 discussed, quiz pending.
- Detours covered: PDF normalization, 0.25°→1° striding + 12h step, denoiser/transformer architecture, GPU vs TPU, attention swap.
