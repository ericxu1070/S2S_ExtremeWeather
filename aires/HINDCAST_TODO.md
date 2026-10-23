# AI+RES hindcast: to do

Written 2026-10-22. Plan, not state: `aires/HANDOFF.md` and `acal/HANDOFF.md` say what is built.

## Goal

1. **3-month hindcast**: the most recent 90 days, one case per day (90 cases).
2. **3-year hindcast**: the most recent 3 years, one case per day (~1,095 cases), published
   somewhere people can browse it.
3. **Full contingency table**: TP, FP, FN, TN (hit rate, false-alarm rate, miss rate,
   correct negatives) across ALL days, including near-mean weeks.

Why: the 42-case acal campaign selected cases on the outcome (every case was an extreme),
so it only measured hits at the tails. It cannot give a false-alarm rate, and nothing is
known about how AI+RES behaves when the verifying week is ordinary. A daily hindcast is not
selected on the outcome, so every cell of the table gets filled. It also sidesteps the
acal declustering problem (`acal/HANDOFF.md`, "the declustering rule over-counts"): every
day is a case, nothing to decluster.

## Step 0: make AI+RES cheap enough (do first)

Current cost is ~33 H100-h per case (median 4.2 h wall on one 8xH100 node).

| | cases | H100-h now | node-h now |
|---|---|---|---|
| 3 months | 90 | ~2,970 | ~370 (~15 days on 1 node) |
| 3 years | ~1,095 | ~36,000 | ~4,500 (not runnable) |

Levers already measured, in order of payoff:

- [ ] **Persistent walk workers**: every leg recompiles GenCast (~8.6 min/leg, ~7 H100-h
      per case) because the 4.17 GB executable exceeds the JAX cache's 2 GiB limit. Fix plan
      is `aires/HANDOFF.md`, "Open problems" item 8.
- [ ] **Score with M=3 instead of M=6** (`aires/gate3_msub.py`, 2026-10-22): keeps ~99% of
      M=6's rho_s at 9/12 d and halves scoring. M=1 after 6 d is untested, not cleared.
- [ ] **FCN3 as the walker (DECIDED 2026-10-22)** (`wbench/`, run small1): 10.2 vs 25.7
      H100-h per case at N=32, ~14x faster per simulated day. Combined with M=3/M=1 the
      projection is ~7.7 H100-h per case at N=64. Bonus: with an FCN3 scorer the walker
      state IS the scorer IC, so the adapter (`aires/adapter.py`) drops out. Fix before
      production:
  - [ ] `clones` check OOM'd on one H100 (tried to allocate 40.7 GiB at B=2). Clone by
        copying state on the host, or rerun the check with B=1.
  - [ ] Persist the noise state in the walker checkpoint: restarting without it moves T2m
        by 0.64 K RMS after one segment, so a resampled walker is not a true clone.
  - [ ] Rerun the stability limit for FCN3 as walker: astab found FCN3 clean to 98 d on
        both test events but diverging at 75 to 129 d on the PNW chains. A 21 d lead is well
        inside that, but check every walker on the global state, not only CONUS.
- [ ] **Deterministic scorer (to try)**: replace the M-member FCN3 score rollout with one
      deterministic forecast per walker. Only worth it if it is **as cheap as or cheaper
      than one FCN3 member** (0.98 s per 6 h step, 3.94 s per simulated day on one H100,
      ~52 GB peak, wbench small1) AND passes Gate 3 like FCN3 M=1 does (rho_s 0.67 at 9 d,
      0.93 at 12 d; worst subset 0.52 / 0.89).
  - [ ] Shortlist from earth2studio and time each with the wbench Layer 1 harness
        (s/sim-day, startup, peak GB). Candidates: SFNO, FourCastNet (AFNO), Pangu,
        GraphCast-small 1 deg, DLWP. (No timings yet; all unverified.)
  - [ ] Coarse models count too: the score is a box mean (CONUS `A_L` is 26x59 deg, lattice
        cells 6 deg), so 1 to 2 deg resolves it. Add ACE2 (1 deg), DLWP-HEALPix / DLESyM
        (~1 deg), NeuralGCM deterministic (1.4 / 2.8 deg). GenCast 1.0 deg is already ruled
        out on cost: 7.01 s/sim-day measured vs FCN3's 3.94. Coarse-specific costs: a
        conservative regrid of the FCN3 state, and the model's own T2m climatology for the
        anomaly (rank skill ignores a constant bias, the DMC weights do not). At 2 deg a
        6 deg lattice cell is only 3x3 points, so check lattice scoring separately.
  - [ ] Count the adapter: anything that is not FCN3 needs an FCN3-state to model-input
        map. Its cost and error count against the "as cheap as 1 member" bar.
  - [ ] Score Gate 3 on the same 80 cases so rho_s is comparable to `gate3_msub.py`.
  - [ ] Watch for blurring: a deterministic model regresses toward climatology at long lead,
        so its `A_L` spread shrinks. rho_s (rank) is immune to that, the DMC weights are
        not, so recheck `rho_sens.json` before trusting it.
  - [ ] Free fallback: FCN3 at M=1 is already "one member" and passes Gate 3 at 9/12 d. A
        deterministic model must beat that on cost, not just match it.
  - [ ] What its rho is worth (`aires/score_noise.py`, HANDOFF "What M buys"): N_eq at 3 sd
        runs 224 at FCN3 M=1's curve to 418 at FCN3's ensemble mean (frac=0). So a
        deterministic scorer at one member's cost pays off in proportion to how close its
        Gate 3 curve gets to rho_inf = 0.77 / 0.86 / 0.96 at 6 / 9 / 12 d. Its curve also
        pins `frac`, which one model cannot.
- [ ] **Storage**: 10.16 GB per case at 21 d lead (0.66 GB init + 9.5 GB run output) means
      ~0.9 TB for 90 cases and ~11 TB for 3 years. Keep only CONUS score cubes + walker
      weights per case; delete global walker states once a case is scored.
- [ ] **Init prep throughput**: aprep built 42 inits in 144 min on the login node (~3.4 min
      each, one subprocess per case, ~5 GB RAM each). 1,095 inits is ~62 h. Fine, but
      start it early, and inits are 673 MB each (~740 GB for 3 years), so build in batches
      and delete after use.

Target after Step 0: ~8 H100-h/case gives 3 months ~720 H100-h (~90 node-h) and 3 years
~8,400 H100-h (~1,050 node-h, ~9 days on 5 nodes). (Projection, not measured end to end.)

## Step 1: design decisions to settle before submitting

- [ ] **Event definition, fixed in advance**: box (CONUS mean `A_L` or 6 deg lattice cells),
      threshold (e.g. |A_L| >= 2/3/4 K, or a percentile, which removes the warming-trend
      bias noted in `acal/HANDOFF.md`), and lead (21 d to the verification-week peak, as acal).
- [ ] **Which tail to tilt**: a one-sided run can never forecast the opposite tail (acal
      opposite-tail check, 2026-10-14). A daily hindcast cannot pick the tail from the
      outcome, so it needs the proposed two-sided log-cosh splitting function, or two runs
      per day (2x cost). Settle this before Step 0 sizing is final.
- [ ] **Forecast to score**: P(event) per day from walker weights, plus the untilted
      baseline (plain FCN3/GenCast ensemble) on the same days so AI+RES has something to
      beat.
- [ ] **Data availability**: confirm ARCO ERA5 coverage reaches the last 90 days
      (inits and truth). The acal index cube currently stops at 2026-08-31. (Unverified:
      ARCO latency for Oct 2026.)

## Step 2: run

- [ ] 3-month hindcast, 90 cases, a3mega Slurm array (job name `Vayuh-s2s`), prep on the
      login node first.
- [ ] Look at it (Step 3) before committing to 3 years.
- [ ] 3-year hindcast, ~1,095 cases.

## Step 3: evaluation

- [ ] Contingency table at each threshold: TP, FP, FN, TN; hit rate (POD), false-alarm
      ratio and rate, CSI, frequency bias.
- [ ] ROC (two-sided) and reliability diagram; Brier score and BSS vs climatology.
- [ ] Behaviour on near-mean weeks: does AI+RES over-forecast extremes when nothing
      happens (tilt leaking into the forecast)?
- [ ] Confidence intervals by **block bootstrap**: consecutive daily inits verify
      overlapping weeks, so 90 days is far fewer than 90 independent samples.
- [ ] Optional: same table for CFSv2 / GEFS / EC46 on the same days (acal board code).

## Step 4: something people can look at

- [ ] Browsable page: per day, forecast P(event) vs what happened, map of the verifying
      week, running hit/false-alarm counts. Format and host still to decide.
