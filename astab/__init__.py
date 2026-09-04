#!/usr/bin/env python
"""How far can the walker and the scorer be pushed before they stop making an atmosphere?

Every AI+RES run to date - the pilot, wave 1, wave 2, all three gates - sits at week-3 lead
(21 d). This package asks the one question that has to be answered before any longer-lead
experiment is worth booking: **at what rollout length does the GenCast walker or the FCN3
score function stop producing a physical atmosphere?**

    4 / 6 / 8 / 10-week lead  x  2 events  x  ONE walker member and ONE FCN3 member

and, since the FCN3 extension, five more leads on the SCORER only:

    12 / 14 / 16 / 18 / 20-week lead  x  2 events  x  both FCN3 arms

Eighteen chains in all, out to 140 days.

It answers **numerical stability only**. It does not, and cannot, say whether the score
still RANKS usefully out there - that needs a rank correlation, which needs an ensemble,
which one member cannot produce. See "Stable is not useful" below.

What was actually in the way
----------------------------
Neither model caps rollout length in code. GenCast's
``rollout.chunked_prediction_generator*`` divides ``num_target_steps`` by a chunk size;
FCN3's ``_default_generator`` is a bare ``while True``. The only cap in the whole stack was
a three-entry dict in two mirrored copies - ``gencast_s2s.config.WEEKS_TO_LEAD_DAYS`` and
``fcn3.fevents.WEEKS_TO_LEAD_DAYS`` - and widening it is purely additive. That plus the
missing init frames was the entire blocker. It now runs 2 .. 20 weeks.

Why the extension is FCN3-only
------------------------------
Job 1189 closed one of the two questions and left the other open. GenCast diverged on 3 of
8 chains by week 6 - a measured limit. FCN3 was clean at week 10 on EVERY rollout of BOTH
arms, which is a FLOOR, not a limit: the sweep stopped before the scorer did. Weeks 12..20
ask the open question and skip the closed one.

At those leads the walker rolls ONE 3-day segment - the adapter-fed arm's launch state -
and both FCN3 arms roll the full horizon. ``sconfig.STAB_WALK_WEEKS`` is the switch, and
what makes that honest rather than convenient is that ``reduce`` refuses to score a lead
the walker never reached: every CSV row carries ``reached_peak``, ``_model_survival`` drops
the (model, week) pairs where it is 0, and neither the headline nor the figure can then
say "GenCast stable through week 20" off three days of rollout. Moving 12..20 into
``STAB_WALK_WEEKS`` turns them into full chains and nothing else in the package changes -
budget it first, a 140-day walker chain is 280 GenCast steps against ~13 min of FCN3.

Layout
------
    sconfig.py   what the sweep is, and where every artifact lives. Nothing else builds
                 a path by hand.
    schedule.py  the segment schedule and the Chain that walks it - both ragged tails,
                 and the rule that the horizon is never a scoring checkpoint
    diag.py      the diagnostic panel and the MEASURED bands it is gated on
    refs.py      ERA5: the scalar reference series, and the T2m truth fields the maps need
    record.py    the per-segment record, written at walk time before the state is pruned
    stages.py    plan / check / calibrate / walk / score / prune
    reduce.py    the panels, the long CSV, the all-chains headline
    plots.py     figures/astab/stability.png
    maps.py      figures/astab/maps/<event>/<model>_wk<NN>.png
    ensemble.py  the ENSEMBLE follow-up: the survival curve, its CSVs and its figure
    run.py       the CLI over all of it

    runs/astab/<event>/walkers/w0N/stepKK/{state.nc,diag.nc,stab.json}
    runs/astab/<event>/scores/w0N_lead03_cube.nc
    runs/astab/<event>/ens/wk<NN>/m<MMM>/stepKK/{state.nc,diag.nc,stab.json}
    runs/astab/<event>/ens/wk<NN>/stab_m<MMM>.json
    runs/astab/refs/{bands.json,gate3_panel.csv,<event>_era5_{ref,t2m}.nc}
    runs/astab/stability.csv
    runs/astab/ensemble_<event>_wk<NN>{,_members}.csv  +  ensemble_<event>_wk<NN>.json
    figures/astab/{stability.png, ensemble_<event>_wk<NN>.png}

The walker slot ``N`` is the lead's index in ``STAB_WEEKS`` and is append-only, so weeks
4/6/8/10 keep slots 0..3 and every artifact job 1189 left on disk stays addressable. The
ensemble tree is keyed by WEEK (``wk<NN>``) and then by MEMBER instead, and the two
namings are why ``Chain`` dispatches every path on ``member``: a week-8 member's walker
slot is still 2, so a path built from the slot lands in the frozen sweep's tree - which
``stage_walk`` ``rmtree``s on a stale or partial segment.

This was ``aires/astab.py``, a tag inside the AI+RES run tree. It is its own package and
its own tree for the reason ``xres`` is separate from ``gencast_s2s``: the two experiments
answer different questions and only one of them should be able to break the other's cache.
It still imports ``aires`` (the walker, the adapter, the scorer, the index) and modifies
nothing there.

Env: every module here imports cleanly in BOTH ``moe`` and ``fcn3`` - like
``aires.aconfig``/``adapter``/``gate2``, and for the same reason: one driver has to span
the two-env handoff. Do not add a JAX import at module level anywhere in this package;
``aires.walker`` is loaded inside the walk stage only.

What the sweep found (job 1189)
-------------------------------
**GenCast stable through week 6 (42 d) - every chain. FCN3 through week 10 (70 d) - every
rollout, both arms.** GenCast diverged on 3 of 8 chains, and NOT monotonically in lead:
PNW's 70-day chain never leaves bounds while Uri's leaves at 30 d. With n = 1 member per
lead this cannot separate "that lead is unstable" from "that member diverged", which is a
finding about the experiment's design rather than a caveat on its result - the follow-up
needs an ensemble for the STABILITY question too, not only for the skill one.

Two of the three divergences were **silent to every CONUS diagnostic**: a textbook CONUS
troposphere on top of a polar stratosphere at 1211 K. The carefully calibrated panel caught
1 of 3; the crude global bounds check caught 3 of 3. The difference is not sophistication,
it is DOMAIN - item 2 is the only item evaluated globally and on every variable and level.
AI+RES's own observable ``A_L`` is a CONUS box mean and would have cloned that walker
happily, so any long-lead production must gate on the global state, not the observable.

What the extension found (job 1194, 2026-08-28)
------------------------------------------------
**FCN3 stable through week 14 (98 d) - every rollout, both arms.** The scorer's limit is
measured now, and it is event-dependent: every Uri rollout is clean to 140 d, while the
PNW chains diverge at week 16 and beyond - the ERA5-IC arm at 75 d of the 112-day chain
and 129 d of the 140-day one (clean to 126 d at week 18), the adapter-fed arm at a
consistent 105-111 d on all three. So FCN3's divergence is not monotonic in lead either,
which is the same n = 1 ambiguity the walker showed. ``stability.csv`` and
``stability.png`` carry all 18 chains; ``aires/HANDOFF.md`` has the table.

The ensemble follow-up (``--ens``, ``astab/ensemble.py``)
--------------------------------------------------------
The finding above is a finding about the EXPERIMENT: with n = 1 member per lead, "PNW's
70-day chain never leaves bounds while its 56-day chain blew up at 45 d" cannot be told
apart from "that member diverged". So the follow-up puts an ensemble on the STABILITY
question::

    python -m astab.run --stage plan --ens 100 --event PNW_HeatDome_2021 --weeks 8 --nshards 64
    sbatch slurm/aires_stab_ens.slurm                       # 8 nodes x 8 GPUs, ~2.5 h
    python -m astab.run --stage reduce --stage figs --ens 100 --event PNW_HeatDome_2021 --weeks 8

100 members of ONE (event, lead), GenCast only, differing only in their diffusion noise
stream. The deliverable is a survival curve - the fraction still inside the physical
bounds at each of the 19 three-day checkpoints, with a 95% Wilson interval - plus the
loud/silent and worst-variable/level breakdown. The same mechanism runs at another lead
with only ``--weeks`` changed, which is how the longest STABLE lead gets found before an
AI+RES production run is booked at one.

**Seeds, and the one control this design buys.** ``walker.segment_key(index, step)`` is
``fold_in(fold_in(PRNGKey(WALKER_BASE_SEED), step), index)`` - the same family
``xres/xinference.py`` and ``gencast_s2s/inference.py`` use for their ensembles - so
member index -> a genuine GenCast diffusion draw. ``Chain.seed`` is the walker SLOT for
the frozen 18 and the MEMBER for an ensemble chain, and the overlap that creates is
deliberate: **member 2 at week 8 draws slot 2's stream at the same init, which IS job
1189's PNW@wk8 chain.** It is a DETERMINISM CONTROL, not an independent draw - bf16/XLA
is not bit-reproducible (the 1189 re-walk gave 297.43 K against 297.42 K), so whether it
blows up at 45 d again is itself a finding about sensitivity. It is counted once, like
any other member: exchangeable a priori. Members 0/1/3 share streams with the week
4/6/10 frozen chains, which is harmless - different inits.

**What the run is allowed to touch.** Nothing of job 1189's. A member chain writes only
under ``ens/wk<NN>/m<MMM>/``; its FCN3 path helpers RAISE rather than alias the frozen
cubes; ``--ens`` refuses ``score``/``prune``/``maps`` outright and refuses a lead outside
``STAB_WALK_WEEKS`` (where the walker rolls 3 d and 100 members would report "clean" off
three days of a 140-day chain). ``astab/tests/test_ensemble.py`` pins each of those.

**What it costs, and what pays for it.** 112 GenCast steps per member x 100 = ~115 GPU-h,
~2.5 h on all 8 nodes (~4.5 h on 4). On disk it is ~1 GB, because the walk SLIMS each member's ``diag.nc``
to T2m once its record is written and RELEASES the states at the chain's end - unslimmed
those diagnostics are 65 GB and unreleased the states are 906 GB, against 267 GB free on
a 98%-full NFS. The slim is irreversible for non-T2m CONUS diagnostics, which is why it
runs strictly after ``ensure_record(force=True)`` and why the surviving variable list is
written into both the file and the record. One member's final state is kept: member 2's.

**What the answer will not cover**, whatever it turns out to be: one event, one lead, one
initial condition. The spread is GenCast's sampling spread with no IC perturbation, so it
measures the model's numerical stability under its own diffusion noise. PNW@wk8 was the
LOUD divergence; Uri@wk8 is the companion run that would make the result general.

**RESULT (job 1197, 2026-09-04): 9 of 100 members diverged by 56 d - 9%, Wilson 95%
5-16% - and every member was clean through 36 d** (never diverged: 99% at 39 d, 97% at
42-45, 95% at 48, 93% at 51, 91% at 56). Onsets 39-56 d, median 48 d, none recovered. One
failure mode, nine times: negative specific humidity at 850 hPa first (past the -5e-3
floor at 39-54 d), then the stratospheric winds at 50/100 hPa. ALL NINE LOUD - the
CONUS-mean T2m anomaly collapses by 10-25 K within ~9 d of onset - so the silent mode of
job 1189's Uri chains did not occur here, which is not evidence that it cannot. Member 2,
on job 1189's own noise stream, never left the bounds where 1189 diverged at 45 d:
bf16/XLA non-reproducibility alone moves a member across the boundary at this lead. Cost:
113 GPU-h on 32 workers over 4 nodes, 4 h 34 m wall, 33.4 s/step median, 1.1 GB on disk.
Table and discussion in ``aires/HANDOFF.md``.

Stable is not useful
--------------------
Gate 2 measured FCN3's rank-equivalence horizon at ~8.5 d; Gate 3 measured score skill
rho_s = 0.797 at 9 d and 0.959 at 12 d. If both models stay stable to 70 d - or FCN3 to
140 - that is PERMISSION to run AI+RES out there, not a reason to: past the predictability
horizon the score function ranks chaos and resampling adds no information. The extension
makes this sharper, not softer: a 140-day rollout is ~16x past the horizon at which the
score was last shown to rank. Nothing in this package is
compared to ERA5 as if agreement were expected past ~12 d - ERA5 enters only as a
magnitude/sign sanity descriptor, never as "error" or "skill".
"""
