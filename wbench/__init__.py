#!/usr/bin/env python
"""How fast is FourCastNet 3 as the AI+RES WALKER, against the GenCast walker it would replace?

Every AI+RES run so far walks GenCast 0.25 deg and scores with FCN3. This package measures
the cost of the swap - FCN3 as the thing that is propagated, checkpointed, cloned and
killed - before anything is built on it. It answers SPEED and WALKER MECHANICS only. It
does not, and cannot, say whether FCN3 walkers would reach the tail, keep the weights
honest, or pass the Gate 3 score-skill test; those need a production run.

Three layers, each built on the one before:

    1. raw step speed     model resident, compile/load excluded; seconds per simulated
                          day per walker. GenCast steps 12 h, FCN3 steps 6 h, so per-step
                          numbers are never compared directly.
    2. walker segment     what a production leg actually pays per walker: read the parent
                          state, roll 3 d, write the restart state (float32 + zlib, as
                          `aires/walker.py` does) and the CONUS diagnostics. Once with one
                          worker per model, once with 8 identical workers on one node -
                          the a3mega pilot showed contention nearly doubling GenCast's
                          per-step cost (28.6 -> 52 s/step), so the solo number alone
                          would mislead.
    3. case cost          H100-h per AI+RES case at N=32 (acal) and N=64 (pilot), worked
                          out from layers 1+2 plus the measured startup (load + compile).
                          The FCN3 score phase is the same for both walkers and is
                          reported separately, so the walker swap's share is visible.

Four model configs:

    gencast_0p25   the production walker (bf16, serial, one member per GPU)
    gencast_1p0    the cheaper GenCast checkpoint (fp32, as the xres 1p0 members ran)
    fcn3_e2s       FCN3 through earth2studio's iterator - the scorer's code path
    fcn3_native    FCN3 through makani directly, B walkers per forward pass

Why two FCN3 paths
------------------
earth2studio's ``FCN3._forward`` loops over the batch dimension in Python and calls the
network once per member, so ``batch_size > 1`` there buys nothing. makani's network and
its noise state are natively batched. ``fcn3_walker.NativeFCN3`` calls makani directly so
the batch sweep measures what batching can actually buy, and ``checks`` pins that at B=1
it reproduces earth2studio bit-for-bit - without that, a native speedup could be a
different (wrong) computation.

Why FCN3 needs walker checks at all
-----------------------------------
FCN3's stochasticity is an AR(1) spherical-diffusion noise process (makani
``DiffusionNoiseS2``) whose state carries over from step to step. earth2studio re-draws
it from the stationary distribution every time an iterator is created, so a walker
restarted through earth2studio is NOT the same trajectory as one rolled straight through.
A production FCN3 walker must persist the noise state with the atmospheric state.
``checks`` verifies that the native walker does, plus determinism, clone divergence,
batch invariance and on-disk integrity.

Layout
------
    wconfig.py        configs, segment sizes, paths, the a3mega reference numbers
    timing.py         timers, summaries, atomic JSON results (stdlib only, both envs)
    gencast_bench.py  layers 1+2 for GenCast (env: gencast / moe)
    fcn3_walker.py    the batched, restartable FCN3 walker (env: fcn3)
    fcn3_bench.py     layers 1+2 for FCN3, both paths, batch sweep (env: fcn3)
    checks.py         FCN3 walker checks (env: fcn3)
    report.py         aggregate + layer 3 projection -> REPORT.md, results.json/.csv
    vm/               Nebius VM runbook: bundle, setup, run_all.sh

Run it: ``wbench/vm/README.md``.
"""
