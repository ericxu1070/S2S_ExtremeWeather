"""AI+RES calibration: does the rare-event probability estimate verify?

`aires/` answers "how much probability mass does AI+RES put out in the tail for THIS
event". That is a statement about one event, and nine of them cannot say whether the
number is *right* - a probability is only checkable against a population. This package
builds that population and scores it.

    aires/          one event  -> P_forecast(A_L >= a)          (the estimate)
    acal/           many cases -> is P_forecast calibrated?      (the verification)

It stands to `aires/` exactly as `astab/` does: it imports the walker, the adapter, the
scorer, the index and the climatology, and it writes to its own tree, `runs/acal/` +
`figures/acal/`.

It modifies those modules in exactly ONE place, and only because a driver cannot resolve
an event that is not registered: `fcn3/fevents.py` loads this package's `cases.csv` into
`KNOWN`/`SEED_ORDER` (see the acal block there). That load is APPEND-ONLY -- `ORDER`,
`ALL_ORDER` and `RES_ORDER` remain prefixes, so every seed the frozen cubes were rolled
with is unchanged -- and the cases stay out of `selected()`, so no published figure gains
a panel. `aires/aindex.py` is genuinely untouched: `box_for` falls back to CONUS, which is
the correct index for a slate selected on the CONUS-wide mean.

The case slate is a SELECTION RULE, not a curated list. Every 6x6 degree window of the
CONUS crop is scanned over 2021-2026 for 7-day box-mean T2m anomalies crossing +/-2, +/-3
and +/-4 K; the crossings are declustered in space and time and each surviving case gets
one box, one peak date and one magnitude bin. The rule is written down in `acases.py` and
the resulting `cases.csv` is frozen, so the population is reproducible and nobody chose
which events went in.

Read `acal/HANDOFF.md` for what is actually built. Read `ccfg.py` for the knobs.
"""
