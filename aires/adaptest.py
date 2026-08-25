#!/usr/bin/env python
"""Does the adapter cost forecast skill? The ten-case version of the question.

Gate 2 asked whether an adapter-built IC forecasts like a native one, and answered yes on
one event. Scoring both ensembles against observed truth afterwards
(``runs/aires/gate2/skill/``) left a residue: on most ensemble-mean scores the adapter run
was marginally the worse of the two, but every interval crossed zero, and the two tightest
reductions disagreed in sign. One event cannot settle that.

This module settles it the only way one event's worth of members cannot: **across events**.

Why more events and not more members
------------------------------------
The limiting quantity is not ensemble size. At M = 24 the paired bootstrap on the PNW heat
dome puts the CONUS RMSE difference at +0.099 K with a paired s.e. of 0.205 K; pushing that
s.e. below the effect would need ~M = 100 on a single event, and would still be one draw of
one atmosphere. The signs of the differences across *independent events* are the statistic
that actually has power here, because under a true null each event's sign is a fair coin
and six of them is a real test. All six events already have native FCN3 cubes on disk from
the FCN3-vs-GenCast experiment, so only the adapter halves cost GPU time.

    PNW_HeatDome_2021     t2m_anom     heat        (Gate 2's event; adapter cube exists)
    HurricaneIan_2022     u850_speed   hurricane
    WinterStorm_Uri_2021  t2m_anom     cold
    p90_20231107          t2m_anom     p90
    p90_20240802          t2m_anom     p90
    p90_20251224          t2m_anom     p90

And the null controls
---------------------
All six of those are extremes by construction, which makes them a biased sample for this
particular question: they are exactly the cases where the atmosphere is least stable, so a
small IC perturbation has its best chance of growing into a visible skill difference. A
positive pooled effect measured only on extremes cannot distinguish "the adapter's IC error
costs skill" from "the adapter's IC error costs skill WHEN the flow is already unstable".

So four quiet days are scored the same way, taken from the month-matched control set
already frozen in ``p90/cases.csv`` -- the four whose CONUS-mean daily T2m anomaly is
nearest zero, one per year and one per season:

    null_20221123         t2m_anom     null        +0.001 K   autumn 2022
    null_20230127         t2m_anom     null        -0.138 K   winter 2023
    null_20240329         t2m_anom     null        +0.033 K   spring 2024
    null_20250526         t2m_anom     null        +0.198 K   late spring 2025

These cost twice the GPU time of an event, because unlike the six they have no native FCN3
cube on disk -- ``slurm/aires_adaptest.slurm`` rolls both halves for them. They buy three
things: four more independent signs for the sign test, a tighter pooled CI, and the
extreme-vs-null CONTRAST, which is the only statistic here that can say whether the effect
is regime-dependent. The contrast is reported alongside each group's own pooled estimate,
because two intervals that both cross zero can still differ from each other and two that
both exclude zero can still be the same effect.

What is measured
----------------
Per event, per region, both ensembles are scored against the cached observed truth for that
event's OWN metric. The differences are then made dimensionless before pooling, because a
wind-speed event and a temperature event cannot be averaged in their native units:

    delta_rel = (adapter_score - native_score) / native_score      lower score is better,
                                                                   so delta_rel > 0 == worse

Three statistics come out of it, in increasing order of how much they can say:

    per-event delta   with a PAIRED bootstrap s.e. over the 24 matched member pairs.
                      Resampling pairs (not members independently) is the whole point:
                      adapter member i and native member i share an internal-noise stream,
                      so the pairing removes the ensemble's own sampling noise from the
                      comparison. Break it and the s.e. roughly doubles.

    sign test         across cases, on the sign of the per-case delta. Exact binomial,
                      two-sided, reported per group and pooled. This is the test with real
                      power, and it is the reason the module exists. Six cases cannot get
                      below p = 0.031 even if unanimous; ten can reach p = 0.002.

    pooled delta_rel  inverse-variance weighted mean of the per-event delta_rel, with a
                      CI and a heterogeneity check. Gives the effect a SIZE, not just a
                      direction.

Traps this design is avoiding
-----------------------------
- **A tally of correlated statistics is not evidence.** The single-event analysis leaned
  "adapter worse" on ten of fourteen signed comparisons, but all fourteen came from the
  same 24 member pairs at the same init, so the tally carried roughly one event's worth of
  information. Only the across-event sign test adds independent draws. Everything here
  reports ONE signed statistic per event per region and pools that.
- **Direction was never a coin flip a priori.** The adapter IC carries real added error
  against the native one (Gate 1: 0.63 m/s on u100m/v100m, 4.6% on tcwv), and adding IC
  error cannot improve a forecast in expectation. A small positive pooled delta is the
  PRIOR. The result worth reporting is therefore its size and its upper bound, not the
  discovery of its sign.
- **Skill scores need an anomaly.** ``t2m_anom`` has a natural zero-skill reference (the
  climatology forecast is the zero field). ``u850_speed`` does not - a zero wind field is
  not a climatology - so no skill score is reported for Ian, only raw RMSE/CRPS and the
  relative delta, which is what pools anyway.
- **Cold events need no sign handling.** RMSE and CRPS are sign-blind, and the A_L
  comparison uses ``|A_L - observed|``, so Uri's negative tail needs no special case. This
  is stated because ``xres.xmetrics.extreme_sign`` exists and it is tempting to reach for.

Stages (two conda envs, as everywhere in this repo)
---------------------------------------------------
    plan    either env, no GPU  : readiness matrix + what is missing + the GPU budget
    prep    login node, moe     : build the adapter ICs. NO GPU, no network.
    score   login node, moe     : per-case skill vs truth + paired bootstrap
    report  login node, moe     : pool per group + overall, sign test, contrast, CSV/JSON
    maps    login node, moe     : the same comparison drawn -- observed / native / adapter,
                                  both error fields, and the adapter-minus-native field

``prep`` delegates to ``aires.gate2.stage_prep`` and ``infer`` is ``aires.gate2``'s as
well - there is no second adapter-IC builder and no second FCN3 driver in this repo, on
purpose. The GPU half is ``slurm/aires_adaptest.slurm``.

    PYTHONPATH=. python -m aires.adaptest --stage plan
    bash scripts/adaptest_prep.sh                               # login node, all inputs
    sbatch slurm/aires_adaptest.slurm                           # a3mega, all ten cases
    PYTHONPATH=. python -m aires.adaptest --stage score
    PYTHONPATH=. python -m aires.adaptest --stage report
    PYTHONPATH=. python -m aires.adaptest --stage maps

``--events`` takes names or a group: ``--events null`` is the four controls,
``--events extreme`` the six events.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from fcn3 import fevents as F

# Every case this module scores, in the frozen fevents order: the six extreme events
# first (indices 0-5, seeds unchanged), then the four null controls (6-9). Everything
# downstream -- the Slurm array index, the CSV row order, the figure panel order -- keys
# off this one tuple.
EVENTS = tuple(F.ALL_ORDER)

# The two families the pooled statistics are reported for separately. An "extreme" is one
# of the six events the experiment was originally built on; a "null" is a quiet day. The
# split exists because pooling them into one number would answer neither question: whether
# the adapter costs skill when it matters, and whether whatever it costs is a generic
# property of the IC error or something the extremes' instability manufactures.
GROUPS = {"extreme": tuple(F.ORDER), "null": tuple(F.CONTROL_ORDER)}


def group_of(name: str) -> str:
    return "null" if name in F.CONTROLS else "extreme"

# Statistics pooled across events. Both are positive-definite and lower-is-better, which
# is what makes (adapter - native) / native a meaningful signed relative effect.
POOLED_STATS = ("rmse_ens_mean", "crps")

N_BOOTSTRAP = int(os.environ.get("AIRES_ADAPTEST_DRAWS", 2000))
BOOTSTRAP_SEED = 20260819          # fixed so a rerun reproduces the published CIs


# --------------------------------------------------------------------------- #
# Paths. A separate tree, so runs/aires/gate2/ stays exactly as Gate 2 left it.
# --------------------------------------------------------------------------- #
ADAPTEST_ROOT = A.RUNS / "aires" / "adaptest"


def score_path(ev: F.Event) -> Path:
    return ADAPTEST_ROOT / "scores" / f"{ev.name}_skill.json"


def report_json_path() -> Path:
    return ADAPTEST_ROOT / "adaptest.json"


def report_csv_path() -> Path:
    return ADAPTEST_ROOT / "adaptest_events.csv"


def hrrr_truth_path(ev: F.Event) -> Path:
    """HRRR observed truth, same naming as the ERA5 one. Not built for every event."""
    return A.RUNS / "observations" / f"{ev.name}_hrrr_verif_{ev.metric}.nc"


def ensure_dirs() -> None:
    for d in (ADAPTEST_ROOT / "scores", ADAPTEST_ROOT):
        d.mkdir(parents=True, exist_ok=True)


def _events(sel: str | None) -> list[F.Event]:
    """Cases to act on. ``sel`` accepts names, or the group keys ``extreme`` / ``null``."""
    if not sel:
        return [F.ALL_EVENTS[n] for n in EVENTS]
    want: list[str] = []
    for tok in (t.strip() for t in sel.split(",")):
        if not tok:
            continue
        want.extend(GROUPS[tok]) if tok in GROUPS else want.append(tok)
    bad = [w for w in want if w not in F.ALL_EVENTS]
    if bad:
        raise SystemExit(f"unknown event(s) {bad}; known: {', '.join(EVENTS)} "
                         f"(or a group: {', '.join(GROUPS)})")
    return [F.ALL_EVENTS[w] for w in want]


# --------------------------------------------------------------------------- #
# Statistics. Pure functions, no I/O - these are what aires/tests/test_adaptest.py pins.
# --------------------------------------------------------------------------- #
def area_weights(lat: np.ndarray, nlon: int) -> np.ndarray:
    """cos(lat) weights broadcast to (nlat, nlon). Matches ``aires.aindex.area_mean``."""
    return np.cos(np.deg2rad(np.asarray(lat, dtype="float64")))[:, None] * np.ones((1, nlon))


def _wmean(field: np.ndarray, w: np.ndarray) -> float:
    return float((field * w).sum() / w.sum())


def crps_spread(members: np.ndarray) -> np.ndarray:
    """``(1 / 2M^2) * sum_ij |x_i - x_j|`` per grid point, in O(M log M) not O(M^2).

    The obvious implementation builds the full M x M pairwise-difference array. At M = 24
    on the CONUS grid that is a 57 MB temporary per call, and the paired bootstrap calls
    this 2 x 2000 times per region per truth source -- 4.2 hours for ten cases, measured.
    It is also entirely avoidable, because for SORTED values the pairwise sum collapses:

        sum_{i<j} |x_i - x_j|  =  sum_k (2k - M + 1) * x_(k)        k = 0 .. M-1

    (each x_(k) exceeds the k values below it and is exceeded by the M-1-k above it, so it
    enters with coefficient k - (M-1-k)). That is the SAME NUMBER, not an approximation --
    ``test_crps_spread_matches_the_pairwise_definition`` pins it against the brute-force
    form, and the two agree to float tolerance.
    """
    m = members.shape[0]
    xs = np.sort(members, axis=0)
    k = np.arange(m).reshape((m,) + (1,) * (members.ndim - 1))
    return ((2 * k - m + 1) * xs).sum(axis=0) / (m * m)


def ensemble_scores(members: np.ndarray, obs: np.ndarray, w: np.ndarray) -> dict:
    """Deterministic + probabilistic scores of one ensemble against one truth field.

    ``members`` is (M, nlat, nlon); ``obs`` is (nlat, nlon); ``w`` is the area weight.
    CRPS is the fair (unbiased) ensemble form
    ``mean_i |x_i - y| - (1 / 2M^2) sum_ij |x_i - x_j|``, evaluated per grid point and then
    area-averaged - not the CRPS of the area mean, which is a different and much smaller
    number.
    """
    em = members.mean(axis=0)
    m = members.shape[0]
    t1 = np.abs(members - obs).mean(axis=0)
    t2 = crps_spread(members)
    al = _wmean(em, w)
    al_members = np.array([_wmean(x, w) for x in members])
    return dict(
        rmse_ens_mean=math.sqrt(_wmean((em - obs) ** 2, w)),
        bias_ens_mean=_wmean(em - obs, w),
        crps=_wmean(t1 - t2, w),
        rmse_member_mean=float(np.mean([math.sqrt(_wmean((x - obs) ** 2, w)) for x in members])),
        spread=math.sqrt(_wmean(members.var(axis=0, ddof=1), w)),
        AL=al,
        AL_sd=float(al_members.std(ddof=1)),
        AL_abs_err=abs(al - _wmean(obs, w)),
        n_members=int(m),
    )


def paired_bootstrap(adapter: np.ndarray, native: np.ndarray, obs: np.ndarray,
                     w: np.ndarray, *, stats=POOLED_STATS,
                     n_draws: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """Bootstrap the adapter-minus-native difference over MATCHED member indices.

    One draw resamples member indices with replacement ONCE and applies the same indices
    to both ensembles. That is what keeps the pairing: the two ensembles share internal
    noise streams member for member, so a common index set removes that shared noise from
    the difference instead of adding two independent copies of it.

    Returns, per statistic: the full-ensemble delta, the bootstrap s.e., a percentile CI,
    and the fraction of draws whose sign disagrees with the full-ensemble delta (a direct,
    assumption-free read on whether even the DIRECTION is established).
    """
    if adapter.shape != native.shape:
        raise ValueError(f"member grids differ: {adapter.shape} vs {native.shape}")
    m = adapter.shape[0]
    rng = np.random.default_rng(seed)

    def delta(idx):
        a = ensemble_scores(adapter[idx], obs, w)
        n = ensemble_scores(native[idx], obs, w)
        return np.array([a[s] - n[s] for s in stats])

    obs_delta = delta(np.arange(m))
    draws = np.array([delta(rng.integers(0, m, m)) for _ in range(n_draws)])

    out = {}
    for k, s in enumerate(stats):
        d, col = float(obs_delta[k]), draws[:, k]
        out[s] = dict(
            delta=d,
            se=float(col.std(ddof=1)),
            ci_lo=float(np.percentile(col, 2.5)),
            ci_hi=float(np.percentile(col, 97.5)),
            frac_sign_flips=float((np.sign(col) != np.sign(d)).mean()),
            n_draws=int(n_draws),
        )
    return out


def sign_test(deltas) -> dict:
    """Exact two-sided binomial test on the signs of independent per-event deltas.

    Under the null "the adapter changes nothing", each event's sign is a fair coin. Zeros
    are dropped rather than split, and the count is reported so a reader can see how many
    events actually voted.
    """
    d = np.asarray([x for x in deltas if x != 0.0], dtype="float64")
    n = int(d.size)
    k = int((d > 0).sum())                      # events where the adapter scored WORSE
    if n == 0:
        return dict(n=0, n_worse=0, p_two_sided=float("nan"))
    # exact two-sided p: sum of binomial pmf over outcomes at least as extreme as k
    pmf = np.array([math.comb(n, i) for i in range(n + 1)], dtype="float64") / 2.0 ** n
    p = float(pmf[pmf <= pmf[k] + 1e-12].sum())
    return dict(n=n, n_worse=k, p_two_sided=min(1.0, p))


def pooled_effect(deltas, ses) -> dict:
    """Inverse-variance weighted mean of the per-event relative deltas.

    Weighting by 1/se^2 is the right pooling when each event measures the same underlying
    effect with a different precision. ``q`` and ``i_squared`` report whether that
    same-effect assumption survives the data: a large I^2 means the events disagree by
    more than their own error bars, and the pooled number should then be read as a
    location estimate rather than as one effect measured six times.
    """
    d = np.asarray(deltas, dtype="float64")
    s = np.asarray(ses, dtype="float64")
    keep = np.isfinite(d) & np.isfinite(s) & (s > 0)
    d, s = d[keep], s[keep]
    n = int(d.size)
    if n == 0:
        return dict(n=0, pooled=float("nan"), se=float("nan"))
    wt = 1.0 / s ** 2
    mu = float((wt * d).sum() / wt.sum())
    se = float(math.sqrt(1.0 / wt.sum()))
    q = float((wt * (d - mu) ** 2).sum())
    df = max(n - 1, 0)
    i2 = float(max(0.0, (q - df) / q) * 100.0) if q > 0 else 0.0
    return dict(n=n, pooled=mu, se=se, ci_lo=mu - 1.96 * se, ci_hi=mu + 1.96 * se,
                q=q, df=df, i_squared=i2,
                unweighted_mean=float(d.mean()),
                detection_limit=1.96 * se)


def group_contrast(extreme: dict, null: dict) -> dict:
    """Two-sample z-test between the two groups' pooled effects.

    This is the statistic the null controls were added to produce, and it answers a
    question neither group's own CI does: is the adapter's cost DIFFERENT on extremes than
    on quiet days? Two intervals that both cross zero can still differ from each other, and
    two that both exclude zero can still be the same effect - so "both CIs overlap" is not
    an answer and is not what this computes.

    Takes two ``pooled_effect`` results. The pooled estimates are independent (disjoint
    cases), so the difference's variance is the sum of theirs.
    """
    if not (extreme.get("n") and null.get("n")):
        return dict(n_extreme=extreme.get("n", 0), n_null=null.get("n", 0),
                    diff=float("nan"), se=float("nan"), z=float("nan"),
                    p_two_sided=float("nan"), regime_dependent=False)
    d = extreme["pooled"] - null["pooled"]
    se = math.sqrt(extreme["se"] ** 2 + null["se"] ** 2)
    z = d / se if se > 0 else float("nan")
    p = math.erfc(abs(z) / math.sqrt(2.0)) if se > 0 else float("nan")
    return dict(n_extreme=extreme["n"], n_null=null["n"],
                extreme=extreme["pooled"], null=null["pooled"],
                diff=d, se=se, z=z, p_two_sided=p,
                ci_lo=d - 1.96 * se, ci_hi=d + 1.96 * se,
                regime_dependent=bool(p < 0.05))


# --------------------------------------------------------------------------- #
# plan - what exists, what is missing, what the GPU half costs
# --------------------------------------------------------------------------- #
def stage_plan(events: list[F.Event]) -> int:
    from aires import gate2 as G2

    rows, missing_ic, missing_cube = [], [], []
    for ev in events:
        src = F.gencast_inputs_path(ev)
        native_ic = F.ic_path(ev)
        native_cube = F.cube_path(ev)
        adapter_ic = G2.adapter_ic_path(ev)
        adapter_cube = G2.adapter_cube_path(ev)
        truth = F.era5_truth_path(ev)
        hrrr = hrrr_truth_path(ev)
        if not adapter_ic.exists():
            missing_ic.append(ev)
        elif not adapter_cube.exists():
            missing_cube.append(ev)
        rows.append(dict(event=ev.name, metric=ev.metric, family=ev.family,
                         group=group_of(ev.name),
                         init=str(ev.init.date()), peak=ev.peak,
                         gencast_inputs=src.exists(), native_ic=native_ic.exists(),
                         native_cube=native_cube.exists(),
                         adapter_ic=adapter_ic.exists(), adapter_cube=adapter_cube.exists(),
                         era5_truth=truth.exists(), hrrr_truth=hrrr.exists()))

    df = pd.DataFrame(rows)
    tick = lambda b: "yes" if b else "--"
    print("\n  READINESS  (runs/aires/adaptest)\n")
    print(f"  {'idx':>3s} {'case':22s} {'group':8s} {'metric':11s} {'gc-in':>6s} "
          f"{'nat-ic':>7s} {'nat-cube':>9s} {'ada-ic':>7s} {'ada-cube':>9s} "
          f"{'era5':>5s} {'hrrr':>5s}")
    for r in rows:
        print(f"  {EVENTS.index(r['event']):>3d} {r['event']:22s} {r['group']:8s} "
              f"{r['metric']:11s} {tick(r['gencast_inputs']):>6s} "
              f"{tick(r['native_ic']):>7s} {tick(r['native_cube']):>9s} "
              f"{tick(r['adapter_ic']):>7s} {tick(r['adapter_cube']):>9s} "
              f"{tick(r['era5_truth']):>5s} {tick(r['hrrr_truth']):>5s}")

    # A missing native CUBE is GPU work the Slurm job does, not a blocker -- the four
    # null controls have never been rolled. What genuinely blocks is a missing input that
    # only the login node can build: init frames, the native IC, or the observed truth.
    blocked = [r["event"] for r in rows
               if not (r["gencast_inputs"] and r["native_ic"] and r["era5_truth"])]
    need_native = [r["event"] for r in rows if r["era5_truth"] and not r["native_cube"]]
    print()
    if blocked:
        print(f"  BLOCKED (an input the adapter test cannot build itself): {blocked}")
        print( "    login node:  bash scripts/adaptest_prep.sh")
    if need_native:
        print(f"  native (ERA5-IC) cube missing for {len(need_native)}: "
              f"{', '.join(need_native)}")
        print( "    the Slurm job rolls it first, then the adapter half -- ~2x the cost")
    if missing_ic:
        print(f"  prep needed for {len(missing_ic)} event(s): "
              f"{', '.join(e.name for e in missing_ic)}")
        print( "    PYTHONPATH=. python -m aires.adaptest --stage prep     # login node, moe")
    need_gpu = [e for e in events if not G2.adapter_cube_path(e).exists()]
    if need_gpu:
        idx = ",".join(str(EVENTS.index(e.name)) for e in need_gpu)
        print(f"  infer needed for {len(need_gpu)} event(s): "
              f"{', '.join(e.name for e in need_gpu)}")
        print(f"    sbatch --array={idx}%2 slurm/aires_adaptest.slurm     # a3mega")
        mins = sum(30 if e.name in set(need_native) else 15 for e in need_gpu)
        print(f"    budget: ~15 min of one 8xH100 node per adapter-only case, ~30 min "
              f"where the native half is missing too")
        print(f"            (~9 min of each is the shared 4 GB model load, not compute)"
              f"  ->  ~{mins} node-min total")
    if not missing_ic and not need_gpu:
        print("  all adapter cubes present - go straight to:")
        print("    PYTHONPATH=. python -m aires.adaptest --stage score")
    print()
    return 1 if blocked else 0


# --------------------------------------------------------------------------- #
# prep - adapter ICs (delegates to Gate 2's builder; there is only one)
# --------------------------------------------------------------------------- #
def stage_prep(events: list[F.Event], force: bool = False) -> int:
    from aires import gate2 as G2

    ensure_dirs()
    rc = 0
    for ev in events:
        print(f"\n[prep] ---- {ev.name} ({ev.metric}) ----")
        r = G2.stage_prep(ev, force=force)
        if r:
            print(f"[prep] {ev.name}: FAILED", file=sys.stderr)
            rc = 1
    return rc


# --------------------------------------------------------------------------- #
# score - per-event skill against truth, with the paired bootstrap
# --------------------------------------------------------------------------- #
def _truth_field(path: Path, metric: str, like: xr.DataArray) -> xr.DataArray:
    """Cached observed field, aligned onto the cube's own grid.

    The cached truth and the cubes are built on the same 0.25 degree CONUS grid, but a
    float32 coordinate round trip can leave them differing in the last bit, which makes
    xarray's automatic alignment silently produce an all-NaN difference. A nearest
    reindex with a tight tolerance either fixes that or raises - never quietly drops rows.
    """
    da = xr.open_dataset(path)[metric]
    return da.reindex_like(like, method="nearest", tolerance=1e-4)


def verification_fields(ev: F.Event) -> tuple[dict, int, dict]:
    """The two ensembles' verification-week fields, member-aligned.

    Returns ``({"adapter": da, "native": da}, n_members, frames)`` where each ``da`` is
    ``(member, lat, lon)``. Shared by ``score`` and ``maps`` so a figure can never be drawn
    from a different reduction than the number printed beside it.
    """
    from aires import aindex as AI
    from aires import gate2 as G2
    from xres import xmetrics as XM

    ada_p, nat_p = G2.adapter_cube_path(ev), F.cube_path(ev)
    for path, what in ((ada_p, "adapter cube"), (nat_p, "native cube")):
        if not path.exists():
            raise FileNotFoundError(f"{ev.name}: missing {what} {path}")

    cubes = {"adapter": xr.open_dataset(ada_p), "native": xr.open_dataset(nat_p)}
    fields, frames = {}, {}
    for k, c in cubes.items():
        frames[k] = AI.check_window(c, ev.peak, ev.metric, where=f"{k} cube {ev.name}")
        fields[k] = XM.forecast_field(c, ev.peak, ev.metric).compute()

    m = min(fields["adapter"].sizes["member"], fields["native"].sizes["member"])
    if m != F.N_MEMBERS:
        print(f"  [warn] {ev.name}: comparing {m} members, not {F.N_MEMBERS} - "
              f"the seed matching assumes the full 8x3 shard split")
    for k in fields:
        fields[k] = fields[k].isel(member=slice(0, m)).transpose("member", "lat", "lon")
    return fields, m, frames


def _score_one(ev: F.Event, plot: bool = False) -> dict:
    from aires import aindex as AI

    fields, m, frames = verification_fields(ev)

    truths = {"era5": F.era5_truth_path(ev)}
    if hrrr_truth_path(ev).exists():
        truths["hrrr"] = hrrr_truth_path(ev)

    res = dict(event=ev.name, metric=ev.metric, family=ev.family, label=ev.label,
               group=group_of(ev.name),
               init=str(ev.init), peak=ev.peak, n_members=m, window_frames=frames,
               is_anomaly=ev.metric == "t2m_anom",
               boxes={}, runs={}, reference={}, paired={}, truth_sources=list(truths))

    for rk, box in AI.boxes_for(ev.name).items():
        res["boxes"][rk] = dict(name=box.name, label=box.label, lat=box.lat, lon=box.lon)

    for tk, tpath in truths.items():
        for rk, box in AI.boxes_for(ev.name).items():
            A_da = AI.crop(fields["adapter"], box) if box is not AI.CONUS else fields["adapter"]
            N_da = AI.crop(fields["native"], box) if box is not AI.CONUS else fields["native"]
            O_da = _truth_field(tpath, ev.metric, A_da.isel(member=0, drop=True))
            O_da = O_da.transpose("lat", "lon")

            w = area_weights(A_da["lat"].values, A_da.sizes["lon"])
            av, nv, ov = A_da.values, N_da.values, O_da.values
            if not (np.isfinite(av).all() and np.isfinite(nv).all() and np.isfinite(ov).all()):
                raise ValueError(f"{ev.name}/{tk}/{rk}: non-finite values after alignment")

            key = f"{tk}/{rk}"
            res["runs"].setdefault("adapter", {})[key] = ensemble_scores(av, ov, w)
            res["runs"].setdefault("native", {})[key] = ensemble_scores(nv, ov, w)
            res["paired"][key] = paired_bootstrap(av, nv, ov, w)
            # relative (dimensionless) deltas - what pools across metrics
            for s in POOLED_STATS:
                base = res["runs"]["native"][key][s]
                p = res["paired"][key][s]
                p["delta_rel"] = p["delta"] / base if base else float("nan")
                p["se_rel"] = p["se"] / base if base else float("nan")
                p["native_base"] = base
            # zero-skill reference: only meaningful where the field IS an anomaly
            if res["is_anomaly"]:
                res["reference"][key] = dict(
                    rmse=math.sqrt(_wmean(ov ** 2, w)), crps=_wmean(np.abs(ov), w),
                    AL=_wmean(ov, w), kind="climatology (zero anomaly)")
            else:
                res["reference"][key] = dict(rmse=None, crps=None, AL=_wmean(ov, w),
                                             kind="none (metric is not an anomaly)")
    return res


def stage_score(events: list[F.Event]) -> int:
    ensure_dirs()
    rc = 0
    for ev in events:
        try:
            res = _score_one(ev)
        except FileNotFoundError as e:
            print(f"[score] skip {ev.name}: {e}")
            rc = 1
            continue
        score_path(ev).write_text(json.dumps(res, indent=1))
        key = "era5/box"
        p = res["paired"][key]["rmse_ens_mean"]
        print(f"[score] {ev.name:22s} {ev.metric:11s} "
              f"box RMSE {res['runs']['adapter'][key]['rmse_ens_mean']:7.3f} vs "
              f"{res['runs']['native'][key]['rmse_ens_mean']:7.3f}  "
              f"delta {p['delta']:+7.3f} +- {p['se']:.3f} "
              f"({p['delta_rel'] * 100:+6.1f}%)  -> {score_path(ev).name}")
    return rc


# --------------------------------------------------------------------------- #
# report - pool across events; this is the verdict
# --------------------------------------------------------------------------- #
def stage_report(events: list[F.Event], plot: bool = True) -> int:
    ensure_dirs()
    have = [ev for ev in events if score_path(ev).exists()]
    if not have:
        print("[report] no per-event scores yet - run --stage score first", file=sys.stderr)
        return 1
    per = {ev.name: json.loads(score_path(ev).read_text()) for ev in have}

    rows = []
    for ev in have:
        r = per[ev.name]
        for key in sorted(r["paired"]):
            tk, rk = key.split("/")
            for s in POOLED_STATS:
                p = r["paired"][key][s]
                rows.append(dict(event=ev.name, family=r["family"],
                                 group=r.get("group", group_of(ev.name)),
                                 metric=r["metric"], truth=tk, region=rk, stat=s,
                                 adapter=r["runs"]["adapter"][key][s],
                                 native=r["runs"]["native"][key][s],
                                 delta=p["delta"], se=p["se"],
                                 delta_rel=p["delta_rel"], se_rel=p["se_rel"],
                                 ci_lo=p["ci_lo"], ci_hi=p["ci_hi"],
                                 frac_sign_flips=p["frac_sign_flips"]))
    df = pd.DataFrame(rows)
    df.to_csv(report_csv_path(), index=False)

    n_ext = sum(1 for ev in have if group_of(ev.name) == "extreme")
    n_null = len(have) - n_ext
    verdict = {"n_events": len(have), "events": [ev.name for ev in have],
               "n_extreme": n_ext, "n_null": n_null,
               "groups": {g: [n for n in names if n in {e.name for e in have}]
                          for g, names in GROUPS.items()},
               "n_members": {ev.name: per[ev.name]["n_members"] for ev in have},
               "n_bootstrap_draws": N_BOOTSTRAP, "pooled": {}, "contrast": {}}

    print(f"\n  ADAPTER TEST  ({len(have)} cases: {n_ext} extreme event(s), "
          f"{n_null} null control(s))\n")
    print("  group    truth region stat            pooled delta_rel        "
          "95% CI            I2   sign test")
    for tk in sorted(df["truth"].unique()):
        for rk in sorted(df["region"].unique()):
            for s in POOLED_STATS:
                base = df[(df.truth == tk) & (df.region == rk) & (df.stat == s)]
                if base.empty:
                    continue
                # each group on its own, then everything pooled together
                for gname in ("extreme", "null", "all"):
                    sub = base if gname == "all" else base[base.group == gname]
                    if sub.empty:
                        continue
                    st = sign_test(sub["delta_rel"].tolist())
                    po = pooled_effect(sub["delta_rel"].tolist(), sub["se_rel"].tolist())
                    verdict["pooled"][f"{gname}/{tk}/{rk}/{s}"] = dict(
                        sign_test=st, pooled=po,
                        per_event=dict(zip(sub.event, sub.delta_rel)))
                    res = ("resolved: adapter WORSE" if po["ci_lo"] > 0 else
                           "resolved: adapter BETTER" if po["ci_hi"] < 0 else
                           "not resolved (CI spans zero)")
                    print(f"  {gname:8s} {tk:5s} {rk:6s} {s:14s}  "
                          f"{po['pooled'] * 100:+6.2f}%  "
                          f"[{po['ci_lo'] * 100:+6.2f}, {po['ci_hi'] * 100:+6.2f}]  "
                          f"I2 {po['i_squared']:4.0f}%   "
                          f"{st['n_worse']}/{st['n']} worse  p {st['p_two_sided']:.3f}   "
                          f"{res}")
                # does the effect DIFFER between extremes and quiet days? A two-sample
                # z on the two pooled estimates. This is the question the controls were
                # added to answer, and it is not the same question as either group's own
                # CI: two intervals that both cross zero can still differ from each other,
                # and two that both exclude zero can still be the same effect.
                pe = verdict["pooled"].get(f"extreme/{tk}/{rk}/{s}")
                pn = verdict["pooled"].get(f"null/{tk}/{rk}/{s}")
                if pe and pn:
                    ct = group_contrast(pe["pooled"], pn["pooled"])
                    if np.isfinite(ct["z"]):
                        verdict["contrast"][f"{tk}/{rk}/{s}"] = ct
                        print(f"  {'contrast':8s} {tk:5s} {rk:6s} {s:14s}  "
                              f"{ct['diff'] * 100:+6.2f}%  extreme minus null,  "
                              f"z {ct['z']:+5.2f}  p {ct['p_two_sided']:.3f}   "
                              f"{'regime-DEPENDENT' if ct['regime_dependent'] else 'no regime dependence'}")

    headline = verdict["pooled"].get("all/era5/box/rmse_ens_mean")
    if headline:
        po, st = headline["pooled"], headline["sign_test"]
        verdict["headline"] = dict(
            key="all/era5/box/rmse_ens_mean", pooled_pct=po["pooled"] * 100,
            ci_pct=[po["ci_lo"] * 100, po["ci_hi"] * 100],
            sign_worse=st["n_worse"], sign_n=st["n"], sign_p=st["p_two_sided"],
            detection_limit_pct=po["detection_limit"] * 100,
            resolved=bool(po["ci_lo"] > 0 or po["ci_hi"] < 0))
        print(f"\n  HEADLINE  (ERA5 truth, event box, ensemble-mean RMSE, all "
              f"{len(have)} cases pooled)")
        print(f"    pooled effect        {po['pooled'] * 100:+.2f}%  "
              f"[{po['ci_lo'] * 100:+.2f}, {po['ci_hi'] * 100:+.2f}]")
        print(f"    sign test            {st['n_worse']} of {st['n']} events worse, "
              f"p = {st['p_two_sided']:.3f}")
        print(f"    detection limit      +-{po['detection_limit'] * 100:.2f}% "
              f"(what this test could have seen)")
        print(f"    verdict              "
              f"{'RESOLVED' if verdict['headline']['resolved'] else 'NOT RESOLVED'}")

    report_json_path().write_text(json.dumps(verdict, indent=1))
    print(f"\n  wrote {report_json_path()}")
    print(f"  wrote {report_csv_path()}")
    if plot:
        try:
            _plot(df, verdict)
        except Exception as e:                        # a missing display must not lose the JSON
            print(f"  [warn] figure skipped: {e}")
    return 0


def _plot(df: pd.DataFrame, verdict: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[(df.truth == "era5") & (df.region == "box") & (df.stat == "rmse_ens_mean")]
    # extremes above, controls below, each block sorted -- so the eye can compare the two
    # populations directly instead of hunting for them in one interleaved sort.
    sub = pd.concat([sub[sub.group == g].sort_values("delta_rel")
                     for g in ("null", "extreme") if (sub.group == g).any()])
    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(sub) + 2.4))
    y = np.arange(len(sub))
    colors = ["#2a78d6" if g == "extreme" else "#c26a1e" for g in sub["group"]]
    for yi, (dr, se, c) in enumerate(zip(sub["delta_rel"], sub["se_rel"], colors)):
        ax.errorbar(dr * 100, yi, xerr=se * 100 * 1.96, fmt="o", ms=7, lw=1.6,
                    capsize=4, color=c, ecolor="#7d8f99")
    n_null = int((sub["group"] == "null").sum())
    if 0 < n_null < len(sub):
        ax.axhline(n_null - 0.5, color="#b8c2c8", lw=1.0, ls=":")
        ax.text(ax.get_xlim()[0], n_null - 0.42, " null controls below / extremes above",
                fontsize=7.5, color="#6b7880", va="bottom")
    ax.axvline(0, color="#131a20", lw=1.2, zorder=0)
    h = verdict.get("headline")
    if h:
        ax.axvspan(h["ci_pct"][0], h["ci_pct"][1], color="#2a78d6", alpha=0.12, zorder=0,
                   label=f"pooled {h['pooled_pct']:+.1f}% "
                         f"[{h['ci_pct'][0]:+.1f}, {h['ci_pct'][1]:+.1f}]")
        ax.axvline(h["pooled_pct"], color="#2a78d6", lw=1.4, ls="--", zorder=1)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e}  ({m})" for e, m in zip(sub["event"], sub["metric"])],
                       fontsize=9)
    for lbl, g in zip(ax.get_yticklabels(), sub["group"]):
        lbl.set_color("#2a78d6" if g == "extreme" else "#c26a1e")
    ax.set_xlabel("ensemble-mean RMSE:  (adapter - native) / native   [%]   "
                  "positive = adapter worse")
    ax.set_title("Does the adapter cost skill? Per-case effect, ERA5 truth, event box\n"
                 "blue = extreme event, orange = null control (quiet day)",
                 fontsize=11, loc="left")
    ax.grid(axis="x", color="#e2e9ec")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = A.fig_dir() / "adaptest_effect.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# maps - the same comparison, drawn, so a reader can check the numbers by eye
#
# Every statistic above is an area mean, and an area mean can hide the two things a reader
# most wants to know: WHERE the adapter and the native run disagree, and whether that
# disagreement looks like a coherent structure (a displaced ridge, a shifted storm track)
# or like ensemble noise. A pooled -0.4% means nothing without that.
#
# Per case, two rows on the same CONUS crop the scores use:
#   row 1  observed | native ens-mean | adapter ens-mean   -- one shared field scale, so
#                                                             the three are comparable
#   row 2  native - obs | adapter - obs | adapter - native -- errors on one shared
#                                                             symmetric scale; the
#                                                             adapter-native difference on
#                                                             its OWN (much tighter) scale
#
# The third panel of row 2 gets its own scale on purpose. It is typically an order of
# magnitude smaller than the forecast error, and drawing it on the error scale renders it
# a uniform white rectangle -- technically honest, visually useless. The colorbar label
# says which scale it is on, and the panel title carries the max magnitude.
# --------------------------------------------------------------------------- #
def _sym(*arrays, pct: float = 99.5) -> float:
    """Symmetric colour limit robust to a single outlier grid point."""
    v = np.concatenate([np.abs(np.asarray(a)).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    return float(np.percentile(v, pct)) if v.size else 1.0


def _range(*arrays, pct: float = 99.5) -> tuple[float, float]:
    v = np.concatenate([np.asarray(a).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    if not v.size:
        return 0.0, 1.0
    return float(np.percentile(v, 100 - pct)), float(np.percentile(v, pct))


def _draw(fig, gs_pos, da, title, *, cmap, vmin, vmax, label, ccrs, box=None):
    from gencast_s2s.plotting import _to_180, _add_conus_features
    from gencast_s2s import config as C

    ax = fig.add_subplot(*gs_pos, projection=ccrs.PlateCarree())
    ax.set_extent(C.CONUS_EXTENT, ccrs.PlateCarree())
    d = _to_180(da)
    m = ax.pcolormesh(d.lon, d.lat, d.values, transform=ccrs.PlateCarree(),
                      cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    _add_conus_features(ax)
    if box is not None:
        lo0, lo1 = box.lon[0], box.lon[1]
        lo0, lo1 = (lo0 - 360 if lo0 > 180 else lo0), (lo1 - 360 if lo1 > 180 else lo1)
        ax.plot([lo0, lo1, lo1, lo0, lo0],
                [box.lat[0], box.lat[0], box.lat[1], box.lat[1], box.lat[0]],
                transform=ccrs.PlateCarree(), color="#111111", lw=1.3, ls="--", zorder=6)
    ax.set_title(title, fontsize=8.5)
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.05, shrink=0.92)
    cb.set_label(label, fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    return ax


def maps_one(ev: F.Event, truth: str = "era5"):
    """Draw the six-panel comparison for one case. Returns the output path."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aires import aindex as AI
    from xres import xmetrics as XM

    fields, m, _ = verification_fields(ev)
    tpath = F.era5_truth_path(ev) if truth == "era5" else hrrr_truth_path(ev)
    if not tpath.exists():
        raise FileNotFoundError(f"{ev.name}: no {truth} truth at {tpath}")

    ada = fields["adapter"].mean("member")
    nat = fields["native"].mean("member")
    obs = _truth_field(tpath, ev.metric, ada).transpose("lat", "lon")
    sp = XM.spec(ev.metric)
    units = sp.units

    # the numbers this figure has to agree with: CONUS, whole-field, same weights as score
    w = area_weights(ada["lat"].values, ada.sizes["lon"])
    sa = ensemble_scores(fields["adapter"].values, obs.values, w)
    sn = ensemble_scores(fields["native"].values, obs.values, w)
    d_rel = (sa["rmse_ens_mean"] - sn["rmse_ens_mean"]) / sn["rmse_ens_mean"] * 100

    diff = ada - nat
    ea, en = ada - obs, nat - obs
    if sp.diverging:
        fv = _sym(obs.values, nat.values, ada.values)
        fmin, fmax = -fv, fv
    else:
        fmin, fmax = _range(obs.values, nat.values, ada.values)
    ev_lim = _sym(ea.values, en.values)
    d_lim = _sym(diff.values)

    box = AI.box_for(ev.name)
    box = None if box is AI.CONUS else box

    fig = plt.figure(figsize=(13.2, 7.0))
    row1 = [(obs, f"Observed ({truth.upper()})"),
            (nat, f"Native IC, {m}-member mean"),
            (ada, f"Adapter IC, {m}-member mean")]
    for i, (da, t) in enumerate(row1):
        _draw(fig, (2, 3, i + 1), da, t, cmap=sp.field_cmap, vmin=fmin, vmax=fmax,
              label=f"{sp.label} [{units}]", ccrs=ccrs, box=box)

    _draw(fig, (2, 3, 4), en, f"Native - observed   (RMSE {sn['rmse_ens_mean']:.3f} {units})",
          cmap="RdBu_r", vmin=-ev_lim, vmax=ev_lim, label=f"forecast error [{units}]",
          ccrs=ccrs, box=box)
    _draw(fig, (2, 3, 5), ea, f"Adapter - observed   (RMSE {sa['rmse_ens_mean']:.3f} {units})",
          cmap="RdBu_r", vmin=-ev_lim, vmax=ev_lim, label=f"forecast error [{units}]",
          ccrs=ccrs, box=box)
    _draw(fig, (2, 3, 6),
          diff, f"Adapter - native   (max |diff| {float(np.abs(diff).max()):.2f} {units})",
          cmap="PuOr_r", vmin=-d_lim, vmax=d_lim,
          label=f"IC effect [{units}]  -- NOTE own scale, {d_lim / ev_lim:.0%} of the error scale",
          ccrs=ccrs, box=box)

    grp = group_of(ev.name)
    tag = "NULL CONTROL" if grp == "null" else f"{ev.family} event"
    # The RMSE quoted here is over the WHOLE CONUS crop -- every grid point drawn. The
    # headline in `--stage report` is over the event BOX (the dashed outline, where there
    # is one), which is a different and usually larger number. Both are in the CSV; saying
    # which is which on the figure keeps a reader from reading one as the other.
    fig.suptitle(
        f"{ev.label}   [{tag}]   {ev.metric}   init {str(ev.init)[:10]} -> "
        f"verification week ending {ev.peak}   (21 d lead)\n"
        f"CONUS ensemble-mean RMSE vs {truth.upper()}: native {sn['rmse_ens_mean']:.3f}, "
        f"adapter {sa['rmse_ens_mean']:.3f} {units}   ->   {d_rel:+.2f}% "
        f"({'adapter worse' if d_rel > 0 else 'adapter better'})"
        + (f"    dashed box = {box.name}, the scored region" if box is not None else ""),
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.0)
    out = A.fig_dir(ev.name) / f"adaptest_maps_{ev.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def maps_overview(events: list[F.Event], truth: str = "era5"):
    """One panel per case: the adapter-minus-native ensemble-mean field, side by side.

    Drawn on a PER-PANEL scale (the cases span temperature in K and wind in m/s, and a
    shared scale would be meaningless across them), so read the shape, not the intensity;
    each panel's own limit is printed in its title. What to look for is whether the null
    controls' differences look like the extremes' differences -- same character, same
    order of magnitude -- or whether the extremes are visibly more disturbed.
    """
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from xres import xmetrics as XM

    have = []
    for ev in events:
        try:
            fields, m, _ = verification_fields(ev)
        except FileNotFoundError:
            continue
        have.append((ev, fields["adapter"].mean("member") - fields["native"].mean("member")))
    if not have:
        raise FileNotFoundError("no cases have both cubes yet")

    # Balance the grid rather than filling rows of 5 and stranding a remainder: 10 cases
    # -> 5 x 2, 6 -> 3 x 2, 4 -> 4 x 1.
    nrow = math.ceil(len(have) / 5)
    ncol = math.ceil(len(have) / nrow)
    fig = plt.figure(figsize=(3.05 * ncol, 3.5 * nrow))
    for i, (ev, diff) in enumerate(have):
        lim = _sym(diff.values)
        grp = group_of(ev.name)
        _draw(fig, (nrow, ncol, i + 1), diff,
              f"{'[null] ' if grp == 'null' else ''}{ev.label}\nmax |diff| "
              f"{float(np.abs(diff).max()):.2f} {XM.spec(ev.metric).units}",
              cmap="PuOr_r", vmin=-lim, vmax=lim,
              label=f"adapter - native [{XM.spec(ev.metric).units}]", ccrs=ccrs)
    fig.suptitle("Adapter IC minus native IC: ensemble-mean verification-week field, "
                 "per case (each panel on its own scale)\n"
                 "[null] = quiet-day control; the rest are extreme events",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.5)
    out = A.fig_dir() / "adaptest_diff_maps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def stage_maps(events: list[F.Event], truth: str = "era5") -> int:
    ensure_dirs()
    rc, drawn = 0, []
    for ev in events:
        try:
            out = maps_one(ev, truth=truth)
        except FileNotFoundError as e:
            print(f"[maps] skip {ev.name}: {e}")
            rc = 1
            continue
        drawn.append(ev)
        print(f"[maps] wrote {out}")
    if drawn:
        print(f"[maps] wrote {maps_overview(drawn, truth=truth)}")
    return rc


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["plan", "prep", "score", "report", "maps"])
    ap.add_argument("--events", default=os.environ.get("AIRES_ADAPTEST_EVENTS"),
                    help="comma-separated subset, or a group name (extreme|null); "
                         "default all ten cases")
    ap.add_argument("--truth", default="era5", choices=["era5", "hrrr"],
                    help="maps: which observed truth to draw against")
    ap.add_argument("--force", action="store_true", help="prep: rebuild a cached adapter IC")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    events = _events(a.events)

    if a.stage == "plan":
        return stage_plan(events)
    if a.stage == "prep":
        return stage_prep(events, force=a.force)
    if a.stage == "score":
        return stage_score(events)
    if a.stage == "maps":
        return stage_maps(events, truth=a.truth)
    return stage_report(events, plot=not a.no_plot)


if __name__ == "__main__":
    sys.exit(main())
