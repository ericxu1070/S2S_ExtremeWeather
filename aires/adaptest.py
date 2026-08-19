#!/usr/bin/env python
"""Does the adapter cost forecast skill? The six-event version of the question.

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

    sign test         across events, on the sign of the per-event delta. Exact binomial,
                      two-sided. This is the test with real power, and it is the reason
                      the module exists.

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
    score   login node, moe     : per-event skill vs truth + paired bootstrap
    report  login node, moe     : pool across events, sign test, CSV/JSON/figure

``prep`` delegates to ``aires.gate2.stage_prep`` and ``infer`` is ``aires.gate2``'s as
well - there is no second adapter-IC builder and no second FCN3 driver in this repo, on
purpose. The GPU half is ``slurm/aires_adaptest.slurm``.

    PYTHONPATH=. python -m aires.adaptest --stage plan
    PYTHONPATH=. python -m aires.adaptest --stage prep          # login node
    sbatch slurm/aires_adaptest.slurm                           # a3mega, 5 events
    PYTHONPATH=. python -m aires.adaptest --stage score
    PYTHONPATH=. python -m aires.adaptest --stage report
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

# The six events, in the frozen fevents order. Everything downstream keys off this.
EVENTS = tuple(F.ORDER)

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
    if not sel:
        return [F.EVENTS[n] for n in EVENTS]
    want = [s.strip() for s in sel.split(",") if s.strip()]
    bad = [w for w in want if w not in F.EVENTS]
    if bad:
        raise SystemExit(f"unknown event(s) {bad}; known: {', '.join(EVENTS)}")
    return [F.EVENTS[w] for w in want]


# --------------------------------------------------------------------------- #
# Statistics. Pure functions, no I/O - these are what aires/tests/test_adaptest.py pins.
# --------------------------------------------------------------------------- #
def area_weights(lat: np.ndarray, nlon: int) -> np.ndarray:
    """cos(lat) weights broadcast to (nlat, nlon). Matches ``aires.aindex.area_mean``."""
    return np.cos(np.deg2rad(np.asarray(lat, dtype="float64")))[:, None] * np.ones((1, nlon))


def _wmean(field: np.ndarray, w: np.ndarray) -> float:
    return float((field * w).sum() / w.sum())


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
    t2 = np.abs(members[:, None] - members[None, :]).mean(axis=(0, 1)) / 2.0
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
                         init=str(ev.init.date()), peak=ev.peak,
                         gencast_inputs=src.exists(), native_ic=native_ic.exists(),
                         native_cube=native_cube.exists(),
                         adapter_ic=adapter_ic.exists(), adapter_cube=adapter_cube.exists(),
                         era5_truth=truth.exists(), hrrr_truth=hrrr.exists()))

    df = pd.DataFrame(rows)
    tick = lambda b: "yes" if b else "--"
    print("\n  READINESS  (runs/aires/adaptest)\n")
    print(f"  {'event':22s} {'metric':11s} {'gc-in':>6s} {'nat-ic':>7s} {'nat-cube':>9s} "
          f"{'ada-ic':>7s} {'ada-cube':>9s} {'era5':>5s} {'hrrr':>5s}")
    for r in rows:
        print(f"  {r['event']:22s} {r['metric']:11s} {tick(r['gencast_inputs']):>6s} "
              f"{tick(r['native_ic']):>7s} {tick(r['native_cube']):>9s} "
              f"{tick(r['adapter_ic']):>7s} {tick(r['adapter_cube']):>9s} "
              f"{tick(r['era5_truth']):>5s} {tick(r['hrrr_truth']):>5s}")

    blocked = [r["event"] for r in rows
               if not (r["gencast_inputs"] and r["native_ic"] and r["native_cube"]
                       and r["era5_truth"])]
    print()
    if blocked:
        print(f"  BLOCKED (an input the adapter test cannot build itself): {blocked}")
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
        print(f"    budget: ~15 min of one 8xH100 node per event "
              f"(~9 min is the shared 4 GB model load, not compute)"
              f"  ->  ~{len(need_gpu) * 15} node-min total")
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


def _score_one(ev: F.Event, plot: bool = False) -> dict:
    from aires import aindex as AI
    from aires import gate2 as G2
    from xres import xmetrics as XM

    ada_p, nat_p = G2.adapter_cube_path(ev), F.cube_path(ev)
    for p, what in ((ada_p, "adapter cube"), (nat_p, "native cube")):
        if not p.exists():
            raise FileNotFoundError(f"{ev.name}: missing {what} {p}")

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
        fields[k] = fields[k].isel(member=slice(0, m))

    truths = {"era5": F.era5_truth_path(ev)}
    if hrrr_truth_path(ev).exists():
        truths["hrrr"] = hrrr_truth_path(ev)

    res = dict(event=ev.name, metric=ev.metric, family=ev.family, label=ev.label,
               init=str(ev.init), peak=ev.peak, n_members=m, window_frames=frames,
               is_anomaly=ev.metric == "t2m_anom",
               boxes={}, runs={}, reference={}, paired={}, truth_sources=list(truths))

    for rk, box in AI.boxes_for(ev.name).items():
        res["boxes"][rk] = dict(name=box.name, label=box.label, lat=box.lat, lon=box.lon)

    for tk, tpath in truths.items():
        for rk, box in AI.boxes_for(ev.name).items():
            A_da = AI.crop(fields["adapter"], box) if box is not AI.CONUS else fields["adapter"]
            N_da = AI.crop(fields["native"], box) if box is not AI.CONUS else fields["native"]
            A_da = A_da.transpose("member", "lat", "lon")
            N_da = N_da.transpose("member", "lat", "lon")
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
                rows.append(dict(event=ev.name, family=r["family"], metric=r["metric"],
                                 truth=tk, region=rk, stat=s,
                                 adapter=r["runs"]["adapter"][key][s],
                                 native=r["runs"]["native"][key][s],
                                 delta=p["delta"], se=p["se"],
                                 delta_rel=p["delta_rel"], se_rel=p["se_rel"],
                                 ci_lo=p["ci_lo"], ci_hi=p["ci_hi"],
                                 frac_sign_flips=p["frac_sign_flips"]))
    df = pd.DataFrame(rows)
    df.to_csv(report_csv_path(), index=False)

    verdict = {"n_events": len(have), "events": [ev.name for ev in have],
               "n_members": {ev.name: per[ev.name]["n_members"] for ev in have},
               "n_bootstrap_draws": N_BOOTSTRAP, "pooled": {}}

    print(f"\n  ADAPTER TEST  ({len(have)} events, "
          f"{', '.join(e.family for e in have)})\n")
    for tk in sorted(df["truth"].unique()):
        for rk in sorted(df["region"].unique()):
            for s in POOLED_STATS:
                sub = df[(df.truth == tk) & (df.region == rk) & (df.stat == s)]
                if sub.empty:
                    continue
                st = sign_test(sub["delta_rel"].tolist())
                po = pooled_effect(sub["delta_rel"].tolist(), sub["se_rel"].tolist())
                verdict["pooled"][f"{tk}/{rk}/{s}"] = dict(sign_test=st, pooled=po,
                                                           per_event=dict(zip(sub.event, sub.delta_rel)))
                res = ("resolved: adapter is worse" if po["ci_lo"] > 0 else
                       "resolved: adapter is better" if po["ci_hi"] < 0 else
                       "not resolved (CI spans zero)")
                print(f"  {tk:5s} {rk:6s} {s:14s}  "
                      f"pooled {po['pooled'] * 100:+6.2f}%  "
                      f"[{po['ci_lo'] * 100:+6.2f}, {po['ci_hi'] * 100:+6.2f}]  "
                      f"I2 {po['i_squared']:4.0f}%   "
                      f"sign {st['n_worse']}/{st['n']} worse  p {st['p_two_sided']:.3f}   "
                      f"{res}")

    headline = verdict["pooled"].get("era5/box/rmse_ens_mean")
    if headline:
        po, st = headline["pooled"], headline["sign_test"]
        verdict["headline"] = dict(
            key="era5/box/rmse_ens_mean", pooled_pct=po["pooled"] * 100,
            ci_pct=[po["ci_lo"] * 100, po["ci_hi"] * 100],
            sign_worse=st["n_worse"], sign_n=st["n"], sign_p=st["p_two_sided"],
            detection_limit_pct=po["detection_limit"] * 100,
            resolved=bool(po["ci_lo"] > 0 or po["ci_hi"] < 0))
        print(f"\n  HEADLINE  (ERA5 truth, event box, ensemble-mean RMSE)")
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
    sub = sub.sort_values("delta_rel")
    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(sub) + 2.2))
    y = np.arange(len(sub))
    ax.errorbar(sub["delta_rel"] * 100, y, xerr=sub["se_rel"] * 100 * 1.96,
                fmt="o", ms=7, lw=1.6, capsize=4, color="#2a78d6", ecolor="#7d8f99")
    ax.axvline(0, color="#131a20", lw=1.2, zorder=0)
    h = verdict.get("headline")
    if h:
        ax.axvspan(h["ci_pct"][0], h["ci_pct"][1], color="#2a78d6", alpha=0.12, zorder=0,
                   label=f"pooled {h['pooled_pct']:+.1f}% "
                         f"[{h['ci_pct'][0]:+.1f}, {h['ci_pct'][1]:+.1f}]")
        ax.axvline(h["pooled_pct"], color="#2a78d6", lw=1.4, ls="--", zorder=1)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e}  ({m})" for e, m in zip(sub["event"], sub["metric"])], fontsize=9)
    ax.set_xlabel("ensemble-mean RMSE:  (adapter - native) / native   [%]   "
                  "positive = adapter worse")
    ax.set_title("Does the adapter cost skill? Per-event effect, ERA5 truth, event box",
                 fontsize=11, loc="left")
    ax.grid(axis="x", color="#e2e9ec")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = A.fig_dir() / "adaptest_effect.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["plan", "prep", "score", "report"])
    ap.add_argument("--events", default=os.environ.get("AIRES_ADAPTEST_EVENTS"),
                    help="comma-separated subset; default all six")
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
    return stage_report(events, plot=not a.no_plot)


if __name__ == "__main__":
    sys.exit(main())
