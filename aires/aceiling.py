#!/usr/bin/env python
"""The model ceiling: is AI+RES's probability limited by the SAMPLER or by GenCast?

Every probability this experiment reports is unbiased for ``P`` **as GenCast defines it**.
If the model's own conditional tail is wrong, every stream inherits that and no amount of
reweighting reveals it. `aires/dmc.py` cannot see this failure; neither can ESS, the
normalization check, or the exceedance curve. It is a different experiment, and this
module is the cheap half of it - the half that runs on data already on disk.

Both sides of the comparison already exist:

    nature's tail    `aires/aclim.py` - 64 years of ERA5 reduced to the SAME observable,
                     ``A_L``, with seasonal-pool percentiles and per-window exceedance
                     rates
    the model's tail the direct ensembles carried in every `compare.json` - GenCast xres
                     (24), Gate 3's free-running GenCast walkers (16), FCN3 (24) - plus
                     the CFSv2 members under `runs/aires/<event>/cfs/` (4 per event)

What this module will not print, and why
----------------------------------------
`aires/INSIGHTS_PLAN.md` (section "Uncertainty: measured quantities only, no new
statistics", 2026-09-10) descoped every invented interval, because this experiment has a
documented history of silent, plausible-looking statistical failure. Two things were cut
from an earlier draft of this file on those grounds, and they must not come back:

**The GEV return period is GONE.** `aclim.gev_return_period` still computes one, and this
module deliberately no longer reads it. The fit is on 30 or 64 seasonal block maxima and
three of the nine events sit AT or BEYOND the sample maximum (`percentile_ex_self` =
100.000), so the quoted figure is pure extrapolation. Measured, on the record on disk:

    event              RP(yr) from 64 blocks   nonparametric 90% CI    resamples giving inf
    PNW 2021                            22.3            [13, 230]                     1%
    California 2022                     37.4            [19, 396]                     5%
    Southwest 2020                     209.5         [47, 68 949]                    13%
    SCentral 2023                     4086.0        [98, 352 357]                    30%

and the same estimator on the 1990-2019 blocks instead of 1959-2022 moves California from
37.4 to 122 years and Southwest from 209.5 years to **infinity** (its fitted upper bound
falls below the observation). A column reading "4086.0" carries none of that, and anyone
reading the table would quote it. The empirical rank is reported instead: it is a count of
windows, it cannot extrapolate, and it is in `aclim`'s own report.

**The Wilcoxon and sign tests on T4 are GONE.** They were computed on 8, 5, 3 and 9 paired
ratios, one per event, over events with different boxes, different seasons and different
tails - not exchangeable draws. A p = 0.039 from eight such numbers is precisely the
confident wrong number the plan's descope is about. The ratios, their exact sampling
intervals and the count below 1 are printed instead.

Four tables, and they do not all carry the same weight
------------------------------------------------------
**T1 - lift over climatology.** The AI+RES conditional probability of the observed ``A_L``
divided by its climatological per-window rate. These are DIFFERENT QUANTITIES (conditional
at 21 d lead vs unconditional) and are not expected to be equal. What makes the ratio
diagnostic is the selection: these nine events were chosen BECAUSE they happened, so a
system with positive resolution should price them ABOVE climatology. A verified event
priced below its own base rate is the signature worth chasing.

Every ``P`` is printed beside the two run-health numbers the plan requires next to it -
the Kish weight ESS out of 64 and the run's own normalization check - and beside the count
of walkers that straddle the target. **A lift is refused outright where that count is 0 or
64**: at 64/64 the indicator is identically 1 and the estimator returns its own
normalization check (`p90_20231107`, 0.5827 against a direct 0.875, `aires/HANDOFF.md`
lines 502 to 519), and at 0/64 the answer is an upper bound, not an estimate.

**T2 - three models at the observed value.** Structural spread at matched inits, matched
observable, with the 95% Wilson interval on the 24-member GenCast direct ensemble - the
one anchor the plan names. Its most useful output is a negative one: at the depth AI+RES
operates every direct ensemble reads 0/24, so multi-model spread bounds nothing there. It
only separates the models at the moderate rungs, and there it is the row where P falls
OUTSIDE the direct interval that carries the finding.

**T3 - fraction of direct members beyond the climatological p90/p95/p99.** Read this ONE
way only. In ABSOLUTE terms it is confounded beyond repair by the same selection that
makes T1 work: an ensemble initialised 21 days before a real extreme SHOULD over-populate
the climatological tail, and that is resolution, not miscalibration. Only the
MODEL-TO-MODEL differences are interpretable, because the selection is identical across
models. The pooled Wilson interval treats members as independent trials and they are not
(24 members share one init), so it is a LOWER bound on the true width; the per-event range
is printed next to it as the honest spread.

**T4 - spread, and this is the one that decides T3.** Past the predictability horizon
(~8.5 d for CONUS T2m) a conditional ensemble's spread should approach the climatological
spread. Spread does not depend on where the ensemble is centred, so it is FREE OF THE
SELECTION CONFOUND that ruins T3's absolute reading. If T3's tail excess were
over-dispersion, T4 would read well above 1.0. It does not - which is what converts T3's
excess from "the models are too wild" into "the models are correctly shifted".

Each ratio carries its exact chi-square sampling interval, and that interval is the point:
a single 24-member ratio of 0.87 has a 95% interval of about [0.68, 1.22] and is not
distinguishable from 1.0 on its own. CFSv2's four members give an interval so wide it says
nothing at all. Only the sign pattern across events is evidence, and it is a count.

The caveat T4 cannot shed
-------------------------
A ratio below 1.0 has two explanations and this module cannot separate them: an
under-dispersed ensemble, or genuine residual predictability at 21 d (a skilful forecast
IS narrower than climatology). A 13-15% deficit is about the size either would produce.
Separating them needs verification across many inits - the hindcast reliability experiment
- not one more reduction of nine events. Reported as a lead, with its own significance.

A second caveat sits under the denominator: the climatological sd is taken over the whole
+-21 d seasonal pool across 64 years, while the ensemble sd is at one fixed valid time.
Anomalies remove the mean seasonal cycle but not the seasonal cycle OF VARIANCE, so the
denominator is mildly inflated by a known amount of unknown sign. It is a reason to read
the model-to-model ordering rather than the absolute level of the ratio.

Not a deck panel
----------------
`aires/INSIGHTS_PLAN.md` does not include this module and nothing here belongs on the
slide deck. `aires/ainsights.py` imports exactly one name from it, `wilson`, as a pinned
utility; it reads none of these tables. This is a standing follow-up, and the experiment it
actually points at is the one T4 keeps naming: hindcast reliability across many inits, the
only thing that can separate an under-dispersed ensemble from genuine residual
predictability at 21 d. The two findings worth carrying into that spec are T4's
model-to-model ordering of the spread ratio and T1's two events priced BELOW their own
climatological base rate (Uri 0.30x, Elliott 0.54x on the 1959-2022 pool).

    PYTHONPATH=. python -m aires.aceiling                     # tables, cached clim report
    PYTHONPATH=. python -m aires.aceiling --rebuild-clim      # re-run aclim.report() first
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from . import aconfig as A

OUT = A.AIRES_ROOT / "ceiling"
CLIM_REPORT = A.AIRES_ROOT / "clim" / "report.json"
TAG = "pilot"

EVENTS = ("PNW_HeatDome_2021", "SCentral_HeatDome_2023", "California_HeatWave_2022",
          "Southwest_HeatWave_2020", "WinterStorm_Uri_2021", "WinterStorm_Elliott_2022",
          "p90_20231107", "p90_20240802", "p90_20251224")
SHORT = {"PNW_HeatDome_2021": "PNW 2021", "SCentral_HeatDome_2023": "SCentral 2023",
         "California_HeatWave_2022": "California 2022",
         "Southwest_HeatWave_2020": "Southwest 2020", "WinterStorm_Uri_2021": "Uri 2021",
         "WinterStorm_Elliott_2022": "Elliott 2022", "p90_20231107": "p90 2023-11-07",
         "p90_20240802": "p90 2024-08-02", "p90_20251224": "p90 2025-12-24"}
# Named exactly as `compare.json`'s `ds` block does, plus CFSv2 which lives elsewhere.
MODELS = ("gencast_xres", "gencast_walkers", "fcn3", "cfs")
MODEL_LABEL = {"gencast_xres": "GenCast xres", "gencast_walkers": "GenCast walkers",
               "fcn3": "FCN3", "cfs": "CFSv2"}
PERIODS = ("baseline_1990_2019", "recent_1993_2022", "full_1959_2022")
# The pool a percentile is read against. `_ex_self` drops the event's own year, so an
# event is never counted as evidence for its own rarity.
POOL = "full_1959_2022"
QUANTILES = ("90", "95", "99")
# The direct ensemble whose binomial interval is the plan's named anchor for `P`.
ANCHOR = "gencast_xres"


# --------------------------------------------------------------------------- #
# Reductions
# --------------------------------------------------------------------------- #
def beyond(x, threshold: float, sign: float) -> np.ndarray:
    """Sign-aware ``x`` at least as extreme as ``threshold``. ``sign=-1`` is a cold tail."""
    return np.asarray(sign * (np.asarray(x, dtype="float64") - threshold) >= 0.0)


def res_probability(cmp_: dict, threshold: float) -> float:
    """``Z * mean(1{beyond} * w)`` - `dmc.DMCResult.expectation`, recomputed from disk.

    Deliberately NOT read off `compare.json`'s 41-threshold curve: the observed value falls
    between curve knots and interpolating a step function in the tail is exactly where it
    would go wrong.
    """
    sign = float(cmp_["ds"]["tail_sign"])
    a = np.asarray(cmp_["realized"]["box"], dtype="float64")
    w = np.asarray(cmp_["weights"], dtype="float64")
    return float(np.exp(cmp_["log_Z"]) * np.mean(beyond(a, threshold, sign) * w))


def weight_ess(weights) -> float:
    """Kish effective sample size of the importance weights, out of ``len(weights)``.

    The number `aires/HANDOFF.md` line 643 says is "the right thing to quote next to a
    `P`", because walkers cloned at the last resampling inherit their parent's ``V_K`` and
    carry identical mass. Reproduces the published values exactly (PNW 5.68, Uri 2.24).
    """
    w = np.asarray(weights, dtype="float64")
    s2 = float((w ** 2).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval. Wald gives zero width at k=0, which is not a statement."""
    if n == 0:
        return (0.0, 1.0)
    p, d = k / n, 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def sd_interval(n: int, conf: float = 0.95) -> tuple[float, float]:
    """Multiplicative 95% sampling interval for a sample sd on ``n`` draws.

    Closed form, not a fit: ``(n-1) s^2 / sigma^2 ~ chi2_{n-1}`` for Gaussian draws, so an
    sd RATIO is uncertain by these factors before any physics enters. At n=24 it is about
    [0.78, 1.40]; at n=4 it is [0.57, 3.73], which is why a four-member CFSv2 sd is shown
    with its interval rather than as a dispersion verdict. Returns (1, 1) if scipy is
    unavailable, i.e. "no interval claimed", never a fabricated one.
    """
    if n < 2:
        return (float("nan"), float("nan"))
    try:
        from scipy import stats
    except Exception:                      # pragma: no cover - scipy is present in `moe`
        return (1.0, 1.0)
    a = (1.0 - conf) / 2.0
    df = n - 1
    return (float(np.sqrt(df / stats.chi2.ppf(1 - a, df))),
            float(np.sqrt(df / stats.chi2.ppf(a, df))))


def resolution(n_beyond: int, n: int) -> str:
    """Does the resampled population STRADDLE the target? The estimator's own precondition.

    `aires/HANDOFF.md` line 513: "the usable operating range is targets the resampled
    population still straddles". At ``n_beyond == n`` the indicator is identically 1 and
    ``P`` collapses to ``Z * mean(w)``, the run's own normalization check - the pinned
    `p90_20231107` failure, silent in `compare.json`. At ``n_beyond == 0`` the estimate is
    identically 0, which is an upper bound rather than a probability.
    """
    if n == 0:
        return "none: no population"
    if n_beyond >= n:
        return "none: population entirely beyond the target, P == normalization check"
    if n_beyond == 0:
        return "none: no walker reached the target, P == 0 is an upper bound"
    if n_beyond <= 2 or n_beyond >= n - 2:
        return f"marginal: only {min(n_beyond, n - n_beyond)} of {n} on one side"
    return "ok"


def check_report_boxes(report: dict) -> None:
    """Refuse a climatology report whose events fell back to the CONUS column.

    The same rule `aires.alift.clim_pool` enforces, applied to the cached JSON, which is
    one remove further from the data than `alift` ever gets: `aclim.report` does
    ``col = name if name in anom.columns else "CONUS"`` with no error, so an event added to
    `aindex.EVENT_BOXES` after the .nc cache was built gets a box and NO climatology
    column. That is not hypothetical. `WinterStorm_Elliott_2022`'s N Plains ``A_L``
    (reaching -17 K) was once scored against the CONUS column (spanning +-5 K), reporting
    ``lift >= 285x`` where the honest number is 0.35x (`aires/HANDOFF.md` line 594).

    `aires/tests/test_aclim.py` pins the .nc cache against `EVENT_BOXES`; nothing pinned
    the DERIVED json, which this module caches separately and can therefore outlive a
    rebuild of either side. Events with no registered box (the ``p90_*`` cases) are defined
    on CONUS in the first place, so for them the CONUS column is correct, not a fallback.
    """
    from . import aindex as AI            # ~1.7 s; kept out of module import on purpose

    bad = {}
    for name, rec in report.get("events", {}).items():
        col = rec.get("column")
        if name in AI.EVENT_BOXES and col != name:
            bad[name] = col
    if bad:
        raise SystemExit(
            f"stale climatology report at {CLIM_REPORT}:\n"
            + "".join(f"  {k} has a registered box but was scored on column {v!r}\n"
                      for k, v in bad.items())
            + "  Those events would be compared against a different region's climate.\n"
              "  Rebuild: python -m aires.aclim --build  (then rerun with --rebuild-clim)")


def climatology_report(rebuild: bool = False) -> dict:
    """`aclim.report()`, cached - it re-reads a 64-year table and takes a few minutes."""
    if CLIM_REPORT.exists() and not rebuild:
        rep = json.loads(CLIM_REPORT.read_text())
        check_report_boxes(rep)
        return rep
    from . import aclim
    rep = aclim.report()
    check_report_boxes(rep)
    CLIM_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CLIM_REPORT.write_text(json.dumps(rep, indent=1, default=str))
    return rep


def collect(events=EVENTS, tag: str = TAG, rebuild_clim: bool = False) -> list[dict]:
    """One row per event: the conditional answer, the climatological rates, the ensembles."""
    report = climatology_report(rebuild_clim)
    rows = []
    for ev in events:
        cmp_path = A.AIRES_ROOT / ev / "res" / tag / "compare.json"
        if not cmp_path.exists():
            print(f"  skip {ev}: no {cmp_path}")
            continue
        cmp_ = json.loads(cmp_path.read_text())
        clim = report["events"][ev]
        sign, obs = float(cmp_["ds"]["tail_sign"]), float(cmp_["observed"])
        pop = np.asarray(cmp_["realized"]["box"], dtype="float64")
        n_beyond = int(beyond(pop, obs, sign).sum())
        pop_extreme = float(pop.max() if sign > 0 else pop.min())

        cfs_path = A.AIRES_ROOT / ev / "cfs" / f"{ev}_cfs_lead21_t2m_anom.json"
        ds = cmp_["ds"]
        ens = {m: ds.get(m, {}).get("box", []) for m in MODELS if m != "cfs"}
        ens["cfs"] = json.loads(cfs_path.read_text())["box"] if cfs_path.exists() else []

        row = dict(event=ev, short=SHORT.get(ev, ev), sign=sign, observed=obs,
                   P_res=res_probability(cmp_, obs),
                   normalization_check=float(cmp_["normalization_check"]),
                   log_Z=float(cmp_["log_Z"]),
                   weight_ess=weight_ess(cmp_["weights"]),
                   n_walkers=int(pop.size), n_beyond=n_beyond,
                   resolution=resolution(n_beyond, int(pop.size)),
                   # The deepest level the run resolved: `P` at the population's own most
                   # extreme member. Where nothing reached the target this is the honest
                   # upper bound (`aires/HANDOFF.md` line 651, "matches the wave-1 table's
                   # < 3e-4"), not a probability of the observation.
                   deepest_resolved=res_probability(cmp_, pop_extreme),
                   min_ess_frac=float(min(cmp_["ess_by_step"])) / cmp_["config"]["n_walkers"],
                   pop_min=float(pop.min()), pop_max=float(pop.max()),
                   ladder=clim[POOL]["week_ladder_ex_self"],
                   clim_column=clim.get("column"),
                   clim_sd=float(clim[POOL]["week_sd"]), ens=ens)
        for per in PERIODS:
            c = clim[per]
            # ex_self: the event's own year is removed from both numerator and pool. The
            # denominator is rescaled by the year count rather than recounted, because the
            # report stores no ex-self window total; windows per year vary by at most a
            # few parts in 2752, so this is exact to well under one window.
            n_pool = c["n_windows"] * c["n_years_ex_self"] / c["n_years"]
            row[f"p_clim__{per}"] = c["n_windows_beyond_ex_self"] / n_pool
            row[f"n_beyond_clim__{per}"] = int(c["n_windows_beyond_ex_self"])
            row[f"n_pool_clim__{per}"] = int(round(n_pool))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _rule(n: int = 118) -> str:
    return "-" * n


def table1(rows) -> None:
    print("\nT1  CONDITIONAL (AI+RES, 21 d lead) vs CLIMATOLOGICAL probability of the "
          "observed A_L")
    print("    Selected on occurrence, so positive resolution should put every lift ABOVE 1.")
    print("    ESS is the Kish weight ESS out of 64; norm is the run's own normalization")
    print("    check; str is how many walkers sit beyond the target. A lift is REFUSED at")
    print("    0/64 and at 64/64 - see the `resolution` note under each such row.")
    hdr = (f"{'event':16s} {'obs A_L':>8s} {'P_AI+RES':>9s} {'ESS':>5s} {'norm':>6s} "
           f"{'str':>6s} | {'p_clim':>8s} {'p_clim':>8s} {'p_clim':>8s} | "
           f"{'lift':>7s} {'lift':>7s}")
    w = len(hdr) + 24                      # the "<-- below climatology" flag hangs off
    print(_rule(w))
    print(hdr)
    print(f"{'':16s} {'':8s} {'':9s} {'/64':>5s} {'':6s} {'/64':>6s} | "
          f"{'90-19':>8s} {'93-22':>8s} {'59-22':>8s} | {'90-19':>7s} {'59-22':>7s}")
    print(_rule(w))
    notes = []
    for r in rows:
        pb, pf = r["p_clim__baseline_1990_2019"], r["p_clim__full_1959_2022"]
        usable = r["resolution"].startswith(("ok", "marginal"))
        if usable:
            lb = f"{r['P_res'] / pb:7.2f}" if pb > 0 else "   n/a "
            lf = f"{r['P_res'] / pf:7.2f}" if pf > 0 else "   n/a "
            flag = "  <-- below climatology" if (pb > 0 and r["P_res"] / pb < 1.0) else ""
        else:
            lb = lf = "refused"
            flag = ""
        print(f"{r['short']:16s} {r['observed']:+8.2f} {r['P_res']:9.4f} "
              f"{r['weight_ess']:5.2f} {r['normalization_check']:6.3f} "
              f"{r['n_beyond']:3d}/{r['n_walkers']:<2d} | {pb:8.4f} "
              f"{r['p_clim__recent_1993_2022']:8.4f} {pf:8.4f} | {lb} {lf}{flag}")
        if not r["resolution"].startswith("ok"):
            notes.append((r["short"], r))
    if notes:
        print(_rule(w))
        for short, r in notes:
            print(f"    {short:16s} {r['resolution']}")
            if r["n_beyond"] == 0:
                print(f"    {'':16s} deepest level this run resolved: "
                      f"P < {r['deepest_resolved']:.1e} at A_L = "
                      f"{(r['pop_max'] if r['sign'] > 0 else r['pop_min']):+.2f} K")
    print(_rule(w))
    for per in ("baseline_1990_2019", "full_1959_2022"):
        z = [r["short"] for r in rows if r[f"n_beyond_clim__{per}"] == 0]
        if z:
            n = rows[0][f"n_pool_clim__{per}"]
            print(f"    p_clim = 0 in {per} means 0 of ~{n} seasonal windows reached the "
                  f"observation: {', '.join(z)}.")
            print(f"    {'':4s}That is a bound on the base rate, so no lift is defined "
                  f"for those rows.")


def table2(rows) -> None:
    print("\nT2  THREE MODELS at the observed value - structural spread at matched inits")
    print(f"    The last column is the 95% Wilson interval on the {MODEL_LABEL[ANCHOR]} "
          f"direct ensemble,")
    print("    the plan's named anchor for P. `!` marks P falling OUTSIDE it.")
    hdr = (f"{'event':16s} {'obs A_L':>8s} | " +
           " ".join(f"{MODEL_LABEL[m]:>15s}" for m in MODELS) +
           f" | {'AI+RES':>8s} {'clim':>7s} {'direct 95% CI':>17s}")
    print(_rule(len(hdr)))
    print(hdr)
    print(_rule(len(hdr)))
    for r in rows:
        cells = []
        for m in MODELS:
            e = r["ens"][m]
            if not len(e):
                cells.append(f"{'-':>15s}")
                continue
            n = int(beyond(e, r["observed"], r["sign"]).sum())
            cells.append(f"{n:>2d}/{len(e):<3d} {n / len(e):6.3f}".rjust(15))
        anchor = r["ens"][ANCHOR]
        if len(anchor):
            k = int(beyond(anchor, r["observed"], r["sign"]).sum())
            lo, hi = wilson(k, len(anchor))
            mark = " " if lo <= r["P_res"] <= hi else "!"
            ci = f"[{lo:.3f},{hi:.3f}]{mark}".rjust(17)
        else:
            ci = f"{'-':>17s}"
        print(f"{r['short']:16s} {r['observed']:+8.2f} | " + " ".join(cells) +
              f" | {r['P_res']:8.4f} {r['p_clim__full_1959_2022']:7.4f} {ci}")
    print(_rule(len(hdr)))
    print("    CFSv2 is 4 members per event. A 0/4 is not evidence of anything; it is")
    print("    printed so the ensemble sizes behind each column are visible.")


def table3(rows) -> dict:
    print("\nT3  Fraction of direct members beyond the climatological p90/p95/p99")
    print(f"    ({POOL} seasonal pool, own year dropped). Calibrated-and-unselected reads")
    print("    0.100 / 0.050 / 0.010. ABSOLUTE VALUES ARE SELECTION-CONFOUNDED - see T4.")
    print(_rule())
    pooled = {m: {q: [0, 0] for q in QUANTILES} for m in MODELS}
    per_event = {m: {q: [] for q in QUANTILES} for m in MODELS}
    for r in rows:
        line = f"{r['short']:16s}"
        for q in QUANTILES:
            thr = r["ladder"][q]
            sub = []
            for m in ("gencast_xres", "fcn3"):
                e = r["ens"][m]
                sub.append("  -- " if not len(e)
                           else f"{int(beyond(e, thr, r['sign']).sum()) / len(e):5.2f}")
            line += f"  p{q}={thr:+7.2f} [" + "/".join(sub) + "]"
        print(line)
        for m in MODELS:
            e = r["ens"][m]
            if not len(e):
                continue
            for q in QUANTILES:
                k = int(beyond(e, r["ladder"][q], r["sign"]).sum())
                pooled[m][q][0] += k
                pooled[m][q][1] += len(e)
                per_event[m][q].append(k / len(e))
    print(_rule())
    print("  POOLED. Only the MODEL-TO-MODEL differences are interpretable, and the")
    print("  interval treats members as independent trials, which 24 members sharing one")
    print("  init are not - read it as a LOWER bound on the width. `evt` is the range of")
    print("  the per-event fractions, which is the spread that is actually resolved.")
    for m in MODELS:
        if pooled[m]["90"][1] == 0:
            continue
        n_ev = len(per_event[m]["90"])
        for q in QUANTILES:
            k, n = pooled[m][q]
            lo, hi = wilson(k, n)
            v = per_event[m][q]
            print(f"    {MODEL_LABEL[m] if q == '90' else '':16s} p{q}: {k:3d}/{n:<3d} "
                  f"{k / n:.3f} [{lo:.3f},{hi:.3f}]  evt {min(v):.2f}-{max(v):.2f} "
                  f"over {n_ev} events")
    print(f"    {'climatology':16s} p90: 0.100   p95: 0.050   p99: 0.010")
    return {m: {q: dict(k=pooled[m][q][0], n=pooled[m][q][1],
                        per_event=per_event[m][q]) for q in QUANTILES}
            for m in MODELS if pooled[m]["90"][1]}


def table4(rows) -> dict:
    print("\nT4  SPREAD - the selection-free test. ensemble sd / climatological sd.")
    print("    1.0 = climatological dispersion. >1 would mean the T3 excess is")
    print("    over-dispersion. Each cell carries the exact chi-square 95% sampling")
    print("    interval of the ratio: at n=24 it is [0.78, 1.40] BEFORE any physics, so a")
    print("    single row is never evidence on its own.")
    hdr = (f"{'event':16s} {'clim sd':>8s} | " +
           " ".join(f"{MODEL_LABEL[m]:>26s}" for m in MODELS))
    print(_rule(len(hdr)))
    print(hdr)
    print(_rule(len(hdr)))
    ratios = {m: [] for m in MODELS}
    sizes = {m: [] for m in MODELS}
    for r in rows:
        cells = []
        for m in MODELS:
            e = r["ens"][m]
            if len(e) < 3:
                cells.append(f"{'-':>26s}")
                continue
            n = len(e)
            sd = float(np.std(np.asarray(e, dtype="float64"), ddof=1))
            ratio = sd / r["clim_sd"]
            lo, hi = sd_interval(n)
            ratios[m].append(ratio)
            sizes[m].append(n)
            cells.append(f"x{ratio:4.2f} [{ratio * lo:4.2f},{ratio * hi:5.2f}] n{n:<2d}"
                         .rjust(26))
        print(f"{r['short']:16s} {r['clim_sd']:8.2f} | " + " ".join(cells))
    print(_rule(len(hdr)))
    out = {}
    for m in MODELS:
        v = np.asarray(ratios[m], dtype="float64")
        if not v.size:
            continue
        gm = float(np.exp(np.mean(np.log(v))))
        out[m] = dict(geometric_mean=gm, n_events=int(v.size),
                      below_one=int((v < 1).sum()), lo=float(v.min()),
                      hi=float(v.max()), ratios=v.tolist(),
                      members_per_event=sizes[m])
        print(f"    {MODEL_LABEL[m]:16s} geo-mean x{gm:.2f}  ({out[m]['below_one']}/{v.size} "
              f"below 1, range {v.min():.2f}-{v.max():.2f}, "
              f"{min(sizes[m])}-{max(sizes[m])} members per event)")
    print("\n    No test statistic is computed across these rows. They are one number per")
    print("    event over different boxes, seasons and tails - not exchangeable draws -")
    print("    and `aires/INSIGHTS_PLAN.md` descopes inventing an interval for them. The")
    print("    count below 1 is the claim; its size is not.")
    print("    A ratio below 1 is under-dispersion OR genuine residual predictability at")
    print("    21 d. This module cannot separate them; the hindcast experiment can.")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--rebuild-clim", action="store_true",
                    help="re-run aclim.report() instead of using the cached JSON")
    ap.add_argument("--json", default=str(OUT / "ceiling.json"))
    a = ap.parse_args(argv)

    rows = collect(tag=a.tag, rebuild_clim=a.rebuild_clim)
    if not rows:
        raise SystemExit("no runs found")
    table1(rows)
    table2(rows)
    pooled = table3(rows)
    spread = table4(rows)

    out = Path(a.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        dict(tag=a.tag, pool=POOL,
             rows=[{k: v for k, v in r.items() if k != "ens"} for r in rows],
             pooled_tail=pooled, spread=spread), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
