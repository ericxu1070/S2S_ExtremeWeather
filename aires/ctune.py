#!/usr/bin/env python
"""Tune the C_k splitting schedule before spending H100-hours on it.

``aires.md`` budgets ~46 H100-h per event for the production run and calls the schedule
"empirical", which means a mis-tuned one is an expensive way to learn something. The tuning
pass it proposes replays the DMC loop cheaply, on a surrogate, and asks two questions:

* does the effective sample size survive to the last step (target: ESS > N/4)?
* do clone multiplicities stay bounded, or does one walker take over the population?

What makes this more than a guess is that the surrogate's one input is **measured**. Gate 3
scored 16 real walkers with FCN3 at five leads and recorded each walker's realized outcome,
which fixes the only thing the schedule's behaviour depends on: how skilful the score is at
each resampling time.

The surrogate
-------------
The five scores of one walker are not five steps of a random walk - they are five forecasts
of the *same* eventual outcome, so they share a common signal rather than diffusing apart.
That is visible in the Gate 3 data: a Markov chain predicts
``corr(z_6d, z_15d) = 0.28`` and the measurement is ``0.61``.

So the model is signal-plus-noise, with the signal a martingale (a forecast sequence has to
be one, or it would be predictably wrong):

    m_k        latent conditional mean outcome,  Var(m_k) = rho_k
    theta_k    = m_k + noise,                    Var(noise) = 1 - rho_k
    A_L        = m_K + noise,                    Var(noise) = 1 - rho_K

with ``rho_k`` the measured Gate 3 correlation at lead ``t_k``. Everything is in
standardized units, so ``A_L ~ N(0, 1)`` and the exceedance probabilities are analytic -
which lets the replay also check the estimator's accuracy in a *realistic skill regime*
rather than only on the Ornstein-Uhlenbeck problem in the unit tests.

This model reproduces the measured cross-lead correlations without being fitted to them:
it predicts ``corr(theta_j, theta_k) = rho_j`` for ``j < k``, and ``--check`` prints that
prediction against the data.

Cloning falls out correctly and for the right reason: a clone inherits its parent's ``m``
at ``t_k`` and then draws its own increments, so siblings diverge at exactly the rate the
score's residual uncertainty says they should.

What this is NOT
----------------
A forecast of the production run's answer. It is a schedule diagnostic on a Gaussian
caricature calibrated to one event's skill curve, and the skill curve itself comes from 16
walkers, so ``rho_k`` carries a standard error of roughly 0.1-0.25. Read the ESS and
multiplicity columns; do not read the exceedance column as a result.

    PYTHONPATH=. python -m aires.ctune --check          # the fit, no replay
    PYTHONPATH=. python -m aires.ctune                  # full sweep + figure
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from aires import aconfig as A
from aires import dmc

DEFAULT_EVENT = "PNW_HeatDome_2021"

# Candidate schedules. The first is aires.md's; the rest probe the two questions Gate 3
# raised - whether the early, skill-free resampling times should be switched off, and how
# much tilt the population can take before it collapses.
SCHEDULES = {
    "aires.md":   (0.0, 1.0, 1.4, 1.8, 2.0),
    "skip 6d":    (0.0, 0.0, 1.4, 1.8, 2.0),
    "paper-like": (0.0, 0.0, 1.6, 1.8, 2.0),
    "gentle":     (0.0, 0.5, 0.8, 1.1, 1.4),
    "strong":     (0.0, 1.5, 2.0, 2.5, 3.0),
    "direct":     (0.0, 0.0, 0.0, 0.0, 0.0),
}


def label(name: str) -> str:
    return f"{name:11s}({','.join(f'{c:g}' for c in SCHEDULES[name])})"


def gate3_path(event: str) -> Path:
    return A.gate3_dir(event) / f"{event}_gate3.json"


# --------------------------------------------------------------------------- #
# Calibration from the Gate 3 measurements
# --------------------------------------------------------------------------- #
def load_skill(event: str, index: str = "box") -> dict:
    """Measured score skill per resampling time, plus the data needed to check the model."""
    p = gate3_path(event)
    if not p.exists():
        raise SystemExit(f"no Gate 3 result at {p}\n"
                         f"  run: PYTHONPATH=. python -m aires.gate3 --stage reduce")
    d = json.loads(p.read_text())
    leads = [float(l) for l in d["leads"]]
    theta = np.array([d["theta"][index][str(l)] for l in leads])       # (K, N)
    realized = np.array(d["realized"][index])

    zs = (theta - theta.mean(1, keepdims=True)) / theta.std(1, ddof=1, keepdims=True)
    a = (realized - realized.mean()) / realized.std(ddof=1)
    rho = np.array([float(np.corrcoef(zs[k], a)[0, 1]) for k in range(len(leads))])

    # The martingale needs a non-decreasing skill curve: a later forecast cannot know less
    # about the outcome than an earlier one. Sampling noise at N=16 can invert a pair, so
    # enforce it and say so rather than letting a negative variance increment appear.
    clipped = np.clip(rho, 0.0, 0.999)
    monotone = np.maximum.accumulate(clipped)
    return dict(event=event, index=index, leads=leads, rho_raw=rho, rho=monotone,
                n_walkers=int(theta.shape[1]), theta_z=zs, realized_z=a,
                adjusted=bool(np.any(np.abs(monotone - rho) > 1e-9)))


def check_model(skill: dict) -> str:
    """Print the surrogate's cross-lead prediction against the data it was not fitted to."""
    zs, leads, rho = skill["theta_z"], skill["leads"], skill["rho"]
    K = len(leads)
    obs = np.corrcoef(zs)
    lines = [f"score skill measured at N = {skill['n_walkers']} walkers "
             f"({skill['index']} index)", ""]
    lines.append("  lead    rho(theta, A_L)")
    for k in range(K):
        note = ""
        if abs(skill["rho_raw"][k] - rho[k]) > 1e-9:
            note = f"   (raw {skill['rho_raw'][k]:+.3f}, clipped to keep skill non-decreasing)"
        lines.append(f"  {leads[k]:4.0f}d   {rho[k]:+.3f}{note}")
    lines += ["", "  cross-lead correlations: the model predicts corr(theta_j, theta_k) = "
                  "rho_j for j < k", "         predicted   measured   diff"]
    err = []
    for j in range(K):
        for k in range(j + 1, K):
            pred, got = rho[j], obs[j, k]
            err.append(abs(pred - got))
            lines.append(f"  {leads[j]:3.0f}d/{leads[k]:3.0f}d  {pred:+8.3f}  {got:+9.3f}  "
                         f"{got - pred:+7.3f}")
    lines.append(f"\n  mean |error| = {np.mean(err):.3f}; the correlation standard error at "
                 f"N = {skill['n_walkers']} is ~{1/np.sqrt(skill['n_walkers']-3):.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The replay
# --------------------------------------------------------------------------- #
def replay(rho, C, n_walkers: int, rng: np.random.Generator) -> tuple:
    """One DMC run on the surrogate. Returns (result, realized A_L per walker)."""
    v = np.asarray(rho, dtype="float64")            # Var(m_k) = rho_k
    K = len(v)

    def propagate(k, parents, states):
        m = np.zeros(n_walkers) if states is None else states[parents]
        if k <= K:
            step_var = v[k - 1] - (v[k - 2] if k >= 2 else 0.0)
        else:
            step_var = 1.0 - v[K - 1]               # the horizon leg reveals the outcome
        if step_var <= 0:
            return m
        return m + rng.normal(scale=np.sqrt(step_var), size=n_walkers)

    def score(k, states):
        return states + rng.normal(scale=np.sqrt(max(1.0 - v[k - 1], 0.0)),
                                   size=n_walkers)

    res = dmc.run(n_walkers, C, propagate, score, rng=rng, standardize_scores=True)
    return res, np.asarray(res.states, dtype="float64")


def sweep(skill: dict, n_walkers: int, repeats: int, thresholds, seed: int = 4242) -> dict:
    from scipy.stats import norm

    rho = skill["rho"]
    out = {}
    for name, C in SCHEDULES.items():
        rng = np.random.default_rng(seed)
        ess, mult, founders, exceed, norms = [], [], [], [], []
        for _ in range(repeats):
            res, a = replay(rho, C, n_walkers, rng)
            ess.append(res.ess_by_step / n_walkers)
            mult.append(res.max_multiplicity_by_step)
            founders.append(res.n_founders)
            exceed.append(res.exceedance(a, thresholds))
            norms.append(res.normalization_check())
        ess = np.array(ess); mult = np.array(mult); exceed = np.array(exceed)
        exact = norm.sf(np.asarray(thresholds, dtype="float64"))
        out[name] = dict(
            C=list(C), ess_mean=ess.mean(0).tolist(), ess_final=float(ess[:, -1].mean()),
            ess_final_p10=float(np.percentile(ess[:, -1], 10)),
            max_mult_mean=mult.mean(0).tolist(), max_mult_p90=np.percentile(mult, 90, axis=0).tolist(),
            founders_mean=float(np.mean(founders)),
            exceed_mean=exceed.mean(0).tolist(), exceed_sd=exceed.std(0, ddof=1).tolist(),
            exceed_relerr=(np.abs(exceed.mean(0) - exact) / exact).tolist(),
            exceed_relsd=(exceed.std(0, ddof=1) / exact).tolist(),
            norm_mean=float(np.mean(norms)),
        )
    out["_exact"] = norm.sf(np.asarray(thresholds, dtype="float64")).tolist()
    return out


def report(skill: dict, res: dict, n_walkers: int, thresholds, repeats: int) -> str:
    leads = skill["leads"]
    target = 0.25
    L = [f"C_k replay on the Gate 3 skill curve - N = {n_walkers} walkers, "
         f"{repeats} repeats", "",
         "  ESS/N per resampling time (aires.md target: > 0.25 at the final step)", "",
         "  schedule                      " + "".join(f"{l:7.0f}d" for l in leads)
         + "   final  p10   founders"]
    for name, r in res.items():
        if name.startswith("_"):
            continue
        bar = "".join(f"{e:7.2f}" for e in r["ess_mean"])
        flag = "  " if r["ess_final"] > target else " !"
        L.append(f"  {label(name):30s}{bar}{flag}{r['ess_final']:5.2f} {r['ess_final_p10']:5.2f} "
                 f"{r['founders_mean']:9.1f}")
    L += ["", "  largest clone multiplicity (mean / 90th percentile over repeats)", "",
          "  schedule                        " + "".join(f"{l:8.0f}d" for l in leads)]
    for name, r in res.items():
        if name.startswith("_"):
            continue
        cells = "".join(f"{m:5.1f}/{p:<3.0f}" for m, p in
                        zip(r["max_mult_mean"], r["max_mult_p90"]))
        L.append(f"  {label(name):30s}{cells}")
    L += ["", f"  tail estimate on the surrogate (analytic answer known; "
              f"thresholds in sd of A_L)", "",
          "  schedule                        " + "".join(f"{t:>10.1f}" for t in thresholds)]
    L.append("  " + " " * 30 + "".join(f"{'relerr/relsd':>10}" for _ in thresholds))
    for name, r in res.items():
        if name.startswith("_"):
            continue
        cells = "".join(f"{e:5.2f}/{s:<4.2f}" for e, s in
                        zip(r["exceed_relerr"], r["exceed_relsd"]))
        L.append(f"  {label(name):30s}{cells}")
    L += ["", "  relerr = |bias| / true probability; relsd = spread / true probability.",
          "  The 'direct' row is plain direct sampling at the same N, for comparison."]
    return "\n".join(L)


def plot(skill: dict, res: dict, n_walkers: int, thresholds, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    leads = skill["leads"]
    names = [n for n in res if not n.startswith("_")]
    cmap = plt.get_cmap("viridis")
    colors = {n: cmap(i / max(len(names) - 1, 1)) for i, n in enumerate(names)}
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    a = ax[0]
    a.plot(leads, skill["rho"], "o-", color="#1f77b4", lw=2, label="used (non-decreasing)")
    a.plot(leads, skill["rho_raw"], "x--", color="#999999", lw=1.2, label="measured")
    a.axhline(0, color="0.85", lw=1)
    a.set_xlabel("resampling time $t_k$ (days lead)")
    a.set_ylabel(r"score skill  $\rho(\theta, A_L)$")
    a.set_title("what the surrogate is calibrated to\n(Gate 3, 16 walkers)")
    a.set_ylim(-0.1, 1.05); a.grid(alpha=.3); a.legend(fontsize=8, loc="lower right")

    a = ax[1]
    for n in names:
        a.plot(leads, res[n]["ess_mean"], "o-", color=colors[n], lw=1.8, label=label(n))
    a.axhline(0.25, color="#d62728", lw=1.4, ls="--")
    a.text(leads[0], 0.27, "  aires.md target: ESS > N/4", color="#d62728", fontsize=8,
           ha="left", va="bottom")
    a.set_xlabel("resampling time $t_k$ (days lead)")
    a.set_ylabel("ESS / N")
    a.set_title(f"does the population survive the schedule?\nN = {n_walkers}")
    a.set_ylim(0, 1.45); a.grid(alpha=.3)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.legend(loc="upper center", ncol=2, framealpha=.95,
             prop={"family": "monospace", "size": 7.5})

    a = ax[2]
    x = np.arange(len(thresholds)); w = 0.8 / len(names)
    for i, n in enumerate(names):
        a.bar(x + i * w - 0.4 + w / 2, res[n]["exceed_relsd"], w,
              color=colors[n], label=label(n))
    a.set_xticks(x); a.set_xticklabels([f"{t:.1f}$\\sigma$" for t in thresholds])
    a.set_xlabel(r"threshold on $A_L$")
    a.set_ylabel("spread / true probability")
    a.set_yscale("log")
    a.set_title("tail estimate: how noisy, vs direct sampling\n(lower is better)")
    a.grid(alpha=.3, axis="y")
    a.legend(fontsize=7.5, prop={"family": "monospace", "size": 7.5}, loc="upper left")

    fig.suptitle(f"C_k schedule replay on the measured Gate 3 skill curve "
                 f"({skill['event']}, {skill['index']} index)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Where the schedule's answer actually comes from: rho at the first scored step
# --------------------------------------------------------------------------- #
def rho_sensitivity(skill: dict, n_walkers: int, repeats: int, thresholds,
                    grid, schedules: dict, seed: int = 4242) -> dict:
    """Sweep rho at the FIRST scored step, holding the rest of the measured curve fixed.

    The recurring proposal is to switch the 6 d resampling off (``C_2 = 0``) and save its
    score - the most expensive leg in the run, since its forecast is the longest. Whether
    that helps depends entirely on ``rho_6d``, which Gate 3 measures at 0.53-0.76 but with
    a standard error of ~0.28 at N = 16. So the decision-relevant quantity is not the point
    estimate, it is the CROSSOVER: how low would rho_6d have to be before skipping wins.

    Below the crossover the resampling is acting on noise, and the damage shows up as
    occasional blow-ups rather than a steady drift - a walker that looks good at 6 d purely
    by chance gets cloned, and the tail estimate inherits its error. That is why the
    keep-6d rows here are NON-MONOTONIC in rho at the low end while the skip-6d rows are
    flat: skipping is insensitive to a skill it never buys.
    """
    from scipy.stats import norm
    exact = norm.sf(np.asarray(thresholds, dtype="float64"))
    rho0 = np.asarray(skill["rho"], dtype="float64")
    out = {}
    for r in grid:
        rho = rho0.copy()
        rho[1] = r
        rho = np.maximum.accumulate(np.clip(rho, 0.0, 0.999))
        row = {}
        for name, C in schedules.items():
            rng = np.random.default_rng(seed)
            ess, exc = [], []
            for _ in range(repeats):
                res, a = replay(rho, C, n_walkers, rng)
                ess.append(res.ess_by_step[-1] / n_walkers)
                exc.append(res.exceedance(a, thresholds))
            exc = np.array(exc)
            row[name] = dict(
                ess_final=float(np.mean(ess)),
                relerr=(np.abs(exc.mean(0) - exact) / exact).tolist(),
                relsd=(exc.std(0, ddof=1) / exact).tolist())
        out[f"{r:.2f}"] = row
    return out


# --------------------------------------------------------------------------- #
# Can a production run stand in for a Gate 3 measurement?  No - checked, it cannot
# --------------------------------------------------------------------------- #
def production_skill(event: str, tag: str = "pilot", index: str = "box") -> dict:
    """Estimate rho_k from a FINISHED production run, via its importance weights.

    Tempting, because only three events have a Gate 3 tree while nine have a production
    run. The population at leg k is tilted, but the weights undo exactly that tilt, so a
    weighted correlation between each final walker's leg-k ancestor score and its realized
    A_L is, in expectation, the prior skill.

    **It does not survive its own validation, and must not be used.** Checked against Gate 3
    on the three events that have both (``--validate-production``): mean |error| 0.263 over
    the scored leads, which is the size of the quantity being measured. PNW at 6 d comes
    back -0.369 against a measured +0.612 - a SIGN FLIP, not a noisy estimate.

    The reason is visible in the runs' own diagnostics. Resampling drives the largest clone
    multiplicity to 11-15, so the 64 final walkers descend from ~15 distinct leg-k
    ancestors; the weighted correlation is then computed over a handful of distinct
    (theta, A_L) pairs and is attenuated toward zero. Uri agrees to 0.012 at 6 d, but its
    weight ESS is 2.2 - that is luck, not reliability, and it is exactly the kind of
    agreement that would license the estimator if only one event had been checked.

    Kept in the tree because the negative result is worth more than the function: a skill
    curve for a new event costs a Gate 3 run (~2.2 h, ~18 H100-h), and there is no way to
    read one off a production run after the fact.
    """
    import pandas as pd  # noqa: F401  (kept for a consistent import surface with aires)

    res_dir = A.res_dir(event, tag) if hasattr(A, "res_dir") else None
    base = res_dir if res_dir is not None else (A.AIRES_ROOT / event / "res" / tag)
    cmp_ = json.loads((Path(base) / "compare.json").read_text())
    res = json.loads((Path(base) / "res_result.json").read_text())

    realized = np.asarray(cmp_["realized"][index], dtype="float64")
    w = np.asarray(cmp_["weights"], dtype="float64")
    lineage = cmp_["realized"]["lineage"]

    def wcorr(x, y, wt):
        wt = wt / wt.sum()
        mx, my = np.sum(wt * x), np.sum(wt * y)
        vx, vy = np.sum(wt * (x - mx) ** 2), np.sum(wt * (y - my) ** 2)
        if vx <= 0 or vy <= 0:
            return float("nan")
        return float(np.sum(wt * (x - mx) * (y - my)) / np.sqrt(vx * vy))

    leads, rho = [], []
    for k, t in enumerate(res["theta"], start=1):
        th = np.asarray(t[index], dtype="float64")
        anc = np.array([lineage[i][str(k)] for i in range(realized.size)], dtype=int)
        leads.append(float(t["lead_days"]))
        rho.append(wcorr(th[anc], realized, w))
    return dict(event=event, leads=leads, rho=np.asarray(rho),
                ess=float(w.sum() ** 2 / np.sum(w ** 2)),
                skipped=[t.get("backend") == "skipped(C=0)" for t in res["theta"]])


def validate_production(events, index: str = "box") -> str:
    """Print the production-run estimator against Gate 3 truth, event by event."""
    L = [f"{'event':<26} {'lead':>5}  {'gate3':>7} {'production':>11}  {'diff':>7}",
         "-" * 64]
    errs = []
    for ev in events:
        try:
            g, p = load_skill(ev, index), production_skill(ev, index=index)
        except (SystemExit, FileNotFoundError) as e:
            L.append(f"{ev:<26}  skipped: {e}")
            continue
        for k, lead in enumerate(g["leads"]):
            gv, pv = g["rho_raw"][k], p["rho"][k]
            note = "   (C=0, score not bought)" if p["skipped"][k] else ""
            if k > 0:
                errs.append(abs(pv - gv))
            L.append(f"{ev:<26} {lead:5.0f}  {gv:+7.3f} {pv:+11.3f}  {pv - gv:+7.3f}{note}")
        L.append(f"{'':<26} {'ESS':>5}  {'':>7} {p['ess']:11.1f}")
        L.append("")
    if errs:
        L += [f"mean |diff| over the scored leads = {np.mean(errs):.3f}",
              "",
              "  VERDICT: the estimator does not survive this check. The error is the size",
              "  of the quantity, and PNW at 6 d flips sign. A new event's skill curve costs",
              "  a Gate 3 run; it cannot be recovered from a finished production run."]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Searching the schedule space, with and without the C_1 = C_2 = 0 constraint
# --------------------------------------------------------------------------- #
def _monotone(values, length, floor=0.0):
    """Non-decreasing tuples - a schedule that tilts less as the forecast sharpens is
    buying uncertainty at the wrong end, so the search does not consider one."""
    out = []

    def rec(prefix, lo):
        if len(prefix) == length:
            out.append(tuple(prefix))
            return
        for v in values:
            if v >= lo:
                rec(prefix + [v], v)
    rec([], floor)
    return out


def score_schedule(rho, C, n_walkers, repeats, thresholds, exact, seed=4242) -> dict:
    """One schedule's cost: relative RMSE of the tail estimate, plus population health.

    relRMSE = sqrt(bias^2 + var) / p_true is what a SINGLE production run actually buys -
    it charges a schedule for being wrong on average and for being unrepeatable, which
    `relerr` and `relsd` each only half-measure.
    """
    rng = np.random.default_rng(seed)
    ess, mult, exc = [], [], []
    for _ in range(repeats):
        res, a = replay(rho, C, n_walkers, rng)
        ess.append(res.ess_by_step[-1] / n_walkers)
        mult.append(np.max(res.max_multiplicity_by_step))
        exc.append(res.exceedance(a, thresholds))
    exc = np.array(exc)
    bias = exc.mean(0) - exact
    sd = exc.std(0, ddof=1)
    rel_rmse = np.sqrt(bias ** 2 + sd ** 2) / exact
    return dict(C=list(C), ess_final=float(np.mean(ess)), max_mult=float(np.mean(mult)),
                rel_rmse=rel_rmse.tolist(), objective=float(np.mean(rel_rmse)))


def search(skill: dict, n_walkers: int, repeats: int, thresholds, values,
           refine_repeats: int, top: int = 5) -> dict:
    """Best schedule with C_1 = C_2 = 0, and best with C_2 free, on one skill curve.

    Two stages on purpose: a coarse pass ranks the whole grid cheaply, then the survivors
    are re-scored with enough repeats that the ranking among them means something. A single
    cheap pass would be choosing between schedules whose separation is smaller than the
    Monte Carlo error on either.
    """
    from scipy.stats import norm
    exact = norm.sf(np.asarray(thresholds, dtype="float64"))
    rho = np.asarray(skill["rho"], dtype="float64")

    fams = {"C1=C2=0": [(0.0, 0.0) + t for t in _monotone(values, 3)],
            "C2 free": [(0.0,) + t for t in _monotone(values, 4) if t[0] > 0]}
    out = {}
    for fam, cands in fams.items():
        coarse = [score_schedule(rho, C, n_walkers, repeats, thresholds, exact)
                  for C in cands]
        coarse.sort(key=lambda r: r["objective"])
        fine = [score_schedule(rho, tuple(r["C"]), n_walkers, refine_repeats,
                               thresholds, exact, seed=99) for r in coarse[:top]]
        fine.sort(key=lambda r: r["objective"])
        out[fam] = dict(n_candidates=len(cands), top=fine)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default=os.environ.get("AIRES_EVENT", DEFAULT_EVENT))
    ap.add_argument("--index", default="box", choices=["box", "conus"])
    ap.add_argument("--walkers", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--thresholds", default="2.0,3.0,3.5,4.0")
    ap.add_argument("--check", action="store_true", help="print the model fit and stop")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--rho-sens", action="store_true",
                    help="sweep rho at the first scored step; find the crossover where "
                         "switching that resampling off starts to pay")
    ap.add_argument("--rho-grid", default="0,0.15,0.3,0.45,0.6,0.75")
    ap.add_argument("--validate-production", action="store_true",
                    help="check the production-run skill estimator against Gate 3 truth")
    ap.add_argument("--search", action="store_true",
                    help="search the schedule grid, with and without C_1 = C_2 = 0")
    ap.add_argument("--grid", default="0.4,0.8,1.2,1.6,2.0,2.4,2.8",
                    help="C values the search may choose from")
    ap.add_argument("--search-repeats", type=int, default=400,
                    help="repeats in the coarse pass; the top few are re-scored at --repeats")
    a = ap.parse_args(argv)

    if a.validate_production:
        print(validate_production(
            ["PNW_HeatDome_2021", "SCentral_HeatDome_2023", "WinterStorm_Uri_2021"],
            a.index))
        return 0

    skill = load_skill(a.event, a.index)
    print(check_model(skill))
    if a.check:
        return 0

    if a.rho_sens:
        thresholds = [float(t) for t in a.thresholds.split(",")]
        grid = [float(g) for g in a.rho_grid.split(",")]
        sens = rho_sensitivity(skill, a.walkers, a.repeats, thresholds, grid,
                               {n: SCHEDULES[n] for n in ("aires.md", "skip 6d", "gentle")})
        print(f"\n  rho at the first scored step vs the schedule that assumes it "
              f"(N = {a.walkers}, {a.repeats} repeats)")
        print(f"  measured here: rho_6d = {skill['rho'][1]:.3f}\n")
        names = ["aires.md", "skip 6d", "gentle"]
        print(f"  {'rho_6d':>7} | " + " | ".join(f"{label(n):^30}" for n in names))
        print(f"  {'':>7} | " + " | ".join(
            f"{'ESS_f  ' + '  '.join(f'relsd{t:g}' for t in thresholds):^30}"
            for _ in names))
        for r, row in sens.items():
            cells = [f"{row[n]['ess_final']:5.2f}  "
                     + "  ".join(f"{v:7.2f}" for v in row[n]['relsd']) for n in names]
            print(f"  {float(r):7.2f} | " + " | ".join(f"{c:^30}" for c in cells))
        out = A.gate3_dir(a.event) / f"{a.event}_rho_sens.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"event": a.event, "measured_rho": skill["rho"].tolist(),
                                   "thresholds": thresholds, "sens": sens}, indent=2))
        print(f"\n  wrote {out}")
        return 0

    if a.search:
        thresholds = [float(t) for t in a.thresholds.split(",")]
        values = [float(v) for v in a.grid.split(",")]
        sr = search(skill, a.walkers, a.search_repeats, thresholds, values, a.repeats)
        print(f"\n  schedule search on {a.event} - objective is mean relative RMSE of the "
              f"tail\n  estimate at {', '.join(f'{t:g}' for t in thresholds)} sigma "
              f"(lower is better); N = {a.walkers}")
        for fam, d in sr.items():
            print(f"\n  {fam}  ({d['n_candidates']} monotone candidates)")
            print(f"    {'C_k':<28} {'objective':>9} {'ESS_f':>6} {'maxmult':>8}  "
                  + "  ".join(f"relRMSE{t:g}" for t in thresholds))
            for r in d["top"]:
                cs = ",".join(f"{c:g}" for c in r["C"])
                print(f"    {cs:<28} {r['objective']:9.2f} {r['ess_final']:6.2f} "
                      f"{r['max_mult']:8.1f}  "
                      + "  ".join(f"{v:9.2f}" for v in r["rel_rmse"]))
        base = {n: SCHEDULES[n] for n in ("aires.md", "skip 6d")}
        from scipy.stats import norm
        exact = norm.sf(np.asarray(thresholds, dtype="float64"))
        print(f"\n    {'-- reference --':<28}")
        for n, C in base.items():
            r = score_schedule(np.asarray(skill["rho"]), C, a.walkers, a.repeats,
                               thresholds, exact, seed=99)
            cs = ",".join(f"{c:g}" for c in r["C"])
            print(f"    {cs:<28} {r['objective']:9.2f} {r['ess_final']:6.2f} "
                  f"{r['max_mult']:8.1f}  "
                  + "  ".join(f"{v:9.2f}" for v in r["rel_rmse"]) + f"   ({n})")
        out = A.gate3_dir(a.event) / f"{a.event}_csearch.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"event": a.event, "thresholds": thresholds,
                                   "grid": values, "search": sr}, indent=2))
        print(f"\n  wrote {out}")
        return 0

    thresholds = [float(t) for t in a.thresholds.split(",")]
    print()
    res = sweep(skill, a.walkers, a.repeats, thresholds)
    print(report(skill, res, a.walkers, thresholds, a.repeats))

    out = A.gate3_dir(a.event) / f"{a.event}_ctune.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"event": a.event, "index": a.index, "n_walkers": a.walkers,
         "repeats": a.repeats, "thresholds": thresholds,
         "leads": skill["leads"], "rho": skill["rho"].tolist(),
         "rho_raw": skill["rho_raw"].tolist(), "schedules": res}, indent=2))
    print(f"\n  wrote {out}")
    if not a.no_plot:
        plot(skill, res, a.walkers, thresholds, A.fig_dir(a.event) / f"ctune_{a.event}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
