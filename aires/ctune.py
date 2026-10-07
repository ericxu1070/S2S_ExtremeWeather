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
    a = ap.parse_args(argv)

    skill = load_skill(a.event, a.index)
    print(check_model(skill))
    if a.check:
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
