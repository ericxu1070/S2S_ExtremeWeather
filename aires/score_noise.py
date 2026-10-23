"""Does a less noisy score buy AI+RES a bigger direct-sampling equivalent? CPU only.

`ctune.replay` uses one skill curve rho_k twice: as the score's noise (1 - rho_k) and as
the walker's predictability (Var(m_k) = rho_k). Sharpen the score there and the simulated
atmosphere becomes more predictable too, so clones made at t_k have less room to diverge,
and a better scorer reads as a WORSE estimate. This module fixes the atmosphere and varies
only the score's noise.

The noise model, per resampling time, in units where Var(A_L) = 1:

    theta_M = m_k + e + mean of M member draws,   Var(m_k) = v, Var(e) = s, member var g
    rho_M   = v / sqrt(v + s + g / M)    =>    1 / rho_M^2 = a + b / M

with a = (v + s) / v^2 and b = g / v^2, fitted to Gate 3's rho at M = 1..6 (every shared
member subset, as in `gate3_msub.py`). Two equations, three unknowns: `g` is NOT the
measured member spread, because common random numbers make part of every member's noise
common to all walkers, and that part drops out of a ranking. So one assumption is needed.
`frac` in [0, 1] places v in its admissible range [1/a, 1] (s >= 0 forces v >= 1/a):
frac = 0 means FCN3's ensemble mean is the true conditional mean (all score error is member
noise); frac > 0 hands part of it to FCN3 model error.

A deterministic scorer is then just a noise level: one that tracks the conditional mean as
well as FCN3's M-member mean is "FCN3 at M_eq", "M=inf" is FCN3's ensemble mean with its
model error s, and "perfect" is the conditional mean itself (noise 0).

    PYTHONPATH=. python -m aires.score_noise [--fracs 0,0.5] [--repeats 30000]
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

from aires import aconfig as A
from aires import ctune as CT
from aires import gate3 as G
from aires import gate3_msub as MS

INF = float("inf")
# (label, M per lead). Lead 3 d carries no score (C_1 = 0), so its M is irrelevant.
SCHEMES = [("M=1", (1, 1, 1, 1, 1)), ("M=3", (3, 3, 3, 3, 3)), ("M=6 (today)", (6, 6, 6, 6, 6)),
           ("M=3 at 6d, M=1 after", (1, 3, 1, 1, 1)), ("M=24", (24,) * 5),
           ("M=inf (FCN3 mean)", (INF,) * 5), ("perfect (cond. mean)", (0,) * 5)]
THRESHOLDS = (2.0, 3.0, 3.5)
SEEDS = (4242, 777, 31)


def subset_rho(df: pd.DataFrame) -> dict[float, np.ndarray]:
    """Pearson rho(theta_M, A_L) for M = 1..6, mean over every member subset, per lead."""
    real = df.groupby("walker")["realized_box"].first().sort_index().to_numpy()
    out = {}
    for l, g in df.groupby("lead_days"):
        p = g.pivot(index="walker", columns="member", values="box").sort_index()
        out[float(l)] = np.array([np.mean([np.corrcoef(p[list(c)].mean(axis=1), real)[0, 1]
                                           for c in combinations(p.columns, m)])
                                  for m in range(1, p.shape[1] + 1)])
    return out


def fit(rho_by_m: dict[float, np.ndarray], min_rho: float = 0.05) -> dict[float, dict]:
    """a, b of 1/rho_M^2 = a + b/M per lead; a lead whose rho never reaches ``min_rho``
    has no skill to fit (3 d here) and is returned with a = b = None."""
    out = {}
    for l, r in rho_by_m.items():
        m = np.arange(1, r.size + 1)
        if np.all(r < min_rho):
            out[l] = dict(a=None, b=None, rho=r.tolist())
            continue
        b, a = np.polyfit(1.0 / m, 1.0 / r ** 2, 1)
        pred = 1.0 / np.sqrt(a + b / m)
        out[l] = dict(a=float(a), b=float(b), rho=r.tolist(),
                      max_resid=float(np.max(np.abs(pred - r))))
    return out


def scenario(fitted: dict[float, dict], frac: float) -> dict[float, dict]:
    """v, s, g per lead with v = 1/a + frac (1 - 1/a); a no-skill lead gets v = 0."""
    out = {}
    for l, f in fitted.items():
        if f["a"] is None:
            out[l] = dict(v=0.0, s=1.0, g=0.0)
            continue
        lo = min(1.0 / f["a"], 0.999)
        v = lo + frac * (0.999 - lo)
        out[l] = dict(v=v, s=f["a"] * v ** 2 - v, g=f["b"] * v ** 2)
    return out


def noise(sc: dict, m) -> float:
    return 0.0 if m == 0 else sc["s"] + (0.0 if m == INF else sc["g"] / m)


def _job(args):
    v, nv, seed, repeats = args
    exact = norm.sf(np.asarray(THRESHOLDS))
    r = CT.score_schedule(v, CT.SCHEDULES["aires.md"], 64, repeats, THRESHOLDS, exact,
                          seed=seed, noise_var=nv)
    return dict(ess=r["ess_final"], max_mult=r["max_mult"],
                n_eq=[(1 - p) / (p * e ** 2) for p, e in zip(exact, r["rel_rmse"])])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--event", default="PNW_HeatDome_2021")
    ap.add_argument("--fracs", default="0,0.5")
    ap.add_argument("--repeats", type=int, default=30000)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args(argv)
    ev = G._event(a.event)
    res = json.loads(G.result_path(ev).read_text())
    fitted = fit(subset_rho(MS.members(ev, res)))
    leads = sorted(fitted)

    print(f"fit 1/rho_M^2 = a + b/M  ({a.event}, M = 1..6)")
    for l in leads:
        f = fitted[l]
        if f["a"] is None:
            print(f"  {l:4.0f}d  no skill (rho_M = {np.round(f['rho'], 3).tolist()})")
        else:
            assert f["max_resid"] < 0.01, (l, f)     # the 1/M noise model must hold
            print(f"  {l:4.0f}d  a={f['a']:.3f} b={f['b']:.3f}  rho_inf={1/np.sqrt(f['a']):.3f}"
                  f"  max resid {f['max_resid']:.4f}")

    rows = []
    for frac in (float(x) for x in a.fracs.split(",")):
        sc = scenario(fitted, frac)
        v = np.array([sc[l]["v"] for l in leads])
        assert np.all(np.diff(v) >= -1e-9), ("predictability must not decrease", v)
        print(f"\nframe frac={frac:g}: v = {np.round(v, 3).tolist()}  "
              f"s = {np.round([sc[l]['s'] for l in leads], 3).tolist()}")
        jobs, meta = [], []
        for label, ms in SCHEMES:
            nv = np.array([noise(sc[l], m) for l, m in zip(leads, ms)])
            rho = [sc[l]["v"] / np.sqrt(sc[l]["v"] + n) if sc[l]["v"] > 0 else 0.0
                   for l, n in zip(leads, nv)]
            for seed in SEEDS:
                jobs.append((v, nv, seed, a.repeats))
                meta.append((label, rho))
        with Pool(a.procs) as pool:
            out = pool.map(_job, jobs)
        for (label, rho), o, (_, _, seed, _) in zip(meta, out, jobs):
            rows.append(dict(frac=frac, scheme=label, seed=seed,
                             rho=np.round(rho, 3).tolist(), ess=o["ess"], max_mult=o["max_mult"],
                             **{f"N_eq_{t:g}sd": n for t, n in zip(THRESHOLDS, o["n_eq"])}))
    t = pd.DataFrame(rows)
    path = A.gate3_dir(ev.name) / f"{ev.name}_score_noise.csv"
    t.to_csv(path, index=False)

    cols = ["ess"] + [f"N_eq_{x:g}sd" for x in THRESHOLDS]
    g = t.groupby(["frac", "scheme"], sort=False)
    m, lo, hi = g[cols].mean(), g[cols].min(), g[cols].max()
    show = m[["ess"]].round(2)
    show.insert(0, "rho 6/9/12/15 d", g["rho"].first().map(lambda r: "/".join(f"{x:.2f}" for x in r[1:])))
    for c in cols[1:]:
        show[c] = [f"{mm:.0f} [{ll:.0f}-{hh:.0f}]" for mm, ll, hh in zip(m[c], lo[c], hi[c])]
    with pd.option_context("display.width", 250):
        print("\n" + show.to_string())
    print(f"\n  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
