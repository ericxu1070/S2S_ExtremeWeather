#!/usr/bin/env python
"""Phase 4 figures: what the AI+RES pilot found.

Four figures, covering the six items `aires.md` lists. Each reads only artifacts that
`aires/run_aires.py` already wrote (`compare.json`, `res_result.json`, `ds_baseline.json`)
plus, for the two that need fields rather than indices, the walker diagnostics cubes. No
GPU, no model, no network.

    trajectory    the algorithm happening: the observable against lead time, every
                  branch the pilot rolled, the barriers, the 7-day window
    exceedance    the headline: does the RES tail contain the observed event when direct
                  sampling does not?                                     [aires.md fig 1]
    diagnostics   score skill per t_k, and whether N=64 held up          [figs 3, 4]
    genealogy     walker ancestry, and whether clones actually separate  [figs 2, 6]
    composite     what the RES-selected extremes look like               [fig 5]

Reading the exceedance figure honestly
--------------------------------------
The RES curve is an estimate with a real uncertainty, and the band drawn around it is a
plain delta-method standard error on the weighted indicator. It is a LOWER bound on the
true uncertainty: standardizing the score across the live population makes the scheme
self-interacting, so the walkers are not independent and no closed-form interval is
available (`aires/dmc.py` states the same caveat for the estimator itself). The direct
sampling curves carry Wilson intervals, which ARE exact for what they are - 16 or 24
independent draws. The comparison the figure is for is qualitative and does not depend on
either interval: whether RES puts probability mass where direct sampling has none.

`--stage compare` must have run. The x axis is in the metric's own units and the tail sign
is applied for cold events, so "further right" always means "more extreme".

    PYTHONPATH=. python -m aires.aplots --event PNW_HeatDome_2021 --tag pilot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from aires import run_aires as R
from fcn3 import fevents as F

FIGURES = ("trajectory", "exceedance", "diagnostics", "genealogy", "composite")

RES_COLOR = "#1f77b4"
DS_STYLE = {
    "gencast_walkers": ("#2ca02c", "GenCast direct (Gate 3 walkers)"),
    "gencast_xres":    ("#8c564b", "GenCast direct (xres 24-member)"),
    "fcn3":            ("#9467bd", "FCN3 direct (24-member)"),
}
OBS_COLOR = "#d62728"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(ctx: R.Ctx) -> dict:
    p = A.res_dir(ctx.event, ctx.tag) / "compare.json"
    if not p.exists():
        raise SystemExit(
            f"no {p}\n  Run the reductions first:\n"
            f"    PYTHONPATH=. python -m aires.run_aires --stage ds --event {ctx.event}\n"
            f"    PYTHONPATH=. python -m aires.run_aires --stage compare "
            f"--event {ctx.event} --tag {ctx.tag}")
    out = R.read_json(p)
    rp = A.res_dir(ctx.event, ctx.tag) / "res_result.json"
    out["res"] = R.read_json(rp) if rp.exists() else {}
    return out


def _fig_path(ctx: R.Ctx, kind: str) -> Path:
    A.fig_dir(ctx.event).mkdir(parents=True, exist_ok=True)
    return A.fig_dir(ctx.event) / f"aires_{kind}_{ctx.event}_{ctx.tag}.png"


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
    return path


# --------------------------------------------------------------------------- #
# Pure helpers (tested without matplotlib)
# --------------------------------------------------------------------------- #
def res_exceedance_se(values, weights, log_Z: float, thresholds, sign: float = 1.0):
    """Delta-method standard error of ``Z * mean(1{sign*x >= sign*a} * w)``.

    A lower bound: it treats the walkers as independent, which standardization makes them
    not. Reported so the curve is not read as exact, not as a confidence statement.
    """
    v = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    Z = float(np.exp(log_Z))
    n = v.size
    out = []
    for a in np.atleast_1d(thresholds):
        f = (sign * v >= sign * float(a)).astype("float64") * w
        out.append(Z * float(np.std(f, ddof=1)) / np.sqrt(n) if n > 1 else np.nan)
    return np.array(out)


def sibling_pairs(parents) -> list[tuple[int, int]]:
    """Slot pairs that came out of one resampling sharing a parent.

    These are the walkers whose separation is entirely the doing of GenCast's diffusion
    noise: a clone starts from a bit-identical state and differs only in its RNG stream.
    If they do not separate, the scheme is propagating duplicates and the paper's
    hand-tuned spherical-harmonic perturbation would be needed.
    """
    p = np.asarray(parents, dtype="int64")
    out = []
    for par in np.unique(p):
        kids = np.flatnonzero(p == par)
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                out.append((int(kids[i]), int(kids[j])))
    return out


def unrelated_pairs(parents, n_max: int = 64, seed: int = 0) -> list[tuple[int, int]]:
    """Slot pairs from DIFFERENT parents - the reference a clone pair has to reach."""
    p = np.asarray(parents, dtype="int64")
    rng = np.random.default_rng(seed)
    cand = [(i, j) for i in range(p.size) for j in range(i + 1, p.size) if p[i] != p[j]]
    if len(cand) > n_max:
        cand = [cand[k] for k in rng.choice(len(cand), n_max, replace=False)]
    return cand


def weighted_quantile(values, weights, q: float) -> float:
    v = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q, c / np.sum(w), v))


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).correlation
    return float(r)


# --------------------------------------------------------------------------- #
# Figure 1 - exceedance / return period
# --------------------------------------------------------------------------- #
def plot_exceedance(ctx: R.Ctx, d: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aires import aindex as AI

    sign = float(d["config"].get("tail_sign", 1.0))
    curve = d["curve"]
    a = np.array([r["threshold"] for r in curve])
    res = np.array([r["res"] for r in curve])
    al = np.array(d["realized"]["box"], dtype=float)
    w = np.array(d["weights"], dtype=float)
    se = res_exceedance_se(al, w, d["log_Z"], a, sign)
    obs = d.get("observed")
    ds = d.get("ds", {})
    unit = "K" if ctx.ev.metric == "t2m_anom" else ""

    # A floor chosen from the data. Clamping zeros to 1e-12 instead would draw every
    # direct-sampling band across twelve decades and bury the comparison the figure is for.
    pos = [v for v in res if v > 0]
    for key in DS_STYLE:
        if key in ds:
            n = len(ds[key]["box"])
            pos.append(1.0 / (2 * n))
    floor = max(min(pos) / 5.0, 1e-9) if pos else 1e-6

    fig, ax = plt.subplots(1, 2, figsize=(14.8, 6.0),
                           gridspec_kw={"width_ratios": [1.6, 1]})

    # --- (0) the exceedance curve ------------------------------------------- #
    a0 = ax[0]
    a0.plot(a, np.maximum(res, floor), "-o", ms=4, color=RES_COLOR, zorder=5,
            label=f"AI+RES  (N={d['config']['n_walkers']}, K={len(d['ess_by_step'])})")
    a0.fill_between(a, np.maximum(res - se, floor), np.maximum(res + se, floor),
                    color=RES_COLOR, alpha=0.20, lw=0, zorder=2)
    for key, (col, lab) in DS_STYLE.items():
        if key not in curve[0]:
            continue
        p_ = np.array([r[key] for r in curve])
        hi = np.array([r[f"{key}_ci"][1] for r in curve])
        n = ds.get(key, {}).get("n", "?")
        m = p_ > 0
        if m.any():
            a0.step(a[m], p_[m], where="post", color=col, lw=1.7,
                    label=f"{lab}, n={n}", zorder=4)
            a0.fill_between(a[m], np.maximum(np.array([r[f"{key}_ci"][0] for r in curve])[m],
                                             floor), hi[m], step="post", color=col,
                            alpha=0.12, lw=0, zorder=1)
        # Where the empirical probability is ZERO the honest statement is the 95% upper
        # bound, not a line at the floor: n draws that all missed do not say "impossible".
        if (~m).any():
            a0.plot(a[~m], hi[~m], ls=(0, (2, 2)), color=col, lw=1.2, zorder=3)
            vmax = max(np.asarray(ds[key]["box"], dtype=float) * sign) * sign
            a0.plot([vmax], [1.0 / n], marker="x", ms=8, mew=2, color=col, zorder=6)
        if not m.any():
            a0.plot([], [], color=col, lw=1.7, label=f"{lab}, n={n}")
    a0.plot([], [], ls=(0, (2, 2)), color="0.4", lw=1.2,
            label="95% upper bound where no member reached it")
    a0.plot([], [], marker="x", ls="none", ms=8, mew=2, color="0.4",
            label="most extreme member drawn")
    if obs is not None:
        a0.axvline(obs, color=OBS_COLOR, lw=1.5, ls="--", zorder=7)
        a0.text(obs, 0.965, " ERA5 observed ", color=OBS_COLOR, fontsize=9,
                transform=a0.get_xaxis_transform(), va="top",
                ha="left" if sign > 0 else "right")
    a0.set_yscale("log")
    a0.set_ylim(floor, 1.6)
    a0.set_xlabel(rf"threshold $a$ on $A_L$   ({unit})")
    a0.set_ylabel(r"$P(A_L \geq a)$" if sign > 0 else r"$P(A_L \leq a)$")
    a0.set_title(f"{ctx.ev.label}: exceedance of the {ctx.ev.metric} index over the "
                 f"{AI.box_for(ctx.event).name} box", fontsize=10)
    a0.grid(alpha=0.25, which="both")
    a0.legend(fontsize=7.5, loc="lower left")
    if sign < 0:
        a0.invert_xaxis()

    a0r = a0.twinx()
    a0r.set_yscale("log")
    a0r.set_ylim(*a0.get_ylim())
    ticks = [t for t in a0.get_yticks() if floor <= t <= 1.0]
    a0r.set_yticks(ticks)
    a0r.set_yticklabels([f"1 in {1/t:,.0f}" if 1 / t >= 1 else "" for t in ticks],
                        fontsize=8)
    a0r.set_ylabel("direct members you would have to draw", fontsize=9)
    a0r.grid(False)

    # --- (1) the population, one lane per source ---------------------------- #
    a1 = ax[1]
    lanes = [("AI+RES (weighted)", None)] + [
        (DS_STYLE[k][1], k) for k in DS_STYLE if k in ds]
    rng = np.random.default_rng(0)
    for y, (lab, key) in enumerate(lanes):
        if key is None:
            j = rng.uniform(-0.16, 0.16, al.size)
            sc = a1.scatter(al, y + j, c=w, cmap="viridis", s=30, edgecolor="0.25",
                            linewidth=0.3, zorder=3)
        else:
            v = np.asarray(ds[key]["box"], dtype=float)
            a1.scatter(v, np.full(v.size, y) + rng.uniform(-0.12, 0.12, v.size),
                       marker="|", s=180, color=DS_STYLE[key][0], alpha=0.9, zorder=2)
    if obs is not None:
        a1.axvline(obs, color=OBS_COLOR, lw=1.5, ls="--", zorder=4)
        a1.text(obs, len(lanes) - 0.45, " ERA5 observed", color=OBS_COLOR, fontsize=8.5,
                ha="left" if sign > 0 else "right")
    fig.colorbar(sc, ax=a1, pad=0.02, label=r"weight $e^{-V_K}$ (undoes the tilt)")
    a1.set_yticks(range(len(lanes)))
    a1.set_yticklabels([l for l, _ in lanes], fontsize=8)
    a1.set_ylim(-0.6, len(lanes) - 0.4)
    a1.invert_yaxis()
    a1.set_xlabel(rf"realized $A_L$   ({unit})")
    reach = "" if obs is None else \
        f"   {int(np.sum(sign * al >= sign * obs))}/{al.size} of the RES population reach it"
    a1.set_title(f"where each method's members landed{reach}", fontsize=10)
    a1.grid(alpha=0.25, axis="x")

    fig.suptitle(f"AI+RES pilot - {ctx.event} ({ctx.tag}): does resampling reach a tail "
                 f"direct sampling misses?", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, _fig_path(ctx, "exceedance"))


# --------------------------------------------------------------------------- #
# Figures 3 + 4 - score skill and population health
# --------------------------------------------------------------------------- #
def plot_diagnostics(ctx: R.Ctx, d: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = d.get("res", {})
    steps = res.get("dmc", {}).get("steps", [])
    thetas = {int(t["leg"]): t for t in res.get("theta", [])}
    al = np.array(d["realized"]["box"], dtype=float)
    sign = float(d["config"].get("tail_sign", 1.0))
    n = al.size
    unit = "K" if ctx.ev.metric == "t2m_anom" else ""

    scored = [s for s in steps if float(s["C"]) != 0.0]
    ncol = max(len(scored), 1)
    span = 2 * ncol                    # so the bottom row can split evenly at any ncol
    fig = plt.figure(figsize=(3.7 * ncol + 1.2, 8.0))
    gs = fig.add_gridspec(2, span, height_ratios=[1, 1.05], hspace=0.42,
                      wspace=0.90)

    # --- top row: theta(t_k) vs the walker's OWN realized outcome ------------ #
    #
    # This is Gate 3's statistic on production data, with one difference that has to be
    # stated: the population at t_k is no longer free-running, so a walker's descendants
    # are not independent draws. It measures whether the score ranked the walkers it
    # actually resampled, which is what the algorithm needed of it.
    for j, s in enumerate(scored):
        a = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        k = int(s["step"])
        rec = thetas.get(k, {})
        th = np.array(rec.get("box", s["theta"]), dtype=float)
        # map each FINAL walker back to the slot it occupied at this resampling
        lin = d["realized"]["lineage"]
        segs = A.res_legs(d["config"]["leads"], d["config"]["horizon_days"],
                          d["config"]["C"])
        seg_k = segs[k - 1].last_segment
        slot = np.array([lin[i][str(seg_k)] for i in range(n)])
        x, y = th[slot], al
        a.scatter(x, y, s=26, color=RES_COLOR, edgecolor="0.3", linewidth=0.3, zorder=3)
        rho = spearman(x, y)
        obs = d.get("observed")
        if obs is not None:
            a.axhline(obs, color=OBS_COLOR, lw=1.2, ls="--", zorder=2)
        a.set_title(rf"$t_k$ = {rec.get('lead_days', '?')} d,  $C_k$ = {s['C']:g}"
                    "\n" rf"$\rho_s$ = {rho:+.2f} (descendants, not independent)",
                    fontsize=9)
        a.set_xlabel(rf"score $\theta$ at $t_k$   ({unit})")
        if j == 0:
            a.set_ylabel(rf"realized $A_L$   ({unit})")
        a.grid(alpha=0.25)

    # --- bottom left: ESS and multiplicities --------------------------------- #
    a = fig.add_subplot(gs[1, :span // 2])
    ks = [int(s["step"]) for s in steps]
    ess = np.array([float(s["ess"]) for s in steps]) / n
    leads = [thetas.get(k, {}).get("lead_days", k) for k in ks]
    a.axhline(0.25, color=OBS_COLOR, ls="--", lw=1.2, zorder=5,
              label="aires.md target: ESS > N/4")
    a.plot(ks, ess, "-o", color=RES_COLOR, zorder=6, label="ESS / N")
    a.set_ylim(0, 1.32)
    a.set_zorder(2)
    a.patch.set_visible(False)          # bars live on the twin axis, behind the line
    a.set_xticks(ks)
    a.set_xticklabels([f"{k}\n{l:g} d" for k, l in zip(ks, leads)], fontsize=8)
    a.set_xlabel("resampling step")
    a.set_ylabel("ESS / N")
    a.grid(alpha=0.25)
    a2 = a.twinx()
    mult = [int(np.max(s["counts"])) for s in steps]
    killed = [int(np.sum(np.asarray(s["counts"]) == 0)) for s in steps]
    a2.bar(np.array(ks) - 0.15, mult, width=0.3, color="0.6", label="max children",
           zorder=1)
    a2.bar(np.array(ks) + 0.15, killed, width=0.3, color="0.85", edgecolor="0.5",
           label="killed", zorder=1)
    a2.set_ylabel("count", fontsize=9, labelpad=1)
    a2.set_zorder(1)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left", ncol=2,
             framealpha=0.95, borderpad=0.3, columnspacing=1.0).set_zorder(9)
    a.set_title("population health", fontsize=10)

    # --- bottom right: the weights the estimator has to undo ----------------- #
    if True:
        a = fig.add_subplot(gs[1, span // 2:])
        w = np.array(d["weights"], dtype=float)
        a.hist(w, bins=min(24, max(6, n // 4)), color=RES_COLOR, alpha=0.85)
        a.axvline(1.0, color="0.4", ls=":", lw=1.2)
        a.set_xlabel(r"final weight $e^{-V_K}$")
        a.set_ylabel("walkers")
        nf = d.get("n_founders", "?")
        a.set_title(f"weights span {w.min():.3g} .. {w.max():.3g};  "
                    f"log Z = {d['log_Z']:+.3f};  normalization check = "
                    f"{d.get('normalization_check', float('nan')):.2f}\n"
                    f"{nf}/{n} of the original walkers still have descendants",
                    fontsize=9)
        a.grid(alpha=0.25)

    fig.suptitle(f"AI+RES pilot - {ctx.event} ({ctx.tag}): did the score rank, and did "
                 f"N={n} hold up?", fontsize=11.5)
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.06, right=0.96)
    return _save(fig, _fig_path(ctx, "diagnostics"))


# --------------------------------------------------------------------------- #
# Figures 2 + 6 - genealogy and clone divergence
# --------------------------------------------------------------------------- #
def _t2m_series(ctx: R.Ctx, slot: int, segment: int) -> xr.DataArray:
    with xr.open_dataset(ctx.diag_path(slot, segment)) as ds:
        da = ds["2m_temperature"]
        if "batch" in da.dims:
            da = da.isel(batch=0)
        return da.load()


def clone_divergence(ctx: R.Ctx, d: dict, max_pairs: int = 48) -> dict:
    """RMS T2m separation of sibling walkers versus lead, against unrelated pairs.

    Measured on the CONUS crop the diagnostics cubes carry, not globally - that is what is
    on disk after the states are pruned, and it is the region the observable lives in.
    """
    cfg = d["config"]
    legs = A.res_legs(cfg["leads"], cfg["horizon_days"], cfg["C"])
    lin = d["realized"]["lineage"]
    n = len(lin)
    last = legs[-1]
    # siblings at the FINAL resampling: final slots whose parent slot agrees
    parents = np.array([lin[i][str(legs[-2].last_segment)] for i in range(n)])
    sib = sibling_pairs(parents)[:max_pairs]
    unr = unrelated_pairs(parents, n_max=max_pairs)
    if not sib:
        return {}

    cache: dict[int, xr.DataArray] = {}

    def series(slot_of_final: int) -> xr.DataArray:
        if slot_of_final not in cache:
            parts = [_t2m_series(ctx, lin[slot_of_final][str(s)], s) for s in last.segments]
            cache[slot_of_final] = xr.concat(parts, dim="time").sortby("time")
        return cache[slot_of_final]

    def rms(pairs):
        out = []
        for i, j in pairs:
            a, b = series(i), series(j)
            out.append(np.sqrt(((a - b) ** 2).mean(dim=("lat", "lon"))).values)
        return np.array(out)

    s_rms, u_rms = rms(sib), rms(unr)
    t = pd.DatetimeIndex(series(sib[0][0])["time"].values)
    lead = (t - pd.Timestamp(ctx.ev.init)).total_seconds() / 86400.0
    return dict(lead_days=lead.tolist(), sibling=s_rms.tolist(),
                unrelated=u_rms.tolist(), n_sibling=len(sib), n_unrelated=len(unr),
                born_at=legs[-2].lead_end_days)


def plot_genealogy(ctx: R.Ctx, d: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    cfg = d["config"]
    legs = A.res_legs(cfg["leads"], cfg["horizon_days"], cfg["C"])
    lin = d["realized"]["lineage"]
    al = np.array(d["realized"]["box"], dtype=float)
    n = al.size
    norm = colors.Normalize(vmin=float(np.min(al)), vmax=float(np.max(al)))
    cmap = matplotlib.colormaps["coolwarm" if ctx.ev.metric == "t2m_anom" else "viridis"]

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.2),
                           gridspec_kw={"width_ratios": [1.35, 1]})

    # --- (0) the tree -------------------------------------------------------- #
    a0 = ax[0]
    segs = sorted({s for leg in legs for s in leg.segments})
    x = [0.0] + [s * A.GATE3_SEG_DAYS for s in segs]
    for i in np.argsort(al):                      # extremes drawn last, on top
        y = [lin[i][str(segs[0])]] + [lin[i][str(s)] for s in segs]
        a0.plot(x, y, "-", lw=1.1, color=cmap(norm(al[i])), alpha=0.85,
                solid_capstyle="round")
    for leg in legs[:-1]:
        a0.axvline(leg.lead_end_days, color="0.75", lw=0.8, ls=":")
        a0.text(leg.lead_end_days, 1.01, f"$C$={leg.C:g}", fontsize=7.5,
                ha="center", color="0.4", transform=a0.get_xaxis_transform())
    a0.set_xlabel("lead (days from the event init)")
    a0.set_ylabel("walker slot")
    a0.set_title(f"genealogy: {d.get('n_founders', '?')}/{n} founders still have "
                 f"descendants at the horizon", fontsize=10, pad=16)
    a0.grid(alpha=0.2)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=a0, pad=0.02,
                 label=r"realized $A_L$ of the descendant")

    # --- (1) do clones actually separate? ------------------------------------ #
    a1 = ax[1]
    div = d.get("divergence") or {}
    if div:
        lead = np.array(div["lead_days"])
        s = np.array(div["sibling"])
        u = np.array(div["unrelated"])
        a1.plot(lead, s.mean(0), "-o", ms=4, color=RES_COLOR,
                label=f"sibling clones (n={div['n_sibling']})")
        a1.fill_between(lead, np.percentile(s, 10, axis=0),
                        np.percentile(s, 90, axis=0), color=RES_COLOR, alpha=0.18, lw=0)
        a1.plot(lead, u.mean(0), "-", color="0.45",
                label=f"unrelated walkers (n={div['n_unrelated']})")
        a1.fill_between(lead, np.percentile(u, 10, axis=0),
                        np.percentile(u, 90, axis=0), color="0.45", alpha=0.12, lw=0)
        a1.axvline(div["born_at"], color=OBS_COLOR, lw=1.2, ls="--")
        a1.text(div["born_at"], 0.02, " cloned here", color=OBS_COLOR, fontsize=8,
                va="bottom", transform=a1.get_xaxis_transform())
        a1.set_title("clone divergence: GenCast's diffusion noise is the only\n"
                     "thing separating a clone from its sibling", fontsize=10)
    else:
        a1.text(0.5, 0.5, "no sibling pairs at the final resampling",
                ha="center", va="center", transform=a1.transAxes, color="0.4")
        a1.set_title("clone divergence", fontsize=10)
    a1.set_xlabel("lead (days from the event init)")
    a1.set_ylabel("RMS 2 m temperature difference over CONUS   (K)")
    a1.grid(alpha=0.25)
    a1.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"AI+RES pilot - {ctx.event} ({ctx.tag}): ancestry, and whether cloning "
                 f"produces distinct walkers", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, _fig_path(ctx, "genealogy"))


# --------------------------------------------------------------------------- #
# Figure 5 - composite maps of the RES-selected extremes
# --------------------------------------------------------------------------- #
def _field_at(ctx: R.Ctx, slot_of_final: int, lin: list, segments, when, var: str,
              level: int | None = None) -> xr.DataArray:
    for s in segments:
        p = ctx.diag_path(lin[slot_of_final][str(s)], s)
        with xr.open_dataset(p) as ds:
            t = pd.DatetimeIndex(ds["time"].values)
            if pd.Timestamp(when) not in set(t):
                continue
            da = ds[var]
            if "batch" in da.dims:
                da = da.isel(batch=0)
            if level is not None:
                da = da.sel(level=level)
            return da.sel(time=pd.Timestamp(when)).load()
    raise KeyError(f"no frame at {when} for final walker {slot_of_final}")


def plot_composite(ctx: R.Ctx, d: dict, top_frac: float = 0.15) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from gencast_s2s.plotting import _to_180, _add_conus_features

    cfg = d["config"]
    legs = A.res_legs(cfg["leads"], cfg["horizon_days"], cfg["C"])
    segs = legs[-1].segments
    lin = d["realized"]["lineage"]
    al = np.array(d["realized"]["box"], dtype=float)
    sign = float(cfg.get("tail_sign", 1.0))
    n = al.size
    k = max(2, int(round(top_frac * n)))
    top = np.argsort(sign * al)[-k:]

    peak = pd.Timestamp(ctx.ev.peak)
    lags = [(-3, peak - pd.Timedelta(days=3)), (0, peak)]
    rows = [("2m_temperature", None, "T2m"), ("geopotential", 500, "Z500")]

    ncol = 3
    nrow = len(rows) * len(lags)
    fig = plt.figure(figsize=(14.0, 2.85 * nrow))
    idx = 0
    for var, lev, vlabel in rows:
        for lag, when in lags:
            allf = xr.concat([_field_at(ctx, i, lin, segs, when, var, lev)
                              for i in range(n)], dim="w")
            mean = allf.mean("w")
            comp = allf.isel(w=top).mean("w")
            diff = comp - mean
            unit = "K" if var == "2m_temperature" else r"m$^2$ s$^{-2}$"
            amp = float(np.nanmax(np.abs(diff.values))) or 1.0
            lo = float(np.nanmin(np.concatenate([comp.values.ravel(), mean.values.ravel()])))
            hi = float(np.nanmax(np.concatenate([comp.values.ravel(), mean.values.ravel()])))
            for c, (da, title, cmap, vmin, vmax) in enumerate([
                (comp, f"RES-selected extremes (top {k}/{n})", "inferno", lo, hi),
                (mean, f"all {n} walkers (mean)", "inferno", lo, hi),
                (diff, "difference", "RdBu_r", -amp, amp),
            ]):
                idx += 1
                ax = fig.add_subplot(nrow, ncol, idx, projection=ccrs.PlateCarree())
                from gencast_s2s import config as GC
                ax.set_extent(GC.CONUS_EXTENT, ccrs.PlateCarree())
                dd = _to_180(da)
                m = ax.pcolormesh(dd.lon, dd.lat, dd.values, transform=ccrs.PlateCarree(),
                                  cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
                _add_conus_features(ax)
                ax.set_title(f"{vlabel}  lag {lag:+d} d   {title}", fontsize=8.5)
                cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.05,
                                  shrink=0.82, aspect=32)
                cb.set_label(unit, fontsize=8)
                cb.ax.tick_params(labelsize=7)

    fig.suptitle(f"AI+RES pilot - {ctx.event} ({ctx.tag}): what the resampling selected. "
                 f"Composites at the event peak and 3 days before.\n"
                 f"Z500 is the raw field: the only climatology on this machine is T2m, so "
                 f"the 'difference' column is the anomaly relative to the walker ensemble "
                 f"itself, not to a climate.", fontsize=10.5, y=0.997, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return _save(fig, _fig_path(ctx, "composite"))


# --------------------------------------------------------------------------- #
# Figure 7 - the trajectory plot (the paper's Fig. 1 shape)
#
# What the other four figures cannot show: the algorithm happening. Every other panel in
# this set is a reduction - a distribution, a scatter, a slot index. This one puts the
# observable itself on the y axis and lead time on the x axis, so a reader sees the
# population being pushed up the tail one barrier at a time, sees which branches were
# killed to pay for it, and sees the 7-day window at the end that turns 64 trajectories
# into 64 numbers.
#
# The y axis is the walker's OWN running index - the cos(lat)-weighted box mean of the
# instantaneous T2m anomaly at each 12 h frame - NOT the score theta. theta is a forecast
# of where the walker will end up and lives at five discrete leads; this is where the
# walker actually is. They are different quantities and mixing them on one axis is how a
# figure ends up claiming the score was right when it is only self-consistent.
# --------------------------------------------------------------------------- #
DEAD_COLOR = "0.72"


# The variables ``aindex.instantaneous_field`` actually consumes, per metric. A walker
# diagnostics cube is ~35 MB of 12 prognostic variables on 13 levels; 64 of them held at
# once is 2.2 GB, and Derecho's login node OOM-kills at less than that. Reading only the
# metric's own variables drops a segment to well under 100 MB.
METRIC_VARS = {
    "t2m_anom": ("2m_temperature",),
    "u850_speed": ("u_component_of_wind", "v_component_of_wind"),
    "u850_speed_max": ("u_component_of_wind", "v_component_of_wind"),
    "tp_total": ("total_precipitation_12hr",),
    "tp_max12h": ("total_precipitation_12hr",),
}


def _seg_index_series(ctx: R.Ctx, segment: int, n_walkers: int):
    """``(lead_days[nt], box[n_walkers, nt])`` - the running index for EVERY slot.

    Loaded a segment at a time and reduced with one call to ``aindex.index_series``, so
    the 1990-2019 climatology is sampled 7 times rather than 448. The slots at a given
    segment share a time axis by construction (they are one leg of one schedule), which
    is what makes the concat legal - and ``concat`` would raise if it were not.
    """
    from aires import aindex as AI

    parts = []
    for w in range(n_walkers):
        p = ctx.diag_path(w, segment)
        if not p.exists():
            raise SystemExit(
                f"no {p}\n  The trajectory figure needs every walker's diagnostics cube, "
                f"including the branches that were killed. On a3mega they are under "
                f"runs/aires/{ctx.event}/res/{ctx.tag}/walkers/w*/step*/diag.nc.")
        want = METRIC_VARS.get(ctx.ev.metric)
        with xr.open_dataset(p) as ds:
            if want is None:
                raise SystemExit(f"no variable list for metric {ctx.ev.metric!r}; add it "
                                 f"to aplots.METRIC_VARS")
            missing = [v for v in want if v not in ds]
            if missing:
                raise SystemExit(f"{p}: diagnostics cube is missing {missing}")
            d = ds[list(want)].load()
        d = d.drop_vars([c for c in ("lead_h",) if c in d.coords])
        if "batch" in d.dims:
            d = d.isel(batch=0, drop=True)
        parts.append(d)
    big = xr.concat(parts, dim="walker", coords="minimal", compat="override")
    ser = AI.index_series(big, ctx.event, ctx.ev.metric)["box"].transpose("walker", "time")
    t = pd.DatetimeIndex(big["time"].values)
    lead = (t - pd.Timestamp(ctx.ev.init)).total_seconds() / 86400.0
    out = (np.asarray(lead, dtype=float), np.asarray(ser.values, dtype=float))
    for d in parts:
        d.close()
    big.close()
    return out


def _init_index_series(ctx: R.Ctx):
    """``(lead_days[2], box[2])`` for the event's ERA5 init frames, or ``None``.

    ONE series, not 64: every walker is launched from these same two analysis frames, and
    the walk's first FORECAST frame is at +12 h. Without this anchor the whole forest
    starts half a day inside the axis, which reads as the trajectories being misaligned
    with the axis rather than as the 12 h step it is. Carrying both frames (-12 h and 0 h)
    also gives the lead-0 point a full centred smoothing window, so the curves leave the
    origin at the same value they would have if the analysis ran on forever.

    Returns ``None`` rather than raising if the init file is absent: it is built by the
    xres prep stage and lives outside ``runs/aires``, so a tree copied to another machine
    can be missing it, and losing the anchor is not worth losing the figure over.
    """
    from aires import aindex as AI

    path = A.gencast_inputs_path(ctx.event)
    if not path.exists():
        print(f"    no init frames at {path}; drawing from the first forecast frame")
        return None
    want = list(METRIC_VARS[ctx.ev.metric])
    with xr.open_dataset(path) as ds:
        if any(v not in ds for v in want):
            return None
        d = ds[want].load()
    if "batch" in d.dims:
        d = d.isel(batch=0, drop=True)
    box = AI.index_series(d, ctx.event, ctx.ev.metric)["box"]
    t = pd.DatetimeIndex(d["time"].values)
    lead = (t - pd.Timestamp(ctx.ev.init)).total_seconds() / 86400.0
    d.close()
    return np.asarray(lead, dtype=float), np.asarray(box.values, dtype=float)


def _from_root(root, lead_s, box_s):
    """A segment-1 polyline extended back onto the ERA5 init, ready for ``daily``."""
    if root is None:
        return daily(lead_s, box_s)
    lead_r, box_r = root
    return daily(np.concatenate([lead_r, lead_s]),
                 np.concatenate([box_r, box_s]), drop_first=len(lead_r) - 1)


def daily(x, y, drop_first: int = 0):
    """A 24 h running mean of a 12-hourly series, CENTRED ON ITS OWN TIME AXIS.

    The raw index carries a real diurnal swing of several kelvin - a heat dome's daytime
    anomaly is much larger than its nighttime one, and the climatology is sampled per hour
    so the anomaly does not remove it - which on 64 lineages reads as noise and hides the
    trend the figure is about.

    The filter is the trapezoid over ``[t-12h, t+12h]``, weights ``(1, 2, 1)/4``, i.e. the
    mean of the two consecutive-pair means straddling ``t``. Like a pair mean it is an
    exact null at the 24 h period for a 12-hourly series, but unlike one it does not move
    the sample off its own timestamp. That distinction is the whole point: an earlier
    version returned the pair means on their midpoints, which slid every curve 6 h to the
    left, so trajectories appeared to start at lead 0.75 d rather than 0.5 d and every
    branch appeared to die a quarter-day BEFORE the barrier that actually killed it.

    At a genuine end of a lineage there is no frame on one side, and the window is closed
    with ``y[0] + y[1] - y[2]`` rather than by mirroring. Mirroring would collapse the
    filter to the pair mean, which is still diurnal-free but is centred half a step inside
    the series, so the last point of every lineage - the one sitting on the event peak,
    where ``A_L`` is taken - would be pulled a quarter-day back down its own trend. This
    padding is the one that makes the end weights ``(0.75, 0.5, -0.25)``: exact on a linear
    trend, still an exact null at 24 h.

    ``drop_first`` trims lead-in frames supplied only so that the first RETAINED point gets
    a full centred window (see ``trajectory_tree``); it is applied after smoothing, so a
    branch drawn from a barrier starts exactly ON that barrier.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return x, y
    if y.size >= 3:
        lo, hi = y[0] + y[1] - y[2], y[-1] + y[-2] - y[-3]
    else:
        lo, hi = y[1], y[-2]                          # only a pair: mirror and accept it
    pad = np.concatenate([[lo], y, [hi]])
    sm = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
    return x[drop_first:], sm[drop_first:]


def _parent_slot(legs, parents_by_leg: dict, walker: int, segment: int):
    """Which slot's state segment ``(walker, segment)`` continued. ``None`` at the root.

    The same rule as ``run_aires.parent_of``: only the first segment of a leg crosses a
    resampling barrier. Duplicated here rather than imported because that function returns
    a path and this needs the index.
    """
    leg = next(l for l in legs if segment in l.segments)
    if segment == 1:
        return None
    if segment == leg.first_segment:
        p = parents_by_leg.get(leg.k)
        return walker if p is None else int(p[walker])
    return walker


def trajectory_tree(ctx: R.Ctx, d: dict) -> dict:
    """Every branch the pilot rolled, alive or dead, as index-vs-lead polylines.

    The surviving 64 come straight out of ``compare.json`` (``realized.series_box``), which
    ``run_aires`` assembled across the genealogy with the same code that produced ``A_L``.
    The dead branches are not in any artifact - nothing downstream needed them - so they
    are rebuilt here from the diagnostics cubes and cross-checked against the surviving
    ones. If the rebuild disagrees with ``series_box``, the genealogy walk is wrong and the
    figure says so instead of drawing a plausible tree.
    """
    cfg = d["config"]
    legs = A.res_legs(cfg["leads"], cfg["horizon_days"], cfg["C"])
    n = int(cfg["n_walkers"])
    segments = sorted({s for leg in legs for s in leg.segments})
    par = R.parents_by_leg(ctx)

    seg = {}
    for s in segments:
        seg[s] = _seg_index_series(ctx, s, n)
        print(f"    segment {s}: {seg[s][1].shape[1]} frames, "
              f"lead {seg[s][0][0]:.1f}-{seg[s][0][-1]:.1f} d")

    root = _init_index_series(ctx)
    lin = d["realized"]["lineage"]
    alive = {(int(lin[i][str(s)]), s) for i in range(len(lin)) for s in segments}

    # Rebuild one surviving trajectory the hard way and check it against the artifact.
    lead_art = np.asarray(d["realized"]["lead_days"], dtype=float)
    rebuilt = np.concatenate([seg[s][1][int(lin[0][str(s)])] for s in segments])
    lead_reb = np.concatenate([seg[s][0] for s in segments])
    art = np.asarray(d["realized"]["series_box"][0], dtype=float)
    if rebuilt.shape != art.shape or not np.allclose(rebuilt, art, atol=1e-6):
        raise SystemExit(
            "the genealogy walk in this figure disagrees with realized.series_box in "
            f"compare.json (max |diff| {np.nanmax(np.abs(rebuilt - art)) if rebuilt.shape == art.shape else float('nan'):.4g}). "
            "Do not read the tree; fix the parent lookup first.")
    if not np.allclose(lead_reb, lead_art, atol=1e-6):
        raise SystemExit("rebuilt lead axis does not match compare.json's lead_days")

    # children, so a branch with no descendant can be drawn as a branch that ENDS
    kids: dict[tuple, list] = {}
    for s in segments:
        if s == 1:
            continue
        for w in range(n):
            kids.setdefault((_parent_slot(legs, par, w, s), s - 1), []).append(w)

    branches, deaths = [], []
    for s in segments:
        lead_s, box_s = seg[s]
        for w in range(n):
            if (w, s) in alive:
                continue                      # drawn as part of a full surviving path
            p = _parent_slot(legs, par, w, s)
            if p is None:
                xs, ys = _from_root(root, lead_s, box_s[w])
            else:
                # TWO lead-in frames from the parent, not one. The branch has to be drawn
                # from the barrier itself, and the point sitting ON the barrier only gets
                # a full centred 24 h window if the frame before it is also present. The
                # first is dropped after smoothing - it belongs to the parent's polyline.
                lead_p, box_p = seg[s - 1]
                k = min(2, lead_p.size)
                x = np.concatenate([lead_p[-k:], lead_s])
                y = np.concatenate([box_p[p][-k:], box_s[w]])
                xs, ys = daily(x, y, drop_first=k - 1)
            branches.append((xs.tolist(), ys.tolist()))
            if not kids.get((w, s)):
                deaths.append((float(xs[-1]), float(ys[-1])))

    killed = {}
    for leg in legs:
        p = par.get(leg.k)
        if p is not None:
            killed[leg.k] = n - len(set(int(x) for x in p))
    return dict(lead=lead_art.tolist(), segments=segments, dead=branches,
                deaths=deaths, killed=killed, n_dead_nodes=len(branches),
                root=None if root is None else [root[0].tolist(), root[1].tolist()])


def plot_trajectory(ctx: R.Ctx, d: dict, tree: dict | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    cfg = d["config"]
    legs = A.res_legs(cfg["leads"], cfg["horizon_days"], cfg["C"])
    n = int(cfg["n_walkers"])
    tree = tree if tree is not None else trajectory_tree(ctx, d)

    lead = np.asarray(tree["lead"], dtype=float)
    series = np.asarray(d["realized"]["series_box"], dtype=float)   # (n, nt)
    al = np.asarray(d["realized"]["box"], dtype=float)
    obs = d.get("observed")
    theta = d.get("res", {}).get("theta") or []

    norm = colors.Normalize(vmin=float(al.min()), vmax=float(al.max()))
    # NOT coolwarm here, unlike the genealogy figure: its midpoint is near-white, and on
    # this panel a mid-A_L survivor drawn in near-white is indistinguishable from a grey
    # killed branch - the one distinction the figure exists to make. A saturated ramp with
    # no neutral point, truncated away from its palest end, keeps every survivor readable.
    base = matplotlib.colormaps["YlOrRd" if ctx.sign > 0 else "YlGnBu"]
    cmap = colors.LinearSegmentedColormap.from_list(
        "survivors", base(np.linspace(0.32, 1.0, 256)))

    fig = plt.figure(figsize=(15.5, 7.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[4.6, 1.0, 0.05], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1], sharey=ax)
    cax = fig.add_subplot(gs[0, 2])

    # --- the x axis: LEAD TIME TO THE EVENT, counted down --------------------------- #
    # Everything upstream carries lead-from-init (0 d at the init, 21 d at the peak),
    # which is the natural coordinate for a forecast and the wrong one for this figure:
    # the reader is watching a population converge on a fixed event, so the axis has to
    # count down to it - 21 d of lead on the left, 0 d at the peak on the right. Time
    # still runs left to right; only the coordinate is reversed. The conversion happens
    # once, HERE, at the drawing boundary, so nothing upstream has to know about it and
    # no annotation can drift out of step with the curves.
    peak_lead = float(cfg["horizon_days"])

    def tb(v):
        """lead-from-init (days) -> lead time remaining to the event peak (days)."""
        return peak_lead - np.asarray(v, dtype=float)

    # --- the 7-day window A_L is the mean over ----------------------------------- #
    win = np.asarray(_window_lead(ctx, d), dtype=float)
    w0, w1 = float(win.min()), float(win.max())
    ax.axvspan(tb(w0), tb(w1), color="#f0c040", alpha=0.16, lw=0, zorder=0)
    ax.text(tb(0.5 * (w0 + w1)), 0.985,
            f"$A_L$ = the 7-day mean\nover these {win.size} frames",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5,
            color="#8a5a12", linespacing=1.35)

    # --- killed branches, then the survivors on top ------------------------------ #
    for x, y in tree["dead"]:
        ax.plot(tb(x), y, "-", lw=0.6, color=DEAD_COLOR, alpha=0.7, zorder=1,
                solid_capstyle="round")
    if tree["deaths"]:
        dx, dy = zip(*tree["deaths"])
        ax.plot(tb(dx), dy, "x", ms=3.6, mew=0.9, color="0.5", zorder=2)
    root = tree.get("root")
    root = None if root is None else (np.asarray(root[0], dtype=float),
                                      np.asarray(root[1], dtype=float))
    for i in np.argsort(al):                            # extremes drawn last, on top
        xs, ys = _from_root(root, lead, series[i])
        ax.plot(tb(xs), ys, "-", lw=1.0, color=cmap(norm(al[i])), alpha=0.9,
                zorder=3, solid_capstyle="round")

    # --- the barriers ------------------------------------------------------------- #
    for leg, th in zip([l for l in legs if l.resampling],
                       list(theta) + [None] * len(legs)):
        xb = float(tb(leg.lead_end_days))
        ax.axvline(xb, color="0.35", lw=0.9, ls=":", zorder=4)
        k = tree["killed"].get(leg.k + 1)
        lab = f"$C$={leg.C:g}"
        if k is not None:
            lab += f"\n{k} killed"
        ax.text(xb, 1.005, lab, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.6, color="0.3", linespacing=1.35)
        if th is not None:
            ax.plot([xb], [float(np.mean(th["box"]))], "D", ms=5.5, color="#175d7d",
                    mec="white", mew=0.8, zorder=6)

    if obs is not None:
        ax.axhline(float(obs), color=OBS_COLOR, lw=1.3, ls="--", zorder=5)
        ax.text(float(tb(0.15)), float(obs),
                f" ERA5 observed $A_L$ = {float(obs):+.2f} K",
                color=OBS_COLOR, fontsize=8.5, va="bottom", zorder=6)

    # Reversed limits rather than invert_xaxis(): the marginal panel shares only y, and a
    # single set_xlim keeps the direction with the data instead of as a later mutation.
    ax.set_xlim(float(tb(-0.4)), float(tb(peak_lead + 0.4)))
    # Ticks every RES_SEG_DAYS so they land ON the barriers rather than between them.
    ax.xaxis.set_major_locator(MultipleLocator(float(A.GATE3_SEG_DAYS)))
    init_s, peak_s = str(cfg["init"])[:10], str(cfg["peak"])[:10]
    ax.set_xlabel(f"lead time to the event peak (days)      "
                  f"init {init_s} at {peak_lead:g} d  \u2192  peak {peak_s} at 0 d")
    ax.set_ylabel(_ylab(ctx))
    n_bar = sum(1 for l in legs if l.resampling)
    n_kill = sum(tree["killed"].values())
    ax.set_title(
        f"{n} GenCast walkers, resampled at {n_bar} barriers on an FCN3 score - "
        f"{n_kill} lineages killed, {d.get('n_founders', '?')}/{n} founders still "
        f"represented at the peak", fontsize=10.5, pad=26)
    ax.grid(alpha=0.2)
    ax.legend(handles=[
        Line2D([], [], color=cmap(0.85), lw=1.4, label="surviving lineage (colored by its $A_L$)"),
        Line2D([], [], color=DEAD_COLOR, lw=1.0,
               label=f"killed branch ({tree['n_dead_nodes']} segments, x where it ends)"),
        Line2D([], [], color="#175d7d", marker="D", ls="none", ms=5,
               label=r"population-mean score $\theta$ at that barrier"),
    ], fontsize=8, loc="upper left", framealpha=0.92)

    # --- the marginal: where each method's members landed ------------------------ #
    _marginal(axm, d, al, obs)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label=r"realized $A_L$   (K)")
    fig.suptitle(f"AI+RES pilot - {ctx.event} ({ctx.tag}): the walk itself, barrier by "
                 f"barrier", fontsize=12)
    fig.subplots_adjust(left=0.055, right=0.945, top=0.865, bottom=0.115)
    return _save(fig, _fig_path(ctx, "trajectory"))


def _window_lead(ctx: R.Ctx, d: dict) -> np.ndarray:
    """The leads of the frames ``A_L`` is the mean of - the same rule as ``xmetrics``.

    Mirrors ``fc_times_in_window`` on the lead axis instead of the datetime axis, so the
    shaded band is the window the number was actually taken over rather than a nominal
    "last 7 days".
    """
    from aires import aindex as AI

    lead = np.asarray(d["realized"]["lead_days"], dtype=float)
    peak = float(d["config"]["horizon_days"])
    lo = peak - 6.0
    m = (lead >= lo - 1e-9) if AI._inclusive_left(ctx.ev.metric) else (lead > lo + 1e-9)
    return lead[m & (lead <= peak + 1e-9)]


def _ylab(ctx: R.Ctx) -> str:
    from aires import aindex as AI

    box = AI.box_for(ctx.event).name
    what = {"t2m_anom": "2 m temperature anomaly",
            "u850_speed": "850 hPa wind speed",
            "u850_speed_max": "850 hPa wind speed",
            "tp_total": "12 h precipitation",
            "tp_max12h": "12 h precipitation"}.get(ctx.ev.metric, ctx.ev.metric)
    unit = "K" if ctx.ev.metric == "t2m_anom" else (
        "m s$^{-1}$" if "speed" in ctx.ev.metric else "mm")
    return f"running index: {box}-box mean {what}, 24 h mean   ({unit})"


def _marginal(axm, d: dict, al: np.ndarray, obs) -> None:
    """Where every method's members ended up, on the main panel's y axis.

    The point of the whole pilot in one column of dots: the resampled population sits on
    top of the observed value, the direct ensembles stop below it. Drawn against the SAME
    y axis as the trajectories so no rescaling can flatter either side.
    """
    short = {"gencast_walkers": "GenCast\ndirect\n(walkers)",
             "gencast_xres": "GenCast\ndirect\n(xres)",
             "fcn3": "FCN3\ndirect"}
    ds = d.get("ds", {}) or {}
    cols = [(f"AI+RES\nresampled\nn={al.size}", al, RES_COLOR)]
    for key, (color, _label) in DS_STYLE.items():
        if key in ds:
            v = np.asarray(ds[key]["box"], dtype=float)
            cols.append((f"{short.get(key, key)}\nn={v.size}", v, color))

    rng = np.random.default_rng(0)
    for i, (label, v, color) in enumerate(cols):
        x = i + rng.uniform(-0.22, 0.22, size=v.size)
        axm.plot(x, v, "o", ms=3.4, color=color, alpha=0.8, mew=0)
        axm.plot([i - 0.34, i + 0.34], [v.mean()] * 2, "-", color=color, lw=1.8)
        if obs is not None:
            k = int(np.sum(v >= float(obs)))
            axm.text(i, 1.005, f"{k}/{v.size}", transform=axm.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=8.5, color=color,
                     fontweight="bold" if k else "normal")
    if obs is not None:
        axm.axhline(float(obs), color=OBS_COLOR, lw=1.3, ls="--")
    axm.set_xticks(range(len(cols)))
    axm.set_xticklabels([c[0] for c in cols], fontsize=7.4)
    axm.set_xlim(-0.6, len(cols) - 0.4)
    axm.tick_params(labelleft=False)
    axm.grid(alpha=0.2, axis="y")
    axm.set_title("reaching the\nobserved value", fontsize=9, pad=16)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default=os.environ.get("AIRES_EVENT", "PNW_HeatDome_2021"))
    ap.add_argument("--tag", default=A.RES_TAG)
    ap.add_argument("--only", default=",".join(FIGURES),
                    help=f"comma-separated subset of {FIGURES}")
    ap.add_argument("--top-frac", type=float, default=0.15,
                    help="composite: fraction of the population counted as 'extreme'")
    a = ap.parse_args(argv)

    ctx = R.Ctx(ev=R._event(a.event), tag=a.tag, n_walkers=A.RES_N_WALKERS,
                members=A.RES_M_MEMBERS, C=A.RES_C, leads=A.RES_LEAD_DAYS,
                horizon_days=A.RES_HORIZON_DAYS, dmc_seed=A.RES_DMC_SEED,
                base_seed=A.WALKER_BASE_SEED)
    d = load(ctx)
    ctx.n_walkers = int(d["config"]["n_walkers"])
    want = [x.strip() for x in a.only.split(",") if x.strip()]
    bad = [x for x in want if x not in FIGURES]
    if bad:
        raise SystemExit(f"unknown figure(s) {bad}; want a subset of {list(FIGURES)}")

    print(f"[aplots] {ctx.event} tag={ctx.tag}: {', '.join(want)}")
    if "trajectory" in want:
        print("  assembling the branch tree from the diagnostics cubes...")
        plot_trajectory(ctx, d)
    if "exceedance" in want:
        plot_exceedance(ctx, d)
    if "diagnostics" in want:
        plot_diagnostics(ctx, d)
    if "genealogy" in want:
        print("  computing clone divergence from the diagnostics cubes...")
        d["divergence"] = clone_divergence(ctx, d)
        plot_genealogy(ctx, d)
    if "composite" in want:
        plot_composite(ctx, d, top_frac=a.top_frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
