#!/usr/bin/env python
"""Phase 4 figures: what the AI+RES pilot found.

Four figures, covering the six items `aires.md` lists. Each reads only artifacts that
`aires/run_aires.py` already wrote (`compare.json`, `res_result.json`, `ds_baseline.json`)
plus, for the two that need fields rather than indices, the walker diagnostics cubes, plus
- for the trajectory figure alone - the cached CFSv2 baseline cube if `aires/cfs.py` has
built one. No GPU, no model, no network: the CFS cube is READ here and fetched there, and
a missing one drops that curve with a message rather than failing the figure.

    trajectory    the algorithm happening, in three panels on ONE shared y axis:
                  the observable against lead time (every branch the pilot rolled,
                  the barriers, the 7-day window, the operational CFSv2 forecast and
                  the ERA5 reanalysis itself), then the final-``A_L`` density, then
                  where each method's members landed         [after AI+RES fig. S9]
    exceedance    the headline: does the RES tail contain the observed event when direct
                  sampling does not?                                     [aires.md fig 1]
    diagnostics   score skill per t_k, and whether N=64 held up          [figs 3, 4]
    genealogy     walker ancestry, and whether clones actually separate  [figs 2, 6]
    composite     what the RES-selected extremes look like               [fig 5]
    walkers       the population as a list: each walker's realized anomaly next to the
                  probability mass it carries

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

Reading the trajectory figure
-----------------------------
Three panels, one y axis, drawn after Lancelin et al.'s figure S9. The middle panel is
the MARGINAL of the left one: it is the same 64 numbers the trajectory lines end at and
the same numbers the right panel scatters, so a bar, a dot and a line-end at the same
height are the same walker. The y limits are therefore computed ONCE over everything
drawn anywhere - walkers, killed branches, CFSv2 members, every scatter population,
every kernel's support, the observation and the ERA5 truth - because a panel that
autoscaled on its own contents would break exactly the correspondence the layout is for.

The left panel also carries the **ERA5 reanalysis** as a single black line with a marker
every 6 h: what actually happened, at the sampling the reanalysis has, against 64
forecasts of it. It comes from `aires/aera5.py` and, like the CFSv2 baseline, a missing
cache costs the line and not the figure.

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

FIGURES = ("trajectory", "exceedance", "diagnostics", "genealogy", "composite", "walkers")

RES_COLOR = "#1f77b4"
DS_STYLE = {
    "gencast_walkers": ("#2ca02c", "GenCast direct (Gate 3 walkers)"),
    "gencast_xres":    ("#8c564b", "GenCast direct (xres 24-member)"),
    "fcn3":            ("#9467bd", "FCN3 direct (24-member)"),
}
OBS_COLOR = "#d62728"
# The CFSv2 operational baseline. Black, dashed, deliberately outside the palette every
# other curve is drawn from: it is the one line on the figure that is not a machine
# learning model, and it has to stay legible against 64 survivors coloured on a saturated
# ramp, ~450 grey killed branches and a red observed line.
CFS_COLOR = "#111111"
# The density panel follows figure S9's two-tone convention rather than this module's
# per-method palette: the AI+RES population in the run colour, the direct-sampling
# ensemble in a neutral mid grey. It is a mid grey (0.50) on purpose - `DEAD_COLOR` is
# 0.72 and lives on a different panel, and the two never share an axis.
DENS_DS_COLOR = "#808080"
# The weighted curve's own blue: `RES_COLOR` at half luminance. It is drawn ON TOP of the
# translucent `RES_COLOR` bars, and mid-blue dashes over a mid-blue fill read as a broken
# WHITE line - the casing showing through the gaps is the only thing with any contrast
# left. A navy at the same hue keeps the curve in the AI+RES family, reads against both
# the fill and the white ground, and can never be mistaken for the solid mid-blue
# unweighted curve: they differ in lightness AND in dash state, not in one of the two.
RES_WEIGHTED_COLOR = "#0f3b5a"
# How much taller than the tallest histogram bar the weighted curve's peak may be before
# the panel stops trying to show all of it. See `_density_panel`.
DENS_WEIGHTED_HEADROOM = 1.6


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


def _save(fig, path: Path, pdf: bool = False) -> Path:
    """Write the PNG, and for the publication figures a vector PDF of the same stem.

    140 dpi is a screen artifact: legible in a browser, and the moment a trajectory
    panel with 450 grey branches is put through a typesetter every one of them turns to
    mush. The PDF is the SAME figure object at the same bbox - one layout, two
    containers - so the two can never drift apart.
    """
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
    if pdf:
        vec = path.with_suffix(".pdf")
        fig.savefig(vec, bbox_inches="tight")
        print(f"  wrote {vec} ({vec.stat().st_size / 1e3:.0f} kB)")
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


def walker_probabilities(weights, log_Z: float, n: int | None = None) -> np.ndarray:
    """The probability mass ``p_i`` the run assigns to each surviving walker.

    A resampled walker is not one member of an equally weighted ensemble. It is a
    trajectory the scheme deliberately over-produced, and the weight ``w_i = e^{-V_K}``
    is what undoes that tilt. The estimator every exceedance number in this experiment
    comes from is

        P(A_L >= a)  =  Z * mean_i( 1{A_L_i >= a} * w_i )   with   Z = e^{log Z}

    so the per-walker term of that sum,

        p_i  =  Z * w_i / N,

    IS the probability GenCast assigns to that walker's outcome: sum the ``p_i`` of the
    walkers past a threshold and the exceedance probability falls out. Summing ALL of them
    gives ``normalization_check``, which should be 1 and is not (open problem 5), so
    ``p_i`` is an estimate on the same footing as the curve it builds - not a quantity
    with an exact interval.

    ``weights`` are the raw ``e^{-V_K}`` values ``compare.json`` stores, and ``n`` defaults
    to their count.
    """
    w = np.asarray(weights, dtype="float64")
    n = int(w.size if n is None else n)
    return float(np.exp(log_Z)) * w / n


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
def metric_unit(metric: str) -> str:
    """The metric's own unit, or ``""`` where it has none. One place, not five inline
    copies of the same conditional - and the thing to extend when a metric is added."""
    return "K" if metric == "t2m_anom" else (
        "m s$^{-1}$" if "speed" in metric else ("mm" if metric.startswith("tp") else ""))


def unit_suffix(ctx) -> str:
    """``" K"`` / ``""`` - the unit as it is appended to a formatted number."""
    u = metric_unit(ctx.ev.metric)
    return f" {u}" if u else ""


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


# --------------------------------------------------------------------------- #
# The final-value density panel (figure S9's middle column)
# --------------------------------------------------------------------------- #
def density_bins(x, h: float) -> np.ndarray:
    """Bin edges for one population of the density panel, by a stated rule.

    The paper had 50,000 direct-sampling members in 30 bins and 400 AI+RES walkers in
    10. We have 64 and 24. At that size the bin count is not a cosmetic choice: too few
    and the panel is three bars that say nothing the scatter column does not; too many
    and it draws Poisson noise as structure. So the count is fenced from both sides by
    quantities the same panel already commits to, and nothing here is a free number:

    - **the target** is Freedman-Diaconis, ``w = 2 IQR n^(-1/3)`` - the usual
      distribution-free width, which adapts to a population whose spread differs by an
      order of magnitude between events (0.30 K on p90_20240802, 4.0 K on Elliott);
    - **the floor** is the square-root rule, ``round(sqrt(n))`` bins - 8 for the 64
      walkers, 5 for a 24-member ensemble - so a population is never drawn coarser than
      that however tight its interquartile range happens to be;
    - **the ceiling** is ``range / h``, i.e. no bin narrower than the KDE bandwidth
      overlaid on the very same axes. ``h`` is Scott's rule on the sample
      (`aires.akde.scott_bandwidth`); a bar finer than the kernel would assert a
      resolution the curve beside it explicitly denies.

    Degenerate samples (fewer than 2 points, or zero range) get a single unit-wide bin
    rather than an exception - the persistence control and Southwest are near-degenerate
    and the figure has to render for them too.
    """
    x = np.asarray(x, dtype="float64").ravel()
    x = x[np.isfinite(x)]
    lo, hi = (float(x.min()), float(x.max())) if x.size else (0.0, 0.0)
    span = hi - lo
    if x.size < 2 or not span > 0:
        return np.array([lo - 0.5, lo + 0.5])
    q1, q3 = np.percentile(x, [25.0, 75.0])
    w_fd = 2.0 * float(q3 - q1) * x.size ** (-1.0 / 3.0)
    k_lo = max(2, int(round(np.sqrt(x.size))))
    k_hi = max(k_lo, int(np.floor(span / h)) if (np.isfinite(h) and h > 0) else k_lo)
    k_fd = int(np.ceil(span / w_fd)) if w_fd > 0 else k_lo
    return np.linspace(lo, hi, int(np.clip(k_fd, k_lo, k_hi)) + 1)


def _density(ctx: R.Ctx) -> dict | None:
    """The three curves of the density panel, or ``None`` if the run is not reduced.

    Everything comes from `aires.akde`, which already encodes the two decisions that
    make a 64-point KDE honest and which must not be re-implemented here: both walker
    curves share ONE bandwidth (Scott's rule on the unweighted positions, because the
    weights reweight the sample and do not enlarge it), and every curve is clipped to
    ``[min, max]`` of its own members, because a Gaussian tail past the observation is
    kernel shape and would draw a reach the experiment denies.

    Degrades the way ``_cfs`` does. ``SystemExit`` is caught by name for the same reason
    it is there: it is how this repo reports "unknown event" / "not reduced" and it does
    not inherit from ``Exception``.
    """
    try:
        from aires import akde as KDE
        from aires import awalkers as WK

        r = WK.run_record(ctx.event, ctx.tag)
        curves = {c["key"]: c for c in KDE.panel_curves(r)}
    except (Exception, SystemExit) as e:                          # noqa: BLE001
        print(f"    final-value density unavailable ({type(e).__name__}: {e}); drawing "
              f"the trajectory figure without that panel")
        return None
    if "aires_unweighted" not in curves:
        return None
    print(f"    density panel: {r['n']} walkers (h = {curves['aires_unweighted']['h']:.2f})"
          + (f", {r['direct'].size} {r['direct_key']} members "
             f"(h = {curves['direct']['h']:.2f})" if "direct" in curves else
             ", no direct ensemble"))
    return dict(r=r, curves=curves)


def _density_panel(axd, dens: dict, obs, unit: str) -> None:
    """Figure S9's marginal: horizontal bars plus KDE, on the trajectories' own y axis.

    The variable is on **y** and density grows rightward from the left spine, so the
    panel is literally the left panel's final column turned into a distribution. Only
    the left spine is drawn and the density axis carries no ticks, no numbers and no
    label: its scale is set by the requirement that the tallest kernel just fits, and a
    reader who measures a bar against it is measuring the wrong thing. What the panel
    says is *shape*, and where the shape sits relative to the red observed line.

    Three populations, and the pairing is the whole argument:

    - **blue bars + blue solid KDE** - the AI+RES walkers UNWEIGHTED. This is the paper's
      blue: the tilted population that resampling actually manufactured. It is what the
      trajectory lines end at and what the scatter panel's first column shows, which is
      what lets the three panels be read across.
    - **grey bars + grey solid KDE** - the direct-sampling ensemble (the paper's DS),
      chosen by ``awalkers.DS_PREFERENCE``. The like-for-like baseline.
    - **navy DASHED KDE, no bars** - ours, not the paper's: the walkers WEIGHTED by
      ``p_i = Z w_i / N``, the estimator's reconstruction of the original forecast law.
      Its mass past the red line is the headline ``P``. It gets no bars because a
      histogram of 64 points whose weights span four decades is a lie about counting -
      the weighted object is a density and only a density. It is the ONLY dashed line on
      the panel, and both other curves are solid exactly as in S9, so "dashed" reads as
      "weighted" and nothing else: the two blue curves are the same 64 walkers and the
      dash is the whole difference between where they are and what they are worth.

    Bin edges are chosen per population from its own range (``density_bins``) and
    deliberately do not align: two grids forced to share edges would invite reading one
    bar against another, which at n = 24 and n = 64 is noise. The density scale is
    LINEAR, unlike `aires.akde`'s own log panel - S9 is linear, and a log axis would
    stop this reading as a marginal of the trajectories beside it.

    Two ways a curve can stop, and they must not look alike
    ------------------------------------------------------
    **A dot is where the sample ran out.** Every curve is drawn only inside
    ``[min, max]`` of its own members (`aires.akde`'s support rule), so all three end in
    mid-air. On the two S9 curves that is invisible - they end near the spine, where a
    kernel has almost no mass left. The WEIGHTED curve does not: its weights concentrate
    on the least extreme walkers, which is the whole point of it, so it ends at the
    least-extreme edge of the support at a density of 0.25 (PNW pilot, 64% of the panel
    width) to 0.75 (the persistence control, 91%). A curve that just stops out there
    reads as a rendering failure. So every curve gets a filled dot at each end: the mark
    says "this is the most extreme member there is", not "the drawing was cut".

    **A caret at the spine is where the drawing WAS cut.** The weighted PDF concentrates,
    so its peak can dwarf the histograms', and a limit that always covered it would
    flatten the bars into invisibility on exactly the runs where the estimator is
    degenerate. The limit is therefore chosen per run:

    - if the weighted peak is within ``DENS_WEIGHTED_HEADROOM`` (1.6x) of the tallest
      bar, the limit covers EVERYTHING drawn - simplest and most honest, and it is what
      fires on all ten runs of the current slate (the largest ratio is 1.52, the
      persistence control);
    - otherwise the limit is set from the bars and the S9 curves alone, the weighted
      curve is clipped at the spine, a caret is drawn where it leaves, and its true peak
      density is printed in its own legend entry. The number is then ON the figure
      rather than lost with the ink.

    The weighted curve is never rescaled to a common peak height. Two densities on one
    axis are comparable only while they share that axis; normalising one of them to fit
    would make the panel's central comparison meaningless.
    """
    import matplotlib.patheffects as pe
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    r, curves = dens["r"], dens["curves"]
    peak_s9 = [0.0]        # bars + the two S9 curves: what has to stay legible
    peak_wtd = [0.0]       # the weighted curve alone: it is allowed to be much taller
    handles = []

    def bars(c, color, alpha, elw, z):
        # The alpha goes on the FACE only, as an RGBA, never on the artist: a patch-level
        # alpha fades the black edge with the fill, and S9's bars are outlined in solid
        # black precisely so two overlapping histograms stay separable where they cross.
        e = density_bins(c["x"], c["h"])
        dy, _ = np.histogram(np.asarray(c["x"], dtype="float64"), bins=e, density=True)
        axd.barh(0.5 * (e[:-1] + e[1:]), dy, height=np.diff(e), align="center",
                 color=to_rgba(color, alpha), edgecolor="black", linewidth=elw, zorder=z)
        peak_s9.append(float(np.max(dy)) if dy.size else 0.0)
        return e.size - 1

    def kde_line(c, color, ls, lw, z, into, halo=False):
        st = dict(color=color, ls=ls, lw=lw, zorder=z, solid_capstyle="round")
        if halo:
            st["path_effects"] = [pe.Stroke(linewidth=lw + 1.7, foreground="white"),
                                  pe.Normal()]
        axd.plot(c["density"], c["grid"], **st)
        d, g = c["density"], c["grid"]
        m = np.flatnonzero(np.isfinite(d))
        if m.size:
            # The two ends of the curve's own support, marked - see the docstring: a dot
            # is the sample running out, and it must not look like the axis cutting it.
            axd.plot(d[m[[0, -1]]], g[m[[0, -1]]], ls="none", marker="o", ms=3.2,
                     mfc=color, mec="white", mew=0.7, zorder=z + 0.1)
            into.append(float(np.max(d[m])))

    # --- draw order is S9's: grey bars, blue bars, grey KDE, blue KDE ------------- #
    direct = curves.get("direct")
    if direct is not None:
        nb = bars(direct, DENS_DS_COLOR, 0.60, 0.55, 2)
        handles.append(Patch(fc=to_rgba(DENS_DS_COLOR, 0.60), ec="black", lw=0.55,
                             label=f"{_ds_short(r['direct_key'])},\n"
                                   f"n={direct['x'].size} ({nb} bins)"))
    unw = curves["aires_unweighted"]
    nb = bars(unw, RES_COLOR, 0.70, 0.75, 3)
    handles.append(Patch(fc=to_rgba(RES_COLOR, 0.70), ec="black", lw=0.75,
                         label=f"AI+RES, unweighted,\nn={unw['x'].size} ({nb} bins)"))
    if direct is not None:
        kde_line(direct, DENS_DS_COLOR, "-", 1.5, 4, peak_s9)
    kde_line(unw, RES_COLOR, "-", 1.5, 5, peak_s9)
    wtd = curves.get("aires_weighted")
    if wtd is not None:
        kde_line(wtd, RES_WEIGHTED_COLOR, (0, (4.5, 1.8)), 1.9, 6, peak_wtd, halo=True)
    if obs is not None:
        axd.axhline(float(obs), color=OBS_COLOR, lw=1.3, ls="--", zorder=7)

    # --- the frame: left spine only, no density ticks, no density label ----------- #
    for side in ("top", "right", "bottom"):
        axd.spines[side].set_visible(False)
    axd.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
    axd.tick_params(axis="y", direction="in", left=True, right=False, labelleft=False)

    # --- how far right the panel reaches, and what happens if a curve is cut ------ #
    s9_hi, wtd_hi = max(peak_s9), max(peak_wtd)
    truncated = wtd_hi > DENS_WEIGHTED_HEADROOM * max(s9_hi, 1e-12)
    axd.set_xlim(0.0, 1.10 * max(s9_hi if truncated else max(s9_hi, wtd_hi), 1e-9))
    if wtd is not None:
        lab = "AI+RES weighted by $p_i$,\nthe estimator's PDF"
        if truncated:
            # It leaves the panel: say where, and say how tall it really is. Without both
            # marks the curve simply vanishes at the spine and the reader cannot tell
            # that from a curve whose data ended there.
            d, g = wtd["density"], wtd["grid"]
            j = int(np.nanargmax(np.where(np.isfinite(d), d, -np.inf)))
            axd.plot([axd.get_xlim()[1]], [g[j]], ls="none", marker=">", ms=7.5,
                     color=RES_WEIGHTED_COLOR, mec="white", mew=0.8, clip_on=False,
                     zorder=9)
            lab += f"\n(peak {wtd_hi:.2f} off-panel $\\rightarrow$)"
        handles.append(Line2D([], [], color=RES_WEIGHTED_COLOR, ls=(0, (4.5, 1.8)),
                              lw=1.9, label=lab))
    axd.set_title("final $A_L$ distribution\n"
                  + (f"(density per {unit})" if unit else "(density)"),
                  fontsize=9, pad=16)

    # --- the legend goes in whichever end of the panel has no ink ----------------- #
    # S9 puts it lower right, which is right for a heat event whose population piles up
    # at the top of a shared axis. On a cold event the same population sits at the
    # BOTTOM of that axis and a fixed corner would land on the bars, so the corner is
    # chosen from where the density actually is. The comparison is over the outer 30% of
    # the axis at each end - roughly the height the legend itself occupies - and NOT
    # over the two halves: a half-and-half split is decided by whatever happens to fall
    # on the near side of the midpoint, which is nowhere near either corner.
    lo, hi = axd.get_ylim()
    span = hi - lo
    top_ink = bot_ink = 0.0
    for c in curves.values():
        d, g = c["density"], c["grid"]
        m = np.isfinite(d)
        if not m.any():
            continue
        up, dn = m & (g > lo + 0.70 * span), m & (g < lo + 0.30 * span)
        top_ink = max(top_ink, float(d[up].max()) if up.any() else 0.0)
        bot_ink = max(bot_ink, float(d[dn].max()) if dn.any() else 0.0)
    axd.legend(handles=handles, loc="lower right" if top_ink >= bot_ink else "upper right",
               fontsize=7.0, frameon=False, handlelength=1.6, handleheight=1.15,
               labelspacing=0.9, borderaxespad=0.3)


def _ds_short(key) -> str:
    """The direct ensemble's name as the scatter panel writes it, on one line."""
    return {"gencast_xres": "GenCast direct (xres)",
            "gencast_walkers": "GenCast direct (walkers)",
            "fcn3": "FCN3 direct"}.get(key, str(key))


# --------------------------------------------------------------------------- #
# The ERA5 reanalysis trajectory
# --------------------------------------------------------------------------- #
def _era5_truth(ctx: R.Ctx) -> dict | None:
    """The observed running index for this event, or ``None`` if it is not cached.

    `aires/aera5.py` builds it; this only reads. Degrades exactly like ``_cfs``: the
    figure stage is offline and must not acquire a data dependency it can fail on, and
    losing the truth line is not worth losing the figure over. The import itself is
    guarded because the module is newer than this one and a checkout may not have it.
    """
    try:
        from aires import aera5

        truth = aera5.load(ctx.event)
    except (Exception, SystemExit) as e:                          # noqa: BLE001
        print(f"    ERA5 truth trajectory unavailable ({type(e).__name__}: {e}); "
              f"drawing without it")
        return None
    if truth is None:
        print(f"    no ERA5 running index cached for {ctx.event} - drawing without the "
              f"reanalysis trajectory.\n"
              f"      build it on the login node: PYTHONPATH=. python -m aires.aera5 "
              f"--event {ctx.event}")
        return None
    try:
        lead = np.asarray(truth["lead"], dtype=float)
        box = np.asarray(truth["box"], dtype=float)
    except (Exception, KeyError) as e:                            # noqa: BLE001
        print(f"    ERA5 truth cache is not shaped as expected ({e}); skipping")
        return None
    if lead.size != box.size or not lead.size:
        print(f"    ERA5 truth cache is empty or ragged ({lead.size} vs {box.size}); "
              f"skipping")
        return None
    step = float(np.median(np.diff(lead))) * 24.0 if lead.size > 1 else float("nan")
    a_l = truth.get("a_l") if isinstance(truth, dict) else None
    print(f"    ERA5 truth: {box.size} frames, {lead.min():.2f}-{lead.max():.2f} d, "
          f"{step:.0f} h sampling"
          + (f", $A_L$ = {float(a_l):+.2f}" if a_l is not None
             and np.isfinite(float(a_l)) else ""))
    return dict(lead=lead, box=box, a_l=a_l)


def _shared_ylim(parts, pad: float = 0.035) -> tuple[float, float]:
    """One y range over EVERYTHING the three panels draw, computed before any of them.

    The panels share a y axis, so whichever one autoscales sets it for all three - and
    the left panel, which is the one matplotlib would autoscale from, is the one whose
    range is largest. Left to itself it would happily clip a direct-sampling member or a
    kernel's support off the bottom of the two panels on its right, and the figure's
    whole claim is that a bar, a dot and a line-end at the same height are the same
    number. So the limits are taken over the union up front.
    """
    vals = [np.asarray(a, dtype=float).ravel() for a in parts if a is not None]
    v = np.concatenate([a for a in vals if a.size]) if vals else np.zeros(0)
    v = v[np.isfinite(v)]
    if not v.size:
        return (-1.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    m = pad * max(hi - lo, 1e-6)
    return lo - m, hi + m


def plot_trajectory(ctx: R.Ctx, d: dict, tree: dict | None = None) -> Path:
    """The walk itself, in three panels on one shared y axis, after AI+RES fig. S9.

        [ trajectories | final-$A_L$ density | where each method landed ] + colorbar

    Left: the observable against lead time - every branch the pilot rolled, the killed
    ones in grey with an x where they end, the resampling barriers with what each one
    killed, the 7-day window ``A_L`` is the mean over, the operational CFSv2 cycles, and
    the ERA5 reanalysis as a black line marked every 6 h. Middle: the same 64 endpoints
    as a probability density, binned and smoothed, against the direct-sampling ensemble
    (`_density_panel`). Right: every method's members as a strip of dots with the count
    that reached the observation (`_marginal`).

    The three panels are ONE picture and the code keeps them that way in three places:
    the y limits are computed once over the union of everything drawn anywhere
    (`_shared_ylim`), the observed value's red dashed line runs across all three, and
    only the left panel carries y numbers. A walker is then the same height in all
    three - a line-end, a bar and a dot - which is the correspondence the layout exists
    to make readable.

    Both the CFSv2 baseline and the ERA5 truth line are optional caches: a missing one
    prints why and drops that curve. The density panel degrades the same way.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    cfg = d["config"]
    legs = A.res_legs(cfg["leads"], cfg["horizon_days"], cfg["C"])
    n = int(cfg["n_walkers"])
    sign = float(cfg.get("tail_sign", ctx.sign))
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

    cfs = _cfs(ctx)
    truth = _era5_truth(ctx)
    dens = _density(ctx)

    # The marginal holds one column per method and gains one when CFS is present; a fixed
    # width ratio would overlap its tick labels rather than shrink the dots. Count the
    # columns `_marginal` ACTUALLY draws: `d["ds"]` also carries `event`, `metric` and
    # `tail_sign`, so its raw length is 8 where 5 columns are drawn, and the panel was
    # being handed 60% more width than it used.
    ds_keys = [k for k in DS_STYLE if k in (d.get("ds", {}) or {})]
    n_cols = 1 + len(ds_keys) + (1 if cfs else 0)
    # S9's proportions: a wide trajectory panel, a narrow density panel beside it, frames
    # top- and bottom-aligned, one tight gap. The scatter strip and the colorbar are ours.
    ratios = [4.6] + ([1.30] if dens else []) + [0.30 * n_cols, 0.05]
    fig = plt.figure(figsize=(15.5 + (2.9 if dens else 0.0), 7.4))
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.075)
    ax = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[0, 1], sharey=ax) if dens else None
    axm = fig.add_subplot(gs[0, -2], sharey=ax)
    cax = fig.add_subplot(gs[0, -1])

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
    # A translucent white plate under it, like every other in-panel label here: the
    # annotation sits at the top of the window and the ERA5 truth line now runs through
    # that corner on the events where the reanalysis is the most extreme thing on the
    # figure (Southwest 2020, where no walker reaches it), which is exactly the event
    # whose caption a reader most needs to be able to read.
    ax.text(tb(0.5 * (w0 + w1)), 0.985,
            f"$A_L$ = the 7-day mean\nover these {win.size} frames",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5,
            color="#8a5a12", linespacing=1.35, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.65))

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
    # Built before they are drawn, because `_shared_ylim` needs the SMOOTHED values: the
    # 24 h filter's end padding can overshoot the raw series by a fraction of a kelvin,
    # and a limit taken from the raw one would clip the very last point of a lineage -
    # the one sitting on the event peak, where A_L is taken.
    surv = [_from_root(root, lead, series[i]) for i in np.argsort(al)]

    ax.set_ylim(*_shared_ylim(
        [y for _x, y in surv]                                   # survivors, as drawn
        + [np.asarray(y, dtype=float) for _x, y in tree["dead"]]  # killed branches
        + [al]                                                   # the walkers' own A_L
        + [np.asarray(v["box"], dtype=float)                     # scatter populations
           for k, v in (d.get("ds", {}) or {}).items() if k in DS_STYLE]
        + ([np.asarray(cfs["series"], dtype=float), np.asarray(cfs["box"], dtype=float),
            np.array([float(cfs["box_mean"])])] if cfs is not None else [])
        + ([truth["box"]] if truth is not None else [])
        + ([c["x"] for c in dens["curves"].values()] if dens else [])  # kernel supports
        + ([np.array([float(obs)])] if obs is not None else [])))

    for (xs, ys), i in zip(surv, np.argsort(al)):       # extremes drawn last, on top
        ax.plot(tb(xs), ys, "-", lw=1.0, color=cmap(norm(al[i])), alpha=0.9,
                zorder=3, solid_capstyle="round")

    # --- the operational baseline -------------------------------------------------- #
    # Drawn ABOVE the survivors, not below: four black curves under 64 saturated ones are
    # invisible, and the whole point of the baseline is that it can be read against them.
    if cfs is not None:
        for row in cfs["series"]:
            ax.plot(tb(cfs["lead"]), row, "--", lw=1.05, color=CFS_COLOR, alpha=0.85,
                    zorder=4.6, dashes=(5, 2.6))
        # A_L is a MEAN over the shaded window, not the endpoint of a curve, so it is
        # drawn as a bar spanning exactly that window at exactly that height. Putting the
        # number anywhere else on this panel would invite reading a curve's last point as
        # A_L, which it is not.
        m = float(cfs["box_mean"])
        ax.plot([tb(w0), tb(w1)], [m, m], "-", lw=2.6, color=CFS_COLOR, zorder=4.8,
                solid_capstyle="butt")
        # Above the ERA5 truth line (zorder 5.5), not below it: on Uri the reanalysis
        # crosses its own CFSv2 bar within a day of this label, and a 2.4 pt black line
        # through the middle of a number is worse than a line with a gap in it.
        ax.text(tb(w0), m, f"CFSv2 $A_L$ = {m:+.2f}{unit_suffix(ctx)}  ", color=CFS_COLOR,
                fontsize=8.5, ha="right", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.72))

    # --- what actually happened ---------------------------------------------------- #
    # The ERA5 reanalysis, on the same running index the walkers are drawn on. One line,
    # solid black, ABOVE everything else: it is the only curve on the panel that is not a
    # forecast, and the figure is asking whether any of the forecasts got near it. A
    # marker on every point rather than a smooth line, because the reanalysis is
    # 6-hourly where the walkers are 12-hourly - the markers are the sampling, and
    # hiding that difference would let the truth read as just another member.
    if truth is not None:
        ax.plot(tb(truth["lead"]), truth["box"], "-", color="black", lw=2.4, marker="o",
                ms=3.4, mfc="white", mec="black", mew=0.9, zorder=5.5,
                solid_capstyle="round")

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
        # Placed at a fixed AXES fraction, clear of the legend box, rather than at a
        # lead. The legend is anchored upper-left and its height grows with the number of
        # methods drawn - adding the CFSv2 entry made it a row taller - so on an event
        # whose observed A_L sits high on the axis (Southwest 2020, +5.14 K) a label
        # pinned to the left edge lands underneath it. An axes fraction cannot collide
        # with a legend that never leaves the left third, and the white bbox keeps the
        # label readable wherever on the y axis the observed value falls.
        ax.text(0.40, float(obs),
                f" ERA5 observed $A_L$ = {float(obs):+.2f}{unit_suffix(ctx)}",
                transform=ax.get_yaxis_transform(), color=OBS_COLOR, fontsize=8.5,
                ha="left", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.72))

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
    handles = [
        Line2D([], [], color=cmap(0.85), lw=1.4, label="surviving lineage (colored by its $A_L$)"),
        Line2D([], [], color=DEAD_COLOR, lw=1.0,
               label=f"killed branch ({tree['n_dead_nodes']} segments, x where it ends)"),
        Line2D([], [], color="#175d7d", marker="D", ls="none", ms=5,
               label=r"population-mean score $\theta$ at that barrier"),
    ]
    if cfs is not None:
        lo, hi = min(cfs["member_lead_days"]), max(cfs["member_lead_days"])
        handles.append(Line2D(
            [], [], color=CFS_COLOR, lw=1.05, ls="--",
            label=f"CFSv2 operational, {cfs['n_members']} lagged cycles "
                  f"({lo:g}-{hi:g} d lead); bar = its $A_L$"))
    if truth is not None:
        handles.append(Line2D(
            [], [], color="black", lw=2.4, marker="o", ms=3.4, mfc="white",
            mec="black", mew=0.9,
            label="ERA5 reanalysis truth (a marker every 6 h)"))
    # Above everything, the ERA5 truth line included. There is no corner of this panel
    # that is empty on all ten runs - on Uri the population starts at +5 K and the legend
    # sits on top of it whatever alpha it carries - so the choice is between a legend the
    # curves are drawn through and a legend that hides some of them. A legend with a
    # trajectory scribbled across its text is not a legend.
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              framealpha=0.92).set_zorder(20)
    # S9's frame: a closed box with ticks on all four sides pointing inward, so the panel
    # reads as one field rather than as two open axes with curves in them.
    ax.tick_params(direction="in", top=True, right=True)

    # --- the marginal of the trajectories, then the strip ------------------------- #
    if axd is not None:
        _density_panel(axd, dens, obs, metric_unit(ctx.ev.metric))
    _marginal(axm, d, al, obs, sign, cfs)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label=r"realized $A_L$" + (f"   ({metric_unit(ctx.ev.metric)})"
                                            if metric_unit(ctx.ev.metric) else ""))
    fig.suptitle(f"AI+RES pilot - {ctx.event} ({ctx.tag}): the walk itself, barrier by "
                 f"barrier", fontsize=12)
    fig.subplots_adjust(left=0.048, right=0.952, top=0.865, bottom=0.115)
    return _save(fig, _fig_path(ctx, "trajectory"), pdf=True)


def _cfs(ctx: R.Ctx) -> dict | None:
    """The CFSv2 baseline for this event, or ``None`` if it has not been built.

    Returns ``None`` rather than raising, for the same reason ``_init_index_series``
    does: this figure is the offline stage and must not acquire a network dependency.
    ``aires/cfs.py`` is what fetches; this only reads what is already cached, says so
    when there is nothing, and draws the rest of the figure unchanged.
    """
    try:
        from aires import cfs as CFS
        cube = CFS.load(ctx.event)
    except (Exception, SystemExit) as e:              # noqa: BLE001 - cfgrib is optional
        # Two ways this legitimately fails and neither may cost the figure: cfgrib is not
        # installed (it is a pip add-on, not part of either conda env's definition), and
        # the event is not in fevents.KNOWN, which is how the unit tests drive this code
        # with synthetic populations.
        #
        # SystemExit is named explicitly because it does NOT inherit from Exception - and
        # SystemExit is precisely how this repo reports "unknown event" and "no cube", so
        # a bare `except Exception` here catches the rare case and misses both common ones.
        print(f"    CFS baseline unavailable ({type(e).__name__}: {e}); skipping")
        return None
    if cube is None:
        print(f"    no CFSv2 cube cached for {ctx.event} - drawing without the "
              f"operational baseline.\n"
              f"      build it on the login node: PYTHONPATH=. python -m aires.cfs "
              f"--event {ctx.event}")
        return None
    try:
        lead, box = CFS.series(ctx.event, cube)
        out = dict(CFS.indices(ctx.event, cube))
    except (Exception, SystemExit) as e:              # noqa: BLE001 - see above
        print(f"    CFS baseline could not be reduced ({type(e).__name__}: {e}); skipping")
        return None
    finally:
        cube.close()
    # CFS is 6-hourly where the walkers are 12-hourly, so the diurnal filter is the 5-point
    # trapezoid rather than aplots.daily's 3-point one - same 24 h null, same centring.
    out["lead"], out["series"] = CFS.daily_mean(lead, box)
    print(f"    CFSv2 baseline: {out['n_members']} cycles, mean $A_L$ = "
          f"{out['box_mean']:+.2f}")
    return out


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
    u = metric_unit(ctx.ev.metric)
    return (f"running index: {box}-box mean {what}, 24 h mean"
            + (f"   ({u})" if u else ""))


def _marginal(axm, d: dict, al: np.ndarray, obs, sign: float = 1.0,
              cfs: dict | None = None) -> None:
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
    if cfs is not None:
        cols.append((f"CFSv2\noperational\nn={len(cfs['box'])}",
                     np.asarray(cfs["box"], dtype=float), CFS_COLOR))

    rng = np.random.default_rng(0)
    for i, (label, v, color) in enumerate(cols):
        x = i + rng.uniform(-0.22, 0.22, size=v.size)
        axm.plot(x, v, "o", ms=3.4, color=color, alpha=0.8, mew=0)
        axm.plot([i - 0.34, i + 0.34], [v.mean()] * 2, "-", color=color, lw=1.8)
        if obs is not None:
            # Sign-aware, exactly as the exceedance panel counts it (see `exceedance`):
            # "reaching" is reaching the EXTREME tail, so on a cold event this counts the
            # members COLDER than the observed freeze, not the ones warmer than it.
            k = int(np.sum(sign * v >= sign * float(obs)))
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
# Figure 7 - the population as a list: what each walker reached, and how likely it is
# --------------------------------------------------------------------------- #
def plot_walkers(ctx: R.Ctx, d: dict) -> Path:
    """One row per walker: the anomaly it reached, and the probability mass it carries.

    The exceedance figure shows the INTEGRAL of this - a curve with the walkers summed
    away. This one keeps them apart, because the two numbers a reader wants per walker are
    "how extreme did it get" and "how much does GenCast believe it", and the second is the
    one a raw ensemble plot cannot show: in a direct ensemble every member carries the same
    1/n, whereas here the masses span three orders of magnitude and the most extreme walker
    is usually NOT the most probable one.

    Rows are ordered by how far into the tail the walker got, most extreme at the top, with
    the tail sign applied - so on a freeze the top row is the coldest walker.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aires import aindex as AI

    sign = float(d["config"].get("tail_sign", 1.0))
    al = np.array(d["realized"]["box"], dtype=float)
    w = np.array(d["weights"], dtype=float)
    p = walker_probabilities(w, d["log_Z"], al.size)
    obs = d.get("observed")
    unit = unit_suffix(ctx)
    ds = d.get("ds", {}) or {}

    order = np.argsort(sign * al)[::-1]          # most extreme first
    al_s, p_s = al[order], p[order]
    y = np.arange(al.size)
    reach = (np.full(al.size, False) if obs is None
             else sign * al_s >= sign * float(obs))

    fig, ax = plt.subplots(1, 2, figsize=(13.6, max(6.4, 0.135 * al.size + 3.0)),
                           sharey=True, gridspec_kw={"width_ratios": [1.35, 1]})

    # --- (0) what each walker reached --------------------------------------- #
    a0 = ax[0]
    a0.barh(y, al_s, height=0.72, color=np.where(reach, OBS_COLOR, RES_COLOR),
            alpha=0.85, edgecolor="none", zorder=3)
    if obs is not None:
        a0.axvline(float(obs), color=OBS_COLOR, lw=1.4, ls="--", zorder=4)
        a0.text(float(obs), -1.4, " ERA5 observed ", color=OBS_COLOR, fontsize=8.5,
                ha="left" if sign > 0 else "right", va="bottom")
    a0.axvline(0.0, color="0.35", lw=0.8, zorder=2)
    a0.set_xlabel(rf"realized $A_L$   ({unit})" if unit else r"realized $A_L$")
    a0.set_ylabel("walker, ranked by how far into the tail it got")
    n_reach = int(reach.sum())
    a0.set_title(f"what each of the {al.size} walkers reached"
                 + (f"   -   {n_reach}/{al.size} reach the observed value"
                    if obs is not None else ""), fontsize=10)
    a0.grid(alpha=0.25, axis="x")
    if sign < 0:
        a0.invert_xaxis()

    # --- (1) the probability mass it carries -------------------------------- #
    a1 = ax[1]
    floor = max(p_s[p_s > 0].min() / 3.0, 1e-12) if (p_s > 0).any() else 1e-12
    a1.hlines(y, floor, np.maximum(p_s, floor), lw=1.1, color="0.6", zorder=2)
    a1.scatter(np.maximum(p_s, floor), y, s=26,
               c=np.where(reach, OBS_COLOR, RES_COLOR), zorder=3, edgecolor="none")
    # Every member of a direct ensemble carries exactly 1/n. Drawn as the reference the
    # weighted masses are read against: a walker far LEFT of it is one the scheme
    # over-produced and the weight has taken back.
    for key, (col, lab) in DS_STYLE.items():
        if key in ds and ds[key].get("n"):
            n_ = int(ds[key]["n"])
            a1.axvline(1.0 / n_, color=col, lw=1.2, ls=":", zorder=4,
                       label=f"a direct member's 1/{n_}")
            break
    a1.set_xscale("log")
    a1.set_xlim(floor, max(1.0, float(p_s.max()) * 2.0))
    a1.set_xlabel(r"probability mass $p_i = Z\,w_i/N$")
    tot = float(p.sum())
    p_obs = float(p_s[reach].sum()) if obs is not None else float("nan")
    head = (rf"$P(A_L \geq $ observed$) = \sum p_i = {p_obs:.4f}$" if sign > 0
            else rf"$P(A_L \leq $ observed$) = \sum p_i = {p_obs:.4f}$")
    a1.set_title("how likely GenCast says each one is"
                 + (f"\n{head}" if obs is not None else "")
                 + f"   -   all {al.size} sum to {tot:.3f}, which would be 1 if the "
                   f"estimator were unbiased", fontsize=9.5)
    a1.grid(alpha=0.25, axis="x", which="both")
    a1.legend(fontsize=8, loc="upper left")

    a0.set_ylim(-2.2, al.size - 0.3)
    a0.invert_yaxis()
    fig.suptitle(f"{ctx.ev.label} - every walker's anomaly next to its probability "
                 f"({AI.box_for(ctx.event).name} box, {ctx.tag})", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, _fig_path(ctx, "walkers"))


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
    if "walkers" in want:
        plot_walkers(ctx, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
