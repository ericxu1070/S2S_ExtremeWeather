#!/usr/bin/env python
"""The population itself: every surviving walker, its anomaly, and the mass it carries.

Every other AI+RES figure reports a *reduction* of the run - an exceedance curve, a
weighted PDF, a map composite. This one shows the raw material those reductions are made
of: all ``N`` walkers that survived to the horizon, where each one landed on the event
index ``A_L``, and the probability

    p_i  =  Z * w_i / N ,        w_i = exp(-V_K^i),   Z = prod_k R_k

that the estimator attaches to it. That definition is not invented here - it is
:func:`aires.alift.walker_mass`, imported rather than re-derived, and it is what makes
this figure exactly consistent with every probability the experiment publishes:

    P(A_L >= a)  =  sum of p_i over the walkers with A_L >= a          (a subset sum)
    sum_i p_i    =  the run's own normalization check                  (should be ~1)

So the headline number of each run is readable straight off the panel as "the total mass
sitting to the right of the red line", and the normalization check is "the total mass on
the panel at all". Nothing else on the figure has to be trusted for that to hold.

What the reference line is for
------------------------------
Each panel carries a horizontal line at ``1/N``. That is what a walker would be worth
under **direct sampling**: draw ``N`` i.i.d. members and each one carries exactly ``1/N``
of the probability, no more and no less. AI+RES buys reach by giving up that uniformity -
it clones the walkers heading for the tail, and then charges them for it. A walker sitting
two decades BELOW the line is one the sampler manufactured; a walker above it is one that
resampling under-represented and the weights are paying back. Reading the panel as
"anomaly on x, price on y" is the whole method in one picture.

The direct ensembles are drawn on the same axes at their own ``1/n`` (24 members -> 0.042,
16 -> 0.062) as tick marks, so where their population stops is visible against where the
RES population continues.

Three things the panels show that a summary table cannot
--------------------------------------------------------
- **The masses are tied in blocks.** Walkers cloned at the LAST resampling inherit their
  parent's ``V_K``, so they carry *identical* weights - up to 9 of them in Uri. The panel
  shows those as points stacked at one height. A run whose 64 walkers hold only 34
  distinct masses has 34 independent pieces of information, not 64, which is what the
  weight ESS in the corner reports.
- **Where the support ends.** ``p90_20251224``'s population starts at +1.25 K: resampling
  killed every walker below it, so no reweighting can put mass down there. That is a
  support limitation, not a weighting error, and it is visible as an empty left half.
- **The pinned case.** In ``p90_20231107`` the observation sits below EVERY walker, so
  the indicator is identically 1, the subset sum is the whole panel, and the reported
  probability collapses onto the normalization check. The panel makes that failure
  legible - the red line is off the left edge of the population - where `compare.json`
  alone shows a plausible-looking 0.583.

    PYTHONPATH=. python -m aires.awalkers                    # every reduced run + the CSV
    PYTHONPATH=. python -m aires.awalkers --per-event        # also one detail figure each
    PYTHONPATH=. python -m aires.awalkers --runs PNW_HeatDome_2021:pilot
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from aires import aconfig as A
from aires import aindex as AI
from aires.alift import walker_mass          # the estimator, not a copy of it
from fcn3 import fevents as F

# Which direct-sampling ensemble a panel compares against, in order of preference. The
# 24-member 0.25 deg xres cube is the baseline the wave tables quote; Gate 3's 16 free
# walkers stand in for events that have no cube, and FCN3 is a different model carried
# for context only - it is the last resort, never the headline comparison.
DS_PREFERENCE = ("gencast_xres", "gencast_walkers", "fcn3")
DS_LABEL = {
    "gencast_xres": "GenCast direct (xres)",
    "gencast_walkers": "GenCast direct (Gate 3 walkers)",
    "fcn3": "FCN3 direct",
}

RES_COLOR = "#1f77b4"
OBS_COLOR = "#d62728"
DIRECT_COLOR = "#8c564b"
REACH_FILL = "#d62728"


# --------------------------------------------------------------------------- #
# Pure pieces - no matplotlib, no network
# --------------------------------------------------------------------------- #
def panel_label(event: str) -> str:
    """The event's figure label, with a stale parenthetical stripped off the p90 cases.

    ``fevents`` labels them "p90 2023-11-07 (+3.1 K)", where the number is the PEAK-DAY
    CONUS-mean anomaly the case was SELECTED on. AI+RES scores the 7-day mean, on which
    that same week verified at +0.21 K. Printing both on one panel invites the reader to
    compare two different observables, so the selection number comes off and the observed
    ``A_L`` - which the panel states explicitly - is the only anomaly on it.
    """
    ev = F.event(event)
    lab = ev.label
    return re.sub(r"\s*\([^)]*\)\s*$", "", lab) if ev.family == "p90" else lab


def direct_ensemble(d: dict) -> tuple[str | None, np.ndarray]:
    """The preferred direct-sampling ensemble in this run's ``ds`` block, and its members."""
    ds = d.get("ds", {}) or {}
    for key in DS_PREFERENCE:
        rec = ds.get(key)
        if isinstance(rec, dict) and rec.get("box"):
            return key, np.asarray(rec["box"], dtype="float64")
    return None, np.zeros(0)


def sigma_depth(d: dict) -> float:
    """``sign * (obs - mean) / sd`` of the observation in the DIRECT ensemble's spread.

    The wave tables order events by this, not by the absolute anomaly, because reach
    tracks how far the observation sits outside the MODEL's forecast distribution: PNW at
    +7.72 K is +2.0 sigma and was reached, Southwest at +5.14 K is +3.3 sigma and was not.
    """
    obs = d.get("observed")
    _, v = direct_ensemble(d)
    if obs is None or v.size < 2:
        return float("nan")
    sd = float(np.std(v, ddof=1))
    sign = float(d["config"].get("tail_sign", 1.0))
    return float("nan") if sd <= 0 else sign * (float(obs) - float(np.mean(v))) / sd


def weight_ess(weights) -> float:
    """Kish ESS of the FINAL weights - how many walkers the reported probability rests on.

    Distinct from the per-step resampling ESS in `aires/dmc.py`: that one says how much of
    the population survived each barrier, this one says how much of it the estimator is
    actually listening to at the end. Uri's is 2.24 of 64.
    """
    w = np.asarray(weights, dtype="float64")
    s2 = float(np.sum(w ** 2))
    return float(np.sum(w) ** 2 / s2) if s2 > 0 else 0.0


def walker_table(event: str, tag: str, d: dict | None = None) -> pd.DataFrame:
    """One row per surviving walker: where it landed and what it is worth.

    ``rank`` counts from the most extreme walker IN THE TAIL DIRECTION, so rank 1 is the
    hottest walker of a heat event and the coldest of a freeze, and ``cum_p`` at rank r is
    ``P(A_L >= A_L of walker r)`` - the exceedance curve, evaluated at the population's
    own points. ``family`` groups walkers by identical final weight, which happens exactly
    when they were cloned from one parent at the last resampling.
    """
    if d is None:
        d = json.loads((A.res_dir(event, tag) / "compare.json").read_text())
    al = np.asarray(d["realized"]["box"], dtype="float64")
    conus = np.asarray(d["realized"].get("conus", np.full(al.size, np.nan)), dtype="float64")
    w = np.asarray(d["weights"], dtype="float64")
    p = walker_mass(d)
    sign = float(d["config"].get("tail_sign", 1.0))
    obs = d.get("observed")
    anc = np.asarray(d.get("ancestry", np.arange(al.size)), dtype="int64")

    order = np.argsort(-sign * al, kind="stable")       # most extreme first
    rank = np.empty(al.size, dtype="int64")
    rank[order] = np.arange(1, al.size + 1)
    cum = np.empty(al.size, dtype="float64")
    cum[order] = np.cumsum(p[order])

    # Identical weight <=> identical V_K <=> same parent at the final resampling: V is
    # C_K * z and distinct walkers have distinct standardized scores, so the coincidence
    # cannot happen any other way. Grouping on the float is exact, not a tolerance.
    _, fam_id, fam_n = np.unique(w, return_inverse=True, return_counts=True)

    return pd.DataFrame({
        "event": event, "tag": tag,
        "walker": np.arange(al.size),
        "founder": anc,
        "family": fam_id,
        "family_size": fam_n[fam_id],
        "A_L": al,
        "A_L_conus": conus,
        "weight": w,
        "p_i": p,
        "p_over_direct": p * al.size,       # p_i / (1/N)
        "rank": rank,
        "cum_p": cum,
        "reached": np.full(al.size, False) if obs is None else (sign * al >= sign * float(obs)),
        "observed": np.nan if obs is None else float(obs),
        "tail_sign": sign,
    })


def run_record(event: str, tag: str) -> dict:
    """Everything one panel needs. Raises ``FileNotFoundError`` if the run is not reduced."""
    path = A.res_dir(event, tag) / "compare.json"
    if not path.exists():
        raise FileNotFoundError(path)
    d = json.loads(path.read_text())
    t = walker_table(event, tag, d)
    key, direct = direct_ensemble(d)
    obs = d.get("observed")
    sign = float(d["config"].get("tail_sign", 1.0))
    return {
        "event": event, "tag": tag, "d": d, "table": t,
        "label": panel_label(event),
        "box": AI.box_for(event).name,
        "metric": d["ds"].get("metric", F.event(event).metric),
        "sign": sign,
        "observed": obs,
        "n": int(t.shape[0]),
        "P": float(t.loc[t["reached"], "p_i"].sum()),
        "n_reached": int(t["reached"].sum()),
        "total_mass": float(t["p_i"].sum()),
        "norm_check": float(d.get("normalization_check", np.nan)),
        # The run's own exceedance evaluated at the population's extreme walker - the
        # deepest probability this run can speak to at all.
        "deepest": float(t.loc[t["rank"] == 1, "cum_p"].iloc[0]),
        "log_Z": float(d["log_Z"]),
        "n_founders": int(d.get("n_founders", t["founder"].nunique())),
        "n_families": int(t["family"].nunique()),
        "weight_ess": weight_ess(d["weights"]),
        "sigma_depth": sigma_depth(d),
        "direct_key": key,
        "direct": direct,
        "direct_p": (1.0 / direct.size) if direct.size else np.nan,
        "direct_reached": (0 if obs is None or not direct.size
                           else int((sign * direct >= sign * float(obs)).sum())),
        # The pinned failure mode: the observation is on the wrong side of the whole
        # population, so 1{A_L >= obs} is identically 1 and P collapses onto sum_i p_i.
        "pinned": bool(obs is not None
                       and np.all(sign * t["A_L"].to_numpy() >= sign * float(obs))),
    }


def metric_unit(metric: str) -> str:
    """The metric's own unit. Mirrors ``aires.aplots.metric_unit`` - kept local so this
    module does not import the figure stack it sits beside."""
    return "K" if metric == "t2m_anom" else (
        "m s$^{-1}$" if "speed" in metric else ("mm" if metric.startswith("tp") else ""))


def _fmt_p(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    return f"{x:.3f}" if x >= 0.01 else f"{x:.2e}"


# --------------------------------------------------------------------------- #
# Figure 1 - the whole slate, one panel per run
# --------------------------------------------------------------------------- #
def plot_walkers(runs: list[dict], out: Path | None = None, ncols: int = 4) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Hardest target first. The ordering is verification-side only - it decides which cell
    # a run is drawn in and never touches a mass - and it is what makes the grid readable
    # as a severity ladder: reach falls off from left to right, and the two runs that
    # cannot answer (Southwest, p90_20231107) sit at the two ends.
    runs = sorted(runs, key=lambda r: (-r["sigma_depth"] if np.isfinite(r["sigma_depth"])
                                       else 99, r["event"], r["tag"]))
    n = len(runs)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    lo = min(float(r["table"]["p_i"].min()) for r in runs)
    hi = max([float(r["table"]["p_i"].max()) for r in runs]
             + [float(r["direct_p"]) for r in runs if np.isfinite(r["direct_p"])])
    ylo, yhi = 10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.55 * ncols, 3.55 * nrows),
                             squeeze=False)
    cmap = matplotlib.colormaps["tab20"]

    for ax, r in zip(axes.ravel(), runs):
        t, sign, obs = r["table"], r["sign"], r["observed"]
        al = t["A_L"].to_numpy()
        unit = metric_unit(r["metric"])

        # --- direct sampling: every member worth exactly 1/n --------------------- #
        if r["direct"].size:
            ax.plot(r["direct"], np.full(r["direct"].size, r["direct_p"]), ls="none",
                    marker="|", ms=13, mew=1.5, color=DIRECT_COLOR, alpha=0.95, zorder=4)
            ax.axhline(r["direct_p"], color=DIRECT_COLOR, lw=0.8, ls=":", alpha=0.55,
                       zorder=2)

        # --- 1/N: what a walker would be worth if nothing had been resampled ----- #
        ax.axhline(1.0 / r["n"], color=RES_COLOR, lw=1.0, ls=":", alpha=0.8, zorder=2)

        # --- the population ------------------------------------------------------ #
        founders = sorted(t["founder"].unique())
        colors = {f: cmap(i % 20) for i, f in enumerate(founders)}
        ax.scatter(al, t["p_i"], s=34,
                   c=[colors[f] for f in t["founder"]],
                   edgecolor="0.25", linewidth=0.35, zorder=5)
        if obs is not None:
            ax.axvline(obs, color=OBS_COLOR, lw=1.5, ls="--", zorder=6)

        ax.set_yscale("log")
        ax.set_ylim(ylo, yhi)

        # x limits are set by hand over walkers + direct members + the observation.
        # Autoscale does not pad for an axvline, so on the runs where the observation
        # lies beyond every walker (Southwest, the persistence control) the red line
        # lands flush on the frame and the region that counts has no width to be seen in.
        xs = [al.min(), al.max()]
        if r["direct"].size:
            xs += [float(r["direct"].min()), float(r["direct"].max())]
        if obs is not None:
            xs.append(float(obs))
        lo_x, hi_x = min(xs), max(xs)
        pad = 0.06 * max(hi_x - lo_x, 1e-6)
        ax.set_xlim(lo_x - pad, hi_x + pad)
        if sign < 0:
            ax.invert_xaxis()

        # --- the reach region: the mass that sums to the headline probability ---- #
        #
        # It runs from the observation to the FAR EDGE OF THE AXIS in the tail direction,
        # not to the most extreme walker. Shading out to the population instead puts the
        # band on the wrong side of the red line whenever nothing reached - which is
        # exactly the case a reader most needs to read correctly. The band is the region
        # that COUNTS; whether anything is in it is what the panel is asking.
        if obs is not None:
            far = (hi_x + pad) if sign > 0 else (lo_x - pad)
            ax.axvspan(min(float(obs), far), max(float(obs), far), color=REACH_FILL,
                       alpha=0.07, lw=0, zorder=0)
        ax.grid(alpha=0.22, which="both")
        ax.tick_params(labelsize=8)

        depth = r["sigma_depth"]
        # With nothing in the band the estimator returns exactly 0, which is not the
        # honest statement. The deepest level the run RESOLVED - the mass of its most
        # extreme walker, i.e. the run's own curve evaluated at the population edge - is,
        # and it is the number the wave table quotes as "< 3e-4" for Southwest.
        head = (f"P = {_fmt_p(r['P'])}  PINNED" if r["pinned"] else
                (f"P = 0;  resolved to {r['deepest']:.1e}" if not r["n_reached"]
                 else f"P = {_fmt_p(r['P'])}"))
        ax.set_title(
            f"{r['label']}" + ("" if r["tag"] == "pilot" else f"  [{r['tag']}]")
            + f"   ({r['box']} box)\n"
            + f"obs {obs:+.2f} {unit}"
            + (f", {depth:+.1f}$\\sigma$" if np.isfinite(depth) else "")
            + f"   |   {r['n_reached']}/{r['n']} reach it   |   {head}",
            fontsize=8.8)
        ax.set_xlabel(rf"walker anomaly $A_L$   ({unit})", fontsize=8.5)

        # Corner block: the two numbers a reader would otherwise have to open
        # compare.json for, plus how much of the population the estimate rests on.
        ax.text(0.025, 0.03,
                f"$\\Sigma p_i$ = {r['total_mass']:.3f}\n"
                f"{r['n_families']} distinct masses\n"
                f"weight ESS {r['weight_ess']:.1f}/{r['n']}",
                transform=ax.transAxes, fontsize=7.2, color="0.30", va="bottom",
                ha="left", linespacing=1.35,
                bbox=dict(fc="white", ec="0.8", lw=0.5, alpha=0.85, pad=2.2))

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for i, ax in enumerate(axes.ravel()[:n]):
        if i % ncols == 0:
            ax.set_ylabel(r"walker mass  $p_i = Z\,w_i/N$", fontsize=9)

    handles = [
        Line2D([], [], marker="o", ls="none", mfc="0.55", mec="0.25", ms=7,
               label="one AI+RES walker (colour = founder lineage)"),
        Line2D([], [], color=RES_COLOR, ls=":", lw=1.2,
               label=r"$1/N$ - what a walker is worth under direct sampling"),
        Line2D([], [], marker="|", ls="none", color=DIRECT_COLOR, ms=13, mew=1.5,
               label="direct-sampling member, at its own $1/n$"),
        Line2D([], [], color=OBS_COLOR, ls="--", lw=1.5, label="ERA5 observed $A_L$"),
        Patch(fc=REACH_FILL, alpha=0.18, ec="none",
              label=r"the mass that sums to $P(A_L \geq {\rm obs})$"),
    ]
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.012), columnspacing=1.8, handletextpad=0.6)
    fig.suptitle("AI+RES: every surviving walker, its anomaly, and the probability it "
                 "carries\n"
                 r"each panel's headline $P$ is the mass right of the red line; "
                 r"$\Sigma p_i$ over the whole panel is the run's normalization check",
                 fontsize=13, y=1.0)
    fig.text(0.5, -0.048,
             "Panels ordered by how far the observation sat outside the DIRECT ensemble's "
             "spread, hardest first. A walker far below the 1/N line is one resampling "
             "manufactured and the weights are charging for;\nwalkers stacked at one "
             "height are clones of a single parent at the last resampling and carry "
             "identical mass by construction. Two runs cannot answer their own question: "
             "Southwest and the persistence control reached nothing, so P is 0 and the\n"
             "honest statement is the deepest level the run resolved; and p90 2023-11-07 "
             "is PINNED - the observation lies below every walker, so the subset sum is "
             "the whole panel and P equals the normalization check, not a probability.",
             ha="center", va="top", fontsize=8.4, color="0.35", linespacing=1.55)
    fig.tight_layout(rect=[0, 0.012, 1, 0.965])

    out = out or (A.fig_dir() / "aires_walkers.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return out


# --------------------------------------------------------------------------- #
# Figure 2 - one run, all N walkers named
# --------------------------------------------------------------------------- #
def plot_run(r: dict, out: Path | None = None) -> Path:
    """All ``N`` walkers of one run, in rank order, with the running total that builds
    the exceedance probability out of them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = r["table"].sort_values("rank").reset_index(drop=True)
    n, sign, obs = r["n"], r["sign"], r["observed"]
    unit = metric_unit(r["metric"])
    x = np.arange(n)
    cmap = matplotlib.colormaps["tab20"]
    founders = sorted(t["founder"].unique())
    colors = {f: cmap(i % 20) for i, f in enumerate(founders)}
    face = [colors[f] for f in t["founder"]]
    al = t["A_L"].to_numpy()

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(max(13.0, 0.21 * n + 4.0), 8.2),
                                 sharex=True, height_ratios=[1.0, 1.15])

    # --- top: where each walker landed --------------------------------------- #
    base = 0.0
    ax.vlines(x, base, al, color="0.75", lw=1.0, zorder=2)
    ax.scatter(x, al, s=40, c=face, edgecolor="0.25", linewidth=0.4, zorder=4)
    if r["direct"].size:
        v = r["direct"]
        ax.axhspan(float(v.min()), float(v.max()), color=DIRECT_COLOR, alpha=0.06, lw=0,
                   zorder=1)
        best = float(v.max() if sign > 0 else v.min())
        ax.axhline(best, color=DIRECT_COLOR, lw=1.2, ls="-.", zorder=3)
        ax.text(n - 0.5, best, f" best {DS_LABEL[r['direct_key']]} member ({best:+.2f})",
                fontsize=8, color=DIRECT_COLOR, va="bottom", ha="right")
    if obs is not None:
        ax.axhline(obs, color=OBS_COLOR, lw=1.6, ls="--", zorder=5)
        ax.text(0.2, obs, f" ERA5 observed {obs:+.2f} {unit}", fontsize=9,
                color=OBS_COLOR, va="bottom", ha="left")
        top = max(float(al.max()), float(obs)) + 0.08 * np.ptp(al)
        bot = min(float(al.min()), float(obs),
                  float(r["direct"].min()) if r["direct"].size else float(al.min()))
        lo_y, hi_y = sorted([bot - 0.08 * np.ptp(al), top])
        ax.set_ylim(lo_y, hi_y)
        far = hi_y if sign > 0 else lo_y
        ax.axhspan(min(float(obs), far), max(float(obs), far), color=REACH_FILL,
                   alpha=0.10, lw=0, zorder=0)
    ax.axhline(base, color="0.5", lw=0.8)
    ax.set_ylabel(rf"realized $A_L$   ({unit})")
    ax.grid(alpha=0.22, axis="y")
    ax.set_title(f"{r['label']} ({r['tag']}) - all {n} surviving walkers, ranked "
                 f"{'hottest' if sign > 0 else 'coldest'} first", fontsize=11)

    # --- bottom: what each is worth ------------------------------------------ #
    bx.bar(x, t["p_i"], width=0.72, color=face, edgecolor="0.25", linewidth=0.35,
           zorder=3)
    bx.axhline(1.0 / n, color=RES_COLOR, lw=1.1, ls=":", zorder=4)
    bx.text(0.2, 1.0 / n, f" 1/N = {1/n:.4f}  (AI+RES walker, if nothing were resampled)",
            fontsize=8, color=RES_COLOR, va="top", ha="left")
    if np.isfinite(r["direct_p"]):
        bx.axhline(r["direct_p"], color=DIRECT_COLOR, lw=1.0, ls=":", zorder=4)
        bx.text(0.2, r["direct_p"],
                f" 1/n = {r['direct_p']:.4f}  ({DS_LABEL[r['direct_key']]} member)",
                fontsize=8, color=DIRECT_COLOR, va="bottom", ha="left")
    bx.set_yscale("log")
    bx.set_ylabel(r"walker mass  $p_i = Z\,w_i/N$")
    bx.set_xlabel("walker slot, ranked by how extreme it got  (colour = founder lineage)")
    bx.grid(alpha=0.22, which="both", axis="y")
    bx.set_xticks(x)
    bx.set_xticklabels([f"{w}" for w in t["walker"]], fontsize=6.5, rotation=90)
    bx.set_xlim(-0.8, n - 0.2)

    # The running total IS the exceedance curve, evaluated at the population's own
    # points: cum_p at rank r equals P(A_L >= that walker's A_L). Drawing it here is what
    # turns "64 bars" into "the number the run reports".
    cx = bx.twinx()
    cx.step(x, t["cum_p"], where="post", color="0.15", lw=1.6, zorder=6)
    cx.set_yscale("log")
    cx.set_ylabel(r"running total $\sum p_i$  =  $P(A_L \geq$ this walker$)$", fontsize=9)
    cx.grid(False)
    # The callout is placed relative to the marker it points at, not at a fixed corner:
    # the marker sits at rank `n_reached`, which is anywhere from the first walker to the
    # last depending on the run, and a fixed anchor sends the arrow arcing out of the
    # axes and across the panel above whenever the two are far apart.
    def _anchor(rank0: int, y: float) -> tuple[float, float]:
        # The running total rises left to right and the bars are tallest on the right, so
        # the upper LEFT is empty in every run - unless the marker itself is up there,
        # which happens when almost every walker counts (the pinned case).
        y0, y1 = cx.get_ylim()
        mf = ((np.log10(max(y, 1e-300)) - np.log10(y0))
              / max(np.log10(y1) - np.log10(y0), 1e-9))
        xf = rank0 / max(n - 1, 1)
        return (0.52 if (xf < 0.25 and mf > 0.80) else 0.035), 0.965

    if obs is not None and r["n_reached"]:
        k = int(r["n_reached"])
        cx.plot([k - 1], [r["P"]], marker="o", ms=9, mfc="none", mec=OBS_COLOR, mew=2,
                zorder=7)
        cx.annotate(rf"$P(A_L \{'geq' if sign > 0 else 'leq'}$ obs$)$ = "
                    f"{_fmt_p(r['P'])}" + "\n" + f"the running total after "
                    f"{k}/{n} walkers",
                    xy=(k - 1, r["P"]), xycoords="data",
                    xytext=_anchor(k - 1, r["P"]), textcoords="axes fraction",
                    fontsize=9.5, color=OBS_COLOR, ha="left", va="top",
                    bbox=dict(fc="white", ec=OBS_COLOR, lw=0.8, alpha=0.92, pad=3),
                    arrowprops=dict(arrowstyle="->", color=OBS_COLOR, lw=1.3,
                                    shrinkB=7), zorder=8)
    elif obs is not None:
        # Nothing reached: the running total never gets to the observation, so the
        # honest statement is where the curve STOPS - the deepest level this run can
        # speak to at all. Annotating the estimator's literal 0 would say more than the
        # run knows.
        cx.plot([0], [r["deepest"]], marker="o", ms=9, mfc="none", mec=OBS_COLOR, mew=2,
                zorder=7)
        cx.annotate("no walker reached the observation.\n"
                    f"the run resolves only to {r['deepest']:.1e},\n"
                    "at its most extreme walker",
                    xy=(0, r["deepest"]), xycoords="data",
                    xytext=_anchor(0, r["deepest"]), textcoords="axes fraction",
                    fontsize=9.5, color=OBS_COLOR, ha="left", va="top",
                    bbox=dict(fc="white", ec=OBS_COLOR, lw=0.8, alpha=0.92, pad=3),
                    arrowprops=dict(arrowstyle="->", color=OBS_COLOR, lw=1.3,
                                    shrinkB=7), zorder=8)
    for i in range(n):
        if bool(t["reached"].iloc[i]):
            bx.get_xticklabels()[i].set_color(OBS_COLOR)

    fam = t.groupby("family")["family_size"].first()
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc="0.55", mec="0.25", ms=7,
               label="one walker (colour = founder lineage)"),
        Line2D([], [], color=OBS_COLOR, ls="--", lw=1.6, label="ERA5 observed"),
        Line2D([], [], color=DIRECT_COLOR, ls="-.", lw=1.2,
               label="best direct member (band = its spread)"),
        Line2D([], [], color="0.15", lw=1.6,
               label=r"running total $\sum p_i$ (lower panel)"),
    ], fontsize=8.5, loc="lower center", ncol=4, columnspacing=1.5,
        framealpha=0.9, borderpad=0.4)
    fig.text(0.5, 0.005,
             f"$\\Sigma p_i$ = {r['total_mass']:.4f} (the run's normalization check) | "
             f"log Z = {r['log_Z']:+.4f} | {r['n_founders']} of {n} founders survive | "
             f"{r['n_families']} distinct masses among {n} walkers "
             f"(largest clone family {int(fam.max())}) | weight ESS "
             f"{r['weight_ess']:.2f}/{n}"
             + ("   |   PINNED: the observation lies below every walker, so P is the "
                "normalization check, not a probability" if r["pinned"] else ""),
             ha="center", va="bottom", fontsize=8.4, color="0.35")
    fig.tight_layout(rect=[0, 0.022, 1, 0.965])

    out = out or (A.fig_dir(r["event"]) / f"aires_walkers_{r['event']}_{r['tag']}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return out


# --------------------------------------------------------------------------- #
def discover() -> list[tuple[str, str]]:
    """Every ``(event, tag)`` under ``runs/aires`` that has a reduced ``compare.json``."""
    return [(p.parents[2].name, p.parent.name)
            for p in sorted(A.AIRES_ROOT.glob("*/res/*/compare.json"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None,
                    help="comma-separated event:tag; default is every reduced run")
    ap.add_argument("--out", default=None, help="path for the cross-event figure")
    ap.add_argument("--csv", default=None,
                    help=f"per-walker table (default {A.AIRES_ROOT / 'aires_walkers.csv'}; "
                         f"'none' to skip)")
    ap.add_argument("--per-event", action="store_true",
                    help="also write one detail figure per run")
    ap.add_argument("--ncols", type=int, default=4)
    a = ap.parse_args(argv)

    runs = ([tuple(x.split(":", 1)) for x in a.runs.split(",")] if a.runs else discover())
    if not runs:
        raise SystemExit("no reduced runs found; run `--stage compare` first")

    recs, tables = [], []
    for ev, tag in runs:
        try:
            r = run_record(ev, tag)
        except FileNotFoundError as e:
            print(f"  skip {ev}:{tag} - no {e}")
            continue
        recs.append(r)
        tables.append(r["table"])
        t = r["table"]
        print(f"  {ev:26s} {tag:8s} N={r['n']:3d} obs={r['observed']:+8.3f} "
              f"depth={r['sigma_depth']:+5.2f}sd  reach={r['n_reached']:2d}/{r['n']}  "
              f"P={_fmt_p(r['P']):>9s}  sum p_i={r['total_mass']:.4f}  "
              f"p_i in [{t['p_i'].min():.2e}, {t['p_i'].max():.2e}]  "
              f"masses={r['n_families']:2d}  wESS={r['weight_ess']:5.2f}"
              + ("  PINNED" if r["pinned"] else ""))

    plot_walkers(recs, Path(a.out) if a.out else None, ncols=a.ncols)
    if a.per_event:
        for r in recs:
            plot_run(r)

    if (a.csv or "").lower() != "none":
        out = Path(a.csv) if a.csv else (A.AIRES_ROOT / "aires_walkers.csv")
        df = pd.concat(tables, ignore_index=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  wrote {out} ({len(df)} walkers x {df.shape[1]} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
