#!/usr/bin/env python
"""Is the forecast saying anything climatology does not? -- the cross-event lift figure.

Every other AI+RES figure anchors on the observation: `aires/aplots.py` reports
``P(A_L >= observed)``, shades "right of the red line", and counts how many walkers got
there. Those are verification statements, and two of them use the outcome to place the
threshold - so they cannot answer the question this module is for:

    when the event WAS extreme, did the run put probability mass out in the tail, and
    when it was ordinary, did it not? -- decided without any hindsight.

The forecast itself never had hindsight. The walkers are initialised 21 days before the
peak, FCN3 scores them, and no truth enters the loop anywhere. What leaks hindsight is
the CHOICE OF THRESHOLD: "how much mass is beyond what happened" is a different question
per event and is not comparable between them. So this figure moves the threshold onto an
a-priori axis - the box's own seasonal CLIMATOLOGY, from `aires/aclim.py`, 1990-2019, the
same 30 years the anomaly is defined against - and asks, at each climatological rarity
level,

    lift(a) = P_forecast(A_L >= a) / P_climatology(A_L >= a)

lift == 1 means the run is saying exactly what climatology says. lift >> 1 in the deep
tail is the run calling an extreme event. Because both sides are in climatological units,
a box whose 99th percentile is +7.06 K (PNW) and one whose 99th is +3.14 K (California)
land on the same axis, and a cold event (``tail_sign = -1``) uses its lower tail with no
change of code.

Why there is no extrapolated climatology on this figure
-------------------------------------------------------
`aires.aclim` gives 64 years of the SAME index - the 6-day box-mean T2m anomaly - pooled
within +-21 days of the event's calendar position. Restricted to the 1990-2019 baseline
the pool holds 1290 overlapping windows, so it resolves exceedance probabilities down to
``1 / n_pool`` = 7.8e-4 and no further. The interesting part of a RES run lies BEYOND
that: for the PNW pilot, 31 of 64 walkers are hotter than anything in 30 years of that
season, carrying 2.5% of the mass.

The obvious move is to continue the climatological curve with a fitted tail. **It was
tried and it does not survive contact with the data.** A GPD fitted to runs-declustered
cluster maxima has only 14-27 clusters to work with, and its shape parameter swings from
-0.11 (threshold p90) to -0.45 (p95) to -1.76 (p97.5) for the PNW box alone - putting the
implied upper endpoint of the climatological distribution anywhere between +8.2 K and
+17.9 K. The PNW walkers reach +10.3 K, i.e. exactly inside the range where the fit
cannot decide whether the event is merely rare or flatly impossible. A lift computed
against that is a number about the fit, not about the forecast.

So the climatology stops where the data stops, and past it the figure reports a **lower
bound** instead: 1290 matched-season windows containing no analogue means
``P_clim <= 1/1290``, hence

    lift  >=  P_forecast(A_L >= pool max) * n_pool

drawn as a right-pointing arrow in the shaded band. That is a weaker claim than a fitted
curve and it is the one the data actually supports.

Two honest caveats, both stated on the figure
---------------------------------------------
- **The events were selected for being extreme.** Four record-breaking heat cases all
  showing lift >> 1 is partly selection, not skill. The discrimination claim needs the
  ``null_*`` cases, which have no RES run yet; the panel is built so they drop straight in.
- **Sum of the walker masses is not 1.** ``sum_i p_i`` is the run's own normalization
  check, and it is noisy at N = 64 (see `aires/dmc.py`). It is printed per run rather than
  divided out, because dividing it out would hide a degenerate run instead of showing it.

    PYTHONPATH=. python -m aires.alift                      # every run that has a compare
    PYTHONPATH=. python -m aires.alift --runs PNW_HeatDome_2021:pilot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from aires import aclim as CL
from aires import aconfig as A
from aires import aindex as AI
from fcn3 import fevents as F

# The a-priori rungs the summary panel reports at, chosen before looking at any run.
# p99.9 is deliberately NOT among them: 1290 pooled windows put only ~1.3 samples above
# it, so its empirical level is a single order statistic and the lift built on it is one
# draw wide. p95 and p99 are resolved by 65 and 13 windows respectively.
RUNGS = (95.0, 99.0)

PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b",
           "#e377c2", "#bcbd22", "#393b79", "#8c6d31", "#5254a3")
CLIM_COLOR = "#444444"


# --------------------------------------------------------------------------- #
# Pure pieces (no matplotlib, no I/O beyond the caches)
# --------------------------------------------------------------------------- #
def walker_mass(d: dict) -> np.ndarray:
    """``p_i = Z * w_i / N``: the probability mass each surviving walker carries.

    Every probability the experiment reports is a subset sum of this vector - it is
    ``DMCResult.expectation`` written per walker rather than summed - so building the
    forecast curve out of it keeps this figure and the exceedance figure exactly
    consistent by construction.
    """
    w = np.asarray(d["weights"], dtype="float64")
    return float(np.exp(d["log_Z"])) * w / w.size


def forecast_exceedance(values, mass, thresholds, sign: float = 1.0) -> np.ndarray:
    """``P(A_L >= a)`` under the original law, for each ``a``, from the walker masses."""
    v = np.asarray(values, dtype="float64")
    m = np.asarray(mass, dtype="float64")
    return np.array([float(m[sign * v >= sign * float(a)].sum())
                     for a in np.atleast_1d(thresholds)])


def clim_exceedance(sample, thresholds, sign: float = 1.0) -> np.ndarray:
    """Empirical ``P(A_L >= a)`` in the seasonal pool. Zero past the pool's own extreme.

    Deliberately not smoothed or extrapolated - see the module docstring. A zero here
    means "no analogue in the pool", which the figure renders as a bound, not as a curve.
    """
    x = np.asarray(sample, dtype="float64")
    return np.array([float((sign * x >= sign * float(a)).mean())
                     for a in np.atleast_1d(thresholds)])


def clim_pool(event: str, anom: pd.DataFrame | None = None) -> tuple[np.ndarray, float]:
    """The seasonal climatological sample of ``A_L`` for an event, and its tail sign.

    The two ways this silently returns the WRONG reference are both checked here, because
    a lift is a ratio and a wrong denominator does not look wrong - it looks like a result.

    1. **A box with no column in the cache.** `aires.aclim` reduces whatever
       ``aindex.EVENT_BOXES`` held WHEN IT WAS BUILT, so an event registered afterwards
       has a box but no climatology, and falling back to the CONUS column compares a
       regional box mean against a continental one. That is exactly the trap CLAUDE.md
       records for the two ``clim_1990_2019`` files: same name, different domain, no
       error. It cost this figure a spurious ``lift >= 285x`` for Elliott, whose N Plains
       ``A_L`` reaches -17 K against a CONUS climatology spanning +-5 K.
    2. **A gappy pool.** ``pool`` drops NaNs, so a box the top-up file never covered comes
       back as a short sample rather than an error, and every quantile shifts.

    Events with no registered box (the ``p90_*`` cases) are defined on CONUS in the first
    place, so for them the CONUS column is the right answer, not a fallback.
    """
    anom = CL.daily_anomaly() if anom is None else anom
    ev = F.event(event)
    if event in AI.EVENT_BOXES and event not in anom.columns:
        raise SystemExit(
            f"{event} has its own box ({AI.box_for(event).name}) but the climatology "
            f"cache has no column for it, so its A_L would be scored against the CONUS "
            f"mean - a different observable.\n  Rebuild the cache:\n"
            f"    PYTHONPATH=. python -m aires.aclim --build\n"
            f"  cache: {CL.CACHE}\n  columns: {sorted(anom.columns)}")
    col = event if event in anom.columns else "CONUS"
    series = CL.window_index(anom[col])
    samp = np.asarray(CL.pool(series, pd.Timestamp(ev.peak), CL.BASELINE), dtype="float64")

    span = CL.BASELINE[1] - CL.BASELINE[0] + 1
    expect = span * (2 * CL.SEASON_HALF_WIDTH + 1)
    if samp.size < 0.9 * expect:
        raise SystemExit(
            f"{event}: seasonal pool for box {col} has {samp.size} windows, expected "
            f"~{expect} ({span} yr x {2 * CL.SEASON_HALF_WIDTH + 1} d). The cache is "
            f"gappy for this box - rebuild it before trusting any rarity from it.")
    return samp, float(getattr(ev, "tail_sign", 1.0))


def lift_curve(event: str, tag: str, anom: pd.DataFrame | None = None,
               n_grid: int = 400) -> dict:
    """Everything the figure needs for one run. No plotting, no hindsight in any curve."""
    path = A.res_dir(event, tag) / "compare.json"
    if not path.exists():
        raise FileNotFoundError(path)
    d = json.loads(path.read_text())
    sign = float(d["config"].get("tail_sign", 1.0))

    al = np.asarray(d["realized"]["box"], dtype="float64")
    mass = walker_mass(d)
    samp, _ = clim_pool(event, anom)

    # Grid spans the climatological median out to the most extreme walker, in the tail
    # direction. Everything on the near side of the median is bulk and says nothing about
    # whether the run called an extreme.
    lo, hi = float(np.median(samp)), float(al.max() if sign > 0 else al.min())
    grid = np.linspace(lo, hi, n_grid)
    p_fc = forecast_exceedance(al, mass, grid, sign)
    p_cl = clim_exceedance(samp, grid, sign)

    # The bound past the pool: no analogue in n_pool windows => P_clim <= 1/n_pool.
    edge = float(samp.max() if sign > 0 else samp.min())
    beyond = float(forecast_exceedance(al, mass, edge, sign)[0])
    n_pool = int(samp.size)

    rungs = {}
    for lv in RUNGS:
        a = float(np.quantile(samp, lv / 100.0 if sign > 0 else 1.0 - lv / 100.0))
        pc = float(clim_exceedance(samp, a, sign)[0])
        pf = float(forecast_exceedance(al, mass, a, sign)[0])
        rungs[lv] = {"threshold": a, "p_clim": pc, "p_fc": pf,
                     "lift": (pf / pc) if pc > 0 else float("nan")}

    obs = d.get("observed")
    return {
        "event": event, "tag": tag, "sign": sign, "box": AI.box_for(event).name,
        "n_walkers": int(d["config"]["n_walkers"]),
        "grid": grid, "p_fc": p_fc, "p_clim": p_cl,
        "pool_edge": edge, "pool_n": n_pool,
        "beyond_pool_mass": beyond,
        "beyond_pool_walkers": int(((al >= edge) if sign > 0 else (al <= edge)).sum()),
        "lift_bound": beyond * n_pool,
        "total_mass": float(mass.sum()),
        "log_Z": float(d["log_Z"]),
        "ess": float(np.sum(d["weights"]) ** 2 / np.sum(np.asarray(d["weights"]) ** 2)),
        "rungs": rungs,
        # --- verification only, never used to build a curve above --------------- #
        "observed": obs,
        "obs_pct": None if obs is None else CL.percentile_of(obs, samp, sign),
        "obs_p_clim": None if obs is None else float(clim_exceedance(samp, obs, sign)[0]),
        "obs_p_fc": None if obs is None else float(forecast_exceedance(al, mass, obs, sign)[0]),
    }


def _lift_str(x) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.2f}x" if x < 1 else (f"{x:.1f}x" if x < 10 else f"{x:.0f}x")


def spread_labels(y, min_ratio: float = 1.9, lo: float = 1e-30, hi: float = 1e30):
    """Nudge label positions apart in LOG space, preserving their order.

    Several runs put nearly the same mass beyond the pool (0.0117 and 0.0135 here), so
    their bound annotations land on top of each other. One pass down the sorted list,
    pushing each label to at least ``min_ratio`` times its neighbour, is enough and keeps
    every label attached to the arrow it belongs to.
    """
    y = np.asarray(y, dtype="float64")
    out = y.copy()
    order = np.argsort(y)
    for a, b in zip(order[:-1], order[1:]):
        if out[a] > 0 and out[b] < out[a] * min_ratio:
            out[b] = out[a] * min_ratio
    return np.clip(out, lo, hi)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_lift(curves: list[dict], out: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Order by how extreme the outcome ACTUALLY was, most extreme first. That ordering is
    # verification, not construction - it changes only which row a run is drawn on, never
    # a curve or a lift - and it is what turns the right panel into a discrimination plot:
    # if the run called extremes and stood down on ordinary weeks, the dots march right to
    # left down the panel. Whether they do is the question the figure is asking.
    curves = sorted(curves, key=lambda c: (c["obs_pct"] is not None,
                                           c["obs_pct"] or 0.0), reverse=True)
    n = len(curves)
    fig = plt.figure(figsize=(16.6, max(7.6, 0.80 * n + 3.0)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])

    floor = min(1.0 / c["pool_n"] for c in curves)
    ylo = 10 ** np.floor(np.log10(max(min(
        [c["p_fc"][c["p_fc"] > 0].min() for c in curves if (c["p_fc"] > 0).any()]),
        1e-6)))
    xlo = floor / 6.0

    # --- the region the pool cannot resolve --------------------------------- #
    ax.axvspan(xlo, floor, color="0.88", alpha=0.7, lw=0, zorder=0)
    ax.text(np.sqrt(xlo * floor), 0.975,
            "no analogue in 30 yr\nof this season:\n"
            rf"$P_{{\rm clim}} \leq 1/{curves[0]['pool_n']}$",
            fontsize=8, color="0.3", ha="center", va="top",
            transform=ax.get_xaxis_transform())

    # --- lift == 1 and its decades ------------------------------------------ #
    ref = np.array([xlo, 1.0])
    decades = ((1, "climatology  (lift = 1)", "--"), (10, "10x", ":"), (100, "100x", ":"))
    for mult, _, st in decades:
        ax.plot(ref, np.minimum(ref * mult, 1.0), color=CLIM_COLOR, ls=st,
                lw=1.4 if mult == 1 else 0.7, alpha=1.0 if mult == 1 else 0.55,
                zorder=2 if mult == 1 else 1)

    bounds = spread_labels([c["beyond_pool_mass"] for c in curves])
    for i, c in enumerate(curves):
        col = PALETTE[i % len(PALETTE)]
        x, y = c["p_clim"], c["p_fc"]
        m = (x > 0) & (y > 0)
        ax.plot(x[m], y[m], color=col, lw=2.0, zorder=5, solid_capstyle="round")
        # past the pool: a bound, drawn as an arrow, not a curve
        if c["beyond_pool_mass"] > 0:
            ax.annotate("", xy=(xlo * 1.9, c["beyond_pool_mass"]),
                        xytext=(floor, c["beyond_pool_mass"]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                        shrinkA=0, shrinkB=0), zorder=6)
            ax.text(xlo * 1.7, bounds[i] * 1.15,
                    rf"$\geq\,${_lift_str(c['lift_bound'])}", color=col, fontsize=8.5,
                    ha="right", va="bottom", zorder=6)
        # The x axis is rarity in the event's OWN tail, so a cold event is directly
        # comparable to a hot one - but the reader has to be told which tail it is.
        tail = " (cold tail)" if c["sign"] < 0 else ""

        # verification marker
        if c["observed"] is not None and c["obs_p_clim"]:
            yv, mk = c["obs_p_fc"], "*"
            if not yv or yv <= 0:
                yv, mk = ylo * 1.7, "v"          # the run gave the outcome zero mass
            ax.plot([c["obs_p_clim"]], [yv], marker=mk, ms=17 if mk == "*" else 9,
                    mfc="none", mec=col, mew=1.8, ls="none", zorder=8)

    ax.plot([], [], marker="*", ms=14, mfc="none", mec="0.3", mew=1.6, ls="none",
            label="VERIFICATION ONLY: where the event landed")
    ax.plot([], [], marker="v", ms=8, mfc="none", mec="0.3", mew=1.6, ls="none",
            label="  ... and the run gave it zero mass")
    ax.plot([], [], color="none", label="curve colours match the rows at right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.0, xlo)                       # inverted: rarer to the RIGHT
    ax.set_ylim(ylo, 1.8)
    ax.set_xlabel(r"climatological exceedance   $P_{\mathrm{clim}}(A_L \geq a)$"
                  "\nsame box, same season, 1990-2019"
                  "                    rarer $\\longrightarrow$")
    ax.set_ylabel(r"forecast probability   $P_{\mathrm{fc}}(A_L \geq a)"
                  r" = \sum_i p_i \, \mathbf{1}\{A_L^i \geq a\}$")
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.96, borderpad=0.5,
              handletextpad=0.8)
    ax.set_title("the threshold axis is climatology, not the outcome:\n"
                 "nothing that builds these curves knows what happened",
                 fontsize=9.5, color="0.3")

    # Decade labels go on LAST: their on-screen angle depends on the axes aspect and on
    # the x axis being inverted, so it has to be measured from the final transform rather
    # than guessed. Anchored low enough to clear the legend and the curves.
    fig.canvas.draw()
    anchor = {1: 4.0e-3, 10: 2.2e-2, 100: 6.0e-3}   # free space, measured off the render
    for mult, lab, _ in decades:
        xa = anchor[mult]
        ya = xa * mult
        if not (ylo < ya < 1.0):
            continue
        p0 = ax.transData.transform((xa, ya))
        p1 = ax.transData.transform((xa / 10.0, ya / 10.0))
        ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
        ax.text(xa, ya * 1.13, lab, color=CLIM_COLOR,
                fontsize=8.5 if mult == 1 else 7.5, rotation=ang,
                rotation_mode="anchor", ha="left", va="bottom",
                alpha=1.0 if mult == 1 else 0.75, zorder=3)

    # --- right: lift at the a-priori rungs ---------------------------------- #
    marks = {RUNGS[0]: ("o", 78, "clim p95  (base rate 0.050)"),
             RUNGS[1]: ("D", 60, "clim p99  (base rate 0.010)")}
    for i, c in enumerate(curves[::-1]):
        col = PALETTE[(n - 1 - i) % len(PALETTE)]
        bx.plot([1e-3, 1e4], [i, i], color="0.92", lw=1.0, zorder=1)
        for lv, (mk, ms, _) in marks.items():
            L = c["rungs"][lv]["lift"]
            if np.isfinite(L) and L > 0:
                bx.scatter([L], [i], marker=mk, s=ms, color=col, edgecolor="0.2",
                           linewidth=0.5, zorder=4)
        if c["lift_bound"] > 0:
            bx.scatter([c["lift_bound"]], [i], marker=">", s=80, color=col,
                       edgecolor="0.2", linewidth=0.5, zorder=4)
    bx.axvline(1.0, color=CLIM_COLOR, lw=1.4, ls="--", zorder=3)
    bx.set_xscale("log")
    bx.set_xlim(5e-3, 5e2)
    bx.set_ylim(-0.55, n - 0.25)
    bx.text(1.06, n - 0.52, "climatology", color=CLIM_COLOR, fontsize=8.5, ha="left")
    # Row labels live INSIDE the panel: as y ticklabels they are wide enough to collide
    # with the left panel, and shortening them would drop the run health numbers that say
    # how much any of these dots is worth.
    for i, c in enumerate(curves[::-1]):
        col = PALETTE[(n - 1 - i) % len(PALETTE)]
        tail = "  (cold tail)" if c["sign"] < 0 else ""
        bx.text(5.6e-3, i + 0.30, f"{c['event'].replace('_', ' ')}  [{c['tag']}]{tail}",
                fontsize=9, ha="left", va="center", color=col, zorder=5)
        obs = ("" if c["obs_pct"] is None else
               f"observed = clim p{c['obs_pct']:.1f}   ")
        bx.text(5.6e-3, i + 0.12,
                f"{obs}$\\Sigma_i\\, p_i$={c['total_mass']:.2f}"
                f"   ESS {c['ess']:.0f}/{c['n_walkers']}",
                fontsize=7.6, ha="left", va="center", color="0.4", zorder=5)
    bx.tick_params(axis="y", length=0)
    bx.set_yticks([])
    bx.set_xlabel(r"lift $= P_{\mathrm{fc}} / P_{\mathrm{clim}}$"
                  "   at an a-priori climatological rung"
                  "\n" r"$\longleftarrow$ says it is ordinary        "
                  r"says it is extreme $\longrightarrow$")
    bx.grid(alpha=0.22, axis="x", which="major")
    bx.legend(handles=[Line2D([], [], marker=mk, ls="none", color="0.35", ms=8, label=lab)
                       for _, (mk, _, lab) in marks.items()]
                      + [Line2D([], [], marker=">", ls="none", color="0.35", ms=9,
                                label="lower bound past the pool")],
              fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3,
              frameon=False, columnspacing=1.6, handletextpad=0.5)
    bx.set_title("did the run call the event, without being told?\n"
                 "rows ordered by how extreme the week ACTUALLY was, most extreme first",
                 fontsize=9.5, color="0.3")

    fig.suptitle("AI+RES: forecast probability against the box's own climatology "
                 "- no hindsight in any curve", fontsize=13, y=0.985)
    y0 = min(ax.get_position().y0, bx.get_position().y0)
    fig.text(0.5, y0 - 0.115,
             "Selection: the six named events were chosen FOR being extreme, so a lift > 1 "
             "on those rows is partly selection, not skill.\nThe three p90_* cases were "
             "drawn on a different index (the daily CONUS-mean anomaly); on this figure's "
             "6-day observable one of them verified at clim p53.6 - an ordinary week -\n"
             "which makes it the closest thing on this slate to a null control. The "
             "purpose-built null_* controls still have no RES run.",
             ha="center", va="top", fontsize=8.5, color="0.35", linespacing=1.5)
    out = out or (A.fig_dir() / "aires_lift.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None, help="also dump the numbers here")
    a = ap.parse_args(argv)

    runs = ([tuple(x.split(":", 1)) for x in a.runs.split(",")] if a.runs else discover())
    if not runs:
        raise SystemExit("no reduced runs found; run `--stage compare` first")

    anom = CL.daily_anomaly()
    curves, rows = [], []
    for ev, tag in runs:
        try:
            c = lift_curve(ev, tag, anom)
        except FileNotFoundError as e:
            print(f"  skip {ev}:{tag} - no {e}")
            continue
        curves.append(c)
        print(f"  {ev:26s} {tag:8s} pool n={c['pool_n']:5d}  "
              + "  ".join(f"p{lv:g}: P_fc={c['rungs'][lv]['p_fc']:.4g} "
                          f"lift={_lift_str(c['rungs'][lv]['lift']):>6s}" for lv in RUNGS)
              + f"   beyond pool: {c['beyond_pool_walkers']:2d} walkers, "
                f"mass {c['beyond_pool_mass']:.4g} => lift >= {_lift_str(c['lift_bound'])}"
                f"   sum p_i={c['total_mass']:.2f}")
        rows.append({k: v for k, v in c.items() if k not in ("grid", "p_fc", "p_clim")})

    plot_lift(curves, Path(a.out) if a.out else None)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=float))
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
