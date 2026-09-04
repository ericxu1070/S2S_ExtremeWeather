#!/usr/bin/env python
"""The population as a density: every walker's ``A_L``, smoothed from 64 points to a curve.

`aires/awalkers.py` draws the raw material - one point per walker at (``A_L``, ``p_i``).
This figure draws the same population as a **probability density over** ``A_L``, so the
distribution the sampler produced reads as a PDF rather than as a scatter. Three curves
per run, each a Gaussian kernel density estimate (KDE):

- **AI+RES walkers, weighted by** ``p_i = Z w_i / N`` - the estimator's own reconstruction
  of the original forecast law. Its mass past the observation is (up to smoothing) the
  headline ``P(A_L >= obs)``. The masses are `aires.alift.walker_mass`, reached through
  `awalkers.run_record`; nothing is re-derived here.
- **AI+RES walkers, unweighted** - where the sampler actually put its walkers: the tilted
  population that resampling manufactured. It is NOT a probability; its mass past the
  observation is 0.4-0.7 on the reached events, because that is where the walkers were
  pushed. It is drawn so the tilt is visible against the weighted curve.
- **The direct-sampling ensemble** (the 24-member 0.25 deg GenCast cube by preference,
  `awalkers.DS_PREFERENCE`), every member worth ``1/n`` - the like-for-like baseline. It
  and the weighted curve estimate the SAME law, so laying one over the other is the reach
  comparison.

Two choices that a KDE forces, and why they went the way they did
-----------------------------------------------------------------
**One bandwidth for both walker curves.** The kernel width ``h`` is Scott's rule on the
64 unweighted positions, ``h = sd * N^(-1/5)`` (scipy's default, which
`gencast_s2s/plotting.py` uses), and the WEIGHTED curve uses that same ``h``. The weights
reweight the positions; they do not add samples, so how finely 64 points resolve ``A_L``
is the same for both curves. scipy's weighted default would instead size the kernel from
the Kish ESS of the weights - 2.2 of 64 for Uri, 1.0 for the persistence control - and
give ``h`` = 2.0 K and 3.7 K respectively, a blob wider than half the population.

**Every curve ends at its own most extreme member.** A Gaussian kernel is never zero, so
an unclipped curve carries mass past the observation even when nothing reached it:
measured on the real runs, the direct ensemble's KDE puts 0.035 (persistence control) and
0.0013 (Southwest) beyond the observation with 0 of 24 members there - the brown curve
would visually assert the reach the experiment denies. So each curve is drawn only inside
``[min, max]`` of its own sample, where the rug ends, and the kernel tail beyond that is
not drawn: it is kernel shape, not data. This is the KDE analogue of the binned figures'
rule that an empty bin breaks the curve rather than crawling the floor.

The weighted curve is self-normalised (integrates to 1 before clipping), so it and the
direct curve are on the same footing. ``--z-scaled`` multiplies it by ``sum_i p_i`` - the
run's normalization check, 8.75 for the persistence control - so that its area past the
observation is the estimator's own un-normalised ``P``. The panel title always prints the
EXACT subset-sum ``P`` from `awalkers.run_record`, never a KDE integral; ``main`` echoes
the KDE's tail mass beside it so the smoothing leakage (1.3-1.75x on Uri, Elliott and
SCentral) is on record rather than hidden.

    PYTHONPATH=. python -m aires.akde                       # slate over every reduced run
    PYTHONPATH=. python -m aires.akde --per-event           # + one figure per run
    PYTHONPATH=. python -m aires.akde --runs PNW_HeatDome_2021:pilot --z-scaled --bw 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import ndtr

from aires import aconfig as A
from aires.awalkers import (DIRECT_COLOR, DS_LABEL, OBS_COLOR, REACH_FILL, RES_COLOR,
                            _fmt_p, discover, metric_unit, order_runs, run_record)

# The y-axis floor, density per K. A floor only: it is never used to break a curve (for a
# KDE the floor crossing encodes the bandwidth, not where the sample ran out). It is
# lowered per slate, down to YMIN_HARD, when a walker of tiny mass would otherwise put
# part of a curve under the frame.
YMIN = 1e-4
YMIN_HARD = 1e-7
NGRID = 400


# --------------------------------------------------------------------------- #
# The KDE itself - pure numpy, tested without matplotlib
# --------------------------------------------------------------------------- #
def scott_bandwidth(x) -> float:
    """``sd(x, ddof=1) * n^(-1/5)``: scipy's ``gaussian_kde`` default in one dimension."""
    x = np.asarray(x, dtype="float64").ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError(f"need at least 2 finite points for a bandwidth, got {x.size}")
    sd = float(np.std(x, ddof=1))
    if not sd > 0:
        raise ValueError("zero-variance sample: no bandwidth can be inferred, pass bw=")
    return sd * x.size ** (-0.2)


def _normalised_weights(x: np.ndarray, weights) -> np.ndarray:
    if weights is None:
        return np.full(x.size, 1.0 / x.size)
    w = np.asarray(weights, dtype="float64").ravel()
    if w.size != x.size:
        raise ValueError(f"{w.size} weights for {x.size} points")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    s = float(w.sum())
    if not s > 0:
        raise ValueError("weights sum to zero")
    return w / s


def _bandwidth(x: np.ndarray, bw) -> float:
    h = scott_bandwidth(x) if bw is None else float(bw)
    if not (np.isfinite(h) and h > 0):
        raise ValueError(f"bandwidth must be positive and finite, got {h}")
    return h


def kde(x, grid, weights=None, bw=None) -> np.ndarray:
    """Gaussian KDE of ``x`` on ``grid``, with an ABSOLUTE bandwidth ``bw`` in x units.

    ``weights`` (one per point) are normalised to sum 1, so the result integrates to 1
    whatever they are; ``weights=None`` is ``scipy.stats.gaussian_kde(x)(grid)`` exactly.
    ``bw=None`` is Scott's rule on the points, weighted or not - see the module docstring
    for why the weights are kept out of the bandwidth.
    """
    x = np.asarray(x, dtype="float64").ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite sample value")
    g = np.asarray(grid, dtype="float64")
    w = _normalised_weights(x, weights)
    h = _bandwidth(x, bw)
    z = (g[..., None] - x) / h
    return (w * np.exp(-0.5 * z * z)).sum(axis=-1) / (h * np.sqrt(2.0 * np.pi))


def tail_mass(x, weights, bw, threshold: float, sign: float = 1.0) -> float:
    """EXACT mass of the (unclipped) KDE past ``threshold`` on the tail side ``sign``:
    ``sum_i w_i Phi(sign (x_i - threshold) / h)``. No grid, so as ``bw -> 0`` this is the
    subset sum the estimator reports, and ``tail(+1) + tail(-1) == 1`` for any ``bw``."""
    x = np.asarray(x, dtype="float64").ravel()
    w = _normalised_weights(x, weights)
    h = _bandwidth(x, bw)
    return float(np.sum(w * ndtr(float(sign) * (x - float(threshold)) / h)))


def support_mask(grid, x) -> np.ndarray:
    """Grid points inside ``[min(x), max(x)]`` - where a curve of ``x`` may be drawn."""
    x = np.asarray(x, dtype="float64")
    g = np.asarray(grid, dtype="float64")
    return (g >= float(x.min())) & (g <= float(x.max()))


# --------------------------------------------------------------------------- #
# From a run record to the curves of one panel
# --------------------------------------------------------------------------- #
def panel_xlim(r: dict, pad: float = 0.06) -> tuple[float, float]:
    """Walkers + direct members + the observation, padded. Same rule as
    `awalkers.plot_walkers`: autoscale does not pad for an axvline, and on the runs where
    the observation lies beyond every walker the red line would land on the frame."""
    al = r["table"]["A_L"].to_numpy(dtype="float64")
    xs = [float(al.min()), float(al.max())]
    if r["direct"].size:
        xs += [float(r["direct"].min()), float(r["direct"].max())]
    if r["observed"] is not None:
        xs.append(float(r["observed"]))
    lo, hi = min(xs), max(xs)
    p = pad * max(hi - lo, 1e-6)
    return lo - p, hi + p


def panel_curves(r: dict, bw: float | None = None, z_scaled: bool = False,
                 ngrid: int = NGRID) -> list[dict]:
    """The curves of one panel, in draw order (the weighted curve last, on top).

    Each entry: ``key``, ``label``, ``style``, ``h`` (the bandwidth used), ``x`` (the
    sample), ``weights``, ``grid``, ``density`` (NaN outside the sample's own support),
    ``scale`` (1, or ``sum p_i`` under ``z_scaled``) and ``tail`` (the exact unclipped
    KDE mass past the observation, times ``scale``; None without an observation).
    """
    t = r["table"]
    al = t["A_L"].to_numpy(dtype="float64")
    p = t["p_i"].to_numpy(dtype="float64")
    sign, obs = float(r["sign"]), r["observed"]
    grid = np.linspace(*panel_xlim(r), ngrid)
    h_w = _bandwidth(al, bw)

    def curve(key, x, weights, h, label, style, scale=1.0):
        dens = kde(x, grid, weights, h) * scale
        dens = np.where(support_mask(grid, x), dens, np.nan)
        tail = None if obs is None else tail_mass(x, weights, h, float(obs), sign) * scale
        return dict(key=key, label=label, style=style, h=h, x=x, weights=weights,
                    grid=grid, density=dens, scale=scale, tail=tail)

    out = [curve(
        "aires_unweighted", al, None, h_w,
        f"AI+RES walkers, unweighted (N={r['n']}): where the sampler placed them - "
        "the tilted population, not a probability",
        dict(color=RES_COLOR, ls=(0, (1, 1.4)), lw=2.0, zorder=4))]
    if r["direct"].size >= 2:
        d = np.asarray(r["direct"], dtype="float64")
        out.append(curve(
            "direct", d, None, _bandwidth(d, bw),
            f"{DS_LABEL[r['direct_key']]} ({d.size} members, each 1/{d.size})",
            dict(color=DIRECT_COLOR, ls=(0, (5, 2)), lw=2.0, zorder=3)))
    out.append(curve(
        "aires_weighted", al, p, h_w,
        r"AI+RES walkers weighted by $p_i = Z\,w_i/N$ (N="
        f"{r['n']}, weight ESS {r['weight_ess']:.1f}): the estimator's PDF",
        dict(color=RES_COLOR, ls="-", lw=2.6, zorder=7),
        scale=float(r["total_mass"]) if z_scaled else 1.0))
    return out


def _slate_ylim(curve_sets: list[list[dict]]) -> tuple[float, float]:
    """One y range for every panel: half-decade above the highest density, and a floor
    lowered (never below ``YMIN_HARD``) when a curve dips under ``YMIN`` inside its
    support - otherwise the frame hides part of a curve the caption says is drawn."""
    vals = np.concatenate([c["density"][np.isfinite(c["density"])]
                           for cs in curve_sets for c in cs])
    vals = vals[vals > 0]
    hi = 10 ** (np.ceil(2.0 * np.log10(float(vals.max()))) / 2.0)
    lo = min(YMIN, 10 ** np.floor(np.log10(float(vals.min()))))
    return max(lo, YMIN_HARD), float(hi)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _title(r: dict, unit: str) -> str:
    """The same title `awalkers.plot_walkers` gives the run, so the two slates read
    side by side: the exact subset-sum P, never anything the KDE produced."""
    obs, depth = r["observed"], r["sigma_depth"]
    head = (f"P = {_fmt_p(r['P'])}  PINNED" if r["pinned"] else
            (f"P = 0;  resolved to {r['deepest']:.1e}" if not r["n_reached"]
             else f"P = {_fmt_p(r['P'])}"))
    return (f"{r['label']}" + ("" if r["tag"] == "pilot" else f"  [{r['tag']}]")
            + f"   ({r['box']} box)\n"
            + (f"obs {obs:+.2f} {unit}" if obs is not None else "no observation")
            + (f", {depth:+.1f}$\\sigma$" if np.isfinite(depth) else "")
            + f"   |   {r['n_reached']}/{r['n']} reach it   |   {head}")


def _panel(ax, r: dict, curves: list[dict], ylim: tuple[float, float], *,
           z_scaled: bool, compact: bool = True):
    import matplotlib.patheffects as pe

    t, sign, obs = r["table"], float(r["sign"]), r["observed"]
    al = t["A_L"].to_numpy(dtype="float64")
    unit = metric_unit(r["metric"])
    lo_x, hi_x = panel_xlim(r)
    fs = 8.0 if compact else 10.0

    # --- the discrete data the curves were smoothed from: a rug at the frame's foot --- #
    # Axes-fraction y (`get_xaxis_transform`), never a data y on a log axis.
    tr = ax.get_xaxis_transform()
    if r["direct"].size:
        ax.plot(r["direct"], np.full(r["direct"].size, 0.062), ls="none", marker="|",
                ms=8 if compact else 11, mew=1.2, color=DIRECT_COLOR, alpha=0.9,
                transform=tr, zorder=5, clip_on=False)
    ax.plot(al, np.full(al.size, 0.022), ls="none", marker="|", ms=8 if compact else 11,
            mew=1.0, color=RES_COLOR, alpha=0.9, transform=tr, zorder=5, clip_on=False)

    # --- the curves, each ending at its own most extreme member -------------------- #
    for c in curves:
        st = dict(c["style"])
        if c["key"] == "aires_weighted":
            st["path_effects"] = [pe.Stroke(linewidth=st["lw"] + 1.8, foreground="white"),
                                  pe.Normal()]
        ax.plot(c["grid"], c["density"], label=c["label"], **st)

    if obs is not None:
        ax.axvline(float(obs), color=OBS_COLOR, lw=1.5, ls="--", zorder=8)
        # The region that COUNTS, from the observation to the far edge of the axis in the
        # tail direction (`awalkers.plot_walkers`) - whether anything is in it is the
        # question the panel asks.
        far = hi_x if sign > 0 else lo_x
        ax.axvspan(min(float(obs), far), max(float(obs), far), color=REACH_FILL,
                   alpha=0.07, lw=0, zorder=0)

    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlim(lo_x, hi_x)
    if sign < 0:
        ax.invert_xaxis()          # house rule: further right is always more extreme
    ax.grid(True, which="both", ls="--", color="0.85", lw=0.5)
    ax.tick_params(labelsize=fs)
    ax.set_title(_title(r, unit), fontsize=8.8 if compact else 11.0)
    ax.set_xlabel(rf"walker anomaly $A_L$   ({unit})", fontsize=fs + 0.5)

    by_key = {c["key"]: c for c in curves}
    hw = by_key["aires_weighted"]["h"]
    hd = by_key["direct"]["h"] if "direct" in by_key else None
    # Top LEFT: the bulk of every population sits at the top of a log axis and the tail
    # runs to the right, so the top-right corner is where the unweighted curve peaks on
    # a reached event; above the direct curve's near end, nothing is drawn.
    ax.text(0.025, 0.965,
            (f"$h$ = {hw:.2f} {unit} (walkers)"
             + (f", {hd:.2f} {unit} (direct)" if hd is not None else "") + "\n"
             + (r"weighted curve $\times\,\Sigma p_i$" if z_scaled
                else "weighted curve self-normalised") + "\n"
             f"$\\Sigma p_i$ = {r['total_mass']:.3f}   weight ESS "
             f"{r['weight_ess']:.1f}/{r['n']}"),
            transform=ax.transAxes, fontsize=fs - 0.8, color="0.30", va="top",
            ha="left", linespacing=1.35,
            bbox=dict(fc="white", ec="0.8", lw=0.5, alpha=0.85, pad=2.2))


def _legend_handles(z_scaled: bool):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [
        Line2D([], [], color=RES_COLOR, ls="-", lw=2.6,
               label=r"AI+RES walkers weighted by $p_i = Z\,w_i/N$ - the estimator's PDF"
                     + (r" ($\times\,\Sigma p_i$)" if z_scaled else "")),
        Line2D([], [], color=RES_COLOR, ls=(0, (1, 1.4)), lw=2.0,
               label="AI+RES walkers unweighted - where the sampler put them, not a "
                     "probability"),
        Line2D([], [], color=DIRECT_COLOR, ls=(0, (5, 2)), lw=2.0,
               label="direct-sampling ensemble, each member 1/n"),
        Line2D([], [], marker="|", ls="none", color=RES_COLOR, ms=9, mew=1.2,
               label="one walker (rug); brown rug = direct members"),
        Line2D([], [], color=OBS_COLOR, ls="--", lw=1.5, label="ERA5 observed $A_L$"),
        Patch(fc=REACH_FILL, alpha=0.18, ec="none",
              label=r"the region whose mass is $P(A_L \geq {\rm obs})$"),
    ]


def plot_kde(runs: list[dict], out: Path | None = None, ncols: int = 4,
             bw: float | None = None, z_scaled: bool = False) -> Path:
    """The slate: one panel per run, hardest target first, one y range throughout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = order_runs(runs)
    n = len(runs)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    curve_sets = [panel_curves(r, bw=bw, z_scaled=z_scaled) for r in runs]
    ylim = _slate_ylim(curve_sets)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.55 * ncols, 3.55 * nrows),
                             squeeze=False)
    for ax, r, cs in zip(axes.ravel(), runs, curve_sets):
        _panel(ax, r, cs, ylim, z_scaled=z_scaled, compact=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for i, ax in enumerate(axes.ravel()[:n]):
        if i % ncols == 0:
            ax.set_ylabel("density  (per K)", fontsize=9)

    fig.legend(handles=_legend_handles(z_scaled), fontsize=8.6, loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.016), columnspacing=1.6,
               handletextpad=0.6)
    fig.suptitle("AI+RES: the walker population as a density over $A_L$\n"
                 "Gaussian KDE of the same 64 walkers per run as aires_walkers.png; the "
                 "weighted curve is the estimator's PDF, the dotted one is where the "
                 "sampler put its walkers, the dashed one is direct sampling",
                 fontsize=12.5, y=1.0)
    fig.text(0.5, -0.062,
             "Every curve ENDS at its own most extreme member, where its rug ends: a "
             "Gaussian kernel is never zero, and drawn past the sample it would put mass "
             "beyond the observation on the two runs where nothing reached it\n(Southwest, "
             "the persistence control). Both walker curves share one bandwidth, Scott's "
             "rule on the 64 positions; the weights reweight those positions, they do "
             "not add samples. Each title's P is the exact subset\nsum of $p_i$ past the "
             "red line, never a KDE integral. Panels ordered by how far the observation "
             "sat outside the direct ensemble's spread, hardest first.",
             ha="center", va="top", fontsize=8.4, color="0.35", linespacing=1.55)
    fig.tight_layout(rect=[0, 0.012, 1, 0.965])

    out = out or (A.fig_dir() / "aires_kde.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return out


def plot_kde_run(r: dict, out: Path | None = None, bw: float | None = None,
                 z_scaled: bool = False) -> Path:
    """One run alone, with the legend on the axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs = panel_curves(r, bw=bw, z_scaled=z_scaled)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    _panel(ax, r, cs, _slate_ylim([cs]), z_scaled=z_scaled, compact=False)
    ax.set_ylabel("density  (per K)", fontsize=10.5)
    # Below the axes, not on them: `loc="best"` only avoids data artists and lands on
    # the rug or the corner text depending on the run.
    fig.legend(handles=_legend_handles(z_scaled), fontsize=8.6, loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.035), columnspacing=1.6,
               handletextpad=0.6)
    fig.text(0.5, 0.006,
             "Curves end at their own most extreme member (the kernel tail beyond it is "
             "shape, not data). Both walker curves share one bandwidth, Scott's rule on "
             "the unweighted positions.",
             ha="center", va="bottom", fontsize=8.4, color="0.35")
    fig.tight_layout(rect=[0, 0.145, 1, 1])

    out = out or (A.fig_dir(r["event"]) / f"aires_kde_{r['event']}_{r['tag']}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None,
                    help="comma-separated event:tag; default is every reduced run")
    ap.add_argument("--out", default=None, help="path for the cross-event figure")
    ap.add_argument("--per-event", action="store_true",
                    help="also write one figure per run")
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--bw", type=float, default=None,
                    help="one absolute kernel width (K) for every curve; default Scott")
    ap.add_argument("--z-scaled", action="store_true",
                    help="scale the weighted curve by sum p_i (the normalization check)")
    a = ap.parse_args(argv)

    runs = ([tuple(x.split(":", 1)) for x in a.runs.split(",")] if a.runs else discover())
    if not runs:
        raise SystemExit("no reduced runs found; run `--stage compare` first")

    recs = []
    for ev, tag in runs:
        try:
            r = run_record(ev, tag)
        except FileNotFoundError as e:
            print(f"  skip {ev}:{tag} - no {e}")
            continue
        recs.append(r)
        cs = {c["key"]: c for c in panel_curves(r, bw=a.bw, z_scaled=a.z_scaled)}
        w = cs["aires_weighted"]
        # The KDE's tail beside the exact subset sum: the gap between them is the
        # smoothing, and it is printed so nobody has to guess how large it is.
        exact = r["P"] / r["total_mass"] if r["total_mass"] > 0 else float("nan")
        print(f"  {ev:26s} {tag:8s} N={r['n']:3d} h={w['h']:.3f}  obs={r['observed']:+8.3f}"
              f"  P={_fmt_p(r['P']):>9s}  sum p_i={r['total_mass']:.4f}  "
              f"kde tail={w['tail']:.4f} vs exact "
              f"{'P' if a.z_scaled else 'P/sum p_i'}="
              f"{(r['P'] if a.z_scaled else exact):.4f}"
              + ("  PINNED" if r["pinned"] else ""))

    plot_kde(recs, Path(a.out) if a.out else None, ncols=a.ncols, bw=a.bw,
             z_scaled=a.z_scaled)
    if a.per_event:
        for r in recs:
            plot_kde_run(r, bw=a.bw, z_scaled=a.z_scaled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
