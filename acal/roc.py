#!/usr/bin/env python
"""ROC curves of AI+RES against the CFSv2 lagged baseline, within same-direction pools. CPU ONLY.

`analyze.py` and `cfsbase.py` score how much probability each forecast put on what
happened. This module asks the discrimination question instead: within a pool of cases,
did the forecast give MORE probability to the cases that went on to cross a deeper
threshold than to the ones that did not? That is a ROC curve, and its area is the
probability that a randomly drawn hit was ranked above a randomly drawn miss.

Why same-direction pools, and why no 2 K curve
----------------------------------------------
Two facts about the slate decide which ROC curves are fair:

  * every case was selected for |A_L| >= 2 K, so at 2 K every case is a hit, there are no
    misses and the ROC is undefined;
  * each AI+RES run was steered toward the tail its event reached (`run.json tail_sign`
    = sign of the observed anomaly). A warm-tail probability from a cold-steered run, or
    the reverse, would carry the observed sign into the forecast.

So the warm tail (+3, +4 K) is scored on the heat cases only (31) and the cold tail
(-3, -4 K) on the cold cases only (11). Within a pool every AI+RES run had the same
configuration (C schedule, N = 32, leads, seeds; only `tail_sign` differs between pools),
so nothing about the case's magnitude reached the forecast. The question each curve
answers is "of the cases that reached 2 K in this direction, which went on to 3 K (4 K)?",
which is a conditional discrimination question, not a statement about false alarms in
ordinary weeks.

The four forecasts (per case, threshold a, tail sign s; outcome o = 1[s obs >= |a|])
-----------------------------------------------------------------------------------
    AI+RES          self-normalized P = sum_i w_i 1[s A_i >= s a] / sum_i w_i over the 32
                    final walkers, = `analyze.Case.p_sn` (the `>=` rule of `stage_compare`)
    CFSv2 raw       mean_j 1[s A_j >= s a] over the 16 lagged members = `cfsbase.cfs_prob`
                    on the `raw_emp` members
    CFSv2 corrected the same on the `corr_emp` members (minus the leave-one-year-out CONUS
                    bias of `bias.csv`, via `cfsbase.members`)
    climatology     `analyze.p_clim` over `analyze.clim_pool` (2021-2025 daily CONUS A_L,
                    +/-30 d of the peak anniversary minus the case's own +/-10 d)

The ROC
-------
Cases are sorted by forecast probability, descending. Exact ties form ONE vertex, so a
group of tied cases is a straight segment, not a staircase whose shape depends on the
sort order. The area is the trapezoid rule over those vertices, which equals the
Mann-Whitney statistic with ties counted 0.5 (tested). With 16 members and 32 weighted
walkers many cases sit at P = 0, so the last segment is often long.

Uncertainty: a stratified, PAIRED case bootstrap. Each replicate resamples the hits and
the misses separately with replacement (so the counts stay fixed and every replicate has a
defined ROC) and uses the SAME resampled case indices for every forecast, so the AUC
differences are paired. 2000 replicates, fixed seed. Bands are the 5-95% hit rate on a
101-point false-alarm grid (the curve's upper envelope at a vertical segment); AUC and
AUC-difference intervals are 90% percentile intervals. With 2 to 6 hits per curve these
intervals are wide, and they are still optimistic: cases in one winter share a regime.

    python -m acal.roc        # runs/acal/analysis/roc_{cases,auc}.csv, figures/acal/roc/*.png
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import cfsbase as CB
from acal import ccfg

CASES_OUT = AN.OUT / "roc_cases.csv"
AUC_OUT = AN.OUT / "roc_auc.csv"
FIG_DIR = ccfg.FIG_ROOT / "roc"

# (pool, family, tail sign, |thresholds| in K). No 2 K: every case crossed it.
POOLS = (("warm", "heat", 1.0, (3.0, 4.0)), ("cold", "cold", -1.0, (3.0, 4.0)))
MODELS = ("ai_res", "cfs_corr", "cfs_raw", "clim", "ai_res_raw")   # raw: table only, not plotted
DIFFS = (("ai_res", "cfs_corr"), ("ai_res", "cfs_raw"))
N_BOOT = 2000
BOOT_SEED = 20261006
CI = 0.90
FAR_GRID = np.linspace(0.0, 1.0, 101)
CHECK_TOL = 5e-6          # published CSVs are written with 6 significant digits
MEAN_TOL = 1e-9           # K; member mean vs build.csv al_mean (same float64 numbers)


# --------------------------------------------------------------------------- #
# ROC and AUC (pure, tested)
# --------------------------------------------------------------------------- #
def roc_curve(p, o) -> tuple[np.ndarray, np.ndarray]:
    """(far, hr) vertices from (0, 0) to (1, 1); exact ties in `p` form one vertex."""
    p, o = np.asarray(p, dtype=float), np.asarray(o, dtype=bool)
    n_hit, n_miss = int(o.sum()), int((~o).sum())
    if n_hit == 0 or n_miss == 0:
        raise ValueError(f"ROC undefined with {n_hit} hits and {n_miss} misses")
    order = np.argsort(-p, kind="stable")
    ps, os_ = p[order], o[order]
    tp, fp = np.cumsum(os_), np.cumsum(~os_)
    last = np.r_[ps[1:] != ps[:-1], True]           # the last case of each tie group
    far = np.r_[0.0, fp[last] / n_miss]
    hr = np.r_[0.0, tp[last] / n_hit]
    return far, hr


def auc(p, o) -> float:
    far, hr = roc_curve(p, o)
    return float(np.sum(np.diff(far) * (hr[1:] + hr[:-1]) / 2.0))


def auc_mann_whitney(p, o) -> float:
    """P(p_hit > p_miss) + 0.5 P(p_hit == p_miss), over all hit-miss pairs."""
    p, o = np.asarray(p, dtype=float), np.asarray(o, dtype=bool)
    d = p[o][:, None] - p[~o][None, :]
    return float(np.mean((d > 0) + 0.5 * (d == 0)))


def hr_on_grid(far, hr, grid=FAR_GRID) -> np.ndarray:
    """Hit rate on a fixed false-alarm grid, linear between vertices.

    A vertical segment (several hits tied above the next miss) is collapsed to its top,
    so the value at that false-alarm rate is the curve's upper envelope.
    """
    u, inv = np.unique(far, return_inverse=True)
    top = np.full(u.size, -np.inf)
    np.maximum.at(top, inv, hr)
    return np.interp(grid, u, top)


def boot_indices(o, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot, n) case indices; hits and misses resampled separately, counts kept."""
    o = np.asarray(o, dtype=bool)
    hits, misses = np.flatnonzero(o), np.flatnonzero(~o)
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(hits, (n_boot, hits.size)),
                           rng.choice(misses, (n_boot, misses.size))], axis=1)


def paired_boot(probs: dict, o, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Per model: (n_boot,) AUCs and (n_boot, grid) hit rates, from SHARED indices."""
    o = np.asarray(o, dtype=bool)
    idx = boot_indices(o, n_boot, seed)
    out = {}
    for m, p in probs.items():
        p = np.asarray(p, dtype=float)
        a = np.empty(n_boot)
        h = np.empty((n_boot, FAR_GRID.size))
        for b in range(n_boot):
            far, hr = roc_curve(p[idx[b]], o[idx[b]])
            a[b] = np.sum(np.diff(far) * (hr[1:] + hr[:-1]) / 2.0)
            h[b] = hr_on_grid(far, hr)
        out[m] = dict(auc=a, hr=h)
    return out


def _ci(x, level: float = CI) -> tuple[float, float]:
    lo, hi = np.percentile(x, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Per-case probabilities, cross-checked against the published numbers
# --------------------------------------------------------------------------- #
def p_ai_res(case: AN.Case, a: float) -> float:
    """Self-normalized `sum(w 1[s A >= s a]) / sum(w)` - written out, equal to `p_sn`."""
    w = case.weights
    return float(np.sum(w * (case.sign * (case.al - a) >= 0.0)) / np.sum(w))


def case_rows() -> pd.DataFrame:
    """One row per (case, pool threshold): outcome and the four forecasts."""
    df, cases = AN.load_all()
    daily = AN.load_daily()
    cfs = CB.load_cfs().set_index("episode_id")
    rows = []
    for r, c in zip(df.itertuples(), cases):
        rec = cfs.loc[r.episode_id]
        pool = AN.clim_pool(daily, r.peak)
        raw = CB.members(rec["al"], rec["bias"], "raw_emp")
        corr = CB.members(rec["al"], rec["bias"], "corr_emp")
        for name, fam, sign, ths in POOLS:
            if r.family != fam:
                continue
            for k in ths:
                a = sign * k
                pc, kc, nc = AN.p_clim(pool, a, sign)
                p_res = p_ai_res(c, a)
                if abs(p_res - c.p_sn(a)) > 1e-12:
                    raise SystemExit(f"[roc] {r.episode_id}: written-out sn P != Case.p_sn")
                rows.append(dict(
                    episode_id=r.episode_id, family=r.family, rung=int(r.rung), peak=r.peak,
                    pool=name, threshold_K=a, obs=c.obs, outcome=int(sign * c.obs >= k),
                    p_ai_res=p_res, p_ai_res_raw=c.p_raw(a),
                    p_cfs_corr=CB.cfs_prob(corr, a, sign, "emp"),
                    p_cfs_raw=CB.cfs_prob(raw, a, sign, "emp"), p_clim=pc,
                    n_walkers_beyond=c.n_beyond(a), normalization_check=c.norm,
                    n_cfs_raw_beyond=int(np.sum(sign * (raw - a) >= 0.0)),
                    n_cfs_corr_beyond=int(np.sum(sign * (corr - a) >= 0.0)),
                    cfs_bias_conus=float(rec["bias"]), cfs_mean_raw=float(np.mean(raw)),
                    k_clim=kc, n_clim=nc))
    return pd.DataFrame(rows)


def checks(rc: pd.DataFrame) -> list[str]:
    """Reproduce numbers already published by analyze/cfsbase; HARD-FAIL on a mismatch."""
    msgs = []
    # AI+RES and climatology vs rungs_cases.csv (F_a_sn = sn P at a; k_clim_a / n pool)
    ru = pd.read_csv(AN.RUNGS_CASES_OUT)
    ru = ru[ru.spec.isin(["3|2", "4|2"])]
    m = rc.assign(a=rc.threshold_K.abs()).merge(ru, on=["episode_id", "a"],
                                                suffixes=("", "_ru"))
    if len(m) != len(rc):
        raise SystemExit(f"[roc] rungs_cases join: {len(m)} rows, want {len(rc)}")
    d = np.abs(m.p_ai_res - m.F_a_sn).max()
    own = ~m.clim_fallback.astype(bool)
    dc = np.abs(m.p_clim[own] - m.k_clim_a[own] / m.n_clim_pool[own]).max()
    if d > CHECK_TOL or dc > 1e-12 or (m.outcome != m.outcome_ru).any():
        raise SystemExit(f"[roc] rungs_cases mismatch: P {d:.2e}, clim {dc:.2e}")
    msgs.append(f"AI+RES sn P vs rungs_cases.csv F_a_sn: max diff {d:.1e} ({len(m)} rows); "
                f"P_clim vs k_clim_a/n_clim_pool: {dc:.1e}; outcomes identical")
    # CFS raw/corrected vs cfs_paired.csv: |p - o| == sqrt(Brier)
    pc = pd.read_csv(CB.PAIRED_CSV).set_index("episode_id")
    worst = 0.0
    for r in rc.itertuples():
        k = abs(r.threshold_K)
        for v, col in (("raw_emp", "p_cfs_raw"), ("corr_emp", "p_cfs_corr")):
            b = pc.loc[r.episode_id, f"brier_{v}_{k:g}K"]
            worst = max(worst, abs(abs(getattr(r, col) - r.outcome) - np.sqrt(b)))
    if worst > CHECK_TOL:
        raise SystemExit(f"[roc] CFS P vs cfs_paired.csv Brier: {worst:.2e}")
    msgs.append(f"CFS raw/corr P vs cfs_paired.csv Brier: max diff {worst:.1e}")
    # CFS member mean vs build.csv al_mean
    bc = pd.read_csv(CB.BUILD_CSV).set_index("episode_id")
    dm = np.abs(rc.cfs_mean_raw.values - bc.loc[rc.episode_id, "al_mean"].values).max()
    if dm > MEAN_TOL or (bc.loc[rc.episode_id, "n_members"] != CB.N_CYCLES).any():
        raise SystemExit(f"[roc] CFS member mean vs build.csv al_mean: {dm:.2e} K")
    msgs.append(f"CFS 16-member mean vs build.csv al_mean: max diff {dm:.1e} K")
    for line in msgs:
        print(f"  check ok: {line}")
    return msgs


# --------------------------------------------------------------------------- #
# AUC table
# --------------------------------------------------------------------------- #
PCOL = {m: f"p_{m}" for m in MODELS}


def analyse(rc: pd.DataFrame, models=MODELS, diffs=DIFFS,
            skip_degenerate: bool = False) -> tuple[pd.DataFrame, dict]:
    """AUC + paired-bootstrap CIs per (pool, threshold, model) and the AUC differences.

    `models` / `diffs` default to the published CFS set; `models_for(source)` gives
    another registry source's. `skip_degenerate` drops a (pool, threshold) with no hits
    or no misses (a partial slate) instead of failing.
    """
    MODELS, DIFFS = models, diffs                    # noqa: N806 - shadow the defaults
    PCOL = {m: f"p_{m}" for m in MODELS}             # noqa: N806
    rows, curves = [], {}
    for (pool, a), d in rc.groupby(["pool", "threshold_K"], sort=False):
        o = d.outcome.values.astype(bool)
        n_hit, n_miss = int(o.sum()), int((~o).sum())
        if skip_degenerate and (n_hit == 0 or n_miss == 0):
            print(f"[roc] {pool} {a:+.0f} K: {n_hit} hits / {n_miss} misses, skipped")
            continue
        probs = {m: d[PCOL[m]].values for m in MODELS}
        bt = paired_boot(probs, o)
        base = dict(pool=pool, threshold_K=a, n_cases=len(d), n_hits=n_hit, n_misses=n_miss)
        cur = {}
        for m in MODELS:
            far, hr = roc_curve(probs[m], o)
            A, mw = auc(probs[m], o), auc_mann_whitney(probs[m], o)
            if abs(A - mw) > 1e-12:
                raise SystemExit(f"[roc] {pool} {a}: trapezoid {A} != Mann-Whitney {mw}")
            lo, hi = _ci(bt[m]["auc"])
            rows.append(dict(**base, quantity="auc", model=m, value=A, ci_lo=lo, ci_hi=hi,
                             boot_frac_positive=np.nan, n_boot=N_BOOT, seed=BOOT_SEED))
            band = np.percentile(bt[m]["hr"], [50 * (1 - CI), 100 - 50 * (1 - CI)], axis=0)
            cur[m] = dict(far=far, hr=hr, auc=A, lo=lo, hi=hi, band=band)
        for x, y in DIFFS:
            dd = bt[x]["auc"] - bt[y]["auc"]
            lo, hi = _ci(dd)
            rows.append(dict(**base, quantity="auc_diff", model=f"{x}_minus_{y}",
                             value=cur[x]["auc"] - cur[y]["auc"], ci_lo=lo, ci_hi=hi,
                             boot_frac_positive=float(np.mean(dd > 0)), n_boot=N_BOOT,
                             seed=BOOT_SEED))
            cur[f"{x}_minus_{y}"] = dict(value=cur[x]["auc"] - cur[y]["auc"], lo=lo, hi=hi)
        curves[(pool, a)] = dict(n_hit=n_hit, n_miss=n_miss, n=len(d), **cur)
    return pd.DataFrame(rows), curves


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
# Model colours (individual figures): black for the headline forecast, then two Okabe-Ito
# hues that stay apart under the common colour-vision deficiencies and avoid the red/blue
# that means warm/cold elsewhere in figures/acal. Line style carries the model as well.
MODEL_STYLE = {
    "ai_res":   dict(label="AI+RES", color="#000000", ls="-", lw=3.0, marker="o"),
    "cfs_corr": dict(label="CFSv2 bias-corrected", color="#009E73", ls="--", lw=2.4,
                     marker="s"),
    "cfs_raw":  dict(label="CFSv2 raw", color="#CC79A7", ls=":", lw=2.6, marker="^"),
    "clim":     dict(label="Climatology", color="#9C6B1F", ls=(0, (5, 3)), lw=1.6,
                     marker=None),
}
# Overlay: colour = threshold (darker = more extreme), within each pool's family.
THRESH_COLOR = {3.0: "#E69F00", 4.0: "#A63A00", -3.0: "#56B4E9", -4.0: "#08427A"}
# Per-source overlays: a warm/cold threshold ramp that reuses no MODEL_STYLE hex (the
# published ramp above coincides with ECCC GEPS orange and BB-SUBS sky blue, so on the
# board it would read as model identity). The published figure keeps THRESH_COLOR.
THRESH_COLOR_BOARD = {3.0: "#FA8072", 4.0: "#99000D", -3.0: "#7B7FD4", -4.0: "#081D58"}
FOOTNOTE = ("21-day lead; CONUS week-mean T2m anomaly; AI+RES 32 walkers (self-normalized), "
            "CFSv2 16 lagged members")
TILT_NOTE = "AI+RES was tilted toward the observed tail direction; CFSv2 was not."
SMALL_N_HITS = 3          # at or below: the bootstrap cannot show the real spread
NO_SKILL = dict(color="0.55", lw=1.0, ls=(0, (2, 2)))
POOL_TEXT = {"warm": ("Warm tail", "warm", "+"), "cold": ("Cold tail", "cold", "-")}

# What the figure functions draw: the published CFS set by default; `source_ctx(source)`
# swaps in any registry source (keys, labels, board colours, footnote, figure dir).
_PUBLISHED = dict(corr="cfs_corr", raw="cfs_raw", style=MODEL_STYLE, fc="CFSv2",
                  short="CFSv2", leg_fs=12, foot=FOOTNOTE, tilt=TILT_NOTE, dir=None)
SHORT = {"cfs13": "CFSv2", "ec46": "EC46", "gefs": "GEFSv12", "geps": "GEPS"}  # overlay legend
_CTX = dict(_PUBLISHED)


def source_ctx(source: str, has_bias: bool, truth: str, fig_dir: Path) -> dict:
    """Figure context of one registry source: AI+RES and the source in the board colours
    (`s2sbase.MODEL_STYLE`; raw = the source's hue darkened), climatology grey."""
    from acal import s2sbase as S2
    from acal import truth as TR
    src = S2.get(source)
    st = S2.MODEL_STYLE.get(source, {"label": src.label, "color": "#0072B2"})
    lab, col = st["label"], st["color"]
    corr, raw = f"{source}_corr", f"{source}_raw"
    style = {"ai_res": dict(label="AI+RES", color=S2.MODEL_STYLE["aires"]["color"], ls="-",
                            lw=3.0, marker="o")}
    if has_bias:
        style[corr] = dict(label=f"{lab} bias-corrected", color=col, ls="--", lw=2.4,
                           marker="s")
    style[raw] = dict(label=f"{lab} raw", color=S2.shade(col), ls=":", lw=2.6, marker="^")
    style["clim"] = dict(label="Climatology", color=S2.MODEL_STYLE["clim"]["color"],
                         ls=(0, (5, 3)), lw=1.6, marker=None)
    win = ("13 frames 00/12Z, peak-6d..peak" if src.obs_window == "13f" else
           "UTC days peak-6..peak-1; truth and AI+RES on 12 frames")
    mem, lead = S2.ens_text(source)      # per-case ranges from the json records
    foot = (f"Lead to peak: AI+RES 21 d, {lab} {lead}. CONUS week-mean T2m anomaly ({win})",
            f"Truth {TR.label_of(TR.get_truth(truth))}; AI+RES 32 walkers (self-normalized), "
            f"{lab} {mem}")
    tilt = f"AI+RES was tilted toward the observed tail direction; {lab} was not."
    return dict(corr=corr if has_bias else None, raw=raw, style=style, fc=lab,
                short=SHORT.get(source, lab), leg_fs=10.5, wrap=112, foot=foot, tilt=tilt,
                thresh_color=THRESH_COLOR_BOARD,
                dir=Path(fig_dir))


def _style(plt) -> None:
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "0.9", "grid.linewidth": 0.8,
                         "axes.axisbelow": True, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})


def _save(fig, name: str) -> Path:
    d = _CTX["dir"] or FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    fig.savefig(p, dpi=200, facecolor="white")
    print(f"  wrote {p} ({p.stat().st_size / 1e3:.0f} kB)")
    return p


def _axes(ax, label_size: int = 14) -> None:
    ax.plot([0, 1], [0, 1], **NO_SKILL, zorder=1)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("False alarm rate", fontsize=label_size)
    ax.set_ylabel("Hit rate", fontsize=label_size)


def _subtitle(pool: str, k: float, cv: dict, n_pool: int) -> str:
    _, word, sg = POOL_TEXT[pool]
    return (f"{n_pool} {word} cases, all observed {'>=' if sg == '+' else '<='} {sg}2 K; "
            f"question: which went on to {sg}{k:g} K")


def _lines(x) -> tuple:
    """A footnote entry as a tuple of lines (the published FOOTNOTE is one string)."""
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _foot(lines, width: int) -> str:
    """Footnote block: each item wrapped at `width` characters, one item per line group.
    An item longer than `width` is wrapped into balanced lines (same line count, lines of
    about equal length), so no single word is left alone on the last line."""
    out = []
    for t in lines:
        n = -(-len(t) // width)
        w = width if n <= 1 else min(width, -(-len(t) // n) + 8)
        txt = textwrap.fill(t, w, break_on_hyphens=False)
        if txt.count("\n") + 1 > n:                     # balancing cost a line: plain wrap
            txt = textwrap.fill(t, width, break_on_hyphens=False)
        out.append(txt)
    return "\n".join(out)


def fig_single(pool: str, a: float, cv: dict, plt) -> Path:
    """One threshold: the three forecasts with their 90% bands, plus climatology.

    Laid out in inches (title block, square axes, legend BELOW the axes so it can never
    cover a curve, footnote block), then converted to figure fractions.
    """
    from matplotlib.lines import Line2D
    k, (head, word, sg) = abs(a), POOL_TEXT[pool]
    W, side, left = 7.6, 4.9, 1.7
    top_h, xlab_h, leg_h, foot_h = 1.15, 0.62, 1.55, 0.92
    H = top_h + side + xlab_h + leg_h + foot_h
    fig = plt.figure(figsize=(W, H))
    y_ax = foot_h + leg_h + xlab_h
    ax = fig.add_axes([left / W, y_ax / H, side / W, side / H])
    _axes(ax)
    STY, corr, raw, fc = _CTX["style"], _CTX["corr"], _CTX["raw"], _CTX["fc"]  # noqa: N806
    for m in [k for k in ("clim", raw, corr, "ai_res") if k]:   # headline drawn last, on top
        st, c = STY[m], cv[m]
        if m != "clim":
            ax.fill_between(FAR_GRID, c["band"][0], c["band"][1], color=st["color"],
                            alpha=0.07 if m == "ai_res" else 0.10, lw=0, zorder=2)
        ax.plot(c["far"], c["hr"], color=st["color"], ls=st["ls"], lw=st["lw"],
                marker=st["marker"], ms=6, mfc=st["color"], mec="white", mew=0.8,
                zorder=4 if m == "ai_res" else 3, clip_on=False)
    handles = []
    for m in STY:
        st, c = STY[m], cv[m]
        handles.append(Line2D([], [], color=st["color"], ls=st["ls"], lw=st["lw"],
                              marker=st["marker"], ms=6, mfc=st["color"], mec="white",
                              label=f"{st['label']}  AUC {c['auc']:.2f} "
                                    f"[{c['lo']:.2f}, {c['hi']:.2f}]"))
    handles.append(Line2D([], [], **NO_SKILL, label="No skill (AUC 0.5)"))
    fig.legend(handles=handles, loc="upper center", fontsize=12, frameon=False,
               handlelength=3.0, ncol=1,
               bbox_to_anchor=(0.5 * (left + side / 2) / (W / 2), (foot_h + leg_h) / H))
    sym = ">=" if sg == "+" else "<="
    fig.text(0.5, 1 - 0.18 / H, f"{head}: did the week reach {sg}{k:g} K?", ha="center",
             va="top", fontsize=17, weight="bold")
    fig.text(0.5, 1 - 0.62 / H, _subtitle(pool, k, cv, cv["n"]) + "\n"
             f"{cv['n_hit']} hits ({sym} {sg}{k:g} K observed), {cv['n_miss']} misses"
             + (f"; only {cv['n_hit']} hits: intervals understate uncertainty"
                if cv["n_hit"] <= SMALL_N_HITS else ""),
             ha="center", va="top", fontsize=12.5, color="0.2", linespacing=1.35)
    d2 = cv[f"ai_res_minus_{raw}"]
    dtxt = f"raw {d2['value']:+.2f} [{d2['lo']:+.2f}, {d2['hi']:+.2f}]"
    if corr:
        d1 = cv[f"ai_res_minus_{corr}"]
        dtxt = (f"corrected {d1['value']:+.2f} [{d1['lo']:+.2f}, {d1['hi']:+.2f}], " + dtxt)
    foot = _foot([
        f"AUC difference, AI+RES minus {fc}: " + dtxt,
        f"Brackets and shading: 90% paired case bootstrap ({N_BOOT} resamples, hits and "
        "misses apart)",
        *_lines(_CTX["foot"]), _CTX["tilt"]], _CTX.get("wrap", 110))
    fig.text(0.5, 0.10 / H, foot, ha="center", va="bottom", fontsize=9.2, color="0.25",
             linespacing=1.35)
    p = _save(fig, f"roc_{pool}_{k:g}K.png")
    plt.close(fig)
    return p


def fig_overlay(curves: dict, plt) -> Path:
    """Both pools side by side, every threshold and forecast, no bands. Legends sit
    BELOW each panel (two columns: one per threshold) so they never cover a curve."""
    from matplotlib.lines import Line2D
    STY, corr, raw, fc = _CTX["style"], _CTX["corr"], _CTX["raw"], _CTX["fc"]  # noqa: N806
    W, side, gap = 13.6, 4.8, 2.6
    top_h, xlab_h, leg_h, foot_h = 1.70, 0.62, 1.05, 0.80
    H = top_h + side + xlab_h + leg_h + foot_h
    fig = plt.figure(figsize=(W, H))
    x0 = (W - 2 * side - gap) / 2 + 0.20            # a little extra room for the y label
    y_ax = foot_h + leg_h + xlab_h
    for i, (pool, fam, sign, ths) in enumerate(POOLS):
        head, word, sg = POOL_TEXT[pool]
        xl = x0 + i * (side + gap)
        ax = fig.add_axes([xl / W, y_ax / H, side / W, side / H])
        _axes(ax)
        handles = []
        for k in ths:
            a = sign * k
            cv, col = curves[(pool, a)], _CTX.get("thresh_color", THRESH_COLOR)[a]
            for m in [k for k in (raw, corr, "ai_res") if k]:
                st = STY[m]
                ax.plot(cv[m]["far"], cv[m]["hr"], color=col, ls=st["ls"], lw=st["lw"],
                        marker=st["marker"], ms=6, mfc=col, mec="white", mew=0.8,
                        zorder=4 if m == "ai_res" else 3, clip_on=False)
            for m in [k for k in ("ai_res", corr, raw) if k]:
                st = STY[m]
                sh = _CTX["short"]
                lab = {"ai_res": "AI+RES", corr: f"{sh} corr.", raw: f"{sh} raw"}[m]
                handles.append(Line2D([], [], color=col, ls=st["ls"], lw=st["lw"],
                                      marker=st["marker"], ms=6, mfc=col, mec="white",
                                      label=f"{sg}{k:g} K  {lab}  AUC {cv[m]['auc']:.2f}"))
        fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=_CTX["leg_fs"],
                   frameon=False, handlelength=2.4, columnspacing=1.2, handletextpad=0.5,
                   bbox_to_anchor=((xl + side / 2) / W, (foot_h + leg_h) / H))
        n = curves[(pool, sign * ths[0])]["n"]
        hm = "; ".join(f"{sg}{k:g} K: {curves[(pool, sign * k)]['n_hit']} hits, "
                       f"{curves[(pool, sign * k)]['n_miss']} misses" for k in ths)
        ax.text(0.5, 1.03, f"{n} {word} cases, all observed {'>=' if sg == '+' else '<='} "
                f"{sg}2 K\n{hm}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12.5, color="0.2", linespacing=1.3)
        ax.text(0.5, 1.145, head, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=16, weight="bold")
    fig.text(0.5, 1 - 0.15 / H, "Which cases went on to the deeper threshold? "
             f"ROC, AI+RES vs {fc}", ha="center", va="top", fontsize=18, weight="bold")
    ls_txt = (f"solid AI+RES, dashed {fc} bias-corrected, dotted {fc} raw" if corr else
              f"solid AI+RES, dotted {fc} raw")
    foot = "\n".join([
        f"Line style = forecast ({ls_txt}); colour = threshold, darker = more extreme; "
        "diagonal = no skill", *_lines(_CTX["foot"]), _CTX["tilt"]])
    fig.text(0.5, 0.10 / H, foot, ha="center", va="bottom", fontsize=10.5, color="0.25",
             linespacing=1.35)
    p = _save(fig, "roc_overlay.png")
    plt.close(fig)
    return p


def figures(curves: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    out = [fig_overlay(curves, plt)]
    for pool, _, sign, ths in POOLS:
        for k in ths:
            out.append(fig_single(pool, sign * k, curves[(pool, sign * k)], plt))
    return out


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Any registry source (acal.s2sbase): its own window, its own truth
# --------------------------------------------------------------------------- #
def models_for(source: str, has_bias: bool) -> tuple[tuple[str, ...], tuple]:
    """(MODELS, DIFFS) for one source, in the published order: for 'cfs' with a bias
    exactly `MODELS` / `DIFFS`."""
    fc = ((f"{source}_corr",) if has_bias else ()) + (f"{source}_raw",)
    return ("ai_res", *fc, "clim", "ai_res_raw"), tuple(("ai_res", m) for m in fc)


def source_rows(source: str, truth: str = "era5") -> pd.DataFrame:
    """`case_rows` for a registry source: outcome = truth on the source's window, AI+RES
    re-reduced on that window (`s2sbase.aires_cases`), one row per (case, pool threshold)."""
    from acal import s2sbase as S2
    from acal import truth as TR
    src, tr = S2.get(source), TR.get_truth(truth)
    recs = S2.load(src, tr)
    cases = S2.aires_cases(tr, src.obs_window)
    daily = TR.pool_series(tr)
    rows = []
    for _, r in recs.iterrows():
        c = cases[r["episode_id"]]
        if abs(c.obs - r["obs"]) > 1e-9:
            raise SystemExit(f"[roc] {r['episode_id']}: AI+RES obs != {source} obs")
        pool = AN.clim_pool(daily, r["peak"])
        raw = CB.members(r["al"], r["bias"], "raw_emp")
        corr = CB.members(r["al"], r["bias"], "corr_emp")
        for name, fam, sign, ths in POOLS:
            if r["family"] != fam:
                continue
            for k in ths:
                a = sign * k
                pc, kc, nc = AN.p_clim(pool, a, sign)
                rows.append({"source": source, "truth": tr.name, "window": src.obs_window,
                             "episode_id": r["episode_id"], "family": r["family"],
                             "rung": r["rung"], "peak": r["peak"], "pool": name,
                             "threshold_K": a, "obs": r["obs"],
                             "outcome": int(sign * r["obs"] >= k),
                             "p_ai_res": p_ai_res(c, a), "p_ai_res_raw": c.p_raw(a),
                             f"p_{source}_corr": (CB.cfs_prob(corr, a, sign, "emp")
                                                  if np.isfinite(r["bias"]) else np.nan),
                             f"p_{source}_raw": CB.cfs_prob(raw, a, sign, "emp"),
                             "p_clim": pc, "n_members": len(raw), "k_clim": kc, "n_clim": nc})
    return pd.DataFrame(rows)


def run_source(source: str, truth: str = "era5", figs: bool = False,
               fig_dir: Path | str | None = None) -> pd.DataFrame:
    """AUC table for one registry source -> runs/acal/analysis/s2s/<truth>/roc_<source>_*.csv.
    `figs` (or a `fig_dir`) also draws the ROC figure set for the source in the board
    colours, to `fig_dir` (default figures/acal/s2s/<truth>/<source>/roc/)."""
    from acal import s2sbase as S2
    rc = source_rows(source, truth)
    has_bias = bool(np.isfinite(rc[f"p_{source}_corr"]).all())
    models, diffs = models_for(source, has_bias)
    tab, curves = analyse(rc, models, diffs, skip_degenerate=True)
    tab.insert(0, "source", source)
    want = {(pool, sign * k) for pool, _, sign, ths in POOLS for k in ths}
    if source == "cfs" and truth != "era5" and want <= set(curves):
        # the published CFS figure set, against this truth, in its figure dir
        global FIG_DIR
        from acal import truth as TR
        keep, FIG_DIR = FIG_DIR, TR.fig_dir(TR.get_truth(truth)) / "roc"
        try:
            figures(curves)
        finally:
            FIG_DIR = keep
    if (figs or fig_dir is not None) and source != "cfs":
        if not want <= set(curves):
            print(f"[roc] {source} vs {truth}: a pool threshold is degenerate, no figures")
        else:
            d = Path(fig_dir) if fig_dir is not None else S2.fig_dir(source, truth) / "roc"
            _CTX.update(source_ctx(source, has_bias, truth, d))
            try:
                figures(curves)
            finally:
                _CTX.clear()
                _CTX.update(_PUBLISHED)
    od = S2.ANALYSIS / truth
    od.mkdir(parents=True, exist_ok=True)
    rc.to_csv(od / f"roc_{source}_cases.csv", index=False, float_format="%.6g")
    tab.to_csv(od / f"roc_{source}_auc.csv", index=False, float_format="%.6g")
    print(f"[roc] {source} vs {truth}: {rc.episode_id.nunique()} cases -> "
          f"{od / f'roc_{source}_auc.csv'}")
    return tab


def run() -> pd.DataFrame:
    rc = case_rows()
    n = rc.groupby("pool").episode_id.nunique().to_dict()
    print(f"[roc] pools: {n}  rows: {len(rc)}")
    if n != {"warm": 31, "cold": 11}:
        raise SystemExit(f"[roc] pool sizes {n}, expected warm 31 / cold 11")
    checks(rc)
    AN.OUT.mkdir(parents=True, exist_ok=True)
    rc.to_csv(CASES_OUT, index=False, float_format="%.6g")
    print(f"  wrote {CASES_OUT}")
    tab, curves = analyse(rc)
    tab.to_csv(AUC_OUT, index=False, float_format="%.6g")
    print(f"  wrote {AUC_OUT}")
    for r in tab.itertuples():
        extra = f"  P(diff>0) {r.boot_frac_positive:.3f}" if r.quantity == "auc_diff" else ""
        print(f"  {r.pool:4s} {r.threshold_K:+.0f} K  hits {r.n_hits:2d} misses "
              f"{r.n_misses:2d}  {r.model:22s} {r.value:+.3f} [{r.ci_lo:+.3f}, "
              f"{r.ci_hi:+.3f}]{extra}")
    figures(curves)
    return tab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", help="a registry source (acal.s2sbase); default = the "
                                     "published CFS ROC")
    ap.add_argument("--truth", default=None,
                    help="era5 | hrrr | hrrr_raw (default era5 with --source); a truth "
                         "without --source scores CFS against it into "
                         "runs/acal/analysis/s2s/<truth>/ - never the published files")
    ap.add_argument("--figures", action="store_true",
                    help="with --source (not cfs): draw the ROC figures to figures/acal/s2s/"
                         "<truth>/<source>/roc/ (or --out-dir)")
    ap.add_argument("--out-dir", default=None, help="figure dir for --figures")
    a = ap.parse_args(argv)
    if a.source or a.truth:
        run_source(a.source or "cfs", a.truth or "era5", a.figures, a.out_dir)
    else:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
