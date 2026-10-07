#!/usr/bin/env python
"""The five cases that reached +/-4 K: what each forecast said, and where its members ended. CPU ONLY.

`roc.py` asks a pooled question (did the cases that reached 4 K get more probability than
the ones that stopped short?). With 3 warm and 2 cold hits that pool is thin, so this
module shows the hits one by one.

Figure A, `prob_4K.png`
-----------------------
One row per hit: P(A >= +4 K) (warm) or P(A <= -4 K) (cold) from AI+RES (self-normalized,
32 weighted walkers), CFSv2 bias-corrected and raw (16 lagged members) and the +/-30 d
climatology, on a log axis. A zero is a real forecast value (no member got there), so it
is drawn in its own "0" column left of an axis break, never dropped. A last row per panel
summarises the same-direction cases that stopped short of 4 K: min-to-max range with a
tick at the median, so a hit's probability can be read against the non-hits'. Values come
from `roc.case_rows` (the same numbers as `roc_cases.csv`, asserted) and are written to
`runs/acal/analysis/prob_4K.csv`.

Figure B, `reach_4K.png` and `reach_<case>.png`
-----------------------------------------------
Strip plot of the members' CONUS week-mean A_L: the 32 final walkers
(`compare.json["realized"]["conus"]`, marker AREA proportional to the normalized weight
`compare.json["weights"]`), and the 16 CFSv2 members raw and bias-corrected
(`cfsbase.members`). Each row is annotated "k of n beyond, P = x", with P weighted for
AI+RES. The per-case figures also give the AI+RES estimate's noise: `ess_by_step[-1]`
(the ESS the run logged at its last resampling step), the Kish ESS of the final weights
(`1 / sum(wn^2)`, the number that governs the variance of P) and the founder count.

Selection caveat, printed on every figure: AI+RES sampled the tail its case went to; in
operations both tails would be run.

    python -m acal.reach      # prob_4K.csv, figures/acal/reach/*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import cfsbase as CB
from acal import ccfg
from acal import roc as R

THRESHOLD = 4.0
PROB_OUT = AN.OUT / "prob_4K.csv"
FIG_DIR = ccfg.FIG_ROOT / "reach"
JITTER_SEED = 20261007
JITTER = 0.17                         # +/- row units; vertical only, within a row
WEIGHT_AREA = 55.0                    # pt^2 for a walker of weight 1/32 (equal weights)
MIN_AREA = 6.0                        # floor so a near-zero weight stays visible
CFS_AREA = 55.0
PROB_TOL = 1e-12

MODELS = ("ai_res", "cfs_corr", "cfs_raw", "clim")
STYLE = {m: R.MODEL_STYLE[m] for m in MODELS}
MARKER = {"ai_res": "o", "cfs_corr": "s", "cfs_raw": "^", "clim": "D"}
TAIL_NOTE = ("AI+RES sampled the tail its case went to (warm runs for warm cases); "
             "in operations both tails would be run.")
LEAD_NOTE = ("21-day lead; CONUS week-mean T2m anomaly; AI+RES 32 walkers (self-normalized), "
             "CFSv2 16 lagged members")
STRIP_NOTE = LEAD_NOTE + ", equally weighted"

# Log-axis layout of figure A: zeros sit in their own column left of a break.
P_LO, P_HI = 2.5e-4, 1.6
P_ZERO, P_BREAK = 7e-5, 1.4e-4


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def short_label(eid: str, peak, obs: float) -> str:
    return f"{eid[:3]}  {pd.Timestamp(peak):%b %Y}  obs {obs:+.2f} K"


def prob_table(rc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows of `roc.case_rows` at +/-4 K, with a `hit` flag; nothing recomputed."""
    rc = R.case_rows() if rc is None else rc
    t = rc[rc.threshold_K.abs() == THRESHOLD].copy()
    t["hit"] = t.outcome.astype(bool)
    return t


def event_members(eid: str, cfs: pd.DataFrame) -> dict:
    """Walker A_L + normalized weights, CFS raw and corrected members, for one case."""
    cmp_ = AN._read(AN.case_dir(eid) / "compare.json")
    al = np.asarray(cmp_["realized"]["conus"], dtype=float)
    w = np.asarray(cmp_["weights"], dtype=float)
    rec = cfs.loc[eid]
    wn = w / w.sum()
    return dict(al=al, wn=wn, raw=CB.members(rec["al"], rec["bias"], "raw_emp"),
                corr=CB.members(rec["al"], rec["bias"], "corr_emp"),
                ess_last=float(cmp_["ess_by_step"][-1]), kish=float(1.0 / np.sum(wn ** 2)),
                n_founders=int(cmp_["n_founders"]), sign=float(rec["sign"]),
                obs=float(rec["obs"]), peak=rec["peak"])


def beyond(x, sign: float, k: float = THRESHOLD) -> np.ndarray:
    """The tail-signed `>=` rule of `aceiling.beyond`, at threshold sign*k."""
    return sign * (np.asarray(x, dtype=float) - sign * k) >= 0.0


def reach_stats(m: dict) -> dict:
    """Per row: (k beyond, n, P). AI+RES P is weighted (self-normalized)."""
    s = m["sign"]
    b = beyond(m["al"], s)
    return {"ai_res": (int(b.sum()), b.size, float(m["wn"][b].sum())),
            "cfs_corr": (int(beyond(m["corr"], s).sum()), m["corr"].size,
                         float(beyond(m["corr"], s).mean())),
            "cfs_raw": (int(beyond(m["raw"], s).sum()), m["raw"].size,
                        float(beyond(m["raw"], s).mean()))}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(plt) -> None:
    R._style(plt)
    plt.rcParams.update({"axes.grid": False})


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=200, facecolor="white")
    print(f"  wrote {p} ({p.stat().st_size / 1e3:.0f} kB)")
    return p


def _px(p) -> np.ndarray:
    """Probability to plotted x: zeros go to the "0" column."""
    p = np.asarray(p, dtype=float)
    return np.where(p > 0, p, P_ZERO)


def _fmt(p: float) -> str:
    if p == 0:
        return "0"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def fig_prob(t: pd.DataFrame, plt) -> Path:
    from matplotlib.lines import Line2D
    pools = [("warm", "Warm tail: P(A >= +4 K)"), ("cold", "Cold tail: P(A <= -4 K)")]
    nrows = {p: int(t[(t.pool == p) & t.hit].shape[0]) + 1 for p, _ in pools}
    W, row_h, left, right = 12.5, 1.15, 3.3, 0.5
    top_h, gap, xlab_h, foot_h = 1.45, 0.95, 0.55, 0.75
    H = top_h + sum(nrows.values()) * row_h + gap + xlab_h + foot_h
    fig = plt.figure(figsize=(W, H))
    y_top = H - top_h
    offs = dict(zip(MODELS, (0.27, 0.09, -0.09, -0.27)))
    for pool, title in pools:
        d = t[t.pool == pool]
        hits = d[d.hit].sort_values("obs", key=lambda s: s.abs())
        miss = d[~d.hit]
        n = nrows[pool]
        h = n * row_h
        ax = fig.add_axes([left / W, (y_top - h) / H, (W - left - right) / W, h / H])
        y_top -= h + gap
        ax.set_xscale("log")
        ax.set_xlim(P_ZERO / 1.9, P_HI)
        ax.set_ylim(-0.6, n - 0.4)
        ax.axvspan(P_ZERO / 1.9, P_BREAK, color="0.96", lw=0, zorder=0)
        for x in (1e-3, 1e-2, 1e-1, 1.0):
            ax.axvline(x, color="0.9", lw=0.8, zorder=0)
        ax.axvline(P_ZERO, color="0.9", lw=0.8, zorder=0)
        labels = []
        for i, r in enumerate(hits.itertuples()):
            y = n - 1 - i
            labels.append((y, short_label(r.episode_id, r.peak, r.obs)))
            for m in MODELS:
                p = getattr(r, R.PCOL[m])
                st = STYLE[m]
                ax.scatter(_px(p), y + offs[m], s=95, marker=MARKER[m], color=st["color"],
                           edgecolor="white", linewidth=0.8, zorder=4, clip_on=False)
                ax.text(_px(p) * 1.22, y + offs[m], _fmt(p), va="center", ha="left",
                        fontsize=10.5, color=st["color"] if m != "clim" else "#6f4c16",
                        zorder=5)
        if hits.shape[0]:
            ax.axhline(0.5, color="0.75", lw=0.8)
        for m in MODELS:
            v = miss[R.PCOL[m]].values
            st = STYLE[m]
            lo, md, hi = _px(v.min()), _px(np.median(v)), _px(v.max())
            # a zero minimum is a cap in the "0" column; the line resumes after the break,
            # never drawn across it as if the gap were a probability range
            start = lo if v.min() > 0 else P_BREAK * 1.35
            ax.plot([start, hi], [offs[m]] * 2, color=st["color"], lw=2.2,
                    solid_capstyle="butt", zorder=3)
            ax.plot([lo, lo], [offs[m] - 0.05, offs[m] + 0.05], color=st["color"], lw=1.4)
            ax.plot([hi, hi], [offs[m] - 0.05, offs[m] + 0.05], color=st["color"], lw=1.4)
            ax.plot([md, md], [offs[m] - 0.085, offs[m] + 0.085], color=st["color"], lw=3.2,
                    zorder=4)
        labels.append((0, f"Stopped short (n={len(miss)})\nrange, | = median"))
        ax.set_yticks([y for y, _ in labels], [s for _, s in labels], fontsize=12.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ticks = [P_ZERO, 1e-3, 1e-2, 1e-1, 1.0]
        ax.set_xticks(ticks, ["0", "0.001", "0.01", "0.1", "1"], fontsize=12.5)
        ax.minorticks_off()
        # axis break marks on the bottom spine
        kw = dict(transform=ax.get_xaxis_transform(), color="0.3", lw=1.2, clip_on=False)
        for xb in (P_BREAK / 1.12, P_BREAK * 1.12):
            ax.plot([xb / 1.06, xb * 1.06], [-0.035, 0.035], **kw)
        ax.set_title(title, fontsize=15, weight="bold", loc="left", x=-left / (W - left - right) + 0.01,
                     pad=8)
    ax.set_xlabel("Forecast probability (log scale; zeros in their own column)", fontsize=13.5)
    handles = [Line2D([], [], ls="none", marker=MARKER[m], ms=10, color=STYLE[m]["color"],
                      mec="white", label=STYLE[m]["label"]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=12.5, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.62 / H), handletextpad=0.3, columnspacing=1.6)
    fig.text(0.5, 1 - 0.15 / H, "The five cases that reached 4 K: probability each forecast "
             "gave beyond 4 K", ha="center", va="top", fontsize=17, weight="bold")
    fig.text(0.5, 0.08 / H, LEAD_NOTE + "\n" + TAIL_NOTE, ha="center", va="bottom",
             fontsize=10.5, color="0.25", linespacing=1.4)
    p = _save(fig, "prob_4K.png")
    plt.close(fig)
    return p


ROWS = (("ai_res", "AI+RES walkers"), ("cfs_corr", "CFSv2 bias-corrected"),
        ("cfs_raw", "CFSv2 raw"))


def _strip(ax, m: dict, xlim, rng, annotate: bool = True) -> None:
    """Three rows of one case on `ax` (y = 2, 1, 0)."""
    s, k = m["sign"], THRESHOLD
    a = s * k
    if s > 0:
        ax.axvspan(a, xlim[1], color="#f3e6e6", lw=0, zorder=0)
    else:
        ax.axvspan(xlim[0], a, color="#e4ecf5", lw=0, zorder=0)
    ax.axvline(a, color="0.25", lw=1.4, ls="--", zorder=1)
    ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)
    st = reach_stats(m)
    for y, (key, _) in zip((2, 1, 0), ROWS):
        x = m["al"] if key == "ai_res" else m["corr" if key == "cfs_corr" else "raw"]
        jy = y + rng.uniform(-JITTER, JITTER, x.size)
        c = STYLE[key]["color"]
        if key == "ai_res":
            area = np.maximum(WEIGHT_AREA * m["wn"] * x.size, MIN_AREA)
            ax.scatter(x, jy, s=area, marker="o", facecolor=c, alpha=0.55, edgecolor=c,
                       linewidth=0.8, zorder=3)
        else:
            ax.scatter(x, jy, s=CFS_AREA, marker=MARKER[key], color=c, alpha=0.8,
                       edgecolor="white", linewidth=0.6, zorder=3)
        if annotate:
            kk, nn, pp = st[key]
            ax.text(1.01, y, f"{kk} of {nn} beyond, P = {_fmt(pp)}",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=11.5, color=c
                    if key != "ai_res" else "0.1")
    ax.scatter([m["obs"]], [2.92], s=260, marker="*", color="#f2c200", edgecolor="0.1",
               linewidth=0.9, zorder=6, clip_on=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, 3.15)
    ax.set_yticks([2, 1, 0], [lab for _, lab in ROWS], fontsize=12)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def _xlims(ms: list[dict]) -> tuple[float, float]:
    vals = np.concatenate([np.r_[m["al"], m["raw"], m["corr"], m["obs"], m["sign"] * THRESHOLD]
                           for m in ms])
    lo, hi = np.floor(vals.min() - 0.3), np.ceil(vals.max() + 0.3)
    return float(lo), float(hi)


def _legend_handles():
    from matplotlib.lines import Line2D
    c = STYLE["ai_res"]["color"]
    return [Line2D([], [], ls="none", marker="o", ms=np.sqrt(WEIGHT_AREA), color=c, alpha=0.55,
                   label="AI+RES walker (area = weight)"),
            Line2D([], [], ls="none", marker="s", ms=np.sqrt(CFS_AREA),
                   color=STYLE["cfs_corr"]["color"], label="CFSv2 corrected member"),
            Line2D([], [], ls="none", marker="^", ms=np.sqrt(CFS_AREA),
                   color=STYLE["cfs_raw"]["color"], label="CFSv2 raw member"),
            Line2D([], [], ls="none", marker="*", ms=17, color="#f2c200", mec="0.1",
                   label="ERA5 observed"),
            Line2D([], [], color="0.25", ls="--", lw=1.4, label="+/-4 K threshold")]


def _legend(fig, H: float, top: float, fontsize: float):
    """One-row legend; HARD-FAIL if it does not sit inside the figure with a margin."""
    leg = fig.legend(handles=_legend_handles(), loc="upper center", ncol=5,
                     fontsize=fontsize, frameon=False, bbox_to_anchor=(0.5, 1 - top / H),
                     handletextpad=0.3, columnspacing=1.3)
    fig.canvas.draw()
    bb, fb = leg.get_window_extent(), fig.bbox
    margin = 0.1 * fig.dpi
    if bb.x0 < fb.x0 + margin or bb.x1 > fb.x1 - margin or bb.y1 > fb.y1 - margin:
        raise SystemExit(f"[reach] legend {bb.x0:.0f}-{bb.x1:.0f} px outside figure "
                         f"{fb.x0:.0f}-{fb.x1:.0f} px")
    return leg


def fig_reach_one(eid: str, m: dict, xlim, plt) -> Path:
    W, H = 12.0, 5.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([2.3 / W, 1.2 / H, 6.6 / W, 2.95 / H])
    _strip(ax, m, xlim, np.random.default_rng(JITTER_SEED))
    ax.set_xlabel("CONUS week-mean T2m anomaly A_L (K)", fontsize=13.5)
    word = "warm" if m["sign"] > 0 else "cold"
    fig.text(0.5, 1 - 0.12 / H, f"{eid[:3]}  {pd.Timestamp(m['peak']):%b %Y}: {word} case, "
             f"observed {m['obs']:+.2f} K", ha="center", va="top", fontsize=17, weight="bold")
    fig.text(0.5, 1 - 0.55 / H, f"AI+RES noise: ESS at last resampling step "
             f"{m['ess_last']:.1f} of 32, Kish ESS of final weights {m['kish']:.1f}, "
             f"{m['n_founders']} founders", ha="center", va="top", fontsize=12, color="0.2")
    _legend(fig, H, 0.85, 11.0)
    fig.text(0.5, 0.06 / H, STRIP_NOTE + "\n" + TAIL_NOTE, ha="center", va="bottom",
             fontsize=9.5, color="0.25", linespacing=1.35)
    p = _save(fig, f"reach_{eid}.png")
    plt.close(fig)
    return p


def fig_reach_all(ms: dict, order: list[str], plt) -> Path:
    W, row_h, gap_ev, gap_tail = 13.0, 2.1, 0.55, 0.45
    top_h, foot_h, xlab_h = 1.75, 0.75, 0.6
    warm = [e for e in order if ms[e]["sign"] > 0]
    cold = [e for e in order if ms[e]["sign"] < 0]
    H = (top_h + (len(warm) + len(cold)) * row_h + (len(warm) - 1 + len(cold) - 1) * gap_ev
         + gap_tail + 2 * xlab_h + foot_h)
    fig = plt.figure(figsize=(W, H))
    left, width = 3.1 / W, 6.9 / W
    y = H - top_h
    rng = np.random.default_rng(JITTER_SEED)
    for grp in (warm, cold):
        xl = _xlims([ms[e] for e in grp])
        for i, e in enumerate(grp):
            y -= row_h
            ax = fig.add_axes([left, y / H, width, row_h / H])
            _strip(ax, ms[e], xl, rng)
            m = ms[e]
            ax.text(-0.33, 1.0, f"{e[:3]}  {pd.Timestamp(m['peak']):%b %Y}  obs {m['obs']:+.2f} K",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=12.5,
                    weight="bold")
            if i < len(grp) - 1:
                ax.set_xticklabels([])
                y -= gap_ev
            else:
                ax.set_xlabel("CONUS week-mean T2m anomaly A_L (K)", fontsize=13.5)
        y -= gap_tail if grp is warm else 0
        y -= xlab_h
    fig.text(0.5, 1 - 0.12 / H, "Where the members ended: the five cases that reached 4 K",
             ha="center", va="top", fontsize=17, weight="bold")
    _legend(fig, H, 0.55, 11.5)
    fig.text(0.5, 0.08 / H, STRIP_NOTE + "\n" + TAIL_NOTE, ha="center", va="bottom",
             fontsize=10.5, color="0.25", linespacing=1.4)
    p = _save(fig, "reach_4K.png")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def run() -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = prob_table()
    hits = t[t.hit]
    print(f"[reach] hits at 4 K: {hits.episode_id.tolist()}")
    cfs = CB.load_cfs().set_index("episode_id")
    ms = {e: event_members(e, cfs) for e in hits.episode_id}
    # The strip plot's P must be the table's P (same members, same rule).
    for r in hits.itertuples():
        st = reach_stats(ms[r.episode_id])
        for key, col in (("ai_res", "p_ai_res"), ("cfs_corr", "p_cfs_corr"),
                         ("cfs_raw", "p_cfs_raw")):
            if abs(st[key][2] - getattr(r, col)) > PROB_TOL:
                raise SystemExit(f"[reach] {r.episode_id} {key}: {st[key][2]} != {getattr(r, col)}")
    out = t[["episode_id", "pool", "threshold_K", "obs", "hit", "p_ai_res", "p_cfs_corr",
             "p_cfs_raw", "p_clim", "n_walkers_beyond", "n_cfs_corr_beyond",
             "n_cfs_raw_beyond"]].copy()
    for e, m in ms.items():
        out.loc[out.episode_id == e, "ess_last_step"] = m["ess_last"]
        out.loc[out.episode_id == e, "kish_ess_final"] = m["kish"]
        out.loc[out.episode_id == e, "n_founders"] = m["n_founders"]
    out.to_csv(PROB_OUT, index=False, float_format="%.6g")
    print(f"  wrote {PROB_OUT}")
    for e, m in ms.items():
        st = reach_stats(m)
        print(f"  {e}: " + "; ".join(f"{k} {v[0]}/{v[1]} P={v[2]:.4f}" for k, v in st.items())
              + f"; ESS last {m['ess_last']:.1f}, Kish {m['kish']:.1f}, founders {m['n_founders']}")
    _style(plt)
    fig_prob(t, plt)
    order = list(hits.sort_values(["pool", "obs"], ascending=[False, True],
                                  key=lambda s: s.abs() if s.name == "obs" else s).episode_id)
    fig_reach_all(ms, order, plt)
    for grp in ([e for e in order if ms[e]["sign"] > 0], [e for e in order if ms[e]["sign"] < 0]):
        xl = _xlims([ms[e] for e in grp])
        for e in grp:
            fig_reach_one(e, ms[e], xl, plt)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
