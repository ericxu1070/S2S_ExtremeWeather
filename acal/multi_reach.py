#!/usr/bin/env python
"""Multi-model reach figures: AI+RES next to every baseline on the cases that hit 4 K. CPU ONLY.

`acal.reach` shows the five cases that reached +/-4 K with ONE baseline (CFSv2). This module
puts AI+RES and EVERY board baseline (`multifig.models()`: CFSv2 13f, EC46, GEFSv12, GEPS),
raw and bias-corrected, in one figure per type.

Figure A, `reach/prob_4K.png`
-----------------------------
One row per hit: P(A >= +4 K) (warm) or P(A <= -4 K) (cold) as given by AI+RES, by each
baseline corrected (squares) and raw (triangles), and by the +/-30 d climatology, on a log
axis. Zeros are a real forecast value and sit in their own "0" column left of an axis
break. The last row of each panel summarises the same-direction cases that stopped short
(min-to-max, tick at the median). Values are read from the per-source tables
(`multifig.table(s, 'prob_4K')`, written by `reach.run_source`), never recomputed.

Two AI+RES markers: filled = the 13-frame window (CFSv2, GEFSv12), hollow = the same
walkers re-reduced on the 12 daily-mean frames (EC46, GEPS). The two differ because the
12-frame window moves the observed A_L and so the walkers beyond the threshold.

Figure B, `reach/reach_4K.png` and `reach/reach_<case>.png`
-----------------------------------------------------------
Strip plot of member CONUS week-mean A_L, one panel per case: a row for AI+RES (area =
normalized weight), then a corrected and a raw row per baseline, annotated "k of N beyond,
P = x". Each baseline row is paired with ITS window: daily-mean sources are verified on 12
frames, so their observed A_L differs slightly from the 13-frame ERA5 star; those rows
(and the 12-frame AI+RES row) carry a black tick at their own observed value. P is the
equal-weight fraction for baselines and the self-normalized weight for AI+RES, via
`reach.reach_stats` rules (`reach.beyond`), cross-checked against the tables.

Caveat printed on every figure: AI+RES sampled the tail its case went to; in operations
both tails would be run.

    python -m acal.multi_reach       # or: python -m acal.multifig --stage reach
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import cfsbase as CB
from acal import multifig as MF
from acal import reach as RE
from acal import roc as R
from acal import s2sbase as S2
from acal import truth as TR

THRESHOLD = RE.THRESHOLD
JITTER_SEED = 20261008
JITTER = 0.30                         # +/- fraction of a row slot; vertical only
CLIM_COLOR = "#666666"
TAIL_NOTE = RE.TAIL_NOTE
PROB_TOL = 1e-6                          # tables are written with 6 significant digits

# Log-axis layout of figure A (same as reach.py): zeros in their own column.
P_LO, P_HI = 2.5e-4, 1.6
P_ZERO, P_BREAK = 7e-5, 1.4e-4


# --------------------------------------------------------------------------- #
# Series layout (pure helpers)
# --------------------------------------------------------------------------- #
def has_daily(models: list[str]) -> bool:
    """True when some baseline is verified on 12 daily-mean frames."""
    return any(S2.get(s).obs_window == "12f" for s in models)


def series(models: list[str], daily: bool | None = None, clim: bool = False) -> list[dict]:
    """Ordered series of one case, top to bottom: AI+RES (13f), AI+RES (12f, only when a
    daily-mean baseline exists), then corrected and raw rows per baseline in board order
    (and, for figure A, climatology). Keys: key, label, color, marker, window, source, kind."""
    daily = has_daily(models) if daily is None else daily
    c = MF.color("aires")
    out = [dict(key="aires13", label="AI+RES (13 frames)", color=c, marker=MF.MARKER["aires"],
                window="13f", source=None, kind="aires")]
    if daily:
        out.append(dict(key="aires12", label="AI+RES (12 frames)", color=c,
                        marker=MF.MARKER["aires"], window="12f", source=None, kind="aires"))
    for s in models:
        w = S2.get(s).obs_window
        out.append(dict(key=f"{s}_corr", label=f"{MF.short(s)} corrected", color=MF.color(s),
                        marker=MF.MARKER["corr"], window=w, source=s, kind="corr"))
        out.append(dict(key=f"{s}_raw", label=f"{MF.short(s)} raw",
                        color=MF.color(s, raw=True), marker=MF.MARKER["raw"], window=w,
                        source=s, kind="raw"))
    if clim:
        out.append(dict(key="clim", label="Climatology", color=CLIM_COLOR, marker="D",
                        window="13f", source=None, kind="clim"))
    return out


def offsets(n: int, half: float = 0.40) -> np.ndarray:
    """n evenly spaced y offsets in a row slot, first series on top."""
    return np.linspace(half, -half, n) if n > 1 else np.zeros(1)


def px(p) -> np.ndarray:
    """Probability to plotted x: zeros go to the "0" column."""
    p = np.asarray(p, dtype=float)
    return np.where(p > 0, p, P_ZERO)


def member_stats(x: np.ndarray, sign: float, wn: np.ndarray | None = None) -> tuple[int, int, float]:
    """(k beyond, n, P): equal-weight fraction, or the weight sum when `wn` is given."""
    b = RE.beyond(x, sign, THRESHOLD)
    p = float(wn[b].sum()) if wn is not None else float(b.mean()) if b.size else float("nan")
    return int(b.sum()), int(b.size), p


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_tables(models: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Per-source prob_4K tables read now (they are not final); a source with no table
    is left out with a printed note."""
    tabs, keep = {}, []
    for s in models:
        p = MF.table(s, "prob_4K")
        if not p.exists():
            print(f"[multi_reach] note: no {p.name} for {s}; left out", flush=True)
            continue
        tabs[s] = pd.read_csv(p).set_index("episode_id")
        keep.append(s)
    return tabs, keep


def hit_cases(tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The cases that reached 4 K on the 13-frame ERA5 truth (the first 13f source's
    table, else the first table), one row each: pool, obs, sign. Printed when another
    source's hit set differs (a 12-frame window can move a case across 4 K)."""
    ref = next((s for s in tabs if S2.get(s).obs_window == "13f"), next(iter(tabs)))
    t = tabs[ref]
    hits = t[t.hit.astype(bool)]
    for s, o in tabs.items():
        h2 = set(o.index[o.hit.astype(bool)])
        if h2 != set(hits.index):
            print(f"[multi_reach] note: {s} hit set {sorted(h2)} != {ref} {sorted(hits.index)}",
                  flush=True)
    return hits.assign(sign=np.sign(hits.threshold_K))


def prob(tabs: dict, s: str, eid: str, kind: str) -> float:
    return float(tabs[s].loc[eid, f"p_{s}_{kind}"])


def series_prob(tabs: dict, ser: dict, eid: str, models: list[str]) -> float:
    """Table probability of one series for one case."""
    if ser["kind"] == "aires":
        src = next((s for s in models if S2.get(s).obs_window == ser["window"]), None)
        return float(tabs[src].loc[eid, "p_ai_res"]) if src else float("nan")
    if ser["kind"] == "clim":
        return float(tabs[models[0]].loc[eid, "p_clim"])
    return prob(tabs, ser["source"], eid, ser["kind"])


def case_members(eid: str, models: list[str], cases: dict, recs: dict) -> dict:
    """Members of one case for every series: {key: dict(x, wn, obs)} plus case metadata.
    AI+RES comes from the truth-switched cases of each window; baselines from their own
    records (`reach.load_records` = `s2sbase.load`)."""
    cmp_ = AN._read(AN.case_dir(eid) / "compare.json")
    w = np.asarray(cmp_["weights"], dtype=float)
    wn = w / w.sum()
    out = {}
    for win, cs in cases.items():
        c = cs[eid]
        if not np.allclose(c.weights, w, rtol=1e-12, atol=0):
            raise SystemExit(f"[multi_reach] {eid}: case weights != compare.json")
        out[f"aires{win[:2]}"] = dict(x=np.asarray(c.al, dtype=float), wn=wn, obs=float(c.obs))
    for s in models:
        rec = recs[s].loc[eid]
        out[f"{s}_corr"] = dict(x=CB.members(rec["al"], rec["bias"], "corr_emp"), wn=None,
                                obs=float(rec["obs"]))
        out[f"{s}_raw"] = dict(x=CB.members(rec["al"], rec["bias"], "raw_emp"), wn=None,
                               obs=float(rec["obs"]))
    out["_meta"] = dict(ess_last=float(cmp_["ess_by_step"][-1]), kish=float(1.0 / np.sum(wn ** 2)),
                        n_founders=int(cmp_["n_founders"]))
    return out


def load_members(hits: pd.DataFrame, models: list[str], tabs: dict) -> dict:
    """{episode_id: case_members}; cross-checks every P against the tables (warning)."""
    tr = TR.get_truth(MF.TRUTH)
    wins = sorted({S2.get(s).obs_window for s in models})
    cases = {w: S2.aires_cases(tr, w) for w in wins}
    recs = {s: RE.load_records(s, MF.TRUTH) for s in models}
    ser = series(models)
    ms = {}
    for eid, r in hits.iterrows():
        m = case_members(eid, models, cases, recs)
        ms[eid] = m
        for sr in ser:
            d = m[sr["key"]] if sr["key"] in m else None
            if d is None:
                continue
            _, _, p = member_stats(d["x"], r.sign, d["wn"])
            tp = series_prob(tabs, sr, eid, models)
            if np.isfinite(tp) and abs(p - tp) > PROB_TOL:
                print(f"[multi_reach] WARNING {eid} {sr['key']}: members give P={p:.5f}, "
                      f"table {tp:.5f} (table stale?)", flush=True)
    return ms


# --------------------------------------------------------------------------- #
# Figure A
# --------------------------------------------------------------------------- #
def _fmt(p: float) -> str:
    return RE._fmt(p)


def _note_lines(models: list[str]) -> str:
    """Footer: per-source window, member count and lead, then the selection/tilt notes."""
    parts = []
    for s in models:
        mem, lead = S2.ens_text(s)
        win = "12 daily-mean frames" if S2.get(s).obs_window == "12f" else "13 frames"
        parts.append(f"{MF.short(s)} {mem}, {lead}, {win}")
    return ("21-day AI+RES lead; 32 walkers (self-normalized); baselines equally weighted. "
            "Each baseline is verified on its own window:\n" + "; ".join(parts[:2]) + ";\n" + "; ".join(parts[2:]) + ".\n"
            + TAIL_NOTE + "\n" + MF.FOOT)


def _legend_handles(ser: list[dict]):
    from matplotlib.lines import Line2D
    hs = []
    for sr in ser:
        hollow = sr["key"] == "aires12"
        hs.append(Line2D([], [], ls="none", marker=sr["marker"], ms=9.5,
                         mfc="white" if hollow else sr["color"], mec=sr["color"] if hollow else "white",
                         mew=1.6 if hollow else 0.8, color=sr["color"], label=sr["label"]))
    return hs


def fig_prob(hits: pd.DataFrame, tabs: dict, models: list[str], plt) -> Path:
    ser = series(models, clim=True)
    offs = dict(zip([s["key"] for s in ser], offsets(len(ser))))
    pools = [("warm", "Warm tail: P(A >= +4 K)"), ("cold", "Cold tail: P(A <= -4 K)")]
    pools = [(p, t) for p, t in pools if (hits.pool == p).any()]
    nrows = {p: int((hits.pool == p).sum()) + 1 for p, _ in pools}
    W, row_h, left, right = 14.0, 2.3, 3.2, 1.9
    top_h, gap, xlab_h, foot_h = 1.5, 0.95, 0.6, 1.25
    H = top_h + sum(nrows.values()) * row_h + gap * (len(pools) - 1) + xlab_h + foot_h
    fig = plt.figure(figsize=(W, H))
    y_top = H - top_h
    first = True
    for pool, title in pools:
        h = hits[hits.pool == pool].sort_values("obs", key=lambda s: s.abs())
        n = nrows[pool]
        hh = n * row_h
        ax = fig.add_axes([left / W, (y_top - hh) / H, (W - left - right) / W, hh / H])
        y_top -= hh + gap
        ax.set_xscale("log")
        ax.set_xlim(P_ZERO / 1.9, P_HI)
        ax.set_ylim(-0.6, n - 0.4)
        ax.axvspan(P_ZERO / 1.9, P_BREAK, color="0.96", lw=0, zorder=0)
        for x in (1e-3, 1e-2, 1e-1, 1.0, P_ZERO):
            ax.axvline(x, color="0.9", lw=0.8, zorder=0)
        labels = []
        for i, (eid, r) in enumerate(h.iterrows()):
            y = n - 1 - i
            labels.append((y, f"{eid[:3]}  {_case_date(eid)}  obs {r.obs:+.2f} K"))
            for sr in ser:
                p = series_prob(tabs, sr, eid, models)
                if not np.isfinite(p):
                    continue
                hollow = sr["key"] == "aires12"
                ax.scatter(px(p), y + offs[sr["key"]], s=70, marker=sr["marker"],
                           facecolor="white" if hollow else sr["color"],
                           edgecolor=sr["color"] if hollow else "white",
                           linewidth=1.5 if hollow else 0.7, zorder=4, clip_on=False)
                ax.text(px(p) * 1.2, y + offs[sr["key"]], _fmt(p), va="center", ha="left",
                        fontsize=8.5, color=sr["color"], zorder=5)
            if i:
                ax.axhline(y + 0.5, color="0.8", lw=0.8, zorder=1)
            if first:                         # direct series names beside the first row
                for sr in ser:
                    ax.text(1.01, y + offs[sr["key"]], sr["label"], transform=ax.get_yaxis_transform(),
                            va="center", ha="left", fontsize=9, color=sr["color"], clip_on=False)
                first = False
        miss = _stopped_short(tabs, models, pool, set(h.index))
        if h.shape[0]:
            ax.axhline(0.5, color="0.75", lw=0.8)
        for sr in ser:
            v = np.array([series_prob(tabs, sr, e, models) for e in miss])
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            o = offs[sr["key"]]
            lo, md, hi = float(px(v.min())), float(px(np.median(v))), float(px(v.max()))
            start = lo if v.min() > 0 else P_BREAK * 1.35   # never draw across the break
            ax.plot([start, hi], [o, o], color=sr["color"], lw=2.0, solid_capstyle="butt", zorder=3)
            for xe in (lo, hi):
                ax.plot([xe, xe], [o - 0.035, o + 0.035], color=sr["color"], lw=1.3)
            ax.plot([md, md], [o - 0.05, o + 0.05], color=sr["color"], lw=3.0, zorder=4)
        labels.append((0, f"Stopped short (n={len(miss)})\nrange, | = median"))
        ax.set_yticks([y for y, _ in labels], [s for _, s in labels], fontsize=12)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([P_ZERO, 1e-3, 1e-2, 1e-1, 1.0], ["0", "0.001", "0.01", "0.1", "1"], fontsize=12)
        ax.minorticks_off()
        kw = dict(transform=ax.get_xaxis_transform(), color="0.3", lw=1.2, clip_on=False)
        for xb in (P_BREAK / 1.12, P_BREAK * 1.12):
            ax.plot([xb / 1.06, xb * 1.06], [-0.02, 0.02], **kw)
        ax.set_title(title, fontsize=14, weight="bold", loc="left",
                     x=-left / (W - left - right) + 0.01, pad=8)
        ax.set_xlabel("Forecast probability (log scale; zeros in their own column)", fontsize=12.5)
    fig.legend(handles=_legend_handles(ser), loc="upper center", ncol=6, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 1 - 0.65 / H), handletextpad=0.3,
               columnspacing=1.3)
    fig.text(0.5, 1 - 0.15 / H, f"The {RE._n_word(len(hits))} cases that reached 4 K: probability "
             "each forecast gave beyond 4 K, AI+RES vs all baselines", ha="center", va="top",
             fontsize=16, weight="bold")
    fig.text(0.5, 0.08 / H, _note_lines(models), ha="center", va="bottom", fontsize=9.5,
             color="0.25", linespacing=1.4)
    return MF.save(fig, "reach/prob_4K.png")


def _stopped_short(tabs: dict, models: list[str], pool: str, hit_ids: set) -> list[str]:
    """Same-pool cases outside `hit_ids`, present in every source's table."""
    ids = None
    for s in models:
        t = tabs[s]
        cur = set(t.index[t.pool == pool])
        ids = cur if ids is None else ids & cur
    return sorted((ids or set()) - hit_ids)


def _case_date(eid: str) -> str:
    """'Jan 2023' from the episode id's trailing yyyymmdd."""
    return pd.Timestamp(eid.split("_")[-1]).strftime("%b %Y")


# --------------------------------------------------------------------------- #
# Figure B
# --------------------------------------------------------------------------- #
def _strip(ax, m: dict, sign: float, ser: list[dict], xlim, rng, obs13: float,
           fs: float, area: float) -> None:
    """One case: a row per series (top to bottom) on `ax`."""
    n = len(ser)
    a = sign * THRESHOLD
    if sign > 0:
        ax.axvspan(a, xlim[1], color="#f3e6e6", lw=0, zorder=0)
    else:
        ax.axvspan(xlim[0], a, color="#e4ecf5", lw=0, zorder=0)
    ax.axvline(a, color="0.25", lw=1.4, ls="--", zorder=1)
    ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)
    ys = []
    for i, sr in enumerate(ser):
        y = n - 1 - i
        ys.append(y)
        d = m[sr["key"]]
        x = d["x"]
        jy = y + rng.uniform(-JITTER, JITTER, x.size)
        c = sr["color"]
        if sr["kind"] == "aires":
            sz = np.maximum(area * d["wn"] * x.size, 5.0)
            hollow = sr["key"] == "aires12"
            ax.scatter(x, jy, s=sz, marker="o", facecolor="none" if hollow else c, alpha=0.6,
                       edgecolor=c, linewidth=0.9, zorder=3)
        else:
            ax.scatter(x, jy, s=area * 0.8, marker=sr["marker"], color=c, alpha=0.8,
                       edgecolor="white", linewidth=0.5, zorder=3)
        k, nn, p = member_stats(x, sign, d["wn"])
        ax.text(1.01, y, f"{k} of {nn} beyond, P = {_fmt(p)}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=fs, color="0.1" if sr["kind"] == "aires" else c)
        if abs(d["obs"] - obs13) > 1e-9:      # own-window observation differs from the star
            ax.plot([d["obs"]] * 2, [y - 0.42, y + 0.42], color="0.05", lw=1.8, zorder=5)
        if i and sr["source"] != ser[i - 1]["source"] and sr["kind"] == "corr":
            ax.axhline(y + 0.5, color="0.88", lw=0.8, zorder=0)
    ax.scatter([obs13], [n - 0.25], s=230, marker="*", color="#f2c200", edgecolor="0.1",
               linewidth=0.9, zorder=6, clip_on=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_yticks(ys, [s["label"] for s in ser], fontsize=fs)
    for t, sr in zip(ax.get_yticklabels(), ser):
        t.set_color("0.1" if sr["kind"] == "aires" else sr["color"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def xlims(ms: list[dict], sign: float, keys: list[str]) -> tuple[float, float]:
    """Shared x range of a tail group: members, observations and the threshold."""
    vals = [np.r_[m[k]["x"], m[k]["obs"]] for m in ms for k in keys]
    vals.append(np.array([sign * THRESHOLD]))
    v = np.concatenate(vals)
    return float(np.floor(v.min() - 0.3)), float(np.ceil(v.max() + 0.3))


def _legend(fig, ser: list[dict], H: float, top: float, fs: float):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="none", marker="o", ms=8, color=MF.color("aires"), alpha=0.6,
                 label="AI+RES walker (area = weight; hollow = 12 frames)"),
          Line2D([], [], ls="none", marker="s", ms=7.5, color="0.35", label="bias-corrected member"),
          Line2D([], [], ls="none", marker="^", ms=7.5, color="0.15", label="raw member"),
          Line2D([], [], ls="none", marker="*", ms=15, color="#f2c200", mec="0.1",
                 label="ERA5 observed (13 frames)"),
          Line2D([], [], color="0.05", lw=1.8, label="observed on the row's own 12-frame window"),
          Line2D([], [], color="0.25", ls="--", lw=1.4, label="+/-4 K threshold")]
    if not has_daily([s["source"] for s in ser if s["source"]]):
        hs = [h for h in hs if "12-frame" not in h.get_label()]
    leg = fig.legend(handles=hs, loc="upper center", ncol=3, fontsize=fs, frameon=False,
                     bbox_to_anchor=(0.5, 1 - top / H), handletextpad=0.3, columnspacing=1.5)
    fig.canvas.draw()
    bb, fb = leg.get_window_extent(), fig.bbox
    margin = 0.1 * fig.dpi
    if bb.x0 < fb.x0 + margin or bb.x1 > fb.x1 - margin or bb.y1 > fb.y1 - margin:
        raise SystemExit(f"[multi_reach] legend outside figure ({bb.x0:.0f}-{bb.x1:.0f} px)")
    return leg


def _title(eid: str, sign: float, obs: float) -> str:
    return f"{eid[:3]}  {_case_date(eid)}: {'warm' if sign > 0 else 'cold'} case, observed {obs:+.2f} K"


def fig_reach(eids: list[str], hits: pd.DataFrame, ms: dict, models: list[str], plt,
              name: str, big: bool = False, xl_fixed=None) -> Path:
    """Panels for `eids` (a case list, one tail group each); `big` = the per-case figure."""
    ser = series(models)
    n = len(ser)
    slot = 0.36 if big else 0.30
    fs = 12.0 if big else 10.5
    area = 60.0 if big else 42.0
    panel_h = n * slot + 0.35
    W, gap, top_h, foot_h, xlab_h = (13.5 if big else 14.0), 0.7, (2.0 if big else 1.55), 1.3, 0.62
    groups = [[e for e in eids if hits.loc[e, "sign"] > 0], [e for e in eids if hits.loc[e, "sign"] < 0]]
    groups = [g for g in groups if g]
    npan = sum(len(g) for g in groups)
    H = top_h + npan * panel_h + (npan - len(groups)) * gap + len(groups) * xlab_h + (len(groups) - 1) * 0.2 + foot_h
    if len(groups) > 1:
        H += 0.3
    fig = plt.figure(figsize=(W, H))
    left, width = 3.0 / W, 6.6 / W
    y = H - top_h
    rng = np.random.default_rng(JITTER_SEED)
    keys = [s["key"] for s in ser]
    obs13 = {e: next(ms[e][k]["obs"] for k in keys if k == "aires13") for e in eids}
    for g in groups:
        sign = hits.loc[g[0], "sign"]
        xl = xl_fixed or xlims([ms[e] for e in g], sign, keys)
        for i, e in enumerate(g):
            y -= panel_h
            ax = fig.add_axes([left, y / H, width, panel_h / H])
            _strip(ax, ms[e], sign, ser, xl, rng, obs13[e], fs, area)
            if not big:                   # the per-case figure carries it as the suptitle
                ax.text(-0.31, 1.07, _title(e, sign, obs13[e]), transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=fs + 1.5, weight="bold")
            if i < len(g) - 1:
                ax.set_xticklabels([])
                y -= gap
            else:
                ax.set_xlabel("CONUS week-mean T2m anomaly A_L (K)", fontsize=fs + 1)
        y -= xlab_h + (0.5 if g is not groups[-1] else 0)
    if big:
        e = eids[0]
        mt = ms[e]["_meta"]
        fig.text(0.5, 1 - 0.12 / H, _title(e, hits.loc[e, "sign"], obs13[e]), ha="center",
                 va="top", fontsize=16, weight="bold")
        fig.text(0.5, 1 - 0.5 / H, f"AI+RES noise: ESS at last resampling step {mt['ess_last']:.1f} "
                 f"of 32, Kish ESS of final weights {mt['kish']:.1f}, {mt['n_founders']} founders",
                 ha="center", va="top", fontsize=11.5, color="0.2")
        _legend(fig, ser, H, 0.82, 10.5)
    else:
        fig.text(0.5, 1 - 0.12 / H, f"Where the members ended: the {RE._n_word(len(eids))} cases "
                 "that reached 4 K, AI+RES vs all baselines", ha="center", va="top", fontsize=16,
                 weight="bold")
        _legend(fig, ser, H, 0.5, 10.5)
    fig.text(0.5, 0.08 / H, _note_lines(models), ha="center", va="bottom", fontsize=9.5,
             color="0.25", linespacing=1.4)
    return MF.save(fig, name)


# --------------------------------------------------------------------------- #
def main() -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tabs, models = load_tables(MF.models())
    if not models:
        print("[multi_reach] no per-source prob_4K tables; nothing to draw")
        return []
    hits = hit_cases(tabs)
    hits = hits.sort_values(["pool", "obs"], ascending=[False, True],
                            key=lambda s: s.abs() if s.name == "obs" else s)
    print(f"[multi_reach] baselines {models}; hits {list(hits.index)}", flush=True)
    ms = load_members(hits, models, tabs)
    RE._style(plt)
    plt.rcParams.update({"axes.grid": False})
    out = [fig_prob(hits, tabs, models, plt)]
    order = list(hits.index)
    out.append(fig_reach(order, hits, ms, models, plt, "reach/reach_4K.png"))
    for g in ([e for e in order if hits.loc[e, "sign"] > 0], [e for e in order if hits.loc[e, "sign"] < 0]):
        for e in g:
            # per-case figures share their tail group's x range (as reach.py does)
            out.append(fig_reach_case(e, g, hits, ms, models, plt))
    return out


def fig_reach_case(eid: str, grp: list[str], hits: pd.DataFrame, ms: dict, models: list[str], plt) -> Path:
    """Per-case figure with the group's shared x range (so cases compare across figures)."""
    sign = hits.loc[eid, "sign"]
    keys = [s["key"] for s in series(models)]
    xl = xlims([ms[e] for e in grp], sign, keys)
    return fig_reach([eid], hits, ms, models, plt, f"reach/reach_{eid}.png", big=True, xl_fixed=xl)


if __name__ == "__main__":
    main()
