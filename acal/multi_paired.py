#!/usr/bin/env python
"""Multi-model paired comparison and scorecard: AI+RES vs every board baseline. CPU only.

Multi-model versions of `acal_cfs_paired.png` and `acal_cfs_scorecard.png`
(`cfsbase.fig_paired` / `cfsbase.fig_scorecard`). Inputs are the per-source tables under
runs/acal/analysis/s2s/era5/ (`<source>_paired.csv`, `<source>_scorecard.csv`, read-only);
the statistics are recomputed here with `cfsbase.paired_stats` so the figures always match
the CSVs, never the json.

  paired.png     One row per (baseline, variant), raw empirical then bias-corrected
                 empirical, in three columns: log(P_RES / P_model) at the observed value,
                 and Brier(model) - Brier(AI+RES) at 3 K and at 4 K. Positive = AI+RES
                 better. Small jittered points are the 42 cases (heat/cold colour, rung
                 shape), the diamond is the mean with a 90% case-bootstrap CI, in the
                 model colour (raw darker), the text at right is wins-ties-losses and the
                 Wilcoxon p. The Gaussian and 4-member rows stay in the per-source tables.
  scorecard.png  Per forecast, the summary scores the CFSv2 scorecard shows: median P(obs)
                 and median conservative lift (all cases, with heat and cold medians), the
                 number of cases with P(obs) = 0 and with lift > 1. Each baseline gets its
                 raw row, its bias-corrected row and an AI+RES row on THE SAME WINDOW
                 (13 frames for CFSv2/GEFSv12, 12 daily-mean frames for EC46/GEPS), so the
                 AI+RES numbers differ a little between windows; that is shown, not hidden.

Each baseline is paired with AI+RES and ERA5 on its own window (`Source.obs_window`); the
per-source tables already do this. Also writes DATA_DIR/paired_summary.csv and
DATA_DIR/scorecard_summary.csv.

    python -m acal.multi_paired
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
from acal import s2sbase as S2

VARIANTS = (("raw", "raw_emp"), ("corrected", "corr_emp"))   # (row label, table suffix)
METRICS = (("logratio", "logratio_%s", "log( P_RES / P_model ) at the observed value"),
           ("brier_3K", "dbrier_%s_3K", "Brier(model) - Brier(AI+RES), 3 K"),
           ("brier_4K", "dbrier_%s_4K", "Brier(model) - Brier(AI+RES), 4 K"))


def _read(source: str, kind: str) -> pd.DataFrame | None:
    """Per-source table, or None (with a note) when it is missing: that source is left out."""
    p = MF.table(source, kind)
    if not p.exists():
        print(f"  [multi_paired] no {kind} table for {source} ({p.name}); left out")
        return None
    return pd.read_csv(p)


def load(kind: str) -> dict[str, pd.DataFrame]:
    out = {}
    for s in MF.models():
        d = _read(s, kind)
        if d is not None:
            out[s] = d
    return out


def paired_rows(tables: dict[str, pd.DataFrame]) -> list[dict]:
    """One row per (source, variant): per-case values per metric plus its paired_stats."""
    rows = []
    for s, d in tables.items():
        for vlab, v in VARIANTS:
            row = dict(source=s, variant=vlab, window=S2.get(s).obs_window, d=d, stats={})
            for key, col, _ in METRICS:
                c = col % v
                row["stats"][key] = CB.paired_stats(d[c].values) if c in d else None
            rows.append(row)
    return rows


def _p_txt(st: dict | None) -> str:
    if st is None or not st["n"]:
        return "n/a"
    p = st["wilcoxon_p"]
    ptxt = "n/a" if not np.isfinite(p) else ("p<0.001" if p < 0.001 else f"p={p:.3f}")
    return f"{st['win']}-{st['tie']}-{st['loss']}  {ptxt}"


def fig_paired(rows: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    AN._style(plt)
    fam_c, mk = AN.FAMILY_COLOR, AN.RUNG_MARKER
    n = len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 0.62 * n + 2.6), sharey=True)
    rng = np.random.default_rng(1)
    ys = -np.arange(n, dtype=float)
    for ax, (key, col, ttl) in zip(axes, METRICS):
        for r, y in zip(rows, ys):
            d, v = r["d"], dict(VARIANTS)[r["variant"]]
            c = col % v
            if c not in d:
                continue
            x = d[c].values
            jit = rng.uniform(-0.27, 0.27, len(d))
            for fam in ("heat", "cold"):
                for g in (2, 3, 4):
                    m = ((d.family == fam) & (d.rung == g)).values
                    if m.any():
                        ax.scatter(x[m], y + jit[m], s=12, marker=mk[g], color=fam_c[fam],
                                   edgecolor="white", linewidth=0.3, alpha=0.75, zorder=3)
            st = r["stats"][key]
            mc = MF.color(r["source"], raw=r["variant"] == "raw")
            if st is not None and st["n"]:
                ax.plot([st["ci_lo"], st["ci_hi"]], [y, y], color=mc, lw=2.6, zorder=4,
                        solid_capstyle="round")
                ax.scatter(st["mean"], y, s=60, marker="D", color=mc, edgecolor="white",
                           linewidth=0.8, zorder=5)
            ax.text(1.02, y, _p_txt(st), transform=ax.get_yaxis_transform(), fontsize=7.5,
                    va="center", color="0.2")
        ax.axvline(0, color="0.3", lw=1, zorder=1)
        # light separator between models (rows come in pairs)
        for k in range(1, n // 2):
            ax.axhline(ys[2 * k] + 0.5, color="0.8", lw=0.8)
        ax.set_ylim(ys[-1] - 0.6, 0.6)
        ax.set_title(ttl, fontsize=9.5)
        n1 = int(max(r["d"][f"o_{key[-2:]}"].sum() for r in rows if f"o_{key[-2:]}" in r["d"])) \
            if key != "logratio" else None
        ax.set_xlabel("positive = AI+RES better" + (f"   ({n1} cases o=1)" if n1 is not None else ""))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [f"{MF.short(r['source'])} {r['variant']}" for r in rows], fontsize=8.5)
    for t, r in zip(axes[0].get_yticklabels(), rows):
        t.set_color(MF.color(r["source"], raw=r["variant"] == "raw"))
    handles = [Line2D([], [], ls="none", marker="o", ms=6, color=fam_c[f], label=f)
               for f in ("heat", "cold")]
    handles += [Line2D([], [], ls="none", marker=mk[g], ms=6, color="0.45", label=f"rung {g} K")
                for g in (2, 3, 4)]
    handles.append(Line2D([], [], ls="none", marker="D", ms=7, color="0.3",
                          label="mean, 90% case-bootstrap CI (model colour; raw darker)"))
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    wins = ", ".join(f"{MF.short(s)} {S2.get(s).obs_window}" for s in dict.fromkeys(r["source"] for r in rows))
    fig.suptitle("AI+RES vs every baseline, paired per case, 21 d lead, 42 cases selected on "
                 "outcome. Text at right: AI+RES wins-ties-losses, Wilcoxon p.\n"
                 f"Empirical probabilities; each baseline on its own ERA5 window ({wins}).",
                 y=1.045, fontsize=9.5)
    fig.text(0.5, -0.005, MF.FOOT, ha="center", va="top", fontsize=6.8, color="0.35", wrap=True)
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.62, top=0.9, bottom=0.08)
    return MF.save(fig, "paired.png")


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #
def _med(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def score_row(sc: pd.DataFrame, pcol: str, lcol: str) -> dict:
    """Summary of one forecast: medians (all/heat/cold) and the two counts, 42-case table."""
    p, l = sc[pcol].values, sc[lcol].values
    out = {}
    for name, m in (("all", np.ones(len(sc), bool)), ("heat", (sc.family == "heat").values),
                    ("cold", (sc.family == "cold").values)):
        out[f"p_med_{name}"] = _med(p[m])
        out[f"lift_med_{name}"] = _med(l[m])
    out["n"] = int(len(sc))
    out["n_zero_heat"] = int(((p == 0) & (sc.family == "heat").values).sum())
    out["n_zero_cold"] = int(((p == 0) & (sc.family == "cold").values).sum())
    out["n_lift_gt1_heat"] = int(((l > 1) & (sc.family == "heat").values).sum())
    out["n_lift_gt1_cold"] = int(((l > 1) & (sc.family == "cold").values).sum())
    out["n_zero"] = out["n_zero_heat"] + out["n_zero_cold"]
    out["n_lift_gt1"] = out["n_lift_gt1_heat"] + out["n_lift_gt1_cold"]
    return out


def scorecard_rows(tables: dict[str, pd.DataFrame]) -> list[dict]:
    """Per source: raw, corrected, then AI+RES on that source's window."""
    rows = []
    for s, sc in tables.items():
        w = S2.get(s).obs_window
        for vlab, v in VARIANTS:
            if f"p_obs_{v}" in sc:
                rows.append(dict(source=s, kind=vlab, window=w,
                                 **score_row(sc, f"p_obs_{v}", f"lift_cons_{v}")))
        if "res_p_obs_sn" in sc:
            rows.append(dict(source=s, kind="aires", window=w,
                             **score_row(sc, "res_p_obs_sn", "res_lift_obs_raw_cons")))
    return rows


def _row_label(r: dict) -> str:
    if r["kind"] == "aires":
        return f"AI+RES ({r['window']})"
    return f"{MF.short(r['source'])} {r['kind']}"


def _row_color(r: dict) -> str:
    return MF.color("aires") if r["kind"] == "aires" else MF.color(r["source"], r["kind"] == "raw")


def fig_scorecard(rows: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    AN._style(plt)
    fam_c = AN.FAMILY_COLOR
    n = len(rows)
    ys = -np.arange(n, dtype=float)
    # blank line between model groups so the AI+RES row of one window is not read as the next
    gap = np.array([0.0] + [0.45 * (rows[i]["source"] != rows[i - 1]["source"]) for i in range(1, n)])
    ys = ys - np.cumsum(gap)
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.42 * n + 2.6), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.3, 1.3, 1, 1]))
    pos = [r[f"p_med_{k}"] for r in rows for k in ("all", "heat", "cold") if r[f"p_med_{k}"] > 0]
    pfloor = (min(pos) if pos else 0.01) / 2.5
    lpos = [r[f"lift_med_{k}"] for r in rows for k in ("all", "heat", "cold") if r[f"lift_med_{k}"] > 0]
    lfloor = (min(lpos) if lpos else 0.1) / 2.5

    def fl(x, f):
        return x if (np.isfinite(x) and x > 0) else f

    for ax, (pre, floor, ttl, xl) in zip(axes[:2], (
            ("p_med", pfloor, "median P(obs)", "median P(observed tail), 0 plotted on the left band"),
            ("lift_med", lfloor, "median lift over climatology",
             "median conservative lift (1 = climatology)"))):
        ax.set_xscale("log")
        hi = max([r[f"{pre}_{k}"] for r in rows for k in ("all", "heat", "cold")] + [floor]) * 2.2
        ax.set_xlim(floor / 1.8, hi)
        ax.axvspan(floor / 1.8, floor * 1.35, color="0.93", zorder=0)
        if pre == "lift_med":
            ax.axvline(1, color="0.7", lw=0.8, zorder=1)
        for r, y in zip(rows, ys):
            c = _row_color(r)
            mk = "o" if r["kind"] == "aires" else ("^" if r["kind"] == "raw" else "s")
            xs = {k: fl(r[f"{pre}_{k}"], floor) for k in ("all", "heat", "cold")}
            ax.plot([min(xs.values()), max(xs.values())], [y, y], color="0.8", lw=1.2, zorder=2)
            ax.scatter(xs["all"], y, s=70, marker=mk, color=c, edgecolor="white", linewidth=0.8, zorder=4)
            # family ticks sit above the marker and are offset (heat up, cold down) so they stay
            # visible when a family median equals the all-case median (discrete k/N values)
            for k, dy in (("heat", 0.17), ("cold", -0.17)):
                ax.scatter(xs[k], y + dy, s=60, marker="|", color=fam_c[k], linewidth=2.0, zorder=6)
        cand = (0.02, 0.03, 0.05, 0.1, 0.2, 0.3) if pre == "p_med" else (0.3, 0.5, 1, 2, 3, 5)
        ticks = [t for t in cand if floor * 1.4 < t < hi]
        ax.set_xticks([floor] + ticks, ["0"] + [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel(xl, fontsize=8)
        ax.grid(axis="y", visible=False)
    for ax, (pre, ttl) in zip(axes[2:], (("n_zero", "cases with P(obs) = 0"),
                                         ("n_lift_gt1", "cases with lift > 1"))):
        for r, y in zip(rows, ys):
            h, c = r[f"{pre}_heat"], r[f"{pre}_cold"]
            ax.barh(y, h, height=0.62, color=fam_c["heat"], alpha=0.85, zorder=3)
            ax.barh(y, c, height=0.62, left=h, color=fam_c["cold"], alpha=0.85, zorder=3)
            ax.text(h + c + 0.8, y, f"{h + c}", va="center", fontsize=8, color="0.2")
        ax.set_xlim(0, 42)
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel("cases of 42 (heat red, cold blue)", fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [_row_label(r) for r in rows], fontsize=8.5)
    for t, r in zip(axes[0].get_yticklabels(), rows):
        t.set_color(_row_color(r))
        if r["kind"] == "aires":
            t.set_fontweight("bold")
    for ax in axes:
        ax.set_ylim(ys[-1] - 0.7, 0.7)
    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=MF.color("aires"), label="AI+RES"),
               Line2D([], [], ls="none", marker="s", ms=7, color="0.45", label="baseline, bias-corrected"),
               Line2D([], [], ls="none", marker="^", ms=7, color="0.45", label="baseline, raw"),
               Line2D([], [], ls="none", marker="|", ms=9, mew=1.8, color=fam_c["heat"], label="heat median"),
               Line2D([], [], ls="none", marker="|", ms=9, mew=1.8, color=fam_c["cold"], label="cold median")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.suptitle("Per-forecast scores, AI+RES and every baseline, 21 d lead, 42 cases selected on "
                 "outcome. Large marker = all cases; AI+RES is re-reduced on each baseline's window.\n"
                 "AI+RES P(obs) is self-normalized; lift counts P_clim = 0 as 1/284.",
                 y=1.045, fontsize=9.5)
    fig.text(0.5, -0.005, MF.FOOT, ha="center", va="top", fontsize=6.8, color="0.35", wrap=True)
    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.12, top=0.9, bottom=0.1)
    return MF.save(fig, "scorecard.png")


# --------------------------------------------------------------------------- #
def write_summaries(prow: list[dict], srow: list[dict]) -> list[Path]:
    MF.DATA_DIR.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in prow:
        for key, _, _ in METRICS:
            st = r["stats"][key]
            if st is not None:
                recs.append(dict(source=r["source"], variant=r["variant"], window=r["window"],
                                 metric=key, **st))
    p1 = MF.DATA_DIR / "paired_summary.csv"
    pd.DataFrame(recs).to_csv(p1, index=False)
    p2 = MF.DATA_DIR / "scorecard_summary.csv"
    pd.DataFrame(srow).to_csv(p2, index=False)
    print(f"  wrote {p1} and {p2.name}")
    return [p1, p2]


def main() -> list[Path]:
    ptab, stab = load("paired"), load("scorecard")
    out = []
    if ptab:
        prow = paired_rows(ptab)
        out.append(fig_paired(prow))
    else:
        prow = []
    srow = scorecard_rows(stab) if stab else []
    if srow:
        out.append(fig_scorecard(srow))
    write_summaries(prow, srow)
    for r in prow:
        s = r["stats"]
        print("  " + f"{r['source']:6s} {r['variant']:9s} " + " | ".join(
            f"{k} {s[k]['mean']:+.3f} [{s[k]['ci_lo']:+.3f},{s[k]['ci_hi']:+.3f}] {_p_txt(s[k])}"
            for k in s if s[k] is not None))
    return out


if __name__ == "__main__":
    main()
