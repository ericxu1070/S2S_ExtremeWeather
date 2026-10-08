"""acal.overall - the headline comparison figures of the acal multi-model board.

Three figures and one table, all drawn from the board (runs/acal/analysis/s2s/board/).
Nothing is re-scored here: paired statistics are copied from board_paired.csv, and the
scoreboard only aggregates board_cases.csv columns (mean / median / root-mean).

  figures/acal/overall/forest.png         paired AI+RES-vs-model forest (headline)
  figures/acal/overall/scoreboard.png     per-model scoreboard, ERA5 and HRRR truth
  figures/acal/overall/hrrr_vs_era5.png   does the verdict survive HRRR truth?
  runs/acal/analysis/s2s/board/overall_numbers.csv   every number plotted

Inputs
  board_paired.csv   paired means, 90% case-bootstrap CIs, W/T/L, Wilcoxon p
  board_cases.csv    per-case scores (P(obs), Brier, CRPS, ens-mean error, field metrics)
  board_summary.json coverage (which registered sources are absent)
  <truth>/maps_land_means_<source>.json   land-mean BSS / CSI of the maps stage (only
                     for the scoreboard's BSS/CSI columns)

A source with no rows on the board (EC46 before its data lands) is absent from every
figure and listed as pending; BB-SUBS appears wherever acal.bbsubs merged estimate rows
(estimate=True), and only on the metrics it estimates (never P(obs), never maps).
Rerun after `python -m acal.board --stage all`:

    python -m acal.overall                  # numbers CSV + all three figures (~20 s)
    python -m acal.overall --stage forest   # one figure
"""
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "runs" / "acal" / "analysis" / "s2s"
BOARD_DIR = ANALYSIS / "board"
FIG_DIR = ROOT / "figures" / "acal" / "overall"
NUMBERS_CSV = BOARD_DIR / "overall_numbers.csv"
HRRR_SUMMARY_JSON = ROOT / "runs" / "acal" / "index_hrrr" / "cases_hrrr_summary.json"

AIRES = "aires"
REFERENCE = ("cfs",)                 # published 25-frame CFSv2: footnote only
FIG_TRUTHS = ("era5", "hrrr")        # forest + scoreboard
ALL_TRUTHS = ("era5", "hrrr", "hrrr_raw")
SUBSETS = ("all", "heat", "cold")
FOREST_METRICS = ("logratio", "dbrier_3K", "dbrier_4K", "dcrps")
HRRR_METRICS = ("logratio", "dcrps")
P_SIG = 0.05

_STYLE_FALLBACK = {
    "aires": dict(label="AI+RES", color="#D55E00"),
    "cfs": dict(label="CFSv2", color="#0072B2", deg="~0.94 deg"),
    "cfs13": dict(label="CFSv2", color="#0072B2", deg="~0.94 deg"),
    "ec46": dict(label="ECMWF IFS (EC46)", color="#009E73", deg="1.5 deg"),
    "gefs": dict(label="GEFSv12", color="#CC79A7", deg="0.5 deg"),
    "geps": dict(label="ECCC GEPS", color="#E69F00", deg="1 deg"),
    "bbsubs": dict(label="BB-SUBS (est.)", color="#56B4E9", hatch="//", ls="--",
                   estimate=True),
    "era5": dict(label="ERA5", color="#000000"),
    "hrrr": dict(label="HRRR", color="#555555"),
    "hrrr_raw": dict(label="HRRR", color="#555555"),
}
try:                                    # the score lane owns the shared style
    from acal.s2sbase import MODEL_STYLE, BOARD_SOURCES, TILT_NOTE
except Exception:                       # pragma: no cover - light envs
    MODEL_STYLE = _STYLE_FALLBACK
    BOARD_SOURCES = ("cfs13", "ec46", "gefs", "geps")
    TILT_NOTE = ("AI+RES was steered toward the observed tail direction (warm walkers on "
                 "warm cases, cold on cold); the baselines were not.")

TRUTH_TEXT = {"era5": "ERA5", "hrrr": "HRRR (offset-corrected)",
              "hrrr_raw": "HRRR raw (sensitivity)"}
TRUTH_SHORT = {"era5": "ERA5", "hrrr": "HRRR", "hrrr_raw": "HRRR raw"}
WINDOW_TEXT = {"13f": "13 frames", "12f": "daily means, 12 frames"}
RES_WINDOW_TEXT = {"13f": "reference, 13 frames",
                   "12f": "reference, 12 frames (daily-mean partner)"}
INK, INK2, INK3 = "#1f1f1f", "#555555", "#8a8a8a"
GRID = "#e4e4e4"
BAND = "#f3f3f3"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_csv(p: Path) -> pd.DataFrame:
    """Board CSVs are read exactly (pandas' default float parser is not round-trip)."""
    return pd.read_csv(p, float_precision="round_trip")


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(("true", "1", "1.0"))


def load_board(board_dir: Path = BOARD_DIR) -> dict:
    board_dir = Path(board_dir)
    paired = read_csv(board_dir / "board_paired.csv")
    cases = read_csv(board_dir / "board_cases.csv")
    for d in (paired, cases):
        d["estimate"] = _as_bool(d["estimate"])
    sj = board_dir / "board_summary.json"
    summary = json.loads(sj.read_text()) if sj.exists() else {}
    return dict(paired=paired, cases=cases, summary=summary)


def style(name: str) -> dict:
    return MODEL_STYLE.get(name) or _STYLE_FALLBACK.get(name) or dict(label=name,
                                                                        color="#777777")


def label(name: str, estimate: bool = False) -> str:
    if estimate and name != "bbsubs":
        return style(name)["label"] + " (est.)"
    return style(name)["label"]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def board_rows(paired: pd.DataFrame) -> list[tuple[str, bool]]:
    """(source, estimate) rows of the figures, in board order: the measured board sources
    present (BOARD_SOURCES order, the 'cfs' reference left out), any other measured
    source after them, then estimate rows (BB-SUBS)."""
    p = paired[["source", "estimate"]].drop_duplicates()
    meas = set(p.source[~p.estimate]) - set(REFERENCE) - {AIRES}
    rows = [s for s in BOARD_SOURCES if s in meas]
    rows += sorted(meas - set(rows))
    est = [s for s in pd.unique(p.source[p.estimate])]
    return [(s, False) for s in rows] + [(s, True) for s in est]


def variants_for(paired: pd.DataFrame, source: str, estimate: bool) -> tuple[str, str | None]:
    """(headline, secondary) variant of a row: raw empirical vs bias-corrected empirical.
    A source with no raw rows (BB-SUBS is anchored on debiased EC46) heads with corr_emp."""
    vs = set(paired.variant[(paired.source == source) & (paired.estimate == estimate)])
    if "raw_emp" in vs:
        return "raw_emp", ("corr_emp" if "corr_emp" in vs else None)
    if "corr_emp" in vs:
        return "corr_emp", None
    return sorted(vs)[0], None


def source_window(cases: pd.DataFrame, source: str, estimate: bool = False) -> str:
    w = cases.window[(cases.source == source) & (cases.estimate == estimate)]
    return str(w.iloc[0]) if len(w) else "13f"


def n_native(cases: pd.DataFrame, source: str, estimate: bool = False) -> str | None:
    """Native member count over the source's cases: '31', or a range '51-101' when the
    count varies by case (EC46 is 51 members on some cases, 101 on others)."""
    n = cases.n_native[(cases.source == source) & (cases.estimate == estimate)].dropna()
    if not len(n):
        return None
    lo, hi = int(round(float(n.min()))), int(round(float(n.max())))
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def forest_table(paired: pd.DataFrame, truths=FIG_TRUTHS,
                 metrics=FOREST_METRICS) -> pd.DataFrame:
    """The board_paired rows the forest draws, verbatim, with a 'role' column
    (headline = raw ERA5-clim anomaly, secondary = bias-corrected)."""
    out = []
    for order, (source, est) in enumerate(board_rows(paired)):
        head, sec = variants_for(paired, source, est)
        for role, v in (("headline", head), ("secondary", sec)):
            if v is None:
                continue
            m = paired[(paired.source == source) & (paired.estimate == est)
                       & (paired.variant == v) & (paired.aires_variant == "sn")
                       & paired.truth.isin(truths) & paired.subset.isin(SUBSETS)
                       & paired.metric.isin(metrics)].copy()
            m.insert(0, "role", role)
            m.insert(0, "row", order)
            out.append(m)
    if not out:
        return pd.DataFrame(columns=["row", "role"] + list(paired.columns))
    return pd.concat(out, ignore_index=True)


# ---- scoreboard ------------------------------------------------------------ #
# key, header, colour scale kind, +1 higher better / -1 lower better, paired metric
SCORE_COLS = (
    ("p_obs", "P(obs)\nmedian", "ratio", +1, "logratio"),
    ("lift", "lift\nmedian", "ratio", +1, None),
    ("brier_2K", "Brier\n2 K", "ratio", -1, "dbrier_2K"),
    ("brier_3K", "Brier\n3 K", "ratio", -1, "dbrier_3K"),
    ("brier_4K", "Brier\n4 K", "ratio", -1, "dbrier_4K"),
    ("crps", "CRPS\n(K)", "ratio", -1, "dcrps"),
    ("rmse_mean", "ens-mean\nRMSE (K)", "ratio", -1, "dsqerr"),
    ("field_rmse", "field\nRMSE (K)", "ratio", -1, "dfield_rmse"),
    ("pattern_r", "pattern\nr", "diff", +1, None),
    ("bss_p2", "land median\nBSS +2 K", "diff", +1, None),
    ("bss_m2", "land median\nBSS -2 K", "diff", +1, None),
    ("csi_p2", "land CSI\n+2 K", "diff", +1, None),
    ("csi_m2", "land CSI\n-2 K", "diff", +1, None),
)
SCORE_STAT = {"p_obs": "median", "lift": "median over p_clim_obs > 0",
              "brier_2K": "mean", "brier_3K": "mean", "brier_4K": "mean", "crps": "mean",
              "rmse_mean": "sqrt(mean(mean_sqerr))", "field_rmse": "mean",
              "pattern_r": "mean", "bss_p2": "land median (maps)", "bss_m2": "land median (maps)",
              "csi_p2": "land mean (maps)", "csi_m2": "land mean (maps)"}
# BSS: the land MEDIAN (maps bss_med). The cos-lat land mean of per-cell BSS is dominated by
# the few cells where P_clim is near 0 (per-cell BSS down to -2e5) and can flip a verdict.
LAND_KEYS = {"bss_p2": ("bss_med", "2.0"), "bss_m2": ("bss_med", "-2.0"),
             "csi_p2": ("csi", "2.0"), "csi_m2": ("csi", "-2.0")}
RATIO_SAT = math.log(2.0)          # colour saturates at a factor of 2
DIFF_SAT = 0.25                    # ... or at 0.25 for r / BSS / CSI


def _fin(x) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    return x[np.isfinite(x)]


def aggregate(g: pd.DataFrame) -> dict:
    """Scoreboard values of one (truth, row) block of board_cases rows."""
    def mean(c):
        x = _fin(g[c]) if c in g else np.array([])
        return float(x.mean()) if x.size else np.nan
    p = _fin(g.p_obs) if "p_obs" in g else np.array([])
    ok = (g.p_clim_obs > 0) & np.isfinite(g.p_obs) if "p_clim_obs" in g else None
    lift = _fin(g.p_obs[ok] / g.p_clim_obs[ok]) if ok is not None else np.array([])
    sq = _fin(g.mean_sqerr) if "mean_sqerr" in g else np.array([])
    return dict(p_obs=float(np.median(p)) if p.size else np.nan,
                lift=float(np.median(lift)) if lift.size else np.nan,
                brier_2K=mean("brier_2K"), brier_3K=mean("brier_3K"),
                brier_4K=mean("brier_4K"), crps=mean("crps"),
                rmse_mean=float(np.sqrt(sq.mean())) if sq.size else np.nan,
                field_rmse=mean("field_rmse"), pattern_r=mean("field_pattern_r"))


def land_means(truth: str, source: str, analysis: Path = ANALYSIS) -> dict | None:
    p = Path(analysis) / truth / f"maps_land_means_{source}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("7d")


def _land_entry(lm: dict | None, key: str) -> dict | None:
    if not lm:
        return None
    return lm.get(key)


def better_t(model: float, ref: float, kind: str, sign: int) -> float:
    """Signed, saturated difference in [-1, 1]; positive = AI+RES better (model worse)."""
    if not (np.isfinite(model) and np.isfinite(ref)):
        return np.nan
    if kind == "ratio":
        if model <= 0 or ref <= 0:
            if model == ref:
                return 0.0
            d = (ref - model) if sign > 0 else (model - ref)
            return float(np.sign(d))
        lr = math.log(model / ref)
        t = (-lr if sign > 0 else lr) / RATIO_SAT
    else:
        t = ((ref - model) if sign > 0 else (model - ref)) / DIFF_SAT
    return float(np.clip(t, -1.0, 1.0))


def scoreboard_table(cases: pd.DataFrame, paired: pd.DataFrame, truths=FIG_TRUTHS,
                     analysis: Path = ANALYSIS) -> pd.DataFrame:
    """Long table: one row per (truth, board row, column) with the value, the matched-window
    AI+RES value it is coloured against, the colour position t and the paired Wilcoxon p."""
    rows = board_rows(paired)
    windows = {(s, e): source_window(cases, s, e) for s, e in rows}
    need_w = sorted(set(windows.values()) | {"13f"}, key=lambda w: (w != "13f", w))
    out = []
    for truth in truths:
        ct = cases[cases.truth == truth]
        # land means: AI+RES per window from the first source json of that window
        res_land = {}
        for s, e in rows:
            if e:
                continue
            w = windows[(s, e)]
            if w not in res_land:
                lm = land_means(truth, s, analysis)
                ent = _land_entry(lm, "AI+RES")
                if ent:
                    res_land[w] = (ent, s)
        ref = {}
        for w in need_w:
            g = ct[(ct.source == AIRES) & (ct.window == w) & (ct.variant == "sn")]
            if g.empty:
                continue
            vals = aggregate(g)
            ent, from_src = res_land.get(w, (None, None))
            for k, (m, thr) in LAND_KEYS.items():
                vals[k] = float(ent[m][thr]) if ent and thr in ent.get(m, {}) else np.nan
            ref[w] = vals
            for key, *_ in SCORE_COLS:
                out.append(dict(truth=truth, source=AIRES, label="AI+RES", estimate=False,
                                variant="sn", window=w, n=len(g), column=key,
                                stat=SCORE_STAT[key], value=vals[key], aires_value=vals[key],
                                t=np.nan, wilcoxon_p=np.nan, paired_mean=np.nan,
                                land_from=from_src if key in LAND_KEYS else None))
        for s, e in rows:
            head, _ = variants_for(paired, s, e)
            w = windows[(s, e)]
            g = ct[(ct.source == s) & (ct.estimate == e) & (ct.variant == head)]
            if g.empty or w not in ref:
                continue
            vals = aggregate(g)
            lm = None if e else land_means(truth, s, analysis)
            tag = "raw" if head.startswith("raw") else "corrected"
            ent = _land_entry(lm, f"{style(s)['label']} {tag}")
            for k, (m, thr) in LAND_KEYS.items():
                vals[k] = float(ent[m][thr]) if ent and thr in ent.get(m, {}) else np.nan
            # colour against AI+RES on THIS row's cases: a source that covers only some
            # cases (EC46 mid-download, a failed cube) must not be judged against all 42
            ga = ct[(ct.source == AIRES) & (ct.window == w) & (ct.variant == "sn")]
            row_ref = dict(ref[w])
            if set(g.episode_id) != set(ga.episode_id):
                row_ref = aggregate(ga[ga.episode_id.isin(set(g.episode_id))])
                print(f"[overall] {truth} {label(s, e)}: {g.episode_id.nunique()} of "
                      f"{ga.episode_id.nunique()} cases - AI+RES reference re-aggregated on "
                      "those cases", flush=True)
                for k in LAND_KEYS:
                    row_ref[k] = np.nan
            ent_res = _land_entry(lm, "AI+RES")     # maps: AI+RES on this source's cases
            if ent_res:
                for k, (m, thr) in LAND_KEYS.items():
                    row_ref[k] = (float(ent_res[m][thr]) if thr in ent_res.get(m, {})
                                  else np.nan)
            pp = paired[(paired.truth == truth) & (paired.source == s)
                        & (paired.estimate == e) & (paired.variant == head)
                        & (paired.subset == "all")].set_index("metric")
            for key, _, kind, sign, pm in SCORE_COLS:
                wp = pmean = np.nan
                if pm is not None and pm in pp.index:
                    wp = float(pp.loc[pm, "wilcoxon_p"])
                    pmean = float(pp.loc[pm, "mean"])
                out.append(dict(truth=truth, source=s, label=label(s, e), estimate=e,
                                variant=head, window=w, n=len(g), column=key,
                                stat=SCORE_STAT[key], value=vals[key],
                                aires_value=row_ref[key],
                                t=better_t(vals[key], row_ref[key], kind, sign),
                                wilcoxon_p=wp, paired_mean=pmean,
                                land_from=s if key in LAND_KEYS and not e else None))
    return pd.DataFrame(out)


# ---- HRRR question ---------------------------------------------------------- #
def obs_table(cases: pd.DataFrame, truths=ALL_TRUTHS, window: str = "13f") -> pd.DataFrame:
    """Observed A_L per case and truth (AI+RES 'sn' rows carry each truth's obs)."""
    g = cases[(cases.source == AIRES) & (cases.variant == "sn") & (cases.window == window)
              & cases.truth.isin(truths)]
    wide = g.pivot_table(index=["case_idx", "episode_id", "family", "rung"],
                         columns="truth", values="obs", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide.sort_values("case_idx").reset_index(drop=True)


def hrrr_table(paired: pd.DataFrame, truths=ALL_TRUTHS,
               metrics=HRRR_METRICS) -> pd.DataFrame:
    """Headline paired rows (all cases) per board row and truth: the HRRR question."""
    out = []
    for order, (source, est) in enumerate(board_rows(paired)):
        head, _ = variants_for(paired, source, est)
        m = paired[(paired.source == source) & (paired.estimate == est)
                   & (paired.variant == head) & (paired.aires_variant == "sn")
                   & paired.truth.isin(truths) & (paired.subset == "all")
                   & paired.metric.isin(metrics)].copy()
        m.insert(0, "row", order)
        out.append(m)
    if not out:
        return pd.DataFrame(columns=["row"] + list(paired.columns))
    return pd.concat(out, ignore_index=True)


# ---- numbers CSV ------------------------------------------------------------ #
NUM_COLS = ["figure", "panel", "truth", "source", "label", "estimate", "role", "variant",
            "aires_variant", "window", "subset", "metric", "stat", "episode_id", "family",
            "value", "ci_lo", "ci_hi", "n", "win", "tie", "loss", "wilcoxon_p",
            "aires_value", "model_value", "t_colour", "note"]


def numbers_table(board: dict, analysis: Path = ANALYSIS) -> pd.DataFrame:
    paired, cases = board["paired"], board["cases"]
    parts = []
    ft = forest_table(paired)
    if len(ft):
        parts.append(pd.DataFrame(dict(
            figure="forest", panel=ft.metric, truth=ft.truth, source=ft.source,
            label=ft.label, estimate=ft.estimate, role=ft.role, variant=ft.variant,
            aires_variant=ft.aires_variant, window=ft.window, subset=ft.subset,
            metric=ft.metric, stat="paired mean (positive = AI+RES better)",
            value=ft["mean"], ci_lo=ft.ci_lo, ci_hi=ft.ci_hi, n=ft.n, win=ft.win,
            tie=ft.tie, loss=ft.loss, wilcoxon_p=ft.wilcoxon_p, aires_value=ft.mean_aires,
            model_value=ft.mean_model)))
    sb = scoreboard_table(cases, paired, analysis=analysis)
    if len(sb):
        parts.append(pd.DataFrame(dict(
            figure="scoreboard", panel=sb.column, truth=sb.truth, source=sb.source,
            label=sb.label, estimate=sb.estimate, role="headline", variant=sb.variant,
            window=sb.window, subset="all", metric=sb.column, stat=sb.stat, value=sb.value,
            n=sb.n, wilcoxon_p=sb.wilcoxon_p, aires_value=sb.aires_value, model_value=sb.value,
            t_colour=sb.t, note=sb.land_from.map(
                lambda s: f"maps_land_means_{s}.json" if isinstance(s, str) else ""))))
    ob = obs_table(cases)
    for t in ALL_TRUTHS:
        if t in ob:
            parts.append(pd.DataFrame(dict(
                figure="hrrr_vs_era5", panel="a", truth=t, source="obs", label=TRUTH_SHORT[t],
                window="13f", subset=ob.family, metric="obs_A_L", stat="observed A_L (K)",
                episode_id=ob.episode_id, family=ob.family, value=ob[t])))
    ht = hrrr_table(paired)
    if len(ht):
        parts.append(pd.DataFrame(dict(
            figure="hrrr_vs_era5", panel=ht.metric.map({"logratio": "b", "dcrps": "c"}),
            truth=ht.truth, source=ht.source, label=ht.label, estimate=ht.estimate,
            role="headline", variant=ht.variant, aires_variant=ht.aires_variant,
            window=ht.window, subset=ht.subset, metric=ht.metric,
            stat="paired mean (positive = AI+RES better)", value=ht["mean"],
            ci_lo=ht.ci_lo, ci_hi=ht.ci_hi, n=ht.n, win=ht.win, tie=ht.tie, loss=ht.loss,
            wilcoxon_p=ht.wilcoxon_p, aires_value=ht.mean_aires, model_value=ht.mean_model)))
    if not parts:
        return pd.DataFrame(columns=NUM_COLS)
    df = pd.concat(parts, ignore_index=True).reindex(columns=NUM_COLS)
    # only drawn numbers: a NaN (BB-SUBS P(obs), a missing map) is never plotted
    return df[np.isfinite(df.value.astype(float))].reset_index(drop=True)


def write_numbers(board: dict, out: Path = NUMBERS_CSV, analysis: Path = ANALYSIS) -> Path:
    df = numbers_table(board, analysis)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"[overall] {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} "
          f"({len(df)} rows)", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "xtick.labelcolor": INK, "ytick.labelcolor": INK,
                         "axes.labelcolor": INK, "text.color": INK})
    return plt


def tint(color: str, f: float = 0.55) -> str:
    """`color` mixed a fraction `f` toward white (the lighter, secondary partner)."""
    c = color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(v + (255 - v) * f):02x}" for v in rgb)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def wtl(r) -> str:
    return f"{int(r.win)}-{int(r.tie)}-{int(r.loss)}"


def _wrap_notes(notes: list[str], width_in: float, fontsize: float) -> str:
    """Footnote paragraphs wrapped to `width_in` inches (one paragraph per note), so a long
    note never widens the canvas past the panels. Numbers stay with their units and
    '|A_L| >= 2 K' stays on one line."""
    n = max(40, int(width_in * 72 / (0.53 * fontsize)))     # ~0.51 em per char, measured
    out = []
    for t in notes:
        t = re.sub(r"(\d) K\b", "\\1\u00a0K", t).replace(" >= ", "\u00a0>=\u00a0")
        out.append(textwrap.fill(t, n))
    return "\n".join(out)


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.png")
    fig.savefig(tmp, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    tmp.replace(out)
    print(f"[overall] {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}",
          flush=True)
    return out


def _axes_in(fig, W, H, left, bottom, width, height):
    """Axes placed in inches from the figure's lower-left corner."""
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def _pending(board: dict) -> list[str]:
    cov = board.get("summary", {}).get("coverage", {})
    return [label(s) for s in cov.get("absent", []) if s in MODEL_STYLE or s in _STYLE_FALLBACK]


def n_cases(board: dict) -> int:
    """Number of board cases (AI+RES rows of one truth and window)."""
    c = board["cases"]
    a = c[(c.source == AIRES) & (c.variant == "sn")]
    if a.empty:
        return int(board["paired"].n.max()) if len(board["paired"]) else 0
    return int(a.groupby(["truth", "window"]).episode_id.nunique().max())


def _rng(lo: float, hi: float) -> str:
    return f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"


def lead_text(board: dict) -> str:
    """'AI+RES lead 21 d (baselines 21-27 d)' from the coverage block: lagged and weekly-init
    baselines do not all start 21 d before the peak."""
    srcs = board.get("summary", {}).get("coverage", {}).get("sources", {})
    res = srcs.get(AIRES, {}).get("lead_days_to_peak") or [21.0, 21.0]
    other = [x for n, v in srcs.items() if n not in (AIRES,) + REFERENCE
             for x in (v.get("member_lead_days") or [])]
    s = f"AI+RES lead {_rng(*res)} d"
    if other and (min(other), max(other)) != tuple(res):
        s += f" (baseline members {_rng(min(other), max(other))} d)"
    return s


def headline_title(board: dict) -> str:
    return (f"{n_cases(board)} cases, {lead_text(board)}, selected on outcome "
            "(|A$_L$| >= 2 K)")


BIAS_TEXT = {"loyo": "leave-one-year-out", "reforecast": "reforecast climatology"}


def bias_note(board: dict) -> str:
    """'Bias correction = ...' for the rows present, from the board coverage."""
    srcs = board.get("summary", {}).get("coverage", {}).get("sources", {})
    groups: dict[str, list[str]] = {}
    for s, est in board_rows(board["paired"]):
        if est:
            continue
        b = srcs.get(s, {}).get("bias") or ("reforecast" if s == "ec46" else "loyo")
        groups.setdefault(BIAS_TEXT.get(b, b), []).append(label(s))
    if not groups:
        return ""
    return "Bias correction = " + "; ".join(f"{k} ({', '.join(v)})" for k, v in groups.items()) + "."


def _ref_cfs_text(paired: pd.DataFrame, truth: str = "era5") -> str:
    r = paired[(paired.source == "cfs") & (paired.truth == truth) & (paired.variant == "raw_emp")
               & (paired.subset == "all") & (paired.metric == "logratio") & ~paired.estimate]
    if r.empty:
        return ""
    r = r.iloc[0]
    return (f"Published 25-frame CFSv2 (reference, {TRUTH_SHORT[truth]}): log ratio "
            f"{r['mean']:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}], {wtl(r)}, {fmt_p(r.wilcoxon_p)}")


def _row_caption(cases: pd.DataFrame, source: str, est: bool, sep: str = ", ") -> str:
    if est:
        return "estimate from" + ("\n" if "\n" in sep else " ") + "debiased EC46"
    n = n_native(cases, source, est)
    w = WINDOW_TEXT.get(source_window(cases, source, est), "")
    return f"{n} members{sep}{w}" if n else w


def sfmt(x: float, spec: str = "+.2f") -> str:
    """Signed format that never prints a negative zero."""
    v = 0.0 if abs(x) < 0.5 * 10 ** -int(spec[-2]) else x
    return format(v, spec)


# --------------------------------------------------------------------------- #
# Figure 1: forest
# --------------------------------------------------------------------------- #
FOREST_PANELS = (
    ("logratio", "(a) log[ P$_{AI+RES}$(obs) / P$_{model}$(obs) ]"),
    ("dbrier_3K", "(b) Brier(model) - Brier(AI+RES), 3 K"),
    ("dbrier_4K", "(c) Brier(model) - Brier(AI+RES), 4 K"),
    ("dcrps", "(d) CRPS(model) - CRPS(AI+RES)  (K)"),
)
SUB_MARK = {"all": "D", "heat": "^", "cold": "v"}
SUB_MS = {"all": 5.6, "heat": 6.6, "cold": 6.6}
# vertical offsets inside one subset row: (role, truth) -> dy
FOREST_DY = {("headline", "era5"): 0.27, ("headline", "hrrr"): 0.09,
             ("secondary", "era5"): -0.09, ("secondary", "hrrr"): -0.27}


def _forest_y(nrows: int, gap: float = 0.8):
    """y of each (row index, subset); rows top-down."""
    ys, top = {}, 0.0
    for i in range(nrows):
        for j, s in enumerate(SUBSETS):
            ys[(i, s)] = top - j
        top -= len(SUBSETS) + gap
    return ys


def _draw_point(ax, x, lo, hi, y, color, marker, ms, filled, light, dashed):
    c = tint(color, 0.55) if light else color
    ls = (0, (3, 2)) if dashed else "-"
    lw = 1.1 if light else 1.7
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=c, lw=lw, ls=ls, solid_capstyle="butt", zorder=3)
    ax.plot([x], [y], marker=marker, ms=ms if not light else ms * 0.85, ls="",
            mfc=c if filled else "white", mec=c, mew=1.3, zorder=4)


def fig_forest(board: dict, out: Path = FIG_DIR / "forest.png") -> Path | None:
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    paired, cases = board["paired"], board["cases"]
    ft = forest_table(paired)
    rows = board_rows(paired)
    if ft.empty or not rows:
        print("[overall] forest: no board rows - skipped", flush=True)
        return None
    ys = _forest_y(len(rows))
    gap = 0.8
    span = len(rows) * len(SUBSETS) + (len(rows) - 1) * gap
    unit = 0.50                                     # inches per subset row
    LEFT, PW, AW, GAP_P, RIGHT = 2.4, 2.55, 1.02, 0.22, 0.1
    TOP, BOT = 1.45, 1.35
    ph = span * unit
    W = LEFT + len(FOREST_PANELS) * (PW + AW) + (len(FOREST_PANELS) - 1) * GAP_P + RIGHT
    H = TOP + ph + BOT
    fig = plt.figure(figsize=(W, H))
    y_lo, y_hi = -span + 0.5, 0.5
    axes = []
    for k, (metric, title) in enumerate(FOREST_PANELS):
        ax = _axes_in(fig, W, H, LEFT + k * (PW + AW + GAP_P), BOT, PW, ph)
        axes.append(ax)
        d = ft[ft.metric == metric]
        # bands
        for i, (s, est) in enumerate(rows):
            y0 = ys[(i, SUBSETS[-1])] - 0.5
            y1 = ys[(i, SUBSETS[0])] + 0.5
            if est:
                ax.axhspan(y0, y1, facecolor="white", edgecolor=tint(style(s)["color"], 0.45),
                           hatch="////", lw=0, zorder=0)
            elif i % 2 == 0:
                ax.axhspan(y0, y1, color=BAND, lw=0, zorder=0)
        vals = []
        for i, (s, est) in enumerate(rows):
            col = style(s)["color"]
            dr = d[(d.source == s) & (d.estimate == est)]
            for sub in SUBSETS:
                for role in ("secondary", "headline"):
                    for truth in FIG_TRUTHS:
                        r = dr[(dr.role == role) & (dr.truth == truth) & (dr.subset == sub)]
                        if r.empty or not np.isfinite(r["mean"].iloc[0]):
                            continue
                        r = r.iloc[0]
                        y = ys[(i, sub)] + FOREST_DY[(role, truth)]
                        _draw_point(ax, r["mean"], r.ci_lo, r.ci_hi, y, col, SUB_MARK[sub],
                                    SUB_MS[sub], truth == "era5", role == "secondary", est)
                        vals += [r["mean"], r.ci_lo, r.ci_hi]
                # annotation: headline W-T-L and Wilcoxon p, E = ERA5, H = HRRR
                for truth, dy in (("era5", 0.19), ("hrrr", -0.19)):
                    r = dr[(dr.role == "headline") & (dr.truth == truth) & (dr.subset == sub)]
                    if r.empty or not np.isfinite(r["mean"].iloc[0]):
                        continue
                    r = r.iloc[0]
                    sig = np.isfinite(r.wilcoxon_p) and r.wilcoxon_p < P_SIG
                    ax.annotate(f"{TRUTH_SHORT[truth][0]} {wtl(r)} {fmt_p(r.wilcoxon_p)}",
                                xy=(1, ys[(i, sub)] + dy), xycoords=("axes fraction", "data"),
                                xytext=(5, 0), textcoords="offset points", ha="left",
                                va="center", fontsize=6.6, color=INK if sig else INK3,
                                fontweight="bold" if sig else "normal",
                                annotation_clip=False)
        v = _fin(vals)
        lo, hi = (min(v.min(), 0.0), max(v.max(), 0.0)) if v.size else (-1.0, 1.0)
        pad = 0.07 * (hi - lo if hi > lo else 1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(y_lo, y_hi)
        ax.axvline(0, color=INK, lw=0.9, zorder=2)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=1)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, fontsize=9, loc="left", pad=6)
        ax.set_xlabel("positive = AI+RES better", fontsize=8, color=INK2)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_yticks([ys[(i, s)] for i in range(len(rows)) for s in SUBSETS])
        if k == 0:
            labs = []
            for i, (s, est) in enumerate(rows):
                for sub in SUBSETS:
                    n = ft[(ft.source == s) & (ft.estimate == est) & (ft.subset == sub)
                           & (ft.truth == "era5") & (ft.role == "headline")].n
                    n = int(n.max()) if len(n) and np.isfinite(n.max()) else None
                    labs.append(f"{sub}" + (f" ({n})" if n else ""))
            ax.set_yticklabels(labs, fontsize=7.5)
            for i, (s, est) in enumerate(rows):
                yc = ys[(i, "heat")]
                ax.annotate(label(s, est), xy=(0, yc + 0.42), xycoords=("axes fraction", "data"),
                            xytext=(-58, 0), textcoords="offset points", ha="right",
                            va="center", fontsize=9.5, fontweight="bold", color=INK,
                            annotation_clip=False)
                ax.annotate(_row_caption(cases, s, est, "\n"), xy=(0, yc - 0.22),
                            xycoords=("axes fraction", "data"), xytext=(-58, 0),
                            textcoords="offset points", ha="right", va="center",
                            fontsize=7, color=INK2, linespacing=1.35,
                            annotation_clip=False)
            # model colour bar at the far left of each band
            for i, (s, est) in enumerate(rows):
                y0 = ys[(i, SUBSETS[-1])] - 0.42
                y1 = ys[(i, SUBSETS[0])] + 0.42
                ax.plot([-0.27, -0.27], [y0, y1], transform=ax.get_yaxis_transform(),
                        color=style(s)["color"], lw=3.0, solid_capstyle="butt",
                        ls=(0, (2, 1.5)) if est else "-", clip_on=False)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        # BB-SUBS rows with no estimate for this metric
        for i, (s, est) in enumerate(rows):
            if not est:
                continue
            dr = d[(d.source == s) & (d.estimate == est)]
            if dr.empty or not np.isfinite(dr["mean"]).any():
                ax.text(0.5, ys[(i, "heat")], "not estimated", transform=ax.get_yaxis_transform(),
                        ha="center", va="center", fontsize=7.5, color=INK2, style="italic",
                        zorder=5, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))

    # legend (neutral ink: colour carries the model, the legend carries the encodings)
    g = "#666666"
    h = [Line2D([], [], marker="o", ls="", mfc=g, mec=g, ms=6, label="ERA5 truth"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=g, mew=1.3, ms=6,
                label="HRRR truth (offset-corrected)"),
         Line2D([], [], color=g, lw=1.7, label="raw ERA5-clim anomaly (headline)"),
         Line2D([], [], color=tint(g, 0.55), lw=1.1, label="bias-corrected (sensitivity)"),
         Line2D([], [], marker="D", ls="", color=g, ms=5, label="all cases"),
         Line2D([], [], marker="^", ls="", color=g, ms=6, label="heat"),
         Line2D([], [], marker="v", ls="", color=g, ms=6, label="cold")]
    if any(e for _, e in rows):
        h.append(Line2D([], [], color=g, lw=1.4, ls=(0, (3, 2)), label="estimate (BB-SUBS)"))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.78 / H),
               ncol=len(h), frameon=False, fontsize=8, handlelength=1.8, columnspacing=1.4)
    fig.text(0.5, 1 - 0.22 / H, "AI+RES vs operational S2S ensembles: " + headline_title(board),
             ha="center", va="top", fontsize=12.5,
             fontweight="bold")
    fig.text(0.5, 1 - 0.52 / H, "Paired per case against AI+RES on the same truth and window. "
             "Marker = case mean, bar = 90% case-bootstrap CI; > 0 = AI+RES better.  "
             "Right of each panel: wins-ties-losses and Wilcoxon p of the headline "
             "(E = ERA5 truth, H = HRRR truth; bold = p < 0.05).",
             ha="center", va="top", fontsize=8.5, color=INK2)
    notes = ["Windows: instantaneous sources on the 13 00/12Z frames peak-6 d..peak; daily-mean "
             "sources (GEPS, EC46) on UTC days peak-6..peak-1, with truth and AI+RES re-reduced "
             "to the matching 12 frames. Native ensemble sizes; the 16-member subsample "
             "sensitivity is in board_paired.csv (variants e16_*).",
             "Selection: every case has |A_L| >= 2 K, so P(obs) and Brier reward mass on the "
             "observed tail rather than calibration, and cases share seasons, so the CIs are "
             "optimistic. " + TILT_NOTE,
             bias_note(board)]
    extra = [x for x in (_ref_cfs_text(paired),) if x]
    pend = _pending(board)
    if pend:
        extra.append("Pending (no data yet): " + ", ".join(pend) + ".")
    if extra:
        notes.append("; ".join(extra))
    fig.text(0.15 / W, 0.62 / H, _wrap_notes(notes, W - 0.3, 7.4), ha="left", va="top",
             fontsize=7.4, color=INK2, linespacing=1.5)
    path = _save(fig, out)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 2: scoreboard
# --------------------------------------------------------------------------- #
FMT = {"p_obs": "{:.3f}", "lift": "{:.2f}", "brier_2K": "{:.3f}", "brier_3K": "{:.3f}",
       "brier_4K": "{:.3f}", "crps": "{:.2f}", "rmse_mean": "{:.2f}", "field_rmse": "{:.2f}",
       "pattern_r": "{:.2f}", "bss_p2": "{:.2f}", "bss_m2": "{:.2f}", "csi_p2": "{:.2f}",
       "csi_m2": "{:.2f}"}
ARROW = {+1: "↑", -1: "↓"}


def fmt_val(key: str, v: float) -> str:
    """Scoreboard cell text: the column's format, '-' for a missing value, no '-0.00'."""
    if not np.isfinite(v):
        return "-"
    t = FMT[key].format(v)
    return t[1:] if t.startswith("-") and float(t) == 0 else t


def _cell_rgb(t: float, cmap):
    if not np.isfinite(t):
        return (1, 1, 1, 1)
    return cmap(0.5 + 0.30 * t)


def _lum(c) -> float:
    r, g, b = c[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def fig_scoreboard(board: dict, out: Path = FIG_DIR / "scoreboard.png",
                   analysis: Path = ANALYSIS) -> Path | None:
    plt = _plt()
    from matplotlib import colormaps
    from matplotlib.patches import Rectangle
    paired, cases = board["paired"], board["cases"]
    sb = scoreboard_table(cases, paired, analysis=analysis)
    if sb.empty:
        print("[overall] scoreboard: empty board - skipped", flush=True)
        return None
    cmap = colormaps["RdBu_r"]
    cols = [c[0] for c in SCORE_COLS]
    ncol = len(cols)
    truths = [t for t in FIG_TRUTHS if (sb.truth == t).any()]
    blocks = []
    for t in truths:
        s = sb[sb.truth == t]
        keys = list(dict.fromkeys(zip(s.source, s.estimate, s.window)))
        res_w = [k for k in keys if k[0] == AIRES]
        mods = [k for k in keys if k[0] != AIRES]
        blocks.append((t, res_w + mods))
    CW, RH, LEFT, HEAD = 0.80, 0.34, 2.35, 0.58
    TOP, BOT, BGAP = 0.78, 0.8, 0.34
    nr = sum(len(b[1]) for b in blocks)
    H = TOP + BOT + sum(HEAD + len(b[1]) * RH for b in blocks) + BGAP * (len(blocks) - 1)
    W = LEFT + ncol * CW + 0.25
    fig = plt.figure(figsize=(W, H))
    y_top = H - TOP
    for t, keys in blocks:
        s = sb[sb.truth == t]
        h = HEAD + len(keys) * RH
        ax = _axes_in(fig, W, H, LEFT, y_top - h, ncol * CW, h)
        ax.set_xlim(0, ncol)
        ax.set_ylim(len(keys) + HEAD / RH, 0)
        ax.axis("off")
        ax.text(-LEFT / CW + 0.05, 0.05, f"Truth: {TRUTH_TEXT[t]}", ha="left", va="top",
                fontsize=10, fontweight="bold")
        for j, (key, head, kind, sign, _) in enumerate(SCORE_COLS):
            ax.text(j + 0.5, HEAD / RH - 0.12, f"{head} {ARROW[sign]}", ha="center",
                    va="bottom", fontsize=7.4, color=INK, linespacing=1.15)
        y0 = HEAD / RH
        for i, (src, est, win) in enumerate(keys):
            y = y0 + i
            r = s[(s.source == src) & (s.estimate == est) & (s.window == win)].set_index("column")
            is_res = src == AIRES
            name = "AI+RES" if is_res else label(src, est)
            sub = (RES_WINDOW_TEXT.get(win, win) if is_res
                   else _row_caption(cases, src, est))
            ax.text(-0.22, y + 0.36, name, ha="right", va="center", fontsize=8.6,
                    fontweight="bold")
            ax.text(-0.22, y + 0.74, sub, ha="right", va="center", fontsize=6.4, color=INK2)
            ax.add_patch(Rectangle((-0.17, y + 0.1), 0.09, 0.8, color=style(src)["color"],
                                   clip_on=False, lw=0))
            for j, key in enumerate(cols):
                v = float(r.loc[key, "value"]) if key in r.index else np.nan
                tt = float(r.loc[key, "t"]) if key in r.index else np.nan
                p = float(r.loc[key, "wilcoxon_p"]) if key in r.index else np.nan
                fc = (0.94, 0.94, 0.94, 1) if is_res else _cell_rgb(tt, cmap)
                if not np.isfinite(v):
                    fc = (1, 1, 1, 1)
                ax.add_patch(Rectangle((j + 0.03, y + 0.05), 0.94, 0.9, facecolor=fc,
                                       edgecolor="none", hatch=None))
                if est:
                    ax.add_patch(Rectangle((j + 0.03, y + 0.05), 0.94, 0.9, facecolor="none",
                                           edgecolor=tint(style(src)["color"], 0.3),
                                           hatch="////", lw=0))
                txt = fmt_val(key, v)
                if np.isfinite(p) and p < P_SIG:
                    txt += "*"
                ink = "white" if _lum(fc) < 0.45 else INK
                ax.text(j + 0.5, y + 0.52, txt, ha="center", va="center",
                        fontsize=8 if not is_res else 8.2, color=ink if np.isfinite(v) else INK3,
                        fontweight="bold" if is_res else "normal",
                        bbox=dict(boxstyle="square,pad=0.08", fc=(1, 1, 1, 0.75), ec="none")
                        if est and np.isfinite(v) else None)
        y_top -= h + BGAP
    fig.text(0.5, 1 - 0.2 / H, "Scoreboard: " + headline_title(board), ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(0.5, 1 - 0.5 / H, "Cell colour vs AI+RES on the same window: red = worse than "
             "AI+RES, blue = better (saturates at a factor of 2, or at 0.25 for r, BSS, CSI). "
             "* = paired Wilcoxon p < 0.05.",
             ha="center", va="top", fontsize=8.3, color=INK2)
    notes = ["Models: raw ERA5-clim anomaly, native ensemble. P(obs), lift = median over cases "
             "(lift = P(obs) / P_clim(obs) where P_clim > 0); Brier, CRPS, field RMSE, pattern r "
             "= case mean; ens-mean RMSE = root of the mean squared",
             "ensemble-mean error; field metrics = ensemble-mean 7-day T2m anomaly map vs truth "
             "over CONUS land; land BSS = maps-stage land median, land CSI = land mean, at "
             "+/-2 K. Every case has |A_L| >= 2 K, so P(obs) and Brier",
             "reward mass on the observed tail. " + TILT_NOTE + " AI+RES 12-frame row = the "
             "partner of the daily-mean sources."]
    pend = _pending(board)
    if pend:
        notes[-1] += " Pending (no data yet): " + ", ".join(pend) + "."
    # one paragraph re-wrapped to the table width (fixed breaks left a ragged last line)
    fig.text(0.12 / W, 0.6 / H, _wrap_notes([" ".join(notes)], W - 0.4, 7), ha="left",
             va="top", fontsize=7, color=INK2, linespacing=1.45)
    path = _save(fig, out)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 3: the HRRR question
# --------------------------------------------------------------------------- #
HRRR_MARK = {"era5": ("o", True), "hrrr": ("o", False), "hrrr_raw": ("s", False)}
HRRR_DY = {"era5": 0.26, "hrrr": 0.0, "hrrr_raw": -0.26}


def _mask_offset_text(path: Path = HRRR_SUMMARY_JSON) -> str:
    """Mean HRRR-minus-ERA5 A_L with ERA5 on the HRRR mask (the like-for-like comparison),
    from acal.hrrrtruth's case summary; empty when that file is absent."""
    try:
        sj = json.loads(path.read_text())
        c, r = (sj[k]["mean_offset_vs_era5_mask"] for k in ("hrrr_13f", "hrrr_raw_13f"))
    except (OSError, KeyError, ValueError, TypeError):
        return ""
    return (f" Against ERA5 on the HRRR mask the mean differences are {c:+.3f} K "
            f"(offset-corrected) and {r:+.3f} K (raw).")


def _hrrr_answer(ht: pd.DataFrame) -> str:
    def one(src, truth):
        r = ht[(ht.source == src) & ~ht.estimate & (ht.truth == truth) & (ht.metric == "logratio")]
        if r.empty:
            return None
        r = r.iloc[0]
        return r
    src = "cfs13" if (ht.source == "cfs13").any() else None
    if src is None:
        return ""
    e, h = one(src, "era5"), one(src, "hrrr")
    if e is None or h is None:
        return ""
    verdict = ("yes" if h.ci_lo > 0 else ("not significantly" if h["mean"] > 0 else "no"))
    return (f"Is AI+RES still better than CFSv2 against HRRR? {verdict}: log ratio "
            f"{h['mean']:+.2f} [{h.ci_lo:+.2f}, {h.ci_hi:+.2f}], {wtl(h)}, {fmt_p(h.wilcoxon_p)} "
            f"(ERA5 truth {e['mean']:+.2f} [{e.ci_lo:+.2f}, {e.ci_hi:+.2f}])")


def fig_hrrr(board: dict, out: Path = FIG_DIR / "hrrr_vs_era5.png") -> Path | None:
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    paired, cases = board["paired"], board["cases"]
    ob = obs_table(cases)
    ht = hrrr_table(paired)
    if ob.empty or "era5" not in ob or ht.empty:
        print("[overall] hrrr_vs_era5: no data - skipped", flush=True)
        return None
    rows = board_rows(paired)
    nrow = len(rows)
    AW_, PW_, ANW = 3.9, 2.7, 1.5
    LEFT, GAPA, GAPB, TOP, BOT = 0.75, 1.85, 0.3, 1.35, 1.35
    ph = max(3.9, nrow * 0.95)
    W = LEFT + AW_ + GAPA + 2 * (PW_ + ANW) + GAPB + 0.1
    H = TOP + ph + BOT
    fig = plt.figure(figsize=(W, H))
    # (a) scatter
    ax = _axes_in(fig, W, H, LEFT, BOT + (ph - AW_) / 2, AW_, AW_)
    x = ob.era5.to_numpy()
    allv = np.concatenate([x] + [ob[t].to_numpy() for t in ("hrrr", "hrrr_raw") if t in ob])
    lim = float(np.nanmax(np.abs(allv))) * 1.08
    ax.axhspan(-2, 2, color=BAND, lw=0, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color=INK3, lw=0.8, zorder=1)
    ax.axhline(0, color=GRID, lw=0.7, zorder=1)
    ax.axvline(0, color=GRID, lw=0.7, zorder=1)
    col = style("hrrr")["color"]
    for fam, mk in (("heat", "^"), ("cold", "v")):
        m = (ob.family == fam).to_numpy()
        if "hrrr" in ob and "hrrr_raw" in ob:
            for xi, a, b in zip(x[m], ob.hrrr_raw.to_numpy()[m], ob.hrrr.to_numpy()[m]):
                ax.plot([xi, xi], [a, b], color=tint(col, 0.5), lw=0.7, zorder=2)
        if "hrrr_raw" in ob:
            ax.plot(x[m], ob.hrrr_raw.to_numpy()[m], mk, ms=5.5, mfc="white", mec=col,
                    mew=1.0, ls="", zorder=3)
        if "hrrr" in ob:
            ax.plot(x[m], ob.hrrr.to_numpy()[m], mk, ms=5.5, mfc=col, mec="white", mew=0.5,
                    ls="", zorder=4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("observed A$_L$, ERA5 full box (K)")
    ax.set_ylabel("observed A$_L$, HRRR mask (K)")
    ax.set_title("(a) observed 7-day anomaly per case", fontsize=9, loc="left", pad=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    lines = []
    for t in ("hrrr", "hrrr_raw"):
        if t in ob:
            d = ob[t].to_numpy() - x
            r = np.corrcoef(x, ob[t].to_numpy())[0, 1]
            below = int((np.abs(ob[t].to_numpy()) < 2).sum())
            lines.append(f"{TRUTH_SHORT[t]} minus ERA5 full box: {sfmt(np.mean(d))} K mean, "
                         f"r = {r:.3f}; |A$_L$| < 2 K in {below}/{len(x)}")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
    ax.text(lim * 0.97, 1.0, "|A$_L$| < 2 K", ha="right", va="center", fontsize=6.8,
            color=INK3)
    h = [Line2D([], [], marker="o", ls="", mfc=col, mec="white", ms=6,
                label="HRRR, offset-corrected"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=col, ms=6, label="HRRR raw"),
         Line2D([], [], marker="^", ls="", color=INK2, ms=6, label="heat"),
         Line2D([], [], marker="v", ls="", color=INK2, ms=6, label="cold")]
    ax.legend(handles=h, loc="lower right", fontsize=7, frameon=True, framealpha=0.95,
              edgecolor=GRID, handletextpad=0.3, borderpad=0.5)
    # (b), (c) paired per truth
    panels = (("logratio", "(b) log[ P$_{AI+RES}$(obs) / P$_{model}$(obs) ]"),
              ("dcrps", "(c) CRPS(model) - CRPS(AI+RES)  (K)"))
    for k, (metric, title) in enumerate(panels):
        bx = _axes_in(fig, W, H, LEFT + AW_ + GAPA + k * (PW_ + ANW + GAPB), BOT, PW_, ph)
        d = ht[ht.metric == metric]
        vals = []
        for i, (s, est) in enumerate(rows):
            yc = -i
            if i % 2 == 0:
                bx.axhspan(yc - 0.5, yc + 0.5, color=BAND, lw=0, zorder=0)
            if est:
                bx.axhspan(yc - 0.5, yc + 0.5, facecolor="white",
                           edgecolor=tint(style(s)["color"], 0.45), hatch="////", lw=0, zorder=0)
            c = style(s)["color"]
            dr = d[(d.source == s) & (d.estimate == est)]
            drew = False
            for t in ALL_TRUTHS:
                r = dr[dr.truth == t]
                if r.empty or not np.isfinite(r["mean"].iloc[0]):
                    continue
                r = r.iloc[0]
                mk, filled = HRRR_MARK[t]
                _draw_point(bx, r["mean"], r.ci_lo, r.ci_hi, yc + HRRR_DY[t], c, mk, 6.2,
                            filled, False, est)
                vals += [r["mean"], r.ci_lo, r.ci_hi]
                sig = np.isfinite(r.wilcoxon_p) and r.wilcoxon_p < P_SIG
                for dx, txt in ((5, TRUTH_SHORT[t]), (51, f"{wtl(r)} {fmt_p(r.wilcoxon_p)}")):
                    bx.annotate(txt, xy=(1, yc + HRRR_DY[t]), xycoords=("axes fraction", "data"),
                                xytext=(dx, 0), textcoords="offset points", ha="left",
                                va="center", fontsize=6.8, color=INK if sig else INK3,
                                fontweight="bold" if sig else "normal", annotation_clip=False)
                drew = True
            if not drew:
                bx.text(0.5, yc, "not estimated", transform=bx.get_yaxis_transform(),
                        ha="center", va="center", fontsize=7.5, color=INK2, style="italic",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
        v = _fin(vals)
        lo, hi = (min(v.min(), 0.0), max(v.max(), 0.0)) if v.size else (-1.0, 1.0)
        pad = 0.07 * (hi - lo if hi > lo else 1.0)
        bx.set_xlim(lo - pad, hi + pad)
        bx.set_ylim(-nrow + 0.5, 0.5)
        bx.axvline(0, color=INK, lw=0.9, zorder=2)
        bx.grid(axis="x", color=GRID, lw=0.6, zorder=1)
        bx.set_axisbelow(True)
        for sp in ("top", "right"):
            bx.spines[sp].set_visible(False)
        bx.set_title(title, fontsize=9, loc="left", pad=6)
        bx.set_xlabel("positive = AI+RES better", fontsize=8, color=INK2)
        bx.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        bx.set_yticks([-i for i in range(nrow)])
        bx.tick_params(axis="y", length=0, pad=13)
        if k == 0:
            bx.set_yticklabels([label(s, e) for s, e in rows], fontsize=8.6, fontweight="bold")
            for i, (s, e) in enumerate(rows):
                bx.plot([-0.03, -0.03], [-i - 0.42, -i + 0.42], transform=bx.get_yaxis_transform(),
                        color=style(s)["color"], lw=3.0, solid_capstyle="butt", clip_on=False,
                        ls=(0, (2, 1.5)) if e else "-")
        else:
            bx.set_yticklabels([])
    g = "#666666"
    hh = [Line2D([], [], marker="o", ls="", mfc=g, mec=g, ms=6, label="ERA5 truth"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=g, mew=1.3, ms=6,
                 label="HRRR truth (offset-corrected)"),
          Line2D([], [], marker="s", ls="", mfc="white", mec=g, mew=1.3, ms=6,
                 label="HRRR raw truth (sensitivity)")]
    fig.legend(handles=hh, loc="upper center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=((LEFT + AW_ + GAPA + PW_ + ANW / 2) / W, 1 - 0.78 / H))
    fig.text(0.5, 1 - 0.2 / H, "Does the verdict survive HRRR truth? "
             + headline_title(board).replace("2 K)", "2 K under ERA5)"), ha="center", va="top",
             fontsize=12, fontweight="bold")
    ans = _hrrr_answer(ht)
    if ans:
        fig.text(0.5, 1 - 0.5 / H, ans, ha="center", va="top", fontsize=8.8, color=INK)
    notes = ["(a) Case A_L on the 13 00/12Z frames. HRRR offset-corrected = HRRR minus its "
             "leave-one-year-out offset to ERA5, fitted on HRRRv4 frames only (2020-12-26 to "
             "2026-08-31; acal.hrrrtruth); HRRR raw = no offset. HRRR A_L is the mean over the "
             "HRRR land mask, ERA5 A_L the full-box mean." + _mask_offset_text() + " Grey band: below the "
             "2 K selection threshold; thin lines join each case's raw and offset-corrected "
             "HRRR value.",
             "(b, c) Headline variant (raw ERA5-clim anomaly, native ensemble), all cases, "
             "paired per case vs AI+RES on the same truth and window; mean, 90% case-bootstrap "
             "CI; text = wins-ties-losses, Wilcoxon p (bold = p < 0.05). Cases selected on "
             "outcome (|A_L| >= 2 K under ERA5). " + TILT_NOTE]
    pend = _pending(board)
    if pend:
        notes.append("Pending (no data yet): " + ", ".join(pend) + ".")
    fig.text(0.12 / W, 0.66 / H, _wrap_notes(notes, W - 0.3, 7.3), ha="left", va="top",
             fontsize=7.3, color=INK2, linespacing=1.5)
    path = _save(fig, out)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
STAGES = ("numbers", "forest", "scoreboard", "hrrr", "all")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--board-dir", type=Path, default=BOARD_DIR)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    a = ap.parse_args(argv)
    if not (Path(a.board_dir) / "board_paired.csv").exists():
        print(f"[overall] no board at {a.board_dir}: run `python -m acal.board --stage all` "
              "first", flush=True)
        return 1
    board = load_board(a.board_dir)
    pend = _pending(board)
    if pend:
        print(f"[overall] absent from the board (no data yet): {', '.join(pend)}", flush=True)
    if a.stage in ("numbers", "all"):
        write_numbers(board, Path(a.board_dir) / NUMBERS_CSV.name)
    if a.stage in ("forest", "all"):
        fig_forest(board, Path(a.fig_dir) / "forest.png")
    if a.stage in ("scoreboard", "all"):
        fig_scoreboard(board, Path(a.fig_dir) / "scoreboard.png")
    if a.stage in ("hrrr", "all"):
        fig_hrrr(board, Path(a.fig_dir) / "hrrr_vs_era5.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
