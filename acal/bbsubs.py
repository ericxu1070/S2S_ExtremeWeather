#!/usr/bin/env python
"""BB-SUBS (Brightband) on the acal board: published context + an ESTIMATE, never a forecast.

BB-SUBS forecasts are not public (free pilot only), so the board can only carry a BB-SUBS
number derived from Brightband's own published skill ratio over debiased EC46, applied to OUR
bias-corrected EC46 scores on the 42 cases. Everything here is labelled "estimate".

Stages (LOGIN NODE, CPU only, seconds)
--------------------------------------
    context   figures/acal/overall/bbsubs_published.png - the published numbers
              (AI Weather Quest RPSS, Brightband MSESS), "not our cases". Runs today.
    estimate  Tier 1: BB-SUBS est = k x EC46(corr) error metric, per case, from
              runs/acal/analysis/s2s/board/board_cases.csv -> board/bbsubs_rows.csv
              (estimate=True). A NO-OP with a message until EC46 exists
              (runs/acal/s2s/ec46/*.json AND ec46 rows in board_cases.csv).
              Tier 2 (case P(obs) via a Gaussian signal-noise model) runs only when the
              model reproduces EC46's own P(obs) (`tier2_validate`); otherwise it records
              "Tier 2 failed validation" in board/bbsubs_tier2_validation.json.
    method    board/bbsubs_method.md - assumptions and the k derivation, for the summary.

    python -m acal.bbsubs --stage {context,estimate,method,all}

Published numbers live in acal/data/{bbsubs_published,aiwq_tas_period_rpss}.csv (tidy copies
with a source_url column; checked against the raw pages on 2026-10-07). They are data.

k derivation (`derive_k`)
-------------------------
    MSE    MSESS = 1 - MSE/MSE_clim on the same truth and region, so
           MSE_BB / MSE_EC46 = (1 - MSESS_BB)/(1 - MSESS_EC46) = 0.71/0.81 = 0.877 (week 3).
           Band from the stated gap +0.10 +/- 0.02: 1 - [0.08, 0.12]/0.81 = [0.852, 0.901].
    CRPS   linear in error: a calibrated Gaussian's CRPS is proportional to its sigma, so
           k_CRPS = sqrt(k_MSE) = 0.936, band [0.923, 0.949]. Also used for RMSE/MAE.
    Brier  RPSS = 1 - RPS/RPS_clim, Brier = two-category RPS, so
           k_Brier = (1 - 0.112)/(1 - 0.062) = 0.947 (AIWQ DJF 2025/26 days 19-25). No CI is
           published; the band is the cross-metric range [k_MSE, k_Brier] = [0.877, 0.947].
    Skill scores vs climatology transform as SS_BB = 1 - k (1 - SS_EC46).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
PUBLISHED_CSV = DATA / "bbsubs_published.csv"
AIWQ_CSV = DATA / "aiwq_tas_period_rpss.csv"
EC46_DIR = ROOT / "runs" / "acal" / "s2s" / "ec46"
BOARD_DIR = ROOT / "runs" / "acal" / "analysis" / "s2s" / "board"
BOARD_CASES = BOARD_DIR / "board_cases.csv"
ROWS_CSV = BOARD_DIR / "bbsubs_rows.csv"
TIER2_JSON = BOARD_DIR / "bbsubs_tier2_validation.json"
METHOD_MD = BOARD_DIR / "bbsubs_method.md"
CONUS_DAILY = ROOT / "runs" / "acal" / "catalog" / "conus_daily_2021_2025.csv"
FIG = ROOT / "figures" / "acal" / "overall" / "bbsubs_published.png"

SOURCE = "bbsubs"
LABEL = "BB-SUBS (est.)"
ANCHOR = "ec46"
WEEK = 3                      # our window (days 15-21) ~ Brightband "week 3"
K_EXPECTED = {3: dict(mse=0.877, crps=0.936, brier=0.947),
              4: dict(mse=0.926, crps=0.962, brier=0.963)}

# Board palette (Okabe-Ito). acal.s2sbase.MODEL_STYLE wins when it defines the key.
_STYLE = {
    "aires": dict(label="AI+RES", color="#D55E00"),
    "cfs": dict(label="CFSv2", color="#0072B2"),
    "ec46": dict(label="ECMWF IFS (EC46)", color="#009E73"),
    "gefs": dict(label="GEFSv12", color="#CC79A7"),
    "geps": dict(label="ECCC GEPS", color="#E69F00"),
    "bbsubs": dict(label="BB-SUBS (est.)", color="#56B4E9", hatch="//", ls="--"),
}
C_OTHER, C_TEXT, C_MUTED, C_GRID = "#9A9A9A", "#222222", "#5A5A5A", "#E3E3E3"


def style(key: str) -> dict:
    out = dict(_STYLE[key])
    try:
        from acal.s2sbase import MODEL_STYLE  # owned by the score lane; may not exist yet
        s = MODEL_STYLE.get(key) or (MODEL_STYLE.get("cfs13") if key == "cfs" else None)
        if isinstance(s, dict):
            out.update({k: v for k, v in s.items() if k in ("color", "hatch")})
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Published numbers and the k derivation
def load_published(path: Path = PUBLISHED_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def load_aiwq(path: Path = AIWQ_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def _pub(pub: pd.DataFrame, chart: str, model_prefix: str, week: int) -> pd.Series:
    m = pub[(pub.chart == chart) & pub.model.str.startswith(model_prefix) & (pub.week == week)]
    if chart == "msess_brightband":
        m = m[m.metric == "MSESS"]
    if len(m) != 1:
        raise ValueError(f"expected one {chart} row for {model_prefix} week {week}, got {len(m)}")
    return m.iloc[0]


def derive_k(pub: pd.DataFrame | None = None) -> dict[int, dict[str, float]]:
    """k per week and score family from the published numbers (see the module docstring)."""
    pub = load_published() if pub is None else pub
    out = {}
    for wk in (3, 4):
        bb = float(_pub(pub, "msess_brightband", "BB-SUBS", wk).value)
        ec = float(_pub(pub, "msess_brightband", "EC46", wk).value)
        gap = pub[(pub.chart == "msess_brightband") & (pub.metric == "MSESS difference")
                  & (pub.week == wk)].iloc[0]
        k_mse = (1 - bb) / (1 - ec)
        lo = 1 - (gap.value + gap.err_halfwidth) / (1 - ec)
        hi = 1 - (gap.value - gap.err_halfwidth) / (1 - ec)
        rbb = float(_pub(pub, "rpss_aiwq", "BB-SUBS", wk).value)
        rec = float(_pub(pub, "rpss_aiwq", "EC46", wk).value)
        k_b = (1 - rbb) / (1 - rec)
        out[wk] = dict(mse=k_mse, mse_lo=lo, mse_hi=hi,
                       crps=np.sqrt(k_mse), crps_lo=np.sqrt(lo), crps_hi=np.sqrt(hi),
                       brier=k_b, brier_lo=min(k_mse, k_b), brier_hi=max(k_mse, k_b),
                       msess_bb=bb, msess_ec46=ec, rpss_bb=rbb, rpss_ec46=rec)
    for wk, exp in K_EXPECTED.items():
        for fam, v in exp.items():
            if abs(out[wk][fam] - v) > 5e-4:
                raise AssertionError(f"k_{fam} week {wk} = {out[wk][fam]:.4f}, expected {v}")
    return out


# --------------------------------------------------------------------------- #
# Tier 1: column classification and the estimate rows
FAMILIES = ("mse", "crps", "brier")
_KEEP_TOKENS = re.compile(r"(^|_)(res|aires|ai_res|clim|era5|hrrr|truth)(_|$)")
_META = {"episode_id", "eid", "case", "family", "rung", "peak", "init", "obs", "tail_sign",
         "season", "month", "year", "lead", "lead_days", "week", "window", "n_members",
         "members", "n_clim", "n_cases", "z", "a_l", "al_obs", "obs_al"}
_DROP = re.compile(r"(^|_)(d(brier|crps|mse|rmse|rps|bs)|delta|diff|logratio|log_ratio|wtl)"
                   r"(_|$)")
_CORR = re.compile(r"(^|_)corr(_|$)")      # variant embedded in the column name (CFS style)
_SS = [(re.compile(r"(^|_)msess(_|$)"), "ss_mse"), (re.compile(r"(^|_)crpss(_|$)"), "ss_crps"),
       (re.compile(r"(^|_)(bss|rpss)(_|$)"), "ss_brier")]
_ERR = [(re.compile(r"(^|_)(mse|se|sqerr|sq_err)(_|$)"), "mse"),
        (re.compile(r"(^|_)(crps|rmse|mae|abserr|abs_err|ae)(_|$)"), "crps"),
        (re.compile(r"(^|_)(brier|bs|rps)(_|$)"), "brier")]


def classify(name: str, numeric: bool = True) -> str:
    """'meta' (copied), 'drop' (set NaN: no estimate), an error family, or 'ss_<family>'."""
    c = name.lower()
    if not numeric or c in _META or c.startswith("o_") or _KEEP_TOKENS.search(c):
        return "meta"
    if _DROP.search(c):
        return "drop"
    for rx, fam in _SS + _ERR:
        if rx.search(c):
            return fam
    return "drop"


def _first(cols, names) -> str | None:
    return next((n for n in names if n in cols), None)


def anchor_rows(board: pd.DataFrame) -> tuple[pd.DataFrame, str, str | None]:
    """The bias-corrected EC46 rows of a board table, the source column and variant column."""
    src_col = _first(board.columns, ("source", "model", "src"))
    if src_col is None:
        raise ValueError("board table has no source/model column")
    var_col = _first(board.columns, ("variant", "bc", "correction"))
    b = board
    if "estimate" in b.columns:
        b = b[~b["estimate"].astype(str).str.lower().isin(("true", "1"))]
    b = b[b[src_col].astype(str).str.lower() == ANCHOR]
    if var_col is not None:
        b = b[b[var_col].astype(str).str.lower().str.contains("corr")]
    elif "bias_corrected" in b.columns:
        b = b[b["bias_corrected"].astype(bool)]
    elif not any(_CORR.search(str(c).lower()) for c in b.columns):
        b = b.iloc[0:0]                     # no bias-corrected EC46 anywhere: nothing to anchor
    return b, src_col, var_col


def _scale(v, fam: str, k: float):
    if fam.startswith("ss_"):
        return 1 - k * (1 - v)
    return k * v


def estimate_rows(board: pd.DataFrame, k: dict[str, float]) -> pd.DataFrame:
    """Tier 1 rows: one per bias-corrected EC46 row, error metrics scaled by k, every other
    forecast metric NaN (no tail probabilities, maps, AUC or CSI for an estimate).

    Wide tables (metric columns) and long tables (`metric`, `value` columns) are both handled.
    Added columns: estimate=True, tier, anchor, k_<family>, and <col>_lo / <col>_hi (wide) or
    k, value_lo, value_hi (long), computed with the k band ends k_lo / k_hi. For an error
    metric _lo is the smaller error; for a skill score (SS = 1 - k (1 - SS_EC46)) _lo is the
    HIGHER skill. Raw and 16-member-subsample EC46 are never scaled.
    """
    rows, src_col, var_col = anchor_rows(board)
    if rows.empty:
        return rows.copy()
    embedded = var_col is None and "bias_corrected" not in rows.columns
    out = rows.copy()
    anchor = (ANCHOR + "/" + rows[var_col].astype(str) if var_col
              else pd.Series(ANCHOR + "/corr", index=rows.index))
    if {"metric", "value"} <= set(out.columns):
        fam = out["metric"].astype(str).map(classify)
        keep = fam.isin(FAMILIES + tuple("ss_" + f for f in FAMILIES))
        out, fam = out[keep].copy(), fam[keep]
        base = fam.str.replace("ss_", "", regex=False)
        v = out["value"].astype(float)
        out["k"] = base.map(lambda f: k[f]).values
        for tag in ("", "_lo", "_hi"):
            kk = base.map(lambda f: k[f + tag]).values
            col = "value" + tag
            out[col] = [_scale(x, f, q) for x, f, q in zip(v, fam, kk)]
        out["k_family"] = base.values
    else:
        new = {}
        for c in out.columns:
            if c in (src_col, var_col, "label", "estimate"):
                continue
            fam = classify(c, pd.api.types.is_numeric_dtype(out[c]))
            if fam == "meta":
                continue
            if fam == "drop" or (embedded and not _CORR.search(c.lower())):
                out[c] = np.nan             # raw / subsample columns are never scaled
                continue
            base = fam.replace("ss_", "")
            v = out[c].astype(float)
            out[c] = _scale(v, fam, k[base])
            new[c + "_lo"] = _scale(v, fam, k[base + "_lo"])
            new[c + "_hi"] = _scale(v, fam, k[base + "_hi"])
        out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
        for f in FAMILIES:
            out["k_" + f] = k[f]
    out[src_col] = SOURCE
    out["label"] = LABEL
    out["estimate"] = True
    out["tier"] = "tier1"
    out["anchor"] = anchor.loc[out.index].values
    return out.reset_index(drop=True)


def ec46_json_count(ec46_dir: Path = EC46_DIR) -> int:
    return len(list(Path(ec46_dir).glob("*.json")))


def run_estimate(board_csv: Path = BOARD_CASES, ec46_dir: Path = EC46_DIR,
                 out_csv: Path = ROWS_CSV, week: int = WEEK, tier2: bool = True,
                 log=print) -> pd.DataFrame | None:
    """Write the Tier 1 rows. Returns None (and writes nothing) while EC46 is absent."""
    n_json = ec46_json_count(ec46_dir)
    if n_json == 0:
        log(f"[bbsubs] no-op: no EC46 case files ({ec46_dir}/*.json). The BB-SUBS estimate is "
            "anchored on bias-corrected EC46; rerun `python -m acal.bbsubs --stage estimate` "
            "after the EC46 token arrives and the board has ec46 rows.")
        return None
    if not Path(board_csv).exists():
        log(f"[bbsubs] no-op: {board_csv} does not exist yet (written by the board stage).")
        return None
    board = pd.read_csv(board_csv)
    k = derive_k()[week]
    est = estimate_rows(board, k)
    if est.empty:
        log(f"[bbsubs] no-op: {board_csv} has no bias-corrected ec46 rows yet "
            f"({n_json} EC46 json files exist; run the ec46 score/board stages first).")
        return None
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".tmp")
    est.to_csv(tmp, index=False)
    tmp.replace(out_csv)
    log(f"[bbsubs] wrote {out_csv} ({len(est)} estimate rows, k week {week}: "
        f"MSE {k['mse']:.3f}, CRPS {k['crps']:.3f}, Brier {k['brier']:.3f})")
    if tier2:
        run_tier2(board, out_dir=out_csv.parent, log=log)
    return est


# --------------------------------------------------------------------------- #
# Tier 2: Gaussian signal-noise model for case P(obs), only if validated on EC46 itself
def gauss_tail(rho2, z):
    """Expected forecast mass at/beyond a standardized observation z for a calibrated
    Gaussian forecast with MSESS rho2: Phi(-z sqrt((1 - rho2)/(1 + rho2)))."""
    from scipy.stats import norm
    rho2 = np.asarray(rho2, float)
    return norm.cdf(-np.abs(np.asarray(z, float)) * np.sqrt((1 - rho2) / (1 + rho2)))


def _boot_ci(x: np.ndarray, n_boot: int, rng, q=(0.05, 0.95)) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, q[0])), float(np.quantile(m, q[1]))


def tier2_validate(p_actual, z, rho2_ec46: float, rho2_pair=(0.29, 0.19),
                   n_boot: int = 4000, seed: int = 0) -> dict:
    """Does G(rho2_ec46, z_i) reproduce EC46's own P(obs) on the cases?

    Pass iff (1) the predicted case-mean P(obs) lies in the 90% case-bootstrap interval of the
    actual case-mean, in each half of the slate split at the median z (the z-dependence is
    what Tier 2 transfers); and (2) the overall level miss |log(mean_pred/mean_actual)| is
    smaller than log of the median Tier-2 adjustment it would apply, i.e. the model's own error
    must be smaller than the effect it is used to translate.
    """
    p = np.asarray(p_actual, float)
    z = np.abs(np.asarray(z, float))
    ok = np.isfinite(p) & np.isfinite(z)
    p, z = p[ok], z[ok]
    rng = np.random.default_rng(seed)
    g = gauss_tail(rho2_ec46, z)
    ratio = gauss_tail(rho2_pair[0], z) / gauss_tail(rho2_pair[1], z)
    tol = float(np.log(np.median(ratio)))
    halves = []
    med = np.median(z)
    for name, sel in (("low_z", z <= med), ("high_z", z > med)):
        if sel.sum() < 3:
            continue
        lo, hi = _boot_ci(p[sel], n_boot, rng)
        mp = float(g[sel].mean())
        halves.append(dict(half=name, n=int(sel.sum()), mean_actual=float(p[sel].mean()),
                           ci90=[lo, hi], mean_pred=mp, inside=bool(lo <= mp <= hi)))
    miss = float(abs(np.log(max(g.mean(), 1e-12) / max(p.mean(), 1e-12))))
    passed = bool(len(halves) == 2 and all(h["inside"] for h in halves) and miss <= tol)
    return dict(passed=passed, n=int(len(p)), rho2_ec46=float(rho2_ec46),
                mean_actual=float(p.mean()), mean_pred=float(g.mean()),
                log_level_miss=miss, log_tolerance=tol,
                median_tier2_ratio=float(np.median(ratio)), halves=halves)


def season_sigma(daily_csv: Path = CONUS_DAILY) -> dict[str, float]:
    """SD of the 7-day CONUS A_L by meteorological season, 2021-2025 (z_i denominator)."""
    d = pd.read_csv(daily_csv, parse_dates=["date"])
    s = d.date.dt.month.map(_season)
    return d.groupby(s)["a_l_conus"].std(ddof=1).to_dict()


def _season(m: int) -> str:
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA"}.get(int(m), "SON")


def ec46_rho2(hind_csv: Path = EC46_DIR / "hind.csv", sig: dict | None = None,
              peaks: dict | None = None) -> float | None:
    """Unconditional CONUS A_L MSESS-equivalent rho^2 of debiased EC46 from its reforecasts:
    squared Pearson correlation of (reforecast member-mean minus the case's leave-one-year-out
    bias) with ERA5 over all hindcast windows, both standardized by the season SD."""
    if not Path(hind_csv).exists():
        return None
    h = pd.read_csv(hind_csv, parse_dates=["peak"])
    g = h.groupby("episode_id")["diff"]
    n = g.transform("count")
    loyo = (g.transform("sum") - h["diff"]) / (n - 1)
    fc = h["al_rf_mean"] - loyo
    sig = season_sigma() if sig is None else sig
    s = h.peak.dt.month.map(_season).map(sig)
    r = np.corrcoef(fc / s, h["al_era5"] / s)[0, 1]
    return float(r * r)


def _p_obs_col(rows: pd.DataFrame, var_col: str | None) -> str | None:
    """EC46 P(obs) column: `p_obs` with a variant column, else a CFS-style embedded
    corr column (gauss preferred: smooth, no zero-member cases)."""
    if var_col is not None:
        return _first(rows.columns, ("p_obs", "p_obs_gauss", "p_obs_emp"))
    cand = [c for c in rows.columns if re.match(r"p_(\w+_)?obs", c) and _CORR.search(c)
            and not _KEEP_TOKENS.search(c)]
    return next((c for c in cand if "gauss" in c), cand[0] if cand else None)


def run_tier2(board: pd.DataFrame, hind_csv: Path | None = None, out_dir: Path | None = None,
              log=print) -> dict | None:
    rows, src_col, var_col = anchor_rows(board)
    pcol = _p_obs_col(rows, var_col)
    if rows.empty or pcol is None or "obs" not in rows.columns:
        log("[bbsubs] Tier 2 not run: the board rows carry no EC46 p_obs/obs columns.")
        return None
    hind_csv = Path(hind_csv) if hind_csv else EC46_DIR / "hind.csv"
    out_dir = Path(out_dir) if out_dir else BOARD_DIR
    rho2 = ec46_rho2(hind_csv)
    if rho2 is None:
        log(f"[bbsubs] Tier 2 not run: no EC46 reforecast table ({hind_csv}).")
        return None
    sig = season_sigma()
    if var_col is not None and rows[var_col].astype(str).str.contains("gauss").any():
        rows = rows[rows[var_col].astype(str).str.contains("gauss")]
    res = {}
    by = [c for c in ("truth",) if c in rows.columns]
    for key, r in (rows.groupby(by) if by else [("all", rows)]):
        r = r.drop_duplicates("episode_id") if "episode_id" in r.columns else r
        peaks = pd.to_datetime(r["peak"]) if "peak" in r.columns else None
        s = peaks.dt.month.map(_season).map(sig) if peaks is not None else np.nan
        z = (r["obs"].abs() / s).values
        v = tier2_validate(r[pcol].values, z, rho2)
        tkey = key[0] if isinstance(key, tuple) else key
        res[str(tkey)] = v
        if v["passed"]:
            t2 = r[[c for c in ("episode_id", "truth", "peak", "obs") if c in r.columns]].copy()
            t2["z"] = z
            t2["p_obs_ec46"] = r[pcol].values
            k = derive_k()[WEEK]
            for name, rb, re_ in (("pub_pair", 0.29, 0.19),
                                  ("fixed_mse", 1 - k["mse"] * (1 - rho2), rho2)):
                t2[f"ratio_{name}"] = gauss_tail(rb, z) / gauss_tail(re_, z)
                t2[f"p_obs_bb_est_{name}"] = t2["p_obs_ec46"] * t2[f"ratio_{name}"]
            t2["estimate"] = True
            t2.to_csv(out_dir / f"bbsubs_tier2_{tkey}.csv", index=False)
    status = {t: ("passed" if v["passed"] else "Tier 2 failed validation")
              for t, v in res.items()}
    js = out_dir / TIER2_JSON.name
    js.write_text(json.dumps(dict(status=status, rho2_ec46=rho2, p_obs_column=pcol,
                                  detail=res), indent=1))
    log(f"[bbsubs] Tier 2 validation: {status} -> {js}")
    return res


# --------------------------------------------------------------------------- #
# Context chart
def _panel_title(ax, letter: str, title: str, sub: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=13, fontweight="bold", pad=24,
                 color=C_TEXT)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=10, color=C_MUTED, va="bottom")


def _clean(ax, grid_axis: str) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBBB")
    ax.grid(axis=grid_axis, color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_TEXT, labelsize=10.5)


def context_figure(out: Path = FIG) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "hatch.linewidth": 1.0})
    pub, lb = load_published(), load_aiwq()
    k = derive_k()
    st = {key: style(key) for key in _STYLE}

    # ---- panel a data: Brightband's RPSS chart + the two board models from the leaderboard
    r = pub[pub.chart == "rpss_aiwq"].pivot_table(index="model", columns="week", values="value")
    r = r.rename(columns={3: "w3", 4: "w4"})
    keymap = dict(pub[pub.chart == "rpss_aiwq"].drop_duplicates("model")
                  .set_index("model").board_key.fillna(""))
    extra = lb[(lb.period == "DJF 2025") & lb.model.isin(["NOAA", "ECCC"])]
    for mdl, sub in extra.groupby("model"):
        name = {"NOAA": "CFSv2", "ECCC": "ECCC GEPS"}[mdl] + " *"
        r.loc[name] = [sub[sub.week == 3].rpss.iloc[0], sub[sub.week == 4].rpss.iloc[0]]
        keymap[name] = {"NOAA": "cfs", "ECCC": "geps"}[mdl]
    r = r.sort_values("w3")
    disp = {"BB-SUBS": "BB-SUBS (self-scored)", "EC46": "ECMWF IFS (EC46)",
            "S2S Multi-Model Mean": "S2S multi-model mean"}

    fig = plt.figure(figsize=(18, 8.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.95, 1.0], wspace=0.33,
                          left=0.115, right=0.985, top=0.80, bottom=0.235)
    ax, bx, cx = (fig.add_subplot(gs[0, i]) for i in range(3))

    y = np.arange(len(r))
    for yi, (name, row) in zip(y, r.iterrows()):
        key = keymap.get(name, "")
        c = st[key]["color"] if key else C_OTHER
        ax.plot([row.w4, row.w3], [yi, yi], color=c if key else "#C8C8C8", lw=2, zorder=2,
                ls="--" if key == "bbsubs" else "-")
        ax.scatter(row.w3, yi, s=95, color=c, zorder=3, edgecolor="white", lw=1.5)
        ax.scatter(row.w4, yi, s=80, facecolor="white", edgecolor=c, lw=2, zorder=3)
        if key:
            ax.text(max(row.w3, row.w4) + 0.008, yi, f"{row.w3:+.3f}", va="center",
                    fontsize=10.5, color=C_TEXT)
    ax.set_yticks(y, [disp.get(n, n) for n in r.index], fontsize=11)
    for t, n in zip(ax.get_yticklabels(), r.index):
        if keymap.get(n):
            t.set_fontweight("bold")
    ax.axvline(0, color="#777777", lw=1)
    ax.set_xlim(-0.11, 0.15)
    ax.set_ylim(-0.7, len(r) - 0.3)
    ax.set_xlabel("RPSS vs climatology (higher is better)", fontsize=11.5)
    _clean(ax, "x")
    _panel_title(ax, "a", "RPSS, global land, DJF 2025/26",
                 "AI Weather Quest quintile RPSS, 13 Thursday inits")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#444444", ms=9, label="week 3 (days 19-25)"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="#444444", mew=2, ms=8.5,
               label="week 4 (days 26-32)")], loc="lower right", fontsize=10, frameon=False)

    # ---- panel b: Brightband MSESS, 30-60N land, 3 winters
    ms = pub[(pub.chart == "msess_brightband") & (pub.metric == "MSESS")]
    w = 0.34
    for j, (key, pre, lab) in enumerate((("bbsubs", "BB-SUBS", "BB-SUBS v5.3 (self-reported)"),
                                         ("ec46", "EC46", "ECMWF IFS (EC46), debiased"))):
        s = ms[ms.model.str.startswith(pre)].sort_values("week")
        xs = np.arange(len(s)) + (j - 0.5) * (w + 0.04)
        c = st[key]["color"]
        kw = (dict(facecolor=to_rgba(c, 0.28), edgecolor=c, hatch=st[key]["hatch"], lw=1.6)
              if key == "bbsubs" else dict(facecolor=c, edgecolor=c, lw=1.6))
        bx.bar(xs, s.value, width=w, label=lab, zorder=2, **kw)
        bx.errorbar(xs, s.value, yerr=[s.value - s.err_lo, s.err_hi - s.value], fmt="none",
                    ecolor="#333333", elinewidth=1.2, capsize=4, zorder=3)
        for xx, v, hi in zip(xs, s.value, s.err_hi):
            bx.text(xx, hi + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5,
                    color=C_TEXT)
    bx.set_xticks([0, 1], ["week 3", "week 4"], fontsize=11.5)
    bx.set_ylim(0, 0.5)
    bx.set_xlim(-0.6, 1.6)
    bx.set_ylabel("MSESS vs climatology (higher is better)", fontsize=11.5)
    _clean(bx, "y")
    _panel_title(bx, "b", "MSESS, 30-60N land, winters 2023-26",
                 "Brightband chart; EC46 debiasing method not stated")
    bx.legend(loc="upper right", fontsize=10, frameon=False)
    k3 = k[3]
    bx.text(0.985, 0.80, "week-3 error ratio k, BB-SUBS / EC46\n"
            f"MSE (MSESS) (1-{k3['msess_bb']:.2f})/(1-{k3['msess_ec46']:.2f})   = {k3['mse']:.3f}\n"
            # the Brier-type k is the RPSS ratio of panel a (Brier = two-category RPS)
            f"RPS (RPSS)  (1-{k3['rpss_bb']:.3f})/(1-{k3['rpss_ec46']:.3f}) = {k3['brier']:.3f}",
            transform=bx.transAxes, ha="right", va="top", fontsize=9, color=C_TEXT,
            family="DejaVu Sans Mono", linespacing=1.5,
            bbox=dict(fc="white", ec="#CCCCCC", boxstyle="round,pad=0.45"))

    # ---- panel c: the same leaderboard across four seasons, week 3
    seasons = ["SON 2025", "DJF 2025", "MAM 2026", "JJA 2026"]
    ticks = ["SON\n2025", "DJF\n2025/26", "MAM\n2026", "JJA\n2026"]
    w3 = lb[lb.week == 3]
    series = [("multimodel_mean", "S2S multi-model mean", None), ("ECMWF", "ECMWF IFS (EC46)",
              "ec46"), ("ECCC", "ECCC GEPS", "geps"), ("NOAA", "CFSv2", "cfs")]
    for mdl, lab, key in series:
        s = w3[w3.model == mdl].set_index("period").reindex(seasons).rpss
        c = st[key]["color"] if key else C_OTHER
        cx.plot(range(4), s.values, color=c, lw=2, marker="o", ms=7, label=lab,
                mec="white", mew=1.2, zorder=3)
    bb = w3[w3.model == "BB-SUBS"].rpss.iloc[0]
    cx.scatter([1], [bb], marker="D", s=80, facecolor=to_rgba(st["bbsubs"]["color"], 0.35),
               edgecolor=st["bbsubs"]["color"], lw=2, hatch=st["bbsubs"]["hatch"], zorder=4,
               label="BB-SUBS (self-scored, DJF only)")
    cx.axhline(0, color="#777777", lw=1)
    cx.set_xticks(range(4), ticks, fontsize=10.5)
    cx.set_xlim(-0.3, 3.3)
    cx.set_ylim(-0.25, 0.2)
    cx.set_ylabel("RPSS, week 3 (days 19-25)", fontsize=11.5)
    _clean(cx, "y")
    _panel_title(cx, "c", "Week-3 RPSS by season",
                 "AIWQ leaderboard, global land; no ECCC entry in MAM/JJA")
    cx.legend(loc="lower left", fontsize=9.5, frameon=False, ncol=1)

    fig.text(0.012, 0.965, "BB-SUBS in context: skill published by Brightband and the AI Weather "
             "Quest, not computed on our 42 cases", fontsize=16, fontweight="bold", color=C_TEXT)
    fig.text(0.012, 0.925, "2 m temperature, weekly-mean skill vs climatology. BB-SUBS numbers "
             "are self-reported by the vendor (winter only, scored after the fact with the "
             "open AIWQ code); every other RPSS matches the official AIWQ leaderboard.",
             fontsize=11, color=C_MUTED)
    fig.text(0.012, 0.035,
             "* Official AIWQ leaderboard entries (S2S Database: NOAA = CFSv2, ECCC = GEPS), not "
             "on Brightband's chart. Gray = the other entries Brightband shows (best model per "
             "team). AIWQ EC46 uses reforecast-calibrated quintiles.\n"
             "MSESS error bars decoded from the chart, interval type not stated. Our acal window "
             "is days 15-21 (CONUS box mean, 42 extremes), earlier than AIWQ week 3; the board's "
             "BB-SUBS row is an estimate scaled from our EC46 by k.\n"
             "Sources: brightband.com/data/subs, "
             "brightband.com/company/news/introducing-brightband-subs, "
             "aiweatherquest.ecmwf.int/leaderboards (tidy copies with URLs: acal/data/).",
             fontsize=9.5, color=C_MUTED, linespacing=1.5)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Methods note
def slate_coverage() -> pd.DataFrame:
    """Each case's BB-SUBS coverage class from init = peak - 21 d (BB-SUBS inits daily 00z)."""
    from acal import aprep
    df = aprep.episodes()[["episode_id", "peak", "init"]].copy()
    init = pd.to_datetime(df.init)
    winter = init.dt.month.isin([10, 11, 12, 1, 2, 3])
    wyear = init.dt.year - (init.dt.month <= 3)          # winter labelled by its Oct year
    df["bb_class"] = np.where(~winter, "apr_sep_no_backtest",
                              np.where(wyear >= 2023, "held_out_2023_26",
                                       "reforecast_training_era"))
    df["season"] = pd.to_datetime(df.peak).dt.month.map(_season)
    return df


def method_note(out: Path = METHOD_MD) -> Path:
    k = derive_k()
    cov = slate_coverage()
    cls = cov.bb_class.value_counts()
    ids = {c: " ".join(e[:3] for e in cov[cov.bb_class == c].episode_id)
           for c in cls.index}
    seas = cov.season.value_counts()
    k3, k4 = k[3], k[4]
    n_json = ec46_json_count()
    if TIER2_JSON.exists():
        t2 = json.loads(TIER2_JSON.read_text()).get("status", {})
        t2_status = "Tier 2 verdict per truth: " + ", ".join(f"{a} {b}" for a, b in t2.items())
    elif n_json == 0:
        t2_status = ("Status: EC46 not yet downloaded (ECDS token pending), so Tier 2 has not "
                     "been run.")
    else:
        t2_status = "Status: EC46 exists but Tier 2 has not been run yet (run the estimate stage)."
    rows_status = (f"`bbsubs_rows.csv` exists ({len(pd.read_csv(ROWS_CSV))} estimate rows)."
                   if ROWS_CSV.exists() else "`bbsubs_rows.csv` has not been written yet.")
    txt = f"""# BB-SUBS estimate - method note

Written by `python -m acal.bbsubs --stage method` (acal/bbsubs.py). Context figure:
`figures/acal/overall/bbsubs_published.png`. Published numbers: `acal/data/bbsubs_published.csv`,
`acal/data/aiwq_tas_period_rpss.csv` (each row carries its source URL).

## What the BB-SUBS row is

An ESTIMATE, not a forecast. BB-SUBS (Brightband, 64-member global AI ensemble, 1.5 deg, daily
00z inits, 32 d) is not publicly downloadable, so the board row is our own bias-corrected EC46
score on each of the 42 cases scaled by Brightband's published skill ratio over debiased EC46:

    BB-SUBS est (case i) = k x EC46_corr error metric (case i)
    skill score vs climatology:  SS_BB = 1 - k (1 - SS_EC46)

Only error-type metrics get an estimate (ensemble-mean squared error, CRPS/RMSE/MAE, Brier/RPS
and their skill scores). Tail probabilities, P(obs), lift, PIT, ROC AUC, CSI and maps are never
estimated (blank in `bbsubs_rows.csv`). The estimate is drawn hatched and labelled "estimate".

## k at week 3 (headline) and week 4 (sensitivity)

| score family | week 3 k | week-3 band | week 4 k | derivation |
|---|---|---|---|---|
| MSE (ens-mean squared error) | {k3['mse']:.3f} | [{k3['mse_lo']:.3f}, {k3['mse_hi']:.3f}] | {k4['mse']:.3f} | (1 - MSESS_BB)/(1 - MSESS_EC46) = (1 - {k3['msess_bb']:.2f})/(1 - {k3['msess_ec46']:.2f}); band from the stated gap +0.10 +/- 0.02 |
| CRPS, RMSE, MAE (linear in error) | {k3['crps']:.3f} | [{k3['crps_lo']:.3f}, {k3['crps_hi']:.3f}] | {k4['crps']:.3f} | sqrt(k_MSE): a calibrated Gaussian's CRPS is proportional to its sigma |
| Brier, RPS (squared probability) | {k3['brier']:.3f} | [{k3['brier_lo']:.3f}, {k3['brier_hi']:.3f}] | {k4['brier']:.3f} | (1 - RPSS_BB)/(1 - RPSS_EC46) = (1 - {k3['rpss_bb']:.3f})/(1 - {k3['rpss_ec46']:.3f}); Brier = two-category RPS; no CI published, band = cross-metric range [k_MSE, k_Brier] |

So BB-SUBS is credited with {100 * (1 - k3['brier']):.0f}-{100 * (1 - k3['mse']):.0f}% lower error than
debiased EC46 at week 3. The week-3 k is used because our window (days 15-21) most likely
matches Brightband's undefined "week 3" (daily product) and is earlier than the AIWQ week 3
(days 19-25). MSESS and RPSS are vs climatology, so the ratio of skill-score complements is the
ratio of errors only if both models are scored against the same truth and reference, which
holds within each published chart.

## Assumptions and caveats (quote these with any BB-SUBS number)

1. **Self-reported.** Every BB-SUBS number is published by the vendor. There is no paper and no
   independent verification; the AIWQ RPSS (0.112 / 0.072) was scored by Brightband after the
   fact with the open AIWQ code on held-out forecasts, not as a blind leaderboard submission.
   All non-BB AIWQ values match the official leaderboard.
2. **Winter only.** The MSESS covers 3 held-out winters 2023-2026 (Oct-Mar); the RPSS one
   season, DJF 2025/26 (13 Thursday inits). Our slate by peak season: DJF {seas.get('DJF', 0)},
   SON {seas.get('SON', 0)}, MAM {seas.get('MAM', 0)}, JJA {seas.get('JJA', 0)}. Applying a winter
   ratio to SON/MAM/JJA cases is an extrapolation.
3. **Region and scale.** MSESS: 30-60N land gridpoints; RPSS: global land (land fraction >= 0.5),
   1.5 deg quintiles. Ours: a CONUS box-mean 7-day anomaly A_L. Area averaging removes noise, so
   absolute skill levels do not transfer; only the RATIO is used, anchored on our EC46.
4. **Average weather, not extremes.** Both published scores are unconditional; our 42 cases are
   selected on outcome (|A_L| >= 2 K). The ratio is an average-weather ratio applied to tail
   events.
5. **Debiasing.** Brightband's EC46 debiasing method is not stated; ours is the reforecast
   model-climate debias (11 members x 20 years, lead-dependent, ruling C12). The anchor is the
   EC46 `corr` variant; raw EC46 is never scaled.
6. **Sampling.** The RPSS is one 13-week season; EC46's own week-3 RPSS ranges 0.039-0.125 over
   the four completed AIWQ seasons (panel c of the context figure). The MSESS error bars
   (+/-0.02, interval type not stated) understate cross-season variation.
7. **Training overlap.** Using init = peak - 21 d: {cls.get('held_out_2023_26', 0)} cases fall in
   BB-SUBS's held-out winters 2023-26 ({ids.get('held_out_2023_26', '')});
   {cls.get('reforecast_training_era', 0)} fall in Oct-Mar winters 2020-21 to 2022-23, almost
   certainly inside BB-SUBS training ({ids.get('reforecast_training_era', '')});
   {cls.get('apr_sep_no_backtest', 0)} have Apr-Sep inits, outside any published BB-SUBS
   coverage ({ids.get('apr_sep_no_backtest', '')}). AI+RES (GenCast <2019, FCN3) is
   out-of-sample on all 42.
8. **Resolution.** BB-SUBS is 1.5 deg; no BB-SUBS maps are estimated.
9. **Truth.** The same k is applied under ERA5, HRRR and HRRR-raw truth; the published ratios
   are ERA5(T)-verified.

## Tier 2 (case-level P(obs)) - conditional

A calibrated Gaussian signal-noise model gives the expected forecast mass at or beyond a
standardized observation z as G(rho^2, z) = Phi(-z sqrt((1 - rho^2)/(1 + rho^2))), MSESS = rho^2.
Tier 2 would scale each case's EC46 P(obs) by G(rho_BB^2, z_i)/G(rho_EC46^2, z_i). It is run only
if the model reproduces EC46's own P(obs) on the 42 cases, with rho_EC46^2 measured from the EC46
reforecasts (debiased member mean vs ERA5, 20 years x 42 dates) and z_i = |A_L,i| / SD of the
7-day CONUS A_L in that season (2021-2025). Tolerance: (1) in each half of the slate split at
the median z, the predicted mean P(obs) must lie inside the 90% case-bootstrap interval of the
actual mean (the z-dependence is what Tier 2 transfers); (2) the overall level miss
|log(mean predicted / mean actual)| must be smaller than log of the median Tier-2 adjustment
(about log 1.3), i.e. the model's own error must be smaller than the effect it would translate.
{t2_status} The verdict ("passed" or "Tier 2 failed validation") is written to
`bbsubs_tier2_validation.json`; per-case Tier 2 estimates (`bbsubs_tier2_<truth>.csv`) exist only
for a truth that passed.

## Status

EC46 case files: {n_json}. {rows_status} The estimator is a no-op until `runs/acal/s2s/ec46/*.json` exist and the board table
(`board_cases.csv`) has bias-corrected ec46 rows; then `python -m acal.bbsubs --stage estimate`
writes `bbsubs_rows.csv`. The real alternative to this estimate is Brightband's free 2-week
pilot (held-out winters 2023-26 and the 2003-2026 reforecast archive), which would let the
{cls.get('held_out_2023_26', 0)} held-out cases be scored exactly like CFSv2; that needs the user.
"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("context", "estimate", "method", "all"), default="all")
    ap.add_argument("--board", default=str(BOARD_CASES))
    a = ap.parse_args(argv)
    if a.stage in ("context", "all"):
        print(f"[bbsubs] wrote {context_figure()}")
    if a.stage in ("estimate", "all"):
        run_estimate(Path(a.board))
    if a.stage in ("method", "all"):           # last: its status lines read the outputs
        print(f"[bbsubs] wrote {method_note()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
