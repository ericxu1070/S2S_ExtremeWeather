#!/usr/bin/env python
"""How much of AI+RES's board advantage could come from steering? CPU ONLY, login node.

Every acal run resampled its walkers toward the OBSERVED tail (`run.json` tail_sign from
the case family: warm walkers cloned on heat cases, cold walkers on cold cases), while the
baselines on the board (CFSv2, GEFSv12, ECCC GEPS) were blind to the outcome. The
importance weights `w_i = exp(-V_K,i)` remove the tilt in expectation, but the board scores
the SELF-NORMALIZED estimate from N = 32 walkers that descend from a handful of founders,
and a ratio estimator at that size can keep part of the tilt. This module measures it from
files that already exist; nothing is re-run.

Three AI+RES forecasts per case, on the board's truths, windows and masks:

    sn        importance weights, self-normalized: the published estimator (board 'sn';
              reproduced bit for bit, asserted against board_paired_cases.csv)
    uniform   the same 32 final walkers with EQUAL weights: the raw tilted population,
              i.e. the forecast with all of the steering left in
    untilted  the FCN3 score forecasts launched at lead 6 d from the 32 PRE-selection
              walker states (6 members each, 192 in all). Step 1 (lead 3 d) has C = 0 and
              identity parents in all 42 runs, so these day-6 states carry no steering,
              and they are the only forecasts in the campaign that do and that reach the
              verification window. They are NOT AI+RES without steering: days 6-21 are
              FCN3, not GenCast.

What the C = 0 leg itself can tell us: nothing about the window. Its walkers stop at lead
3 d, and its `theta` is the persistence index (the instantaneous day-3 anomaly), not a
forecast. No GenCast walker state inside the verification window (lead 15-21 d) is
untilted; every final walker descends from four tilted selections (C = 1, 1.4, 1.8, 2).

    python -m acal.tilt --stage lead6     # tilt_lead6/<case>.csv (~10 s per case, cached)
    python -m acal.tilt --stage table     # tilt_check.csv, tilt_check_paired.csv, .json
    python -m acal.tilt --stage figure    # figures/acal/overall/tilt_check.png
    python -m acal.tilt --stage all

Sign conventions follow the board: positive paired differences mean AI+RES better;
`err_*` = s * (forecast mean - obs), so > 0 is a forecast beyond the observation on the
tail side and < 0 one short of it.

The BB-SUBS estimate (`acal/bbsubs.py`, k x bias-corrected EC46, error scores only) is paired
the same way (source 'bbsubs', variant 'corr_emp', 12f; no log ratio, because it has no
P(obs)). The json block `estimate` adds its CRPS pairing at every published k (week-3 band
ends and week-4) and on the cases where the EC46 lead is 21 d, the BB-SUBS lead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import aprep
from acal import board as BD
from acal import cfsbase as CB
from acal import s2sbase as S2
from acal import truth as TR

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = S2.ANALYSIS
LEAD6_DIR = ANALYSIS / "tilt_lead6"
CASES_OUT = ANALYSIS / "tilt_check.csv"
PAIRED_OUT = ANALYSIS / "tilt_check_paired.csv"
JSON_OUT = ANALYSIS / "tilt_check.json"
FIG_OUT = ROOT / "figures" / "acal" / "overall" / "tilt_check.png"

TRUTHS = ("era5", "hrrr", "hrrr_raw")
WINDOWS = BD.AIRES_WINDOWS                      # ('13f', '12f')
MODELS = ("cfs13", "gefs", "geps", "ec46")      # board rows with data
MODEL_VARIANTS = ("raw_emp", "corr_emp")        # headline + bias-corrected sensitivity
ESTIMATES = {"bbsubs": ("corr_emp",)}           # board estimate rows paired here (error scores)
ESTIMATE_ANCHOR = ("ec46", "corr_emp")          # BB-SUBS estimate = k x this row's CRPS
ESTIMATE_LEAD_DAYS = 21                         # BB-SUBS starts daily: its lead to the peak
VARIANTS = ("sn", "uniform", "untilted", "untilted_f32")
FIG_VARIANTS = ("sn", "uniform", "untilted")
METRICS = ("logratio", "dcrps", "dbrier_2K", "dbrier_3K", "dbrier_4K", "dsqerr")
SIDE = {"logratio": "logp_floor", "dcrps": "crps", "dsqerr": "mean_sqerr",
        **{f"dbrier_{k:g}K": f"brier_{k:g}K" for k in BD.BRIER_K}}
LEAD6_DAYS = 6.0
N_LEAD6_MEMBERS = 6
CHECK_TOL = 1e-4                                # K; 25-frame mean vs theta[1]['conus']
PARITY_TOL = 1e-12


# --------------------------------------------------------------------------- #
# The untilted lead-6 FCN3 forecasts
# --------------------------------------------------------------------------- #
def lead6_case(eid: str, force: bool = False) -> Path:
    """Per-member A_L of the lead-6 FCN3 score forecasts of one case (cached CSV).

    Columns al13/al12 (CONUS, the ERA5 truth's reduction) and al13_hrrrmask/al12_hrrrmask
    (the HRRR truths' mask), from the 12-hourly frames of `truth.window_times`; al25 is the
    6-hourly 25-frame mean that `run_aires.theta_from_fcn3` scored, checked against the
    stored theta to CHECK_TOL so the cube and the run's own record are the same forecast.
    """
    import xarray as xr
    from aires import aindex as AI

    p = LEAD6_DIR / f"{eid}.csv"
    if p.exists() and not force:
        return p
    d = AN.case_dir(eid)
    run = json.loads((d / "run.json").read_text())
    rec = json.loads((d / "res_result.json").read_text())
    th, dmc = rec["theta"][1], rec["dmc"]
    n = int(dmc["n_walkers"])
    if not (dmc["C"][0] == 0.0 and dmc["genealogy"][0] == list(range(n))):
        raise SystemExit(f"[tilt] {eid}: step 1 is not untilted (C {dmc['C'][0]}, "
                         "parents not identity) - lead-6 states would carry steering")
    if not (th["lead_days"] == LEAD6_DAYS and th["backend"] == "fcn3"
            and th["members"] == N_LEAD6_MEMBERS):
        raise SystemExit(f"[tilt] {eid}: theta[1] is not the lead-6 FCN3 score: "
                         f"{th['lead_days']} {th['backend']} {th['members']}")
    peak = pd.Timestamp(run["peak"])
    t25 = pd.date_range(peak - pd.Timedelta(days=6), peak, freq="6h")
    t13, t12 = TR.window_times(peak, "13f"), TR.window_times(peak, "12f")
    mask = TR.get_truth("hrrr").mask()
    rows = []
    for w in range(n):
        f = d / "scores" / f"w{w:02d}_lead06_cube.nc"
        with xr.open_dataset(f) as cube:
            if float(cube.attrs["score_lead_days"]) != LEAD6_DAYS:
                raise SystemExit(f"[tilt] {f}: score_lead_days {cube.attrs['score_lead_days']}")
            inst = AI.instantaneous_field(cube.sel(time=t25).load(), "t2m_anom")
        inst = inst.astype("float64")
        inst = inst.assign_coords(lat=inst["lat"].astype("float64"))
        a25 = TR.area_mean(inst.mean("time")).values
        if abs(a25.mean() - th["conus"][w]) > CHECK_TOL:
            raise SystemExit(f"[tilt] {eid} w{w:02d}: 25-frame mean {a25.mean():+.5f} != "
                             f"theta {th['conus'][w]:+.5f}")
        f13, f12 = inst.sel(time=t13).mean("time"), inst.sel(time=t12).mean("time")
        cols = dict(al25=a25, al13=TR.area_mean(f13).values, al12=TR.area_mean(f12).values,
                    al13_hrrrmask=TR.area_mean(f13, mask).values,
                    al12_hrrrmask=TR.area_mean(f12, mask).values)
        for m in range(a25.size):
            rows.append(dict(eid=eid, walker=w, member=m,
                             **{k: float(v[m]) for k, v in cols.items()}))
    tab = pd.DataFrame(rows)
    if len(tab) != n * N_LEAD6_MEMBERS:
        raise SystemExit(f"[tilt] {eid}: {len(tab)} lead-6 members, want {n * N_LEAD6_MEMBERS}")
    LEAD6_DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    tab.to_csv(tmp, index=False, float_format="%.9g")
    os.replace(tmp, p)
    return p


def lead6_all(force: bool = False) -> None:
    for r in aprep.episodes().itertuples():
        t0 = time.time()
        p = lead6_case(r.episode_id, force=force)
        print(f"[tilt] {r.episode_id}: {p.name} ({time.time() - t0:.0f} s)", flush=True)


def lead6_members(eid: str, truth: TR.Truth, window: str) -> np.ndarray:
    p = LEAD6_DIR / f"{eid}.csv"
    if not p.exists():
        raise SystemExit(f"[tilt] missing {p}; run --stage lead6")
    tab = pd.read_csv(p, float_precision="round_trip")
    return tab[f"al{window[:2]}{TR.mask_tag(truth)}"].to_numpy(dtype="float64")


# --------------------------------------------------------------------------- #
# Per-case diagnostics and scores
# --------------------------------------------------------------------------- #
def weight_stats(al, w, s: float) -> dict:
    """Shape of the final importance weights and where the walkers sit around the
    self-normalized mean (pure numpy; unit-tested)."""
    al, w = np.asarray(al, dtype="float64"), np.asarray(w, dtype="float64")
    wn = w / w.sum()
    n = al.size
    mu = float(np.sum(wn * al))
    nz = wn[wn > 0]
    h = float(-np.sum(nz * np.log(nz)))
    return dict(ess=float(1.0 / np.sum(wn ** 2)), ess_frac=float(1.0 / np.sum(wn ** 2)) / n,
                entropy=h, entropy_frac=h / np.log(n), w_max=float(wn.max()),
                n_tilted_side=int(np.sum(s * (al - mu) > 0)),
                n_tilted_side_uniform=int(np.sum(s * (al - al.mean()) > 0)))


def case_metrics(c: AN.Case, l6: np.ndarray) -> dict[str, dict]:
    """The three AI+RES forecasts of one case, scored with the board's own functions."""
    s, obs = float(c.sign), float(c.obs)
    return {"sn": BD.aires_metrics(c),
            "uniform": BD.ensemble_metrics(c.al, obs, s, "emp", len(c.al)),
            "untilted": BD.ensemble_metrics(l6, obs, s, "emp", len(l6)),
            # same forecast, log-ratio floor at the walkers' 1/33 instead of 1/193
            "untilted_f32": BD.ensemble_metrics(l6, obs, s, "emp", len(c.al))}


KEEP = ("p_obs", "p_floor", "logp_floor", "crps", "fc_mean", "mean_err_signed", "mean_sqerr",
        "spread", "brier_2K", "brier_3K", "brier_4K")


def case_rows() -> pd.DataFrame:
    slate = aprep.episodes()
    rows = []
    for tname in TRUTHS:
        tr = TR.get_truth(tname)
        for win in WINDOWS:
            cases = S2.aires_cases(tr, win)
            for i, r in enumerate(slate.itertuples()):
                c = cases[r.episode_id]
                l6 = lead6_members(r.episode_id, tr, win)
                m = case_metrics(c, l6)
                rec = dict(truth=tname, window=win, case_idx=i, episode_id=r.episode_id,
                           family=r.family, rung=int(r.rung), tail_sign=float(c.sign),
                           obs=float(c.obs), n_walkers=int(len(c.al)),
                           n_untilted=int(len(l6)),
                           **weight_stats(c.al, c.weights, c.sign))
                for v in VARIANTS:
                    for k in KEEP:
                        rec[f"{k}_{v}"] = float(m[v][k])
                # how far the weights pulled the mean back from the tilted population
                rec["pullback"] = rec["mean_err_signed_uniform"] - rec["mean_err_signed_sn"]
                rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Paired against the board's models
# --------------------------------------------------------------------------- #
def _paired_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Scored model rows (raw and corrected) plus the board's estimate rows in ESTIMATES."""
    est = df.estimate.astype(bool)
    keep = (df.source.isin(MODELS)) & (df.variant.isin(MODEL_VARIANTS)) & (~est)
    for src, variants in ESTIMATES.items():
        keep |= (df.source == src) & (df.variant.isin(variants)) & est
    return df[keep]


def _board_cases() -> pd.DataFrame:
    return _paired_sources(pd.read_csv(BD.CASES_CSV, float_precision="round_trip"))


def paired_cases(cs: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    """Per-case differences, model minus AI+RES variant (positive = AI+RES better)."""
    out = []
    for (tname, src, mv), g in models.groupby(["truth", "source", "variant"], sort=False):
        win = g.window.iloc[0]
        a = cs[(cs.truth == tname) & (cs.window == win)].set_index("episode_id")
        g = g.sort_values("case_idx")
        a = a.loc[g.episode_id]
        if np.max(np.abs(a.obs.to_numpy() - g.obs.to_numpy())) > BD.OBS_TOL:
            raise SystemExit(f"[tilt] {tname} {src} {mv}: obs differ - not like-for-like")
        for v in VARIANTS:
            d = {"logratio": np.log(a[f"p_floor_{v}"].to_numpy() / g.p_floor.to_numpy())}
            for mname in METRICS[1:]:
                col = SIDE[mname]
                d[mname] = g[col].to_numpy() - a[f"{col}_{v}"].to_numpy()
            for j, row in enumerate(g.itertuples()):
                rec = dict(truth=tname, source=src, label=BD.label(src), variant=mv,
                           window=win, aires_variant=v, case_idx=int(row.case_idx),
                           episode_id=row.episode_id, family=row.family, rung=int(row.rung),
                           obs=float(row.obs))
                rec.update({k: float(x[j]) for k, x in d.items()})
                out.append(rec)
    return pd.DataFrame(out)


def parity(pc: pd.DataFrame) -> float:
    """Max |difference| between our 'sn' pairs and the board's own (must be ~0)."""
    bp = _paired_sources(pd.read_csv(BD.PAIRED_CASES_CSV, float_precision="round_trip"))
    ours = pc[pc.aires_variant == "sn"]
    m = ours.merge(bp, on=["truth", "source", "variant", "episode_id"], suffixes=("", "_b"))
    if len(m) != len(ours):
        raise SystemExit(f"[tilt] parity: {len(ours)} sn pairs, {len(m)} matched on the board")
    worst = 0.0
    for k in METRICS:
        d = np.abs(m[k] - m[f"{k}_b"]).to_numpy()
        if np.isnan(m[k].to_numpy()).sum() != np.isnan(m[f"{k}_b"].to_numpy()).sum():
            raise SystemExit(f"[tilt] parity: {k} is blank on one side only")
        if np.isfinite(d).any():
            worst = max(worst, float(np.nanmax(d)))
    if worst > PARITY_TOL:
        raise SystemExit(f"[tilt] parity: 'sn' pairs differ from the board by {worst:.3g}")
    return worst


def paired_table(pc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in pc.groupby(["truth", "source", "variant", "aires_variant"], sort=False):
        for sub in BD.SUBSETS:
            d = g if sub == "all" else g[g.family == sub]
            for mname in METRICS:
                if not np.isfinite(g[mname].to_numpy()).any():
                    continue                    # e.g. no log ratio for the BB-SUBS estimate
                rows.append(dict(zip(("truth", "source", "variant", "aires_variant"), key),
                                 label=g.label.iloc[0], window=g.window.iloc[0], subset=sub,
                                 metric=mname, **CB.paired_stats(d[mname].to_numpy())))
    return pd.DataFrame(rows)


def estimate_block(cs: pd.DataFrame) -> dict:
    """dCRPS of the BB-SUBS estimate minus each AI+RES forecast ('sn' published, 'untilted'
    blind; positive = AI+RES better) at every published CRPS ratio k (week-3 value and band
    ends, week 4), and at the week-3 k on the cases whose EC46 lead is 21 d (BB-SUBS starts
    daily, so its own lead would be 21 d on every case)."""
    from acal import bbsubs as BB
    from acal import s2s_ec46 as E
    k = BB.derive_k()
    ks = {"wk3": float(k[3]["crps"]), "wk3_lo": float(k[3]["crps_lo"]),
          "wk3_hi": float(k[3]["crps_hi"]), "wk4": float(k[4]["crps"])}
    bc = pd.read_csv(BD.CASES_CSV, float_precision="round_trip")
    est_flag = bc.estimate.astype(bool)
    lead = pd.read_csv(E.REQUESTS_CSV).set_index("episode_id").lead_days
    out = dict(k_crps=ks, lead_days=ESTIMATE_LEAD_DAYS, anchor="/".join(ESTIMATE_ANCHOR))
    src, var = ESTIMATE_ANCHOR
    for tname in TRUTHS:
        g = bc[(bc.truth == tname) & (bc.source == src) & (bc.variant == var) & ~est_flag]
        e = bc[(bc.truth == tname) & (bc.source == "bbsubs") & (bc.variant == "corr_emp")
               & est_flag]
        if g.empty or e.empty:
            continue
        g = g.sort_values("case_idx").set_index("episode_id")
        e = e.set_index("episode_id").loc[g.index]
        win = g.window.iloc[0]
        a = cs[(cs.truth == tname) & (cs.window == win)].set_index("episode_id").loc[g.index]
        anchor = g.crps.to_numpy()
        if not np.allclose(e.crps.to_numpy(), ks["wk3"] * anchor, rtol=1e-6, atol=0):
            raise SystemExit(f"[tilt] {tname}: BB-SUBS estimate is not k x {src} {var} CRPS")
        m21 = (lead.loc[g.index] == ESTIMATE_LEAD_DAYS).to_numpy()
        blk = dict(window=win, n=int(len(g)), n_lead21=int(m21.sum()))
        for v in ("sn", "untilted"):
            side = a[f"crps_{v}"].to_numpy()
            for kname, kv in ks.items():
                blk[f"dcrps_{kname}_{v}"] = CB.paired_stats(kv * anchor - side)
            blk[f"dcrps_lead21_{v}"] = CB.paired_stats((ks["wk3"] * anchor - side)[m21])
        out[tname] = blk
    return out


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _q(x) -> dict:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                min=float(x.min()), max=float(x.max()))


def family_summary(cs: pd.DataFrame) -> dict:
    """Per truth/window and family: means of the tilt diagnostics, plus the paired
    weight effect sn - uniform (model-free: the model side cancels in every metric)."""
    out = {}
    cols = ("mean_err_signed_sn", "mean_err_signed_uniform", "mean_err_signed_untilted",
            "pullback", "ess", "entropy_frac", "w_max", "n_tilted_side",
            "n_tilted_side_uniform", "logp_floor_sn", "logp_floor_uniform",
            "logp_floor_untilted", "crps_sn", "crps_uniform", "crps_untilted",
            "p_obs_sn", "p_obs_uniform", "p_obs_untilted")
    for (tname, win), g in cs.groupby(["truth", "window"], sort=False):
        blk = {}
        for sub in BD.SUBSETS:
            d = g if sub == "all" else g[g.family == sub]
            e = {k: _q(d[k]) for k in cols}
            e["weight_effect"] = dict(
                logp=CB.paired_stats(d.logp_floor_sn - d.logp_floor_uniform),
                crps=CB.paired_stats(d.crps_uniform - d.crps_sn),
                err_toward_tail=CB.paired_stats(d.pullback))
            blk[sub] = e
        out.setdefault(tname, {})[win] = blk
    return out


def verdicts(pt: pd.DataFrame) -> dict:
    """Per (truth, model, raw_emp): mean logratio and dCRPS of the three AI+RES forecasts,
    and the share of the uniform (fully tilted) advantage the published 'sn' keeps."""
    out = {}
    h = pt[(pt.variant == "raw_emp") & (pt.subset == "all")]
    for (tname, src), g in h.groupby(["truth", "source"], sort=False):
        blk = dict(label=g.label.iloc[0], window=g.window.iloc[0])
        for mname in ("logratio", "dcrps"):
            x = g[g.metric == mname].set_index("aires_variant")
            e = {v: dict(mean=float(x.loc[v, "mean"]), ci_lo=float(x.loc[v, "ci_lo"]),
                         ci_hi=float(x.loc[v, "ci_hi"]), win=int(x.loc[v, "win"]),
                         tie=int(x.loc[v, "tie"]), loss=int(x.loc[v, "loss"]),
                         wilcoxon_p=float(x.loc[v, "wilcoxon_p"])) for v in VARIANTS}
            u, sn = e["uniform"]["mean"], e["sn"]["mean"]
            e["sn_share_of_uniform"] = float(sn / u) if u != 0 else float("nan")
            blk[mname] = e
        out.setdefault(tname, {})[src] = blk
    return out


CONVENTIONS = dict(
    variants=dict(
        sn="published AI+RES: final 32 walkers, importance weights self-normalized "
           "(board 'sn', bit-identical)",
        uniform="the same 32 final walkers with equal weights: the tilted population "
                "(all steering kept; an upper bound on what steering can buy)",
        untilted="FCN3 score forecasts launched at lead 6 d from the 32 pre-selection walker "
                 "states, 6 members each (192): no steering, but FCN3 (not GenCast) over "
                 "lead 6-21 d",
        untilted_f32="'untilted' with its log-ratio floor at 1/33 (the walkers' N = 32) "
                     "instead of 1/193: only logp_floor / logratio differ"),
    c0_leg="Step 1 (lead 3 d) has C = 0 and identity parents in all 42 runs; its theta is the "
           "persistence index at day 3, not a forecast, and no GenCast walker state in the "
           "verification window (lead 15-21 d) is untilted.",
    signs="err_* = s * (mean - obs) (> 0 beyond obs on the tail side); pullback = "
          "err_uniform - err_sn (> 0: the weights moved the mean away from the tail); paired "
          "differences positive = AI+RES better (board convention); logratio floors each side "
          "at 1/(N+1) with its own N (32 walkers, 192 untilted members, model members)",
    stats="cfsbase.paired_stats: mean, median, 90% case-bootstrap CI (n=5000, seed 20261006), "
          "W/T/L, Wilcoxon p",
    weight_effect="sn minus uniform per case (logp, -crps, -err): the model side cancels, so "
                  "it is the same against every baseline",
    estimate="source 'bbsubs' rows are the BB-SUBS ESTIMATE (k x bias-corrected EC46 per case, "
             "acal/bbsubs.py), not a forecast: error scores only, no logratio; json 'estimate' "
             "= its dCRPS at each published k and on the EC46-lead-21 d cases",
)


def table() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cs = case_rows()
    pc = paired_cases(cs, _board_cases())
    worst = parity(pc)
    pt = paired_table(pc)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for df, p in ((cs, CASES_OUT), (pt, PAIRED_OUT)):
        tmp = p.with_suffix(".tmp.csv")
        df.to_csv(tmp, index=False, float_format="%.9g")
        os.replace(tmp, p)
    summ = dict(generated=str(pd.Timestamp.now(tz="UTC")), conventions=CONVENTIONS,
                n_cases=int(cs.episode_id.nunique()), board_parity_max_abs=worst,
                models=list(MODELS), estimates={k: list(v) for k, v in ESTIMATES.items()},
                truths=list(TRUTHS), families=family_summary(cs), verdicts=verdicts(pt),
                estimate=estimate_block(cs))
    tmp = JSON_OUT.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(BD._clean(summ), indent=1))
    os.replace(tmp, JSON_OUT)
    print(f"[tilt] {CASES_OUT.name} {len(cs)} rows, {PAIRED_OUT.name} {len(pt)} rows, "
          f"{JSON_OUT.name}; board parity max |d| {worst:.2g}")
    return cs, pt, summ


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
INK, INK2, INK3, GRID = "#1f1f1f", "#555555", "#8a8a8a", "#e4e4e4"
VSTYLE = {
    "uniform": dict(label="uniform weights (tilted walkers)", color=S2.MODEL_STYLE["aires"]["color"],
                    marker="s", mfc="white", ms=5.2),
    "sn": dict(label="self-normalized weights (published)",
               color=S2.MODEL_STYLE["aires"]["color"], marker="o", mfc=None, ms=5.4),
    "untilted": dict(label="untilted lead-6 FCN3 (blind, 192 members)", color="#555555",
                     marker="D", mfc=None, ms=4.4),
}
FIG_TRUTHS = ("era5", "hrrr")


def figure(cs: pd.DataFrame | None = None, pt: pd.DataFrame | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if cs is None:
        cs = pd.read_csv(CASES_OUT, float_precision="round_trip")
    if pt is None:
        pt = pd.read_csv(PAIRED_OUT, float_precision="round_trip")
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
                         "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK,
                         "ytick.labelcolor": INK, "axes.labelcolor": INK, "text.color": INK})
    fig = plt.figure(figsize=(10.0, 7.6), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.36, wspace=0.08,
                          left=0.165, right=0.985, top=0.885, bottom=0.1)

    # (a) per-case signed mean error, ERA5 13f
    ax = fig.add_subplot(gs[0, :])
    d = cs[(cs.truth == "era5") & (cs.window == "13f")].copy()
    d["fam_order"] = (d.family == "cold").astype(int)
    d = d.sort_values(["fam_order", "mean_err_signed_uniform"]).reset_index(drop=True)
    x = np.arange(len(d))
    for xi, r in zip(x, d.itertuples()):
        ax.plot([xi, xi], [r.mean_err_signed_uniform, r.mean_err_signed_sn], color=INK3,
                lw=0.9, zorder=2)
    for v in ("untilted", "uniform", "sn"):
        st = VSTYLE[v]
        ax.plot(x, d[f"mean_err_signed_{v}"], ls="", marker=st["marker"], ms=st["ms"],
                color=st["color"], mfc=st["mfc"] or st["color"], mec=st["color"], mew=1.0,
                zorder=4 if v != "untilted" else 3)
    nh = int((d.family == "heat").sum())
    ax.axvline(nh - 0.5, color=INK3, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color=INK, lw=0.9, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(-0.8, len(d) - 0.2)
    ax.set_xticks([])
    ax.set_ylabel("mean - obs toward the tail (K)")
    lo, hi = ax.get_ylim()
    means = []
    for fam, x0, x1 in (("heat", 0, nh - 1), ("cold", nh, len(d) - 1)):
        g = d[d.family == fam]
        ax.text((x0 + x1) / 2, hi + 0.03 * (hi - lo), f"{fam} cases (n = {len(g)})",
                ha="center", va="bottom", fontsize=8, color=INK2)
        means.append(f"{fam}: uniform {g.mean_err_signed_uniform.mean():+.2f}, "
                     f"self-normalized {g.mean_err_signed_sn.mean():+.2f}, "
                     f"untilted {g.mean_err_signed_untilted.mean():+.2f}")
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    ax.set_xlabel("case (heat left, cold right; sorted by the uniform-weight error within each "
                  "family)\nfamily means (K), " + ";  ".join(means), fontsize=7.8, color=INK2,
                  linespacing=1.5)
    ax.set_title("(a) Ensemble-mean A_L error toward the observed tail, ERA5, 13-frame window",
                 loc="left", fontsize=9.2, color=INK)

    # (b) logratio and (c) dCRPS against each baseline
    rows = [(t, m) for m in MODELS for t in FIG_TRUTHS]
    h = pt[(pt.variant == "raw_emp") & (pt.subset == "all")]
    offs = {"uniform": -0.24, "sn": 0.0, "untilted": 0.24}
    for col, (mname, title, xl) in enumerate((
            ("logratio", "(b) Paired log ratio log(P_AI+RES / P_model) at obs",
             "mean over 42 cases, positive = AI+RES better"),
            ("dcrps", "(c) Paired CRPS difference (model - AI+RES)",
             "mean over 42 cases (K), positive = AI+RES better"))):
        ax = fig.add_subplot(gs[1, col])
        for i, (t, m) in enumerate(rows):
            y = len(rows) - 1 - i
            if i % 4 == 0:                    # shade every other model (2 rows each)
                ax.axhspan(y - 1.5, y + 0.5, color="#f4f4f4", zorder=0, lw=0)
            for v in FIG_VARIANTS:
                r = h[(h.truth == t) & (h.source == m) & (h.metric == mname)
                      & (h.aires_variant == v)].iloc[0]
                st = VSTYLE[v]
                yy = y - offs[v]
                ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=st["color"], lw=1.4, zorder=3,
                        solid_capstyle="round")
                ax.plot([r["mean"]], [yy], ls="", marker=st["marker"], ms=st["ms"],
                        color=st["color"], mfc=st["mfc"] or st["color"], mec=st["color"],
                        mew=1.0, zorder=4)
        ax.axvline(0, color=INK, lw=0.9, zorder=2)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=1)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(range(len(rows)))
        if col == 0:
            ax.set_yticklabels([f"{BD.label(m)}, {'ERA5' if t == 'era5' else 'HRRR'}"
                                for t, m in rows[::-1]], fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xl, fontsize=7.8, color=INK2)
        ax.set_title(title, loc="left", fontsize=9.2, color=INK)

    handles = [Line2D([], [], ls="", marker=VSTYLE[v]["marker"], ms=VSTYLE[v]["ms"] + 0.6,
                      color=VSTYLE[v]["color"], mfc=VSTYLE[v]["mfc"] or VSTYLE[v]["color"],
                      mec=VSTYLE[v]["color"], label=VSTYLE[v]["label"])
               for v in ("uniform", "sn", "untilted")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.2,
               bbox_to_anchor=(0.56, 0.945), handletextpad=0.4, columnspacing=1.6)
    fig.suptitle("AI+RES steering check: walkers were cloned toward the observed tail; "
                 "the baselines were blind", x=0.56, y=0.985, fontsize=10.5,
                 fontweight="bold", color=INK)
    fig.text(0.5, 0.008,
             "Bars: 90% case-bootstrap CI over 42 cases. HRRR = offset-corrected truth on the "
             "HRRR mask. Baselines raw (no bias correction); GEPS and EC46 paired on the 12-frame window."
             "\nUniform = the final 32 walkers with equal weights (all steering kept). Untilted = "
             "FCN3 forecasts from the 32 pre-selection day-6 walker states (not GenCast "
             "over lead 6-21 d).", ha="center", va="bottom", fontsize=7, color=INK2,
             linespacing=1.45)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[tilt] {FIG_OUT}")
    return FIG_OUT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("lead6", "table", "figure", "all"), default="all")
    ap.add_argument("--force", action="store_true", help="recompute the lead-6 cache")
    a = ap.parse_args(argv)
    if a.stage in ("lead6", "all"):
        lead6_all(force=a.force)
    cs = pt = None
    if a.stage in ("table", "all"):
        cs, pt, _ = table()
    if a.stage in ("figure", "all"):
        figure(cs, pt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
