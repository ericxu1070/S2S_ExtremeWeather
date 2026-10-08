#!/usr/bin/env python
"""The multi-model S2S board: the one table every overall figure and the summary read. LOGIN NODE.

    python -m acal.board --stage collect                 # board_cases.csv (all truths)
    python -m acal.board --stage paired                  # board_paired*.csv + board_summary.json
    python -m acal.board --stage all                     # both
    python -m acal.board --stage all --truth era5 --sources cfs13 gefs --no-fields

Writes runs/acal/analysis/s2s/board/:

board_cases.csv
    One row per (truth, source, window, variant, case). Sources: 'aires' (AI+RES, once per
    truth window '13f' and '12f' so every model pairs like-for-like), the board sources with
    data (`s2sbase.BOARD_SOURCES`: cfs13, ec46, gefs, geps; a source without cubes is absent
    and joins on the next run once its files exist), the published 25-frame CFS 'cfs' as a
    reference row (role 'reference', the parity source), and BB-SUBS estimate rows merged
    from bbsubs_rows.csv when acal.bbsubs wrote them in this run (estimate=True).
    `obs` is the truth's A_L on the row's window and mask (`s2sbase.load` / `aires_cases`).
    Variants: AI+RES 'sn' (importance weights, self-normalized, as published); models
    `Source.variants` (raw_emp headline, raw_gauss, corr_*, s16_emp fixed 16-member subset,
    sub_emp CFS last 4) plus 'e16_<v>' = the EXPECTED score over uniformly random
    16-member subsets (and 'e16_sn' for AI+RES walkers).
    Columns: p_obs, p_floor = max(p, 1/(N+1)), logp_floor, pit, o/p/brier at 2/3/4 K
    (tail-signed: o = s*obs >= k, p = P(s*A >= k)), crps (CRPS of the forecast distribution:
    members or importance-weighted walkers; closed form for a Gaussian variant), crps_fair
    (ensemble-size-adjusted), fc_mean, mean_err (fc_mean - obs), mean_err_signed
    (s * mean_err; < 0 = short of the extreme), mean_sqerr, spread (unbiased sd; reliability-
    weighted for AI+RES), n_members (scored), n_native, n_eff, p_clim_obs, and per-case field
    metrics of the raw ensemble mean (field_bias / field_rmse / field_pattern_r: 7-day mean
    T2m anomaly vs the truth over CONUS land on the truth's mask, cos-lat weighted,
    `errmaps.wstats`) on rows raw_emp / raw_gauss / sn (NaN elsewhere: no field-level bias
    correction or subsample mean is invented).

board_paired.csv (long: one row per truth, source, variant, subset, metric)
    Each model variant against the AI+RES row on the SAME truth window and mask: 'sn' for
    native/fixed-subset variants, 'e16_sn' for 'e16_*' (both sides at 16 members).
    Metrics (positive = AI+RES better throughout):
        logratio      log(P_AIRES / P_model), each floored at 1/(N+1) (cfsbase.log_ratio);
                      for e16 rows E[log floored P] of each side (sides independent)
        dbrier_<k>K   Brier_model - Brier_AIRES at k = 2, 3, 4 K
        dcrps         CRPS_model - CRPS_AIRES
        dsqerr        ensemble-mean squared error, model - AI+RES
        dfield_rmse   7-day field RMSE of the ensemble mean, model - AI+RES (raw rows only)
    Stats per (subset all/heat/cold): `cfsbase.paired_stats` (mean, median, 90% case-
    bootstrap CI n=5000 seed 20261006, W/T/L, Wilcoxon p) + mean of each side.
board_paired_cases.csv: the per-case differences behind it.
board_summary.json: per (truth, source, window, variant, subset) means/medians, the paired
    table, a 'coverage' block (cases, members, lead range, window, native grid per source)
    and the conventions.

Expected scores over random 16-member subsets ('e16_')
    empirical model variants  EXACT: the reaching count in a 16-subset is hypergeometric, so
                              P, the floored log P and the Brier scores are exact sums over
                              its pmf; E[CRPS] = mean|x - y| - (m-1)/(2m) * mean_{i!=j}|x_i - x_j|;
                              E[(mean_S - y)^2] = (mean - y)^2 + var_pop/m * (n-m)/(n-1).
    Gaussian model variants   N_RESAMPLE_GAUSS seeded subsets (mean, sd refitted per subset).
    AI+RES walkers            N_RESAMPLE_AIRES seeded 16-walker subsets, weights self-
                              normalized within the subset (an approximation of a 16-walker
                              run: the cloning history is the 32-walker one).
    A source with N <= 16 is its own e16 (exactly).

Parity: source 'cfs' + truth 'era5' reproduces the published runs/acal/analysis/
cfs_summary.json paired numbers exactly (acal/tests/test_board.py).
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
from acal import cfsbase as CB
from acal import s2sbase as S2
from acal import truth as TR

BOARD_DIR = S2.ANALYSIS / "board"
CASES_CSV = BOARD_DIR / "board_cases.csv"
PAIRED_CSV = BOARD_DIR / "board_paired.csv"
PAIRED_CASES_CSV = BOARD_DIR / "board_paired_cases.csv"
SUMMARY_JSON = BOARD_DIR / "board_summary.json"
BBSUBS_ROWS_CSV = BOARD_DIR / "bbsubs_rows.csv"

TRUTHS = ("era5", "hrrr", "hrrr_raw")
REFERENCE = ("cfs",)                     # published 25-frame CFS: reference + parity
SOURCES = REFERENCE + tuple(S2.BOARD_SOURCES)
AIRES = "aires"
AIRES_WINDOWS = ("13f", "12f")
N_RES = CB.N_RES                         # 32 walkers
BRIER_K = CB.BRIER_K
N_SUB = CB.N_FIXED                       # 16
E16_BASE = ("raw_emp", "raw_gauss", "corr_emp", "corr_gauss")
N_RESAMPLE_GAUSS = 400
N_RESAMPLE_AIRES = 1000
RESAMPLE_SEED = 20261008
OBS_TOL = 1e-9                           # K; model and AI+RES obs must be the same number
TIE_TOL = 1e-12                          # e16 logratio: |d| below this is a float pseudo-tie
SUBSETS = ("all", "heat", "cold")
FIELD_VARIANTS = ("raw_emp", "raw_gauss", "sn")
PAIRED_METRICS = ("logratio", "dbrier_2K", "dbrier_3K", "dbrier_4K", "dcrps", "dsqerr",
                  "dfield_rmse")
SIDE_COL = {"logratio": "logp_floor", "dcrps": "crps", "dsqerr": "mean_sqerr",
            "dfield_rmse": "field_rmse", **{f"dbrier_{k:g}K": f"brier_{k:g}K"
                                            for k in CB.BRIER_K}}
SUMMARY_METRICS = ("p_obs", "logp_floor", "pit", "brier_2K", "brier_3K", "brier_4K", "crps",
                   "crps_fair", "mean_err", "mean_err_signed", "mean_sqerr", "spread",
                   "n_eff", "p_clim_obs", "field_bias", "field_rmse", "field_pattern_r")
_STYLE_FALLBACK = {"aires": "AI+RES", "cfs": "CFSv2", "cfs13": "CFSv2",
                   "ec46": "ECMWF IFS (EC46)", "gefs": "GEFSv12", "geps": "ECCC GEPS",
                   "bbsubs": "BB-SUBS (est.)"}


def label(name: str) -> str:
    st = getattr(S2, "MODEL_STYLE", {})
    return st.get(name, {}).get("label", _STYLE_FALLBACK.get(name, name))


# --------------------------------------------------------------------------- #
# Scores (pure numpy, unit-tested)
# --------------------------------------------------------------------------- #
def _norm_w(x: np.ndarray, w) -> np.ndarray:
    if w is None:
        return np.full(x.size, 1.0 / x.size)
    w = np.asarray(w, dtype="float64")
    if w.shape != x.shape or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, same shape as x, positive sum")
    return w / w.sum()


def crps_ensemble(x, y: float, w=None, fair: bool = False) -> float:
    """CRPS of the (weighted) ensemble distribution sum_i w_i delta(x_i) at outcome y.

        CRPS = sum_i w_i |x_i - y| - 1/2 sum_ij w_i w_j |x_i - x_j|      (w normalized)

    = the integral of (F(z) - 1[z >= y])^2 for the step CDF F, i.e. the CRPS of the
    forecast distribution as issued. Equal weights give the usual ensemble CRPS.
    fair=True divides the spread term's off-diagonal sum by 1 - sum w_i^2 (an unbiased
    E|X - X'| for exchangeable draws; equal weights give the 'fair' CRPS, m/(m-1)).
    """
    x = np.asarray(x, dtype="float64").ravel()
    if x.size == 0 or not np.all(np.isfinite(x)) or not np.isfinite(y):
        return np.nan
    wn = _norm_w(x, w)
    t1 = float(np.sum(wn * np.abs(x - y)))
    t2 = float(np.sum(wn[:, None] * wn[None, :] * np.abs(x[:, None] - x[None, :])))
    if fair:
        s2 = float(np.sum(wn ** 2))
        if s2 >= 1.0:
            return np.nan
        t2 = t2 / (1.0 - s2)
    return t1 - 0.5 * t2


def crps_gauss(mu, sd, y):
    """CRPS of N(mu, sd^2) at y (closed form); sd <= 0 is a point mass: |mu - y|."""
    from scipy.stats import norm
    mu, sd, y = (np.asarray(v, dtype="float64") for v in (mu, sd, y))
    good = np.isfinite(sd) & (sd > 0)
    s = np.where(good, sd, 1.0)
    z = (y - mu) / s
    c = s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    out = np.where(good, c, np.abs(mu - y))
    return float(out) if out.ndim == 0 else out


def weighted_sd(x, wn) -> float:
    """Unbiased (reliability-weights) sd: sum w (x - mu)^2 / (1 - sum w^2); ddof=1 for equal w."""
    x, wn = np.asarray(x, dtype="float64"), np.asarray(wn, dtype="float64")
    s2 = float(np.sum(wn ** 2))
    if x.size < 2 or s2 >= 1.0:
        return np.nan
    mu = float(np.sum(wn * x))
    return float(np.sqrt(np.sum(wn * (x - mu) ** 2) / (1.0 - s2)))


def p_floor(p: float, n: int) -> float:
    return max(float(p), 1.0 / (n + 1))


def _tail_o(obs: float, s: float) -> dict:
    return {k: float(s * obs >= k) for k in BRIER_K}


def ensemble_metrics(a, obs: float, s: float, kind: str, n_floor: int) -> dict:
    """Scores of one equal-weight ensemble (kind 'emp' or 'gauss'), cfsbase rules."""
    a = np.asarray(a, dtype="float64")
    p = CB.cfs_prob(a, obs, s, kind)
    pf = p_floor(p, n_floor)
    mu = float(a.mean())
    sd = float(a.std(ddof=1)) if a.size > 1 else np.nan
    o = _tail_o(obs, s)
    out = dict(p_obs=p, p_floor=pf, logp_floor=float(np.log(pf)),
               pit=CB.cfs_pit(a, obs, s, kind), n_members=int(a.size), n_eff=float(a.size))
    for k in BRIER_K:
        pk = CB.cfs_prob(a, s * k, s, kind)
        out[f"o_{k:g}K"], out[f"p_{k:g}K"] = o[k], pk
        out[f"brier_{k:g}K"] = CB.brier(pk, o[k])
    if kind == "emp":
        out["crps"] = crps_ensemble(a, obs)
        out["crps_fair"] = crps_ensemble(a, obs, fair=True) if a.size > 1 else np.nan
    else:
        out["crps"] = out["crps_fair"] = crps_gauss(mu, sd, obs)
    out.update(_mean_cols(mu, obs, s, (mu - obs) ** 2, sd))
    return out


def _mean_cols(mu: float, obs: float, s: float, sqerr: float, spread: float) -> dict:
    return dict(fc_mean=mu, mean_err=mu - obs, mean_err_signed=s * (mu - obs),
                mean_sqerr=float(sqerr), spread=spread)


def aires_metrics(c: AN.Case) -> dict:
    """Scores of one AI+RES case: probabilities via `Case.p_sn` (the published estimator),
    CRPS / moments with the self-normalized importance weights."""
    al, s, obs = np.asarray(c.al, dtype="float64"), float(c.sign), float(c.obs)
    wn = _norm_w(al, c.weights)
    p = c.p_sn(obs)
    pf = p_floor(p, N_RES)
    mu = float(np.sum(wn * al))
    o = _tail_o(obs, s)
    out = dict(p_obs=p, p_floor=pf, logp_floor=float(np.log(pf)),
               pit=c.cdf_raw(obs) / c.norm, n_members=int(al.size),
               n_eff=float(1.0 / np.sum(wn ** 2)))
    for k in BRIER_K:
        pk = c.p_sn(s * k)
        out[f"o_{k:g}K"], out[f"p_{k:g}K"] = o[k], pk
        out[f"brier_{k:g}K"] = CB.brier(pk, o[k])
    out["crps"] = crps_ensemble(al, obs, wn)
    out["crps_fair"] = crps_ensemble(al, obs, wn, fair=True)
    out.update(_mean_cols(mu, obs, s, (mu - obs) ** 2, weighted_sd(al, wn)))
    return out


# --------------------------------------------------------------------------- #
# Expected scores over random m-member subsets
# --------------------------------------------------------------------------- #
def subset_idx(n: int, m: int, r: int, key) -> np.ndarray:
    """(r, m) indices of r uniformly random m-subsets of range(n), seeded by `key`."""
    rng = np.random.default_rng([RESAMPLE_SEED, *[int(k) for k in key]])
    return np.argsort(rng.random((r, n)), axis=1)[:, :m]


def _hyper(n: int, K: int, m: int):
    from scipy.stats import hypergeom
    k = np.arange(max(0, m - (n - K)), min(K, m) + 1)
    return k, hypergeom.pmf(k, n, K, m)


def _mean_logp_floor(p, fl: float) -> float:
    """Mean of log(max(p, fl)), clipped to [log fl, 0]. Every term lies in that range, so
    the clip only removes float noise: 1000 subsets that are all floored must give
    exactly log(fl), not log(fl) - 4e-16 (which a paired test would count as a loss)."""
    lf = float(np.log(fl))
    return float(min(max(np.log(np.maximum(p, fl)).mean(), lf), 0.0))


def e16_emp(a, obs: float, s: float, m: int = N_SUB) -> dict:
    """EXACT expected empirical scores of a uniformly random m-subset of the members `a`."""
    a = np.asarray(a, dtype="float64")
    n = a.size
    m = min(m, n)

    def expect(thr, f):
        K = int(np.sum(s * (a - thr) >= 0.0))
        k, pmf = _hyper(n, K, m)
        return float(np.sum(pmf * f(k / m)))
    fl = 1.0 / (m + 1)
    o = _tail_o(obs, s)
    out = dict(p_obs=expect(obs, lambda q: q),
               p_floor=expect(obs, lambda q: np.maximum(q, fl)),
               logp_floor=min(max(expect(obs, lambda q: np.log(np.maximum(q, fl))),
                                  float(np.log(fl))), 0.0),
               pit=float(np.mean(s * a < s * obs)), n_members=m, n_eff=float(m))
    for k in BRIER_K:
        out[f"o_{k:g}K"] = o[k]
        out[f"p_{k:g}K"] = expect(s * k, lambda q: q)
        out[f"brier_{k:g}K"] = expect(s * k, lambda q, ok=o[k]: (q - ok) ** 2)
    A = float(np.mean(np.abs(a - obs)))
    if n > 1:
        dbar = float(np.sum(np.abs(a[:, None] - a[None, :]))) / (n * (n - 1))
        out["crps"] = A - 0.5 * (m - 1) / m * dbar
        out["crps_fair"] = A - 0.5 * dbar if m > 1 else np.nan
        var_mean = float(a.var()) / m * (n - m) / (n - 1)
        spread = float(a.std(ddof=1))
    else:
        out["crps"], out["crps_fair"], var_mean, spread = A, np.nan, 0.0, np.nan
    mu = float(a.mean())
    out.update(_mean_cols(mu, obs, s, (mu - obs) ** 2 + var_mean, spread))
    return out


def e16_gauss(a, obs: float, s: float, idx: np.ndarray) -> dict:
    """Expected Gaussian-fit scores over the subsets `idx` (r, m) of the members `a`."""
    from scipy.stats import norm
    a = np.asarray(a, dtype="float64")
    X = a[idx]
    m = X.shape[1]
    mu = X.mean(1)
    sd = X.std(1, ddof=1)
    good = sd > 0

    def prob(thr):
        emp = np.mean(s * (X - thr) >= 0.0, axis=1)
        g = norm.sf((s * thr - s * mu) / np.where(good, sd, 1.0))
        return np.where(good, g, emp)
    p = prob(obs)
    fl = 1.0 / (m + 1)
    o = _tail_o(obs, s)
    out = dict(p_obs=float(p.mean()), p_floor=float(np.maximum(p, fl).mean()),
               logp_floor=_mean_logp_floor(p, fl),
               pit=float((1.0 - p).mean()), n_members=int(m), n_eff=float(m))
    for k in BRIER_K:
        pk = prob(s * k)
        out[f"o_{k:g}K"], out[f"p_{k:g}K"] = o[k], float(pk.mean())
        out[f"brier_{k:g}K"] = float(((pk - o[k]) ** 2).mean())
    out["crps"] = out["crps_fair"] = float(np.mean(crps_gauss(mu, sd, obs)))
    out.update(_mean_cols(float(mu.mean()), obs, s, float(((mu - obs) ** 2).mean()),
                          float(np.sqrt(np.mean(sd ** 2)))))
    return out


def e16_weighted(al, w, obs: float, s: float, idx: np.ndarray) -> dict:
    """Expected self-normalized scores over walker subsets `idx` (r, m)."""
    al, w = np.asarray(al, dtype="float64"), np.asarray(w, dtype="float64")
    X, W = al[idx], w[idx]
    W = W / W.sum(1, keepdims=True)
    m = X.shape[1]
    fl = 1.0 / (m + 1)

    def prob(thr):
        return np.sum(W * (s * (X - thr) >= 0.0), axis=1)
    p = prob(obs)
    o = _tail_o(obs, s)
    s2 = np.sum(W ** 2, axis=1)
    out = dict(p_obs=float(p.mean()), p_floor=float(np.maximum(p, fl).mean()),
               logp_floor=_mean_logp_floor(p, fl),
               pit=float(np.sum(W * (s * X < s * obs), axis=1).mean()), n_members=int(m),
               n_eff=float(np.mean(1.0 / s2)))
    for k in BRIER_K:
        pk = prob(s * k)
        out[f"o_{k:g}K"], out[f"p_{k:g}K"] = o[k], float(pk.mean())
        out[f"brier_{k:g}K"] = float(((pk - o[k]) ** 2).mean())
    t1 = np.sum(W * np.abs(X - obs), axis=1)
    t2 = np.einsum("ri,rj,rij->r", W, W, np.abs(X[:, :, None] - X[:, None, :]))
    out["crps"] = float(np.mean(t1 - 0.5 * t2))
    out["crps_fair"] = float(np.mean(t1 - 0.5 * t2 / (1.0 - s2)))
    mu = np.sum(W * X, axis=1)
    var = np.sum(W * (X - mu[:, None]) ** 2, axis=1) / (1.0 - s2)
    out.update(_mean_cols(float(mu.mean()), obs, s, float(((mu - obs) ** 2).mean()),
                          float(np.sqrt(var.mean()))))
    return out


# --------------------------------------------------------------------------- #
# Collect: board_cases rows
# --------------------------------------------------------------------------- #
def variant_flags(v: str) -> dict:
    base = v[4:] if v.startswith("e16_") else v
    ens = ("e16" if v.startswith("e16_") else "fixed16" if base.startswith("s16")
           else "last4" if base.startswith("sub") else "native")
    kind = "weighted" if base == "sn" else base.split("_")[1]
    return dict(variant=v, ens=ens, kind=kind, bias_corrected=base.startswith("corr"),
                headline=base in ("raw_emp", "sn") and ens == "native")


def _case_meta(r, i: int) -> dict:
    return dict(case_idx=i, episode_id=r.episode_id, family=r.family, rung=int(r.rung),
                peak=str(pd.Timestamp(r.peak).date()), tail_sign=1.0 if r.family == "heat"
                else -1.0)


def aires_rows(tr: TR.Truth, slate: pd.DataFrame, daily: pd.Series) -> list[dict]:
    rows = []
    for win in AIRES_WINDOWS:
        cases = S2.aires_cases(tr, win)
        for i, r in enumerate(slate.itertuples()):
            c = cases[r.episode_id]
            meta = _case_meta(r, i)
            if c.sign != meta["tail_sign"]:
                raise SystemExit(f"[board] {r.episode_id}: AI+RES sign != slate family")
            pc = AN.p_clim(AN.clim_pool(daily, r.peak), c.obs, c.sign)[0]
            head = dict(truth=tr.name, source=AIRES, label=label(AIRES), role="aires",
                        estimate=False, window=win, fc_window=win, **meta, obs=float(c.obs),
                        n_native=int(len(c.al)), p_clim_obs=pc)
            rows.append({**head, **variant_flags("sn"), **aires_metrics(c)})
            idx = subset_idx(len(c.al), N_SUB, N_RESAMPLE_AIRES, (i, len(c.al), 0))
            rows.append({**head, **variant_flags("e16_sn"),
                         **e16_weighted(c.al, c.weights, c.obs, c.sign, idx)})
    return rows


def _nan_metrics(m: dict) -> dict:
    return {k: (np.nan if isinstance(v, float) and not k.startswith("o_") else v)
            for k, v in m.items()}


def source_rows(src: S2.Source, tr: TR.Truth, slate: pd.DataFrame,
                daily: pd.Series) -> list[dict]:
    recs = S2.load(src, tr)
    if recs.empty:
        return []
    variants = src.variants(int(recs.al.map(len).max()))
    srows = {r.episode_id: (i, r) for i, r in enumerate(slate.itertuples())}
    role = "reference" if src.name in REFERENCE else "board"
    rows = []
    for rec in recs.to_dict("records"):
        eid = rec["episode_id"]
        i, srow = srows[eid]
        meta = _case_meta(srow, i)
        al_raw = np.asarray(rec["al"], dtype="float64")
        s, obs, bias = float(rec["sign"]), float(rec["obs"]), float(rec["bias"])
        if s != meta["tail_sign"]:
            raise SystemExit(f"[board] {src.name} {eid}: sign != slate family")
        pc = AN.p_clim(AN.clim_pool(daily, rec["peak"]), obs, s)[0]
        head = dict(truth=tr.name, source=src.name, label=label(src.name), role=role,
                    estimate=False, window=src.obs_window, fc_window=src.window, **meta,
                    obs=obs, n_native=int(al_raw.size), p_clim_obs=pc, bias_conus=bias)
        for v in variants:
            a, kind = CB.members(al_raw, bias, v), v.split("_")[1]
            m = ensemble_metrics(a, obs, s, kind, CB.n_floor(v, al_raw.size))
            if v.startswith("corr") and not np.isfinite(bias):
                m = _nan_metrics(m)
            rows.append({**head, **variant_flags(v), **m})
        for v in E16_BASE:
            if v not in variants:
                continue
            a, kind = CB.members(al_raw, bias, v), v.split("_")[1]
            mm = min(N_SUB, a.size)
            if kind == "emp":
                m = e16_emp(a, obs, s, mm)
            else:
                m = e16_gauss(a, obs, s, subset_idx(a.size, mm, N_RESAMPLE_GAUSS,
                                                    (i, a.size, 1)))
            if v.startswith("corr") and not np.isfinite(bias):
                m = _nan_metrics(m)
            rows.append({**head, **variant_flags("e16_" + v), **m})
    return rows


def aires_mean_field(eid: str, window: str) -> "xr.DataArray":
    """Importance-weighted (self-normalized) AI+RES ensemble-mean field of one case."""
    f, w = S2.aires_fields(eid, window)
    wn = np.asarray(w, dtype="float64") / np.sum(w)
    dim = [d for d in f.dims if d not in ("lat", "lon")]
    if len(dim) != 1 or f.sizes[dim[0]] != wn.size:
        raise SystemExit(f"[board] {eid}: AI+RES fields {dict(f.sizes)} vs {wn.size} weights")
    import xarray as xr
    return (f.astype("float64") * xr.DataArray(wn, dims=dim)).sum(dim[0])


def field_metrics(tr: TR.Truth, sources: list[str], workers: int = 8) -> pd.DataFrame:
    """Per (source, window, case): bias / RMSE / pattern r of the raw ensemble-mean 7-day
    field against the truth over CONUS land on the truth's mask (errmaps' grid and stats)."""
    from acal import errmaps as E
    b = E.build_board(tr.name, workers=workers)
    rows = []

    def add(source, window, F):
        O = b.obs[window]
        for i, e in enumerate(b.cases.episode_id):
            if not np.isfinite(F[i]).any():
                continue
            bias, rmse, r = E.wstats(F[i], O[i], b.valid, b.coslat)
            rows.append(dict(source=source, window=window, episode_id=e, field_bias=bias,
                             field_rmse=rmse, field_pattern_r=r))
    for win in AIRES_WINDOWS:
        F = np.stack([E._grid_like(aires_mean_field(e, win), b.lat, b.lon, f"aires {e}")
                      for e in b.cases.episode_id])
        add(AIRES, win, F)
    for m in b.models:
        if m.name != "aires" and m.name in sources:
            add(m.name, m.window, m.F)
    if "cfs" in sources:
        ds = E.source_ensmean("cfs", b.cases, b.lat, b.lon, workers)
        if ds is not None:
            add("cfs", "13f", E._grid_like(ds["ensmean"], b.lat, b.lon, "cfs")
                .astype("float64"))
    return pd.DataFrame(rows)


def collect_truth(truth: str, sources=SOURCES, fields: bool = True,
                  workers: int = 8) -> pd.DataFrame:
    """All board_cases rows of one truth (measured rows only, no estimates)."""
    t0 = time.time()
    tr = TR.get_truth(truth)
    slate = aprep.episodes().reset_index(drop=True)
    daily = TR.pool_series(tr)
    rows = aires_rows(tr, slate, daily)
    present = []
    for name in sources:
        if not S2.has_data(name):
            print(f"[board] {truth}: {name} has no data yet - absent from the board",
                  flush=True)
            continue
        src = S2.get(name)
        got = source_rows(src, tr, slate, daily)
        print(f"[board] {truth}: {name} {len(got)} rows ({time.time() - t0:.0f} s)",
              flush=True)
        if got:
            present.append(name)
        rows += got
    df = pd.DataFrame(rows)
    for c in ("field_bias", "field_rmse", "field_pattern_r"):
        df[c] = np.nan
    if fields:
        fm = field_metrics(tr, present, workers)
        if not fm.empty:
            key = ["source", "window", "episode_id"]
            sel = df.variant.isin(FIELD_VARIANTS)
            j = df.loc[sel, key].merge(fm, on=key, how="left")
            for c in ("field_bias", "field_rmse", "field_pattern_r"):
                df.loc[sel, c] = j[c].to_numpy()
    print(f"[board] {truth}: {len(df)} rows in {time.time() - t0:.0f} s", flush=True)
    return df


def merge_estimates(df: pd.DataFrame, rows_csv: Path = BBSUBS_ROWS_CSV,
                    not_before: float | None = None) -> pd.DataFrame:
    """Append BB-SUBS estimate rows (acal.bbsubs writes them) flagged estimate=True.

    A rows file older than `not_before` (an epoch time) is stale - written from an older
    board - and is left out with a message."""
    rows_csv = Path(rows_csv)
    if not rows_csv.exists():
        return df
    if not_before is not None and rows_csv.stat().st_mtime < not_before:
        print(f"[board] {rows_csv.name} predates this board: not merged", flush=True)
        return df
    est = pd.read_csv(rows_csv)
    if est.empty:
        return df
    est = est[[c for c in est.columns if c in df.columns or c in ("tier", "anchor")]].copy()
    est["estimate"] = True
    est["role"] = "estimate"
    # bbsubs blanks every numeric column it does not scale or copy, case_idx included
    est["case_idx"] = est.episode_id.map(dict(zip(df.episode_id, df.case_idx)))
    est["source"] = est.get("source", "bbsubs")
    if "variant" in est:
        for k in ("ens", "kind", "bias_corrected", "headline"):
            est[k] = [variant_flags(str(v))[k] for v in est.variant]
    for c in ("field_bias", "field_rmse", "field_pattern_r"):
        if c in est:
            est[c] = np.nan                     # an estimate never carries maps
    print(f"[board] merged {len(est)} BB-SUBS estimate rows from {rows_csv.name}", flush=True)
    return pd.concat([df, est], ignore_index=True)


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    order = {n: i for i, n in enumerate((AIRES,) + SOURCES + ("bbsubs",))}
    vorder = {v: i for i, v in enumerate(
        ("sn", "e16_sn", "raw_emp", "raw_gauss", "corr_emp", "corr_gauss", "s16_emp",
         "sub_emp", "e16_raw_emp", "e16_raw_gauss", "e16_corr_emp", "e16_corr_gauss"))}
    t = {n: i for i, n in enumerate(TRUTHS)}
    k = pd.DataFrame(dict(
        t=df.truth.map(t).fillna(99), s=df.source.map(order).fillna(99),
        w=df.window.map({"13f": 0, "12f": 1}).fillna(9), e=df.estimate.astype(bool),
        v=df.variant.map(vorder).fillna(99), c=df.case_idx))
    return df.loc[k.sort_values(list(k.columns), kind="stable").index].reset_index(drop=True)


def _atomic_csv(df: pd.DataFrame, p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}.csv")
    df.to_csv(tmp, index=False)                 # full precision: paired reads it back
    os.replace(tmp, p)
    return p


def _collect_job(args) -> pd.DataFrame:
    return collect_truth(*args)


def collect(truths=TRUTHS, sources=SOURCES, fields: bool = True, workers: int = 8,
            bbsubs: bool = True, jobs: int = 3) -> pd.DataFrame:
    """Rebuild the rows of `truths` (other truths in an existing board_cases.csv are kept),
    write board_cases.csv, let acal.bbsubs derive its estimate rows from it, merge them.
    `jobs` > 1 runs the truths in parallel processes (each re-reduces the cubes on its
    mask, ~3-5 min for a masked truth); their field step then runs single-process."""
    jobs = max(1, min(jobs, len(truths)))
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            parts = list(ex.map(_collect_job, [(t, sources, fields, 1) for t in truths]))
    else:
        parts = [collect_truth(t, sources, fields, workers) for t in truths]
    new = pd.concat(parts, ignore_index=True)
    if CASES_CSV.exists():
        old = pd.read_csv(CASES_CSV, float_precision="round_trip")
        old = old[~old.estimate.astype(bool)]
        keep = ~old.truth.isin(truths) | ~old.source.isin(set(new.source))
        new = pd.concat([old[keep], new], ignore_index=True)
    new = _sort(new)
    t_written = time.time()
    _atomic_csv(new, CASES_CSV)
    if bbsubs:
        try:
            from acal import bbsubs as BB
            BB.run_estimate(board_csv=CASES_CSV, out_csv=BBSUBS_ROWS_CSV, tier2=False,
                            log=lambda m: print(m, flush=True))
        except ImportError:
            print("[board] acal.bbsubs not importable: BB-SUBS rows only from a file",
                  flush=True)
        merged = merge_estimates(new, BBSUBS_ROWS_CSV, not_before=t_written - 1.0)
        if len(merged) > len(new):
            new = _sort(merged)
            _atomic_csv(new, CASES_CSV)
    print(f"[board] {len(new)} rows -> {CASES_CSV}", flush=True)
    return new


# --------------------------------------------------------------------------- #
# Paired
# --------------------------------------------------------------------------- #
def aires_partner(variant: str) -> str:
    return "e16_sn" if variant.startswith("e16_") else "sn"


def paired_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Per-case differences of every model row against its AI+RES partner (same truth,
    same truth window, 'sn' or 'e16_sn'); positive = AI+RES better."""
    res = df[df.source == AIRES]
    out = []
    for (truth, source, est, variant), g in df[df.source != AIRES].groupby(
            ["truth", "source", "estimate", "variant"], sort=False):
        win = g.window.iloc[0]
        a = res[(res.truth == truth) & (res.window == win)
                & (res.variant == aires_partner(variant))].set_index("episode_id")
        g = g[g.episode_id.isin(a.index)].sort_values("case_idx")
        if g.empty:
            continue
        r = a.loc[g.episode_id]
        dobs = np.abs(g.obs.to_numpy() - r.obs.to_numpy())
        if np.nanmax(dobs) > OBS_TOL:
            raise SystemExit(f"[board] {truth} {source} {variant}: obs != AI+RES obs "
                             f"(max |d| {np.nanmax(dobs):.3g} K) - not like-for-like")
        if variant.startswith("e16_"):
            lr = r.logp_floor.to_numpy() - g.logp_floor.to_numpy()
            lr = np.where(np.abs(lr) < TIE_TOL, 0.0, lr)   # float pseudo-ties -> ties
        else:                                   # cfsbase.log_ratio, bit for bit
            lr = np.log(r.p_floor.to_numpy() / g.p_floor.to_numpy())
        d = dict(logratio=lr)
        for k in BRIER_K:
            d[f"dbrier_{k:g}K"] = g[f"brier_{k:g}K"].to_numpy() - r[f"brier_{k:g}K"].to_numpy()
        d["dcrps"] = g.crps.to_numpy() - r.crps.to_numpy()
        d["dsqerr"] = g.mean_sqerr.to_numpy() - r.mean_sqerr.to_numpy()
        d["dfield_rmse"] = g.field_rmse.to_numpy() - r.field_rmse.to_numpy()
        side = SIDE_COL
        base = dict(truth=truth, source=source, label=g.label.iloc[0], estimate=bool(est),
                    variant=variant, aires_variant=aires_partner(variant), window=win)
        for j, (_, row) in enumerate(g.iterrows()):
            rec = dict(base, case_idx=int(row.case_idx), episode_id=row.episode_id,
                       family=row.family, rung=int(row.rung), obs=float(row.obs))
            for mname, vals in d.items():
                rec[mname] = float(vals[j])
                col = side[mname]
                rec[f"aires_{col}"] = float(r[col].iloc[j])
                rec[f"model_{col}"] = float(row[col])
            out.append(rec)
    return pd.DataFrame(out)


def paired_table(pc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (truth, source, est, variant), g in pc.groupby(
            ["truth", "source", "estimate", "variant"], sort=False):
        for sub in SUBSETS:
            d = g if sub == "all" else g[g.family == sub]
            for mname in PAIRED_METRICS:
                ac = SIDE_COL[mname]
                x = d[mname].to_numpy(dtype="float64")
                ok = np.isfinite(x)
                st = CB.paired_stats(x)
                rows.append(dict(truth=truth, source=source, label=g.label.iloc[0],
                                 estimate=bool(est), variant=variant,
                                 aires_variant=g.aires_variant.iloc[0],
                                 window=g.window.iloc[0], subset=sub, metric=mname,
                                 **st,
                                 mean_aires=float(np.mean(d[f"aires_{ac}"].to_numpy()[ok]))
                                 if ok.any() else np.nan,
                                 mean_model=float(np.mean(d[f"model_{ac}"].to_numpy()[ok]))
                                 if ok.any() else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _stat(x) -> dict:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if not x.size:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)))


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def case_summary(df: pd.DataFrame) -> dict:
    out: dict = {}
    for (truth, source, est, win, variant), g in df.groupby(
            ["truth", "source", "estimate", "window", "variant"], sort=False):
        key = f"{source} (estimate)" if est else source
        node = out.setdefault(truth, {}).setdefault(key, {}).setdefault(win, {})
        node[variant] = {}
        for sub in SUBSETS:
            d = g if sub == "all" else g[g.family == sub]
            s = {m: _stat(d[m]) for m in SUMMARY_METRICS if m in d}
            s["n_cases"] = int(len(d))
            s["n_zero_obs"] = int((d.p_obs == 0).sum()) if "p_obs" in d else None
            s["mae"] = _stat(np.abs(d.mean_err)).get("mean") if "mean_err" in d else None
            ms = d.mean_sqerr.to_numpy(dtype="float64") if "mean_sqerr" in d else np.array([])
            ms = ms[np.isfinite(ms)]
            s["rmse_of_mean"] = float(np.sqrt(ms.mean())) if ms.size else None
            node[variant][sub] = s
    return out


def paired_summary(pt: pd.DataFrame) -> dict:
    out: dict = {}
    keys = ("n", "mean", "median", "ci_lo", "ci_hi", "win", "tie", "loss", "wilcoxon_p",
            "mean_aires", "mean_model")
    for r in pt.itertuples():
        key = f"{r.source} (estimate)" if r.estimate else r.source
        node = (out.setdefault(r.truth, {}).setdefault(key, {})
                .setdefault(r.variant, {}).setdefault(r.subset, {}))
        node[r.metric] = {k: getattr(r, k) for k in keys}
    return out


def coverage(df: pd.DataFrame) -> dict:
    slate = aprep.episodes().set_index("episode_id")
    cov = {}
    lead_res = (pd.to_datetime(slate.peak) - pd.to_datetime(slate.init)).dt.days
    cov[AIRES] = dict(label=label(AIRES), role="aires", n_cases=int(len(slate)),
                      n_members=N_RES, member_kind="DMC walkers, importance-weighted",
                      lead_days_to_peak=[float(lead_res.min()), float(lead_res.max())],
                      windows=list(AIRES_WINDOWS), native_grid="0.25 deg GenCast",
                      time_kind="instant")
    for name in sorted(set(df.source) - {AIRES}, key=lambda n: (SOURCES + ("bbsubs",)).index(n)
                       if n in SOURCES + ("bbsubs",) else 99):
        d = df[(df.source == name) & (df.variant == "raw_emp") & ~df.estimate.astype(bool)]
        if name == "bbsubs" or d.empty:
            e = df[df.source == name]
            cov[name] = dict(label=label(name), role="estimate", estimate=True,
                             n_cases=int(e.episode_id.nunique()),
                             anchor=sorted(set(map(str, e.get("anchor", pd.Series()).dropna()))),
                             native_grid="none (estimate, never maps)")
            continue
        src = S2.get(name)
        leads = []
        for eid in d.episode_id.unique():
            p = src.json_path(eid)
            if p.exists():
                leads += [float(x) for x in json.loads(p.read_text()).get("member_lead_days",
                                                                          [])]
        st = S2.MODEL_STYLE.get(name, {}) if hasattr(S2, "MODEL_STYLE") else {}
        cov[name] = dict(
            label=label(name), role="reference" if name in REFERENCE else "board",
            n_cases=int(d.episode_id.nunique()),
            n_members=[int(d.n_native.min()), int(d.n_native.max())],
            member_lead_days=[min(leads), max(leads)] if leads else None,
            window=src.window, truth_window=src.obs_window, time_kind=src.time_kind,
            native_grid=st.get("deg", f"{src.native_deg:g} deg"), native_deg=src.native_deg,
            init_rule=src.init_rule, bias=src.bias, bias_available=src.has_bias(),
            bias_note=src.bias_note, dataset_id=src.dataset_id, url=src.url,
            variants=sorted(set(df[df.source == name].variant)))
    absent = [n for n in SOURCES if n not in cov]
    return dict(sources=cov, absent=absent,
                truths=sorted(set(df.truth), key=lambda t: TRUTHS.index(t)
                              if t in TRUTHS else 99))


CONVENTIONS = dict(
    sign="tail-signed: s = +1 heat, -1 cold; o_k = 1[s*obs >= k]; P = P(s*A >= s*a)",
    p_obs="AI+RES self-normalized importance weights (Case.p_sn, as published); models "
          "empirical (fraction of members, >=) or Gaussian fit (ddof=1)",
    logratio="log(P_AIRES/P_model), floors 1/(N+1) each side (N = 32 walkers, model N "
             "scored); e16 rows: E[log floored P] per side over random 16-member subsets",
    differences="model minus AI+RES for Brier / CRPS / squared error / field RMSE: "
                "positive = AI+RES better (as cfsbase.paired)",
    crps="CRPS of the forecast distribution (members, or walkers with self-normalized "
         "weights; Gaussian closed form for *_gauss); crps_fair = ensemble-size-adjusted",
    windows="instant sources and AI+RES '13f' on the 13 00/12Z frames peak-6d..peak; "
            "daily-mean sources (GEPS, EC46) on UTC days peak-6..peak-1 with truth and "
            "AI+RES on the 12 frames peak-6d..peak-1d 12Z",
    headline="variant raw_emp (models) vs sn (AI+RES), native ensemble sizes, truth hrrr "
             "for the HRRR board and era5 for the ERA5 board",
    e16="expected value over uniformly random 16-member subsets: exact (hypergeometric) "
        f"for empirical variants, {N_RESAMPLE_GAUSS} seeded resamples for Gaussian ones, "
        f"{N_RESAMPLE_AIRES} seeded 16-walker resamples for AI+RES (seed {RESAMPLE_SEED}). "
        "AI+RES subsets re-normalize the importance weights within the subset: with n_eff "
        "~5 of 32 that ratio estimator moves the subset mean toward the extreme (era5 13f: "
        "mean signed error -0.97 K native, -0.76 K e16), so e16 is a sensitivity, not a "
        "16-walker run",
    bootstrap=f"case bootstrap, {int(CB.CI * 100)}% CI, n={CB.N_BOOT}, seed={CB.BOOT_SEED}",
    caveat="All 42 cases are selected on |obs| >= 2 K: P(obs) and Brier at 2 K are mass on "
           "the observed tail, not calibration; cases share seasons, so CIs are optimistic.",
    field="field_* = raw ensemble-mean 7-day T2m anomaly vs truth, CONUS land on the truth's "
          "mask, cos-lat weighted (errmaps.wstats); raw_emp/raw_gauss/sn rows only",
)


def headline(df: pd.DataFrame, pt: pd.DataFrame) -> dict:
    """Compact digest: per truth and source, the AI+RES-vs-model paired means (raw_emp vs
    sn, native and e16) and each side's mean CRPS / median P(obs) on the matched window."""
    out: dict = {}
    keep = ("mean", "ci_lo", "ci_hi", "win", "tie", "loss", "wilcoxon_p")
    for (truth, source, est), g in pt[pt.subset == "all"].groupby(
            ["truth", "source", "estimate"], sort=False):
        node = out.setdefault(truth, {}).setdefault(
            f"{source} (estimate)" if est else source, {})
        for v in ("raw_emp", "corr_emp", "e16_raw_emp"):
            gv = g[g.variant == v]
            if gv.empty:
                continue
            node[v] = {r.metric: {k: getattr(r, k) for k in keep} for r in gv.itertuples()
                       if r.metric in ("logratio", "dcrps", "dbrier_3K", "dbrier_4K")}
        m = df[(df.truth == truth) & (df.source == source) & (df.estimate == est)
               & (df.variant == "raw_emp")]
        if not m.empty:
            a = df[(df.truth == truth) & (df.source == AIRES) & (df.variant == "sn")
                   & (df.window == m.window.iloc[0])]
            node["window"] = m.window.iloc[0]
            node["crps_mean"] = {"aires": float(a.crps.mean()), "model": float(m.crps.mean())}
            node["p_obs_median"] = {"aires": float(a.p_obs.median()),
                                    "model": float(m.p_obs.median())}
    return out


def paired(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None:
        if not CASES_CSV.exists():
            raise SystemExit(f"[board] {CASES_CSV} missing: run --stage collect")
        df = pd.read_csv(CASES_CSV, float_precision="round_trip")
    pc = paired_cases(df)
    pt = paired_table(pc)
    _atomic_csv(pc, PAIRED_CASES_CSV)
    _atomic_csv(pt, PAIRED_CSV)
    summ = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), conventions=CONVENTIONS,
                headline=headline(df, pt), coverage=coverage(df), cases=case_summary(df),
                paired=paired_summary(pt))
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY_JSON.with_suffix(f".tmp{os.getpid()}.json")
    tmp.write_text(json.dumps(_clean(summ), indent=1))
    os.replace(tmp, SUMMARY_JSON)
    print(f"[board] {len(pc)} paired case rows -> {PAIRED_CASES_CSV}\n"
          f"[board] {len(pt)} paired stats -> {PAIRED_CSV}\n[board] summary -> {SUMMARY_JSON}")
    head = pt[(pt.subset == "all") & (pt.metric.isin(("logratio", "dcrps")))
              & pt.variant.isin(("raw_emp", "corr_emp", "e16_raw_emp"))]
    for r in head.itertuples():
        print(f"  {r.truth:8s} {r.source:7s}{' est' if r.estimate else '    '} "
              f"{r.variant:12s} {r.metric:8s} {r.mean:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] "
              f"W/T/L {r.win}/{r.tie}/{r.loss} p={r.wilcoxon_p:.3f}")
    return pc, pt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("collect", "paired", "all"), default="all")
    ap.add_argument("--truth", nargs="+", default=list(TRUTHS), choices=TRUTHS)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--no-fields", action="store_true",
                    help="skip the per-case field metrics (errmaps grid, ~30 s per truth)")
    ap.add_argument("--no-bbsubs", action="store_true",
                    help="do not call acal.bbsubs.run_estimate after writing the cases")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=3, help="truths collected in parallel")
    a = ap.parse_args(argv)
    bad = [s for s in a.sources if s not in SOURCES]
    if bad:
        ap.error(f"unknown sources {bad}; board sources: {SOURCES}")
    df = None
    if a.stage in ("collect", "all"):
        df = collect(tuple(a.truth), tuple(a.sources), fields=not a.no_fields,
                     workers=a.workers, bbsubs=not a.no_bbsubs, jobs=a.jobs)
    if a.stage in ("paired", "all"):
        paired(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
