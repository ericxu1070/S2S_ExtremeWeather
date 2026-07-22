"""Verification metrics for the p90 T2m-anomaly experiment.

Turns the model cubes (CUBE SCHEMA, see ``pconfig``) plus the ERA5 truth / persistence
files into per-case skill scores, using the EXACT xres verification recipe: the forecast
target is the week-mean T2m anomaly over [peak-6d, peak] vs the 1990-2019 climatology,
built by ``xres.xmetrics.forecast_field(cube, peak, "t2m_anom")`` and scored against the
matching ERA5 anomaly truth on the native 0.25deg CONUS grid.

Two views of the ensemble are scored:

  * the week-mean anomaly FIELD (member, lat, lon) -> grid CRPS, rank histograms, ensemble
    RMSE / bias / ACC / spread and the spread-error ratio;
  * the daily CONUS-mean anomaly SERIES per member -> the event probability p_event (the
    fraction of members that cross the p90 threshold on any day of the verification week),
    which feeds Brier / reliability / ROC scoring against the observed event label.

Everything is pure numpy/pandas here (NaN-safe, cos-lat weighted); the cube -> field and
cube -> series reductions delegate to ``xres.xmetrics`` and ``gencast_s2s.data`` (imported
lazily so this module loads in any environment). CRPS uses the standard ensemble estimator
(identical to ``properscoring.crps_ensemble``) via the O(M log M) sorted pairwise formula.

Env vars: none read directly (all case selection / paths flow through ``pconfig``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import pconfig as P

DAY = pd.Timedelta(days=1)


# --------------------------------------------------------------------------- #
# Latitude weighting (mirrors xres.xscores._wmean)
# --------------------------------------------------------------------------- #
def wmean(field, lat) -> float:
    """Cos-lat weighted mean of a (lat, lon) field, NaN-tolerant (NaNs dropped)."""
    field = np.asarray(field, dtype=float)
    lat = np.asarray(lat, dtype=float)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(field)
    m = np.isfinite(field)
    if not m.any():
        return float("nan")
    return float((field[m] * w[m]).sum() / w[m].sum())


# --------------------------------------------------------------------------- #
# Grid CRPS (standard ensemble estimator == properscoring.crps_ensemble)
# --------------------------------------------------------------------------- #
def crps_field(members, truth):
    """Per-gridpoint CRPS of an ensemble against a deterministic truth.

    members[M, ...], truth[...] -> [...] (member axis removed). Estimator:
        CRPS = mean_i |x_i - y| - 0.5 * E|X - X'|
    with the pairwise term via the O(M log M) sorted formulation
        E|X - X'| = (2 / M^2) * sum_k (2k - 1 - M) x_(k),  x_(k) ascending, k = 1..M.
    NaN propagates: any point with a NaN truth or a NaN in any member is NaN."""
    members = np.asarray(members, dtype=float)
    truth = np.asarray(truth, dtype=float)
    M = members.shape[0]
    t1 = np.mean(np.abs(members - truth[None]), axis=0)
    xs = np.sort(members, axis=0)                       # ascending along member axis
    k = np.arange(1, M + 1).reshape((M,) + (1,) * (members.ndim - 1))
    coef = 2 * k - 1 - M
    ex = (2.0 / M ** 2) * np.sum(coef * xs, axis=0)     # E|X - X'|
    crps = t1 - 0.5 * ex
    bad = ~np.isfinite(truth) | ~np.all(np.isfinite(members), axis=0)
    return np.where(bad, np.nan, crps)


# --------------------------------------------------------------------------- #
# Cube reductions (delegated; imports kept lazy so this module loads anywhere)
# --------------------------------------------------------------------------- #
def week_anom_members(cube, peak):
    """Per-member week-mean T2m anomaly field (member, lat, lon) - the xres recipe."""
    from xres import xmetrics as XM
    return XM.forecast_field(cube, peak, "t2m_anom")


def _sel_times(da, times):
    """Select forecast times; use isel when the cube time coord has duplicates."""
    t = pd.DatetimeIndex(da["time"].values)
    want = pd.DatetimeIndex(times)
    if t.is_unique:
        return da.sel(time=want)
    return da.isel(time=t.isin(want))


def daily_conus_series(cube, peak) -> np.ndarray:
    """Per-member daily CONUS-mean T2m anomaly over the verification week.

    Returns [n_member, 7]: for each calendar day d in [peak-6d, peak] take the 4 frames at
    P.HOURS of day d (all must be present), mean over those frames (per member), subtract
    the daily-mean climatology (D.clim_for over the same 4 timestamps), then cos-lat mean
    over CONUS. Mirrors the daily anomaly convention in find_p90_t2m_events.py."""
    from gencast_s2s import data as D

    peak = pd.Timestamp(peak)
    da = cube["2m_temperature"]
    lat = da["lat"].values
    cube_times = pd.DatetimeIndex(da["time"].values)
    days = pd.date_range(peak - 6 * DAY, peak, freq="1D")
    n_member = da.sizes["member"]
    out = np.empty((n_member, 7), dtype=float)
    for j, d in enumerate(days):
        times = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for h in P.HOURS])
        missing = [str(t) for t in times if t not in cube_times]
        assert not missing, f"daily_conus_series: missing frames {missing} for day {d.date()}"
        day_mean = _sel_times(da, times).mean("time")               # (member, lat, lon)
        clim = D.clim_for(times).mean("time")                       # (lat, lon)
        anom = (day_mean - clim).values                             # aligns on coords
        for mi in range(n_member):
            out[mi, j] = wmean(anom[mi], lat)
    return out


def p_event(cube, peak) -> float:
    """Fraction of members whose daily CONUS-mean anomaly crosses THR_K on any week day."""
    s = daily_conus_series(cube, peak)
    return float((s >= P.THR_K).any(axis=1).mean())


# --------------------------------------------------------------------------- #
# Rank histogram (mirrors xres.xscores._rank_hist)
# --------------------------------------------------------------------------- #
def rank_hist_field(members, truth):
    """Bincount of the number of members strictly below the truth, over finite points.
    Length M+1 (ranks 0..M); flat = calibrated, U = under-dispersed, spikes = biased."""
    e = np.asarray(members, dtype=float)
    o = np.asarray(truth, dtype=float)
    M = e.shape[0]
    valid = np.isfinite(o) & np.isfinite(e).all(axis=0)
    below = (e < o[None]).sum(axis=0)
    return np.bincount(below[valid].ravel(), minlength=M + 1)


# --------------------------------------------------------------------------- #
# Brier score, Murphy decomposition, skill score
# --------------------------------------------------------------------------- #
def brier(p, o) -> float:
    """Mean squared error of forecast probability p against binary outcome o."""
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    return float(np.mean((p - o) ** 2))


def brier_decomposition(p, o, nbins: int = 10):
    """Murphy partition of the Brier score into (reliability, resolution, uncertainty).

    Equal-width probability bins over [0, 1]; p is clipped into [0, 1] then digitized;
    empty bins are skipped. The identity BS = REL - RES + UNC holds exactly when p is
    constant within each bin (e.g. p sits on the bin centres).
        REL = (1/N) sum_k n_k (pbar_k - obar_k)^2
        RES = (1/N) sum_k n_k (obar_k - obar)^2
        UNC = obar (1 - obar)"""
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    N = len(p)
    obar = float(o.mean())
    unc = obar * (1.0 - obar)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    pc = np.clip(p, 0.0, 1.0)
    idx = np.clip(np.digitize(pc, edges[1:-1]), 0, nbins - 1)
    rel = res = 0.0
    for k in range(nbins):
        sel = idx == k
        nk = int(sel.sum())
        if nk == 0:
            continue
        pbar = float(pc[sel].mean())
        obar_k = float(o[sel].mean())
        rel += nk * (pbar - obar_k) ** 2
        res += nk * (obar_k - obar) ** 2
    return rel / N, res / N, unc


def bss(p, o) -> float:
    """Brier skill score vs the climatological base rate; nan if obar is 0 or 1."""
    o = np.asarray(o, dtype=float)
    obar = float(o.mean())
    if obar in (0.0, 1.0):
        return float("nan")
    return float(1.0 - brier(p, o) / (obar * (1.0 - obar)))


# --------------------------------------------------------------------------- #
# ROC curve + AUC
# --------------------------------------------------------------------------- #
def roc_curve(p, o):
    """(far, hr, auc): sweep thresholds over sorted unique p plus 0 and 1+eps endpoints so
    the curve spans (0,0)-(1,1). Thresholds are walked from strict to lenient so far / hr
    are non-decreasing; auc = trapezoid area under hr(far)."""
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    ev = o > 0.5
    n_pos = float(ev.sum())
    n_neg = float((~ev).sum())
    thrs = np.unique(np.concatenate([[0.0], np.unique(p), [1.0 + 1e-9]]))[::-1]
    far = np.empty(len(thrs))
    hr = np.empty(len(thrs))
    for i, thr in enumerate(thrs):
        pred = p >= thr
        hr[i] = float((pred & ev).sum()) / n_pos if n_pos > 0 else np.nan
        far[i] = float((pred & ~ev).sum()) / n_neg if n_neg > 0 else np.nan
    auc = float(np.trapz(hr, far))
    return far, hr, auc


# --------------------------------------------------------------------------- #
# Anomaly correlation + ensemble summary stats
# --------------------------------------------------------------------------- #
def acc(f, o, lat) -> float:
    """Cos-lat weighted UNCENTERED anomaly correlation between fields f and o.
        ACC = sum(w f o) / sqrt(sum(w f^2) sum(w o^2))  over jointly finite points."""
    f = np.asarray(f, dtype=float)
    o = np.asarray(o, dtype=float)
    lat = np.asarray(lat, dtype=float)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(f)
    m = np.isfinite(f) & np.isfinite(o)
    if not m.any():
        return float("nan")
    num = (w[m] * f[m] * o[m]).sum()
    den = math.sqrt((w[m] * f[m] ** 2).sum() * (w[m] * o[m] ** 2).sum())
    return float(num / den) if den > 0 else float("nan")


def ens_stats(members, truth, lat) -> dict:
    """Ensemble-mean skill + spread diagnostics for an anomaly field.
        rmse         sqrt(wmean((ens_mean - o)^2))
        bias         wmean(ens_mean - o)
        acc          uncentered ACC of ens_mean vs o
        spread       sqrt(wmean(var over members, ddof=1))
        spread_error spread * sqrt((M+1)/M) / rmse  (~1 for a calibrated ensemble)"""
    members = np.asarray(members, dtype=float)
    truth = np.asarray(truth, dtype=float)
    M = members.shape[0]
    ens_mean = np.nanmean(members, axis=0)
    diff = ens_mean - truth
    rmse = math.sqrt(wmean(diff ** 2, lat))
    bias = wmean(diff, lat)
    a = acc(ens_mean, truth, lat)
    var = np.nanvar(members, axis=0, ddof=1)
    spread = math.sqrt(wmean(var, lat))
    spread_error = (spread * math.sqrt((M + 1) / M) / rmse) if rmse > 0 else float("nan")
    return dict(rmse=rmse, bias=bias, acc=a, spread=spread, spread_error=spread_error)


# --------------------------------------------------------------------------- #
# Full per-model scoring pass
# --------------------------------------------------------------------------- #
def compute_all(model: str) -> Path:
    """Score every available case for ``model`` and write the results.

    Loops P.cases(); a case missing its cube or truth file is skipped with a warning.
    Per case: the week-mean anomaly ensemble is scored against the ERA5 anomaly truth
    (grid CRPS, rank histogram, ensemble stats) and the daily-series p_event is compared
    to the observed event label. Reference CRPS scores use climatology-zero
    (crps_clim0 = wmean(|truth|)) and persistence (crps_persist = wmean(|persist - truth|)).
    Rank histograms are pooled separately for events and controls.

    Writes scores_dir(model)/case_scores.csv (one row per scored case) and
    scores_dir(model)/pooled.npz (pooled rank histograms + per-kind p/o arrays); prints a
    compact summary. Returns the CSV path."""
    import xarray as xr

    P.ensure_dirs(model)
    # Score the FULL design (all 90 cases), never the env-filtered subset P.cases()
    # returns: a leaked P90_KIND/P90_CASES_SEL/P90_MAX_CASES would otherwise collapse the
    # balanced 45/45 contrast (base rate 0 or 1 -> BSS/ROC undefined). Cases without a
    # cube/truth pair are skipped below, so smoke runs still work.
    df = P.all_cases()
    rows = []
    rank_event = rank_control = None
    p_by_kind = {"event": [], "control": []}
    o_by_kind = {"event": [], "control": []}
    n_members = 0

    for _, row in df.iterrows():
        name = row["case"]
        cpath = P.cube_path(model, name)
        tpath = P.truth_path(name)
        if not cpath.exists():
            print(f"[compute_all] skip {name}: missing cube {cpath}")
            continue
        if not tpath.exists():
            print(f"[compute_all] skip {name}: missing truth {tpath}")
            continue

        cube = xr.open_dataset(cpath)
        truth_ds = xr.open_dataset(tpath)
        peak = pd.Timestamp(row["peak"])
        kind = row["kind"]

        mem_da = week_anom_members(cube, peak)                  # (member, lat, lon)
        lat = mem_da["lat"].values
        members = mem_da.values
        truth = truth_ds["t2m_anom"].values
        M = members.shape[0]
        n_members = M

        crps = wmean(crps_field(members, truth), lat)
        crps_clim0 = wmean(np.abs(truth), lat)
        crpss_clim0 = 1.0 - crps / crps_clim0 if crps_clim0 else float("nan")

        ppath = P.persist_path(name)
        if ppath.exists():
            persist = xr.open_dataset(ppath)["t2m_anom"].values
            crps_persist = wmean(np.abs(persist - truth), lat)
            crpss_persist = 1.0 - crps / crps_persist if crps_persist else float("nan")
        else:
            print(f"[compute_all] {name}: missing persistence {ppath}; crps_persist=nan")
            crps_persist = crpss_persist = float("nan")

        pe = p_event(cube, peak)
        oe = int(row["obs_event"])
        es = ens_stats(members, truth, lat)
        ens_mean = np.nanmean(members, axis=0)

        rh = rank_hist_field(members, truth)
        if kind == "event":
            rank_event = rh.copy() if rank_event is None else rank_event + rh
        else:
            rank_control = rh.copy() if rank_control is None else rank_control + rh
        p_by_kind[kind].append(pe)
        o_by_kind[kind].append(oe)

        rows.append(dict(
            case=name, kind=kind, month=int(row["month"]),
            obs_event=oe, peak_anom_K=float(row["peak_anom_K"]),
            crps=crps, crps_clim0=crps_clim0, crps_persist=crps_persist,
            crpss_clim0=crpss_clim0, crpss_persist=crpss_persist,
            p_event=pe,
            rmse=es["rmse"], bias=es["bias"], acc=es["acc"],
            spread=es["spread"], spread_error=es["spread_error"],
            ens_conus_anom=wmean(ens_mean, lat), obs_conus_anom=wmean(truth, lat),
        ))
        cube.close()
        truth_ds.close()

    scores = P.scores_dir(model)
    csv_path = scores / "case_scores.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    if rank_event is None:
        rank_event = np.zeros(n_members + 1, dtype=int)
    if rank_control is None:
        rank_control = np.zeros(n_members + 1, dtype=int)
    np.savez(
        scores / "pooled.npz",
        rank_hist_event=rank_event, rank_hist_control=rank_control,
        n_members=n_members,
        p_event=np.array(p_by_kind["event"], dtype=float),
        o_event=np.array(o_by_kind["event"], dtype=float),
        p_control=np.array(p_by_kind["control"], dtype=float),
        o_control=np.array(o_by_kind["control"], dtype=float),
    )

    if rows:
        dd = pd.DataFrame(rows)
        allp = dd["p_event"].to_numpy()
        allo = dd["obs_event"].to_numpy()
        _, _, auc = roc_curve(allp, allo)
        print(f"[compute_all] {model}: scored {len(rows)} cases; "
              f"mean CRPS={dd['crps'].mean():.4f}; Brier(p_event)={brier(allp, allo):.4f}; "
              f"AUC={auc:.4f}")
    else:
        print(f"[compute_all] {model}: no cases scored")
    return csv_path
