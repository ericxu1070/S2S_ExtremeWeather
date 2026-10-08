"""acal.board: weighted CRPS (vs the unweighted formula and a brute-force integral), the
exact 16-member-subset expectations, the BB-SUBS hook, and parity with the published CFS
paired numbers."""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acal import board as B  # noqa: E402
from acal import cfsbase as CB  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_trapz = getattr(np, "trapezoid", None) or np.trapz
PUB_SUMMARY = ROOT / "runs" / "acal" / "analysis" / "cfs_summary.json"


# --------------------------------------------------------------------------- #
# CRPS
# --------------------------------------------------------------------------- #
def crps_integral(x, y, w=None, n_grid=400_001):
    """Brute force: integrate (F(z) - 1[z >= y])^2 on a fine grid (trapezoid)."""
    x = np.asarray(x, float)
    w = np.full(x.size, 1 / x.size) if w is None else np.asarray(w, float) / np.sum(w)
    lo, hi = min(x.min(), y) - 1.0, max(x.max(), y) + 1.0
    z = np.linspace(lo, hi, n_grid)
    order = np.argsort(x)
    F = np.concatenate([[0.0], np.cumsum(w[order])])[np.searchsorted(x[order], z, "right")]
    H = (z >= y).astype(float)
    return float(_trapz((F - H) ** 2, z))


def crps_exact_steps(x, y, w=None):
    """Exact integral of the step-function integrand, piece by piece."""
    x = np.asarray(x, float)
    w = np.full(x.size, 1 / x.size) if w is None else np.asarray(w, float) / np.sum(w)
    pts = np.sort(np.concatenate([x, [y]]))
    tot = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        m = 0.5 * (a + b)
        F = np.sum(w[x <= m])
        tot += (F - float(m >= y)) ** 2 * (b - a)
    return tot


def test_crps_unweighted_formula():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=11), 0.7
    want = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert B.crps_ensemble(x, y) == pytest.approx(want, abs=1e-14)
    assert B.crps_ensemble(x, y, np.full(11, 3.0)) == pytest.approx(want, abs=1e-14)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_crps_equals_brute_force_integral(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 2, size=32)
    w = np.exp(rng.normal(0, 1.5, size=32))          # importance-like weights
    for y in (x.min() - 0.5, np.median(x), x.max() + 1.3, x[3]):
        got = B.crps_ensemble(x, y, w)
        assert got == pytest.approx(crps_exact_steps(x, y, w), abs=1e-12)
        assert got == pytest.approx(crps_integral(x, y, w), abs=2e-4)


def test_weighted_crps_integer_weights_equal_replicated_ensemble():
    rng = np.random.default_rng(3)
    x, k = rng.normal(size=6), np.array([1, 3, 2, 5, 1, 4])
    y = 0.2
    assert B.crps_ensemble(x, y, k) == pytest.approx(B.crps_ensemble(np.repeat(x, k), y),
                                                     abs=1e-13)


def test_fair_crps_equal_weights():
    rng = np.random.default_rng(4)
    x, y, m = rng.normal(size=9), -0.4, 9
    want = (np.mean(np.abs(x - y))
            - np.sum(np.abs(x[:, None] - x[None, :])) / (2 * m * (m - 1)))
    assert B.crps_ensemble(x, y, fair=True) == pytest.approx(want, abs=1e-14)


def test_crps_gauss_brute_force():
    from scipy.integrate import quad
    from scipy.stats import norm
    for mu, sd, y in ((0.0, 1.0, 0.3), (1.5, 0.4, -1.0), (-2.0, 3.0, 4.0)):
        lo, hi = mu - 40 * sd - abs(y), mu + 40 * sd + abs(y)
        brute = (quad(lambda z: norm.cdf(z, mu, sd) ** 2, lo, y, limit=200)[0]
                 + quad(lambda z: norm.sf(z, mu, sd) ** 2, y, hi, limit=200)[0])
        assert B.crps_gauss(mu, sd, y) == pytest.approx(brute, abs=1e-8)
    assert B.crps_gauss(1.0, 0.0, 3.0) == pytest.approx(2.0)


def test_weighted_sd_equal_weights_is_ddof1():
    x = np.array([1.0, 2.5, -0.3, 4.0])
    assert B.weighted_sd(x, np.full(4, 0.25)) == pytest.approx(x.std(ddof=1), abs=1e-14)


# --------------------------------------------------------------------------- #
# Expected scores over random subsets
# --------------------------------------------------------------------------- #
def test_e16_emp_exact_vs_enumeration():
    rng = np.random.default_rng(5)
    n, m, s = 9, 4, 1.0
    a = rng.normal(2.0, 1.5, size=n)
    obs = float(np.sort(a)[6])
    got = B.e16_emp(a, obs, s, m)
    subs = [B.ensemble_metrics(a[list(c)], obs, s, "emp", m)
            for c in itertools.combinations(range(n), m)]
    for k in ("p_obs", "p_floor", "logp_floor", "brier_2K", "brier_3K", "brier_4K", "crps",
              "crps_fair", "mean_sqerr", "pit"):
        assert got[k] == pytest.approx(np.mean([d[k] for d in subs]), abs=1e-12), k
    assert got["fc_mean"] == pytest.approx(a.mean())


def test_e16_emp_full_size_is_native():
    rng = np.random.default_rng(6)
    a = rng.normal(size=16)
    nat = B.ensemble_metrics(a, 0.5, -1.0, "emp", 16)
    e = B.e16_emp(a, 0.5, -1.0, 16)
    for k in ("p_obs", "p_floor", "logp_floor", "brier_3K", "crps", "mean_sqerr"):
        assert e[k] == pytest.approx(nat[k], abs=1e-12), k


def test_e16_gauss_and_weighted_full_set_reduce_to_native():
    rng = np.random.default_rng(7)
    a, w = rng.normal(1.0, 2.0, size=12), np.exp(rng.normal(size=12))
    idx = np.arange(12)[None, :]
    g, nat = B.e16_gauss(a, 2.1, 1.0, idx), B.ensemble_metrics(a, 2.1, 1.0, "gauss", 12)
    for k in ("p_obs", "logp_floor", "brier_2K", "crps", "mean_sqerr", "spread"):
        assert g[k] == pytest.approx(nat[k], abs=1e-12), k
    ew = B.e16_weighted(a, w, 2.1, 1.0, idx)
    wn = w / w.sum()
    assert ew["p_obs"] == pytest.approx(np.sum(wn * (a >= 2.1)), abs=1e-14)
    assert ew["crps"] == pytest.approx(B.crps_ensemble(a, 2.1, w), abs=1e-12)
    assert ew["crps_fair"] == pytest.approx(B.crps_ensemble(a, 2.1, w, fair=True), abs=1e-12)
    assert ew["spread"] == pytest.approx(B.weighted_sd(a, wn), abs=1e-12)


def test_e16_all_floored_is_exact_tie():
    """Every 16-subset floored on both sides must give logratio exactly 0 (a tie), not a
    -4e-16 float 'loss' from averaging 1000 copies of log(1/17)."""
    a = np.linspace(-3.0, -1.0, 32)                  # no member reaches obs = +2 K
    w = np.exp(np.random.default_rng(3).normal(size=32))
    idx = B.subset_idx(32, 16, 1000, (1, 32, 0))
    fl = np.log(1.0 / 17.0)
    assert B.e16_weighted(a, w, 2.0, 1.0, idx)["logp_floor"] == fl
    assert B.e16_gauss(a, 9.0, 1.0, idx[:400])["logp_floor"] == fl
    assert B.e16_emp(a, 2.0, 1.0)["logp_floor"] == fl
    df = _toy_board()
    df = df[df.variant != "corr_emp"].copy()
    df["variant"] = np.where(df.source == "aires", "e16_sn", "e16_raw_emp")
    df.loc[df.source == "aires", "logp_floor"] = fl - 4.4e-16
    df.loc[df.source == "gefs", "logp_floor"] = fl
    pc = B.paired_cases(df)
    assert (pc.logratio == 0.0).all()
    r = B.paired_table(pc).query("subset == 'all' and metric == 'logratio'").iloc[0]
    assert (r.win, r.tie, r.loss) == (0, 2, 0)


def test_subset_idx_seeded_and_distinct():
    i1, i2 = B.subset_idx(32, 16, 50, (3, 32, 0)), B.subset_idx(32, 16, 50, (3, 32, 0))
    assert np.array_equal(i1, i2)
    assert all(len(set(r)) == 16 for r in i1)
    assert not np.array_equal(i1, B.subset_idx(32, 16, 50, (4, 32, 0)))


def test_variant_flags():
    assert B.variant_flags("raw_emp")["headline"]
    assert B.variant_flags("sn")["kind"] == "weighted"
    f = B.variant_flags("e16_corr_gauss")
    assert (f["ens"], f["kind"], f["bias_corrected"], f["headline"]) == \
        ("e16", "gauss", True, False)
    assert B.variant_flags("s16_emp")["ens"] == "fixed16"
    assert B.variant_flags("sub_emp")["ens"] == "last4"


# --------------------------------------------------------------------------- #
# BB-SUBS hook and pairing guards (synthetic)
# --------------------------------------------------------------------------- #
def _toy_board():
    rows = []
    for src, var, p, crps in (("aires", "sn", 0.5, 1.0), ("gefs", "raw_emp", 0.25, 2.0),
                              ("gefs", "corr_emp", 0.25, 1.5)):
        for i, (eid, fam) in enumerate((("e01", "heat"), ("e02", "cold"))):
            rows.append(dict(truth="era5", source=src, label=src, role="x", estimate=False,
                             window="13f", variant=var, case_idx=i, episode_id=eid,
                             family=fam, rung=2, obs=2.5 if fam == "heat" else -2.5,
                             p_floor=p, logp_floor=np.log(p), brier_2K=(1 - p) ** 2,
                             brier_3K=0.0, brier_4K=0.0, crps=crps, mean_sqerr=crps,
                             field_rmse=np.nan, field_bias=np.nan))
    return pd.DataFrame(rows)


def test_merge_estimates_flags_and_staleness(tmp_path):
    df = _toy_board()
    est = df[df.source == "gefs"].assign(source="bbsubs", field_rmse=9.0, tier="tier1",
                                         anchor="ec46/corr_emp", case_idx=np.nan,
                                         p_floor=np.nan, logp_floor=np.nan)
    p = tmp_path / "bbsubs_rows.csv"
    est.to_csv(p, index=False)
    out = B.merge_estimates(df, p)
    e = out[out.estimate.astype(bool)]
    assert len(e) == len(est) and set(e.role) == {"estimate"} and e.field_rmse.isna().all()
    assert set(e.ens) == {"native"} and e.case_idx.notna().all()
    pc = B.paired_cases(out)
    pe = pc[pc.estimate.astype(bool)]
    assert len(pe) == len(est) and pe.logratio.isna().all() and np.isfinite(pe.dcrps).all()
    # the paired table carries an estimate only on the metrics it estimates (no P(obs), no maps)
    pt = B.paired_table(pc)
    ept, mpt = pt[pt.estimate.astype(bool)], pt[~pt.estimate.astype(bool)]
    assert not ({"logratio", "dfield_rmse"} & set(ept.metric))
    assert {"dcrps", "dsqerr", "dbrier_2K"} <= set(ept.metric) and (ept.n > 0).all()
    assert {"logratio", "dfield_rmse"} <= set(mpt.metric)      # measured rows keep every metric
    # integer columns stay integer although the estimate rows leave them blank
    df2 = df.assign(n_native=16)
    out2 = B.merge_estimates(df2, p)
    assert pd.api.types.is_integer_dtype(out2.n_native)
    assert out2.n_native[~out2.estimate.astype(bool)].eq(16).all()
    assert len(B.merge_estimates(df, p, not_before=time.time() + 10)) == len(df)
    assert len(B.merge_estimates(df, tmp_path / "absent.csv")) == len(df)


def test_paired_cases_signs_and_obs_guard():
    df = _toy_board()
    pc = B.paired_cases(df)
    raw = pc[pc.variant == "raw_emp"]
    assert np.allclose(raw.logratio, np.log(0.5 / 0.25))       # AI+RES higher P: positive
    assert np.allclose(raw.dcrps, 1.0)                         # model CRPS worse: positive
    pt = B.paired_table(pc)
    r = pt[(pt.variant == "raw_emp") & (pt.subset == "all") & (pt.metric == "dcrps")].iloc[0]
    assert (r.n, r.win, r.mean_aires, r.mean_model) == (2, 2, 1.0, 2.0)
    bad = df.copy()
    bad.loc[(bad.source == "gefs") & (bad.case_idx == 0), "obs"] += 0.01
    with pytest.raises(SystemExit):
        B.paired_cases(bad)


# --------------------------------------------------------------------------- #
# Parity with the published CFS paired numbers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cfs_board():
    if not PUB_SUMMARY.exists() or not CB.json_path("e01_h2_20210119").exists():
        pytest.skip("published CFS files not on this machine")
    df = B.collect_truth("era5", sources=("cfs",), fields=False)
    return df, B.paired_table(B.paired_cases(df))


def test_cfs_parity_headline(cfs_board):
    _, pt = cfs_board
    r = pt[(pt.source == "cfs") & (pt.variant == "raw_emp") & (pt.subset == "all")]
    lr = r[r.metric == "logratio"].iloc[0]
    assert round(lr["mean"], 2) == 0.49
    assert (round(lr.ci_lo, 2), round(lr.ci_hi, 2)) == (0.19, 0.79)
    assert round(lr.wilcoxon_p, 3) == 0.026
    assert (lr.win, lr.tie, lr.loss) == (24, 0, 18)
    assert round(r[r.metric == "dbrier_2K"].iloc[0]["mean"], 3) == 0.270


def test_cfs_parity_every_variant_exact(cfs_board):
    _, pt = cfs_board
    pub = json.loads(PUB_SUMMARY.read_text())["paired"]["variants"]
    for v, sub, (m, pm) in itertools.product(
            CB.VARIANTS, ("all", "heat", "cold"),
            (("logratio", "logratio"), ("dbrier_2K", "brier_2K"),
             ("dbrier_3K", "brier_3K"), ("dbrier_4K", "brier_4K"))):
        r = pt[(pt.source == "cfs") & (pt.variant == v) & (pt.subset == sub)
               & (pt.metric == m)].iloc[0]
        p = pub[v][sub][pm]
        for k in ("n", "mean", "median", "win", "tie", "loss", "ci_lo", "ci_hi",
                  "wilcoxon_p"):
            assert r[k] == pytest.approx(p[k], abs=1e-12), (v, sub, m, k)
        if pm != "logratio":
            assert r.mean_model == pytest.approx(p["mean_brier_cfs"], abs=1e-12)
            assert r.mean_aires == pytest.approx(p["mean_brier_res"], abs=1e-12)


def test_cfs_board_rows_are_complete(cfs_board):
    df, _ = cfs_board
    res = df[(df.source == "aires") & (df.variant == "sn") & (df.window == "13f")]
    assert len(res) == 42 and res.p_obs.between(0, 1).all() and (res.crps > 0).all()
    cfs = df[(df.source == "cfs") & (df.variant == "raw_emp")]
    assert len(cfs) == 42 and (cfs.n_members == 16).all()
    # the board's model P(obs) is the published scorecard's
    sc = pd.read_csv(ROOT / "runs" / "acal" / "analysis" / "cfs_scorecard.csv")
    got = cfs.set_index("episode_id").p_obs.reindex(sc.episode_id).to_numpy()
    assert np.allclose(got, sc.p_obs_raw_emp.to_numpy(), atol=1e-6)
    # e16 of a 16-member source is the source itself
    e16 = df[(df.source == "cfs") & (df.variant == "e16_raw_emp")]
    assert np.allclose(e16.crps.to_numpy(), cfs.crps.to_numpy(), atol=1e-12)


def test_board_file_reproduces_published_cfs():
    """The board_paired.csv on disk (written via the CSV round trip) carries the published
    CFS paired numbers exactly, and every board source is like-for-like with AI+RES."""
    if not (B.PAIRED_CSV.exists() and PUB_SUMMARY.exists()):
        pytest.skip("board not built on this machine")
    pt = pd.read_csv(B.PAIRED_CSV, float_precision="round_trip")
    pub = json.loads(PUB_SUMMARY.read_text())["paired"]["variants"]
    for m, pm in (("logratio", "logratio"), ("dbrier_2K", "brier_2K")):
        r = pt[(pt.truth == "era5") & (pt.source == "cfs") & (pt.variant == "raw_emp")
               & (pt.subset == "all") & (pt.metric == m)].iloc[0]
        for k in ("mean", "ci_lo", "ci_hi", "wilcoxon_p", "win", "loss"):
            assert r[k] == pytest.approx(pub["raw_emp"]["all"][pm][k], abs=1e-12)
    win = pt[pt.variant == "raw_emp"].groupby("source").window.first()
    assert all(win.get(s, w) == w for s, w in (("cfs13", "13f"), ("gefs", "13f"),
                                              ("geps", "12f"), ("ec46", "12f")))
