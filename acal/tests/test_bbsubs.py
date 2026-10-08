"""BB-SUBS estimate (acal/bbsubs.py): k derivation, published data, Tier 1 rows on synthetic
board tables (wide, CFS-style embedded variants, long), the EC46-absent no-op, Tier 2
validation and the context figure."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from acal import bbsubs as B

K3 = dict(mse=0.877, crps=0.936, brier=0.947)


def test_derive_k_matches_published():
    k = B.derive_k()
    for fam, v in K3.items():
        assert k[3][fam] == pytest.approx(v, abs=5e-4)
        assert k[3][fam + "_lo"] <= k[3][fam] <= k[3][fam + "_hi"]
    assert k[3]["mse_lo"] == pytest.approx(1 - 0.12 / 0.81, abs=1e-9)
    assert k[3]["mse_hi"] == pytest.approx(1 - 0.08 / 0.81, abs=1e-9)
    assert k[3]["crps"] == pytest.approx(np.sqrt(k[3]["mse"]))
    assert k[4]["mse"] == pytest.approx(0.88 / 0.95)
    assert all(k[4][f] > k[3][f] for f in K3)            # less skill gain at week 4


def test_published_data_tidy_and_consistent():
    pub, lb = B.load_published(), B.load_aiwq()
    for df in (pub, lb):
        assert df.source_url.str.contains("https://").all()
    assert len(pub[pub.chart == "rpss_aiwq"]) == 20 and len(lb) == 328
    # every non-BB value on Brightband's RPSS chart equals the official leaderboard
    djf = lb[lb.period == "DJF 2025"].set_index(["team", "model", "week"]).rpss
    chart = pub[(pub.chart == "rpss_aiwq") & (pub.kind == "leaderboard_verified")]
    for _, r in chart.iterrows():
        team, model = (x.strip() for x in r.aiwq_entry.split("/"))
        assert djf.loc[(team, model, r.week)] == pytest.approx(r.value)
    bb = lb[lb.model == "BB-SUBS"]
    assert set(bb.kind) == {"self_reported"} and set(bb.period) == {"DJF 2025"}


@pytest.mark.parametrize("name,fam", [
    ("brier_corr_emp_3K", "brier"), ("brier_3K", "brier"), ("rps", "brier"),
    ("crps", "crps"), ("rmse_ensmean", "crps"), ("se_ensmean", "mse"), ("mse", "mse"),
    ("msess", "ss_mse"), ("bss_3K", "ss_brier"), ("crpss", "ss_crps"),
    ("p_obs", "drop"), ("lift_corr_emp", "drop"), ("pit", "drop"), ("roc_auc", "drop"),
    ("csi_land", "drop"), ("dbrier_corr_emp_3K", "drop"), ("logratio_raw_emp", "drop"),
    ("cfs_mean", "drop"),
    ("brier_res_3K", "meta"), ("p_res_obs", "meta"), ("p_clim_obs", "meta"), ("obs", "meta"),
    ("o_3K", "meta"), ("rung", "meta"), ("n_members", "meta"), ("crps_clim", "meta"),
])
def test_classify(name, fam):
    assert B.classify(name) == fam


def _wide_board(n=6, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for src in ("aires", "cfs13", "ec46"):
        for var in (["raw_emp"] if src == "aires" else ["raw_emp", "corr_emp", "corr_gauss"]):
            for truth in ("era5", "hrrr"):
                for i in range(n):
                    rows.append(dict(
                        episode_id=f"e{i:02d}", source=src, variant=var, truth=truth,
                        n_members=101 if src == "ec46" else 16, obs=3.0 + i * 0.1, o_3K=1,
                        p_obs=rng.uniform(0.01, 0.3), brier_3K=rng.uniform(0.2, 0.9),
                        brier_res_3K=0.5, crps=rng.uniform(0.5, 2.0),
                        se_ensmean=rng.uniform(1, 9), msess=rng.uniform(-0.2, 0.4),
                        dbrier_3K=0.1, roc_auc=0.7))
    return pd.DataFrame(rows)


def test_estimate_rows_wide():
    board = _wide_board()
    k = B.derive_k()[3]
    est = B.estimate_rows(board, k)
    anc = board[(board.source == "ec46") & board.variant.str.startswith("corr")]
    assert len(est) == len(anc) == 2 * 2 * 6
    assert (est.source == "bbsubs").all() and est.estimate.all()
    assert (est.label == "BB-SUBS (est.)").all() and (est.tier == "tier1").all()
    assert set(est.anchor) == {"ec46/corr_emp", "ec46/corr_gauss"}
    a = anc.reset_index(drop=True)
    np.testing.assert_allclose(est.brier_3K, k["brier"] * a.brier_3K)
    np.testing.assert_allclose(est.crps, k["crps"] * a.crps)
    np.testing.assert_allclose(est.se_ensmean, k["mse"] * a.se_ensmean)
    np.testing.assert_allclose(est.msess, 1 - k["mse"] * (1 - a.msess))
    np.testing.assert_allclose(est.brier_3K_lo, k["brier_lo"] * a.brier_3K)
    np.testing.assert_allclose(est.crps_hi, k["crps_hi"] * a.crps)
    assert (est.brier_3K <= a.brier_3K).all() and (est.msess >= a.msess).all()
    # no tail probability / AUC / paired difference for an estimate; truth & AI+RES copied
    assert est[["p_obs", "roc_auc", "dbrier_3K"]].isna().all().all()
    for c in ("obs", "o_3K", "brier_res_3K", "episode_id", "truth", "variant", "n_members"):
        assert (est[c].values == a[c].values).all()
    # raw EC46 is never the anchor
    assert not (est.anchor == "ec46/raw_emp").any()


def test_estimate_rows_embedded_variants():
    """CFS-style: one row per case, the variant inside the column name."""
    board = pd.DataFrame(dict(episode_id=["e01", "e02", "e01"], source=["ec46", "ec46", "cfs13"],
                              obs=[3.0, -4.0, 3.0], brier_raw_emp_3K=[0.4, 0.6, 0.5],
                              brier_corr_emp_3K=[0.3, 0.5, 0.4], brier_res_3K=[0.2, 0.3, 0.2],
                              p_obs_corr_emp=[0.1, 0.2, 0.1]))
    est = B.estimate_rows(board, B.derive_k()[3])
    assert len(est) == 2 and (est.anchor == "ec46/corr").all()
    np.testing.assert_allclose(est.brier_corr_emp_3K, 0.947 * np.array([0.3, 0.5]), atol=1e-3)
    assert est.brier_raw_emp_3K.isna().all() and est.p_obs_corr_emp.isna().all()
    np.testing.assert_allclose(est.brier_res_3K, [0.2, 0.3])


def test_estimate_rows_long():
    board = pd.DataFrame(dict(
        source=["ec46"] * 4 + ["ec46"], variant=["corr_emp"] * 4 + ["raw_emp"],
        episode_id=["e01"] * 5, metric=["brier_3K", "crps", "p_obs", "msess", "brier_3K"],
        value=[0.5, 1.0, 0.2, 0.3, 0.6]))
    est = B.estimate_rows(board, B.derive_k()[3]).set_index("metric")
    assert set(est.index) == {"brier_3K", "crps", "msess"}           # p_obs and raw dropped
    assert est.loc["brier_3K", "value"] == pytest.approx(0.947 * 0.5, abs=1e-3)
    assert est.loc["crps", "value"] == pytest.approx(0.936, abs=1e-3)
    assert est.loc["msess", "value"] == pytest.approx(1 - 0.877 * 0.7, abs=1e-3)
    assert est.loc["crps", "value_lo"] < est.loc["crps", "value"] < est.loc["crps", "value_hi"]
    assert est.estimate.all()


def test_run_estimate_noop_until_ec46(tmp_path):
    ec, out = tmp_path / "ec46", tmp_path / "board" / "bbsubs_rows.csv"
    ec.mkdir()
    board_csv = tmp_path / "board_cases.csv"
    msgs = []
    # 1. no EC46 json -> no-op, nothing written
    assert B.run_estimate(board_csv, ec, out, tier2=False, log=msgs.append) is None
    assert "no EC46 case files" in msgs[-1] and not out.exists()
    # 2. EC46 json exists, board not written yet
    (ec / "e01.json").write_text("{}")
    assert B.run_estimate(board_csv, ec, out, tier2=False, log=msgs.append) is None
    assert "does not exist yet" in msgs[-1]
    # 3. board without ec46 rows
    b = _wide_board()
    b[b.source != "ec46"].to_csv(board_csv, index=False)
    assert B.run_estimate(board_csv, ec, out, tier2=False, log=msgs.append) is None
    assert "no bias-corrected ec46 rows" in msgs[-1] and not out.exists()
    # 4. full -> rows written
    b.to_csv(board_csv, index=False)
    est = B.run_estimate(board_csv, ec, out, tier2=False, log=msgs.append)
    back = pd.read_csv(out)
    assert len(back) == len(est) == 24 and back.estimate.all()
    # an existing estimate row in the board is never re-anchored
    pd.concat([b, back]).to_csv(board_csv, index=False)
    assert len(B.run_estimate(board_csv, ec, out, tier2=False, log=msgs.append)) == 24


def test_gauss_tail_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for rho2, z in ((0.19, 2.0), (0.5, 1.5), (0.05, 3.0)):
        m = rho2 * z + np.sqrt(rho2 * (1 - rho2)) * rng.standard_normal(400_000)
        mc = norm.sf((z - m) / np.sqrt(1 - rho2)).mean()
        assert B.gauss_tail(rho2, z) == pytest.approx(mc, rel=0.01)
        assert B.gauss_tail(rho2, -z) == B.gauss_tail(rho2, z)     # tail-direction free


def _sim_p(rho2, z, rng):
    m = rho2 * z + np.sqrt(rho2 * (1 - rho2)) * rng.standard_normal(z.size)
    return norm.sf((z - m) / np.sqrt(1 - rho2))


def test_tier2_validate_pass_and_fail():
    rng = np.random.default_rng(3)
    z = rng.uniform(1.2, 3.2, 42)
    ok = B.tier2_validate(_sim_p(0.25, z, rng), z, 0.25)
    assert ok["passed"], ok
    assert ok["log_tolerance"] == pytest.approx(np.log(ok["median_tier2_ratio"]))
    bad = B.tier2_validate(_sim_p(0.6, z, rng), z, 0.05)       # EC46 far sharper than modelled
    assert not bad["passed"]
    assert bad["log_level_miss"] > bad["log_tolerance"]


def test_ec46_rho2_from_hind(tmp_path):
    rng = np.random.default_rng(5)
    rows = []
    for i in range(42):
        b = rng.normal(0, 0.8)                                   # per-case model bias
        for y in range(20):
            obs = rng.normal(0, 1.5)
            sig = obs * 0.5 + rng.normal(0, 1.5 * np.sqrt(0.75))  # corr 0.5 with obs
            rows.append(dict(episode_id=f"e{i:02d}", year=2000 + y, peak="2022-01-15",
                             al_rf_mean=sig + b, al_era5=obs, diff=sig + b - obs))
    p = tmp_path / "hind.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    r2 = B.ec46_rho2(p, sig={"DJF": 1.5, "MAM": 1.5, "JJA": 1.5, "SON": 1.5})
    assert r2 == pytest.approx(0.25, abs=0.05)
    assert B.ec46_rho2(tmp_path / "missing.csv") is None


def test_slate_coverage_counts():
    cov = B.slate_coverage()
    n = cov.bb_class.value_counts()
    assert (n["held_out_2023_26"], n["reforecast_training_era"], n["apr_sep_no_backtest"]) == \
        (22, 15, 5)


def test_context_figure(tmp_path):
    out = B.context_figure(tmp_path / "ctx.png")
    assert out.exists() and out.stat().st_size > 50_000


def _hind(tmp_path, rho, seed=7):
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 1.5, 840)
    fc = rho * obs + rng.normal(0, 1.5 * np.sqrt(1 - rho ** 2), 840)
    h = pd.DataFrame(dict(episode_id=np.repeat([f"e{i:02d}" for i in range(42)], 20),
                          year=np.tile(np.arange(2000, 2020), 42), peak="2022-01-15",
                          al_rf_mean=fc, al_era5=obs, diff=fc - obs))
    p = tmp_path / "hind.csv"
    h.to_csv(p, index=False)
    return p


@pytest.mark.parametrize("rho2_true,expect", [(None, "passed"), (0.7, "Tier 2 failed validation")])
def test_run_tier2_end_to_end(tmp_path, rho2_true, expect):
    hind = _hind(tmp_path, rho=0.5)
    rho2 = B.ec46_rho2(hind)
    sig = B.season_sigma()["DJF"]
    rng = np.random.default_rng(11)
    z = rng.uniform(1.3, 3.0, 42)
    p = _sim_p(rho2 if rho2_true is None else rho2_true, z, rng)
    board = pd.DataFrame(dict(episode_id=[f"e{i:02d}" for i in range(42)], source="ec46",
                              variant="corr_gauss", truth="era5", peak="2024-01-20",
                              obs=-z * sig, p_obs=p, brier_3K=0.5))
    res = B.run_tier2(board, hind_csv=hind, out_dir=tmp_path, log=lambda m: None)
    js = json.loads((tmp_path / "bbsubs_tier2_validation.json").read_text())
    assert js["status"] == {"era5": expect}
    assert js["rho2_ec46"] == pytest.approx(rho2) and js["p_obs_column"] == "p_obs"
    t2 = tmp_path / "bbsubs_tier2_era5.csv"
    assert t2.exists() == (expect == "passed") == res["era5"]["passed"]
    if t2.exists():
        t = pd.read_csv(t2)
        assert (t.ratio_pub_pair > 1).all() and (t.p_obs_bb_est_pub_pair > t.p_obs_ec46).all()
        assert t.estimate.all()


def test_run_tier2_without_reforecasts(tmp_path):
    board = _wide_board()
    msgs = []
    assert B.run_tier2(board, hind_csv=tmp_path / "none.csv", out_dir=tmp_path,
                       log=msgs.append) is None
    assert "no EC46 reforecast table" in msgs[-1]


def test_tier2_sentence_reports_the_numbers():
    det = dict(passed=False, mean_actual=0.115, mean_pred=0.060, log_level_miss=0.649,
               log_tolerance=0.372, median_tier2_ratio=1.45,
               halves=[dict(half="low_z", inside=True, mean_pred=0.099, ci90=[0.08, 0.14]),
                       dict(half="high_z", inside=False, mean_pred=0.022, ci90=[0.082, 0.157])])
    s = B._tier2_sentence(dict(rho2_ec46=0.179, detail={"era5": det, "hrrr": det}))
    assert s.startswith("Tier 2 failed validation under every truth")
    assert "0.179" in s and "era5 failed" in s and "hrrr failed" in s
    assert "high-z half model 0.022 outside the 90% interval [0.082, 0.157]" in s
    assert "low-z" not in s                                  # only the halves that missed
    ok = dict(det, passed=True)
    assert B._tier2_sentence(dict(rho2_ec46=0.2, detail={"era5": ok})).startswith(
        "Tier 2 passed validation under every truth")
    assert "only" in B._tier2_sentence(dict(rho2_ec46=0.2, detail={"era5": ok, "hrrr": det}))
