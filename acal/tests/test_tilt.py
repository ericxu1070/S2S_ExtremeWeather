"""acal.tilt: weight diagnostics, and the published outputs agree with the board and each other."""
import json

import numpy as np
import pandas as pd
import pytest

from acal import board as BD
from acal import tilt as T

# The CSVs are written with 9 significant digits; the in-memory board parity (exact, 0) is
# asserted by `tilt.table` itself and recorded as board_parity_max_abs in the json.
CSV_RTOL, CSV_ATOL = 1e-8, 1e-9


def test_weight_stats_uniform_and_one_hot():
    al = np.linspace(-1.0, 2.0, 32)
    u = T.weight_stats(al, np.ones(32), +1.0)
    assert u["ess"] == pytest.approx(32.0) and u["ess_frac"] == pytest.approx(1.0)
    assert u["entropy_frac"] == pytest.approx(1.0)
    assert u["n_tilted_side"] == u["n_tilted_side_uniform"] == 16
    w = np.zeros(32)
    w[3] = 5.0
    o = T.weight_stats(al, w, -1.0)
    assert o["ess"] == pytest.approx(1.0) and o["entropy"] == pytest.approx(0.0)
    assert o["w_max"] == pytest.approx(1.0)
    # cold sign: walkers colder than al[3] are on the tilted side
    assert o["n_tilted_side"] == 3


def test_uniform_scores_equal_equal_weight_crps():
    rng = np.random.default_rng(0)
    al = rng.normal(1.0, 1.0, 32)
    m = BD.ensemble_metrics(al, 2.0, 1.0, "emp", 32)
    assert m["crps"] == pytest.approx(BD.crps_ensemble(al, 2.0, np.ones(32)), abs=1e-12)
    assert m["p_obs"] == pytest.approx(np.mean(al >= 2.0))


needs = pytest.mark.skipif(not (T.CASES_OUT.exists() and T.PAIRED_OUT.exists()
                                and T.JSON_OUT.exists()), reason="tilt outputs not built")


@needs
def test_sn_rows_equal_board_aires_rows():
    cs = pd.read_csv(T.CASES_OUT, float_precision="round_trip")
    b = pd.read_csv(BD.CASES_CSV, float_precision="round_trip")
    b = b[(b.source == "aires") & (b.variant == "sn")]
    m = cs.merge(b, on=["truth", "window", "episode_id"])
    assert len(m) == len(cs) == 3 * 2 * 42
    for k in ("p_obs", "logp_floor", "crps", "mean_err_signed", "brier_3K"):
        assert np.allclose(m[f"{k}_sn"], m[k], rtol=CSV_RTOL, atol=CSV_ATOL), k
    assert np.allclose(m.ess, m.n_eff, rtol=CSV_RTOL, atol=CSV_ATOL)


@needs
def test_sn_pairs_equal_board_paired_and_json_matches_csv():
    pt = pd.read_csv(T.PAIRED_OUT, float_precision="round_trip")
    bp = pd.read_csv(BD.PAIRED_CSV, float_precision="round_trip")
    bp = bp[(bp.source.isin(T.MODELS)) & (bp.variant == "raw_emp") & (~bp.estimate)]
    ours = pt[(pt.aires_variant == "sn") & (pt.variant == "raw_emp")]
    m = ours.merge(bp, on=["truth", "source", "variant", "subset", "metric"],
                   suffixes=("", "_b"))
    assert len(m) == len(ours) > 0
    for k in ("mean", "ci_lo", "ci_hi", "win", "loss"):
        assert np.allclose(m[k], m[f"{k}_b"], rtol=CSV_RTOL, atol=CSV_ATOL), k
    js = json.loads(T.JSON_OUT.read_text())
    assert js["board_parity_max_abs"] <= T.PARITY_TOL
    for t, blk in js["verdicts"].items():
        for src, e in blk.items():
            for v in T.VARIANTS:
                r = pt[(pt.truth == t) & (pt.source == src) & (pt.variant == "raw_emp")
                       & (pt.subset == "all") & (pt.metric == "logratio")
                       & (pt.aires_variant == v)].iloc[0]
                assert e["logratio"][v]["mean"] == pytest.approx(r["mean"], rel=CSV_RTOL,
                                                                 abs=CSV_ATOL)


@needs
def test_untilted_lead6_matches_theta_and_counts():
    cs = pd.read_csv(T.CASES_OUT, float_precision="round_trip")
    assert (cs.n_untilted == 192).all() and (cs.n_walkers == 32).all()
    eid = cs.episode_id.iloc[0]
    tab = pd.read_csv(T.LEAD6_DIR / f"{eid}.csv", float_precision="round_trip")
    rec = json.loads((T.AN.case_dir(eid) / "res_result.json").read_text())
    got = tab.groupby("walker").al25.mean().to_numpy()
    assert np.max(np.abs(got - np.asarray(rec["theta"][1]["conus"]))) < T.CHECK_TOL
    e = cs[(cs.truth == "era5") & (cs.window == "13f") & (cs.episode_id == eid)].iloc[0]
    assert e.fc_mean_untilted == pytest.approx(tab.al13.mean(), abs=1e-9)


@needs
def test_bbsubs_estimate_pairs_match_board_and_blind_json():
    """The BB-SUBS estimate rows: 'sn' equals the board's own pairing, there is no log ratio
    (the estimate has no P(obs)), and the json k=week-3 blind dCRPS equals the csv row."""
    pt = pd.read_csv(T.PAIRED_OUT, float_precision="round_trip")
    e = pt[pt.source == "bbsubs"]
    assert set(e.variant) == {"corr_emp"} and set(e.window) == {"12f"}
    assert "logratio" not in set(e.metric)
    bp = pd.read_csv(BD.PAIRED_CSV, float_precision="round_trip")
    bp = bp[(bp.source == "bbsubs") & (bp.variant == "corr_emp")]
    ours = e[e.aires_variant == "sn"]
    m = ours.merge(bp, on=["truth", "source", "variant", "subset", "metric"], suffixes=("", "_b"))
    assert len(m) == len(ours) > 0
    for k in ("mean", "ci_lo", "ci_hi", "win", "loss"):
        assert np.allclose(m[k], m[f"{k}_b"], rtol=CSV_RTOL, atol=CSV_ATOL), k
    js = json.loads(T.JSON_OUT.read_text())["estimate"]
    assert js["k_crps"]["wk3_lo"] < js["k_crps"]["wk3"] < js["k_crps"]["wk3_hi"] < js["k_crps"]["wk4"]
    for t in ("era5", "hrrr"):
        r = e[(e.truth == t) & (e.aires_variant == "untilted") & (e.subset == "all")
              & (e.metric == "dcrps")].iloc[0]
        got = js[t]["dcrps_wk3_untilted"]
        assert got["n"] == js[t]["n"] == 42 and 0 < js[t]["n_lead21"] < 42
        for k in ("mean", "ci_lo", "ci_hi", "win", "loss"):
            assert got[k] == pytest.approx(r[k], rel=CSV_RTOL, abs=CSV_ATOL), (t, k)
