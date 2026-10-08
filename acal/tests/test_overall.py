"""Tests for acal.overall - the data assembly behind the overall figures (not pixels).

A small synthetic board (6 cases, 3 truths) stands in for runs/acal/analysis/s2s/board;
the last test checks the real board, when present, against board_paired.csv."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acal import overall as OV

TRUTHS = ("era5", "hrrr", "hrrr_raw")
EIDS = [f"e{i:02d}_x" for i in range(1, 7)]
FAM = ["heat", "heat", "heat", "heat", "cold", "cold"]
METRICS = ("logratio", "dbrier_2K", "dbrier_3K", "dbrier_4K", "dcrps", "dsqerr",
           "dfield_rmse")
SRC = {  # source -> (window, n_native, variants, estimate)
    "cfs": ("13f", 16, ("raw_emp", "corr_emp"), False),
    "cfs13": ("13f", 16, ("raw_emp", "corr_emp", "e16_raw_emp"), False),
    "gefs": ("13f", 31, ("raw_emp", "corr_emp"), False),
    "geps": ("12f", 21, ("raw_emp", "corr_emp"), False),
    "ec46": ("12f", 101, ("raw_emp", "corr_emp"), False),
    "bbsubs": ("12f", np.nan, ("corr_emp",), True),
}


def _case_rows(rng, truth, source, window, n_native, variant, est):
    out = []
    for i, (eid, fam) in enumerate(zip(EIDS, FAM)):
        r = dict(truth=truth, source=source, label=OV.label(source), role="board",
                 estimate=est, window=window, case_idx=i, episode_id=eid, family=fam,
                 rung=2, obs=(2.5 + i) * (1 if fam == "heat" else -1)
                 + (0.1 if truth != "era5" else 0) + (0.05 if window == "12f" else 0),
                 n_native=n_native, p_clim_obs=0.0 if i == 0 else 0.02 + 0.01 * i,
                 variant=variant, p_obs=rng.uniform(0.01, 0.4),
                 brier_2K=rng.uniform(0, 1), brier_3K=rng.uniform(0, 1),
                 brier_4K=rng.uniform(0, 1), crps=rng.uniform(0.5, 2),
                 mean_sqerr=rng.uniform(1, 9), field_rmse=rng.uniform(2, 6),
                 field_pattern_r=rng.uniform(-0.2, 0.8))
        if est:
            r.update(p_obs=np.nan, field_rmse=np.nan, field_pattern_r=np.nan)
        out.append(r)
    return out


def synthetic_board(with_ec46: bool = False, with_bbsubs: bool = False, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    cases, paired = [], []
    for truth in TRUTHS:
        for w in ("13f", "12f"):
            cases += _case_rows(rng, truth, "aires", w, 32, "sn", False)
        for s, (w, n, variants, est) in SRC.items():
            if (s == "ec46" and not with_ec46) or (s == "bbsubs" and not with_bbsubs):
                continue
            for v in variants:
                cases += _case_rows(rng, truth, s, w, n, v, est)
                for sub in OV.SUBSETS:
                    for m in METRICS:
                        nan = est and m in ("logratio", "dfield_rmse")
                        mu = np.nan if nan else rng.normal(0.3, 0.3)
                        paired.append(dict(
                            truth=truth, source=s, label=OV.label(s), estimate=est,
                            variant=v, aires_variant="e16_sn" if v.startswith("e16") else "sn",
                            window=w, subset=sub, metric=m, n=0 if nan else 6, mean=mu,
                            median=mu, win=4, tie=0, loss=2,
                            ci_lo=mu - 0.2, ci_hi=mu + 0.2,
                            wilcoxon_p=np.nan if nan else rng.uniform(0, 0.2),
                            mean_aires=rng.uniform(0, 1), mean_model=rng.uniform(0, 1)))
    absent = [] if with_ec46 else ["ec46"]
    return dict(cases=pd.DataFrame(cases), paired=pd.DataFrame(paired),
                summary=dict(coverage=dict(absent=absent)))


def _write_land(root: Path, truth: str, source: str, res: float, mod: float):
    d = root / truth
    d.mkdir(parents=True, exist_ok=True)
    ent = lambda v: {"bss": {"2.0": v - 5, "-2.0": v - 5.1},      # mean: never plotted
                     "bss_med": {"2.0": v, "-2.0": v - 0.1},
                     "csi": {"2.0": v + 0.3, "-2.0": v + 0.2}}
    lab = OV.style(source)["label"]
    (d / f"maps_land_means_{source}.json").write_text(json.dumps(
        {"7d": {"AI+RES": ent(res), f"{lab} raw": ent(mod), f"{lab} corrected": ent(mod + 1)},
         "daily": {}}))


# --------------------------------------------------------------------------- #
def test_board_rows_order_absent_and_pickup():
    b = synthetic_board()
    assert OV.board_rows(b["paired"]) == [("cfs13", False), ("gefs", False), ("geps", False)]
    b = synthetic_board(with_ec46=True, with_bbsubs=True)
    assert OV.board_rows(b["paired"]) == [("cfs13", False), ("ec46", False), ("gefs", False),
                                          ("geps", False), ("bbsubs", True)]


def test_variants_headline_and_secondary():
    p = synthetic_board(with_ec46=True, with_bbsubs=True)["paired"]
    assert OV.variants_for(p, "cfs13", False) == ("raw_emp", "corr_emp")
    assert OV.variants_for(p, "bbsubs", True) == ("corr_emp", None)


def test_forest_table_is_board_paired_verbatim():
    b = synthetic_board(with_ec46=True, with_bbsubs=True)
    ft = OV.forest_table(b["paired"])
    key = ["truth", "source", "estimate", "variant", "subset", "metric"]
    m = ft.merge(b["paired"], on=key, suffixes=("", "_b"), validate="one_to_one")
    assert len(m) == len(ft)
    for c in ("mean", "ci_lo", "ci_hi", "win", "tie", "loss", "wilcoxon_p", "n"):
        np.testing.assert_array_equal(m[c].to_numpy(), m[c + "_b"].to_numpy())
    assert set(ft.truth) == set(OV.FIG_TRUTHS)
    assert set(ft.metric) == set(OV.FOREST_METRICS)
    assert not (ft.source == "cfs").any()                      # reference never a row
    assert set(ft.variant) == {"raw_emp", "corr_emp"}          # no e16 on the forest
    hl = ft[ft.role == "headline"]
    assert set(hl.variant[hl.source != "bbsubs"]) == {"raw_emp"}
    assert set(hl.variant[hl.source == "bbsubs"]) == {"corr_emp"}


def test_better_t_orientation():
    assert OV.better_t(2.0, 1.0, "ratio", -1) == pytest.approx(1.0)    # model error 2x
    assert OV.better_t(0.5, 1.0, "ratio", -1) == pytest.approx(-1.0)
    assert OV.better_t(0.05, 0.1, "ratio", +1) == pytest.approx(1.0)   # model P(obs) half
    assert OV.better_t(0.25, 0.5, "diff", +1) == pytest.approx(1.0)
    assert OV.better_t(0.6, 0.5, "diff", +1) == pytest.approx(-0.4)
    assert OV.better_t(0.0, 0.1, "ratio", +1) == 1.0                   # P = 0 saturates
    assert np.isnan(OV.better_t(np.nan, 1.0, "ratio", -1))


def test_scoreboard_aggregates_and_window_pairing(tmp_path):
    b = synthetic_board(with_bbsubs=True)
    for t in ("era5", "hrrr"):
        _write_land(tmp_path, t, "cfs13", 0.2, 0.1)
        _write_land(tmp_path, t, "gefs", 0.2, 0.3)
        _write_land(tmp_path, t, "geps", 0.25, 0.05)
    sb = OV.scoreboard_table(b["cases"], b["paired"], analysis=tmp_path)
    c = b["cases"]
    g = c[(c.truth == "era5") & (c.source == "gefs") & (c.variant == "raw_emp")]
    v = sb[(sb.truth == "era5") & (sb.source == "gefs")].set_index("column")
    assert v.loc["p_obs", "value"] == np.median(g.p_obs)
    ok = g.p_clim_obs > 0
    assert ok.sum() == 5
    assert v.loc["lift", "value"] == np.median(g.p_obs[ok] / g.p_clim_obs[ok])
    assert v.loc["rmse_mean", "value"] == pytest.approx(np.sqrt(g.mean_sqerr.mean()))
    assert v.loc["crps", "value"] == pytest.approx(g.crps.mean())
    assert v.loc["pattern_r", "value"] == pytest.approx(g.field_pattern_r.mean())
    assert v.loc["bss_p2", "value"] == 0.3 and v.loc["csi_m2", "value"] == pytest.approx(0.5)
    # GEPS (12f) is coloured against AI+RES on 12 frames, GEFS against 13 frames
    a12 = c[(c.truth == "era5") & (c.source == "aires") & (c.window == "12f")]
    a13 = c[(c.truth == "era5") & (c.source == "aires") & (c.window == "13f")]
    gp = sb[(sb.truth == "era5") & (sb.source == "geps")].set_index("column")
    assert gp.loc["crps", "aires_value"] == pytest.approx(a12.crps.mean())
    assert v.loc["crps", "aires_value"] == pytest.approx(a13.crps.mean())
    assert gp.loc["bss_p2", "aires_value"] == 0.25 and v.loc["bss_p2", "aires_value"] == 0.2
    # colour sign follows the AI+RES-better orientation
    exp = np.sign(v.loc["crps", "value"] - v.loc["crps", "aires_value"])
    assert np.sign(v.loc["crps", "t"]) == exp
    # paired p comes from the board, headline variant, all cases
    pp = b["paired"]
    want = pp[(pp.truth == "era5") & (pp.source == "gefs") & (pp.variant == "raw_emp")
              & (pp.subset == "all") & (pp.metric == "dcrps")].wilcoxon_p.iloc[0]
    assert v.loc["crps", "wilcoxon_p"] == want
    # BB-SUBS: corr_emp, no P(obs), no maps; AI+RES rows for both windows
    bb = sb[(sb.truth == "era5") & (sb.source == "bbsubs")].set_index("column")
    assert set(bb.variant) == {"corr_emp"}
    assert np.isnan(bb.loc["p_obs", "value"]) and np.isnan(bb.loc["bss_p2", "value"])
    assert np.isfinite(bb.loc["crps", "value"])
    assert set(sb.window[sb.source == "aires"]) == {"13f", "12f"}


def test_scoreboard_partial_source_vs_aires_on_same_cases(tmp_path):
    """A source covering only some cases is coloured against AI+RES on THOSE cases."""
    b = synthetic_board(with_ec46=True)
    c = b["cases"]
    ids = sorted(c.episode_id.unique())
    keep = set(ids[: len(ids) // 2])
    c = c[(c.source != "ec46") | c.episode_id.isin(keep)]
    for t in ("era5", "hrrr"):
        _write_land(tmp_path, t, "ec46", 0.4, 0.1)
    sb = OV.scoreboard_table(c, b["paired"], analysis=tmp_path)
    ec = sb[(sb.truth == "era5") & (sb.source == "ec46")].set_index("column")
    a12 = c[(c.truth == "era5") & (c.source == "aires") & (c.window == "12f")
            & (c.variant == "sn")]
    assert ec.loc["crps", "aires_value"] == pytest.approx(a12[a12.episode_id.isin(keep)].crps.mean())
    assert ec.loc["crps", "aires_value"] != pytest.approx(a12.crps.mean())
    assert ec.loc["bss_p2", "aires_value"] == 0.4          # AI+RES from ec46's own maps json
    gp = sb[(sb.truth == "era5") & (sb.source == "geps")].set_index("column")
    assert gp.loc["crps", "aires_value"] == pytest.approx(a12.crps.mean())


def test_obs_and_hrrr_tables():
    b = synthetic_board(with_ec46=True)
    ob = OV.obs_table(b["cases"])
    assert list(ob.episode_id) == EIDS
    assert {"era5", "hrrr", "hrrr_raw"} <= set(ob.columns)
    np.testing.assert_allclose(ob.hrrr - ob.era5, 0.1)
    ht = OV.hrrr_table(b["paired"])
    assert set(ht.truth) == set(TRUTHS) and set(ht.subset) == {"all"}
    assert set(ht.metric) == set(OV.HRRR_METRICS)
    assert set(ht.source) == {"cfs13", "ec46", "gefs", "geps"}


def test_numbers_table_holds_every_plotted_number(tmp_path):
    b = synthetic_board(with_bbsubs=True)
    _write_land(tmp_path, "era5", "cfs13", 0.2, 0.1)
    nt = OV.numbers_table(b, analysis=tmp_path)
    assert set(nt.figure) == {"forest", "scoreboard", "hrrr_vs_era5"}
    assert np.isfinite(nt.value).all()                         # only drawn numbers
    f = nt[nt.figure == "forest"]
    ft = OV.forest_table(b["paired"])
    assert len(f) == int(np.isfinite(ft["mean"]).sum())
    # BB-SUBS has no log ratio on the board, so none in the CSV
    assert not ((f.source == "bbsubs") & (f.metric == "logratio")).any()
    assert ((f.source == "bbsubs") & (f.metric == "dcrps")).any()
    a = nt[(nt.figure == "hrrr_vs_era5") & (nt.panel == "a")]
    assert len(a) == len(EIDS) * 3


def test_figures_render_with_ec46_and_bbsubs(tmp_path):
    b = synthetic_board(with_ec46=True, with_bbsubs=True)
    assert OV.fig_forest(b, tmp_path / "f.png").stat().st_size > 10_000
    assert OV.fig_scoreboard(b, tmp_path / "s.png", analysis=tmp_path).stat().st_size > 10_000
    assert OV.fig_hrrr(b, tmp_path / "h.png").stat().st_size > 10_000


def test_cli_without_board(tmp_path, capsys):
    assert OV.main(["--board-dir", str(tmp_path)]) == 1
    assert "no board" in capsys.readouterr().out


@pytest.mark.skipif(not (OV.BOARD_DIR / "board_paired.csv").exists(), reason="no real board")
def test_real_board_numbers_match_board_paired():
    b = OV.load_board()
    nt = OV.numbers_table(b)
    p = b["paired"]
    key = ["truth", "source", "estimate", "variant", "subset", "metric"]
    for fig in ("forest", "hrrr_vs_era5"):
        f = nt[(nt.figure == fig) & (nt.metric != "obs_A_L")]
        m = f.merge(p, on=key, suffixes=("", "_b"), validate="one_to_one")
        assert len(m) == len(f) > 0
        np.testing.assert_array_equal(m.value.to_numpy(), m["mean"].to_numpy())
        for c in ("ci_lo", "ci_hi", "wilcoxon_p", "win", "tie", "loss"):
            np.testing.assert_array_equal(m[c].to_numpy(dtype=float),
                                          m[c + "_b"].to_numpy(dtype=float))
    # scoreboard AI+RES P(obs) median == the board summary's
    s = b["summary"]["cases"]["era5"]["aires"]["13f"]["sn"]["all"]["p_obs"]["median"]
    sb = nt[(nt.figure == "scoreboard") & (nt.source == "aires") & (nt.truth == "era5")
            & (nt.window == "13f") & (nt.metric == "p_obs")]
    assert sb.value.iloc[0] == pytest.approx(s, abs=1e-12)
