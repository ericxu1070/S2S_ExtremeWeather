"""Per-source scoring and figure plumbing: board style, data detection, figure contexts,
and the AI+RES 12-frame probability used against daily-mean sources."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acal import reach as RE
from acal import roc as R
from acal import s2sbase as S2

BOARD_LABELS = {"aires": "AI+RES", "cfs13": "CFSv2", "ec46": "ECMWF IFS (EC46)",
                "gefs": "GEFSv12", "geps": "ECCC GEPS", "bbsubs": "BB-SUBS (est.)"}
BOARD_COLORS = {"aires": "#D55E00", "cfs13": "#0072B2", "ec46": "#009E73",
                "gefs": "#CC79A7", "geps": "#E69F00", "bbsubs": "#56B4E9"}


def test_model_style_labels_and_colours():
    for k, lab in BOARD_LABELS.items():
        assert S2.MODEL_STYLE[k]["label"] == lab
        assert S2.MODEL_STYLE[k]["color"].upper() == BOARD_COLORS[k].upper()
    assert S2.MODEL_STYLE["bbsubs"].get("estimate") is True
    assert S2.MODEL_STYLE["era5"]["color"] == "#000000"
    assert S2.MODEL_STYLE["hrrr"]["color"] == "#555555"
    for k in ("cfs13", "gefs", "geps", "ec46"):           # native-grid caveat on maps
        assert "deg" in S2.MODEL_STYLE[k]["deg"]


def test_shade_is_darker_same_hue():
    assert S2.shade("#CC79A7", 0.0).lower() == "#cc79a7"
    assert S2.shade("#ffffff", 1.0) == "#000000"
    d = S2.shade("#E69F00")
    rgb = [int(d[i:i + 2], 16) for i in (1, 3, 5)]
    assert rgb == [round(v * 0.55) for v in (0xE6, 0x9F, 0x00)]


def test_has_data_needs_a_cube(tmp_path, monkeypatch):
    assert S2.has_data("cfs13")
    assert not S2.has_data("nonexistent")
    src = S2.get("gefs")
    monkeypatch.setattr(S2, "S2S_ROOT", tmp_path)          # Source.root follows S2S_ROOT
    assert src.root == tmp_path / "gefs"
    assert not S2.has_data("gefs")
    (tmp_path / "gefs").mkdir()
    (tmp_path / "gefs" / "bias.nc").write_bytes(b"")       # bias alone is not data
    assert not S2.has_data("gefs")
    (tmp_path / "gefs" / "e01_h2_20210119.nc").write_bytes(b"")
    assert S2.has_data("gefs")


def test_fig_dir_is_never_published():
    d = S2.fig_dir("gefs", "hrrr")
    assert d.parts[-4:] == ("acal", "s2s", "hrrr", "gefs")


def test_roc_published_context_unchanged():
    assert R._CTX == R._PUBLISHED
    assert R._PUBLISHED["style"] is R.MODEL_STYLE
    assert (R._PUBLISHED["corr"], R._PUBLISHED["raw"], R._PUBLISHED["fc"]) == \
        ("cfs_corr", "cfs_raw", "CFSv2")


def test_roc_source_context(tmp_path):
    ctx = R.source_ctx("geps", True, "hrrr", tmp_path)
    assert list(ctx["style"]) == ["ai_res", "geps_corr", "geps_raw", "clim"]
    assert ctx["style"]["ai_res"]["color"] == "#D55E00"
    assert ctx["style"]["geps_corr"]["color"] == "#E69F00"
    assert ctx["style"]["geps_raw"]["color"] == S2.shade("#E69F00")
    assert "12 frames" in ctx["foot"][0] and "ECCC GEPS 21 members" in ctx["foot"][1]
    assert "Lead to peak" in ctx["foot"][0] and "21-day" not in ctx["foot"][0]
    assert ctx["thresh_color"] == R.THRESH_COLOR_BOARD
    assert not set(ctx["thresh_color"].values()) & {v["color"] for v in S2.MODEL_STYLE.values()}
    # no orphan word: every wrapped footnote line is at least a third of the longest
    lines = R._foot(R._lines(ctx["foot"]), ctx["wrap"]).split("\n")
    assert max(map(len, lines)) <= ctx["wrap"]
    assert min(map(len, lines)) >= max(map(len, lines)) / 3
    assert ctx["dir"] == tmp_path
    nob = R.source_ctx("gefs", False, "era5", tmp_path)
    assert nob["corr"] is None and "gefs_corr" not in nob["style"]


def test_reach_published_context_unchanged():
    assert RE._FIG == RE._PUBLISHED
    assert RE._rows() == RE.ROWS
    assert RE._PUBLISHED["lead"] + ", equally weighted" == RE.STRIP_NOTE


def test_reach_source_context(tmp_path):
    ctx = RE.source_ctx("gefs", "era5", tmp_path)
    assert ctx["fc"] == "GEFSv12" and "GEFSv12 31 members" in ctx["lead"]
    # legend and row labels use the short name (the long one overflowed the one-row legend);
    # the full label stays in the lead note
    ec = RE.source_ctx("ec46", "era5", tmp_path)
    assert ec["fc"] == "EC46" and "ECMWF IFS (EC46)" in ec["lead"]
    keep = dict(RE._FIG)
    try:
        RE._FIG.update(ctx)
        assert RE._rows()[1][1] == "GEFSv12 bias-corrected"
    finally:
        RE._FIG.clear()
        RE._FIG.update(keep)


def test_aires_prob12_estimator(monkeypatch):
    """Self-normalized weighted exceedance, the estimator `maps.fields` uses."""
    from acal import maps as M
    rng = np.random.default_rng(0)
    F = rng.normal(0, 3, (5, 3, 4)).astype("float32")
    w = rng.uniform(0.1, 2.0, 5)
    da = xr.DataArray(F, dims=("walker", "lat", "lon"))
    monkeypatch.setattr(S2, "aires_fields", lambda eid, window="13f": (da, w))
    got = M.aires_prob12("eX")
    assert got.shape == (len(M.THRESHOLDS), 3, 4)
    wn = w / w.sum()
    for j, a in enumerate(M.THRESHOLDS):
        want = (wn[:, None, None] * (np.sign(a) * F >= np.sign(a) * a)).sum(0)
        np.testing.assert_allclose(got[j], want, rtol=1e-6)


def test_aires_fields_window_refused():
    with pytest.raises(ValueError):
        S2.aires_fields("e02_c4_20210218", "d6")


@pytest.mark.skipif(not S2.aires_fields_path("e02_c4_20210218").exists(),
                    reason="AI+RES field cache not built")
def test_aires_fields_cache_matches_tables():
    """13f fields reproduce compare.json's walker A_L, 12f the window table's al12, and
    the 13f fields equal the published maps path (`maps.walker_fields`) to float32."""
    tab = pd.read_csv(S2.AIRES_WINDOWS_CSV)
    g = tab[tab.eid == "e02_c4_20210218"].sort_values("walker")
    for win, col in (("13f", "al13"), ("12f", "al12")):
        F, w = S2.aires_fields("e02_c4_20210218", win)
        got = np.asarray(S2.AI.area_mean(F), dtype="float64")
        np.testing.assert_allclose(got, g[col].to_numpy(), atol=S2.CONSISTENCY_TOL)
        np.testing.assert_allclose(w, g.weight.to_numpy(), rtol=1e-8)   # table precision


def test_driver_script_exists_and_lists_stages():
    p = Path(__file__).resolve().parents[2] / "scripts" / "acal_board_all.sh"
    if not p.exists():
        pytest.skip("driver not written yet")
    txt = p.read_text()
    for k in ("--stage with_data", "--stage score", "--stage paired", "--stage maps",
              "acal.roc", "acal.reach", "acal.sidebyside", "acal.board"):
        assert k in txt, k


def test_maps_aires_scored_on_source_cases():
    """A partial source: AI+RES maps samples outside the source's (case, day) are NaN, so
    both sides are scored on the same samples; full coverage leaves AI+RES untouched."""
    from acal import maps as M
    cases = ["e01", "e02", "e03"]
    shp = (3, 2, 2, 2)
    res = xr.Dataset({"prob": (("case", "threshold", "lat", "lon"), np.full(shp, 0.5))},
                     coords=dict(case=cases))
    src = xr.Dataset({"prob_raw": (("case", "threshold", "lat", "lon"), np.full(shp, 0.2))},
                     coords=dict(case=cases))
    assert M.on_source_cases(res, "prob", src) is res
    src["prob_raw"][1] = np.nan                                  # e02 not built
    got = M.on_source_cases(res, "prob", src)
    assert bool(got.prob.sel(case="e02").isnull().all())
    assert bool(got.prob.sel(case=["e01", "e03"]).notnull().all())
    assert bool(res.prob.notnull().all())                        # input not modified
    part = src.sel(case=["e01", "e03"])                          # case missing from the file
    assert bool(M.on_source_cases(res, "prob", part).prob.sel(case="e02").isnull().all())
    # daily source vs daily AI+RES: per (case, day)
    rd = xr.Dataset({"prob": (("case", "day", "threshold"), np.full((3, 2, 2), 0.5))},
                    coords=dict(case=cases))
    sd = xr.Dataset({"prob_raw": (("case", "day", "threshold"), np.full((3, 2, 2), 0.2))},
                    coords=dict(case=cases))
    sd["prob_raw"][0, 0] = np.nan
    g = M.on_source_cases(rd, "prob", sd)
    assert int(g.prob.isnull().sum()) == 2 and bool(g.prob[0, 0].isnull().all())


def test_reach_all_one_tail_only(monkeypatch):
    """A partial source with 4 K hits of one sign only: no crash on the empty tail."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(RE, "_strip", lambda *a, **k: None)
    monkeypatch.setattr(RE, "_save", lambda fig, name: name)
    m = dict(al=np.r_[4.0, 5.0], raw=np.r_[3.0], corr=np.r_[3.5], obs=4.2, sign=1.0,
             peak="2021-06-30")
    ms = {"e01_h4": m, "e02_h4": dict(m, obs=4.5)}
    assert RE._xlims(list(ms.values())) == (2.0, 6.0)
    assert RE.fig_reach_all(ms, list(ms), plt) == "reach_4K.png"
    assert RE.fig_reach_all({}, [], plt) is None
    plt.close("all")


def test_write_json_rewrites_record_older_than_cube(tmp_path, monkeypatch):
    """A json record older than its cube (the cube was rebuilt) is rewritten; a fresh one
    is reused."""
    import json
    import os
    monkeypatch.setattr(S2, "S2S_ROOT", tmp_path)
    src = S2.get("gefs")
    src.root.mkdir(parents=True)
    eid = S2._slate().episode_id.iloc[0]
    calls = []

    def fake_record(s, r, cube, truth):
        calls.append(float(cube["x"].values))
        return dict(episode_id=r.episode_id, obs=0.0, obs_window=0.0, al_mean=calls[-1],
                    n_members=1, n_reach=0, al=[calls[-1]])
    monkeypatch.setattr(S2, "case_record", fake_record)
    cp = src.cube_path(eid)
    xr.Dataset({"x": 1.0}).to_netcdf(cp)
    S2.write_json(src, cases=[eid])
    S2.write_json(src, cases=[eid])                    # json newer than cube: reused
    assert calls == [1.0]
    xr.Dataset({"x": 2.0}).to_netcdf(cp)                # cube rebuilt
    t = src.json_path(eid).stat().st_mtime
    os.utime(cp, (t + 10, t + 10))
    df = S2.write_json(src, cases=[eid])
    assert calls == [1.0, 2.0] and df.al_mean.iloc[0] == 2.0
    assert json.loads(src.json_path(eid).read_text())["al_mean"] == 2.0
