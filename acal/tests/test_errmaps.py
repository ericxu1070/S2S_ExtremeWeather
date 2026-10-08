"""acal.errmaps on synthetic fields: composite weighting, masks, statistics, absent sources."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acal import errmaps as E  # noqa: E402

LAT = np.linspace(24.0, 50.0, 9)
COS = np.cos(np.deg2rad(LAT))


def _fields(rng, n=5, ny=9, nx=12):
    return rng.normal(size=(n, ny, nx)), rng.normal(size=(n, ny, nx))


def test_composite_is_equal_weight_over_selected_cases():
    rng = np.random.default_rng(0)
    F, O = _fields(rng)
    sel = np.array([True, False, True, True, False])
    cp = E.composite(F, O, sel)
    assert cp["n"] == 3
    np.testing.assert_allclose(cp["err"], (F[sel] - O[sel]).sum(0) / 3)
    np.testing.assert_allclose(cp["fc"], F[sel].mean(0))
    np.testing.assert_allclose(cp["obs"], O[sel].mean(0))
    np.testing.assert_allclose(cp["mae"], np.abs(F[sel] - O[sel]).mean(0))


def test_composite_drops_cases_without_a_forecast():
    rng = np.random.default_rng(1)
    F, O = _fields(rng)
    F[2] = np.nan                                     # source has no cube for case 2
    sel = np.ones(5, bool)
    cp = E.composite(F, O, sel)
    keep = np.array([1, 1, 0, 1, 1], bool)
    assert cp["n"] == 4
    np.testing.assert_allclose(cp["err"], (F[keep] - O[keep]).mean(0))
    # the truth composite of that source is over ITS cases, not all five
    np.testing.assert_allclose(cp["obs"], O[keep].mean(0))
    none = E.composite(F, O, np.zeros(5, bool))
    assert none["n"] == 0 and np.isnan(none["err"]).all()


def test_composite_keeps_masked_cells_nan():
    rng = np.random.default_rng(2)
    F, O = _fields(rng)
    O[:, :2, :] = np.nan                              # truth off-mask (HRRR) rows
    cp = E.composite(F, O, np.ones(5, bool))
    assert np.isnan(cp["err"][:2]).all() and np.isfinite(cp["err"][2:]).all()


def test_wstats_constant_offset_and_perfect_pattern():
    rng = np.random.default_rng(3)
    O = rng.normal(size=(9, 12))
    valid = np.ones_like(O, bool)
    b, rmse, r = E.wstats(O + 1.5, O, valid, COS)
    assert b == pytest.approx(1.5) and rmse == pytest.approx(1.5) and r == pytest.approx(1.0)
    b, rmse, r = E.wstats(-2 * O + 3, O, valid, COS)
    assert r == pytest.approx(-1.0)


def test_wstats_is_coslat_weighted():
    O = np.zeros((9, 12))
    F = np.zeros((9, 12))
    F[0] = 1.0                                        # error only on the southern row
    b, _, _ = E.wstats(F, O, np.ones_like(O, bool), COS)
    assert b == pytest.approx(COS[0] / COS.sum())


def test_mask_excludes_cells_from_every_statistic():
    rng = np.random.default_rng(4)
    O = rng.normal(size=(9, 12))
    F = O + rng.normal(scale=0.3, size=O.shape)
    valid = np.ones_like(O, bool)
    valid[:, :4] = False                              # "sea"
    ref = E.wstats(F[:, 4:], O[:, 4:], np.ones((9, 8), bool), COS)
    F2 = F.copy()
    F2[:, :4] = 1e6                                   # garbage off the mask is ignored
    got = E.wstats(F2, O, valid, COS)
    np.testing.assert_allclose(got, ref)
    O2 = O.copy()
    O2[:, :4] = np.nan                                # NaN truth off the mask too
    np.testing.assert_allclose(E.wstats(F, O2, np.ones_like(O, bool), COS), ref)
    assert E.wmean(F2, valid, COS) == pytest.approx(E.wmean(F[:, 4:], np.ones((9, 8), bool), COS))
    assert np.isnan(E.wstats(F, O, np.zeros_like(valid), COS)[0])


def test_pooled_stats_pool_cases_and_skip_missing():
    rng = np.random.default_rng(5)
    F, O = _fields(rng, n=4)
    valid = np.ones((9, 12), bool)
    F[1] = np.nan
    mae, rmse, r = E.pooled_stats(F, O, valid, COS, np.ones(4, bool))
    use = [0, 2, 3]
    want_mae = np.mean([E.wmean(np.abs(F[i] - O[i]), valid, COS) for i in use])
    want_rmse = np.sqrt(np.mean([E.wstats(F[i], O[i], valid, COS)[1] ** 2 for i in use]))
    want_r = np.mean([E.wstats(F[i], O[i], valid, COS)[2] for i in use])
    assert (mae, rmse, r) == pytest.approx((want_mae, want_rmse, want_r))


def test_source_without_cubes_is_absent(monkeypatch, tmp_path):
    class _Src:
        window = "d6"

        def cube_path(self, eid):
            return tmp_path / f"{eid}.nc"            # never exists

    monkeypatch.setattr(E.S2, "get", lambda name: _Src())
    cases = pd.DataFrame(dict(episode_id=["e01_x", "e02_y"], family=["heat", "cold"],
                              peak=pd.to_datetime(["2021-01-19", "2021-02-18"])))
    assert E.source_ensmean("ec46", cases, LAT, np.arange(12.0)) is None


def test_style_labels_match_the_board():
    want = {"aires": "AI+RES", "cfs13": "CFSv2", "ec46": "ECMWF IFS (EC46)",
            "gefs": "GEFSv12", "geps": "ECCC GEPS"}
    for k, v in want.items():
        assert E.style(k)["label"] == v
        assert E.style(k)["color"].startswith("#")
    assert E.BOARD[0] == "aires" and set(E.BOARD) == set(want)
