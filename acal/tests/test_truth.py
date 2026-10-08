"""The truth switch: ERA5 parity, a fake truth's mask on forecasts and truth alike, and the
P_clim pool following the truth. Runs on the real runs/acal tree (skipped elsewhere)."""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acal import analyze as AN
from acal import cfsbase as CB
from acal import maps as M
from acal import s2sbase as S
from acal import truth as TR

needs_runs = pytest.mark.skipif(
    not (CB.BUILD_CSV.exists() and AN.SCORE_OUT.exists() and M.FIELDS.exists()),
    reason="runs/acal not here")
OFFSET = 0.5                                    # K, the fake truth's constant shift


@functools.lru_cache(maxsize=1)
def _era5_index() -> xr.DataArray:
    return M.era5_index()


def west_mask() -> xr.DataArray:
    """True west of 265 E: a mask that cuts CONUS roughly in half."""
    lat, lon = S._clim_grid()
    m = np.broadcast_to(lon[None, :] < 265.0, (lat.size, lon.size))
    return xr.DataArray(m.copy(), dims=("lat", "lon"), coords=dict(lat=lat, lon=lon))


class FakeTruth(TR.Truth):
    """ERA5 + a constant offset, optionally NaN outside a mask."""
    name, label = "fake", "Fake"

    def __init__(self, offset: float = OFFSET, mask: xr.DataArray | None = None,
                 mask_name: str = "fakemask"):
        self.offset, self._mask, self.mask_name = offset, mask, mask_name

    def mask(self):
        return self._mask

    def frames(self, eid):
        return TR.apply_mask(TR.get_truth("era5").frames(eid) + self.offset, self._mask)

    def daily_conus(self):
        return TR.get_truth("era5").daily_conus() + self.offset

    def index(self):
        return TR.apply_mask(_era5_index() + self.offset, self._mask)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
def test_get_truth_dispatch():
    assert TR.get_truth("era5").name == "era5"
    with pytest.raises(KeyError):
        TR.get_truth("gfs")


def test_dirs_and_tags():
    era5 = TR.get_truth("era5")
    assert TR.analysis_dir(era5).parts[-3:] == ("analysis", "s2s", "era5")
    assert TR.fig_dir(era5).parts[-2:] == ("s2s", "era5")       # never the published dir
    assert TR.mask_tag(era5) == ""
    fake = FakeTruth(mask=west_mask())
    assert TR.mask_tag(fake) == "_fakemask" and S._mask_tag(fake) == "_fakemask"

    class H:                                  # lane B's duck-typed provider shape
        name = "hrrr_raw"
    assert TR.fig_dir(H()).parts[-2:] == ("hrrr", "raw")
    H.name = "hrrr"
    assert TR.fig_dir(H()).parts[-1] == "hrrr"


@needs_runs
def test_pool_series_era5_is_the_catalog():
    """ERA5's pool input is the published catalog, value for value (parity)."""
    era5 = TR.get_truth("era5")
    assert TR.pool_series(era5).equals(AN.load_daily())
    full = era5.daily_conus()                         # may extend through 2026-08-31
    assert full.index.min() == pd.Timestamp("2021-01-01")
    assert full.loc[:"2025-12-31"].equals(AN.load_daily())


@needs_runs
def test_pool_switches_with_truth():
    """The pool is the truth's own series on the same days; a constant shift of obs AND
    pool leaves P_clim unchanged (shift invariance), which a pool left on ERA5 would not."""
    era5, fake = TR.get_truth("era5"), FakeTruth()
    pe, pf = TR.pool_series(era5), TR.pool_series(fake)
    assert pe.index.equals(pf.index)
    np.testing.assert_allclose(pf.values, pe.values + OFFSET)
    peak, obs, sign = pd.Timestamp("2021-02-18"), -5.0, -1.0
    p_e = AN.p_clim(AN.clim_pool(pe, peak), obs, sign)
    p_f = AN.p_clim(AN.clim_pool(pf, peak), obs + OFFSET, sign)
    assert p_e == p_f
    # ... and a truth-shifted obs against the ERA5 pool would have moved
    assert AN.p_clim(AN.clim_pool(pe, peak), obs + 2.0, sign) != p_e


# --------------------------------------------------------------------------- #
# ERA5 parity of the threaded scorers
# --------------------------------------------------------------------------- #
@needs_runs
def test_scorecard_era5_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(TR, "analysis_dir", lambda t: tmp_path / t.name)
    AN.scorecard("era5")
    assert (tmp_path / "era5" / "scorecard.csv").read_bytes() == AN.SCORE_OUT.read_bytes()
    assert (tmp_path / "era5" / "summary.json").read_bytes() == AN.SUMMARY_OUT.read_bytes()


@needs_runs
def test_reach_truth_rows_era5_parity():
    from acal import reach as RE
    from acal import roc as R
    a = R.case_rows()
    b, _ = RE.truth_rows("era5")
    cols = ["episode_id", "pool", "threshold_K", "obs", "outcome", "p_ai_res", "p_ai_res_raw",
            "p_cfs_corr", "p_cfs_raw", "p_clim", "k_clim", "n_clim", "n_walkers_beyond",
            "n_cfs_raw_beyond", "n_cfs_corr_beyond"]
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True),
                                  b[cols].reset_index(drop=True), check_dtype=False)


@needs_runs
def test_maps_truth_dataset_era5_parity_and_fake_mask():
    with xr.open_dataset(M.FIELDS) as b:
        base = b.isel(case=slice(0, 2)).load()
    era5 = TR.get_truth("era5")
    era5.index = _era5_index                         # same values, loaded once
    t = M.truth_dataset(base, era5, daily=False)
    for v in ("truth", "clim", "prob"):
        assert np.array_equal(t[v].values, base[v].values, equal_nan=True), v
    assert "valid" not in t

    mask = west_mask()
    f = M.truth_dataset(base, FakeTruth(mask=mask), daily=False)
    m = mask.values
    assert f["valid"].values.astype(bool).tolist() == m.tolist()
    # truth: the offset inside the mask, NaN outside (applied to the truth)
    np.testing.assert_allclose(f["truth"].values[:, m], base["truth"].values[:, m] + OFFSET,
                               atol=1e-4)
    assert np.isnan(f["truth"].values[:, ~m]).all()
    assert np.isnan(f["clim"].values[:, :, ~m]).all()
    # forecasts: per-cell probabilities kept, but scored on the mask only
    np.testing.assert_array_equal(f["prob"].values, base["prob"].values)
    sc = M.scores(f)
    assert sc["csi"].isel(threshold=0).values[~m].size and \
        np.isnan(sc["csi"].values[:, ~m]).all()
    assert np.isfinite(sc["accuracy"].values[:, m]).any()


# --------------------------------------------------------------------------- #
# The mask reaches the forecasts
# --------------------------------------------------------------------------- #
@needs_runs
def test_source_members_rereduced_on_the_mask(monkeypatch):
    """A masked truth re-reduces every source member on the SAME mask as its obs; an
    all-True mask reproduces the published member A_L."""
    src = S.get("cfs13")
    eps = S.aprep.episodes()
    two = eps[eps.episode_id.isin(["e01_h2_20210119", "e02_c4_20210218"])]
    if len(two) != 2 or not all(src.json_path(e).exists() for e in two.episode_id):
        pytest.skip("cfs13 json not built")
    monkeypatch.setattr(S.aprep, "episodes", lambda: two.copy())

    pub = S.load(src, TR.get_truth("era5"))
    lat, lon = S._clim_grid()
    full = FakeTruth(mask=xr.DataArray(np.ones((lat.size, lon.size), bool), dims=("lat", "lon"),
                                       coords=dict(lat=lat, lon=lon)))
    got = S.load(src, full)
    for a, b in zip(pub.al, got.al):
        np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_allclose(got.obs.values, pub.obs.values + OFFSET, atol=2e-3)

    mask = west_mask()
    fake = FakeTruth(mask=mask)
    got = S.load(src, fake)
    for r, a in zip(got.itertuples(), got.al):
        with xr.open_dataset(src.cube_path(r.episode_id)) as cube:
            want = S.member_al(cube.load(), r.peak, src.window, mask)
        np.testing.assert_allclose(a, want, atol=1e-6)
        assert np.abs(np.asarray(a) - np.asarray(pub.set_index("episode_id")
                                                  .loc[r.episode_id, "al"])).max() > 0.01
        f = TR.get_truth("era5").frames(r.episode_id).astype("float64")
        f = f.assign_coords(lat=f["lat"].astype("float64"))   # obs_al reduces in float64
        era5_masked = float(TR.area_mean(f.mean("time"), mask))
        assert r.obs == pytest.approx(era5_masked + OFFSET, abs=1e-5)


@needs_runs
def test_aires_cases_use_masked_walkers():
    """AI+RES walkers are re-reduced on a masked truth's mask (aires_al_windows.csv
    al13_<tag>); a missing mask column is refused, never silently unmasked."""
    with pytest.raises(SystemExit):
        S.aires_cases(FakeTruth(mask=west_mask(), mask_name="nosuchmask"), "13f")
    tab = pd.read_csv(S.AIRES_WINDOWS_CSV)
    if "al13_hrrrmask" not in tab:
        pytest.skip("masked walker columns not built (s2sbase --stage aires_mask)")
    fake = FakeTruth(mask=west_mask(), mask_name="hrrrmask")
    fake.obs_al = lambda eid, window="13f": 9.0          # obs from the truth, not the slate
    cases = S.aires_cases(fake, "13f")
    c = cases["e02_c4_20210218"]
    g = tab[tab.eid == c.episode_id].sort_values("walker")
    np.testing.assert_array_equal(c.al, g["al13_hrrrmask"].to_numpy())
    assert c.obs == 9.0
    unmasked = S.aires_cases(FakeTruth(), "13f")["e02_c4_20210218"]
    np.testing.assert_array_equal(unmasked.al, g["al13"].to_numpy())


@needs_runs
def test_fake_truth_scorecard_moves_obs_and_pool_together(tmp_path, monkeypatch):
    """Against ERA5 + 0.5 K (no mask): P_RES is evaluated at the shifted obs and P_clim is
    unchanged (obs and pool shifted together)."""
    monkeypatch.setattr(TR, "analysis_dir", lambda t: tmp_path / t.name)
    sc = AN.scorecard(FakeTruth())
    pub = pd.read_csv(AN.SCORE_OUT)
    np.testing.assert_allclose(sc.obs.values, pub.obs.values + OFFSET, atol=1e-3)
    np.testing.assert_allclose(sc.p_clim_obs.values, pub.p_clim_obs.values, atol=1e-6)  # %.6g
    assert sc.catalog_match.all()
    _, cases = AN.load_all()
    c = cases[0]
    assert sc.p_obs_raw.iloc[0] == pytest.approx(c.p_raw(sc.obs.iloc[0]))


# --------------------------------------------------------------------------- #
# End to end on HRRR (lane B's provider), when its index exists
# --------------------------------------------------------------------------- #
@needs_runs
def test_hrrr_end_to_end():
    try:
        h = TR.get_truth("hrrr")
        mask = h.mask()
    except (SystemExit, ImportError, FileNotFoundError) as e:
        pytest.skip(f"HRRR truth not available: {e}")
    assert TR.mask_tag(h) == "_hrrrmask" and int(mask.sum()) == 23902
    pe, ph = TR.pool_series(TR.get_truth("era5")), TR.pool_series(h)
    assert pe.index.equals(ph.index)                    # same pool days, HRRR values
    eid = "e02_c4_20210218"
    obs = h.obs_al(eid, "13f")
    assert abs(ph.loc["2021-02-18"] - obs) <= AN.CATALOG_TOL   # pool and obs: one statistic
    fr = TR.frames_on(h, eid, "12f")
    assert fr.sizes["time"] == 12 and np.isnan(fr.values[:, ~mask.values]).all()
