"""HRRR truth: bin-mean remap, geometric mask, LOYO offset, provider interface."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acal import hrrrtruth as H


# --------------------------------------------------------------------------- remap
def _fine_grid(lat0, lat1, lon0, lon1, step):
    la = np.arange(lat0, lat1 + 1e-9, step)
    lo = np.arange(lon0, lon1 + 1e-9, step)
    return np.meshgrid(la, lo, indexing="ij")


def test_bin_mean_linear_field_is_cell_centre_value():
    tlat = np.arange(30.0, 31.01, 0.25)
    tlon = np.arange(260.0, 261.01, 0.25)
    # 0.05 deg points, offset so no point sits on a cell edge; symmetric around each centre
    hla, hlo = _fine_grid(29.9, 31.1, 259.9, 261.1, 0.05)
    flat, cnt = H.bin_index(hla, hlo, tlat, tlon)
    f = 3.0 * hla - 2.0 * hlo + 7.0
    b = H.bin_mean(f, flat, cnt)
    TLA, TLO = np.meshgrid(tlat, tlon, indexing="ij")
    np.testing.assert_allclose(b, 3.0 * TLA - 2.0 * TLO + 7.0, atol=1e-9)
    assert (cnt == 25).all()                     # 5 x 5 fine points per 0.25 cell


def test_bin_mean_empty_cells_nan_and_outside_points_dropped():
    tlat = np.array([30.0, 30.25, 30.5])
    tlon = np.array([260.0, 260.25])
    # points only near (30.0, 260.0) plus points far outside the target grid
    hla = np.array([[30.01, 29.99, 45.0]])
    hlo = np.array([[260.02, -99.99, 200.0]])    # -99.99 E == 260.01 E (wrapped)
    flat, cnt = H.bin_index(hla, hlo, tlat, tlon)
    assert flat.tolist() == [0, 0, -1]
    b = H.bin_mean(np.array([[1.0, 3.0, 1e6]]), flat, cnt)
    assert b[0, 0] == 2.0
    assert np.isnan(b).sum() == b.size - 1


def test_corner_mask_is_strict_subset_of_centre_inside():
    import pyproj
    P = pyproj.Proj(**H.LCC)
    nx = ny = 120
    dx = 3000.0
    xc, yc = P(262.5, 38.5)
    x0, y0 = xc - nx / 2 * dx, yc - ny / 2 * dx
    tlat = np.arange(36.0, 41.01, 0.25)
    tlon = np.arange(258.0, 267.01, 0.25)
    m = H.corner_mask(tlat, tlon, x0=x0, y0=y0, nx=nx, ny=ny, dx=dx)
    TLA, TLO = np.meshgrid(tlat, tlon, indexing="ij")
    x, y = P(TLO, TLA)
    i, j = (x - x0) / dx, (y - y0) / dx
    centre = (i >= -0.5) & (i <= nx - 0.5) & (j >= -0.5) & (j <= ny - 0.5)
    assert m.any() and (centre & ~m).any()       # some edge cells: centre in, a corner out
    assert not (m & ~centre).any()
    assert m[np.argmin(abs(tlat - 38.5)), np.argmin(abs(tlon - 262.5))]
    # every corner of every masked cell projects inside the source rectangle
    for a in (-0.125, 0.125):
        for b in (-0.125, 0.125):
            cx, cy = P(TLO[m] + b, TLA[m] + a)
            assert ((cx - x0) / dx >= -0.5).all() and ((cx - x0) / dx <= nx - 0.5).all()
            assert ((cy - y0) / dx >= -0.5).all() and ((cy - y0) / dx <= ny - 0.5).all()


def test_area_mean_coslat_and_mask():
    lat = np.array([0.0, 60.0])
    a = np.array([[1.0, 1.0], [4.0, np.nan]])
    m = np.array([[True, True], [True, False]])
    # weights 1, 1, 0.5 -> (1 + 1 + 0.5*4) / 2.5
    assert H.area_mean(a, lat, m) == pytest.approx(4.0 / 2.5)
    assert H.area_mean(np.ones((3, 2, 2)), lat).shape == (3,)


# --------------------------------------------------------------------------- offset
def _times(years):
    t = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31T12", freq="12h")
    return t[t.year.isin(years)]


def test_loyo_offset_excludes_own_year():
    times = _times([2021, 2022, 2023])
    diff = np.zeros((len(times), 2, 3), "float32")
    for i, t in enumerate(times):
        diff[i] = {2021: 1.0, 2022: 2.0, 2023: 4.0}[t.year]
    offs, S, N = H.loyo_offset(diff, times)
    np.testing.assert_allclose(offs[2021], 3.0)          # (2 + 4) / 2
    np.testing.assert_allclose(offs[2022], 2.5)
    np.testing.assert_allclose(offs[2023], 1.5)
    # a huge value in the own year never leaks into its own offset
    diff2 = diff.copy()
    diff2[times.year == 2022] = 1e6
    offs2, _, _ = H.loyo_offset(diff2, times)
    np.testing.assert_allclose(offs2[2022], 2.5)
    assert offs2[2021].min() > 1e5


def test_loyo_offset_by_month_and_hour_and_nan_cells():
    times = _times([2021, 2022])
    diff = np.zeros((len(times), 1, 2), "float32")
    for i, t in enumerate(times):
        diff[i] = t.month + (0.5 if t.hour == 12 else 0.0)
    diff[:, 0, 1] = np.nan                                # a cell never observed
    offs, _, _ = H.loyo_offset(diff, times)
    for m in range(12):
        assert offs[2021][m, 0, 0, 0] == pytest.approx(m + 1)
        assert offs[2021][m, 1, 0, 0] == pytest.approx(m + 1.5)
    assert np.isnan(offs[2021][:, :, 0, 1]).all()


def test_window_times():
    t = H.window_times("2021-02-18", "13f")
    assert len(t) == 13 and t[0] == pd.Timestamp("2021-02-12T00") \
        and t[-1] == pd.Timestamp("2021-02-18T00") and set(t.hour) == {0, 12}
    t12 = H.window_times("2021-02-18", "12f")
    assert len(t12) == 12 and t12[-1] == pd.Timestamp("2021-02-17T12")
    with pytest.raises(ValueError):
        H.window_times("2021-02-18", "25f")


# --------------------------------------------------------------------------- interface
@pytest.fixture
def synthetic_index(tmp_path, monkeypatch):
    """A one-year HRRR + ERA5 index on a 4 x 5 grid with a known mask and offset."""
    times = pd.date_range("2020-12-26", "2021-01-31", freq="12h")
    lat = np.array([30.0, 30.25, 30.5, 30.75], "float32")
    lon = np.array([260.0, 260.25, 260.5, 260.75, 261.0], "float32")
    rng = np.random.default_rng(3)
    era = rng.normal(size=(len(times), 4, 5)).astype("float32")
    mask = np.ones((4, 5), bool)
    mask[0, :2] = False
    raw = (era + 0.3).astype("float32")
    raw[:, ~mask] = np.nan
    anom = (raw - 0.3).astype("float32")
    hd, ed = tmp_path / "index_hrrr", tmp_path / "index"
    hd.mkdir(); ed.mkdir()
    xr.Dataset({"anom": (("time", "lat", "lon"), anom),
                "anom_raw": (("time", "lat", "lon"), raw),
                "mask": (("lat", "lon"), mask.astype("int8"))},
               coords={"time": times, "lat": lat, "lon": lon}
               ).to_netcdf(hd / "hrrr_t2m_anom_12h_2021.nc")
    xr.Dataset({"t2m_anom": (("time", "lat", "lon"), era)},
               coords={"time": times, "lat": lat, "lon": lon}
               ).to_netcdf(ed / "era5_t2m_anom_12h_2021.nc")
    ep = tmp_path / "episodes.csv"
    pd.DataFrame(dict(episode_id=["eX"], family=["heat"], rung=[2], peak=["2021-01-20"],
                      a_l_conus=[0.0])).to_csv(ep, index=False)
    monkeypatch.setattr(H, "OUT", hd)
    monkeypatch.setattr(H, "ERA5_INDEX", ed)
    monkeypatch.setattr(H, "YEARS", (2021,))
    monkeypatch.setattr(H, "EPISODES", ep)
    monkeypatch.setattr(H, "DAILY_CSV", hd / "daily.csv")
    monkeypatch.setattr(H, "INDEX_END", pd.Timestamp("2021-01-31"))
    monkeypatch.setattr(H, "DAILY_END", "2021-01-31")
    for f in (H._index, H._mask, H._peaks, H.get_truth):
        f.cache_clear()
    yield dict(times=times, lat=lat, era=era, raw=raw, anom=anom, mask=mask)
    for f in (H._index, H._mask, H._peaks, H.get_truth):
        f.cache_clear()


def test_provider_interface(synthetic_index):
    s = synthetic_index
    hr, raw = H.get_truth("hrrr"), H.get_truth("hrrr_raw")
    assert (hr.name, raw.name) == ("hrrr", "hrrr_raw")
    m = hr.mask()
    assert m.dtype == bool and m.dims == ("lat", "lon") and int(m.sum()) == s["mask"].sum()
    f = hr.frames("eX")
    assert f.dims == ("time", "lat", "lon") and f.sizes["time"] == 13
    assert pd.Timestamp(f.time.values[0]) == pd.Timestamp("2021-01-14T00")
    assert pd.Timestamp(f.time.values[-1]) == pd.Timestamp("2021-01-20T00")
    assert np.isnan(f.values[:, ~s["mask"]]).all() and np.isfinite(f.values[:, s["mask"]]).all()
    assert hr.frames("eX", "12f").sizes["time"] == 12
    k = s["times"].get_indexer(H.window_times("2021-01-20"))
    want = H.area_mean(s["anom"][k].mean(0), s["lat"], s["mask"])
    assert hr.obs_al("eX") == pytest.approx(float(want), abs=1e-6)
    assert raw.obs_al("eX") - hr.obs_al("eX") == pytest.approx(0.3, abs=1e-5)
    assert hr.obs_al("eX", "12f") != hr.obs_al("eX", "13f")


def test_daily_conus_definition(synthetic_index):
    s = synthetic_index
    d = H.get_truth("hrrr").daily_conus()
    assert isinstance(d, pd.Series) and d.index[0] == pd.Timestamp("2021-01-01") \
        and d.index[-1] == pd.Timestamp("2021-01-31") and len(d) == 31
    k = s["times"].get_indexer(H.window_times("2021-01-10"))
    want = H.area_mean(s["anom"][k], s["lat"], s["mask"]).mean()
    assert d.loc["2021-01-10"] == pytest.approx(want, abs=1e-4)
    dr = H.get_truth("hrrr_raw").daily_conus()
    np.testing.assert_allclose((dr - d).values, 0.3, atol=2e-4)


def test_get_truth_rejects_unknown():
    with pytest.raises(ValueError):
        H.HrrrTruth("era5")


# --------------------------------------------------------------------------- built index
def test_built_index_matches_era5_axis():
    files = H.index_files()
    if not all(f.exists() for f in files):
        pytest.skip("HRRR index not built")
    for f in files:
        y = int(f.stem[-4:])
        with xr.open_dataset(f) as h, xr.open_dataset(H.ERA5_INDEX / f"era5_t2m_anom_12h_{y}.nc") as e:
            assert np.array_equal(h.time.values, e.time.values)
            assert np.array_equal(h.lat.values, e.lat.values)
            assert np.array_equal(h.lon.values, e.lon.values)
            m = h["mask"].values > 0
            assert int(m.sum()) == 23902
            for v in ("anom", "anom_raw"):
                x = h[v].isel(time=[0, h.sizes["time"] // 2, -1]).values
                assert np.isfinite(x[:, m]).all() and np.isnan(x[:, ~m]).all()
