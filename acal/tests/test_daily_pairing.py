"""Ruling C2: a daily-mean source's UTC day D is verified on the 12-hourly pair
(00Z D, 12Z D) - truth, climatology, AI+RES and the daily bias alike - not on the
(12Z D, 00Z D+1) pair the instantaneous sources' daily maps use."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acal import cfsbase as CB
from acal import maps as M
from acal import s2sbase as S
from acal import truth as TR

LAT = np.array([30.0, 40.0, 50.0])
LON = np.array([250.0, 260.0, 270.0, 280.0])
PEAK = pd.Timestamp("2022-07-20")
RATE = 0.1                                    # K per hour, the synthetic anomaly ramp


def _ramp(times) -> xr.DataArray:
    """Anomaly = RATE x hours since PEAK, the same in every cell: a pair's mean is the
    ramp at its midpoint, so a 12 h pairing error shows up as RATE x 12 = 1.2 K."""
    t = pd.DatetimeIndex(times)
    h = np.asarray((t - PEAK) / pd.Timedelta(hours=1), dtype="float64")
    v = RATE * h[:, None, None] + np.zeros((1, LAT.size, LON.size))
    return xr.DataArray(v, dims=("time", "lat", "lon"),
                        coords=dict(time=t.values, lat=LAT, lon=LON))


def _at(days, hour) -> np.ndarray:
    """The ramp value at `hour` UTC of each day."""
    return RATE * np.asarray((pd.DatetimeIndex(days) + pd.Timedelta(hours=hour) - PEAK)
                             / pd.Timedelta(hours=1), dtype="float64")


def test_utc_pairs_are_the_12f_frames():
    days = S.daily_days(PEAK)
    t12 = TR.window_times(PEAK, "12f")
    a = _ramp(TR.window_times(PEAK, "13f"))
    p = S.utc_pairs(a, days)
    assert p.dims == ("day", "lat", "lon") and p.sizes["day"] == 6
    np.testing.assert_allclose(p.values[:, 0, 0], _at(days, 6))       # midpoint 06Z D
    np.testing.assert_allclose(p.mean("day").values, a.sel(time=t12).mean("time").values)
    with pytest.raises(ValueError):
        S.utc_pairs(a, days + pd.Timedelta(hours=12))


def test_daily_bias_of_a_daily_mean_source_uses_the_utc_pair(monkeypatch):
    """A daily-mean forecast equal to the ERA5 (00Z D, 12Z D) pair has zero daily bias on
    maps days 1..6 (day 0 NaN). The pre-C2 pairing (12Z D, 00Z D+1) would read 12 h of
    ramp, -1.2 K, as model bias."""
    monkeypatch.setattr(S, "GRID", (LAT.size, LON.size))
    monkeypatch.setattr(S, "_clim_grid", lambda: (LAT, LON))
    clim = lambda times: xr.DataArray(                          # noqa: E731
        np.full((len(times), LAT.size, LON.size), 280.0), dims=("time", "lat", "lon"),
        coords=dict(time=pd.DatetimeIndex(times).values, lat=LAT, lon=LON))
    monkeypatch.setattr(S, "clim_at", clim)
    era = _ramp(pd.date_range(PEAK - pd.Timedelta(hours=156), PEAK, freq="12h"))
    monkeypatch.setattr(CB, "era5_window", lambda peak: era)
    days = pd.date_range(PEAK - pd.Timedelta(days=8), PEAK, freq="D")
    n = 5
    T = 280.0 + _at(days, 6)[None, :, None, None] + np.linspace(-1, 1, n)[:, None, None, None]
    cube = xr.Dataset({"2m_temperature": (("member", "time", "lat", "lon"),
                                          np.broadcast_to(T, (n, days.size, LAT.size,
                                                              LON.size)).copy())},
                      coords=dict(member=np.arange(n), time=days.values, lat=LAT, lon=LON),
                      attrs=dict(time_kind="daily_mean", step_h=24))
    src = S.Source(name="fake", label="F", native_deg=1.5, time_kind="daily_mean",
                   n_members=n, init_rule="-", bias="loyo", dataset_id="-", url="-")
    b = S._daily_bias(src, cube, PEAK)
    assert b.shape == (7, LAT.size, LON.size) and np.isnan(b[0]).all()
    np.testing.assert_allclose(b[1:], 0.0, atol=1e-9)
    old = (CB.daily_pairs(era, PEAK).values[1:]
           - S.utc_pairs(era, S.daily_days(PEAK)).values)
    np.testing.assert_allclose(old, RATE * 12.0)                # what C2 removed


def test_stale_daily_mean_hind_files_are_rebuilt(tmp_path):
    p = tmp_path / "h.nc"
    xr.Dataset(dict(x=0.0)).to_netcdf(p)
    dm = S.Source(name="fake", label="F", native_deg=1.5, time_kind="daily_mean",
                  n_members=4, init_rule="-", bias="loyo", dataset_id="-", url="-")
    inst = S.Source(name="fake2", label="F", native_deg=1.0, time_kind="instant",
                    n_members=4, init_rule="-", bias="loyo", dataset_id="-", url="-")
    assert not S._hind_red_current(dm, p) and S._hind_red_current(inst, p)
    q = tmp_path / "h2.nc"
    xr.Dataset(dict(x=0.0), attrs=dict(daily_pair=S.DAILY_PAIR)).to_netcdf(q)
    assert S._hind_red_current(dm, q)


def test_era5_daily_utc_pairs_and_gaps():
    t = pd.date_range(PEAK - pd.Timedelta(days=4), PEAK, freq="12h")
    t = t[t != PEAK - pd.Timedelta(days=2) + pd.Timedelta(hours=12)]   # drop one 12Z
    d = M.era5_daily_utc(_ramp(t))
    want = pd.DatetimeIndex([PEAK - pd.Timedelta(days=k) for k in (4, 3, 1)])
    assert list(pd.DatetimeIndex(d["time"].values)) == list(want)
    np.testing.assert_allclose(d.values[:, 0, 0], _at(want, 6))


def test_aires_prob_daily_estimator(monkeypatch):
    """Self-normalized weighted exceedance per UTC day; maps day 0 NaN."""
    rng = np.random.default_rng(0)
    F = rng.normal(0, 3, (5, 6, 3, 4)).astype("float32")
    w = rng.uniform(0.1, 2.0, 5)
    da = xr.DataArray(F, dims=("walker", "day", "lat", "lon"))
    monkeypatch.setattr(S, "aires_daily", lambda eid: (da, w))
    got = M.aires_prob_daily("eX")
    assert got.shape == (7, len(M.THRESHOLDS), 3, 4) and np.isnan(got[0]).all()
    wn = w / w.sum()
    for j, a in enumerate(M.THRESHOLDS):
        want = (wn[:, None, None, None] * (np.sign(a) * F >= np.sign(a) * a)).sum(0)
        np.testing.assert_allclose(got[1:, j], want, rtol=1e-6)


class _RampTruth(TR.Truth):
    name = "ramp"

    def __init__(self, idx):
        self._idx = idx

    def index(self):
        return self._idx

    def mask(self):
        return None

    def obs_al(self, eid, window="13f"):
        return float(self._idx.sel(time=TR.window_times(PEAK, window)).mean())


def test_utc_daily_truth_and_clim(monkeypatch):
    """Maps day k = UTC day peak-7+k on its (00Z, 12Z) pair; clim over the same pairs on
    the `analyze.clim_pool` days; the six days average to the 12f obs."""
    from acal import analyze as AN
    t = pd.date_range("2021-06-01", "2022-08-31T12", freq="12h")
    idx = 5.0 * np.sin(_ramp(t) / (RATE * 24 * 7.3) * 2 * np.pi)   # mixed exceedances
    eps = pd.DataFrame(dict(episode_id=["eX"], peak=[PEAK]))
    monkeypatch.setattr(M.aprep, "episodes", lambda: eps)
    base = xr.Dataset(dict(truth=(("case", "day", "lat", "lon"), np.zeros((1, 7, 3, 4))),
                           clim=(("case", "threshold", "lat", "lon"),
                                 np.zeros((1, len(M.THRESHOLDS), 3, 4)))),
                      coords=dict(case=["eX"], day=np.arange(7),
                                  threshold=list(M.THRESHOLDS), lat=LAT, lon=LON))
    truth, clim = M.utc_daily_truth(_RampTruth(idx), base)
    assert np.isnan(truth[0, 0]).all()
    days = S.daily_days(PEAK)
    want = [(float(idx.sel(time=D)[0, 0]) + float(idx.sel(time=D + pd.Timedelta(hours=12))[0, 0]))
            / 2 for D in days]
    np.testing.assert_allclose(truth[0, 1:, 0, 0], want, rtol=1e-5, atol=1e-6)
    d = M.era5_daily_utc(idx)
    pool = d.sel(time=AN.clim_pool(pd.Series(0.0, index=pd.DatetimeIndex(d["time"].values)),
                                   PEAK).index)
    pt = pd.DatetimeIndex(pool["time"].values)       # +/-30 d of 2021-07-20 and 2022-07-20,
    assert pool.sizes["time"] == 61 + 40 and PEAK - pd.Timedelta(days=365) in pt   # own +/-10 d out
    assert PEAK not in pt
    fr = []
    for j, a in enumerate(M.THRESHOLDS):
        want = float((np.sign(a) * pool.values[:, 0, 0] >= np.sign(a) * a).mean())
        assert clim[0, j, 0, 0] == pytest.approx(want, abs=1e-6)
        fr.append(want)
    assert 0.0 < min(fr) and max(fr) < 1.0


EID = "e02_c4_20210218"


@pytest.mark.skipif(not S.aires_daily_path(EID).exists(), reason="AI+RES daily cache not built")
def test_aires_daily_cache_averages_to_12f():
    """Real walkers: the six UTC-day fields average to the cached '12f' field."""
    F, w = S.aires_daily(EID)
    F12, w12 = S.aires_fields(EID, "12f")
    np.testing.assert_allclose(F.mean("day").values, F12.values, atol=2e-5)
    np.testing.assert_array_equal(w, w12)


_GEPS_DAILY = S.ANALYSIS / "era5" / "maps_daily_geps.nc"


@pytest.mark.skipif(not _GEPS_DAILY.exists(), reason="GEPS daily maps file not built")
def test_geps_daily_maps_truth_is_the_utc_pair():
    """Real file: GEPS day k truth = ERA5 (00Z, 12Z) of UTC day peak-7+k; prob_aires
    present; day 0 has no GEPS forecast."""
    with xr.open_dataset(_GEPS_DAILY) as d:
        d = d.sel(case=EID).load()
    assert "prob_aires" in d and bool(d["prob_raw"].sel(day=0).isnull().all())
    fr = TR.frames_on(TR.get_truth("era5"), EID, "12f")
    pk = TR.peak_of(EID)
    want = S.utc_pairs(fr, S.daily_days(pk)).values
    np.testing.assert_allclose(d["truth"].values[1:], want, atol=2e-4)
    assert bool(d["truth"].sel(day=0).isnull().all())
