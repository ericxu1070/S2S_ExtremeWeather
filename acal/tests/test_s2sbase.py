"""Tests for the multi-model registry and window reducers (acal/s2sbase.py, acal/truth.py)."""
from __future__ import annotations

import json
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acal import analyze as AN
from acal import cfsbase as CB
from acal import s2sbase as S
from acal import truth as TR

LAT = np.array([30.0, 40.0, 50.0])
LON = np.array([250.0, 260.0, 270.0, 280.0])
PEAK = pd.Timestamp("2022-07-20")
AMP = 6.0                                     # K, diurnal half-range of the synthetic clim


def _clim(times) -> xr.DataArray:
    """Sinusoidal diurnal clim (warmest 21Z, the CONUS afternoon) on a slow seasonal ramp."""
    t = pd.DatetimeIndex(times)
    h = t.hour.values + t.minute.values / 60.0
    v = 290.0 + AMP * np.cos(2 * np.pi * (h - 21.0) / 24.0) + 0.01 * t.dayofyear.values
    base = v[:, None, None] + 0.1 * LAT[None, :, None] + 0.0 * LON[None, None, :]
    return xr.DataArray(base, dims=("time", "lat", "lon"),
                        coords=dict(time=t.values, lat=LAT, lon=LON))


@pytest.fixture
def synth(monkeypatch):
    """Point the reducers at a 3x4 grid and the synthetic clim (no files read)."""
    monkeypatch.setattr(S, "GRID", (LAT.size, LON.size))
    monkeypatch.setattr(S, "_clim_grid", lambda: (LAT, LON))
    monkeypatch.setattr(S, "clim_at", _clim)


def _anom(n_mem: int) -> np.ndarray:
    return np.linspace(-3.0, 3.0, n_mem)[:, None, None] + np.zeros((1, LAT.size, LON.size))


def instant_cube(n_mem: int = 5, step_h: int = 6, attrs=True) -> xr.Dataset:
    t = pd.date_range(PEAK - pd.Timedelta(days=8), PEAK + pd.Timedelta(days=1), freq=f"{step_h}h")
    T = _clim(t).values[None] + _anom(n_mem)[:, None]
    ds = xr.Dataset(dict(**{"2m_temperature": (("member", "time", "lat", "lon"),
                                               T.astype("float32"))}),
                    coords=dict(member=np.arange(n_mem), time=t.values, lat=LAT, lon=LON,
                                cycle=("member", [PEAK - pd.Timedelta(days=21)] * n_mem),
                                member_lead_days=("member", [21.0] * n_mem)))
    if attrs:
        ds.attrs.update(time_kind="instant", step_h=step_h)
    return ds


def daily_cube(n_mem: int = 5, attrs=True, stamp_h: int = 0) -> xr.Dataset:
    days = pd.date_range(PEAK - pd.Timedelta(days=8), PEAK, freq="D")
    # the TRUE daily mean of an hourly series = clim's daily mean + the anomaly
    hourly = [_clim(pd.date_range(d, periods=24, freq="h")).mean("time").values for d in days]
    T = np.stack(hourly)[None] + _anom(n_mem)[:, None]
    ds = xr.Dataset(dict(**{"2m_temperature": (("member", "time", "lat", "lon"),
                                               T.astype("float64"))}),
                    coords=dict(member=np.arange(n_mem),
                                time=(days + pd.Timedelta(hours=stamp_h)).values,
                                lat=LAT, lon=LON,
                                cycle=("member", [PEAK - pd.Timedelta(days=21)] * n_mem),
                                member_lead_days=("member", [21.0] * n_mem)))
    if attrs:
        ds.attrs.update(time_kind="daily_mean", step_h=24)
    return ds


# --------------------------------------------------------------------------- #
# Reducers
# --------------------------------------------------------------------------- #
def test_instant_reducer_recovers_anomaly(synth):
    cube = instant_cube()
    for w in ("13f", "12f"):
        f = S.reduce_instant(cube, PEAK, w)
        np.testing.assert_allclose(f.values, _anom(5), atol=2e-4)       # float32 cube
    # a 12-hourly cube gives the same 13f field (only 00/12Z frames are used)
    f12 = S.reduce_instant(instant_cube(step_h=12), PEAK, "13f")
    np.testing.assert_allclose(f12.values, S.reduce_instant(cube, PEAK, "13f").values, atol=1e-5)


def test_daily_reducer_interval_clim(synth):
    cube = daily_cube()
    f = S.reduce_daily(cube, PEAK)
    # the 4-synoptic-hour mean of a pure sinusoid equals its 24 h mean exactly
    np.testing.assert_allclose(f.values, _anom(5), atol=1e-9)
    pd_ = S.reduce_daily(cube, PEAK, per_day=True)
    assert pd_.sizes["time"] == 6
    assert pd.DatetimeIndex(pd_["time"].values)[0] == PEAK - pd.Timedelta(days=6)
    assert pd.DatetimeIndex(pd_["time"].values)[-1] == PEAK - pd.Timedelta(days=1)
    # the trap: subtracting the 00Z clim from a daily mean puts the diurnal cycle in
    days = S.daily_days(PEAK)
    naive = (cube["2m_temperature"].sel(time=days) - _clim(days)).mean("time")
    err = float((naive - f).mean())
    assert abs(err - (-AMP * np.cos(2 * np.pi * (0 - 21) / 24))) < 1e-6
    assert abs(err) > 4.0


def test_reduce_dispatch_and_refusals(synth):
    inst, day = instant_cube(), daily_cube()
    np.testing.assert_allclose(S.reduce(inst, PEAK).values,
                               S.reduce_instant(inst, PEAK, "13f").values)
    np.testing.assert_allclose(S.reduce(day, PEAK).values, S.reduce_daily(day, PEAK).values)
    with pytest.raises(ValueError, match="daily_mean cube"):
        S.reduce_instant(day, PEAK)
    with pytest.raises(ValueError, match="instantaneous cube"):
        S.reduce_daily(inst, PEAK)
    with pytest.raises(ValueError):
        S.reduce(inst, PEAK, "d6")
    with pytest.raises(ValueError):
        S.reduce(day, PEAK, "13f")
    # no attr: sub-daily step -> instant (the CFS cubes); daily step -> refuse to guess
    assert S.time_kind(instant_cube(attrs=False)) == "instant"
    with pytest.raises(ValueError, match="refusing to guess"):
        S.time_kind(daily_cube(attrs=False))
    with pytest.raises(ValueError, match="stamped 00Z"):
        S.reduce_daily(daily_cube(stamp_h=12), PEAK)
    short = inst.sel(time=slice(None, PEAK - pd.Timedelta(hours=12)))
    with pytest.raises(ValueError, match="lacks"):
        S.reduce_instant(short, PEAK, "13f")
    S.reduce_instant(short, PEAK, "12f")                    # 12f stops at peak-1d 12Z


def test_check_cube(synth):
    S.check_cube(instant_cube(), PEAK)
    S.check_cube(daily_cube(), PEAK)
    c = instant_cube()
    c["2m_temperature"] = c["2m_temperature"] - 273.15
    with pytest.raises(ValueError, match="Celsius"):
        S.check_cube(c)
    with pytest.raises(ValueError, match="dims"):
        S.check_cube(instant_cube().transpose("time", "member", "lat", "lon"))
    with pytest.raises(ValueError, match="cycle"):
        S.check_cube(instant_cube().drop_vars("cycle"))
    with pytest.raises(ValueError, match="grid"):
        S.check_cube(instant_cube().isel(lat=slice(0, 2)))
    with pytest.raises(ValueError, match="cover"):
        S.check_cube(instant_cube(), PEAK + pd.Timedelta(days=3))


def test_masked_area_mean():
    rng = np.random.default_rng(0)
    da = xr.DataArray(rng.normal(size=(2, LAT.size, LON.size)), dims=("member", "lat", "lon"),
                      coords=dict(lat=LAT, lon=LON))
    full = xr.DataArray(np.ones((LAT.size, LON.size), bool), dims=("lat", "lon"),
                        coords=dict(lat=LAT, lon=LON))
    assert np.array_equal(TR.area_mean(da, full).values, TR.area_mean(da).values)
    m = full.copy(data=np.zeros_like(full.values))
    m[0, 0] = True
    np.testing.assert_allclose(TR.area_mean(da, m).values, da.values[:, 0, 0])
    with pytest.raises(TypeError):
        TR.area_mean(da, full.astype(int))


def test_window_times():
    t13, t12 = TR.window_times(PEAK, "13f"), TR.window_times(PEAK, "12f")
    assert len(t13) == 13 and t13[0] == PEAK - pd.Timedelta(days=6) and t13[-1] == PEAK
    assert len(t12) == 12 and t12[-1] == PEAK - pd.Timedelta(hours=12)
    assert set(t13.hour) == {0, 12}
    with pytest.raises(ValueError):
        TR.window_times(PEAK + pd.Timedelta(hours=6))


# --------------------------------------------------------------------------- #
# Variants, floors, registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [16, 21, 31, 51, 101])
def test_n_floor_and_fixed_subset(n):
    assert CB.n_floor("raw_emp", n) == n and CB.n_floor("corr_gauss", n) == n
    assert CB.n_floor("sub_emp", n) == CB.N_SUBSET
    assert CB.n_floor("s16_emp", n) == 16
    al = np.arange(n, dtype=float)
    sub = CB.members(al, 0.0, "s16_emp")
    assert sub.size == 16 and np.unique(sub).size == 16 and np.all(np.diff(sub) > 0)
    np.testing.assert_array_equal(sub, CB.members(al, 9.9, "s16_emp"))   # raw: no bias
    np.testing.assert_array_equal(CB.fixed_subset(n), CB.fixed_subset(n))  # deterministic
    # log-ratio floor follows the variant's N
    assert CB.log_ratio(0.5, 0.0, CB.n_floor("raw_emp", n)) == pytest.approx(np.log(0.5 * (n + 1)))


def test_cfs_defaults_unchanged():
    assert CB.n_floor("raw_emp") == 16 and CB.n_floor("sub_emp") == 4
    assert S.get("cfs").variants() == CB.VARIANTS


def test_registry_variants(tmp_path, monkeypatch):
    cfs, c13 = S.get("cfs"), S.get("cfs13")
    assert (cfs.window, c13.window, c13.obs_window) == ("all", "13f", "13f")
    assert cfs.cube_path("x") == c13.cube_path("x") == CB.cube_path("x")
    assert cfs.json_path("x") == CB.json_path("x")
    assert c13.json_path("x") == S.S2S_ROOT / "cfs" / "x.json"
    src = S.Source(name="fake", label="F", native_deg=1.5, time_kind="daily_mean",
                   n_members=51, init_rule="-", bias="loyo", dataset_id="-", url="-")
    monkeypatch.setattr(S, "S2S_ROOT", tmp_path)
    assert src.window == "d6" and src.obs_window == "12f"
    assert src.variants(51) == ("raw_emp", "raw_gauss", "s16_emp")
    assert src.variants(16) == ("raw_emp", "raw_gauss")
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "bias.csv").write_text("episode_id,bias_conus\n")
    assert src.variants(101) == ("raw_emp", "raw_gauss", "corr_emp", "corr_gauss", "s16_emp")
    with pytest.raises(ValueError):
        S.Source(name="bad", label="", native_deg=1, time_kind="weekly", n_members=1,
                 init_rule="", bias="", dataset_id="", url="")
    with pytest.raises(KeyError):
        S.get("seas5")
    assert {"cfs", "cfs13"} <= set(S.available())


def test_nan_corr_for_missing_bias():
    row = dict(episode_id="e", family="heat", rung=2, peak="2022-07-20", obs=2.5, sign=1.0,
               al=list(np.linspace(0, 4, 20)), bias=np.nan)
    daily = pd.Series(np.linspace(-3, 3, 400),
                      index=pd.date_range("2021-01-01", periods=400, freq="D"))
    v = ("raw_emp", "corr_emp", "s16_emp")
    out = S._nan_corr(CB.score_cfs_case(row, daily, v), v, row["bias"])
    assert np.isnan(out["p_obs_corr_emp"]) and np.isnan(out["n_reach_corr_emp"])
    assert out["p_obs_raw_emp"] == pytest.approx(np.mean(np.asarray(row["al"]) >= 2.5))
    assert np.isfinite(out["p_obs_s16_emp"])


# --------------------------------------------------------------------------- #
# JSON schema (synthetic)
# --------------------------------------------------------------------------- #
class _FakeTruth(TR.Truth):
    name = "fake"

    def obs_al(self, eid, window="13f"):
        return {"13f": 2.5, "12f": 2.4}[window]


CFS_KEYS = {"episode_id", "family", "obs", "sign", "n_members", "cycles", "member_lead_days",
            "al", "al_mean", "al_std", "al_sub_mean", "n_reach", "n_reach_sub", "init_anom",
            "cycles_skipped", "archives"}
Row = namedtuple("Row", "episode_id peak init family a_l_conus")


@pytest.mark.parametrize("kind", ["instant", "daily_mean"])
def test_case_record_schema(synth, kind):
    src = S.Source(name="fake", label="F", native_deg=1.0, time_kind=kind, n_members=21,
                   init_rule="-", bias="loyo", dataset_id="-", url="-")
    cube = instant_cube(21) if kind == "instant" else daily_cube(21)
    row = Row("e99", PEAK, PEAK - pd.Timedelta(days=21), "heat", 2.5)
    rec = S.case_record(src, row, cube, _FakeTruth())
    assert CFS_KEYS <= set(rec)
    assert {"window", "time_kind", "obs_window"} <= set(rec)
    assert rec["time_kind"] == kind and rec["n_members"] == 21 == len(rec["al"])
    assert rec["window"] == ("13f" if kind == "instant" else "d6")
    assert rec["obs_window"] == (2.5 if kind == "instant" else 2.4)
    np.testing.assert_allclose(rec["al"], _anom(21)[:, 0, 0], atol=2e-4)
    json.dumps(rec, allow_nan=False)                         # strict JSON (NaN -> null)
    assert rec["al_sub_mean"] == pytest.approx(np.mean(np.asarray(rec["al"])[CB.fixed_subset(21)]))


# --------------------------------------------------------------------------- #
# On the real runs/ tree (skipped elsewhere)
# --------------------------------------------------------------------------- #
needs_runs = pytest.mark.skipif(not (CB.BUILD_CSV.exists() and AN.SCORE_OUT.exists()),
                                reason="runs/acal not here")


@needs_runs
def test_cfs_score_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ANALYSIS", tmp_path)
    S.score(S.get("cfs"), TR.get_truth("era5"))
    got = (tmp_path / "era5" / "cfs_scorecard.csv").read_bytes()
    assert got == CB.SCORE_CSV.read_bytes()


@needs_runs
def test_cfs_paired_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ANALYSIS", tmp_path)
    S.paired(S.get("cfs"), TR.get_truth("era5"))
    assert (tmp_path / "era5" / "cfs_paired.csv").read_bytes() == CB.PAIRED_CSV.read_bytes()


@needs_runs
def test_cfs13_json_and_reduction():
    src = S.get("cfs13")
    eid = "e02_c4_20210218"
    p = src.json_path(eid)
    if not p.exists():
        pytest.skip("cfs13 json not built")
    rec = json.loads(p.read_text())
    assert CFS_KEYS <= set(rec) and rec["window"] == "13f" and rec["time_kind"] == "instant"
    pub = json.loads(CB.json_path(eid).read_text())
    assert rec["cycles"] == pub["cycles"] and rec["obs"] == pub["obs"]
    # independent route: keep only the 00/12Z frames, then the published aindex reduction
    from aires import aindex as AI
    with xr.open_dataset(src.cube_path(eid)) as cube:
        t = pd.DatetimeIndex(cube["time"].values)
        c12 = cube.isel(time=np.flatnonzero(t.hour.isin([0, 12]))).load()
    peak = pd.Timestamp("2021-02-18")
    assert AI.check_window(c12, peak, "t2m_anom") == 13
    want = AI.area_mean(AI.field(c12, peak, "t2m_anom")).values
    np.testing.assert_allclose(rec["al"], want, atol=1e-5)


@needs_runs
def test_aires_windows_table():
    if not S.AIRES_WINDOWS_CSV.exists():
        pytest.skip("aires_al_windows.csv not built")
    tab = pd.read_csv(S.AIRES_WINDOWS_CSV)
    assert tab.eid.nunique() == 42 and len(tab) == 42 * 32
    assert {"eid", "walker", "weight", "al13", "al12"} <= set(tab.columns)
    _, cases = AN.load_all()
    c = cases[1]
    g = tab[tab.eid == c.episode_id].sort_values("walker")
    np.testing.assert_allclose(g.al13.values, c.al, atol=1e-6)
    assert np.allclose(g.weight_sn.sum(), 1.0)
