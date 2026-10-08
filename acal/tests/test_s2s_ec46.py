"""EC46 source (acal/s2s_ec46.py): dates, requests, GRIB decode, regrid, cube, bias.

Everything is synthetic and offline: the GRIB2 files are written here with eccodes in the
layout ECDS serves (daily-mean 2t, PDT 11 for forecasts, PDT 61 with a model-version
date for reforecasts), so no ECDS token or network is needed.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acal import aprep
from acal import s2s_ec46 as E
from aires import cfs

# (EC46 init, reforecast reference date) per case, from the ECDS constraints lists
# (research/ec46_cases.csv, 2026-10-07). Changing the init rule must break this.
EXPECTED = dict(x.split(":") for x in """
e01:20201228/20201228 e02:20210128/20210128 e03:20210318/20210318 e04:20210520/20210520
e05:20210920/20210920 e06:20211115/20211115 e07:20211125/20211125 e08:20211206/20211206
e09:20220207/20220207 e10:20220221/20220221 e11:20220228/20220228 e12:20221031/20221031
e13:20221201/20221201 e14:20221212/20221212 e15:20221226/20221226 e16:20230112/20230112
e17:20230227/20230227 e18:20230915/20230914 e19:20231004/20231005 e20:20231013/20231012
e21:20231030/20231030 e22:20231119/20231120 e23:20231207/20231207 e24:20231229/20231228
e25:20240110/20240111 e26:20240121/20240122 e27:20240207/20240208 e28:20240226/20240226
e29:20240328/20240328 e30:20240910/20240909 e31:20241006/20241007 e32:20241130/20241129
e33:20241211/20241211 e34:20250104/20250103 e35:20250119/20250119 e36:20250202/20250201
e37:20250209/20250209 e38:20250224/20250223 e39:20250310/20250309 e40:20250909/20250909
e41:20251030/20251029 e42:20251207/20251207""".split())

# Types of the ECDS process inputs (process_fc.json / process_rf.json, 2026-10-07).
STRING = {"origin", "year", "month", "day", "time", "level_type", "forecast_type",
          "data_format"}
ARRAY = {"variable", "leadtime_hour", "area", "hyear", "hmonth", "hday"}

NATIVE_LAT = np.arange(54.0, 20.99, -1.5)          # ECDS area crop, N -> S
NATIVE_LON = np.arange(231.0, 297.01, 1.5)


def linear(lat, lon, member=0, day=0):
    """A field bilinear interpolation reproduces exactly: plane + per-message offset."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return 250.0 + 0.5 * la + 0.1 * (lo - 231.0) + member + 0.01 * day


def write_grib(path, msgs):
    """``msgs``: dicts with number, date, start (h), mvd (reforecast ref date) or None."""
    import eccodes as ec
    with open(path, "wb") as f:
        for m in msgs:
            h = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            ec.codes_set(h, "centre", "ecmf")
            ec.codes_set(h, "productDefinitionTemplateNumber", 61 if m.get("mvd") else 11)
            for k, v in dict(discipline=0, parameterCategory=0, parameterNumber=0,
                             typeOfFirstFixedSurface=103, scaleFactorOfFirstFixedSurface=0,
                             scaledValueOfFirstFixedSurface=2).items():
                ec.codes_set(h, k, v)
            ec.codes_set(h, "dataDate", int(m["date"]))
            ec.codes_set(h, "dataTime", 0)
            ec.codes_set(h, "typeOfEnsembleForecast", 1 if m["number"] == 0 else 3)
            ec.codes_set(h, "perturbationNumber", m["number"])
            ec.codes_set(h, "numberOfForecastsInEnsemble", 51)
            ec.codes_set(h, "typeOfStatisticalProcessing", m.get("stat", 0))
            ec.codes_set(h, "stepRange", f"{m['start']}-{m['start'] + 24}")
            if m.get("mvd"):
                mvd = int(m["mvd"])
                ec.codes_set(h, "YearOfModelVersion", mvd // 10000)
                ec.codes_set(h, "MonthOfModelVersion", mvd // 100 % 100)
                ec.codes_set(h, "DayOfModelVersion", mvd % 100)
            for k, v in dict(Ni=NATIVE_LON.size, Nj=NATIVE_LAT.size,
                             latitudeOfFirstGridPointInDegrees=54.0,
                             longitudeOfFirstGridPointInDegrees=231.0,
                             latitudeOfLastGridPointInDegrees=21.0,
                             longitudeOfLastGridPointInDegrees=297.0,
                             iDirectionIncrementInDegrees=1.5,
                             jDirectionIncrementInDegrees=1.5).items():
                ec.codes_set(h, k, v)
            vals = linear(NATIVE_LAT, NATIVE_LON, m["number"], m["start"] // 24)
            ec.codes_set_values(h, vals.ravel())
            ec.codes_write(h, f)
            ec.codes_release(h)
    return path


def plan_of(prefix):
    df = aprep.episodes()
    return E.case_plan(next(df[df.episode_id.str.startswith(prefix)].itertuples()))


# --------------------------------------------------------------------------- dates
def test_init_and_ref_dates_match_ecds_for_all_42_cases():
    pl = E.plans()
    assert len(pl) == 42
    for p in pl:
        init, ref = EXPECTED[p["episode_id"][:3]].split("/")
        assert f"{p['init']:%Y%m%d}" == init, p["episode_id"]
        assert f"{p['ref_date']:%Y%m%d}" == ref, p["episode_id"]
        assert 0 <= p["lag_d"] <= 3 and p["lead_days"] == 21 + p["lag_d"]
        assert p["n_members"] == (51 if p["init"] < pd.Timestamp("2023-06-27") else 101)
        assert p["hyears"] == list(range(p["ref_date"].year - 20, p["ref_date"].year))
        # the window days peak-6..peak-1 plus a margin day each side are fetched
        assert p["fc_days"][0] == 0 and p["fc_days"][-1] == p["lead_days"]
        assert p["rf_days"] == list(range(p["lead_days"] - 7, p["lead_days"] + 1))


def test_schedule_edges():
    assert not E.is_init_date("2023-06-27") and E.is_init_date("2023-06-28")
    assert E.is_init_date("2021-01-04") and not E.is_init_date("2021-01-05")   # Mon / Tue
    assert E.ec46_init("2021-01-06") == pd.Timestamp("2021-01-04")
    assert E.n_members_for("2023-06-26") == 51 and E.n_members_for("2023-06-28") == 101
    # odd-day reference dates from 2024-11-13; equidistant -> the earlier one
    assert E.ref_date("2024-11-30") == pd.Timestamp("2024-11-29")
    assert E.ref_date("2023-10-04") == pd.Timestamp("2023-10-05")             # Wed -> Thu
    assert E.hdate("2024-02-29", 2005) == pd.Timestamp("2005-02-28")


# --------------------------------------------------------------------------- requests
@pytest.mark.parametrize("kind", ["fc", "rf"])
def test_requests_follow_the_ecds_schema(kind):
    p = plan_of("e42")
    for ftype in E.FORECAST_TYPES:
        ds, req = E.request(kind, p, ftype)
        assert ds == ("s2s-forecasts" if kind == "fc" else "s2s-reforecasts")
        for k, v in req.items():
            if k in STRING:
                assert isinstance(v, str), k
            else:
                assert k in ARRAY and isinstance(v, list), k
        assert req["area"] == [54, -129, 21, -62]
        assert req["origin"] == "ecmwf" and req["variable"] == ["2_m_temperature"]
        assert req["forecast_type"] == ftype and req["time"] == "00:00"
        days = p["fc_days"] if kind == "fc" else p["rf_days"]
        assert req["leadtime_hour"] == [f"{24 * d}_{24 * d + 24}" for d in days]
        if kind == "rf":
            assert req["hyear"] == [str(y) for y in range(2005, 2025)]
            assert req["hmonth"] == ["12"] and req["hday"] == ["07"]
        else:
            assert (req["year"], req["month"], req["day"]) == ("2025", "12", "07")
    assert E.expected_messages("fc", p, "perturbed_forecast") == 100 * 22
    assert E.expected_messages("rf", p, "perturbed_forecast") == 10 * 8 * 20


def test_requests_csv(tmp_path):
    out = E.write_requests_csv(E.plans(), tmp_path / "requests.csv")
    df = pd.read_csv(out)
    assert len(df) == 42 and df.n_hyears.eq(20).all()
    assert set(df.n_members) == {51, 101} and df.lead_days.between(21, 24).all()


# --------------------------------------------------------------------------- token
def test_token_file_parsing_and_absence(tmp_path, monkeypatch):
    monkeypatch.delenv("ECDS_KEY", raising=False)
    rc = tmp_path / "ecdsapirc"
    with pytest.raises(E.TokenMissing) as ei:
        E.read_token(rc)
    assert "s2s-reforecasts" in str(ei.value)
    rc.write_text("url: https://ecds.ecmwf.int/api\nkey: abc-123:secret\n")
    assert E.read_token(rc) == ("https://ecds.ecmwf.int/api", "abc-123:secret")
    rc.write_text("url: https://ecds.ecmwf.int/api\n")
    monkeypatch.setenv("ECDS_KEY", "envkey")
    assert E.read_token(rc)[1] == "envkey"


def test_fetch_without_token_waits_and_touches_no_network(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ECDS_KEY", raising=False)
    monkeypatch.setattr(E, "TOKEN_FILE", tmp_path / "absent")
    monkeypatch.setattr(E, "REQUESTS_CSV", tmp_path / "requests.csv")
    monkeypatch.setattr(E, "RAW", tmp_path / "raw")
    assert E.main(["--stage", "fetch", "--case", "e02_c4_20210218"]) == 3
    assert "waiting_token" in capsys.readouterr().out
    assert (tmp_path / "requests.csv").exists() and not (tmp_path / "raw").exists()


def test_cache_is_keyed_on_the_request(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "RAW", tmp_path)
    p = plan_of("e02")
    f = E.raw_path("fc", p, "control_forecast")
    f.write_bytes(b"GRIB")
    assert not E.is_cached("fc", p, "control_forecast")              # no sidecar
    E._sidecar(f).write_text(json.dumps({"request": E.request("fc", p, "control_forecast")[1]}))
    assert E.is_cached("fc", p, "control_forecast")
    E._sidecar(f).write_text(json.dumps({"request": {"area": [1, 2, 3, 4]}}))
    assert not E.is_cached("fc", p, "control_forecast")


# --------------------------------------------------------------------------- decode
def test_decode_forecast_grib(tmp_path):
    msgs = [dict(number=n, date=20210128, start=24 * d) for n in range(3) for d in range(4)]
    df = E.read_grib(write_grib(tmp_path / "fc.grib", msgs))
    nat = E.native_cube(df)
    assert nat.dims == ("member", "time", "lat", "lon")
    assert list(nat.member.values) == [0, 1, 2]
    assert list(pd.DatetimeIndex(nat.time.values).strftime("%m-%d")) == \
        ["01-28", "01-29", "01-30", "01-31"]                      # day = init + startStep
    assert nat.lat.values[0] == 21.0 and nat.lat.values[-1] == 54.0
    assert nat.lon.values[0] == 231.0 and nat.lon.values[-1] == 297.0
    want = linear(nat.lat.values, nat.lon.values, member=2, day=3)
    assert np.allclose(nat.isel(member=2, time=3).values, want, atol=0.01)


def test_decode_refuses_instantaneous_and_holes(tmp_path):
    bad = [dict(number=0, date=20210128, start=0, stat=2)]           # max, not avg
    with pytest.raises(ValueError, match="avg"):
        E.read_grib(write_grib(tmp_path / "bad.grib", bad))
    holey = [dict(number=n, date=20210128, start=24 * d) for n in range(2) for d in range(2)]
    df = E.read_grib(write_grib(tmp_path / "h.grib", holey[:-1]))
    with pytest.raises(ValueError, match="incomplete"):
        E.native_cube(df)


def test_decode_reforecast_groups_by_hindcast_date(tmp_path):
    msgs = [dict(number=n, date=y * 10000 + 128, start=24 * d, mvd=20210128)
            for y in (2001, 2002) for n in range(2) for d in range(2)]
    df = E.read_grib(write_grib(tmp_path / "rf.grib", msgs))
    assert set(df.init) == {pd.Timestamp("2001-01-28"), pd.Timestamp("2002-01-28")}
    assert set(df.model_version) == {pd.Timestamp("2021-01-28")}


# --------------------------------------------------------------------------- regrid / cube
def test_regrid_reproduces_a_plane_and_checks_the_halo():
    lat, lon = cfs.target_grid()
    nat = xr.DataArray(linear(NATIVE_LAT, NATIVE_LON)[None, None],
                       dims=("member", "time", "lat", "lon"),
                       coords=dict(member=[0], time=[np.datetime64("2021-01-28")],
                                   lat=NATIVE_LAT, lon=NATIVE_LON)).sortby("lat")
    out = E.regrid(nat, lat, lon)
    assert out.shape == (1, 1, 105, 237) and out.dtype == np.float32
    assert np.allclose(out.values[0, 0], linear(lat, lon), atol=1e-4)
    with pytest.raises(ValueError, match="halo"):
        E.regrid(nat.sel(lat=slice(22, 54)), lat, lon)


def _cube(init="2021-01-28", peak="2021-02-18", n=3, days=None, offset=0.0):
    lat, lon = cfs.target_grid()
    init, peak = pd.Timestamp(init), pd.Timestamp(peak)
    days = days if days is not None else pd.date_range(init, peak, freq="D")
    spread = np.linspace(-0.2, 0.2, n)[:, None, None, None]      # member mean = offset
    v = 280.0 + offset + spread + np.zeros((n, len(days), lat.size, lon.size))
    da = xr.DataArray(v.astype("float32"), dims=("member", "time", "lat", "lon"),
                      coords=dict(member=np.arange(n), time=days.values, lat=lat, lon=lon))
    return E.make_cube(da, init=init, peak=peak, attrs=dict(archives="test", kind="t"))


def test_cube_contract_and_atomic_write(tmp_path):
    c = _cube()
    for k in ("source", "dataset_id", "url", "init", "peak", "lead_days", "time_kind",
              "step_h", "native_grid", "archives"):
        assert k in c.attrs, k
    assert c.attrs["time_kind"] == "daily_mean" and c.attrs["step_h"] == 24
    assert float(c.member_lead_days[0]) == 21.0
    assert pd.Timestamp(c.cycle.values[0]) == pd.Timestamp("2021-01-28")
    out = E.write_cube(c, tmp_path / "x.nc")
    assert [p.name for p in tmp_path.iterdir()] == ["x.nc"]
    with xr.open_dataset(out) as d:
        assert d["2m_temperature"].encoding.get("zlib") and d["2m_temperature"].dtype == "float32"
    with pytest.raises(ValueError, match="Celsius"):
        _cube(offset=-260.0)
    with pytest.raises(ValueError, match="window days"):
        _cube(days=pd.date_range("2021-01-28", "2021-02-14", freq="D"))


def test_build_case_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "RAW", tmp_path / "raw")
    monkeypatch.setattr(E, "ROOT", tmp_path)
    (tmp_path / "raw").mkdir()
    p = plan_of("e02")
    for ftype, nums in (("control_forecast", [0]), ("perturbed_forecast", [1, 2])):
        write_grib(E.raw_path("fc", p, ftype),
                   [dict(number=n, date=20210128, start=24 * d)
                    for n in nums for d in p["fc_days"]])
    out = E.build_case(p["episode_id"])
    with xr.open_dataset(out) as c:
        E.check_cube(c, p["peak"])
        assert c.sizes["member"] == 3 and c.attrs["members_expected"] == 51
        lat, lon = c.lat.values, c.lon.values
        got = c["2m_temperature"].sel(member=1, time="2021-02-17").values
        assert np.allclose(got, linear(lat, lon, member=1, day=20), atol=0.01)


# --------------------------------------------------------------------------- reduce / bias
def test_daily_anomaly_uses_the_interval_mean_clim(monkeypatch):
    from gencast_s2s import data as D
    lat, lon = cfs.target_grid()

    def clim_for(times):                     # clim = the hour of day
        t = pd.DatetimeIndex(times)
        return xr.DataArray(np.broadcast_to(t.hour.values[:, None, None].astype(float),
                                            (t.size, lat.size, lon.size)),
                            dims=("time", "lat", "lon"))
    monkeypatch.setattr(D, "clim_for", clim_for)
    c = _cube()
    days = E.window_days(pd.Timestamp("2021-02-18"))
    a = E.daily_anom(c, days)
    raw = c["2m_temperature"].sel(time=days.values).values
    assert a.dims == ("member", "day", "lat", "lon") and a.sizes["day"] == 7
    assert np.allclose(a.values, raw - 9.0)                 # mean(0, 6, 12, 18) = 9


def test_bias_recovers_a_known_offset_in_the_cfs_schema(tmp_path, monkeypatch):
    lat, lon = cfs.target_grid()
    cases = aprep.episodes().iloc[[1, 41]].reset_index(drop=True)
    monkeypatch.setattr(aprep, "episodes", lambda rung=None: cases.copy())
    for k in ("ROOT", "HIND_ROOT", "ERA5_ROOT"):
        monkeypatch.setattr(E, k, tmp_path / k.lower())
    monkeypatch.setattr(E, "BIAS_NC", tmp_path / "bias.nc")
    monkeypatch.setattr(E, "BIAS_CSV", tmp_path / "bias.csv")
    monkeypatch.setattr(E, "HIND_CSV", tmp_path / "hind.csv")
    monkeypatch.setattr(E, "MIN_YEARS", 2)
    monkeypatch.setattr(E, "interval_clim", lambda days: xr.DataArray(
        np.full((len(days), lat.size, lon.size), 280.0), dims=("time", "lat", "lon")))
    for r in cases.itertuples():
        p = E.case_plan(r)
        yrs = p["hyears"][-3:]
        e12 = np.zeros((3, lat.size, lon.size), np.float32)
        pair = np.zeros((3, 7, lat.size, lon.size), np.float32)
        peaks = []
        for j, y in enumerate(yrs):
            h = E.hdate(p["ref_date"], y)
            peak_y = h + pd.Timedelta(days=p["lead_days"])
            days = pd.date_range(peak_y - pd.Timedelta(days=7), peak_y, freq="D")
            c = _cube(init=h, peak=peak_y, n=11, days=days, offset=1.5 + 0.1 * j)
            E.write_cube(c, E.hind_path(r.episode_id, y))
            e12[j] = 0.1 * j
            pair[j] = 0.1 * j
            peaks.append(peak_y.to_datetime64())
        E.ERA5_ROOT.mkdir(parents=True, exist_ok=True)
        xr.Dataset(dict(t2m_anom_12f=(("year", "lat", "lon"), e12),
                        t2m_anom_pair=(("year", "day", "lat", "lon"), pair),
                        peak=(("year",), np.asarray(peaks))),
                   coords=dict(year=yrs, day=np.arange(7), lat=lat, lon=lon),
                   attrs=dict(daily_pair=E.DAILY_PAIR)).to_netcdf(E.era5_path(r.episode_id))
    E.bias()
    with xr.open_dataset(E.BIAS_NC) as b:
        assert set(b.data_vars) == {"bias_conus", "bias_sd", "n_years", "bias7", "bias_daily"}
        assert b.bias7.dims == ("case", "lat", "lon")
        assert b.bias_daily.dims == ("case", "day", "lat", "lon") and b.sizes["day"] == 7
        assert list(b.case.values) == list(cases.episode_id) and "family" in b.coords
        assert np.allclose(b.bias_conus, 1.5, atol=1e-4)
        assert np.allclose(b.bias7, 1.5, atol=1e-4) and np.allclose(b.bias_daily, 1.5, atol=1e-4)
        assert np.allclose(b.bias_sd, 0.0, atol=1e-4)
        assert (b.n_years == 3).all()
    tab = pd.read_csv(E.BIAS_CSV)
    assert list(tab.columns) == ["episode_id", "n_years", "years", "bias_conus", "bias_sd",
                                 "spread_hind"]
    assert len(pd.read_csv(E.HIND_CSV)) == 6


def test_era5_frames_cover_both_truth_conventions():
    """14 frames = the (00Z D, 12Z D) pair of UTC days peak-7 .. peak-1 (ruling C2); the
    last 12 are the '12f' window (ruling C9)."""
    t = E.era5_times("2021-02-18")
    assert t.size == 14 and t[0] == pd.Timestamp("2021-02-11T00")
    assert t[-1] == pd.Timestamp("2021-02-17T12")
    assert t[2] == pd.Timestamp("2021-02-12T00") and t[13] == pd.Timestamp("2021-02-17T12")
    S = pytest.importorskip("acal.s2sbase")
    from acal import truth as TR
    assert list(t[2:]) == list(TR.window_times("2021-02-18", "12f"))
    days = E.window_days("2021-02-18")
    assert list(t[0::2]) == list(days) and (t[1::2] - t[0::2] == pd.Timedelta(hours=12)).all()
    assert E.DAILY_PAIR == S.DAILY_PAIR


def test_era5_case_pairs_utc_days(tmp_path, monkeypatch):
    """t2m_anom_pair day k = mean(00Z, 12Z) of UTC day peak-7+k; days 1..6 average to the
    12f mean; the s2sbase pairing (`utc_pairs`) gives the same numbers (ruling C2)."""
    S = pytest.importorskip("acal.s2sbase")
    lat, lon = cfs.target_grid()
    monkeypatch.setattr(E, "ERA5_ROOT", tmp_path)
    peak = pd.Timestamp("2021-02-18")
    monkeypatch.setattr(E, "case_plan", lambda row: dict(
        hyears=[2020], ref_date=dt.date(2021, 1, 28), lead_days=21.0))
    monkeypatch.setattr(E, "case_row", lambda eid: None)
    monkeypatch.setattr(E, "hdate", lambda ref, y: peak - pd.Timedelta(days=21))
    seen = []

    def frames(times):
        seen.append(pd.DatetimeIndex(times))
        h = (pd.DatetimeIndex(times) - peak) / pd.Timedelta(hours=1)
        return np.broadcast_to(np.asarray(h, np.float32)[:, None, None],
                               (len(times), lat.size, lon.size)).copy()
    monkeypatch.setattr(E, "era5_frames", frames)
    E.era5_case("e99")
    with xr.open_dataset(E.era5_path("e99")) as d:
        pair = d["t2m_anom_pair"].sel(year=2020).values[:, 0, 0]
        e12 = float(d["t2m_anom_12f"].sel(year=2020).values[0, 0])
        assert d.attrs["daily_pair"] == S.DAILY_PAIR
    # UTC day peak-7+k: 00Z at -168+24k h, 12Z at -156+24k h -> mean -162+24k
    np.testing.assert_allclose(pair, -162.0 + 24.0 * np.arange(7))
    assert e12 == pytest.approx(pair[1:].mean())
    inst = xr.DataArray(frames(seen[0])[:, :2, :2], dims=("time", "lat", "lon"),
                        coords=dict(time=seen[0]))
    np.testing.assert_allclose(S.utc_pairs(inst, E.window_days(peak)).values[:, 0, 0], pair)
    # a file written before ruling C2 (no daily_pair attr) is not a cache hit
    with xr.open_dataset(E.era5_path("e99")) as d:
        old = d.load()
    old.attrs.pop("daily_pair")
    old.to_netcdf(tmp_path / "old.nc")
    assert not E._era5_current(tmp_path / "old.nc")
    assert E._era5_current(E.era5_path("e99"))


def test_parity_with_the_s2sbase_daily_reducer():
    """The bias must sit on the quantity s2sbase scores: same days, same clim."""
    S = pytest.importorskip("acal.s2sbase")
    peak = pd.Timestamp("2021-02-18")
    c = _cube()
    rng = np.random.default_rng(1)
    c["2m_temperature"].values += rng.normal(0, 2, c["2m_temperature"].shape).astype("float32")
    S.check_cube(c, peak)
    mine = E.daily_anom(c, E.window_days(peak)).isel(day=slice(1, None)).mean("day")
    theirs = S.reduce_daily(c, peak)
    assert np.allclose(mine.values, theirs.transpose(*mine.dims).values, atol=1e-4)
    assert S.time_kind(c) == "daily_mean"
    src = S.get("ec46")
    assert src.time_kind == "daily_mean" and src.bias == "reforecast"
