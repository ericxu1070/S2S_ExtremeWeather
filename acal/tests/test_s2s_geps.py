"""Tests for the ECCC GEPS (SubX) source (acal/s2s_geps.py): start rules, the daily-mean
stamping, the canonical cube contract, and the guards. Synthetic and offline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acal import s2s_geps as G

LAT = np.arange(24.0, 50.0001, 0.25)          # the 105 x 237 acal grid
LON = np.arange(235.0, 294.0001, 0.25)


def inv_of(*rows) -> pd.DataFrame:
    return pd.DataFrame([dict(model=m, start=pd.Timestamp(s)) for m, s in rows])


def weekly(model: str, first: str, last: str, freq: str = "W-THU") -> list[tuple[str, str]]:
    return [(model, str(d.date())) for d in pd.date_range(first, last, freq=freq)]


def synth_raw(start="2021-01-28", n_members=21, max_lead=31.5, value=None) -> xr.Dataset:
    """A SubX-like raw crop whose field is LINEAR in lead, lat and lon (bilinear is exact)."""
    lead = np.arange(0.5, max_lead + 0.01, 1.0)
    lat = np.arange(21.0, 53.01, 1.0)
    lon = np.arange(232.0, 297.01, 1.0)
    m = np.arange(n_members)
    if value is None:
        f = (280.0 + 0.5 * lead[None, :, None, None] + 0.1 * (lat[None, None, :, None] - 21)
             + 0.01 * (lon[None, None, None, :] - 232) + 0.001 * m[:, None, None, None])
    else:
        f = np.full((m.size, lead.size, lat.size, lon.size), value)
    f = np.broadcast_to(f, (m.size, lead.size, lat.size, lon.size)).astype("float32").copy()
    return xr.Dataset({"tas": (("member", "lead", "lat", "lon"), f)},
                      coords=dict(member=m, lead=lead, lat=lat, lon=lon),
                      attrs=dict(start=start))


def cube_e02(raw=None) -> xr.Dataset:
    return G.make_cube(synth_raw() if raw is None else raw, model="GEPS6",
                       start="2021-01-28", init="2021-01-28", peak="2021-02-18",
                       lat=LAT, lon=LON)


# --------------------------------------------------------------------------- #
# Source record and stamping
# --------------------------------------------------------------------------- #
def test_source_record_follows_the_contract():
    s = G.SOURCE
    assert set(s) >= {"name", "label", "native_deg", "time_kind", "n_members", "init_rule",
                      "bias", "dataset_id", "url"}
    assert s["name"] == "geps" and s["time_kind"] == "daily_mean" and s["bias"] == "loyo"
    assert s["n_members"] == 21 and s["native_deg"] == 1.0


def test_lead_of_day_is_utc_day_s_plus_k():
    # L = k + 0.5 is UTC day S + k; the 00Z start day itself is L = 0.5
    assert G.lead_of_day("2021-01-28", "2021-01-28") == 0.5
    assert G.lead_of_day("2021-01-28", "2021-02-12") == 15.5
    assert G.lead_of_day("2021-01-28", "2021-02-17 00:00") == 20.5


def test_window_days_are_peak_minus_6_to_peak_minus_1():
    w = G.window_days("2021-02-18")
    assert list(w) == list(pd.date_range("2021-02-12", "2021-02-17", freq="D"))
    c = G.cube_days("2021-02-18")
    assert c[0] == pd.Timestamp("2021-02-11") and c[-1] == pd.Timestamp("2021-02-18")
    assert set(w) <= set(c)


# --------------------------------------------------------------------------- #
# Start selection
# --------------------------------------------------------------------------- #
def test_pick_start_latest_on_or_before_init_newest_version_on_a_shared_date():
    inv = inv_of(("GEPS6", "2021-11-25"), ("GEPS6", "2021-12-02"), ("GEPS7", "2021-12-02"),
                 ("GEPS7", "2021-12-09"))
    s = G.pick_start("2021-12-05", inv)
    assert (s.model, s.start) == ("GEPS7", pd.Timestamp("2021-12-02"))
    s = G.pick_start("2021-12-09", inv)          # same-day start counts (<=)
    assert s.start == pd.Timestamp("2021-12-09")
    s = G.pick_start("2021-12-01", inv)
    assert (s.model, s.start) == ("GEPS6", pd.Timestamp("2021-11-25"))
    assert G.pick_start("2021-11-24", inv) is None


def test_pick_hind_start_nearest_then_earlier_then_none():
    inv = inv_of(("GEPS7", "2023-01-19"), ("GEPS7", "2023-01-26"))
    # 2023-01-24 is 5 d after the 19th and 2 d before the 26th -> the 26th
    assert G.pick_hind_start("2023-01-24", inv).start == pd.Timestamp("2023-01-26")
    # a tie (3.5 d is impossible on dates, so build one): 2 d each side -> the earlier start
    inv2 = inv_of(("GEPS8", "2025-03-03"), ("GEPS8", "2025-03-07"))
    assert G.pick_hind_start("2025-03-05", inv2).start == pd.Timestamp("2025-03-03")
    assert G.pick_hind_start("2023-03-01", inv) is None   # nothing within 7 d


def test_hind_years_bounds_and_feb29():
    assert G.hind_years("2021-01-19") == [2022, 2023, 2024, 2025, 2026]
    # 2026-10-29 lies past the truth record (PEAK_END 2026-08-31)
    assert G.hind_years("2025-10-29") == [2021, 2022, 2023, 2024]
    # 2021-01-01 lies before the ERA5 index can supply peak-6d
    assert 2021 not in G.hind_years("2022-01-01")
    assert G.shifted_peak("2024-02-29", 2023) == pd.Timestamp("2023-02-28")


def test_hind_plan_keeps_the_case_lead_within_a_few_days():
    inv = pd.DataFrame(
        [dict(model="GEPS6", start=d) for d in pd.date_range("2020-06-04", "2021-11-25", freq="W-THU")]
        + [dict(model="GEPS7", start=d) for d in pd.date_range("2021-12-02", "2024-06-06", freq="W-THU")]
        + [dict(model="GEPS8", start=d) for d in pd.date_range("2024-06-13", "2026-10-05", freq="D")
           if d.dayofweek in (0, 3)])
    row = pd.Series(dict(episode_id="e02_c4_20210218", peak="2021-02-18", init="2021-01-28"))
    plan = G.hind_plan(row, inv)
    assert [h["year"] for h in plan] == [2022, 2023, 2024, 2025, 2026]
    for h in plan:
        assert h["start"] is not None and abs(h["shift_d"]) <= 3
        assert h["target"] == h["peak"] - pd.Timedelta(days=21)   # case lag is 0 d
        assert (h["peak"] - h["start"]).days in range(18, 25)
    assert all(h["model"] == "GEPS8" for h in plan if h["year"] >= 2025)


# --------------------------------------------------------------------------- #
# Reading SubX by dimension name
# --------------------------------------------------------------------------- #
class FakeVar:
    def __init__(self, data, dims):
        self.data, self.dimensions = data, dims

    def __getitem__(self, idx):
        return np.ma.masked_greater(self.data[idx], 1e20)


def test_read_indexes_by_name_for_both_dim_orders_and_masks_fill():
    rng = np.random.default_rng(0)
    smlyx = rng.normal(280, 5, (3, 4, 5, 6, 7))          # S, M, L, Y, X
    smlyx[1, 2, 3, 4, 5] = 9.999e20
    msl = np.transpose(smlyx, (1, 0, 2, 3, 4))           # GEPS8: M, S, L, Y, X
    sel = {"S": 1, "Y": slice(1, 5), "X": slice(2, 7)}
    a1, d1 = G._read(FakeVar(smlyx, ("S", "M", "L", "Y", "X")), sel)
    a2, d2 = G._read(FakeVar(msl, ("M", "S", "L", "Y", "X")), sel)
    assert d1 == d2 == ["M", "L", "Y", "X"]
    np.testing.assert_array_equal(a1, a2)
    assert np.isnan(a1[2, 3, 3, 3]) and np.isfinite(a1).sum() == a1.size - 1


# --------------------------------------------------------------------------- #
# The canonical cube
# --------------------------------------------------------------------------- #
def test_cube_contract_and_daily_stamping():
    c = cube_e02()
    t = c["2m_temperature"]
    assert t.dims == ("member", "time", "lat", "lon") and t.dtype == np.float32
    assert c.sizes["lat"] == 105 and c.sizes["lon"] == 237 and c.sizes["member"] == 21
    days = pd.DatetimeIndex(c.time.values)
    assert (days == days.normalize()).all()                       # stamped 00Z
    assert days[0] == pd.Timestamp("2021-02-11") and days[-1] == pd.Timestamp("2021-02-18")
    assert np.all(c.cycle.values == np.datetime64("2021-01-28"))
    assert np.all(c.member_lead_days.values == 21.0)
    a = c.attrs
    assert a["time_kind"] == "daily_mean" and a["step_h"] == 24 and a["lead_days"] == 21.0
    for k in ("source", "dataset_id", "url", "init", "peak", "native_grid", "archives"):
        assert a[k]
    # the value stamped on day d is the raw lead L = (d - start) + 0.5, bilinear-exact
    d = pd.Timestamp("2021-02-12")
    want = (280.0 + 0.5 * G.lead_of_day("2021-01-28", d) + 0.1 * (LAT - 21)[:, None]
            + 0.01 * (LON - 232)[None, :] + 0.001 * 5)
    np.testing.assert_allclose(t.sel(time=d, member=5).values, want, atol=2e-4)


def test_incomplete_member_dropped_and_recorded_too_few_refused():
    raw = synth_raw()
    raw["tas"][3, 17, 10, 10] = np.nan                  # lead 17.5 = 2021-02-14: in the cube
    raw["tas"][4, 2, 10, 10] = np.nan                   # lead 2.5: outside the cube, harmless
    c = cube_e02(raw)
    assert c.sizes["member"] == 20 and 3 not in c.member.values and 4 in c.member.values
    assert c.attrs["members_dropped"] == "3"
    raw["tas"][:10, 18] = np.nan
    with pytest.raises(SystemExit, match="complete members"):
        cube_e02(raw)


def test_refuses_celsius_and_missing_leads():
    with pytest.raises(SystemExit, match="Celsius"):
        cube_e02(synth_raw(value=12.0))
    with pytest.raises(SystemExit, match="no lead"):
        cube_e02(synth_raw(max_lead=15.5))


def test_regrid_requires_a_halo():
    raw = synth_raw().sel(lat=slice(24, 50))
    with pytest.raises(SystemExit, match="halo"):
        cube_e02(raw)


def test_atomic_write_roundtrip(tmp_path):
    c = cube_e02()
    out = G._write_cube(c, tmp_path / "e02.nc")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["e02.nc"]   # no tmp left behind
    with xr.open_dataset(out) as r:
        assert r["2m_temperature"].encoding.get("zlib")
        np.testing.assert_array_equal(r["2m_temperature"].values, c["2m_temperature"].values)
        assert r.attrs["time_kind"] == "daily_mean"


# --------------------------------------------------------------------------- #
# Interval-mean climatology and the sanity reduction
# --------------------------------------------------------------------------- #
def fake_clim_for(times):
    t = pd.DatetimeIndex(times)
    base = xr.DataArray(np.zeros((LAT.size, LON.size)), dims=("lat", "lon"),
                        coords=dict(lat=LAT, lon=LON))
    hour = xr.DataArray(np.asarray(t.hour, float), dims="time")
    return (270.0 + hour + base).assign_coords(hour=("time", t.hour),
                                               dayofyear=("time", t.dayofyear))


def test_interval_clim_is_the_00_06_12_18_mean(monkeypatch):
    import gencast_s2s.data as D

    monkeypatch.setattr(D, "clim_for", fake_clim_for)
    c = G.interval_clim(G.window_days("2021-02-18"))
    assert c.dims == ("time", "lat", "lon")
    np.testing.assert_allclose(c.values, 270.0 + (0 + 6 + 12 + 18) / 4)


def test_member_al_against_interval_clim(monkeypatch):
    import gencast_s2s.data as D

    monkeypatch.setattr(D, "clim_for", fake_clim_for)
    c = cube_e02(synth_raw(value=270.0 + 9.0 + 1.5))     # clim interval mean is 279
    al = G.member_al(c, "2021-02-18")
    assert al.shape == (21,)
    np.testing.assert_allclose(al, 1.5, atol=1e-4)


def test_era5_al12_reads_the_december_tail_from_the_next_years_file(tmp_path, monkeypatch):
    """The 2021 index file starts 2020-12-26 and each later file repeats the December tail;
    a window crossing New Year must find those frames and must not double-count them."""
    idx = tmp_path / "index"
    idx.mkdir()
    lat, lon = LAT[:3], LON[:4]
    for y in (2021, 2022):
        t = pd.date_range(f"{y - 1}-12-26", f"{y}-12-31 12:00", freq="12h")
        v = np.where(t.year == y, float(y - 2019), 9.0)          # 2021->2, 2022->3
        if y == 2021:
            v = np.where(t.year == 2020, 1.0, v)                 # 2020 tail -> 1
        v = v[:, None, None]
        da = xr.DataArray(np.broadcast_to(v, (t.size, lat.size, lon.size)).astype("float32"),
                          dims=("time", "lat", "lon"), coords=dict(time=t, lat=lat, lon=lon))
        da.to_dataset(name="t2m_anom").to_netcdf(idx / f"era5_t2m_anom_12h_{y}.nc")
    monkeypatch.setattr(G.ccfg, "ACAL_ROOT", tmp_path)
    # peak 2021-01-03: frames 2020-12-28 00Z .. 2021-01-02 12Z = 8 December + 4 January
    assert G.era5_al12("2021-01-03") == pytest.approx((8 * 1.0 + 4 * 2.0) / 12)
    # peak 2022-01-03: the December 2021 frames sit in BOTH files (2.0 and 9.0); the
    # earlier file wins, as a concat + drop-duplicates keeps the first
    assert G.era5_al12("2022-01-03") == pytest.approx((8 * 2.0 + 4 * 3.0) / 12)
