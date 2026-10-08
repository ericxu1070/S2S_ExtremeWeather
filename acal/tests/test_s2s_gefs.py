"""Tests for the GEFSv12 baseline source (acal/s2s_gefs.py): idx parsing, step selection,
and the canonical cube contract on a synthetic GRIB2 written with eccodes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from acal import s2s_gefs as G

IDX = """\
1:0:d=2021012800:HGT:10 mb:504 hour fcst:ENS=+1
62:11477372:d=2021012800:ICETK:surface:504 hour fcst:ENS=+1
63:11506477:d=2021012800:TMP:2 m above ground:504 hour fcst:ENS=+1
64:11745641:d=2021012800:RH:2 m above ground:504 hour fcst:ENS=+1
65:11960875:d=2021012800:TMAX:2 m above ground:498-504 hour max fcst:ENS=+1
66:12098319:d=2021012800:TMIN:2 m above ground:498-504 hour min fcst:ENS=+1
"""

E02 = "e02_c4_20210218"          # peak 2021-02-18, AI+RES init 2021-01-28


# --------------------------------------------------------------------------- #
# idx parsing
# --------------------------------------------------------------------------- #
def test_idx_range_is_the_instant_tmp2m_record():
    start, end = G.parse_idx(IDX, 504)
    assert (start, end) == (11506477, 11745641 - 1)


def test_idx_never_matches_tmax_or_another_hour():
    # TMAX/TMIN share "2 m above ground" but are "498-504 hour max fcst"
    tail = "65:100:d=2021012800:TMAX:2 m above ground:498-504 hour max fcst:ENS=+1\n"
    with pytest.raises(KeyError):
        G.parse_idx(tail, 504)
    with pytest.raises(KeyError):
        G.parse_idx(IDX, 498)          # the file is f504; asking for 498 must not match


def test_idx_last_record_is_open_ended():
    txt = "1:0:d=2021012800:HGT:10 mb:504 hour fcst:ENS=+1\n" \
          "2:500:d=2021012800:TMP:2 m above ground:504 hour fcst:ENS=+1\n"
    assert G.parse_idx(txt, 504) == (500, None)


def test_idx_rejects_non_increasing_offsets():
    txt = "1:900:d=2021012800:TMP:2 m above ground:504 hour fcst:ENS=+1\n" \
          "2:500:d=2021012800:RH:2 m above ground:504 hour fcst:ENS=+1\n"
    with pytest.raises(ValueError):
        G.parse_idx(txt, 504)


# --------------------------------------------------------------------------- #
# dates, steps, urls
# --------------------------------------------------------------------------- #
def test_case_dates_and_steps_for_e02():
    init, peak = G.case_dates(G.case_row(E02))
    assert init == pd.Timestamp("2021-01-28") and peak == pd.Timestamp("2021-02-18")
    fh = G.fhours(init, peak)
    assert fh[0] == 336 and fh[-1] == 504 and len(fh) == 29
    assert np.all(np.diff(fh) == 6)
    # the contract window [peak-6d 00Z, peak 00Z] is inside, at 25 native frames
    assert sum(360 <= f <= 504 for f in fh) == 25
    # the 13 00Z/12Z frames the board reduces on
    assert [f for f in fh if 360 <= f and f % 12 == 0] == list(range(360, 505, 12))


def test_fhours_refuses_non_00z_and_beyond_35_days():
    with pytest.raises(ValueError):
        G.fhours(pd.Timestamp("2021-01-28 06:00"), pd.Timestamp("2021-02-18 06:00"))
    with pytest.raises(ValueError):
        G.fhours(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-18"))


def test_msg_url_layout():
    u = G.msg_url(pd.Timestamp("2021-01-28"), "p07", 342)
    assert u == ("https://noaa-gefs-pds.s3.amazonaws.com/gefs.20210128/00/atmos/pgrb2ap5/"
                 "gep07.t00z.pgrb2a.0p50.f342")
    assert G.MEMBERS[0] == "c00" and G.MEMBERS[-1] == "p30" and len(G.MEMBERS) == 31
    assert G.HIND_MEMBERS == G.MEMBERS[:11]


def test_hind_years_are_cfs_loyo_years():
    from acal import cfsbase as CB
    assert G.hind_jobs([E02]) == CB.hind_jobs([E02])
    years = [y for _e, y, _i, _p in G.hind_jobs([E02])]
    assert 2021 not in years and len(years) >= 4
    assert G.hind_case(E02, 2021) is None        # its own year is never a hindcast


# --------------------------------------------------------------------------- #
# cube contract on synthetic GRIB2
# --------------------------------------------------------------------------- #
LAT = np.arange(24.0, 50.01, 0.25)[:9]          # a small ascending target grid
LON = np.arange(235.0, 294.01, 0.25)[:13]


def truth_field(lat, lon, hours):
    """Linear in lat/lon so bilinear interpolation reproduces it exactly."""
    return 300.0 - 0.5 * lat[:, None] + 0.1 * (lon[None, :] - 235.0) + 0.01 * hours


def write_grib(path, init, fhrs, offset=0.0):
    """A GEFS-like member file: 0.5 deg, lat DESCENDING (as GEFS), one 2t msg per hour."""
    eccodes = pytest.importorskip("eccodes")
    nlat = np.arange(60.0, 10.0 - 0.01, -0.5)          # north to south, like GEFS
    nlon = np.arange(220.0, 310.0 + 0.01, 0.5)
    sample = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    with open(path, "wb") as fh:
        for f in fhrs:
            h = eccodes.codes_clone(sample)
            for k, v in (("Ni", nlon.size), ("Nj", nlat.size),
                         ("latitudeOfFirstGridPointInDegrees", nlat[0]),
                         ("latitudeOfLastGridPointInDegrees", nlat[-1]),
                         ("longitudeOfFirstGridPointInDegrees", nlon[0]),
                         ("longitudeOfLastGridPointInDegrees", nlon[-1]),
                         ("iDirectionIncrementInDegrees", 0.5),
                         ("jDirectionIncrementInDegrees", 0.5),
                         ("jScansPositively", 0)):
                eccodes.codes_set(h, k, v)
            eccodes.codes_set(h, "dataDate", int(f"{init:%Y%m%d}"))
            eccodes.codes_set(h, "dataTime", 0)
            eccodes.codes_set(h, "discipline", 0)
            eccodes.codes_set(h, "parameterCategory", 0)
            eccodes.codes_set(h, "parameterNumber", 0)
            eccodes.codes_set(h, "typeOfFirstFixedSurface", 103)
            eccodes.codes_set(h, "scaledValueOfFirstFixedSurface", 2)
            eccodes.codes_set(h, "scaleFactorOfFirstFixedSurface", 0)
            eccodes.codes_set(h, "forecastTime", int(f))
            eccodes.codes_set(h, "bitsPerValue", 24)
            vals = truth_field(nlat, nlon, f) + offset
            eccodes.codes_set_values(h, vals.ravel())
            eccodes.codes_write(h, fh)
            eccodes.codes_release(h)
    eccodes.codes_release(sample)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "RAW", tmp_path / "raw")
    monkeypatch.setattr(G.cfs, "target_grid", lambda: (LAT, LON))
    init, peak = pd.Timestamp("2021-01-28"), pd.Timestamp("2021-02-18")
    fh = G.fhours(init, peak)
    for k, mem in enumerate(("c00", "p01", "p03")):              # p02 is absent
        p = G.raw_path(init, mem)
        p.parent.mkdir(parents=True, exist_ok=True)
        write_grib(p, init, fh, offset=float(k))
    return init, peak, fh


def test_member_cube_flips_lat_and_uses_valid_time(synthetic):
    init, peak, fh = synthetic
    c = G.member_cube(G.raw_path(init, "c00"), peak, LAT, LON)
    t = pd.DatetimeIndex(c["time"].values)
    assert t[0] == peak - pd.Timedelta(days=7) and t[-1] == peak and t.size == 29
    assert np.all(np.diff(c["lat"].values) > 0)
    want = truth_field(LAT, LON, np.array(fh)[:, None, None])
    np.testing.assert_allclose(c["2m_temperature"].values, want, atol=2e-3)


def test_assemble_meets_the_cube_contract(synthetic, tmp_path):
    init, peak, _fh = synthetic
    cube = G.assemble(init, peak, ("c00", "p01", "p02", "p03"), dict(event=E02))
    assert cube["2m_temperature"].dims == ("member", "time", "lat", "lon")
    assert cube["2m_temperature"].dtype == np.float32
    assert list(cube["member"].values) == [0, 1, 3]              # perturbation numbers
    assert np.all(pd.DatetimeIndex(cube["cycle"].values) == init)
    np.testing.assert_allclose(cube["member_lead_days"].values, 21.0)
    for k in ("source", "dataset_id", "url", "init", "peak", "lead_days", "time_kind",
              "step_h", "native_grid", "archives"):
        assert k in cube.attrs, k
    assert cube.attrs["time_kind"] == "instant" and cube.attrs["step_h"] == 6
    assert cube.attrs["members_skipped"] == "p02"
    # member k carries offset k: members are not mixed up
    d = cube["2m_temperature"].mean(("time", "lat", "lon")).values
    np.testing.assert_allclose(d - d[0], [0.0, 1.0, 2.0], atol=1e-3)
    out = G.write_cube(cube, tmp_path / "out" / "x.nc")
    with xr.open_dataset(out) as back:
        assert back["2m_temperature"].dtype == np.float32
        assert back["2m_temperature"].encoding.get("zlib")
        assert dict(back.sizes) == dict(member=3, time=29, lat=LAT.size, lon=LON.size)
    assert not list((tmp_path / "out").glob("*.tmp.*"))


def test_frames_13_are_00z_12z_in_the_window(synthetic):
    init, peak, _fh = synthetic
    cube = G.assemble(init, peak, ("c00", "p01"), {})
    f13 = G.frames_13(cube, peak)
    t = pd.DatetimeIndex(f13["time"].values)
    assert t.size == 13 and t[0] == peak - pd.Timedelta(days=6) and t[-1] == peak
    assert set(t.hour) == {0, 12}


def test_assemble_refuses_celsius(synthetic, monkeypatch):
    init, peak, _fh = synthetic
    real = G.cfs.grib_to_cube

    def celsius(path, metric, lat, lon):
        c = real(path, metric, lat, lon)
        c["2m_temperature"] = c["2m_temperature"] - 273.15
        return c

    monkeypatch.setattr(G.cfs, "grib_to_cube", celsius)
    with pytest.raises(RuntimeError, match="Celsius"):
        G.assemble(init, peak, ("c00", "p01"), {})


def test_assemble_refuses_too_few_members(synthetic):
    init, peak, _fh = synthetic
    with pytest.raises(RuntimeError):
        G.assemble(init, peak, ("c00", "p02"), {})


def test_source_record():
    s = G.SOURCE
    assert s["name"] == "gefs" and s["time_kind"] == "instant" and s["bias"] == "loyo"
    assert s["n_members"] == 31 and s["native_deg"] == 0.5
    for k in ("label", "init_rule", "dataset_id", "url"):
        assert s[k]
