"""Contract tests for the CFSv2 operational baseline.

The failure modes here are all silent ones - the module talks to an external archive and
hands its output to the same reduction that scores the walkers, so anything it gets wrong
comes back as a plausible number on a figure rather than as an exception:

* the inventory URL. On NCEI the sidecar REPLACES ``.grb2`` with ``.inv``; appending
  instead gives a 404 on a cycle that is perfectly present, which is exactly what
  happened on the first run and would have been read as "the archive is missing
  2021-06-06 06Z". The AWS mirror's convention is the exact opposite (``.idx`` APPENDED),
  so both directions are pinned.
* the archive fallback. NCEI is missing all of 2024 and December 2025, which is every
  cycle two of the p90 cases need; those events read as "absent from the archive" and
  lost their baseline entirely until the mirror was tried. The fallback must fire on a
  404 and must NOT quietly become the primary.
* the lag ensemble's direction. ``trailing`` must never produce a member at a SHORTER
  lead than the walkers it is a baseline for - that is the whole reason it is the default.
* the inventory parser, on the multi-field layout (``wnd850`` writes a day of UGRD, then
  the same day of VGRD), where forecast hour is NOT monotone in record number and a
  naive "cut at the first record past the horizon" drops every v component.
* the 24 h filter, which on the 6-hourly CFS step is a 5-point trapezoid; the 3-point
  kernel ``aplots.daily`` uses for the 12-hourly walkers is not a diurnal null here and
  would leave the diurnal swing on the curve.

None of these need the network. The one test that would - that the archive still serves
what we think it serves - is the CLI, not a unit test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aires import cfs as CFS

INIT = pd.Timestamp("2021-06-07")


# --------------------------------------------------------------------------- #
# URLs
# --------------------------------------------------------------------------- #
def test_inventory_url_replaces_the_grib_suffix():
    """``...daily.inv``, never ``...daily.grb2.inv``.

    The second form is a clean 404 against a cycle that exists, so it reads as missing
    data rather than as a wrong URL. Pinned because it cost a debugging cycle already.
    """
    grb = CFS.cfs_url(INIT, "tmp2m")
    inv = CFS.cfs_url(INIT, "tmp2m", "inv")
    assert grb.endswith("/tmp2m.01.2021060700.daily.grb2")
    assert inv.endswith("/tmp2m.01.2021060700.daily.inv")
    assert not inv.endswith(".grb2.inv")


def test_url_path_carries_the_cycle_hour():
    """The archive nests YYYY/YYYYMM/YYYYMMDD/YYYYMMDDHH - the last level is the CYCLE.

    Dropping the hour still yields a valid-looking path for the 00Z cycle and a 404 for
    every other one, i.e. it would look like "only 00Z is archived".
    """
    u = CFS.cfs_url(pd.Timestamp("2022-12-01 18:00"), "tmp2m")
    assert "/time-series/2022/202212/20221201/2022120118/" in u
    assert "wnd850" not in u


# --------------------------------------------------------------------------- #
# The AWS mirror, tried when NCEI 404s a cycle
# --------------------------------------------------------------------------- #
def test_mirror_inventory_url_appends_rather_than_replaces():
    """The mirror's convention is the EXACT OPPOSITE of NCEI's, and both are live.

    NCEI replaces ``.grb2`` with ``.inv``; AWS appends ``.idx`` to the whole name. Apply
    either convention to the other archive and you get a 404 against an object that is
    present - the same misreading the NCEI test above exists to prevent, one archive over.
    """
    grb = CFS.aws_url(INIT, "tmp2m")
    inv = CFS.aws_url(INIT, "tmp2m", "inv")
    assert grb.endswith("/tmp2m.01.2021060700.daily.grb2")
    assert inv.endswith("/tmp2m.01.2021060700.daily.grb2.idx")
    assert not inv.endswith(".daily.idx")


def test_mirror_url_carries_the_cycle_hour_as_its_own_level():
    """``cfs.<YYYYMMDD>/<HH>/time_grib_01/`` - the date and the hour are SEPARATE levels,
    unlike NCEI where the last level is the full YYYYMMDDHH stamp."""
    u = CFS.aws_url(pd.Timestamp("2022-12-01 18:00"), "tmp2m")
    assert "/cfs.20221201/18/time_grib_01/" in u


def test_mirror_refuses_an_extension_it_does_not_know():
    with pytest.raises(ValueError):
        CFS.aws_url(INIT, "tmp2m", "grib")


def test_ncei_is_tried_before_the_mirror():
    """Order is the contract: NCEI is the archive of record and reaches back to 2011;
    the mirror is a rolling archive that only starts 2023-04-22."""
    assert [n for n, _ in CFS.ARCHIVES] == ["NCEI", "AWS"]


def _fake_archive(monkeypatch, present: str, payload: bytes = b"GRIB"):
    """Route ``_get`` to a single archive, 404-ing every other one.

    ``present`` is matched against the host, so the fake is keyed on the thing that
    actually distinguishes the two archives rather than on call order.
    """
    def fake(url, byte_range=None, missing_ok=False):
        if present not in url:
            if missing_ok:
                return None
            raise CFS.MissingCycle(url)
        return SINGLE_FIELD.encode() if url.endswith((".inv", ".idx")) else payload
    monkeypatch.setattr(CFS, "_get", fake)


def test_fetch_window_falls_back_to_the_mirror(tmp_path, monkeypatch, capsys):
    """NCEI's 2024 and Dec-2025 holes are the reason this module has a second archive.

    Without the fallback those cycles read as "absent from the archive" and the event
    silently loses its baseline - which is what p90_20240802 and p90_20251224 did.
    """
    _fake_archive(monkeypatch, "amazonaws.com")
    dest = tmp_path / "tmp2m.grb2"
    assert CFS.fetch_window(pd.Timestamp("2024-07-11 06:00"), "tmp2m", 24, dest) == "AWS"
    assert dest.read_bytes() == b"GRIB"
    assert "absent from NCEI, served by AWS" in capsys.readouterr().out


def test_fetch_window_stays_on_ncei_when_it_has_the_cycle(tmp_path, monkeypatch, capsys):
    """The fallback must not become the default: NCEI serving normally is silent, and
    nothing goes to the mirror."""
    _fake_archive(monkeypatch, "ncei.noaa.gov")
    dest = tmp_path / "tmp2m.grb2"
    assert CFS.fetch_window(INIT, "tmp2m", 24, dest) == "NCEI"
    assert "served by" not in capsys.readouterr().out


def test_a_cycle_in_no_archive_is_still_a_missing_cycle(tmp_path, monkeypatch):
    """``build`` degrades the lag ensemble to n-1 members on ``MissingCycle``. Adding a
    second archive must not turn a genuinely absent cycle into some other exception that
    kills the whole event instead."""
    _fake_archive(monkeypatch, "no-such-host")
    with pytest.raises(CFS.MissingCycle) as e:
        CFS.fetch_window(INIT, "tmp2m", 24, tmp_path / "x.grb2")
    assert "NCEI" in str(e.value) and "AWS" in str(e.value)


# --------------------------------------------------------------------------- #
# The lag ensemble
# --------------------------------------------------------------------------- #
def test_trailing_cycles_are_never_at_a_shorter_lead():
    """Every trailing member's lead >= the walkers', with the last one exactly equal.

    This is the property that makes the baseline unimpeachable: CFS is never handed a
    forecast closer to the event than the thing it is compared against.
    """
    cycles = CFS.cycles_for(INIT, 4, "trailing")
    assert cycles[-1] == INIT
    assert cycles == sorted(cycles)
    assert all(c <= INIT for c in cycles)
    assert (INIT - cycles[0]) == pd.Timedelta(hours=18)


def test_sameday_cycles_run_the_other_way():
    cycles = CFS.cycles_for(INIT, 4, "sameday")
    assert cycles[0] == INIT
    assert all(c >= INIT for c in cycles)


def test_off_cycle_init_is_refused():
    """CFSv2 runs at 00/06/12/18Z. An init at 03Z has no cycle and must not be rounded."""
    with pytest.raises(SystemExit):
        CFS.cycles_for(pd.Timestamp("2021-06-07 03:00"), 4)


def test_unknown_lag_mode_is_refused():
    with pytest.raises(SystemExit):
        CFS.cycles_for(INIT, 4, "centred")


# --------------------------------------------------------------------------- #
# The inventory parser
# --------------------------------------------------------------------------- #
SINGLE_FIELD = """1:0:d=2021060700:TMP:2 m above ground:6 hour fcst:
2:78723:d=2021060700:TMP:2 m above ground:12 hour fcst:
3:157561:d=2021060700:TMP:2 m above ground:18 hour fcst:
4:236387:d=2021060700:TMP:2 m above ground:24 hour fcst:
"""

# The real wnd850 layout: a day of UGRD, then the SAME day of VGRD, then the next day.
MULTI_FIELD = """1:0:d=2021060700:UGRD:850 mb:6 hour fcst:
2:57406:d=2021060700:UGRD:850 mb:12 hour fcst:
3:114307:d=2021060700:UGRD:850 mb:18 hour fcst:
4:171223:d=2021060700:UGRD:850 mb:24 hour fcst:
5:228194:d=2021060700:VGRD:850 mb:6 hour fcst:
6:285682:d=2021060700:VGRD:850 mb:12 hour fcst:
7:342605:d=2021060700:VGRD:850 mb:18 hour fcst:
8:399530:d=2021060700:VGRD:850 mb:24 hour fcst:
9:456500:d=2021060700:UGRD:850 mb:30 hour fcst:
"""


def test_parse_inv_reads_record_offset_and_hour():
    inv = CFS.parse_inv(SINGLE_FIELD)
    assert inv == [(1, 0, 6), (2, 78723, 12), (3, 157561, 18), (4, 236387, 24)]


def test_parse_inv_rejects_something_that_is_not_an_inventory():
    with pytest.raises(SystemExit):
        CFS.parse_inv("<html><body>404 Not Found</body></html>")


def test_forecast_hour_is_not_monotone_in_record_number():
    """The premise of ``fetch_window``'s "take the LARGEST matching record" rule.

    If this ever became monotone the rule would still be correct, but the comment
    explaining it would be wrong - and the naive rule it warns against would start
    passing for the wrong reason.
    """
    hours = [fh for _r, _o, fh in CFS.parse_inv(MULTI_FIELD)]
    assert hours != sorted(hours)


def test_a_24h_cut_keeps_both_wind_components():
    """Selecting on hour and taking the max record must not truncate at the first UGRD run.

    Cutting at the first record whose hour EXCEEDS 24 would stop at record 4 and take
    every u with no v at all - and ``grib_to_cube`` would then build a cube with a u
    component and no v, from which ``sqrt(u^2 + v^2)`` is simply missing.
    """
    inv = CFS.parse_inv(MULTI_FIELD)
    wanted = [i for i, (_r, _o, fh) in enumerate(inv) if fh <= 24]
    last = max(wanted)
    assert inv[last][0] == 8                       # the last VGRD of day 1, not the 4th u
    kept = {line.split(":")[3] for line in MULTI_FIELD.splitlines()[:last + 1]}
    assert kept == {"UGRD", "VGRD"}


# --------------------------------------------------------------------------- #
# The 24 h filter
# --------------------------------------------------------------------------- #
def _diurnal(step_h: float, days: float = 8.0, amp: float = 4.0, slope: float = 0.0):
    lead = np.arange(0.0, days + 1e-9, step_h / 24.0)
    return lead, amp * np.sin(2 * np.pi * lead) + slope * lead


def test_daily_mean_kills_the_diurnal_cycle_at_a_6h_step():
    """A pure 24 h sine must come out flat. This is what the 3-point kernel would NOT do.

    The diurnal swing on a box-mean T2m anomaly is several kelvin, i.e. comparable to the
    signal the trajectory figure is about; leaving it on makes four CFS curves look like
    noise against 64 smooth walker curves.
    """
    lead, y = _diurnal(6.0)
    _, sm = CFS.daily_mean(lead, y)
    # Everywhere, ENDS INCLUDED. The last point is the one sitting on the event peak,
    # where A_L is read; plain linear end padding leaves a quarter of the amplitude there.
    assert np.abs(sm).max() < 1e-9
    assert abs(sm[0]) < 1e-9 and abs(sm[-1]) < 1e-9


def test_the_padding_reduces_to_the_walker_rule_at_a_12h_step():
    """``aplots.daily`` pads with ``y[0] + y[1] - y[2]``. That is this rule at P = 2.

    Pinned so the two functions stay one rule at two steps: if this ever diverges, the
    walkers and the CFS baseline are being filtered by different filters and the figure
    is comparing curves that were not treated alike.
    """
    lead, y = _diurnal(12.0, days=10.0, amp=3.0, slope=0.4)
    _, mine = CFS.daily_mean(lead, y)
    from aires.aplots import daily
    _, theirs = daily(lead, y)
    assert np.allclose(mine, theirs, atol=1e-12)


def test_the_walker_3point_kernel_would_not_kill_it_at_6h():
    """Why this module does not just call ``aplots.daily``.

    ``daily``'s ``(1, 2, 1)/4`` is an exact null at the 24 h period for a 12-HOURLY
    series. At a 6 h step the same kernel spans 12 h, not 24, and leaves most of the
    swing behind.
    """
    _lead, y = _diurnal(6.0)
    three = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
    assert np.abs(three).max() > 1.0


def test_daily_mean_kills_a_diurnal_cycle_riding_on_a_trend():
    """The real case: a heat dome building under a diurnal swing.

    Nulling the diurnal and reproducing the trend are separate properties, and the end
    padding has to deliver both AT ONCE - which is why it is "one period back, shifted by
    one period of trend" rather than either a mirror or a straight extrapolation.
    """
    lead, y = _diurnal(6.0, days=8.0, amp=4.0, slope=0.9)
    _, sm = CFS.daily_mean(lead, y)
    assert np.allclose(sm, 0.9 * lead, atol=1e-9)


def test_daily_mean_is_exact_on_a_linear_trend_including_the_ends():
    """The endpoint is where ``A_L`` is read, so it may not be pulled back down the trend.

    ``aplots.daily`` documents the same requirement for the walkers; a mirrored pad would
    fail here, which is the point of testing the ends rather than the interior.
    """
    lead = np.arange(0.0, 8.0 + 1e-9, 0.25)
    y = 2.0 + 0.75 * lead
    out_lead, sm = CFS.daily_mean(lead, y)
    assert np.allclose(out_lead, lead)
    assert np.allclose(sm, y, atol=1e-9)


def test_daily_mean_is_centred_on_its_own_timestamp():
    """No half-step slide. The bug this pins cost the walker figure a quarter-day once.

    A filter that returned pair means on their midpoints would shift the whole curve and
    make every CFS member appear to reach the peak before it does. Tested as the SYMMETRY
    of the impulse response rather than as the location of its maximum: the trapezoid has
    three tied interior weights, so an argmax would report the first of the tie and fail a
    perfectly centred filter.
    """
    lead = np.arange(0.0, 8.0 + 1e-9, 0.25)
    i0 = int(np.argmin(np.abs(lead - 4.0)))
    y = np.zeros_like(lead)
    y[i0] = 1.0
    out_lead, sm = CFS.daily_mean(lead, y)
    assert out_lead.size == sm.size == lead.size
    for j in range(1, 5):
        assert np.isclose(sm[i0 - j], sm[i0 + j]), j
    assert sm[i0] > 0
    assert np.isclose(sm.sum(), 1.0)               # the filter conserves the total


def test_daily_mean_preserves_the_member_axis():
    """It is applied to a ``(member, time)`` array, not to one curve at a time."""
    lead, y = _diurnal(6.0)
    stack = np.stack([y, y + 3.0, y - 3.0])
    out_lead, sm = CFS.daily_mean(lead, stack)
    assert sm.shape == stack.shape
    assert np.allclose(out_lead, lead)
    assert np.allclose(sm[1] - sm[0], 3.0)
    assert np.allclose(sm[2] - sm[0], -3.0)


def test_daily_mean_passes_a_series_too_short_to_filter_through_unchanged():
    lead = np.array([0.0, 0.25])
    y = np.array([1.0, 2.0])
    out_lead, sm = CFS.daily_mean(lead, y)
    assert np.allclose(sm, y) and np.allclose(out_lead, lead)


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #
def test_every_metric_this_module_claims_maps_to_a_real_grib_field():
    """``METRIC_VAR`` and ``GRIB_TO_GENCAST`` have to agree on what a cube will contain."""
    names = {name for name, _lev in CFS.GRIB_TO_GENCAST.values()}
    assert "2m_temperature" in names
    assert {"u_component_of_wind", "v_component_of_wind"} <= names
    for metric, var in CFS.METRIC_VAR.items():
        assert var in ("tmp2m", "wnd850"), metric


def test_precip_metrics_are_refused_rather_than_guessed():
    """CFS ships a RATE; the metric wants a 12 h accumulation. Refuse, do not convert."""
    assert "tp_total" not in CFS.METRIC_VAR
    assert "tp_max12h" not in CFS.METRIC_VAR


def test_all_events_includes_the_pilot_event():
    """``fevents.RES_EVENTS`` does not contain PNW_HeatDome_2021 - it predates that
    registry - so an ``--all`` built from ``res_events()`` alone skips the flagship
    event's figure without saying anything."""
    events = CFS.all_events()
    assert "PNW_HeatDome_2021" in events
    assert len(events) == len(set(events))


# --------------------------------------------------------------------------- #
# The figure must survive the baseline being absent
# --------------------------------------------------------------------------- #
def test_load_returns_none_for_an_event_with_no_cube(tmp_path, monkeypatch):
    """``aplots`` treats ``None`` as "draw without the CFS curve"."""
    from aires import aconfig as A
    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path)
    assert CFS.load("PNW_HeatDome_2021") is None


def test_aplots_drops_the_curve_instead_of_failing(tmp_path, monkeypatch, capsys):
    """A figure must never fail because the optional baseline is missing or unbuildable.

    Both routes are covered: an event with no cube, and an event name that is not in the
    registry at all - which is how ``test_aplots`` drives these functions with synthetic
    populations, and which raises ``SystemExit`` out of ``fevents.event``.
    """
    from aires import aconfig as A
    from aires import aplots as AP
    from aires import run_aires as R
    from fcn3 import fevents as F

    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path)
    ctx = R.Ctx(ev=F.EVENTS["PNW_HeatDome_2021"], tag="unit", n_walkers=4, members=6,
                C=(0.0,), leads=(3.0,), horizon_days=6.0, dmc_seed=0, base_seed=0)
    assert AP._cfs(ctx) is None
    assert "no CFSv2 cube cached" in capsys.readouterr().out

    bogus = F.Event("not_an_event", "2021-06-28", "t2m_anom", "heat", "x", "aires")
    ctx2 = R.Ctx(ev=bogus, tag="unit", n_walkers=4, members=6, C=(0.0,), leads=(3.0,),
                 horizon_days=6.0, dmc_seed=0, base_seed=0)
    assert AP._cfs(ctx2) is None
    assert "unavailable" in capsys.readouterr().out
