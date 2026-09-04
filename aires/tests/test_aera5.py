"""Contract tests for the ERA5 truth series.

The curve this module builds is drawn on the same axes as 64 walkers, a CFS baseline and
a horizontal rule labelled with the same event's ``A_L``. So the things that can go wrong
are all about AGREEMENT with what is already on that figure, not about the fetch:

* the lead axis has to be the walkers' axis - anchored at the AI+RES init, 0 .. 21 d;
* the 7-day window has to be the one the published scalar was averaged over, 25 frames
  inclusive at both ends, or the curve and the rule are two different numbers;
* the smoothing has to be the figure's 24 h mean at a 6 h step, not the walkers' 3-point
  kernel, which spans only 12 h here;
* a missing cache has to read as "no line", not as an exception; and
* an ARCO read has to prove its epoch before it is believed.

No GPU, no network, no climatology file, no real cache on disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aires import aconfig as A
from aires import aera5 as E

EVENT = "PNW_HeatDome_2021"


# --------------------------------------------------------------------------- #
# The lead axis
# --------------------------------------------------------------------------- #
def test_the_lead_axis_is_the_trajectory_figures_axis():
    """0 .. horizon in 0.25 d steps, anchored at the init the walkers count lead from."""
    init, peak, times = E.window(EVENT)
    lead = E.lead_days(EVENT, times)
    assert times.size == int(A.RES_HORIZON_DAYS * 24 / E.STEP_H) + 1 == 85
    assert lead[0] == 0.0 and lead[-1] == pytest.approx(float(A.RES_HORIZON_DAYS))
    assert np.allclose(np.diff(lead), E.STEP_H / 24.0)
    assert times[0] == init and times[-1] == peak


def test_every_event_lands_on_the_same_axis():
    """A per-event horizon would put one panel's truth on a different x axis from its
    walkers, silently. ``window`` refuses that rather than returning it."""
    for ev in E.all_events():
        init, peak, times = E.window(ev)
        assert (peak - init) == pd.Timedelta(days=float(A.RES_HORIZON_DAYS))
        assert times.size == 85


# --------------------------------------------------------------------------- #
# The verification window - where the curve has to meet the rule
# --------------------------------------------------------------------------- #
def test_the_verification_window_is_25_frames_inclusive_at_both_ends():
    """``[peak - 6 d, peak]`` at 6 h, both endpoints kept - the window
    ``gencast_s2s.data.true_verif_anom`` averaged and ``aindex.check_window`` demands.
    One frame off at either end and the self-check compares two different 7-day means."""
    _init, peak, times = E.window(EVENT)
    m = E.verif_mask(EVENT, times)
    sel = times[m]
    assert sel.size == 25
    assert sel[0] == peak - pd.Timedelta(days=6) and sel[-1] == peak
    assert E.lead_days(EVENT, sel)[0] == pytest.approx(15.0)


def test_the_seven_day_mean_is_the_last_25_frames():
    """On a ramp the answer is analytic: the mean of ``lead`` over [15, 21] is 18."""
    _init, _peak, times = E.window(EVENT)
    lead, m = E.lead_days(EVENT, times), E.verif_mask(EVENT, times)
    assert lead[m].mean() == pytest.approx(18.0)


def test_check_a_l_accepts_the_published_scalar_and_refuses_anything_else():
    """The self-check is the whole reason to trust the curve, so it is tested in both
    directions: a series whose 7-day mean IS the published ``A_L`` passes, and one that
    is 10 K away raises instead of being cached."""
    stored, _tag = E.stored_observed(EVENT)
    if stored is None:
        pytest.skip("no reduced AI+RES run for this event on this box")
    _init, _peak, times = E.window(EVENT)
    a_l, got, _ = E.check_a_l(EVENT, np.full(times.size, stored), times)
    assert a_l == pytest.approx(stored) and got == pytest.approx(stored)
    with pytest.raises(SystemExit, match="7-day mean"):
        E.check_a_l(EVENT, E.lead_days(EVENT, times), times)


def test_a_shifted_window_would_be_caught():
    """The failure this pins: a series that stops one frame short of the peak. Its own
    7-day mean is then a different quantity, and ``verif_mask`` says so instead of
    quietly averaging 24 frames."""
    _init, _peak, times = E.window(EVENT)
    with pytest.raises(SystemExit, match="24 frames"):
        E.verif_mask(EVENT, times[:-1])


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
def _diurnal(step_h: float, days: float = 6.0, amp: float = 5.0, slope: float = 0.0):
    lead = np.arange(0.0, days + 1e-9, step_h / 24.0)
    return lead, amp * np.sin(2 * np.pi * lead) + slope * lead


def test_the_smoothing_is_the_figures_24h_mean_at_a_6h_step():
    """A pure diurnal cycle comes out flat everywhere, ENDS INCLUDED - the last sample is
    the one sitting on the event peak, where the curve is read as a number."""
    lead, y = _diurnal(E.STEP_H)
    sm = E.smooth(lead, y)
    assert np.abs(sm).max() < 1e-9
    assert abs(sm[0]) < 1e-9 and abs(sm[-1]) < 1e-9


def test_the_walkers_3point_kernel_is_not_used():
    """``aplots.daily`` is an exact null at the diurnal period for a 12-HOURLY series. At
    this module's 6 h step it spans 12 h, not 24, and leaves most of the swing on the
    curve - several kelvin on a box-mean T2m anomaly. This is why ``smooth`` routes to
    ``cfs.daily_mean``, and the test fails if someone swaps them back."""
    lead, y = _diurnal(E.STEP_H)
    from aires.aplots import daily
    assert np.abs(daily(lead, y)[1]).max() > 1.0
    assert np.abs(E.smooth(lead, y)).max() < 1e-9


def test_the_smoothing_is_cfs_daily_mean_and_preserves_a_trend():
    """Identical to the CFS baseline's filter, and exact on the ramp underneath the
    diurnal - the two curves share one figure and must be treated alike."""
    from aires.cfs import daily_mean
    lead, y = _diurnal(E.STEP_H, days=8.0, amp=4.0, slope=0.9)
    assert np.array_equal(E.smooth(lead, y), daily_mean(lead, y)[1])
    assert np.allclose(E.smooth(lead, y), 0.9 * lead, atol=1e-9)


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #
def test_load_returns_none_when_there_is_no_cache(monkeypatch, tmp_path):
    """The figure drops the black line and still renders; it never raises."""
    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path / "runs" / "aires")
    assert not E.cache_path(EVENT).exists()
    assert E.load(EVENT) is None


def test_the_cache_round_trips_the_series_and_its_labels(monkeypatch, tmp_path):
    """``load`` is the figure's whole view of this module, so the keys it promises are
    pinned here: ``box`` is the SMOOTHED array (the line), ``raw`` the unsmoothed one,
    ``lead`` their axis, and the box's NAME is ``box_name`` - not ``box``."""
    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path / "runs" / "aires")
    _init, _peak, times = E.window(EVENT)
    lead = E.lead_days(EVENT, times)
    raw = {"box": 3.0 + 0.2 * lead, "conus": 1.0 + 0.1 * lead}
    ds = E.cache_dataset(EVENT, raw, source="unit", a_l=6.6, stored=6.6, tag="unit")
    p = E.cache_path(EVENT)
    p.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(p)

    d = E.load(EVENT)
    assert set(d) >= {"lead", "box", "raw", "times", "event", "box_name", "a_l"}
    assert d["event"] == EVENT and d["box_name"] == "PNW" and d["a_l"] == pytest.approx(6.6)
    assert np.allclose(d["lead"], lead) and d["times"].dtype == np.dtype("datetime64[ns]")
    assert np.allclose(d["raw"], raw["box"])
    # A linear series is its own 24 h mean, ends included - so the smoothed array is the
    # raw one here, which is exactly what makes this a round trip and not a re-derivation.
    assert np.allclose(d["box"], raw["box"], atol=1e-9)
    assert d["box"].shape == d["raw"].shape == d["lead"].shape == d["times"].shape


# --------------------------------------------------------------------------- #
# The ARCO epoch guard
# --------------------------------------------------------------------------- #
class _FakeTimeAxis:
    """Just enough of a zarr array: ``units``, a length, and a stored value per index."""

    def __init__(self, units: str, n: int = 10 ** 7, offset: int = 0):
        self.attrs = {"units": units}
        self.shape = (n,)
        self._offset = offset

    def __getitem__(self, i):
        return np.int64(int(i) + self._offset)


def _root(axis):
    return {"time": axis}


def test_a_correct_arco_store_passes_the_epoch_guard():
    E._check_arco_epoch(_root(_FakeTimeAxis("hours since 1900-01-01 00:00:00")),
                        pd.Timestamp("2023-11-07"))


def test_the_1959_epoch_is_refused():
    """The quarantined bug: the retired ARCO store's epoch against the v3 store's index
    shifted every read back 59 years and raised nothing."""
    with pytest.raises(SystemExit, match="1900-01-01"):
        E._check_arco_epoch(_root(_FakeTimeAxis("hours since 1959-01-01 00:00:00")),
                            pd.Timestamp("2023-11-07"))


def test_an_index_that_decodes_to_the_wrong_time_is_refused():
    """A store whose time axis is not the identity on hours-since-1900 - the general form
    of the same failure, caught by asking the store what it actually holds."""
    axis = _FakeTimeAxis("hours since 1900-01-01 00:00:00", offset=24)
    with pytest.raises(SystemExit, match="decodes to"):
        E._check_arco_epoch(_root(axis), pd.Timestamp("2023-11-07"))


def test_a_time_past_the_end_of_the_store_is_missing_truth_not_a_crash():
    """An event ERA5 has not reached yet is a legitimate outcome: skip it and report it.
    ``MissingTruth`` is what the CLI catches; a fabricated or interpolated frame is not
    an option."""
    axis = _FakeTimeAxis("hours since 1900-01-01 00:00:00", n=10)
    with pytest.raises(E.MissingTruth):
        E._check_arco_epoch(_root(axis), pd.Timestamp("2025-12-24"))
