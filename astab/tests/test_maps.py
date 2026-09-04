"""Contract tests for the lead-by-lead T2m maps (``astab/maps.py``).

The maps exist to be COMPARED - this model against that one, this lead against that one -
so what can quietly ruin them is not the drawing, it is anything that makes two figures
mean different things: a color scale computed per figure, a column subsample that drops the
last lead, a truth field on a different grid that xarray silently aligns instead of
refusing. Each of those is pinned below.

The grid one is this repo's own recurring failure. ``CLAUDE.md``'s climatology trap was two
files with the same name and different grids, and it produced silently wrong anomalies
rather than an error. Subtracting ERA5 from a forecast is exactly that operation again.

No GPU, no model, no file in ``runs/``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from astab import maps as MP
from astab import sconfig as SC
from astab import schedule as SCH
from gencast_s2s import config as C

LAT = np.arange(24.0, 50.01, 0.25, dtype="float32")
LON = np.arange(235.0, 294.01, 0.25, dtype="float32")


def _series(times, base=290.0, sd=3.0, seed=0) -> xr.DataArray:
    """A (time, lat, lon) T2m field on the real CONUS model grid."""
    rng = np.random.default_rng(seed)
    a = base + sd * rng.standard_normal((len(times), LAT.size, LON.size))
    return xr.DataArray(a.astype("float32"), dims=("time", "lat", "lon"),
                        coords=dict(time=pd.DatetimeIndex(times), lat=LAT, lon=LON))


def _truth_on_disk(tmp_path, monkeypatch, event, times, **kw):
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    p = SC.ref_field_path(event)
    p.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"2m_temperature": _series(times, **kw)}).to_netcdf(p)
    return p


# --------------------------------------------------------------------------- #
# The grid: subtracted directly, never aligned
# --------------------------------------------------------------------------- #
def test_the_truth_cache_is_on_the_model_grid():
    """105 x 237 is what ``diag.nc`` and every FCN3 cube carry. ``refs`` crops ERA5 with
    the SAME ``config.LAT``/``LON`` slices, so no regridding ever happens."""
    assert LAT.size == 105 and LON.size == 237
    assert (C.LAT.start, C.LAT.stop) == (24, 50)
    assert (C.LON.start, C.LON.stop) == (235, 294)


def test_a_grid_mismatch_is_refused_not_aligned(tmp_path, monkeypatch):
    """CLAUDE.md's climatology trap, in the one operation that would repeat it. xarray
    would happily align two different grids to their intersection and return a difference
    field over whatever overlapped; that is a wrong figure, not an error."""
    ev = SC.STAB_EVENTS[0]
    ch = SCH.Chain(ev, 4)
    times = [ch.valid_at(l) for l in SCH.scoring_checkpoints(ch.horizon_days)]
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    p = SC.ref_field_path(ev)
    p.parent.mkdir(parents=True, exist_ok=True)
    coarse = _series(times).isel(lat=slice(0, 14), lon=slice(0, 30))     # the 2 deg file
    xr.Dataset({"2m_temperature": coarse}).to_netcdf(p)
    monkeypatch.setattr(MP, "forecast_t2m", lambda c, m: _series(times))
    with pytest.raises(SystemExit, match="never regridded"):
        MP.draw_chain_maps(ch, "gencast", force=True)


# --------------------------------------------------------------------------- #
# The scales are fixed, so two figures mean the same thing
# --------------------------------------------------------------------------- #
def test_the_difference_scale_is_a_constant_not_a_per_figure_number():
    """A scale computed per figure makes a 20 K error at 60 d look like a 2 K error at
    3 d, which inverts the finding the maps exist to show."""
    assert isinstance(MP.DIFF_MAX_K, float)
    assert MP.DIFF_MAX_K == 12.0


def test_the_absolute_scale_is_per_event_and_the_same_for_every_model(tmp_path,
                                                                     monkeypatch):
    """One scale per event - a June heat dome and a February freeze genuinely need
    different ranges - but the SAME one for every model and every lead of that event."""
    ev = SC.STAB_EVENTS[0]
    times = pd.date_range("2021-05-01", periods=12, freq="3D")
    _truth_on_disk(tmp_path, monkeypatch, ev, times, base=295.0, sd=4.0)
    a = MP.event_t2m_scale(ev)
    assert a == MP.event_t2m_scale(ev)                     # deterministic
    assert a[0] < a[1]


def test_off_scale_is_measured_against_what_era5_actually_does(tmp_path, monkeypatch):
    """The color scale clips 2% of ERA5's own points by construction, so a HEALTHY
    forecast leaves it on nearly every frame - flagging that put a red label on every
    panel of the first version. ``event_t2m_range`` is the honest reference and is
    strictly wider."""
    ev = SC.STAB_EVENTS[0]
    times = pd.date_range("2021-05-01", periods=20, freq="3D")
    _truth_on_disk(tmp_path, monkeypatch, ev, times, base=295.0, sd=5.0)
    lo, hi = MP.event_t2m_scale(ev)
    flo, fhi = MP.event_t2m_range(ev)
    assert flo < lo and fhi > hi


# --------------------------------------------------------------------------- #
# The columns
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("weeks", SC.STAB_WEEKS)
def test_columns_are_three_day_checkpoints_both_sides_have(weeks, tmp_path, monkeypatch):
    ch = SCH.Chain(SC.STAB_EVENTS[0], weeks)
    times = [ch.valid_at(l) for l in SCH.scoring_checkpoints(ch.horizon_days)]
    f = _series(times)
    cols, _ = MP._columns(ch, f, f, 0)
    for t in cols:
        lead = (pd.Timestamp(t) - ch.init).total_seconds() / 86400.0
        assert abs(lead % 3.0) < 1e-9 and lead > 0


def test_a_subsample_always_keeps_the_first_and_last_lead():
    """The last column is the whole point of the figure; a subsample that dropped it would
    end the story one column early."""
    ch = SCH.Chain(SC.STAB_EVENTS[0], 10)
    times = [ch.valid_at(l) for l in SCH.scoring_checkpoints(ch.horizon_days)]
    f = _series(times)
    allc, none = MP._columns(ch, f, f, 0)
    few, dropped = MP._columns(ch, f, f, 5)
    assert none == 0 and dropped == len(allc) - len(few)
    assert len(few) <= 5
    assert few[0] == allc[0] and few[-1] == allc[-1]


def test_the_drop_count_is_reported_so_the_cap_is_never_silent():
    """CLAUDE.md: if a figure bounds coverage, say what was dropped - silent truncation
    reads as 'covered everything'."""
    ch = SCH.Chain(SC.STAB_EVENTS[0], 10)
    times = [ch.valid_at(l) for l in SCH.scoring_checkpoints(ch.horizon_days)]
    f = _series(times)
    few, dropped = MP._columns(ch, f, f, 4)
    assert dropped > 0
    assert len(few) + dropped == len(MP._columns(ch, f, f, 0)[0])


def test_a_model_with_no_frames_yields_no_columns():
    ch = SCH.Chain(SC.STAB_EVENTS[0], 4)
    empty = xr.DataArray(np.zeros((0, 0, 0)), dims=("time", "lat", "lon"))
    assert MP._columns(ch, empty, _series([ch.peak]), 8) == ([], 0)


def test_the_adapter_fed_arm_simply_has_no_early_columns():
    """It launches at lead 3 d, so a column at 3 d is its own step zero and everything
    before the launch is absent. That is not an error and must not be one."""
    ch = SCH.Chain(SC.STAB_EVENTS[0], 8)
    all_cp = [ch.valid_at(l) for l in SCH.scoring_checkpoints(ch.horizon_days)]
    truth = _series(all_cp)
    fed = _series([t for t in all_cp if t >= ch.valid_at(9.0)])
    cols, _ = MP._columns(ch, fed, truth, 0)
    assert cols and min(cols) == ch.valid_at(9.0)
    assert len(cols) < len(all_cp)


# --------------------------------------------------------------------------- #
# Assembling the walker's own field
# --------------------------------------------------------------------------- #
def test_overlapping_segments_are_not_counted_twice(tmp_path, monkeypatch):
    """Segments share no frames, but a re-walk that overlapped would silently double one
    and put two panels at the same valid time."""
    ev = SC.STAB_EVENTS[0]
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    ch = SCH.Chain(ev, 4)
    times = pd.date_range(ch.init + pd.Timedelta(hours=12), periods=6, freq="12h")
    for k in (1, 2):
        p = SC.diag_path(ev, ch.walker, k)
        p.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"2m_temperature": _series(times, seed=k)}).to_netcdf(p)
    out = MP._walker_t2m(ch)
    assert out.sizes["time"] == len(times)
    assert pd.DatetimeIndex(out["time"].values).is_monotonic_increasing


def test_an_unknown_model_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown model"):
        MP.forecast_t2m(SCH.Chain(SC.STAB_EVENTS[0], 4), "persistence")


def test_the_walker_is_not_drawn_at_an_fcn3_extension_lead(tmp_path, monkeypatch, capsys):
    """One 3-day segment is not a filmstrip. Drawn beside the other figures it would imply
    a 140-day walker trajectory that was never rolled - the same claim
    ``reduce._model_survival`` refuses to make."""
    monkeypatch.setattr(SC, "RUNS", tmp_path)
    monkeypatch.setattr(SC, "FIGS", tmp_path / "figs")
    ext = [w for w in SC.STAB_WEEKS if not SC.walks_to_peak(w)]
    if not ext:
        pytest.skip("no FCN3-extension leads configured")
    chs = SCH.chains([SC.STAB_EVENTS[0]], [ext[0]])
    assert MP.stage_maps(chs, models=("gencast",)) == 0
    out = capsys.readouterr().out
    assert "no trajectory to draw" in out
    assert "not drawn (FCN3-extension leads)" in out


def test_every_model_on_the_stability_curve_has_a_map_source():
    """The set of map figures is one per curve on ``stability.png``; a model on the plot
    with no map source would be a gap the reader cannot see."""
    assert set(MP.MODEL_LABEL) == set(SC.MODELS)
    from astab.plots import MODEL_STYLE
    assert set(MODEL_STYLE) == set(SC.MODELS)


# --------------------------------------------------------------------------- #
# Where the figures land
# --------------------------------------------------------------------------- #
def test_one_figure_per_event_model_and_initialization():
    paths = {SC.map_path(e, m, w)
             for e in SC.STAB_EVENTS for m in SC.MODELS for w in SC.STAB_WEEKS}
    assert len(paths) == len(SC.STAB_EVENTS) * len(SC.MODELS) * len(SC.STAB_WEEKS)
    assert all(SC.FIGS in p.parents for p in paths)


def test_the_week_is_zero_padded_so_ten_sorts_after_four():
    """``wk4`` and ``wk10`` in one directory sort wrong without the pad, and these are
    read as a filmstrip in filename order."""
    ev = SC.STAB_EVENTS[0]
    names = sorted(SC.map_path(ev, "gencast", w).name for w in (4, 6, 8, 10))
    assert names == ["gencast_wk04.png", "gencast_wk06.png", "gencast_wk08.png",
                     "gencast_wk10.png"]
