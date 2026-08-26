"""``aires/aclim.py`` -- the climatological reference for ``A_L``.

Nothing here touches the network or the 1.5 MB cache: the tests pin the bookkeeping that
is easy to get silently wrong (which day a rolling window is labelled by, which tail a
percentile is taken in, whether a gap in the daily series can be averaged across) rather
than the numbers, which are data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aires import aclim as AC


def test_window_index_is_labelled_by_the_peak_not_the_last_day():
    """``A_L`` is the mean of the days BEFORE the peak; the series must be indexed by the
    peak so ``idx[peak]`` lines up with the experiment's number rather than being one day
    early. This is the whole reason ``window_index`` shifts."""
    days = pd.date_range("2021-06-01", "2021-06-30", freq="D")
    s = pd.Series(np.arange(len(days), dtype="float64"), index=days)
    w = AC.window_index(s, days=6)
    peak = pd.Timestamp("2021-06-28")
    # mean of the 6 values on 2021-06-22 .. 2021-06-27, i.e. indices 21..26
    assert w.loc[peak] == pytest.approx(np.arange(21, 27).mean())


def test_window_index_refuses_to_average_across_a_gap():
    """The ARCO top-up is separated from the stored record by months. A rolling mean over
    a gappy index would bridge it silently, so the anomaly frame is reindexed onto a
    complete calendar first and the window must come out NaN."""
    idx = (pd.date_range("2023-01-01", "2023-01-10", freq="D")
           .append(pd.date_range("2023-06-21", "2023-06-24", freq="D")))
    s = pd.Series(1.0, index=idx).reindex(
        pd.date_range(idx.min(), idx.max(), freq="D"))
    w = AC.window_index(s, days=6)
    # window_index drops the NaNs, so a bridged window must be ABSENT, not wrong.
    assert pd.Timestamp("2023-06-25") not in w.index
    assert pd.Timestamp("2023-01-08") in w.index      # a real, complete window survives


def test_season_mask_wraps_the_year_end():
    """p90_20251224 peaks on 24 December; its +-21 d pool has to reach into January."""
    idx = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    m = AC._season_mask(idx, pd.Timestamp("2025-12-24"), half=21)
    assert m[idx.get_loc(pd.Timestamp("2000-01-05"))]
    assert m[idx.get_loc(pd.Timestamp("2000-12-24"))]
    assert not m[idx.get_loc(pd.Timestamp("2000-06-24"))]


def test_percentile_and_ladder_follow_the_cold_tail():
    """Winter Storm Uri's extreme is the NEGATIVE tail. ``aindex.tail_sign`` is the only
    place that convention lives, and every statistic here has to honour it or the freeze
    gets scored as a mild day."""
    x = np.linspace(-10.0, 10.0, 2001)               # symmetric about zero
    assert AC.percentile_of(-9.0, x, sign=-1.0) > 90
    assert AC.percentile_of(-9.0, x, sign=+1.0) < 10
    hi = AC.quantile_ladder(x, sign=+1.0)
    lo = AC.quantile_ladder(x, sign=-1.0)
    assert hi[99] > hi[95] > hi[50]
    assert lo[99] < lo[95] < lo[50]
    assert hi[50] == pytest.approx(lo[50])            # the median has no tail
    assert AC.exceedance(9.0, x, sign=+1.0) == pytest.approx(
        AC.exceedance(-9.0, x, sign=-1.0), abs=1e-12)


def test_conus_slice_matches_the_climatology_grid():
    """The cubes, the climatology file and this reduction must all be on the same 105x237
    CONUS window -- the failure mode CLAUDE.md's climatology-grid incident describes."""
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.arange(0.0, 360.0, 0.25)
    ilat, ilon = AC._conus_slice(lat, lon)
    assert lat[ilat].size == 105 and lon[ilon].size == 237
    assert lat[ilat].min() == pytest.approx(24.0)
    assert lat[ilat].max() == pytest.approx(50.0)
    assert lon[ilon].min() == pytest.approx(235.0)
    assert lon[ilon].max() == pytest.approx(294.0)


def test_boxes_cover_every_reported_t2m_event():
    """A missing box silently falls back to the CONUS mean, which scores a different
    event. Every event ``--report`` names must resolve to a column the cache carries."""
    from fcn3 import fevents as F

    cols = set(AC._boxes())
    for name in AC.REPORT_EVENTS:
        if F.event(name).metric != "t2m_anom":
            continue
        col = name if name in cols else "CONUS"
        assert col in cols
        if name.startswith("p90_"):
            assert col == "CONUS"      # the p90 index IS the CONUS mean, by design
        else:
            assert col == name, f"{name} lost its box and would be scored on CONUS"


def test_the_cache_on_disk_is_not_stale_relative_to_the_registered_boxes():
    """``test_boxes_cover_every_reported_t2m_event`` checks ``_boxes()``, which is built
    from ``EVENT_BOXES`` at call time - so it passes even when the CACHED FILE predates a
    newly registered box. That is the gap that shipped: `WinterStorm_Elliott_2022` was
    added to EVENT_BOXES after the cache was built, `daily_anomaly()` had no column for
    it, and its N Plains ``A_L`` (reaching -17 K) was scored against the CONUS climatology
    (spanning +-5 K) with no error anywhere. Check the file, not the registry.

    Skipped where the cache has not been built; a rebuild is
    ``PYTHONPATH=. python -m aires.aclim --build``.
    """
    if not AC.CACHE.exists():
        pytest.skip(f"no climatology cache at {AC.CACHE}")
    with xr.open_dataset(AC.CACHE) as ds:
        have = {str(b) for b in ds["box"].values}
    missing = set(AC._boxes()) - have
    assert not missing, (f"{AC.CACHE.name} has no column for {sorted(missing)}; those "
                         f"events would silently be scored on the CONUS mean. Rebuild: "
                         f"PYTHONPATH=. python -m aires.aclim --build")
