"""acal/multi_reach.py: series layout, probability axis mapping and member statistics.
Pure helpers only; no data is loaded."""
from __future__ import annotations

import numpy as np

from acal import multi_reach as MR


def test_series_layout_has_both_windows_and_unique_keys():
    ser = MR.series(["cfs13", "ec46", "gefs", "geps"], clim=True)
    keys = [s["key"] for s in ser]
    assert keys[:2] == ["aires13", "aires12"] and keys[-1] == "clim"
    assert len(keys) == len(set(keys)) == 2 + 4 * 2 + 1
    assert [s["window"] for s in ser if s["source"] in ("ec46", "geps")] == ["12f"] * 4


def test_no_12f_row_without_daily_source():
    ser = MR.series(["cfs13", "gefs"])
    assert [s["key"] for s in ser][:2] == ["aires13", "cfs13_corr"]
    assert not MR.has_daily(["cfs13", "gefs"]) and MR.has_daily(["ec46"])


def test_offsets_and_px():
    o = MR.offsets(5)
    assert o[0] > o[-1] and np.allclose(np.diff(o), np.diff(o)[0])
    assert np.all(MR.px([0.0, 0.2]) == [MR.P_ZERO, 0.2])


def test_member_stats_weighted_and_equal():
    x = np.array([5.0, 4.0, 3.0, -1.0])
    assert MR.member_stats(x, 1.0) == (2, 4, 0.5)          # >= rule: 4.0 counts
    wn = np.array([0.1, 0.2, 0.3, 0.4])
    k, n, p = MR.member_stats(x, 1.0, wn)
    assert (k, n) == (2, 4) and abs(p - 0.3) < 1e-12
    assert MR.member_stats(-x, -1.0)[0] == 2                # cold tail mirrored
