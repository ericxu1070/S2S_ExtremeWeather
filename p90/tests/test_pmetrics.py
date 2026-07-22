"""Unit tests for p90.pmetrics.

Pure synthetic numpy/xarray - no network, no GPU, no reads under runs/. Runs under pytest
(``PYTHONPATH=. python -m pytest p90/tests -q``) or as a plain script
(``PYTHONPATH=. python p90/tests/test_pmetrics.py``) via the __main__ block, which calls
every test_ function and prints PASS. The properscoring cross-check skips cleanly if the
package is absent; everything else is self-contained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

try:
    import pytest
except ImportError:                     # plain-script fallback when pytest is absent
    class _Skipped(Exception):
        pass

    class _Approx:
        def __init__(self, value, rel=1e-6, abs=1e-12):
            self.value, self.rel, self.abs = value, rel, abs

        def __eq__(self, other):
            return abs(float(other) - float(self.value)) <= max(
                self.abs, self.rel * abs(float(self.value)))

        def __repr__(self):
            return f"approx({self.value})"

    class _PytestShim:
        Skipped = _Skipped

        @staticmethod
        def approx(value, rel=1e-6, abs=1e-12):
            return _Approx(value, rel, abs)

        @staticmethod
        def importorskip(name):
            import importlib
            try:
                return importlib.import_module(name)
            except ImportError:
                raise _Skipped(f"missing {name}")

    pytest = _PytestShim()

from p90 import pmetrics as PM
from p90 import pconfig as P


# --------------------------------------------------------------------------- #
# CRPS
# --------------------------------------------------------------------------- #
def test_crps_matches_properscoring():
    ps = pytest.importorskip("properscoring")
    rng = np.random.default_rng(12345)
    members = rng.normal(size=(17, 6, 5))
    truth = rng.normal(size=(6, 5))
    got = PM.crps_field(members, truth)
    ref = ps.crps_ensemble(truth, np.moveaxis(members, 0, -1))
    assert np.allclose(got, ref, atol=1e-9)


def test_crps_analytic():
    # M = 1 -> CRPS == |x - y| (no pairwise term).
    m1 = np.array([[[3.0]]])
    assert np.allclose(PM.crps_field(m1, np.array([[1.0]])), 2.0)
    # M = 2, members {0, 2}, truth 1 -> 0.5(1+1) - |0-2|/4 = 0.5.
    m2 = np.array([[[0.0]], [[2.0]]])
    assert np.allclose(PM.crps_field(m2, np.array([[1.0]])), 0.5)


def test_crps_nan_propagates():
    members = np.array([[[1.0, np.nan]], [[2.0, 3.0]]])
    truth = np.array([[1.5, 2.0]])
    out = PM.crps_field(members, truth)
    assert np.isnan(out[0, 1]) and np.isfinite(out[0, 0])


# --------------------------------------------------------------------------- #
# Rank histogram
# --------------------------------------------------------------------------- #
def test_rank_hist_total_and_example():
    members = np.array([[[1.0, 0.0]], [[2.0, 0.0]], [[3.0, 0.0]]])   # M=3
    truth = np.array([[2.5, -1.0]])                                  # below = 2, 0
    h = PM.rank_hist_field(members, truth)
    assert list(h) == [1, 0, 1, 0]           # len M+1 = 4
    assert h.sum() == 2                       # == number of finite points


def test_rank_hist_ignores_nan_points():
    members = np.array([[[1.0, np.nan]], [[2.0, 5.0]]])
    truth = np.array([[1.5, 4.0]])
    h = PM.rank_hist_field(members, truth)
    assert h.sum() == 1                       # only the finite column counts


# --------------------------------------------------------------------------- #
# Brier decomposition + skill score
# --------------------------------------------------------------------------- #
def test_brier_decomposition_exact_on_bin_centers():
    centers = (np.arange(10) + 0.5) / 10.0     # bin centres, constant within each bin
    p = np.repeat(centers, 20)
    rng = np.random.default_rng(7)
    o = (rng.random(p.size) < p).astype(float)
    rel, res, unc = PM.brier_decomposition(p, o, nbins=10)
    assert abs((rel - res + unc) - PM.brier(p, o)) < 1e-10


def test_bss_sign():
    o = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert PM.bss(o.copy(), o) == pytest.approx(1.0)     # perfect -> skill 1
    assert PM.bss(1.0 - o, o) < 0                          # reversed -> negative skill
    assert np.isnan(PM.bss(np.full_like(o, 0.5), np.ones_like(o)))  # obar==1 -> nan


# --------------------------------------------------------------------------- #
# ROC
# --------------------------------------------------------------------------- #
def test_roc_perfect_and_reversed():
    o = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    far, hr, auc = PM.roc_curve(o.copy(), o)
    assert auc == pytest.approx(1.0)
    assert far[0] == pytest.approx(0.0) and far[-1] == pytest.approx(1.0)   # endpoints
    far_r, hr_r, auc_r = PM.roc_curve(1.0 - o, o)
    assert (1.0 - auc_r) == pytest.approx(1.0)              # reversed -> auc 0


# --------------------------------------------------------------------------- #
# Weighted mean + ACC
# --------------------------------------------------------------------------- #
def test_wmean_uniform():
    lat = np.array([10.0, 20.0, 30.0])
    field = np.full((3, 4), 7.5)
    assert PM.wmean(field, lat) == pytest.approx(7.5)


def test_wmean_coslat_handcheck():
    lat = np.array([0.0, 60.0])                # weights 1.0 and 0.5
    field = np.array([[3.0], [0.0]])
    assert PM.wmean(field, lat) == pytest.approx((3.0 + 0.5 * 0.0) / 1.5)   # = 2.0


def test_wmean_nan_tolerant():
    lat = np.array([0.0, 0.0])
    field = np.array([[np.nan, 4.0], [2.0, np.nan]])   # NaNs dropped, rest averaged
    assert PM.wmean(field, lat) == pytest.approx(3.0)


def test_acc_perfect():
    lat = np.array([10.0, 20.0])
    f = np.array([[1.0, -2.0], [3.0, 0.5]])
    assert PM.acc(f, 2.0 * f, lat) == pytest.approx(1.0)   # uncentered, colinear -> 1


# --------------------------------------------------------------------------- #
# Ensemble spread / error ratio
# --------------------------------------------------------------------------- #
def test_spread_error_calibrated():
    # Reliable ensemble: truth is an extra draw from the same distribution as the members.
    rng = np.random.default_rng(0)
    M, H, W = 200, 40, 40
    members = rng.normal(size=(M, H, W))
    truth = rng.normal(size=(H, W))
    lat = np.linspace(24, 50, H)
    es = PM.ens_stats(members, truth, lat)
    assert abs(es["spread_error"] - 1.0) < 0.05


# --------------------------------------------------------------------------- #
# Daily series + p_event on a synthetic cube (clim monkeypatched to a constant)
# --------------------------------------------------------------------------- #
def _synthetic_cube():
    """88-frame cube (member, time, lat, lon); member m is a spatially-constant field
    280 + m K at every frame, so its daily anomaly vs a 280 K climatology is exactly m."""
    peak = pd.Timestamp("2022-07-01T00:00:00")
    init = peak - pd.Timedelta(days=P.LEAD_DAYS)
    times = pd.date_range(init, periods=P.NSTEPS + 1, freq=f"{P.STEP_H}h")
    lat = np.linspace(24.0, 50.0, 5)
    lon = np.linspace(235.0, 294.0, 4)
    M = 6
    data = np.empty((M, len(times), len(lat), len(lon)), dtype="float32")
    for m in range(M):
        data[m] = 280.0 + m
    cube = xr.Dataset(
        {"2m_temperature": (("member", "time", "lat", "lon"), data)},
        coords={"member": np.arange(M), "time": times, "lat": lat, "lon": lon},
    )
    return cube, peak, lat, lon


def _fake_clim_factory(lat, lon, value=280.0):
    def fake_clim_for(times):
        times = pd.DatetimeIndex(times)
        arr = np.full((len(times), len(lat), len(lon)), value, dtype="float32")
        return xr.DataArray(arr, dims=("time", "lat", "lon"),
                            coords={"time": times, "lat": lat, "lon": lon})
    return fake_clim_for


def test_daily_series_and_p_event(monkeypatch):
    import gencast_s2s.data as gd
    cube, peak, lat, lon = _synthetic_cube()
    monkeypatch.setattr(gd, "clim_for", _fake_clim_factory(lat, lon))
    series = PM.daily_conus_series(cube, peak)
    assert series.shape == (6, 7)
    for m in range(6):
        assert np.allclose(series[m], float(m))           # daily anomaly == m every day
    # p_event: members with m >= THR_K (2.3662) are 3,4,5 -> 3 of 6.
    pe = PM.p_event(cube, peak)
    assert pe == pytest.approx(3.0 / 6.0)


# --------------------------------------------------------------------------- #
# Plain-script runner
# --------------------------------------------------------------------------- #
class _MonkeyPatch:
    """Minimal monkeypatch shim so the __main__ path needs no pytest."""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        old = getattr(target, name)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def undo(self):
        for target, name, old in reversed(self._undo):
            setattr(target, name, old)
        self._undo = []


def _run_all():
    import inspect
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = skipped = 0
    for name, fn in fns:
        mp = _MonkeyPatch()
        try:
            kwargs = {}
            if "monkeypatch" in inspect.signature(fn).parameters:
                kwargs["monkeypatch"] = mp
            fn(**kwargs)
            print(f"PASS {name}")
            passed += 1
        except Exception as exc:  # importorskip raises Skipped; treat as skip
            if type(exc).__name__ in ("Skipped", "OutcomeException"):
                print(f"SKIP {name}: {exc}")
                skipped += 1
            else:
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
                raise
        finally:
            mp.undo()
    print(f"\n{passed} passed, {skipped} skipped")


if __name__ == "__main__":
    _run_all()
