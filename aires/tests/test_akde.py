"""Contract tests for the KDE population figure.

The figure's two claims are that the smooth curve is an honest picture of the same 64
walkers the scatter shows, and that it never asserts more than the run knows. So: the
unweighted estimator IS scipy's (no home-grown kernel drifting from the house one); the
weighted estimator collapses onto the exact subset sum as the kernel narrows; both walker
curves share one bandwidth; and every curve stops at its own sample's edge, so a Gaussian
tail cannot put mass past an observation nothing reached.

No GPU, no network, no real run on disk.
"""
from __future__ import annotations

import numpy as np
import pytest

from aires import akde as K
from aires import awalkers as W
from aires.tests.test_awalkers import _compare_dict, _write

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _record(monkeypatch, tmp_path, event="PNW_HeatDome_2021", **kw):
    _write(monkeypatch, tmp_path, event, "unit", _compare_dict(**kw))
    return W.run_record(event, "unit")


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
def test_unweighted_kde_is_scipys():
    """Scott's rule and the kernel reproduce ``gaussian_kde`` - the KDE
    `gencast_s2s/plotting.py` draws - to round-off."""
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    grid = np.linspace(-4, 4, 200)
    assert K.kde(x, grid) == pytest.approx(gaussian_kde(x)(grid), rel=1e-10)
    assert K.scott_bandwidth(x) == pytest.approx(gaussian_kde(x).factor * x.std(ddof=1))


def test_uniform_weights_are_the_unweighted_curve_and_any_weights_integrate_to_one():
    rng = np.random.default_rng(1)
    x = rng.normal(size=30)
    h = K.scott_bandwidth(x)
    grid = np.linspace(x.min() - 8 * h, x.max() + 8 * h, 4000)
    assert np.allclose(K.kde(x, grid, np.full(30, 0.37)), K.kde(x, grid), rtol=1e-12)
    w = rng.exponential(size=30) ** 3            # wildly uneven, like p_i
    assert _trapz(K.kde(x, grid, w), grid) == pytest.approx(1.0, rel=1e-6)


def test_the_weighted_tail_collapses_onto_the_subset_sum_as_the_kernel_narrows():
    """With ``bw`` far below the point spacing the KDE's mass past the observation is
    the estimator's own ``P / sum p_i`` - for both tails. 0.5 sits strictly between two
    walkers, as in `test_awalkers`, so nothing is on the threshold."""
    for sign in (+1.0, -1.0):
        d = _compare_dict(sign=sign, obs=0.5)
        t = W.walker_table("PNW_HeatDome_2021", "unit", d)
        al, p = t["A_L"].to_numpy(), t["p_i"].to_numpy()
        exact = p[sign * al >= sign * 0.5].sum() / p.sum()
        assert K.tail_mass(al, p, 1e-3, 0.5, sign) == pytest.approx(exact, rel=1e-12)
    # and for ANY bandwidth the two tails partition the mass
    al = np.linspace(-1, 6, 8)
    h = K.scott_bandwidth(al)
    assert (K.tail_mass(al, None, h, 0.5, +1.0) + K.tail_mass(al, None, h, 0.5, -1.0)
            == pytest.approx(1.0, rel=1e-12))


def test_guards():
    with pytest.raises(ValueError):
        K.scott_bandwidth([1.0])                       # one point
    with pytest.raises(ValueError):
        K.scott_bandwidth([2.0, 2.0, 2.0])             # zero variance
    with pytest.raises(ValueError):
        K.kde([0.0, 1.0], [0.5], bw=0.0)               # --bw 0
    with pytest.raises(ValueError):
        K.kde([0.0, 1.0], [0.5], weights=[1.0, -1.0])  # a negative mass
    assert K.kde([2.0, 2.0], [2.0], bw=0.5).size == 1  # zero variance is fine with bw=


# --------------------------------------------------------------------------- #
# The panel
# --------------------------------------------------------------------------- #
def test_both_walker_curves_share_one_bandwidth_and_direct_has_its_own(monkeypatch, tmp_path):
    r = _record(monkeypatch, tmp_path)
    cs = {c["key"]: c for c in K.panel_curves(r)}
    al = r["table"]["A_L"].to_numpy()
    assert cs["aires_weighted"]["h"] == cs["aires_unweighted"]["h"] == K.scott_bandwidth(al)
    assert cs["direct"]["h"] == pytest.approx(K.scott_bandwidth(r["direct"]))
    assert cs["aires_weighted"]["h"] != cs["direct"]["h"]
    assert [c["key"] for c in K.panel_curves(r)][-1] == "aires_weighted"   # drawn on top


def test_a_run_without_a_direct_baseline_has_no_direct_curve(monkeypatch, tmp_path):
    d = _compare_dict()
    d["ds"] = {}
    _write(monkeypatch, tmp_path, "PNW_HeatDome_2021", "unit", d)
    r = W.run_record("PNW_HeatDome_2021", "unit")
    assert "direct" not in {c["key"] for c in K.panel_curves(r)}
    lo, hi = K.panel_xlim(r)
    assert lo < r["table"]["A_L"].min() and hi > r["table"]["A_L"].max()


def test_z_scaled_multiplies_only_the_weighted_curve_by_the_normalization_check(monkeypatch, tmp_path):
    r = _record(monkeypatch, tmp_path)
    plain = {c["key"]: c for c in K.panel_curves(r)}
    z = {c["key"]: c for c in K.panel_curves(r, z_scaled=True)}
    ok = np.isfinite(plain["aires_weighted"]["density"])
    assert np.allclose(z["aires_weighted"]["density"][ok],
                       r["total_mass"] * plain["aires_weighted"]["density"][ok], rtol=1e-12)
    assert z["aires_weighted"]["tail"] == pytest.approx(
        r["total_mass"] * plain["aires_weighted"]["tail"], rel=1e-12)
    for key in ("aires_unweighted", "direct"):
        assert np.array_equal(z[key]["density"], plain[key]["density"], equal_nan=True)


@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_every_curve_ends_at_its_own_most_extreme_member(monkeypatch, tmp_path, sign):
    """Southwest and the persistence control: the observation lies beyond every walker.
    The kernel tail must NOT be drawn out to it - a curve past the sample would show
    mass in the band the run reports empty."""
    al = sign * np.linspace(-1.0, 6.0, 8)
    # 20 K is beyond every walker but within float range of the kernel, so the reported
    # (unclipped) tail is tiny and non-zero rather than an underflow to exactly 0.
    r = _record(monkeypatch, tmp_path, sign=sign, obs=sign * 20.0, al=al)
    assert r["n_reached"] == 0
    lo, hi = K.panel_xlim(r)
    assert (hi > 20.0) if sign > 0 else (lo < -20.0)      # the grid reaches the obs
    for c in K.panel_curves(r):
        g, y, x = c["grid"], c["density"], np.asarray(c["x"])
        inside = (g >= x.min()) & (g <= x.max())
        assert np.all(np.isfinite(y[inside])) and np.all(y[inside] > 0)
        assert np.all(np.isnan(y[~inside]))
        assert inside.any() and (~inside).any()
        # the unclipped tail is still reported, so the leakage is on record, and for
        # this run it is tiny but not zero
        assert 0 < c["tail"] < 1e-6


def test_panel_xlim_covers_walkers_direct_and_observation_for_both_signs(monkeypatch, tmp_path):
    for sign, obs in ((+1.0, 7.5), (-1.0, -7.5)):
        r = _record(monkeypatch, tmp_path, sign=sign, obs=obs,
                    al=sign * np.linspace(-1.0, 6.0, 8))
        lo, hi = K.panel_xlim(r)
        pts = np.concatenate([r["table"]["A_L"].to_numpy(), r["direct"], [obs]])
        assert lo < pts.min() and hi > pts.max()


def test_order_runs_is_the_severity_ladder():
    """Hardest first, no-baseline runs last, ties broken by name - the order both slates
    share, pinned against the inline sort `plot_walkers` used before it was extracted."""
    runs = [dict(sigma_depth=np.nan, event="b", tag="pilot"),
            dict(sigma_depth=1.0, event="a", tag="pilot"),
            dict(sigma_depth=3.3, event="c", tag="pilot"),
            dict(sigma_depth=1.0, event="a", tag="persist")]
    got = [(r["event"], r["tag"]) for r in W.order_runs(runs)]
    assert got == [("c", "pilot"), ("a", "persist"), ("a", "pilot"), ("b", "pilot")]


# --------------------------------------------------------------------------- #
# Rendering (smoke only)
# --------------------------------------------------------------------------- #
def test_the_slate_and_the_per_run_figure_render(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, "PNW_HeatDome_2021", "unit", _compare_dict())
    _write(monkeypatch, tmp_path, "WinterStorm_Uri_2021", "unit",
           _compare_dict(sign=-1.0, obs=-0.5, al=-np.linspace(-1.0, 6.0, 8)))
    _write(monkeypatch, tmp_path, "Southwest_HeatWave_2020", "unit",
           _compare_dict(obs=99.0))
    recs = [W.run_record(e, "unit") for e in
            ("PNW_HeatDome_2021", "WinterStorm_Uri_2021", "Southwest_HeatWave_2020")]
    out = K.plot_kde(recs, tmp_path / "k.png", ncols=2, z_scaled=True)
    assert out.exists() and out.stat().st_size > 10_000
    out = K.plot_kde_run(recs[1], tmp_path / "r.png")
    assert out.exists() and out.stat().st_size > 10_000
