"""Tests for the AI+RES PDF figures.

Only the estimator is pinned here, and it is pinned against the thing it claims to
generalize. `aires/apdfs.py::weighted_pdf` exists so an importance-weighted population can
be drawn on the same axes as unweighted ones; if it is not the house
`xres/xcombined.py::PDF_histogram` recipe with weights bolted on, then the weighted curve
and the ERA5 curve beside it are two different estimators and the figure's whole
bin-for-bin comparison is meaningless - while still looking perfectly reasonable.

Four identities do that job:

    weights=None            ==  PDF_histogram, bin for bin
    weights=ones            ==  weights=None
    mode='z'                ==  z_factor x the self-normalized curve, exactly
    self-normalized mass    ==  1 whenever the bin range covers the data

No matplotlib, no cartopy, no run tree: everything below is arithmetic on arrays.
(`xres.xcombined` itself pulls in matplotlib, so it is imported lazily inside the two
tests that need it rather than at module scope.)
"""
from __future__ import annotations

import numpy as np
import pytest

from aires import apdfs as P

Z_FACTOR = 0.5683748752526481          # the pilot's DMC normalization_check


def _house():
    """`xres.xcombined`, imported on demand (~28 s of plotting stack on first call)."""
    from xres import xcombined as XC
    return XC


def _sample(shape=(12, 40), seed=0):
    """A population shaped the way the figure sees one: (member, points)."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=1.5, scale=2.0, size=shape)


# --------------------------------------------------------------------------- #
# The identity that matters: unweighted == the house recipe
# --------------------------------------------------------------------------- #
def test_unweighted_reproduces_the_house_pdf_histogram():
    x = _sample()
    XC = _house()
    want_pts, want_dens = XC.PDF_histogram(x)
    got_pts, got_dens = P.weighted_pdf(x, None)
    assert np.allclose(got_pts, want_pts, rtol=0, atol=1e-12)
    assert np.allclose(got_dens, want_dens, rtol=0, atol=1e-12)


def test_unweighted_reproduces_the_house_recipe_on_a_shared_bin_range():
    """The figures always pass an explicit shared (xmin, xmax); the identity has to hold
    there too, not only on each curve's own mean +/- 4 sigma default."""
    x = _sample(seed=3)
    XC = _house()
    lo, hi = -4.25, 7.75
    want_pts, want_dens = XC.PDF_histogram(x, xmin=lo, xmax=hi)
    got_pts, got_dens = P.weighted_pdf(x, None, xmin=lo, xmax=hi)
    assert np.allclose(got_pts, want_pts, rtol=0, atol=1e-12)
    assert np.allclose(got_dens, want_dens, rtol=0, atol=1e-12)


def test_house_style_constants_match_xcombined():
    """NBINS/YMIN are mirrored rather than imported (import cost); they must not drift."""
    XC = _house()
    assert P.NBINS == XC.NBINS
    assert P.YMIN == XC.YMIN


# --------------------------------------------------------------------------- #
# Equal weights are no weights
# --------------------------------------------------------------------------- #
def test_uniform_weights_equal_no_weights():
    x = _sample(seed=1)
    a = P.weighted_pdf(x, np.ones(x.shape[0]), xmin=-6.0, xmax=9.0)
    b = P.weighted_pdf(x, None, xmin=-6.0, xmax=9.0)
    assert np.allclose(a[0], b[0], rtol=0, atol=1e-12)
    assert np.allclose(a[1], b[1], rtol=0, atol=1e-12)


def test_uniform_weights_equal_no_weights_on_the_default_range():
    """The default range is derived from the same finite values in both paths."""
    x = _sample(seed=7)
    a = P.weighted_pdf(x, np.ones(x.shape[0]))
    b = P.weighted_pdf(x, None)
    assert np.allclose(a[0], b[0], rtol=0, atol=1e-12)
    assert np.allclose(a[1], b[1], rtol=0, atol=1e-12)


def test_a_rescaled_weight_vector_changes_nothing():
    """Self-normalization means only the weights' RATIOS can move the curve."""
    x = _sample(seed=5)
    w = np.abs(np.random.default_rng(11).normal(size=x.shape[0])) + 0.05
    a = P.weighted_pdf(x, w, xmin=-6.0, xmax=9.0)[1]
    b = P.weighted_pdf(x, 137.0 * w, xmin=-6.0, xmax=9.0)[1]
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_weighting_actually_moves_the_curve():
    """The guard on the identities above: an unequal weight vector must NOT reproduce
    the unweighted curve, or the tests would pass on a function that ignores weights."""
    x = _sample(seed=2)
    w = np.linspace(0.01, 5.0, x.shape[0])
    a = P.weighted_pdf(x, w, xmin=-6.0, xmax=9.0)[1]
    b = P.weighted_pdf(x, None, xmin=-6.0, xmax=9.0)[1]
    assert not np.allclose(a, b, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------- #
# The Z factor is a rescaling, never a reshaping
# --------------------------------------------------------------------------- #
def test_z_mode_is_exactly_the_normalization_factor_times_selfnorm():
    x = _sample(seed=4)
    w = np.abs(np.random.default_rng(4).normal(size=x.shape[0])) + 0.1
    _, self_norm = P.weighted_pdf(x, w, xmin=-6.0, xmax=9.0, mode="selfnorm")
    _, scaled = P.weighted_pdf(x, w, xmin=-6.0, xmax=9.0, mode="z", z_factor=Z_FACTOR)
    assert np.allclose(scaled, Z_FACTOR * self_norm, rtol=1e-12, atol=0)


def test_z_mode_needs_its_factor():
    x = _sample()
    with pytest.raises(ValueError):
        P.weighted_pdf(x, np.ones(x.shape[0]), mode="z")


def test_unknown_mode_is_refused():
    x = _sample()
    with pytest.raises(ValueError):
        P.weighted_pdf(x, None, mode="density")


# --------------------------------------------------------------------------- #
# Mass
# --------------------------------------------------------------------------- #
def test_self_normalized_mass_is_one_when_the_range_covers_the_data():
    x = _sample(seed=6)
    w = np.abs(np.random.default_rng(6).normal(size=x.shape[0])) + 0.1
    lo, hi = float(x.min()) - 1.0, float(x.max()) + 1.0
    for weights in (None, np.ones(x.shape[0]), w):
        _, dens = P.weighted_pdf(x, weights, xmin=lo, xmax=hi)
        assert float(dens.sum()) == pytest.approx(1.0, abs=1e-12)


def test_mass_falls_below_one_when_the_range_clips_the_data():
    x = _sample(seed=8)
    _, dens = P.weighted_pdf(x, None, xmin=float(x.min()), xmax=float(np.median(x)))
    assert 0.0 < float(dens.sum()) < 1.0


def test_empty_bins_are_zero_not_nan():
    """`weighted_pdf` must stay comparable to `PDF_histogram` bin for bin; breaking the
    curve at empty bins is the plotting layer's job, not the estimator's."""
    x = np.array([[0.0, 0.0, 10.0, 10.0]])
    _, dens = P.weighted_pdf(x, np.ones(1), xmin=0.0, xmax=10.0, nbins=4)
    assert np.isfinite(dens).all()
    assert float(dens.sum()) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Shape contract and the small helpers around the estimator
# --------------------------------------------------------------------------- #
def test_one_weight_per_slab_is_enforced():
    x = _sample(shape=(6, 20))
    with pytest.raises(ValueError):
        P.weighted_pdf(x, np.ones(5))
    with pytest.raises(ValueError):
        P.weighted_pdf(x.ravel(), np.ones(6))       # no slab axis at all


def test_every_grid_point_of_a_walker_carries_that_walker_s_weight():
    """Two walkers, disjoint values, weights 3:1 -> mass splits 0.75/0.25."""
    x = np.array([[0.5, 0.5, 0.5, 0.5], [9.5, 9.5, 9.5, 9.5]])
    _, dens = P.weighted_pdf(x, [3.0, 1.0], xmin=0.0, xmax=10.0, nbins=2)
    assert dens[0] == pytest.approx(0.75, abs=1e-12)
    assert dens[1] == pytest.approx(0.25, abs=1e-12)


def test_shared_range_is_the_house_mean_plus_minus_four_sigma_clipped():
    a = np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    b = np.array([0.0, 1.0])
    lo, hi = P.shared_range([a, b])
    allv = np.concatenate([a, b])
    mu, sd = allv.mean(), allv.std()
    assert lo == pytest.approx(max(allv.min(), mu - 4 * sd))
    assert hi == pytest.approx(min(allv.max(), mu + 4 * sd))
    assert lo >= allv.min() and hi <= allv.max()


def test_tail_probability_sums_the_bins_at_or_above_the_threshold():
    pts = np.array([1.0, 5.0, 9.0, 13.0])
    dens = np.array([0.5, 0.3, 0.15, 0.05])
    assert P.tail_probability(pts, dens, 9.0) == pytest.approx(0.20)
    assert P.tail_probability(pts, dens, 0.0) == pytest.approx(1.00)


def test_a_cold_tail_takes_the_mass_BELOW_the_threshold():
    """The sign is the whole reason this table means anything on a freeze.

    Unsigned, a cold event's `P(point >= threshold)` is the mass on the WARM side of a
    cold threshold - the complement of what the figure is about, and for a
    PNW-style +9 K constant it is identically zero for every curve.
    """
    pts = np.array([-15.0, -10.0, -5.0, 0.0])
    dens = np.array([0.05, 0.15, 0.3, 0.5])
    assert P.tail_probability(pts, dens, -10.0, -1.0) == pytest.approx(0.20)
    assert P.tail_probability(pts, dens, 0.0, -1.0) == pytest.approx(1.00)
    # the same call unsigned is the complement, which is the bug being pinned out
    assert P.tail_probability(pts, dens, -10.0, +1.0) == pytest.approx(0.95)


def test_the_four_wave_one_events_keep_the_published_nine_kelvin_threshold():
    """Those four `P(point >= 9 K)` tables are quoted in aires/HANDOFF.md and sit in
    figures/aires/. Deriving them instead would silently redraw published numbers."""
    era5 = np.linspace(-3.0, 3.0, 601)          # nowhere near +9 K: a derived
    for name in ("PNW_HeatDome_2021", "SCentral_HeatDome_2023",
                 "California_HeatWave_2022", "Southwest_HeatWave_2020"):
        thr, src = P.tail_threshold_for(name, era5, +1.0)
        assert thr == 9.0, name                 # threshold would be ~+2.9 K
        assert "pinned" in src


def test_a_new_event_derives_its_threshold_from_its_own_observed_field():
    """Event-intrinsic and sign-aware: the warm side takes p95, the cold side p5."""
    era5 = np.linspace(0.0, 100.0, 101)
    thr, src = P.tail_threshold_for("p90_20240802", era5, +1.0)
    assert thr == pytest.approx(95.0)
    assert "p95" in src
    cold, csrc = P.tail_threshold_for("WinterStorm_Elliott_2022", era5, -1.0)
    assert cold == pytest.approx(5.0)
    assert "p5" in csrc
    # and it scales with the event rather than with a constant
    mild, _ = P.tail_threshold_for("p90_20231107", era5 / 100.0, +1.0)
    assert mild == pytest.approx(0.95)


def test_tail_op_and_metric_unit_follow_the_sign_and_the_metric():
    assert (P.tail_op(+1.0), P.tail_op(-1.0)) == (">=", "<=")
    assert P.metric_unit("t2m_anom") == "K"
    assert P.metric_unit("u850_speed") == "m s$^{-1}$"
    assert P.metric_unit("tp_total") == "mm"


def test_ess_of_uniform_weights_is_the_count():
    assert P._ess(np.ones(64)) == pytest.approx(64.0)
    assert P._ess(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_bin_range_sources_are_the_three_headline_curves():
    """The context curves (Gate 3 walkers, FCN3) must not move the bin edges: adding one
    would silently renumber every published per-bin probability."""
    assert P.BIN_RANGE_SOURCES == ("era5", "aires_unweighted", "gencast_xres")
