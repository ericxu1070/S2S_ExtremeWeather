"""The arithmetic behind the trajectory figure's middle panel.

`aires.aplots.density_bins` and `aires.aplots._shared_ylim` are the two pure pieces of
the three-panel trajectory figure, and both fail SILENTLY: a bad bin count draws a
plausible histogram of the wrong shape, and a y limit that misses one population clips
dots off the bottom of a panel that the caption says shares its axis with the others.
Neither would raise, and neither is visible in a thumbnail. So they are pinned here.

The rendering itself is not smoke-tested: `plot_trajectory` needs every killed branch's
`diag.nc`, which is 450 files of real walker output, and there is no synthetic stand-in
for those. `test_aplots.py` covers the parts of that figure that can be driven from a
dict.
"""
from __future__ import annotations

import numpy as np
import pytest

from aires import aplots as P


# --------------------------------------------------------------------------- #
# density_bins: Freedman-Diaconis, floored by sqrt(n), capped by the bandwidth
# --------------------------------------------------------------------------- #
def test_edges_span_exactly_the_sample():
    """No member may fall outside the bins - `np.histogram` would drop it in silence."""
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.5, size=64)
    e = P.density_bins(x, 0.65)
    assert e[0] == pytest.approx(x.min())
    assert e[-1] == pytest.approx(x.max())
    counts, _ = np.histogram(x, bins=e)
    assert counts.sum() == x.size


def test_freedman_diaconis_is_used_when_it_sits_between_the_fences():
    """A wide, well-behaved sample: neither fence binds, so FD's own count comes out."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=64)
    q1, q3 = np.percentile(x, [25.0, 75.0])
    k_fd = int(np.ceil((x.max() - x.min()) / (2.0 * (q3 - q1) * 64 ** (-1.0 / 3.0))))
    h = (x.max() - x.min()) / (k_fd + 5)          # a bandwidth loose enough not to cap
    assert 8 < k_fd < (x.max() - x.min()) / h
    assert P.density_bins(x, h).size - 1 == k_fd


def test_a_degenerate_interquartile_range_falls_back_to_the_square_root_rule():
    """62 identical values and two outliers: IQR is 0, so FD's width is 0 and its count
    is undefined. The floor has to catch that, not a ZeroDivisionError or 10^9 bins."""
    x = np.concatenate([np.full(62, 3.0), [0.0, 6.0]])
    assert np.percentile(x, 75) - np.percentile(x, 25) == 0.0
    assert P.density_bins(x, 0.5).size - 1 == int(round(np.sqrt(64)))    # 8


def test_no_bin_is_ever_narrower_than_the_kernel_drawn_over_it():
    """The ceiling. A bar finer than the KDE bandwidth on the same axes would claim a
    resolution the curve beside it explicitly denies."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 0.3, size=64)             # tight IQR -> FD wants many bins
    h = 0.5 * (x.max() - x.min())                 # a deliberately fat kernel
    e = P.density_bins(x, h)
    # The floor outranks the ceiling by construction (see below), so the guarantee that
    # holds unconditionally is the one the floor allows.
    assert e.size - 1 <= max(int(round(np.sqrt(64))), int((x.max() - x.min()) / h))


def test_the_floor_outranks_the_ceiling_when_they_disagree():
    """A bandwidth wider than the whole sample would ask for zero bins. The floor wins:
    a histogram with no bars is not a more honest figure than a coarse one."""
    x = np.linspace(0.0, 1.0, 64)
    assert P.density_bins(x, 10.0).size - 1 == int(round(np.sqrt(64)))


def test_a_smaller_ensemble_gets_a_coarser_floor():
    """24 direct members are floored at 5 bins, not at the walkers' 8: the floor is a
    rule of n, so the two grids differ and do not invite bar-against-bar reading."""
    x = np.concatenate([np.full(22, 1.0), [0.0, 2.0]])       # IQR 0 -> floor binds
    assert P.density_bins(x, 0.4).size - 1 == int(round(np.sqrt(24)))    # 5


@pytest.mark.parametrize("x", [np.array([2.5]), np.array([]), np.full(30, 4.0)])
def test_a_degenerate_sample_returns_one_bin_rather_than_raising(x):
    """Southwest and the persistence control are near-degenerate populations and the
    figure has to render for them; a raise here would cost the whole three-panel plot."""
    e = P.density_bins(x, 0.5)
    assert e.size == 2 and e[1] > e[0]
    if x.size:
        assert e[0] <= x.min() and x.max() <= e[-1]


# --------------------------------------------------------------------------- #
# _shared_ylim: one range over every panel's data
# --------------------------------------------------------------------------- #
def test_the_shared_limit_contains_every_population_it_is_given():
    lo, hi = P._shared_ylim([np.array([1.0, 2.0]), np.array([-7.0]), np.array([11.0])])
    assert lo < -7.0 and hi > 11.0


def test_the_shared_limit_ignores_missing_and_empty_inputs():
    """A run with no CFSv2 cube and no ERA5 truth hands this ``None`` and empty arrays;
    it must not turn either into a NaN limit that blanks the whole figure."""
    lo, hi = P._shared_ylim([None, np.zeros(0), np.array([0.0, 4.0]), np.array([np.nan])])
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < 0.0 and hi > 4.0


def test_the_shared_limit_pads_so_a_line_on_the_extreme_never_lands_on_the_frame():
    """The observed value is regularly the most extreme number on the figure, and it is
    drawn as a horizontal rule across all three panels - on the frame it is invisible."""
    lo, hi = P._shared_ylim([np.array([0.0, 10.0])])
    assert lo < 0.0 and hi > 10.0
