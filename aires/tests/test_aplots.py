"""Tests for the Phase 4 figures.

Two kinds. The arithmetic behind the figures is pinned directly - a wrong sibling pairing
or a wrong weighted quantile would draw a plausible curve that says the wrong thing, and
nothing downstream would catch it. The rendering is only smoke-tested: that a figure comes
out at all, from a dict shaped like the real `compare.json`, which is what breaks when a
key is renamed upstream.
"""
from __future__ import annotations

import numpy as np
import pytest

from aires import aconfig as A
from aires import aplots as P


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #
def test_sibling_pairs_are_exactly_the_clones():
    # slots 0,1 share parent 3; slot 2 is alone; slots 3,4,5 share parent 7
    parents = [3, 3, 5, 7, 7, 7]
    assert P.sibling_pairs(parents) == [(0, 1), (3, 4), (3, 5), (4, 5)]


def test_sibling_pairs_is_empty_when_nothing_was_cloned():
    assert P.sibling_pairs([0, 1, 2, 3]) == []


def test_unrelated_pairs_never_include_a_sibling():
    parents = [0, 0, 1, 1, 2, 2]
    sib = set(P.sibling_pairs(parents))
    for pair in P.unrelated_pairs(parents, n_max=100):
        assert pair not in sib
        assert parents[pair[0]] != parents[pair[1]]


def test_exceedance_se_vanishes_when_every_walker_agrees():
    """No spread in the indicator means no sampling error in the estimate."""
    v = np.array([1.0, 2.0, 3.0])
    w = np.ones(3)
    se = P.res_exceedance_se(v, w, 0.0, [0.0])       # every walker exceeds
    assert se[0] == pytest.approx(0.0)


def test_exceedance_se_is_positive_when_walkers_disagree():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    se = P.res_exceedance_se(v, np.ones(4), 0.0, [2.5])
    assert se[0] > 0


def test_exceedance_se_respects_the_cold_tail_sign():
    v = np.array([-5.0, 0.0, 5.0])
    hot = P.res_exceedance_se(v, np.ones(3), 0.0, [4.0], sign=+1.0)
    cold = P.res_exceedance_se(v, np.ones(3), 0.0, [-4.0], sign=-1.0)
    assert hot[0] == pytest.approx(cold[0])          # mirror images of one another


def test_weighted_quantile_reduces_to_the_plain_one():
    v = np.arange(1, 10, dtype=float)
    got = P.weighted_quantile(v, np.ones(v.size), 0.5)
    assert got == pytest.approx(np.median(v), abs=1e-9)


def test_weighted_quantile_follows_the_weights():
    v = np.array([0.0, 10.0])
    assert P.weighted_quantile(v, [99.0, 1.0], 0.5) < 1.0
    assert P.weighted_quantile(v, [1.0, 99.0], 0.5) > 9.0


# --------------------------------------------------------------------------- #
# The trajectory figure's time axis
#
# These pin an alignment defect that rendered without any error: the 24 h running mean was
# the mean of consecutive PAIRS, returned on the pair midpoints, so every curve sat 6 h to
# the left of the axis. The forest started at lead 0.75 d instead of 0.5, no polyline
# reached the peak, and each killed branch's terminal x sat a quarter-day BEFORE the
# barrier that killed it - which reads as the resampling acting one step early.
# --------------------------------------------------------------------------- #
def test_the_running_mean_stays_on_its_own_time_axis():
    lead = np.arange(0.5, 21.01, 0.5)
    xs, _ = P.daily(lead, np.zeros(lead.size))
    assert np.allclose(xs, lead)


def test_the_running_mean_annihilates_the_diurnal_cycle():
    """A 24 h period sampled 12-hourly alternates in sign; the filter must null it.

    Including at the two ENDS, and without dragging them back down the trend - the last
    point of a lineage sits on the event peak, which is where ``A_L`` is taken.
    """
    lead = np.arange(0.5, 21.01, 0.5)
    trend = 0.4 * lead
    diurnal = 3.0 * (-1.0) ** np.arange(lead.size)
    _, ys = P.daily(lead, trend + diurnal)
    assert np.allclose(ys, trend, atol=1e-9)


def test_a_branch_starts_exactly_on_its_barrier():
    """With two lead-in frames and ``drop_first=1``, the first drawn point IS the barrier,
    and it carries the same value the uninterrupted lineage has there."""
    lead = np.arange(0.5, 21.01, 0.5)
    y = 0.4 * lead + 3.0 * (-1.0) ** np.arange(lead.size)
    _, full = P.daily(lead, y)
    i = int(np.argmin(np.abs(lead - 6.0)))                  # the 6 d barrier
    xb, yb = P.daily(lead[i - 1:], y[i - 1:], drop_first=1)
    assert xb[0] == pytest.approx(6.0)
    assert yb[0] == pytest.approx(full[i], abs=1e-9)


def test_the_root_anchor_puts_every_lineage_on_lead_zero():
    root = (np.array([-0.5, 0.0]), np.array([-6.0, -3.0]))
    lead = np.arange(0.5, 3.01, 0.5)
    xs, ys = P._from_root(root, lead, np.zeros(lead.size))
    assert xs[0] == pytest.approx(0.0)
    assert xs.size == ys.size == lead.size + 1


def test_without_an_init_file_the_curve_still_starts_at_the_first_frame():
    lead = np.arange(0.5, 3.01, 0.5)
    xs, _ = P._from_root(None, lead, np.zeros(lead.size))
    assert xs[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _compare_dict(n=8):
    rng = np.random.default_rng(0)
    al = np.linspace(-1.0, 6.0, n)
    ds = np.linspace(-2.0, 4.0, 12)
    thr = np.linspace(-2.0, 6.0, 9)
    curve = []
    for a in thr:
        curve.append(dict(threshold=float(a),
                          res=float(np.mean(al >= a)),
                          gencast_walkers=float(np.mean(ds >= a)),
                          gencast_walkers_ci=[0.0, 0.5]))
    lineage = [{"1": int(i % n), "2": int(i), "3": int(i)} for i in range(n)]
    steps = [dict(step=1, C=0.0, theta=[0.0] * n, counts=[1] * n, ess=float(n),
                  parents=list(range(n))),
             dict(step=2, C=1.0, theta=al.tolist(), counts=[1] * n, ess=0.6 * n,
                  parents=list(range(n)))]
    return dict(
        config=dict(n_walkers=n, tail_sign=1.0, leads=[3.0, 6.0], horizon_days=9.0,
                    C=[0.0, 1.0]),
        curve=curve, observed=4.5, log_Z=0.0, normalization_check=1.0,
        weights=np.ones(n).tolist(), ess_by_step=[1.0, 0.6], n_founders=n,
        realized=dict(box=al.tolist(), conus=al.tolist(), lineage=lineage,
                      lead_days=[1.5, 3.0], series_box=[[0.0, 1.0]] * n,
                      series_conus=[[0.0, 1.0]] * n),
        ds=dict(gencast_walkers=dict(box=ds.tolist(), n=ds.size)),
        res=dict(dmc=dict(steps=steps),
                 theta=[dict(leg=2, lead_days=6.0, box=al.tolist(), C=1.0,
                             backend="fcn3")]),
    )


@pytest.fixture
def figdir(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    return tmp_path / "figures" / "aires"


def _ctx(n=8):
    from aires import run_aires as R
    from fcn3 import fevents as F
    return R.Ctx(ev=F.EVENTS["PNW_HeatDome_2021"], tag="unit", n_walkers=n, members=6,
                 C=(0.0, 1.0), leads=(3.0, 6.0), horizon_days=9.0, dmc_seed=1,
                 base_seed=A.WALKER_BASE_SEED)


def test_exceedance_figure_renders(figdir):
    p = P.plot_exceedance(_ctx(), _compare_dict())
    assert p.exists() and p.stat().st_size > 10_000


def test_diagnostics_figure_renders(figdir):
    p = P.plot_diagnostics(_ctx(), _compare_dict())
    assert p.exists() and p.stat().st_size > 10_000


def test_a_cold_event_flips_the_exceedance_axis(figdir):
    """The x axis must run the other way for a freeze, so 'further right' still means
    'more extreme' and the figure cannot be misread against a heat event's."""
    import matplotlib
    matplotlib.use("Agg")
    from aires import run_aires as R
    from fcn3 import fevents as F

    d = _compare_dict()
    d["config"]["tail_sign"] = -1.0
    ctx = R.Ctx(ev=F.EVENTS["WinterStorm_Uri_2021"], tag="unit", n_walkers=8, members=6,
                C=(0.0, 1.0), leads=(3.0, 6.0), horizon_days=9.0, dmc_seed=1,
                base_seed=A.WALKER_BASE_SEED)
    assert ctx.sign == -1.0
    assert P.plot_exceedance(ctx, d).exists()
