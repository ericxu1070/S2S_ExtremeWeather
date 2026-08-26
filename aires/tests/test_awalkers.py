"""Contract tests for the per-walker figure.

The figure's whole claim is that nothing on it is a re-derivation: the mass attached to
each walker is the estimator's own, and every probability the experiment publishes is a
subset sum of the vector the panel draws. Two things would break that claim without
looking broken - a mass that no longer sums to the run's normalization check, and a rank
order that ignores the tail sign, which on a freeze would rank the WARMEST walker first
and quietly report the wrong tail. Both are pinned here.

No GPU, no network, no model, no real run on disk.
"""
from __future__ import annotations

import numpy as np
import pytest

from aires import aconfig as A
from aires import awalkers as W
from aires import dmc as D


# --------------------------------------------------------------------------- #
def _compare_dict(n=8, sign=1.0, obs=4.5, log_Z=-0.3, weights=None, al=None):
    """A dict shaped like a real ``compare.json``, small enough to check by hand."""
    al = np.linspace(-1.0, 6.0, n) if al is None else np.asarray(al, dtype="float64")
    w = np.linspace(0.2, 2.0, n) if weights is None else np.asarray(weights, dtype="float64")
    ds = np.linspace(-2.0, 4.0, 12)
    return dict(
        config=dict(n_walkers=n, tail_sign=sign, leads=[3.0, 6.0], horizon_days=9.0,
                    C=[0.0, 1.0]),
        observed=obs, log_Z=log_Z, weights=w.tolist(),
        normalization_check=float(np.exp(log_Z) * w.mean()),
        n_founders=3, ancestry=[i % 3 for i in range(n)],
        realized=dict(box=al.tolist(), conus=al.tolist()),
        ds=dict(metric="t2m_anom",
                gencast_xres=dict(box=ds.tolist(), n=ds.size),
                gencast_walkers=dict(box=(ds[:6] * 0.5).tolist(), n=6),
                fcn3=dict(box=(ds * 2).tolist(), n=ds.size)),
    )


# --------------------------------------------------------------------------- #
# The mass is the estimator's, not a copy of it
# --------------------------------------------------------------------------- #
def test_the_masses_sum_to_the_runs_normalization_check():
    """``sum_i p_i`` IS ``DMCResult.normalization_check()``. If the figure's total and
    the number printed in `compare.json` ever disagree, one of them is lying."""
    d = _compare_dict()
    t = W.walker_table("PNW_HeatDome_2021", "unit", d)
    assert t["p_i"].sum() == pytest.approx(d["normalization_check"], rel=1e-12)


def test_the_reached_subset_sum_is_the_dmc_estimator():
    """``P(A_L >= obs)`` off the panel must equal ``DMCResult.expectation`` of the same
    indicator - the figure adds up bars, the experiment multiplies Z by a mean, and they
    have to be the same arithmetic."""
    d = _compare_dict()
    al = np.asarray(d["realized"]["box"])
    r = D.DMCResult(n_walkers=al.size, C=(0.0,))
    r.final_V = -np.log(np.asarray(d["weights"]))
    r.log_Z = d["log_Z"]
    t = W.walker_table("PNW_HeatDome_2021", "unit", d)
    assert (t.loc[t["reached"], "p_i"].sum()
            == pytest.approx(r.expectation(al >= d["observed"]), rel=1e-12))


def test_the_module_imports_the_estimator_rather_than_reimplementing_it():
    """`alift.walker_mass` is the single definition of ``p_i``. A local copy here would
    drift from the lift figure the first time either is touched."""
    from aires import alift as LF
    assert W.walker_mass is LF.walker_mass


# --------------------------------------------------------------------------- #
# The tail sign
# --------------------------------------------------------------------------- #
def test_a_cold_run_ranks_the_coldest_walker_first():
    """Rank 1 is the most extreme IN THE TAIL DIRECTION. Unsigned, a freeze would rank
    its warmest walker first and the running total would build the wrong tail."""
    d = _compare_dict(sign=-1.0, obs=-0.5)
    t = W.walker_table("WinterStorm_Uri_2021", "unit", d).sort_values("rank")
    assert t["A_L"].iloc[0] == pytest.approx(min(d["realized"]["box"]))
    assert bool(t["reached"].iloc[0])                       # coldest reaches a cold target


def test_reached_uses_the_signed_comparison():
    # 0.5 sits between two walkers on purpose: the comparison is `>=`, so a walker
    # exactly ON the threshold would count for BOTH tails and the partition below would
    # not be a partition.
    hot = W.walker_table("PNW_HeatDome_2021", "unit", _compare_dict(sign=+1.0, obs=0.5))
    cold = W.walker_table("WinterStorm_Uri_2021", "unit", _compare_dict(sign=-1.0, obs=0.5))
    # the same population against the same threshold, opposite tails - and every walker
    # is on exactly one side of it
    assert int(hot["reached"].sum()) + int(cold["reached"].sum()) == hot.shape[0]
    assert int(hot["reached"].sum()) and int(cold["reached"].sum())


# --------------------------------------------------------------------------- #
# The running total, the ties, and the failure the panel exists to expose
# --------------------------------------------------------------------------- #
def test_cum_p_is_the_exceedance_curve_at_the_populations_own_points():
    d = _compare_dict()
    t = W.walker_table("PNW_HeatDome_2021", "unit", d)
    mass, al = t["p_i"].to_numpy(), t["A_L"].to_numpy()
    for i in range(t.shape[0]):
        assert t["cum_p"].iloc[i] == pytest.approx(mass[al >= al[i]].sum(), rel=1e-12)


def test_identical_weights_are_grouped_as_one_clone_family():
    """Walkers cloned from one parent at the last resampling inherit its ``V_K`` and so
    carry identical mass; the panel counts distinct masses, not walkers."""
    d = _compare_dict(n=6, weights=[1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    t = W.walker_table("PNW_HeatDome_2021", "unit", d)
    assert t["family"].nunique() == 3
    assert sorted(t.groupby("family")["family_size"].first()) == [1, 2, 3]


def test_a_target_below_every_walker_is_flagged_pinned(monkeypatch, tmp_path):
    """The silent failure `p90_20231107` hit: the indicator is identically 1, so the
    reported probability collapses onto the normalization check and is not a probability
    at all. `compare.json` alone shows a plausible number; the record must say otherwise."""
    d = _compare_dict(obs=-99.0)
    _write(monkeypatch, tmp_path, "PNW_HeatDome_2021", "unit", d)
    r = W.run_record("PNW_HeatDome_2021", "unit")
    assert r["pinned"] and r["n_reached"] == r["n"]
    assert r["P"] == pytest.approx(r["norm_check"], rel=1e-12)


def test_an_ordinary_target_is_not_flagged_pinned(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, "PNW_HeatDome_2021", "unit", _compare_dict())
    assert not W.run_record("PNW_HeatDome_2021", "unit")["pinned"]


# --------------------------------------------------------------------------- #
# Baselines and labels
# --------------------------------------------------------------------------- #
def test_the_direct_baseline_prefers_the_xres_cube():
    """The wave tables quote the 24-member xres cube; Gate 3 walkers and FCN3 are
    fallbacks. Silently comparing against a different ensemble would move every
    sigma-depth on the figure."""
    key, v = W.direct_ensemble(_compare_dict())
    assert key == "gencast_xres"
    d = _compare_dict()
    d["ds"].pop("gencast_xres")
    assert W.direct_ensemble(d)[0] == "gencast_walkers"
    d["ds"].pop("gencast_walkers")
    assert W.direct_ensemble(d)[0] == "fcn3"
    d["ds"] = {}
    key, empty = W.direct_ensemble(d)
    assert key is None and empty.size == 0
    assert not np.isfinite(W.sigma_depth(d))       # no baseline => no depth, not a guess
    assert v.size == 12


def test_sigma_depth_is_signed_by_the_tail():
    ds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    d = _compare_dict(obs=6.0)
    d["ds"] = dict(metric="t2m_anom", gencast_xres=dict(box=ds.tolist(), n=ds.size))
    assert W.sigma_depth(d) > 0
    d["config"]["tail_sign"] = -1.0
    assert W.sigma_depth(d) < 0                 # a hot target is not deep on a cold tail


def test_weight_ess_bounds():
    assert W.weight_ess(np.ones(10)) == pytest.approx(10.0)
    assert W.weight_ess([1.0] + [0.0] * 9) == pytest.approx(1.0)


def test_p90_labels_drop_the_selection_number_but_named_events_keep_theirs():
    """The p90 parenthetical is the PEAK-DAY CONUS anomaly the case was selected on, not
    the 7-day box mean the panel reports - two different observables on one panel."""
    assert "(" not in W.panel_label("p90_20231107")
    assert W.panel_label("PNW_HeatDome_2021") == "PNW Heat Dome 2021"


# --------------------------------------------------------------------------- #
# Rendering (smoke only - that a figure comes out from a real-shaped dict)
# --------------------------------------------------------------------------- #
def _write(monkeypatch, tmp_path, event, tag, d):
    import json
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path / "runs" / "aires")
    p = A.res_dir(event, tag)
    p.mkdir(parents=True, exist_ok=True)
    (p / "compare.json").write_text(json.dumps(d))
    return p


def test_the_cross_event_figure_renders(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, "PNW_HeatDome_2021", "unit", _compare_dict())
    _write(monkeypatch, tmp_path, "WinterStorm_Uri_2021", "unit",
           _compare_dict(sign=-1.0, obs=-0.5, al=-np.linspace(-1.0, 6.0, 8)))
    recs = [W.run_record(e, "unit")
            for e in ("PNW_HeatDome_2021", "WinterStorm_Uri_2021")]
    out = W.plot_walkers(recs, tmp_path / "w.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_the_per_run_figure_renders_for_a_cold_event(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, "WinterStorm_Uri_2021", "unit",
           _compare_dict(sign=-1.0, obs=-0.5, al=-np.linspace(-1.0, 6.0, 8)))
    out = W.plot_run(W.run_record("WinterStorm_Uri_2021", "unit"), tmp_path / "r.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_a_run_with_no_reach_still_renders(monkeypatch, tmp_path):
    """Southwest and the persistence control: the observation lies beyond every walker,
    so the band that counts is empty. It must still be DRAWN, on the correct side."""
    _write(monkeypatch, tmp_path, "Southwest_HeatWave_2020", "unit",
           _compare_dict(obs=99.0))
    r = W.run_record("Southwest_HeatWave_2020", "unit")
    assert r["n_reached"] == 0 and r["P"] == 0.0 and r["deepest"] > 0
    assert W.plot_walkers([r], tmp_path / "n.png").exists()
