"""Contract tests for the three new insights-deck figures.

The deck's whole claim is that nothing on it is a re-derivation and nothing on it is
scraped from prose. Five things would break that without looking broken:

- a **sigma-depth** taken from `aires/HANDOFF.md` instead of derived, or derived with the
  population sd, which moves PNW from +1.98 to +2.02 and Southwest from +3.32 to +3.39
  (the HANDOFF prose rounds that one to "~3.3", which is why the table below carries a
  per-row tolerance rather than one number for all nine);
- a **baseline coverage** table that assumes three direct-sampling rows, when four of the
  nine events carry exactly one and `ds_baseline.json` simply omits the others;
- a **probability printed without its normalization check**, which runs 0.568 to 1.769
  across the productions and 8.75 on the persistence control;
- a **stability verdict read off `runs/astab/stability.csv`'s `status` column**, which
  gives a GenCast failure at day 24 of the week-4 chain against a published verdict of
  "stable through week 6";
- a **persistence control averaged into the production slate**, where its `sum p_i` of
  8.75 against 0.57 on the production beside it would move any cross-event summary and
  its miss at PNW's own depth would close a reach boundary that is not closed.

All five are pinned here. No GPU, no network, no model; the tests that need a real run on
disk skip when it is not there, in the manner of `test_aceiling.py`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aires import aceiling as AC
from aires import aconfig as A
from aires import ainsights as I
from aires import awalkers as W
from aires.tests.test_awalkers import _compare_dict, _write

# The published ladder, `aires/HANDOFF.md` lines 181-187 and 205-212, with the precision
# it was published at. Six rows are printed to two decimals in the wave-2 table and are
# held to 0.01; three are prose approximations ("CA ~2.9 sigma", "SW ~3.3", "~2.2") and
# carry their own rounding, so they are held to 0.06. The point of the test is that the
# FORMULA reproduces the table, not that a rounded quote is exact.
PUBLISHED_SIGMA = {
    "PNW_HeatDome_2021": (+1.98, 0.01),
    "WinterStorm_Uri_2021": (+2.28, 0.01),
    "WinterStorm_Elliott_2022": (+2.81, 0.01),
    "p90_20251224": (+1.91, 0.01),
    "p90_20240802": (0.00, 0.01),
    "p90_20231107": (-0.97, 0.01),
    "California_HeatWave_2022": (+2.90, 0.06),
    "Southwest_HeatWave_2020": (+3.30, 0.06),
    "SCentral_HeatDome_2023": (+2.20, 0.06),
}

# `aires/INSIGHTS_PLAN.md` section 2.4, read off the files it cites.
PUBLISHED_COVERAGE = {
    "PNW_HeatDome_2021": dict(gencast_xres=24, gencast_walkers=16, fcn3=24),
    "WinterStorm_Uri_2021": dict(gencast_xres=24, gencast_walkers=16, fcn3=24),
    "SCentral_HeatDome_2023": dict(gencast_xres=None, gencast_walkers=16, fcn3=None),
    "California_HeatWave_2022": dict(gencast_xres=24, gencast_walkers=None, fcn3=None),
    "Southwest_HeatWave_2020": dict(gencast_xres=24, gencast_walkers=None, fcn3=None),
    "WinterStorm_Elliott_2022": dict(gencast_xres=24, gencast_walkers=None, fcn3=None),
    "p90_20231107": dict(gencast_xres=24, gencast_walkers=None, fcn3=24),
    "p90_20240802": dict(gencast_xres=24, gencast_walkers=None, fcn3=24),
    "p90_20251224": dict(gencast_xres=24, gencast_walkers=None, fcn3=24),
}

# How many events carry each slot. Stated separately from the map above on purpose: the
# map is nine rows anyone can retype wrong, these three totals are the sentence the deck
# actually prints, and a row edited in either place has to break one of them.
COVERAGE_TOTALS = dict(gencast_xres=8, gencast_walkers=3, fcn3=5)

# Every reduced run under `runs/aires`, hardest target first - `awalkers.order_runs` on
# the real tree. The two PNW rows are the SAME target at the same depth; the tie breaks
# on the tag, so the control is drawn immediately before the production it controls.
SLATE_ORDER = [
    ("Southwest_HeatWave_2020", "pilot"),
    ("California_HeatWave_2022", "pilot"),
    ("WinterStorm_Elliott_2022", "pilot"),
    ("WinterStorm_Uri_2021", "pilot"),
    ("SCentral_HeatDome_2023", "pilot"),
    ("PNW_HeatDome_2021", "persist"),
    ("PNW_HeatDome_2021", "pilot"),
    ("p90_20251224", "pilot"),
    ("p90_20240802", "pilot"),
    ("p90_20231107", "pilot"),
]


# --------------------------------------------------------------------------- #
def _ds_dict(box=None, conus=None, observed=1.0, observed_conus=None, sign=1.0,
             keys=("gencast_xres",)):
    """A dict shaped like a real ``ds_baseline.json``, with only the slots named."""
    box = np.linspace(-2.0, 4.0, 12) if box is None else np.asarray(box, dtype="float64")
    conus = box * 0.5 if conus is None else np.asarray(conus, dtype="float64")
    d = dict(event="unit", metric="t2m_anom", tail_sign=sign,
             observed=dict(box=float(observed),
                           conus=float(observed if observed_conus is None
                                       else observed_conus)))
    for k in keys:
        d[k] = dict(box=box.tolist(), conus=conus.tolist(), n=int(box.size))
    return d


def _write_ds(event, d):
    p = I.ds_baseline_path(event)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))
    return p


def _run(monkeypatch, tmp_path, event="PNW_HeatDome_2021", tag="unit", ds_keys=None,
         **kw):
    """One reduced run plus its baseline file, both under a temporary tree."""
    d = _compare_dict(**kw)
    if ds_keys is not None:
        d["ds"] = {k: v for k, v in d["ds"].items()
                   if k == "metric" or k in ds_keys}
    _write(monkeypatch, tmp_path, event, tag, d)
    _write_ds(event, _ds_dict(observed=d["observed"], sign=d["config"]["tail_sign"],
                              keys=tuple(k for k in I.BASELINE_KEYS
                                         if k in d["ds"])))
    return W.run_record(event, tag)


def _real_runs():
    """Every reduced run on this machine, or a skip. Reads `runs/aires`, writes nothing."""
    runs = W.discover()
    if not runs:
        pytest.skip(f"no reduced runs under {A.AIRES_ROOT}")
    return [W.run_record(ev, tag) for ev, tag in runs]


# --------------------------------------------------------------------------- #
# Nothing here is a re-derivation
# --------------------------------------------------------------------------- #
def test_the_module_imports_the_estimator_the_depth_and_the_interval():
    """Three single definitions, three imports. A local copy of any of them would drift
    from the number the run published the first time either side was touched."""
    from aires import alift as LF

    assert I.walker_mass is LF.walker_mass
    assert I.sigma_depth is W.sigma_depth
    assert I.wilson is AC.wilson


# --------------------------------------------------------------------------- #
# Baseline coverage - the trap the deck's first page exists to defuse
# --------------------------------------------------------------------------- #
def test_baseline_coverage_reports_the_slots_a_run_does_not_have(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path / "runs" / "aires")
    _write_ds("lonely", _ds_dict(keys=("gencast_xres",)))
    cov = I.baseline_coverage("lonely")
    assert cov["gencast_xres"] == 12
    assert cov["gencast_walkers"] is None and cov["fcn3"] is None
    assert cov["n_direct"] == 1
    assert cov["cfs"] is None                       # no CFS file written

    _write_ds("full", _ds_dict(keys=I.BASELINE_KEYS))
    assert I.baseline_coverage("full")["n_direct"] == 3


def test_baseline_coverage_reads_the_cfs_member_count(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path / "runs" / "aires")
    _write_ds("ev", _ds_dict())
    p = A.cfs_json_path("ev", 21.0, "t2m_anom")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(n_members=4, lead_days=21.0)))
    assert I.baseline_coverage("ev")["cfs"] == 4


def test_an_empty_slot_is_not_a_zero_member_ensemble(monkeypatch, tmp_path):
    """A key present but empty must read as absent, not as an ensemble of size 0 - the
    difference is "never run" against "ran and reached nothing"."""
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path / "runs" / "aires")
    d = _ds_dict(keys=("gencast_xres",))
    d["fcn3"] = dict(box=[], n=0)
    _write_ds("ev", d)
    cov = I.baseline_coverage("ev")
    assert cov["fcn3"] is None and cov["n_direct"] == 1


def test_a_missing_baseline_file_raises_rather_than_guessing(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "AIRES_ROOT", tmp_path / "runs" / "aires")
    with pytest.raises(FileNotFoundError):
        I.baseline_coverage("nothing_here")


# --------------------------------------------------------------------------- #
# The sigma-depth formula - the x axis of the whole deck
# --------------------------------------------------------------------------- #
def test_sigma_depth_divides_by_the_sample_sd_and_a_hand_case_proves_it():
    """Five members 0..4 against an observation of 6, so the arithmetic is checkable by
    hand and needs no run on disk.

    mean 2; ddof=1 sd is sqrt(2.5), ddof=0 sd is sqrt(2). The depth is 4/sqrt(2.5) =
    2.5298, and the population sd would report 2.8284 for the same five numbers. The two
    differ by exactly sqrt(n/(n-1)), which is the entire bug: it is a smooth rescaling of
    every marker on panel 2 at once, so no single number looks wrong.
    """
    d = _compare_dict(obs=6.0)
    d["ds"] = dict(metric="t2m_anom",
                   gencast_xres=dict(box=[0.0, 1.0, 2.0, 3.0, 4.0], n=5))
    got = W.sigma_depth(d)
    assert got == pytest.approx(4.0 / np.sqrt(2.5), rel=1e-12)
    assert got == pytest.approx(2.52982, abs=1e-5)
    population = 4.0 / np.sqrt(2.0)
    assert population == pytest.approx(2.82843, abs=1e-5)
    assert got != pytest.approx(population, rel=1e-6)
    assert population / got == pytest.approx(np.sqrt(5.0 / 4.0), rel=1e-12)


def test_the_ddof_guard_passes_on_the_real_import_and_fires_on_a_population_sd(
        monkeypatch):
    """`check_sigma_ddof1` is the deck's own tripwire. A tripwire that cannot be made to
    fire is decoration, so this test makes it fire.

    The memo list has to be cleared for the negative case: the guard is memoized on the
    grounds that an import does not change mid-process, which is true in production and
    false here."""
    monkeypatch.setattr(I, "_SIGMA_GUARD", [])
    I.check_sigma_ddof1()                        # the real import, no exception

    def population_sd(d):
        v = np.asarray(d["ds"]["gencast_xres"]["box"], dtype="float64")
        return float(d["config"]["tail_sign"]) * (d["observed"] - v.mean()) / v.std(ddof=0)

    monkeypatch.setattr(I, "_SIGMA_GUARD", [])
    monkeypatch.setattr(I, "sigma_depth", population_sd)
    with pytest.raises(AssertionError, match="ddof=1"):
        I.check_sigma_ddof1()


def test_the_guard_is_memoized_but_the_memo_does_not_swallow_a_real_failure(monkeypatch):
    """Memoizing a contract check is only safe if the memo is set AFTER it passes. Set
    before, the first call would arm the guard and every later one would return silently
    no matter what the import had become."""
    monkeypatch.setattr(I, "_SIGMA_GUARD", [])
    monkeypatch.setattr(I, "sigma_depth", lambda d: 0.0)
    with pytest.raises(AssertionError):
        I.check_sigma_ddof1()
    assert I._SIGMA_GUARD == []                  # a failed check must not arm the memo


def test_sigma_ddof0_is_the_population_ladder_and_is_never_what_a_record_reports(
        monkeypatch, tmp_path):
    """The caption quotes the shift, so the shift has to be derived. On an n-member direct
    ensemble the two depths differ by exactly sqrt(n / (n - 1)) and by nothing else."""
    r = _run(monkeypatch, tmp_path, obs=4.5)
    n = int(r["direct"].size)
    assert n == 12
    assert I.sigma_ddof0(r) / r["sigma_depth"] == pytest.approx(np.sqrt(n / (n - 1.0)),
                                                               rel=1e-12)
    assert I.sigma_ddof0(r) != pytest.approx(r["sigma_depth"], rel=1e-6)


def test_sigma_ddof0_returns_nan_rather_than_a_number_it_cannot_have():
    """One member is not a spread, and a zero-spread ensemble is a division by zero. Both
    must be nan, not 0.0 and not an exception: a 0.0 would draw a marker at the origin."""
    assert not np.isfinite(I.sigma_ddof0(dict(direct=np.array([1.0]), observed=2.0,
                                              sign=1.0)))
    assert not np.isfinite(I.sigma_ddof0(dict(direct=np.ones(5), observed=2.0, sign=1.0)))
    assert not np.isfinite(I.sigma_ddof0(dict(direct=np.arange(5.0), observed=None,
                                              sign=1.0)))


def test_normalization_range_separates_the_productions_from_the_control():
    """The control is not an outlier to be averaged in: it is the failure being SHOWN, so
    the range is the productions' and the control is named on its own. A mean over all
    four rows below would report 2.8, which is neither a production nor the control."""
    rows = [dict(control=False, total_mass=0.568, weight_ess=5.68, n=64),
            dict(control=False, total_mass=1.769, weight_ess=2.24, n=64),
            dict(control=False, total_mass=0.689, weight_ess=7.24, n=64),
            dict(control=True, total_mass=8.750, weight_ess=1.01, n=64)]
    v = I.normalization_range(rows)
    assert (v["n_prod"], v["n_control"]) == (3, 1)
    assert v["lo"] == pytest.approx(0.568) and v["hi"] == pytest.approx(1.769)
    assert v["control_mass"] == pytest.approx(8.750)
    assert v["control_ess"] == pytest.approx(1.01)
    assert v["control_n"] == 64
    # the control never moves the production range
    assert I.normalization_range(rows[:3])["hi"] == pytest.approx(v["hi"])


def test_normalization_range_with_no_control_says_nan_rather_than_zero():
    """A slate drawn without the control is a legitimate figure; a control mass of 0.0
    printed under it is a claim that a control ran and behaved perfectly."""
    v = I.normalization_range([dict(control=False, total_mass=0.7, weight_ess=5.0, n=64)])
    assert v["n_control"] == 0 and v["control_n"] == 0
    assert not np.isfinite(v["control_mass"]) and not np.isfinite(v["control_ess"])
    v = I.normalization_range([])
    assert not np.isfinite(v["lo"]) and not np.isfinite(v["hi"])


# --------------------------------------------------------------------------- #
# The ladder point: what a marker is allowed to assert
# --------------------------------------------------------------------------- #
def test_a_run_that_reached_plots_its_probability(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path)
    p = I.ladder_point(r)
    assert p["kind"] == "reached" and p["y"] == pytest.approx(r["P"])
    assert not p["control"]


def test_a_run_that_reached_nothing_plots_an_upper_bound_not_a_zero(monkeypatch, tmp_path):
    """``P`` is exactly 0 there, which cannot go on a log axis and is not the honest
    statement either. The deepest level the run resolved is."""
    r = _run(monkeypatch, tmp_path, event="Southwest_HeatWave_2020", obs=99.0)
    p = I.ladder_point(r)
    assert r["P"] == 0.0
    assert p["kind"] == "bound" and p["y"] == pytest.approx(r["deepest"]) and p["y"] > 0


def test_a_pinned_run_is_marked_pinned_and_its_y_is_the_normalization_check(
        monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, event="p90_20231107", obs=-99.0)
    p = I.ladder_point(r)
    assert p["kind"] == "pinned"
    assert p["y"] == pytest.approx(r["norm_check"], rel=1e-12)
    assert p["total_mass"] == pytest.approx(r["norm_check"], rel=1e-12)


def test_every_ladder_point_carries_its_own_normalization_check(monkeypatch, tmp_path):
    """A bare ``P`` in this experiment is misleading, so the point that draws one has to
    carry the check beside it or the figure cannot honour the rule."""
    r = _run(monkeypatch, tmp_path)
    p = I.ladder_point(r)
    assert p["total_mass"] == pytest.approx(r["table"]["p_i"].sum(), rel=1e-12)
    assert p["weight_ess"] == pytest.approx(r["weight_ess"])


def test_direct_sampling_gets_a_wilson_interval_and_nothing_else_does(monkeypatch,
                                                                     tmp_path):
    r = _run(monkeypatch, tmp_path, obs=3.0)
    p = I.ladder_point(r)
    assert (p["direct_lo"], p["direct_hi"]) == AC.wilson(p["direct_k"], p["direct_n"])
    assert p["direct_lo"] <= p["direct_p"] <= p["direct_hi"]
    assert "P_lo" not in p and "P_hi" not in p      # no invented interval on AI+RES


def test_the_reach_boundary_brackets_the_hits_and_ignores_the_control():
    """The control missed at PNW's own depth because the SCORE was swapped, not because
    the target was deeper. Counting it would close a gap that is not closed."""
    pts = [dict(x=1.98, kind="reached", control=False),
           dict(x=2.85, kind="reached", control=False),
           dict(x=3.32, kind="bound", control=False),
           dict(x=1.98, kind="bound", control=True)]
    assert I.reach_boundary(pts) == (2.85, 3.32)
    assert I.reach_boundary([p for p in pts if p["control"]])[0] != 1.98


def test_the_reach_boundary_says_nothing_when_a_side_is_empty():
    lo, hi = I.reach_boundary([dict(x=1.0, kind="reached", control=False)])
    assert np.isfinite(lo) and not np.isfinite(hi)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def test_label_offsets_fan_near_neighbours_highest_first():
    """Two points 0.075 sigma apart get one label above and one below, and it is the
    HIGHER point that goes above - the only ordering that cannot drop a label between
    two markers."""
    pts = [dict(x=1.906, y=0.0661), dict(x=1.981, y=0.0539), dict(x=-0.97, y=0.58)]
    offs = I.label_offsets(pts)
    assert offs[0]["va"] == "bottom" and offs[1]["va"] == "top"
    assert offs[2] == I.TOP_FAN[0]                   # alone, so the default slot
    # reversing the heights reverses which one goes above
    flip = I.label_offsets([dict(x=1.906, y=0.01), dict(x=1.981, y=0.9)])
    assert flip[0]["va"] == "top" and flip[1]["va"] == "bottom"


def test_label_offsets_uses_the_named_y_field():
    pts = [dict(x=2.808, y=0.008, total_mass=0.811),
           dict(x=2.849, y=0.019, total_mass=0.777)]
    by_p = I.label_offsets(pts, ykey="y")
    by_mass = I.label_offsets(pts, ykey="total_mass")
    assert by_p[0]["va"] == "top" and by_p[1]["va"] == "bottom"
    assert by_mass[0]["va"] == "bottom" and by_mass[1]["va"] == "top"


def test_box_extent_reads_a_360_degree_longitude_as_west():
    """`aindex` stores longitude on [0, 360); a reader places a box in degrees west. The
    CONUS box is the case that wraps at BOTH ends (235.0 and 294.0), so a half-applied
    conversion would print "235E-66W" and still look like a longitude range."""
    assert I.box_extent("PNW_HeatDome_2021") == "44N-50N, 124W-118W"
    assert I.box_extent("p90_20231107") == "24N-50N, 125W-66W"
    assert I.box_extent("WinterStorm_Uri_2021") == "26N-37N, 106W-94W"


# --------------------------------------------------------------------------- #
# The anchors
# --------------------------------------------------------------------------- #
def test_an_anchor_scores_the_index_it_was_asked_for(monkeypatch, tmp_path):
    """The box and the CONUS index are DIFFERENT quantities - INSIGHTS_PLAN section 3.5
    measures four of nine events moving to or below the median between them - so an
    anchor that silently scored the wrong one would look entirely plausible."""
    ev = "PNW_HeatDome_2021"
    d = _compare_dict(obs=4.5)
    d["realized"]["conus"] = (np.asarray(d["realized"]["box"]) * 0.25).tolist()
    _write(monkeypatch, tmp_path, ev, "unit", d)
    _write_ds(ev, _ds_dict(observed=4.5, observed_conus=0.6))

    box = I.anchor_record(ev, "unit", "box")
    conus = I.anchor_record(ev, "unit", "conus")
    assert box["observed"] == 4.5 and conus["observed"] == 0.6
    assert box["n_reached"] != conus["n_reached"]
    assert conus["index_label"].startswith("CONUS secondary")

    # and the AI+RES number is the estimator's own subset sum on that index
    al = np.asarray(d["realized"]["conus"])
    p = I.walker_mass(d)
    assert conus["P"] == pytest.approx(p[al >= 0.6].sum(), rel=1e-12)
    assert conus["total_mass"] == pytest.approx(p.sum(), rel=1e-12)


def test_an_anchor_carries_the_wilson_interval_and_says_whether_it_is_inside(
        monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, "PNW_HeatDome_2021", "unit", _compare_dict(obs=3.6))
    _write_ds("PNW_HeatDome_2021", _ds_dict(observed=3.6))
    a = I.anchor_record("PNW_HeatDome_2021", "unit", "box")
    assert (a["direct_lo"], a["direct_hi"]) == AC.wilson(a["direct_k"], a["direct_n"])
    assert a["inside"] == (a["direct_lo"] <= a["P"] <= a["direct_hi"])


def test_an_anchor_on_an_index_no_ensemble_carries_raises(monkeypatch, tmp_path):
    d = _compare_dict()
    _write(monkeypatch, tmp_path, "PNW_HeatDome_2021", "unit", d)
    ds = _ds_dict(observed=d["observed"], keys=("gencast_xres",))
    ds["gencast_xres"].pop("conus")
    _write_ds("PNW_HeatDome_2021", ds)
    with pytest.raises(KeyError):
        I.anchor_record("PNW_HeatDome_2021", "unit", "conus")


def test_the_two_anchors_are_the_ones_the_plan_names():
    assert I.ANCHORS == (("p90_20251224", "pilot", "box"),
                         ("WinterStorm_Elliott_2022", "pilot", "conus"))


# --------------------------------------------------------------------------- #
# The stability verdict - never the status column
# --------------------------------------------------------------------------- #
def _stability_csv(path):
    """A sweep table where the first ``FAIL`` and the reduce verdict disagree.

    The soft `t2m_anom_conus` panel fails at day 24 of the week-4 chain; no `bounds` or
    `nonfinite` row fails anywhere. Reading the status column gives "GenCast broke at
    24 d"; `astab.reduce` gives "clean at every tested week", which is the shape of the
    real disagreement `aires/HANDOFF.md` records.
    """
    rows = []
    for wk, days in ((4, (12.0, 24.0)), (6, (12.0, 24.0))):
        for ev in ("A", "B"):
            for day in days:
                soft = "FAIL" if (wk == 4 and ev == "A" and day == 24.0) else "ok"
                for metric, status in (("bounds", "ok"), ("nonfinite", "ok"),
                                       ("bounds_severity", "report"),
                                       ("t2m_anom_conus", soft)):
                    rows.append(dict(event=ev, weeks=wk, lead_days=wk * 7.0,
                                     model="gencast", checkpoint_day=day, metric=metric,
                                     value=0.0, band="", status=status, extrapolated=0,
                                     reached_peak=1, valid_time="2021-06-28"))
    # a lead the walker never rolled to the peak: it must be dropped, not counted clean
    for metric in ("bounds", "nonfinite", "bounds_severity"):
        rows.append(dict(event="A", weeks=8, lead_days=56.0, model="gencast",
                         checkpoint_day=3.0, metric=metric, value=0.0, band="",
                         status="ok", extrapolated=0, reached_peak=0,
                         valid_time="2021-06-28"))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_the_stability_verdict_comes_from_reduce_and_not_from_the_status_column(tmp_path):
    csv = _stability_csv(tmp_path / "stability.csv")
    df = pd.read_csv(csv)
    # what reading the column directly would say
    first_fail = df[(df.model == "gencast") & (df.status == "FAIL")].checkpoint_day.min()
    assert first_fail == 24.0
    v = I.stability_verdict(csv)
    assert v["gencast_weeks"] == 6                  # not "broken at 24 d"
    assert v["gencast"]["tested"] == [4, 6]         # week 8 dropped, not counted clean
    assert v["gencast"]["untested"] == [8]


def test_the_stability_verdict_reports_a_model_with_no_rows_as_nothing(tmp_path):
    csv = _stability_csv(tmp_path / "stability.csv")
    v = I.stability_verdict(csv)
    assert v["fcn3_era5"]["clean_through"] is None and v["fcn3_weeks"] is None


# --------------------------------------------------------------------------- #
# The slate table
# --------------------------------------------------------------------------- #
def test_the_slate_is_ordered_hardest_first_and_carries_the_coverage(monkeypatch,
                                                                    tmp_path):
    shallow = _run(monkeypatch, tmp_path, event="p90_20240802", tag="unit", obs=0.5,
                   ds_keys=("gencast_xres",))
    deep = _run(monkeypatch, tmp_path, event="Southwest_HeatWave_2020", tag="unit",
                obs=3.9, ds_keys=("gencast_xres",))
    t = I.slate_table([shallow, deep])
    assert list(t["event"]) == ["Southwest_HeatWave_2020", "p90_20240802"]
    assert set(t["n_direct"]) == {1}
    assert t["total_mass"].iloc[0] == pytest.approx(deep["table"]["p_i"].sum(), rel=1e-12)
    # the provenance columns carry VALUES, not just headers: the extent is the box
    # `aindex` scored, converted out of [0, 360)
    assert list(t["extent"]) == ["29N-35N, 109W-103W", "24N-50N, 125W-66W"]
    assert list(t["control"]) == [False, False]
    assert list(t["direct_n"]) == [12, 12]
    # a config with no init/peak yields an empty cell, never the string "None"
    assert list(t["init"]) == ["", ""] and list(t["peak"]) == ["", ""]


# --------------------------------------------------------------------------- #
# Rendering (smoke only)
# --------------------------------------------------------------------------- #
def test_the_three_figures_render(monkeypatch, tmp_path):
    hot = _run(monkeypatch, tmp_path, event="PNW_HeatDome_2021", tag="unit")
    cold = _run(monkeypatch, tmp_path, event="WinterStorm_Uri_2021", tag="unit",
                sign=-1.0, obs=-0.5, al=-np.linspace(-1.0, 6.0, 8))
    miss = _run(monkeypatch, tmp_path, event="Southwest_HeatWave_2020", tag="unit",
                obs=99.0)
    recs = [hot, cold, miss]
    for fn, name in ((I.plot_slate, "s.png"), (I.plot_ladder, "l.png")):
        out = fn(recs, tmp_path / name)
        assert out == tmp_path / name and out.stat().st_size > 10_000
    anchors = [I.anchor_record("PNW_HeatDome_2021", "unit", "box"),
               I.anchor_record("WinterStorm_Uri_2021", "unit", "conus")]
    out = I.plot_anchors(anchors, tmp_path / "a.png")
    assert out == tmp_path / "a.png" and out.stat().st_size > 10_000


def test_the_deck_writes_its_figures_under_figures_and_never_under_runs(monkeypatch,
                                                                        tmp_path):
    """`_save` creates its parent directory, so a default output path that pointed into
    the run tree would quietly manufacture folders inside `runs/aires`. The deck reads
    `runs/`; it writes `figures/`."""
    monkeypatch.setattr(A, "ROOT", tmp_path)
    default = A.fig_dir()
    assert default == tmp_path / "figures" / "aires"
    assert "runs" not in default.parts
    assert not (tmp_path / "runs").exists()


def test_the_ladder_renders_with_only_a_pinned_run(monkeypatch, tmp_path):
    """The leftmost run of the real slate. Its ``P`` is a normalization check, so the
    figure must still draw and must still say so."""
    r = _run(monkeypatch, tmp_path, event="p90_20231107", tag="unit", obs=-99.0)
    assert I.plot_ladder([r], tmp_path / "p.png").stat().st_size > 10_000


# --------------------------------------------------------------------------- #
# Against what was published (needs the real run tree; skips without it)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("event", sorted(PUBLISHED_SIGMA))
def test_the_derived_sigma_depth_reproduces_the_published_ladder(event):
    """`sigma = tail_sign * (obs - mean(direct)) / std(direct, ddof=1)`, from the run's
    own files. The published table is a QUOTE of this number, not its source."""
    path = A.res_dir(event, "pilot") / "compare.json"
    if not path.exists():
        pytest.skip(f"no run at {path}")
    got = W.sigma_depth(json.loads(path.read_text()))
    published, tol = PUBLISHED_SIGMA[event]
    assert got == pytest.approx(published, abs=tol)


@pytest.mark.parametrize("event,ddof0", [("PNW_HeatDome_2021", 2.02),
                                         ("Southwest_HeatWave_2020", 3.39)])
def test_ddof_1_is_load_bearing(event, ddof0):
    """The population sd is a different ladder. It moves PNW past 2 sigma and Southwest
    past 3.35, and every marker on panel 2 with it."""
    path = A.res_dir(event, "pilot") / "compare.json"
    if not path.exists():
        pytest.skip(f"no run at {path}")
    d = json.loads(path.read_text())
    _, v = W.direct_ensemble(d)
    sign = float(d["config"]["tail_sign"])
    got = sign * (float(d["observed"]) - v.mean()) / np.std(v, ddof=0)
    assert got == pytest.approx(ddof0, abs=0.01)
    assert abs(got - W.sigma_depth(d)) > 0.03
    # and the two published numbers are not two independent constants: on a 24-member
    # baseline the shift is exactly sqrt(n / (n - 1)), the same factor on both events
    assert v.size == 24
    assert got / W.sigma_depth(d) == pytest.approx(np.sqrt(24.0 / 23.0), rel=1e-12)


@pytest.mark.parametrize("event", sorted(PUBLISHED_COVERAGE))
def test_the_baseline_coverage_on_disk_is_the_plans_table(event):
    if not I.ds_baseline_path(event).exists():
        pytest.skip(f"no {I.ds_baseline_path(event)}")
    cov = I.baseline_coverage(event)
    want = PUBLISHED_COVERAGE[event]
    assert {k: cov[k] for k in I.BASELINE_KEYS} == want
    assert cov["n_direct"] == sum(1 for v in want.values() if v)


def test_the_coverage_map_is_eight_five_and_three_with_all_three_only_twice():
    """The map above is nine rows; the deck prints three totals off it. Stating both and
    deriving one from the other is what stops a mistyped row from surviving a review."""
    got = {k: sum(1 for c in PUBLISHED_COVERAGE.values() if c[k]) for k in I.BASELINE_KEYS}
    assert got == COVERAGE_TOTALS
    assert [e for e, c in PUBLISHED_COVERAGE.items() if not c["gencast_xres"]] == [
        "SCentral_HeatDome_2023"]
    assert sorted(e for e, c in PUBLISHED_COVERAGE.items()
                  if all(c[k] for k in I.BASELINE_KEYS)) == ["PNW_HeatDome_2021",
                                                             "WinterStorm_Uri_2021"]


def test_the_coverage_on_disk_totals_eight_five_and_three():
    """The same three totals, counted off the nine `ds_baseline.json` files.

    The per-event test above parametrizes, so it skips one event at a time and would stay
    green with eight of the nine files missing. This one counts them, which is the form
    the sentence on page one is actually printed in: eight events have the 24-member xres
    cube, five have FCN3, three have the Gate 3 walkers, and only PNW and Uri have all
    three. CFSv2 is the one uniform slot, four cycles on every event.
    """
    present = [e for e in PUBLISHED_COVERAGE if I.ds_baseline_path(e).exists()]
    if len(present) < len(PUBLISHED_COVERAGE):
        pytest.skip("not every event's ds_baseline.json is on disk")
    cov = {e: I.baseline_coverage(e) for e in present}
    assert {k: sum(1 for c in cov.values() if c[k]) for k in I.BASELINE_KEYS} == (
        COVERAGE_TOTALS)
    assert [e for e, c in cov.items() if not c["gencast_xres"]] == ["SCentral_HeatDome_2023"]
    assert sorted(e for e, c in cov.items() if c["n_direct"] == 3) == [
        "PNW_HeatDome_2021", "WinterStorm_Uri_2021"]
    assert {c["cfs"] for c in cov.values()} == {4}


def test_four_events_carry_exactly_one_direct_ensemble_not_three():
    """INSIGHTS_PLAN's Panel 1 prose says three events have only one baseline row; its
    own coverage table in section 2.4 says four, and the files agree with the table.
    Derived, so the deck cannot inherit the wrong sentence."""
    present = [e for e in PUBLISHED_COVERAGE if I.ds_baseline_path(e).exists()]
    if len(present) < len(PUBLISHED_COVERAGE):
        pytest.skip("not every event's ds_baseline.json is on disk")
    lonely = [e for e in present if I.baseline_coverage(e)["n_direct"] == 1]
    assert sorted(lonely) == ["California_HeatWave_2022", "SCentral_HeatDome_2023",
                              "Southwest_HeatWave_2020", "WinterStorm_Elliott_2022"]


@pytest.mark.parametrize("event,tag,index,published_P,sigma,mass,ratio", [
    ("p90_20251224", "pilot", "box", 0.0661, 1.91, 0.689, 1.59),
    ("WinterStorm_Elliott_2022", "pilot", "conus", 0.136, 1.89, 0.811, 3.26),
])
def test_the_calibration_anchors_reproduce_the_published_numbers(
        event, tag, index, published_P, sigma, mass, ratio):
    """`aires/HANDOFF.md` lines 324-331 and 431-435, the two rungs where direct sampling
    still returns a non-zero count in the tail.

    Both direct counts are 1 of 24, so both carry the IDENTICAL Wilson interval and the
    whole comparison rests on where AI+RES lands inside it: 0.0661 against 0.0417 on
    p90_20251224 and 0.136 against the same 0.0417 on Elliott's CONUS index, i.e. 1.6x
    and 3.3x the direct rate. The normalization check is pinned beside each one, because
    a probability quoted without it is the failure this deck's own rule forbids.

    Parametrized rather than looped: a missing run must skip its own row and not take the
    other anchor's assertions down with it, which is what the loop form did.
    """
    if not (A.res_dir(event, tag) / "compare.json").exists():
        pytest.skip(f"no {event} run on disk")
    a = I.anchor_record(event, tag, index)
    assert a["index"] == index
    assert a["direct_k"] == 1 and a["direct_n"] == 24
    assert a["direct_p"] == pytest.approx(1.0 / 24.0, rel=1e-12)
    assert a["direct_p"] == pytest.approx(0.0417, abs=0.0001)
    assert (a["direct_lo"], a["direct_hi"]) == AC.wilson(1, 24)
    assert a["direct_lo"] == pytest.approx(0.00739, abs=1e-5)
    assert a["direct_hi"] == pytest.approx(0.202, abs=0.001)
    assert a["P"] == pytest.approx(published_P, abs=0.0005)
    assert a["sigma"] == pytest.approx(sigma, abs=0.01)
    assert a["total_mass"] == pytest.approx(mass, abs=0.001)
    assert a["P"] / a["direct_p"] == pytest.approx(ratio, abs=0.01)
    assert a["inside"] and a["P"] > a["direct_p"]


def test_elliotts_conus_anchor_is_not_elliotts_score():
    """Panel 6 labels it a SECOND anchor. On its own N Plains box Elliott is a +2.81
    sigma target that direct sampling misses outright; the two must never be confused."""
    path = A.res_dir("WinterStorm_Elliott_2022", "pilot") / "compare.json"
    if not path.exists():
        pytest.skip(f"no run at {path}")
    box = I.anchor_record("WinterStorm_Elliott_2022", "pilot", "box")
    conus = I.anchor_record("WinterStorm_Elliott_2022", "pilot", "conus")
    assert box["direct_k"] == 0 and conus["direct_k"] == 1
    assert box["sigma"] > conus["sigma"] > 1.5
    assert box["P"] < conus["P"]


# --------------------------------------------------------------------------- #
# The production slate, and the one run that is not part of it
# --------------------------------------------------------------------------- #
def test_the_walker_csv_is_ten_runs_of_sixty_four_walkers():
    """`runs/aires/aires_walkers.csv` is the table every published per-walker number is
    read off, and its shape is a contract: ten (event, tag) pairs, 64 walkers each, 640
    rows.

    A short group is a run whose reduce stopped partway, and it is silent: the masses
    still sum to that run's own `Z`, so `sum p_i` still looks like a normalization check
    and every subset sum off it is quietly conditioned on a truncated population.
    """
    csv = A.AIRES_ROOT / "aires_walkers.csv"
    if not csv.exists():
        pytest.skip(f"no {csv}")
    g = pd.read_csv(csv).groupby(["event", "tag"]).size()
    assert len(g) == 10
    assert set(g) == {64}
    assert int(g.sum()) == 640
    assert sorted(map(tuple, g.index)) == sorted(SLATE_ORDER)


def test_the_persistence_control_is_the_one_run_not_on_the_production_slate():
    """PNW carries TWO runs at the same +1.98 sigma target: the production and job 1180's
    persistence control, which differs only in the score function.

    The control has to be on the deck and has to be excluded from every statement the
    deck makes about the productions. Its ``sum p_i`` is 8.75 against 0.568 on the
    production standing next to it, so anything that averages the ten runs is moved by
    the control alone; and it misses at PNW's own depth because the SCORE was swapped,
    not because the target was deeper, so counting its miss would close a reach boundary
    that is not closed. `is_control` reads ``config.backend``, never the tag.
    """
    recs = _real_runs()
    assert len(recs) == 10
    ctrl = [r for r in recs if I.is_control(r)]
    assert [(r["event"], r["tag"]) for r in ctrl] == [("PNW_HeatDome_2021", "persist")]
    assert str(ctrl[0]["d"]["config"]["backend"]) == "persistence"

    prod = [r for r in recs if not I.is_control(r)]
    assert len(prod) == 9
    assert sorted(r["event"] for r in prod) == sorted(PUBLISHED_SIGMA)
    assert {str(r["d"]["config"].get("backend", "fcn3")) for r in prod} == {"fcn3"}

    # on the ladder it is drawn, flagged, and ignored by the boundary
    pts = [I.ladder_point(r) for r in recs]
    c = next(p for p in pts if p["control"])
    assert c["kind"] == "bound" and c["x"] == pytest.approx(1.98, abs=0.01)
    assert I.reach_boundary(pts) == pytest.approx(
        I.reach_boundary([p for p in pts if not p["control"]]))


def test_the_reach_boundary_on_the_real_slate_is_california_to_southwest():
    """Deepest target AI+RES reached at all, shallowest it did not: +2.85 (California) to
    +3.32 (Southwest). The control's miss at +1.98 sits well inside that gap and must not
    move either edge."""
    pts = [I.ladder_point(r) for r in _real_runs()]
    hit, miss = I.reach_boundary(pts)
    assert (hit, miss) == pytest.approx((2.849, 3.321), abs=0.001)
    by_kind = {p["label"]: p["kind"] for p in pts if not p["control"]}
    assert by_kind["California Heat Wave 2022"] == "reached"
    assert by_kind["Southwest Heat Wave 2020"] == "bound"
    # the control is the shallowest miss on the whole ladder, and it is NOT the boundary
    assert min(p["x"] for p in pts if p["kind"] == "bound") == pytest.approx(1.98,
                                                                            abs=0.01)
    assert hit < miss


def test_the_real_slate_is_ordered_hardest_first_and_the_pnw_tie_breaks_on_the_tag():
    """One ordering for the whole deck, `awalkers.order_runs`. The PNW pair is the case
    that matters: identical depth, so only the tag separates them, and the control has to
    land next to the production it controls rather than somewhere else on the ladder."""
    t = I.slate_table(_real_runs())
    assert list(zip(t["event"], t["tag"])) == SLATE_ORDER
    pnw = t[t["event"] == "PNW_HeatDome_2021"]
    assert pnw["sigma"].nunique() == 1
    assert list(pnw["control"]) == [True, False]
    assert list(pnw["tag"]) == ["persist", "pilot"]
    # depth falls monotonically down the table, which is what makes it readable as a ladder
    assert list(t["sigma"]) == sorted(t["sigma"], reverse=True)


def test_every_production_normalization_check_is_order_one_and_the_control_is_not():
    """`sum p_i` across the slate: 0.568 to 1.769 on the nine productions, 8.75 on the
    control. That spread is the reason the deck's rule is that no probability is printed
    without its check beside it - at 8.75 the number has stopped being a normalization
    check at all, and the ``P`` derived from it is not a probability."""
    recs = _real_runs()
    prod = sorted(r["total_mass"] for r in recs if not I.is_control(r))
    ctrl = [r["total_mass"] for r in recs if I.is_control(r)]
    assert len(prod) == 9
    assert prod[0] == pytest.approx(0.568, abs=0.001)
    assert prod[-1] == pytest.approx(1.769, abs=0.001)
    assert ctrl == pytest.approx([8.750], abs=0.001)
    assert prod[-1] < 2.0 < ctrl[0]
    # and the check is the run's own, not a re-derivation
    for r in recs:
        assert r["total_mass"] == pytest.approx(r["table"]["p_i"].sum(), rel=1e-12)


def test_the_weight_ess_across_the_slate_runs_to_eighteen_not_to_nine():
    """`aires/HANDOFF.md` line 621 publishes four of the ten weight ESS values (PNW 5.68,
    Uri 2.24, the p90 anchor 7.24, the control 1.01), and a reader who takes the largest
    quoted number as the slate maximum lands on Elliott's 8.61.

    The six unquoted runs include three that are larger: South-Central 11.40, California
    11.03 and Southwest 18.81. All ten are pinned here so that no caption or summary can
    describe 8.61 as the top of a range that contains them.
    """
    recs = _real_runs()
    ess = {f"{r['event']}:{r['tag']}": r["weight_ess"] for r in recs}
    assert min(ess.values()) == pytest.approx(1.014, abs=0.001)
    assert max(ess.values()) == pytest.approx(18.805, abs=0.001)
    assert ess["PNW_HeatDome_2021:persist"] == pytest.approx(1.01, abs=0.01)
    assert ess["PNW_HeatDome_2021:pilot"] == pytest.approx(5.68, abs=0.01)
    assert ess["WinterStorm_Uri_2021:pilot"] == pytest.approx(2.24, abs=0.01)
    assert ess["p90_20251224:pilot"] == pytest.approx(7.24, abs=0.01)
    assert ess["WinterStorm_Elliott_2022:pilot"] == pytest.approx(8.61, abs=0.01)
    assert sum(1 for v in ess.values() if v > 8.61) == 3


def test_the_stability_verdict_on_disk_is_the_published_one():
    """`aires/HANDOFF.md`: GenCast clean through week 6, FCN3 clean through week 14 on
    BOTH arms. Weeks 12 to 20 are the FCN3-only extension, where the walker rolls one
    segment and never reaches the peak, so GenCast reports them untested rather than
    clean - three days of rollout must not read as a clean 140-day chain."""
    from astab import reduce as R

    if not R.csv_path().exists():
        pytest.skip(f"no {R.csv_path()}")
    v = I.stability_verdict()
    assert v["gencast_weeks"] == 6
    assert v["fcn3_weeks"] == 14
    assert v["fcn3_era5"]["clean_through"] == 14
    assert v["fcn3_adapter"]["clean_through"] == 14
    assert v["gencast"]["tested"] == [4, 6, 8, 10]
    assert v["gencast"]["untested"] == [12, 14, 16, 18, 20]
