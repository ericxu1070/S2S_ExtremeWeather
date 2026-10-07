"""Tests for the model-ceiling reduction.

Three of the things this module can get wrong are silent, so each is pinned here:

* **the cold-tail sign.** Every event in the low tail is compared with ``>=`` against a
  NEGATIVE threshold. An inverted `beyond` still returns a plausible-looking fraction, and
  the two events where it matters (Uri, Elliott) are the two carrying the finding.
* **the recomputed probability.** `res_probability` re-derives `P` from `realized.box`,
  `weights` and `log_Z` rather than reading `compare.json`'s curve. If it drifts from what
  the runs actually reported, the whole comparison is against a number nobody published;
  the wave-2 table is therefore pinned digit for digit.
* **the ex-self pool.** Dropping the event's own year from the numerator but not the
  denominator (or vice versa) shifts every climatological rate by ~1/64 - small enough to
  survive review, large enough to move a lift of 0.9 across 1.0.

And three things this module must keep REFUSING to do, each of which a later edit could
quietly restore:

* **quote a GEV return period.** `aclim.gev_return_period` still exists and is still
  cheap to call. It is fitted on 30 to 64 block maxima, three of the nine events sit at or
  beyond the sample maximum, and a nonparametric resample of SCentral 2023's 64 blocks
  puts its "4086 year" figure in a 90% interval of [98, 352357] years with 30% of
  resamples returning infinity. `test_no_return_period_is_quoted_anywhere` keeps it out.
* **report a probability where the estimator has no resolution.** At 64/64 walkers beyond
  the target the indicator is identically 1 and `P` IS the normalization check
  (`p90_20231107`: 0.582662 both ways, direct sampling 0.875). The number is pinned, the
  classification is pinned, and the refusal to divide it by a base rate is pinned.
* **score an event against another region's climatology.** The `aclim` cache reduces
  whatever `aindex.EVENT_BOXES` held when it was written, and the cached report JSON this
  module keeps is one further remove from the data. `WinterStorm_Elliott_2022` was once
  scored on the CONUS column and reported a lift of 285x where the truth is 0.35x.

`wilson` is pinned at the ends because that is where the intervals are actually read, and
`sd_interval` because the whole point of T4's cells is that the interval is wide.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from aires import aceiling as AC

# The published wave-1/wave-2 numbers (aires/HANDOFF.md), each with HALF A UNIT IN ITS OWN
# LAST QUOTED PLACE as the tolerance. The HANDOFF quotes these to 3 or 4 significant
# decimals depending on the row, so a single fixed tolerance would either be unmeetable on
# the 3-dp rows or vacuous on the 4-dp ones.
PUBLISHED_P = {
    "PNW_HeatDome_2021": (0.054, 5e-4),
    "WinterStorm_Uri_2021": (0.0090, 5e-5),
    "WinterStorm_Elliott_2022": (0.0079, 5e-5),
    "p90_20251224": (0.0661, 5e-5),
    "p90_20240802": (0.649, 5e-4),
    "p90_20231107": (0.5827, 5e-5),
}

# `aires/HANDOFF.md` line 643: the Kish weight ESS is "the right thing to quote next to a
# P". Only these two are published as numbers, so only these two are pinned.
PUBLISHED_WEIGHT_ESS = {"PNW_HeatDome_2021": 5.68, "WinterStorm_Uri_2021": 2.24}


def _compare(event: str) -> dict:
    path = AC.A.AIRES_ROOT / event / "res" / AC.TAG / "compare.json"
    if not path.exists():
        pytest.skip(f"no run at {path}")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Sign handling
# --------------------------------------------------------------------------- #
def _require_clim_report():
    """Skip before `collect()` runs, not after it returns.

    `aceiling.collect` calls `climatology_report`, which on a cache MISS re-reads a
    64-year table (minutes) and WRITES `runs/aires/clim/report.json`. A test must never
    do that: `runs/` is experiment output, not a test fixture. The `if not rows: skip`
    checks further down fire after the write has already happened, so they do not
    protect anything. Call this first instead.
    """
    if not AC.CLIM_REPORT.exists():
        pytest.skip(f"no cached climatology report at {AC.CLIM_REPORT}")


def test_beyond_is_inclusive_and_sign_aware():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert AC.beyond(x, 1.0, +1.0).tolist() == [False, False, False, True, True]
    assert AC.beyond(x, -1.0, -1.0).tolist() == [True, True, False, False, False]
    # the threshold itself counts, on both tails
    assert AC.beyond([2.0], 2.0, +1.0).tolist() == [True]
    assert AC.beyond([-2.0], -2.0, -1.0).tolist() == [True]


def test_cold_tail_is_not_just_a_negated_warm_tail():
    """A -1 sign with a negative threshold must select the COLD members, not the warm."""
    members = np.array([-12.0, -8.0, -2.0, +4.0])
    cold = AC.beyond(members, -7.40, -1.0)
    assert cold.tolist() == [True, True, False, False]
    assert AC.beyond(members, -7.40, +1.0).sum() == 2   # the wrong sign picks the other two


# --------------------------------------------------------------------------- #
# The estimator, against what was published
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("event", sorted(PUBLISHED_P))
def test_res_probability_reproduces_the_published_table(event):
    cmp_ = _compare(event)
    got = AC.res_probability(cmp_, float(cmp_["observed"]))
    published, tol = PUBLISHED_P[event]
    assert got == pytest.approx(published, abs=tol)


def test_res_probability_at_an_unreachable_threshold_is_zero():
    """Past the population's own maximum the indicator is empty - not a small number."""
    cmp_ = dict(ds=dict(tail_sign=1.0), log_Z=0.0,
                realized=dict(box=[1.0, 2.0, 3.0]), weights=[1.0, 1.0, 1.0])
    assert AC.res_probability(cmp_, 10.0) == 0.0
    assert AC.res_probability(cmp_, 0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("event", sorted(PUBLISHED_WEIGHT_ESS))
def test_weight_ess_reproduces_the_published_values(event):
    cmp_ = _compare(event)
    assert AC.weight_ess(cmp_["weights"]) == pytest.approx(
        PUBLISHED_WEIGHT_ESS[event], abs=5e-3)


def test_weight_ess_is_n_for_equal_weights_and_one_for_a_point_mass():
    assert AC.weight_ess(np.ones(64)) == pytest.approx(64.0)
    w = np.zeros(64)
    w[7] = 3.0
    assert AC.weight_ess(w) == pytest.approx(1.0)
    assert AC.weight_ess(np.zeros(64)) == 0.0      # degenerate, not a ZeroDivisionError


# --------------------------------------------------------------------------- #
# Resolution: the estimator's own precondition
# --------------------------------------------------------------------------- #
def test_resolution_flags_both_degenerate_ends():
    assert AC.resolution(64, 64).startswith("none")      # indicator identically 1
    assert AC.resolution(0, 64).startswith("none")       # indicator identically 0
    assert AC.resolution(63, 64).startswith("marginal")
    assert AC.resolution(1, 64).startswith("marginal")
    assert AC.resolution(42, 64) == "ok"
    assert AC.resolution(0, 0).startswith("none")


def test_the_pinned_silent_failure_is_classified_as_unresolved():
    """`p90_20231107`: every walker is beyond the target, so `P` IS the normalization
    check. `compare.json` alone shows a plausible 0.583 against a direct 21/24 = 0.875
    (`aires/HANDOFF.md` lines 502 to 519). The module must name that, not price it."""
    cmp_ = _compare("p90_20231107")
    pop = np.asarray(cmp_["realized"]["box"], dtype="float64")
    obs = float(cmp_["observed"])
    sign = float(cmp_["ds"]["tail_sign"])
    n_beyond = int(AC.beyond(pop, obs, sign).sum())
    assert n_beyond == pop.size
    assert AC.resolution(n_beyond, int(pop.size)).startswith("none")
    # the equality that makes it a failure rather than a coincidence
    assert AC.res_probability(cmp_, obs) == pytest.approx(
        float(cmp_["normalization_check"]), abs=1e-9)


def test_table1_refuses_a_lift_where_the_estimator_has_no_resolution(capsys):
    _require_clim_report()
    rows = AC.collect(events=("p90_20231107", "PNW_HeatDome_2021"))
    if len(rows) < 2:
        pytest.skip("runs or climatology report missing")
    AC.table1(rows)
    out = capsys.readouterr().out
    lines = out.splitlines()
    degenerate = [l for l in lines if l.startswith(AC.SHORT["p90_20231107"])]
    healthy = [l for l in lines if l.startswith(AC.SHORT["PNW_HeatDome_2021"])]
    assert degenerate and "refused" in degenerate[0]
    assert healthy and "refused" not in healthy[0]
    # and it says WHY, rather than leaving a blank cell
    assert "P == normalization check" in out


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #
def test_wilson_stays_a_proper_interval_at_the_ends():
    lo, hi = AC.wilson(0, 24)
    assert lo == 0.0 and 0.0 < hi < 1.0        # Wald would give (0, 0)
    lo, hi = AC.wilson(24, 24)
    assert hi == 1.0 and 0.0 < lo < 1.0
    lo, hi = AC.wilson(0, 0)
    assert (lo, hi) == (0.0, 1.0)              # no data is not a zero-width statement


def test_wilson_brackets_the_point_estimate():
    for k, n in ((1, 36), (14, 192), (56, 192)):
        lo, hi = AC.wilson(k, n)
        assert lo < k / n < hi


def test_sd_interval_brackets_one_and_is_wide_at_small_n():
    lo24, hi24 = AC.sd_interval(24)
    assert lo24 < 1.0 < hi24
    assert lo24 == pytest.approx(0.777, abs=0.01)
    assert hi24 == pytest.approx(1.403, abs=0.01)
    lo4, hi4 = AC.sd_interval(4)
    # CFSv2's four members: the interval spans a factor of six, so no dispersion verdict
    # can be read off a single CFSv2 row.
    assert lo4 < lo24 and hi4 > hi24
    assert hi4 / lo4 > 6.0
    assert np.isnan(AC.sd_interval(1)[0])


def test_sd_interval_tightens_monotonically_with_n():
    widths = [AC.sd_interval(n)[1] / AC.sd_interval(n)[0] for n in (4, 8, 16, 24, 64)]
    assert widths == sorted(widths, reverse=True)


# --------------------------------------------------------------------------- #
# The climatological pool
# --------------------------------------------------------------------------- #
def test_ex_self_drops_the_event_year_from_both_sides():
    """Numerator and denominator must lose the same year, or every rate shifts by ~1/64."""
    _require_clim_report()
    rows = AC.collect(events=("PNW_HeatDome_2021",))
    if not rows:
        pytest.skip("no PNW run / climatology report on disk")
    rep = AC.climatology_report()["events"]["PNW_HeatDome_2021"][AC.POOL]
    n_pool = rep["n_windows"] * rep["n_years_ex_self"] / rep["n_years"]
    assert rep["n_years_ex_self"] == rep["n_years"] - 1
    assert rows[0][f"p_clim__{AC.POOL}"] == pytest.approx(
        rep["n_windows_beyond_ex_self"] / n_pool)
    # and it is strictly the stricter of the two readings for an event that set a record
    assert rep["n_windows_beyond_ex_self"] <= rep["n_windows_beyond"]


def test_a_report_that_lost_an_event_box_is_refused():
    """The Elliott failure mode, at the level this module caches it.

    `aclim.report` falls back to the CONUS column with no error when the .nc cache has no
    column for a registered box, so a stale cached report scores an event against a
    different region's climate. `aires/alift.py::clim_pool` raises on exactly this; the
    cached JSON needed its own guard, because it survives a rebuild of either side.
    """
    stale = {"events": {"WinterStorm_Elliott_2022": {"column": "CONUS"}}}
    with pytest.raises(SystemExit) as e:
        AC.check_report_boxes(stale)
    assert "WinterStorm_Elliott_2022" in str(e.value)
    # a p90 case IS the CONUS mean by design, so CONUS is correct there, not a fallback
    AC.check_report_boxes({"events": {"p90_20231107": {"column": "CONUS"}}})


def test_the_report_on_disk_still_carries_every_events_own_column():
    if not AC.CLIM_REPORT.exists():
        pytest.skip(f"no cached report at {AC.CLIM_REPORT}")
    rep = json.loads(AC.CLIM_REPORT.read_text())
    AC.check_report_boxes(rep)               # raises if any boxed event fell back
    from aires import aindex as AI
    for name, r in rep["events"].items():
        assert r["column"] == (name if name in AI.EVENT_BOXES else "CONUS")


# --------------------------------------------------------------------------- #
# What the module must keep refusing to say
# --------------------------------------------------------------------------- #
def test_no_return_period_is_quoted_anywhere():
    """The GEV was removed on 2026-09-11 and must not come back without an interval.

    `aclim` still computes it and `collect` still reads the same report block, so a single
    line restores it. The fit is 30 to 64 block maxima with events at or beyond the sample
    maximum; SCentral 2023's 4086-year figure resamples to [98, 352357] years with 30% of
    resamples infinite. If a return period is ever wanted again it carries that interval.
    """
    _require_clim_report()
    rows = AC.collect(events=("PNW_HeatDome_2021",))
    if not rows:
        pytest.skip("no PNW run / climatology report on disk")
    assert not [k for k in rows[0] if "gev" in k.lower() or "return_period" in k.lower()]
    src = (AC.__file__ and open(AC.__file__).read()) or ""
    body = src.split('"""', 2)[-1]           # exclude the docstring, which explains the cut
    assert "gev" not in body.lower()


def test_no_p_value_is_computed_across_the_spread_ratios():
    """Nine events over different boxes, seasons and tails are not exchangeable draws.

    `aires/INSIGHTS_PLAN.md` descopes inventing an interval for them, so T4 returns the
    ratios, their per-row sampling intervals and a count, and no test statistic.
    """
    _require_clim_report()
    rows = AC.collect(events=("PNW_HeatDome_2021", "WinterStorm_Uri_2021"))
    if len(rows) < 2:
        pytest.skip("runs or climatology report missing")
    spread = AC.table4(rows)
    for m, d in spread.items():
        assert not [k for k in d if "p" == k or k.endswith("_p") or "pvalue" in k], m
        assert set(d) == {"geometric_mean", "n_events", "below_one", "lo", "hi",
                          "ratios", "members_per_event"}


def test_every_spread_ratio_is_reported_with_its_member_count():
    """A four-member sd and a 24-member sd must not be printable as the same claim."""
    _require_clim_report()
    rows = AC.collect(events=("PNW_HeatDome_2021",))
    if not rows:
        pytest.skip("no PNW run / climatology report on disk")
    spread = AC.table4(rows)
    for m, d in spread.items():
        assert len(d["members_per_event"]) == len(d["ratios"]) == d["n_events"]
        assert all(n >= 3 for n in d["members_per_event"])
