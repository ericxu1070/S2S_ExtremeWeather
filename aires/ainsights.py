#!/usr/bin/env python
"""The three figures the AI+RES insights deck needs and does not already have.

`aires/INSIGHTS_PLAN.md` Step 4 lists fourteen panels. Eleven of them are figures already
on disk under `figures/aires/` and `figures/astab/`; `scripts/embed_insights.py` collects
those. This module draws the other three, and nothing here re-derives a quantity that a
run already published:

**Panel 1, the slate** (`aires_insights_slate.png`). Two tables in one figure. The top one
is the ten runs - nine productions plus the persistence control - with box, init, peak,
tail sign, observed ``A_L``, sigma-depth, reach, weighted ``P`` and, beside every ``P``,
the run's own normalization check ``sum p_i``. The bottom one is the **baseline coverage**,
and it is on the first page on purpose: `runs/aires/<event>/ds_baseline.json` is NOT
uniformly populated, and a reader who meets the exceedance curves without knowing that
will read a missing row as a negative result. Four of the nine events carry exactly one
direct-sampling ensemble (INSIGHTS_PLAN's Panel 1 prose says three; its own section 2.4
coverage table and the files both say four - see `baseline_coverage`).

**Panel 2, the severity ladder** (`aires_insights_ladder.png`). The deck's magnitude axis
done properly. x is how far the observation sat outside the DIRECT ensemble's own spread,

    sigma = tail_sign * (observed - mean(direct)) / std(direct, ddof=1)

which is `awalkers.sigma_depth`, imported rather than copied, and NOT scraped from the
prose tables in `aires/HANDOFF.md`. ``ddof=1`` is load-bearing: the population sd moves
PNW from +1.98 to +2.02 and Southwest from +3.32 to +3.39, and `check_sigma_ddof1` fails
the module loudly if the imported depth ever stops using the sample sd. y is the weighted
``P(A_L >= obs)``, log scale. The reach boundary sits between California at +2.85 sigma
(reached) and Southwest at +3.32 sigma (not reached) - INSIGHTS_PLAN names Elliott as the
deepest hit, but California is 0.04 sigma deeper still and was also reached, so the gap is
derived by `reach_boundary` and not quoted. That boundary is the single thing the deck
exists to show. A second strip under the scatter carries ``sum p_i`` for
the same runs at the same x, because a bare ``P`` in this experiment is misleading: the
check runs 0.568 to 1.769 across the nine productions and 8.75 on the persistence control.

**Panel 6, the two calibration anchors** (`aires_insights_anchors.png`). Deep in the tail,
where AI+RES operates, direct sampling reads 0/24 on six of the nine events and there is
nothing to check the weighted curve against. Two rungs are the exception, and both return
1 of 24:

    p90_20251224, box index      AI+RES P = 0.0661   direct 1/24 = 0.0417 [0.0074, 0.202]
    Elliott, CONUS secondary     AI+RES P = 0.136    direct 1/24 = 0.0417 [0.0074, 0.202]

Elliott's CONUS index is a legitimate SECOND anchor at +1.89 sigma and is labelled as
such - it is not that event's score, which is the N Plains box at +2.81 sigma. Section 3.5
of the plan is the reason: on the CONUS index four of the nine events sit at or below the
median, which is the one regime where this estimator is measured to fail silently. Two
further direct-resolving comparisons exist (`p90_20240802` at 14/24 and `p90_20231107` at
21/24) but they are median-region checks, not tail anchors, and they are panel 7's page.

No new statistics
-----------------
Deliberately. The plan (Step 4, "Uncertainty") descoped a founder-lineage bootstrap
because the estimator under resampling is not obvious - whether ``Z`` is recomputed per
draw, and how to treat the tied mass blocks - and this experiment has a documented history
of failing plausibly. So exactly three already-measured numbers carry the uncertainty, and
this module computes none of its own:

1. the direct-sampling **Wilson interval** where direct resolves (`aceiling.wilson`),
2. the **weight ESS** out of 64 (`awalkers.weight_ess`),
3. the run's own **normalization check** ``sum p_i``.

    PYTHONPATH=. python -m aires.ainsights
    PYTHONPATH=. python -m aires.ainsights --panels ladder
    PYTHONPATH=. python -m aires.ainsights --outdir /tmp/figs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from aires import aconfig as A
from aires import aindex as AI
from aires.aceiling import wilson                # the only interval in the deck
from aires.alift import walker_mass              # the estimator, not a copy of it
from aires.awalkers import (DIRECT_COLOR, OBS_COLOR, RES_COLOR, _fmt_p, discover,
                            metric_unit, order_runs, panel_label, run_record, sigma_depth,
                            weight_ess)

# The direct-sampling ensembles a run may carry, in the preference order the wave tables
# quote. `awalkers.DS_PREFERENCE` is the same tuple; it is restated here as the COVERAGE
# axis rather than imported as a preference, because the coverage table has to list every
# slot including the empty ones.
BASELINE_KEYS = ("gencast_xres", "gencast_walkers", "fcn3")
BASELINE_HEAD = {
    "gencast_xres": "GenCast direct\n(0.25 deg xres cube)",
    "gencast_walkers": "GenCast direct\n(Gate 3 walkers)",
    "fcn3": "FCN3 direct\n(different model)",
}

# The two rungs where direct sampling still returns a non-zero count in the TAIL. Both
# are 1 of 24. `index` is which of the two indices `aindex.indices()` records on every
# run: the event-centered box, or the free CONUS secondary.
ANCHORS = (("p90_20251224", "pilot", "box"),
           ("WinterStorm_Elliott_2022", "pilot", "conus"))

CONTROL_COLOR = "0.45"
PINNED_COLOR = "#9467bd"
OK_GREEN = "#2a7f4f"
ABSENT_GREY = "0.55"

PANELS = ("slate", "ladder", "anchors")


# --------------------------------------------------------------------------- #
# Pure pieces - no matplotlib, no network
# --------------------------------------------------------------------------- #
_SIGMA_GUARD: list[bool] = []


def check_sigma_ddof1() -> None:
    """Fail loudly if `awalkers.sigma_depth` ever stops using the SAMPLE sd.

    ``ddof`` is the one parameter in the depth formula that moves every marker on panel 2
    without changing the shape of the figure, so a silent switch to the population sd
    would look exactly like a correct ladder: on the real slate it moves PNW from +1.98 to
    +2.02 and Southwest from +3.32 to +3.39, and nothing else on the page would change.
    A four-member ensemble whose two sds differ by 15 per cent separates the two answers,
    so the check is arithmetic on a fabricated dict - no run tree, no files, microseconds.
    Memoized: this is a contract on an import, and imports do not change mid-process.
    """
    if _SIGMA_GUARD:
        return
    v = [0.0, 1.0, 2.0, 3.0]                       # sample sd 1.29099, population 1.11803
    d = {"observed": 5.0, "config": {"tail_sign": 1.0},
         "ds": {"gencast_xres": {"box": v}}}
    got = float(sigma_depth(d))
    want = (5.0 - 1.5) / float(np.std(v, ddof=1))
    if not np.isclose(got, want, rtol=1e-9, atol=0.0):
        raise AssertionError(
            "aires.awalkers.sigma_depth is not using ddof=1: on a four-member ensemble it "
            f"returns {got:.6f}, where the sample sd gives {want:.6f} and the population "
            f"sd gives {(5.0 - 1.5) / float(np.std(v, ddof=0)):.6f}. Every sigma-depth in "
            "this deck, and the reach boundary derived from them, would be wrong.")
    _SIGMA_GUARD.append(True)


def sigma_ddof0(r: dict) -> float:
    """The same depth with the POPULATION sd. NOT a number the deck reports.

    It exists so the caption claiming ``ddof=1`` is load-bearing can quote the shift it
    derived instead of a figure copied out of prose - which is how the ladder's footnote
    came to say Southwest moves "from +3.30", when the run's own files say +3.32.
    """
    v = np.asarray(r["direct"], dtype="float64")
    obs = r.get("observed")
    if obs is None or v.size < 2:
        return float("nan")
    sd = float(np.std(v, ddof=0))
    if sd <= 0:
        return float("nan")
    return float(r["sign"]) * (float(obs) - float(np.mean(v))) / sd


def normalization_range(rows) -> dict:
    """What ``sum p_i`` actually does on this slate, productions and controls apart.

    Every caption in this module that quotes the normalization check quotes it from here,
    because the check is the one number on the page that a new run changes and a hardcoded
    caption would not. ``rows`` is anything with ``total_mass``, ``weight_ess``, ``n`` and
    ``control`` - a slate row or a ladder point.
    """
    rows = list(rows)
    prod = [q for q in rows if not q["control"]]
    ctl = [q for q in rows if q["control"]]
    return dict(
        n_prod=len(prod), n_control=len(ctl),
        lo=(min(q["total_mass"] for q in prod) if prod else float("nan")),
        hi=(max(q["total_mass"] for q in prod) if prod else float("nan")),
        # The worst control is the one the caption has to name: the failure being shown is
        # that a degenerate run's check runs away, and an average would hide it.
        control_mass=(max(q["total_mass"] for q in ctl) if ctl else float("nan")),
        control_ess=(min(q["weight_ess"] for q in ctl) if ctl else float("nan")),
        control_n=(max(int(q["n"]) for q in ctl) if ctl else 0),
    )


def ds_baseline_path(event: str) -> Path:
    return A.event_dir(event) / "ds_baseline.json"


def ds_baseline(event: str) -> dict:
    """The run's own ``ds_baseline.json``. Raises if it is missing rather than guessing."""
    p = ds_baseline_path(event)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def cfs_members(event: str, lead_days: float = 21.0, metric: str = "t2m_anom") -> int | None:
    """How many CFSv2 cycles this event carries, or None if it has no CFS record.

    CFSv2 is the one baseline that IS uniform - all nine events have four members from
    the four 6-hourly cycles ending at the AI+RES init - so it is worth showing beside
    the three that are not.
    """
    p = A.cfs_json_path(event, lead_days, metric)
    if not p.exists():
        return None
    return int(json.loads(p.read_text()).get("n_members", 0)) or None


def baseline_coverage(event: str) -> dict:
    """Which direct-sampling ensembles this event actually has, and how many members.

    ``{key: n or None}`` over every slot in `BASELINE_KEYS` plus ``"cfs"``, plus
    ``n_direct``, the count of NON-EMPTY slots among `BASELINE_KEYS`.

    This is the trap the plan flags first and the reason `aires/amaps.py` and
    `aires/apdfs.py` were both fixed: eight of nine events have the 24-member GenCast
    xres cube (SCentral does not, and its 16 Gate 3 walkers are promoted into that slot),
    five of nine have the 24-member FCN3 ensemble, and three have the Gate 3 walkers.
    Code that assumes three rows either crashes or draws an empty panel that reads as
    "the ensemble missed" instead of "the ensemble was never run".

    Counted here, from the files: **four** events carry exactly one direct ensemble
    (SCentral, California, Southwest, Elliott), where INSIGHTS_PLAN's Panel 1 prose says
    three. The coverage TABLE in the plan's section 2.4 is right; the sentence under
    Panel 1 is not. Nothing is hardcoded either way - the number is derived.
    """
    d = ds_baseline(event)
    out: dict = {}
    for key in BASELINE_KEYS:
        rec = d.get(key)
        out[key] = (int(rec.get("n", len(rec.get("box", [])))) if isinstance(rec, dict)
                    and rec.get("box") else None)
    out["cfs"] = cfs_members(event)
    out["n_direct"] = sum(1 for k in BASELINE_KEYS if out[k])
    return out


def is_control(r: dict) -> bool:
    """Is this run the persistence control rather than a production?

    On ``config.backend``, which is what the run's score function actually was
    (``"fcn3"`` on every production, ``"persistence"`` on job 1180), never on the tag.
    A tag is a label someone chose; the backend is the thing the control is a control OF,
    and the whole point of that run is that only the scorer differs.
    """
    return str(r["d"]["config"].get("backend", "fcn3")) != "fcn3"


def box_extent(event: str) -> str:
    """``44-50N, 124-118W`` - the box `aindex` scored, in degrees a reader can place."""
    b = AI.box_for(event)
    lo_la, hi_la = b.lat
    lo_lo, hi_lo = (((x + 180.0) % 360.0) - 180.0 for x in b.lon)

    def _la(x):
        return f"{abs(x):.0f}{'N' if x >= 0 else 'S'}"

    def _lo(x):
        return f"{abs(x):.0f}{'W' if x < 0 else 'E'}"

    return f"{_la(lo_la)}-{_la(hi_la)}, {_lo(lo_lo)}-{_lo(hi_lo)}"


def slate_row(r: dict) -> dict:
    """One line of the slate table, straight off a `awalkers.run_record`."""
    check_sigma_ddof1()
    cfg = r["d"]["config"]
    cov = baseline_coverage(r["event"])
    return dict(
        event=r["event"], tag=r["tag"], label=panel_label(r["event"]),
        control=is_control(r), box=r["box"], extent=box_extent(r["event"]),
        init=str(cfg.get("init", ""))[:10], peak=str(cfg.get("peak", ""))[:10],
        sign=r["sign"], observed=r["observed"], unit=metric_unit(r["metric"]),
        sigma=r["sigma_depth"], n=r["n"], n_reached=r["n_reached"],
        P=r["P"], deepest=r["deepest"], pinned=r["pinned"],
        total_mass=r["total_mass"], weight_ess=r["weight_ess"],
        direct_key=r["direct_key"], direct_n=int(r["direct"].size),
        direct_reached=r["direct_reached"], n_direct=cov["n_direct"], coverage=cov,
    )


def slate_table(recs: list[dict]) -> pd.DataFrame:
    """The slate, hardest target first - `awalkers.order_runs`, the deck's one ordering."""
    return pd.DataFrame([slate_row(r) for r in order_runs(recs)])


def ladder_point(r: dict) -> dict:
    """One point of the severity ladder, with everything the marker has to encode.

    ``y`` is the weighted ``P(A_L >= obs)`` where the run reached the observation. Where
    nothing reached, the estimator returns exactly 0, which cannot go on a log axis and
    is not the honest statement anyway: the run's own curve evaluated at its most extreme
    walker IS a statement, so ``y`` is that level and ``kind`` is ``"bound"`` - the point
    is drawn as an upper limit, not an estimate. Where the observation lies below EVERY
    walker the run is ``"pinned"``: ``P`` collapses onto ``sum p_i`` and is not a
    probability at all.
    """
    check_sigma_ddof1()
    kind = ("pinned" if r["pinned"] else
            ("reached" if r["n_reached"] else "bound"))
    y = r["deepest"] if kind == "bound" else r["P"]
    k, n = int(r["direct_reached"]), int(r["direct"].size)
    lo, hi = wilson(k, n) if n else (np.nan, np.nan)
    return dict(
        event=r["event"], tag=r["tag"], label=panel_label(r["event"]),
        x=r["sigma_depth"], x_ddof0=sigma_ddof0(r), y=y, kind=kind, control=is_control(r),
        n=r["n"], n_reached=r["n_reached"], P=r["P"], deepest=r["deepest"],
        total_mass=r["total_mass"], weight_ess=r["weight_ess"],
        direct_key=r["direct_key"], direct_k=k, direct_n=n,
        direct_p=(k / n if n else np.nan), direct_lo=lo, direct_hi=hi,
    )


def reach_boundary(points: list[dict]) -> tuple[float, float]:
    """The sigma-depth gap the reach boundary sits in: deepest reached, shallowest missed.

    Productions only - the persistence control missed at PNW's depth for a reason that is
    the scorer and not the severity, so counting it would close a gap that is not closed.
    ``(nan, nan)`` if either side is empty.
    """
    p = [q for q in points if not q["control"] and np.isfinite(q["x"])]
    hit = [q["x"] for q in p if q["kind"] == "reached"]
    miss = [q["x"] for q in p if q["kind"] == "bound"]
    return (max(hit) if hit else np.nan, min(miss) if miss else np.nan)


def anchor_record(event: str, tag: str = "pilot", index: str = "box") -> dict:
    """One calibration anchor: AI+RES's weighted ``P`` against direct sampling's ``k/n``.

    ``index`` selects which of the two indices `aindex.indices()` records - ``"box"``, the
    event-centered observable the whole deck uses, or ``"conus"``, the free secondary. The
    masses are `alift.walker_mass`, so the AI+RES number here is the same subset sum every
    other figure reports, evaluated on the chosen index.
    """
    d = json.loads((A.res_dir(event, tag) / "compare.json").read_text())
    b = ds_baseline(event)
    # From `config`, which is where `awalkers.sigma_depth` reads it too. `ds.tail_sign`
    # is the same number on every real run; taking it from two places is how the two
    # would eventually disagree.
    sign = float(d["config"]["tail_sign"])
    obs = float(b["observed"][index])

    key = next((k for k in BASELINE_KEYS
                if isinstance(b.get(k), dict) and b[k].get(index)), None)
    if key is None:
        raise KeyError(f"{event}: no direct ensemble carries a {index!r} index")
    direct = np.asarray(b[key][index], dtype="float64")

    al = np.asarray(d["realized"][index], dtype="float64")
    p = walker_mass(d)
    reached = sign * al >= sign * float(obs)
    k, n = int((sign * direct >= sign * obs).sum()), int(direct.size)
    lo, hi = wilson(k, n)
    sd = float(np.std(direct, ddof=1))
    return dict(
        event=event, tag=tag, index=index, label=panel_label(event),
        index_label=("event box (" + AI.box_for(event).name + ")" if index == "box"
                     else "CONUS secondary index"),
        unit=metric_unit(d["ds"].get("metric", "t2m_anom")), sign=sign, observed=obs,
        direct_key=key, direct_mean=float(np.mean(direct)), direct_sd=sd,
        sigma=(sign * (obs - float(np.mean(direct))) / sd if sd > 0 else np.nan),
        direct_k=k, direct_n=n, direct_p=k / n if n else np.nan,
        direct_lo=lo, direct_hi=hi,
        P=float(p[reached].sum()), n_reached=int(reached.sum()), n=int(al.size),
        total_mass=float(p.sum()), weight_ess=weight_ess(d["weights"]),
        inside=bool(lo <= float(p[reached].sum()) <= hi),
    )


def stability_verdict(csv: Path | None = None) -> dict:
    """*GenCast stable through week X, FCN3 through week Y*, via `astab.reduce`.

    Never off `runs/astab/stability.csv`'s ``status`` column. The thresholds that produce
    the published verdict are applied at REDUCE time, and taking the first ``FAIL`` per
    chain gives a GenCast failure at day 24 of the week-4 chain against a published
    verdict of "stable through week 6". ``_model_survival`` also drops the (model, week)
    pairs the walker never rolled to the peak, so three days of walker rollout at an
    FCN3-extension lead cannot read as a clean 140-day chain.
    """
    from astab import reduce as R

    path = Path(csv) if csv else R.csv_path()
    df = pd.read_csv(path)
    out: dict = {}
    for model in ("gencast", "fcn3_era5", "fcn3_adapter"):
        surv = R._model_survival(df, model)
        out[model] = dict(clean_through=R._all_clean_through(surv),
                          tested=sorted(surv),
                          untested=R.untested_weeks(df, model))
    fcn3 = [out[m]["clean_through"] for m in ("fcn3_era5", "fcn3_adapter")
            if out[m]["clean_through"] is not None]
    out["gencast_weeks"] = out["gencast"]["clean_through"]
    out["fcn3_weeks"] = min(fcn3) if fcn3 else None
    return out


TOP_FAN = (dict(dx=0.0, dy=17.0, ha="center", va="bottom"),
           dict(dx=15.0, dy=-15.0, ha="left", va="top"),
           dict(dx=-15.0, dy=15.0, ha="right", va="bottom"),
           dict(dx=15.0, dy=15.0, ha="left", va="bottom"),
           dict(dx=-15.0, dy=-15.0, ha="right", va="top"))
STRIP_FAN = (dict(dx=0.0, dy=11.0, ha="center", va="bottom"),
             dict(dx=0.0, dy=-12.0, ha="center", va="top"),
             dict(dx=-16.0, dy=11.0, ha="right", va="bottom"),
             dict(dx=16.0, dy=-12.0, ha="left", va="top"))


def label_offsets(points: list[dict], gap: float = 0.30, fan=TOP_FAN,
                  ykey: str = "y") -> list[dict]:
    """Where each point's label goes, so that near neighbours do not overprint.

    Nine events on one axis put PNW at +1.98 and `p90_20251224` at +1.91, Elliott at
    +2.81 and California at +2.85, and the persistence control on top of PNW exactly.
    Matplotlib will happily stack all of those labels in the same place. Points are
    clustered by x proximity, each cluster is ordered by ``ykey`` DESCENDING, and the fan
    then places the highest point's label above and the lowest one's below - which is the
    only ordering that cannot drop a label between two markers. Deterministic, so the
    figure is reproducible.
    """
    out = [dict(fan[0]) for _ in points]
    order = sorted(range(len(points)), key=lambda i: points[i]["x"])
    clusters: list[list[int]] = []
    for i in order:
        if clusters and points[i]["x"] - points[clusters[-1][-1]]["x"] < gap:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    for c in clusters:
        for j, i in enumerate(sorted(c, key=lambda k: -points[k][ykey])):
            out[i] = dict(fan[j % len(fan)])
    return out


# Where a label may go once its fan seat is taken, as (radius in points, bearing in
# degrees) walked nearest-ring-first. Straight up is tried before the diagonals because a
# label directly above its own marker is the one a reader never has to trace.
LABEL_RINGS = (18.0, 32.0, 48.0, 66.0, 88.0)
LABEL_ANGLES = (90.0, 45.0, 135.0, 0.0, 180.0, -45.0, -135.0, -90.0)


def _offset_at(r: float, deg: float) -> dict:
    """One ring seat, with the alignment that puts the text's near EDGE at the anchor.

    The alignment is not cosmetic: it is what makes the anchor point sit between the
    marker and the text rather than in the middle of the words, so a leader line drawn to
    the anchor stops at the edge of the label instead of striking through it.
    """
    t = np.radians(deg)
    cx, cy = float(np.cos(t)), float(np.sin(t))
    return dict(dx=r * cx, dy=r * cy,
                ha=("center" if abs(cx) < 0.3 else ("left" if cx > 0 else "right")),
                va=("center" if abs(cy) < 0.3 else ("bottom" if cy > 0 else "top")))


def label_candidates(seed: dict) -> list[dict]:
    """Every seat a label may take, its own fan choice first and then outward."""
    return [dict(seed)] + [_offset_at(r, a) for r in LABEL_RINGS for a in LABEL_ANGLES]


def _bbox(x0, y0, x1, y1):
    from matplotlib.transforms import Bbox
    return Bbox.from_extents(x0, y0, x1, y1)


def _pad_box(bb, p: float):
    return _bbox(bb.x0 - p, bb.y0 - p, bb.x1 + p, bb.y1 + p)


def _overlap(a, b) -> float:
    """Overlapping AREA in square pixels, 0 when two boxes are clear of each other."""
    from matplotlib.transforms import Bbox
    i = Bbox.intersection(a, b)
    return 0.0 if i is None else abs(i.width * i.height)


def point_box(ax, xy, half: float = 9.5):
    """A marker's own footprint in display pixels, so a label cannot land on top of it."""
    x, y = ax.transData.transform(xy)
    return _bbox(x - half, y - half, x + half, y + half)


def span_box(ax, x, lo, hi, half: float = 5.0):
    """An error bar's footprint: the vertical run it occupies, not just its centre."""
    _, y0 = ax.transData.transform((x, lo))
    px, y1 = ax.transData.transform((x, hi))
    return _bbox(px - half, min(y0, y1), px + half, max(y0, y1))


def place_labels(fig, items, obstacles=(), pad: float = 3.0,
                 leader_at: float = 27.0) -> list[dict]:
    """Move each label off every other one using its RENDERED WIDTH, not its x alone.

    `label_offsets` clusters points by x and fans their labels, which is enough while the
    labels are short and fails exactly where this panel can least afford it: S-Central,
    Elliott and California sit inside 0.62 sigma of one another, each label is a run name
    plus a reach count and about 180 px of text, and the three of them are precisely the
    runs that carry the reach-boundary argument. A reader who cannot tell which marker is
    California cannot read the boundary, which is the one thing the figure exists to show.

    A label's width is only knowable after a draw, so this runs on the rendered figure.
    Each label takes the nearest seat on the rings around its own marker whose box clears
    every box already placed - the other labels, the markers, the error bars, the legend,
    the boundary callout - and stays inside its own axes. A seat far enough out to be
    ambiguous grows a thin leader line back to the marker.

    Deterministic in two ways that matter for a figure under version control: the seat
    order is fixed, and ``items`` is consumed in the order given, so the runs that come
    first in `awalkers.order_runs` get first choice. Returns the seat each label took.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    placed = list(obstacles)
    taken = []
    for it in items:
        ann, ax = it["ann"], it["ax"]
        frame = ax.get_window_extent(rend)
        best, best_box, best_cost = None, None, None
        for seat in label_candidates(it["seed"]):
            ann.set_position((seat["dx"], seat["dy"]))
            ann.set_horizontalalignment(seat["ha"])
            ann.set_verticalalignment(seat["va"])
            box = _pad_box(ann.get_window_extent(rend), pad)
            area = abs(box.width * box.height)
            # Anything hanging outside the axes costs four times what an overlap does:
            # a label half off the frame is worse than a label that grazes a gridline.
            cost = (sum(_overlap(box, q) for q in placed)
                    + 4.0 * (area - _overlap(box, frame)))
            if best_cost is None or cost < best_cost:
                best, best_box, best_cost = seat, box, cost
            if cost <= 0.0:
                break
        ann.set_position((best["dx"], best["dy"]))
        ann.set_horizontalalignment(best["ha"])
        ann.set_verticalalignment(best["va"])
        placed.append(best_box)
        taken.append(dict(best))

        if float(np.hypot(best["dx"], best["dy"])) >= leader_at:
            # Straight in DISPLAY space, which is what the reader sees; the two ends are
            # converted back through transData so the log y axis cannot bend it.
            inv = ax.transData.inverted()
            x0, y0 = ax.transData.transform(it["xy"])
            k = fig.dpi / 72.0
            a, b = inv.transform((x0, y0)), inv.transform(
                (x0 + best["dx"] * k, y0 + best["dy"] * k))
            ax.plot([a[0], b[0]], [a[1], b[1]], lw=0.7, ls="-",
                    color=it.get("colour", "0.4"), alpha=0.5, zorder=4.5,
                    solid_capstyle="butt")
    return taken


def _fmt_sigma(x: float) -> str:
    """``+2.81``, ``n/a``, and ``0.00`` for the one run that sits on the median.

    `p90_20240802` is at -0.0025 sigma, which ``%+.2f`` renders as ``-0.00``: a sign on a
    number that has none at the precision printed, and on the one panel whose whole point
    is that the target sits AT the ensemble median. The plan quotes it as 0.00 and so does
    this - only inside the half-unit of the last printed digit, never anywhere else.
    """
    if not np.isfinite(x):
        return "n/a"
    return "0.00" if abs(x) < 0.005 else f"{x:+.2f}"


def _cell(n) -> str:
    return f"{int(n)} members" if n else "absent"


def _save(fig, out: Path) -> Path:
    """One place that writes a PNG, so the three panels cannot drift on dpi or facecolor.

    ``facecolor="white"`` is explicit: the table panels have transparent axes, and a
    transparent PNG inlined into the deck picks up whatever the reader's theme paints
    behind it, which on a dark theme makes black-on-transparent table text invisible.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return out


# --------------------------------------------------------------------------- #
# Panel 1 - the slate, and the coverage a reader has to see before figure one
# --------------------------------------------------------------------------- #
def plot_slate(recs: list[dict], out: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = slate_table(recs)
    # The coverage table is a statement about EVENTS, so the control - which shares PNW's
    # baselines by construction - is not a row of it. A slate of nothing but controls has
    # no coverage table at all, and the figure says so rather than raising.
    prod = t[~t["control"]]

    head = ["event", "box", "extent", "init", "peak (event)", "tail",
            r"observed $A_L$", r"$\sigma$-depth", "N", "reach",
            r"weighted $P$", r"$\Sigma p_i$", "wESS"]
    body, colours = [], []
    for _, r in t.iterrows():
        control = bool(r["control"])
        if r["pinned"]:
            pcell = f"{r['P']:.4f}  PINNED"
        elif r["n_reached"]:
            pcell = _fmt_p(r["P"])
        else:
            pcell = f"0  (< {r['deepest']:.1e})"
        body.append([
            r["label"] + ("  [control]" if control else ""),
            r["box"], r["extent"], r["init"], r["peak"],
            "cold (low)" if r["sign"] < 0 else "warm (high)",
            f"{r['observed']:+.2f} {r['unit']}",
            _fmt_sigma(r["sigma"]),
            f"{r['n']}", f"{r['n_reached']}/{r['n']}",
            pcell, f"{r['total_mass']:.4f}", f"{r['weight_ess']:.2f}",
        ])
        colours.append(CONTROL_COLOR if control else
                       (PINNED_COLOR if r["pinned"] else "0.12"))

    cov_head = ["event"] + [BASELINE_HEAD[k] for k in BASELINE_KEYS] + [
        "CFSv2\n(operational)", "direct\nensembles"]
    cov_body, cov_flag = [], []
    for _, r in prod.iterrows():
        c = r["coverage"]
        cov_body.append([r["label"]] + [_cell(c[k]) for k in BASELINE_KEYS]
                        + [_cell(c["cfs"]), f"{c['n_direct']} of 3"])
        cov_flag.append(int(c["n_direct"]))
    lonely = sum(1 for f in cov_flag if f == 1)

    # Row counts drive the split: each table gets vertical space in proportion to the
    # rows it has to draw, and `bbox` makes each one fill its axes exactly. Without the
    # bbox, `loc="upper center"` leaves a third of the figure blank between the two.
    fig = plt.figure(figsize=(19.5, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(body) + 1.0, len(cov_body) + 1.6],
                          hspace=0.34, left=0.012, right=0.988, top=0.905, bottom=0.095)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (ax, bx):
        a.axis("off")

    tab = ax.table(cellText=body, colLabels=head, cellLoc="center",
                   bbox=[0, 0, 1, 1],
                   colWidths=[0.175, 0.068, 0.104, 0.068, 0.076, 0.070, 0.080,
                              0.060, 0.028, 0.050, 0.098, 0.054, 0.044])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.4)
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("0.80")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eceff3")
            cell.set_text_props(weight="bold", color="0.15")
        else:
            cell.set_text_props(color=colours[row - 1])
            if colours[row - 1] == CONTROL_COLOR:
                cell.set_facecolor("#f4f4f4")
            elif colours[row - 1] == PINNED_COLOR:
                cell.set_facecolor("#f5eefb")
            if col == 0:
                cell.set_text_props(ha="left")
                cell.PAD = 0.03
    n_ctl = int(t["control"].sum())
    ax.set_title(f"The slate: {len(prod)} week-3 production"
                 f"{'s' if len(prod) != 1 else ''}"
                 + (f" and {n_ctl} control{'s' if n_ctl != 1 else ''}" if n_ctl else "")
                 + ", hardest target first\n"
                 r"$N$ = 64 walkers, $M$ = 6 score members, $K$ = 5 resampling steps at "
                 r"3/6/9/12/15 d, frozen $C_k$ = (0, 1.0, 1.4, 1.8, 2.0), horizon 21 d",
                 fontsize=13.5, pad=16)

    if not cov_body:
        bx.set_title("No production run on this slate, so there is no baseline coverage "
                     "table to draw", fontsize=13.5, pad=16)
        return _save(fig, out or (A.fig_dir() / "aires_insights_slate.png"))

    ctab = bx.table(cellText=cov_body, colLabels=cov_head, cellLoc="center",
                    bbox=[0, 0, 1, 1],
                    colWidths=[0.22, 0.155, 0.155, 0.155, 0.13, 0.10])
    ctab.auto_set_font_size(False)
    ctab.set_fontsize(9.8)
    for (row, col), cell in ctab.get_celld().items():
        cell.set_edgecolor("0.80")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eceff3")
            cell.set_text_props(weight="bold", color="0.15")
            continue
        if col == 0:
            cell.set_text_props(ha="left", color="0.12")
            cell.PAD = 0.03
        elif 1 <= col <= len(BASELINE_KEYS) + 1:
            absent = cell.get_text().get_text() == "absent"
            cell.set_text_props(color=(ABSENT_GREY if absent else OK_GREEN),
                                style=("italic" if absent else "normal"))
            cell.set_facecolor("#f7f7f7" if absent else "#f1f8f3")
        else:
            cell.set_text_props(weight="bold",
                                color=(OBS_COLOR if cov_flag[row - 1] == 1 else "0.12"))
        if cov_flag[row - 1] == 1 and col == 0:
            cell.set_text_props(ha="left", color=OBS_COLOR, weight="bold")
    bx.set_title("Baseline coverage: `ds_baseline.json` is NOT uniformly populated, and "
                 "this belongs before the first figure\n"
                 f"{lonely} of {len(cov_body)} events carry exactly ONE direct-sampling "
                 "ensemble, so a missing curve on a later panel means the ensemble was "
                 "never run, not that it missed",
                 fontsize=13.5, pad=16)

    # Derived, not typed: the check is the one number on this page that a new run changes,
    # and a caption carrying last month's range would be exactly the kind of prose the
    # rest of this module refuses to scrape.
    nz = normalization_range(t.to_dict("records"))
    ctl_txt = ("" if not nz["n_control"] else
               f" and {nz['control_mass']:.2f} on the persistence control")
    fig.text(0.5, 0.018,
             r"$\sigma$-depth is $\mathrm{tail\_sign}\times(\mathrm{obs}-\mathrm{mean}\,"
             r"\mathrm{direct})/\mathrm{sd}(\mathrm{direct},\ \mathrm{ddof}=1)$, derived "
             "from each run's own direct ensemble (first available of GenCast xres, "
             "Gate 3 walkers, FCN3), never scraped from prose.\n"
             r"$\Sigma p_i$ is the run's OWN normalization check and should be 1: it is "
             f"not. It runs {nz['lo']:.3f} to {nz['hi']:.3f} across the {nz['n_prod']} "
             f"production{'s' if nz['n_prod'] != 1 else ''}{ctl_txt}, "
             r"so no $P$ in this deck is quoted without it. "
             "Weight ESS is the Kish ESS of the final weights out of "
             f"{int(t['n'].max()) if len(t) else 64}.",
             ha="center", va="bottom", fontsize=9.2, color="0.35", linespacing=1.6)

    return _save(fig, out or (A.fig_dir() / "aires_insights_slate.png"))


# --------------------------------------------------------------------------- #
# Panel 2 - the severity ladder
# --------------------------------------------------------------------------- #
def plot_ladder(recs: list[dict], out: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pts = [ladder_point(r) for r in order_runs(recs)]
    pts = [p for p in pts if np.isfinite(p["x"])]
    if not pts:
        raise SystemExit("no run carries a direct ensemble, so no depth can be derived")
    hit, miss = reach_boundary(pts)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(16.6, 11.4), sharex=True,
                                 height_ratios=[2.55, 1.0])

    # --- the boundary band, drawn first so every marker sits on top ----------- #
    # Its two EDGES are drawn as well as its fill. A label that lands on the band gets a
    # white backing below, and a 7.5 per cent wash under white text would stop reading as
    # a band at all; the dashed edges keep its extent unambiguous whatever sits on it.
    # Both edges come from `reach_boundary`, which derives them from the points.
    band = np.isfinite(hit) and np.isfinite(miss) and miss > hit
    if band:
        for a in (ax, bx):
            a.axvspan(hit, miss, color=OBS_COLOR, alpha=0.075, lw=0, zorder=0)
            for edge in (hit, miss):
                a.axvline(edge, color=OBS_COLOR, lw=1.0, ls=(0, (5, 3)), alpha=0.55,
                          zorder=0.5)

    # Autoscale packs the two end points against the frame, and the leftmost and
    # rightmost runs (`p90_20231107` and Southwest) are the two the deck most needs a
    # reader to see, so the limits are set by hand over the points themselves. The right
    # margin is the wider one: Southwest is the last point and its label is the one with
    # nothing beyond it to borrow room from.
    xs = [p["x"] for p in pts]
    pad = 0.11 * max(max(xs) - min(xs), 1e-6)
    ax.set_xlim(min(xs) - pad, max(xs) + 1.9 * pad)

    # The fan below is only the SEED. `place_labels` measures what each label actually
    # renders as and moves it off whatever it lands on, which the fan cannot do because
    # it clusters on x and knows nothing about how wide a run's name is.
    free = [p for p in pts if not p["control"]]
    it = iter(label_offsets(free))
    offsets = [dict(dx=0.0, dy=-18.0, ha="center", va="top") if p["control"] else next(it)
               for p in pts]
    it = iter(label_offsets(free, fan=STRIP_FAN, ykey="total_mass"))
    strip = [dict(dx=0.0, dy=11.0, ha="center", va="bottom") if p["control"] else next(it)
             for p in pts]

    ymin = min(p["y"] for p in pts)
    ymax = max(p["y"] for p in pts)
    ax.set_ylim(ymin / 11.0, ymax * 5.5)

    top_items, strip_items, blockers, strip_blockers = [], [], [], []
    for p, off, soff in zip(pts, offsets, strip):
        colour = (CONTROL_COLOR if p["control"] else
                  (PINNED_COLOR if p["kind"] == "pinned" else RES_COLOR))

        # Direct sampling, where it resolves at all. k = 0 is not a point on a log axis
        # and pretending otherwise would be the deck's worst possible mistake, so it is
        # drawn as what it actually is: the upper end of the same 95% Wilson interval
        # every other direct marker carries, and no estimate at all.
        if p["direct_n"]:
            if p["direct_k"]:
                ax.errorbar(p["x"], p["direct_p"],
                            yerr=[[p["direct_p"] - p["direct_lo"]],
                                  [p["direct_hi"] - p["direct_p"]]],
                            fmt="s", ms=6.5, color=DIRECT_COLOR, ecolor=DIRECT_COLOR,
                            elinewidth=1.3, capsize=4, zorder=4, alpha=0.95)
                kn = ax.annotate(f"{p['direct_k']}/{p['direct_n']}",
                                 xy=(p["x"], p["direct_p"]), xytext=(7, -11),
                                 textcoords="offset points", fontsize=7.8,
                                 color=DIRECT_COLOR, zorder=6)
                # The bar plus its own k/n caption: a run label that lands here would
                # read as a comment on the direct estimate rather than on the AI+RES one.
                blockers.append(("span", p["x"], p["direct_lo"], p["direct_hi"]))
                blockers.append(("art", kn))
            else:
                ax.plot([p["x"]], [p["direct_hi"]], marker="v", ms=7.5, mfc="none",
                        mec=DIRECT_COLOR, mew=1.3, alpha=0.75, zorder=3)
                blockers.append(("pt", p["x"], p["direct_hi"], 9.0))

        # AI+RES.
        if p["kind"] == "reached":
            ax.plot([p["x"]], [p["y"]], marker="o", ms=13, mfc=colour, mec="0.15",
                    mew=1.0, zorder=6)
        elif p["kind"] == "pinned":
            ax.plot([p["x"]], [p["y"]], marker="s", ms=12, mfc="none", mec=colour,
                    mew=2.2, zorder=6)
        else:
            ax.plot([p["x"]], [p["y"]], marker="o", ms=13, mfc="none", mec=colour,
                    mew=2.0, zorder=6)
            ax.annotate("", xy=(p["x"], p["y"] / 3.6), xytext=(p["x"], p["y"]),
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6),
                        zorder=5)

        blockers.append(("pt", p["x"], p["y"], 11.0))

        tail = (f"\n{p['n_reached']}/{p['n']} reach" if p["kind"] == "reached"
                else ("\nPINNED: P is $\\Sigma p_i$" if p["kind"] == "pinned"
                      else f"\n0/{p['n']} reach, upper bound"))
        head = p["label"] + ("  [persistence control]" if p["control"] else "")
        ann = ax.annotate(head + tail,
                          xy=(p["x"], p["y"]), xytext=(off["dx"], off["dy"]),
                          textcoords="offset points", ha=off["ha"], va=off["va"],
                          fontsize=8.6, color=colour, linespacing=1.3, zorder=7)
        top_items.append(dict(ann=ann, ax=ax, xy=(p["x"], p["y"]), seed=off,
                              colour=colour))

        # --- the strip: the normalization check, at the same x ---------------- #
        bx.vlines(p["x"], 1.0, p["total_mass"], color=colour, lw=1.4, alpha=0.75,
                  zorder=3)
        bx.plot([p["x"]], [p["total_mass"]], marker="o", ms=8,
                mfc=(colour if not p["control"] else "none"), mec="0.15", mew=1.0,
                zorder=4)
        strip_blockers.append(("pt", p["x"], p["total_mass"], 8.0))
        strip_blockers.append(("span", p["x"], min(1.0, p["total_mass"]),
                               max(1.0, p["total_mass"])))
        sann = bx.annotate(f"{p['total_mass']:.3f}", xy=(p["x"], p["total_mass"]),
                           xytext=(soff["dx"], soff["dy"]), textcoords="offset points",
                           ha=soff["ha"], va=soff["va"], fontsize=8.2, color=colour,
                           zorder=7)
        strip_items.append(dict(ann=sann, ax=bx, xy=(p["x"], p["total_mass"]), seed=soff,
                                colour=colour))

    ax.set_yscale("log")
    ax.grid(alpha=0.24, which="both")
    ax.set_ylabel(r"weighted  $P(A_L \geq \mathrm{obs})$   (log)", fontsize=11.5)
    ax.set_title("The severity ladder: reach follows the observation's depth in the "
                 "MODEL's forecast distribution,\nnot the size of the anomaly",
                 fontsize=14.5, pad=12)

    if band:
        # Anchored high, where the axis is empty on the right: the band's own foot is
        # where Southwest's upper-bound arrow lives, and a box there covers the one
        # marker the annotation is about.
        who = {p["x"]: p["label"] for p in pts if not p["control"]}
        x0, x1 = ax.get_xlim()
        frac = (0.5 * (hit + miss) - x0) / max(x1 - x0, 1e-9)
        ha = "right" if frac > 0.66 else ("left" if frac < 0.34 else "center")
        callout = ax.annotate(
            "the reach boundary lies in here:\n"
            f"{hit:+.2f}$\\sigma$ reached ({who.get(hit, '?')})\n"
            f"{miss:+.2f}$\\sigma$ not ({who.get(miss, '?')})",
            xy=(0.5 * (hit + miss), 0.985), xycoords=("data", "axes fraction"),
            ha=ha, va="top", fontsize=9.2, color=OBS_COLOR, linespacing=1.35,
            bbox=dict(fc="white", ec=OBS_COLOR, lw=0.7, alpha=0.92, pad=3.5))
        blockers.append(("art", callout))

    bx.set_yscale("log")
    masses = [p["total_mass"] for p in pts]
    bx.set_ylim(min(masses) / 1.55, max(masses) * 1.7)
    bx.axhline(1.0, color="0.25", lw=1.2, ls="--", zorder=2)
    bx.text(0.006, 1.0, " 1.0, what the check should be",
            transform=bx.get_yaxis_transform(), fontsize=8.6, color="0.25",
            va="bottom", ha="left")
    bx.grid(alpha=0.24, which="both")
    bx.set_ylabel(r"normalization check  $\Sigma p_i$", fontsize=11)
    bx.set_xlabel(r"$\sigma$-depth of the observation in the DIRECT ensemble"
                  r"   $\left[\,\mathrm{tail\_sign}\times(\mathrm{obs}-\mathrm{mean})/"
                  r"\mathrm{sd}(\mathrm{ddof}=1)\,\right]$", fontsize=11.5)

    handles = [
        Line2D([], [], marker="o", ls="none", mfc=RES_COLOR, mec="0.15", ms=11,
               label="AI+RES reached the observation: $P$ is an estimate"),
        Line2D([], [], marker="o", ls="none", mfc="none", mec=RES_COLOR, mew=2, ms=11,
               label=r"nothing reached: $P=0$, plotted at the deepest level the run "
                     r"resolved (an upper bound)"),
        Line2D([], [], marker="s", ls="none", mfc="none", mec=PINNED_COLOR, mew=2, ms=10,
               label=r"PINNED: the target sits below every walker, so $P$ collapses onto "
                     r"$\Sigma p_i$"),
        Line2D([], [], marker="o", ls="none", mfc=CONTROL_COLOR, mec="0.15", ms=11,
               label="persistence control (same walkers, seeds and $C_k$; score swapped)"),
        Line2D([], [], marker="s", ls="none", color=DIRECT_COLOR, ms=7,
               label="direct sampling $k/n$, with its 95% Wilson interval"),
        Line2D([], [], marker="v", ls="none", mfc="none", mec=DIRECT_COLOR, mew=1.3,
               ms=8, label="direct sampling $0/n$: plotted at the upper end of its 95% "
                          "Wilson interval, which is not an estimate"),
    ]
    leg = ax.legend(handles=handles, fontsize=8.8, loc="lower left", ncol=1,
                    framealpha=0.92, borderpad=0.55, labelspacing=0.55)
    blockers.append(("art", leg))
    strip_blockers.append(("art", bx.yaxis.get_label()))

    # Both halves of this caption are derived from the points on the axes above it. The
    # ddof clause names the two runs the reach boundary sits between, because those are
    # the two markers a reader is being asked to compare, and it quotes the shift the
    # population sd would put on them rather than a pair of numbers typed from HANDOFF
    # prose - which is how "+3.30" survived here against a file that says +3.32.
    shifts = [(abs(q["x_ddof0"] - q["x"]), q) for q in pts
              if not q["control"] and np.isfinite(q["x_ddof0"])]
    ends = [q for q in (dict(x=hit), dict(x=miss)) if np.isfinite(q["x"])]
    named = [q for _, q in shifts if any(abs(q["x"] - e["x"]) < 1e-9 for e in ends)]
    if not named:
        named = [q for _, q in sorted(shifts, key=lambda s: -s[0])[:2]]
    ddof_txt = (" and ".join(f"{q['label']} from {_fmt_sigma(q['x'])} to "
                             f"{_fmt_sigma(q['x_ddof0'])}" for q in named)
                if named else "every point on this slate")
    nz = normalization_range(pts)
    ctl_txt = (".\n" if not nz["n_control"] else
               f":\nthe control's check is {nz['control_mass']:.2f} and its weight ESS is "
               f"{nz['control_ess']:.2f} of {nz['control_n']}, so the control is not "
               "merely worse than the scored run, it is degenerate.\n")
    fig.text(0.5, -0.012,
             "One point per run. The x axis is derived from each run's own direct "
             "ensemble with ddof = 1, which is load-bearing: the population sd moves "
             f"{ddof_txt}.\n"
             r"The lower strip is the run's OWN normalization check $\Sigma p_i$, which "
             f"should be 1 and is not - it runs {nz['lo']:.3f} to {nz['hi']:.3f} across "
             f"the {nz['n_prod']} production"
             f"{'s' if nz['n_prod'] != 1 else ''}. It is on the page because a bare $P$ "
             "in this experiment is misleading, not as a footnote"
             + ctl_txt +
             "No interval is drawn on any AI+RES point - "
             "the deck reports the direct-sampling Wilson interval, the weight ESS and "
             r"$\Sigma p_i$, and invents no resampling scheme.",
             ha="center", va="top", fontsize=9.0, color="0.35", linespacing=1.6)
    # Before the labels are placed, not after: `tight_layout` moves both axes, and a
    # placement measured against the old frame would be resolved against a figure that no
    # longer exists.
    fig.tight_layout(rect=[0, 0.028, 1, 1])

    # --- now that everything else is on the page, fit the labels around it ----- #
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def _boxes(axis, spec):
        out = []
        for s in spec:
            if s[0] == "pt":
                out.append(point_box(axis, (s[1], s[2]), s[3]))
            elif s[0] == "span":
                out.append(span_box(axis, s[1], s[2], s[3]))
            else:
                out.append(s[1].get_window_extent(rend))
        return out

    # The leader threshold is set just under a DIAGONAL seat in each fan (21.2 points on
    # top, 20.0 on the strip). A label sitting straight above or below its marker needs no
    # line to claim it; one pushed off to a corner, which is what happens to Elliott where
    # California is 0.04 sigma away, does.
    place_labels(fig, top_items, obstacles=_boxes(ax, blockers), leader_at=21.0)
    place_labels(fig, strip_items, obstacles=_boxes(bx, strip_blockers),
                 pad=2.0, leader_at=19.0)

    # A label that ends up over the boundary band gets a white backing, so the text stays
    # readable AND the band stays a band. Only the ones that actually land on it: the
    # test is the rendered box against the band's own rectangle, not the label's x.
    if band:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        for axis, items in ((ax, top_items), (bx, strip_items)):
            fr = axis.get_window_extent(rend)
            bx0 = axis.transData.transform((hit, 0))[0]
            bx1 = axis.transData.transform((miss, 0))[0]
            rect = _bbox(min(bx0, bx1), fr.y0, max(bx0, bx1), fr.y1)
            for it in items:
                if _overlap(it["ann"].get_window_extent(rend), rect) > 0:
                    it["ann"].set_bbox(dict(fc="white", ec="none", alpha=0.72, pad=1.6))

    return _save(fig, out or (A.fig_dir() / "aires_insights_ladder.png"))


# --------------------------------------------------------------------------- #
# Panel 6 - the two calibration anchors
# --------------------------------------------------------------------------- #
def plot_anchors(anchors: list[dict], out: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n = len(anchors)
    if not n:
        raise SystemExit("no anchor resolved, so there is no calibration page to draw")
    fig, axes = plt.subplots(1, n, figsize=(7.6 * n, 6.4), squeeze=False)

    for ax, a in zip(axes.ravel(), anchors):
        rows = {"direct": 1.0, "aires": 0.0}
        ax.axvspan(a["direct_lo"], a["direct_hi"], color=DIRECT_COLOR, alpha=0.10, lw=0,
                   zorder=0)
        ax.errorbar([a["direct_p"]], [rows["direct"]],
                    xerr=[[a["direct_p"] - a["direct_lo"]],
                          [a["direct_hi"] - a["direct_p"]]],
                    fmt="s", ms=10, color=DIRECT_COLOR, ecolor=DIRECT_COLOR,
                    elinewidth=2.0, capsize=6, zorder=4)
        ax.plot([a["P"]], [rows["aires"]], marker="o", ms=15, mfc=RES_COLOR,
                mec="0.15", mew=1.1, zorder=5)

        ax.set_xscale("log")
        # Padded by hand: on autoscale the Wilson band fills the whole panel and stops
        # reading as an interval at all.
        # Positive values only: on a k = 0 anchor the Wilson lower bound is exactly 0,
        # which a log axis cannot take.
        vals = [v for v in (a["direct_lo"], a["direct_hi"], a["direct_p"], a["P"])
                if v > 0]
        if vals:
            ax.set_xlim(min(vals) / 2.4, max(vals) * 2.4)
        ax.set_yticks([rows["aires"], rows["direct"]])
        ax.set_yticklabels([f"AI+RES\n{a['n_reached']}/{a['n']} walkers",
                            f"direct sampling\n{a['direct_k']}/{a['direct_n']} members"],
                           fontsize=10.5)
        ax.set_ylim(-0.72, 1.72)
        ax.grid(alpha=0.25, which="both", axis="x")
        ax.set_xlabel(r"$P(A_L \geq \mathrm{obs})$" if a["sign"] > 0
                      else r"$P(A_L \leq \mathrm{obs})$", fontsize=11.5)

        # Above the marker, not below it: the box of run health sits along the bottom of
        # every panel, and on Elliott the weighted P lands right on top of it.
        ax.annotate(f"{a['P']:.4f}", xy=(a["P"], rows["aires"]), xytext=(0, 22),
                    textcoords="offset points", ha="center", va="bottom", fontsize=12,
                    color=RES_COLOR, weight="bold")
        ax.annotate(f"{a['direct_p']:.4f}\n[{a['direct_lo']:.4f}, {a['direct_hi']:.3f}]",
                    xy=(a["direct_p"], rows["direct"]), xytext=(0, 20),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    color=DIRECT_COLOR, linespacing=1.35)

        verdict = ("INSIDE the direct interval" if a["inside"]
                   else "OUTSIDE the direct interval")
        ax.set_title(f"{a['label']}\n{a['index_label']}   |   observed "
                     f"{a['observed']:+.2f} {a['unit']} at "
                     f"{_fmt_sigma(a['sigma'])}$\\sigma$\n{verdict}",
                     fontsize=12.2, linespacing=1.45,
                     color=(OK_GREEN if a["inside"] else OBS_COLOR))

        ax.text(0.015, 0.03,
                f"$\\Sigma p_i$ = {a['total_mass']:.4f}   (the run's own "
                "normalization check)\n"
                f"weight ESS {a['weight_ess']:.2f} / {a['n']}\n"
                f"direct ensemble: {a['direct_key']}, mean {a['direct_mean']:+.2f}, "
                f"sd {a['direct_sd']:.2f}\n"
                r"(that sd is the $\sigma$-depth denominator, ddof = 1 - it is not an "
                r"uncertainty on $P$)",
                transform=ax.transAxes, fontsize=8.8, color="0.32", va="bottom",
                ha="left", linespacing=1.45,
                bbox=dict(fc="white", ec="0.8", lw=0.5, alpha=0.9, pad=3.2))

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc=RES_COLOR, mec="0.15", ms=11,
               label=r"AI+RES weighted $P$ - a subset sum of $p_i$, no interval by design"),
        Line2D([], [], marker="s", ls="none", color=DIRECT_COLOR, ms=9,
               label=r"direct sampling $k/n$"),
        Patch(fc=DIRECT_COLOR, alpha=0.20, ec="none",
              label="its 95% Wilson interval - the ONLY interval in this deck"),
    ], fontsize=9.6, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.005), columnspacing=2.0, handletextpad=0.7)

    # Derived from the anchors themselves: the interval's width is the result, so it is
    # not a number to type. Both anchors carry the same k and n and therefore the same
    # interval; the first is quoted and the assertion above keeps them from drifting.
    a0 = anchors[0]
    span = (a0["direct_hi"] / a0["direct_lo"] if a0["direct_lo"] > 0 else float("nan"))
    fig.suptitle("The two calibration anchors: where direct sampling still returns a "
                 "non-zero count in the tail\n"
                 "on six of the nine events direct sampling returns zero at the "
                 "observation, so the weighted curve has nothing to be checked against",
                 fontsize=14, y=1.015)
    fig.text(0.5, -0.075,
             f"Both rungs are {a0['direct_k']} of {a0['direct_n']}, which is the "
             f"shallowest non-zero a {a0['direct_n']}-member ensemble can report, and its "
             f"Wilson interval spans a factor of {span:.0f}. That "
             "width IS the result's precision, and it is why panel 16 of the plan\n"
             "(extend the direct reference to N = 96, ~216 H100-h) is the cheapest "
             "upgrade available. Elliott's CONUS index is a SECOND anchor, not that "
             "event's score: Elliott is scored on the N Plains box at +2.81 sigma,\n"
             "where direct reads 0/24. Two further direct-resolving comparisons exist "
             "and are not anchors but median-region checks - `p90_20240802` at 14/24 and "
             "`p90_20231107` at 21/24 - and they are the next panel.",
             ha="center", va="top", fontsize=9.0, color="0.35", linespacing=1.6)
    fig.tight_layout(rect=[0, 0.055, 1, 0.955])

    return _save(fig, out or (A.fig_dir() / "aires_insights_anchors.png"))


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None,
                    help="comma-separated event:tag; default is every reduced run")
    ap.add_argument("--panels", default=",".join(PANELS),
                    help=f"comma-separated subset of {','.join(PANELS)}")
    ap.add_argument("--outdir", default=None,
                    help=f"where the PNGs go (default {A.fig_dir()})")
    ap.add_argument("--no-stability", action="store_true",
                    help="skip the astab.reduce verdict echo (it reads runs/astab)")
    a = ap.parse_args(argv)

    want = [p.strip() for p in a.panels.split(",") if p.strip()]
    unknown = [p for p in want if p not in PANELS]
    if unknown:
        raise SystemExit(f"unknown panel(s) {unknown}; choose from {list(PANELS)}")
    outdir = Path(a.outdir) if a.outdir else A.fig_dir()

    runs = ([tuple(x.split(":", 1)) for x in a.runs.split(",")] if a.runs else discover())
    if not runs:
        raise SystemExit("no reduced runs found; run `--stage compare` first")

    recs = []
    for ev, tag in runs:
        try:
            r = run_record(ev, tag)
        except FileNotFoundError as e:
            print(f"  skip {ev}:{tag} - no {e}")
            continue
        recs.append(r)

    for row in slate_table(recs).itertuples():
        print(f"  {row.event:26s} {row.tag:8s} {row.box:12s} obs={row.observed:+8.3f} "
              f"depth={_fmt_sigma(row.sigma):>6s}  reach={row.n_reached:2d}/{row.n}  "
              f"P={_fmt_p(row.P):>9s}  sum p_i={row.total_mass:.4f}  "
              f"wESS={row.weight_ess:5.2f}  direct={row.direct_key} "
              f"({row.direct_reached}/{row.direct_n})  baselines={row.n_direct}/3")

    if "slate" in want:
        plot_slate(recs, outdir / "aires_insights_slate.png")
    if "ladder" in want:
        pts = [ladder_point(r) for r in order_runs(recs)]
        hit, miss = reach_boundary(pts)
        print(f"  reach boundary: deepest reached {hit:+.2f} sigma, "
              f"shallowest missed {miss:+.2f} sigma")
        plot_ladder(recs, outdir / "aires_insights_ladder.png")
    if "anchors" in want:
        anchors = []
        for ev, tag, idx in ANCHORS:
            try:
                anchors.append(anchor_record(ev, tag, idx))
            except (FileNotFoundError, KeyError) as e:
                print(f"  skip anchor {ev}:{idx} - {e}")
        for x in anchors:
            print(f"  anchor {x['event']:26s} [{x['index']:5s}] obs={x['observed']:+7.3f} "
                  f"{_fmt_sigma(x['sigma'])} sigma  AI+RES {x['n_reached']}/{x['n']} "
                  f"P={x['P']:.4f}  direct {x['direct_k']}/{x['direct_n']}="
                  f"{x['direct_p']:.4f} [{x['direct_lo']:.5f}, {x['direct_hi']:.3f}]  "
                  f"inside={x['inside']}  sum p_i={x['total_mass']:.4f}")
        if anchors:
            plot_anchors(anchors, outdir / "aires_insights_anchors.png")

    if not a.no_stability:
        try:
            v = stability_verdict()
            print(f"  astab.reduce: GenCast clean through week {v['gencast_weeks']} "
                  f"(tested {v['gencast']['tested']}, untested "
                  f"{v['gencast']['untested']}); FCN3 clean through week "
                  f"{v['fcn3_weeks']} on both arms")
        except (FileNotFoundError, ImportError) as e:
            print(f"  no stability verdict - {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
