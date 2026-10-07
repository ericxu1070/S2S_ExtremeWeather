#!/usr/bin/env python
"""The ensemble follow-up: 100 GenCast walkers at ONE lead, and what fraction survive it.

Job 1189 rolled ONE member per lead. It found that divergence is not monotonic in lead -
PNW's week-8 chain blew up at 45 d while its week-10 chain never left bounds - and with
n = 1 it could not separate *that lead is unstable* from *that member diverged*. This
module reduces the run that can: N members of the same chain, same event, same lead, same
deterministic ERA5 init, differing only in GenCast's diffusion noise stream.

The deliverable is a SURVIVAL CURVE - the fraction of members still inside the physical
bounds at each 3-day checkpoint, with a 95% interval - plus the failure-mode breakdown
(loud vs silent, which variable, which level) that job 1189 showed a CONUS-only panel
cannot produce.

Four things this module is careful about
----------------------------------------
* **Wilson, not Wald.** The interesting answers are at the ends: 0 of 100 diverged by
  21 d, or 100 of 100 by 56 d. Wald gives both a zero-width interval, which is not a
  statement about anything. Wilson stays a proper interval at k = 0 and k = n.
* **First out of bounds and DIVERGED are different dates**, and the distinction is
  measured rather than assumed: PNW@wk8 left bounds at 42 d by 0.5% of the bound span (a
  humidity excursion inside the noise of where that floor was drawn) and only became an
  unrecoverable blow-up at 45 d. Both dates are reported for every member; the survival
  curve draws both.
* **Incomplete members are named, never averaged in.** A member whose shard was killed is
  a job failure, not a physical result. Per-checkpoint fractions use the members actually
  at risk at that checkpoint; the headline at the final lead uses the complete members and
  says how many that was.
* **BOUNDS is applied once, in ``reduce``.** Every member summary is built from
  ``reduce.walker_panel``, so the severity, the SST-mask growth rule and the phase-matched
  drift baseline are the sweep's, not a second implementation that could drift from it.

What the answer does NOT cover: one event, one lead, one initial condition. The spread is
GenCast's sampling spread alone - there is no IC perturbation - so this measures the
model's numerical stability under its own diffusion noise, and says nothing about whether
the score function still RANKS out there (that needs a rank correlation against an
ensemble of outcomes, which is the other experiment).

Env: imports cleanly in BOTH ``moe`` and ``fcn3``. No JAX, no torch; matplotlib is
imported inside the figure stage.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from aires import aconfig as A
from astab import sconfig as SC
from astab.diag import (BOUNDS, HARD_METRICS, MARGINAL_SEVERITY, load_bands,
                        status_for)
from astab.record import jsonable
from astab.reduce import CSV_METRICS, walker_panel
from astab.schedule import Chain

# The two-sided 95% normal quantile, spelled out so a change is a visible edit.
Z95 = 1.959963984540054

# The CONUS diagnostic panel. A divergence that trips any of these is LOUD; one that trips
# only the global bounds check is SILENT. Job 1189 measured that split at 1 loud to 2
# silent, and the silent pair is why the sweep's own conclusion is that a long-lead AI+RES
# run must gate on the global state rather than on its CONUS observable.
#
# Mirrors ``stages.CALIB_METRICS`` - the metrics ``--stage calibrate`` measured bands for -
# and is pinned equal to it by the tests rather than imported, so this module does not
# depend on the stage module.
LOUD_METRICS = ("t2m_anom_conus", "t2m_conus_drift", "diurnal_ratio", "conus_t2m_sd",
                "z500_hi_wavenumber_frac")

# Fates, worst first. This IS the precedence order used by ``member_summary``.
FATES = ("diverged", "marginal", "incomplete", "clean")


# --------------------------------------------------------------------------- #
# The interval
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k/n``.

    Wilson rather than Wald because the answers this run cares about live at the ends. At
    k = 0 of 100, Wald gives ``[0, 0]`` - "we are certain no member ever diverges" off a
    sample that could not have shown one - while Wilson gives ``[0, 0.037]``, which is
    the honest statement. At k = n it is the mirror image, and "100/100 diverged" is
    exactly the result a 56-day run might produce.

    ``n = 0`` returns the whole interval: no members at risk is no information.
    """
    n = int(n)
    k = int(k)
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"wilson: k={k} outside 0..n={n}")
    denom = n + z * z
    centre = (k + z * z / 2.0) / denom
    half = (z / denom) * math.sqrt(k * (n - k) / n + z * z / 4.0)
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- #
# One member
# --------------------------------------------------------------------------- #
def member_summary(chain: Chain, bands: dict | None = None) -> dict:
    """Everything one member contributes to the curve, from its per-segment records.

    Built on ``reduce.walker_panel``: the bounds verdict, the severity, the SST-mask
    growth rule and the phase-matched drift baseline are the sweep's own, so a member here
    and a chain in ``stability.csv`` are scored by identical code.

    The fate precedence is ``diverged > marginal > incomplete > clean`` and the order
    matters in exactly one case: a member that blew up and whose shard then died. Calling
    that "incomplete" would hide a real physical result behind a job failure - so
    divergence wins, and ``complete`` is carried separately so the member still appears in
    every incompleteness count. What must never happen is the other way round: an
    unfinished member that never left bounds reading as "clean", which is why
    ``incomplete`` outranks it.
    """
    bands = load_bands(chain.event) if bands is None else bands
    rows = sorted(walker_panel(chain), key=lambda r: float(r["lead_days"]))
    segs = chain.schedule
    leads = [float(r["lead_days"]) for r in rows]
    sev = [float(r.get("bounds_severity", 0.0) or 0.0) for r in rows]
    bnd = [float(r.get("bounds", 0.0) or 0.0) for r in rows]
    nf = [float(r.get("nonfinite", 0.0) or 0.0) for r in rows]

    complete = (len(rows) == len(segs)
                and bool(leads) and abs(leads[-1] - chain.walk_days) < 1e-9)

    first_oob = first_oob_var = None
    first_oob_sev = float("nan")
    for r, s, b, f in zip(rows, sev, bnd, nf):
        if b > 0 or f > 0:
            first_oob = float(r["lead_days"])
            first_oob_var = str(r.get("bounds_worst_var", "") or "")
            first_oob_sev = s
            break

    diverged = diverged_var = None
    for r, s, f in zip(rows, sev, nf):
        if s > MARGINAL_SEVERITY or f > 0:
            diverged = float(r["lead_days"])
            diverged_var = str(r.get("bounds_worst_var", "") or "")
            break

    worst_sev = max(sev, default=0.0)
    worst_i = int(np.argmax(sev)) if sev else -1
    worst_var = str(rows[worst_i].get("bounds_worst_var", "") or "") if rows else ""
    worst_day = float(rows[worst_i]["lead_days"]) if rows else float("nan")
    worst_step = int(rows[worst_i]["step"]) if rows else -1
    worst_level = _worst_level(chain, worst_step, worst_var) if worst_var else None

    # RECOVERED: it left the bounds hard and came back inside them. Reported rather than
    # folded into "diverged", because a model that returns to a physical state is a
    # different object from one that runs away - and the survival curve is cumulative, so
    # a recovered member still counts as diverged from its onset date onward.
    recovered = bool(diverged is not None and sev and sev[-1] <= MARGINAL_SEVERITY)

    if diverged is not None:
        fate = "diverged"
    elif first_oob is not None:
        fate = "marginal"
    elif not complete:
        fate = "incomplete"
    else:
        fate = "clean"

    mode, mode_metrics, mode_extrap = _failure_mode(rows, bands, diverged)

    return dict(
        event=chain.event, weeks=int(chain.weeks), member=int(chain.member or 0),
        seed=int(chain.seed), label=chain.label, slug=chain.slug,
        walk_days=int(chain.walk_days), n_segments=len(segs), n_records=len(rows),
        complete=bool(complete), last_lead=(leads[-1] if leads else None),
        first_oob=first_oob, first_oob_var=first_oob_var,
        first_oob_sev=float(first_oob_sev),
        diverged=diverged, diverged_var=diverged_var,
        worst_sev=float(worst_sev), worst_var=worst_var, worst_day=worst_day,
        worst_level=worst_level, recovered=recovered, fate=fate,
        mode=mode, mode_metrics=mode_metrics, mode_extrapolated=bool(mode_extrap),
        leads=leads, sev=sev, bounds=bnd, nonfinite=nf,
        status=[_status_at(s, b, f) for s, b, f in zip(sev, bnd, nf)],
        t2m_anom=[float(r.get("t2m_anom_conus", float("nan"))) for r in rows],
        rows=rows,
    )


def _status_at(sev: float, bounds: float, nonfinite: float) -> str:
    """Per checkpoint, the three states the heatmap draws."""
    if sev > MARGINAL_SEVERITY or nonfinite > 0:
        return "diverged"
    if bounds > 0 or sev > 0:
        return "marginal"
    return "ok"


def _failure_mode(rows, bands, diverged) -> tuple[str, list, bool]:
    """LOUD or SILENT: did any CONUS diagnostic fire at or after the divergence?

    The same ``diag.status_for`` call the sweep's reduce makes, so a member and a chain
    are judged loud or silent by identical thresholds. The bands are measured to 21 d and
    EXTRAPOLATED beyond, so whether the verdict rests on an extrapolated band is carried
    alongside it rather than left for the reader to work out.

    A member that never diverged has no mode; the field is empty rather than "silent",
    which would read as a claim about a member that never failed at all.
    """
    if diverged is None:
        return ("", [], False)
    fired: list[str] = []
    extrap = False
    for r in rows:
        if float(r["lead_days"]) < diverged - 1e-9:
            continue
        for m in LOUD_METRICS:
            if m not in r:
                continue
            st, ex = status_for(m, r[m], bands, float(r["lead_days"]))
            if st == "FAIL":
                fired.append(m)
                extrap |= bool(ex)
    return ("loud" if fired else "silent", sorted(set(fired)), extrap)


def _worst_level(chain: Chain, step: int, var: str):
    """Which LEVEL of ``var`` carried the worst excursion, from the segment's record.

    Job 1189's two silent divergences were ``temperature`` at 50 hPa over Antarctica while
    the CONUS troposphere stayed textbook-normal. With 100 members and no retained states,
    ``extremes_global`` (written at walk time, last frame) and ``ranges_global_levels``
    (both frames, per level) are the only surviving attribution.

    Returns ``None`` for a single-level variable, for a record written before those fields
    existed, or when the variable is inside its bounds.
    """
    if step < 1 or var not in BOUNDS:
        return None
    p = chain.record_path(step)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text())
    except Exception:
        return None
    lo, hi = BOUNDS[var]
    ext = (rec.get("extremes_global") or {}).get(var) or {}
    for end, ok in (("min", lambda v: v < lo), ("max", lambda v: v > hi)):
        e = ext.get(end)
        if isinstance(e, dict) and "level" in e and ok(float(e.get("value", 0.0))):
            return float(e["level"])
    # Fall back to the per-level ranges, which cover BOTH frames and therefore match the
    # severity the panel actually reported when the worse value sat in the earlier frame.
    per = (rec.get("ranges_global_levels") or {}).get(var) or {}
    span = hi - lo
    best, best_lv = 0.0, None
    for lv, r in per.items():
        mn, mx = float(r.get("min", np.nan)), float(r.get("max", np.nan))
        if not np.isfinite(mn) or not np.isfinite(mx):
            return float(lv)
        e = max((lo - mn) / span, (mx - hi) / span)
        if e > best:
            best, best_lv = e, float(lv)
    return best_lv


def format_member_verdict(s: dict) -> str:
    """The one line a shard log prints when a member finishes.

    ``m007: complete; left bounds 45 d (temperature 12% past); DIVERGED 45 d (silent);
    worst 64% at 56 d``. It exists so a 4.5 h log answers "did this member blow up?"
    without waiting for the reduce - a diverged member is the interesting one and must not
    be discoverable only in aggregate.
    """
    bits = [f"m{s['member']:03d}:",
            "complete" if s["complete"] else
            f"INCOMPLETE ({s['n_records']}/{s['n_segments']} segments)"]
    if s["first_oob"] is None:
        bits.append("never left bounds")
    else:
        lv = "" if s["worst_level"] is None else f" @ {s['worst_level']:g} hPa"
        bits.append(f"left bounds {s['first_oob']:g} d "
                    f"({s['first_oob_var'] or '?'}{lv} "
                    f"{s['first_oob_sev'] * 100:.1f}% past)")
    if s["diverged"] is not None:
        bits.append(f"DIVERGED {s['diverged']:g} d ({s['mode']}"
                    + (", band extrapolated" if s["mode_extrapolated"] else "") + ")")
        if s["recovered"]:
            bits.append("recovered by the end")
    if s["worst_sev"] > 0:
        bits.append(f"worst {s['worst_sev'] * 100:.0f}% at {s['worst_day']:g} d "
                    f"({s['worst_var'] or '?'})")
    return "; ".join([bits[0] + " " + bits[1]] + bits[2:])


# --------------------------------------------------------------------------- #
# The population
# --------------------------------------------------------------------------- #
def survival_curve(summaries: list[dict], checkpoints: list[float]) -> list[dict]:
    """Per checkpoint: how many members are still at risk, clean, and undiverged.

    Three counts, and they answer three different questions:

    * ``n_clean``       - cumulative: has NEVER left the bounds by this lead. This is the
                          survival curve in the usual sense and it is monotone.
    * ``n_undiverged``  - cumulative, using ``MARGINAL_SEVERITY``: has never been past the
                          bounds by more than a marginal excursion. Also monotone, and it
                          is the one the headline quotes, because "left bounds" at 0.5%
                          past a floor measured on 18 states is not "stopped being an
                          atmosphere".
    * ``n_inbounds_now`` - INSTANTANEOUS: inside the bounds at this checkpoint, whatever
                          it did earlier. Not monotone, and the gap between it and
                          ``n_clean`` is exactly the recovered members.

    ``n_at_risk`` counts the members with a RECORD at this checkpoint - not the members in
    the run. A shard killed at 30 d contributes to every checkpoint up to 30 d and to none
    after it, so the denominator shrinks rather than the numerator being quietly wrong.
    """
    out = []
    for c in checkpoints:
        c = float(c)
        at_risk = [s for s in summaries if any(abs(l - c) < 1e-9 for l in s["leads"])]
        n = len(at_risk)
        n_clean = sum(1 for s in at_risk
                      if s["first_oob"] is None or s["first_oob"] > c + 1e-9)
        n_undiv = sum(1 for s in at_risk
                      if s["diverged"] is None or s["diverged"] > c + 1e-9)
        n_now = sum(1 for s in at_risk if float(_at(s, c, "bounds") or 0.0) == 0.0)
        lo_c, hi_c = wilson(n_clean, n)
        lo_u, hi_u = wilson(n_undiv, n)
        out.append(dict(
            checkpoint_day=c, n_at_risk=n,
            n_clean=n_clean, frac_clean=(n_clean / n if n else float("nan")),
            clean_lo=lo_c, clean_hi=hi_c,
            n_undiverged=n_undiv,
            frac_undiverged=(n_undiv / n if n else float("nan")),
            undiverged_lo=lo_u, undiverged_hi=hi_u,
            n_inbounds_now=n_now,
            frac_inbounds_now=(n_now / n if n else float("nan")),
            n_diverged=n - n_undiv,
            frac_diverged=((n - n_undiv) / n if n else float("nan")),
            diverged_lo=1.0 - hi_u, diverged_hi=1.0 - lo_u,
            median_sev=_median([_at(s, c, "sev") for s in at_risk])))
    return out


def _median(values) -> float:
    v = np.asarray([x for x in values if x is not None], dtype="float64")
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _at(s: dict, c: float, field: str):
    """One member's value of ``field`` at checkpoint ``c``; ``None``/nan when absent."""
    for i, l in enumerate(s["leads"]):
        if abs(l - c) < 1e-9:
            return s[field][i]
    return None if field == "status" else float("nan")


# --------------------------------------------------------------------------- #
# stage: reduce  (login node, either env)
# --------------------------------------------------------------------------- #
def stage_ens_reduce(chs: list[Chain]) -> int:
    """The long CSV, the members CSV, the summary JSON and the printed headline.

    Tolerates missing and partial members on purpose: this is also the MID-FLIGHT monitor.
    Running it against a 4.5 h job that is two hours in gives the survival curve so far,
    over the members that have reached each checkpoint - which is the only cheap way to
    find out that the run is producing what was expected before it finishes.
    """
    groups: dict[tuple[str, int], list[Chain]] = {}
    for ch in chs:
        if ch.member is None:
            print(f"[ens-reduce] skipping {ch.label}: not a member chain. The ensemble "
                  f"reduce reads runs/astab/<event>/ens/, never the frozen sweep's "
                  f"walkers/ tree.", file=sys.stderr)
            continue
        groups.setdefault((ch.event, ch.weeks), []).append(ch)
    if not groups:
        print("[ens-reduce] ERROR: no member chains selected. Pass --ens N.",
              file=sys.stderr)
        return 1

    rc = 0
    for (event, weeks), members in sorted(groups.items()):
        rc |= _reduce_one(event, weeks, sorted(members, key=lambda c: c.member))
    return rc


def _reduce_one(event: str, weeks: int, members: list[Chain]) -> int:
    bands = load_bands(event)
    summaries = [member_summary(ch, bands) for ch in members]
    with_data = [s for s in summaries if s["n_records"]]
    if not with_data:
        print(f"[ens-reduce] ERROR: {event} @ wk{weeks:02d}: not one of "
              f"{len(members)} member(s) has a segment record yet "
              f"({SC.ens_dir(event, weeks)}).", file=sys.stderr)
        return 1

    ch0 = members[0]
    checkpoints = [float(l) for (_k, l, _n) in ch0.schedule]

    # -- the long table: one row per (member, checkpoint, metric) ------------ #
    long_rows = []
    for s in summaries:
        for r in s["rows"]:
            for m in CSV_METRICS:
                if m not in r:
                    continue
                v = r[m]
                if m in HARD_METRICS:
                    status, extrap, band = ("FAIL" if float(v or 0) > 0 else "ok"), False, ""
                else:
                    status, extrap = status_for(m, v, bands, r["lead_days"])
                    b = bands.get(m)
                    band = f"[{b.lo:.4g},{b.hi:.4g}]" if b else ""
                long_rows.append(dict(
                    event=event, weeks=weeks, lead_days=ch0.horizon_days,
                    model="gencast", member=s["member"], seed=s["seed"], fate=s["fate"],
                    checkpoint_day=r["lead_days"], metric=m, value=v, band=band,
                    status=status, extrapolated=int(bool(extrap)),
                    reached_peak=int(bool(s["complete"])),
                    valid_time=r.get("valid_time", "")))
    long = pd.DataFrame(long_rows)
    out_csv = SC.ens_csv_path(event, weeks)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_csv, index=False)

    # -- the wide table: one row per member ---------------------------------- #
    wide = pd.DataFrame([{k: v for k, v in s.items()
                          if k not in ("rows", "leads", "sev", "bounds", "nonfinite",
                                       "status", "t2m_anom", "mode_metrics")}
                         | {"mode_metrics": ",".join(s["mode_metrics"])}
                         for s in summaries])
    wide.to_csv(SC.ens_members_csv_path(event, weeks), index=False)

    curve = survival_curve(summaries, checkpoints)
    payload = dict(
        event=event, weeks=int(weeks), horizon_days=ch0.horizon_days,
        init=str(ch0.init), peak=str(ch0.peak), walk_days=ch0.walk_days,
        n_members=len(members), n_with_records=len(with_data),
        n_complete=sum(1 for s in summaries if s["complete"]),
        checkpoints=checkpoints,
        provenance=_provenance(event, weeks, members),
        bands={k: b.as_dict() for k, b in bands.items()},
        curve=curve,
        members=[{k: v for k, v in s.items() if k != "rows"} for s in summaries],
        caveats=CAVEATS,
    )
    head = _headline(event, weeks, summaries, curve, checkpoints)
    payload["headline"] = head
    p = SC.ens_summary_path(event, weeks)
    tmp = p.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=jsonable))
    os.replace(tmp, p)

    print(f"[ens-reduce] {event} @ wk{weeks:02d}: {len(long)} row(s) -> {out_csv}")
    print(f"[ens-reduce]   {len(summaries)} member row(s) -> "
          f"{SC.ens_members_csv_path(event, weeks)}")
    print(f"[ens-reduce]   summary -> {p}")
    print(head)
    return 0


CAVEATS = [
    "n members are GenCast diffusion draws from ONE deterministic ERA5 init: this is "
    "sampling spread, not initial-condition spread, and the run says nothing about "
    "sensitivity to the IC.",
    "One event and one lead. Divergence was already measured to be non-monotonic in lead "
    "(job 1189), so this fraction may not transfer to another lead of the same event.",
    "The diagnostic bands were measured on Gate 3 walkers to 21 d and are EXTRAPOLATED "
    "beyond; the loud/silent split past 21 d rests on that extrapolation. The bounds "
    "check itself is absolute and is not extrapolated.",
    "Numerical stability only. Staying inside physical bounds at 56 d is PERMISSION to "
    "run AI+RES there, not evidence that the score function still ranks: Gate 2 put "
    "FCN3's rank-equivalence horizon at ~8.5 d.",
]


def _provenance(event: str, weeks: int, members: list[Chain]) -> dict:
    """What produced these members. The wrong-checkpoint incident (jobs 1155/1156) is why
    the requested checkpoint AND the one stamped on a state are both recorded."""
    kw = {}
    try:
        kw = A.walker_model_kwargs()
    except Exception:
        pass
    stamped = ""
    for ch in members:
        p = ch.record_path(1)
        if p.exists():
            try:
                stamped = str(json.loads(p.read_text()).get("params_file", ""))
            except Exception:
                stamped = ""
            if stamped:
                break
    return dict(
        params_file=kw.get("params_file", ""), res=kw.get("res"),
        resolution=A.WALKER_RES, stamped_on_record=stamped,
        base_seed=A.WALKER_BASE_SEED,
        seed_rule="segment_key(member, step) = fold_in(fold_in(PRNGKey(base_seed), "
                  "step), member) - the same family xres and gencast_s2s use for their "
                  "ensembles",
        control_member=sorted(SC.ENS_KEEP_STATE_MEMBERS),
        seg_steps=A.RES_SEG_STEPS, step_h=A.WALKER_STEP_H,
        marginal_severity=MARGINAL_SEVERITY, diag_vars=list(SC.ENS_DIAG_VARS),
        members=[int(c.member) for c in members])


# The frozen sweep's single member at PNW week 8, for the one comparison this run can make
# against a result that already exists. Job 1189: left bounds 42 d, diverged 45 d, loud.
JOB_1189 = dict(event="PNW_HeatDome_2021", weeks=8, member=2,
                first_oob=42.0, diverged=45.0, mode="loud")


def _headline(event: str, weeks: int, summaries: list[dict], curve: list[dict],
              checkpoints: list[float]) -> str:
    n_all = len(summaries)
    complete = [s for s in summaries if s["complete"]]
    incomplete = [s for s in summaries if not s["complete"]]
    horizon = max(checkpoints) if checkpoints else float("nan")
    div = [s for s in complete if s["diverged"] is not None]
    oob = [s for s in complete if s["first_oob"] is not None]
    rec = [s for s in complete if s["recovered"]]
    n = len(complete)
    lo, hi = wilson(len(div), n)
    onsets = sorted(s["diverged"] for s in div)
    modes = Counter(s["mode"] for s in div)
    var_counts = Counter(f"{s['worst_var'] or '?'}"
                         + ("" if s["worst_level"] is None
                            else f" ({s['worst_level']:g} hPa)")
                         for s in div)

    L = ["", "=" * 78, "  ENSEMBLE HEADLINE"]
    L.append(f"    {event} @ {horizon:g} d (week {weeks}), "
             f"{n} complete of {n_all} member(s)")
    L.append(f"    diverged {len(div)} ({len(div) / n * 100:.0f}%, Wilson 95% "
             f"{lo * 100:.0f}-{hi * 100:.0f}%)  |  left bounds {len(oob)}  |  "
             f"recovered {len(rec)}"
             if n else "    no complete member yet")
    if onsets:
        L.append(f"    onset: first at {onsets[0]:g} d, median "
                 f"{float(np.median(onsets)):g} d, last at {onsets[-1]:g} d")
        L.append(f"    modes: loud {modes.get('loud', 0)} / silent "
                 f"{modes.get('silent', 0)}   (loud = a CONUS diagnostic fired too; "
                 f"silent = only the GLOBAL bounds check saw it)")
        L.append(f"    worst variable: "
                 + ", ".join(f"{k} {v}" for k, v in var_counts.most_common()))
    else:
        L.append("    no complete member has diverged")

    L.append("")
    L.append(f"    {'lead':>6s}  {'at risk':>7s}  {'never OOB':>18s}  "
             f"{'never diverged':>18s}  {'in bounds now':>13s}")
    for row in curve:
        if row["n_at_risk"] == 0:
            continue
        L.append(f"    {row['checkpoint_day']:6.0f}  {row['n_at_risk']:7d}  "
                 f"{row['n_clean']:4d} {row['frac_clean'] * 100:5.1f}% "
                 f"[{row['clean_lo'] * 100:4.1f},{row['clean_hi'] * 100:5.1f}]  "
                 f"{row['n_undiverged']:4d} {row['frac_undiverged'] * 100:5.1f}% "
                 f"[{row['undiverged_lo'] * 100:4.1f},{row['undiverged_hi'] * 100:5.1f}]  "
                 f"{row['n_inbounds_now']:6d} "
                 f"{row['frac_inbounds_now'] * 100:5.1f}%")

    if incomplete:
        L.append("")
        L.append(f"    {len(incomplete)} member(s) INCOMPLETE - a job failure, not a "
                 f"physical result. They are excluded from the headline")
        L.append(f"    fractions and counted per checkpoint only where they have a "
                 f"record:")
        for s in sorted(incomplete, key=lambda s: s["member"]):
            L.append(f"      m{s['member']:03d}: {s['n_records']}/{s['n_segments']} "
                     f"segment(s), last lead "
                     f"{s['last_lead'] if s['last_lead'] is not None else '-'}")

    # The one comparison against a result that already exists.
    if event == JOB_1189["event"] and int(weeks) == JOB_1189["weeks"]:
        m2 = next((s for s in summaries if s["member"] == JOB_1189["member"]), None)
        L.append("")
        if m2 is None:
            L.append(f"    member {JOB_1189['member']} (job 1189's noise stream) is not "
                     f"in this selection")
        else:
            L.append(f"    member {JOB_1189['member']} shares job 1189's noise stream "
                     f"(same slot, same init) - a DETERMINISM CONTROL, not an")
            L.append(f"    independent draw. 1189: left bounds "
                     f"{JOB_1189['first_oob']:g} d, diverged {JOB_1189['diverged']:g} d, "
                     f"{JOB_1189['mode']}.")
            L.append(f"    here: {format_member_verdict(m2)}")
            L.append(f"    bf16/XLA is not bit-reproducible (the 1189 re-walk gave "
                     f"297.43 vs 297.42 K), so a difference is a finding about "
                     f"sensitivity,")
            L.append(f"    not a bug. It is counted once, like any other member: "
                     f"exchangeable a priori.")

    L.append("")
    for c in CAVEATS:
        L += ["    " + line for line in _wrap(c, 74)]
    L.append("=" * 78)
    return "\n".join(L)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- #
# stage: figs
#
# The figure answers three questions in four panels and as few words as it can:
#   A  how many members are still an atmosphere at each lead (the deliverable);
#   B  how far past a physical bound each one went, in % of the bound's span - the raw
#      quantity behind A;
#   C  directly under B on the same lead axis: what that did to the CONUS-mean T2m, in
#      kelvin - the observable AI+RES resamples on. A crossing in B and a collapse in C
#      are the same lead by eye, and the dot on each trajectory marks that lead;
#   D  kelvin against percent for every checkpoint that was past a bound at all, with
#      the 2% line drawn through it. This is the panel that shows the threshold is a
#      real one: every checkpoint past 2% sits below every clean checkpoint, and no
#      trajectory ever comes back up.
#
# The reference against which "cold" is judged is the run's OWN clean population - the
# members that never touched a bound, at every checkpoint - rather than the Gate 3 band.
# The band was measured to 21 d and extrapolated beyond; the clean members are measured
# at every lead the figure draws, so the extrapolation caveat is not needed here. (The
# band still decides loud vs silent in the CSV and JSON; nothing about that changes.)
#
# Palette - every chromatic value below was CHECKED, not chosen by eye
# (dataviz/scripts/validate_palette.py, light mode, surface #fcfcfb):
#
#   panel A's two curves are ORDINAL, not categorical: two THRESHOLDS on the same
#   quantity (ever out of bounds / ever past the marginal severity), so they take one
#   hue in two lightness steps. `--ordinal`: monotone L PASS, adjacent dL PASS, light-end
#   contrast 2.36:1 PASS, single hue (spread 19 deg) PASS.
#
#   clean -> marginal -> diverged is a STATUS palette with reserved meaning, and it
#   keeps `#d62728` because that is the red `astab/plots.py` marks a divergence with -
#   the two figures have to read as one system. `--pairs all`: CVD dE 16.8 deutan /
#   18.6 normal PASS. The sweep's reference chain is drawn in neutral ink, dashed: it is
#   not a member and takes no hue.
#
# `#6baed6` (2.36:1) and `#ff7f0e` (2.47:1) sit below 3:1 against the surface, which the
# validator WARNs on and which obligates a relief channel. The relief is shipped: every
# curve is direct-labelled with its own final value, every legend entry carries its
# count, and `ensemble_<event>_wk<NN>_members.csv` is the table-view twin.
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
SURV_CLEAN = "#6baed6"        # never left the bounds at all      (light step)
SURV_UNDIV = "#08519c"        # never past MARGINAL_SEVERITY      (dark step, headline)
ST_MARGINAL = "#ff7f0e"
ST_DIVERGED = "#d62728"       # the same red astab/plots.py marks a divergence with
INK = "#1a1a19"
INK_MUTED = "#6b6b66"
GRID = "#dcdcd8"
# A LINE of the "clean" state needs more value than a fill would: `#c6c6c0` is the
# lightest grey that is still a visible 1 px line on the surface.
FATE_COLOR = {"clean": "#c6c6c0", "marginal": ST_MARGINAL, "diverged": ST_DIVERGED,
              "incomplete": "#7a7a73"}
FATE_DASH = {"clean": "-", "marginal": "-", "diverged": "-", "incomplete": (0, (3, 2))}
REF_DASH = (0, (5, 2))
REF_LABEL = "job 1189, the sweep's single chain"
# Severity is plotted on a log axis and a clean member's severity is exactly 0. Zeros
# are drawn ON the axis floor rather than dropped, so a clean member is a visible flat
# line at the bottom instead of an absence the eye reads as missing data; the floor's
# tick is labelled "0" for the same reason.
SEV_FLOOR = 1e-4
SEV_TICKS = ((1e-4, "0"), (1e-3, "0.1%"), (1e-2, "1%"), (1e-1, "10%"), (1.0, "100%"))
PCT_TICKS = ((0.002, "0.2%"), (0.005, "0.5%"), (0.01, "1%"), (0.02, "2%"), (0.05, "5%"),
             (0.1, "10%"), (0.2, "20%"), (0.5, "50%"), (1.0, "100%"))


def marginal_in_units() -> list[tuple[str, str]]:
    """What ``MARGINAL_SEVERITY`` of a bound's span IS, in each field's own units.

    Computed from ``BOUNDS`` rather than typed, so the figure cannot quote a number the
    gate does not use: 2% of the 160 K T2m span is 3.2 K past 180 K or 340 K, 2% of the
    400 m/s wind span is 8 m/s past 200 m/s, and so on.
    """
    out = []
    for var, label, scale, unit, fmt in (
            ("2m_temperature", "T2m", 1.0, "K", "{:.1f}"),
            ("temperature", "T aloft", 1.0, "K", "{:.0f}"),
            ("u_component_of_wind", "wind", 1.0, "m/s", "{:.0f}"),
            ("specific_humidity", "q", 1e3, "g/kg", "{:.1f}")):
        if var in BOUNDS:
            lo, hi = BOUNDS[var]
            out.append((label, fmt.format(MARGINAL_SEVERITY * (hi - lo) * scale)
                        + f" {unit}"))
    return out


def stage_ens_figs(chs: list[Chain]) -> int:
    groups: dict[tuple[str, int], list[Chain]] = {}
    for ch in chs:
        if ch.member is not None:
            groups.setdefault((ch.event, ch.weeks), []).append(ch)
    if not groups:
        print("[ens-figs] ERROR: no member chains selected. Pass --ens N.",
              file=sys.stderr)
        return 1
    rc = 0
    for (event, weeks), members in sorted(groups.items()):
        rc |= _fig_one(event, weeks, sorted(members, key=lambda c: c.member))
    return rc


def _reference_chain(event: str, weeks: int, bands: dict) -> dict | None:
    """The frozen sweep's own chain at this (event, lead), summarised by the same code
    as the members, when there is one on disk. Job 1189's PNW@wk8 is the single chain
    the ensemble was run to put in context, so it is drawn through panels B-D as a
    dashed reference. ``None`` when the sweep tree is absent (a test RUNS, another box):
    the figure must not depend on it."""
    if event != JOB_1189["event"] or int(weeks) != JOB_1189["weeks"]:
        return None
    try:
        s = member_summary(Chain(event, weeks), bands)
    except Exception:
        return None
    return s if s["n_records"] else None


def _fig_one(event: str, weeks: int, members: list[Chain]) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bands = load_bands(event)
    summaries = [member_summary(ch, bands) for ch in members]
    live = [s for s in summaries if s["n_records"]]
    if not live:
        print(f"[ens-figs] ERROR: {event} @ wk{weeks:02d}: no member has a record yet.",
              file=sys.stderr)
        return 1
    ch0 = members[0]
    checkpoints = [float(l) for (_k, l, _n) in ch0.schedule]
    curve = survival_curve(summaries, checkpoints)
    complete = [s for s in summaries if s["complete"]]
    div = [s for s in complete if s["diverged"] is not None]
    ref = _reference_chain(event, weeks, bands)

    fig, axd = plt.subplot_mosaic(
        [["A", "A", "A"], ["B", "B", "D"], ["C", "C", "D"]], figsize=(16.5, 11.0),
        gridspec_kw=dict(height_ratios=[0.78, 0.6, 0.8], width_ratios=[1.0, 1.0, 1.12]))
    for ax in axd.values():
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

    _panel_survival(axd["A"], curve, checkpoints)
    _panel_severity(axd["B"], summaries, checkpoints, ref)
    _panel_anomaly(axd["C"], summaries, checkpoints, ref, Line2D)
    _panel_kelvin_vs_percent(axd["D"], summaries, ref, Line2D)

    n_c = len(complete)
    lo, hi = wilson(len(div), n_c)
    onsets = sorted(s["diverged"] for s in div)
    n_rec = sum(1 for s in div if s["recovered"])
    if n_c:
        head = (f"{len(div)}/{n_c} diverged by {max(checkpoints):g} d "
                f"({len(div) / n_c * 100:.0f}%, Wilson 95% {lo * 100:.0f}-{hi * 100:.0f}%)")
        if onsets:
            head += (f", onsets {onsets[0]:g}-{onsets[-1]:g} d, "
                     f"{n_rec if n_rec else 'none'} recovered")
    else:
        head = f"{len(live)} member(s) in flight, no complete member yet"
    fig.suptitle(
        f"GenCast walker survival to {ch0.horizon_days} d  -  {event.replace('_', ' ')}, "
        f"week {weeks} (init {ch0.init.date()} -> peak {ch0.peak.date()}), "
        f"{len(members)} diffusion members from one ERA5 init\n"
        f"{head}.   Diverged = some field more than {MARGINAL_SEVERITY * 100:g}% of its "
        f"physical span past a bound, anywhere on the globe.",
        fontsize=11, y=0.995, va="top", color=INK)
    # Explicit margins rather than tight_layout: the right-edge labels of A and C and
    # the shared-axis pair B/C are laid out by hand.
    fig.subplots_adjust(left=0.055, right=0.975, top=0.915, bottom=0.065,
                        hspace=0.34, wspace=0.42)
    out = SC.ens_fig_path(event, weeks)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[ens-figs] wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return 0


# --------------------------------------------------------------------------- #
# Shared pieces
# --------------------------------------------------------------------------- #
def _trajectories(summaries: list[dict], ref: dict | None):
    """Every curve the lead-axis panels draw: (summary, color, dash, weight, alpha, z).
    Diverged members over clean ones over the reference, so the nine are never buried
    under the ninety-one and the reference is never mistaken for a member."""
    out = []
    for s in summaries:
        if not s["leads"]:
            continue
        is_div = s["diverged"] is not None
        out.append((s, FATE_COLOR.get(s["fate"], FATE_COLOR["clean"]),
                    FATE_DASH.get(s["fate"], "-"),
                    1.3 if is_div else 0.9, 0.6 if is_div else 0.3, 4 if is_div else 3))
    if ref is not None and ref["leads"]:
        out.append((ref, INK, REF_DASH, 1.4, 0.85, 5))
    return out


def _onset_dots(ax, summaries: list[dict], ref: dict | None, field: str, fn=None) -> None:
    """A filled dot on each diverged trajectory at its FIRST checkpoint past 2%.

    The same lead in B and C, so the crossing and the collapse are matched by eye; and
    the surface ring keeps a dot legible where nine trajectories cross."""
    fn = fn or (lambda v: v)
    curves = [(s, ST_DIVERGED) for s in summaries] + ([(ref, INK)] if ref else [])
    for s, color in curves:
        if s["diverged"] is None:
            continue
        v = _at(s, s["diverged"], field)
        if v is None or not np.isfinite(float(v)):
            continue
        ax.plot([s["diverged"]], [fn(float(v))], "o", ms=5.5, mfc=color, mec=SURFACE,
                mew=1.2, zorder=7)


def _clean_population(summaries: list[dict]) -> list[dict]:
    """The members "cold" is judged against: never touched a bound. On a mid-flight
    reduce nothing is complete yet, so it falls back to whatever has not diverged."""
    clean = [s for s in summaries if s["leads"] and s["fate"] == "clean"]
    return clean or [s for s in summaries if s["leads"] and s["diverged"] is None]


def _finite(values) -> np.ndarray:
    v = np.asarray(list(values), dtype="float64")
    return v[np.isfinite(v)]


# --------------------------------------------------------------------------- #
# The panels
# --------------------------------------------------------------------------- #
def _panel_survival(ax, curve, checkpoints) -> None:
    """A. The deliverable: the fraction still standing at each checkpoint, with a 95% CI.

    A step function, because that is what it is: a member's state changes at a checkpoint
    and holds until the next one. Drawn from lead 0 at 1.0 so the reader sees the whole
    survival, not the part after the first checkpoint.
    """
    x = [0.0] + [r["checkpoint_day"] for r in curve]
    ends = []
    for key, lo_k, hi_k, color, lw, label in (
            ("frac_undiverged", "undiverged_lo", "undiverged_hi", SURV_UNDIV, 2.2,
             f"never diverged (never past {MARGINAL_SEVERITY * 100:g}% of a span)"),
            ("frac_clean", "clean_lo", "clean_hi", SURV_CLEAN, 2.0,
             "never left the bounds")):
        # A checkpoint no member has reached yet has n_at_risk = 0, and wilson(0, 0)
        # is the whole interval [0, 1] by design ("no information"). Drawing that as a
        # wash would put a 0-100% band over the un-walked part of a mid-flight figure,
        # which reads as a measurement. Leave it blank instead, like the curve itself.
        y = [1.0] + [r[key] for r in curve]
        lo = [1.0] + [r[lo_k] if r["n_at_risk"] else np.nan for r in curve]
        hi = [1.0] + [r[hi_k] if r["n_at_risk"] else np.nan for r in curve]
        ax.fill_between(x, lo, hi, step="post", color=color, alpha=0.16, lw=0, zorder=2)
        ax.step(x, y, where="post", color=color, lw=lw, label=label, zorder=4,
                solid_capstyle="round")
        if np.isfinite(y[-1]):
            ends.append((float(y[-1]), color))
    # Direct label at the right edge - the relief channel the contrast WARN obligates,
    # and the number a reader actually wants off this panel. ONE label when the two
    # curves end on the same value, or the two print on top of each other.
    seen: set = set()
    for yv, color in ends:
        k = round(yv, 3)
        if k in seen:
            continue
        seen.add(k)
        ax.annotate(f" {yv * 100:.0f}%", (x[-1], yv), color=INK, fontsize=10,
                    fontweight="bold", va="center", ha="left", xytext=(4, 0),
                    textcoords="offset points", zorder=6)

    # How many members each fraction is over, printed ONLY if the denominator moves
    # (a shard killed mid-run). A survival curve whose denominator changes without
    # saying so is not a measurement; a row of identical "100"s is noise.
    live = [(i, r) for i, r in enumerate(curve) if r["n_at_risk"]]
    changes = [(i, r) for j, (i, r) in enumerate(live)
               if j and r["n_at_risk"] != live[j - 1][1]["n_at_risk"]]
    if changes:
        for i, r in [live[0]] + changes:
            ax.annotate(f"{r['n_at_risk']}", (r["checkpoint_day"], 0.0),
                        xytext=(0, -30), textcoords="offset points", ha="center",
                        fontsize=8, color=INK_MUTED, annotation_clip=False)
        ax.annotate("members at risk", (0.0, 0.0), xytext=(-8, -30),
                    textcoords="offset points", ha="right", fontsize=8, color=INK_MUTED,
                    annotation_clip=False)

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, max(checkpoints) * 1.06 if checkpoints else 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("fraction of members", fontsize=10, color=INK)
    ax.set_xlabel("lead (days from init)", fontsize=10, color=INK,
                  labelpad=30 if changes else 4)
    ax.set_title("A.  survival, with 95% Wilson intervals", fontsize=11.5, loc="left",
                 color=INK)
    ax.legend(fontsize=9, loc="lower left", frameon=False)


def _sev_floor(v) -> np.ndarray:
    """Severity onto the log axis: NaN and zero on the floor, +inf left as NaN so it is
    drawn by the marker branch and not by the line."""
    a = np.asarray(v, dtype="float64")
    a = np.where(np.isposinf(a), np.nan, a)
    return np.clip(np.nan_to_num(a, nan=SEV_FLOOR), SEV_FLOOR, None)


def _panel_severity(ax, summaries, checkpoints, ref) -> None:
    """B. Every member's excursion past its bounds, in % of the bound's span - the raw
    quantity behind panel A. No x tick labels: it shares its lead axis with C below."""
    top = SEV_FLOOR
    for s, color, dash, lw, alpha, z in _trajectories(summaries, ref):
        y = np.asarray(s["sev"], dtype="float64")
        finite = y[np.isfinite(y)]
        top = max(top, float(finite.max()) if finite.size else SEV_FLOOR)
        ax.plot(s["leads"], _sev_floor(y), lw=lw, alpha=alpha, color=color, ls=dash,
                zorder=z)
    # An infinite severity is a field with no finite value left in it - item 1's
    # failure, off the top of any axis. Drawn as an open marker at the ceiling so it
    # is present and obviously not a number.
    for s in summaries:
        inf_leads = [l for l, v in zip(s["leads"], s["sev"]) if not np.isfinite(v)]
        if inf_leads:
            ax.plot(inf_leads, [top * 4] * len(inf_leads), "o", mfc="none",
                    mec=ST_DIVERGED, ms=7, mew=1.4, zorder=6)
    _onset_dots(ax, summaries, ref, "sev", fn=lambda v: float(_sev_floor([v])[0]))

    ax.axhline(MARGINAL_SEVERITY, color=INK_MUTED, lw=1.2, ls="--", zorder=5)
    ax.annotate(f" {MARGINAL_SEVERITY * 100:g}% of the span = diverged",
                (0.01, MARGINAL_SEVERITY), xycoords=ax.get_yaxis_transform(),
                fontsize=8.5, color=INK_MUTED, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(SEV_FLOOR / 2, max(top * 6, 0.3))
    ticks = [(t, l) for t, l in SEV_TICKS if t <= top * 6]
    ax.set_yticks([t for t, _l in ticks])
    ax.set_yticklabels([l for _t, l in ticks])
    ax.minorticks_off()
    ax.set_xlim(0, max(checkpoints) if checkpoints else 1)
    ax.tick_params(labelbottom=False)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("worst excursion past a bound\n(% of the bound's span)", fontsize=9.5,
                  color=INK)
    ax.set_title("B.  how far past a physical bound", fontsize=11.5, loc="left",
                 color=INK)


def _panel_anomaly(ax, summaries, checkpoints, ref, Line2D) -> None:
    """C. The same trajectories at the surface, in kelvin, on B's lead axis.

    The reference for "cold" is the coldest checkpoint of any CLEAN member - measured at
    every lead drawn, so no extrapolated band is needed. The two ranges at the right edge
    are the direct labels: what the diverged and the clean members read at the last
    checkpoint.
    """
    counts = Counter(s["fate"] for s in summaries)
    for s, color, dash, lw, alpha, z in _trajectories(summaries, ref):
        ax.plot(s["leads"], s["t2m_anom"], lw=lw, alpha=alpha, color=color, ls=dash,
                zorder=z)
    _onset_dots(ax, summaries, ref, "t2m_anom")

    clean = _clean_population(summaries)
    floor = _finite(v for s in clean for v in s["t2m_anom"])
    floor = float(floor.min()) if floor.size else float("nan")
    if np.isfinite(floor):
        ax.axhline(floor, color=INK_MUTED, lw=1.0, ls=(0, (1, 2)), zorder=2)
        ax.annotate(f" coldest clean checkpoint, {floor:+.1f} K", (0.01, floor),
                    xycoords=ax.get_yaxis_transform(), fontsize=8, color=INK_MUTED,
                    va="top")
    ax.axhline(0.0, color=GRID, lw=1.0, zorder=1)

    last = max(checkpoints) if checkpoints else float("nan")

    def _ends(pop):
        return _finite(s["t2m_anom"][-1] for s in pop
                       if s["leads"] and abs(s["leads"][-1] - last) < 1e-9)

    all_vals = _finite(v for s in summaries for v in s["t2m_anom"])
    if ref is not None:
        all_vals = np.concatenate([all_vals, _finite(ref["t2m_anom"])])
    y0 = float(min(all_vals.min(), floor if np.isfinite(floor) else 0.0, 0.0)) \
        if all_vals.size else -1.0
    y1 = float(max(all_vals.max(), 0.0)) if all_vals.size else 1.0
    pad = 0.12 * max(y1 - y0, 1.0)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlim(0, last if np.isfinite(last) else 1)

    d_end = _ends([s for s in summaries if s["diverged"] is not None])
    c_end = _ends(clean)
    # Ink, not the series colour: the label sits at the end of the bundle it describes,
    # and position carries the identity (text wears text tokens).
    labels = []
    if d_end.size:
        labels.append((float(d_end.mean()), INK,
                       f" {d_end.max():.0f} to {d_end.min():.0f} K"))
    if c_end.size:
        labels.append((float(c_end.mean()), INK,
                       f" {c_end.min():+.0f} to {c_end.max():+.0f} K"))
    # Two labels closer than a line's height get pushed apart, never overprinted.
    if len(labels) == 2 and abs(labels[0][0] - labels[1][0]) < 0.08 * (y1 - y0 + 2 * pad):
        mid = (labels[0][0] + labels[1][0]) / 2
        off = 0.045 * (y1 - y0 + 2 * pad)
        labels = [(mid - off, labels[0][1], labels[0][2]),
                  (mid + off, labels[1][1], labels[1][2])]
    for yv, color, text in labels:
        ax.annotate(text, (last, yv), color=color, fontsize=9, fontweight="bold",
                    va="center", ha="left", xytext=(3, 0), textcoords="offset points",
                    zorder=6, annotation_clip=False)

    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("CONUS-mean T2m anomaly  (K)", fontsize=9.5, color=INK)
    ax.set_xlabel("lead (d)", fontsize=9.5, color=INK)
    ax.set_title("C.  the same members at the surface, in kelvin", fontsize=11.5,
                 loc="left", color=INK)
    handles = [Line2D([], [], color=FATE_COLOR[f], lw=2.2, ls=FATE_DASH[f],
                      label=f"{f} ({counts.get(f, 0)})")
               for f in FATES if counts.get(f)]
    if ref is not None:
        handles.append(Line2D([], [], color=INK, lw=1.6, ls=REF_DASH, label=REF_LABEL))
    handles.append(Line2D([], [], marker="o", ms=6, mfc=ST_DIVERGED, mec=SURFACE,
                          ls="none", label=f"first checkpoint past "
                                           f"{MARGINAL_SEVERITY * 100:g}%"))
    ax.legend(handles=handles, fontsize=8, loc="lower left", frameon=False)


def _panel_kelvin_vs_percent(ax, summaries, ref, Line2D) -> None:
    """D. Kelvin against percent: every checkpoint of every trajectory that was past a
    bound at all, hollow below 2% and filled at or past it, joined in time order.

    The grey band is the whole clean population at every checkpoint. The panel makes
    the threshold's case without a sentence: the hollow points sit in or at the band,
    every filled point sits below it, and every path runs down and to the right.
    """
    clean = _clean_population(summaries)
    cv = _finite(v for s in clean for v in s["t2m_anom"])
    lo_c, hi_c = (float(cv.min()), float(cv.max())) if cv.size else (np.nan, np.nan)

    paths = []
    for s in summaries:
        if not s["leads"]:
            continue
        pts = [(float(x), float(y)) for x, y in zip(s["sev"], s["t2m_anom"])
               if np.isfinite(x) and x > 0 and np.isfinite(y)]
        if pts:
            color = ST_DIVERGED if s["diverged"] is not None else ST_MARGINAL
            paths.append((pts, color, "-", 0.5))
    if ref is not None:
        pts = [(float(x), float(y)) for x, y in zip(ref["sev"], ref["t2m_anom"])
               if np.isfinite(x) and x > 0 and np.isfinite(y)]
        if pts:
            paths.append((pts, INK, REF_DASH, 0.8))

    if np.isfinite(lo_c):
        ax.axhspan(lo_c, hi_c, color=INK_MUTED, alpha=0.09, lw=0, zorder=0)
        for edge in (lo_c, hi_c):
            ax.axhline(edge, color=INK_MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.annotate(f" the {len(clean)} clean members, every checkpoint", (0.01, hi_c),
                    xycoords=ax.get_yaxis_transform(), fontsize=8, color=INK_MUTED,
                    va="bottom")
    ax.axhline(0.0, color=GRID, lw=1.0, zorder=1)

    for pts, color, dash, alpha in paths:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, lw=1.0, ls=dash, alpha=alpha, color=color, zorder=3)
        for x, y in pts:
            filled = x >= MARGINAL_SEVERITY
            ax.plot([x], [y], "o", ms=6, mfc=color if filled else SURFACE,
                    mec=SURFACE if filled else color, mew=1.2 if filled else 1.4,
                    zorder=6 if filled else 5)

    # The line needs no label of its own: "2%" is a tick on this axis.
    ax.axvline(MARGINAL_SEVERITY, color=INK_MUTED, lw=1.2, ls="--", zorder=4)
    # What the crossing reads in kelvin, from the data: the anomaly at each diverged
    # member's first checkpoint past 2%. Placed just right of the onset cluster, at its
    # own height - the far-past points are all much lower, so nothing is there.
    onset = _finite(_at(s, s["diverged"], "t2m_anom") for s in summaries
                    if s["diverged"] is not None)
    onset_x = _finite(_at(s, s["diverged"], "sev") for s in summaries
                      if s["diverged"] is not None)
    if onset.size and onset_x.size:
        ax.annotate(f"first checkpoint past {MARGINAL_SEVERITY * 100:g}%:\n"
                    f"{onset.max():.0f} to {onset.min():.0f} K",
                    (float(onset_x.max()) * 1.6, float(onset.max())), ha="left",
                    va="top", fontsize=8.5, color=INK, zorder=8)

    xs_all = _finite(x for pts, *_r in paths for x, _y in pts)
    ys_all = _finite(y for pts, *_r in paths for _x, y in pts)
    x0 = min(0.003, float(xs_all.min()) * 0.6) if xs_all.size else 0.003
    x1 = max(0.06, float(xs_all.max()) * 2.5) if xs_all.size else 0.06
    ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ticks = [(t, l) for t, l in PCT_TICKS if x0 <= t <= x1]
    ax.set_xticks([t for t, _l in ticks])
    ax.set_xticklabels([l for _t, l in ticks])
    ax.minorticks_off()
    ylo = min(float(ys_all.min()) if ys_all.size else 0.0, lo_c if np.isfinite(lo_c) else 0.0)
    yhi = max(float(ys_all.max()) if ys_all.size else 0.0, hi_c if np.isfinite(hi_c) else 0.0)
    pad = 0.12 * max(yhi - ylo, 1.0)
    ax.set_ylim(ylo - pad, yhi + pad)
    if not paths:
        ax.annotate("no member has left the bounds", (0.5, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize=10, color=INK_MUTED)

    # 2% of a span, in the units of the fields that carried it - top right, above the
    # band, where no trajectory goes (every point past a bound is at or below it).
    note = f"{MARGINAL_SEVERITY * 100:g}% of a span = " + "  ·  ".join(
        f"{v} {lbl}" for lbl, v in marginal_in_units())
    ax.annotate(note, (0.99, 0.99), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color=INK_MUTED)

    ax.grid(axis="both", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("worst excursion past a bound  (% of the bound's span)", fontsize=9.5,
                  color=INK)
    ax.set_ylabel("CONUS-mean T2m anomaly  (K)", fontsize=9.5, color=INK)
    ax.set_title("D.  kelvin against percent, every checkpoint past a bound",
                 fontsize=11.5, loc="left", color=INK)
    handles = [Line2D([], [], marker="o", ms=6, mfc=SURFACE, mec=ST_DIVERGED, mew=1.4,
                      ls="none", label=f"below {MARGINAL_SEVERITY * 100:g}%"),
               Line2D([], [], marker="o", ms=6, mfc=ST_DIVERGED, mec=SURFACE, ls="none",
                      label=f"at or past {MARGINAL_SEVERITY * 100:g}%")]
    if ref is not None:
        handles.append(Line2D([], [], color=INK, lw=1.6, ls=REF_DASH, label=REF_LABEL))
    ax.legend(handles=handles, fontsize=8, loc="lower left", frameon=False)
