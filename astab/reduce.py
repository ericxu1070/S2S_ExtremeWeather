#!/usr/bin/env python
"""stage: reduce - three panels per chain, one long CSV, and the headline.

One row per (chain, model, checkpoint, metric). The three models are the GenCast walker,
FCN3 launched from an ERA5 IC (adapter-free) and FCN3 launched from the walker's lead-3 d
state (adapter-fed); their curves are assembled from different artifacts but land in the
same table so a single query answers "where did anything first fail".

Two rules this module exists to enforce
---------------------------------------
* **"Stable through week X" means EVERY chain at that week, not the best one.** Job 1189 is
  why: PNW's 70-day chain never leaves bounds while Uri's leaves at 30 d, so a
  max-over-chains rule would have reported "stable through week 10" for a model that blew
  up on one of the two chains at that lead. ``_model_survival`` and ``_all_clean_through``
  are the all-chains rule, and both the printed headline and the figure's subtitle go
  through them.
* **A model is only credited with a lead it was actually rolled to.** At an
  FCN3-extension lead (weeks 12..20) the GenCast walker rolls 3 d, not 140, purely to
  launch the adapter-fed arm. Its panel there is one clean checkpoint - and a survival rule
  that counted it would report "GenCast stable through week 20" off three days of rollout.
  ``reached_peak`` is carried per CSV row and ``_model_survival`` drops any (model, week)
  where it is 0: not a failure, not a pass, simply not tested. The same flag catches the
  other way a model can fail to cover a lead - an FCN3 cube that stops short, which at 560
  steps is a live possibility - so it is MEASURED for the scorer, not assumed.
* **First out of bounds and DIVERGED are different dates.** PNW@wk8 first leaves bounds at
  42 d by 0.5% of the bound span - a humidity excursion inside the noise of where that
  floor was drawn - and only becomes an unrecoverable blow-up at 45 d. Reporting the first
  date with the worst magnitude would date the divergence early; ``MARGINAL_SEVERITY``
  carries the distinction and the raw excursion goes in the CSV, where the choice stays
  visible and revisable.

``BOUNDS`` is applied HERE, to the ranges the walk stage stored - never read back from a
verdict a record baked in. See ``diag.field_ranges``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from aires import aindex as AI
from astab import sconfig as SC
from astab.diag import (HARD_METRICS, MARGINAL_SEVERITY, NONFINITE_STATIC, bound_severity,
                        conus_series, diurnal_ratio_series, field_ranges, hard_nonfinite,
                        load_bands, nonfinite_fractions, status_for,
                        violations_from_ranges, zonal_high_wavenumber_fraction,
                        _phase_matched_first)
from astab.record import jsonable
from astab.schedule import Chain
from gencast_s2s import config as C

def _fcn3_panel(chain: Chain, *, arm: str) -> list[dict]:
    """The scorer's curve: one row per FCN3 output frame that lands on a 3-day boundary.

    CONUS items come from the cube; the global items (4 and 7) come from the retained
    pre-crop Zarr, because the cube is CONUS-cropped before it reaches disk.
    """
    if arm == "free":
        cube_p, zarr_p, launch = (chain.fcn3_free_cube_path(),
                                  chain.fcn3_free_zarr_path(), chain.init)
    else:
        cube_p, zarr_p, launch = (chain.score_cube_path(), chain.score_zarr_path(),
                                  chain.valid_at(SC.SCORE_LEAD_DAYS))
    if not cube_p.exists():
        return []
    rows = []
    with xr.open_dataset(cube_p) as cube:
        cube = AI._squeeze(cube.load())
        if "member" in cube.dims:
            cube = cube.isel(member=0, drop=True)
        t = pd.DatetimeIndex(cube["time"].values)
        ser = conus_series(cube)
        # FCN3's step is 6 h, so 00Z/12Z pairs are 2 frames apart; the diurnal helper
        # needs adjacent frames to BE the pair, so it is fed the 12-hourly subsample.
        sub = cube.isel(time=[i for i, x in enumerate(t) if x.hour in (0, 12)])
        dur = diurnal_ratio_series(sub)
        base = _phase_matched_first(t, ser["t2m_mean"].values, hour=0)
        for i, ti in enumerate(t):
            lead = (ti - launch).total_seconds() / 86400.0
            if abs(lead % A.GATE3_SEG_DAYS) > 1e-9 or lead <= 0:
                continue
            # Per FRAME, not per cube: "first hard failure at lead X" is the headline, and
            # reducing the whole cube to one flag would date every failure to lead 3 d.
            frame = cube.isel(time=i)
            nf = nonfinite_fractions(frame)
            bv = violations_from_ranges(field_ranges(frame))
            rows.append(dict(
                lead_days=float((ti - chain.init).total_seconds() / 86400.0),
                fcn3_lead_days=float(lead), valid_time=str(ti),
                t2m_conus_mean=float(ser["t2m_mean"].iloc[i]),
                t2m_anom_conus=float(ser["t2m_anom"].iloc[i]),
                conus_t2m_sd=float(ser["t2m_sd"].iloc[i]),
                t2m_conus_drift=float(ser["t2m_mean"].iloc[i] - base),
                diurnal_ratio=(float(dur.loc[ti, "ratio"]) if ti in dur.index else np.nan),
                nonfinite=hard_nonfinite(nf),
                bounds=(1.0 if bv else 0.0),
                bounds_severity=bound_severity(field_ranges(frame)),
            ))
    if zarr_p.exists():
        try:
            _add_fcn3_global(rows, zarr_p, launch)
        except Exception as e:
            print(f"  [reduce] {chain.label} {arm}: global items unavailable from "
                  f"{zarr_p.name} ({type(e).__name__}: {e})")
    else:
        print(f"  [reduce] {chain.label} {arm}: no retained Zarr at {zarr_p}; the global "
              f"items (4 and 7) are unavailable for this arm. Re-run the score stage "
              f"with --keep-zarr.")
    return rows


def _add_fcn3_global(rows: list[dict], zarr_p: Path, launch) -> None:
    """Global-mean T2m/z500 and the zonal spectrum from the pre-crop Zarr."""
    ds = xr.open_zarr(zarr_p)
    by_lead = {r["fcn3_lead_days"]: r for r in rows}
    lead_h = pd.TimedeltaIndex(ds["lead_time"].values)
    # FCN3 channel -> the GenCast variable whose BOUNDS entry applies to it. This is what
    # lets the GLOBAL bounds check run on the scorer at all: its cube is CONUS-cropped, and
    # job 1189 showed a CONUS-only check misses a stratospheric blow-up entirely.
    GLOBAL_BOUNDED = {"t2m": "2m_temperature", "t50": "temperature",
                      "z500": "geopotential"}
    for j, lt in enumerate(lead_h):
        lead = lt.total_seconds() / 86400.0
        r = by_lead.get(float(lead))
        if r is None:
            continue
        sl = dict(ensemble=0, time=0, lead_time=j)
        ranges = {}
        for ch_name, gname in GLOBAL_BOUNDED.items():
            if ch_name not in ds:
                continue
            a = ds[ch_name].isel(**sl).load()
            v = np.asarray(a.values, dtype="float64")
            fin = np.isfinite(v)
            ranges[gname] = dict(min=float(v[fin].min()) if fin.any() else float("nan"),
                                 max=float(v[fin].max()) if fin.any() else float("nan"))
            if ch_name == "z500":
                r["global_z500"] = float(AI.area_mean(a))
                r["z500_hi_wavenumber_frac"] = zonal_high_wavenumber_fraction(a)
            elif ch_name == "t2m":
                r["global_t2m"] = float(AI.area_mean(a))
            elif ch_name == "t50":
                r["global_t50"] = float(AI.area_mean(a))
            del a, v
        if ranges:
            sev, var = bound_severity(ranges, which=True)
            # The CONUS check already ran; keep whichever is worse, and say which variable.
            if sev > float(r.get("bounds_severity", 0.0)):
                r["bounds_severity"] = sev
                r["bounds_worst_var"] = var
            if sev > 0:
                r["bounds"] = 1.0
    ds.close()


def _walker_panel(chain: Chain) -> list[dict]:
    """The walker's curve, assembled from the per-segment records written at walk time."""
    rows = []
    base = None
    nf_base: dict = {}
    for (k, lead, n_steps) in chain.schedule:
        p = chain.record_path(k)
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        conus = rec.get("conus", {})
        glob = rec.get("global", {})
        if base is None and conus.get("t2m_mean"):
            # Segment boundaries are always 00Z (3-day multiples of a 00Z init), but a
            # segment's FIRST frame is 12Z. Comparing across that phase would put a ~4 K
            # diurnal offset into every drift number.
            base = _phase_matched_first(pd.DatetimeIndex(conus["times"]),
                                        conus["t2m_mean"],
                                        hour=pd.Timestamp(rec["valid_time"]).hour)
        # The chain's own first segment is the non-finite baseline; see NONFINITE_STATIC.
        if not nf_base:
            nf_base = {**rec.get("nonfinite_global", {}),
                       **rec.get("nonfinite_conus", {})}
        nf = max(hard_nonfinite(rec.get("nonfinite_global", {}), nf_base),
                 hard_nonfinite(rec.get("nonfinite_conus", {}), nf_base))
        sst_nf = max([rec.get("nonfinite_global", {}).get(v, 0.0)
                      for v in NONFINITE_STATIC] or [0.0])
        # BOUNDS is applied HERE, to the stored ranges - never read back from a verdict a
        # walk-time record baked in. See field_ranges.
        sg, vg = bound_severity(rec.get("ranges_global", {}), which=True)
        sc, vc = bound_severity(rec.get("ranges_conus", {}), which=True)
        sev, sev_var = (sg, vg) if sg >= sc else (sc, vc)
        bad = sev > 0
        rows.append(dict(
            lead_days=float(lead), step=int(k), n_steps=int(n_steps),
            valid_time=rec.get("valid_time"),
            t2m_conus_mean=conus.get("t2m_mean_last", np.nan),
            t2m_anom_conus=conus.get("t2m_anom_last", np.nan),
            conus_t2m_sd=conus.get("t2m_sd_last", np.nan),
            t2m_conus_drift=(float(conus["t2m_mean_last"]) - base
                             if base is not None and "t2m_mean_last" in conus else np.nan),
            diurnal_ratio=conus.get("diurnal_ratio_mean", np.nan),
            global_t2m=glob.get("global_t2m", np.nan),
            global_z500=glob.get("global_z500", np.nan),
            z500_hi_wavenumber_frac=glob.get("z500_hi_wavenumber_frac", np.nan),
            era5_global_t2m=rec.get("era5", {}).get("global_t2m", np.nan),
            era5_global_z500=rec.get("era5", {}).get("global_z500", np.nan),
            era5_hi_wavenumber_frac=rec.get("era5", {}).get("z500_hi_wavenumber_frac",
                                                           np.nan),
            nonfinite=float(nf), bounds=(1.0 if bad else 0.0),
            nonfinite_sst=float(sst_nf), bounds_severity=float(sev),
            bounds_worst_var=sev_var,
        ))
    return rows


# The ONE public name for the walker's curve. ``astab.ensemble`` builds every member
# summary from it, so the BOUNDS logic, the phase-matched drift baseline and the
# SST-mask growth rule are reused rather than re-implemented - which matters because a
# second implementation would drift from this one and the two would disagree about which
# members diverged.
walker_panel = _walker_panel

CSV_METRICS = ("nonfinite", "bounds", "t2m_anom_conus", "t2m_conus_drift",
               "diurnal_ratio", "conus_t2m_sd", "z500_hi_wavenumber_frac",
               "global_t2m", "global_z500", "global_t50", "nonfinite_sst",
               "bounds_severity")


def csv_path() -> Path:
    return SC.csv_path()


def _first_hard_failure(rows: list[dict]) -> float | None:
    """Lead in days of the first HARD failure (item 1 or 2), or ``None``."""
    for r in sorted(rows, key=lambda r: r["lead_days"]):
        if float(r.get("nonfinite", 0) or 0) > 0 or float(r.get("bounds", 0) or 0) > 0:
            return float(r["lead_days"])
    return None


def _merge_panels(panels: dict, chain: Chain) -> dict:
    """Carry forward any per-checkpoint field a fresh panel could no longer compute."""
    p = chain.result_path()
    if not p.exists():
        return panels
    try:
        prev = json.loads(p.read_text()).get("panels", {})
    except Exception:
        return panels
    carried = 0
    for model, rows in panels.items():
        old_by_lead = {float(r["lead_days"]): r for r in prev.get(model, [])}
        for r in rows:
            for k, v in old_by_lead.get(float(r["lead_days"]), {}).items():
                if k not in r and v is not None:
                    r[k] = v
                    carried += 1
    if carried:
        print(f"  [reduce] {chain.label}: carried {carried} value(s) forward from the "
              f"previous panel (their source has been pruned)")
    return panels


def stage_reduce(chs: list[Chain]) -> int:
    long_rows = []
    verdicts = []
    for ch in chs:
        bands = load_bands(ch.event)
        panels = {"gencast": _walker_panel(ch),
                  "fcn3_era5": _fcn3_panel(ch, arm="free"),
                  "fcn3_adapter": _fcn3_panel(ch, arm="fed")}
        # MERGE with what a previous reduce computed. `--stage prune` deletes the pre-crop
        # Zarrs, so a reduce run afterwards cannot recompute the FCN3 global items - and
        # without this it would overwrite them with a panel that silently lacks those
        # columns. Measured once, kept; that is the same rule the walk-time records follow.
        panels = _merge_panels(panels, ch)
        result = dict(
            event=ch.event, weeks=ch.weeks, walker=ch.walker, tag=SC.STAB_TAG,
            horizon_days=ch.horizon_days, init=str(ch.init), peak=str(ch.peak),
            walk_days=ch.walk_days, walks_to_peak=bool(ch.walks_to_peak),
            segments=[[int(k), float(l), int(n)] for (k, l, n) in ch.schedule],
            scoring_checkpoints=ch.checkpoints,
            n_walker_members=SC.N_WALKER_MEMBERS, n_fcn3_members=SC.N_FCN3_MEMBERS,
            provenance=_provenance(ch),
            bands={k: b.as_dict() for k, b in bands.items()},
            panels=panels,
            first_hard_failure={m: _first_hard_failure(p) for m, p in panels.items()},
        )
        # Did each model actually cover this chain? For the walker it is a configuration
        # question (`walks_to_peak`); for the FCN3 arms it is a measurement, because a
        # rollout can also stop early. At 560 steps that is no longer hypothetical, and a
        # cube that ends at step 200 would otherwise contribute a full set of clean
        # checkpoints and read as a clean 140-day rollout.
        last_ckpt = int(ch.horizon_days / A.GATE3_SEG_DAYS) * A.GATE3_SEG_DAYS
        reached = {m: (ch.walks_to_peak if m == "gencast" else
                       bool(rows) and max(r["lead_days"] for r in rows)
                       >= last_ckpt - 1e-9)
                   for m, rows in panels.items()}
        for model, rows in panels.items():
            if rows and not reached[model]:
                print(f"  [reduce] {ch.label} {model}: covers {max(r['lead_days'] for r in rows):g} d "
                      f"of {ch.horizon_days} d - INCOMPLETE, so it is reported but not "
                      f"counted as a verdict at this lead")
            for r in rows:
                for m in CSV_METRICS:
                    if m not in r:
                        continue
                    v = r[m]
                    if m in HARD_METRICS:
                        status = "FAIL" if float(v or 0) > 0 else "ok"
                        extrap = False
                        band = ""
                    else:
                        status, extrap = status_for(m, v, bands, r["lead_days"])
                        b = bands.get(m)
                        band = f"[{b.lo:.4g},{b.hi:.4g}]" if b else ""
                    long_rows.append(dict(
                        event=ch.event, weeks=ch.weeks, lead_days=ch.horizon_days,
                        model=model, checkpoint_day=r["lead_days"], metric=m,
                        value=v, band=band, status=status,
                        extrapolated=int(bool(extrap)),
                        # Did THIS model roll to the peak on THIS chain? 0 for the walker
                        # at an FCN3-extension lead, and 0 for an FCN3 arm whose cube stops
                        # short. `_model_survival` refuses to draw a verdict from a row
                        # marked 0 - see the module docstring.
                        reached_peak=int(reached[model]),
                        valid_time=r.get("valid_time", "")))
        result["reached_peak"] = {m: bool(v) for m, v in reached.items()}
        p = ch.result_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2, default=jsonable))
        print(f"[reduce] {ch.label}: wrote {p}  "
              f"(gencast {len(panels['gencast'])} ckpt, "
              f"fcn3-era5 {len(panels['fcn3_era5'])}, "
              f"fcn3-adapter {len(panels['fcn3_adapter'])})")
        verdicts.append((ch, result["first_hard_failure"]))

    if not long_rows:
        print("[reduce] ERROR: nothing to reduce - no per-segment records and no FCN3 "
              "cubes on disk.", file=sys.stderr)
        return 1
    df = pd.DataFrame(long_rows)
    out = csv_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[reduce] wrote {out} ({len(df)} rows)")

    print(_headline(verdicts, df))
    return 0


def _provenance(ch: Chain) -> dict:
    """What made these artifacts. Prescribed by HANDOFF after the wrong-checkpoint
    incident: nothing on disk from jobs 1155/1156 said which model produced it."""
    kw = {}
    try:
        kw = A.walker_model_kwargs()
    except Exception:
        pass
    stamped = ""
    p = ch.state_path(1)
    if p.exists():
        with xr.open_dataset(p) as s:
            stamped = str(s.attrs.get("params_file", ""))
    return dict(params_file=kw.get("params_file", ""), res=kw.get("res"),
                resolution=A.WALKER_RES, stamped_on_state=stamped,
                base_seed=A.WALKER_BASE_SEED, seg_steps=A.RES_SEG_STEPS,
                step_h=A.WALKER_STEP_H, tag=SC.STAB_TAG,
                schedule=[[int(k), float(l), int(n)] for (k, l, n) in ch.schedule])


def untested_weeks(df: pd.DataFrame, model: str) -> list[int]:
    """Weeks present in the table where ``model`` was not rolled to the peak.

    The walker at an FCN3-extension lead. Reported rather than hidden: "GenCast: weeks
    12-20 not tested" is a different statement from "GenCast: weeks 12-20 clean", and the
    figure and the headline both have to make it.
    """
    if "reached_peak" not in df.columns:
        return []                                   # a CSV from before the extension
    sub = df[df.model == model]
    return sorted({int(w) for w, g in sub.groupby("weeks")
                   if not bool(g.reached_peak.astype(int).min())})


def _model_survival(df: pd.DataFrame, model: str) -> dict:
    """Per WEEK: did EVERY chain stay inside bounds, and where did the first one leave?

    "Stable through week X" has to mean every chain at that week, not the best one. Job
    1189 is why: PNW's 70-day chain never leaves bounds while Uri's leaves at 30 d, so a
    max-over-chains rule would have reported "stable through week 10" for a model that
    blew up on one of the two chains at that lead.

    A week the model was NOT rolled to the peak at is absent from the result entirely -
    neither clean nor failed. Without that, the walker's three clean days at an
    FCN3-extension lead would read as a clean 140-day chain and ``_all_clean_through``
    would carry the verdict straight past week 10 into leads nobody walked.
    """
    out: dict[int, dict] = {}
    sub = df[df.model == model]
    skip = set(untested_weeks(df, model))
    for wk, g in sub.groupby("weeks"):
        if int(wk) in skip:
            continue
        chains, failed = {}, {}
        for ev, gg in g.groupby("event"):
            hard = gg[(gg.metric == "bounds") & (gg.status == "FAIL")]
            nf = gg[(gg.metric == "nonfinite") & (gg.status == "FAIL")]
            first = None
            if len(hard) or len(nf):
                first = float(pd.concat([hard, nf]).checkpoint_day.min())
            sev = gg[gg.metric == "bounds_severity"]
            # FIRST out of bounds and DIVERGED are different dates and must not be
            # conflated: PNW@wk8 first leaves bounds at 42 d by 0.5% of the bound span -
            # a humidity excursion inside the noise of where that floor was drawn - and
            # only becomes an unrecoverable blow-up at 56 d, 64% past. Reporting the
            # first date with the worst magnitude would date the divergence two weeks early.
            big = sev[sev.value > MARGINAL_SEVERITY]
            chains[str(ev)] = dict(
                first=first,
                first_sev=(float(sev[sev.checkpoint_day == first].value.max())
                           if first is not None and len(sev[sev.checkpoint_day == first])
                           else float("nan")),
                diverged=(float(big.checkpoint_day.min()) if len(big) else None),
                worst=float(sev.value.max()) if len(sev) else 0.0)
            if first is not None:
                failed[str(ev)] = chains[str(ev)]
        out[int(wk)] = dict(chains=chains, n=len(chains), n_failed=len(failed))
    return out


def _all_clean_through(surv: dict) -> int | None:
    """The largest week at which every chain stayed inside bounds, weeks in order."""
    best = None
    for wk in sorted(surv):
        if surv[wk]["n_failed"]:
            break
        best = wk
    return best


def _headline(verdicts, df: pd.DataFrame) -> str:
    """*GenCast stable through week X, FCN3 through week Y* - and what that does not mean."""
    lines = ["", "=" * 78, "  HEADLINE"]
    surv = {m: _model_survival(df, m)
            for m in ("gencast", "fcn3_era5", "fcn3_adapter")}
    gc = _all_clean_through(surv["gencast"])
    f_free = _all_clean_through(surv["fcn3_era5"])
    f_fed = _all_clean_through(surv["fcn3_adapter"])
    fc = min([w for w in (f_free, f_fed) if w is not None], default=None)

    def wk(w):
        return f"week {w} ({C.lead_days_for(w)} d)" if w else "(no week clean)"

    lines.append(f"    GenCast stable through {wk(gc)}   - every chain")
    lines.append(f"    FCN3    stable through {wk(fc)}   - every rollout, both arms "
                 f"[ERA5-IC {f_free or '-'}, adapter-fed {f_fed or '-'}]")

    # What each verdict covers. The two models are no longer rolled over the same set of
    # leads, and a headline that did not say so would read as if they were.
    tested = {m: sorted(surv[m]) for m in surv}
    untested = untested_weeks(df, "gencast")
    lines.append(f"    tested to the peak: GenCast week(s) {tested['gencast'] or '-'}, "
                 f"FCN3 week(s) {tested['fcn3_era5'] or '-'}")
    if untested:
        lines += [
            f"    NOT a GenCast result at week(s) {untested}: the walker rolls "
            f"{SC.SCORE_LEAD_DAYS:g} d there, only to launch the",
            f"    adapter-fed arm, and FCN3 carries the rest. Those chains say nothing "
            f"about the walker at that lead -",
            f"    they are the SCORER's question, which week 10 left open by never "
            f"failing."]

    # Where each model actually left bounds, and by how much - a binary verdict cannot
    # distinguish a 0.5%-past humidity excursion from a 33%-past temperature blow-up.
    any_fail = False
    for model in ("gencast", "fcn3_era5", "fcn3_adapter"):
        rows = []
        for w in sorted(surv[model]):
            for ev, c in surv[model][w]["chains"].items():
                if c["first"] is not None:
                    fs = ("" if not np.isfinite(c["first_sev"])
                          else f" ({c['first_sev'] * 100:.1f}% past"
                               + (", marginal)" if c["first_sev"] <= MARGINAL_SEVERITY
                                  else ")"))
                    tail = (f"; DIVERGED by {c['diverged']:g} d, worst "
                            f"{c['worst'] * 100:.0f}% past"
                            if c["diverged"] is not None else
                            "; never past the marginal threshold")
                    rows.append(f"      {model:>13s}  {ev}@wk{w:<2d} left bounds at "
                                f"{c['first']:g} d{fs}{tail}")
        if rows:
            any_fail = True
            lines += rows
    if not any_fail:
        lines.append("      no chain of either model ever left the physical bounds")

    # Non-monotonicity: the finding that decides what the follow-up has to be.
    gcs = surv["gencast"]
    clean_beyond = [w for w in sorted(gcs)
                    if gc is not None and w > gc and gcs[w]["n_failed"] < gcs[w]["n"]]
    if clean_beyond:
        detail = ", ".join(
            f"week {w}: {gcs[w]['n'] - gcs[w]['n_failed']}/{gcs[w]['n']} chains clean"
            for w in clean_beyond)
        lines += [
            "",
            "    DIVERGENCE IS NOT MONOTONIC IN LEAD: " + detail + ", while every",
            "    chain at the SHORTER week 8 lead diverged. A chain that survives 70 d",
            "    sits beside one that fails at 30 d, so this",
            "    run cannot separate 'that lead is unstable' from 'that member diverged'.",
            "    With n = 1 member per lead it was never able to. The follow-up needs an",
            "    ensemble for the STABILITY question too, not only for the skill one -",
            "    which is a finding about the experiment, not a caveat on it."]

    deg = df[(df.status == "FAIL") & (~df.metric.isin(HARD_METRICS))]
    if len(deg):
        lines.append(f"")
        lines.append(f"    {len(deg)} band exceedance(s) short of a hard failure "
                     f"(degraded, not diverged):")
        for (model, metric), g in deg.groupby(["model", "metric"]):
            lines.append(f"      {model:>13s}  {metric:<24s} first at "
                         f"{g.checkpoint_day.min():g} d"
                         + ("  (band extrapolated past 21 d)"
                            if int(g.extrapolated.max()) else ""))

    lines += [
        "",
        "    STABLE IS NOT USEFUL, and it gets less useful the further out this goes.",
        "    Gate 2 put FCN3's rank-equivalence horizon at ~8.5 d",
        "    and Gate 3 measured score skill rho_s = 0.797 at 9 d. Stability out here is",
        "    PERMISSION to run AI+RES at long lead, not a reason to: past the",
        "    predictability horizon the score function ranks chaos. Establishing the",
        "    USEFUL horizon needs a rank correlation, which needs an ensemble, which one",
        "    member cannot produce.",
        "=" * 78]
    return "\n".join(lines)


