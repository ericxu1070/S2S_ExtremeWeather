#!/usr/bin/env python
"""Build the committed p90 case list: 45 p90 events + 45 season-matched controls.

Run ONCE on Derecho (where the daily anomaly cube lives) and commit the outputs
(p90/cases.csv + p90/cases_meta.json) so every other machine (a3mega) gets the exact
same experiment definition from git, without needing the cube.

Inputs (Derecho only):
  runs/p90_t2m_events_2022_2025/events_summary.csv            (from find_p90_t2m_events.py)
  runs/p90_t2m_events_2022_2025/daily_conus_t2m_anom_2022_2025.nc

Why controls exist: all 45 events are observed "yes" by construction, so Brier/ROC/
reliability would be degenerate (no false-positive axis). Each event gets one control
day drawn from the same calendar month (any year 2022-2025) at least 7 days away from
every event window, giving 45 observed-"no" cases with matched seasonality.

Case fields:
  case        event_NN_<peakYYYYMMDD> | control_NN_<peakYYYYMMDD>
  kind        event | control
  event_id    original event number (controls: the event they are matched to)
  peak        verification-week end (00Z); init = peak - 21 days
  obs_event   1 if any day in [peak-6d, peak] has CONUS-mean daily T2m anom >= thr
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import gencast_s2s.config as C

SEED = 20260722
PCTL = 90.0
LEAD_DAYS = 21
BUFFER_DAYS = 7          # controls must be > this far from any event day

SRC_DIR = C.RUNS / "p90_t2m_events_2022_2025"
CUBE_F = SRC_DIR / "daily_conus_t2m_anom_2022_2025.nc"
SUMMARY_F = SRC_DIR / "events_summary.csv"

PKG = Path(__file__).resolve().parent
CASES_CSV = PKG / "cases.csv"
META_JSON = PKG / "cases_meta.json"


def obs_event(series: pd.Series, peak: pd.Timestamp, thr: float) -> int:
    win = series.loc[peak - pd.Timedelta(days=6): peak]
    assert len(win) == 7, f"verif week for {peak.date()} not fully inside the cube"
    return int((win >= thr).any())


def main() -> None:
    ev = pd.read_csv(SUMMARY_F, parse_dates=["start", "end", "peak_day"])
    cube = xr.open_dataset(CUBE_F)
    series = cube["conus_mean_t2m_anom"].to_pandas()
    thr = float(np.percentile(series.values, PCTL))
    days = series.index

    excluded = set()
    for _, r in ev.iterrows():
        for d in pd.date_range(r.start - pd.Timedelta(days=BUFFER_DAYS),
                               r.end + pd.Timedelta(days=BUFFER_DAYS)):
            excluded.add(d)

    rng = np.random.default_rng(SEED)
    rows, taken = [], set()
    for _, r in ev.iterrows():
        peak = r.peak_day
        rows.append(dict(
            case=f"event_{r.event:02d}_{peak:%Y%m%d}", kind="event", event_id=int(r.event),
            peak=f"{peak:%Y-%m-%d}T00:00:00", init=f"{peak - pd.Timedelta(days=LEAD_DAYS):%Y-%m-%dT00:00:00}",
            month=int(peak.month), start=str(r.start.date()), end=str(r.end.date()),
            n_days=int(r.n_days), peak_anom_K=float(r.peak_anom_K), mean_anom_K=float(r.mean_anom_K),
            obs_event=obs_event(series, peak, thr),
        ))

    def candidates(months: set[int]) -> list[pd.Timestamp]:
        return [d for d in days
                if d.month in months and d >= days[0] + pd.Timedelta(days=6)
                and d not in excluded and d not in taken]

    for _, r in ev.iterrows():
        peak = r.peak_day
        m = peak.month
        # same calendar month first; widen to adjacent months if saturated (Feb is)
        pool = candidates({m}) or candidates({(m - 2) % 12 + 1, m % 12 + 1})
        assert pool, f"no control candidates for event {r.event} (month {m} +-1)"
        ctrl = pd.Timestamp(rng.choice(np.array(sorted(pool), dtype="datetime64[ns]")))
        taken.add(ctrl)
        rows.append(dict(
            case=f"control_{r.event:02d}_{ctrl:%Y%m%d}", kind="control", event_id=int(r.event),
            peak=f"{ctrl:%Y-%m-%d}T00:00:00", init=f"{ctrl - pd.Timedelta(days=LEAD_DAYS):%Y-%m-%dT00:00:00}",
            month=int(ctrl.month), start="", end="", n_days=0,
            peak_anom_K=float(series.loc[ctrl]), mean_anom_K=float(series.loc[ctrl]),
            obs_event=obs_event(series, ctrl, thr),
        ))

    df = pd.DataFrame(rows)
    n_ev, n_ct = (df.kind == "event").sum(), (df.kind == "control").sum()
    assert (df.loc[df.kind == "event", "obs_event"] == 1).all(), "an event scored obs_event=0"
    assert (df.loc[df.kind == "control", "obs_event"] == 0).all(), "a control scored obs_event=1"
    df.to_csv(CASES_CSV, index=False)

    META_JSON.write_text(json.dumps(dict(
        seed=SEED, pctl=PCTL, thr_K=round(thr, 4), lead_days=LEAD_DAYS,
        buffer_days=BUFFER_DAYS, n_events=int(n_ev), n_controls=int(n_ct),
        clim="ERA5 1990-2019 WB2 hourly climatology (gencast_s2s C.CLIM_FILE)",
        series="cos(lat)-weighted CONUS-mean daily-mean T2m anomaly, 2022-2025",
        source_cube=str(CUBE_F), source_summary=str(SUMMARY_F),
        generated_by="p90/make_cases.py",
    ), indent=2) + "\n")
    print(f"wrote {CASES_CSV} ({n_ev} events + {n_ct} controls), thr = {thr:.4f} K")
    print(df.groupby(["kind", "month"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
