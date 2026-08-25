#!/usr/bin/env python
"""How rare is each AI+RES event? -- the climatological distribution of ``A_L``.

``aires/aindex.py`` defines the observable the experiment sorts walkers by:

    A_L = cos(lat)-weighted area mean, over the event's box, of the 7-day-mean
          T2m anomaly field ending at the event peak

and every figure in the run reports a *forecast* distribution of it: where the
observation sits relative to what GenCast (or AI+RES) sampled. That says how hard the
event was to FORECAST at 21 days. It says nothing about how rare the event was, because
the reference is a 24-to-64 member ensemble, not the climate.

This module supplies the missing reference: the same index, computed from 64 years of
ERA5, so ``A_L = +7.72 K`` can be quoted as a percentile and a return period rather than
as "2.0 sigma of a 24-member ensemble".

Why a daily-mean source
-----------------------
The experiment's window is 25 six-hourly frames, ``[peak - 6 d 00Z, peak 00Z]``
inclusive (``aires.aindex.check_window``; ``gencast_s2s.data.true_verif_anom``). The first
24 of those frames are exactly the six daily means of days ``peak-6 .. peak-1``, so

    A_L = (24 * A6 + a(peak, 00Z)) / 25,   A6 = mean of the 6 daily-mean anomalies

and the final frame carries weight 1/25. ``A6`` is therefore the same index to within a
few hundredths of a K (measured per event by ``--report``), and it is computable from
WeatherBench-2's **daily-mean** 0.25 degree ERA5 store -- one 4.2 MB frame per day instead
of four, i.e. the whole 1959-2023 record for ~97 GB of read instead of ~390 GB.

Why only box means are cached
-----------------------------
The area mean is linear, so ``mean_box(T - clim) == mean_box(T) - mean_box(clim)``: the
anomaly can be formed AFTER the spatial reduction. The cache is therefore a
(time x box) table of absolute box-mean T2m -- a few hundred kB for 64 years -- and every
anomaly, window length and reference period below is a reduction of that table. Rebuilding
a percentile does not re-download anything.

Reference period
----------------
Anomalies use the same 1990-2019 climatology the experiment uses
(``gencast_s2s.config.CLIM_FILE``), so the numbers are commensurable with the published
``A_L``. Percentiles are reported against BOTH that 30-year baseline period and the full
1959-2022 record; the events are all 2020+, so the baseline-period percentile is the
"relative to the climatology this anomaly is defined against" number and the full-record
one is the "relative to everything ERA5 has seen" number.

Usage
-----
    python -m aires.aclim --build              # ~6 min, one time, cached
    python -m aires.aclim --report             # the tables
    python -m aires.aclim --report --json out.json
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from gencast_s2s import config as C

from . import aconfig as A
from . import aindex as AI

# --------------------------------------------------------------------------- #
# Source: WeatherBench-2's daily-mean 0.25 degree ERA5 (the "s2s" derived store).
# 23386 daily frames, 1959-01-01 .. 2023-01-10, chunked (1, 721, 1440).
# --------------------------------------------------------------------------- #
DAILY_STORE = ("weatherbench2/datasets/era5_daily/"
               "1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr")
VAR = "2m_temperature"

CACHE = Path(os.environ.get("AIRES_CLIM_CACHE",
                            A.AIRES_ROOT / "clim" / "box_t2m_daily_1959_2023.nc"))
# The daily store stops at 2023-01-10, but four of the events registered here peak after
# it. Their days are topped up from ARCO-ERA5 (hourly, continuously updated) and averaged
# over 24 h -- the same daily mean the WB2 store holds, so the two concatenate cleanly.
TOPUP = Path(os.environ.get("AIRES_CLIM_TOPUP",
                            A.AIRES_ROOT / "clim" / "box_t2m_daily_topup.nc"))

# The window the observable averages over, in DAILY means. 6 days == the 24 six-hourly
# frames of the experiment's 25-frame window; see the module docstring.
WINDOW_DAYS = 6

# Half-width of the seasonal pool a percentile is taken within. An event's index is
# compared against the same calendar season, not against the whole year: a +5 K weekly
# anomaly in January and one in July are not the same event, and pooling them would
# report the seasonal cycle of variance as if it were rarity.
SEASON_HALF_WIDTH = 21

BASELINE = (1990, 2019)          # the period CLIM_FILE averages; the anomaly's own base
RECENT = (1993, 2022)            # the last 30 complete years -- the warmed base
FULL = (1959, 2022)              # every complete year in the store
PERIODS = {"baseline_1990_2019": BASELINE, "recent_1993_2022": RECENT,
           "full_1959_2022": FULL}

# Boxes to reduce. Everything aires/aindex.py knows about, keyed by box name so several
# events can share one (CONUS is the p90 cases' own event-defining index).
def _boxes() -> dict[str, AI.Box]:
    out = {"CONUS": AI.CONUS}
    for ev, box in AI.EVENT_BOXES.items():
        out[ev] = box
    return out


# The events this module reports on: everything the AI+RES stack has run an index for.
REPORT_EVENTS = (
    "PNW_HeatDome_2021",
    "SCentral_HeatDome_2023",
    "California_HeatWave_2022",
    "Southwest_HeatWave_2020",
    "WinterStorm_Uri_2021",
    "p90_20231107",
    "p90_20240802",
    "p90_20251224",
)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _conus_slice(lat: np.ndarray, lon: np.ndarray) -> tuple[slice, slice]:
    """Index slices into the store's global grid for the CONUS crop every cube uses."""
    lat_lo, lat_hi = AI.CONUS_LAT
    lon_lo, lon_hi = AI.CONUS_LON
    ii = np.where((lat >= lat_lo - 1e-6) & (lat <= lat_hi + 1e-6))[0]
    jj = np.where((lon >= lon_lo - 1e-6) & (lon <= lon_hi + 1e-6))[0]
    return slice(ii[0], ii[-1] + 1), slice(jj[0], jj[-1] + 1)


def build(threads: int = 32, verbose: bool = True) -> xr.Dataset:
    """Read every daily frame, reduce to box means, cache. ~97 GB of read, ~6 min."""
    import gcsfs
    import zarr

    fs = gcsfs.GCSFileSystem(token="anon")
    root = zarr.open(fs.get_mapper(DAILY_STORE), mode="r")
    arr = root[VAR]
    lat = np.asarray(root["latitude"][:], dtype="float64")
    lon = np.asarray(root["longitude"][:], dtype="float64")
    # Decode the CF time axis by hand (zarr gives raw ints; the store is
    # "days since 1959-01-01 00:00:00", proleptic_gregorian).
    units = str(root["time"].attrs.get("units", ""))
    step, _, origin = units.partition(" since ")
    unit = {"hours": "h", "days": "D"}.get(step.strip())
    if unit is None:
        raise RuntimeError(f"unhandled time units {units!r} in {DAILY_STORE}")
    time = pd.DatetimeIndex(
        pd.Timestamp(origin.strip())
        + pd.to_timedelta(np.asarray(root["time"][:]), unit=unit))
    ilat, ilon = _conus_slice(lat, lon)
    clat, clon = lat[ilat], lon[ilon]
    if clat[0] > clat[-1]:           # store descends 90 -> -90; aindex wants ascending
        flip = True
        clat = clat[::-1]
    else:
        flip = False

    boxes = _boxes()
    # Per-box normalised cos(lat) weight fields on the CONUS crop, precomputed once.
    w = {}
    for name, box in boxes.items():
        m_lat = (clat >= box.lat[0] - 1e-6) & (clat <= box.lat[1] + 1e-6)
        m_lon = (clon >= box.lon[0] - 1e-6) & (clon <= box.lon[1] + 1e-6)
        if not m_lat.any() or not m_lon.any():
            raise ValueError(f"box {name} selects nothing inside the CONUS crop")
        ww = np.outer(np.cos(np.deg2rad(clat)) * m_lat, m_lon.astype("float64"))
        w[name] = ww / ww.sum()

    names = list(boxes)
    out = np.full((len(time), len(names)), np.nan, dtype="float64")

    def one(i: int) -> tuple[int, np.ndarray]:
        f = np.asarray(arr[i, ilat, ilon], dtype="float64")
        if flip:
            f = f[::-1]
        return i, np.array([float((w[n] * f).sum()) for n in names])

    n_done = 0
    with ThreadPoolExecutor(threads) as ex:
        for i, vals in ex.map(one, range(len(time))):
            out[i] = vals
            n_done += 1
            if verbose and n_done % 1000 == 0:
                print(f"   {n_done}/{len(time)} frames", flush=True)

    ds = xr.Dataset(
        {"t2m_box_mean": (("time", "box"), out)},
        coords={"time": time, "box": names},
        attrs={
            "source": f"gs://{DAILY_STORE}",
            "variable": VAR,
            "note": ("cos(lat)-weighted box-mean daily-mean 2 m temperature (K). "
                     "Anomalies are formed after the spatial reduction; the area mean "
                     "is linear so mean_box(T - clim) == mean_box(T) - mean_box(clim)."),
            "boxes": json.dumps({n: {"lat": list(b.lat), "lon": list(b.lon)}
                                 for n, b in boxes.items()}),
        })
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    tmp.replace(CACHE)
    if verbose:
        print(f"wrote {CACHE} ({CACHE.stat().st_size/1e3:.0f} kB)")
    return ds


def _box_weights(clat: np.ndarray, clon: np.ndarray) -> dict[str, np.ndarray]:
    w = {}
    for name, box in _boxes().items():
        m_lat = (clat >= box.lat[0] - 1e-6) & (clat <= box.lat[1] + 1e-6)
        m_lon = (clon >= box.lon[0] - 1e-6) & (clon <= box.lon[1] + 1e-6)
        ww = np.outer(np.cos(np.deg2rad(clat)) * m_lat, m_lon.astype("float64"))
        w[name] = ww / ww.sum()
    return w


def topup(dates, threads: int = 32, verbose: bool = True) -> xr.Dataset:
    """Daily box means for days the WB2 daily store does not cover, read from ARCO.

    24 hourly frames per day, averaged -- i.e. the same daily mean the WB2 daily store
    holds, not a 4-sample synoptic mean, so a topped-up day is interchangeable with a
    stored one.
    """
    from gencast_s2s import data as D

    dates = pd.DatetimeIndex(sorted({pd.Timestamp(d).normalize() for d in dates}))
    lat_c, lon_c = D._arco_lat_lon_coords(conus=True, stride=1)
    w = _box_weights(lat_c, lon_c)
    names = list(_boxes())
    hours = [(d, h) for d in dates for h in range(24)]

    def one(args):
        d, h = args
        f = np.asarray(D.arco_read("2m_temperature", d + pd.Timedelta(hours=h),
                                   conus=True), dtype="float64")
        return (d, h), np.array([float((w[n] * f).sum()) for n in names])

    acc = {d: np.zeros(len(names)) for d in dates}
    n_done = 0
    with ThreadPoolExecutor(threads) as ex:
        for (d, _h), vals in ex.map(one, hours):
            acc[d] += vals / 24.0
            n_done += 1
            if verbose and n_done % 200 == 0:
                print(f"   {n_done}/{len(hours)} hourly frames", flush=True)

    ds = xr.Dataset(
        {"t2m_box_mean": (("time", "box"),
                          np.stack([acc[d] for d in dates]))},
        coords={"time": dates, "box": names},
        attrs={"source": D.ARCO_STORE, "variable": "2m_temperature",
               "note": "24-hourly-mean daily box means; top-up past the WB2 daily store"})
    if TOPUP.exists():
        with xr.open_dataset(TOPUP) as old:
            ds = xr.concat([old.load(), ds], dim="time").drop_duplicates("time")
        ds = ds.sortby("time")
    TOPUP.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOPUP.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    tmp.replace(TOPUP)
    if verbose:
        print(f"wrote {TOPUP} ({ds.sizes['time']} days)")
    return ds


def topup_dates_for(events=None, before: int = 10, after: int = 3) -> list[pd.Timestamp]:
    """Days around each registered peak that the WB2 daily store does not cover."""
    from fcn3 import fevents as F

    with xr.open_dataset(CACHE) as ds:
        end = pd.Timestamp(pd.DatetimeIndex(ds["time"].values).max())
    out = []
    for name in (events or REPORT_EVENTS):
        peak = pd.Timestamp(F.event(name).peak)
        days = pd.date_range(peak - pd.Timedelta(days=before),
                             peak + pd.Timedelta(days=after), freq="D")
        out += [d for d in days if d > end]
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# Anomalies
# --------------------------------------------------------------------------- #
def clim_box_means() -> pd.DataFrame:
    """The 1990-2019 climatology, reduced to the same box means. (dayofyear x box).

    Read from the SAME file the experiment's anomalies use (``C.CLIM_FILE``), averaged
    over its four synoptic hours to give a daily-mean climatology.
    """
    with xr.open_dataset(C.CLIM_FILE) as ds:
        clim = ds[VAR].mean("hour").load()          # (dayofyear, lat, lon)
    rows = {}
    for name, box in _boxes().items():
        rows[name] = AI.area_mean(clim, box).values
    return pd.DataFrame(rows, index=clim["dayofyear"].values).rename_axis("dayofyear")


def daily_anomaly() -> pd.DataFrame:
    """Daily-mean box-mean T2m ANOMALY, 1959-2023, one column per box.

    Uses ``dayofyear`` exactly as ``gencast_s2s.data.clim_for`` does -- no leap-year
    realignment -- so an anomaly here means what an anomaly means everywhere else in
    this repo.
    """
    if not CACHE.exists():
        raise SystemExit(f"no cache at {CACHE}; run `python -m aires.aclim --build` first")
    frames = []
    for path in (CACHE, TOPUP):
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            frames.append(pd.DataFrame(
                ds["t2m_box_mean"].values,
                index=pd.DatetimeIndex(ds["time"].values),
                columns=[str(b) for b in ds["box"].values]))
    raw = pd.concat(frames)
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    # Reindex onto a COMPLETE daily calendar. The top-up is separated from the stored
    # record by a gap, and a rolling mean over a gappy index would silently average
    # across it -- a 6-day window built from days months apart, with no error.
    raw = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="D"))
    clim = clim_box_means()
    doy = raw.index.dayofyear
    base = clim.reindex(doy)[raw.columns].to_numpy()
    return pd.DataFrame(raw.to_numpy() - base, index=raw.index, columns=raw.columns)


def window_index(anom: pd.Series, days: int = WINDOW_DAYS) -> pd.Series:
    """``A6``-style index: the mean of the ``days`` daily anomalies ENDING the day before
    ``t``. Indexed by the event peak ``t``, so ``idx[peak]`` lines up with ``A_L``."""
    r = anom.rolling(days).mean()
    r.index = r.index + pd.Timedelta(days=1)
    return r.dropna()


# --------------------------------------------------------------------------- #
# Percentiles
# --------------------------------------------------------------------------- #
def _season_mask(idx: pd.DatetimeIndex, peak: pd.Timestamp,
                 half: int = SEASON_HALF_WIDTH) -> np.ndarray:
    """Within ``half`` days of the peak's calendar position, in any year."""
    d = (idx.dayofyear - peak.dayofyear + 366) % 366
    return np.minimum(d, 366 - d) <= half


def _years(idx: pd.DatetimeIndex, span: tuple[int, int]) -> np.ndarray:
    return (idx.year >= span[0]) & (idx.year <= span[1])


def pool(series: pd.Series, peak, span: tuple[int, int],
         half: int = SEASON_HALF_WIDTH) -> pd.Series:
    peak = pd.Timestamp(peak)
    m = _season_mask(series.index, peak, half) & _years(series.index, span)
    return series[m].dropna()


def percentile_of(value: float, sample: np.ndarray, sign: float = 1.0) -> float:
    """Percent of the sample the value is at least as extreme as, in the event's tail."""
    x = np.asarray(sample, dtype="float64")
    if sign > 0:
        return 100.0 * float((x <= value).mean())
    return 100.0 * float((x >= value).mean())


def exceedance(value: float, sample: np.ndarray, sign: float = 1.0) -> float:
    x = np.asarray(sample, dtype="float64")
    return float((x >= value).mean()) if sign > 0 else float((x <= value).mean())


def seasonal_extremes(series: pd.Series, peak, span: tuple[int, int],
                      sign: float = 1.0, half: int = SEASON_HALF_WIDTH) -> pd.Series:
    """One value per year: the most extreme window index inside that year's season pool.

    Overlapping 7-day windows are heavily autocorrelated, so an empirical percentile over
    all of them understates how uncertain the tail is. Block extremes are independent by
    construction and are what a return period should be built on.
    """
    s = pool(series, peak, span, half)
    g = s.groupby(s.index.year)
    return g.max() if sign > 0 else g.min()


def gev_return_period(value: float, blocks: np.ndarray, sign: float = 1.0) -> dict:
    """Return period in years from a GEV fit to the seasonal block extremes.

    The fit is on ``sign * blocks`` so a cold event is handled by the same high-tail code.
    Reported alongside the empirical rank, never instead of it: a 30-to-64 point GEV
    extrapolated past the sample maximum is an estimate with a wide interval, and the
    figure is here to say "roughly once a century" rather than "once every 137 years".
    """
    from scipy import stats

    x = sign * np.asarray(blocks, dtype="float64")
    v = sign * float(value)
    c, loc, scale = stats.genextreme.fit(x)
    p = float(stats.genextreme.sf(v, c, loc=loc, scale=scale))
    return {"shape": float(c), "loc": float(loc), "scale": float(scale),
            "p_exceed_per_season": p,
            "return_period_years": (float("inf") if p <= 0 else 1.0 / p)}


LADDER = (50, 75, 90, 95, 99, 99.5, 99.9)


def quantile_ladder(sample: np.ndarray, sign: float = 1.0,
                    levels=LADDER) -> dict[float, float]:
    x = np.asarray(sample, dtype="float64")
    return {lv: float(np.quantile(x, lv / 100.0 if sign > 0 else 1.0 - lv / 100.0))
            for lv in levels}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def observed_A_L(event: str) -> dict:
    """The experiment's own published ``A_L``, recomputed from the ERA5 truth file."""
    from fcn3 import fevents as F

    ev = F.event(event)
    path = F.era5_truth_path(ev)
    if not path.exists():
        return {}
    with xr.open_dataset(path) as ds:
        da = ds[ev.metric].load()
    box = AI.box_for(event)
    return {"A_L": float(AI.area_mean(da, box)),
            "A_L_conus": float(AI.area_mean(da, AI.CONUS)),
            "grid_max": float(da.max()), "grid_min": float(da.min())}


def p90_peak_anomaly(event: str) -> float | None:
    """The DAILY CONUS-mean anomaly a p90 case was selected on (``p90/cases.csv``).

    The p90 cases are defined by a 1-day index, not the 7-day one the AI+RES observable
    uses, and their peaks are past the daily store's 2023-01-10 end -- so the selection
    value is read from the frozen case table rather than recomputed.
    """
    from fcn3 import fevents as F

    case = F.P90_SOURCE_CASE.get(event)
    if case is None:
        return None
    path = A.ROOT / "p90" / "cases.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path).set_index("case")
    return float(df.loc[case, "peak_anom_K"]) if case in df.index else None


def report(events=REPORT_EVENTS, half: int = SEASON_HALF_WIDTH) -> dict:
    from fcn3 import fevents as F

    anom = daily_anomaly()
    out = {"window_days": WINDOW_DAYS, "season_half_width": half,
           "periods": {k: list(v) for k, v in PERIODS.items()}, "events": {}}

    for name in events:
        ev = F.event(name)
        if ev.metric != "t2m_anom":
            continue
        peak = pd.Timestamp(ev.peak)
        box = AI.box_for(name)
        col = name if name in anom.columns else "CONUS"
        sign = AI.tail_sign(name)
        wk = window_index(anom[col])
        day = anom[col]

        rec = {"peak": ev.peak, "box": box.name, "box_label": box.label,
               "tail": "high" if sign > 0 else "low", "column": col,
               "observed": observed_A_L(name)}

        # The same index, measured from this daily series, for the event's own week --
        # the check that A6 and A_L are the same number.
        rec["A6_from_daily"] = (float(wk.loc[peak]) if peak in wk.index else None)
        # The 1-day value at the peak. Past the daily store's end (2023-01-10) fall back
        # to the frozen p90 selection table, which is the same quantity for those cases.
        rec["daily_at_peak"] = (float(day.loc[peak]) if peak in day.index
                                else p90_peak_anomaly(name))
        rec["daily_at_peak_source"] = ("era5_daily" if peak in day.index
                                       else ("p90/cases.csv" if rec["daily_at_peak"]
                                             is not None else None))

        # The frozen peak date fixes ONE window. The event's own worst window may sit a
        # few days either side of it -- the PNW dome's does -- and quoting only the
        # frozen window understates the event as a meteorological fact (though it is
        # exactly the number the experiment forecasts, so both belong here).
        near = wk[(wk.index >= peak - pd.Timedelta(days=7))
                  & (wk.index <= peak + pd.Timedelta(days=7))].dropna()
        if near.size:
            i = near.idxmax() if sign > 0 else near.idxmin()
            rec["event_best_window"] = {"value": float(near.loc[i]),
                                        "window_end": str(i.date()),
                                        "offset_days": int((i - peak).days)}

        obs = rec["observed"].get("A_L")
        obs_day = rec["daily_at_peak"]
        for tag, span in PERIODS.items():
            s = pool(wk, peak, span, half)
            d = pool(day, peak, span, half)
            blocks = seasonal_extremes(wk, peak, span, sign, half)
            # The event's own occurrence is in the record. Its overlapping windows all
            # exceed the threshold, so an exceedance count that includes them says
            # "6 windows this extreme" about ONE event. The ex-self pool drops the
            # event's own calendar year; both are reported.
            keep = s.index.year != peak.year
            s_ex = s[keep]
            blocks_ex = blocks[blocks.index != peak.year]
            # Overlapping windows are not independent draws: consecutive 6-day means
            # share 5 days. The effective count is roughly one per window length per
            # year, and it is what the top of the ladder is actually resolved by --
            # p99.9 of 2752 overlapping windows can be three windows of one event.
            n_eff = int(round(blocks.size * (2 * half + 1) / WINDOW_DAYS))
            e = {
                "n_windows": int(s.size), "n_years": int(blocks.size),
                "n_effective": n_eff,
                "week_ladder": quantile_ladder(s.values, sign),
                "day_ladder": quantile_ladder(d.values, sign),
                "week_ladder_ex_self": quantile_ladder(s[keep].values, sign),
                "day_ladder_ex_self": quantile_ladder(
                    d[d.index.year != peak.year].values, sign),
                "week_mean": float(s.mean()), "week_sd": float(s.std(ddof=1)),
                "week_record": float(s.max() if sign > 0 else s.min()),
                "week_record_date": str((s.idxmax() if sign > 0 else s.idxmin()).date()),
                "day_record": float(d.max() if sign > 0 else d.min()),
            }
            if obs is not None:
                beyond = (s.values >= obs) if sign > 0 else (s.values <= obs)
                beyond_ex = (s_ex.values >= obs) if sign > 0 else (s_ex.values <= obs)
                e["percentile"] = percentile_of(obs, s.values, sign)
                e["percentile_ex_self"] = percentile_of(obs, s_ex.values, sign)
                e["p_exceed_per_window"] = exceedance(obs, s.values, sign)
                e["sigma"] = (obs - e["week_mean"]) / e["week_sd"] * sign
                e["n_windows_beyond"] = int(beyond.sum())
                e["n_windows_beyond_ex_self"] = int(beyond_ex.sum())
                e["years_beyond"] = sorted({int(y) for y in s.index[beyond].year})
                e["n_years_ex_self"] = int(blocks_ex.size)
                try:
                    e["gev"] = gev_return_period(obs, blocks.values, sign)
                except Exception as exc:                      # scipy fit can fail
                    e["gev"] = {"error": repr(exc)}
            if obs_day is not None:
                keep_d = d.index.year != peak.year
                e["day_percentile"] = percentile_of(obs_day, d.values, sign)
                e["day_percentile_ex_self"] = percentile_of(obs_day, d[keep_d].values,
                                                            sign)
                e["day_sigma"] = ((obs_day - float(d.mean()))
                                  / float(d.std(ddof=1)) * sign)
            if "event_best_window" in rec:
                e["best_window_percentile"] = percentile_of(
                    rec["event_best_window"]["value"], s.values, sign)
            # Year-round pool, no season mask. This is the reference the p90 case set
            # used (`p90/cases_meta.json`: the 90th percentile of the daily CONUS-mean
            # anomaly over ALL days of 2022-2025), and it is what makes a +2.4 K August
            # day "marginal" there and a seasonal record here: the year-round variance of
            # a CONUS mean is set by winter.
            yr_w = wk[_years(wk.index, span)].dropna()
            yr_d = day[_years(day.index, span)].dropna()
            e["year_round"] = {
                "n_windows": int(yr_w.size),
                "week_ladder": quantile_ladder(yr_w.values, sign),
                "day_ladder": quantile_ladder(yr_d.values, sign),
            }
            if obs is not None:
                e["year_round"]["percentile"] = percentile_of(obs, yr_w.values, sign)
            if obs_day is not None:
                e["year_round"]["day_percentile"] = percentile_of(obs_day, yr_d.values,
                                                                  sign)
            rec[tag] = e
        out["events"][name] = rec
    return out


def _fmt(v, w=6, p=2):
    return "   n/a" if v is None else f"{v:+{w}.{p}f}"


def print_report(rep: dict) -> None:
    print(f"AI+RES events vs the ERA5 climatological distribution of A_L")
    print(f"  index      : cos(lat)-weighted BOX MEAN of the {rep['window_days']}-day-mean "
          f"T2m anomaly ending at the peak")
    print(f"  season pool: +-{rep['season_half_width']} d around the peak's calendar date, "
          f"every year in the reference period")
    print()
    hdr = (f"{'event':26s} {'box':13s} {'A_L':>7s} {'A6':>7s} "
           f"{'pct 90-19':>10s} {'pct 59-22':>10s} {'sigma':>6s} "
           f"{'RP (yr)':>8s} {'yrs>=obs':>9s} {'record':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in rep["events"].items():
        b = r["baseline_1990_2019"]
        f = r["full_1959_2022"]
        rp = f.get("gev", {}).get("return_period_years")
        rp_s = "n/a" if rp is None else (">1e4" if rp > 1e4 else f"{rp:.0f}")
        yrs = f.get("years_beyond", [])
        yrs_s = f"{len(yrs)}/{f['n_years']}"
        print(f"{name:26s} {r['box']:13s} {_fmt(r['observed'].get('A_L'))} "
              f"{_fmt(r['A6_from_daily'])} {b.get('percentile', float('nan')):9.3f}% "
              f"{f.get('percentile', float('nan')):9.3f}% "
              f"{f.get('sigma', float('nan')):+6.2f} {rp_s:>8s} {yrs_s:>9s} "
              f"{f['week_record']:+8.2f}")
    print()
    for name, r in rep["events"].items():
        f = r["full_1959_2022"]
        b = r["baseline_1990_2019"]
        rc = r["recent_1993_2022"]
        print(f"{name}  --  {r['box']} ({r['box_label']}), peak {r['peak']}, "
              f"{r['tail']} tail")
        print(f"    observed  {rep['window_days']}-d box mean A_L = "
              f"{_fmt(r['observed'].get('A_L'))} K   "
              f"| 1-day box mean at the peak "
              f"{_fmt(r['daily_at_peak'])} K"
              f" (p{f.get('day_percentile', float('nan')):.2f})"
              f" | worst grid point "
              f"{_fmt(r['observed'].get('grid_max'))} K")
        print(f"    percentile of A_L: 1990-2019 {b.get('percentile', float('nan')):.3f}"
              f"   1993-2022 {rc.get('percentile', float('nan')):.3f}"
              f"   1959-2022 {f.get('percentile', float('nan')):.3f}"
              f"   (windows at least this extreme: {f.get('n_windows_beyond')}"
              f" of {f['n_windows']}, {f.get('n_windows_beyond_ex_self')}"
              f" once its own year is dropped)")
        lad = f["week_ladder"]
        print("    " + f"{rep['window_days']}-day box-mean anomaly, 1959-2022 season pool "
              f"(n={f['n_windows']}):")
        print("      " + "  ".join(f"p{lv:g}={lad[lv]:+.2f}" for lv in LADDER))
        lad = f["day_ladder"]
        print("    1-day box-mean anomaly, same pool:")
        print("      " + "  ".join(f"p{lv:g}={lad[lv]:+.2f}" for lv in LADDER))
        print(f"    record {rep['window_days']}-d window in 1959-2022: "
              f"{f['week_record']:+.2f} K ({f['week_record_date']});  "
              f"years with a window at least as extreme as the event: "
              f"{f.get('years_beyond')}")
        print()


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    i = argv.index("--threads") + 1 if "--threads" in argv else None
    threads = int(argv[i]) if i else 32
    if "--build" in argv:
        build(threads=threads)
        return
    if "--topup" in argv:
        dates = topup_dates_for()
        if not dates:
            print("nothing to top up: every registered peak is inside the daily store")
            return
        print(f"topping up {len(dates)} days from ARCO "
              f"({dates[0].date()} .. {dates[-1].date()})")
        topup(dates, threads=threads)
        return
    if "--report" in argv or not argv:
        half = SEASON_HALF_WIDTH
        if "--half" in argv:
            half = int(argv[argv.index("--half") + 1])
        rep = report(half=half)
        print_report(rep)
        if "--json" in argv:
            p = Path(argv[argv.index("--json") + 1])
            p.write_text(json.dumps(rep, indent=2))
            print(f"wrote {p}")
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
