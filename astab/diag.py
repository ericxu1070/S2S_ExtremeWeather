#!/usr/bin/env python
"""The diagnostic panel: is this still an atmosphere, and how would we know?

Anchored on the ONE measured failure on record (``aires/HANDOFF.md``, the wrong-checkpoint
incident, jobs 1155/1156): CONUS-mean T2m lost its diurnal cycle inside the first 3-day
segment (00Z equal to 12Z to 0.03 K), drifted 26 K cold over 21 days, and z500 finished
1780 m2 s-2 low. The healthy 3-day reference is 298.56 +- 0.162 K CONUS-mean T2m.

Cadence: CONUS items every 12 h off ``diag.nc`` (whose ``time`` is absolute, so segments
concatenate cleanly across the ragged tail); global items at each segment boundary off
``state.nc``, computed at walk time before the state is pruned.

Domain is the thing to get right, not sophistication
----------------------------------------------------
Job 1189's verdict: the carefully calibrated CONUS panel caught 1 of GenCast's 3
divergences and the crude global bounds check caught 3 of 3. Two of the three were confined
to ``temperature`` at 50 hPa over Antarctica (1211 K at 70 d lead) while the CONUS
troposphere stayed textbook-normal. Item 2 below is the only item evaluated GLOBALLY and on
every variable and level, and that - not its sophistication - is why it is the one that
works. Any longer-lead panel needs at least one such item.

The bands are MEASURED, not guessed (see the second half of this module and
``stages.stage_calibrate``): the Gate 3 tree already holds 16 free-running, physically
validated 21-day walkers per event, so items 3, 5, 6 and 7 are distributions out to 21 d
rather than thresholds someone chose. Past 21 d every band is an EXTRAPOLATION and every
consumer says so.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aires import aindex as AI
from astab import sconfig as SC

# Diagnostics
#
# Anchored on the ONE measured failure on record (aires/HANDOFF.md, the wrong-checkpoint
# incident, jobs 1155/1156): CONUS-mean T2m lost its diurnal cycle inside the first 3-day
# segment (00Z equal to 12Z to 0.03 K), drifted 26 K cold over 21 days, and z500 finished
# 1780 m2 s-2 low. The healthy 3-day reference is 298.56 +- 0.162 K CONUS-mean T2m.
#
# Cadence: CONUS items every 12 h off diag.nc (whose `time` is absolute, so segments
# concatenate cleanly across the ragged tail); global items at each segment boundary off
# state.nc, computed at walk time before the state is pruned.
# --------------------------------------------------------------------------- #
# Item 2. Absolute, and deliberately generous: this is the line between an atmosphere and
# a diverged array, not a skill test. The 50 hPa geopotential entry is the same class as
# CLAUDE.md's float16 trap - ~2e5 m2 s-2 is where a narrow dtype silently stores inf.
# Every entry below was checked against 18 known-good Gate 3 GLOBAL states (both events,
# walkers 0/5/15, leads 3/12/21 d) and against ERA5 itself at six chain dates. Measured
# walker ranges, for the record:
#
#     2m_temperature           [ 194.4,     324.0 ]      ERA5 global: [201.0, 320.5]
#     temperature              [ 181.1,     324.9 ]
#     mean_sea_level_pressure  [ 93080,    106400 ] Pa   ERA5 global: [93500, 105200]
#     specific_humidity        [-1.28e-3, 2.84e-2 ]
#     u/v (10 m and upper)     [ -84.9,     109.5 ] m/s
#     geopotential             [ -5501,   2.08e+5 ] m2/s2
#
# The geopotential maximum is 2.08e5, which is where CLAUDE.md's float16 trap lives:
# float16's ceiling is 65504, so a narrowed archive would store `inf` for the top of every
# column, silently and on write.
BOUNDS: dict[str, tuple[float, float]] = {
    # NOT (200, 330). That is a CONUS range applied to a GLOBAL field: ERA5's own global
    # minimum is 201.0 K on 2021-06-28 (Antarctic plateau, austral winter) and healthy
    # walkers reach 194.4 K, so a 200 K floor fires on the real atmosphere. The smoke run
    # tripped it at 6 d lead with a global minimum of 197.8 K. 180 K clears the worst
    # measured walker by 14 K and ERA5 by 21 K and is still a diverged-array detector.
    #
    # This item is deliberately NOT what catches the job-1155 failure: a 26 K cold CONUS
    # mean leaves the field minima far inside these bounds. Items 3 and 5 catch that, by
    # 8x and 200x respectively. Item 2 is the coarse net for genuine blow-up.
    "2m_temperature": (180.0, 340.0),
    "temperature": (150.0, 350.0),
    "mean_sea_level_pressure": (87000.0, 109000.0),
    # NOT `q >= 0`, which the plan specified and which fires on every healthy walker.
    # Measured over 18 known-good Gate 3 states (both events, walkers 0/5/15, leads
    # 3/12/21 d): the minimum is -1.28e-3 kg/kg and up to 0.19% of points are negative -
    # the ordinary small-negative-humidity artifact of an ML weather model, present from
    # the first segment and not growing with lead. The floor is ~4x the worst measured
    # value, so it is still a divergence detector and not a nuisance alarm. This is
    # CLAUDE.md's "measure the noise floor before setting a threshold", applied to a
    # threshold that looked unarguable.
    "specific_humidity": (-5.0e-3, 0.1),
    "10m_u_component_of_wind": (-200.0, 200.0),
    "10m_v_component_of_wind": (-200.0, 200.0),
    "u_component_of_wind": (-200.0, 200.0),
    "v_component_of_wind": (-200.0, 200.0),
    "geopotential": (-1.0e4, 5.0e5),
}
HARD_METRICS = ("nonfinite", "bounds")

# Variables whose non-finite mask is a property of the FIELD, not of the rollout.
# `sea_surface_temperature` is NaN over land by construction: measured at exactly 0.3389
# of the global grid on every one of 18 known-good Gate 3 states - bit-identical across
# both events, three walkers and three leads - and a further 0.004225 of the OCEAN points
# (coastal and ice-edge cells the source ERA5 mask leaves empty). Hard-failing on "any
# non-finite value" would therefore have marked all eight chains diverged at lead 3 d and
# reported "stable through week (none)". These are gated on GROWTH against the chain's own
# first segment instead, which still catches a real NaN blow-up in SST.
NONFINITE_STATIC = ("sea_surface_temperature",)
NONFINITE_GROWTH_TOL = 1e-4

# Item 7. Wavenumber 20 at 45 N is ~2000 km; power piling up above it is the classic
# AI-weather instability signature and the plan's bet on which item fires first.
HI_WAVENUMBER = 20
SPECTRUM_LAT = (30.0, 60.0)
SPECTRUM_BAND_DEG = 5.0
SPECTRUM_MIN_LON_SPAN = 300.0     # degrees; a CONUS crop spans 59 and must be refused

# Item 4 has no in-repo baseline of any kind, so it ships as a reported number with no
# gate. Inventing a threshold for it would be worse than having none.
UNGATED = ("global_t2m", "global_z500", "global_t50", "t2m_conus_mean",
           "nonfinite_sst", "bounds_severity")

# The broken-run floor drawn on panel 3: -1780 m2 s-2 = -181 m of geopotential height.
BROKEN_Z500_M = -1780.0 / 9.80665


def nonfinite_fractions(ds: xr.Dataset) -> dict[str, float]:
    """Item 1. Fraction of non-finite values per variable. Hard fail on anything > 0.

    **Entirely empty pressure levels are excluded**, because they are a coordinate
    artifact rather than model output. Putting two single-level channels in one Dataset
    makes xarray union their ``level`` axis and fill the cross terms with NaN: asking FCN3
    for ``z500`` and ``t50`` yields a cube on ``level = [50, 500]`` where
    ``geopotential@50`` and ``temperature@500`` are 100% NaN. Counting those reported all
    16 FCN3 rollouts as non-finite from lead 3 d and turned "FCN3 stable through week 10"
    into "no week clean".

    This cannot be fixed by reshaping the Dataset first - a Dataset has ONE ``level``
    coordinate, so it cannot represent "geopotential at 500 only" beside "temperature at
    50 only" without the padding. It has to be handled where the fraction is computed.

    Only slices that are *entirely* empty are skipped. A partially non-finite field is
    exactly the failure item 1 exists to catch and is counted in full.
    """
    out = {}
    for v in ds.data_vars:
        da = ds[v]
        a = np.asarray(da.values)
        if not np.issubdtype(a.dtype, np.floating):
            continue
        if "level" in da.dims and da.sizes["level"] > 1:
            keep, total, bad = [], 0, 0
            for lv in da["level"].values:
                sl = np.asarray(da.sel(level=lv).values)
                if not np.isfinite(sl).any():
                    continue                      # empty level: alignment padding
                keep.append(lv)
                total += sl.size
                bad += int((~np.isfinite(sl)).sum())
            out[str(v)] = (bad / total) if total else 0.0
            continue
        out[str(v)] = float((~np.isfinite(a)).sum()) / max(a.size, 1)
    return out


def field_ranges(ds: xr.Dataset) -> dict[str, dict]:
    """``{var: {min, max}}`` over the finite values of every bounded variable.

    The record stores THIS, and the reduce stage applies ``BOUNDS`` to it - the thresholds
    are never baked into an artifact at walk time. The smoke run made the reason concrete:
    ``BOUNDS["specific_humidity"]`` was corrected from ``0.0`` to ``-5e-3`` AFTER the
    segments were written, and records carrying a pre-computed verdict still reported the
    healthy chain as diverged. A measurement is durable; a verdict on it is not.
    """
    out: dict[str, dict] = {}
    for v in BOUNDS:
        if v not in ds:
            continue
        a = np.asarray(ds[v].values, dtype="float64")
        finite = np.isfinite(a)
        out[str(v)] = dict(min=float(a[finite].min()) if finite.any() else float("nan"),
                           max=float(a[finite].max()) if finite.any() else float("nan"))
    return out


# How far past a bound counts as DIVERGED rather than a marginal excursion, as a fraction
# of the bound's own span. Job 1189 is why this exists: three chains left bounds, at 0.5%,
# 4.1% and 33.1% past. The first is `specific_humidity` reaching -5.48e-3 against a floor
# set at -5e-3 - itself 4x the worst value measured on a known-good walker - and calling
# that "the walker stopped producing an atmosphere" would be an artifact of where the floor
# was drawn. The last is upper-air temperature at 416 K, 66 K above anything ever measured.
# A binary in/out verdict cannot tell those apart, so the severity is carried alongside it
# and the raw excursion goes in the CSV, where the choice of 2% stays visible and revisable.
MARGINAL_SEVERITY = 0.02


def bound_severity(ranges: dict, *, which: bool = False):
    """Worst excursion past a bound, as a fraction of that bound's span. 0.0 if inside.

    ``which=True`` also returns the variable responsible. Job 1189 is why that matters:
    two of its three divergences were confined to `temperature` at 50 hPa over Antarctica
    (1211 K at lead 70 d) while the CONUS troposphere stayed textbook-normal, so "which
    variable" is the difference between an attributable failure and a mystery.
    """
    worst, worst_var = 0.0, ""
    for v, r in (ranges or {}).items():
        if v not in BOUNDS:
            continue
        lo, hi = BOUNDS[v]
        span = hi - lo
        mn, mx = float(r.get("min", np.nan)), float(r.get("max", np.nan))
        if not np.isfinite(mn) or not np.isfinite(mx):
            return (float("inf"), str(v)) if which else float("inf")
        e = max((lo - mn) / span, (mx - hi) / span)
        if e > worst:
            worst, worst_var = e, str(v)
    worst = float(max(worst, 0.0))
    return (worst, worst_var) if which else worst


def violations_from_ranges(ranges: dict) -> dict[str, dict]:
    """Apply ``BOUNDS`` to a stored ``field_ranges`` mapping. The reduce-time gate."""
    bad = {}
    for v, r in (ranges or {}).items():
        if v not in BOUNDS:
            continue
        lo, hi = BOUNDS[v]
        mn, mx = float(r.get("min", np.nan)), float(r.get("max", np.nan))
        if not np.isfinite(mn) or not np.isfinite(mx) or mn < lo or mx > hi:
            bad[v] = dict(lo=lo, hi=hi, **r)
    return bad


def hard_nonfinite(fractions: dict, baseline: dict | None = None) -> float:
    """The number item 1 is gated on: the worst UNEXPLAINED non-finite fraction.

    Any non-finite value in a variable outside ``NONFINITE_STATIC`` is a failure outright.
    For the static-mask variables it is the GROWTH over the chain's own first segment that
    counts, because their baseline mask is the land-sea mask and is not a rollout product.
    """
    worst = 0.0
    for v, f in fractions.items():
        if v in NONFINITE_STATIC:
            base = (baseline or {}).get(v, f)
            worst = max(worst, max(0.0, float(f) - float(base) - NONFINITE_GROWTH_TOL))
        else:
            worst = max(worst, float(f))
    return worst


def bound_violations(ds: xr.Dataset) -> dict[str, dict]:
    """Item 2. ``{var: {lo, hi, min, max, frac}}`` for every variable out of bounds.

    Reports the fraction as well as the extremes: one bad grid point and a globally
    diverged field are both violations, and the number that distinguishes them belongs in
    the record.
    """
    out: dict[str, dict] = {}
    for v, (lo, hi) in BOUNDS.items():
        if v not in ds:
            continue
        a = np.asarray(ds[v].values, dtype="float64")
        finite = np.isfinite(a)
        if not finite.any():
            out[str(v)] = dict(lo=lo, hi=hi, min=float("nan"), max=float("nan"), frac=1.0)
            continue
        bad = finite & ((a < lo) | (a > hi))
        if bad.any():
            out[str(v)] = dict(lo=lo, hi=hi, min=float(a[finite].min()),
                               max=float(a[finite].max()),
                               frac=float(bad.sum()) / a.size)
    return out


def conus_mean_t2m(diag: xr.Dataset) -> xr.DataArray:
    """cos(lat)-weighted CONUS-mean T2m, frame by frame. The 298.56 K reference quantity."""
    return AI.area_mean(diag["2m_temperature"])


def conus_series(diag: xr.Dataset) -> pd.DataFrame:
    """Items 3 and 6, per 12-h frame: CONUS-mean T2m, its spatial sd, and its anomaly.

    The anomaly column is ``NaN`` when no climatology is reachable, so the drift
    diagnostic works on a machine that has no ``clim_1990_2019_t2m_conus.nc`` - the
    absolute drift from the chain's own first frame carries the -26 K signature on its
    own, and the anomaly is the version that is comparable ACROSS chains running through
    different seasons.
    """
    t2m = AI._squeeze(diag["2m_temperature"])
    t = pd.DatetimeIndex(t2m["time"].values)
    mean = np.asarray(AI.area_mean(t2m).values, dtype="float64")
    # Spatial sd over the CONUS window, unweighted on purpose: this is a structure
    # statistic, and cos-lat weighting it would blend "the field went flat" with "the
    # field went flat in the north".
    sd = np.asarray(t2m.std(("lat", "lon")).values, dtype="float64")
    anom = np.full(mean.shape, np.nan)
    try:
        anom = np.asarray(AI.area_mean(t2m - AI.clim_on(diag, t)).values, dtype="float64")
    except Exception as e:                     # no climatology on this box, or off-grid
        print(f"  [astab] climatology unavailable ({type(e).__name__}: {e}); "
              f"reporting absolute drift only")
    return pd.DataFrame(dict(t2m_mean=mean, t2m_sd=sd, t2m_anom=anom), index=t)


def drift_from_start(series) -> float:
    """Last minus first, PHASE MATCHED. The -26 K signature, with no climatology needed.

    Phase matching is not a refinement, it is the difference between a diagnostic and
    noise: frames alternate 00Z and 12Z, the CONUS diurnal range is ~8 K, and a raw
    last-minus-first therefore carries a +-4 K offset decided by nothing but whether the
    series has an odd or an even number of frames. The comparison is made against the
    first frame sharing the LAST frame's hour.
    """
    s = pd.Series(series).dropna()
    if s.size < 2:
        return float("nan")
    v = np.asarray(s.values, dtype="float64")
    return float(v[-1] - _phase_matched_first(s.index, v, hour=None))


def _phase_matched_first(times, values, hour: int | None) -> float:
    """The first value whose timestamp hour is ``hour`` (default: the LAST frame's hour).

    Falls back to the plain first value when the index is not datetime-like or no frame
    matches - a fallback that is only ever wrong by the diurnal range, and is loudly
    better than raising in a diagnostic.
    """
    v = np.asarray(values, dtype="float64")
    try:
        t = pd.DatetimeIndex(times)
    except (TypeError, ValueError):
        return float(v[0])
    want = int(t[-1].hour) if hour is None else int(hour)
    m = np.asarray(t.hour == want)
    return float(v[m][0]) if m.any() else float(v[0])


def diurnal_amplitudes(series: xr.DataArray | pd.Series) -> pd.DataFrame:
    """Item 5, first half: ``T2m(12Z) - T2m(00Z)`` over each adjacent pair of frames.

    GenCast's step is 12 h and every chain here inits at 00Z, so the frames alternate
    12Z/00Z and each adjacent pair IS a half-diurnal difference. Over CONUS (66-125 W),
    12Z is 04-07 local - near the minimum - and 00Z is 16-19 local, near the maximum, so
    a healthy amplitude is NEGATIVE by this convention and the sign is meaningful.

    Indexed by the later frame's time; ``n - 1`` rows for ``n`` frames.
    """
    if isinstance(series, xr.DataArray):
        t = pd.DatetimeIndex(series["time"].values)
        v = np.asarray(series.values, dtype="float64").ravel()
    else:
        s = pd.Series(series)
        t, v = pd.DatetimeIndex(s.index), np.asarray(s.values, dtype="float64")
    if t.size < 2:
        return pd.DataFrame(dict(amp=[], hour_first=[], hour_second=[]),
                            index=pd.DatetimeIndex([]))
    rows, idx = [], []
    for i in range(t.size - 1):
        h0, h1 = int(t[i].hour), int(t[i + 1].hour)
        if {h0, h1} != {0, 12}:
            continue                       # not a 00Z/12Z pair; no diurnal meaning
        # Orient the difference as 12Z minus 00Z whichever way round the pair sits.
        amp = (v[i] - v[i + 1]) if h0 == 12 else (v[i + 1] - v[i])
        rows.append(dict(amp=amp, hour_first=h0, hour_second=h1))
        idx.append(t[i + 1])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def diurnal_ratio_series(diag: xr.Dataset) -> pd.DataFrame:
    """Item 5, in full: the model's diurnal amplitude over the climatology's own.

    The best-grounded item on the panel, because it is a ratio against a real seasonal
    reference: the climatology's ``hour=12`` minus ``hour=0`` slice at the same
    ``dayofyear``, area-meaned identically. That self-calibration is what makes a single
    band valid for chains running April-June and December-February alike; a raw amplitude
    would not be comparable across them.

    A healthy chain sits near 1. The job-1155 collapse (0.03 K against a ~8 K reference)
    sits at ~0.004.
    """
    t2m = AI._squeeze(diag["2m_temperature"])
    model = diurnal_amplitudes(AI.area_mean(t2m))
    if model.empty:
        return model.assign(amp_clim=[], ratio=[])
    t = pd.DatetimeIndex(t2m["time"].values)
    try:
        clim = AI.area_mean(AI.clim_on(diag, t))
        cl = diurnal_amplitudes(clim)
    except Exception as e:
        print(f"  [astab] climatology unavailable ({type(e).__name__}: {e}); "
              f"diurnal ratio not computed")
        return model.assign(amp_clim=np.nan, ratio=np.nan)
    out = model.join(cl["amp"].rename("amp_clim"), how="left")
    out["ratio"] = out["amp"] / out["amp_clim"].where(np.abs(out["amp_clim"]) > 1e-6)
    return out


def zonal_high_wavenumber_fraction(z: xr.DataArray, *, cutoff: int = HI_WAVENUMBER,
                                   lat_band=SPECTRUM_LAT,
                                   band_deg: float = SPECTRUM_BAND_DEG) -> float:
    """Item 7. Fraction of zonal z500 power above wavenumber ``cutoff``, 30-60 N.

    Spectra are formed per latitude row, cos-lat averaged within each ``band_deg`` band,
    and the per-band fractions are then averaged - so a single very noisy row cannot carry
    the number, and neither can the density of rows in a band.

    **Global input only.** The CONUS crop spans 59 degrees of longitude
    (``config.LON = slice(235, 294)``); an FFT over that window is a perfectly
    well-formed number that means nothing as a zonal wavenumber, which is exactly the
    quiet wrongness this repo keeps getting bitten by. Refused loudly instead.
    """
    lon = np.asarray(z["lon"].values, dtype="float64")
    span = float(lon.max() - lon.min())
    if lon.size < 4 * cutoff or span < SPECTRUM_MIN_LON_SPAN:
        raise ValueError(
            f"the zonal spectrum needs a global longitude axis; got {lon.size} points "
            f"spanning {span:.0f} deg. A CONUS-cropped diag.nc cannot supply it - read "
            f"state.nc, which is global by contract.")
    z = AI._squeeze(z)
    for d in ("time", "batch", "level", "member"):
        if d in z.dims:
            z = z.isel({d: -1}, drop=True)
    # GenCast states ascend in latitude, FCN3's Zarr descends, and a slice on a descending
    # axis selects NOTHING while raising nothing. Sort first, always.
    if float(z["lat"][0]) > float(z["lat"][-1]):
        z = z.sortby("lat")
    band = z.sel(lat=slice(min(lat_band), max(lat_band)))
    lat = np.asarray(band["lat"].values, dtype="float64")
    if lat.size == 0:
        raise ValueError(f"no latitudes inside {lat_band}")
    a = np.asarray(band.transpose("lat", "lon").values, dtype="float64")
    a = a - a.mean(axis=1, keepdims=True)           # drop the zonal mean, k = 0
    power = np.abs(np.fft.rfft(a, axis=1)) ** 2     # (lat, k), k = 0 .. nlon/2
    w = np.cos(np.deg2rad(lat))
    edges = np.arange(min(lat_band), max(lat_band) + 1e-9, band_deg)
    fracs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lat >= lo) & (lat < hi + 1e-9)
        if not m.any():
            continue
        p = np.average(power[m], axis=0, weights=w[m])
        tot = p[1:].sum()
        if tot <= 0:
            continue
        fracs.append(float(p[cutoff + 1:].sum() / tot))
    if not fracs:
        raise ValueError(f"no populated {band_deg:g} deg bands inside {lat_band}")
    return float(np.mean(fracs))


def global_means(ds: xr.Dataset) -> dict[str, float]:
    """Item 4. cos(lat)-weighted global means of T2m and z500 at the state's valid time.

    Global, not CONUS: item 3 cannot see a runaway that happens outside the crop, and by
    70 days there is no reason a divergence should be polite enough to start over CONUS.
    """
    out: dict[str, float] = {}
    if "2m_temperature" in ds:
        out["global_t2m"] = float(AI.area_mean(_last_frame(ds["2m_temperature"])))
    if "geopotential" in ds:
        z = _last_frame(ds["geopotential"])
        if "level" in z.dims:
            z = z.sel(level=500)
        out["global_z500"] = float(AI.area_mean(z))
    return out


def _last_frame(da: xr.DataArray) -> xr.DataArray:
    da = AI._squeeze(da)
    for d in ("time", "batch", "member"):
        if d in da.dims:
            da = da.isel({d: -1}, drop=True)
    return da


# --------------------------------------------------------------------------- #
# Bands - measured, not guessed
# --------------------------------------------------------------------------- #
# How far past the observed range of 32 known-good walkers a value may sit before it is
# called out. 0.5 half-widths: wide enough that the Gate 3 population itself never trips
# it, narrow enough that the 250x diurnal collapse and the 26 K drift do.
BAND_WIDEN = 0.5


@dataclass(frozen=True)
class Band:
    """A measured operating range for one diagnostic, and how far it was measured.

    ``max_lead_days`` is the honest edge of the measurement: the Gate 3 walkers reach 21 d
    and no further, so past that the band is an EXTRAPOLATION and every consumer - the
    CSV, the figure caption - has to say so rather than draw it like a measurement.
    """
    metric: str
    lo: float
    hi: float
    n: int
    max_lead_days: float
    p05: float = float("nan")
    p50: float = float("nan")
    p95: float = float("nan")
    source: str = "gate3_walkers"

    def status_at(self, lead_days: float, value: float | None = None) -> tuple[str, bool]:
        """``(status, extrapolated)``. Status is ``ok``/``FAIL``/``report``."""
        extrap = float(lead_days) > self.max_lead_days + 1e-9
        if value is None or not np.isfinite(value):
            return ("report", extrap)
        return ("ok" if self.lo <= value <= self.hi else "FAIL", extrap)

    def as_dict(self) -> dict:
        return dict(metric=self.metric, lo=self.lo, hi=self.hi, n=self.n,
                    max_lead_days=self.max_lead_days, p05=self.p05, p50=self.p50,
                    p95=self.p95, source=self.source)


def band_from_samples(metric: str, samples, max_lead_days: float,
                      source: str = "gate3_walkers") -> Band:
    v = np.asarray(list(samples), dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Band(metric, float("nan"), float("nan"), 0, max_lead_days, source=source)
    lo, hi = float(v.min()), float(v.max())
    pad = BAND_WIDEN * (hi - lo) / 2.0
    return Band(metric, lo - pad, hi + pad, int(v.size), float(max_lead_days),
                float(np.percentile(v, 5)), float(np.percentile(v, 50)),
                float(np.percentile(v, 95)), source)


def status_for(metric: str, value, bands: dict, lead_days: float = 0.0) -> tuple[str, bool]:
    """``(status, extrapolated)`` for one measurement, given the measured bands.

    Metrics in ``UNGATED`` always come back ``report``: they have no in-repo baseline, and
    a threshold invented for the figure would read as a measurement.
    """
    if metric in UNGATED or metric not in bands:
        return ("report", False)
    b = bands[metric]
    b = b if isinstance(b, Band) else Band(**{k: v for k, v in b.items()})
    return b.status_at(lead_days, value)


def bands_path() -> Path:
    return SC.bands_path()


def load_bands(event: str | None = None) -> dict[str, Band]:
    p = bands_path()
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    per = raw.get("events", {})
    src = per.get(event) if event and event in per else raw.get("pooled", {})
    return {k: Band(**v) for k, v in src.items()}


