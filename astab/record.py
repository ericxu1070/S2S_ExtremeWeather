#!/usr/bin/env python
"""The per-segment record - written at WALK time, from the state before it is pruned.

Items 4 and 7 are GLOBAL and cannot be recovered from the CONUS crop afterwards, so a
record that is not written here is a hole in the curve that only a re-walk could fill.
``stab.json`` and ``diag.nc`` are for that reason the two things ``--stage prune`` never
touches: every diagnostic in the sweep is derived from them.

What is stored is a MEASUREMENT, never a verdict. ``field_ranges`` goes in; ``BOUNDS`` is
applied to it at reduce time. The smoke run made the reason concrete:
``BOUNDS["specific_humidity"]`` was corrected from ``0.0`` to ``-5e-3`` AFTER the segments
were written, and records carrying a pre-computed verdict still reported the healthy chain
as diverged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from astab.diag import (BOUNDS, conus_series, diurnal_ratio_series, field_ranges,
                        global_means, nonfinite_fractions,
                        zonal_high_wavenumber_fraction)
from astab.refs import reference_at
from astab.schedule import Chain

def segment_record(chain: Chain, step: int, lead_days: float, n_steps: int) -> dict:
    """Everything the panel needs from one segment: CONUS items from ``diag.nc``, global
    items from ``state.nc``.

    Written while the state is still on disk. Items 4 and 7 are GLOBAL and cannot be
    recovered from the CONUS crop afterwards, so a record that is not written here is a
    hole in the curve that only a re-walk could fill. For the ensemble follow-up that goes
    further: the state is RELEASED and the diag is SLIMMED at the end of every member
    chain, so this file is the only surviving description of that member's atmosphere.

    Which is why it carries three things the frozen sweep's records do not need:

    * ``member`` and ``seed`` - self-identification. 100 records under
      ``ens/wk08/m007/`` that do not say which member they are cannot be re-associated if
      a directory is ever moved, and ``seed`` is what says which GenCast noise stream the
      member drew (member 2 is job 1189's, deliberately).
    * ``diag_vars`` - what ``diag.nc`` still held when this record was computed. After the
      slim it holds T2m only, so a record built from a slimmed cube has a narrower
      ``nonfinite_conus``/``ranges_conus`` than one built at walk time, and nothing else
      on disk would say so.
    * ``ranges_global_levels`` and ``extremes_global`` - WHERE the state left its bounds.
      Job 1189's two silent divergences were ``temperature`` at 50 hPa over Antarctica
      while CONUS stayed textbook-normal; ``ranges_global`` says the variable and this
      says the level and the grid point. With 100 members and no states retained, that
      attribution has to be captured at walk time or it does not exist.

    Nothing is renamed: ``reduce._walker_panel`` reads ``conus``/``global``/``ranges_*``/
    ``nonfinite_*``/``era5``/``valid_time`` and must keep reading job 1189's records too.
    """
    rec: dict = dict(event=chain.event, weeks=chain.weeks, walker=chain.walker,
                     member=(None if chain.member is None else int(chain.member)),
                     seed=int(chain.seed),
                     step=int(step), lead_days=float(lead_days), n_steps=int(n_steps),
                     valid_time=str(chain.valid_at(lead_days)))

    dp = chain.diag_path(step)
    if dp.exists():
        with xr.open_dataset(dp) as d:
            d = d.load()
        rec["diag_vars"] = sorted(str(v) for v in d.data_vars)
        ser = conus_series(d)
        dur = diurnal_ratio_series(d)
        rec["conus"] = dict(
            times=[str(t) for t in ser.index],
            t2m_mean=ser["t2m_mean"].tolist(),
            t2m_sd=ser["t2m_sd"].tolist(),
            t2m_anom=ser["t2m_anom"].tolist(),
            t2m_mean_last=float(ser["t2m_mean"].iloc[-1]),
            t2m_anom_last=float(ser["t2m_anom"].iloc[-1]),
            t2m_sd_last=float(ser["t2m_sd"].iloc[-1]),
            diurnal_amp_last=(float(dur["amp"].iloc[-1]) if not dur.empty else float("nan")),
            diurnal_ratio_last=(float(dur["ratio"].iloc[-1]) if not dur.empty
                                else float("nan")),
            diurnal_ratio_mean=(float(np.nanmean(dur["ratio"].values)) if not dur.empty
                                else float("nan")),
        )
        rec["nonfinite_conus"] = nonfinite_fractions(d)
        rec["ranges_conus"] = field_ranges(d)
        del d

    sp = chain.state_path(step)
    if sp.exists():
        with xr.open_dataset(sp) as s:
            s = s.load()
        rec["global"] = global_means(s)
        rec["global"]["z500_hi_wavenumber_frac"] = zonal_high_wavenumber_fraction(
            s["geopotential"].sel(level=500))
        rec["nonfinite_global"] = nonfinite_fractions(s)
        rec["ranges_global"] = field_ranges(s)
        rec["ranges_global_levels"] = ranges_by_level(s)
        rec["extremes_global"] = global_extremes(s)
        rec["params_file"] = str(s.attrs.get("params_file", ""))
        rec["resolution"] = str(s.attrs.get("resolution", ""))
        del s
    ref = reference_at(chain.event, chain.valid_at(lead_days))
    if ref:
        rec["era5"] = ref
    return rec


def ranges_by_level(ds: xr.Dataset) -> dict[str, dict]:
    """``{var: {level: {min, max}}}`` for the bounded variables that have levels.

    The same reduction ``field_ranges`` makes - finite values only, over BOTH frames of
    the state - one level at a time, so ``min(min over levels) == field_ranges[var].min``
    exactly. That identity is what makes this an attribution of the existing number rather
    than a second, subtly different measurement of it.

    It exists because "temperature left its bounds" and "temperature at 50 hPa left its
    bounds while every tropospheric level was fine" are different findings, and job 1189
    could only tell them apart by keeping the 477 MB state. A 100-member run cannot.
    """
    out: dict[str, dict] = {}
    for v in BOUNDS:
        if v not in ds or "level" not in ds[v].dims:
            continue
        da = ds[v]
        per: dict[str, dict] = {}
        for lv in np.asarray(da["level"].values).ravel():
            a = np.asarray(da.sel(level=lv).values, dtype="float64")
            fin = np.isfinite(a)
            per[_level_key(lv)] = dict(
                min=float(a[fin].min()) if fin.any() else float("nan"),
                max=float(a[fin].max()) if fin.any() else float("nan"))
        out[str(v)] = per
    return out


def _level_key(lv) -> str:
    f = float(lv)
    return str(int(f)) if f == int(f) else str(f)


def global_extremes(ds: xr.Dataset) -> dict[str, dict]:
    """``{var: {'min'|'max': {value, lat, lon[, level]}}}`` on the LAST FRAME ONLY.

    Deliberately not "both frames like ``field_ranges``": a located extremum needs one
    frame's index space to be meaningful, and the last frame is the one the segment landed
    on - the state a clone would continue from and the one the next segment inherits. So
    ``extremes_global[v]['max']['value']`` can be slightly INSIDE
    ``ranges_global[v]['max']`` when the earlier frame held the worse value. That is the
    intended relationship and both are stored, rather than one being derived from the
    other.

    Non-finite points are excluded from the argmin/argmax; a field with no finite value at
    all reports ``None``, which is item 1's business rather than this one's.
    """
    out: dict[str, dict] = {}
    for v in BOUNDS:
        if v not in ds:
            continue
        da = _last_frame(ds[v])
        a = np.asarray(da.values, dtype="float64")
        fin = np.isfinite(a)
        if not fin.any():
            out[str(v)] = dict(min=None, max=None)
            continue
        lo = np.unravel_index(int(np.argmin(np.where(fin, a, np.inf))), a.shape)
        hi = np.unravel_index(int(np.argmax(np.where(fin, a, -np.inf))), a.shape)
        out[str(v)] = dict(min=_locate(da, a, lo), max=_locate(da, a, hi))
    return out


def _last_frame(da: xr.DataArray) -> xr.DataArray:
    for d in ("batch", "time", "member", "sample"):
        if d in da.dims:
            da = da.isel({d: -1}, drop=True)
    return da


def _locate(da: xr.DataArray, values: np.ndarray, idx) -> dict:
    """The value at ``idx`` plus whichever of lat/lon/level the array is indexed by."""
    loc: dict = dict(value=float(values[idx]))
    for dim, i in zip(da.dims, idx):
        if dim not in da.coords:
            continue
        c = np.asarray(da[dim].values).ravel()
        loc[str(dim)] = (float(c[i]) if np.issubdtype(c.dtype, np.number)
                         else str(c[i]))
    return loc


def ensure_record(chain: Chain, step: int, lead_days: float, n_steps: int, *,
                  force: bool = False) -> dict:
    p = chain.record_path(step)
    if p.exists() and not force:
        return json.loads(p.read_text())
    rec = segment_record(chain, step, lead_days, n_steps)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(rec, indent=2, default=jsonable))
    os.replace(tmp, p)
    return rec


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, Path)):
        return str(o)
    raise TypeError(f"not JSON-serialisable: {type(o)}")


