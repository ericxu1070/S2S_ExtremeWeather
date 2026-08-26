#!/usr/bin/env python
"""CFSv2 as the operational baseline: what a real forecast centre had at 21 days.

Every other curve on the trajectory figure is a machine-learning model. That makes the
figure an argument between AI systems, and leaves the obvious question - *would the
operational dynamical model have seen this event either?* - unanswered. This module
answers it, at the same lead, over the same box, with the same reduction.

    A_L^CFS  =  aires.aindex.a_index( CFSv2 forecast, peak, metric, event box )

The forecast is NCEP's operational CFSv2 9-month run, initialised at the AI+RES init and
verified over the identical 7-day window ending at the event peak - so the number it
produces is directly comparable to a walker's realized ``A_L`` and to the direct-sampling
baselines already in ``compare.json``.

The ensemble
------------
CFSv2 runs once every 6 h and each cycle is one deterministic member; an operational
ensemble at a fixed target is built by LAGGING cycles. The default here is the four
cycles ENDING at the AI+RES init - ``init-18h, init-12h, init-6h, init`` - so leads run
21.0 to 21.75 d. Two reasons for trailing rather than the four cycles of the init *day*:

  * the 00Z member is then EXACTLY lead-matched to the GenCast walkers, and
  * no member is at a shorter lead than the thing it is being compared against, so the
    baseline can never be accused of having been handed an easier forecast.

``--lag sameday`` gives the other convention (init, +6h, +12h, +18h) if it is ever wanted.

The climatology caveat - READ THIS BEFORE QUOTING A NUMBER
----------------------------------------------------------
``t2m_anom`` is an anomaly, and the anomaly here is *CFSv2 minus the ERA5 1990-2019
climatology* - NOT CFSv2 minus its own reforecast climatology, which is what NCEP's own
anomaly products use. So ``A_L^CFS`` carries whatever CFSv2 model drift and
representativity offset it has, on top of the forecast signal.

This is deliberate and it is the consistent choice: the GenCast walkers and the FCN3
cubes are scored against the exact same ERA5 climatology with no bias correction, so
every model on the figure is treated identically. It is NOT a bias-corrected forecast and
must not be described as one.

How big is that offset, measured rather than assumed. On the PNW dome, at 12 h into the
forecast - early enough that nothing has drifted - the CFS box-mean T2m is **0.65 K**
below a GenCast walker's over the event box (0.28 K over CONUS), on a ~0.94 deg Gaussian
grid against 0.25 deg in mountainous terrain. That is the scale of the representativity
term, and it is small against the +7.7 K the event itself registers. It is not zero and a
1 K-level result should not be argued from this cube alone.

A second, smaller term with the same cause: a 6 deg event box is only ~6.4 CFS cells
across, so "the box mean" is itself resolution-dependent. Regridding to 0.25 deg and
averaging there differs from a cos(lat) mean over the native cells whose centres fall in
the box by **-0.09 K** on average over the PNW forecast (-0.26 K over the verification
window, 0.57 K at worst on a single frame). Neither estimator is the truth - the native
one has ragged box edges, the regridded one is what makes CFS and a walker share a
reduction - and the spread between them is the discretisation uncertainty on A_L^CFS.

What is NOT an offset: ``init_anom``, the forecast's own day-0/1 box anomaly, is the
WEATHER it starts in, not a bias. On the PNW dome it is -4.98 K because the Pacific
Northwest genuinely was ~5 K below climatology three weeks before the dome built - the
GenCast walkers, initialised from ERA5, start at -3.95 K over the same box. It is printed
because it is where the CFS curve enters the trajectory figure, and the figure draws it
next to the walkers so any real offset between the two is visible rather than argued.

Where the data comes from
-------------------------
NCEI's object store, ``prod-cfs-operational-forecast``, ``time-series/`` product: one
GRIB2 file per cycle per variable holding the whole 9-month run at 6 h resolution
(~95 MB for ``tmp2m``). Each ships a wgrib ``.inv`` sidecar giving the byte offset of
every record, and the store honours HTTP range requests - so a 21-day window is a single
~7 MB prefix read instead of a 95 MB download. Four members per event costs ~27 MB and a
few seconds.

The AWS mirror (``noaa-cfs-pds``) is NOT usable here: it is a rolling archive that starts
in 2023 and every event in this experiment except SCentral_HeatDome_2023 predates it.

Metrics
-------
``t2m_anom`` (``tmp2m``) and ``u850_speed``/``u850_speed_max`` (``wnd850``, which carries
UGRD and VGRD) are supported. The precipitation metrics are NOT: CFS ships ``prate``, a
6 h mean RATE, and turning that into the 12 h accumulation ``xres.xmetrics`` expects is a
unit-and-window conversion with nothing in this repo to validate it against. It raises
rather than guessing.

    # build the cube + A_L for one event (login node; needs internet, no GPU)
    PYTHONPATH=. python -m aires.cfs --event PNW_HeatDome_2021
    PYTHONPATH=. python -m aires.cfs --all           # every registered AI+RES event

``aires/aplots.py`` READS the cache and never builds it, so the figure stage stays
offline. A missing cube drops the CFS curve and says so; it does not fail the figure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from aires import aindex as AI
from fcn3 import fevents as F

# --------------------------------------------------------------------------- #
# The archive
# --------------------------------------------------------------------------- #
CFS_BASE = os.environ.get(
    "AIRES_CFS_BASE",
    "https://www.ncei.noaa.gov/oa/prod-cfs-operational-forecast")
CFS_PRODUCT = "time-series"
CFS_MEMBER = "01"          # the operational time series has one member per cycle
CYCLE_H = 6                # CFSv2 runs 00/06/12/18Z
N_CYCLES = int(os.environ.get("AIRES_CFS_CYCLES", 4))
LAG_MODE = os.environ.get("AIRES_CFS_LAG", "trailing")   # trailing | sameday

# The CFS time-series variable that carries each metric, and what cfgrib names the
# fields inside it once decoded.
METRIC_VAR = {
    "t2m_anom":       "tmp2m",
    "u850_speed":     "wnd850",
    "u850_speed_max": "wnd850",
}
# cfgrib short name -> (GenCast variable name, pressure level or None). The cube this
# module writes uses GenCast's names so aires.aindex reduces it with the identical code
# path that scores a walker - the same trick fcn3/fevents.py::FCN3_TO_GENCAST plays.
GRIB_TO_GENCAST = {
    "t2m": ("2m_temperature", None),
    "u":   ("u_component_of_wind", 850),
    "v":   ("v_component_of_wind", 850),
}

HTTP_ATTEMPTS = 4
HTTP_BACKOFF = 5.0         # seconds, doubled per retry


# --------------------------------------------------------------------------- #
# Cycles and URLs
# --------------------------------------------------------------------------- #
def cycles_for(init, n: int = N_CYCLES, mode: str = LAG_MODE) -> list[pd.Timestamp]:
    """The ``n`` CFSv2 cycles that make up the lagged ensemble for an init.

    ``trailing`` (default) ends ON the init, so every member is at a lead >= the walkers'.
    ``sameday`` starts on it, the convention that treats the init DAY's four cycles as
    the ensemble - those three later members are at 6/12/18 h SHORTER lead.
    """
    init = pd.Timestamp(init)
    if init.hour % CYCLE_H:
        raise SystemExit(f"init {init} is not on a {CYCLE_H} h CFSv2 cycle")
    if mode == "trailing":
        out = [init - pd.Timedelta(hours=CYCLE_H * i) for i in range(n)][::-1]
    elif mode == "sameday":
        out = [init + pd.Timedelta(hours=CYCLE_H * i) for i in range(n)]
    else:
        raise SystemExit(f"unknown lag mode {mode!r}; want trailing|sameday")
    return out


def cfs_url(cycle: pd.Timestamp, var: str, ext: str = "grb2") -> str:
    """The archive URL for one cycle's time series.

    ``ext`` picks the GRIB2 payload or its wgrib inventory. The inventory REPLACES the
    ``.grb2`` suffix (``tmp2m.01.2021060700.daily.inv``) rather than appending to it -
    appending yields a clean 404 on a cycle that is perfectly present.
    """
    stamp = cycle.strftime("%Y%m%d%H")
    return (f"{CFS_BASE}/{CFS_PRODUCT}/{cycle:%Y}/{cycle:%Y%m}/{cycle:%Y%m%d}/{stamp}/"
            f"{var}.{CFS_MEMBER}.{stamp}.daily.{ext}")


# --------------------------------------------------------------------------- #
# HTTP with byte ranges
# --------------------------------------------------------------------------- #
class MissingCycle(Exception):
    """The archive has no such object. Not retryable, and not always fatal."""


def _get(url: str, byte_range: str | None = None, missing_ok: bool = False):
    """One GET, retried. ``byte_range`` is an HTTP Range value like ``0-6659994``.

    A 404 is never retried - it is an absent object, not a flaky link. With
    ``missing_ok`` it returns ``None`` so the caller can fall back; otherwise it raises
    ``MissingCycle``, which ``build`` turns into a skipped member rather than a dead run.
    """
    last = None
    for attempt in range(HTTP_ATTEMPTS):
        req = urllib.request.Request(url, headers={"User-Agent": "aires/cfs"})
        if byte_range:
            req.add_header("Range", f"bytes={byte_range}")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                if missing_ok:
                    return None
                raise MissingCycle(url) from None
            last = e
        except Exception as e:                        # noqa: BLE001 - network is network
            last = e
        if attempt < HTTP_ATTEMPTS - 1:
            time.sleep(HTTP_BACKOFF * 2 ** attempt)
    raise SystemExit(f"giving up on {url} after {HTTP_ATTEMPTS} attempts: {last!r}")


def parse_inv(text: str) -> list[tuple[int, int, int]]:
    """``[(record, byte offset, forecast hour), ...]`` from a wgrib inventory.

    Forecast hour is NOT monotone in record number for multi-field files: ``wnd850``
    writes a day of UGRD, then the same day of VGRD, then the next day. Callers must
    select on the hour and take the largest matching record, never assume a sorted list.
    """
    out = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            continue
        fh = [p for p in parts if p.endswith(" hour fcst")]
        if not fh:
            continue
        out.append((int(parts[0]), int(parts[1]), int(fh[0].split()[0])))
    if not out:
        raise SystemExit("could not parse a CFS inventory; the archive layout changed")
    return out


def fetch_window(cycle: pd.Timestamp, var: str, max_hours: int, dest: Path) -> Path:
    """Download only the records at forecast hour <= ``max_hours``, as one range read.

    Records are laid out in increasing byte order, so "every record we want" is a byte
    PREFIX of the file - one request, no stitching. The end of that prefix is the offset
    of the first record after the last one we want, which is why the whole inventory is
    parsed rather than just the part below the cut.
    """
    url = cfs_url(cycle, var)
    raw = _get(cfs_url(cycle, var, "inv"), missing_ok=True)
    if raw is None:
        # The inventory is a sidecar and the archive is not uniform about shipping it
        # (tmp2m at 2020-07-25 06Z has the GRIB2 but no .inv). Without offsets there is
        # no safe prefix to ask for - a guessed range ends mid-message and eccodes reads
        # garbage - so the whole 9-month run comes down and the extra steps are dropped
        # on the time subset. ~95 MB instead of ~7, on a handful of cycles.
        print(f"    no .inv for {cycle:%Y-%m-%d %HZ}; fetching the whole run instead")
        blob = _get(url)
    else:
        inv = parse_inv(raw.decode())
        wanted = [i for i, (_r, _o, fh) in enumerate(inv) if fh <= max_hours]
        if not wanted:
            raise SystemExit(f"{url}: no record at or below forecast hour {max_hours}")
        last = max(wanted)
        if inv[last][2] < max_hours:
            raise SystemExit(
                f"{url} stops at forecast hour {inv[last][2]} but the verification window "
                f"needs {max_hours} h. The 9-month run should reach ~7150 h; this file is "
                f"truncated and must not be scored.")
        rng = "0-" if last + 1 >= len(inv) else f"0-{inv[last + 1][1] - 1}"
        blob = _get(url, byte_range=rng)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return dest


# --------------------------------------------------------------------------- #
# GRIB -> a cube aires.aindex can score
# --------------------------------------------------------------------------- #
def target_grid() -> tuple[np.ndarray, np.ndarray]:
    """The 0.25 degree CONUS grid every cube in this repo lives on.

    Taken from the climatology file rather than from a config constant on purpose: the
    anomaly is ``field - clim_for(times)``, so a CFS cube regridded to anything else
    would either hard-fail on shape or - the CLAUDE.md climatology trap - silently
    broadcast. Asserting the shape here is the same 105 x 237 check that incident asks
    for, applied before any interpolation rather than after a figure is drawn.
    """
    from gencast_s2s import config as C

    with xr.open_dataset(C.CLIM_FILE) as ds:
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
    if lat.size != 105 or lon.size != 237:
        raise SystemExit(
            f"{C.CLIM_FILE} is {lat.size} x {lon.size}, not the 105 x 237 0.25 degree "
            f"CONUS grid. This is the stale-climatology trap in CLAUDE.md - verify the "
            f"grid, not the filename.")
    return lat, lon


def grib_to_cube(path: Path, metric: str, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """One cycle's GRIB2 prefix -> a GenCast-schema cube on the 0.25 degree CONUS grid.

    Three things happen and all three are load-bearing:

      * ``latitude`` is flipped to ASCENDING. CFS writes 90 -> -90; ``aindex.check_box``
        refuses a descending cube precisely so a slice-based crop cannot silently select
        an empty region.
      * ``step`` is swapped for ``valid_time``, so the cube's time axis is wall-clock and
        ``fc_times_in_window`` can find the verification window without knowing the init.
      * the T126 Gaussian grid (190 x 384, ~0.94 deg) is bilinearly interpolated to the
        0.25 deg CONUS grid. Interpolation to a FINER grid adds no information and
        changes an area mean only at the rounding level, but it is what lets the exact
        same ``aindex`` reduction - and the exact same climatology - score CFS and a
        walker. A CONUS-sized subset is cut first (with a 3 degree halo so the edge cells
        interpolate rather than extrapolate) because interpolating the globe is 20x the
        work for identical output.
    """
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        want = {k: v for k, v in GRIB_TO_GENCAST.items() if k in ds}
        if not want:
            raise SystemExit(f"{path}: none of {list(GRIB_TO_GENCAST)} in {list(ds)}")
        sub = ds[list(want)].rename({"latitude": "lat", "longitude": "lon"})
        sub = sub.assign_coords(
            time=("step", pd.DatetimeIndex(ds["valid_time"].values))
        ).swap_dims({"step": "time"})
        sub = sub.drop_vars([c for c in ("step", "valid_time", "heightAboveGround",
                                         "isobaricInhPa", "surface")
                             if c in sub.coords or c in sub.variables])
        sub = sub.sortby("lat")
        halo = 3.0
        sub = sub.sel(lat=slice(lat[0] - halo, lat[-1] + halo),
                      lon=slice(lon[0] - halo, lon[-1] + halo)).load()
        if sub.sizes["lat"] < 4 or sub.sizes["lon"] < 4:
            raise SystemExit(f"{path}: CONUS subset is {dict(sub.sizes)} - wrong grid?")
        out = sub.interp(lat=lat, lon=lon, method="linear")
    finally:
        ds.close()

    ren = {}
    for short, (name, level) in want.items():
        da = out[short].astype("float32")
        if level is not None:
            da = da.expand_dims(level=[float(level)])
        ren[name] = da
    cube = xr.Dataset(ren)
    if metric == "t2m_anom" and float(cube["2m_temperature"].max()) < 100:
        raise SystemExit(f"{path}: t2m looks like Celsius, not kelvin - unit change?")
    return cube


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(event: str, *, n_cycles: int = N_CYCLES, mode: str = LAG_MODE,
          force: bool = False, workdir: Path | None = None) -> Path:
    """Download, regrid and cache the CFSv2 baseline cube for one event."""
    ev = F.event(event)
    metric = ev.metric
    if metric not in METRIC_VAR:
        raise SystemExit(
            f"{event} is scored on {metric!r}, which this module does not support. CFS "
            f"ships 'prate' (a 6 h mean RATE), and converting it to the 12 h accumulation "
            f"xres.xmetrics expects has nothing in this repo to validate against - see "
            f"the module docstring. Supported: {', '.join(METRIC_VAR)}.")
    var = METRIC_VAR[metric]
    init, peak = pd.Timestamp(ev.init), pd.Timestamp(ev.peak)
    lead_days = (peak - init) / pd.Timedelta(days=1)
    out = A.cfs_cube_path(event, lead_days, metric)
    if out.exists() and not force:
        print(f"[cfs] {event}: cached {out}")
        return out

    cycles = cycles_for(init, n_cycles, mode)
    lat, lon = target_grid()
    work = Path(workdir) if workdir else out.parent / ".grib"
    print(f"[cfs] {event}: {var}, peak {peak.date()}, init {init} ({lead_days:g} d lead)")

    cubes, skipped = [], []
    for cyc in cycles:
        max_hours = int(round((peak - cyc) / pd.Timedelta(hours=1)))
        grb = work / f"{var}.{cyc:%Y%m%d%H}.grb2"
        if not grb.exists() or force:
            try:
                fetch_window(cyc, var, max_hours, grb)
            except MissingCycle as e:
                # One absent cycle must not cost the whole event its baseline. The
                # ensemble is a lag ensemble, so it degrades to n-1 members cleanly; what
                # it may NOT do is silently look like a full one, hence the record in the
                # cube attrs and the printed line.
                print(f"    {cyc:%Y-%m-%d %HZ}  ABSENT from the archive, skipping ({e})")
                skipped.append(str(cyc))
                continue
        c = grib_to_cube(grb, metric, lat, lon)
        # Trim every member to the SAME lead axis. Trailing cycles start before the
        # AI+RES init, so their raw axes are ragged relative to the 00Z member's; a
        # concat over ragged axes silently outer-joins and pads with NaN, which then
        # poisons any reduction taken outside the verification window (it is what made
        # the day-0 drift diagnostic come out nan the first time). The common window is
        # [init + one cycle, peak] - one CFS step in, so the 00Z member is included.
        c = c.sel(time=slice(init + pd.Timedelta(hours=CYCLE_H), peak))
        n = AI.check_window(c, peak, metric, where=f"CFS {cyc:%Y-%m-%d %HZ}")
        lead = (peak - cyc) / pd.Timedelta(days=1)
        print(f"    {cyc:%Y-%m-%d %HZ}  lead {lead:5.2f} d  "
              f"{c.sizes['time']:3d} frames, {n} in the window  "
              f"({grb.stat().st_size / 1e6:.1f} MB)")
        cubes.append(c.expand_dims(member=[len(cubes)]).assign_coords(
            cycle=("member", [cyc.to_datetime64()]),
            member_lead_days=("member", [lead])))

    if len(cubes) < 2:
        raise SystemExit(f"{event}: only {len(cubes)} CFS cycle(s) available; a one-member "
                         f"lag ensemble is not a baseline. Try --cycles 6.")
    # join="exact" rather than the default: the trim above is supposed to have made the
    # four time axes identical, and if it ever does not, that must be an error rather
    # than a quietly NaN-padded cube.
    cube = xr.concat(cubes, dim="member", coords="minimal", compat="override",
                     join="exact")
    if np.isnan(cube[list(cube.data_vars)[0]].values).any():
        raise SystemExit("the assembled CFS cube contains NaN; refusing to cache it")
    AI.check_window(cube, peak, metric, where=f"CFS {event} (assembled)")
    cube.attrs.update(
        source="NCEP CFSv2 operational 9-month forecast (NCEI time-series product)",
        url=CFS_BASE, cfs_variable=var, event=event, metric=metric,
        init=str(init), peak=str(peak), lead_days=float(lead_days), lag_mode=mode,
        cycles_requested=len(cycles), cycles_skipped=",".join(skipped) or "none",
        climatology="ERA5 1990-2019 (runs/models/clim_1990_2019_t2m_conus.nc) - NOT the "
                    "CFSv2 reforecast climatology; anomalies carry CFSv2 model drift, "
                    "exactly as the GenCast and FCN3 cubes carry theirs",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in cube.data_vars}
    tmp = out.with_suffix(f".tmp.{os.getpid()}.nc")
    cube.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, out)
    print(f"[cfs] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    summarize(event, write=True)
    return out


def load(event: str) -> xr.Dataset | None:
    """The cached CFS cube, or ``None`` if it has not been built."""
    ev = F.event(event)
    if ev.metric not in METRIC_VAR:
        return None
    lead = (pd.Timestamp(ev.peak) - pd.Timestamp(ev.init)) / pd.Timedelta(days=1)
    p = A.cfs_cube_path(event, lead, ev.metric)
    return xr.open_dataset(p) if p.exists() else None


# --------------------------------------------------------------------------- #
# Reductions - the same two the walkers report
# --------------------------------------------------------------------------- #
def indices(event: str, cube: xr.Dataset | None = None) -> dict:
    """``A_L`` per member over the event box and over CONUS, plus the drift diagnostic."""
    ev = F.event(event)
    own = cube is None
    cube = load(event) if own else cube
    if cube is None:
        raise SystemExit(f"no CFS cube for {event}; run `python -m aires.cfs "
                         f"--event {event}` first")
    try:
        peak = pd.Timestamp(ev.peak)
        idx = AI.indices(cube, event, peak, ev.metric, where=f"CFS {event}")
        box = np.asarray(idx["box"].values, dtype=float)
        conus = np.asarray(idx["conus"].values, dtype=float)
        lead, ser = series(event, cube)
        # Where the CFS curve ENTERS the figure: its own day-0/1 box anomaly. This is the
        # weather at the init, not a model bias (see the module docstring) - it is worth
        # carrying because a CFS curve that starts far from where the walkers start is
        # the one case where the two are not being read on the same footing.
        first = ser[:, (lead > 0) & (lead <= 1.0)]
        return dict(
            event=event, metric=ev.metric, lead_days=float(
                (peak - pd.Timestamp(ev.init)) / pd.Timedelta(days=1)),
            cycles=[str(pd.Timestamp(c)) for c in np.atleast_1d(cube["cycle"].values)],
            member_lead_days=[float(x) for x in
                              np.atleast_1d(cube["member_lead_days"].values)],
            box=box.tolist(), conus=conus.tolist(),
            box_mean=float(box.mean()), conus_mean=float(conus.mean()),
            init_anom=float(first.mean()) if first.size else float("nan"),
            n_members=int(box.size),
        )
    finally:
        if own:
            cube.close()


def series(event: str, cube: xr.Dataset | None = None):
    """``(lead_days[nt], box[n_members, nt])`` - the running index, lead from the AI+RES init.

    Lead is measured from the EVENT's init, not from each member's own cycle, so the
    curves land on the trajectory figure's axis alongside the walkers. Trailing members
    therefore start at a slightly negative lead; that is correct and the axis clips it.
    """
    ev = F.event(event)
    own = cube is None
    cube = load(event) if own else cube
    if cube is None:
        raise SystemExit(f"no CFS cube for {event}")
    try:
        box = AI.index_series(cube, event, ev.metric)["box"]
        box = box.transpose("member", "time")
        t = pd.DatetimeIndex(cube["time"].values)
        lead = (t - pd.Timestamp(ev.init)).total_seconds() / 86400.0
        return np.asarray(lead, dtype=float), np.asarray(box.values, dtype=float)
    finally:
        if own:
            cube.close()


def daily_mean(lead, y):
    """24 h centred running mean of a series at ANY step that divides 24 h.

    The same filter ``aplots.daily`` applies to the 12-hourly walker index - the
    trapezoid over ``[t-12h, t+12h]``, an exact null at the 24 h period, centred on the
    sample's own timestamp - generalised to the CFS cube's 6 h step. At a 6 h step the
    trapezoid is ``(0.5, 1, 1, 1, 0.5)/4``; ``daily``'s 3-point ``(1, 2, 1)/4`` spans only
    12 h here and leaves most of the diurnal swing on the curve (``test_cfs.py`` pins
    both halves of that statement).

    The ends are the subtle part, and they matter because the LAST point of every curve
    sits on the event peak, where the eye reads ``A_L`` off the figure. The window is
    closed by padding

        y[-j]  =  y[P-j] - (y[P] - y[0])        P = one 24 h period in samples

    i.e. the value one period earlier, shifted back by the trend accumulated over that
    period. That is simultaneously exact on a linear trend (so the endpoint is not pulled
    back down its own slope) and an exact periodic extension of a 24 h cycle (so the
    endpoint is as diurnal-free as the interior). Plain linear extrapolation gives the
    first property and NOT the second - it leaves up to a quarter of the diurnal
    amplitude, ~1 K on a box-mean T2m anomaly, on the one point that is read as a number.

    At ``P = 2`` this reduces to ``y[0] + y[1] - y[2]``, which is exactly the padding
    ``aplots.daily`` uses - so the two functions are the same rule at two steps rather
    than two rules that happen to look alike.
    """
    lead = np.asarray(lead, dtype=float)
    y = np.asarray(y, dtype=float)
    if lead.size < 3:
        return lead, y
    step = float(np.median(np.diff(lead)))
    k = int(round(0.5 / step))                       # samples in a 12 h half-window
    P = 2 * k                                        # samples in 24 h
    if k < 1 or y.shape[-1] < P + 1:
        return lead, y
    w = np.ones(P + 1)
    w[0] = w[-1] = 0.5
    w /= w.sum()
    # The two pads run in OPPOSITE index order: the leading pad is laid out
    # p[-k] ... p[-1] (j counting down), the trailing one p[+1] ... p[+k] (j counting up).
    # Sharing one j array silently reverses the trailing pad, which stays exact at the
    # 24 h null and quietly breaks the linear-trend property at the last few points.
    jl, jh = np.arange(k, 0, -1), np.arange(1, k + 1)
    lo = y[..., P - jl] - (y[..., P:P + 1] - y[..., 0:1])
    hi = y[..., -1 - P + jh] + (y[..., -1:] - y[..., -1 - P:-P])
    pad = np.concatenate([lo, y, hi], axis=-1)
    sm = np.apply_along_axis(lambda v: np.convolve(v, w, mode="valid"), -1, pad)
    return lead, sm


def all_events() -> list[str]:
    """Every event this baseline is wanted for, in registry order.

    The AI+RES extension registry (``fevents.RES_EVENTS``) UNION whatever already has a
    run on disk. The union matters: PNW_HeatDome_2021 is the pilot event and the one with
    the most figures, but it lives in ``fevents.EVENTS`` - it predates the extension
    registry - so an ``--all`` built from ``res_events()`` alone silently skips it.
    """
    have = {p.parent.parent.parent.name
            for p in sorted(A.AIRES_ROOT.glob("*/res/*/compare.json"))}
    want = list(F.RES_ORDER) + [n for n in F.KNOWN_ORDER if n in have]
    return list(dict.fromkeys(want))


def observed(event: str, tag: str | None = None) -> float | None:
    """The event's observed ``A_L``, read from an AI+RES run if one exists.

    Only so the CLI can print the baseline next to the thing it is a baseline FOR. It is
    ``None`` for an event whose run has not been done - the CFS cube is perfectly
    buildable before there is anything to compare it against.
    """
    p = A.res_dir(event, tag) / "compare.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["observed"])
    except (KeyError, TypeError, ValueError):
        return None


def summarize(event: str, write: bool = False) -> dict:
    """Print (and optionally cache) the A_L table for one event."""
    d = indices(event)
    ev = F.event(event)
    box = AI.box_for(event)
    print(f"[cfs] {event}: A_L over {box.name} ({box.label}), {d['lead_days']:g} d lead")
    for c, l, b in zip(d["cycles"], d["member_lead_days"], d["box"]):
        print(f"    {c[:16]}  lead {l:5.2f} d   A_L = {b:+7.3f}")
    print(f"    ensemble mean A_L = {d['box_mean']:+.3f} "
          f"(CONUS {d['conus_mean']:+.3f})")
    print(f"    day-0/1 box anomaly = {d['init_anom']:+.3f}  <- where the forecast "
          f"starts (the weather at the init), not a bias")
    obs = observed(event)
    if obs is not None:
        k = int(np.sum(AI.tail_sign(event) * np.asarray(d["box"])
                       >= AI.tail_sign(event) * obs))
        print(f"    observed A_L = {obs:+.3f}; CFS members reaching it: "
              f"{k}/{d['n_members']}")
    if write:
        p = A.cfs_json_path(event, d["lead_days"], ev.metric)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2, sort_keys=True))
        print(f"[cfs] wrote {p}")
    return d


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default=os.environ.get("AIRES_EVENT"))
    ap.add_argument("--all", action="store_true",
                    help="every AI+RES event whose metric CFS can score")
    ap.add_argument("--cycles", type=int, default=N_CYCLES)
    ap.add_argument("--lag", default=LAG_MODE, choices=("trailing", "sameday"))
    ap.add_argument("--force", action="store_true", help="re-download and rebuild")
    ap.add_argument("--summary-only", action="store_true",
                    help="print A_L from the cached cube; no network")
    ap.add_argument("--keep-grib", action="store_true",
                    help="keep the downloaded GRIB prefixes (default: delete)")
    a = ap.parse_args(argv)

    if a.all:
        events = [n for n in all_events() if F.event(n).metric in METRIC_VAR]
    elif a.event:
        events = [a.event]
    else:
        raise SystemExit("give --event NAME or --all")

    rows = []
    for name in events:
        if a.summary_only:
            rows.append(summarize(name))
            continue
        build(name, n_cycles=a.cycles, mode=a.lag, force=a.force)
        rows.append(indices(name))
        if not a.keep_grib:
            g = A.cfs_dir(name) / ".grib"
            for f in sorted(g.glob("*.grb2")):
                f.unlink()
            if g.exists() and not any(g.iterdir()):
                g.rmdir()
        print()

    if len(rows) > 1:
        # "most extreme" is tail-signed, like everything else in this experiment: on a
        # cold event the extreme member is the most NEGATIVE one, and a bare max() would
        # report the warmest member of a freeze as its best forecast.
        print(f"\n{'event':28s} {'lead':>6s} {'A_L mean':>9s} {'A_L tail':>9s} "
              f"{'day0':>7s}  {'observed':>9s}")
        for d in rows:
            b = np.asarray(d["box"])
            sgn = AI.tail_sign(d["event"])
            tail = float(b[np.argmax(sgn * b)])
            obs = observed(d["event"])
            print(f"{d['event']:28s} {d['lead_days']:5.0f}d {d['box_mean']:+9.3f} "
                  f"{tail:+9.3f} {d['init_anom']:+7.3f}  "
                  + (f"{obs:+9.3f}" if obs is not None else f"{'-':>9s}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
