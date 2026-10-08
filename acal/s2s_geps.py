"""ECCC GEPS week-3 baseline for the 42 acal cases, from SubX (IRIDL anonymous OPeNDAP).

What it builds
--------------
One canonical cube per case, ``runs/acal/s2s/geps/<eid>.nc``: all 21 members of the latest
operational ECCC GEPS extended-range start on or before the AI+RES init (peak - 21 d, 00Z),
2 m temperature daily means bilinearly regridded from SubX's 1 degree grid to the 0.25 degree
acal grid (``aires.cfs.target_grid``), plus a leave-one-year-out hind cube per (case, other
year) under ``runs/acal/s2s/geps/hind/`` for the bias-corrected sensitivity.

Data route
----------
``https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS{6,7,8}/.forecast/.tas/dods``.
The IRIDL HTML pages now demand a login; the ``dods`` endpoints still answer anonymously.
That is undocumented and may close, so ``--stage fetch`` downloads every start any case or
hind year needs to ``raw/`` first (all members, all leads, a 1 degree crop with a 3 degree
halo) and ``build``/``hind`` never touch the network.

Three SubX datasets cover the period, one per GEPS version, with no gap and no overlap
except one shared date (2021-12-02, the GEPS7 implementation week; the newer version wins):

=======  =========================  ==================  =======  =====================
dataset  starts with data           schedule            members  leads L (days)
=======  =========================  ==================  =======  =====================
GEPS6    2019-06-27 .. 2021-12-02   weekly, Thursday    21       0.5 .. 31.5
GEPS7    2021-12-02 .. 2024-06-06   weekly, Thursday    21       0.5 .. 31.5
GEPS8    2024-06-13 .. 2026-10-05   Monday + Thursday   21       0.5 .. 38.5
=======  =========================  ==================  =======  =====================

``tas`` is ``[S, M, L, Y, X]`` for GEPS6/7 but ``[M, S, L, Y, X]`` for GEPS8, so every read
here indexes by dimension NAME. ``Y`` ascends (-90..90), ``X`` is 0..359, ``S`` is days since
1960-01-01 at 00:00 (every start is a 00Z run), missing values arrive masked or as ~1e21.

What one SubX daily value is (measured, not assumed)
-----------------------------------------------------
The IRIDL metadata say only ``L`` = "Lead" in days with values k + 0.5, the IRIDL dataset
pages now sit behind a login, and the SubX protocol paper (Pegion et al. 2019, BAMS,
appendix A) says only "daily values". So it was measured: the 21-member mean at k = 0 and 1
(where the forecast is still close to the analysis) was compared with ERA5 hourly 2 m
temperature averaged over candidate windows, cos-lat RMSE over the 1 degree CONUS box, for
8 starts spread over GEPS6/7/8 (2021-2026). Scripts, per-start CSVs and logs:
``runs/acal/s2s/geps/convention/convention{,2}.{py,csv,log}``.

* Which day: the UTC day ``S + k``. A 24 h window shifted by a full day either way is
  1.0-1.5 K worse (first probe, 5 starts, k = 0: 1.81 K at day S, 2.96 K at day S-1,
  3.33 K at day S+1).
* Which hours: among windows a forecast can actually produce at k = 0 (nothing exists
  before ``S`` 00Z), the mean of the 00, 06, 12 and 18Z samples fits best (k=0/1: 1.68/1.87
  K), ahead of 3-hourly 00..21Z (1.69/1.87), hourly 00..23Z (1.71/1.89), the trapezoid
  00..24Z (1.76/1.95) and 06..24Z (1.76/1.92). Windows starting 3 h earlier fit marginally
  better (1.64/1.80) at k = 0 as well, where that window cannot be the definition, so that
  residual is a GEPS diurnal-phase bias, not the averaging window.

So **``L = k + 0.5`` is the mean over UTC day ``S + k``** (00Z included, the next 00Z not),
and this module stamps it ``S + k`` at 00Z. That is exactly the daily-mean contract the
acal reducers expect: days ``peak-6 .. peak-1`` scored against the interval-mean
climatology (mean of the 00/06/12/18Z climatology of each day), truth on the 12 frames
``peak-6d 00Z .. peak-1d 12Z`` (ruling C9). The research note's SubX window
``L in [(p-s)-6+0.5, (p-s)+0.5]`` is one day late (it is days peak-6 .. peak) and is not
used.

Leave-one-year-out hind rule
----------------------------
For case peak ``P`` scored from start ``S0`` and every other year ``y`` in 2021-2026 whose
calendar-shifted peak ``P_y = P + (y - P.year) years`` (Feb 29 -> Feb 28) lies inside the
ERA5/HRRR truth record (2021-01-02 .. ``ccfg.PEAK_END``): the hind start is the GEPS start
NEAREST to ``P_y - (P - S0)``, i.e. the same calendar start day and the same lead to the
same calendar window. Ties go to the earlier start (the longer lead). GEPS starts at least
weekly with no gaps, so the shift is at most 3 days (GEPS8: 2); anything beyond
``HIND_MAX_SHIFT_D`` is skipped and recorded. The hind cube follows the case cube
contract with ``peak = P_y`` and ``init = P_y - 21 d``; ``member_lead_days`` records the
actual lead.

CLI
---
``python -m acal.s2s_geps --stage fetch [--case EID] [--workers N]``   network, resumable
``python -m acal.s2s_geps --stage build [--case EID]``                 offline
``python -m acal.s2s_geps --stage hind  [--case EID] [--workers N]``   offline
``python -m acal.s2s_geps --stage table``                              sanity A_L vs catalog
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from acal import ccfg

NAME = "geps"
SUBX_BASE = os.environ.get(
    "ACAL_SUBX_ECCC", "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC")
DATASETS = ("GEPS6", "GEPS7", "GEPS8")        # oldest -> newest; a shared date goes to the newest
VARIABLE = "tas"
SUBX_EPOCH = pd.Timestamp("1960-01-01")
MISSING_ABOVE = 1e10                           # SubX fill is 9.999e20
N_MEMBERS = 21
MIN_MEMBERS = 15                               # fewer complete members -> refuse the cube

SOURCE = dict(
    name=NAME,
    label="ECCC GEPS",
    native_deg=1.0,
    time_kind="daily_mean",
    n_members=N_MEMBERS,
    init_rule="latest SubX GEPS start (00Z; weekly Thu for GEPS6/7, Mon+Thu for GEPS8) "
              "on or before the AI+RES init = peak - 21 d",
    bias="loyo",
    dataset_id="IRIDL SubX ECCC GEPS6/GEPS7/GEPS8 .forecast .tas (daily mean)",
    url=SUBX_BASE + "/.GEPS{6,7,8}/.forecast/.tas/dods",
)

ROOT = ccfg.ACAL_ROOT / "s2s" / NAME
RAW = ROOT / "raw"
HIND = ROOT / "hind"
STARTS_CSV = RAW / "starts.csv"
FETCH_LOG = RAW / "fetch_log.csv"
SANITY_CSV = ROOT / "sanity.csv"

LEAD_DAYS = ccfg.LEAD_DAYS                     # 21: the AI+RES init is peak - 21 d
WINDOW_DAYS = 6                                # scored days: peak-6 .. peak-1 (UTC)
MARGIN_DAYS = 1                                # cube carries peak-7 .. peak
HALO_DEG = 3.0
HIND_YEARS = range(2021, 2027)
HIND_FIRST_PEAK = pd.Timestamp("2021-01-02")   # the ERA5 index starts 2020-12-26 00Z
HIND_LAST_PEAK = pd.Timestamp(ccfg.PEAK_END)
HIND_MAX_SHIFT_D = 7
MIN_HIND_YEARS = 4
HTTP_ATTEMPTS = 5
HTTP_BACKOFF = 10.0                            # seconds, doubled per retry
PROBE_LATLON = (40.0, 260.0)                   # one central-CONUS land point


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def url(model: str) -> str:
    return f"{SUBX_BASE}/.{model}/.forecast/.{VARIABLE}/dods"


def cube_path(eid: str) -> Path:
    return ROOT / f"{eid}.nc"


def hind_path(eid: str, year: int) -> Path:
    return HIND / f"{eid}_{year}.nc"


def raw_path(model: str, start) -> Path:
    return RAW / f"{model}_{pd.Timestamp(start):%Y%m%d}.nc"


# --------------------------------------------------------------------------- #
# Cases and start selection (pure; tested offline)
# --------------------------------------------------------------------------- #
def episodes() -> pd.DataFrame:
    from acal import aprep

    return aprep.episodes()


def case_row(eid: str):
    df = episodes()
    hit = df[df.episode_id == eid]
    if hit.empty:
        raise SystemExit(f"[geps] unknown case {eid}")
    return hit.iloc[0]


def case_times(row) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(init, peak) at 00Z. init is peak - 21 d, as the catalog's own ``init`` column says."""
    peak = pd.Timestamp(row.peak).normalize()
    init = peak - pd.Timedelta(days=LEAD_DAYS)
    cat_init = getattr(row, "init", None)
    if cat_init is not None and pd.notna(cat_init) and pd.Timestamp(cat_init).normalize() != init:
        raise SystemExit(f"[geps] {row.episode_id}: catalog init {row.init} != peak - "
                         f"{LEAD_DAYS} d")
    return init, peak


def window_days(peak) -> pd.DatetimeIndex:
    """The UTC days (stamped 00Z) a daily-mean source is scored on: peak-6 .. peak-1."""
    peak = pd.Timestamp(peak).normalize()
    return pd.date_range(peak - pd.Timedelta(days=WINDOW_DAYS), periods=WINDOW_DAYS, freq="D")


def cube_days(peak) -> pd.DatetimeIndex:
    """The days a cube carries: the scored window plus ``MARGIN_DAYS`` on each side."""
    peak = pd.Timestamp(peak).normalize()
    return pd.date_range(peak - pd.Timedelta(days=WINDOW_DAYS + MARGIN_DAYS),
                         peak, freq="D")


def lead_of_day(start, day) -> float:
    """SubX lead ``L`` holding UTC day ``day`` of a 00Z start: day S+k is L = k + 0.5."""
    k = (pd.Timestamp(day).normalize() - pd.Timestamp(start).normalize()).days
    return k + 0.5


def _rank(model: str) -> int:
    return DATASETS.index(model)


def pick_start(init, inv: pd.DataFrame) -> pd.Series | None:
    """Latest start on or before ``init``; on a shared date the newer GEPS version wins."""
    init = pd.Timestamp(init)
    c = inv[inv.start <= init]
    if c.empty:
        return None
    c = c.assign(rank=c.model.map(_rank)).sort_values(["start", "rank"])
    return c.iloc[-1]


def shifted_peak(peak, year: int) -> pd.Timestamp:
    peak = pd.Timestamp(peak)
    return (peak + pd.DateOffset(years=year - peak.year)).normalize()  # Feb 29 -> Feb 28


def pick_hind_start(target, inv: pd.DataFrame,
                    max_shift_d: int = HIND_MAX_SHIFT_D) -> pd.Series | None:
    """The start nearest ``target``; a tie goes to the earlier start, a shared date to the
    newer GEPS version. None if nothing lies within ``max_shift_d`` days."""
    target = pd.Timestamp(target)
    d = (inv.start - target).dt.days
    c = inv.assign(absd=d.abs(), d=d, rank=inv.model.map(_rank))
    c = c[c.absd <= max_shift_d]
    if c.empty:
        return None
    c = c.sort_values(["absd", "d", "rank"], ascending=[True, True, False])
    return c.iloc[0]


def hind_years(peak) -> list[int]:
    peak = pd.Timestamp(peak)
    return [y for y in HIND_YEARS if y != peak.year
            and HIND_FIRST_PEAK <= shifted_peak(peak, y) <= HIND_LAST_PEAK]


def hind_plan(row, inv: pd.DataFrame) -> list[dict]:
    """Every (year, start) the leave-one-year-out bias of this case needs."""
    init, peak = case_times(row)
    s0 = pick_start(init, inv)
    if s0 is None:
        return []
    lead = peak - s0.start
    out = []
    for y in hind_years(peak):
        p = shifted_peak(peak, y)
        target = p - lead
        s = pick_hind_start(target, inv)
        out.append(dict(episode_id=row.episode_id, year=y, peak=p, target=target,
                        model=None if s is None else s.model,
                        start=None if s is None else s.start,
                        shift_d=None if s is None else int((s.start - target).days)))
    return out


# --------------------------------------------------------------------------- #
# Network (fetch stage only)
# --------------------------------------------------------------------------- #
_OPEN: dict = {}


def _open(model: str):
    """One netCDF4 OPeNDAP handle per model per PROCESS (netCDF4 is not thread-safe)."""
    import netCDF4

    if model in _OPEN:
        return _OPEN[model]
    err = None
    for i in range(HTTP_ATTEMPTS):
        try:
            ds = netCDF4.Dataset(url(model))
            v = ds.variables[VARIABLE]
            ax = dict(
                S=SUBX_EPOCH + pd.to_timedelta(np.asarray(ds.variables["S"][:], float),
                                               unit="D"),
                M=np.asarray(ds.variables["M"][:]),
                L=np.asarray(ds.variables["L"][:], float),
                Y=np.asarray(ds.variables["Y"][:], float),
                X=np.asarray(ds.variables["X"][:], float),
            )
            _OPEN[model] = (ds, v, ax)
            return _OPEN[model]
        except Exception as e:                       # noqa: BLE001 - retried, then raised
            err = e
            time.sleep(HTTP_BACKOFF * 2 ** i)
    raise SystemExit(f"[geps] cannot open {url(model)}: {err}")


def _read(v, sel: dict) -> tuple[np.ndarray, list[str]]:
    """Index a SubX variable by dimension NAME; returns (array, remaining dims)."""
    idx = tuple(sel.get(d, slice(None)) for d in v.dimensions)
    a = v[idx]
    a = np.ma.filled(np.ma.asarray(a).astype("float64"), np.nan)
    a[np.abs(a) > MISSING_ABOVE] = np.nan
    dims = [d for d, i in zip(v.dimensions, idx) if not isinstance(i, (int, np.integer))]
    return a, dims


def inventory(force: bool = False) -> pd.DataFrame:
    """Every GEPS start with finite data at one central-CONUS point (member 0, L = 0.5).

    The S axes are padded (GEPS6 is a daily axis with weekly data), so the starts that
    exist have to be found by looking, not read off the axis."""
    if STARTS_CSV.exists() and not force:
        return read_inventory()
    rows = []
    for m in DATASETS:
        _, v, ax = _open(m)
        yi = int(np.argmin(np.abs(ax["Y"] - PROBE_LATLON[0])))
        xi = int(np.argmin(np.abs(ax["X"] - PROBE_LATLON[1])))
        a, dims = _read(v, {"M": 0, "L": 0, "Y": yi, "X": xi})
        assert dims == ["S"], dims
        for s, val in zip(ax["S"], a):
            if np.isfinite(val):
                rows.append(dict(model=m, start=pd.Timestamp(s), probe_K=round(float(val), 2)))
        print(f"[geps] inventory {m}: {sum(r['model'] == m for r in rows)} starts", flush=True)
    inv = pd.DataFrame(rows)
    if (inv.start.dt.hour != 0).any():
        raise SystemExit("[geps] a SubX start is not at 00Z - the day stamping assumes 00Z")
    RAW.mkdir(parents=True, exist_ok=True)
    _atomic_csv(inv, STARTS_CSV)
    return inv


def read_inventory() -> pd.DataFrame:
    inv = pd.read_csv(STARTS_CSV, parse_dates=["start"])
    return inv.sort_values(["start", "model"]).reset_index(drop=True)


def _crop_idx(vals: np.ndarray, lo: float, hi: float) -> slice:
    i = np.where((vals >= lo) & (vals <= hi))[0]
    if i.size == 0 or np.any(np.diff(i) != 1):
        raise SystemExit(f"[geps] crop {lo}..{hi} is not a contiguous slice of the axis")
    return slice(int(i[0]), int(i[-1]) + 1)


def fetch_start(model: str, start, force: bool = False) -> Path:
    """All members x all leads of one start over CONUS + halo at 1 degree -> raw/."""
    start = pd.Timestamp(start)
    out = raw_path(model, start)
    if out.exists() and not force:
        return out
    from aires.cfs import target_grid

    lat, lon = target_grid()
    ds, v, ax = _open(model)
    hit = np.where(ax["S"] == start)[0]
    if hit.size != 1:
        raise SystemExit(f"[geps] {model} has no start {start:%Y-%m-%d}")
    ysl = _crop_idx(ax["Y"], lat[0] - HALO_DEG, lat[-1] + HALO_DEG)
    xsl = _crop_idx(ax["X"], lon[0] - HALO_DEG, lon[-1] + HALO_DEG)
    err = None
    for i in range(HTTP_ATTEMPTS):
        try:
            a, dims = _read(v, {"S": int(hit[0]), "Y": ysl, "X": xsl})
            break
        except Exception as e:                       # noqa: BLE001 - retried, then raised
            err = e
            _OPEN.pop(model, None)
            time.sleep(HTTP_BACKOFF * 2 ** i)
            ds, v, ax = _open(model)
    else:
        raise SystemExit(f"[geps] {model} {start:%Y-%m-%d}: fetch failed: {err}")
    a = np.transpose(a, [dims.index(d) for d in ("M", "L", "Y", "X")]).astype("float32")
    raw = xr.Dataset(
        {VARIABLE: (("member", "lead", "lat", "lon"), a)},
        coords=dict(member=np.asarray(ax["M"]).astype(int), lead=ax["L"],
                    lat=ax["Y"][ysl], lon=ax["X"][xsl]),
        attrs=dict(model=model, start=str(start), url=url(model),
                   subx_dims=",".join(v.dimensions), units="K",
                   fetched_utc=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                   note="raw SubX crop: lead L = k + 0.5 is the mean over UTC day start + k"),
    )
    raw = raw.sortby("lat")
    _atomic_nc(raw, out)
    return out


def _fetch_job(job: tuple[str, str]) -> dict:
    model, start = job
    t0 = time.time()
    try:
        p = fetch_start(model, start)
        with xr.open_dataset(p) as r:
            fin = np.isfinite(r[VARIABLE].values).all(axis=(1, 2, 3))
        return dict(model=model, start=start, ok=True, n_full_members=int(fin.sum()),
                    mb=round(p.stat().st_size / 1e6, 2), secs=round(time.time() - t0, 1))
    except BaseException as e:                       # noqa: BLE001 - reported, not swallowed
        return dict(model=model, start=start, ok=False, error=str(e)[:300],
                    secs=round(time.time() - t0, 1))


def needed_starts(cases: list[str] | None = None, inv: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
    """Every (model, start) a case cube or a hind cube needs, with who needs it."""
    inv = read_inventory() if inv is None else inv
    df = episodes()
    if cases:
        df = df[df.episode_id.isin(cases)]
    rows = []
    for r in df.itertuples():
        init, _ = case_times(r)
        s = pick_start(init, inv)
        if s is not None:
            rows.append(dict(model=s.model, start=s.start, use=f"{r.episode_id}:case"))
        for h in hind_plan(r, inv):
            if h["start"] is not None:
                rows.append(dict(model=h["model"], start=h["start"],
                                 use=f"{r.episode_id}:{h['year']}"))
    return pd.DataFrame(rows)


def fetch(cases: list[str] | None = None, workers: int = 6) -> pd.DataFrame:
    inv = inventory()
    need = needed_starts(cases, inv)
    for ds, _, _ in _OPEN.values():          # do not hand open curl handles to forked workers
        ds.close()
    _OPEN.clear()
    jobs = sorted({(m, str(pd.Timestamp(s).date())) for m, s in zip(need.model, need.start)})
    todo = [j for j in jobs if not raw_path(*j).exists()]
    print(f"[geps] fetch: {len(jobs)} starts needed, {len(todo)} to download, "
          f"{workers} processes", flush=True)
    res = []
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_fetch_job, j) for j in todo]
            for n, f in enumerate(as_completed(futs), 1):
                r = f.result()
                res.append(r)
                print(f"[geps] {n}/{len(todo)} {r['model']} {r['start']} "
                      f"{'ok' if r['ok'] else 'FAIL ' + r.get('error', '')} "
                      f"{r.get('n_full_members', '')} members {r['secs']}s", flush=True)
    log = pd.DataFrame(res)
    if FETCH_LOG.exists() and not log.empty:
        log = pd.concat([pd.read_csv(FETCH_LOG), log], ignore_index=True)
    if not log.empty:
        _atomic_csv(log, FETCH_LOG)
    bad = [j for j in jobs if not raw_path(*j).exists()]
    print(f"[geps] fetch: {len(jobs) - len(bad)}/{len(jobs)} raw starts on disk"
          + (f"; MISSING {bad}" if bad else ""), flush=True)
    return log


# --------------------------------------------------------------------------- #
# Cube assembly (offline)
# --------------------------------------------------------------------------- #
def raw_to_daily(raw: xr.Dataset, start, days) -> xr.DataArray:
    """Pick the leads holding ``days`` and stamp each with 00Z of its UTC day."""
    start = pd.Timestamp(start)
    days = pd.DatetimeIndex(days)
    leads = np.array([lead_of_day(start, d) for d in days])
    have = np.asarray(raw["lead"].values, float)
    miss = [float(x) for x in leads if not np.any(np.isclose(have, x))]
    if miss:
        raise SystemExit(f"[geps] start {start:%Y-%m-%d} has no lead(s) {miss} "
                         f"(max L {have.max()}) for days {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}")
    da = raw[VARIABLE].sel(lead=leads, method="nearest", tolerance=1e-3)
    da = da.assign_coords(time=("lead", days)).swap_dims({"lead": "time"}).drop_vars("lead")
    return da.transpose("member", "time", "lat", "lon")


def regrid(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Bilinear 1 deg -> 0.25 deg. The raw crop carries a 3 deg halo, so nothing extrapolates."""
    if float(da.lat.min()) > lat[0] - 1 or float(da.lat.max()) < lat[-1] + 1 \
            or float(da.lon.min()) > lon[0] - 1 or float(da.lon.max()) < lon[-1] + 1:
        raise SystemExit("[geps] native crop does not cover the target grid with a halo")
    return da.interp(lat=lat, lon=lon, method="linear").astype("float32")


def make_cube(raw: xr.Dataset, *, model: str, start, init, peak, lat, lon,
              extra_attrs: dict | None = None) -> xr.Dataset:
    start, init, peak = (pd.Timestamp(x) for x in (start, init, peak))
    days = cube_days(peak)
    da = raw_to_daily(raw, start, days)
    full = np.isfinite(da.values).all(axis=(1, 2, 3))
    dropped = [int(m) for m, ok in zip(da.member.values, full) if not ok]
    da = da.isel(member=np.where(full)[0])
    if da.sizes["member"] < MIN_MEMBERS:
        raise SystemExit(f"[geps] {model} {start:%Y-%m-%d}: only {da.sizes['member']} "
                         f"complete members (< {MIN_MEMBERS})")
    out = regrid(da, lat, lon)
    if float(out.max()) < 100:
        raise SystemExit(f"[geps] {model} {start:%Y-%m-%d}: t2m looks like Celsius")
    if not np.isfinite(out.values).all():
        raise SystemExit(f"[geps] {model} {start:%Y-%m-%d}: NaN after regrid")
    n = out.sizes["member"]
    lead_d = float((peak - start) / pd.Timedelta(days=1))
    cube = xr.Dataset({"2m_temperature": out.rename(None)})
    cube = cube.assign_coords(
        member=np.asarray(out.member.values).astype(int),
        cycle=("member", np.full(n, np.datetime64(start, "ns"))),
        member_lead_days=("member", np.full(n, lead_d)),
    )
    cube["2m_temperature"].attrs.update(units="K", long_name="2 m temperature, daily mean")
    cube.attrs.update(
        source=f"ECCC GEPS ({model}) extended-range ensemble via SubX",
        dataset_id=f"SubX.ECCC.{model}.forecast.tas",
        url=url(model),
        init=str(init), peak=str(peak), lead_days=lead_d,
        start=str(start), lag_days=float((init - start) / pd.Timedelta(days=1)),
        time_kind="daily_mean", step_h=24,
        time_stamp="00Z of the UTC day; SubX L = k + 0.5 is the mean over UTC day start + k "
                   "(00/06/12/18Z), measured against ERA5 hourly",
        native_grid="1.0 deg regular lat-lon (SubX/IRIDL), regridded bilinear to the 0.25 deg "
                    "acal grid from a 3 deg halo crop",
        archives="IRIDL SubX OPeNDAP (anonymous)",
        members_dropped=",".join(map(str, dropped)) or "none",
        climatology="none stored; score against the ERA5 1990-2019 interval-mean clim "
                    "(mean of 00/06/12/18Z) of each day",
    )
    if extra_attrs:
        cube.attrs.update(extra_attrs)
    return cube


def _write_cube(cube: xr.Dataset, out: Path) -> Path:
    enc = {"2m_temperature": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    enc |= {c: {"_FillValue": None} for c in ("lat", "lon", "member_lead_days") if c in cube.coords}
    _atomic_nc(cube, out, enc)
    return out


def build_case(eid: str, force: bool = False) -> Path:
    out = cube_path(eid)
    if out.exists() and not force:
        return out
    from aires.cfs import target_grid

    row = case_row(eid)
    init, peak = case_times(row)
    s = pick_start(init, read_inventory())
    if s is None:
        raise SystemExit(f"[geps] {eid}: no GEPS start on or before {init:%Y-%m-%d}")
    rp = raw_path(s.model, s.start)
    if not rp.exists():
        raise SystemExit(f"[geps] {eid}: {rp} missing - run --stage fetch first")
    lat, lon = target_grid()
    with xr.open_dataset(rp) as raw:
        cube = make_cube(raw.load(), model=s.model, start=s.start, init=init, peak=peak,
                         lat=lat, lon=lon, extra_attrs=dict(episode_id=eid))
    _write_cube(cube, out)
    print(f"[geps] {eid}: {s.model} start {s.start:%Y-%m-%d} (lag {cube.attrs['lag_days']:.0f} d, "
          f"lead {cube.attrs['lead_days']:.0f} d) {cube.sizes['member']} members -> {out.name}",
          flush=True)
    return out


def hind_case(eid: str, year: int, force: bool = False) -> Path | None:
    out = hind_path(eid, year)
    if out.exists() and not force:
        return out
    from aires.cfs import target_grid

    row = case_row(eid)
    inv = read_inventory()
    plan = {h["year"]: h for h in hind_plan(row, inv)}
    h = plan.get(int(year))
    if h is None or h["start"] is None:
        return None
    rp = raw_path(h["model"], h["start"])
    if not rp.exists():
        raise SystemExit(f"[geps] {eid} {year}: {rp} missing - run --stage fetch first")
    p = pd.Timestamp(h["peak"])
    lat, lon = target_grid()
    with xr.open_dataset(rp) as raw:
        cube = make_cube(raw.load(), model=h["model"], start=h["start"],
                         init=p - pd.Timedelta(days=LEAD_DAYS), peak=p, lat=lat, lon=lon,
                         extra_attrs=dict(episode_id=eid, hind_year=int(year),
                                          case_peak=str(pd.Timestamp(row.peak)),
                                          hind_rule="GEPS start nearest (case start shifted "
                                                    "to this year); tie -> earlier",
                                          hind_shift_days=int(h["shift_d"])))
    _write_cube(cube, out)
    return out


def _hind_job(job: tuple[str, int, bool]) -> dict:
    eid, y, force = job
    try:
        p = hind_case(eid, y, force=force)
        return dict(episode_id=eid, year=y, ok=p is not None, path=str(p) if p else "")
    except BaseException as e:                       # noqa: BLE001 - reported
        return dict(episode_id=eid, year=y, ok=False, error=str(e)[:300])


def hind(cases: list[str] | None = None, workers: int = 8, force: bool = False) -> pd.DataFrame:
    inv = read_inventory()
    df = episodes()
    if cases:
        df = df[df.episode_id.isin(cases)]
    jobs, skipped = [], []
    for r in df.itertuples():
        for h in hind_plan(r, inv):
            (jobs if h["start"] is not None else skipped).append((r.episode_id, h["year"]))
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_hind_job, (*j, force)) for j in jobs]):
            res.append(f.result())
    res = pd.DataFrame(res)
    for eid, y in skipped:
        print(f"[geps] hind {eid} {y}: no GEPS start within {HIND_MAX_SHIFT_D} d - skipped")
    bad = res[~res.ok] if not res.empty else res
    for r in bad.itertuples():
        print(f"[geps] hind {r.episode_id} {r.year} FAILED: {getattr(r, 'error', '')}")
    n = res[res.ok].groupby("episode_id").size() if not res.empty else pd.Series(dtype=int)
    short = sorted(set(df.episode_id) - set(n[n >= MIN_HIND_YEARS].index))
    print(f"[geps] hind: {int(res.ok.sum()) if not res.empty else 0}/{len(jobs)} cubes; "
          f"cases with < {MIN_HIND_YEARS} years: {short or 'none'}", flush=True)
    return res


# --------------------------------------------------------------------------- #
# Sanity table (the scorers live in acal/s2sbase.py; this is a build check only)
# --------------------------------------------------------------------------- #
def interval_clim(days, lat=None, lon=None) -> xr.DataArray:
    """ERA5 1990-2019 climatology averaged over 00/06/12/18Z of each UTC day."""
    import gencast_s2s.data as D

    days = pd.DatetimeIndex(days)
    parts = []
    for h in (0, 6, 12, 18):
        c = D.clim_for(days + pd.Timedelta(hours=h))
        c = c.drop_vars([k for k in ("hour", "dayofyear") if k in c.coords])
        parts.append(c.assign_coords(time=days))
    return (sum(parts) / 4.0).transpose("time", "lat", "lon")


def member_al(cube: xr.Dataset, peak) -> np.ndarray:
    """Per-member CONUS A_L on days peak-6..peak-1 against the interval-mean clim."""
    from aires import aindex as AI

    days = window_days(peak)
    t = cube["2m_temperature"].sel(time=days)
    if t.sizes["time"] != WINDOW_DAYS:
        raise SystemExit("[geps] cube does not cover days peak-6..peak-1")
    clim = interval_clim(days)
    clim = clim.sel(lat=t.lat, lon=t.lon, method="nearest", tolerance=1e-3)
    clim = clim.assign_coords(lat=t.lat, lon=t.lon)
    anom = (t - clim).mean("time")
    return np.asarray(AI.area_mean(anom).values, float)


def era5_al12(peak) -> float:
    """ERA5 A_L on the 12 frames peak-6d 00Z .. peak-1d 12Z (the daily-source truth)."""
    from aires import aindex as AI

    peak = pd.Timestamp(peak).normalize()
    fr = pd.date_range(peak - pd.Timedelta(days=WINDOW_DAYS), periods=2 * WINDOW_DAYS,
                       freq="12h")
    # each yearly file also carries the tail of the previous December (2021 starts 2020-12-26)
    files = [ccfg.ACAL_ROOT / "index" / f"era5_t2m_anom_12h_{y}.nc"
             for y in range(fr[0].year, fr[-1].year + 2)]
    da = xr.concat([xr.open_dataset(f)["t2m_anom"] for f in files if f.exists()], "time")
    da = da.sel(time=~da.indexes["time"].duplicated()).sel(time=fr)
    return float(AI.area_mean(da.mean("time")).values)


def hind_check(eid: str) -> pd.DataFrame:
    """Member-mean A_L minus ERA5 (12 frames) for every hind year of one case."""
    rows = []
    for f in sorted(HIND.glob(f"{eid}_*.nc")):
        with xr.open_dataset(f) as c:
            p = pd.Timestamp(c.attrs["peak"])
            fc = float(member_al(c, p).mean())
            rows.append(dict(episode_id=eid, year=int(c.attrs["hind_year"]),
                             model=c.attrs["dataset_id"].split(".")[2], start=c.attrs["start"][:10],
                             shift_d=int(c.attrs["hind_shift_days"]), lead_d=c.attrs["lead_days"],
                             fc=round(fc, 3), obs12=round(era5_al12(p), 3)))
    d = pd.DataFrame(rows)
    if not d.empty:
        d["bias"] = (d.fc - d.obs12).round(3)
    return d


def table(cases: list[str] | None = None) -> pd.DataFrame:
    """Build check: per-case GEPS A_L (raw and LOYO-corrected) next to the ERA5 truth.

    ``obs13`` is the catalog A_L (13 frames); ``obs12`` the 12-frame daily-source truth;
    ``bias_loyo`` the mean of the hind years' member-mean minus ERA5. Writes ``sanity.csv``
    and ``hind_check.csv``. The board's scoring lives in ``acal/s2sbase.py``, not here."""
    df = episodes()
    if cases:
        df = df[df.episode_id.isin(cases)]
    rows, hinds = [], []
    for r in df.itertuples():
        p = cube_path(r.episode_id)
        rec = dict(episode_id=r.episode_id, family=r.family, peak=r.peak,
                   obs13=float(r.a_l_conus))
        if not p.exists():
            rows.append(rec | dict(status="no cube"))
            continue
        _, peak = case_times(r)
        with xr.open_dataset(p) as cube:
            al = member_al(cube, peak)
            a = cube.attrs
        o12 = era5_al12(peak)
        h = hind_check(r.episode_id)
        hinds.append(h)
        b = float(h.bias.mean()) if not h.empty else float("nan")
        sgn = 1.0 if r.family == "heat" else -1.0
        rec |= dict(status="ok", model=a["dataset_id"].split(".")[2], start=a["start"][:10],
                    lag_d=a["lag_days"], lead_d=a["lead_days"], n_members=int(al.size),
                    obs12=round(o12, 3), al_mean=round(float(al.mean()), 3),
                    al_sd=round(float(al.std(ddof=1)), 3),
                    err=round(float(al.mean()) - o12, 3),
                    n_reach=int(np.sum(sgn * al >= sgn * o12)),
                    n_hind_years=len(h), bias_loyo=round(b, 3),
                    al_corr=round(float(al.mean()) - b, 3),
                    err_corr=round(float(al.mean()) - b - o12, 3))
        rows.append(rec)
    t = pd.DataFrame(rows)
    _atomic_csv(t, SANITY_CSV)
    if hinds:
        _atomic_csv(pd.concat(hinds, ignore_index=True), ROOT / "hind_check.csv")
    return t


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def _atomic_nc(ds: xr.Dataset, out: Path, encoding: dict | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if encoding is None:
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    tmp = out.with_name(f".{out.stem}.tmp.{os.getpid()}.nc")
    try:
        ds.to_netcdf(tmp, encoding=encoding)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_csv(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m acal.s2s_geps", description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("fetch", "build", "hind", "table"))
    ap.add_argument("--case", action="append", default=None,
                    help="episode id (repeatable or comma-separated); default all 42")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true",
                    help="build/hind: rewrite cubes; fetch: re-probe the start inventory only")
    a = ap.parse_args(argv)
    cases = None
    if a.case:
        cases = [c for x in a.case for c in x.split(",") if c]
    if a.stage == "fetch":
        if a.force:
            inventory(force=True)
        fetch(cases, workers=a.workers)
    elif a.stage == "build":
        ids = cases or list(episodes().episode_id)
        fails = []
        for eid in ids:
            try:
                build_case(eid, force=a.force)
            except SystemExit as e:
                fails.append((eid, str(e)))
                print(f"[geps] {eid}: FAILED {e}", flush=True)
        print(f"[geps] build: {len(ids) - len(fails)}/{len(ids)} cubes", flush=True)
        return 1 if fails else 0
    elif a.stage == "hind":
        hind(cases, workers=a.workers, force=a.force)
    else:
        t = table(cases)
        with pd.option_context("display.width", 200, "display.max_rows", 100):
            print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
