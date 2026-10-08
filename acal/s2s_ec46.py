#!/usr/bin/env python
"""ECMWF IFS extended-range ensemble (EC46) as an acal baseline. LOGIN NODE (CPU, internet).

What it is
----------
The operational IFS ENS extended range ("EC46"), as ECMWF issued it at the time, taken
from the ECMWF Data Store (ECDS) S2S archive: dataset ``s2s-forecasts``, origin
``ecmwf``, 2 m temperature DAILY MEAN, 1.5 deg, control + perturbed members. There is no
instantaneous 6-hourly 2t in the S2S archive, so this is a ``daily_mean`` source.

  * Init: the latest EC46 init at or before the AI+RES init (peak - 21 d). EC46 ran
    Mon + Thu 00Z up to 2023-06-26 and daily 00Z from 2023-06-28 (2023-06-27 is absent);
    both rules were checked date by date against the ECDS constraints list for
    2020-11..2026-10. 33 of 42 cases have lag 0; the rest lag 1-3 d (lead 22-24 d).
  * Members: 51 (cf + 50 pf) before Cy48r1 (2023-06-27), 101 (cf + 100 pf) after.
  * Leads fetched: every daily range from the init day to the peak day, so the cube
    holds the whole trajectory and the window days peak-6..peak-1 plus one margin day
    on each side (peak-7 for the daily bias maps, the peak day itself).
  * Crop: ``area=[54, -129, 21, -62]`` (ruling C13), i.e. lat 21..54 x lon 231..297 E on
    the 1.5 deg grid, a 3 deg halo around the 0.25 deg CONUS grid for bilinear regrid.

Bias correction (sensitivity, ruling C12): the S2S reforecasts (``s2s-reforecasts``,
cf + 10 pf, 20 hindcast years ref.year-20..ref.year-1 at the same month/day, ERA5
initial conditions) from the reference date nearest the EC46 init (0-1 d away; ties go
to the earlier date). Reference dates are Mon + Thu up to 2024-11-11 and odd days of
the month from 2024-11-13 (Cy49r1). The bias is lead dependent: the reforecast is read
at the SAME leads as the forecast, and compared with ERA5 on the same days of each
hindcast year. ERA5 comes from the WeatherBench2 GCS zarr for years < 2021 and from the
acal index (``runs/acal/index``) for 2021+; the two are bit-identical where they
overlap (checked on 2021-06-01 00Z and 2022-02-10 12Z). The bias is defined on the
scored quantity of a daily-mean source (ruling C9):

    forecast  mean over UTC days peak-6..peak-1 of (daily mean T - interval-mean clim),
              interval-mean clim = mean of the ERA5 1990-2019 clim at 00/06/12/18Z
    ERA5      mean of the 12 frames peak-6d 00Z .. peak-1d 12Z of (T - clim at that hour)

``runs/acal/s2s/ec46/bias.{nc,csv}`` follow the ``runs/acal/cfs/bias.{nc,csv}`` schema.
``bias_daily`` day k is reforecast UTC day D = peak-7+k minus the ERA5 pair (00Z D, 12Z D)
(``s2sbase.DAILY_PAIR``, ruling C2): the ``s2sbase._daily_bias`` convention for daily-mean
sources, filled at day 0 too because the reforecast carries the margin day peak-7.

Credentials: ``~/.ecdsapirc`` with the two lines ``url: https://ecds.ecmwf.int/api`` and
``key: <token>`` (or the ``ECDS_KEY`` environment variable). The key is read, never
printed. Client: ``cdsapi>=0.7.7`` (installed into my-env 2026-10-07). The user creates
the token once: ECMWF account -> accept the S2S licence on the Download tab of BOTH
``s2s-forecasts`` and ``s2s-reforecasts`` -> token from https://ecds.ecmwf.int/how-to-api.
Both datasets need the one licence ``s2s-licence``. A token without it authenticates, but
every retrieve returns HTTP 403 "required licences not accepted" (seen 2026-10-08).
``--stage fetch`` exits 3 without a token, 4 without the licence (no retries) and 5 when
another fetch holds ``runs/acal/s2s/ec46/.fetch.lock``.

    python -m acal.s2s_ec46 --stage plan                 # runs/acal/s2s/ec46/requests.csv
    python -m acal.s2s_ec46 --stage era5 --workers 8     # ERA5 at the hindcast dates (no token)
    python -m acal.s2s_ec46 --stage fetch --workers 4    # THE command: download, then build,
                                                         #   hind, bias and verify (cache-aware)
    python -m acal.s2s_ec46 --stage build [--case EID]   # cubes from the raw GRIB
    python -m acal.s2s_ec46 --stage hind                 # reforecast cubes per hindcast year
    python -m acal.s2s_ec46 --stage bias                 # bias.nc / bias.csv / hind.csv
    python -m acal.s2s_ec46 --stage verify               # acceptance checks

ECDS takes one init date and ONE forecast type per request (``forecast_type``, ``year``,
``month`` and ``day`` are single-choice in the process schema), so each case is four
requests: forecast cf, forecast pf, reforecast cf (20 years), reforecast pf (20 years).
Raw downloads live only under ``runs/acal/s2s/ec46/raw/``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acal import aprep, ccfg  # noqa: E402
from aires import aindex as AI  # noqa: E402
from aires import cfs  # noqa: E402

NAME = "ec46"
API_URL = "https://ecds.ecmwf.int/api"
FC_DATASET = "s2s-forecasts"
RF_DATASET = "s2s-reforecasts"
SOURCE = dict(
    name=NAME,
    label="ECMWF IFS (EC46)",
    native_deg=1.5,
    time_kind="daily_mean",
    n_members=101,
    init_rule="latest EC46 00Z init <= AI+RES init (Mon/Thu to 2023-06-26, daily from "
              "2023-06-28); 51 members before 2023-06-27, 101 after",
    bias="reforecast",
    dataset_id="ECDS s2s-forecasts origin=ecmwf 2_m_temperature daily mean 1.5 deg",
    url="https://ecds.ecmwf.int/datasets/s2s-forecasts",
)

ROOT = ccfg.ACAL_ROOT / "s2s" / NAME
RAW = ROOT / "raw"
HIND_ROOT = ROOT / "hind"
ERA5_ROOT = ROOT / "era5_hdates"
REQUESTS_CSV = ROOT / "requests.csv"
BIAS_NC = ROOT / "bias.nc"
BIAS_CSV = ROOT / "bias.csv"
HIND_CSV = ROOT / "hind.csv"
TOKEN_FILE = Path(os.environ.get("ECDS_RC", str(Path.home() / ".ecdsapirc")))

AREA = [54, -129, 21, -62]               # N, W, S, E (ruling C13: 2 rows of margin)
HALO = 3.0                               # degrees around the 0.25 deg CONUS grid
CY48R1 = dt.date(2023, 6, 27)            # 101 members and daily inits from here
DAILY_FROM = dt.date(2023, 6, 28)        # 2023-06-27 itself has no 2t in the archive
RF_ODD_DAYS_FROM = dt.date(2024, 11, 13)  # Cy49r1 reference dates: odd days of month
N_HYEARS = 20
RF_MEMBERS = 11                          # cf + 10 pf
FORECAST_TYPES = ("control_forecast", "perturbed_forecast")
WINDOW_DAYS = 6                          # scored UTC days peak-6 .. peak-1
DAILY_DAYS = 7                           # bias_daily days peak-7 .. peak-1
MIN_YEARS = 15                           # fewer reforecast years -> refuse the bias
ERA5_SPLIT = pd.Timestamp("2021-01-01")  # < : WB2 zarr; >= : acal index
CLIM_HOURS = (0, 6, 12, 18)
DAILY_PAIR = "00Z D + 12Z D"           # = s2sbase.DAILY_PAIR (ruling C2; no import cycle)


class TokenMissing(RuntimeError):
    """No ECDS credentials on this machine yet."""


# One licence covers both datasets (ECDS catalogue, 2026-10-08: id s2s-licence, revision 5).
LICENCE_ID = "s2s-licence"
LICENCE_PAGE = f"https://ecds.ecmwf.int/datasets/{FC_DATASET}?tab=download#manage-licences"


class LicenceMissing(RuntimeError):
    """ECDS accepts the token, but the account has not accepted the S2S licence."""


def is_licence_error(e: BaseException) -> bool:
    """True for the HTTP 403 ECDS returns when a dataset licence is not accepted."""
    s = str(e).lower()
    return "licence" in s and "not accepted" in s


# --------------------------------------------------------------------------- #
# Dates: init, reference date, hindcast dates, leads
# --------------------------------------------------------------------------- #
def _date(x) -> dt.date:
    return pd.Timestamp(x).date()


def is_init_date(d) -> bool:
    d = _date(d)
    if d < CY48R1:
        return d.weekday() in (0, 3)         # Mon, Thu
    return d >= DAILY_FROM


def ec46_init(aires_init) -> pd.Timestamp:
    """Latest EC46 init (00Z) at or before the AI+RES init: never a shorter lead."""
    d = _date(aires_init)
    for k in range(8):
        c = d - dt.timedelta(days=k)
        if is_init_date(c):
            return pd.Timestamp(c)
    raise ValueError(f"no EC46 init within 7 d before {d}")


def n_members_for(init) -> int:
    return 51 if _date(init) < CY48R1 else 101


def is_ref_date(d) -> bool:
    d = _date(d)
    if d < RF_ODD_DAYS_FROM:
        return d.weekday() in (0, 3)
    return d.day % 2 == 1


def ref_date(init) -> pd.Timestamp:
    """Reforecast reference date nearest the EC46 init; ties go to the earlier date."""
    d = _date(init)
    for k in range(4):
        for c in (d - dt.timedelta(days=k), d + dt.timedelta(days=k)):
            if is_ref_date(c):
                return pd.Timestamp(c)
    raise ValueError(f"no reforecast reference date within 3 d of {d}")


def hdate(ref, year: int) -> pd.Timestamp:
    """The hindcast start in ``year`` for reference date ``ref`` (Feb 29 -> Feb 28)."""
    r = _date(ref)
    day = 28 if (r.month == 2 and r.day == 29) else r.day
    return pd.Timestamp(dt.date(year, r.month, day))


def hyears(ref) -> list[int]:
    y = _date(ref).year
    return list(range(y - N_HYEARS, y))


def lead_range(d: int) -> str:
    return f"{24 * d}_{24 * (d + 1)}"


def case_row(eid: str):
    df = aprep.episodes()
    r = df[df.episode_id == eid]
    if r.empty:
        raise SystemExit(f"[ec46] no such case: {eid}")
    return next(r.itertuples())


def case_plan(row) -> dict:
    """Everything about one case's requests, derived from the catalog row alone."""
    peak = pd.Timestamp(row.peak)
    ai_init = peak - pd.Timedelta(days=ccfg.LEAD_DAYS)       # the catalog's `init`
    init = ec46_init(ai_init)
    lead = int((peak - init) / pd.Timedelta(days=1))
    ref = ref_date(init)
    fc_days = list(range(0, lead + 1))                       # init day .. peak day
    rf_days = list(range(lead - DAILY_DAYS, lead + 1))       # peak-7 .. peak
    return dict(
        episode_id=row.episode_id, family=row.family, peak=peak, aires_init=ai_init,
        init=init, lag_d=int((ai_init - init) / pd.Timedelta(days=1)), lead_days=lead,
        n_members=n_members_for(init), ref_date=ref,
        ref_offset_d=int((ref - init) / pd.Timedelta(days=1)), hyears=hyears(ref),
        fc_days=fc_days, rf_days=rf_days,
        fc_leads=[lead_range(d) for d in fc_days], rf_leads=[lead_range(d) for d in rf_days],
    )


def plans(cases: list[str] | None = None) -> list[dict]:
    df = aprep.episodes()
    if cases:
        df = df[df.episode_id.isin(cases)]
        if df.empty:
            raise SystemExit(f"[ec46] no such case(s): {cases}")
    return [case_plan(r) for r in df.itertuples()]


# --------------------------------------------------------------------------- #
# Requests and the request plan
# --------------------------------------------------------------------------- #
def _ftag(ftype: str) -> str:
    return "cf" if ftype == "control_forecast" else "pf"


def raw_path(kind: str, plan: dict, ftype: str) -> Path:
    d = plan["init"] if kind == "fc" else plan["ref_date"]
    return RAW / f"{plan['episode_id']}_{kind}_{_ftag(ftype)}_{d:%Y%m%d}.grib"


def request(kind: str, plan: dict, ftype: str) -> tuple[str, dict]:
    """``(dataset, request)`` for one ECDS retrieve. Single-choice fields are strings."""
    d = plan["init"] if kind == "fc" else plan["ref_date"]
    req = {
        "origin": "ecmwf", "year": f"{d:%Y}", "month": f"{d:%m}", "day": f"{d:%d}",
        "time": "00:00", "level_type": "single_level", "variable": ["2_m_temperature"],
        "forecast_type": ftype, "data_format": "grib", "area": list(AREA),
    }
    if kind == "fc":
        req["leadtime_hour"] = list(plan["fc_leads"])
        return FC_DATASET, req
    if kind != "rf":
        raise ValueError(kind)
    hd = [hdate(d, y) for y in plan["hyears"]]
    req["hyear"] = [f"{h:%Y}" for h in hd]
    req["hmonth"] = sorted({f"{h:%m}" for h in hd})
    req["hday"] = sorted({f"{h:%d}" for h in hd})
    req["leadtime_hour"] = list(plan["rf_leads"])
    return RF_DATASET, req


def expected_messages(kind: str, plan: dict, ftype: str) -> int:
    n_lead = len(plan["fc_leads"] if kind == "fc" else plan["rf_leads"])
    if kind == "fc":
        n_mem = 1 if ftype == "control_forecast" else plan["n_members"] - 1
        return n_mem * n_lead
    n_mem = 1 if ftype == "control_forecast" else RF_MEMBERS - 1
    return n_mem * n_lead * len(plan["hyears"])


def write_requests_csv(pl: list[dict], out: Path | None = None) -> Path:
    out = Path(out) if out is not None else REQUESTS_CSV
    rows = []
    for p in pl:
        rows.append(dict(
            episode_id=p["episode_id"], family=p["family"], peak=f"{p['peak']:%Y-%m-%d}",
            aires_init=f"{p['aires_init']:%Y-%m-%d}", ec46_init=f"{p['init']:%Y-%m-%d}",
            weekday=f"{p['init']:%a}", lag_d=p["lag_d"], lead_days=p["lead_days"],
            n_members=p["n_members"],
            fc_leads=f"{p['fc_leads'][0]}..{p['fc_leads'][-1]}", n_fc_leads=len(p["fc_leads"]),
            fc_messages=sum(expected_messages("fc", p, f) for f in FORECAST_TYPES),
            ref_date=f"{p['ref_date']:%Y-%m-%d}", ref_offset_d=p["ref_offset_d"],
            hyears=f"{p['hyears'][0]}..{p['hyears'][-1]}", n_hyears=len(p["hyears"]),
            rf_members=RF_MEMBERS,
            rf_leads=f"{p['rf_leads'][0]}..{p['rf_leads'][-1]}", n_rf_leads=len(p["rf_leads"]),
            rf_messages=sum(expected_messages("rf", p, f) for f in FORECAST_TYPES),
            fc_files=";".join(raw_path("fc", p, f).name for f in FORECAST_TYPES),
            rf_files=";".join(raw_path("rf", p, f).name for f in FORECAST_TYPES),
        ))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp.{os.getpid()}.csv")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"[ec46] wrote {out} ({len(rows)} cases, {4 * len(rows)} ECDS requests)")
    return out


# --------------------------------------------------------------------------- #
# Credentials and download
# --------------------------------------------------------------------------- #
def read_token(path: Path | None = None) -> tuple[str, str]:
    """``(url, key)`` from ``~/.ecdsapirc`` (``url:`` / ``key:`` lines) or ``ECDS_KEY``."""
    path = Path(path) if path is not None else TOKEN_FILE
    url, key = API_URL, None
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.partition(":")
            if not sep:
                continue
            k, v = k.strip().lower(), v.strip()
            if k == "url" and v:
                url = v
            elif k == "key" and v:
                key = v
    if key is None and os.environ.get("ECDS_KEY"):
        key = os.environ["ECDS_KEY"].strip()
    if not key:
        raise TokenMissing(
            f"no ECDS token: {path} is absent (or has no 'key:' line) and ECDS_KEY is unset. "
            f"Create an ECMWF account, accept the S2S licence on the Download tab of "
            f"https://ecds.ecmwf.int/datasets/s2s-forecasts and .../s2s-reforecasts, copy "
            f"the token from https://ecds.ecmwf.int/how-to-api, and write {path} with two "
            f"lines: 'url: {API_URL}' and 'key: <token>'.")
    return url, key


def token_present(path: Path | None = None) -> bool:
    try:
        read_token(path)
        return True
    except TokenMissing:
        return False


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".request.json")


def count_messages(path: Path) -> int:
    import eccodes
    with open(path, "rb") as f:
        return int(eccodes.codes_count_in_file(f))


def is_cached(kind: str, plan: dict, ftype: str) -> bool:
    """Cached = the GRIB exists AND was downloaded for exactly this request."""
    out = raw_path(kind, plan, ftype)
    side = _sidecar(out)
    if not (out.exists() and out.stat().st_size > 0 and side.exists()):
        return False
    try:
        rec = json.loads(side.read_text())
    except ValueError:
        return False
    return rec.get("request") == request(kind, plan, ftype)[1]


_REFUSED = threading.Event()                 # set by the first licence refusal of a fetch


def fetch_one(job, url: str, key: str, attempts: int = 3) -> str:
    kind, plan, ftype = job
    out = raw_path(kind, plan, ftype)
    if is_cached(kind, plan, ftype):
        return f"cached {out.name}"
    if _REFUSED.is_set():
        raise LicenceMissing(f"{out.name}: not requested, ECDS already refused the licence")
    import cdsapi
    ds, req = request(kind, plan, ftype)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    err = None
    for a in range(attempts):
        try:
            t0 = time.time()
            c = cdsapi.Client(url=url, key=key, quiet=True, progress=False)
            c.retrieve(ds, req, str(part))
            n = count_messages(part)
            want = expected_messages(kind, plan, ftype)
            if n == 0:
                raise RuntimeError(f"{out.name}: the download holds no GRIB messages")
            os.replace(part, out)
            _sidecar(out).write_text(json.dumps(dict(
                dataset=ds, request=req, messages=n, expected=want,
                seconds=round(time.time() - t0, 1),
                fetched=pd.Timestamp.now(tz="UTC").isoformat()), indent=1))
            note = "" if n == want else f"  WARNING: {n} messages, expected {want}"
            return f"wrote {out.name} {out.stat().st_size / 1e6:.1f} MB {n} msgs " \
                   f"{time.time() - t0:.0f}s{note}"
        except Exception as e:                       # noqa: BLE001 - retry, then report
            err = e
            part.unlink(missing_ok=True)
            if is_licence_error(e):                  # a retry cannot fix this; stop at once
                _REFUSED.set()
                raise LicenceMissing(
                    f"{ds} refuses the request: the ECDS account has not accepted the "
                    f"'{LICENCE_ID}' licence. Accept it (logged in) at {LICENCE_PAGE}, "
                    f"then rerun. ECDS said: {str(e).strip()[:400]}") from e
            if a + 1 < attempts:
                time.sleep(30 * (a + 1))
    raise RuntimeError(f"{out.name}: {type(err).__name__}: {err}")


def fetch(cases: list[str] | None = None, workers: int = 4,
          kinds: tuple[str, ...] = ("fc", "rf")) -> list:
    pl = plans(cases)
    url, key = read_token()
    jobs = [(k, p, f) for p in pl for k in kinds for f in FORECAST_TYPES]
    todo = [j for j in jobs if not is_cached(*j)]
    _REFUSED.clear()
    print(f"[ec46] fetch: {len(jobs)} requests, {len(todo)} to run, {workers} in flight",
          flush=True)
    fails = []
    with ThreadPoolExecutor(max(1, workers)) as ex:
        fut = {ex.submit(fetch_one, j, url, key): j for j in todo}
        for i, f in enumerate(as_completed(fut), 1):
            try:
                print(f"  [{i}/{len(todo)}] {f.result()}", flush=True)
            except LicenceMissing:                   # every other request would fail too
                for g in fut:
                    g.cancel()
                raise
            except Exception as e:                   # noqa: BLE001 - keep going
                fails.append(fut[f])
                print(f"  [{i}/{len(todo)}] FAILED {e}", flush=True)
    return fails


# --------------------------------------------------------------------------- #
# GRIB decode
# --------------------------------------------------------------------------- #
def _get(h, key, default=None):
    import eccodes
    try:
        return eccodes.codes_get(h, key)
    except Exception:                                # noqa: BLE001 - key absent
        return default


def _geti(h, key, default: int = -1) -> int:
    """Integer value of a GRIB key (``codes_get`` returns code-table abbreviations)."""
    import eccodes
    try:
        return int(eccodes.codes_get_long(h, key))
    except Exception:                                # noqa: BLE001 - key absent
        return default


def _grid(h) -> tuple[np.ndarray, np.ndarray]:
    ni, nj = _geti(h, "Ni"), _geti(h, "Nj")
    la1, la2 = float(_get(h, "latitudeOfFirstGridPointInDegrees")), \
        float(_get(h, "latitudeOfLastGridPointInDegrees"))
    lo1, lo2 = float(_get(h, "longitudeOfFirstGridPointInDegrees")) % 360.0, \
        float(_get(h, "longitudeOfLastGridPointInDegrees")) % 360.0
    if _geti(h, "iScansNegatively", 0):
        raise ValueError("iScansNegatively grids are not supported")
    if lo2 < lo1:
        lo2 += 360.0
    lat = np.linspace(la1, la2, nj)
    lon = np.linspace(lo1, lo2, ni)
    return np.round(lat, 6), np.round(lon, 6)


def read_grib(path: Path) -> pd.DataFrame:
    """One row per 2 m temperature message: init, member, day, grid, values (K).

    Keyed on GRIB2 (discipline, category, number) = (0, 0, 0) at 2 m above ground and
    a 24 h averaging interval, not on the shortName, which eccodes versions spell
    ``2t``, ``avg_2t`` or ``mean2t``. The init is ``dataDate`` (the hindcast start
    for a reforecast, whose reference date is ``modelVersionDate``).
    """
    import eccodes
    rows = []
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                if _geti(h, "edition", 2) != 2:
                    raise ValueError(f"{path.name}: GRIB edition {_geti(h, 'edition')}")
                param = (_geti(h, "discipline"), _geti(h, "parameterCategory"),
                         _geti(h, "parameterNumber"))
                surf = (_geti(h, "typeOfFirstFixedSurface"), _geti(h, "level"))
                if param != (0, 0, 0) or surf != (103, 2):
                    continue
                s0, s1 = _geti(h, "startStep"), _geti(h, "endStep")
                stype = str(_get(h, "stepType", ""))
                if s1 - s0 != 24 or s0 % 24 or stype != "avg":
                    raise ValueError(f"{path.name}: step {s0}-{s1} {stype!r}, want a 24 h "
                                     f"average (stepType avg) starting at 00Z")
                if _geti(h, "dataTime", 0) != 0:
                    raise ValueError(f"{path.name}: dataTime {_geti(h, 'dataTime')}, want 00Z")
                lat, lon = _grid(h)
                vals = np.asarray(eccodes.codes_get_values(h), dtype=np.float64)
                if _geti(h, "bitmapPresent", 0):
                    miss = float(_get(h, "missingValue", 9999))
                    vals = np.where(vals == miss, np.nan, vals)
                vals = vals.reshape(lat.size, lon.size)
                init = pd.Timestamp(str(_geti(h, "dataDate")))
                mvd = _geti(h, "modelVersionDate", 0)
                rows.append(dict(
                    init=init, member=_geti(h, "number", 0),
                    day=init + pd.Timedelta(hours=s0), lead_d=s0 // 24,
                    model_version=pd.Timestamp(str(int(mvd))) if mvd else pd.NaT,
                    lat=lat, lon=lon, values=vals))
            finally:
                eccodes.codes_release(h)
    if not rows:
        raise ValueError(f"{path}: no 2 m temperature daily-mean messages")
    return pd.DataFrame(rows)


def native_cube(df: pd.DataFrame) -> xr.DataArray:
    """``(member, time, lat, lon)`` at the native 1.5 deg grid, lat ascending, lon 0-360.

    Refuses duplicate or missing (member, day) pairs: a hole in the member x day table
    is a partial download, and a NaN-padded cube must not be cached.
    """
    lat, lon = df.lat.iloc[0], df.lon.iloc[0]
    for a, b in zip(df.lat, df.lon):
        if a.shape != lat.shape or b.shape != lon.shape or not (
                np.allclose(a, lat) and np.allclose(b, lon)):
            raise ValueError("messages on different grids")
    if df.duplicated(["member", "day"]).any():
        raise ValueError("duplicate (member, day) messages")
    members = np.sort(df.member.unique())
    days = pd.DatetimeIndex(np.sort(df.day.unique()))
    if len(df) != members.size * days.size:
        raise ValueError(f"incomplete member x day table: {len(df)} messages for "
                         f"{members.size} members x {days.size} days")
    arr = np.full((members.size, days.size, lat.size, lon.size), np.nan, np.float32)
    mi = {m: i for i, m in enumerate(members)}
    di = {d: i for i, d in enumerate(days)}
    for r in df.itertuples():
        arr[mi[r.member], di[r.day]] = r.values
    da = xr.DataArray(arr, dims=("member", "time", "lat", "lon"),
                      coords=dict(member=members.astype(int), time=days.values,
                                  lat=lat, lon=lon), name="2m_temperature")
    return da.sortby("lat").sortby("lon")


# --------------------------------------------------------------------------- #
# Regrid and the canonical cube
# --------------------------------------------------------------------------- #
def regrid(native: xr.DataArray, lat: np.ndarray, lon: np.ndarray,
           halo: float = HALO) -> xr.DataArray:
    """Bilinear 1.5 deg -> 0.25 deg after a CONUS cut with a ``halo`` degree margin."""
    eps = 1e-6
    nl, no = native["lat"].values, native["lon"].values
    if nl[0] > lat[0] - halo + eps or nl[-1] < lat[-1] + halo - eps or \
            no[0] > lon[0] - halo + eps or no[-1] < lon[-1] + halo - eps:
        raise ValueError(f"native grid lat {nl[0]}..{nl[-1]} lon {no[0]}..{no[-1]} does "
                         f"not cover the target plus a {halo} deg halo")
    sub = native.sel(lat=slice(lat[0] - halo - eps, lat[-1] + halo + eps),
                     lon=slice(lon[0] - halo - eps, lon[-1] + halo + eps))
    out = sub.interp(lat=lat, lon=lon, method="linear")
    return out.astype("float32")


def make_cube(t2m: xr.DataArray, *, init, peak, attrs: dict) -> xr.Dataset:
    init, peak = pd.Timestamp(init), pd.Timestamp(peak)
    n = t2m.sizes["member"]
    lead = float((peak - init) / pd.Timedelta(days=1))
    cube = xr.Dataset({"2m_temperature": t2m.astype("float32")})
    cube = cube.assign_coords(
        member=np.arange(n), cycle=("member", np.full(n, init.to_datetime64())),
        member_lead_days=("member", np.full(n, lead)))
    cube.attrs.update(
        source="ECMWF IFS extended-range ensemble (EC46), ECDS S2S archive",
        dataset_id=SOURCE["dataset_id"], url=SOURCE["url"], init=str(init),
        peak=str(peak), lead_days=lead, time_kind="daily_mean", step_h=24,
        native_grid=f"1.5 deg regular lat-lon, ECDS area crop {AREA} (N,W,S,E), bilinear "
                    f"to the 0.25 deg CONUS grid with a {HALO:g} deg halo",
        time_convention="one value per UTC day, stamped 00Z of that day = mean over "
                        "[day 00Z, day+1 00Z)",
        climatology="none stored; score against the ERA5 1990-2019 clim as an "
                    "interval mean (00/06/12/18Z) for daily-mean sources",
        members_note="member 0 = control (cf), 1..N-1 = perturbed (GRIB number)",
        **attrs)
    check_cube(cube, peak)
    return cube


def check_cube(cube: xr.Dataset, peak) -> None:
    """The contract-2 checks for a daily-mean cube."""
    peak = pd.Timestamp(peak)
    t = cube["2m_temperature"]
    if t.dims != ("member", "time", "lat", "lon"):
        raise ValueError(f"dims {t.dims}")
    lat, lon = cube["lat"].values, cube["lon"].values
    if lat.size != 105 or lon.size != 237 or lat[0] > lat[-1]:
        raise ValueError(f"grid {lat.size}x{lon.size}, lat {lat[0]}..{lat[-1]}")
    times = pd.DatetimeIndex(cube["time"].values)
    if (times != times.normalize()).any():
        raise ValueError("daily-mean frames must be stamped 00Z")
    need = pd.date_range(peak - pd.Timedelta(days=WINDOW_DAYS),
                         peak - pd.Timedelta(days=1), freq="D")
    missing = need.difference(times)
    if len(missing):
        raise ValueError(f"cube misses window days {list(missing.strftime('%Y-%m-%d'))}")
    v = t.values
    if np.isnan(v).any():
        raise ValueError("cube contains NaN")
    if float(np.nanmax(v)) < 100:
        raise ValueError("t2m looks like Celsius, not kelvin")


def write_cube(cube: xr.Dataset, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = {"2m_temperature": {"zlib": True, "complevel": 4}}
    tmp = out.with_suffix(f".tmp.{os.getpid()}.nc")
    cube.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, out)
    return out


def cube_path(eid: str) -> Path:
    return ROOT / f"{eid}.nc"


def hind_path(eid: str, year: int) -> Path:
    return HIND_ROOT / f"{eid}_{year}.nc"


def _load_raw(kind: str, plan: dict) -> pd.DataFrame:
    parts = []
    for f in FORECAST_TYPES:
        p = raw_path(kind, plan, f)
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing - run `python -m acal.s2s_ec46 "
                                    f"--stage fetch --case {plan['episode_id']}`")
        parts.append(read_grib(p))
    return pd.concat(parts, ignore_index=True)


def build_case(eid: str, force: bool = False) -> Path:
    out = cube_path(eid)
    if out.exists() and not force:
        return out
    plan = case_plan(case_row(eid))
    df = _load_raw("fc", plan)
    inits = set(df.init)
    if inits != {plan["init"]}:
        raise ValueError(f"{eid}: GRIB inits {sorted(inits)} != planned {plan['init']}")
    nat = native_cube(df)
    lat, lon = cfs.target_grid()
    t2m = regrid(nat, lat, lon)
    n = t2m.sizes["member"]
    note = "" if n == plan["n_members"] else f" (expected {plan['n_members']})"
    cube = make_cube(t2m, init=plan["init"], peak=plan["peak"], attrs=dict(
        archives=f"ECDS {FC_DATASET} (cf + pf requests)", kind="forecast",
        n_members=n, aires_init=str(plan["aires_init"]), lag_days=plan["lag_d"],
        members_expected=plan["n_members"]))
    write_cube(cube, out)
    print(f"[ec46] {eid}: wrote {out.name} members {n}{note}, days "
          f"{pd.Timestamp(cube.time.values[0]):%m-%d}..{pd.Timestamp(cube.time.values[-1]):%m-%d}"
          f", lead {plan['lead_days']} d", flush=True)
    return out


_RF_CACHE: dict = {}


def _rf_native(plan: dict) -> dict:
    """Decode the case's reforecast GRIBs once: ``{year: native (member, time, lat, lon)}``."""
    eid = plan["episode_id"]
    paths = [raw_path("rf", plan, f) for f in FORECAST_TYPES]
    key = (eid, tuple(p.stat().st_mtime if p.exists() else 0 for p in paths))
    if key in _RF_CACHE:
        return _RF_CACHE[key]
    df = _load_raw("rf", plan)
    mvd = df.model_version.dropna().unique()
    if len(mvd) and set(pd.DatetimeIndex(mvd)) != {plan["ref_date"]}:
        raise ValueError(f"{eid}: modelVersionDate {list(mvd)} != ref date {plan['ref_date']}")
    want = {hdate(plan["ref_date"], y): y for y in plan["hyears"]}
    got = set(df.init)
    if not got <= set(want):
        raise ValueError(f"{eid}: reforecast inits {sorted(got)[:3]}... are not the hindcast "
                         f"dates of {plan['ref_date']:%Y-%m-%d}; inspect with grib_ls -p "
                         f"dataDate,modelVersionDate")
    out = {want[i]: native_cube(g) for i, g in df.groupby("init")}
    _RF_CACHE.clear()
    _RF_CACHE[key] = out
    return out


def hind_case(eid: str, year: int, force: bool = False) -> Path | None:
    """Reforecast cube for hindcast ``year`` (same leads as the forecast), or None."""
    plan = case_plan(case_row(eid))
    if year not in plan["hyears"]:
        return None
    out = hind_path(eid, year)
    if out.exists() and not force:
        return out
    nat = _rf_native(plan).get(year)
    if nat is None:
        print(f"[ec46] {eid} {year}: hindcast year absent from the reforecast GRIB")
        return None
    h = hdate(plan["ref_date"], year)
    peak_y = h + pd.Timedelta(days=plan["lead_days"])
    lat, lon = cfs.target_grid()
    cube = make_cube(regrid(nat, lat, lon), init=h, peak=peak_y, attrs=dict(
        archives=f"ECDS {RF_DATASET} (cf + pf requests)", kind="reforecast",
        n_members=int(nat.sizes["member"]), ref_date=str(plan["ref_date"]),
        hdate=str(h), hyear=int(year), case_peak=str(plan["peak"]),
        case_init=str(plan["init"])))
    return write_cube(cube, out)


# --------------------------------------------------------------------------- #
# Daily-mean reduction and ERA5 at the hindcast dates
# --------------------------------------------------------------------------- #
def interval_clim(days) -> xr.DataArray:
    """``(time, lat, lon)``: mean ERA5 1990-2019 clim over 00/06/12/18Z of each UTC day."""
    from gencast_s2s import data as D
    days = pd.DatetimeIndex(days)
    t = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in CLIM_HOURS])
    c = np.asarray(D.clim_for(t).values, dtype=np.float64)
    c = c.reshape(days.size, len(CLIM_HOURS), *c.shape[1:]).mean(axis=1)
    lat, lon = cfs.target_grid()
    return xr.DataArray(c, dims=("time", "lat", "lon"),
                        coords=dict(time=days.values, lat=lat, lon=lon))


def daily_anom(cube: xr.Dataset, days) -> xr.DataArray:
    """``(member, day, lat, lon)`` daily-mean anomaly on the given UTC days (00Z stamps)."""
    days = pd.DatetimeIndex(days)
    t = cube["2m_temperature"].sel(time=days.values).astype("float64")
    a = t - interval_clim(days).values[None]
    return a.rename(time="day").assign_coords(day=np.arange(days.size))


def window_days(peak, n: int = DAILY_DAYS) -> pd.DatetimeIndex:
    """The ``n`` UTC days ending at the peak 00Z (last = peak-1)."""
    peak = pd.Timestamp(peak)
    return pd.date_range(peak - pd.Timedelta(days=n), peak - pd.Timedelta(days=1), freq="D")


def era5_path(eid: str) -> Path:
    return ERA5_ROOT / f"{eid}.nc"


_WB2_T2M = None


def _wb2_t2m() -> xr.DataArray:
    """WB2 ERA5 2 m temperature, lat ascending, opened WITHOUT dask.

    ``gencast_s2s.data._surf`` opens the store with ``chunks={'time': 1}``; slicing that
    93,544-chunk dask array frame by frame grew one process to 4.5 GB RSS, and eight of
    them were killed on the login node. The lazy backend reads only the chunks a
    selection touches, and a multi-time selection lets zarr fetch them concurrently.
    """
    global _WB2_T2M
    if _WB2_T2M is None:
        from gencast_s2s import data as D
        ds = xr.open_zarr(D.SURF_STORE, storage_options={"token": "anon"}, chunks=None)
        if "latitude" in ds.coords:
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        v = ds["2m_temperature"]
        if float(v["lat"][0]) > float(v["lat"][-1]):
            v = v.isel(lat=slice(None, None, -1))
        _WB2_T2M = v
    return _WB2_T2M


def era5_frames(times) -> np.ndarray:
    """``(time, lat, lon)`` ERA5 T2m anomaly (vs the 1990-2019 clim) at 00Z/12Z ``times``."""
    from gencast_s2s import data as D
    times = pd.DatetimeIndex(times)
    lat, lon = cfs.target_grid()
    out = np.full((times.size, lat.size, lon.size), np.nan, np.float32)
    old = np.where(times < ERA5_SPLIT)[0]
    if old.size:
        v = _wb2_t2m().sel(time=times[old].values,
                           lat=slice(lat[0], lat[-1]), lon=slice(lon[0], lon[-1]))
        if v.shape[1:] != (lat.size, lon.size) or not np.allclose(v["lat"], lat) \
                or not np.allclose(v["lon"], lon):
            raise ValueError(f"WB2 CONUS cut is {v.shape[1:]}, not the 105 x 237 grid")
        out[old] = np.asarray(v.values, dtype=np.float64) - D.clim_for(times[old]).values
    new = np.where(times >= ERA5_SPLIT)[0]
    for y in sorted({times[i].year for i in new}):
        idx = [i for i in new if times[i].year == y]
        p = ccfg.ACAL_ROOT / "index" / f"era5_t2m_anom_12h_{y}.nc"
        with xr.open_dataset(p) as d:
            out[idx] = d["t2m_anom"].sel(time=times[idx].values).values
    if np.isnan(out).any():
        raise ValueError("ERA5 frames contain NaN")
    return out


def era5_times(peak_y) -> pd.DatetimeIndex:
    """The 14 ERA5 frames a hindcast year needs: peak-7d 00Z .. peak-1d 12Z, 12-hourly,
    i.e. the (00Z D, 12Z D) pair of each UTC day D = peak-7 .. peak-1."""
    peak_y = pd.Timestamp(peak_y)
    return pd.date_range(peak_y - pd.Timedelta(days=DAILY_DAYS),
                         peak_y - pd.Timedelta(hours=12), freq="12h")


def era5_case(eid: str, force: bool = False) -> str:
    """ERA5 truth for every hindcast year of a case, on both conventions the bias needs.

    ``t2m_anom_12f(year)``: mean of the 12 frames peak_y-6d 00Z .. peak_y-1d 12Z (the
    truth a daily-mean source is scored against). ``t2m_anom_pair(year, day=7)``: day k
    is the pair (00Z D, 12Z D) of UTC day D = peak_y-7+k (``s2sbase.DAILY_PAIR``, ruling
    C2), the truth the daily maps and ``s2sbase._daily_bias`` verify that UTC day on.
    Days 1..6 are the 12f frames, so they average to ``t2m_anom_12f``. A file written
    before ruling C2 (no ``daily_pair`` attr) is rebuilt.
    """
    out = era5_path(eid)
    if out.exists() and not force and _era5_current(out):
        return f"{eid}: cached"
    t0 = time.time()
    plan = case_plan(case_row(eid))
    lat, lon = cfs.target_grid()
    ny = len(plan["hyears"])
    e12 = np.empty((ny, lat.size, lon.size), np.float32)
    pair = np.empty((ny, DAILY_DAYS, lat.size, lon.size), np.float32)
    peaks = []
    for j, y in enumerate(plan["hyears"]):
        peak_y = hdate(plan["ref_date"], y) + pd.Timedelta(days=plan["lead_days"])
        fr = era5_frames(era5_times(peak_y)).astype(np.float64)       # (14, lat, lon)
        e12[j] = fr[2:].mean(axis=0)                     # peak-6d 00Z .. peak-1d 12Z
        pair[j] = fr.reshape(DAILY_DAYS, 2, lat.size, lon.size).mean(axis=1)
        peaks.append(peak_y.to_datetime64())
    ds = xr.Dataset(
        dict(t2m_anom_12f=(("year", "lat", "lon"), e12),
             t2m_anom_pair=(("year", "day", "lat", "lon"), pair),
             peak=(("year",), np.asarray(peaks))),
        coords=dict(year=plan["hyears"], day=np.arange(DAILY_DAYS), lat=lat, lon=lon),
        attrs=dict(case=eid, ref_date=str(plan["ref_date"]), lead_days=plan["lead_days"],
                   daily_pair=DAILY_PAIR,
                   note="ERA5 T2m minus the ERA5 1990-2019 clim. t2m_anom_12f = mean of "
                        "the 12 frames peak-6d 00Z..peak-1d 12Z; t2m_anom_pair day k = "
                        "mean of the frames 00Z and 12Z of UTC day peak-7+k. peak = "
                        "hindcast start + lead_days. WB2 zarr before 2021, acal index "
                        "from 2021."))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp.{os.getpid()}.nc")
    enc = {v: {"zlib": True, "complevel": 4} for v in ("t2m_anom_12f", "t2m_anom_pair")}
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, out)
    return f"{eid}: {ny} years {time.time() - t0:.0f}s"


def _era5_current(path: Path) -> bool:
    """The cached hindcast-date ERA5 carries the ruling-C2 daily pair."""
    with xr.open_dataset(path) as d:
        return d.attrs.get("daily_pair") == DAILY_PAIR


def era5(cases: list[str] | None = None, workers: int = 8, force: bool = False) -> None:
    import multiprocessing as mp
    pl = plans(cases)
    todo = [p["episode_id"] for p in pl if force or not era5_path(p["episode_id"]).exists()
            or not _era5_current(era5_path(p["episode_id"]))]
    print(f"[ec46] era5: {len(pl)} cases, {len(todo)} to build, {workers} workers", flush=True)
    fails = []
    with ProcessPoolExecutor(max(1, workers), mp_context=mp.get_context("spawn")) as ex:
        fut = {ex.submit(era5_case, e, force): e for e in todo}
        for i, f in enumerate(as_completed(fut), 1):
            try:
                print(f"  [{i}/{len(todo)}] {f.result()}", flush=True)
            except Exception as e:                   # noqa: BLE001
                fails.append(fut[f])
                print(f"  [{i}/{len(todo)}] {fut[f]} FAILED {e!r}", flush=True)
    if fails:
        raise SystemExit(f"[ec46] era5 failed for {fails}; rerun to retry")


# --------------------------------------------------------------------------- #
# Bias: reforecast model climate minus ERA5, same leads, same hindcast days
# --------------------------------------------------------------------------- #
def hind_record(eid: str, year: int, era: xr.Dataset) -> dict:
    """One hindcast year: member A_L, ERA5 A_L, the 6-day and the daily difference maps.

    bias7 is on the scored quantity (UTC days peak-6..peak-1 minus the interval-mean
    clim, against the 12-frame ERA5 mean). bias_daily day k is reforecast UTC day
    D = peak-7+k minus the ERA5 pair (00Z D, 12Z D), the convention of
    ``s2sbase._daily_bias`` for daily-mean sources (ruling C2), extended to day 0 because
    the reforecast holds the margin day peak-7.
    """
    with xr.open_dataset(hind_path(eid, year)) as cube:
        cube.load()
    peak_y = pd.Timestamp(cube.attrs["peak"])
    if pd.Timestamp(era["peak"].sel(year=year).values) != peak_y:
        raise ValueError(f"{eid} {year}: ERA5 peak != hindcast peak {peak_y}")
    fd = daily_anom(cube, window_days(peak_y))                   # (member, day=7, ...)
    f6 = fd.isel(day=slice(DAILY_DAYS - WINDOW_DAYS, None)).mean("day")
    e6 = era["t2m_anom_12f"].sel(year=year).astype("float64")
    ep = era["t2m_anom_pair"].sel(year=year).astype("float64")
    al_rf = AI.area_mean(f6).values.astype(float)
    al_era = float(AI.area_mean(e6))
    return dict(year=year, peak=peak_y, al_rf=al_rf, al_era5=al_era,
                bias7=(f6.mean("member") - e6.values),
                bias_daily=(fd.mean("member") - ep.values))


def bias(cases: list[str] | None = None) -> Path:
    df = aprep.episodes()
    if cases:
        raise SystemExit("[ec46] the bias stage always covers the whole slate")
    rows, hrows, b7, bd = [], [], [], []
    for eid in df.episode_id:
        plan = case_plan(case_row(eid))
        yrs = [y for y in plan["hyears"] if hind_path(eid, y).exists()]
        if len(yrs) < MIN_YEARS:
            raise SystemExit(f"[ec46] {eid}: {len(yrs)} reforecast years, need {MIN_YEARS}")
        if not _era5_current(era5_path(eid)):
            raise SystemExit(f"[ec46] {era5_path(eid)} predates the ruling-C2 daily pair: "
                             "rerun --stage era5")
        with xr.open_dataset(era5_path(eid)) as era:
            era.load()
        recs = [hind_record(eid, y, era) for y in yrs]
        d = [float(r["al_rf"].mean() - r["al_era5"]) for r in recs]
        rows.append(dict(episode_id=eid, n_years=len(recs),
                         years=",".join(str(r["year"]) for r in recs),
                         bias_conus=float(np.mean(d)), bias_sd=float(np.std(d, ddof=1)),
                         spread_hind=float(np.mean([r["al_rf"].std(ddof=1) for r in recs]))))
        for r, x in zip(recs, d):
            hd = hdate(plan["ref_date"], r["year"])
            hrows.append(dict(episode_id=eid, year=r["year"], hdate=f"{hd:%Y-%m-%d}",
                              peak=f"{r['peak']:%Y-%m-%d}", n_members=r["al_rf"].size,
                              al_rf_mean=float(r["al_rf"].mean()),
                              al_rf_sd=float(r["al_rf"].std(ddof=1)),
                              al_era5=r["al_era5"], diff=x))
        b7.append(xr.concat([r["bias7"] for r in recs], "year").mean("year").astype("float32"))
        bd.append(xr.concat([r["bias_daily"] for r in recs], "year").mean("year")
                  .astype("float32"))
    tab = pd.DataFrame(rows)
    out = xr.Dataset(
        dict(bias_conus=("case", tab.bias_conus.values),
             bias_sd=("case", tab.bias_sd.values),
             n_years=("case", tab.n_years.values),
             bias7=xr.concat(b7, "case"), bias_daily=xr.concat(bd, "case")),
        coords=dict(case=tab.episode_id.values, family=("case", df.family.values)),
        attrs=dict(note="EC46 reforecast member-mean minus ERA5 at the same hindcast dates "
                        "and leads, averaged over the 20 hindcast years of the reference "
                        "date nearest the EC46 init (model-climate debias, ruling C12). "
                        "Daily-mean convention: forecast = UTC days peak-6..peak-1 minus "
                        "the 00/06/12/18Z interval-mean clim; ERA5 = 12 frames peak-6d 00Z"
                        "..peak-1d 12Z. bias_daily day k = reforecast UTC day D = peak-7+k "
                        "minus the ERA5 pair (00Z D, 12Z D). "
                        "Subtract from EC46 to correct.", source=NAME))
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = BIAS_NC.with_suffix(f".tmp.{os.getpid()}.nc")
    out.to_netcdf(tmp)
    os.replace(tmp, BIAS_NC)
    tab.to_csv(BIAS_CSV, index=False)
    pd.DataFrame(hrows).to_csv(HIND_CSV, index=False)
    for fam, g in tab.merge(df[["episode_id", "family"]]).groupby("family"):
        print(f"[ec46] {fam}: bias_conus mean {g.bias_conus.mean():+.3f} K, range "
              f"[{g.bias_conus.min():+.2f}, {g.bias_conus.max():+.2f}]")
    print(f"[ec46] wrote {BIAS_NC}, {BIAS_CSV}, {HIND_CSV}")
    return BIAS_NC


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def build(cases: list[str] | None = None, force: bool = False) -> list[str]:
    fails = []
    for p in plans(cases):
        try:
            build_case(p["episode_id"], force)
        except Exception as e:                       # noqa: BLE001
            fails.append(p["episode_id"])
            print(f"[ec46] {p['episode_id']}: build FAILED {e}", flush=True)
    return fails


def hind(cases: list[str] | None = None, force: bool = False) -> list[str]:
    fails = []
    for p in plans(cases):
        eid = p["episode_id"]
        try:
            n = sum(hind_case(eid, y, force) is not None for y in p["hyears"])
            print(f"[ec46] {eid}: {n}/{len(p['hyears'])} reforecast years", flush=True)
        except Exception as e:                       # noqa: BLE001
            fails.append(eid)
            print(f"[ec46] {eid}: hind FAILED {e}", flush=True)
    return fails


def verify() -> bool:
    """Acceptance: 42 cubes on contract, 20 hindcast cubes each, bias files complete.

    A cube is on contract when it passes both this module's daily-mean checks and
    ``s2sbase.check_cube`` (contract 2, the scorer's own gate), and its init and lead
    are the planned ones (21-24 d by the init rule).
    """
    from acal import s2sbase as S                    # lazy: s2sbase imports sources lazily
    df = aprep.episodes()
    ok = True
    n_cube = 0
    for eid in df.episode_id:
        p = cube_path(eid)
        if not p.exists():
            print(f"  {eid}: cube MISSING")
            ok = False
            continue
        plan = case_plan(case_row(eid))
        with xr.open_dataset(p) as c:
            try:
                check_cube(c, plan["peak"])
                S.check_cube(c, plan["peak"])
                lead = float(c.attrs["lead_days"])
                if pd.Timestamp(c.attrs["init"]) != plan["init"] or \
                        lead != plan["lead_days"] or not 21 <= lead <= 24:
                    raise ValueError(f"init {c.attrs['init']} lead {lead} d, planned "
                                     f"{plan['init']:%Y-%m-%d} lead {plan['lead_days']} d")
                if c.sizes["member"] != plan["n_members"]:
                    print(f"  {eid}: {c.sizes['member']} members, expected {plan['n_members']}")
                n_cube += 1
            except ValueError as e:
                print(f"  {eid}: cube BAD {e}")
                ok = False
        nh = sum(hind_path(eid, y).exists() for y in plan["hyears"])
        if nh < MIN_YEARS:
            print(f"  {eid}: {nh} reforecast years")
            ok = False
    print(f"[ec46] verify: {n_cube}/{len(df)} cubes on contract")
    if not (BIAS_NC.exists() and BIAS_CSV.exists()):
        print("[ec46] verify: bias files MISSING")
        return False
    with xr.open_dataset(BIAS_NC) as b:
        good = (b.sizes.get("case") == len(df) and list(b["case"].values) == list(df.episode_id)
                and not np.isnan(b["bias7"].values).any())
        print(f"[ec46] verify: bias.nc cases {b.sizes.get('case')}, NaN-free {good}, "
              f"bias_conus mean {float(b.bias_conus.mean()):+.3f} K")
    return ok and good


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["plan", "fetch", "era5", "build", "hind", "bias", "verify"])
    ap.add_argument("--case", action="append", help="episode id (repeatable)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--kind", choices=["fc", "rf", "both"], default="both",
                    help="fetch: forecasts, reforecasts or both")
    ap.add_argument("--no-chain", action="store_true",
                    help="fetch: stop after the downloads (default: build, hind, bias, verify)")
    a = ap.parse_args(argv)
    cases = a.case
    if a.stage == "plan":
        write_requests_csv(plans(None))
        return 0
    if a.stage == "era5":
        era5(cases, workers=a.workers, force=a.force)
        return 0
    if a.stage == "fetch":
        if not REQUESTS_CSV.exists():
            write_requests_csv(plans(None))
        # One fetch at a time: two processes would share the .part files. The lock is
        # released when the process exits, however it exits.
        import fcntl
        lock_path = RAW.parent / ".fetch.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = open(lock_path, "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"[ec46] another fetch holds {lock_path}; not starting a second one",
                  flush=True)
            return 5
        try:
            kinds = ("fc", "rf") if a.kind == "both" else (a.kind,)
            fails = fetch(cases, workers=a.workers, kinds=kinds)
        except TokenMissing as e:
            print(f"[ec46] waiting_token: {e}", flush=True)
            return 3
        except LicenceMissing as e:
            print(f"[ec46] waiting_licence: {e}", flush=True)
            return 4
        if fails:
            print(f"[ec46] {len(fails)} request(s) failed; rerun the same command to resume")
            return 1
        if a.no_chain or a.kind != "both" or cases:
            return 0
        era5(None, workers=8)
        bad = build(None) + hind(None)
        if bad:
            print(f"[ec46] build/hind failed for {sorted(set(bad))}")
            return 1
        bias()
        return 0 if verify() else 1
    if a.stage == "build":
        return 1 if build(cases, a.force) else 0
    if a.stage == "hind":
        return 1 if hind(cases, a.force) else 0
    if a.stage == "bias":
        bias(cases)
        return 0
    if a.stage == "verify":
        return 0 if verify() else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
