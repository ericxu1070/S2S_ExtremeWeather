#!/usr/bin/env python
"""NOAA GEFSv12 35-day baseline for the multi-model acal board. LOGIN NODE (internet, CPU).

The forecast
------------
GEFSv12's 00Z cycle is the only one that runs to 35 days (f840); 06/12/18Z stop at f384.
For each acal case the 00Z cycle OF THE AI+RES INIT DATE is used, so the lead at the peak
is exactly 21 d (lag 0), the same lead the walkers and the newest CFSv2 cycle have. All 31
members (c00 control + p01..p30) are kept, at the native 6-hourly step (f240 onwards is
6-hourly in pgrb2a).

Only `TMP:2 m above ground` is read, one GRIB2 message per (member, forecast hour), as an
HTTP byte range located from the `.idx` sidecar - ~237 KB instead of the ~21 MB file.
The frames kept are peak-7d 00Z .. peak 00Z (f336..f504 at a 21 d lead, 29 frames): the
contract asks for at least [peak-6d 00Z, peak 00Z]; the extra day lets a daily pairing
(the 00Z frame and the 12Z before it, `cfsbase.daily_pairs`) reach day 0 too. The 29
messages of one (init date, member) are concatenated into ONE raw GRIB2 file - the
resumable unit - under `runs/acal/s2s/gefs/raw/<YYYYMMDD>/`.

Decoding and regridding reuse `aires.cfs.grib_to_cube` unchanged (CONUS subset with a
3 deg halo, latitude flipped to ascending, `step` swapped for wall-clock `valid_time`,
bilinear to the 0.25 deg `aires.cfs.target_grid()`, Celsius refused), so a GEFS cube and
a CFS cube come out of the same code path.

Outputs (acal multi-model contract 2)
-------------------------------------
    runs/acal/s2s/gefs/<eid>.nc              2m_temperature(member=31, time=29, lat=105, lon=237)
    runs/acal/s2s/gefs/hind/<eid>_<year>.nc  the same, 11 members (c00 + p01..p10)
    runs/acal/s2s/gefs/sanity.csv            member A_L (25 and 13 frames) vs ERA5 and CFSv2

`member` is the GEFS perturbation number (0 = c00). The cube stores raw kelvin; the
anomaly is taken at reduction time against the ERA5 1990-2019 climatology, as for CFS.

The hindcast (leave-one-year-out bias, sensitivity only)
--------------------------------------------------------
Same 00Z calendar init in every OTHER year whose verification window the ERA5 index
covers - the exact (case, year) list `acal.cfsbase.hind_jobs` gives CFS, so both sources'
corrections are taken over the same years. 11 members are enough for a member-mean bias
(ruling C11); cubes and hind come from the same route, so no product offset enters it.

    python -m acal.s2s_gefs --stage fetch --workers 16      # raw GRIB, 42 cases x 31 members
    python -m acal.s2s_gefs --stage build --procs 8         # the 42 cubes (fetches what is missing)
    python -m acal.s2s_gefs --stage hind --workers 16 --procs 8
    python -m acal.s2s_gefs --stage check                   # sanity.csv
    python -m acal.s2s_gefs --stage build --case e02_c4_20210218
"""
from __future__ import annotations

import argparse
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

NAME = "gefs"
BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
PRODUCT = "atmos/pgrb2ap5"
RECORD = ("TMP", "2 m above ground")
MEMBERS = ("c00",) + tuple(f"p{i:02d}" for i in range(1, 31))
HIND_MEMBERS = MEMBERS[:11]
STEP_H = 6
SPAN_H = 7 * 24                 # frames from peak-7d 00Z to peak 00Z
WINDOW_H = 6 * 24               # the contract's [peak-6d 00Z, peak 00Z]
MIN_MEMBERS = 2

ROOT = ccfg.ACAL_ROOT / "s2s" / NAME
RAW = ROOT / "raw"
HIND_ROOT = ROOT / "hind"
SANITY_CSV = ROOT / "sanity.csv"

SOURCE = dict(
    name=NAME,
    label="NOAA GEFSv12",
    native_deg=0.5,
    time_kind="instant",
    n_members=len(MEMBERS),
    init_rule="00Z cycle of the AI+RES init date (lead 21 d at the peak, lag 0)",
    bias="loyo",
    dataset_id="noaa-gefs-pds GEFSv12 00Z pgrb2ap5 TMP:2 m above ground",
    url=BUCKET,
)


HTTP_TIMEOUT = (10, 60)         # (connect, read) seconds; a stalled socket must not idle 5 min
HTTP_ATTEMPTS = 8
HTTP_BACKOFF = 2.0              # seconds, doubled per retry, capped at 60


class MissingMember(Exception):
    """A (date, member, hour) message is absent from the bucket (404, not retried)."""


_local = threading.local()


def _session():
    import requests
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
        s.headers["User-Agent"] = "acal/s2s_gefs"
    return s


def http_get(url: str, byte_range: str | None = None) -> bytes | None:
    """One GET with keep-alive and short timeouts, retried; ``None`` on a 404.

    `aires.cfs._get` waits 300 s per attempt, which is right for a 95 MB CFS prefix but
    not for ~29 x 2 small requests per member: on the Derecho login node a few sockets
    per minute stall with data stuck in the send queue (seen 2026-10-07, 24 threads,
    throughput fell to ~2 files/min), and every stall then cost 5 min. Here a stall
    costs one 60 s read timeout and a reconnect.
    """
    last = None
    for attempt in range(HTTP_ATTEMPTS):
        try:
            hdr = {"Range": f"bytes={byte_range}"} if byte_range else None
            r = _session().get(url, headers=hdr, timeout=HTTP_TIMEOUT)
            if r.status_code == 404:
                return None
            if r.status_code in (200, 206):
                return r.content
            last = f"HTTP {r.status_code}"
        except Exception as e:                        # noqa: BLE001 - network is network
            last = repr(e)
            _local.s = None                           # drop a possibly wedged pool
        if attempt < HTTP_ATTEMPTS - 1:
            time.sleep(min(60.0, HTTP_BACKOFF * 2 ** attempt))
    raise RuntimeError(f"giving up on {url} after {HTTP_ATTEMPTS} attempts: {last}")


# --------------------------------------------------------------------------- #
# Dates and steps
# --------------------------------------------------------------------------- #
def case_row(eid: str):
    df = aprep.episodes()
    hit = df[df.episode_id == eid]
    if hit.empty:
        raise SystemExit(f"[gefs] no such case: {eid}")
    return next(hit.itertuples())


def case_dates(row) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(init, peak) of a case: init is 00Z of the AI+RES init date, peak - LEAD_DAYS."""
    peak = pd.Timestamp(row.peak)
    init = peak - pd.Timedelta(days=ccfg.LEAD_DAYS)
    if peak != peak.normalize():
        raise SystemExit(f"[gefs] {row.episode_id}: peak {peak} is not 00Z")
    if hasattr(row, "init") and pd.Timestamp(row.init) != init:
        raise SystemExit(f"[gefs] {row.episode_id}: catalog init {row.init} != peak - "
                         f"{ccfg.LEAD_DAYS} d ({init.date()})")
    return init, peak


def fhours(init: pd.Timestamp, peak: pd.Timestamp) -> list[int]:
    """Forecast hours of the frames kept: peak-7d .. peak at the native 6 h step."""
    lead_h = int(round((peak - init) / pd.Timedelta(hours=1)))
    if init != init.normalize():
        raise ValueError(f"GEFS 35-day runs only at 00Z; init {init} is not 00Z")
    if lead_h > 840:
        raise ValueError(f"lead {lead_h} h is beyond the 35-day (f840) horizon")
    lo = lead_h - SPAN_H
    if lo < 240:
        raise ValueError(f"f{lo} is in the 3-hourly range; this module assumes f240+")
    return list(range(lo, lead_h + 1, STEP_H))


def msg_url(init: pd.Timestamp, mem: str, fhr: int) -> str:
    return (f"{BUCKET}/gefs.{init:%Y%m%d}/00/{PRODUCT}/"
            f"ge{mem}.t00z.pgrb2a.0p50.f{fhr:03d}")


def raw_path(init: pd.Timestamp, mem: str) -> Path:
    return RAW / f"{init:%Y%m%d}" / f"ge{mem}.t00z.tmp2m.grib2"


def missing_path(init: pd.Timestamp, mem: str) -> Path:
    return raw_path(init, mem).with_suffix(".missing")


def cube_path(eid: str) -> Path:
    return ROOT / f"{eid}.nc"


def hind_path(eid: str, year: int) -> Path:
    return HIND_ROOT / f"{eid}_{year}.nc"


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def parse_idx(text: str, fhr: int, record: tuple[str, str] = RECORD) -> tuple[int, int | None]:
    """Byte range ``(start, end_inclusive)`` of one record in a GEFS ``.idx``.

    ``end`` is ``None`` when the record is the last message (an open-ended range). The
    record must be the INSTANTANEOUS ``<fhr> hour fcst`` one: ``TMAX``/``TMIN`` carry
    ``498-504 hour max fcst`` and must never match, and neither may a different hour.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for k, ln in enumerate(lines):
        p = ln.split(":")
        if len(p) < 6:
            continue
        if (p[3], p[4]) == record and p[5].strip() == f"{fhr} hour fcst":
            start = int(p[1])
            end = int(lines[k + 1].split(":")[1]) - 1 if k + 1 < len(lines) else None
            if end is not None and end <= start:
                raise ValueError(f"idx offsets not increasing at record {p[0]}")
            return start, end
    raise KeyError(f"{':'.join(record)} at {fhr} hour fcst not in the idx")


def fetch_message(init: pd.Timestamp, mem: str, fhr: int) -> bytes:
    url = msg_url(init, mem, fhr)
    idx = http_get(url + ".idx")
    if idx is None:
        raise MissingMember(f"{url}.idx")
    start, end = parse_idx(idx.decode(), fhr)
    blob = http_get(url, byte_range=f"{start}-{'' if end is None else end}")
    if blob is None:
        raise MissingMember(url)
    if blob[:4] != b"GRIB" or blob[-4:] != b"7777":
        raise ValueError(f"{url} bytes {start}-{end}: not one whole GRIB message")
    return blob


def fetch_member(init: pd.Timestamp, mem: str, fhrs: list[int],
                 force: bool = False) -> str:
    """All ``fhrs`` of one member into one raw GRIB2 file. Resumable per file."""
    dest = raw_path(init, mem)
    if dest.exists() and not force:
        return "cached"
    if missing_path(init, mem).exists() and not force:
        return "missing (marker)"
    try:
        blobs = [fetch_message(init, mem, f) for f in fhrs]
    except MissingMember as e:
        missing_path(init, mem).parent.mkdir(parents=True, exist_ok=True)
        missing_path(init, mem).write_text(json.dumps(dict(missing=str(e))))
        return f"MISSING {e}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f".tmp.{os.getpid()}.{id(blobs)}")
    tmp.write_bytes(b"".join(blobs))
    os.replace(tmp, dest)
    return f"ok {dest.stat().st_size / 1e6:.1f} MB"


def fetch_many(tasks: list[tuple[pd.Timestamp, str, list[int]]], workers: int = 16,
               force: bool = False) -> list[tuple]:
    """Fetch ``(init, member, fhrs)`` tasks with a thread pool; returns the failures."""
    todo = [t for t in dict.fromkeys((t[0], t[1], tuple(t[2])) for t in tasks)
            if force or not (raw_path(t[0], t[1]).exists()
                             or missing_path(t[0], t[1]).exists())]
    print(f"[gefs] fetch: {len(tasks)} member files, {len(todo)} to download, "
          f"{workers} threads", flush=True)
    fails, t0 = [], time.time()

    def one(t):
        try:
            return fetch_member(t[0], t[1], list(t[2]), force)
        except Exception as e:                # noqa: BLE001 - report, keep going
            return f"FAILED {e!r}"

    with ThreadPoolExecutor(workers) as ex:
        fut = {ex.submit(one, t): t for t in todo}
        for k, f in enumerate(as_completed(fut), 1):
            t, msg = fut[f], f.result()
            if msg.startswith(("FAILED", "MISSING")):
                fails.append((t[0], t[1], msg))
            if msg.startswith(("FAILED", "MISSING")) or k % 25 == 0 or k == len(todo):
                print(f"  [{k}/{len(todo)}] {t[0]:%Y-%m-%d} {t[1]}: {msg}  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    return fails


# --------------------------------------------------------------------------- #
# Decode + assemble
# --------------------------------------------------------------------------- #
def member_cube(path: Path, peak: pd.Timestamp, lat: np.ndarray,
                lon: np.ndarray) -> xr.Dataset:
    """One raw member file -> (time, lat, lon) on the target grid, frames peak-7d..peak."""
    c = cfs.grib_to_cube(path, "t2m_anom", lat, lon)
    c = c.drop_vars([v for v in c.coords if v not in ("time", "lat", "lon")])
    c = c.sel(time=slice(peak - pd.Timedelta(hours=SPAN_H), peak))
    want = pd.date_range(peak - pd.Timedelta(hours=SPAN_H), peak, freq=f"{STEP_H}h")
    got = pd.DatetimeIndex(c["time"].values)
    if not got.equals(want):
        raise ValueError(f"{path}: frames {got[0] if len(got) else None}.."
                         f"{got[-1] if len(got) else None} ({len(got)}), want "
                         f"{want[0]}..{want[-1]} ({len(want)})")
    return c


def assemble(init: pd.Timestamp, peak: pd.Timestamp, members: tuple[str, ...],
             attrs: dict) -> xr.Dataset:
    """Stack the members whose raw file exists into the canonical cube."""
    lat, lon = cfs.target_grid()
    parts, skipped = [], []
    lead = (peak - init) / pd.Timedelta(days=1)
    for mem in members:
        p = raw_path(init, mem)
        if not p.exists():
            skipped.append(mem)
            continue
        c = member_cube(p, peak, lat, lon)
        pert = 0 if mem == "c00" else int(mem[1:])
        parts.append(c.expand_dims(member=[pert]).assign_coords(
            cycle=("member", [init.to_datetime64()]),
            member_lead_days=("member", [float(lead)])))
    if len(parts) < MIN_MEMBERS:
        raise RuntimeError(f"only {len(parts)} GEFS member(s) for init {init:%Y-%m-%d}")
    cube = xr.concat(parts, dim="member", coords="minimal", compat="override",
                     join="exact")
    t2m = cube["2m_temperature"]
    if np.isnan(t2m.values).any():
        raise RuntimeError(f"GEFS cube for init {init:%Y-%m-%d} contains NaN")
    if float(t2m.max()) < 100:
        raise RuntimeError("t2m looks like Celsius, not kelvin")
    AI.check_window(cube, peak, "t2m_anom", where=f"GEFS {init:%Y-%m-%d}")
    cube["member"] = cube["member"].astype("int32")
    cube.attrs.update(
        source="NOAA GEFSv12 operational 35-day ensemble, 00Z cycle (pgrb2a 0.5 deg)",
        dataset_id=SOURCE["dataset_id"], url=BUCKET,
        archives=f"AWS noaa-gefs-pds gefs.{init:%Y%m%d}/00/{PRODUCT}",
        init=str(init), peak=str(peak), lead_days=float(lead), time_kind="instant",
        step_h=STEP_H,
        native_grid="0.5 deg regular lat-lon (361 x 720); CONUS subset with a 3 deg halo, "
                    "bilinear to the 0.25 deg acal grid",
        members_requested=",".join(members), members_skipped=",".join(skipped) or "none",
        climatology="none stored; anomalies are taken against ERA5 1990-2019 "
                    "(runs/models/clim_1990_2019_t2m_conus.nc) at reduction time, so they "
                    "carry GEFSv12 model drift, as the CFS and walker cubes do",
        **attrs)
    return cube


def write_cube(cube: xr.Dataset, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = {"2m_temperature": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    tmp = out.with_suffix(f".tmp.{os.getpid()}.nc")
    cube.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, out)
    return out


# --------------------------------------------------------------------------- #
# Contract entry points
# --------------------------------------------------------------------------- #
def case_tasks(eid: str) -> list[tuple[pd.Timestamp, str, list[int]]]:
    init, peak = case_dates(case_row(eid))
    fh = fhours(init, peak)
    return [(init, m, fh) for m in MEMBERS]


def build_case(eid: str, force: bool = False, workers: int = 16) -> Path:
    """Canonical 31-member cube for one case (fetches any missing raw first)."""
    out = cube_path(eid)
    if out.exists() and not force:
        return out
    init, peak = case_dates(case_row(eid))
    fetch_many(case_tasks(eid), workers)
    cube = assemble(init, peak, MEMBERS, dict(event=eid, case=eid))
    write_cube(cube, out)
    print(f"[gefs] {eid}: {cube.sizes['member']} members, {cube.sizes['time']} frames "
          f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out


def hind_jobs(cases: list[str] | None = None):
    """``(eid, year, init, peak)``: CFS's leave-one-year-out list, verbatim."""
    from acal import cfsbase as CB
    return CB.hind_jobs(cases)


def hind_tasks(jobs) -> list[tuple[pd.Timestamp, str, list[int]]]:
    return [(i, m, fhours(i, p)) for _e, _y, i, p in jobs for m in HIND_MEMBERS]


def hind_case(eid: str, year: int, force: bool = False, workers: int = 16) -> Path | None:
    """11-member cube at the same calendar 00Z init in ``year`` (None if not a LOYO year)."""
    jobs = [j for j in hind_jobs([eid]) if j[1] == year]
    if not jobs:
        return None
    _e, _y, init, peak = jobs[0]
    out = hind_path(eid, year)
    if out.exists() and not force:
        return out
    fetch_many(hind_tasks(jobs), workers)
    try:
        cube = assemble(init, peak, HIND_MEMBERS,
                        dict(event=eid, case=eid, hind_year=int(year), role="hind"))
    except RuntimeError as e:
        print(f"[gefs] hind {eid} {year}: {e}", flush=True)
        return None
    write_cube(cube, out)
    return out


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def _cases(case: list[str] | None) -> list[str]:
    ids = list(aprep.episodes().episode_id)
    if case:
        bad = [c for c in case if c not in ids]
        if bad:
            raise SystemExit(f"[gefs] no such case(s): {bad}")
        return case
    return ids


def _run_procs(fn, args: list[tuple], procs: int) -> list[str]:
    fails = []
    with ProcessPoolExecutor(procs) as ex:
        fut = {ex.submit(fn, *a): a for a in args}
        for k, f in enumerate(as_completed(fut), 1):
            a = fut[f]
            try:
                r = f.result()
                print(f"  [{k}/{len(args)}] {a[:2]}: {r}", flush=True)
            except BaseException as e:            # noqa: BLE001 - report, keep going
                fails.append(f"{a[:2]}: {e!r}")
                print(f"  [{k}/{len(args)}] {a[:2]}: FAILED {e!r}", flush=True)
    return fails


def _build_only(eid: str, force: bool) -> str:
    out = build_case(eid, force, workers=1)
    with xr.open_dataset(out) as d:
        return f"{d.sizes['member']} members, skipped {d.attrs.get('members_skipped')}"


def _hind_only(eid: str, year: int, force: bool) -> str:
    out = hind_case(eid, year, force, workers=1)
    if out is None:
        return "no cube"
    with xr.open_dataset(out) as d:
        return f"{d.sizes['member']} members"


def stage_fetch(cases, workers, force=False):
    tasks = [t for e in cases for t in case_tasks(e)]
    fails = fetch_many(tasks, workers, force)
    print(f"[gefs] fetch done: {len(fails)} member file(s) missing/failed", flush=True)
    return fails


def stage_build(cases, workers, procs, force=False):
    stage_fetch(cases, workers)
    todo = [e for e in cases if force or not cube_path(e).exists()]
    print(f"[gefs] build: {len(todo)} of {len(cases)} cubes to write", flush=True)
    fails = _run_procs(_build_only, [(e, force) for e in todo], procs)
    if fails:
        raise SystemExit(f"[gefs] {len(fails)} build(s) failed; rerun to retry")


def stage_hind(cases, workers, procs, force=False):
    jobs = hind_jobs(cases)
    print(f"[gefs] hind: {len(jobs)} (case, year) jobs", flush=True)
    fails = fetch_many(hind_tasks(jobs), workers, force)
    print(f"[gefs] hind fetch done: {len(fails)} member file(s) missing/failed", flush=True)
    todo = [(e, y, force) for e, y, _i, _p in jobs if force or not hind_path(e, y).exists()]
    fails = _run_procs(_hind_only, todo, procs)
    if fails:
        raise SystemExit(f"[gefs] {len(fails)} hind build(s) failed; rerun to retry")


def frames_13(cube: xr.Dataset, peak: pd.Timestamp) -> xr.Dataset:
    """The 13 00Z/12Z frames in [peak-6d 00Z, peak 00Z] (ruling C10)."""
    want = pd.date_range(peak - pd.Timedelta(hours=WINDOW_H), peak, freq="12h")
    return cube.sel(time=want)


def sanity_row(eid: str) -> dict:
    row = case_row(eid)
    init, peak = case_dates(row)
    with xr.open_dataset(cube_path(eid)) as cube:
        cube = cube.load()
    al25 = AI.indices(cube, eid, peak, "t2m_anom", where=f"GEFS {eid}")["conus"].values
    al13 = AI.a_index(frames_13(cube, peak), peak, "t2m_anom",
                      where=f"GEFS {eid} 13f").values
    s = 1.0 if row.family == "heat" else -1.0
    obs = float(row.a_l_conus)
    hy = [y for e, y, _i, _p in hind_jobs([eid]) if hind_path(e, y).exists()]
    return dict(episode_id=eid, family=row.family, obs=obs, n_members=int(al25.size),
                al25_mean=float(al25.mean()), al25_sd=float(al25.std(ddof=1)),
                al13_mean=float(al13.mean()), al13_sd=float(al13.std(ddof=1)),
                d13_25_maxabs=float(np.abs(al13 - al25).max()),
                al25_min=float(al25.min()), al25_max=float(al25.max()),
                n_reach13=int(np.sum(s * al13 >= s * obs)),
                hind_years=",".join(map(str, hy)), n_hind=len(hy))


def stage_check(cases, procs) -> pd.DataFrame:
    ok = [e for e in cases if cube_path(e).exists()]
    with ProcessPoolExecutor(procs) as ex:
        rows = list(ex.map(sanity_row, ok))
    tab = pd.DataFrame(rows)
    cfs_csv = ccfg.ACAL_ROOT / "cfs" / "build.csv"
    if cfs_csv.exists():
        c = pd.read_csv(cfs_csv)[["episode_id", "al_mean", "al_std"]].rename(
            columns={"al_mean": "cfs_al25_mean", "al_std": "cfs_al25_sd"})
        tab = tab.merge(c, on="episode_id", how="left")
    ROOT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(SANITY_CSV, index=False, float_format="%.4f")
    r = np.corrcoef(tab.al25_mean, tab.obs)[0, 1] if len(tab) > 2 else float("nan")
    print(f"[gefs] wrote {SANITY_CSV}: {len(tab)} cases, members "
          f"{tab.n_members.min()}-{tab.n_members.max()}, corr(GEFS mean, obs) {r:+.2f}, "
          f"hind years min {tab.n_hind.min() if len(tab) else 0}", flush=True)
    return tab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("fetch", "build", "hind", "check"))
    ap.add_argument("--case", action="append", help="episode id (repeatable); default all 42")
    ap.add_argument("--workers", type=int, default=16, help="HTTP threads")
    ap.add_argument("--procs", type=int, default=8, help="decode processes")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cases = _cases(a.case)
    if a.stage == "fetch":
        stage_fetch(cases, a.workers, a.force)
    elif a.stage == "build":
        stage_build(cases, a.workers, a.procs, a.force)
    elif a.stage == "hind":
        stage_hind(cases, a.workers, a.procs, a.force)
    else:
        stage_check(cases, a.procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
