#!/usr/bin/env python
"""Multi-model S2S board: the forecast-source registry and the window reducers. LOGIN NODE.

`acal/cfsbase.py` scores ONE baseline (CFSv2) against AI+RES. This module makes the source a
parameter, so CFSv2, GEFSv12, ECCC GEPS and ECMWF EC46 go through the same scorers, on the
same 42 cases, against the same truth (`acal.truth`).

Registry
--------
    cfs     the PUBLISHED CFSv2 adapter: runs/acal/cfs/<eid>_cfs16.{nc,json}, bias.{nc,csv},
            nothing rebuilt or moved. Window 'all' = all 25 six-hourly frames in
            [peak-6d, peak], exactly as `cfsbase` scored it. Parity source.
    cfs13   the same 16 CFS members re-reduced on the 13 00Z/12Z frames (ruling C10): the
            CFSv2 row of the multi-model board. Written under runs/acal/s2s/cfs/; the bias
            files are the published CFS ones (25-frame LOYO, see `Source.bias_note`).
    gefs, geps, ec46
            imported lazily from `acal.s2s_<name>` (contract 1: a `SOURCE` dict, `build_case`,
            `hind_case`). A module that does not exist yet is skipped by `available()`.

Windows (rulings C9, C10)
-------------------------
    instant sources ('13f')   mean of the 13 frames at 00Z/12Z in [peak-6d 00Z, peak 00Z]
                              minus the mean of the climatology at those frame hours
    daily-mean sources ('d6') mean of the daily means of UTC days peak-6 .. peak-1 (stamped
                              00Z of the day) minus the INTERVAL-MEAN climatology of each
                              day (mean of the clim at 00/06/12/18Z). Subtracting the 00Z
                              clim from a 24 h mean would put the diurnal cycle (several K)
                              into the anomaly.
    their truth               '13f' for instant sources; '12f' (peak-6d 00Z .. peak-1d 12Z)
                              for daily-mean sources, and AI+RES is re-reduced on the same
                              12 frames whenever it is paired with one.

`reduce` dispatches on the cube's `time_kind` attr; `reduce_instant` refuses a daily cube
and `reduce_daily` an instantaneous one. A cube without the attr is accepted as instant only
at a sub-daily step (the CFS cubes predate the attr).

Variants per source
-------------------
    raw_emp, raw_gauss          always (raw_emp is the headline, as for CFS)
    corr_emp, corr_gauss        only when the source has bias files
    sub_emp                     CFS only: the last 4 members (the aires convention)
    s16_emp                     sources with N > 16: a FIXED 16-member subsample (indices
                                depend on N only; members are exchangeable), the
                                ensemble-size sensitivity

    python -m acal.s2sbase --source cfs13 --stage json        # cfs13 json + 13f vs 25f shift
    python -m acal.s2sbase --stage aires                      # aires_al_windows.csv
    python -m acal.s2sbase --source gefs --stage hind         # delegates to s2s_gefs.hind_case
    python -m acal.s2sbase --source gefs --stage bias         # LOYO bias.nc / bias.csv
    python -m acal.s2sbase --source cfs --stage score         # parity with cfs_scorecard.csv
    python -m acal.s2sbase --source cfs13 --stage paired --truth era5
    python -m acal.s2sbase --source cfs13 --stage maps

Outputs: runs/acal/s2s/<dir>/ (cubes, json, bias), runs/acal/analysis/s2s/<truth>/
(<source>_scorecard.csv, <source>_paired.csv, s2s_summary.json keyed by source,
maps_{fields,daily}_<source>.nc). Nothing under runs/acal/cfs or the published
runs/acal/analysis files is written.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import aprep, ccfg
from acal import cfsbase as CB
from acal import truth as TR
from aires import aindex as AI

S2S_ROOT = ccfg.ACAL_ROOT / "s2s"
ANALYSIS = AN.OUT / "s2s"
FIG_DIR = ccfg.FIG_ROOT / "s2s"
AIRES_WINDOWS_CSV = ANALYSIS / "aires_al_windows.csv"
MODULE_SOURCES = ("gefs", "geps", "ec46")
BUILTIN = ("cfs", "cfs13")
TIME_KINDS = ("instant", "daily_mean")
GRID = (105, 237)
GRID_TOL = 1e-6
CONSISTENCY_TOL = 1e-3                   # K; json al vs a recomputed field mean
MIN_YEARS = CB.MIN_YEARS
N_FIXED = CB.N_FIXED


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Source:
    name: str
    label: str
    native_deg: float
    time_kind: str                       # 'instant' | 'daily_mean'
    n_members: int | str
    init_rule: str
    bias: str                            # 'loyo' | 'reforecast' | 'none'
    dataset_id: str
    url: str
    module: str | None = None            # 'acal.s2s_gefs'; None = adapter over existing files
    window: str = ""                     # 'all' | '13f' | 'd6'; '' = from time_kind
    dirname: str = ""                    # runs/acal/s2s/<dirname>; '' = name
    bias_note: str = ""

    def __post_init__(self):
        if self.time_kind not in TIME_KINDS:
            raise ValueError(f"{self.name}: time_kind {self.time_kind!r} not in {TIME_KINDS}")
        if not self.window:
            object.__setattr__(self, "window",
                               "13f" if self.time_kind == "instant" else "d6")

    # -- paths ------------------------------------------------------------- #
    @property
    def root(self) -> Path:
        return S2S_ROOT / (self.dirname or self.name)

    @property
    def obs_window(self) -> str:
        """The truth window this source is verified on."""
        return "12f" if self.time_kind == "daily_mean" else "13f"

    def cube_path(self, eid: str) -> Path:
        return CB.cube_path(eid) if self.name in BUILTIN else self.root / f"{eid}.nc"

    def json_path(self, eid: str) -> Path:
        return CB.json_path(eid) if self.name == "cfs" else self.root / f"{eid}.json"

    @property
    def bias_nc(self) -> Path:
        return CB.BIAS_NC if self.name in BUILTIN else self.root / "bias.nc"

    @property
    def bias_csv(self) -> Path:
        return CB.CFS_ROOT / "bias.csv" if self.name in BUILTIN else self.root / "bias.csv"

    def hind_cube_path(self, eid: str, year: int) -> Path:
        return self.root / "hind" / f"{eid}_{year}.nc"

    def hind_red_path(self, eid: str, year: int) -> Path:
        return self.root / "hind_red" / f"{eid}_{year}.nc"

    def has_bias(self) -> bool:
        return self.bias_csv.exists()

    def variants(self, n_members: int | None = None) -> tuple[str, ...]:
        v = ["raw_emp", "raw_gauss"]
        if self.has_bias():
            v += ["corr_emp", "corr_gauss"]
        if self.name in BUILTIN:
            v.append("sub_emp")
        elif n_members is None or n_members > N_FIXED:
            v.append("s16_emp")
        return tuple(v)

    def mod(self):
        if self.module is None:
            raise RuntimeError(f"{self.name} is an adapter over existing files: no module")
        return importlib.import_module(self.module)


CFS_URL = "https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system"
_CFS = dict(label="CFSv2", native_deg=0.94, time_kind="instant", n_members=CB.N_CYCLES,
            init_rule="16 trailing 6-hourly cycles ending on the AI+RES init (21.0-24.75 d)",
            bias="loyo", dataset_id="NCEP CFSv2 operational 6-hourly tmp2m (NCEI/NOMADS)",
            url=CFS_URL)
_BUILTIN = {
    "cfs": Source(name="cfs", window="all", dirname="cfs", **_CFS,
                  bias_note="published 25-frame LOYO bias (runs/acal/cfs/bias.csv)"),
    "cfs13": Source(name="cfs13", window="13f", dirname="cfs", **_CFS,
                    bias_note="published CFS LOYO bias, defined on the 25-frame CFS mean; "
                              "the 13-frame CFS mean differs by a case mean of ~0.03 K, "
                              "which this bias does not remove"),
}
SOURCE_KEYS = ("name", "label", "native_deg", "time_kind", "n_members", "init_rule", "bias",
               "dataset_id", "url")

# Board style shared by every multi-model figure (Okabe-Ito). `label` is the exact board
# label; `deg` the native grid spacing printed as the map-panel caveat. Figure modules
# import this dict (and fall back to the same literal when s2sbase cannot be imported).
MODEL_STYLE = {
    "aires": dict(label="AI+RES", color="#D55E00"),
    "cfs": dict(label="CFSv2", color="#0072B2", deg="~0.94 deg"),
    "cfs13": dict(label="CFSv2", color="#0072B2", deg="~0.94 deg"),
    "ec46": dict(label="ECMWF IFS (EC46)", color="#009E73", deg="1.5 deg"),
    "gefs": dict(label="GEFSv12", color="#CC79A7", deg="0.5 deg"),
    "geps": dict(label="ECCC GEPS", color="#E69F00", deg="1 deg"),
    "bbsubs": dict(label="BB-SUBS (est.)", color="#56B4E9", hatch="//", ls="--",
                   estimate=True),
    "era5": dict(label="ERA5", color="#000000"),
    "hrrr": dict(label="HRRR", color="#555555"),
    "hrrr_raw": dict(label="HRRR", color="#555555"),
    "clim": dict(label="Climatology", color="#999999"),
}
BOARD_SOURCES = ("cfs13", "ec46", "gefs", "geps")       # board order; cfs = 25f reference

# Claim discipline printed on every multi-model board figure (roc.TILT_NOTE and
# reach.TAIL_NOTE carry the same two facts on the per-source figures).
SELECTION_NOTE = "cases selected on outcome (|A_L| >= 2 K)"
TILT_NOTE = ("AI+RES was steered toward the observed tail direction (warm walkers on warm "
             "cases, cold on cold); the baselines were not.")


def style(name: str) -> dict:
    """MODEL_STYLE entry of a source or truth (KeyError for an unknown name)."""
    return MODEL_STYLE[name]


def ens_ranges(name: str) -> dict:
    """Native member count and member lead-to-peak ranges over the source's json records
    (they vary by case: EC46 is 51 or 101 members, GEPS weekly inits lead the peak by
    21-27 d); the registry member count when no json exists yet."""
    src = get(name)
    n, lead = [], []
    for eid in aprep.episodes().episode_id:
        p = src.json_path(eid)
        if p.exists():
            rec = json.loads(p.read_text())
            n.append(int(rec.get("n_members", len(rec.get("al", [])))))
            lead += [float(x) for x in rec.get("member_lead_days", [])]
    if not n:
        n = [int(src.n_members)]
    return dict(n=(min(n), max(n)), lead=(min(lead), max(lead)) if lead else None)


def ens_text(name: str) -> tuple[str, str]:
    """('31 members' or '51-101 members', '21 d' or '21-27 d') for figure footers."""
    r = ens_ranges(name)

    def rng(lo, hi):
        return f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"
    return f"{rng(*r['n'])} members", (rng(*r["lead"]) + " d") if r["lead"] else "n/a"


def shade(color: str, f: float = 0.45) -> str:
    """`color` mixed a fraction `f` toward black: the raw-forecast partner of a source's
    bias-corrected colour (same hue, darker), so the pair reads as one source."""
    c = color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(v * (1 - f)):02x}" for v in rgb)


def has_data(name: str) -> bool:
    """True when source `name` is registered AND has at least one canonical cube (the
    published CFS cubes for cfs/cfs13). A source with code but no data yet (EC46 before
    its download) is absent from the board until its files exist."""
    try:
        src = get(name)
    except KeyError:
        return False
    if name in BUILTIN:
        return CB.cube_path(aprep.episodes().episode_id.iloc[0]).exists()
    return any(src.root.glob("e[0-9][0-9]_*.nc"))


def fig_dir(source: str, truth: str) -> Path:
    """Per-source figure dir: figures/acal/s2s/<truth>/<source>/."""
    return FIG_DIR / truth / source


def get(name: str) -> Source:
    """The registered source `name`; module sources are imported on first use."""
    if name in _BUILTIN:
        return _BUILTIN[name]
    if name not in MODULE_SOURCES:
        raise KeyError(f"unknown source {name!r}; known: {BUILTIN + MODULE_SOURCES}")
    modname = f"acal.s2s_{name}"
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name == modname:
            raise KeyError(f"source {name!r}: {modname} does not exist yet") from e
        raise
    spec = dict(getattr(mod, "SOURCE"))
    missing = [k for k in SOURCE_KEYS if k not in spec]
    if missing:
        raise ValueError(f"{modname}.SOURCE lacks {missing}")
    if spec["name"] != name:
        raise ValueError(f"{modname}.SOURCE name {spec['name']!r} != {name!r}")
    return Source(**{k: spec[k] for k in SOURCE_KEYS}, module=modname)


def available() -> list[str]:
    """Every source that can be used now (built-ins plus the modules that import)."""
    out = list(BUILTIN)
    for n in MODULE_SOURCES:
        try:
            get(n)
            out.append(n)
        except KeyError:
            continue
    return out


# --------------------------------------------------------------------------- #
# Cube contract and reducers
# --------------------------------------------------------------------------- #
def _step_h(cube) -> float:
    t = pd.DatetimeIndex(cube["time"].values).unique().sort_values()
    if t.size < 2:
        raise ValueError(f"need >= 2 time frames to infer the step, got {t.size}")
    return float(np.median(np.diff(t.values).astype("timedelta64[m]").astype(float)) / 60.0)


def time_kind(cube) -> str:
    """The cube's `time_kind` attr; absent, 'instant' at a sub-daily step, else refuse."""
    k = cube.attrs.get("time_kind")
    if k is not None:
        if k not in TIME_KINDS:
            raise ValueError(f"cube time_kind {k!r} not in {TIME_KINDS}")
        return k
    if _step_h(cube) < 24.0:
        return "instant"
    raise ValueError("cube has no time_kind attr and a >= 24 h step: refusing to guess "
                     "whether it holds snapshots or daily means")


def _clim_grid() -> tuple[np.ndarray, np.ndarray]:
    from gencast_s2s import data as D
    c = D._clim_src()
    return c["lat"].values, c["lon"].values


def check_grid(obj) -> None:
    """105 x 237, ascending lat, on the climatology grid (the CLAUDE.md climatology trap)."""
    if (obj.sizes.get("lat"), obj.sizes.get("lon")) != GRID:
        raise ValueError(f"grid {obj.sizes.get('lat')}x{obj.sizes.get('lon')} is not "
                         f"{GRID[0]}x{GRID[1]}")
    lat, lon = _clim_grid()
    if not (np.allclose(obj["lat"].values, lat, atol=GRID_TOL)
            and np.allclose(obj["lon"].values, lon, atol=GRID_TOL)):
        raise ValueError("cube lat/lon != the climatology grid (ascending lat, 235..294 E)")


def check_cube(cube: xr.Dataset, peak=None) -> None:
    """Contract 2 for a canonical source cube; raises on any violation."""
    if "2m_temperature" not in cube:
        raise ValueError("cube lacks 2m_temperature")
    da = cube["2m_temperature"]
    if tuple(da.dims) != ("member", "time", "lat", "lon"):
        raise ValueError(f"2m_temperature dims {da.dims} != (member, time, lat, lon)")
    check_grid(cube)
    for c in ("cycle", "member_lead_days"):
        if c not in cube.coords or cube[c].dims != ("member",):
            raise ValueError(f"cube lacks coord {c}(member)")
    if float(da.max()) < 100.0:
        raise ValueError("2m_temperature max < 100: Celsius, not K")
    k = time_kind(cube)
    if peak is not None:
        peak = pd.Timestamp(peak)
        t = pd.DatetimeIndex(cube["time"].values)
        need = (TR.window_times(peak, "13f") if k == "instant"
                else daily_days(peak))
        if not need.isin(t).all():
            raise ValueError(f"cube time {t[0]}..{t[-1]} does not cover the {k} window "
                             f"for peak {peak}")


def daily_days(peak) -> pd.DatetimeIndex:
    """UTC days peak-6 .. peak-1, each stamped 00Z (ruling C9)."""
    peak = pd.Timestamp(peak)
    return pd.date_range(peak - pd.Timedelta(days=6), peak - pd.Timedelta(days=1), freq="D")


# The 12-hourly pair a daily-mean source's UTC day D is verified on (ruling C2): a
# 00/06/12/18Z daily mean weights the instantaneous frames ~0.375 f00 + 0.5 f12 + 0.125
# f24, so (00Z D, 12Z D) is the closest 00/12Z pair; the six pairs of `daily_days` are
# exactly the '12f' frames, so they average to the 7-day truth of ruling C9.
DAILY_PAIR = "00Z D + 12Z D"


def utc_pairs(inst: xr.DataArray, days) -> xr.DataArray:
    """(day, ..., lat, lon) mean of the 00Z and the 12Z frame of each UTC day in `days`
    (00Z stamps) of a 12-hourly field: the `DAILY_PAIR` convention. Every frame must
    exist (`.sel` raises otherwise)."""
    days = pd.DatetimeIndex(days)
    if (days != days.normalize()).any():
        raise ValueError("utc_pairs: days must be stamped 00Z")
    d = [inst.sel(time=[t, t + pd.Timedelta(hours=12)]).mean("time") for t in days]
    return xr.concat(d, dim="day").assign_coords(day=np.arange(days.size))


def clim_at(times) -> xr.DataArray:
    """1990-2019 clim at `times` (time, lat, lon), through `gencast_s2s.data.clim_for`."""
    return AI.clim_on(None, times)


def interval_clim(days) -> xr.DataArray:
    """(time=day, lat, lon) mean of the clim at 00/06/12/18Z of each UTC day."""
    days = pd.DatetimeIndex(days)
    hrs = [days + pd.Timedelta(hours=h) for h in (0, 6, 12, 18)]
    c = sum(clim_at(h).values for h in hrs) / 4.0
    ref = clim_at(days)
    return ref.copy(data=c)


def _on_clim_grid(da: xr.DataArray, clim: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """`(da, clim)` with the clim carrying `da`'s exact lat/lon (both checked against the
    climatology grid first), so arithmetic never inner-joins and the cube keeps its own
    (float64) coordinates - the cos-lat weights of the published reduction."""
    check_grid(da)
    check_grid(clim)
    return da, clim.assign_coords(lat=da["lat"].values, lon=da["lon"].values)


def reduce_instant(cube: xr.Dataset, peak, window: str = "13f",
                   mask: xr.DataArray | None = None) -> xr.DataArray:
    """(member, lat, lon) window-mean T2m anomaly of an INSTANTANEOUS cube.

    '13f' / '12f' select the 00Z/12Z frames of `truth.window_times` and require every one;
    'all' is every frame in [peak-6d, peak] through `aires.aindex.field` (the published CFS
    reduction, kept for parity).
    """
    if time_kind(cube) != "instant":
        raise ValueError("reduce_instant got a daily_mean cube; use reduce_daily")
    cube = AI._squeeze(cube)
    if window == "all":
        AI.check_window(cube, peak, "t2m_anom")
        f = AI._squeeze(AI.field(cube, peak, "t2m_anom"))
        return TR.apply_mask(f, mask)
    want = TR.window_times(peak, window)
    t = pd.DatetimeIndex(cube["time"].values)
    if not t.is_unique:
        raise ValueError("cube time axis has duplicates")
    miss = want[~want.isin(t)]
    if len(miss):
        raise ValueError(f"cube lacks {len(miss)} of the {len(want)} {window} frames "
                         f"(first missing {miss[0]})")
    T, clim = _on_clim_grid(cube["2m_temperature"].sel(time=want), clim_at(want))
    f = T.mean("time") - clim.mean("time")
    return TR.apply_mask(f, mask)


def reduce_daily(cube: xr.Dataset, peak, mask: xr.DataArray | None = None,
                 per_day: bool = False) -> xr.DataArray:
    """(member, lat, lon) mean over UTC days peak-6..peak-1 of a DAILY-MEAN cube minus the
    interval-mean clim; `per_day=True` returns (member, time=day, lat, lon) unaveraged."""
    if time_kind(cube) != "daily_mean":
        raise ValueError("reduce_daily got an instantaneous cube; use reduce_instant")
    t = pd.DatetimeIndex(cube["time"].values)
    if (t != t.normalize()).any():
        raise ValueError("daily_mean cube times must be stamped 00Z of their UTC day")
    days = daily_days(peak)
    miss = days[~days.isin(t)]
    if len(miss):
        raise ValueError(f"daily cube lacks {len(miss)} of days peak-6..peak-1 "
                         f"(first missing {miss[0]:%Y-%m-%d})")
    T, clim = _on_clim_grid(cube["2m_temperature"].sel(time=days), interval_clim(days))
    a = T - clim
    return TR.apply_mask(a if per_day else a.mean("time"), mask)


def reduce(cube: xr.Dataset, peak, window: str | None = None,
           mask: xr.DataArray | None = None) -> xr.DataArray:
    """Dispatch on `time_kind`: instant -> `reduce_instant(window or '13f')`, daily -> 'd6'."""
    if time_kind(cube) == "instant":
        if window == "d6":
            raise ValueError("window 'd6' is for daily_mean cubes")
        return reduce_instant(cube, peak, window or "13f", mask)
    if window not in (None, "d6"):
        raise ValueError(f"daily_mean cube: window must be 'd6', got {window!r}")
    return reduce_daily(cube, peak, mask)


def member_al(cube: xr.Dataset, peak, window: str | None = None,
              mask: xr.DataArray | None = None) -> np.ndarray:
    """(member,) cos-lat area mean of the window field over the mask."""
    return np.asarray(TR.area_mean(reduce(cube, peak, window, mask)).values, dtype=float)


def init_anom(cube: xr.Dataset, init) -> float:
    """CONUS anomaly over (init, init + 1 d], the CFS json's drift diagnostic; NaN if the
    cube does not reach back to the init (window-only cubes)."""
    init = pd.Timestamp(init)
    t = pd.DatetimeIndex(cube["time"].values)
    if time_kind(cube) == "instant":
        sel = t[(t > init) & (t <= init + pd.Timedelta(days=1))]
        if not len(sel):
            return float("nan")
        T, clim = _on_clim_grid(cube["2m_temperature"].sel(time=sel), clim_at(sel))
        a = T - clim
    else:
        if init.normalize() not in t:
            return float("nan")
        days = pd.DatetimeIndex([init.normalize()])
        T, clim = _on_clim_grid(cube["2m_temperature"].sel(time=days), interval_clim(days))
        a = T - clim
    return float(TR.area_mean(a.mean("time")).mean())


# --------------------------------------------------------------------------- #
# Stage json - the per-(source, case) record every scorer reads
# --------------------------------------------------------------------------- #
def _atomic_json(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(rec, indent=2))
    os.replace(tmp, p)


def _sub_idx(src: Source, n: int) -> np.ndarray:
    """Indices of the json's `al_sub_mean` members: CFS last 4, others the fixed 16."""
    if src.name in BUILTIN:
        return np.arange(n)[-CB.N_SUBSET:]
    return CB.fixed_subset(n) if n > N_FIXED else np.arange(n)


def case_record(src: Source, row, cube: xr.Dataset, truth: TR.Truth) -> dict:
    """The json record: the CFS json keys plus window, time_kind, obs_window."""
    eid, peak = row.episode_id, pd.Timestamp(row.peak)
    check_cube(cube, peak)
    al = member_al(cube, peak, src.window)
    s = 1.0 if row.family == "heat" else -1.0
    obs = float(row.a_l_conus)
    obs_w = truth.obs_al(eid, src.obs_window)
    o = obs if src.obs_window == "13f" else obs_w           # what this source is scored on
    sub = _sub_idx(src, al.size)
    init = pd.Timestamp(row.init)
    return dict(
        episode_id=eid, family=row.family, obs=obs, sign=s, n_members=int(al.size),
        cycles=[str(pd.Timestamp(c)) for c in np.atleast_1d(cube["cycle"].values)],
        member_lead_days=[float(x) for x in np.atleast_1d(cube["member_lead_days"].values)],
        al=al.tolist(), al_mean=float(al.mean()), al_std=float(al.std(ddof=1)),
        al_sub_mean=float(al[sub].mean()),
        n_reach=int(np.sum(s * al >= s * o)), n_reach_sub=int(np.sum(s * al[sub] >= s * o)),
        init_anom=_nan_none(init_anom(cube, init)),
        cycles_skipped=cube.attrs.get("cycles_skipped", "none"),
        archives=cube.attrs.get("archives", ""),
        source=src.name, window=src.window, time_kind=time_kind(cube),
        obs_window=obs_w, obs_window_name=src.obs_window, truth=truth.name,
        sub_rule=("last 4 members" if src.name in BUILTIN
                  else f"fixed {N_FIXED}-member subset (seed {CB.SUB_SEED})"),
    )


def _nan_none(x: float):
    return None if not np.isfinite(x) else float(x)


def write_json(src: Source, cases: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    if src.name == "cfs":
        raise SystemExit("[s2sbase] cfs is the published adapter: its json already exists")
    truth = TR.get_truth("era5")
    df = _slate(cases)
    rows, wrote = [], False
    for r in df.itertuples():
        p = src.json_path(r.episode_id)
        cp = src.cube_path(r.episode_id)
        # a record is reused only while it is newer than its cube: a rebuilt cube (EC46
        # re-download, a fixed builder) must not leave a stale `al` behind
        if p.exists() and not force and not (cp.exists()
                                             and cp.stat().st_mtime > p.stat().st_mtime):
            rows.append(json.loads(p.read_text()))
            continue
        if not cp.exists():
            print(f"[s2sbase] {src.name} {r.episode_id}: no cube {cp}, skipped", flush=True)
            continue
        with xr.open_dataset(cp) as cube:
            rec = case_record(src, r, cube.load(), truth)
        _atomic_json(p, rec)
        rows.append(rec)
        wrote = True
        print(f"[s2sbase] {src.name} {r.episode_id}: obs {rec['obs']:+.2f} "
              f"(window {rec['obs_window']:+.2f})  mean {rec['al_mean']:+.2f} "
              f"n={rec['n_members']}  reach {rec['n_reach']}", flush=True)
    # the shift is a whole-slate number; rewritten only when a record changed (a rewrite
    # would make every downstream table look stale to the board driver's cache check)
    if (src.name == "cfs13" and not cases
            and (wrote or not any(src.root.glob("shift_13f_vs_25f.*")))):
        cfs13_shift(rows)
    return pd.DataFrame(rows)


def cfs13_shift(rows: list[dict]) -> dict:
    """13-frame minus published 25-frame member A_L, once, for the methods text."""
    out = []
    for rec in rows:
        pub = np.asarray(json.loads(CB.json_path(rec["episode_id"]).read_text())["al"])
        d = np.asarray(rec["al"]) - pub
        out.append(dict(episode_id=rec["episode_id"], n=d.size, mean=float(d.mean()),
                        max_abs=float(np.abs(d).max())))
    tab = pd.DataFrame(out)
    allm = np.concatenate([np.asarray(r["al"]) - np.asarray(json.loads(
        CB.json_path(r["episode_id"]).read_text())["al"]) for r in rows]) if rows else np.r_[0.]
    summ = dict(n_cases=len(rows), n_members=int(allm.size), mean=float(allm.mean()),
                mean_abs=float(np.abs(allm).mean()), max_abs=float(np.abs(allm).max()),
                case_mean_min=float(tab["mean"].min()) if len(tab) else np.nan,
                case_mean_max=float(tab["mean"].max()) if len(tab) else np.nan)
    root = get("cfs13").root
    tab.to_csv(root / "shift_13f_vs_25f.csv", index=False, float_format="%.6g")
    _atomic_json(root / "shift_13f_vs_25f.json", summ)
    print(f"[s2sbase] cfs13 - published 25-frame member A_L over {summ['n_cases']} cases: "
          f"mean {summ['mean']:+.4f} K, mean |d| {summ['mean_abs']:.4f} K, "
          f"max |d| {summ['max_abs']:.4f} K; case means "
          f"{summ['case_mean_min']:+.3f}..{summ['case_mean_max']:+.3f} K")
    return summ


def _slate(cases: list[str] | None = None) -> pd.DataFrame:
    df = aprep.episodes()
    if cases:
        df = df[df.episode_id.isin(cases)]
        if df.empty:
            raise SystemExit(f"[s2sbase] no such case(s): {cases}")
    return df


# --------------------------------------------------------------------------- #
# AI+RES walkers on the other windows
# --------------------------------------------------------------------------- #
def aires_windows_from_series(c: AN.Case) -> pd.DataFrame:
    """Per-walker A_L on '13f' and '12f' from compare.json's per-frame CONUS series.

    `realized.series_conus[w]` is the walker's instantaneous CONUS anomaly at every
    12-hourly frame (lead 0.5 .. 21 d); the A_L is linear in the frames, so the 13-frame
    mean reproduces `realized.conus` (checked to 1e-4 K) and dropping the peak frame gives
    the 12-frame value without re-reading 224 diag files.
    """
    r = c.cmp["realized"]
    ser = np.asarray(r["series_conus"], dtype="float64")
    lead = np.asarray(r["lead_days"], dtype="float64")
    win = lead >= lead[-1] - 6.0 - 1e-9
    if win.sum() != 13 or abs(lead[-1] - ccfg.LEAD_DAYS) > 1e-9:
        raise SystemExit(f"[s2sbase] {c.episode_id}: series has {win.sum()} window frames, "
                         f"last lead {lead[-1]}")
    al13 = ser[:, win].mean(1)
    al12 = ser[:, win][:, :12].mean(1)
    if np.abs(al13 - c.al).max() > 1e-4:
        raise SystemExit(f"[s2sbase] {c.episode_id}: series 13-frame mean != walker A_L")
    w = np.asarray(c.weights, dtype="float64")
    return pd.DataFrame(dict(eid=c.episode_id, walker=np.arange(c.al.size), weight=w,
                             weight_sn=w / w.sum(), al13=c.al, al12=al12))


def walker_frames(c: AN.Case) -> xr.DataArray:
    """(walker, time=13, lat, lon) instantaneous anomaly of every final walker over the
    13f window, from the diag files (the path `maps.walker_fields` uses). For masked
    re-reductions (HRRR): `TR.area_mean(f.sel(time=...).mean('time'), mask)`."""
    from acal import maps as M
    peak = pd.Timestamp(c.run["peak"])
    want = TR.window_times(peak, "13f")
    out = []
    for w, slots in enumerate(c.cmp["realized"]["lineage"]):
        traj = M.walker_traj(c, w, slots, peak)
        inst = AI._squeeze(AI.instantaneous_field(traj, "t2m_anom")).sel(time=want)
        got = float(AI.area_mean(inst.mean("time")))
        if abs(got - c.al[w]) > CONSISTENCY_TOL:
            raise SystemExit(f"[s2sbase] {c.episode_id} w{w:02d}: field mean {got:+.4f} != "
                             f"A_L {c.al[w]:+.4f}")
        out.append(inst.astype("float32"))
    return xr.concat(out, dim="walker")


def _write_windows(tab: pd.DataFrame) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    tmp = AIRES_WINDOWS_CSV.with_suffix(".tmp.csv")
    tab.to_csv(tmp, index=False, float_format="%.9g")
    os.replace(tmp, AIRES_WINDOWS_CSV)


def aires_windows() -> pd.DataFrame:
    _, cases = AN.load_all()
    tab = pd.concat([aires_windows_from_series(c) for c in cases], ignore_index=True)
    if AIRES_WINDOWS_CSV.exists():                 # keep masked columns already computed
        old = pd.read_csv(AIRES_WINDOWS_CSV)
        extra = [k for k in old.columns if k not in tab.columns]
        if extra:
            tab = tab.merge(old[["eid", "walker"] + extra], on=["eid", "walker"], how="left")
    _write_windows(tab)
    d = (tab.al12 - tab.al13).to_numpy()
    print(f"[s2sbase] {tab.eid.nunique()} cases x 32 walkers -> {AIRES_WINDOWS_CSV}; "
          f"al12 - al13 mean {d.mean():+.4f} K, max |d| {np.abs(d).max():.4f} K")
    return tab


def _mask_tag(truth: TR.Truth) -> str:
    return TR.mask_tag(truth)


AIRES_MASK_DIR = ANALYSIS / "aires_mask"


def aires_mask_case(eid: str, truth_name: str, force: bool = False) -> Path:
    """Per-walker A_L of one case on the truth's mask, '13f' and '12f', from the walker
    frames (`walker_frames`, ~224 diag files). Cached per case (resumable)."""
    truth = TR.get_truth(truth_name)
    tag = _mask_tag(truth)
    if not tag:
        raise SystemExit(f"[s2sbase] truth {truth_name} has no mask: nothing to re-reduce")
    p = AIRES_MASK_DIR / f"{eid}{tag}.csv"
    if p.exists() and not force:
        return p
    c = AN.load_case(next(r for r in aprep.episodes().itertuples() if r.episode_id == eid))
    f = walker_frames(c)
    peak, mask = pd.Timestamp(c.run["peak"]), truth.mask()
    al13 = TR.area_mean(f.mean("time"), mask).values.astype("float64")
    al12 = TR.area_mean(f.sel(time=TR.window_times(peak, "12f")).mean("time"),
                        mask).values.astype("float64")
    out = pd.DataFrame({"eid": eid, "walker": np.arange(al13.size),
                        f"al13{tag}": al13, f"al12{tag}": al12})
    AIRES_MASK_DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    out.to_csv(tmp, index=False, float_format="%.9g")
    os.replace(tmp, p)
    print(f"[s2sbase] {eid}: masked walker A_L{tag} 13f mean {al13.mean():+.3f} "
          f"(full {c.al.mean():+.3f})", flush=True)
    return p


def _mask_case_job(args) -> str:
    aires_mask_case(*args)
    return args[0]


AIRES_FIELDS_DIR = ANALYSIS / "aires_fields"


def aires_fields_path(eid: str) -> Path:
    return AIRES_FIELDS_DIR / f"{eid}.nc"


def _case_frames(eid: str) -> tuple[AN.Case, xr.DataArray]:
    c = AN.load_case(next(r for r in aprep.episodes().itertuples() if r.episode_id == eid))
    return c, walker_frames(c)


def aires_fields_case(eid: str, force: bool = False, frames=None) -> Path:
    """(walker, lat, lon) AI+RES 7-day-mean anomaly of one case on the '13f' AND the '12f'
    window (peak-6d 00Z .. peak-1d 12Z), from one pass over the walker frames
    (`walker_frames`, ~224 diag files, ~25 s). 13f is the board's AI+RES field; 12f is
    the one paired with a daily-mean source (ruling C9). CONUS means asserted against
    compare.json (13f, inside `walker_frames`) and aires_al_windows.csv `al12`. Cached."""
    p = aires_fields_path(eid)
    if p.exists() and not force:
        return p
    c, fr = frames if frames is not None else _case_frames(eid)
    peak = pd.Timestamp(c.run["peak"])
    f13 = fr.mean("time")
    f12 = fr.sel(time=TR.window_times(peak, "12f")).mean("time")
    tab = pd.read_csv(AIRES_WINDOWS_CSV)
    want = tab[tab.eid == eid].sort_values("walker").al12.to_numpy("float64")
    got = np.asarray(AI.area_mean(f12), dtype="float64")
    if got.shape != want.shape or np.max(np.abs(got - want)) > CONSISTENCY_TOL:
        raise SystemExit(f"[s2sbase] {eid}: 12f walker field means != aires_al_windows al12")
    ds = xr.Dataset(dict(t2m_anom_13f=f13.astype("float32"), t2m_anom_12f=f12.astype("float32"),
                         weight=("walker", np.asarray(c.weights, dtype="float64"))),
                    attrs=dict(case=eid, peak=str(peak),
                               note="AI+RES final-walker 7-day-mean T2m anomaly (K); weight "
                                    "= importance weight (self-normalize before use)"))
    AIRES_FIELDS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}.nc")
    ds.to_netcdf(tmp)
    os.replace(tmp, p)
    print(f"[s2sbase] {eid}: AI+RES 13f/12f fields -> {p}", flush=True)
    return p


def aires_fields(eid: str, window: str = "13f") -> tuple[xr.DataArray, np.ndarray]:
    """(fields (walker, lat, lon), raw importance weights) of one case on `window`."""
    if window not in ("13f", "12f"):
        raise ValueError(f"window {window!r}: AI+RES fields exist for '13f' / '12f' only")
    with xr.open_dataset(aires_fields_case(eid)) as f:
        f = f.load()
    return f[f"t2m_anom_{window}"], f["weight"].values


AIRES_DAILY_DIR = ANALYSIS / "aires_daily"


def aires_daily_path(eid: str) -> Path:
    return AIRES_DAILY_DIR / f"{eid}.nc"


def aires_daily_case(eid: str, force: bool = False, frames=None) -> Path:
    """(walker, day=6, lat, lon) AI+RES anomaly of the UTC days peak-6..peak-1, each the
    mean of its 00Z and 12Z walker frames (`utc_pairs`, ruling C2): the daily partner of a
    daily-mean source, whose day D is verified on the same pair. The six days average to
    the '12f' field, so their CONUS means are asserted against aires_al_windows.csv
    `al12`. Built from the walker frames (`walker_frames`, ~25 s). Cached."""
    p = aires_daily_path(eid)
    if p.exists() and not force:
        return p
    c, fr = frames if frames is not None else _case_frames(eid)
    peak = pd.Timestamp(c.run["peak"])
    days = daily_days(peak)
    d = utc_pairs(fr, days).transpose("walker", "day", "lat", "lon")
    tab = pd.read_csv(AIRES_WINDOWS_CSV)
    want = tab[tab.eid == eid].sort_values("walker").al12.to_numpy("float64")
    got = np.asarray(AI.area_mean(d.mean("day")), dtype="float64")
    if got.shape != want.shape or np.max(np.abs(got - want)) > CONSISTENCY_TOL:
        raise SystemExit(f"[s2sbase] {eid}: UTC-day walker fields do not average to al12")
    ds = xr.Dataset(dict(t2m_anom_utc=d.astype("float32"),
                         weight=("walker", np.asarray(c.weights, dtype="float64"))),
                    coords=dict(utc_day=("day", days.values)),
                    attrs=dict(case=eid, peak=str(peak), daily_pair=DAILY_PAIR,
                               note="AI+RES final-walker daily T2m anomaly (K) of the UTC "
                                    "days peak-6..peak-1, each the mean of its 00Z and 12Z "
                                    "frames; weight = importance weight (self-normalize "
                                    "before use)"))
    AIRES_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}.nc")
    ds.to_netcdf(tmp, encoding={"t2m_anom_utc": {"zlib": True, "complevel": 4}})
    os.replace(tmp, p)
    print(f"[s2sbase] {eid}: AI+RES UTC-day fields -> {p}", flush=True)
    return p


def aires_daily(eid: str) -> tuple[xr.DataArray, np.ndarray]:
    """((walker, day=6, lat, lon) UTC-day fields, raw importance weights) of one case."""
    with xr.open_dataset(aires_daily_case(eid)) as f:
        f = f.load()
    return f["t2m_anom_utc"], f["weight"].values


def _fields_job(args) -> str:
    """Build whichever of the two AI+RES caches is missing, from one walker-frame pass."""
    eid, force = args
    need = [fn for fn, path in ((aires_fields_case, aires_fields_path),
                                (aires_daily_case, aires_daily_path))
            if force or not path(eid).exists()]
    frames = _case_frames(eid) if need else None
    for fn in need:
        fn(eid, force, frames)
    return eid


def aires_fields_all(workers: int = 8, cases: list[str] | None = None,
                     force: bool = False) -> None:
    """Cache the 13f/12f AI+RES walker fields and the UTC-day daily fields of every case
    (process pool)."""
    from concurrent.futures import ProcessPoolExecutor
    eids = list(_slate(cases).episode_id)
    jobs = [(e, force) for e in eids
            if force or not (aires_fields_path(e).exists() and aires_daily_path(e).exists())]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
            for e in ex.map(_fields_job, jobs):
                print(f"[s2sbase] done {e}", flush=True)
    else:
        for j in jobs:
            _fields_job(j)
    print(f"[s2sbase] aires_fields: {len(jobs)} built, "
          f"{sum(aires_fields_path(e).exists() for e in eids)}/{len(eids)} cached; "
          f"aires_daily {sum(aires_daily_path(e).exists() for e in eids)}/{len(eids)} cached")


def aires_mask(truth_name: str = "hrrr", workers: int = 8, cases: list[str] | None = None,
               force: bool = False) -> pd.DataFrame:
    """Add the truth's masked columns (al13_<tag>, al12_<tag>) to aires_al_windows.csv."""
    from concurrent.futures import ProcessPoolExecutor
    tag = _mask_tag(TR.get_truth(truth_name))
    eids = list(_slate(cases).episode_id)
    jobs = [(e, truth_name, force) for e in eids]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for e in ex.map(_mask_case_job, jobs):
                print(f"[s2sbase] done {e}", flush=True)
    else:
        for j in jobs:
            _mask_case_job(j)
    all_eids = list(aprep.episodes().episode_id)
    parts = [AIRES_MASK_DIR / f"{e}{tag}.csv" for e in all_eids]
    if not all(p.exists() for p in parts):
        n = sum(p.exists() for p in parts)
        print(f"[s2sbase] {n}/{len(parts)} cases masked; table not updated yet")
        return pd.DataFrame()
    m = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    tab = pd.read_csv(AIRES_WINDOWS_CSV)
    tab = tab.drop(columns=[k for k in m.columns if k not in ("eid", "walker")
                            and k in tab.columns])
    tab = tab.merge(m, on=["eid", "walker"], how="left", validate="one_to_one")
    if tab[[f"al13{tag}", f"al12{tag}"]].isna().any().any():
        raise SystemExit("[s2sbase] masked columns incomplete after merge")
    _write_windows(tab)
    d = (tab[f"al13{tag}"] - tab.al13).to_numpy()
    print(f"[s2sbase] {AIRES_WINDOWS_CSV.name} += al13{tag}, al12{tag}; masked - full "
          f"mean {d.mean():+.4f} K, max |d| {np.abs(d).max():.4f} K")
    return tab


def aires_cases(truth: TR.Truth, window: str) -> dict[str, AN.Case]:
    """AI+RES `Case`s whose `al` and `obs` are re-reduced on `window` against `truth`.

    era5 + '13f' returns the published cases unchanged. Otherwise `al` comes from
    aires_al_windows.csv (column al13/al12, plus the truth's mask tag) and `obs` from the
    truth. The importance weights do not depend on the window.
    """
    _, cases = AN.load_all()
    if truth.name == "era5" and window == "13f":
        return {c.episode_id: c for c in cases}
    if not AIRES_WINDOWS_CSV.exists():
        raise SystemExit(f"[s2sbase] missing {AIRES_WINDOWS_CSV}; run --stage aires")
    tab = pd.read_csv(AIRES_WINDOWS_CSV)
    col = f"al{window[:2]}{_mask_tag(truth)}"
    if col not in tab:
        raise SystemExit(f"[s2sbase] {AIRES_WINDOWS_CSV.name} has no column {col}")
    out = {}
    for c in cases:
        g = tab[tab.eid == c.episode_id].sort_values("walker")
        if not np.allclose(g.weight.to_numpy(), c.weights, rtol=1e-8, atol=0):
            raise SystemExit(f"[s2sbase] {c.episode_id}: table weights != compare.json")
        out[c.episode_id] = dataclasses.replace(
            c, al=g[col].to_numpy(dtype="float64"), obs=truth.obs_al(c.episode_id, window))
    return out


# --------------------------------------------------------------------------- #
# Records, score, paired
# --------------------------------------------------------------------------- #
def load(src: Source, truth: TR.Truth) -> pd.DataFrame:
    """`cfsbase.load_cfs` for any source and truth: one row per case, slate order.

    era5 + a 13f/all window keeps the slate's obs (the published numbers); otherwise obs is
    the truth's A_L on the source's window. A masked truth re-reduces the members from the
    cube on the mask.
    """
    if src.name == "cfs" and truth.name == "era5":
        return CB.load_cfs()
    df = aprep.episodes()
    bias = (pd.read_csv(src.bias_csv).set_index("episode_id")
            if src.has_bias() else None)
    mask = truth.mask()
    rows = []
    for r in df.itertuples():
        p = src.json_path(r.episode_id)
        if not p.exists():
            print(f"[s2sbase] {src.name} {r.episode_id}: no json, case left out", flush=True)
            continue
        rec = json.loads(p.read_text())
        want = 1.0 if r.family == "heat" else -1.0
        if rec["sign"] != want or abs(rec["obs"] - r.a_l_conus) > AN.CATALOG_TOL:
            raise SystemExit(f"[s2sbase] {src.name} {r.episode_id}: json sign/obs != slate")
        al = rec["al"]
        if mask is not None:
            with xr.open_dataset(src.cube_path(r.episode_id)) as cube:
                al = member_al(cube.load(), r.peak, src.window, mask).tolist()
        if truth.name == "era5" and src.obs_window == "13f":
            obs = rec["obs"]
        else:
            obs = truth.obs_al(r.episode_id, src.obs_window)
        b = (float(bias.loc[r.episode_id, "bias_conus"])
             if bias is not None and r.episode_id in bias.index else np.nan)
        rows.append(dict(episode_id=r.episode_id, family=r.family, rung=int(r.rung),
                         peak=r.peak, obs=obs, sign=rec["sign"], al=al, bias=b))
    return pd.DataFrame(rows)


def out_dir(truth: TR.Truth) -> Path:
    return ANALYSIS / truth.name


def _nan_corr(out: dict, variants, bias: float) -> dict:
    """A case without a bias gets NaN in every corr_* column (not a silent bias of 0)."""
    if np.isfinite(bias):
        return out
    for k in list(out):
        if any(k.endswith(f"_{v}") or f"_{v}_" in k for v in variants if v.startswith("corr")):
            out[k] = np.nan
    return out


def _res_columns(sc: pd.DataFrame, src: Source, truth: TR.Truth, daily: pd.Series):
    cols = ("p_obs_raw", "p_obs_sn", "lift_obs_raw", "lift_obs_sn", "lift_obs_raw_cons",
            "pit_sn", "n_beyond_obs")
    if truth.name == "era5" and src.obs_window == "13f":
        res = pd.read_csv(AN.SCORE_OUT).set_index("episode_id")
        for c in cols:
            sc[f"res_{c}"] = sc.episode_id.map(res[c])
        if not np.allclose(sc.p_clim_obs, sc.episode_id.map(res.p_clim_obs)):
            raise SystemExit("[s2sbase] P_clim differs from analyze scorecard - pool mismatch")
        return sc
    cases = aires_cases(truth, src.obs_window)
    slate = aprep.episodes().set_index("episode_id")
    recs = {e: AN.score_case(cases[e], slate.loc[e, "rung"], slate.loc[e, "peak"], daily)
            for e in sc.episode_id}
    for c in cols:
        sc[f"res_{c}"] = sc.episode_id.map({e: r[c] for e, r in recs.items()})
    return sc


def score(src: Source, truth: TR.Truth) -> pd.DataFrame:
    rows, daily = load(src, truth), TR.pool_series(truth)
    if rows.empty:
        raise SystemExit(f"[s2sbase] {src.name}: no case records; run --stage json")
    n_max = int(rows.al.map(len).max())
    variants = src.variants(n_max)
    sc = pd.DataFrame([_nan_corr(CB.score_cfs_case(r, daily, variants), variants, r["bias"])
                       for _, r in rows.iterrows()])
    sc = _res_columns(sc, src, truth, daily)
    od = out_dir(truth)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{src.name}_scorecard.csv"
    sc.to_csv(p, index=False, float_format="%.6g")
    summ = dict(n_cases=int(len(sc)), window=src.window, obs_window=src.obs_window,
                truth=truth.name, variants={v: CB._summ(sc, v) for v in variants},
                ai_res={k: AN._q(sc[f"res_{k}"]) for k in
                        ("p_obs_raw", "p_obs_sn", "lift_obs_raw", "lift_obs_sn",
                         "lift_obs_raw_cons", "pit_sn")},
                p_clim_obs=AN._q(sc.p_clim_obs), bias_note=src.bias_note)
    _merge_summary(truth, src.name, "score", summ)
    print(f"[s2sbase] {src.name} vs {truth.name}: {len(sc)} cases -> {p}")
    for v in variants:
        q = summ["variants"][v]["overall"]["p_obs"]
        print(f"  {v:10s} P(obs) median {q['median']:.3f} [{q['q25']:.3f}, {q['q75']:.3f}]  "
              f"P=0: {summ['variants'][v]['n_zero_obs']}")
    return sc


def paired(src: Source, truth: TR.Truth) -> pd.DataFrame:
    rows = load(src, truth)
    if rows.empty:
        raise SystemExit(f"[s2sbase] {src.name}: no case records; run --stage json")
    variants = src.variants(int(rows.al.map(len).max()))
    cases = aires_cases(truth, src.obs_window)
    pc = pd.DataFrame([_nan_corr(CB.paired_case(r, cases[r["episode_id"]], variants),
                                 variants, r["bias"]) for _, r in rows.iterrows()])
    od = out_dir(truth)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{src.name}_paired.csv"
    pc.to_csv(p, index=False, float_format="%.6g")
    groups = {"all": pc, "heat": pc[pc.family == "heat"], "cold": pc[pc.family == "cold"]}
    summ = {}
    for v in variants:
        summ[v] = {}
        for name, d in groups.items():
            summ[v][name] = {"logratio": CB.paired_stats(d[f"logratio_{v}"])}
            for k in CB.BRIER_K:
                summ[v][name][f"brier_{k:g}K"] = CB.paired_stats(d[f"dbrier_{v}_{k:g}K"])
    _merge_summary(truth, src.name, "paired", dict(window=src.obs_window, variants=summ))
    print(f"[s2sbase] {src.name} vs {truth.name}: {len(pc)} paired -> {p}")
    for v in variants:
        st = summ[v]["all"]["logratio"]
        print(f"  {v:10s} logR {st['mean']:+.3f} W/T/L {st['win']}/{st['tie']}/{st['loss']} "
              f"p={st['wilcoxon_p']:.3f}")
    return pc


def _merge_summary(truth: TR.Truth, source: str, key: str, val: dict) -> None:
    p = out_dir(truth) / "s2s_summary.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.setdefault(source, {})[key] = val
    _atomic_json(p, old)


# --------------------------------------------------------------------------- #
# Bias - leave-one-year-out from hind cubes (sources with bias 'loyo')
# --------------------------------------------------------------------------- #
def era5_on(peak, window: str) -> xr.DataArray:
    """ERA5 anomaly (time, lat, lon) for an arbitrary peak (hind years), 12-hourly."""
    w = CB.era5_window(pd.Timestamp(peak))                    # 14 frames [peak-6.5d, peak]
    return w.sel(time=TR.window_times(peak, window))


def _daily_bias(src: Source, cube: xr.Dataset, peak) -> np.ndarray:
    """(day=7, lat, lon) member-mean minus ERA5 on the `maps.daily` day convention.

    Instant: the CFS pairing (00Z frame + the 12Z before it, day 6 ends at the peak).
    Daily mean: UTC day D = peak-7+k fills maps day k (k = 1..6; day 0 has no daily mean),
    against the ERA5 pair (00Z D, 12Z D) (`utc_pairs`, ruling C2), the same pair the
    daily maps verify that day on (`maps.utc_daily_truth`).
    """
    era = CB.era5_window(pd.Timestamp(peak))                      # 12Z peak-7 .. 00Z peak
    if time_kind(cube) == "instant":
        inst = AI._squeeze(AI.instantaneous_field(cube, "t2m_anom")).mean("member")
        return (CB.daily_pairs(inst, peak) - CB.daily_pairs(era, peak)).values
    a = reduce_daily(cube, peak, per_day=True).mean("member")      # (time=6, lat, lon)
    e = utc_pairs(era, daily_days(peak)).values                    # maps days 1..6
    out = np.full((7,) + a.shape[1:], np.nan, "float64")
    out[1:] = a.values - e
    return out


def _hind_red_current(src: Source, path: Path) -> bool:
    """A daily-mean source's reduced hind file must carry the `DAILY_PAIR` bias_daily;
    one written before ruling C2 (no `daily_pair` attr) is rebuilt."""
    if src.time_kind != "daily_mean":
        return True
    with xr.open_dataset(path) as d:
        return d.attrs.get("daily_pair") == DAILY_PAIR


def hind_reduce(src: Source, eid: str, year: int, init, peak) -> Path | None:
    """Reduce one hind cube to the CFS hind schema (al_cfs, al_era5, bias7, bias_daily)."""
    out = src.hind_red_path(eid, year)
    cp = src.hind_cube_path(eid, year)
    if not cp.exists():
        return None
    if (out.exists() and out.stat().st_mtime >= cp.stat().st_mtime   # rebuilt cube -> redo
            and _hind_red_current(src, out)):
        return out
    with xr.open_dataset(cp) as cube:
        cube = cube.load()
    check_cube(cube, peak)
    f = reduce(cube, peak, src.window)
    e = era5_on(peak, src.obs_window).mean("time")
    al = np.asarray(TR.area_mean(f).values, dtype=float)
    ds = xr.Dataset(
        dict(al_cfs=("member", al), al_era5=float(TR.area_mean(e)),
             bias7=(("lat", "lon"), (f.mean("member").values - e.values).astype("float32")),
             bias_daily=(("day", "lat", "lon"),
                         _daily_bias(src, cube, peak).astype("float32"))),
        coords=dict(member=np.arange(al.size), day=np.arange(7),
                    lat=e["lat"].values, lon=e["lon"].values),
        attrs=dict(case=eid, year=year, init=str(init), peak=str(peak), source=src.name,
                   window=src.window, obs_window=src.obs_window,
                   **({"daily_pair": DAILY_PAIR} if src.time_kind == "daily_mean" else {})))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    os.replace(tmp, out)
    return out


def hind(src: Source, cases: list[str] | None = None, force: bool = False) -> None:
    mod = src.mod()
    jobs = CB.hind_jobs(cases)
    print(f"[s2sbase] {src.name} hind: {len(jobs)} (case, year) jobs", flush=True)
    for eid, y, _i, _p in jobs:
        try:
            p = mod.hind_case(eid, y, force=force)
            print(f"  {eid} {y}: {p}", flush=True)
        except Exception as e:                       # noqa: BLE001 - report, keep going
            print(f"  {eid} {y}: FAILED {e!r}", flush=True)


def bias(src: Source) -> Path:
    """LOYO bias.nc + bias.csv in the published CFS schema, from the hind cubes."""
    if src.bias == "reforecast":
        raise SystemExit(f"[s2sbase] {src.name} debiases from its reforecast: its module "
                         "writes bias.nc / bias.csv; s2sbase only reads them")
    if src.name in BUILTIN:
        raise SystemExit("[s2sbase] the CFS bias is published (runs/acal/cfs/bias.*)")
    df = aprep.episodes()
    want = {}
    for eid, y, i, p in CB.hind_jobs():
        want.setdefault(eid, []).append((y, i, p))
    rows, b7, bd, fams, ok = [], [], [], [], []
    lat, lon = _clim_grid()
    for r in df.itertuples():
        paths = [hind_reduce(src, r.episode_id, y, i, p) for y, i, p in want.get(r.episode_id, [])]
        ds = [xr.open_dataset(x) for x in paths if x is not None]
        if len(ds) < MIN_YEARS:
            print(f"[s2sbase] {src.name} {r.episode_id}: {len(ds)} hind years < {MIN_YEARS}, "
                  "no bias (corr variants NaN)", flush=True)
            for x in ds:
                x.close()
            continue
        d = [float(x.al_cfs.mean() - x.al_era5) for x in ds]
        rows.append(dict(episode_id=r.episode_id, n_years=len(ds),
                         years=",".join(str(x.attrs["year"]) for x in ds),
                         bias_conus=float(np.mean(d)), bias_sd=float(np.std(d, ddof=1)),
                         spread_hind=float(np.mean([x.al_cfs.std(ddof=1) for x in ds]))))
        b7.append(xr.concat([x.bias7 for x in ds], "year").mean("year"))
        bd.append(xr.concat([x.bias_daily for x in ds], "year").mean("year"))
        fams.append(r.family)
        for x in ds:
            x.close()
    if not rows:
        raise SystemExit(f"[s2sbase] {src.name}: no case has {MIN_YEARS} hind years")
    tab = pd.DataFrame(rows)
    out = xr.Dataset(
        dict(bias_conus=("case", tab.bias_conus.values), bias_sd=("case", tab.bias_sd.values),
             n_years=("case", tab.n_years.values),
             bias7=xr.concat(b7, "case"), bias_daily=xr.concat(bd, "case")),
        coords=dict(case=tab.episode_id.values, family=("case", np.asarray(fams))),
        attrs=dict(note=f"{src.label} member-mean minus ERA5 on window {src.window} (ERA5 "
                        f"{src.obs_window}), averaged over the case's other years "
                        "(leave-one-year-out). Subtract from the forecast to correct.",
                   source=src.name))
    src.root.mkdir(parents=True, exist_ok=True)
    tmp = src.bias_nc.with_suffix(".tmp.nc")
    out.to_netcdf(tmp)
    os.replace(tmp, src.bias_nc)
    tab.to_csv(src.bias_csv, index=False)
    print(f"[s2sbase] {src.name}: bias for {len(tab)}/{len(df)} cases -> {src.bias_nc}")
    return src.bias_nc


# --------------------------------------------------------------------------- #
# Maps
# --------------------------------------------------------------------------- #
def maps_stage(src: Source, truth: TR.Truth) -> list[Path]:
    """maps_{fields,daily}_<source>.nc under runs/acal/analysis/s2s/<truth>/: the source's
    probabilities with the truth's truth/clim (and, for a masked truth, `valid`), built on
    `maps.truth_base(truth)` - the AI+RES maps file against that truth."""
    from acal import maps as M
    od = out_dir(truth)
    od.mkdir(parents=True, exist_ok=True)
    out = []
    for daily, name in ((False, "maps_fields"), (True, "maps_daily")):
        base = M.truth_base(truth, daily)
        p = od / f"{name}_{src.name}.nc"
        tmp = p.with_suffix(".tmp.nc")
        M.source_dataset(src.name, base, daily, truth).to_netcdf(tmp)
        os.replace(tmp, p)
        print(f"[s2sbase] wrote {p}")
        out.append(p)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="cfs13")
    p.add_argument("--stage", required=True,
                   choices=("list", "with_data", "json", "aires", "aires_mask", "aires_fields",
                            "hind",
                            "bias", "score", "paired", "maps"))
    p.add_argument("--truth", default="era5")
    p.add_argument("--case", action="append", help="episode id; repeatable")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if a.stage == "aires_mask":
        aires_mask(a.truth, a.workers, a.case, a.force)
        return 0
    if a.stage == "aires_fields":
        aires_fields_all(a.workers, a.case, a.force)
        return 0
    if a.stage == "list":
        for n in available():
            s = get(n)
            print(f"{n:6s} {s.label:24s} {s.time_kind:10s} window {s.window:4s} "
                  f"N={s.n_members} bias={s.bias} has_bias={s.has_bias()}")
        return 0
    if a.stage == "with_data":     # the driver's source list: "name bias root bias_csv"
        for n in BOARD_SOURCES:
            if has_data(n):
                s = get(n)
                print(f"{n} {s.bias} {s.root} {s.bias_csv}")
        return 0
    if a.stage == "aires":
        aires_windows()
        return 0
    src, truth = get(a.source), TR.get_truth(a.truth)
    if a.stage == "json":
        write_json(src, a.case, a.force)
    elif a.stage == "hind":
        hind(src, a.case, a.force)
    elif a.stage == "bias":
        bias(src)
    elif a.stage == "score":
        score(src, truth)
    elif a.stage == "paired":
        paired(src, truth)
    elif a.stage == "maps":
        maps_stage(src, truth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
