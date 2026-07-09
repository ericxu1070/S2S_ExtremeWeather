"""Static configuration for the cross-resolution (0.25deg vs 1.0deg) experiment.

Reuses the shared caches from ``gencast_s2s`` (checkpoint params dir, normalization stats,
statics, T2m climatology, ERA5/HRRR observed truth under ``runs/observations``) and adds a
separate output tree under ``runs/xres/`` so the original week2/3/4 outputs are untouched.
"""
from __future__ import annotations

import os
from pathlib import Path

from gencast_s2s import config as C

# --------------------------------------------------------------------------- #
# Output layout (under the shared runs/ NFS root). Observed truth stays shared
# in runs/observations (resolution independent); forecasts/caches are per resolution.
# --------------------------------------------------------------------------- #
XRES_ROOT = C.RUNS / "xres"
XFIG_DIR  = XRES_ROOT / "figures"          # cross-resolution comparison figures


def res_dir(res: str, weeks: int) -> Path:
    return XRES_ROOT / res / f"week{weeks}"


def inputs_dir(res: str, weeks: int) -> Path:
    return res_dir(res, weeks) / "inputs"


def cache_dir(res: str, weeks: int) -> Path:
    """Full GenCast output cube per event (all vars / steps / members, CONUS box)."""
    return res_dir(res, weeks) / "cache"


def verif_dir(res: str, weeks: int) -> Path:
    """Per-member verification fields derived from the cube (one file per metric)."""
    return res_dir(res, weeks) / "verif"


def figures_dir(res: str, weeks: int) -> Path:
    return res_dir(res, weeks) / "figures"


def ensure_dirs(res: str | None = None, weeks: int | None = None) -> None:
    C.ensure_dirs()                                   # shared models/ + observations/
    XRES_ROOT.mkdir(parents=True, exist_ok=True)
    XFIG_DIR.mkdir(parents=True, exist_ok=True)
    if res is not None and weeks is not None:
        for d in (inputs_dir(res, weeks), cache_dir(res, weeks),
                  verif_dir(res, weeks), figures_dir(res, weeks) / "maps"):
            d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Resolutions. Both are full <2019 checkpoints; ``stride`` coarsens the 0.25deg ERA5
# stores to the model's native grid (0.25 -> 1, 1.0 -> 4).
# --------------------------------------------------------------------------- #
# Line styling: one color FAMILY per resolution (reds = 0.25deg, blues = 1.0deg) with
# the DARK shade for ERA5 truth and the LIGHT shade for the forecast, plus a distinct
# marker per curve so panels stay readable under color-vision deficiency (lightness,
# line style and marker all encode the same distinction redundantly).
RES_SPECS = {
    "0p25": dict(params="GenCast 0p25deg <2019.npz", res=0.25, stride=1,
                 color="#f4562b", truth_color="#99000d", marker="o",
                 label="GenCast 0.25deg"),
    "1p0":  dict(params="GenCast 1p0deg <2019.npz",  res=1.0,  stride=4,
                 color="#4f9ad1", truth_color="#08306b", marker="^",
                 label="GenCast 1.0deg"),
}
RES_ORDER = ("0p25", "1p0")                            # left-to-right in figures


def res_spec(res: str) -> dict:
    if res not in RES_SPECS:
        raise ValueError(f"res must be one of {tuple(RES_SPECS)}, got {res!r}")
    return RES_SPECS[res]


def params_for(res: str) -> str:
    return res_spec(res)["params"]


# Default 2-week horizon only (weeks 3/4 held for this experiment).
WEEKS = int(os.environ.get("XRES_WEEKS", 2))

# Ensemble size per resolution (both 24 by default, for an apples-to-apples PDF compare).
N_MEMBERS = {
    "0p25": int(os.environ.get("XRES_N_MEMBERS_0P25",
                               os.environ.get("XRES_N_MEMBERS", 24))),
    "1p0":  int(os.environ.get("XRES_N_MEMBERS_1P0",
                               os.environ.get("XRES_N_MEMBERS", 24))),
}


# --------------------------------------------------------------------------- #
# Events: (peak_date, headline_metric). 6 heat/cold (T2m) + 3 hurricanes (u850 speed)
# + 3 extreme-rain (tp total). All post-2019 -> out-of-sample for the <2019 checkpoints
# and covered by the HRRR archive (2015-2025).
# --------------------------------------------------------------------------- #
XRES_EVENTS = {
    # Heat (T2m anomaly)
    "PNW_HeatDome_2021":        ("2021-06-28", "t2m_anom"),
    "Southwest_HeatWave_2020":  ("2020-08-16", "t2m_anom"),
    "California_HeatWave_2022": ("2022-09-06", "t2m_anom"),
    # Cold (T2m anomaly)
    "WinterStorm_Uri_2021":     ("2021-02-15", "t2m_anom"),
    "PolarVortex_2019":         ("2019-01-30", "t2m_anom"),
    "WinterStorm_Elliott_2022": ("2022-12-23", "t2m_anom"),
    # Out-of-distribution CONUS hurricanes (u850 wind speed)
    "HurricaneIda_2021":        ("2021-08-29", "u850_speed"),   # LA Cat-4 + NE flooding
    "HurricaneIan_2022":        ("2022-09-28", "u850_speed"),   # SW Florida Cat-4/5
    "HurricaneIdalia_2023":     ("2023-08-30", "u850_speed"),   # FL Big Bend Cat-3
    # Out-of-distribution CONUS extreme precipitation (total precip)
    "California_AR_Floods_2023":("2023-01-10", "tp_total"),     # Jan 2023 atmospheric rivers
    "Kentucky_Floods_2022":     ("2022-07-28", "tp_total"),     # Eastern KY flash floods
    "Vermont_Floods_2023":      ("2023-07-10", "tp_total"),     # Jul 2023 Northeast floods
}

# Cold events (extreme = coldest member, negative T2m anomaly). Heat events default to +1.
COLD_EVENTS = {"WinterStorm_Uri_2021", "PolarVortex_2019", "WinterStorm_Elliott_2022"}

# Optional subsetting for smoke tests (mirrors gencast_s2s):
#   XRES_EVENTS_SEL="HurricaneIda_2021,Kentucky_Floods_2022"
#   XRES_MAX_EVENTS=1
_sel = os.environ.get("XRES_EVENTS_SEL")
if _sel:
    _want = [s.strip() for s in _sel.split(",") if s.strip()]
    XRES_EVENTS = {k: XRES_EVENTS[k] for k in _want if k in XRES_EVENTS}
_maxn = os.environ.get("XRES_MAX_EVENTS")
if _maxn:
    XRES_EVENTS = dict(list(XRES_EVENTS.items())[:int(_maxn)])


def events() -> dict:
    return XRES_EVENTS


def event_peak(name: str) -> str:
    return XRES_EVENTS[name][0]


def event_metric(name: str) -> str:
    return XRES_EVENTS[name][1]


def metrics_in_use() -> list[str]:
    seen = []
    for _, m in XRES_EVENTS.values():
        if m not in seen:
            seen.append(m)
    return seen
