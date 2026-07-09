"""Static configuration: events, horizons, model spec, CONUS box, folder layout.

All output lives under ``runs/`` inside the project directory (NFS-shared, so the
login node and every a3mega compute node see the same files). Nothing is written to
Google Drive — the notebook's ``USE_DRIVE`` path is gone.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project / output layout
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]          # .../S2S_ExtremeWeather
RUNS = Path(os.environ.get("GENCAST_RUNS", ROOT / "runs"))

MODELS_DIR = RUNS / "models"                         # shared across horizons
PARAMS_DIR = MODELS_DIR / "params"
STATS_DIR  = MODELS_DIR / "stats"
OBS_DIR    = RUNS / "observations"                   # shared verification truth
LOGS_DIR   = ROOT / "logs"

# Local copy of the small static fields (orography + land-sea mask), shared too.
STATICS_FILE = MODELS_DIR / "statics.nc"
# Local copy of the 1990-2019 T2m climatology, cropped to CONUS, so the infer stage
# (anomaly conversion) is fully offline on the compute node.
CLIM_FILE = MODELS_DIR / "clim_1990_2019_t2m_conus.nc"

# --------------------------------------------------------------------------- #
# HRRR observed overlay. The local HRRR v3 netCDF archive (3km CONUS, 2015-2025,
# 6-hourly; one file per timestamp with t2m, u/v on pressure levels and
# log_tp = ln(6h precip mm + 1e-5) on the native 1059x1799 Lambert grid) gives a
# second, higher-resolution observed truth to overlay alongside ERA5. Anomalies
# reuse the ERA5 1990-2019 climatology above (no HRRR-native climatology exists
# offline), so a small HRRR-vs-ERA5 model bias is folded into the overlay.
# --------------------------------------------------------------------------- #
HRRR_NC = Path(os.environ.get(
    "HRRR_NC", "/glade/derecho/scratch/mdarman/hrrr_work/hrrr_nc_v3"))
HRRR_REMAP_CACHE = MODELS_DIR / "hrrr_to_conus_nn_index.npy"


def week_dir(weeks: int) -> Path:
    return RUNS / f"week{weeks}"


def inputs_dir(weeks: int) -> Path:
    return week_dir(weeks) / "inputs"


def forecasts_dir(weeks: int) -> Path:
    return week_dir(weeks) / "forecasts"


def figures_dir(weeks: int) -> Path:
    return week_dir(weeks) / "figures"


def ensure_dirs(weeks: int | None = None) -> None:
    """Create the full output tree (shared dirs always; per-week dirs if given)."""
    for d in (RUNS, MODELS_DIR, PARAMS_DIR, STATS_DIR, OBS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if weeks is not None:
        for d in (inputs_dir(weeks), forecasts_dir(weeks),
                  figures_dir(weeks) / "pdf",
                  figures_dir(weeks) / "maps",
                  figures_dir(weeks) / "scores"):
            d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Experiment horizons. weeks -> lead in days. init = peak - lead_days.
# Verification window is always the 7 days ending at the peak, i.e. forecast
# lead days (lead_days - 6) .. lead_days.
# --------------------------------------------------------------------------- #
WEEKS_TO_LEAD_DAYS = {2: 14, 3: 21, 4: 28}
VALID_WEEKS = tuple(WEEKS_TO_LEAD_DAYS)


def lead_days_for(weeks: int) -> int:
    if weeks not in WEEKS_TO_LEAD_DAYS:
        raise ValueError(f"weeks must be one of {VALID_WEEKS}, got {weeks}")
    return WEEKS_TO_LEAD_DAYS[weeks]


# --------------------------------------------------------------------------- #
# Events. (peak_date, is_out_of_sample_post_2019). We only run out-of-sample
# events because the GenCast "<2019" checkpoint would otherwise be scored on its
# own training period (apparent skill there could be memorization).
# --------------------------------------------------------------------------- #
ALL_EVENTS = {
    # Heat
    "PNW_HeatDome_2021":        ("2021-06-28", True),
    "Chicago_HeatWave_1995":    ("1995-07-14", False),
    "NAmerica_HeatWave_2012":   ("2012-07-04", False),
    "SCentral_HeatWave_2011":   ("2011-07-20", False),
    "SCentral_HeatDome_2023":   ("2023-06-27", False),
    "Southwest_HeatWave_2020":  ("2020-08-16", True),
    "California_HeatWave_2022": ("2022-09-06", True),
    # Cold
    "WinterStorm_Uri_2021":     ("2021-02-15", True),
    "PolarVortex_2019":         ("2019-01-30", True),
    "PolarVortex_2014":         ("2014-01-06", False),
    "WinterStorm_Elliott_2022": ("2022-12-23", True),
    # Out-of-distribution CONUS hurricanes (peak = landfall day; all post-2019, so
    # out-of-sample for the <2019 GenCast checkpoint and covered by the HRRR archive).
    "HurricaneIda_2021":        ("2021-08-29", True),   # LA landfall + NE flooding
    "HurricaneIan_2022":        ("2022-09-28", True),   # SW Florida, Cat 4-5
    "HurricaneIdalia_2023":     ("2023-08-30", True),   # FL Big Bend
    # Out-of-distribution CONUS extreme-precipitation events.
    "California_AR_Floods_2023":("2023-01-10", True),   # Jan 2023 atmospheric rivers
    "Kentucky_Floods_2022":     ("2022-07-28", True),   # Eastern KY flash floods
    "Vermont_Floods_2023":      ("2023-07-10", True),   # Jul 2023 Northeast synoptic flooding
}
EVENTS = {k: v[0] for k, v in ALL_EVENTS.items() if v[1]}   # name -> peak (OOS)

# Optional subsetting for smoke tests, via env (applied to every stage):
#   GENCAST_EVENTS="PNW_HeatDome_2021,WinterStorm_Uri_2021"  explicit names
#   GENCAST_MAX_EVENTS=1                                       first N events
_sel = os.environ.get("GENCAST_EVENTS")
if _sel:
    _want = [s.strip() for s in _sel.split(",") if s.strip()]
    EVENTS = {k: EVENTS[k] for k in _want if k in EVENTS}
_maxn = os.environ.get("GENCAST_MAX_EVENTS")
if _maxn:
    EVENTS = dict(list(EVENTS.items())[:int(_maxn)])


# --------------------------------------------------------------------------- #
# Model spec (GenCast 1.0deg "Mini", the lightweight diffusion checkpoint).
# --------------------------------------------------------------------------- #
N_MEMBERS = int(os.environ.get("GENCAST_N_MEMBERS", 24))   # multiple of 8 -> 3 waves on 8 GPUs

# Diffusion ensemble; the 14-day notebook also ran a single deterministic GraphCast
# control. We default to GenCast-only (the ensemble is the whole point and it finishes
# comfortably on one 8-GPU node). "graphcast" is left out of MODELS on purpose.
MODELS = ["gencast"]

P13 = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
ATM = ["temperature", "geopotential", "u_component_of_wind",
       "v_component_of_wind", "vertical_velocity", "specific_humidity"]

# CONUS verification box (lon in 0..360: -125 -> 235, -66 -> 294).
LAT = slice(24, 50)
LON = slice(235, 294)
# Explicit numeric extent for cartopy (deg east in -180..180, deg north).
CONUS_EXTENT = [-125.0, -66.0, 24.0, 50.0]

GENCAST_PARAMS = "GenCast 1p0deg Mini <2019.npz"
PARAMS_DIR_GCS = "gs://dm_graphcast/gencast/params/"
STATS_DIR_GCS  = "gs://dm_graphcast/gencast/stats/"
STATS_FILES = ("diffs_stddev_by_level.nc", "mean_by_level.nc",
               "stddev_by_level.nc", "min_by_level.nc")


def model_cfg(model: str = "gencast", res: float = 1.0,
              params_file: str | None = None) -> dict:
    """GenCast task spec. ``res`` selects the grid (1.0 or 0.25 deg) and therefore the
    coarsening stride from the 0.25deg ERA5 stores; ``params_file`` overrides the default
    Mini checkpoint (e.g. the full 1p0deg/0p25deg <2019 checkpoints used by the xres
    cross-resolution experiment). Both default to the original Mini/1.0deg behaviour so the
    existing week2/3/4 pipeline is unchanged."""
    if model == "gencast":
        return dict(
            step_h=12, levels=13, levels_hpa=list(P13), res=res,
            surf=["2m_temperature", "mean_sea_level_pressure",
                  "10m_u_component_of_wind", "10m_v_component_of_wind",
                  "sea_surface_temperature"],
            precip_in="total_precipitation_12hr",
            params_file=params_file or GENCAST_PARAMS,
        )
    raise ValueError(f"unsupported model {model!r}")


def n_rollout_steps(weeks: int, model: str = "gencast") -> int:
    """Number of model steps to reach the end of the forecast horizon."""
    return lead_days_for(weeks) * 24 // model_cfg(model)["step_h"]
