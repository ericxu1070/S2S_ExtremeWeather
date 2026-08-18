"""Static contracts and paths for the AI+RES stack.

Two models with two different state representations meet here, so this module is the
single place that states how they line up. Everything downstream imports these names
rather than re-deriving them.

    GenCast (JAX/GraphCast)                FourCastNet3 (PyTorch/earth2studio)
    ---------------------------            -----------------------------------
    xr.Dataset, named variables            xr.DataArray, 72 stacked channels
    lat ASCENDING  -90 -> +90              lat DESCENDING +90 -> -90
    lon 0 -> 359.75                        lon 0 -> 359.75            (same)
    13 levels, 50..1000 hPa                13 levels, 50..1000 hPa    (same)
    K / Pa / m s-1 / m2 s-2 / kg kg-1      identical units            (same)

So the conversion is a rename plus a latitude flip plus three derived channels
(u100m, v100m, tcwv) that GenCast does not carry. FCN3 needs exactly ONE time frame,
which is why GenCast's 12 h step versus FCN3's 6 h step is not an issue.
"""
from __future__ import annotations

import os
from pathlib import Path

from gencast_s2s import config as C

# --------------------------------------------------------------------------- #
# FCN3's channel contract.
#
# The authoritative list lives in earth2studio (``models/px/fcn3.py::VARIABLES``),
# which is only importable in the ``fcn3`` conda env. Calibration and Gate 1 run in
# the GenCast env, where it is not, so the list is mirrored here and checked against
# the package whenever the package IS importable (``verify_against_package``) and
# against any on-disk IC file (``verify_against_ic``). Do not edit by hand.
# --------------------------------------------------------------------------- #
P13 = tuple(C.P13)                      # (50, 100, ..., 1000) - shared by both models
LEVEL_PREFIXES = ("u", "v", "z", "t", "q")
SURFACE_VARS = ("u10m", "v10m", "u100m", "v100m", "t2m", "msl", "tcwv")

FCN3_VARIABLES = tuple(SURFACE_VARS) + tuple(
    f"{pfx}{lev}" for pfx in LEVEL_PREFIXES for lev in P13
)
assert len(FCN3_VARIABLES) == 72, len(FCN3_VARIABLES)

# FCN3 grid. earth2studio asserts these exactly; a flipped or rolled axis is silent
# garbage rather than an error, so the adapter checks them on every conversion.
FCN3_NLAT, FCN3_NLON = 721, 1440

# --------------------------------------------------------------------------- #
# GenCast -> FCN3 name maps. Direct, 1:1, no unit conversion anywhere.
# --------------------------------------------------------------------------- #
SURFACE_FROM_GENCAST = {
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
    "t2m":  "2m_temperature",
    "msl":  "mean_sea_level_pressure",
}
LEVEL_FROM_GENCAST = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "t": "temperature",
    "q": "specific_humidity",
}
# The only channels GenCast cannot supply directly; see aires/adapter.py.
DERIVED_VARS = ("u100m", "v100m", "tcwv")

# GenCast state variables the adapter actually reads (the rest of GenCast's target
# set - vertical_velocity, sea_surface_temperature, total_precipitation_12hr - is
# carried by the walker but never consumed by FCN3).
GENCAST_NEEDED = tuple(SURFACE_FROM_GENCAST.values()) + tuple(LEVEL_FROM_GENCAST.values())

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = C.ROOT
RUNS = Path(os.environ.get("GENCAST_RUNS", ROOT / "runs"))
AIRES_ROOT = RUNS / "aires"

# Calibration for the three derived channels, fitted once from ERA5 (see calibrate.py).
CALIB_PATH = Path(os.environ.get("AIRES_CALIB", AIRES_ROOT / "calib" / "derived_calib.nc"))

# ERA5 sources for calibration/validation. Both are FCN3 IC files (all 72 channels,
# including the true u100m/v100m/tcwv), so no network access is needed.
CALIB_IC_GLOB = str(RUNS / "p90_t2m" / "ic" / "*.nc")            # 90 cases, 2021-2025: FIT
HOLDOUT_IC_GLOB = str(RUNS / "fcn3" / "week3" / "ic" / "*_ic.nc")  # the 6 events: TEST

# Gate 1 pass criteria (aires.md). Evaluated on HOLDOUT_IC_GLOB only.
GATE1 = {
    "u100m_rmse_max": 1.0,      # m/s
    "u100m_abs_bias_max": 0.2,  # m/s
    "tcwv_rel_rmse_max": 0.05,  # fraction
    "tcwv_abs_rel_bias_max": 0.02,
}


# --------------------------------------------------------------------------- #
# Walker configuration and the per-event run tree.
#
# A walker is a GenCast trajectory that can be STOPPED, checkpointed and restarted, so
# unlike the xres cubes its checkpoint must be GLOBAL: FCN3 consumes the whole sphere, and
# a CONUS-cropped state cannot initialise it (xres/xinference.py crops before the host
# transfer, which is why those cached cubes are useless here).
#
#     runs/aires/<event>/walkers/w<ii>/step<k>/state.nc   global 2-frame restart state
#                                             /diag.nc    CONUS-cropped, every step
#                       /scores/                          FCN3 scores theta(t_k)
#                       /gate3/                           Gate 3 artifacts
# --------------------------------------------------------------------------- #
WALKER_RES = os.environ.get("AIRES_RES", "0p25")   # GenCast checkpoint resolution
WALKER_STEP_H = 12                                 # GenCast native step
WALKER_WEEKS = int(os.environ.get("AIRES_WEEKS", 3))  # lead of the pilot init (21 d)

# Base seed for walker segment RNGs. A segment's key is fold_in(fold_in(base, step),
# walker), so a cloned walker gets a stream no sibling has ever used -- which IS the
# clone-perturbation mechanism (the paper has to hand-tune spherical-harmonic noise
# because its walker, PlaSim, is deterministic; GenCast's diffusion noise does it for us).
WALKER_BASE_SEED = int(os.environ.get("AIRES_BASE_SEED", 20260101))

# Global state grid, asserted on every read/write. Same numbers as FCN3's, deliberately:
# the walker state feeds the adapter directly.
STATE_NLAT, STATE_NLON = FCN3_NLAT, FCN3_NLON

# The 12 time-varying variables GenCast predicts, plus the 2 statics. A restart state must
# carry all 14: the prognostics are what the rollout advances, the statics are what
# data_utils needs to rebuild the input batch.
STATE_PROGNOSTIC = (
    "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
    "geopotential", "mean_sea_level_pressure", "sea_surface_temperature",
    "specific_humidity", "temperature", "total_precipitation_12hr",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
)
STATE_STATIC = ("geopotential_at_surface", "land_sea_mask")

# Gate 3 (aires.md): free-running walkers checkpointed at these leads, each scored by FCN3;
# the score must rank the eventual outcome.
GATE3_LEAD_DAYS = (3, 6, 9, 12, 15)
GATE3 = {
    "spearman_min": 0.5,          # required at t_k = 9 d AND 12 d
    "required_leads": (9, 12),
}


def fig_dir() -> Path:
    return ROOT / "figures" / "aires"


def event_dir(event: str) -> Path:
    return AIRES_ROOT / event


def walker_dir(event: str, walker: int) -> Path:
    return event_dir(event) / "walkers" / f"w{walker:02d}"


def segment_dir(event: str, walker: int, step: int) -> Path:
    return walker_dir(event, walker) / f"step{step:02d}"


def state_path(event: str, walker: int, step: int) -> Path:
    """Global 2-frame restart state at the END of segment ``step``."""
    return segment_dir(event, walker, step) / "state.nc"


def diag_path(event: str, walker: int, step: int) -> Path:
    """CONUS-cropped every-step diagnostics for segment ``step``."""
    return segment_dir(event, walker, step) / "diag.nc"


def gencast_inputs_path(event: str, res: str | None = None, weeks: int | None = None) -> Path:
    """The event's ERA5 init frames, in GenCast schema -- a walker's step-0 state.

    Built by the xres pipeline; global and 2-frame already, which is exactly the walker
    checkpoint format, so step 0 needs no conversion.
    """
    res = res or WALKER_RES
    weeks = WALKER_WEEKS if weeks is None else weeks
    return RUNS / "xres" / res / f"week{weeks}" / "inputs" / f"{event}_inputs.nc"


def ensure_dirs(event: str | None = None) -> None:
    dirs = [AIRES_ROOT, CALIB_PATH.parent, fig_dir()]
    if event:
        dirs += [event_dir(event), event_dir(event) / "walkers",
                 event_dir(event) / "scores", event_dir(event) / "gate3"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Contract checks
# --------------------------------------------------------------------------- #
def verify_against_package() -> None:
    """Assert our mirrored channel list matches earth2studio's. No-op if not importable."""
    try:
        from earth2studio.models.px.fcn3 import VARIABLES as PKG
    except Exception:
        return
    if tuple(PKG) != FCN3_VARIABLES:
        raise AssertionError(
            "FCN3_VARIABLES drifted from earth2studio.\n"
            f"  package: {list(PKG)}\n  aires  : {list(FCN3_VARIABLES)}"
        )


def verify_against_ic(path) -> None:
    """Assert an on-disk FCN3 IC file has exactly our channels, in order."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        got = tuple(str(s) for s in ds["variable"].values)
        if got != FCN3_VARIABLES:
            raise AssertionError(f"channel mismatch vs {path}:\n  file : {got}\n  aires: {FCN3_VARIABLES}")
        if ds.sizes["lat"] != FCN3_NLAT or ds.sizes["lon"] != FCN3_NLON:
            raise AssertionError(f"grid mismatch vs {path}: {dict(ds.sizes)}")
        if ds["lat"].values[0] < ds["lat"].values[-1]:
            raise AssertionError(f"{path} has ascending lat; FCN3 expects 90 -> -90")
