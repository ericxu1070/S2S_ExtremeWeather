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


def fig_dir() -> Path:
    return ROOT / "figures" / "aires"


def ensure_dirs() -> None:
    for d in (AIRES_ROOT, CALIB_PATH.parent, fig_dir()):
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
