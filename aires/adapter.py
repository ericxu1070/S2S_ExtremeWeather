"""Convert a GenCast model state into the 72-channel initial condition FCN3 wants.

Why this exists
---------------
In AI+RES the GenCast walker is stopped at a resampling time and its state must be
scored by an FCN3 ensemble forecast. FCN3 consumes a single frame of 72 channels;
GenCast carries 69 of them under different names, on a latitude axis running the
other way. This module is that translation, plus the three channels GenCast lacks.

The three derived channels
--------------------------
``u100m``/``v100m``
    The 100 m wind is the 10 m wind sped up and veered by boundary-layer shear. Both
    effects are one complex number per grid point: writing W = u + iv, we model
    ``W100 = c(x) * W10`` where ``|c|`` is the speed-up and ``arg(c)`` the veering
    angle. c is dominated by surface roughness, which is static, so it is fitted once
    against ERA5 (``aires/calibrate.py``) and reused. A real-valued ratio would
    capture the speed-up but throw away the veering, which reaches 10-20 degrees over
    land.

``tcwv``
    Total column water vapour is ``(1/g) * integral of q dp``. GenCast carries q on 13
    pressure levels, so any estimate is a quadrature ``tcwv = sum_l w_l * q_l``. The
    physical choice is the trapezoid, but it misses the layer below 1000 hPa and, over
    terrain, integrates levels that are underground - errors that are large and static
    in space. So the weights themselves are fitted per grid point
    (``aires/calibrate.py``), ridge-shrunk toward the trapezoid so the fit corrects the
    physics rather than replacing it. Held out, that beats a scaled trapezoid
    (4.6% vs 5.2% worst-case relative RMSE). The residual is information the 13-level
    state genuinely does not contain.

Nothing here needs a GPU, network access, or the ``fcn3`` conda env: it is pure
xarray/numpy and runs wherever the GenCast env runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from aires import aconfig as A

G = 9.80665  # m s-2, standard gravity (matches the ERA5/IFS definition)

# Fallbacks used when no calibration file is supplied. The power-law exponent 1/7 is
# the textbook neutral-stability value; no veering is assumed. Good enough for smoke
# tests, NOT good enough for production scoring - fit the calibration.
DEFAULT_WIND_COEF = complex((100.0 / 10.0) ** (1.0 / 7.0), 0.0)
DEFAULT_TCWV_SCALE = 1.0


# --------------------------------------------------------------------------- #
# Channel-name plumbing
# --------------------------------------------------------------------------- #
def split_channel(name: str) -> tuple[str, int | None]:
    """``'u850' -> ('u', 850)``; ``'t2m' -> ('t2m', None)``."""
    if name in A.SURFACE_VARS:
        return name, None
    prefix, digits = name[0], name[1:]
    if prefix in A.LEVEL_PREFIXES and digits.isdigit() and int(digits) in A.P13:
        return prefix, int(digits)
    raise KeyError(f"not an FCN3 channel: {name!r}")


# --------------------------------------------------------------------------- #
# Derivations (shared with calibrate.py so the fit and the application cannot drift)
# --------------------------------------------------------------------------- #
def raw_tcwv(q: np.ndarray, levels_hpa) -> np.ndarray:
    """Trapezoidal ``(1/g) * int q dp`` over the model levels. Returns kg m-2.

    ``q`` has shape ``(nlev, ...)`` with ``nlev`` matching ``levels_hpa`` (ascending
    in pressure, i.e. top of atmosphere first).
    """
    p = np.asarray(levels_hpa, dtype="float64") * 100.0          # hPa -> Pa
    order = np.argsort(p)
    q64 = np.asarray(q, dtype="float64")[order]
    # np.trapezoid is the modern spelling; np.trapz on older numpy.
    trap = getattr(np, "trapezoid", None) or np.trapz
    return trap(q64, x=p[order], axis=0) / G


def apply_tcwv_weights(q: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Learned quadrature ``tcwv = sum_l w_l q_l``. ``weights`` is ``(lat, lon, nlev)``."""
    return np.einsum("xyl,lxy->xy", np.asarray(weights, dtype="float64"),
                     np.asarray(q, dtype="float64"))


def apply_wind_coef(u10: np.ndarray, v10: np.ndarray, coef) -> tuple[np.ndarray, np.ndarray]:
    """Rotate-and-scale the 10 m wind to 100 m: ``W100 = coef * W10``."""
    w = np.asarray(u10, dtype="float64") + 1j * np.asarray(v10, dtype="float64")
    w100 = np.asarray(coef) * w
    return w100.real, w100.imag


# --------------------------------------------------------------------------- #
# Calibration I/O
# --------------------------------------------------------------------------- #
def load_calibration(path=None) -> xr.Dataset | None:
    """Load the fitted derived-channel calibration, or None if it has not been built.

    Returning None rather than raising lets smoke tests run on the physical defaults;
    production callers should pass ``require=True`` via :func:`require_calibration`.
    """
    path = Path(path or A.CALIB_PATH)
    if not path.exists():
        return None
    return xr.open_dataset(path).load()


def require_calibration(path=None) -> xr.Dataset:
    cal = load_calibration(path)
    if cal is None:
        raise FileNotFoundError(
            f"no derived-channel calibration at {path or A.CALIB_PATH}.\n"
            "Build it first:  PYTHONPATH=. python -m aires.calibrate --fit"
        )
    return cal


# --------------------------------------------------------------------------- #
# The conversion
# --------------------------------------------------------------------------- #
def _orient(da: xr.DataArray) -> xr.DataArray:
    """Put a field on FCN3's grid: lat descending 90 -> -90, lon ascending 0 -> 359.75."""
    if da["lat"].values[0] < da["lat"].values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    if not np.all(np.diff(da["lon"].values) > 0):
        da = da.sortby("lon")
    return da


def gencast_to_fcn3(
    state: xr.Dataset,
    *,
    calib: xr.Dataset | None = None,
    time_index: int = -1,
    valid_time=None,
) -> xr.DataArray:
    """Build FCN3's ``(time, variable, lat, lon)`` initial condition from a GenCast state.

    Parameters
    ----------
    state
        GenCast variables on the 0.25 degree global grid. Extra dims (``batch``,
        ``time``) are reduced; extra variables (vertical_velocity, SST, precip) are
        ignored. Must be GLOBAL - a CONUS-cropped cube cannot drive FCN3.
    calib
        Output of :func:`load_calibration`. If None, the physical defaults are used
        and a warning-grade attribute is set on the result.
    time_index
        Which frame to take when ``state`` still has a time dim. Default -1 = the most
        recent, which is the frame a walker restart hands over.
    valid_time
        Value for the output ``time`` coordinate. Defaults to the selected frame's own
        time, falling back to ``state.datetime`` if present.
    """
    # ---- reduce to a single frame -----------------------------------------------
    if "batch" in state.dims:
        state = state.isel(batch=0, drop=True)
    if valid_time is None and "time" in state.coords and state["time"].size:
        try:
            valid_time = np.asarray(state["time"].values).ravel()[time_index]
        except (IndexError, TypeError):
            valid_time = None
    if "time" in state.dims:
        state = state.isel(time=time_index, drop=True)
    if valid_time is None and "datetime" in state.coords:
        valid_time = np.asarray(state["datetime"].values).ravel()[-1]
    if valid_time is None:
        raise ValueError("cannot determine the valid time; pass valid_time=...")

    missing = [v for v in A.GENCAST_NEEDED if v not in state]
    if missing:
        raise KeyError(f"GenCast state is missing {missing}")

    lev = np.asarray(state["level"].values)
    if set(lev.tolist()) != set(A.P13):
        raise ValueError(f"expected levels {A.P13}, got {lev.tolist()}")

    # ---- orient onto FCN3's grid -------------------------------------------------
    fields = {v: _orient(state[v]) for v in A.GENCAST_NEEDED}
    ref = fields[A.SURFACE_FROM_GENCAST["t2m"]]
    lat, lon = ref["lat"].values, ref["lon"].values
    if lat.size != A.FCN3_NLAT or lon.size != A.FCN3_NLON:
        raise ValueError(
            f"FCN3 needs the global {A.FCN3_NLAT}x{A.FCN3_NLON} grid, got {lat.size}x{lon.size}. "
            "A CONUS-cropped GenCast cube cannot be used here."
        )

    # ---- derived channels ---------------------------------------------------------
    u10 = fields[A.SURFACE_FROM_GENCAST["u10m"]].values
    v10 = fields[A.SURFACE_FROM_GENCAST["v10m"]].values
    q = fields[A.LEVEL_FROM_GENCAST["q"]].transpose("level", "lat", "lon").values

    if calib is None:
        coef, calib_tag = DEFAULT_WIND_COEF, "defaults(uncalibrated)"
        tcwv = raw_tcwv(q, state["level"].values) * DEFAULT_TCWV_SCALE
    else:
        coef = calib["wind_coef_real"].values + 1j * calib["wind_coef_imag"].values
        calib_tag = str(calib.attrs.get("provenance", "fitted"))
        # tcwv_weights must be ordered like state["level"]; calibrate.py stores them
        # on the shared P13 axis, so reindex rather than assume the order matches.
        w = calib["tcwv_weights"].sel(level=list(state["level"].values)).transpose("lat", "lon", "level")
        tcwv = apply_tcwv_weights(q, w.values)

    u100, v100 = apply_wind_coef(u10, v10, coef)

    derived = {"u100m": u100, "v100m": v100, "tcwv": tcwv}

    # ---- stack in FCN3's exact channel order ---------------------------------------
    planes = np.empty((len(A.FCN3_VARIABLES), lat.size, lon.size), dtype="float32")
    for i, name in enumerate(A.FCN3_VARIABLES):
        if name in derived:
            planes[i] = derived[name]
            continue
        key, level = split_channel(name)
        if level is None:
            planes[i] = fields[A.SURFACE_FROM_GENCAST[key]].values
        else:
            planes[i] = fields[A.LEVEL_FROM_GENCAST[key]].sel(level=level).values

    out = xr.DataArray(
        planes[None, ...],
        dims=("time", "variable", "lat", "lon"),
        coords={
            "time": np.asarray([valid_time], dtype="datetime64[ns]"),
            "variable": list(A.FCN3_VARIABLES),
            "lat": lat,
            "lon": lon,
        },
        name="ic",
    )
    out.attrs["source"] = "gencast"
    out.attrs["derived_channels"] = ",".join(A.DERIVED_VARS)
    out.attrs["calibration"] = calib_tag
    return out


# --------------------------------------------------------------------------- #
# earth2studio DataSource
# --------------------------------------------------------------------------- #
class GenCastICSource:
    """An earth2studio ``DataSource`` backed by a GenCast state.

    earth2studio's DataSource is a bare Protocol - anything with
    ``__call__(time, variable) -> xr.DataArray`` works - which is what
    ``fcn3/run_fcn3.py::LocalIC`` already exploits for cached NetCDF ICs. Handing FCN3
    a GenCast state therefore needs no change on the FCN3 side at all.
    """

    def __init__(self, ic: xr.DataArray):
        if tuple(str(s) for s in ic["variable"].values) != A.FCN3_VARIABLES:
            raise ValueError("ic does not carry FCN3's 72 channels in order")
        self.da = ic

    @classmethod
    def from_gencast(cls, state: xr.Dataset, **kw) -> "GenCastICSource":
        return cls(gencast_to_fcn3(state, **kw))

    def __call__(self, time, variable) -> xr.DataArray:
        t = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        v = list(np.atleast_1d(variable))
        return self.da.sel(time=t, variable=v)
