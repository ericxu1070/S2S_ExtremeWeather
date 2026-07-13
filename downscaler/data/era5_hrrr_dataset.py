"""ERA5 (coarse, 1 deg) -> HRRR (fine, 3 km CONUS) downscaling dataset.

Each sample is a SINGLE timestamp T:
    input  (conditioning): ERA5 drivers at T, bilinearly regridded from the
                           1-degree lat/lon grid onto the HRRR 1059x1799
                           Lambert grid, z-scored, + static channels (orog, lsm).
    target:                the HRRR analysis at T (full-field prognostic state
                           + diagnostics), z-scored.

The UNet then denoises the HRRR field conditioned on the upsampled ERA5 -> it
learns to add ~fine-scale (3 km) structure the coarse driver cannot resolve.

------------------------------------------------------------------------------
!!  DATA-FORMAT TOUCHPOINTS  -- verify these against your real ERA5/HRRR files !!
    (kept in config so they are edits, not code changes):
      * era5_variables / hrrr_prognostic_variables / hrrr_diagnostic_variables
      * era5_path_template / hrrr_path_template   (strftime patterns)
      * era5_lat_name / era5_lon_name             (coordinate names)
      * era5_level_name                           (vertical coord, if used)
      * match_era5_lon_to_360                     (HRRR lon convention)
    The 1-degree ERA5 is not downloaded yet (Derecho scratch down); until then
    use `data.use_dummy=true`. For production, prefer regridding ERA5 offline
    once (see scripts/precompute_stats.py notes) instead of per-__getitem__.
------------------------------------------------------------------------------
"""

import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# --- Per-channel transforms, applied to a coarse ERA5 field on read ---------------
# Precipitation is the reason this exists. Raw ERA5 tp is in METRES, HRRR's target `log_tp`
# is ln(mm + 1e-5) (see moein/hrrr_work/download_hrrr_v3.py), and the distribution is ~77%
# dry with a p99.99 of ~100 mm. Z-scoring a field that skewed is nearly useless, and the
# input would live on a scale 1000x off from the target. So the ERA5 precip channel gets the
# SAME log treatment as the target, on the same units.
ERA5_TRANSFORMS = {
    # ERA5 tp (metres) -> mm -> ln(mm + 1e-5). Matches the HRRR log_tp transform exactly.
    "log_precip_m_to_mm": lambda x: np.log(np.maximum(x, 0.0) * 1000.0 + 1e-5),
    # Already-mm precip -> same log transform.
    "log_precip_mm": lambda x: np.log(np.maximum(x, 0.0) + 1e-5),
}


def _parse_var_spec(spec):
    """Parse a variable spec -> (name, level|None, transform|None).

    Accepts:
        't2m'                                        -> ('t2m', None, None)
        'u:850'                                      -> ('u', 850.0, None)
        {'name': 'u', 'level': 850}                  -> ('u', 850.0, None)
        {'name': 'tp', 'transform': 'log_precip_m_to_mm'}
    """
    if isinstance(spec, dict):
        name = spec["name"]
        level = spec.get("level", None)
        transform = spec.get("transform", None)
    elif isinstance(spec, str) and ":" in spec:
        name, level = spec.split(":", 1)
        level, transform = float(level), None
    else:
        name, level, transform = str(spec), None, None

    if transform is not None and transform not in ERA5_TRANSFORMS:
        raise ValueError(
            f"unknown transform {transform!r} for variable {name!r}; "
            f"known: {sorted(ERA5_TRANSFORMS)}"
        )
    return str(name), (float(level) if level is not None else None), transform


class Era5HrrrDataset(Dataset):
    """Paired ERA5(T) -> HRRR(T) samples for conditional-diffusion downscaling.

    Args:
        timestamps: list of ISO timestamp strings (see data.dataloader.load_timestamps).
        era5_dir, hrrr_dir: roots for ERA5 and HRRR files.
        grid_meta_path: HRRR grid reference NetCDF with 2-D `latitude`/`longitude`.
        norm: Era5HrrrNorm (or DummyNormalizationStats).
        era5_variables: list of ERA5 driver channel specs.
        hrrr_prognostic_variables: HRRR full-field target channels.
        hrrr_diagnostic_variables: HRRR diagnostic target channels.
        era5_path_template, hrrr_path_template: strftime path patterns, relative to
            era5_dir/hrrr_dir (e.g. "%Y/era5.%Y%m%d.t%Hz.nc").
        era5_lat_name, era5_lon_name, era5_level_name: ERA5 coordinate names.
        orog_var, lsm_var: static variable names in the HRRR files.
        match_era5_lon_to_360: shift HRRR longitudes into [0,360) before regridding
            (ERA5/ARCO is 0..360; HRRR reference is often -180..180).
    """

    def __init__(
        self,
        timestamps: list[str],
        era5_dir: str,
        hrrr_dir: str,
        grid_meta_path: str,
        norm,
        era5_variables: list,
        hrrr_prognostic_variables: list,
        hrrr_diagnostic_variables: list,
        era5_path_template: str,
        hrrr_path_template: str,
        era5_lat_name: str = "latitude",
        era5_lon_name: str = "longitude",
        era5_level_name: str = "level",
        orog_var: str = "orog",
        lsm_var: str = "lsm",
        match_era5_lon_to_360: bool = True,
        rotate_hrrr_winds: bool = True,
        hrrr_wind_pairs: list | None = None,
    ):
        self.timestamps = list(timestamps)
        self.era5_dir = era5_dir
        self.hrrr_dir = hrrr_dir
        self.norm = norm
        self.era5_variables = [_parse_var_spec(s) for s in era5_variables]
        self.hrrr_prognostic_variables = [_parse_var_spec(s) for s in hrrr_prognostic_variables]
        self.hrrr_diagnostic_variables = [_parse_var_spec(s) for s in hrrr_diagnostic_variables]
        self.era5_path_template = era5_path_template
        self.hrrr_path_template = hrrr_path_template
        self.era5_lat_name = era5_lat_name
        self.era5_lon_name = era5_lon_name
        self.era5_level_name = era5_level_name
        self.orog_var = orog_var
        self.lsm_var = lsm_var

        # --- HRRR target grid (2-D lat/lon) — the regridding destination ---
        import netCDF4
        gref = netCDF4.Dataset(grid_meta_path, "r")
        self.hrrr_lat = np.asarray(gref.variables["latitude"][:], dtype=np.float64)   # (H, W)
        hrrr_lon = np.asarray(gref.variables["longitude"][:], dtype=np.float64)        # (H, W)
        gref.close()
        if match_era5_lon_to_360:
            hrrr_lon = np.mod(hrrr_lon, 360.0)
        self.hrrr_lon = hrrr_lon
        self.H, self.W = self.hrrr_lat.shape
        # Flattened destination points (lat, lon) for RegularGridInterpolator.
        self._dst_pts = np.stack([self.hrrr_lat.ravel(), self.hrrr_lon.ravel()], axis=-1)

        # --- Wind rotation: HRRR u/v are GRID-relative, ERA5's are EARTH-relative ---
        # (hrrr_grid_reference.nc: uv_relative_to_grid=1). Without this the model would be
        # trained to map Earth-relative drivers onto grid-relative targets, silently
        # absorbing the Lambert convergence angle (+-23 deg at the edges of CONUS) into its
        # weights, and every wind verification against an Earth-relative source would be
        # measuring that projection artifact as if it were error. See data/wind_rotation.py.
        self.rotate_hrrr_winds = bool(rotate_hrrr_winds)
        self._wind_idx = []
        if self.rotate_hrrr_winds:
            from data.wind_rotation import rotation_terms

            cos_t, sin_t = rotation_terms(self.hrrr_lat, self.hrrr_lon)
            self._cos_t, self._sin_t = cos_t, sin_t
            prog_names = [n for n, _, _ in self.hrrr_prognostic_variables]
            for u_name, v_name in (hrrr_wind_pairs or []):
                if u_name not in prog_names or v_name not in prog_names:
                    raise ValueError(
                        f"hrrr_wind_pairs names ({u_name}, {v_name}) must both appear in "
                        f"hrrr_prognostic_variables {prog_names}"
                    )
                self._wind_idx.append((prog_names.index(u_name), prog_names.index(v_name)))
            if not self._wind_idx:
                log.warning(
                    "rotate_hrrr_winds=True but hrrr_wind_pairs is empty — no winds will be "
                    "rotated. If any target channel is a wind component, this is a bug."
                )

        # --- Static channels (orog normalized to [0,1], lsm binary), cached once ---
        self.statics = self._load_statics()

        log.info(
            f"Era5HrrrDataset: {len(self.timestamps)} timestamps | "
            f"{len(self.era5_variables)} ERA5 cond ch -> "
            f"{len(self.hrrr_prognostic_variables)} prog + "
            f"{len(self.hrrr_diagnostic_variables)} diag HRRR ch | grid {self.H}x{self.W}"
        )

    # ------------------------------------------------------------------ paths
    def _ts_to_dt(self, ts: str) -> datetime:
        return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")

    def _era5_path(self, ts: str) -> str:
        return os.path.join(self.era5_dir, self._ts_to_dt(ts).strftime(self.era5_path_template))

    def _hrrr_path(self, ts: str) -> str:
        return os.path.join(self.hrrr_dir, self._ts_to_dt(ts).strftime(self.hrrr_path_template))

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def _open(path: str):
        """Open a NetCDF file or a zarr store as an xarray Dataset."""
        import xarray as xr
        engine = "zarr" if path.endswith(".zarr") or os.path.isdir(path) else None
        return xr.open_dataset(path, engine=engine)

    def _read_2d(self, ds, name: str, level, transform=None):
        """Select a single 2-D field (optionally at a pressure level) as float32 (Hc, Wc).

        The transform (e.g. the precip log) is applied on the COARSE field, before
        regridding. That ordering is deliberate: interpolating in log space keeps the
        interpolant from smearing a heavy-precip core across neighbouring dry cells, and it
        is what makes the channel comparable to the HRRR log_tp target.
        """
        da = ds[name]
        if level is not None and self.era5_level_name in da.dims:
            da = da.sel({self.era5_level_name: level}, method="nearest")
        da = da.squeeze(drop=True)
        field = np.asarray(da.values, dtype=np.float32)
        if transform is not None:
            field = ERA5_TRANSFORMS[transform](field).astype(np.float32)
        return field

    # ---------------------------------------------------------------- regrid
    def regrid_era5(self, field: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray) -> np.ndarray:
        """Bilinearly regrid a coarse (Hc, Wc) field onto the HRRR grid -> (H, W).

        src_lat/src_lon are the ERA5 1-D coordinate vectors. RegularGridInterpolator
        needs strictly-ascending axes, so descending latitude is flipped.
        """
        from scipy.interpolate import RegularGridInterpolator

        lat, lon = np.asarray(src_lat, np.float64), np.asarray(src_lon, np.float64)
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            field = field[::-1, :]
        interp = RegularGridInterpolator(
            (lat, lon), field, method="linear", bounds_error=False, fill_value=None
        )
        out = interp(self._dst_pts).reshape(self.H, self.W)
        return out.astype(np.float32)

    def rotate_prognostic_winds(self, prog: np.ndarray) -> np.ndarray:
        """Rotate every configured (u, v) channel pair of a (C, H, W) prognostic stack
        from grid-relative to Earth-relative. No-op if rotation is off / no pairs.

        Exposed (not private) because inference has to undo nothing but verification code
        needs to apply exactly the same transform to HRRR truth — sharing this method is
        what keeps the two from drifting apart.
        """
        if not self.rotate_hrrr_winds or not self._wind_idx:
            return prog
        from data.wind_rotation import grid_to_earth

        prog = prog.copy()
        for iu, iv in self._wind_idx:
            prog[iu], prog[iv] = grid_to_earth(prog[iu], prog[iv], self._cos_t, self._sin_t)
        return prog

    def _load_statics(self) -> torch.Tensor:
        """Read orog + lsm from the first HRRR file, normalize, cache as (2, H, W)."""
        import netCDF4
        path = self._hrrr_path(self.timestamps[0])
        ds = netCDF4.Dataset(path, "r")
        orog = torch.from_numpy(np.asarray(ds.variables[self.orog_var][:], np.float32))
        lsm = torch.from_numpy(np.asarray(ds.variables[self.lsm_var][:], np.float32))
        ds.close()
        orog_norm = self.norm.normalize_orog(orog)
        return torch.stack([orog_norm, lsm], dim=0)  # (2, H, W)

    # ------------------------------------------------------------------ item
    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> dict:
        ts = self.timestamps[idx]

        # --- ERA5 conditioning: read coarse drivers, regrid to HRRR grid, z-score ---
        eds = self._open(self._era5_path(ts))
        src_lat = np.asarray(eds[self.era5_lat_name].values, np.float64)
        src_lon = np.mod(np.asarray(eds[self.era5_lon_name].values, np.float64), 360.0)
        era5_ch = []
        for name, level, transform in self.era5_variables:
            coarse = self._read_2d(eds, name, level, transform)
            era5_ch.append(self.regrid_era5(coarse, src_lat, src_lon))
        eds.close()
        era5 = torch.from_numpy(np.stack(era5_ch, axis=0))          # (n_era5, H, W)
        era5 = self.norm.normalize_era5(era5)
        x_cond = torch.cat([era5, self.statics], dim=0)             # (n_era5 + 2, H, W)

        # --- HRRR target: full-field prognostic state + diagnostics, z-score ---
        import netCDF4
        hds = netCDF4.Dataset(self._hrrr_path(ts), "r")
        prog = np.stack(
            [np.asarray(hds.variables[n][:], np.float32).reshape(self.H, self.W)
             for n, _, _ in self.hrrr_prognostic_variables],
            axis=0,
        )
        diag = np.stack(
            [np.asarray(hds.variables[n][:], np.float32).reshape(self.H, self.W)
             for n, _, _ in self.hrrr_diagnostic_variables],
            axis=0,
        )
        hds.close()

        # Rotate HRRR winds grid-relative -> Earth-relative BEFORE normalizing, so that the
        # stats in stats.npz (computed through this same __getitem__) describe the rotated
        # fields the model actually sees. Doing it after normalization would mix the two
        # channels' z-scores and be wrong.
        prog = self.rotate_prognostic_winds(prog)

        target_prognostic = self.norm.normalize_hrrr(torch.from_numpy(prog))
        target_diagnostic = self.norm.normalize_diagnostic(torch.from_numpy(diag))

        return {
            "x_cond": x_cond,
            "target_prognostic": target_prognostic,
            "target_diagnostic": target_diagnostic,
            "timestamp": idx,
        }
