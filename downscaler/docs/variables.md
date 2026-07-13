# Dataset Variables — ERA5 (1°) and HRRR (3 km)

Two NetCDF datasets used by the downscaler: coarse global **ERA5** input and
high-resolution regional **HRRR** target.

---

## ERA5 — 1.0° global

- **Path:** `~/Vayuh/data/eric/era5_1deg/<year>/`
- **File pattern:** `era5_1deg.YYYYMMDD.tHHz.nc`  (HH ∈ {00, 06, 12, 18})
- **Dimensions:** `time`(1) × `level`(13) × `lat`(181) × `lon`(360)
- **Grid:** 1.0° global — lat −90→90, lon 0→359
- **Pressure levels (hPa):** 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
- **Format:** GenCast 1.0° input; coarsened from native 0.25° by 1-in-4 grid-point subsampling
- **Source:** ERA5 via NCAR RDA ds633.0

**14 data variables total.**

### Upper-air — `(time, level, lat, lon)`, 13 levels each

| Variable | Units | Description |
|---|---|---|
| `geopotential` | m² s⁻² | Height of each pressure surface (g·z) |
| `temperature` | K | Air temperature on pressure levels |
| `u_component_of_wind` | m s⁻¹ | Eastward wind |
| `v_component_of_wind` | m s⁻¹ | Northward wind |
| `vertical_velocity` | Pa s⁻¹ | Pressure-vertical velocity (omega) |
| `specific_humidity` | kg kg⁻¹ | Water-vapor mass fraction |

### Surface / single-level — `(time, lat, lon)`

| Variable | Units | Description |
|---|---|---|
| `2m_temperature` | K | Air temperature at 2 m |
| `mean_sea_level_pressure` | Pa | Pressure reduced to mean sea level |
| `10m_u_component_of_wind` | m s⁻¹ | Eastward wind at 10 m |
| `10m_v_component_of_wind` | m s⁻¹ | Northward wind at 10 m |
| `sea_surface_temperature` | K | Ocean surface skin temperature |
| `total_precipitation_12hr` | m (unlabeled) | Precipitation accumulated over 12 h |

### Static — `(lat, lon)`, time-invariant

| Variable | Units | Description |
|---|---|---|
| `geopotential_at_surface` | m² s⁻² | Surface elevation as geopotential (orography) |
| `land_sea_mask` | 0–1 | Land fraction of each cell (0 sea → 1 land) |

---

## HRRR — 3 km CONUS (v3)

- **Path:** `~/Vayuh/data/moein/hrrr_work/hrrr_nc_v3/<year>/`
- **File pattern:** `hrrr.YYYYMMDD.tHHz.v3.nc`  (HH ∈ {00, 06, 12, 18}), named by valid time
- **Dimensions:** `isobaricInhPa8`(8) / `isobaricInhPa5`(5) × `y`(1059) × `x`(1799)
- **Grid:** 1799 × 1059, 3 km Lambert Conformal, CONUS. `latitude`/`longitude` are 2D arrays.
- **8 pressure levels (hPa):** 1000, 925, 850, 700, 500, 400, 300, 200
- **5 moisture levels (hPa):** 1000, 925, 850, 700, 500
- **Timing:** atmospheric/surface fields are analysis (fxx=0) at time T; precip is 6-h accumulated (T−6h → T)
- **Note:** U/V winds are **grid-relative**, not Earth-relative
- **Source:** HRRR "prs" product (AWS S3)

**44 channels across 12 data variables.**

### Pressure-level — `(isobaricInhPa8, y, x)`, 8 levels each

| Variable (cfgrib) | Units | Channels | Description |
|---|---|---|---|
| `gh` | m | 8 | Geopotential height |
| `t` | K | 8 | Temperature |
| `u` | m/s | 8 | U-wind (grid-relative) |
| `v` | m/s | 8 | V-wind (grid-relative) |

### Moisture — `(isobaricInhPa5, y, x)`, 5 levels

| Variable (cfgrib) | Units | Channels | Description |
|---|---|---|---|
| `q` | kg/kg | 5 | Specific humidity |

### Surface — `(y, x)`

| Variable (cfgrib) | Units | Channels | Description |
|---|---|---|---|
| `t2m` | K | 1 | 2 m temperature |
| `u10` | m/s | 1 | 10 m U-wind |
| `v10` | m/s | 1 | 10 m V-wind |
| `log_sp` | ln(Pa) | 1 | ln(surface pressure) |
| `log_tp` | ln(kg/m²) | 1 | ln(6-h accumulated precip + 1e-5) |

### Static — `(y, x)`, time-invariant

| Variable (cfgrib) | Units | Channels | Description |
|---|---|---|---|
| `orog` | m | 1 | Orography (surface elevation) |
| `lsm` | 0–1 | 1 | Land-sea mask |

**Channel count:** (gh, t, u, v) 4×8 = 32 + q 5 + (t2m, u10, v10, log_sp, log_tp) 5 + (orog, lsm) 2 = **44**.

---

### Quick comparison

| | ERA5 | HRRR |
|---|---|---|
| Resolution | 1.0° (~111 km) global | 3 km CONUS |
| Grid | 181 × 360 (regular lat/lon) | 1059 × 1799 (Lambert Conformal) |
| Pressure levels | 13 | 8 (5 for humidity) |
| Data variables | 14 | 12 (44 channels) |
| Cadence | 6-hourly | 6-hourly |
| Wind convention | Earth-relative | Grid-relative |
| Precip | `total_precipitation_12hr` (m) | `log_tp`, ln(6-h accum) |
| Pressure var | `mean_sea_level_pressure` (Pa) | `log_sp`, ln(surface Pa) |
