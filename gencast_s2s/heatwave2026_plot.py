"""Figures for the June-July 2026 heat-wave: Observed | Vayuh | GenCast over CONUS.

Rebuilds the three figure families from the Vayuh.ai report with our GenCast 1.0deg forecast
added as a column/line:

    1. peak-period CONUS map   (pooled leads 14-18, mean over 29 Jun-2 Jul)
       columns: Observed | Vayuh | GenCast mean | warmest member | coldest member
    2. per-lead map grid        (one figure per peak target date; rows = leads 14-18)
       same five columns
    3. East-US daily time series (absolute °C + anomaly; one panel-row per lead)

Plus a combined ``Vayuh_plus_GenCast_Heatwave2026.pdf``.

NOTE on CFSv2: the report's third column is raw CFSv2, but no CFSv2 data ships with this
repo (only the Vayuh forecast h5 + ERA5 via ARCO). CFSv2 is therefore omitted here; the
comparison is Observed (ERA5 truth) | Vayuh | GenCast. All anomalies are vs the 1990-2019
ERA5 climatology used throughout this repo; Vayuh's ``preds_*`` are its own pre-computed
anomalies (shown as delivered).

Maps and ACC/MAE are clipped to contiguous-US land (Natural Earth admin-0 USA ∩ CONUS
box) so Mexico and ocean cells do not colour the panels or enter the scores. Extreme
members are the ensemble members with the highest / lowest latitude-weighted US-mean
T2m anomaly (same selection idea as ``combinedextremepdf.png``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from . import config as C                            # noqa: E402
from . import heatwave2026 as H                      # noqa: E402

VMAX = 6.0
CMAP = "RdBu_r"
EAST_US_LON_MIN = 262.0        # ~98 W: "East-US" region for the time series
DAY = pd.Timedelta(days=1)
K2C = 273.15


# --------------------------------------------------------------------------- #
# Contiguous-US land mask (Natural Earth admin-0 ∩ CONUS box)
# --------------------------------------------------------------------------- #
def _conus_usa_geom():
    """Unary union of contiguous-US state polygons from the local cartopy cache."""
    from pathlib import Path

    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union

    # Prefer a local cartopy cache path so we never hit naturalearth.s3 (Derecho
    # login nodes often cannot download). Fall back to natural_earth() only if
    # the cached admin-1 file is missing.
    cached = (Path.home() / ".local/share/cartopy/shapefiles/natural_earth/cultural"
              / "ne_50m_admin_1_states_provinces_lakes.shp")
    if cached.is_file():
        fname = str(cached)
    else:
        fname = shpreader.natural_earth(
            resolution="50m", category="cultural",
            name="admin_1_states_provinces_lakes")

    geoms = []
    for rec in shpreader.Reader(fname).records():
        attrs = rec.attributes
        admin = attrs.get("admin") or attrs.get("ADMIN") or ""
        if admin != "United States of America":
            continue
        name = attrs.get("name") or attrs.get("NAME") or ""
        if name in ("Alaska", "Hawaii"):
            continue
        geoms.append(rec.geometry)
    if not geoms:
        raise RuntimeError(f"no CONUS state geometries in {fname}")
    return unary_union(geoms)


def _us_land_mask(lat: xr.DataArray, lon: xr.DataArray, usa_geom=None) -> xr.DataArray:
    """Boolean (lat, lon) mask: True on contiguous-US land only.

    Built from the cached Natural Earth admin-1 states shapefile (no network): keep
    US states inside the CONUS box, drop Alaska / Hawaii / territories. Mexico and
    ocean cells stay False even when the rectangular CONUS box covers them.
    """
    from shapely.geometry import Point
    from shapely.prepared import prep

    usa_raw = usa_geom if usa_geom is not None else _conus_usa_geom()
    usa = prep(usa_raw)
    west, east, south, north = C.CONUS_EXTENT

    lon180 = (((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0)
    latv = np.asarray(lat, dtype=float)
    Lon, Lat = np.meshgrid(lon180, latv)
    mask = np.zeros(Lon.shape, dtype=bool)
    for i in range(Lon.shape[0]):
        y = float(Lat[i, 0])
        if y < south or y > north:
            continue
        for j in range(Lon.shape[1]):
            x = float(Lon[i, j])
            if x < west or x > east:
                continue
            if usa.contains(Point(x, y)):
                mask[i, j] = True
    return xr.DataArray(mask, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def _apply_us(da: xr.DataArray | None, us_mask: xr.DataArray) -> xr.DataArray | None:
    if da is None:
        return None
    return da.where(us_mask)


# --------------------------------------------------------------------------- #
# Data loading (everything aligned onto the Vayuh 1 deg grid)
# --------------------------------------------------------------------------- #
class Data:
    def __init__(self):
        r = H.root()
        self.vayuh = xr.open_dataset(r / "vayuh_forecast.nc")["pred"].load()   # (lead,start_date,lat,lon)
        self.vayuh = self.vayuh.assign_coords(start_date=pd.DatetimeIndex(self.vayuh.start_date.values))
        self.grid_lat = self.vayuh.lat
        self.grid_lon = self.vayuh.lon
        self.usa_geom = _conus_usa_geom()
        self.us_mask = _us_land_mask(self.grid_lat, self.grid_lon, usa_geom=self.usa_geom)
        print(f"[plot] US land mask: {int(self.us_mask.sum())} / "
              f"{self.us_mask.size} cells on Vayuh grid")

        od = H.obs_dir()
        self.obs = self._to_grid(xr.open_dataset(od / "obs_daily_t2m_anom.nc")["t2m_anom"].load())
        self.obs = self.obs.assign_coords(date=pd.DatetimeIndex(self.obs.date.values))
        self.clim = self._to_grid(xr.open_dataset(od / "clim_daily_t2m.nc")["t2m_clim"].load())
        self.clim = self.clim.assign_coords(date=pd.DatetimeIndex(self.clim.date.values))
        self.obs_abs = self._to_grid(xr.open_dataset(od / "obs_daily_t2m.nc")["t2m"].load())
        self.obs_abs = self.obs_abs.assign_coords(date=pd.DatetimeIndex(self.obs_abs.date.values))

        # GenCast per-init daily-mean absolute tmp2m: ens-mean + full member cube.
        self.gc = {}
        self.gc_members = {}
        for f in sorted(H.forecasts_dir().glob("init_*_daily_t2m_members.nc")):
            init = pd.Timestamp(f.name.split("_")[1]).normalize()
            da = self._to_grid(xr.open_dataset(f)["t2m"].load())   # (member, lead_day, lat, lon)
            self.gc_members[init] = da
            self.gc[init] = da.mean("member")

    def _to_grid(self, da: xr.DataArray) -> xr.DataArray:
        out = da.sel(lat=self.grid_lat, lon=self.grid_lon, method="nearest")
        return out.assign_coords(lat=self.grid_lat, lon=self.grid_lon)

    # -- field accessors: anomaly DataArray(lat,lon) on the Vayuh grid, or None -----------
    def obs_field(self, target: pd.Timestamp):
        target = pd.Timestamp(target).normalize()
        if target not in pd.DatetimeIndex(self.obs.date.values):
            return None
        return self.obs.sel(date=target)

    def vayuh_field(self, target: pd.Timestamp, lead: int):
        sd = pd.Timestamp(target).normalize() - lead * DAY
        if int(lead) not in [int(x) for x in self.vayuh.lead.values]:
            return None
        if sd not in pd.DatetimeIndex(self.vayuh.start_date.values):
            return None
        return self.vayuh.sel(lead=int(lead), start_date=sd)

    def gencast_field(self, target: pd.Timestamp, lead: int):
        target = pd.Timestamp(target).normalize()
        init = target - lead * DAY
        da = self.gc.get(init)
        if da is None or int(lead) not in [int(x) for x in da.lead_day.values]:
            return None
        clim = self.clim.sel(date=target) if target in pd.DatetimeIndex(self.clim.date.values) else None
        if clim is None:
            return None
        return da.sel(lead_day=int(lead)) - clim        # absolute -> anomaly

    def gencast_members_field(self, target: pd.Timestamp, lead: int):
        """Per-member anomaly (member, lat, lon), or None."""
        target = pd.Timestamp(target).normalize()
        init = target - lead * DAY
        da = self.gc_members.get(init)
        if da is None or int(lead) not in [int(x) for x in da.lead_day.values]:
            return None
        clim = self.clim.sel(date=target) if target in pd.DatetimeIndex(self.clim.date.values) else None
        if clim is None:
            return None
        return da.sel(lead_day=int(lead)) - clim

    def gencast_extreme_fields(self, target: pd.Timestamp, lead: int):
        """Warmest / coldest member by latitude-weighted US-mean anomaly.

        Returns ``(hot, hot_idx, cold, cold_idx)``; any element may be None if the
        ensemble is unavailable.
        """
        fc = self.gencast_members_field(target, lead)
        if fc is None:
            return None, None, None, None
        w = np.cos(np.deg2rad(fc.lat))
        mean = fc.where(self.us_mask).weighted(w).mean(("lat", "lon"))
        hot_i = int(mean.argmax("member"))
        cold_i = int(mean.argmin("member"))
        return (fc.isel(member=hot_i), hot_i,
                fc.isel(member=cold_i), cold_i)

    def gencast_bestfit_field(self, target: pd.Timestamp, lead: int):
        """Ensemble member closest to observed by latitude-weighted US-land MAE.

        Returns ``(field, member_idx)``; both None if the ensemble or obs are missing.
        """
        fc = self.gencast_members_field(target, lead)
        obs = self.obs_field(target)
        if fc is None or obs is None:
            return None, None
        w = np.cos(np.deg2rad(fc.lat))
        mae = (np.abs(fc - obs).where(self.us_mask)
               .weighted(w).mean(("lat", "lon")))
        best_i = int(mae.argmin("member"))
        return fc.isel(member=best_i), best_i

    # -- absolute-temperature accessors (Kelvin) for the °C series column -----------------
    def clim_field(self, target: pd.Timestamp):
        target = pd.Timestamp(target).normalize()
        if target not in pd.DatetimeIndex(self.clim.date.values):
            return None
        return self.clim.sel(date=target)

    def obs_abs_field(self, target: pd.Timestamp):
        target = pd.Timestamp(target).normalize()
        if target not in pd.DatetimeIndex(self.obs_abs.date.values):
            return None
        return self.obs_abs.sel(date=target)

    def gencast_abs_field(self, target: pd.Timestamp, lead: int):
        target = pd.Timestamp(target).normalize()
        da = self.gc.get(target - lead * DAY)
        if da is None or int(lead) not in [int(x) for x in da.lead_day.values]:
            return None
        return da.sel(lead_day=int(lead))                     # ens-mean absolute (K)

    def vayuh_abs_field(self, target: pd.Timestamp, lead: int):
        # Vayuh delivers anomalies only; reconstruct absolute = anomaly + 1990-2019 ERA5
        # clim (Vayuh's own climatology is unknown, so this carries a small clim-basis bias).
        an = self.vayuh_field(target, lead)
        cl = self.clim_field(target)
        if an is None or cl is None:
            return None
        return an + cl                                        # degC + K -> K

    def field(self, model: str, target, lead):
        return {"obs": self.obs_field, "vayuh": self.vayuh_field,
                "gencast": self.gencast_field}[model](target, lead) if model != "obs" \
            else self.obs_field(target)


# --------------------------------------------------------------------------- #
# Scoring: latitude-weighted spatial ACC + MAE over US land only
# --------------------------------------------------------------------------- #
def acc_mae(fc: xr.DataArray, obs: xr.DataArray, us_mask: xr.DataArray | None = None):
    if fc is None or obs is None:
        return np.nan, np.nan
    f = fc.values
    o = obs.values
    w = np.cos(np.deg2rad(fc.lat.values))
    w2 = np.repeat(w[:, None], fc.sizes["lon"], axis=1)
    m = np.isfinite(f) & np.isfinite(o) & np.isfinite(w2)
    if us_mask is not None:
        m = m & np.asarray(us_mask.values, dtype=bool)
    f, o, w2 = f[m], o[m], w2[m]
    if f.size < 5:
        return np.nan, np.nan
    mae = float(np.sum(w2 * np.abs(f - o)) / np.sum(w2))
    denom = np.sqrt(np.sum(w2 * f * f) * np.sum(w2 * o * o))
    acc = float(np.sum(w2 * f * o) / denom) if denom > 0 else np.nan
    return acc, mae


# --------------------------------------------------------------------------- #
# Cartopy panel
# --------------------------------------------------------------------------- #
def _to_180(da):
    return da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")


def _features(ax):
    import cartopy.feature as cfeature
    for feat, kw in ((cfeature.COASTLINE, dict(lw=0.6, zorder=3)),
                     (cfeature.BORDERS, dict(lw=0.5, zorder=3)),
                     (cfeature.STATES, dict(lw=0.3, edgecolor="grey", zorder=3))):
        try:
            ax.add_feature(feat, **kw)
        except Exception:
            pass


def _clip_to_geom(mesh, ax, geom):
    """Clip a pcolormesh to the (lon/lat) USA border polygon.

    Keeps the colour on contiguous-US land only while leaving Mexico / Canada / ocean
    visible underneath (as coastline + border outlines) instead of overpainting them
    white. The clip follows the actual coastline/border and the Great-Lakes holes, so
    the panel edge is crisp rather than blocky whole-cell staircases.
    """
    from cartopy.mpl.patch import geos_to_path
    from matplotlib.path import Path

    paths = geos_to_path(geom)
    if not paths:
        return
    clip = Path.make_compound_path(*paths)
    mesh.set_clip_path(clip, transform=ax.transData)


def _map_panel(fig, pos, da, title, ccrs, us_mask=None, usa_geom=None):
    """Draw one CONUS map panel: US data clipped to the US border, neighbours kept."""
    _ = us_mask  # scoring uses the mask; the panel clips to usa_geom for a crisp edge
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    ax.set_extent(C.CONUS_EXTENT, ccrs.PlateCarree())
    ax.set_facecolor("white")
    m = None
    if da is not None:
        d = _to_180(da)
        m = ax.pcolormesh(d.lon, d.lat, d.values, transform=ccrs.PlateCarree(),
                          cmap=CMAP, vmin=-VMAX, vmax=VMAX, shading="auto",
                          zorder=1)
        if usa_geom is not None:
            _clip_to_geom(m, ax, usa_geom)
    _features(ax)
    ax.set_title(title, fontsize=8)
    return ax, m


# --------------------------------------------------------------------------- #
# Figure 1: peak-period CONUS map (pooled leads, mean over peak targets)
# --------------------------------------------------------------------------- #
def _pool(data: Data, model: str):
    """Mean field over (peak targets x leads 14-18) for a model; obs is lead-independent."""
    fields = []
    for T in H.PEAK_TARGETS:
        if model == "obs":
            f = data.obs_field(T)
            if f is not None:
                fields.append(f)
        else:
            for L in H.LEADS:
                f = data.field(model, T, L)
                if f is not None:
                    fields.append(f)
    if not fields:
        return None
    return xr.concat(fields, dim="s").mean("s")


def _pool_extremes(data: Data):
    """Pool warmest / coldest / best-fit member fields over (peak targets x leads).

    Members are not aligned across inits, so for each (T, L) we pick the hot / cold /
    best-fit member of that ensemble and average those selected fields (mirrors
    ens-mean pooling). "best-fit" = member closest to that day's observed (US-land MAE).
    """
    hot_fields, cold_fields, best_fields = [], [], []
    for T in H.PEAK_TARGETS:
        for L in H.LEADS:
            hot, _, cold, _ = data.gencast_extreme_fields(T, L)
            best, _ = data.gencast_bestfit_field(T, L)
            if hot is not None:
                hot_fields.append(hot)
            if cold is not None:
                cold_fields.append(cold)
            if best is not None:
                best_fields.append(best)
    hot = xr.concat(hot_fields, dim="s").mean("s") if hot_fields else None
    cold = xr.concat(cold_fields, dim="s").mean("s") if cold_fields else None
    best = xr.concat(best_fields, dim="s").mean("s") if best_fields else None
    return hot, cold, best


def fig_peak_map(data: Data, ccrs):
    us = data.us_mask
    geom = data.usa_geom
    obs, vay, gc = (_pool(data, m) for m in ("obs", "vayuh", "gencast"))
    hot, cold, best = _pool_extremes(data)
    a_v, m_v = acc_mae(vay, obs, us)
    a_g, m_g = acc_mae(gc, obs, us)
    a_b, m_b = acc_mae(best, obs, us)
    a_h, m_h = acc_mae(hot, obs, us)
    a_c, m_c = acc_mae(cold, obs, us)
    fig = plt.figure(figsize=(26.4, 4.4))
    _map_panel(fig, (1, 6, 1), obs, "Observed (ERA5)", ccrs, us, geom)
    _map_panel(fig, (1, 6, 2), vay,
               f"Vayuh (wk 3-4 pooled)\nACC {a_v:+.2f}  MAE {m_v:.1f}°C", ccrs, us, geom)
    _map_panel(fig, (1, 6, 3), gc,
               f"GenCast mean (leads 14-18)\nACC {a_g:+.2f}  MAE {m_g:.1f}°C", ccrs, us, geom)
    _map_panel(fig, (1, 6, 4), best,
               f"GenCast best-fit member\nACC {a_b:+.2f}  MAE {m_b:.1f}°C", ccrs, us, geom)
    _map_panel(fig, (1, 6, 5), hot,
               f"GenCast warmest member\nACC {a_h:+.2f}  MAE {m_h:.1f}°C", ccrs, us, geom)
    _, m = _map_panel(fig, (1, 6, 6), cold,
                      f"GenCast coldest member\nACC {a_c:+.2f}  MAE {m_c:.1f}°C", ccrs, us, geom)
    fig.suptitle("June/July 2026 NE-US heat wave — peak (29 Jun-2 Jul) tmp2m anomaly, "
                 "subseasonal leads 14-18 (US land only)", y=1.02, fontsize=12)
    cax = fig.add_axes([0.25, 0.02, 0.5, 0.03])
    fig.colorbar(m, cax=cax, orientation="horizontal",
                 label="tmp2m anomaly (°C)")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# --------------------------------------------------------------------------- #
# Figure 2: per-lead map grid for one target date
# --------------------------------------------------------------------------- #
def fig_per_lead_grid(data: Data, target: pd.Timestamp, ccrs):
    target = pd.Timestamp(target)
    us = data.us_mask
    geom = data.usa_geom
    leads = H.LEADS
    nrows = len(leads)
    fig = plt.figure(figsize=(21.6, 2.7 * nrows))
    obs = data.obs_field(target)
    lastm = None
    for i, L in enumerate(leads):
        init = target.normalize() - L * DAY
        vay = data.vayuh_field(target, L)
        gc = data.gencast_field(target, L)
        best, best_i = data.gencast_bestfit_field(target, L)
        hot, hot_i, cold, cold_i = data.gencast_extreme_fields(target, L)
        a_v, m_v = acc_mae(vay, obs, us)
        a_g, m_g = acc_mae(gc, obs, us)
        a_b, m_b = acc_mae(best, obs, us)
        a_h, m_h = acc_mae(hot, obs, us)
        a_c, m_c = acc_mae(cold, obs, us)
        best_lab = f"#{best_i}" if best_i is not None else "?"
        hot_lab = f"#{hot_i}" if hot_i is not None else "?"
        cold_lab = f"#{cold_i}" if cold_i is not None else "?"
        _map_panel(fig, (nrows, 6, 6 * i + 1),
                   obs, ("Observed" if i == 0 else "") + f"\nlead {L}d (init {init:%b %d})",
                   ccrs, us, geom)
        _map_panel(fig, (nrows, 6, 6 * i + 2), vay,
                   f"Vayuh  ACC {a_v:+.2f} MAE {m_v:.1f}", ccrs, us, geom)
        _map_panel(fig, (nrows, 6, 6 * i + 3), gc,
                   f"GenCast mean  ACC {a_g:+.2f} MAE {m_g:.1f}", ccrs, us, geom)
        _map_panel(fig, (nrows, 6, 6 * i + 4), best,
                   f"Best-fit {best_lab}  ACC {a_b:+.2f} MAE {m_b:.1f}", ccrs, us, geom)
        _map_panel(fig, (nrows, 6, 6 * i + 5), hot,
                   f"Warmest {hot_lab}  ACC {a_h:+.2f} MAE {m_h:.1f}", ccrs, us, geom)
        _, lastm = _map_panel(fig, (nrows, 6, 6 * i + 6), cold,
                              f"Coldest {cold_lab}  ACC {a_c:+.2f} MAE {m_c:.1f}",
                              ccrs, us, geom)
    fig.suptitle(f"CONUS heat-wave forecast anomaly by lead — TARGET {target:%Y-%m-%d (%a)}\n"
                 f"Observed | Vayuh | GenCast mean | best-fit | warmest | coldest  (US land only)",
                 y=1.005, fontsize=12)
    if lastm is not None:
        cax = fig.add_axes([0.25, 0.005, 0.5, 0.012])
        fig.colorbar(lastm, cax=cax, orientation="horizontal", label="tmp2m anomaly (°C)")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    return fig


# --------------------------------------------------------------------------- #
# Figure 3: East-US daily time series per lead
# --------------------------------------------------------------------------- #
def _east_us_mean(da: xr.DataArray, us_mask: xr.DataArray | None = None):
    if da is None:
        return np.nan
    sub = da.where(da.lon >= EAST_US_LON_MIN)
    if us_mask is not None:
        sub = sub.where(us_mask)
    w = np.cos(np.deg2rad(sub.lat))
    return float(sub.weighted(w.fillna(0)).mean(("lat", "lon")))


def _toC(vals):
    return [v - K2C if np.isfinite(v) else np.nan for v in vals]


def fig_east_us_series(data: Data):
    """Two columns: left = absolute tmp2m (°C, read accuracy in real units); right = anomaly.
    Rows = leads 14-18. Observed | Vayuh | GenCast (+ climatology on the absolute column)."""
    targets = H._target_dates()
    leads = H.LEADS
    us = data.us_mask
    fig, axes = plt.subplots(len(leads), 2, figsize=(13.5, 2.2 * len(leads)),
                             sharex=True, squeeze=False)
    for row, L in enumerate(leads):
        axA, axB = axes[row, 0], axes[row, 1]
        for ax in (axA, axB):
            ax.axvspan(H.PEAK_TARGETS[0], H.PEAK_TARGETS[-1], color="red", alpha=0.10,
                       label="heat-wave peak")
            ax.grid(alpha=0.25)

        # left: absolute °C (with climatology dashed)
        oc = _toC([_east_us_mean(data.obs_abs_field(T), us) for T in targets])
        cc = _toC([_east_us_mean(data.clim_field(T), us) for T in targets])
        vc = _toC([_east_us_mean(data.vayuh_abs_field(T, L), us) for T in targets])
        gcc = _toC([_east_us_mean(data.gencast_abs_field(T, L), us) for T in targets])
        axA.plot(targets, cc, color="grey", ls="--", lw=1.2, label="Climatology")
        axA.plot(targets, oc, "k-o", ms=3, lw=1.8, label="Observed")
        axA.plot(targets, vc, color="tab:blue", marker="s", ms=3, lw=1.4, label="Vayuh")
        axA.plot(targets, gcc, color="tab:green", marker="^", ms=3, lw=1.4, label="GenCast")
        axA.set_ylabel("°C")
        axA.set_title(f"lead {L}d — absolute °C", fontsize=9, loc="left")

        # right: anomaly
        o = [_east_us_mean(data.obs_field(T), us) for T in targets]
        v = [_east_us_mean(data.vayuh_field(T, L), us) for T in targets]
        g = [_east_us_mean(data.gencast_field(T, L), us) for T in targets]
        axB.axhline(0, color="grey", lw=0.6)
        axB.plot(targets, o, "k-o", ms=3, lw=1.8, label="Observed")
        axB.plot(targets, v, color="tab:blue", marker="s", ms=3, lw=1.4, label="Vayuh")
        axB.plot(targets, g, color="tab:green", marker="^", ms=3, lw=1.4, label="GenCast")
        axB.set_ylabel("anomaly °C")
        axB.set_title(f"lead {L}d — anomaly", fontsize=9, loc="left")

    axes[0, 0].legend(fontsize=7, ncol=2, loc="upper left")
    for ax in (axes[-1, 0], axes[-1, 1]):
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45); lbl.set_ha("right")
    fig.suptitle("East-US (≥ 98°W) mean tmp2m vs target date — absolute °C (left) & anomaly (right)\n"
                 "Observed | Vayuh | GenCast  (US land only)", y=0.998, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# --------------------------------------------------------------------------- #
def make_all():
    import cartopy.crs as ccrs
    data = Data()
    outdir = H.figures_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[plot] {len(data.gc)} GenCast inits loaded: "
          f"{sorted(str(k.date()) for k in data.gc)}")

    figs = []
    f1 = fig_peak_map(data, ccrs)
    f1.savefig(outdir / "peak_map.png", dpi=140, bbox_inches="tight")
    figs.append(("peak_map", f1))

    grid_figs = []
    for T in H.PEAK_TARGETS:
        fg = fig_per_lead_grid(data, T, ccrs)
        fg.savefig(outdir / f"per_lead_grid_{T:%Y%m%d}.png", dpi=130, bbox_inches="tight")
        grid_figs.append((f"per_lead_grid_{T:%Y%m%d}", fg))

    f3 = fig_east_us_series(data)
    f3.savefig(outdir / "east_us_series.png", dpi=140, bbox_inches="tight")

    pdf_path = H.root() / "Vayuh_plus_GenCast_Heatwave2026.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(f1, bbox_inches="tight")
        for _, fg in grid_figs:
            pdf.savefig(fg, bbox_inches="tight")
        pdf.savefig(f3, bbox_inches="tight")
    for _, fg in [("east", f3)] + grid_figs + figs:
        plt.close(fg)
    print(f"[plot] wrote figures -> {outdir}")
    print(f"[plot] combined PDF -> {pdf_path}")
