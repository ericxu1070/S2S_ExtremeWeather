"""Figures for the June-July 2026 heat-wave: Observed | Vayuh | GenCast over CONUS.

Rebuilds the three figure families from the Vayuh.ai report with our GenCast 1.0deg forecast
added as a column/line:

    1. peak-period CONUS map   (pooled leads 14-18, mean over 29 Jun-2 Jul)
    2. per-lead map grid        (one figure per peak target date; rows = leads 14-18)
    3. East-US daily time series (one panel per lead)

Plus a combined ``Vayuh_plus_GenCast_Heatwave2026.pdf``.

NOTE on CFSv2: the report's third column is raw CFSv2, but no CFSv2 data ships with this
repo (only the Vayuh forecast h5 + ERA5 via ARCO). CFSv2 is therefore omitted here; the
comparison is Observed (ERA5 truth) | Vayuh | GenCast. All anomalies are vs the 1990-2019
ERA5 climatology used throughout this repo; Vayuh's ``preds_*`` are its own pre-computed
anomalies (shown as delivered).
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

        od = H.obs_dir()
        self.obs = self._to_grid(xr.open_dataset(od / "obs_daily_t2m_anom.nc")["t2m_anom"].load())
        self.obs = self.obs.assign_coords(date=pd.DatetimeIndex(self.obs.date.values))
        self.clim = self._to_grid(xr.open_dataset(od / "clim_daily_t2m.nc")["t2m_clim"].load())
        self.clim = self.clim.assign_coords(date=pd.DatetimeIndex(self.clim.date.values))

        # GenCast per-init daily-mean absolute tmp2m -> ens-mean, keyed by init date.
        self.gc = {}
        for f in sorted(H.forecasts_dir().glob("init_*_daily_t2m_members.nc")):
            init = pd.Timestamp(f.name.split("_")[1])
            da = xr.open_dataset(f)["t2m"].load()
            self.gc[init.normalize()] = self._to_grid(da.mean("member"))       # (lead_day,lat,lon) abs
        self.vayuh_mask = np.isfinite(self.vayuh.isel(lead=0, start_date=0))

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

    def field(self, model: str, target, lead):
        return {"obs": self.obs_field, "vayuh": self.vayuh_field,
                "gencast": self.gencast_field}[model](target, lead) if model != "obs" \
            else self.obs_field(target)


# --------------------------------------------------------------------------- #
# Scoring: latitude-weighted spatial ACC + MAE over finite points
# --------------------------------------------------------------------------- #
def acc_mae(fc: xr.DataArray, obs: xr.DataArray):
    if fc is None or obs is None:
        return np.nan, np.nan
    f = fc.values.ravel(); o = obs.values.ravel()
    w = np.cos(np.deg2rad(fc.lat.values))
    w2 = np.repeat(w[:, None], fc.sizes["lon"], axis=1).ravel()
    m = np.isfinite(f) & np.isfinite(o) & np.isfinite(w2)
    if m.sum() < 5:
        return np.nan, np.nan
    f, o, w2 = f[m], o[m], w2[m]
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
    for feat, kw in ((cfeature.COASTLINE, dict(lw=0.6)),
                     (cfeature.BORDERS, dict(lw=0.5)),
                     (cfeature.STATES, dict(lw=0.3, edgecolor="grey"))):
        try:
            ax.add_feature(feat, **kw)
        except Exception:
            pass


def _map_panel(fig, pos, da, title, ccrs, add_cbar=False):
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    ax.set_extent(C.CONUS_EXTENT, ccrs.PlateCarree())
    m = None
    if da is not None:
        d = _to_180(da)
        m = ax.pcolormesh(d.lon, d.lat, d.values, transform=ccrs.PlateCarree(),
                          cmap=CMAP, vmin=-VMAX, vmax=VMAX, shading="auto")
    _features(ax)
    ax.set_title(title, fontsize=9)
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


def fig_peak_map(data: Data, ccrs):
    obs, vay, gc = (_pool(data, m) for m in ("obs", "vayuh", "gencast"))
    a_v, m_v = acc_mae(vay, obs)
    a_g, m_g = acc_mae(gc, obs)
    fig = plt.figure(figsize=(14.5, 4.4))
    _, _ = _map_panel(fig, (1, 3, 1), obs, "Observed (ERA5)", ccrs)
    _map_panel(fig, (1, 3, 2), vay, f"Vayuh (wk 3-4 pooled)\nACC {a_v:+.2f}  MAE {m_v:.1f}°C", ccrs)
    _, m = _map_panel(fig, (1, 3, 3), gc, f"GenCast (leads 14-18)\nACC {a_g:+.2f}  MAE {m_g:.1f}°C", ccrs)
    fig.suptitle("June/July 2026 NE-US heat wave — peak (29 Jun-2 Jul) tmp2m anomaly, "
                 "subseasonal leads 14-18", y=1.02, fontsize=12)
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
    leads = H.LEADS
    nrows = len(leads)
    fig = plt.figure(figsize=(11, 2.7 * nrows))
    obs = data.obs_field(target)
    lastm = None
    for i, L in enumerate(leads):
        init = target.normalize() - L * DAY
        vay = data.vayuh_field(target, L)
        gc = data.gencast_field(target, L)
        a_v, m_v = acc_mae(vay, obs)
        a_g, m_g = acc_mae(gc, obs)
        _map_panel(fig, (nrows, 3, 3 * i + 1),
                   obs, ("Observed" if i == 0 else "") + f"\nlead {L}d (init {init:%b %d})", ccrs)
        _map_panel(fig, (nrows, 3, 3 * i + 2), vay,
                   f"Vayuh  ACC {a_v:+.2f} MAE {m_v:.1f}", ccrs)
        _, lastm = _map_panel(fig, (nrows, 3, 3 * i + 3), gc,
                              f"GenCast  ACC {a_g:+.2f} MAE {m_g:.1f}", ccrs)
    fig.suptitle(f"CONUS heat-wave forecast anomaly by lead — TARGET {target:%Y-%m-%d (%a)}\n"
                 f"Observed | Vayuh | GenCast", y=1.005, fontsize=12)
    if lastm is not None:
        cax = fig.add_axes([0.25, 0.005, 0.5, 0.012])
        fig.colorbar(lastm, cax=cax, orientation="horizontal", label="tmp2m anomaly (°C)")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    return fig


# --------------------------------------------------------------------------- #
# Figure 3: East-US daily time series per lead
# --------------------------------------------------------------------------- #
def _east_us_mean(da: xr.DataArray):
    if da is None:
        return np.nan
    sub = da.where(da.lon >= EAST_US_LON_MIN)
    w = np.cos(np.deg2rad(sub.lat))
    return float(sub.weighted(w.fillna(0)).mean(("lat", "lon")))


def fig_east_us_series(data: Data):
    targets = H._target_dates()
    leads = H.LEADS
    fig, axes = plt.subplots(len(leads), 1, figsize=(9, 2.2 * len(leads)),
                             sharex=True, squeeze=False)
    for ax, L in zip(axes.ravel(), leads):
        o = [_east_us_mean(data.obs_field(T)) for T in targets]
        v = [_east_us_mean(data.vayuh_field(T, L)) for T in targets]
        g = [_east_us_mean(data.gencast_field(T, L)) for T in targets]
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvspan(H.PEAK_TARGETS[0], H.PEAK_TARGETS[-1], color="red", alpha=0.10,
                   label="heat-wave peak")
        ax.plot(targets, o, "k-o", ms=3, lw=1.8, label="Observed")
        ax.plot(targets, v, color="tab:blue", marker="s", ms=3, lw=1.4, label="Vayuh")
        ax.plot(targets, g, color="tab:green", marker="^", ms=3, lw=1.4, label="GenCast")
        ax.set_ylabel("anomaly °C")
        ax.set_title(f"lead {L}d", fontsize=9, loc="left")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=7, ncol=4, loc="upper left")
    for lbl in axes.ravel()[-1].get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha("right")
    fig.suptitle("East-US (≥ 98°W) mean tmp2m anomaly vs target date — "
                 "Observed | Vayuh | GenCast", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
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
