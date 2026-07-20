#!/usr/bin/env python
"""Find 90th-percentile-hot CONUS T2m-anomaly events, 2022-2025, and plot them.

Definition (chosen with the user):
  * metric  = daily-mean 2m_temperature ANOMALY vs the 1990-2019 climatology
              (same clim + hour/dayofyear matching convention as
              gencast_s2s.data.true_verif_anom).
  * ranking = cos(lat)-weighted CONUS-mean of that daily anomaly map.
  * events  = calendar days whose CONUS-mean anomaly >= the 90th percentile of
              the full 2022-2025 daily series, with CONSECUTIVE hot days merged
              into one event. Each event is plotted as its event-mean anomaly map.

Two stages, cache-aware:
  --stage cube  : read ERA5 (ARCO) T2m over CONUS for every day 2022-01-01..2025-12-31,
                  build the daily anomaly cube + CONUS-mean series -> one NetCDF.
                  Needs internet (ARCO). Heavy-ish I/O -> run via PBS (develop queue).
  --stage plot  : from the cached cube, pick p90 events, write per-event NetCDFs and
                  CONUS maps + an overview panel. Offline, light -> ok anywhere.
  --stage all   : cube then plot (default).
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import gencast_s2s.config as C
import gencast_s2s.data as D

START = "2022-01-01"
END = "2025-12-31"
HOURS = [0, 6, 12, 18]           # synoptic times; matches clim `hour` axis
PCTL = 90.0

OUT_DIR = C.RUNS / "p90_t2m_events_2022_2025"
CUBE_F = OUT_DIR / "daily_conus_t2m_anom_2022_2025.nc"
EVENTS_DIR = OUT_DIR / "events"
MAPS_DIR = OUT_DIR / "maps"
SUMMARY_CSV = OUT_DIR / "events_summary.csv"


# --------------------------------------------------------------------------- #
def _cos_weights(lat: np.ndarray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat))
    return xr.DataArray(w, dims="lat", coords={"lat": lat})


def build_cube() -> xr.Dataset:
    """Daily-mean T2m anomaly (day, lat, lon) over CONUS + cos-lat CONUS-mean series."""
    days = pd.date_range(START, END, freq="D")
    clim = D._clim_src()                    # (hour, dayofyear, lat, lon), 0.25 CONUS
    lat_c, lon_c = D._arco_lat_lon_coords(conus=True, stride=1)
    assert lat_c.shape[0] == clim.lat.size and lon_c.shape[0] == clim.lon.size, \
        f"grid mismatch: ARCO {lat_c.shape[0]}x{lon_c.shape[0]} vs clim " \
        f"{clim.lat.size}x{clim.lon.size}"

    anom_frames = []
    print(f"[cube] {len(days)} days x {len(HOURS)} hours from ARCO-ERA5 ...", flush=True)
    for i, d in enumerate(days):
        times = [d + pd.Timedelta(hours=h) for h in HOURS]
        # daily-mean observed T2m over the 4 synoptic hours
        obs = np.mean([D.arco_read("2m_temperature", t, conus=True, stride=1)
                       for t in times], axis=0)
        # daily-mean climatology over the SAME hours / this dayofyear
        cl = clim.sel(hour=xr.DataArray(HOURS, dims="t"),
                      dayofyear=int(d.dayofyear)).mean("t").values
        anom_frames.append(obs - cl)
        D.release_stores()
        gc.collect()
        if i % 60 == 0:
            print(f"  {i+1}/{len(days)}  {d.date()}", flush=True)

    anom = xr.DataArray(
        np.stack(anom_frames).astype("float32"),
        dims=("day", "lat", "lon"),
        coords={"day": days, "lat": lat_c, "lon": lon_c},
        name="t2m_anom",
        attrs={"units": "K", "long_name": "daily-mean 2m T anomaly vs 1990-2019 clim"},
    )
    w = _cos_weights(lat_c)
    conus_mean = anom.weighted(w).mean(("lat", "lon"))
    conus_mean.name = "conus_mean_t2m_anom"
    conus_mean.attrs = {"units": "K", "long_name": "cos(lat)-weighted CONUS-mean daily t2m anomaly"}

    ds = xr.Dataset({"t2m_anom": anom, "conus_mean_t2m_anom": conus_mean})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    D._atomic_to_netcdf(ds, CUBE_F)
    print(f"[cube] wrote {CUBE_F}", flush=True)
    return ds


# --------------------------------------------------------------------------- #
def find_events(series: xr.DataArray) -> tuple[float, list[dict]]:
    """Merge consecutive above-p90 days into events."""
    thr = float(np.percentile(series.values, PCTL))
    hot = series.values >= thr
    days = pd.DatetimeIndex(series["day"].values)

    events = []
    i = 0
    n = len(hot)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hot[j + 1] and (days[j + 1] - days[j]).days == 1:
            j += 1
        idx = np.arange(i, j + 1)
        vals = series.values[idx]
        pk = idx[int(np.argmax(vals))]
        events.append({
            "start": days[i], "end": days[j], "n_days": int(j - i + 1),
            "idx": idx, "peak_day": days[pk],
            "peak_anom_K": float(series.values[pk]),
            "mean_anom_K": float(vals.mean()),
        })
        i = j + 1
    return thr, events


def plot_event_map(ax, da: xr.DataArray, title: str, vmax: float):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    lon = da.lon.values.copy()
    lon = np.where(lon > 180, lon - 360, lon)      # 0..360 -> -180..180 for cartopy
    order = np.argsort(lon)
    m = ax.pcolormesh(lon[order], da.lat.values, da.values[:, order],
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      transform=ccrs.PlateCarree(), shading="auto")
    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    ax.add_feature(cfeature.BORDERS, lw=0.4)
    ax.add_feature(cfeature.STATES, lw=0.25, edgecolor="0.5")
    ax.set_extent(C.CONUS_EXTENT, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=9)
    return m


def do_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    ds = xr.open_dataset(CUBE_F)
    series = ds["conus_mean_t2m_anom"]
    anom = ds["t2m_anom"]
    thr, events = find_events(series)
    print(f"[plot] p{PCTL:.0f} threshold = {thr:.3f} K ; {len(events)} events "
          f"({int((series.values>=thr).sum())} hot days)", flush=True)

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    event_maps = []
    proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=38)
    for k, ev in enumerate(events, 1):
        emap = anom.isel(day=ev["idx"]).mean("day")
        tag = f"{ev['start']:%Y%m%d}_{ev['end']:%Y%m%d}"
        name = f"event_{k:02d}_{tag}"
        # save per-event map NetCDF
        emap.to_dataset(name="t2m_anom").assign_attrs(
            start=str(ev["start"].date()), end=str(ev["end"].date()),
            n_days=ev["n_days"], peak_day=str(ev["peak_day"].date()),
            peak_anom_K=ev["peak_anom_K"], mean_anom_K=ev["mean_anom_K"],
        ).to_netcdf(EVENTS_DIR / f"{name}.nc")
        event_maps.append((name, ev, emap))
        rows.append({
            "event": k, "start": ev["start"].date(), "end": ev["end"].date(),
            "n_days": ev["n_days"], "peak_day": ev["peak_day"].date(),
            "peak_anom_K": round(ev["peak_anom_K"], 3),
            "mean_anom_K": round(ev["mean_anom_K"], 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(df.to_string(index=False), flush=True)

    vmax = max(3.0, float(np.percentile(np.abs(anom.values), 99)))

    # per-event maps
    for name, ev, emap in event_maps:
        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        m = plot_event_map(
            ax, emap,
            f"{ev['start']:%Y-%m-%d} to {ev['end']:%Y-%m-%d}  "
            f"({ev['n_days']}d)\nCONUS-mean +{ev['mean_anom_K']:.2f} K, "
            f"peak +{ev['peak_anom_K']:.2f} K", vmax)
        fig.colorbar(m, ax=ax, shrink=0.7, label="T2m anomaly (K)")
        fig.tight_layout()
        fig.savefig(MAPS_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # overview panel
    ncol = 4
    nrow = int(np.ceil(len(event_maps) / ncol))
    fig = plt.figure(figsize=(4 * ncol, 3 * nrow))
    for i, (name, ev, emap) in enumerate(event_maps, 1):
        ax = fig.add_subplot(nrow, ncol, i, projection=proj)
        m = plot_event_map(ax, emap,
                           f"{ev['start']:%Y-%m-%d} ({ev['n_days']}d)\n"
                           f"+{ev['mean_anom_K']:.2f} K", vmax)
    fig.suptitle(
        f"CONUS 90th-pct daily-mean T2m-anomaly events, {START[:4]}-{END[:4]}\n"
        f"threshold = +{thr:.2f} K CONUS-mean; {len(events)} events", fontsize=12)
    cax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    fig.colorbar(m, cax=cax, orientation="horizontal", label="T2m anomaly (K)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(MAPS_DIR / "overview_all_events.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {len(events)} event maps + overview to {MAPS_DIR}", flush=True)

    # time-series diagnostic
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(pd.DatetimeIndex(series["day"].values), series.values, lw=0.7, color="0.4")
    ax.axhline(thr, color="crimson", ls="--", lw=1, label=f"p{PCTL:.0f} = +{thr:.2f} K")
    for ev in events:
        ax.axvspan(ev["start"], ev["end"], color="orange", alpha=0.35)
    ax.set_ylabel("CONUS-mean t2m anom (K)")
    ax.set_title(f"Daily CONUS-mean T2m anomaly {START[:4]}-{END[:4]} (shaded = p{PCTL:.0f} events)")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.99, 0.97, "de-meaned with 1990-2019 climatology",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
    fig.tight_layout()
    fig.savefig(MAPS_DIR / "conus_mean_timeseries.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cube", "plot", "all"], default="all")
    args = ap.parse_args()

    if args.stage in ("cube", "all"):
        if CUBE_F.exists():
            print(f"[cube] cache hit: {CUBE_F} (skipping build)", flush=True)
        else:
            build_cube()
    if args.stage in ("plot", "all"):
        do_plot()


if __name__ == "__main__":
    main()
