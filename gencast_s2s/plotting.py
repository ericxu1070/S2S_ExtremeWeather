"""Figures: week-N T2m anomaly PDFs, CONUS cartopy maps (truth | mean | error), scoring.

Reads the saved per-member forecasts (``runs/week{N}/forecasts``) and the shared observed
truth (``runs/observations``) and writes PNGs + ``scores.csv`` under
``runs/week{N}/figures/{pdf,maps,scores}``.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")                      # headless / batch
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
import xarray as xr                        # noqa: E402

from . import config as C                  # noqa: E402

YMIN = math.exp(-5)                        # floor for the log-y PDF axis


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _truth(name) -> xr.DataArray:
    return xr.open_dataset(C.OBS_DIR / f"{name}_verif_t2m_anom.nc")["t2m_anom"]


def _hrrr_truth(name):
    """Observed HRRR week-mean T2m anomaly overlay (built by gencast_s2s.hrrr), or None."""
    p = C.OBS_DIR / f"{name}_hrrr_verif_t2m_anom.nc"
    return xr.open_dataset(p)["t2m_anom"] if p.exists() else None


def _forecast(name, weeks, model="gencast"):
    p = C.forecasts_dir(weeks) / f"{name}_{model}_verif_t2m_anom_members.nc"
    return xr.open_dataset(p)["t2m_anom"] if p.exists() else None


def _to_180(da: xr.DataArray) -> xr.DataArray:
    """Map a 0..360 longitude axis to -180..180 for cartopy, sorted ascending."""
    return da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")


def _events_with_forecast(weeks, model="gencast"):
    out = []
    for name, peak in C.EVENTS.items():
        fc = _forecast(name, weeks, model=model)
        if fc is not None and (C.OBS_DIR / f"{name}_verif_t2m_anom.nc").exists():
            out.append(name)
    return out


def _truth_on_fc(truth, fc):
    """Observed anomaly nearest-neighbour sampled onto the forecast (CONUS) grid."""
    obs = truth.sel(lat=fc["lat"], lon=fc["lon"], method="nearest")
    return obs.assign_coords(lat=fc["lat"], lon=fc["lon"])


def _best_member(fc, truth):
    """The single ensemble member closest to observations over CONUS.

    "Closest" = smallest latitude-weighted RMSE of the week-N T2m anomaly field
    against the observed anomaly, computed on the forecast grid. Returns
    ``(member_field, member_index, rmse)``.
    """
    obs = _truth_on_fc(truth, fc)
    w = np.cos(np.deg2rad(fc["lat"]))
    rmse = np.sqrt(((fc - obs) ** 2).weighted(w).mean(("lat", "lon")))  # dim: member
    idx = int(rmse.argmin("member"))
    return fc.isel(member=idx), idx, float(rmse.isel(member=idx))


def _event_sign(name) -> int:
    """+1 for heat events (extreme = hottest), -1 for cold events (extreme = coldest)."""
    cold_keys = ("winter", "polar", "vortex", "cold", "freeze", "elliott", "uri")
    return -1 if any(k in name.lower() for k in cold_keys) else 1


def _extreme_member(fc, name):
    """The member that most aggressively predicts the event's extreme over CONUS.

    Extremeness = latitude-weighted CONUS-mean anomaly taken in the event's own
    direction (heat -> warmest member, cold -> coldest member). This favours a
    member that overshoots the extreme rather than a miss (false positive > no
    positive). Returns ``(member_field, member_index, mean_anomaly)``.
    """
    sign = _event_sign(name)
    w = np.cos(np.deg2rad(fc["lat"]))
    mean_anom = fc.weighted(w).mean(("lat", "lon"))           # dim: member
    idx = int((sign * mean_anom).argmax("member"))
    return fc.isel(member=idx), idx, float(mean_anom.isel(member=idx))


# --------------------------------------------------------------------------- #
# PDF figures
# --------------------------------------------------------------------------- #
def _pdf_on_grid(sample, xgrid):
    from scipy.stats import gaussian_kde
    sample = sample[np.isfinite(sample)]
    if sample.size < 5 or np.allclose(sample, sample[0]):
        d = np.full_like(xgrid, YMIN)
        d[np.argmin(np.abs(xgrid - np.nanmean(sample)))] = 1.0
        return d
    return gaussian_kde(sample)(xgrid)


def _plot_pdf(ax, name, weeks, model="gencast"):
    truth = _truth(name)
    fc = _forecast(name, weeks, model=model)
    series = {"truth": truth.values.ravel()}
    hrrr = _hrrr_truth(name)
    if hrrr is not None:
        series["hrrr"] = hrrr.values.ravel()
    nm = int(fc.sizes["member"]) if fc is not None else 0   # actual ensemble size in the file
    best_idx = None
    if fc is not None:
        series[model] = fc.values.ravel()
        best, best_idx, _ = _best_member(fc, truth)
        series["best"] = best.values.ravel()
    allv = np.concatenate([v[np.isfinite(v)] for v in series.values()])
    xgrid = np.linspace(allv.min() - 1, allv.max() + 1, 400)
    style = {
        "truth": dict(color="k", lw=2.5, label="Observed (ERA5)"),
        "hrrr": dict(color="tab:red", lw=2.0, label="Observed (HRRR)"),
        "gencast": dict(color="tab:blue", lw=2.0, label=f"GenCast {nm}-member"),
        "best": dict(color="tab:green", lw=2.0, ls="--",
                     label=f"Best member (#{best_idx})"),
    }
    for key in ("truth", "hrrr", model, "best"):
        if key not in series:
            continue
        d = np.clip(_pdf_on_grid(series[key], xgrid), YMIN, None)
        ax.plot(xgrid, d, **style.get(key, dict(label=key)))
    ax.axvline(0, color="grey", lw=0.8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(YMIN, None)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel(f"week-{weeks} T2m anomaly (K)")
    ax.set_ylabel("density")


def plot_pdfs(weeks, model="gencast"):
    names = _events_with_forecast(weeks, model=model)
    outdir = C.figures_dir(weeks) / "pdf"
    outdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        fig, ax = plt.subplots(figsize=(6, 4))
        _plot_pdf(ax, name, weeks, model=model)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"{name}_pdf.png", dpi=140)
        plt.close(fig)
    if not names:
        return
    cols = 3
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        _plot_pdf(ax, name, weeks, model=model)
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Week-{weeks} T2m anomaly PDFs over CONUS (init = peak - {C.lead_days_for(weeks)}d)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(outdir / "all_events_pdf.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[pdf week{weeks}] wrote {len(names)} + combined -> {outdir}")


# --------------------------------------------------------------------------- #
# Cartopy CONUS maps: observed | ensemble-mean | error
# --------------------------------------------------------------------------- #
def _add_conus_features(ax):
    import cartopy.feature as cfeature
    for feat, kw in ((cfeature.COASTLINE, dict(lw=0.6)),
                     (cfeature.BORDERS, dict(lw=0.5)),
                     (cfeature.STATES, dict(lw=0.3, edgecolor="grey"))):
        try:
            ax.add_feature(feat, **kw)            # needs Natural Earth data (cached on NFS)
        except Exception as e:                    # offline & uncached -> draw without it
            print(f"[maps] skipped a map feature ({type(e).__name__}: {e})")


def _panel(fig, gs_pos, da, title, cmap, vmax, ccrs):
    # gs_pos may be a 3-digit int (e.g. 131) or an (nrows, ncols, index) tuple;
    # add_subplot only accepts the tuple form when it is unpacked into 3 args.
    pos = gs_pos if isinstance(gs_pos, tuple) else (gs_pos,)
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    ax.set_extent(C.CONUS_EXTENT, ccrs.PlateCarree())
    da = _to_180(da)
    m = ax.pcolormesh(da.lon, da.lat, da.values, transform=ccrs.PlateCarree(),
                      cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    _add_conus_features(ax)
    ax.set_title(title, fontsize=10)
    fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.04, shrink=0.9,
                 label="K")
    return ax


def _maps_for_event(name, weeks, model="gencast"):
    """Return (truth, ens_mean, error, n_members) on the forecast (CONUS) grid."""
    truth = _truth(name)
    fc = _forecast(name, weeks, model=model)
    if fc is None:
        return None
    nm = int(fc.sizes["member"])
    ens_mean = fc.mean("member")
    # Align truth (0.25deg) onto the forecast grid (1deg) for a point-for-point error.
    truth_on_fc = truth.sel(lat=ens_mean["lat"], lon=ens_mean["lon"], method="nearest")
    truth_on_fc = truth_on_fc.assign_coords(lat=ens_mean["lat"], lon=ens_mean["lon"])
    error = ens_mean - truth_on_fc
    return truth, ens_mean, error, nm


def plot_maps(weeks, model="gencast"):
    import cartopy.crs as ccrs
    names = _events_with_forecast(weeks, model=model)
    outdir = C.figures_dir(weeks) / "maps"
    outdir.mkdir(parents=True, exist_ok=True)
    lead = C.lead_days_for(weeks)

    for name in names:
        res = _maps_for_event(name, weeks, model=model)
        if res is None:
            continue
        truth, ens_mean, error, nm = res
        amax = float(np.nanmax(np.abs(np.concatenate(
            [truth.values.ravel(), ens_mean.values.ravel()])))) or 1.0
        emax = float(np.nanmax(np.abs(error.values))) or 1.0
        fig = plt.figure(figsize=(15, 4.2))
        _panel(fig, 131, truth, "Observed (ERA5)", "RdBu_r", amax, ccrs)
        _panel(fig, 132, ens_mean, f"GenCast {nm}-member mean", "RdBu_r", amax, ccrs)
        _panel(fig, 133, error, "Error (forecast - obs)", "RdBu_r", emax, ccrs)
        fig.suptitle(f"{name} — week-{weeks} T2m anomaly (init = peak - {lead}d)", y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / f"{name}_t2m_map.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # Combined grid: one row per event, columns observed | mean | error.
    if not names:
        return
    nrows = len(names)
    fig = plt.figure(figsize=(15, 4.0 * nrows))
    for i, name in enumerate(names):
        res = _maps_for_event(name, weeks, model=model)
        if res is None:
            continue
        truth, ens_mean, error, nm = res
        amax = float(np.nanmax(np.abs(np.concatenate(
            [truth.values.ravel(), ens_mean.values.ravel()])))) or 1.0
        emax = float(np.nanmax(np.abs(error.values))) or 1.0
        _panel(fig, (nrows, 3, 3 * i + 1), truth, f"{name}\nObserved (ERA5)", "RdBu_r", amax, ccrs)
        _panel(fig, (nrows, 3, 3 * i + 2), ens_mean, f"GenCast {nm}-member mean", "RdBu_r", amax, ccrs)
        _panel(fig, (nrows, 3, 3 * i + 3), error, "Error (fc - obs)", "RdBu_r", emax, ccrs)
    fig.suptitle(f"Week-{weeks} CONUS T2m anomaly (init = peak - {lead}d)", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "all_events_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[maps week{weeks}] wrote {len(names)} + combined -> {outdir}")


def _best_maps_for_event(name, weeks, model="gencast"):
    """Return (truth, best_member, error, member_index) on the forecast grid."""
    truth = _truth(name)
    fc = _forecast(name, weeks, model=model)
    if fc is None:
        return None
    best, idx, _ = _best_member(fc, truth)
    error = best - _truth_on_fc(truth, fc)
    return truth, best, error, idx


def plot_best_member_maps(weeks, model="gencast"):
    """CONUS maps using the single best ensemble member: observed | best | error."""
    import cartopy.crs as ccrs
    names = _events_with_forecast(weeks, model=model)
    outdir = C.figures_dir(weeks) / "maps"
    outdir.mkdir(parents=True, exist_ok=True)
    lead = C.lead_days_for(weeks)

    for name in names:
        res = _best_maps_for_event(name, weeks, model=model)
        if res is None:
            continue
        truth, best, error, idx = res
        amax = float(np.nanmax(np.abs(np.concatenate(
            [truth.values.ravel(), best.values.ravel()])))) or 1.0
        emax = float(np.nanmax(np.abs(error.values))) or 1.0
        fig = plt.figure(figsize=(15, 4.2))
        _panel(fig, 131, truth, "Observed (ERA5)", "RdBu_r", amax, ccrs)
        _panel(fig, 132, best, f"GenCast best member (#{idx})", "RdBu_r", amax, ccrs)
        _panel(fig, 133, error, "Error (best member - obs)", "RdBu_r", emax, ccrs)
        fig.suptitle(f"{name} — week-{weeks} T2m anomaly, best member (init = peak - {lead}d)",
                     y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / f"{name}_t2m_best_member_map.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    if not names:
        return
    nrows = len(names)
    fig = plt.figure(figsize=(15, 4.0 * nrows))
    for i, name in enumerate(names):
        res = _best_maps_for_event(name, weeks, model=model)
        if res is None:
            continue
        truth, best, error, idx = res
        amax = float(np.nanmax(np.abs(np.concatenate(
            [truth.values.ravel(), best.values.ravel()])))) or 1.0
        emax = float(np.nanmax(np.abs(error.values))) or 1.0
        _panel(fig, (nrows, 3, 3 * i + 1), truth, f"{name}\nObserved (ERA5)", "RdBu_r", amax, ccrs)
        _panel(fig, (nrows, 3, 3 * i + 2), best, f"Best member (#{idx})", "RdBu_r", amax, ccrs)
        _panel(fig, (nrows, 3, 3 * i + 3), error, "Error (best - obs)", "RdBu_r", emax, ccrs)
    fig.suptitle(f"Week-{weeks} CONUS T2m anomaly, best ensemble member (init = peak - {lead}d)",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "all_events_best_member_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[best-member maps week{weeks}] wrote {len(names)} + combined -> {outdir}")


# --------------------------------------------------------------------------- #
# Scoring: ensemble CRPS + rank histograms
# --------------------------------------------------------------------------- #
def score(weeks, model="gencast"):
    import properscoring as ps
    names = _events_with_forecast(weeks, model=model)
    outdir = C.figures_dir(weeks) / "scores"
    outdir.mkdir(parents=True, exist_ok=True)

    rows, rankhists = [], {}
    for name in names:
        truth = _truth(name)
        ens = _forecast(name, weeks, model=model)                  # (member, lat, lon)
        obs = truth.sel(lat=ens["lat"], lon=ens["lon"], method="nearest").values
        e = ens.values
        crps = ps.crps_ensemble(obs, np.moveaxis(e, 0, -1))        # members -> last axis
        rows.append({"event": name,
                     f"{model}_crps": float(np.nanmean(crps)),
                     f"{model}_spread": float(np.nanstd(e, axis=0).mean())})
        below = (e < obs[None]).sum(axis=0)                        # rank of obs among members
        rankhists[name] = np.bincount(below.ravel(), minlength=e.shape[0] + 1)

    if not rows:
        return
    scores = pd.DataFrame(rows)
    scores.to_csv(outdir / "scores.csv", index=False)
    print(f"[scores week{weeks}]\n{scores.to_string(index=False)}")

    # Rank histograms (per event + pooled).
    n = len(rankhists)
    cols = 3
    nrows = math.ceil(n / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 3.2 * nrows), squeeze=False)
    pooled = None
    for ax, (name, h) in zip(axes.ravel(), rankhists.items()):
        pooled = h.copy() if pooled is None else pooled + h
        ax.bar(np.arange(len(h)), h / h.sum(), color="tab:blue", alpha=0.8)
        ax.axhline(1 / len(h), color="k", ls="--", lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("rank of obs among members")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Week-{weeks} rank histograms (flat = calibrated, U = under-dispersed)", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "rank_histograms.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    if pooled is not None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(np.arange(len(pooled)), pooled / pooled.sum(), color="tab:purple", alpha=0.8)
        ax.axhline(1 / len(pooled), color="k", ls="--", lw=1)
        ax.set_title(f"Week-{weeks} pooled rank histogram (all OOS events)")
        ax.set_xlabel("rank of obs among members")
        fig.tight_layout()
        fig.savefig(outdir / "rank_hist_pooled.png", dpi=140)
        plt.close(fig)

    # CRPS bar summary.
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(scores))
    ax.bar(x, scores[f"{model}_crps"], 0.6, color="tab:blue", label="GenCast CRPS")
    ax.set_xticks(x)
    ax.set_xticklabels(scores["event"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("K")
    ax.legend()
    ax.set_title(f"Week-{weeks} T2m ensemble CRPS")
    fig.tight_layout()
    fig.savefig(outdir / "crps_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_all(weeks, model="gencast"):
    plot_pdfs(weeks, model=model)
    plot_maps(weeks, model=model)
    plot_best_member_maps(weeks, model=model)
    score(weeks, model=model)
