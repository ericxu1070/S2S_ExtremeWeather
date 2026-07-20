#!/usr/bin/env python
"""ERA5->HRRR downscaler demo + GenCast (xres 1deg) -> downscaler coupling.

Produces three figures:

  1. TEST-SET DEMO      real ERA5 analysis at an out-of-sample (2024) timestamp,
                        downscaled to 3 km HRRR, vs HRRR truth.
                        cols: [ ERA5 1deg -> HRRR grid | HRRR 3km truth | HRRR 3km predicted ]

  2. COUPLING DEMO      one coarse 1deg GenCast *forecast* member of an out-of-distribution
                        extreme event (from an xres cube, a 2-3 week lead), downscaled to
                        3 km HRRR, vs the HRRR analysis truth at the event time.
                        cols: [ GenCast 1deg forecast | HRRR 3km truth | HRRR 3km predicted ]

  3. PDFs               spatial distributions, over the CONUS interior actually covered by the
                        GenCast box, of HRRR truth vs predicted HRRR vs the coarse forecast.

This is the "thin adapter" CLAUDE.md anticipates: it does NOT import gencast_s2s / xres. It
only READS a GenCast output cube (a netCDF) and feeds a member's coarse fields through the
SAME regrid + normalization path the downscaler's Era5HrrrDataset uses for real ERA5, so the
two stacks stay decoupled and the conditioning the model sees is identical in construction.

Stages (GPU only for `infer`):

  prep  (CPU, login node): read ERA5 / HRRR / GenCast cubes, regrid the coarse drivers onto
                           the HRRR grid in PHYSICAL units, read HRRR truth, cache to npz.
  infer (GPU, sbatch):     load the trained EDM downscaler (EMA weights), normalize the cached
                           conditioning, draw one conditional sample per case, cache the
                           physical predictions. Pure GPU compute -- no data building here.
  plot  (CPU, login node): assemble the three figures from the caches.

    python scripts/xres_downscale_demo.py --stage prep
    sbatch  slurm/xres_downscale_infer.sbatch        # runs --stage infer on one H100
    python scripts/xres_downscale_demo.py --stage plot
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # downscaler/
REPO_ROOT = os.path.dirname(ROOT)            # S2S_ExtremeWeather/ (holds runs/xres/)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.era5_hrrr_dataset import ERA5_TRANSFORMS, Era5HrrrDataset  # noqa: E402

# --- CONUS interior box actually covered by the GenCast xres cube -------------------------
# The HRRR grid runs lat 21.1..52.6, lon 235.9..299.1; the GenCast box is lat 24..50,
# lon 235..294. Outside that box the bilinear regridder EXTRAPOLATES from the box edge (the
# model was trained on GLOBAL ERA5 and never saw extrapolated edges), so we score / show only
# the interior. This box is the intersection, shrunk 1 cell to stay clear of the edge.
BOX = dict(lat0=24.5, lat1=49.5, lon0=236.0, lon1=293.0)  # lon in 0..360

CACHE = os.path.join(ROOT, "outputs", "xres_demo_cache")

# --- Cases -------------------------------------------------------------------------------
# caseA: real ERA5 analysis, out-of-sample year (2024 test split) -> the in-distribution
#        "does the downscaler work" demo.
# caseB: a 1deg GenCast forecast member of an OOD extreme event -> the coupling demo.
CASES = {
    "testset": dict(
        kind="era5",
        title="Downscaler test set  |  ERA5 analysis 2024-07-16 00Z (out-of-sample)",
        valid="2024-07-16T00:00:00",
        # real ERA5 covers the whole HRRR grid, so no interior restriction is needed;
        # we still show a CONUS view for comparability.
        box=None,
    ),
    "idalia": dict(
        kind="gencast",
        title="GenCast 1° forecast → downscaler  |  Hurricane Idalia 2023, week-3 lead (member {m})",
        cube="runs/xres/1p0/week3/cache/HurricaneIdalia_2023_cube.nc",
        valid="2023-08-30T00:00:00",   # event peak / landfall (last cube time)
        member=0,
        box=BOX,
    ),
}


# ----------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------
class _IdentityNorm:
    """No-op normalizer: prep works in PHYSICAL units; infer applies the real z-scores."""

    def normalize_orog(self, x):
        return x

    def normalize_era5(self, x):
        return x

    def normalize_hrrr(self, y):
        return y

    def normalize_diagnostic(self, d):
        return d


def _load_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.load(os.path.join(ROOT, "configs", "data", "era5_hrrr.yaml"))


def _build_probe(cfg, norm):
    """An Era5HrrrDataset used only for its grid, cached bilinear regridder and statics.

    era5_variables is the MODEL's conditioning list (order matters -- the sample() input
    channels must line up with what the UNet was trained on).
    """
    ts0 = CASES["testset"]["valid"]  # any real HRRR file works for the (time-invariant) grid/statics
    return Era5HrrrDataset(
        timestamps=[ts0],
        era5_dir=cfg.era5_dir,
        hrrr_dir=cfg.hrrr_dir,
        grid_meta_path=cfg.grid_meta_path,
        norm=norm,
        era5_variables=list(cfg.era5_variables),
        hrrr_prognostic_variables=list(cfg.hrrr_prognostic_variables),
        hrrr_diagnostic_variables=list(cfg.hrrr_diagnostic_variables),
        era5_path_template=cfg.era5_path_template,
        hrrr_path_template=cfg.hrrr_path_template,
        era5_lat_name=cfg.era5_lat_name,
        era5_lon_name=cfg.era5_lon_name,
        era5_level_name=cfg.era5_level_name,
        orog_var=cfg.orog_var,
        lsm_var=cfg.lsm_var,
        match_era5_lon_to_360=cfg.match_era5_lon_to_360,
        rotate_hrrr_winds=cfg.rotate_hrrr_winds,
        hrrr_wind_pairs=[list(p) for p in cfg.hrrr_wind_pairs],
    )


def _coarse_source(cfg, case):
    """Return (dataset_for_read, src_lat, src_lon) for the coarse conditioning source.

    For an ERA5 case that is the real 1deg analysis file; for a GenCast case it is the cube
    subset to one member and the event valid time. Both expose the driver variables under the
    same CF names, on ascending/whatever lat and 0..360 lon, so probe._read_2d + regrid_era5
    consume them identically.
    """
    import xarray as xr

    if case["kind"] == "era5":
        dt = datetime.strptime(case["valid"][:19], "%Y-%m-%dT%H:%M:%S")
        path = os.path.join(cfg.era5_dir, dt.strftime(cfg.era5_path_template))
        if not os.path.exists(path):
            raise SystemExit(f"missing ERA5 file: {path}")
        eds = xr.open_dataset(path)
        src_lat = np.asarray(eds[cfg.era5_lat_name].values, np.float64)
        src_lon = np.mod(np.asarray(eds[cfg.era5_lon_name].values, np.float64), 360.0)
        return eds, src_lat, src_lon

    # GenCast cube (runs/xres/ lives at the repo root, above downscaler/)
    path = os.path.join(REPO_ROOT, case["cube"])
    if not os.path.exists(path):
        raise SystemExit(f"missing GenCast cube: {path}")
    cube = xr.open_dataset(path)
    sub = cube.isel(member=int(case["member"])).sel(
        time=np.datetime64(case["valid"]), method="nearest"
    )
    got = str(np.datetime64(sub.time.values))
    print(f"    cube member={case['member']} valid={got} (requested {case['valid']})")
    src_lat = np.asarray(cube.lat.values, np.float64)
    src_lon = np.mod(np.asarray(cube.lon.values, np.float64), 360.0)
    return sub, src_lat, src_lon


def _hrrr_path(cfg, valid):
    dt = datetime.strptime(valid[:19], "%Y-%m-%dT%H:%M:%S")
    return os.path.join(cfg.hrrr_dir, dt.strftime(cfg.hrrr_path_template))


def _interior_mask(lat, lon, box):
    if box is None:
        return np.ones(lat.shape, bool)
    lon = np.mod(lon, 360.0)
    return (
        (lat >= box["lat0"]) & (lat <= box["lat1"]) & (lon >= box["lon0"]) & (lon <= box["lon1"])
    )


def _derive_display(t2m, u10, v10, log_tp):
    """Physical driver/target/pred channels -> the three display fields (K, m/s, mm)."""
    return dict(
        t2m=t2m,                                   # K
        wspd=np.hypot(u10, v10),                   # m/s (rotation-invariant)
        precip=np.exp(log_tp) - 1e-5,              # mm (inverse of ln(mm+1e-5))
    )


# ----------------------------------------------------------------------------------------
# Stage: prep  (CPU / login node)
# ----------------------------------------------------------------------------------------
def stage_prep():
    import netCDF4

    os.makedirs(CACHE, exist_ok=True)
    cfg = _load_cfg()
    probe = _build_probe(cfg, _IdentityNorm())
    H, W = probe.H, probe.W

    # Save the grid once (2-D lat/lon on the HRRR Lambert grid) for the plot stage.
    np.savez(
        os.path.join(CACHE, "grid.npz"),
        lat=probe.hrrr_lat.astype(np.float32),
        lon=probe.hrrr_lon.astype(np.float32),
    )
    print(f"grid: {H} x {W}  ->  {os.path.join(CACHE, 'grid.npz')}")

    # Which cube variables map to the [t2m, u10, v10] display + the log_tp precip channel.
    var_index = {name: i for i, (name, _, _) in enumerate(probe.era5_variables)}
    i_t2m = var_index["2m_temperature"]
    i_u10 = var_index["10m_u_component_of_wind"]
    i_v10 = var_index["10m_v_component_of_wind"]
    # precip channel carries the log_precip_m_to_mm transform (ln(mm+1e-5))
    i_tp = next(i for i, (_, _, t) in enumerate(probe.era5_variables) if t is not None)

    for key, case in CASES.items():
        print(f"\n=== prep [{key}] {case['kind']} valid={case['valid']} ===")

        # --- coarse ERA5/GenCast drivers -> HRRR grid, PHYSICAL, model channel order ---
        eds, src_lat, src_lon = _coarse_source(cfg, case)
        cond = []
        for name, level, transform in probe.era5_variables:
            coarse = probe._read_2d(eds, name, level, transform)
            cond.append(probe.regrid_era5(coarse, src_lat, src_lon))
        cond = np.stack(cond, axis=0).astype(np.float32)  # (n_era5, H, W)
        try:
            eds.close()
        except Exception:
            pass

        # coarse display fields: precip channel is in log space -> invert to mm
        coarse_disp = _derive_display(cond[i_t2m], cond[i_u10], cond[i_v10], cond[i_tp])

        # --- HRRR truth at the same valid time ---
        hp = _hrrr_path(cfg, case["valid"])
        if not os.path.exists(hp):
            raise SystemExit(f"missing HRRR truth file: {hp}")
        hds = netCDF4.Dataset(hp, "r")
        tt2m = np.asarray(hds.variables["t2m"][:], np.float32).reshape(H, W)
        tu10 = np.asarray(hds.variables["u10"][:], np.float32).reshape(H, W)
        tv10 = np.asarray(hds.variables["v10"][:], np.float32).reshape(H, W)
        tlog_tp = np.asarray(hds.variables["log_tp"][:], np.float32).reshape(H, W)
        hds.close()
        truth_disp = _derive_display(tt2m, tu10, tv10, tlog_tp)  # wind SPEED needs no rotation

        mask = _interior_mask(probe.hrrr_lat, probe.hrrr_lon, case["box"])

        out = os.path.join(CACHE, f"{key}_prep.npz")
        np.savez(
            out,
            cond_phys=cond,                       # (n_era5, H, W)  -> infer
            coarse_t2m=coarse_disp["t2m"],
            coarse_wspd=coarse_disp["wspd"],
            coarse_precip=coarse_disp["precip"],
            truth_t2m=truth_disp["t2m"],
            truth_wspd=truth_disp["wspd"],
            truth_precip=truth_disp["precip"],
            mask=mask,
        )
        print(f"    cond {cond.shape} | truth t2m {tt2m.mean():.1f}K "
              f"| interior pixels {int(mask.sum()):,} -> {out}")

    print("\nprep done. Verify caches, then: sbatch slurm/xres_downscale_infer.sbatch")


# ----------------------------------------------------------------------------------------
# Stage: infer  (GPU / sbatch)
# ----------------------------------------------------------------------------------------
def _load_eval_state(ckpt):
    """Weights to evaluate: EMA shadow overlaid on the model state dict, else raw.

    (Same logic as scripts/compare_timestep.load_eval_state; inlined to avoid a fragile
    cross-script import in the sbatch environment.)
    """
    state = dict(ckpt["model_state_dict"])
    shadow = (ckpt.get("ema_state_dict") or {}).get("shadow")
    if not shadow:
        return state, False
    unknown = [k for k in shadow if k not in state]
    if unknown:
        raise SystemExit(f"EMA shadow keys not in model_state_dict: {unknown[:3]} ...")
    state.update(shadow)
    return state, True


def stage_infer(ckpt_path, n_steps, seed):
    import torch
    from omegaconf import OmegaConf

    from data.normalization import Era5HrrrNorm
    from evaluation.downscale import Downscaler
    from model.preconditioning import EDMPreconditioning
    from model.unet import HRRRUNet

    cfg = _load_cfg()
    if not os.path.exists(cfg.stats_path):
        raise SystemExit(f"no norm stats at {cfg.stats_path}")
    if not torch.cuda.is_available():
        raise SystemExit("infer needs a GPU (run via slurm/xres_downscale_infer.sbatch)")
    device = torch.device("cuda")

    mcfg = OmegaConf.load(os.path.join(ROOT, "configs", "model", "unet.yaml"))
    tcfg = OmegaConf.load(os.path.join(ROOT, "configs", "training", "default.yaml"))
    norm = Era5HrrrNorm(cfg.stats_path).to(device)

    unet = HRRRUNet(
        n_input_ch=mcfg.n_input_ch, n_output_ch=mcfg.n_output_ch, C_base=mcfg.C_base,
        n_res_blocks=list(mcfg.n_res_blocks), emb_dim=mcfg.emb_dim, num_groups=mcfg.num_groups,
        use_attention=mcfg.use_attention, attn_depth=mcfg.attn_depth,
        attn_window_size=mcfg.attn_window_size, attn_num_heads=mcfg.attn_num_heads,
        attn_mlp_ratio=mcfg.attn_mlp_ratio,
    )
    model = EDMPreconditioning(unet, sigma_data=tcfg.sigma_data).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state, used_ema = _load_eval_state(ckpt)
    model.load_state_dict(state)
    print(f"loaded {ckpt_path} (step {ckpt.get('global_step', '?')}, "
          f"{'EMA' if used_ema else 'raw'} weights)")

    # statics (orog, lsm) on the HRRR grid, normalized with the real stats.
    probe = _build_probe(cfg, norm)
    n_prog = len(cfg.hrrr_prognostic_variables)
    n_diag = len(cfg.hrrr_diagnostic_variables)
    prog_names = [str(v) for v in cfg.hrrr_prognostic_variables]
    ip_t2m, ip_u10, ip_v10 = (prog_names.index(x) for x in ("t2m", "u10", "v10"))
    dsc = Downscaler(model, norm, probe.statics.to(device), n_prognostic=n_prog, n_diagnostic=n_diag)

    for key in CASES:
        prep = np.load(os.path.join(CACHE, f"{key}_prep.npz"))
        cond = torch.from_numpy(prep["cond_phys"]).unsqueeze(0).to(device)  # (1, n_era5, H, W)
        torch.manual_seed(seed)                                             # reproducible sample
        x_cond = dsc.build_conditioning(cond)                              # normalize + statics
        prog, diag = dsc.sample(x_cond, n_steps=n_steps)                   # physical HRRR fields
        prog = prog[0].cpu().numpy()
        diag = diag[0].cpu().numpy()
        disp = _derive_display(prog[ip_t2m], prog[ip_u10], prog[ip_v10], diag[0])
        out = os.path.join(CACHE, f"{key}_pred.npz")
        np.savez(out, pred_t2m=disp["t2m"], pred_wspd=disp["wspd"], pred_precip=disp["precip"])
        print(f"[{key}] sampled -> t2m {disp['t2m'].mean():.1f}K "
              f"wspd<={disp['wspd'].max():.1f} precip<={disp['precip'].max():.1f}mm -> {out}")

    print("\ninfer done. Now: python scripts/xres_downscale_demo.py --stage plot")


# ----------------------------------------------------------------------------------------
# Stage: plot  (CPU / login node)
# ----------------------------------------------------------------------------------------
# variable -> (display transform, unit label, colormap, "diverging?")
VARS = [
    ("t2m",    lambda a: a - 273.15, "2 m temperature [°C]",  "RdYlBu_r", "seq"),
    ("wspd",   lambda a: a,          "10 m wind speed [m/s]",      "viridis",  "seq"),
    ("precip", lambda a: a,          "precipitation [mm]",         "cmo.rain", "precip"),
]


def _resolve_cmap(name):
    """Resolve a colormap spec: 'cmo.<x>' -> a cmocean colormap object; else pass through
    (matplotlib resolves plain names / Colormap objects itself)."""
    if isinstance(name, str) and name.startswith("cmo."):
        import cmocean
        return getattr(cmocean.cm, name[4:])
    return name


def _geo_ax(fig, gs, lat, lon):
    import cartopy.crs as ccrs

    ax = fig.add_subplot(gs, projection=ccrs.PlateCarree())
    ax.set_extent([-125, -66.5, 23.5, 50.5], crs=ccrs.PlateCarree())
    return ax


def _draw(ax, lat, lon180, field, vmin, vmax, cmap):
    import cartopy.feature as cfeature

    im = ax.pcolormesh(lon180, lat, field, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto",
                       transform=_pc(), rasterized=True)
    for feat, lw in ((cfeature.COASTLINE, 0.5), (cfeature.BORDERS, 0.4)):
        try:
            ax.add_feature(feat, linewidth=lw, edgecolor="0.15")
        except Exception:
            pass
    try:
        ax.add_feature(cfeature.STATES, linewidth=0.25, edgecolor="0.45")
    except Exception:
        pass
    return im


def _pc():
    import cartopy.crs as ccrs

    return ccrs.PlateCarree()


def _vrange(fields, kind):
    """Shared colour limits across the panels of one variable row."""
    stack = np.concatenate([f[np.isfinite(f)].ravel() for f in fields])
    if kind == "precip":
        return 0.0, max(float(np.percentile(stack, 99.5)), 1.0)
    lo, hi = np.percentile(stack, [1, 99])
    return float(lo), float(hi)


def _rmse(a, b, mask):
    d = (a - b)[mask]
    return float(np.sqrt(np.mean(d * d)))


def _maps_figure(key, col_order, col_titles, out_png, suptitle):
    """(field columns | error) x (t2m | wspd | precip) map figure.

    A dedicated right-hand column shows the downscaler error (predicted − truth) on a
    diverging colour scale, symmetric about zero, over the same interior mask used for the
    metrics. Field columns share one colourbar per row; the error column has its own.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    S = 2  # render stride: the HRRR grid (1059x1799) over-samples the figure at this DPI, so
           # subsampling for pcolormesh is visually lossless and ~4x faster. RMSE/PDFs use full res.
    grid = np.load(os.path.join(CACHE, "grid.npz"))
    lat = grid["lat"].astype(np.float64)
    lon180 = np.mod(grid["lon"].astype(np.float64) + 180.0, 360.0) - 180.0
    prep = np.load(os.path.join(CACHE, f"{key}_prep.npz"))
    pred = np.load(os.path.join(CACHE, f"{key}_pred.npz"))
    mask = prep["mask"].astype(bool)
    box = CASES[key]["box"]

    # source arrays keyed by column tag
    src = {"coarse": prep, "truth": prep, "pred": pred}
    pfx = {"coarse": "coarse_", "truth": "truth_", "pred": "pred_"}

    K = len(col_order)
    # columns: K field maps | field colourbar | error map | error colourbar
    ncol = K + 3
    width_ratios = [1.0] * K + [0.055, 1.0, 0.055]
    fig = plt.figure(figsize=(4.35 * (K + 1) + 1.0, 3.5 * len(VARS)))
    gs = GridSpec(len(VARS), ncol, figure=fig, width_ratios=width_ratios, hspace=0.14, wspace=0.10)

    for r, (vkey, tf, label, cmap, kind) in enumerate(VARS):
        truth = tf(prep["truth_" + vkey].astype(np.float64))
        predf = tf(pred["pred_" + vkey].astype(np.float64))

        # --- field columns (shared colour scale) ---
        cols = []
        for tag in col_order:
            arr = tf(src[tag][pfx[tag] + vkey].astype(np.float64)).copy()
            # blank the extrapolated rim on the coarse GenCast panel (outside the box)
            if tag == "coarse" and box is not None:
                arr[~mask] = np.nan
            cols.append(arr)
        vmin, vmax = _vrange(cols, kind)
        fcmap = _resolve_cmap(cmap)
        im = None
        for c, (tag, arr) in enumerate(zip(col_order, cols)):
            ax = _geo_ax(fig, gs[r, c], lat, lon180)
            im = _draw(ax, lat[::S, ::S], lon180[::S, ::S], arr[::S, ::S], vmin, vmax, fcmap)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, pad=4)
            if c == 0:
                ax.text(-0.02, 0.5, label, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=11)
            # RMSE badge on predicted vs truth (only where both are the fine-scale HRRR)
            if tag == "pred":
                ax.text(0.98, 0.03, f"RMSE {_rmse(predf, truth, mask):.2f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=8.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="0.15", ec="none", alpha=0.7))
        fig.colorbar(im, cax=fig.add_subplot(gs[r, K]))

        # --- error column: predicted − truth (diverging, symmetric about 0) ---
        err = predf - truth                                   # finite on the full grid
        if box is not None:
            err = err.copy()
            err[~mask] = np.nan                               # blank the extrapolated rim
        lim = float(np.percentile(np.abs(err[mask]), 99)) or 1.0
        ecmap = "BrBG" if kind == "precip" else "RdBu_r"      # brown=too dry / blue=too cold, etc.
        axe = _geo_ax(fig, gs[r, K + 1], lat, lon180)
        ime = _draw(axe, lat[::S, ::S], lon180[::S, ::S], err[::S, ::S], -lim, lim, _resolve_cmap(ecmap))
        if r == 0:
            axe.set_title("error\n(predicted − truth)", fontsize=11, pad=4)
        axe.text(0.98, 0.03, f"bias {np.mean(err[mask]):+.2f}", transform=axe.transAxes,
                 ha="right", va="bottom", fontsize=8.5, color="black",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        fig.colorbar(ime, cax=fig.add_subplot(gs[r, K + 2]))

    fig.suptitle(suptitle, fontsize=13, y=0.995)
    fig.savefig(out_png, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {out_png}")


def _pdf_figure(key, out_png, suptitle):
    """Overlaid PDFs of HRRR truth vs predicted HRRR vs coarse forecast, over the interior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prep = np.load(os.path.join(CACHE, f"{key}_prep.npz"))
    pred = np.load(os.path.join(CACHE, f"{key}_pred.npz"))
    mask = prep["mask"].astype(bool)

    series = [
        ("HRRR truth (3 km)",            "truth_",  prep, "0.1",  "-"),
        ("downscaled HRRR (predicted)",  "pred_",   pred, "#1f77b4", "-"),
        ("GenCast 1° forecast",     "coarse_", prep, "#d62728", "-"),
    ]

    fig, axes = plt.subplots(1, len(VARS), figsize=(5.2 * len(VARS), 4.2))
    for ax, (vkey, tf, label, _cmap, kind) in zip(axes, VARS):
        vals = {tag: tf(store[tag + vkey].astype(np.float64))[mask] for _, tag, store, _, _ in series}
        if kind == "precip":
            # heavily zero-dominated: compare the WET-pixel distribution on log axes.
            wet = {k: v[v > 0.1] for k, v in vals.items()}
            allw = np.concatenate(list(wet.values())) if any(len(w) for w in wet.values()) else np.array([1.0])
            bins = np.logspace(np.log10(0.1), np.log10(max(allw.max(), 1.0)), 40)
            for (name, tag, _s, color, ls) in series:
                w = wet[tag]
                if len(w):
                    ax.hist(w, bins=bins, density=True, histtype="step", color=color, lw=1.8, ls=ls, label=name)
            ax.set_xscale("log")
            ax.set_xlabel(label + "   (wet pixels > 0.1 mm)")
            ax.text(0.5, 1.02, "coarse forecast = 12 h accum; HRRR = 6 h", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8, color="0.4")
        else:
            allv = np.concatenate(list(vals.values()))
            bins = np.linspace(np.percentile(allv, 0.2), np.percentile(allv, 99.8), 60)
            for (name, tag, _s, color, ls) in series:
                ax.hist(vals[tag], bins=bins, density=True, histtype="step", color=color, lw=1.8, ls=ls, label=name)
                ax.axvline(np.mean(vals[tag]), color=color, lw=0.8, ls=":", alpha=0.6)
            ax.set_xlabel(label)
        ax.set_ylabel("probability density")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=125)
    plt.close(fig)
    print(f"  figure: {out_png}")


def stage_plot():
    outdir = os.path.join(ROOT, "outputs")
    os.makedirs(outdir, exist_ok=True)

    print("Figure 1: test-set demo")
    _maps_figure(
        "testset",
        col_order=["coarse", "truth", "pred"],
        col_titles=["ERA5 1° → HRRR grid\n(bilinear input)", "HRRR 3 km truth",
                    "HRRR 3 km predicted\n(downscaler)"],
        out_png=os.path.join(outdir, "fig1_downscaler_testset.png"),
        suptitle=CASES["testset"]["title"],
    )

    print("Figure 2: GenCast -> downscaler coupling")
    m = CASES["idalia"]["member"]
    _maps_figure(
        "idalia",
        col_order=["coarse", "truth", "pred"],
        col_titles=["GenCast 1° forecast\n(regridded to HRRR)", "HRRR 3 km truth\n(analysis at event)",
                    "HRRR 3 km predicted\n(downscaled forecast)"],
        out_png=os.path.join(outdir, "fig2_gencast_downscaled.png"),
        suptitle=CASES["idalia"]["title"].format(m=m),
    )

    print("Figure 3: PDFs")
    _pdf_figure(
        "idalia",
        out_png=os.path.join(outdir, "fig3_pdfs.png"),
        suptitle="Spatial distributions over the CONUS interior  |  " + CASES["idalia"]["title"].format(m=m),
    )


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["prep", "infer", "plot"])
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "latest.pt"))
    ap.add_argument("--n-steps", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stage == "prep":
        stage_prep()
    elif args.stage == "infer":
        stage_infer(args.ckpt, args.n_steps, args.seed)
    else:
        stage_plot()


if __name__ == "__main__":
    main()
