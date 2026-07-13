"""Compare one ERA5 timestep against the HRRR analysis at the same timestep.

    python scripts/compare_timestep.py --timestamp 2023-07-15T18
    python scripts/compare_timestep.py --timestamp 2023-07-15T18 --ckpt checkpoints/latest.pt

Two things happen here, and it is worth being clear about which is which:

1. WITHOUT --ckpt (the default) this is a *baseline*, not a model evaluation. It
   bilinearly regrids the coarse ERA5 onto the HRRR grid — exactly the operation
   Era5HrrrDataset performs to build the conditioning input — and scores that
   interpolated field against the HRRR truth. This is the no-skill reference the
   downscaler has to beat: it is what you get from interpolation alone, with no
   generative model at all. It also proves the ERA5/HRRR data contract (paths,
   coord names, variable names, grids, units) actually holds on real files.

2. WITH --ckpt it additionally draws one conditional sample from the trained
   diffusion model and scores that too, so you can read the model's RMSE next to
   the interpolation baseline in the same table.

   An UNTRAINED checkpoint will score far *worse* than the baseline — a random-weight
   EDM sample is noise. That is expected and is not a bug; it means nothing until the
   model has actually been trained.

The comparison variables are the ones that are physically like-for-like between the two
datasets (ERA5 2 m temperature vs HRRR t2m, ERA5 10 m winds vs HRRR u10/v10). Note these
are NOT identical to the model's *conditioning* channels in configs/data/era5_hrrr.yaml,
which use 850 hPa winds — conditioning inputs and comparable fields are different questions.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.era5_hrrr_dataset import Era5HrrrDataset  # noqa: E402

# ERA5 field -> HRRR field, for fields that mean the same physical thing in both datasets.
#   (era5_name, era5_level, hrrr_name, units, label)
COMPARE_PAIRS = [
    ("2m_temperature",          None, "t2m", "K",   "2 m temperature"),
    ("10m_u_component_of_wind", None, "u10", "m/s", "10 m u-wind"),
    ("10m_v_component_of_wind", None, "v10", "m/s", "10 m v-wind"),
]


class _IdentityNorm:
    """No-op normalizer: this script compares fields in PHYSICAL units, not z-scores."""

    def normalize_orog(self, x):
        return x

    def normalize_era5(self, x):
        return x

    def normalize_hrrr(self, y):
        return y

    def normalize_diagnostic(self, d):
        return d


def parse_ts(s: str) -> str:
    """Accept 2023-07-15T18, 2023-07-15T18:00, 2023071518, ... -> full ISO."""
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H",
                "%Y-%m-%d %H", "%Y%m%d%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    raise SystemExit(f"could not parse timestamp: {s!r}")


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Error of `pred` against `truth`, plus the spatial std of each (the fine-scale test).

    std(pred) vs std(truth) is the interesting one for downscaling: bilinear interpolation
    is smooth, so it systematically UNDER-disperses. Closing that variance gap is precisely
    the job of the generative model.
    """
    err = pred - truth
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "corr": float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1]),
        "std_pred": float(np.std(pred)),
        "std_truth": float(np.std(truth)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timestamp", required=True, help="e.g. 2023-07-15T18")
    p.add_argument("--config", default="configs/data/era5_hrrr.yaml")
    p.add_argument("--ckpt", default=None, help="optional trained checkpoint -> also sample the model")
    p.add_argument("--n-steps", type=int, default=18, help="EDM sampler steps")
    p.add_argument("--out", default="outputs/compare_timestep.png")
    args = p.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    ts = parse_ts(args.timestamp)

    # ---------------------------------------------------------------- 1. files exist?
    print(f"\n=== timestep {ts} ===")
    # The comparison stack is [t2m, u10, v10], so the wind pair sits at channels (1, 2).
    probe = Era5HrrrDataset(
        timestamps=[ts],
        era5_dir=cfg.era5_dir,
        hrrr_dir=cfg.hrrr_dir,
        grid_meta_path=cfg.grid_meta_path,
        norm=_IdentityNorm(),
        era5_variables=[n if lv is None else f"{n}:{int(lv)}" for n, lv, _, _, _ in COMPARE_PAIRS],
        hrrr_prognostic_variables=[h for _, _, h, _, _ in COMPARE_PAIRS],
        hrrr_diagnostic_variables=[],
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
    print(f"  wind rotation: {'ON (HRRR u/v -> Earth-relative)' if cfg.rotate_hrrr_winds else 'OFF'}")
    era5_path, hrrr_path = probe._era5_path(ts), probe._hrrr_path(ts)
    for tag, path in (("ERA5", era5_path), ("HRRR", hrrr_path)):
        if not os.path.exists(path):
            raise SystemExit(f"missing {tag} file: {path}")
        print(f"  {tag}: {path}")
    print(f"  HRRR grid: {probe.H} x {probe.W}")

    # ---------------------------------------------------------------- 2. read + regrid
    import netCDF4

    eds = probe._open(era5_path)
    src_lat = np.asarray(eds[cfg.era5_lat_name].values, np.float64)
    src_lon = np.mod(np.asarray(eds[cfg.era5_lon_name].values, np.float64), 360.0)
    hds = netCDF4.Dataset(hrrr_path, "r")

    # HRRR truth as a (3, H, W) stack so the wind pair can be rotated together, using the
    # SAME dataset method training uses. Keep the raw stack too, to show what the rotation
    # is actually worth.
    truth_raw = np.stack(
        [np.asarray(hds.variables[h][:], np.float32).reshape(probe.H, probe.W)
         for _, _, h, _, _ in COMPARE_PAIRS],
        axis=0,
    )
    truth_rot = probe.rotate_prognostic_winds(truth_raw)

    rows = []
    for k, (era5_name, level, hrrr_name, units, label) in enumerate(COMPARE_PAIRS):
        coarse = probe._read_2d(eds, era5_name, level)                     # (181, 360) global
        interp = probe.regrid_era5(coarse, src_lat, src_lon)               # (H, W) on HRRR grid
        rows.append(dict(
            label=label, units=units, coarse=coarse, interp=interp,
            truth=truth_rot[k],                       # Earth-relative: the honest comparison
            m=metrics(interp, truth_rot[k]),
            m_raw=metrics(interp, truth_raw[k]),      # vs grid-relative: the artifact
            rotated=not np.allclose(truth_raw[k], truth_rot[k]),
        ))
    eds.close()
    hds.close()

    # Wind SPEED is invariant under the rotation, so it isolates magnitude error from the
    # convergence-angle artifact: if speed RMSE is good but u/v RMSE is bad, the problem is
    # direction/rotation, not the wind field itself.
    iu, iv = 1, 2
    rows.append(dict(
        label="10 m wind speed", units="m/s",
        interp=np.hypot(rows[iu]["interp"], rows[iv]["interp"]),
        truth=np.hypot(truth_rot[iu], truth_rot[iv]),
        rotated=False,
    ))
    rows[-1]["m"] = metrics(rows[-1]["interp"], rows[-1]["truth"])
    rows[-1]["m_raw"] = rows[-1]["m"]

    # ---------------------------------------------------------------- 3. optional model
    model_rows = None
    if args.ckpt:
        model_rows = run_model(cfg, probe, ts, args)

    # ---------------------------------------------------------------- 4. report
    print(f"\n{'field':<18} {'units':>5} {'RMSE':>8} {'bias':>8} {'MAE':>8} "
          f"{'corr':>6} {'std(interp)':>12} {'std(HRRR)':>10}")
    print("-" * 82)
    for r in rows:
        m = r["m"]
        print(f"{r['label']:<18} {r['units']:>5} {m['rmse']:8.3f} {m['bias']:8.3f} "
              f"{m['mae']:8.3f} {m['corr']:6.3f} {m['std_pred']:12.3f} {m['std_truth']:10.3f}")
    print("\n  Above = BILINEAR-INTERPOLATION BASELINE (no model). This is the number the")
    print("  downscaler must beat. Note std(interp) < std(HRRR): interpolation is too smooth,")
    print("  and recovering that missing fine-scale variance is the model's whole job.")

    # What the wind rotation is worth: same baseline scored against grid-relative HRRR.
    if any(r["rotated"] for r in rows):
        print(f"\n  Effect of the wind rotation (RMSE of the same interpolated ERA5 field):")
        print(f"    {'field':<18} {'vs grid-rel HRRR':>17} {'vs Earth-rel HRRR':>18} {'change':>9}")
        for r in rows:
            if not r["rotated"]:
                continue
            a, b = r["m_raw"]["rmse"], r["m"]["rmse"]
            print(f"    {r['label']:<18} {a:17.3f} {b:18.3f} {(b - a) / a * 100:+8.1f}%")
        print("    Scoring against un-rotated HRRR charges the projection's convergence angle")
        print("    to the model as if it were error. The rotated column is the honest one.")

    if model_rows:
        print(f"\n{'field':<18} {'units':>5} {'RMSE':>8} {'bias':>8} {'MAE':>8} "
              f"{'corr':>6} {'std(model)':>12} {'std(HRRR)':>10}")
        print("-" * 82)
        for r in model_rows:
            m = r["m"]
            print(f"{r['label']:<18} {r['units']:>5} {m['rmse']:8.3f} {m['bias']:8.3f} "
                  f"{m['mae']:8.3f} {m['corr']:6.3f} {m['std_pred']:12.3f} {m['std_truth']:10.3f}")

    plot(rows, model_rows, ts, args.out)
    return 0


def run_model(cfg, probe, ts, args):
    """Draw one conditional sample from a checkpoint and score it against HRRR."""
    import netCDF4
    from omegaconf import OmegaConf

    from data.normalization import Era5HrrrNorm
    from evaluation.downscale import Downscaler
    from model.preconditioning import EDMPreconditioning
    from model.unet import HRRRUNet

    if not os.path.exists(cfg.stats_path):
        raise SystemExit(f"no norm stats at {cfg.stats_path} — run scripts/precompute_stats.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = OmegaConf.load("configs/model/unet.yaml")
    norm = Era5HrrrNorm(cfg.stats_path).to(device)

    unet = HRRRUNet(
        n_input_ch=mcfg.n_input_ch, n_output_ch=mcfg.n_output_ch, C_base=mcfg.C_base,
        n_res_blocks=list(mcfg.n_res_blocks), emb_dim=mcfg.emb_dim,
        num_groups=mcfg.num_groups,
        use_attention=mcfg.use_attention, attn_depth=mcfg.attn_depth,
        attn_window_size=mcfg.attn_window_size, attn_num_heads=mcfg.attn_num_heads,
        attn_mlp_ratio=mcfg.attn_mlp_ratio,
    )
    tcfg = OmegaConf.load("configs/training/default.yaml")
    model = EDMPreconditioning(unet, sigma_data=tcfg.sigma_data).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Prefer the EMA weights — that is what you evaluate a diffusion model with.
    state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"\n  loaded {args.ckpt} (step {ckpt.get('global_step', '?')}), "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    # Conditioning must be the model's OWN channel list, not the comparison list.
    cond_ds = Era5HrrrDataset(
        timestamps=[ts], era5_dir=cfg.era5_dir, hrrr_dir=cfg.hrrr_dir,
        grid_meta_path=cfg.grid_meta_path, norm=norm,
        era5_variables=list(cfg.era5_variables),
        hrrr_prognostic_variables=list(cfg.hrrr_prognostic_variables),
        hrrr_diagnostic_variables=list(cfg.hrrr_diagnostic_variables),
        era5_path_template=cfg.era5_path_template, hrrr_path_template=cfg.hrrr_path_template,
        era5_lat_name=cfg.era5_lat_name, era5_lon_name=cfg.era5_lon_name,
        era5_level_name=cfg.era5_level_name, orog_var=cfg.orog_var, lsm_var=cfg.lsm_var,
        match_era5_lon_to_360=cfg.match_era5_lon_to_360,
    )
    sample = cond_ds[0]
    x_cond = sample["x_cond"].unsqueeze(0).to(device)

    dsc = Downscaler(model, norm, cond_ds.statics.to(device),
                     n_prognostic=len(cfg.hrrr_prognostic_variables),
                     n_diagnostic=len(cfg.hrrr_diagnostic_variables))
    prog, _ = dsc.sample(x_cond, n_steps=args.n_steps)   # x_cond already built by the dataset
    prog = prog[0].cpu().numpy()

    prog_names = [str(v) for v in cfg.hrrr_prognostic_variables]
    hds = netCDF4.Dataset(probe._hrrr_path(ts), "r")
    out = []
    for _, _, hrrr_name, units, label in COMPARE_PAIRS:
        if hrrr_name not in prog_names:
            continue   # model does not predict this channel
        pred = prog[prog_names.index(hrrr_name)]
        truth = np.asarray(hds.variables[hrrr_name][:], np.float32).reshape(probe.H, probe.W)
        out.append(dict(label=label, units=units, pred=pred, truth=truth,
                        m=metrics(pred, truth)))
    hds.close()
    return out


def plot(rows, model_rows, ts, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_label = {r["label"]: r for r in (model_rows or [])}
    ncol = 4 if model_rows else 3
    fig, axes = plt.subplots(len(rows), ncol, figsize=(5.0 * ncol, 3.4 * len(rows)))
    axes = np.atleast_2d(axes)

    for i, r in enumerate(rows):
        truth, interp = r["truth"], r["interp"]
        vmin, vmax = np.percentile(truth, [1, 99])
        common = dict(origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto")

        panels = [
            (interp, f"ERA5 1° -> HRRR grid (bilinear)\nstd={r['m']['std_pred']:.2f}", common),
            (truth, f"HRRR 3 km (truth)\nstd={r['m']['std_truth']:.2f}", common),
        ]
        if model_rows and r["label"] in by_label:
            mr = by_label[r["label"]]
            panels.append((mr["pred"], f"downscaler sample\nRMSE={mr['m']['rmse']:.2f}", common))
        diff = interp - truth
        lim = float(np.percentile(np.abs(diff), 99))
        panels.append((diff, f"interp − HRRR\nRMSE={r['m']['rmse']:.2f} {r['units']}",
                       dict(origin="lower", vmin=-lim, vmax=lim, cmap="coolwarm", aspect="auto")))

        for j, (field, title, kw) in enumerate(panels[:ncol]):
            ax = axes[i, j]
            im = ax.imshow(field, **kw)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
        axes[i, 0].set_ylabel(f"{r['label']} [{r['units']}]", fontsize=10)

    fig.suptitle(f"ERA5 vs HRRR — {ts}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"\n  figure: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
