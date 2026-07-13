"""Hydra entry point for ERA5 -> HRRR downscaler training.

Architecture (UNet + EDM preconditioning + EMA + DDP training loop) is ported
verbatim from the HRRR EDM emulator in
    /home/ubuntu/Vayuh/data/moein/regional/hrrr_edm
The task differs: this model DOWNSCALES coarse ERA5 (1 deg) to HRRR (3 km),
conditioning on the upsampled ERA5 field and predicting the HRRR *state*
(not a T->T+6h tendency). See README.md.

Usage:
    # Single GPU / CPU smoke test — no real data needed:
    python train.py data.use_dummy=true training.max_steps=5

    # Multi-GPU DDP:
    torchrun --nproc_per_node=4 train.py training.max_steps=100

    # Override config:
    torchrun --nproc_per_node=4 train.py model.C_base=32 training.lr_max=2e-4
"""

import logging
import os
import sys
from datetime import timedelta

import hydra
from omegaconf import DictConfig
import torch

log = logging.getLogger(__name__)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from model.unet import HRRRUNet
from model.preconditioning import EDMPreconditioning
from model.ema import EMAModel
from training.trainer import Trainer
from training.loss import EDMLoss
from training.sigma_sampler import EDMSigmaSampler
from training.variable_weighting import OcampoWeighting


def setup_ddp():
    """Initialize DDP if running under torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0


def build_model(cfg: DictConfig, device: torch.device) -> EDMPreconditioning:
    """Build EDMPreconditioning-wrapped HRRRUNet from config."""
    unet = HRRRUNet(
        n_input_ch=cfg.model.n_input_ch,
        n_output_ch=cfg.model.n_output_ch,
        C_base=cfg.model.C_base,
        n_res_blocks=list(cfg.model.n_res_blocks),
        emb_dim=cfg.model.emb_dim,
        num_groups=cfg.model.num_groups,
        dropout=cfg.model.dropout,
        use_strided_conv=cfg.model.use_strided_conv,
        use_attention=cfg.model.use_attention,
        attn_depth=cfg.model.attn_depth,
        attn_window_size=cfg.model.attn_window_size,
        attn_num_heads=cfg.model.attn_num_heads,
        attn_mlp_ratio=cfg.model.attn_mlp_ratio,
    )
    model = EDMPreconditioning(unet, sigma_data=cfg.training.sigma_data)
    return model.to(device)


def build_dataloaders(cfg: DictConfig, world_size: int, rank: int):
    """Build train and validation DataLoaders."""
    if cfg.data.use_dummy:
        from data.dummy import DummyDataset

        common = dict(
            n_era5_cond=cfg.data.n_era5_cond,
            n_prognostic=cfg.data.n_prognostic,
            n_diagnostic=cfg.data.n_diagnostic,
            n_static=cfg.data.n_static,
            height=cfg.data.grid_height,
            width=cfg.data.grid_width,
        )
        train_ds = DummyDataset(length=100, **common)
        val_ds = DummyDataset(length=20, **common)
    else:
        from data.normalization import Era5HrrrNorm
        from data.era5_hrrr_dataset import Era5HrrrDataset
        from data.dataloader import load_timestamps

        norm = Era5HrrrNorm(cfg.data.stats_path)
        ds_kwargs = dict(
            era5_dir=cfg.data.era5_dir,
            hrrr_dir=cfg.data.hrrr_dir,
            grid_meta_path=cfg.data.grid_meta_path,
            norm=norm,
            era5_variables=list(cfg.data.era5_variables),
            hrrr_prognostic_variables=list(cfg.data.hrrr_prognostic_variables),
            hrrr_diagnostic_variables=list(cfg.data.hrrr_diagnostic_variables),
            era5_path_template=cfg.data.era5_path_template,
            hrrr_path_template=cfg.data.hrrr_path_template,
            era5_lat_name=cfg.data.era5_lat_name,
            era5_lon_name=cfg.data.era5_lon_name,
            era5_level_name=cfg.data.era5_level_name,
            orog_var=cfg.data.orog_var,
            lsm_var=cfg.data.lsm_var,
            match_era5_lon_to_360=cfg.data.match_era5_lon_to_360,
            rotate_hrrr_winds=cfg.data.rotate_hrrr_winds,
            hrrr_wind_pairs=[list(p) for p in cfg.data.hrrr_wind_pairs],
        )
        train_ts = load_timestamps(cfg.data.index_path, int(cfg.data.train_start[:4]), int(cfg.data.train_end[:4]))
        val_ts = load_timestamps(cfg.data.index_path, int(cfg.data.val_start[:4]), int(cfg.data.val_end[:4]))
        train_ds = Era5HrrrDataset(train_ts, **ds_kwargs)
        val_ds = Era5HrrrDataset(val_ts, **ds_kwargs)

    use_ddp = dist.is_initialized()
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

    mp_ctx = os.environ.get("DATA_MP_CONTEXT") or None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.per_gpu_batch,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
        multiprocessing_context=mp_ctx,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.per_gpu_batch,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
        multiprocessing_context=mp_ctx,
    )
    return train_loader, val_loader


def validate_channel_counts(cfg: DictConfig) -> None:
    """Fail fast if the variable lists, channel counts, and model I/O disagree."""
    d, m = cfg.data, cfg.model
    n_out = d.n_prognostic + d.n_diagnostic
    checks = [
        (d.n_era5_cond == len(d.era5_variables),
         f"n_era5_cond({d.n_era5_cond}) != len(era5_variables)({len(d.era5_variables)})"),
        (d.n_prognostic == len(d.hrrr_prognostic_variables),
         f"n_prognostic({d.n_prognostic}) != len(hrrr_prognostic_variables)({len(d.hrrr_prognostic_variables)})"),
        (d.n_diagnostic == len(d.hrrr_diagnostic_variables),
         f"n_diagnostic({d.n_diagnostic}) != len(hrrr_diagnostic_variables)({len(d.hrrr_diagnostic_variables)})"),
        (m.n_output_ch == n_out,
         f"model.n_output_ch({m.n_output_ch}) != n_prognostic+n_diagnostic({n_out})"),
        (m.n_input_ch == n_out + d.n_era5_cond + d.n_static,
         f"model.n_input_ch({m.n_input_ch}) != n_output_ch+n_era5_cond+n_static"
         f"({n_out}+{d.n_era5_cond}+{d.n_static}={n_out + d.n_era5_cond + d.n_static})"),
    ]
    errors = [msg for ok, msg in checks if not ok]
    if errors:
        raise ValueError("Channel-count config is inconsistent:\n  - " + "\n  - ".join(errors))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    validate_channel_counts(cfg)
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0

    if is_main:
        log.info(f"Device: {device}, World size: {world_size}")

    torch.manual_seed(cfg.seed + rank)

    model = build_model(cfg, device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"Model parameters: {n_params:,}")

    ema = EMAModel(model, decay=cfg.training.ema_decay)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    train_loader, val_loader = build_dataloaders(cfg, world_size, local_rank)

    # Loss — HRRR-grid cos(lat) spatial weights for prognostic channels (real data only).
    if not cfg.data.use_dummy:
        from data.normalization import Era5HrrrNorm
        _stats = Era5HrrrNorm(cfg.data.stats_path)
        cos_lat_w = _stats.cos_lat_weights / _stats.cos_lat_weights.sum()
    else:
        cos_lat_w = None

    loss_fn = EDMLoss(
        sigma_data=cfg.training.sigma_data,
        n_prog=cfg.data.n_prognostic,
        n_diag=cfg.data.n_diagnostic,
        spatial_weights=cos_lat_w,
        diagnostic_loss_weight=cfg.training.diagnostic_loss_weight,
    ).to(device)

    sigma_sampler = EDMSigmaSampler(
        P_mean=cfg.training.P_mean,
        P_std=cfg.training.P_std,
        sigma_min=cfg.training.sigma_min,
        sigma_max=cfg.training.sigma_max,
    )

    # Adaptive per-variable weighting. The log_sp/log_tp caps are only meaningful
    # if those channels exist at these indices; harmless otherwise (guarded by
    # `idx < n_vars` inside OcampoWeighting).
    variable_weighting = OcampoWeighting(
        n_vars=cfg.data.n_prognostic + cfg.data.n_diagnostic,
        start_epoch=cfg.training.weight_update_start_epoch,
        update_every=cfg.training.weight_update_every_n_epochs,
        logsp_idx=cfg.data.n_prognostic - 1,
        logtp_idx=cfg.data.n_prognostic,
    )

    trainer = Trainer(
        model=model,
        ema=ema,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        sigma_sampler=sigma_sampler,
        variable_weighting=variable_weighting,
        cfg=cfg,
    )

    ckpt_path = cfg.get("resume_from", None)
    if ckpt_path:
        trainer.load_checkpoint(ckpt_path)

    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
