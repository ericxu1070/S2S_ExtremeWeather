# Downscaler hyperparameters

Snapshot of the Hydra config tree at `downscaler/configs/` (defaults: `model=unet`,
`training=default`, `data=era5_hrrr`, `normalization=default`). These are the values a
plain `python train.py` run uses; check the config files themselves for authoritative
current values.

| Category | Hyperparameter | Value |
|---|---|---|
| **Model (U-Net)** | `C_base` | 64 |
| | `n_res_blocks` | [1, 2, 2] |
| | `num_groups` (GroupNorm) | 32 |
| | `emb_dim` (noise embedding) | 256 |
| | `dropout` | 0.0 |
| | `use_strided_conv` | true |
| **Attention (Swin bottleneck)** | `use_attention` | true |
| | `attn_depth` | 4 |
| | `attn_window_size` | 16 |
| | `attn_num_heads` | 8 (must divide bottleneck channels = 512) |
| | `attn_mlp_ratio` | 4.0 |
| **Channels** | `n_input_ch` | 16 (5 noisy target + 9 ERA5 + 2 static) |
| | `n_output_ch` | 5 (4 prognostic + 1 diagnostic) |
| **Diffusion (EDM)** | `sigma_data` | 1.0 |
| | `P_mean` | −1.2 |
| | `P_std` | 1.2 |
| | `sigma_min` | 0.002 |
| | `sigma_max` | 80.0 |
| **Optimizer** | `lr_max` | 1e-4 |
| | `lr_min` | 1e-8 (cosine decay) |
| | `weight_decay` | 1e-5 |
| | `grad_clip` | 1.0 |
| | `fused_optimizer` | true |
| | `ema_decay` | 0.9999 |
| | `precision` | bf16 |
| **Batch** | `per_gpu_batch` | 1 |
| | `accumulation_steps` | 16 (effective 128 on 8 GPUs) |
| **Schedule** | `n_epochs` | 200 |
| | `phase2_start_epoch` | 160 |
| | `val_every_n_epochs` | 5 |
| | `rollout_every_n_epochs` | 20 |
| | `rollout_start_epoch` | 150 |
| **Variable weighting (Ocampo)** | `weight_update_start_epoch` | 20 |
| | `weight_update_every_n_epochs` | 10 |
| | `logsp_weight_cap` | 0.5 |
| | `logtp_weight_cap` | 0.5 |
| **Loss / regularization** | `diagnostic_loss_weight` | 1.0 |
| | `spectral_reg_weight` | 0.05 |
| | `spectral_reg_start_epoch` | 161 |
| **Checkpointing** | `save_every_n_steps` | 100 |
| | `keep_last_n_checkpoints` | 3 |
| | `rebase_lr_on_resume` | true |
| **Data** | ERA5 drivers | 9 (t2m, u10/v10, u/v@850, MSLP, q@850/700, tp) |
| | HRRR targets | t2m, u10, v10, log_sp + log_tp (diagnostic) |
| | Grid | 1059 × 1799 (3 km CONUS) |
| | Split | train 2015–22 / val 2023 / test 2024–25 |
| | Normalization | per-channel z-score (orog min-max, lsm raw) |
| | `num_workers` / `prefetch_factor` | 12 / 2 |
| **Misc** | `seed` | 42 |
