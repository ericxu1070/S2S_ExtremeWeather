"""End-to-end smoke tests for the ERA5 -> HRRR downscaler (CPU, no real data).

Exercises the exact tensor contract the trainer relies on:
  dataset dict -> EDMPreconditioning -> EDMLoss -> backward.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dummy import DummyDataset, DummyNormalizationStats
from model.unet import HRRRUNet
from model.preconditioning import EDMPreconditioning
from training.loss import EDMLoss
from training.sigma_sampler import EDMSigmaSampler


N_ERA5, N_PROG, N_DIAG, N_STATIC = 4, 4, 1, 2
H, W = 24, 32


def test_dummy_item_shapes():
    ds = DummyDataset(N_ERA5, N_PROG, N_DIAG, N_STATIC, height=H, width=W, length=3)
    item = ds[0]
    assert set(item) == {"x_cond", "target_prognostic", "target_diagnostic", "timestamp"}
    assert item["x_cond"].shape == (N_ERA5 + N_STATIC, H, W)
    assert item["target_prognostic"].shape == (N_PROG, H, W)
    assert item["target_diagnostic"].shape == (N_DIAG, H, W)


def test_statics_constant_across_samples():
    ds = DummyDataset(N_ERA5, N_PROG, N_DIAG, N_STATIC, height=H, width=W, length=3)
    s0 = ds[0]["x_cond"][-N_STATIC:]
    s1 = ds[1]["x_cond"][-N_STATIC:]
    assert torch.equal(s0, s1)


def test_one_training_step():
    """Full denoise step: shapes line up and gradients flow, no NaNs."""
    n_out = N_PROG + N_DIAG
    n_in = n_out + N_ERA5 + N_STATIC
    unet = HRRRUNet(n_input_ch=n_in, n_output_ch=n_out, C_base=16,
                    n_res_blocks=[1, 1], emb_dim=64, num_groups=8)
    model = EDMPreconditioning(unet, sigma_data=1.0)
    loss_fn = EDMLoss(sigma_data=1.0, n_prog=N_PROG, n_diag=N_DIAG)
    sampler = EDMSigmaSampler()

    ds = DummyDataset(N_ERA5, N_PROG, N_DIAG, N_STATIC, height=H, width=W, length=2)
    item = ds[0]
    x_cond = item["x_cond"].unsqueeze(0)
    target = torch.cat([item["target_prognostic"], item["target_diagnostic"]], dim=0).unsqueeze(0)
    assert target.shape[1] == n_out

    sigma = sampler(1, torch.device("cpu"))
    x_noisy = target + sigma[:, None, None, None] * torch.randn_like(target)

    D_out = model(x_noisy, sigma, x_cond)
    assert D_out.shape == target.shape

    loss = loss_fn(D_out, target, sigma)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
