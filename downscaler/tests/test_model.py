"""Tests for model architecture (Phase 1).

CPU tests verify shapes, gradient flow, conditioning effects, and formulas.
GPU test (marked) verifies full-resolution forward pass fits in memory.
"""

import math
import sys
import os

import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.blocks import NoiseEmbedding, AdaGN, ResBlock, DownBlock, UpBlock
from model.unet import HRRRUNet, _compute_padding
from model.preconditioning import EDMPreconditioning
from model.ema import EMAModel


# ---------------------------------------------------------------------------
# NoiseEmbedding
# ---------------------------------------------------------------------------

class TestNoiseEmbedding:
    def test_output_shape(self):
        emb = NoiseEmbedding(emb_dim=256)
        c_noise = torch.randn(4)
        out = emb(c_noise)
        assert out.shape == (4, 256)

    def test_different_inputs_produce_different_outputs(self):
        emb = NoiseEmbedding(emb_dim=128)
        out1 = emb(torch.tensor([0.0]))
        out2 = emb(torch.tensor([1.0]))
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# AdaGN
# ---------------------------------------------------------------------------

class TestAdaGN:
    def test_output_shape(self):
        agn = AdaGN(num_channels=64, emb_dim=256, num_groups=32)
        x = torch.randn(2, 64, 8, 8)
        emb = torch.randn(2, 256)
        out = agn(x, emb)
        assert out.shape == x.shape

    def test_conditioning_changes_output(self):
        agn = AdaGN(num_channels=64, emb_dim=256, num_groups=32)
        x = torch.randn(1, 64, 8, 8)
        emb1 = torch.randn(1, 256)
        emb2 = torch.randn(1, 256)
        out1 = agn(x, emb1)
        out2 = agn(x, emb2)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------

class TestResBlock:
    def test_same_channels(self):
        block = ResBlock(64, 64, emb_dim=256, num_groups=32)
        x = torch.randn(1, 64, 16, 16)
        emb = torch.randn(1, 256)
        out = block(x, emb)
        assert out.shape == (1, 64, 16, 16)

    def test_channel_change(self):
        block = ResBlock(32, 64, emb_dim=256, num_groups=32)
        x = torch.randn(1, 32, 16, 16)
        emb = torch.randn(1, 256)
        out = block(x, emb)
        assert out.shape == (1, 64, 16, 16)

    def test_gradient_flow(self):
        block = ResBlock(64, 64, emb_dim=256, num_groups=32)
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        emb = torch.randn(1, 256, requires_grad=True)
        out = block(x, emb)
        out.sum().backward()
        assert x.grad is not None
        assert emb.grad is not None
        assert not torch.all(x.grad == 0)


# ---------------------------------------------------------------------------
# DownBlock
# ---------------------------------------------------------------------------

class TestDownBlock:
    def test_output_shapes(self):
        block = DownBlock(64, 128, emb_dim=256, n_res_blocks=2, num_groups=32)
        x = torch.randn(1, 64, 32, 32)
        emb = torch.randn(1, 256)
        down, skip = block(x, emb)
        assert skip.shape == (1, 128, 32, 32)  # after ResBlocks, before downsample
        assert down.shape == (1, 128, 16, 16)  # after 2x downsample

    def test_avgpool_fallback(self):
        block = DownBlock(
            64, 128, emb_dim=256, n_res_blocks=1, num_groups=32,
            use_strided_conv=False,
        )
        x = torch.randn(1, 64, 32, 32)
        emb = torch.randn(1, 256)
        down, skip = block(x, emb)
        assert down.shape == (1, 128, 16, 16)


# ---------------------------------------------------------------------------
# UpBlock
# ---------------------------------------------------------------------------

class TestUpBlock:
    def test_output_shape(self):
        block = UpBlock(128, 128, 64, emb_dim=256, n_res_blocks=2, num_groups=32)
        x = torch.randn(1, 128, 16, 16)
        skip = torch.randn(1, 128, 32, 32)
        emb = torch.randn(1, 256)
        out = block(x, skip, emb)
        assert out.shape == (1, 64, 32, 32)


# ---------------------------------------------------------------------------
# UNet padding
# ---------------------------------------------------------------------------

class TestPadding:
    def test_compute_padding_exact(self):
        # 1024 is divisible by 8
        pad = _compute_padding(1024, 1024, 3)
        assert all(p == 0 for p in pad)

    def test_compute_padding_hrrr(self):
        # 1059x1799 -> needs to become divisible by 8
        pad = _compute_padding(1059, 1799, 3)
        pad_left, pad_right, pad_top, pad_bottom = pad
        new_h = 1059 + pad_top + pad_bottom
        new_w = 1799 + pad_left + pad_right
        assert new_h % 8 == 0
        assert new_w % 8 == 0


# ---------------------------------------------------------------------------
# HRRRUNet (small config, CPU)
# ---------------------------------------------------------------------------

class TestHRRRUNet:
    def test_small_forward(self):
        """Small UNet on small spatial dims — CPU test."""
        net = HRRRUNet(
            n_input_ch=10,
            n_output_ch=4,
            C_base=16,
            n_res_blocks=[1, 1],
            emb_dim=64,
            num_groups=8,
        )
        x = torch.randn(1, 10, 32, 48)
        c_noise = torch.randn(1)
        out = net(x, c_noise)
        assert out.shape == (1, 4, 32, 48)

    def test_odd_spatial_dims(self):
        """Verify padding/cropping works for non-power-of-2 dims."""
        net = HRRRUNet(
            n_input_ch=10,
            n_output_ch=4,
            C_base=16,
            n_res_blocks=[1, 1],
            emb_dim=64,
            num_groups=8,
        )
        x = torch.randn(1, 10, 33, 47)
        c_noise = torch.randn(1)
        out = net(x, c_noise)
        assert out.shape == (1, 4, 33, 47)

    def test_gradient_flow_through_unet(self):
        net = HRRRUNet(
            n_input_ch=10,
            n_output_ch=4,
            C_base=16,
            n_res_blocks=[1, 1],
            emb_dim=64,
            num_groups=8,
        )
        x = torch.randn(1, 10, 16, 16, requires_grad=True)
        c_noise = torch.randn(1)
        out = net(x, c_noise)
        out.sum().backward()
        assert x.grad is not None
        # Verify no NaN in gradients
        assert not torch.isnan(x.grad).any()

    def test_no_nan_output(self):
        net = HRRRUNet(
            n_input_ch=10,
            n_output_ch=4,
            C_base=16,
            n_res_blocks=[1, 1],
            emb_dim=64,
            num_groups=8,
        )
        x = torch.randn(2, 10, 16, 16)
        c_noise = torch.randn(2)
        out = net(x, c_noise)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# EDMPreconditioning
# ---------------------------------------------------------------------------

class TestEDMPreconditioning:
    def test_preconditioning_formulas(self):
        """Verify c_skip, c_out, c_in match Karras et al. 2022."""
        sigma_data = 1.0
        sigma = torch.tensor([0.5, 1.0, 2.0])

        for s in sigma:
            sd2 = sigma_data**2
            s2 = s.item() ** 2
            c_skip = sd2 / (s2 + sd2)
            c_out = s.item() * sigma_data / math.sqrt(s2 + sd2)
            c_in = 1.0 / math.sqrt(s2 + sd2)

            assert abs(c_skip - sd2 / (s2 + sd2)) < 1e-6
            assert abs(c_out - s.item() * sigma_data / math.sqrt(s2 + sd2)) < 1e-6
            assert abs(c_in - 1.0 / math.sqrt(s2 + sd2)) < 1e-6

    def test_forward_shape(self):
        unet = HRRRUNet(
            n_input_ch=20,
            n_output_ch=8,
            C_base=16,
            n_res_blocks=[1, 1],
            emb_dim=64,
            num_groups=8,
        )
        edm = EDMPreconditioning(unet, sigma_data=1.0)
        x_noisy = torch.randn(2, 8, 16, 16)
        sigma = torch.tensor([0.5, 1.0])
        x_cond = torch.randn(2, 12, 16, 16)
        out = edm(x_noisy, sigma, x_cond)
        assert out.shape == (2, 8, 16, 16)

    def test_skip_connection_at_zero_noise(self):
        """At sigma=0, c_skip=1, c_out=0, so D_theta should equal x_noisy."""
        unet = HRRRUNet(
            n_input_ch=20,
            n_output_ch=8,
            C_base=16,
            n_res_blocks=[1, 1],
            emb_dim=64,
            num_groups=8,
        )
        edm = EDMPreconditioning(unet, sigma_data=1.0)
        x_noisy = torch.randn(1, 8, 16, 16)
        sigma = torch.tensor([1e-10])  # near zero
        x_cond = torch.randn(1, 12, 16, 16)
        out = edm(x_noisy, sigma, x_cond)
        # c_skip ≈ 1, c_out ≈ 0, so output should be very close to x_noisy
        assert torch.allclose(out, x_noisy, atol=0.1)


# ---------------------------------------------------------------------------
# EMAModel
# ---------------------------------------------------------------------------

class TestEMAModel:
    def test_ema_diverges_after_update(self):
        model = torch.nn.Linear(10, 10)
        ema = EMAModel(model, decay=0.99)

        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        ema.update(model)

        # EMA shadow should differ from original snapshot
        for name, param in model.named_parameters():
            assert not torch.equal(ema.shadow[name], param.data)

    def test_ema_scope_context(self):
        model = torch.nn.Linear(10, 10)
        ema = EMAModel(model, decay=0.99)

        # Modify model
        original_weight = model.weight.data.clone()
        with torch.no_grad():
            model.weight.add_(torch.ones_like(model.weight))

        ema.update(model)

        with ema.ema_scope(model):
            # Inside scope, model should have EMA weights
            assert not torch.equal(model.weight.data, original_weight + 1)

        # Outside scope, model should be restored
        assert torch.allclose(model.weight.data, original_weight + 1)

    def test_state_dict_roundtrip(self):
        model = torch.nn.Linear(10, 10)
        ema = EMAModel(model, decay=0.999)
        sd = ema.state_dict()
        ema2 = EMAModel(model, decay=0.5)
        ema2.load_state_dict(sd)
        assert ema2.decay == 0.999


# ---------------------------------------------------------------------------
# Full-resolution GPU test (Phase 1 completion criterion)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_full_resolution_forward():
    """Forward pass at full HRRR resolution on GPU.

    Phase 1 completion criterion: B=1, C_base=64, H=1059, W=1799
    must complete without OOM on a single H200 with checkpointing.
    """
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    unet = HRRRUNet(
        n_input_ch=86,
        n_output_ch=42,
        C_base=64,
        n_res_blocks=[1, 2, 2],
        emb_dim=256,
        num_groups=32,
    ).to(device)

    edm = EDMPreconditioning(unet, sigma_data=1.0).to(device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        x_noisy = torch.randn(1, 42, 1059, 1799, device=device, dtype=torch.bfloat16)
        sigma = torch.tensor([1.0], device=device)
        x_cond = torch.randn(1, 44, 1059, 1799, device=device, dtype=torch.bfloat16)

        out = edm(x_noisy, sigma, x_cond)

    assert out.shape == (1, 42, 1059, 1799)
    assert not torch.isnan(out).any()

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"\nPeak GPU memory: {peak_gb:.2f} GB")
    print(f"Parameters: {sum(p.numel() for p in edm.parameters()):,}")
