"""Tests for the transformer bottleneck (Swin-style windowed attention)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.attention import TransformerBlock, WindowedSelfAttention
from model.unet import HRRRUNet


# ---------------------------------------------------------------------------
# WindowedSelfAttention
# ---------------------------------------------------------------------------

class TestWindowedSelfAttention:
    def test_shape_preserved_non_divisible_dims(self):
        """Dims not divisible by the window size are padded and cropped back."""
        attn = WindowedSelfAttention(dim=32, window_size=8, num_heads=4)
        x = torch.randn(2, 13, 21, 32)  # (B, H, W, C), 13 and 21 not multiples of 8
        out = attn(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_unshifted_windows_are_local(self):
        """Without shift, a perturbation in one window cannot reach another."""
        torch.manual_seed(0)
        attn = WindowedSelfAttention(dim=16, window_size=8, num_heads=4).eval()
        x = torch.randn(1, 16, 16, 16)
        x2 = x.clone()
        x2[:, :8, :8, :] += 1.0  # perturb only the top-left window
        with torch.no_grad():
            base, pert = attn(x), attn(x2)
        assert not torch.allclose(base[:, :8, :8, :], pert[:, :8, :8, :])
        assert torch.allclose(base[:, 8:, :, :], pert[:, 8:, :, :], atol=1e-6)
        assert torch.allclose(base[:, :8, 8:, :], pert[:, :8, 8:, :], atol=1e-6)

    def test_shifted_windows_cross_boundaries(self):
        """With shift, information crosses the unshifted window boundary."""
        torch.manual_seed(0)
        attn = WindowedSelfAttention(dim=16, window_size=8, num_heads=4, shift=True).eval()
        x = torch.randn(1, 16, 16, 16)
        x2 = x.clone()
        x2[:, :8, :8, :] += 1.0
        with torch.no_grad():
            base, pert = attn(x), attn(x2)
        assert not torch.allclose(base[:, 8:, :, :], pert[:, 8:, :, :], atol=1e-6)

    def test_padding_is_masked_out(self):
        """Padded tokens must not change real outputs: a 12x12 input tiled by
        8x8 windows gets padded to 16x16; outputs must be finite and the op
        must still be exactly local to each window's real tokens."""
        torch.manual_seed(0)
        attn = WindowedSelfAttention(dim=16, window_size=8, num_heads=4).eval()
        x = torch.randn(1, 12, 12, 16)
        x2 = x.clone()
        x2[:, :8, :8, :] += 1.0
        with torch.no_grad():
            base, pert = attn(x), attn(x2)
        # bottom-right 4x4 block lives in a different window from the perturbation
        assert torch.allclose(base[:, 8:, 8:, :], pert[:, 8:, 8:, :], atol=1e-6)
        assert not torch.isnan(base).any()

    def test_gradient_flow(self):
        attn = WindowedSelfAttention(dim=16, window_size=4, num_heads=2)
        x = torch.randn(1, 9, 11, 16, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert attn.relative_position_bias_table.grad is not None


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def test_identity_at_init(self):
        """adaLN-Zero gates make a fresh block an exact identity map."""
        block = TransformerBlock(dim=32, emb_dim=64, window_size=8, num_heads=4)
        x = torch.randn(2, 32, 12, 18)
        emb = torch.randn(2, 64)
        out = block(x, emb)
        assert torch.equal(out, x)

    def test_noise_conditioning_changes_output(self):
        torch.manual_seed(0)
        block = TransformerBlock(dim=32, emb_dim=64, window_size=8, num_heads=4)
        with torch.no_grad():  # open the adaLN-Zero gates
            block.adaLN.weight.normal_(std=0.1)
            block.adaLN.bias.normal_(std=0.1)
        x = torch.randn(1, 32, 12, 18)
        out1 = block(x, torch.randn(1, 64))
        out2 = block(x, torch.randn(1, 64))
        assert not torch.allclose(out1, out2)
        assert not torch.allclose(out1, x)

    def test_gradient_flow_with_open_gates(self):
        torch.manual_seed(0)
        block = TransformerBlock(dim=32, emb_dim=64, window_size=8, num_heads=4)
        with torch.no_grad():
            block.adaLN.weight.normal_(std=0.1)
            block.adaLN.bias.normal_(std=0.1)
        x = torch.randn(1, 32, 12, 18, requires_grad=True)
        emb = torch.randn(1, 64, requires_grad=True)
        block(x, emb).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)
        assert emb.grad is not None
        assert block.attn.qkv.weight.grad is not None
        assert not torch.all(block.attn.qkv.weight.grad == 0)


# ---------------------------------------------------------------------------
# HRRRUNet with the transformer bottleneck
# ---------------------------------------------------------------------------

class TestUNetWithTransformerBottleneck:
    def _build(self):
        return HRRRUNet(
            n_input_ch=10,
            n_output_ch=4,
            C_base=16,
            n_res_blocks=[1, 1],
            emb_dim=64,
            num_groups=8,
            use_attention=True,
            attn_depth=2,
            attn_window_size=4,
            attn_num_heads=4,
        )

    def test_forward_odd_spatial_dims(self):
        net = self._build()
        x = torch.randn(1, 10, 33, 47)
        c_noise = torch.randn(1)
        out = net(x, c_noise)
        assert out.shape == (1, 4, 33, 47)
        assert not torch.isnan(out).any()

    def test_blocks_alternate_shift(self):
        net = self._build()
        assert len(net.mid_attn) == 2
        assert net.mid_attn[0].attn.shift is False
        assert net.mid_attn[1].attn.shift is True

    def test_attention_params_train(self):
        net = self._build()
        with torch.no_grad():  # open the gates so gradients reach the attn weights
            for blk in net.mid_attn:
                blk.adaLN.weight.normal_(std=0.1)
        x = torch.randn(1, 10, 16, 16)
        c_noise = torch.randn(1)
        net(x, c_noise).sum().backward()
        for blk in net.mid_attn:
            assert blk.attn.qkv.weight.grad is not None
            assert not torch.all(blk.attn.qkv.weight.grad == 0)
