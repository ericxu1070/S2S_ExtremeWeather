"""Swin-style windowed self-attention for the UNet bottleneck.

WindowedSelfAttention is the bare multi-head attention op over
non-overlapping (optionally cyclically shifted) windows with a learned
relative position bias (Liu et al. 2021). TransformerBlock wraps it into
a pre-norm transformer block (W-MSA + MLP) whose LayerNorms are modulated
by the EDM noise embedding via adaLN-Zero (Peebles & Xie 2023): the
per-branch gates are zero-initialized, so a freshly built block is an
exact identity and enabling attention does not perturb early training.

Blocks are meant to be stacked with alternating shift=False/True so
information propagates across window boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_MASK_NEG = -1e4  # additive mask value; exp() underflows to 0, safe in bf16


def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, ws*ws, C). H and W must be multiples of ws."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def _window_reverse(windows: torch.Tensor, ws: int, B: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws*ws, C) -> (B, H, W, C)."""
    C = windows.shape[-1]
    x = windows.view(B, H // ws, W // ws, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


class WindowedSelfAttention(nn.Module):
    """Multi-head self-attention within ws x ws windows.

    Operates on channels-last feature maps (B, H, W, C). Spatial dims are
    padded up to multiples of the window size internally and padded tokens
    are masked out of attention. With shift=True the windows are cyclically
    shifted by ws//2 and the standard Swin region mask keeps wrapped-around
    tokens from attending to each other.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 16,
        num_heads: int = 8,
        shift: bool = False,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift = shift

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Learned relative position bias, one table entry per (dy, dx) offset
        ws = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        ).flatten(1)                                      # (2, N)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)  # (N, N, 2)
        rel[..., 0] += ws - 1
        rel[..., 1] += ws - 1
        rel[..., 0] *= 2 * ws - 1
        self.register_buffer(
            "relative_position_index", rel.sum(-1), persistent=False
        )

        self._mask_cache: dict = {}

    def _region_mask(
        self, H: int, W: int, Hp: int, Wp: int, shift: int, device: torch.device
    ) -> torch.Tensor | None:
        """Additive (nW, N, N) mask handling both padding and cyclic shift.

        Tokens carry a region label; attention is allowed only within the
        same region. Labels follow the Swin slice construction (defined in
        the post-roll frame); padded tokens get offset labels so no real
        token attends to padding. Returns None when no masking is needed.
        """
        if shift == 0 and Hp == H and Wp == W:
            return None
        key = (H, W, Hp, Wp, shift, device)
        if key in self._mask_cache:
            return self._mask_cache[key]

        ws = self.window_size
        region = torch.zeros(1, Hp, Wp, 1, device=device)
        if shift > 0:
            cnt = 0
            for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
                for wsl in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
                    region[:, hs, wsl, :] = cnt
                    cnt += 1
        # Padding lives at the bottom/right in the pre-roll frame; roll it
        # into the same frame as the region labels. Offset by 9 (the max
        # number of shift regions) so padded labels never collide with real ones.
        pad = torch.zeros(1, Hp, Wp, 1, device=device)
        pad[:, H:, :, :] = 1.0
        pad[:, :, W:, :] = 1.0
        if shift > 0:
            pad = torch.roll(pad, shifts=(-shift, -shift), dims=(1, 2))
        region = region + 9.0 * pad

        rw = _window_partition(region, ws).squeeze(-1)    # (nW, N)
        diff = rw.unsqueeze(1) - rw.unsqueeze(2)          # (nW, N, N)
        mask = torch.where(diff == 0, 0.0, _MASK_NEG)
        self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) channels-last feature map
        Returns:
            (B, H, W, C)
        """
        B, H, W, C = x.shape
        ws = self.window_size
        shift = ws // 2 if self.shift else 0
        N = ws * ws

        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        Hp, Wp = H + pad_b, W + pad_r
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        if shift:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))

        qkv = _window_partition(self.qkv(x), ws)          # (B*nW, N, 3C)
        q, k, v = (
            qkv.view(-1, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )                                                 # each (B*nW, nH, N, hd)

        bias = (
            self.relative_position_bias_table[self.relative_position_index]
            .view(N, N, -1)
            .permute(2, 0, 1)
        )                                                 # (nH, N, N)
        region = self._region_mask(H, W, Hp, Wp, shift, x.device)
        if region is not None:
            attn_mask = (bias.unsqueeze(0) + region.unsqueeze(1)).repeat(B, 1, 1, 1)
        else:
            attn_mask = bias.unsqueeze(0)                 # broadcasts over B*nW

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(q.dtype))
        out = out.transpose(1, 2).reshape(-1, N, C)
        out = self.proj(out)

        x = _window_reverse(out, ws, B, Hp, Wp)
        if shift:
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
        if pad_b or pad_r:
            x = x[:, :H, :W, :]
        return x


class TransformerBlock(nn.Module):
    """Pre-norm windowed transformer block with adaLN-Zero noise conditioning.

    Takes and returns channels-first maps (B, C, H, W) so it drops in
    between ResBlocks. Wrapped in activation checkpointing like ResBlock.
    """

    def __init__(
        self,
        dim: int,
        emb_dim: int,
        window_size: int = 16,
        num_heads: int = 8,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = WindowedSelfAttention(dim, window_size, num_heads, shift, qkv_bias)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # adaLN-Zero: zero-init so gates start at 0 and the block is an identity
        self.adaLN = nn.Linear(emb_dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        t = x.permute(0, 2, 3, 1)                         # (B, H, W, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN(F.silu(emb))[:, None, None, :].chunk(6, dim=-1)
        )
        h = self.norm1(t) * (1 + scale_msa) + shift_msa
        t = t + gate_msa * self.attn(h)
        h = self.norm2(t) * (1 + scale_mlp) + shift_mlp
        t = t + gate_mlp * self.mlp(h)
        return t.permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, x, emb, use_reentrant=False)
