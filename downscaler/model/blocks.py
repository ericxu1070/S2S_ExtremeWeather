"""Core building blocks for the HRRR EDM UNet.

NoiseEmbedding: Sinusoidal embedding of continuous sigma for EDM.
AdaGN: Adaptive Group Normalization conditioned on noise embedding.
ResBlock: Residual block with AdaGN and activation checkpointing.
DownBlock: Encoder block with ResBlocks and spatial downsampling.
UpBlock: Decoder block with skip connections and spatial upsampling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class NoiseEmbedding(nn.Module):
    """Sinusoidal embedding of EDM noise level c_noise = ln(sigma)/4.

    Adapted from LUCIE's get_time_embedding, but takes continuous float
    instead of integer timestep.
    """

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_noise: (B,) noise level embedding input (ln(sigma)/4)
        Returns:
            (B, emb_dim) noise embedding
        """
        half_dim = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=c_noise.device, dtype=torch.float32)
            / half_dim
        )
        # (B, half_dim)
        args = c_noise[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class AdaGN(nn.Module):
    """Adaptive Group Normalization.

    Applies GroupNorm then modulates with learned scale and shift
    derived from the noise embedding.
    """

    def __init__(self, num_channels: int, emb_dim: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)
        self.proj = nn.Linear(emb_dim, 2 * num_channels)
        # Initialize scale=1, shift=0 so AdaGN starts as standard GroupNorm
        nn.init.ones_(self.proj.weight[:num_channels])
        nn.init.zeros_(self.proj.weight[num_channels:])
        nn.init.ones_(self.proj.bias[:num_channels])
        nn.init.zeros_(self.proj.bias[num_channels:])

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            emb: (B, emb_dim) noise embedding
        Returns:
            (B, C, H, W) normalized and modulated
        """
        h = self.norm(x)
        scale_shift = self.proj(F.silu(emb))  # (B, 2*C)
        scale, shift = scale_shift.chunk(2, dim=-1)
        # (B, C) -> (B, C, 1, 1) for broadcasting
        return scale[:, :, None, None] * h + shift[:, :, None, None]


class ResBlock(nn.Module):
    """Residual block with AdaGN noise conditioning.

    Two Conv2d layers, each followed by AdaGN and SiLU activation.
    Skip connection via 1x1 conv if channel dimensions differ.
    Wrapped in activation checkpointing to save memory.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.adagn1 = AdaGN(out_ch, emb_dim, num_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.adagn2 = AdaGN(out_ch, emb_dim, num_groups)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.adagn1(h, emb)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.adagn2(h, emb)
        h = self.act(h)
        return h + self.skip(x)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, x, emb, use_reentrant=False)


class DownBlock(nn.Module):
    """Encoder block: ResBlocks followed by 2x spatial downsampling.

    Args:
        use_strided_conv: If True, use stride-2 Conv2d for downsampling.
            If False, use AvgPool2d + Conv2d (more stable fallback).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        n_res_blocks: int = 2,
        num_groups: int = 32,
        dropout: float = 0.0,
        use_strided_conv: bool = True,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(n_res_blocks):
            self.res_blocks.append(
                ResBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    emb_dim,
                    num_groups,
                    dropout,
                )
            )

        if use_strided_conv:
            self.downsample = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
        else:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (downsampled_output, skip_connection)
            skip is the output after ResBlocks, before downsampling.
        """
        for block in self.res_blocks:
            x = block(x, emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Decoder block: bilinear upsample, skip concat, then ResBlocks."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        emb_dim: int,
        n_res_blocks: int = 2,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.upsample_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.res_blocks = nn.ModuleList()
        for i in range(n_res_blocks):
            self.res_blocks.append(
                ResBlock(
                    (in_ch + skip_ch) if i == 0 else out_ch,
                    out_ch,
                    emb_dim,
                    num_groups,
                    dropout,
                )
            )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, H, W) from previous decoder level
            skip: (B, skip_ch, 2H, 2W) from encoder
            emb: (B, emb_dim) noise embedding
        """
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upsample_conv(x)
        # Handle spatial size mismatch from odd dimensions
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]])
        x = torch.cat([x, skip], dim=1)
        for block in self.res_blocks:
            x = block(x, emb)
        return x
