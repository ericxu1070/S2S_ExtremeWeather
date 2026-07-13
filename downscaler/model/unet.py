"""HRRRUNet: UNet architecture for HRRR regional weather prediction.

Designed for the full CONUS grid (1059x1799). Only lightweight Conv2d
layers operate at full resolution (encoder stem + decoder head). All
ResBlocks, noise conditioning, and skip connections operate at 2x, 4x,
or 8x downsampled scales.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import NoiseEmbedding, ResBlock, DownBlock, UpBlock


def _compute_padding(h: int, w: int, n_levels: int) -> tuple[int, int, int, int]:
    """Compute padding to make spatial dims divisible by 2^n_levels.

    Returns:
        (pad_left, pad_right, pad_top, pad_bottom) for F.pad
    """
    divisor = 2**n_levels
    new_h = math.ceil(h / divisor) * divisor
    new_w = math.ceil(w / divisor) * divisor
    pad_h = new_h - h
    pad_w = new_w - w
    # Distribute padding: more on bottom/right
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


class HRRRUNet(nn.Module):
    """UNet for HRRR EDM weather prediction.

    Args:
        n_input_ch: Total input channels (noisy scaled + conditioning).
        n_output_ch: Output channels (tendency + diagnostic).
        C_base: Base channel width.
        n_res_blocks: Number of ResBlocks per level, length = n_levels.
        emb_dim: Noise embedding dimension.
        num_groups: GroupNorm groups.
        dropout: Dropout rate in ResBlocks.
        use_strided_conv: Strided conv vs AvgPool for downsampling.
        use_attention: Enable the transformer bottleneck (Swin-style
            windowed attention blocks between the two mid ResBlocks).
        attn_depth: Number of transformer blocks at the bottleneck.
            Blocks alternate unshifted/shifted windows.
        attn_window_size: Attention window size in bottleneck pixels.
        attn_num_heads: Attention heads; must divide the bottleneck
            channel count (C_base * 2**n_levels).
        attn_mlp_ratio: Hidden/dim ratio of each block's MLP.
    """

    def __init__(
        self,
        n_input_ch: int = 90,
        n_output_ch: int = 44,
        C_base: int = 64,
        n_res_blocks: list[int] | None = None,
        emb_dim: int = 256,
        num_groups: int = 32,
        dropout: float = 0.0,
        use_strided_conv: bool = True,
        use_attention: bool = False,
        attn_depth: int = 4,
        attn_window_size: int = 16,
        attn_num_heads: int = 8,
        attn_mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if n_res_blocks is None:
            n_res_blocks = [1, 2, 2]

        self.n_levels = len(n_res_blocks)
        self.n_input_ch = n_input_ch
        self.n_output_ch = n_output_ch

        # Channel widths per level: C_base, 2*C_base, 4*C_base, 8*C_base
        channel_mults = [2**i for i in range(self.n_levels + 1)]
        channels = [C_base * m for m in channel_mults]

        # Noise embedding
        self.noise_emb = NoiseEmbedding(emb_dim)

        # Encoder stem — lightweight, full resolution
        self.conv_in = nn.Conv2d(n_input_ch, channels[0], 3, padding=1)

        # Encoder levels
        self.down_blocks = nn.ModuleList()
        for i in range(self.n_levels):
            self.down_blocks.append(
                DownBlock(
                    channels[i],
                    channels[i + 1],
                    emb_dim,
                    n_res_blocks[i],
                    num_groups,
                    dropout,
                    use_strided_conv,
                )
            )

        # Bottleneck
        bot_ch = channels[-1]
        self.mid_block1 = ResBlock(bot_ch, bot_ch, emb_dim, num_groups, dropout)
        self.mid_block2 = ResBlock(bot_ch, bot_ch, emb_dim, num_groups, dropout)

        if use_attention:
            from model.attention import TransformerBlock

            self.mid_attn = nn.ModuleList(
                TransformerBlock(
                    bot_ch,
                    emb_dim,
                    window_size=attn_window_size,
                    num_heads=attn_num_heads,
                    shift=(i % 2 == 1),
                    mlp_ratio=attn_mlp_ratio,
                )
                for i in range(attn_depth)
            )
        else:
            self.mid_attn = None

        # Decoder levels (reverse order)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(self.n_levels)):
            self.up_blocks.append(
                UpBlock(
                    channels[i + 1],     # from previous decoder level
                    channels[i + 1],     # skip from encoder (after ResBlocks, same as out_ch of DownBlock)
                    channels[i],         # output channels
                    emb_dim,
                    n_res_blocks[i],
                    num_groups,
                    dropout,
                )
            )

        # Decoder head — lightweight, full resolution
        self.norm_out = nn.GroupNorm(num_groups, channels[0])
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[0], n_output_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_input_ch, H, W) — concatenated scaled noisy + conditioning
            c_noise: (B,) — noise embedding input (ln(sigma)/4)

        Returns:
            (B, n_output_ch, H, W) — raw network output F_theta
        """
        orig_h, orig_w = x.shape[-2:]

        # Pad to make divisible by 2^n_levels
        pad = _compute_padding(orig_h, orig_w, self.n_levels)
        if any(p > 0 for p in pad):
            # pad order: (left, right, top, bottom)
            x = F.pad(x, pad, mode="reflect")

        # Noise embedding
        emb = self.noise_emb(c_noise)

        # Encoder
        h = self.conv_in(x)
        skips = []
        for down in self.down_blocks:
            h, skip = down(h, emb)
            skips.append(skip)

        # Bottleneck
        h = self.mid_block1(h, emb)
        if self.mid_attn is not None:
            for block in self.mid_attn:
                h = block(h, emb)
        h = self.mid_block2(h, emb)

        # Decoder
        for up in self.up_blocks:
            skip = skips.pop()
            h = up(h, skip, emb)

        # Head
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        # Crop back to original size
        if any(p > 0 for p in pad):
            pad_left, pad_right, pad_top, pad_bottom = pad
            h = h[
                ...,
                pad_top: h.shape[-2] - (pad_bottom if pad_bottom > 0 else None),
                pad_left: h.shape[-1] - (pad_right if pad_right > 0 else None),
            ]

        assert h.shape[-2:] == (orig_h, orig_w), (
            f"Output shape {h.shape[-2:]} != input shape ({orig_h}, {orig_w})"
        )
        return h
