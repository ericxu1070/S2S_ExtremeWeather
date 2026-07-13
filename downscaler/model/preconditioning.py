"""EDM preconditioning wrapper (Karras et al. 2022).

Wraps the raw UNet F_theta with skip-connection preconditioning:
    D_theta(x_noisy, sigma, x_cond) = c_skip * x_noisy + c_out * F_theta(c_in * x_noisy, c_noise, x_cond)

sigma_data = 1.0 is valid because both normalized tendencies (prognostic)
and normalized diagnostic full field have std ~= 1 by construction.
"""

import torch
import torch.nn as nn

from model.unet import HRRRUNet


class EDMPreconditioning(nn.Module):
    """EDM preconditioning wrapper around HRRRUNet.

    Args:
        unet: The raw UNet network F_theta.
        sigma_data: Data standard deviation (1.0 for properly normalized targets).
        n_cond_ch: Number of conditioning channels (appended to scaled noisy input).
    """

    def __init__(
        self,
        unet: HRRRUNet,
        sigma_data: float = 1.0,
    ):
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data

    def forward(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        x_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_noisy: (B, 44, H, W) noisy target (tendency + diagnostic)
            sigma: (B,) noise levels
            x_cond: (B, 46, H, W) conditioning input (normalized prognostic + diag + statics)

        Returns:
            D_theta: (B, 44, H, W) denoised prediction
        """
        sigma = sigma.reshape(-1, 1, 1, 1)  # (B, 1, 1, 1)
        sd2 = self.sigma_data**2

        c_skip = sd2 / (sigma**2 + sd2)
        c_out = sigma * self.sigma_data / (sigma**2 + sd2).sqrt()
        c_in = 1.0 / (sigma**2 + sd2).sqrt()
        c_noise = sigma.reshape(-1).log() / 4.0  # (B,)

        # Scale noisy input and concatenate conditioning
        net_input = torch.cat([c_in * x_noisy, x_cond], dim=1)

        # Raw UNet output
        F_out = self.unet(net_input, c_noise)

        # EDM preconditioning: skip connection + scaled network output
        D_out = c_skip * x_noisy + c_out * F_out

        return D_out
