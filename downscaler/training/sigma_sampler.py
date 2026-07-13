"""EDM noise level sampler (Karras et al. 2022).

Samples sigma from a log-normal distribution:
    ln(sigma) ~ N(P_mean, P_std^2), clipped to [sigma_min, sigma_max]

P_mean and P_std are high-priority hyperparameters to sweep early —
the default values (-1.2, 1.2) are tuned for natural images and may
not be optimal for atmospheric tendencies.
"""

import torch


class EDMSigmaSampler:
    """Log-normal noise level sampler for EDM training.

    Args:
        P_mean: Mean of ln(sigma) distribution.
        P_std: Std of ln(sigma) distribution.
        sigma_min: Minimum sigma (clamp).
        sigma_max: Maximum sigma (clamp).
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample sigma values for a batch.

        Returns:
            (batch_size,) tensor of sigma values.
        """
        log_sigma = self.P_mean + self.P_std * torch.randn(
            batch_size, device=device
        )
        sigma = log_sigma.exp().clamp(self.sigma_min, self.sigma_max)
        return sigma
