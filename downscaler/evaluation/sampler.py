"""EDM deterministic ODE sampler (Karras et al. 2022, Algorithm 1).

Heun 2nd-order solver for the probability flow ODE. Default 18 NFE.
"""

import torch


def get_sigma_schedule(
    n_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
) -> torch.Tensor:
    """EDM noise schedule (Karras et al. 2022, Eq. 5).

    Returns (n_steps + 1,) sigma values from sigma_max down to 0.
    """
    step_indices = torch.arange(n_steps, device=device, dtype=torch.float64)
    sigmas = (
        sigma_max ** (1 / rho)
        + step_indices / (n_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    # Append final sigma=0 for the last step
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=torch.float64)])
    return sigmas.float()


@torch.no_grad()
def edm_heun_sample(
    model,
    x_cond: torch.Tensor,
    n_target_ch: int,
    n_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    s_churn: float = 0.0,
) -> torch.Tensor:
    """Sample from EDM model using Heun 2nd-order ODE solver.

    Args:
        model: EDMPreconditioning module (callable as model(x_noisy, sigma, x_cond)).
        x_cond: (B, C_cond, H, W) conditioning input.
        n_target_ch: Number of output channels (42 = 41 tendency + 1 diagnostic).
        n_steps: Number of denoising steps (NFE = 2*n_steps - 1 for Heun).
        sigma_min, sigma_max, rho: Noise schedule parameters.
        s_churn: Stochasticity parameter (0 = deterministic ODE).

    Returns:
        (B, n_target_ch, H, W) denoised sample.
    """
    B, _, H, W = x_cond.shape
    device = x_cond.device
    dtype = x_cond.dtype

    sigmas = get_sigma_schedule(n_steps, sigma_min, sigma_max, rho, device)

    # Initialize from pure noise scaled by sigma_max
    x = torch.randn(B, n_target_ch, H, W, device=device, dtype=dtype) * sigmas[0]

    for i in range(n_steps):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Optional stochastic churn (s_churn > 0 turns Heun into a stochastic SDE solver)
        if s_churn > 0 and sigma_cur > sigma_min:
            gamma = min(s_churn / n_steps, 2**0.5 - 1)
            sigma_hat = sigma_cur * (1 + gamma)
            eps = torch.randn_like(x) * (sigma_hat**2 - sigma_cur**2).sqrt()
            x_hat = x + eps
        else:
            sigma_hat = sigma_cur
            x_hat = x

        # Apply model with current sigma
        sigma_in = sigma_hat.expand(B)
        denoised = model(x_hat, sigma_in, x_cond)

        # Euler step
        d_cur = (x_hat - denoised) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d_cur

        # Heun correction (skip if sigma_next == 0)
        if sigma_next > 0:
            sigma_in = sigma_next.expand(B)
            denoised_next = model(x_next, sigma_in, x_cond)
            d_next = (x_next - denoised_next) / sigma_next
            x_next = x_hat + (sigma_next - sigma_hat) * (d_cur + d_next) / 2

        x = x_next

    return x
