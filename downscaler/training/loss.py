"""EDM loss function for HRRR weather prediction.

Implements Karras et al. 2022 denoising score matching loss with:
- Lambda weighting by noise level
- cos(latitude) spatial weighting for prognostic channels
- Per-variable Ocampo weighting
- Unweighted L2 for diagnostic channel (no spatial weighting)
"""

import torch
import torch.nn as nn


class EDMLoss(nn.Module):
    """EDM denoising loss with heterogeneous channel treatment.

    It is an MSE at heart -- (D_out - target)^2 on the DENOISED prediction (not on the noise)
    in normalized z-score space -- with three weightings layered on:
      1. lambda(sigma), the Karras et al. 2022 noise-level weight;
      2. cos(latitude) area weighting, prognostic channels only;
      3. per-variable (Ocampo) weights, supplied by the caller.

    Prognostic channels (0:n_prog): lambda-weighted, area-weighted, per-variable-weighted
        MSE on the normalized full HRRR field. (The downscaler predicts full fields; the
        HRRR emulator this was ported from predicted tendencies.)
    Diagnostic channels (n_prog:): lambda-weighted MSE, no area weighting.

    Args:
        sigma_data: Data std for lambda weighting (1.0 for normalized targets).
        n_prog: Number of prognostic channels.
        n_diag: Number of diagnostic channels.
        spatial_weights: (H, W) cos(lat) weights, ALREADY normalized to sum=1 -- the
            prognostic reduction is a weighted sum, so unnormalized weights would silently
            scale the prognostic loss by ~H*W relative to the diagnostic one.
        diagnostic_loss_weight: weight of each diagnostic channel relative to each
            prognostic channel in the final per-channel mean. 1.0 = all channels equal.
    """

    def __init__(
        self,
        sigma_data: float = 1.0,
        n_prog: int = 41,
        n_diag: int = 1,
        spatial_weights: torch.Tensor | None = None,
        diagnostic_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.n_prog = n_prog
        self.n_diag = n_diag
        self.diagnostic_loss_weight = float(diagnostic_loss_weight)

        assert n_prog + n_diag > 0

        if spatial_weights is not None:
            # (H, W) -> registered buffer
            self.register_buffer("spatial_weights", spatial_weights)
        else:
            self.spatial_weights = None

    def forward(
        self,
        D_out: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        variable_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            D_out: (B, n_prog+n_diag, H, W) denoised prediction from EDM.
            target: (B, n_prog+n_diag, H, W) clean normalized target.
            sigma: (B,) noise levels.
            variable_weights: (n_prog+n_diag,) per-variable weights from Ocampo.
                If None, uses uniform weights.

        Returns:
            Dict with 'loss' (scalar), 'loss_prog' (scalar), 'loss_diag' (scalar),
            and 'per_variable_loss' ((n_prog+n_diag,) tensor).
        """
        B = D_out.shape[0]

        # Lambda weighting: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        sigma_v = sigma.reshape(B, 1, 1, 1)
        sd2 = self.sigma_data**2
        lam = (sigma_v**2 + sd2) / (sigma_v * self.sigma_data) ** 2  # (B, 1, 1, 1)

        # Squared error per element
        sq_err = (D_out - target) ** 2  # (B, C, H, W)

        # Lambda-weighted squared error
        weighted_sq_err = lam * sq_err  # (B, C, H, W)

        # Split prognostic and diagnostic
        prog_err = weighted_sq_err[:, : self.n_prog]  # (B, n_prog, H, W)
        diag_err = weighted_sq_err[:, self.n_prog :]  # (B, n_diag, H, W)

        # --- Prognostic loss: spatially weighted ---
        if self.spatial_weights is not None:
            # (H, W) broadcast to (B, n_prog, H, W)
            W = self.spatial_weights[None, None, :, :]
            # Weighted spatial mean per variable per sample
            per_var_prog = (prog_err * W).sum(dim=(-2, -1))  # (B, n_prog)
        else:
            per_var_prog = prog_err.mean(dim=(-2, -1))  # (B, n_prog)

        # --- Diagnostic loss: NO spatial weighting ---
        per_var_diag = diag_err.mean(dim=(-2, -1))  # (B, n_diag)

        # Per-variable losses across batch
        per_var_all = torch.cat([per_var_prog, per_var_diag], dim=1)  # (B, C)
        per_var_mean = per_var_all.mean(dim=0)  # (C,)

        # Apply variable weights
        if variable_weights is not None:
            assert variable_weights.shape[0] == per_var_mean.shape[0], (
                f"Variable weights shape {variable_weights.shape} != "
                f"per_var_mean shape {per_var_mean.shape}"
            )
            weighted_per_var = per_var_mean * variable_weights
        else:
            weighted_per_var = per_var_mean

        # Reported separately for logging / Ocampo: these stay plain per-group means.
        loss_prog = weighted_per_var[: self.n_prog].mean()
        loss_diag = weighted_per_var[self.n_prog :].mean()

        # The TRAINED objective is a per-channel mean, not loss_prog + loss_diag. Summing the
        # two group means hands each GROUP half the gradient regardless of how many channels
        # it holds: with 4 prognostic + 1 diagnostic, precipitation alone would be 50% of the
        # loss and each of t2m/u10/v10/log_sp just 12.5%. Weighting per channel gives every
        # channel 1/5. diagnostic_loss_weight is the deliberate escape hatch — set it >1 to
        # genuinely prioritize precip, rather than getting that emphasis by accident.
        w = torch.ones(self.n_prog + self.n_diag, device=weighted_per_var.device)
        if self.n_diag > 0:
            w[self.n_prog :] = self.diagnostic_loss_weight
        loss = (weighted_per_var * w).sum() / w.sum()

        return {
            "loss": loss,
            "loss_prog": loss_prog,
            "loss_diag": loss_diag,
            "per_variable_loss": per_var_mean.detach(),
        }
