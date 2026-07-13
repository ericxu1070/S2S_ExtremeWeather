"""Ocampo et al. (2024) validation-loss-scaled variable weighting.

After a warmup period (epoch >= start_epoch), variable weights are
computed as w_i = base_weight / val_loss_i, updated every N epochs.
log_sp and log_tp weights are capped to prevent instability from
extreme-variance channels (following LUCIE-3D).
"""

import torch


class OcampoWeighting:
    """Adaptive per-variable loss weighting.

    Args:
        n_vars: Total number of variables (prognostic + diagnostic).
        start_epoch: Epoch at which weighting activates (uniform before).
        update_every: Epochs between weight updates.
        base_weight: Numerator in w_i = base_weight / val_loss_i.
        logsp_idx: Index of log_sp channel (for weight capping).
        logtp_idx: Index of log_tp channel (for weight capping).
        logsp_cap: Cap multiplier for log_sp weight.
        logtp_cap: Cap multiplier for log_tp weight.
    """

    def __init__(
        self,
        n_vars: int = 42,
        start_epoch: int = 20,
        update_every: int = 10,
        base_weight: float = 0.005,
        logsp_idx: int = 40,
        logtp_idx: int = 41,
        logsp_cap: float = 0.5,
        logtp_cap: float = 0.5,
    ):
        self.n_vars = n_vars
        self.start_epoch = start_epoch
        self.update_every = update_every
        self.base_weight = base_weight
        self.logsp_idx = logsp_idx
        self.logtp_idx = logtp_idx
        self.logsp_cap = logsp_cap
        self.logtp_cap = logtp_cap

        # Current weights — uniform until activated
        self.weights = torch.ones(n_vars)

    def update(self, epoch: int, val_per_var_loss: torch.Tensor) -> None:
        """Update weights based on validation per-variable loss.

        Args:
            epoch: Current epoch number.
            val_per_var_loss: (n_vars,) per-variable validation loss.
        """
        if epoch < self.start_epoch:
            return
        if (epoch - self.start_epoch) % self.update_every != 0:
            return

        # w_i = base_weight / val_loss_i
        # Clamp val_loss to avoid division by zero
        safe_loss = val_per_var_loss.clamp(min=1e-8)
        self.weights = self.base_weight / safe_loss

        # Cap log_sp and log_tp weights
        if self.logsp_idx < self.n_vars:
            uncapped = self.weights[self.logsp_idx].item()
            self.weights[self.logsp_idx] = min(uncapped, uncapped * self.logsp_cap)

        if self.logtp_idx < self.n_vars:
            uncapped = self.weights[self.logtp_idx].item()
            self.weights[self.logtp_idx] = min(uncapped, uncapped * self.logtp_cap)

    def get_weights(self, epoch: int, device: torch.device) -> torch.Tensor | None:
        """Get current variable weights.

        Returns None if weighting is not yet active (before start_epoch),
        which signals the loss function to use uniform weights.
        """
        if epoch < self.start_epoch:
            return None
        return self.weights.to(device)
