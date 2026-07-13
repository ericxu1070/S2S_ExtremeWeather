"""Ocampo et al. (2024) validation-loss-scaled variable weighting.

After a warmup period (epoch >= start_epoch), variable weights are computed as
w_i ∝ base_weight / val_loss_i and then normalized to mean 1, so switching the
weighting on never rescales the TOTAL loss (an absolute rescale — 1/val_loss can
be 10-100x — would silently shift the effective LR and the meaning of grad_clip
mid-run). log_sp and log_tp raw weights are capped (following LUCIE-3D) before
the normalization: those channels have tiny validation losses, so uncapped
1/val_loss would hand them most of the gradient.

update() is driven by the trainer's validation cadence (val_every_n_epochs): it
fires on the first validation epoch >= start_epoch and then on the first
validation epoch at least update_every epochs after the previous update. The two
schedules do NOT need to be phase-aligned — the previous modular-arithmetic
schedule ((epoch - start) % update_every == 0) never intersected the trainer's
val epochs (epoch ≡ 4 mod 5) for the default start=20/every=10, so the feature
was silently dead.
"""

import torch


class OcampoWeighting:
    """Adaptive per-variable loss weighting.

    Args:
        n_vars: Total number of variables (prognostic + diagnostic).
        start_epoch: Epoch at which weighting activates (uniform before).
        update_every: Minimum number of epochs between weight updates.
        base_weight: Numerator in the raw weight base_weight / val_loss_i.
        logsp_idx: Index of log_sp channel (for weight capping).
        logtp_idx: Index of log_tp channel (for weight capping).
        logsp_cap: Absolute ceiling on the raw log_sp weight (pre-normalization).
        logtp_cap: Absolute ceiling on the raw log_tp weight (pre-normalization).
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
        self.last_update_epoch: int | None = None

    def update(self, epoch: int, val_per_var_loss: torch.Tensor) -> None:
        """Recompute weights from the per-variable validation loss.

        Called by the trainer on every validation epoch; rate-limits itself to one
        update per `update_every` epochs.

        Args:
            epoch: Current epoch number.
            val_per_var_loss: (n_vars,) per-variable validation loss.
        """
        if epoch < self.start_epoch:
            return
        if (
            self.last_update_epoch is not None
            and epoch - self.last_update_epoch < self.update_every
        ):
            return

        # Raw weights: w_i = base_weight / val_loss_i (clamped away from zero).
        raw = self.base_weight / val_per_var_loss.clamp(min=1e-8)

        # Cap the extreme-variance channels IN RAW UNITS, before normalization.
        # min(w, cap) — an actual ceiling; the previous min(w, w*cap) was just an
        # unconditional halving, not a cap.
        for idx, cap in ((self.logsp_idx, self.logsp_cap), (self.logtp_idx, self.logtp_cap)):
            if 0 <= idx < self.n_vars:
                raw[idx] = raw[idx].clamp(max=cap)

        # Normalize to mean 1: only the RELATIVE weights carry information.
        self.weights = raw * (self.n_vars / raw.sum())
        self.last_update_epoch = epoch

    def get_weights(self, epoch: int, device: torch.device) -> torch.Tensor | None:
        """Get current variable weights.

        Returns None if weighting is not yet active (before start_epoch),
        which signals the loss function to use uniform weights.
        """
        if epoch < self.start_epoch:
            return None
        return self.weights.to(device)

    # --- checkpoint plumbing: weights are training state and must survive resume ---

    def state_dict(self) -> dict:
        return {"weights": self.weights, "last_update_epoch": self.last_update_epoch}

    def load_state_dict(self, state: dict) -> None:
        self.weights = state["weights"]
        self.last_update_epoch = state.get("last_update_epoch")
