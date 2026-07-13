"""Exponential Moving Average (EMA) model wrapper.

Maintains a shadow copy of model parameters with exponential decay.
EMA model is used exclusively for validation, early stopping, and
test evaluation. Raw model weights are used for training only.
"""

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average of model parameters.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate (default 0.9999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Swap model parameters with EMA shadow parameters.

        Call restore() afterwards to revert.
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore model parameters from backup (undo apply_shadow)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    @contextmanager
    def ema_scope(self, model: nn.Module):
        """Context manager to temporarily use EMA parameters."""
        self.apply_shadow(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)
