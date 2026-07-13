"""Normalization statistics for the ERA5 -> HRRR downscaler.

Two independent z-score systems + one diagnostic system + statics:

  System E (era5):  z-score for the coarse ERA5 conditioning input.
      x_norm = (x - era5_mu) / era5_sigma            (n_era5 channels)

  System H (hrrr):  z-score for the HRRR prognostic *state* target.
      Unlike the HRRR emulator (which normalized a zero-mean *tendency*),
      the downscaler predicts the full HRRR field, so the mean is
      subtracted here:  y_norm = (y - hrrr_mu) / hrrr_sigma   (n_prognostic ch)

  System D (diag):  z-score for the HRRR diagnostic target (e.g. log_tp).
      d_norm = (d - diag_mu) / diag_sigma            (n_diagnostic channels)

Statics (orog min-max to [0,1]; lsm binary) live on the HRRR grid and are
normalized in the dataset, not here.

`sigma_data = 1.0` in the EDM preconditioning is valid because all three
systems produce ~unit-variance normalized targets by construction.

Expected stats.npz keys (see scripts/precompute_stats.py):
    era5_mu, era5_sigma          (n_era5,)
    hrrr_mu, hrrr_sigma          (n_prognostic,)
    diag_mu, diag_sigma          (n_diagnostic,)
    orog_min, orog_max           (scalars)
    cos_lat_weights              (H, W)   -- HRRR grid area weights
    era5_channel_names           (n_era5,)      [optional]
    hrrr_channel_names           (n_prognostic,)[optional]
"""

import numpy as np
import torch


class Era5HrrrNorm:
    """Loads stats.npz and provides normalize/denormalize for the downscaler.

    Mirrors the DummyNormalizationStats interface so the training loop cannot
    tell real and dummy data apart.
    """

    def __init__(self, stats_path: str):
        data = np.load(stats_path, allow_pickle=True)

        # System E: ERA5 conditioning input
        self.era5_mu = torch.from_numpy(data["era5_mu"].copy()).float()
        self.era5_sigma = torch.from_numpy(data["era5_sigma"].copy()).float()

        # System H: HRRR prognostic full-field target
        self.hrrr_mu = torch.from_numpy(data["hrrr_mu"].copy()).float()
        self.hrrr_sigma = torch.from_numpy(data["hrrr_sigma"].copy()).float()

        # System D: HRRR diagnostic target
        self.diag_mu = torch.from_numpy(data["diag_mu"].copy()).float()
        self.diag_sigma = torch.from_numpy(data["diag_sigma"].copy()).float()

        # Static normalization
        self.orog_min = float(data["orog_min"])
        self.orog_max = float(data["orog_max"])

        # Metadata
        self.n_era5 = len(self.era5_mu)
        self.n_prognostic = len(self.hrrr_mu)
        self.n_diagnostic = len(self.diag_mu)
        self.era5_channel_names = list(data["era5_channel_names"]) if "era5_channel_names" in data else None
        self.hrrr_channel_names = list(data["hrrr_channel_names"]) if "hrrr_channel_names" in data else None
        self.cos_lat_weights = torch.from_numpy(data["cos_lat_weights"].copy()).float()

    @staticmethod
    def _shape(x: torch.Tensor) -> tuple[int, ...]:
        return (-1, 1, 1) if x.dim() == 3 else (1, -1, 1, 1)

    # --- System E: ERA5 conditioning input ---
    def normalize_era5(self, x: torch.Tensor) -> torch.Tensor:
        s = self._shape(x)
        return (x - self.era5_mu.reshape(s)) / self.era5_sigma.reshape(s)

    def denormalize_era5(self, x: torch.Tensor) -> torch.Tensor:
        s = self._shape(x)
        return x * self.era5_sigma.reshape(s) + self.era5_mu.reshape(s)

    # --- System H: HRRR prognostic full-field target ---
    def normalize_hrrr(self, y: torch.Tensor) -> torch.Tensor:
        s = self._shape(y)
        return (y - self.hrrr_mu.reshape(s)) / self.hrrr_sigma.reshape(s)

    def denormalize_hrrr(self, y: torch.Tensor) -> torch.Tensor:
        s = self._shape(y)
        return y * self.hrrr_sigma.reshape(s) + self.hrrr_mu.reshape(s)

    # --- System D: HRRR diagnostic target ---
    def normalize_diagnostic(self, d: torch.Tensor) -> torch.Tensor:
        s = self._shape(d)
        return (d - self.diag_mu.reshape(s)) / self.diag_sigma.reshape(s)

    def denormalize_diagnostic(self, d: torch.Tensor) -> torch.Tensor:
        s = self._shape(d)
        return d * self.diag_sigma.reshape(s) + self.diag_mu.reshape(s)

    # --- Statics ---
    def normalize_orog(self, orog: torch.Tensor) -> torch.Tensor:
        return (orog - self.orog_min) / max(self.orog_max - self.orog_min, 1e-8)

    def to(self, device: torch.device) -> "Era5HrrrNorm":
        for attr in ("era5_mu", "era5_sigma", "hrrr_mu", "hrrr_sigma",
                     "diag_mu", "diag_sigma", "cos_lat_weights"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self
