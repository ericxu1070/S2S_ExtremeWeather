"""Dummy dataset for the ERA5 -> HRRR downscaler (profiling / smoke tests).

Returns tensors with the same shapes, dtypes, and keys as Era5HrrrDataset but
filled with random normalized data (std=1). Lets you exercise the full training
loop end-to-end with NO real data — useful right now while the 1-degree ERA5
transfer from Derecho is pending.

    python train.py data.use_dummy=true training.max_steps=5
"""

import torch
from torch.utils.data import Dataset


class DummyNormalizationStats:
    """Identity normalization (mu=0, sigma=1). Mirrors Era5HrrrNorm's interface."""

    def __init__(self, n_era5: int = 42, n_prognostic: int = 41, n_diagnostic: int = 1):
        self.n_era5 = n_era5
        self.n_prognostic = n_prognostic
        self.n_diagnostic = n_diagnostic
        self.era5_mu = torch.zeros(n_era5)
        self.era5_sigma = torch.ones(n_era5)
        self.hrrr_mu = torch.zeros(n_prognostic)
        self.hrrr_sigma = torch.ones(n_prognostic)
        self.diag_mu = torch.zeros(n_diagnostic)
        self.diag_sigma = torch.ones(n_diagnostic)
        self.orog_min, self.orog_max = 0.0, 1.0

    def normalize_era5(self, x): return x
    def denormalize_era5(self, x): return x
    def normalize_hrrr(self, y): return y
    def denormalize_hrrr(self, y): return y
    def normalize_diagnostic(self, d): return d
    def denormalize_diagnostic(self, d): return d
    def normalize_orog(self, o): return o
    def to(self, device): return self


class DummyDataset(Dataset):
    """Random-tensor dataset matching Era5HrrrDataset's interface.

    Args:
        n_era5_cond: ERA5 driver channels (already regridded onto the HRRR grid).
        n_prognostic: HRRR prognostic output channels (full-field state target).
        n_diagnostic: HRRR diagnostic output channels (e.g. log precip).
        n_static: static conditioning channels (orog, lsm).
        height, width: HRRR grid dims.
        length: number of samples.
    """

    def __init__(
        self,
        n_era5_cond: int = 42,
        n_prognostic: int = 41,
        n_diagnostic: int = 1,
        n_static: int = 2,
        height: int = 1059,
        width: int = 1799,
        length: int = 100,
    ):
        self.n_era5_cond = n_era5_cond
        self.n_prognostic = n_prognostic
        self.n_diagnostic = n_diagnostic
        self.n_static = n_static
        self.height = height
        self.width = width
        self.length = length

        # Cached static channels (normalized orog + lsm), constant across samples.
        self.statics = torch.randn(n_static, height, width)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        h, w = self.height, self.width

        # Conditioning: upsampled-ERA5 drivers + statics.
        x_cond = torch.randn(self.n_era5_cond + self.n_static, h, w)
        x_cond[-self.n_static:] = self.statics

        # Target: normalized HRRR full-field state + diagnostic.
        target_prognostic = torch.randn(self.n_prognostic, h, w)
        target_diagnostic = torch.randn(self.n_diagnostic, h, w)

        return {
            "x_cond": x_cond,
            "target_prognostic": target_prognostic,
            "target_diagnostic": target_diagnostic,
            "timestamp": idx,
        }
