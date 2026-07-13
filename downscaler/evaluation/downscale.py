"""Single-shot ERA5 -> HRRR downscaling inference.

This REPLACES the HRRR emulator's autoregressive rollout (evaluation/evaluator.py
in the source project). Downscaling is not a rollout: given the coarse ERA5
field at time T, you draw ONE conditional sample of the HRRR field at T. Draw
several with different seeds to get an ensemble / uncertainty.

    ds = Downscaler(model, norm, statics, n_prognostic=4, n_diagnostic=1)
    x_cond = ds.build_conditioning(era5_on_hrrr_grid)   # physical ERA5, (B, n_era5, H, W)
    prog, diag = ds.sample(x_cond, n_steps=18)          # physical HRRR fields
"""

import torch

from evaluation.sampler import edm_heun_sample


class Downscaler:
    """Conditional-diffusion downscaler wrapping an EDMPreconditioning model.

    Args:
        model: EDMPreconditioning module (callable as model(x_noisy, sigma, x_cond)).
        norm: Era5HrrrNorm (already .to(device)).
        statics: (n_static, H, W) normalized static channels (orog, lsm).
        n_prognostic, n_diagnostic: HRRR output channel split.
    """

    def __init__(self, model, norm, statics: torch.Tensor,
                 n_prognostic: int, n_diagnostic: int):
        self.model = model
        self.norm = norm
        self.statics = statics
        self.n_prognostic = n_prognostic
        self.n_diagnostic = n_diagnostic
        self.device = next(model.parameters()).device

    def build_conditioning(self, era5_physical: torch.Tensor) -> torch.Tensor:
        """Normalize the (already HRRR-gridded) coarse ERA5 field and append statics.

        Args:
            era5_physical: (B, n_era5, H, W) ERA5 drivers regridded onto the HRRR
                grid, in physical units (see Era5HrrrDataset.regrid_era5).
        Returns:
            x_cond: (B, n_era5 + n_static, H, W).
        """
        B = era5_physical.shape[0]
        era5_norm = self.norm.normalize_era5(era5_physical.to(self.device))
        statics_b = self.statics.unsqueeze(0).expand(B, -1, -1, -1).to(self.device)
        return torch.cat([era5_norm, statics_b], dim=1)

    @torch.no_grad()
    def sample(self, x_cond: torch.Tensor, n_steps: int = 18, s_churn: float = 0.0):
        """Draw one HRRR sample conditioned on x_cond. Returns physical (prog, diag)."""
        n_target = self.n_prognostic + self.n_diagnostic
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            sample = edm_heun_sample(
                self.model, x_cond, n_target_ch=n_target, n_steps=n_steps, s_churn=s_churn
            )
        sample = sample.float()
        prog_norm = sample[:, : self.n_prognostic]
        diag_norm = sample[:, self.n_prognostic :]
        prog = self.norm.denormalize_hrrr(prog_norm)
        diag = self.norm.denormalize_diagnostic(diag_norm)
        return prog, diag

    @torch.no_grad()
    def ensemble(self, x_cond: torch.Tensor, n_members: int = 8, n_steps: int = 18):
        """Draw n_members samples (stochastic churn) -> stacked (M, B, C, H, W) tensors."""
        progs, diags = [], []
        for _ in range(n_members):
            p, d = self.sample(x_cond, n_steps=n_steps, s_churn=1.0)
            progs.append(p)
            diags.append(d)
        return torch.stack(progs), torch.stack(diags)
