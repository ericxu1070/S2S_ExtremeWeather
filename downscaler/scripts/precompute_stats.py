"""Compute normalization stats.npz for the ERA5 -> HRRR downscaler.

Streams a sample of training timestamps, accumulating per-channel mean/std via
Welford for: ERA5 conditioning (regridded), HRRR prognostic state, HRRR
diagnostics. Also records orog min/max and HRRR-grid cos(lat) area weights.

Trick: it builds an Era5HrrrDataset with IDENTITY normalization, so __getitem__
returns physical (un-normalized) fields through the exact same read+regrid path
training will use — no separate reader to drift out of sync.

Run once the 1-degree ERA5 has landed and configs/data/era5_hrrr.yaml is filled:
    python scripts/precompute_stats.py data.use_dummy=false stats.n_samples=500
Writes to cfg.data.stats_path.
"""

import logging
import os
import sys

import hydra
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

log = logging.getLogger(__name__)


class _Welford:
    """Streaming per-channel mean/variance over (C, H, W) samples, every pixel an observation.

    Chan's parallel merge: reduce each incoming sample to (count, mean, M2) with vectorized
    numpy, then combine that block into the running aggregate in O(C). The obvious
    element-at-a-time Welford loop is exact too, but it costs ~4 us per pixel in Python and
    an HRRR field has 1.9 M of them -- roughly 1.3 hours for a 200-sample run, versus a few
    seconds here. Accumulate in float64: summing 1.9 M squared residuals in float32 loses
    precision badly.
    """

    def __init__(self, n_ch: int):
        self.n = 0
        self.mean = np.zeros(n_ch, dtype=np.float64)
        self.m2 = np.zeros(n_ch, dtype=np.float64)

    def update(self, field: np.ndarray) -> None:
        x = np.asarray(field, dtype=np.float64).reshape(field.shape[0], -1)  # (C, N)
        n_b = x.shape[1]
        if n_b == 0:
            return
        mean_b = x.mean(axis=1)
        m2_b = ((x - mean_b[:, None]) ** 2).sum(axis=1)

        if self.n == 0:
            self.n, self.mean, self.m2 = n_b, mean_b, m2_b
            return

        n_a = self.n
        total = n_a + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / total)
        self.m2 = self.m2 + m2_b + delta**2 * (n_a * n_b / total)
        self.n = total

    def finalize(self):
        var = self.m2 / max(self.n - 1, 1)
        return self.mean.astype(np.float32), np.sqrt(var).astype(np.float32)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    from data.dummy import DummyNormalizationStats
    from data.era5_hrrr_dataset import Era5HrrrDataset
    from data.dataloader import load_timestamps

    n_samples = int(cfg.get("stats", {}).get("n_samples", 200))

    identity = DummyNormalizationStats(
        n_era5=cfg.data.n_era5_cond,
        n_prognostic=cfg.data.n_prognostic,
        n_diagnostic=cfg.data.n_diagnostic,
    )
    ts = load_timestamps(cfg.data.index_path, int(cfg.data.train_start[:4]), int(cfg.data.train_end[:4]))
    # Even stride so the sample spans all seasons/years.
    stride = max(1, len(ts) // n_samples)
    ts = ts[::stride][:n_samples]
    log.info(f"Accumulating stats over {len(ts)} timestamps")

    ds = Era5HrrrDataset(
        ts,
        era5_dir=cfg.data.era5_dir, hrrr_dir=cfg.data.hrrr_dir,
        grid_meta_path=cfg.data.grid_meta_path, norm=identity,
        era5_variables=list(cfg.data.era5_variables),
        hrrr_prognostic_variables=list(cfg.data.hrrr_prognostic_variables),
        hrrr_diagnostic_variables=list(cfg.data.hrrr_diagnostic_variables),
        era5_path_template=cfg.data.era5_path_template,
        hrrr_path_template=cfg.data.hrrr_path_template,
        era5_lat_name=cfg.data.era5_lat_name, era5_lon_name=cfg.data.era5_lon_name,
        era5_level_name=cfg.data.era5_level_name,
        orog_var=cfg.data.orog_var, lsm_var=cfg.data.lsm_var,
        match_era5_lon_to_360=cfg.data.match_era5_lon_to_360,
        # Stats MUST be accumulated on the rotated winds, since that is what the model sees.
        rotate_hrrr_winds=cfg.data.rotate_hrrr_winds,
        hrrr_wind_pairs=[list(p) for p in cfg.data.hrrr_wind_pairs],
    )

    w_era5 = _Welford(cfg.data.n_era5_cond)
    w_hrrr = _Welford(cfg.data.n_prognostic)
    w_diag = _Welford(cfg.data.n_diagnostic)
    for i in range(len(ds)):
        item = ds[i]
        # statics are identity-normalized here, so x_cond[:n_era5] is physical ERA5
        w_era5.update(item["x_cond"][: cfg.data.n_era5_cond].numpy())
        w_hrrr.update(item["target_prognostic"].numpy())
        w_diag.update(item["target_diagnostic"].numpy())
        if (i + 1) % 25 == 0:
            log.info(f"  {i + 1}/{len(ds)}")

    era5_mu, era5_sigma = w_era5.finalize()
    hrrr_mu, hrrr_sigma = w_hrrr.finalize()
    diag_mu, diag_sigma = w_diag.finalize()

    # Statics + area weights from the raw first-file statics / grid reference.
    orog = ds.statics[0].numpy()  # identity-normalized == physical orog
    cos_lat = np.cos(np.deg2rad(ds.hrrr_lat)).astype(np.float32)

    out_path = cfg.data.stats_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        era5_mu=era5_mu, era5_sigma=era5_sigma,
        hrrr_mu=hrrr_mu, hrrr_sigma=hrrr_sigma,
        diag_mu=diag_mu, diag_sigma=diag_sigma,
        orog_min=np.float64(orog.min()), orog_max=np.float64(orog.max()),
        cos_lat_weights=cos_lat,
        era5_channel_names=np.array([str(v) for v in cfg.data.era5_variables]),
        hrrr_channel_names=np.array([str(v) for v in cfg.data.hrrr_prognostic_variables]),
    )
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
