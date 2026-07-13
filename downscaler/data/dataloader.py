"""DataLoader utilities for the ERA5 -> HRRR downscaler.

Downscaling is a *same-time* mapping ERA5(T) -> HRRR(T): each training sample is
a single timestamp, not a (T, T+6h) pair. `load_timestamps` reads a newline-
separated index of ISO timestamps for which BOTH an ERA5 field and a HRRR
analysis exist, filtered by year.
"""

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


def load_timestamps(index_path: str, start_year: int, end_year: int) -> list[str]:
    """Load and filter an index of valid timestamps by year range.

    Args:
        index_path: newline-separated file of ISO timestamps (e.g. 2020-01-01T00:00:00).
        start_year: first year (inclusive).
        end_year: last year (inclusive).

    Returns:
        List of ISO timestamp strings.
    """
    timestamps = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # tolerate a 2-column (era5<TAB>hrrr) index by taking the first field
            ts = line.split("\t")[0]
            year = int(ts[:4])
            if start_year <= year <= end_year:
                timestamps.append(ts)
    return timestamps


def build_dataloader(dataset, cfg, is_train: bool = True) -> DataLoader:
    """Build a DataLoader with optional DDP sampler."""
    use_ddp = dist.is_initialized()
    sampler = DistributedSampler(dataset, shuffle=is_train) if use_ddp else None
    return DataLoader(
        dataset,
        batch_size=cfg.training.per_gpu_batch,
        sampler=sampler,
        shuffle=(is_train and sampler is None),
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
    )
