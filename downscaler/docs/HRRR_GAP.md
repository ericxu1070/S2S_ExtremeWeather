# HRRR 2019-2021 deleted, 2026-09-14

`/home/ubuntu/Vayuh/data/moein/hrrr_work/hrrr_nc_v3_rebuilt/{2019,2020,2021}` — 4,379
files, 383 GB — were **deleted** to free space for the `acal` calibration campaign.
Eric's decision, made with the cost below understood; the years are to be re-downloaded
before the next downscaler training run.

File-by-file manifest: `hrrr_deleted_2026-09-14.txt` (4,379 names).

## What this breaks right now

**`valid_timestamps.txt` is STALE.** It still lists 16,019 timestamps, 4,379 of which no
longer have an HRRR file. `Era5HrrrDataset` will raise on the first missing one. Worse, the
`norm_stats/stats.npz` that training loads was computed over the old index — the exact trap
in the root `CLAUDE.md`: a stale `stats.npz` crashes training **at the first batch, after
the Slurm allocation is granted**.

Do not start a training run until the years are restored or the index is rebuilt.

## To restore

```bash
# 1. re-download (Herbie, 3 calls per timestamp; see moein/hrrr_work/README.md)
python /home/ubuntu/Vayuh/data/moein/hrrr_work/download_hrrr_v3.py   # needs a year filter
# 2. rebuild index AND norm stats together - never one without the other
cd downscaler && FORCE_REBUILD=1 bash bash/02_build_index_and_stats.sh   # ~35 min
# 3. confirm
wc -l valid_timestamps.txt      # expect 16019 again
```

## To train WITHOUT restoring

Rebuild the index over what is left (11,640 timestamps) and accept the gap:

```bash
cd downscaler && FORCE_REBUILD=1 bash bash/02_build_index_and_stats.sh
```

The split in `configs/data/era5_hrrr.yaml` is train 2015-2022 / val 2023 / test 2024-25, so
the gap costs **three of eight training years** and leaves validation and test intact. If
you adopt `train_end: 2018-12-31` — which the root `CLAUDE.md` recommends so all 12 xres
events become out-of-sample — the gap costs **nothing at all**, which is why 2019-2021 were
chosen over the older years.

## Not affected

- ERA5 (`eric/era5_1deg/`, 2015-2025) — untouched, still complete.
- xres HRRR overlays — they read `runs/observations/*_hrrr_verif_*.nc`, not this tree.
- `hrrr_work/shards_v1/` (2.2 TB) and `hrrr_nc_v3/` (40 GB) — untouched.
