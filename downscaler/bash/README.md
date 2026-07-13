# Running the downscaler training (handoff runbook)

This folder is for whoever operates the GPUs. You do **not** need to know the science or
the codebase — six numbered scripts, run in order, and the job is explicitly designed to
be **stopped at any moment and resumed later**, so it can soak up idle time between your
own jobs without ever holding a node hostage.

What it trains: a diffusion model that sharpens coarse (1°) ERA5 weather fields into
3 km HRRR-resolution fields over CONUS. Curious? `../README.md` is the project page,
`../docs/model_reference.html` the model spec (open in a browser). Not required reading.

## TL;DR

```bash
cd /home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/downscaler

bash bash/00_preflight.sh              # read-only checks           (~1 min)
bash bash/01_smoke.sh                  # harness self-test, no GPU  (~3 min)
bash bash/02_build_index_and_stats.sh  # one-time data prep, CPU    (~35 min — see below)
bash bash/03_train.sh                  # start / resume training

bash bash/05_status.sh                 # is it running? what step? healthy?
bash bash/04_stop.sh                   # need the GPUs back? this. (~seconds)
bash bash/03_train.sh                  # ...and later, same command resumes
```

00–02 are idempotent and safe on a login node. Only 03 uses GPUs.

## The two commands that matter day-to-day

**Stop (give the GPUs back):** `bash bash/04_stop.sh` — drops a sentinel file; the
trainer checkpoints everything (weights, optimizer, EMA) at the next step boundary and
exits cleanly. No deadline, nothing is killed, and at most **100 optimizer steps** of
work are redone on resume. If you're in a genuine hurry, plain `scancel` / `kill` is
also survivable — the trainer catches SIGTERM and checkpoints, and every save is atomic,
so even a hard kill mid-write cannot corrupt the resume point.

**Start/resume:** `bash bash/03_train.sh` — the same command both starts and resumes; it
auto-loads `checkpoints/latest.pt` if one exists. On a machine with Slurm it submits
`slurm/train.sbatch` (1 node, 8 GPUs, 24 h walltime — resubmit to continue); on a plain
GPU box it launches `torchrun` on all visible GPUs under `nohup` (limit with `GPUS=4`).
Hydra overrides pass through: `bash bash/03_train.sh training.per_gpu_batch=2`.

## How to tell it's healthy

`bash bash/05_status.sh`, or watch the newest log for heartbeat lines:

```
[hb] step=1230 loss=0.0841 peak_gb=62.3
```

Step number climbing and loss (noisily) falling = healthy. The loss starts near ~1.0.
No heartbeat for >10 min = look at the tail of the log. On a fresh resume you should see
`Auto-resumed from .../latest.pt` near the top.

## Footprint & expectations

| | |
|---|---|
| GPU | 1 node × 8 GPUs (H100-class); per-GPU batch 1, bf16, ~60–70 GB VRAM/GPU |
| Disk | ~3 GB of checkpoints (3 rolling snapshots; `latest.pt` is a hardlink), under `checkpoints/` |
| Data (read-only) | ERA5 + HRRR netCDF trees on NFS — the scripts never write into them |
| Wall time | multi-day for the full 200 epochs — that's *why* it's interruptible; run it in whatever gaps you have |
| Env | conda env `moe` (PyTorch). Nothing to install; preflight verifies it |
| Network | none needed. Metrics log to a JSONL file in the repo (`metrics/train_metrics.jsonl`; plot with `python scripts/plot_metrics.py`). W&B is opt-in (`logging=wandb`, needs a login) and a failure to reach it never kills the run |

## Troubleshooting

- **`02` looks hung** — it isn't: the stats pass streams 500 samples through the real
  read+regrid path at ~4 s each (~35 min total; progress lines count samples). Do NOT
  start a second instance to "help" — the final stats write is not atomic and two writers
  can corrupt it.
- **Stopping a run that used a non-default checkpoint dir** — tell the stop script where:
  `CKPT_DIR=/path/to/dir bash bash/04_stop.sh`.
- **`03_train.sh` refuses to start** — it either found a run already active (stop it or
  let it be) or the one-time prerequisites are missing (`bash bash/02_build_index_and_stats.sh`).
- **A run dies instantly at startup** — read the first 30 lines of its log. A stale `STOP`
  sentinel is *not* the cause (the trainer clears those itself); a channel-count config
  error aborts with an explicit message before touching the GPUs.
- **DataLoader hangs/crashes on netCDF reads** — should not happen (the scripts set
  `HDF5_USE_FILE_LOCKING=FALSE`, the known NFS deadlock), but if you see it, that env var
  is the first thing to check.
- **Restart from scratch instead of resuming** — point at a fresh dir:
  `bash bash/03_train.sh ckpt_dir=/path/to/new_dir` (or move `checkpoints/` aside).
  A run *always* resumes from an existing `latest.pt` in its checkpoint dir.
- **Something smells off in the harness itself** — `bash bash/01_smoke.sh` re-verifies the
  whole checkpoint/stop/resume cycle in ~3 min without touching a GPU.

Slurm-specific detail (signals, walltime handling, `scancel` semantics) lives in
`../slurm/README.md`. The scripts here are a superset: they call into the same machinery
when Slurm is present and fall back to plain `torchrun` when it isn't.
