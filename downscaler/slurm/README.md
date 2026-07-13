# Interruptible training on a3mega

Runbook for training the downscaler in the gaps between the `serve-*` inference jobs.
The a3mega nodes are whole-node (`OverSubscribe=NO`), so training **cannot** share a node
with a serving job — it queues for one and gives it back on demand.

| | |
|---|---|
| **Smoke test** | `sbatch slurm/smoke.sbatch` — CPU only, no GPU burned |
| **Start** | `sbatch slurm/train.sbatch` |
| **Watch** | `tail -f logs/slurm-downscaler-<jobid>.out` |
| **Stop** | `bash slurm/stop.sh` |
| **Resume** | `sbatch slurm/train.sbatch` — same command, auto-resumes |

Nothing else is needed. Stop and start as often as you like; each stop costs at most
`training.save_every_n_steps` (default 100) optimizer steps of redone work.

## Smoke test (`debug` partition, ~3 min)

```bash
sbatch slurm/smoke.sbatch
tail -f logs/slurm-downscaler-smoke-<jobid>.out
```

Runs the whole start/stop/resume cycle on a 1-vCPU `debug` node against a tiny dummy model,
so it costs no GPU time and cannot disturb a real run (it checkpoints to
`checkpoints/_smoke/`, never the real `latest.pt`). It asserts 13 things, the load-bearing
ones being: a fresh run checkpoints mid-epoch; `latest.pt` is a hardlink rather than a copy;
a second run resumes at the exact step the first stopped at; and the `STOP` sentinel makes
the trainer save, consume the sentinel and **exit 0** — a clean Slurm completion, not a
failure.

Run it after touching `training/trainer.py` or anything under `slurm/`. It passing means the
harness is sound; it says nothing about the science, since it trains on random tensors.

**Timing.** ~50 s of actual work. The `debug` nodes are cloud nodes that power down when
idle, so a cold submit waits ~100 s in `CF` (CONFIGURING) while GCP boots the VM — call it
**3 minutes end to end from cold, under a minute if a debug node is already up.**

## Starting

```bash
cd /home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather/downscaler
sbatch slurm/train.sbatch
```

If every node is busy the job sits in `PENDING` and Slurm starts it the moment one frees —
submit it anyway. Check with `squeue -u $USER`, and see when it would land with
`sbatch --test-only slurm/train.sbatch`.

Hydra overrides pass straight through:

```bash
sbatch slurm/train.sbatch training.per_gpu_batch=2 model.C_base=128
sbatch --nodes=2 slurm/train.sbatch          # scale out; the LR cosine rebases on resume
```

## Stopping

`bash slurm/stop.sh` drops a `STOP` sentinel in the checkpoint dir. The trainer polls for
it at micro-step boundaries, writes a full checkpoint (model + optimizer + scheduler +
EMA), deletes the sentinel, and exits 0 — the job completes normally and the node returns
to the queue. **There is no deadline on this path**, which is why it is the default.

Two alternatives, in decreasing order of safety:

```bash
scancel -s USR1 <jobid>   # same graceful path, no kill timer. Fine.
scancel <jobid>           # SIGTERM, then SIGKILL 30 s later (KillWait).
```

Plain `scancel` is handled — the trainer catches SIGTERM and checkpoints — but it is
racing a 30-second timer. A save is normally a few seconds, so it will almost always win;
`stop.sh` and `scancel -s USR1` remove the race entirely.

Walltime (`--time=24:00:00`) is handled the same way: `--signal=USR1@300` gives the
trainer a 5-minute warning to checkpoint and exit before Slurm kills it.

## Resuming

Just resubmit. `Trainer.train()` auto-resumes from `<ckpt_dir>/latest.pt` — you should see

```
Loaded checkpoint from during epoch 12 (step 1100) — resuming at epoch 12
Auto-resumed from .../checkpoints/latest.pt
```

Weights, optimizer moments, EMA shadow and `global_step` are restored exactly. An epoch
that was cut short is re-run from the top with a fresh shuffle: no training progress is
lost, only the within-epoch sample coverage is resampled, which is immaterial over
hundreds of epochs.

To start over instead of resuming, point at a fresh dir: `sbatch slurm/train.sbatch
ckpt_dir=/path/to/new_dir` (or delete the old one). A run will *always* pick up an
existing `latest.pt` in its `ckpt_dir`.

## What's in the checkpoint dir

```
latest.pt                    # resume pointer — a hardlink, not a copy
checkpoint_step00001100.pt   # archival snapshots, newest keep_last_n_checkpoints (3) kept
STOP                         # only exists between `stop.sh` and the trainer noticing it
```

Saves are atomic (temp file + `os.replace`), so a kill mid-write cannot corrupt an
existing checkpoint. Pruning an old snapshot never invalidates `latest.pt` — the hardlink
keeps the data alive.

Knobs live in `configs/training/default.yaml` under *Checkpointing / interruptible
training*.

## Note: this still runs on dummy data

`configs/data/era5_hrrr.yaml` defaults to `use_dummy: true`, so the job as shipped trains
on random tensors — it exercises the full pipeline but means nothing scientifically. The
real-data prerequisites (rebuild the HRRR netCDF archive from `shards_v1`, fix the ERA5
variable/coord names and path template, build the timestamp index, precompute norm stats)
are listed in `../README.md`. Once those are done, add `data.use_dummy=false`.
