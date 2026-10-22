# Walker benchmark: FCN3 vs GenCast

Run `small1` on `computeinstance-e00fjhc4cz8n2e69gz`: 1x NVIDIA H100 80GB HBM3 (driver 550.163.01), Intel(R) Xeon(R) Platinum 8468, 206.1 GB RAM. 9 result records.

## Answer

- **One AI+RES case at N=32:** 25.7 H100-h with GenCast walkers vs 10.2 with fcn3_e2s (B=1). The FCN3 score phase is 8.8 H100-h of both totals.

## FCN3 walker checks

| check | verdict | key numbers |
|---|---|---|
| native_matches_e2s | PASS | t2m_rms_k=[0.0, 0.0, 0.0, 0.0]; max_abs_diff=[0.0, 0.0, 0.0, 0.0]; bitwise=[True, True, True, True]; note=WARN = same model to rounding (different kernel order), not bitwise |
| determinism | PASS | max_abs_diff=0.0; bitwise=True |
| restart_exact_nc | PASS | t2m_rms_k=0.0; max_abs_diff=0.0; state_bytes=196469310 |
| restart_exact_npz | PASS | t2m_rms_k=0.0; max_abs_diff=0.0; state_bytes=307374244 |
| noise_renewal | INFO | t2m_rms_k_after_segment=0.6399236073965024; note=restart without the persisted noise state vs with it, same seeds; compare with clone_divergence at the same lead |
| clones | ERROR | error=OutOfMemoryError: CUDA out of memory. Tried to allocate 40.73 GiB. GPU 0 has a total capacity of 79.10 GiB of which 6.57 GiB is free. Including non-PyTorch memory, this process has 72.52 GiB memory in use. Of the allocated memory 54.32 GiB is allocated by PyTorch, and 17.53 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_ |
| batch_invariance | SKIP | note=B=2 does not fit on this card |
| state_integrity | PASS | state_bytes=196194542 |

## Layer 1: raw step speed (one worker per model, model resident)

GenCast steps 12 h and FCN3 6 h, so compare the per-day columns. Compute is the model call alone; roll adds what the walker loop does per step (host copy, CONUS crop).

| config | precision | s/step compute | s/step roll | s/sim-day compute | s/sim-day roll | speedup (compute) | peak GPU GB | startup s |
|---|---|---|---|---|---|---|---|---|
| GenCast 0.25 deg (production walker) | bf16 | 27.71 | 28.44 | 55.41 | 56.88 | 1.00 | 22.70 | 390.92 |
| GenCast 1.0 deg | fp32 | 3.51 | 3.56 | 7.01 | 7.12 | 7.90 | 3.00 | 50.27 |
| FCN3 via earth2studio | bf16-autocast | 0.98 | 0.99 | 3.94 | 3.94 | 14.06 | 52.60 | 5.23 |
| FCN3 native batched | bf16-autocast | 0.99 | 0.99 | 3.94 | 3.95 | 14.06 | 53.00 | 5.19 |

### FCN3 batch sweep

| config | worker | B | s/step | walker-steps/s | s per walker-day | gain vs B=1 | peak GB | OOM |
|---|---|---|---|---|---|---|---|---|
| fcn3_e2s | 0 | 1 | 0.98 | 1.02 | 3.94 | 1.00 | 52.64 |  |
| fcn3_e2s | 0 | 2 | 1.96 | 1.02 | 3.91 | 1.01 | 52.95 |  |
| fcn3_native | 0 | 1 | 0.99 | 1.01 | 3.94 | 1.00 | 52.95 |  |
| fcn3_native | 0 | 2 | - | - | - | - | 58.01 | yes |
| fcn3_native | 1 | 1 | 0.99 | 1.01 | 3.94 | 1.00 | 52.95 |  |
| fcn3_native | 1 | 2 | - | - | - | - | 58.01 | yes |

## Layer 2: walker segments (3 d, checkpoint I/O included)

`solo` = one worker per model (other configs on the other GPUs); `node8` = 8 identical workers on one node, the production layout. Per-walker phase times are seconds per 3-day segment.

| config | phase | B | io | workers | s/walker-day | range | speedup | read | roll | write state | write diag | state MB | noise kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gencast_0p25 | solo | 1 | nc-zlib4 | 1 | 62.55 | 62.55..62.55 | 1.00 | 2.27 | 170.63 | 13.76 | 1.00 | 472.70 | n/a |
| gencast_1p0 | solo | 1 | nc-zlib4 | 1 | 7.53 | 7.53..7.53 | 8.31 | 0.15 | 21.35 | 0.99 | 0.09 | 31.20 | n/a |
| fcn3_e2s | solo | 1 | nc-zlib4 | 1 | 6.91 | 6.91..6.91 | 9.06 | 0.95 | 11.83 | 5.99 | 1.77 | 192.10 | no |
| fcn3_native | solo | 1 | nc-zlib4 | 1 | 7.05 | 7.05..7.05 | 8.87 | 1.10 | 11.86 | 6.21 | 1.80 | 196.20 | yes |
| fcn3_native | solo | 1 | npz-raw | 1 | 4.79 | 4.79..4.79 | 13.06 | 0.25 | 11.83 | 0.32 | 1.78 | 307.40 | yes |

## Layer 3: one AI+RES case

Horizon 21 d, 6 legs, resampling at 3, 6, 9, 12, 15 d with C_k = 0, 1, 1.4, 1.8, 2; score M = 6 FCN3 members at 0.985 s/member-step (measured (fcn3_e2s, B=1)). `relaunch` = every leg restarts its workers (today's runner); `persistent` = workers live across legs (the fix proposed in aires/HANDOFF.md).

| N | walker | B | rate from | walk H100-h | score H100-h | startup (relaunch) | total (relaunch) | total (persistent) | node wall h |
|---|---|---|---|---|---|---|---|---|---|
| 32 | gencast_0p25 | 1 | solo | 11.68 | 8.83 | 5.21 | 25.71 | 21.37 | 3.21 |
| 32 | gencast_1p0 | 1 | solo | 1.40 | 8.83 | 0.67 | 10.90 | 10.34 | 1.36 |
| 32 | fcn3_e2s | 1 | solo | 1.29 | 8.83 | 0.07 | 10.18 | 10.13 | 1.27 |
| 32 | fcn3_native | 1 | solo | 1.32 | 8.83 | 0.07 | 10.21 | 10.15 | 1.28 |
| 32 | gencast_0p25 @ a3mega rates | 1 | reference | 19.41 | 12.54 | 12.00 | 43.96 | 33.96 | 5.49 |
| 64 | gencast_0p25 | 1 | solo | 23.35 | 17.65 | 5.21 | 46.22 | 41.87 | 5.78 |
| 64 | gencast_1p0 | 1 | solo | 2.81 | 17.65 | 0.67 | 21.13 | 20.57 | 2.64 |
| 64 | fcn3_e2s | 1 | solo | 2.58 | 17.65 | 0.07 | 20.30 | 20.24 | 2.54 |
| 64 | fcn3_native | 1 | solo | 2.63 | 17.65 | 0.07 | 20.35 | 20.30 | 2.54 |
| 64 | gencast_0p25 @ a3mega rates | 1 | reference | 38.83 | 25.09 | 12.00 | 75.91 | 65.91 | 9.49 |

## What this does not measure

- Whether FCN3 walkers reach the tail or keep the importance weights honest. That needs Gate 3 and a production case with FCN3 walking.
- FCN3's clone separation versus GenCast's. `clone_divergence` shows that clones separate, not whether they separate as fast as the resampling needs.
- Shared-filesystem I/O. The VM writes states to its own disk, while a3mega wrote them to NFS. The a3mega reference row shows what NFS contention cost there.
- torch.compile, CUDA graphs and multi-GPU spatial parallelism for FCN3. All of them would only make FCN3 faster.
