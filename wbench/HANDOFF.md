# wbench - live state

Walker benchmark: how fast FCN3 is as the AI+RES WALKER compared with GenCast, to be run on
a Nebius 8xH100 VM. The design is in `wbench/__init__.py`. The VM runbook is
`wbench/vm/README.md`. This file is the state of the work. **Trust `ls` and the commands
below over any prose here.**

Started 2026-10-15 on Derecho (`derecho1`). Nothing is committed: `wbench/` is untracked
on `main` (commit only when asked). The pre-existing user edit to `acal/HANDOFF.md` was
not touched.

## Where things stand

| piece | state |
|---|---|
| design (agreed with the user) | **done.** 3 layers (raw step speed, 3-day walker segment with checkpoint I/O at 1 and 8 workers per node, H100-h per AI+RES case at N=32/64), 4 configs (GenCast 0.25 bf16, GenCast 1.0 fp32, FCN3 via earth2studio, FCN3 native batched), plus FCN3 walker checks. Bare 8xH100 VM, bundle relayed through the laptop, micromamba setup script, ~1.5 h budget |
| code (`wbench/*.py`) | **written, CPU-tested, never run on a GPU** |
| VM scripts (`wbench/vm/`) | **written.** `run_all.sh` dry-run tested on the login node; `setup_vm.sh` and `make_bundle.sh` **never executed** |
| GPU-path review | **interrupted mid-way.** Was re-reading `fcn3_walker.py` at `NativeFCN3.roll` (line ~214); `fcn3_bench.py` and `checks.py` not yet re-read |
| bundle | **not built** |
| Nebius run | **`small1` ran 2026-10-21** (1xH100, eu-north1, speed only). Results in R2 `vayuh/eric/wbench/small1/` (35 files). The rows above predate it; the follow-ups live in the vault note "Nebius experiments - FCN3 AI+RES" |

## Nebius and R2 operations (learned 2026-10-23)

Read this before booking or cleaning up a Nebius VM.

- **Tenant:** `olive-wolf-tenant-2zm`, CLI profile `eric`. One default project per region; **H100 exists only in `default-project-eu-north1`** (`project-e00g90k7pr0032bsy9gfff`). Pass that `--parent-id`; the profile default is us-central1, which has no H100.
- **No CPU-only VMs:** quota `compute.instance.non-gpu.vcpu` is 0 tenant-wide. The cheapest box is `gpu-l40s-a` / `1gpu-8vcpu-32gb` (~$0.75/h spot, $1.55/h on-demand, 2026-10-23).
- **Preemptible flags:** `--preemptible-on-preemption stop` plus a pricing model (`--follows-spot-price` or `--spot-pricing-policy-id`) plus `--recovery-policy fail`. Each one missing is a separate API error.
- **Spot cap goes away 2026-11-02.** After that, "follow spot price" has no ceiling. Use the max-price policy `h100-max-2` (`pricingpolicy-e00nf9i4ktm3hily5azuy`, $2/GPU-h, eu-north1): `--spot-pricing-policy-id pricingpolicy-e00nf9i4ktm3hily5azuy`. If spot rises above $2 the VM is stopped, so checkpoint.
- **Disks can't attach read-only** (API rejects it). To read a disk safely, attach read-write and `mount -o ro,noload`.
- **R2:** bucket `vayuh` is in Cloudflare account "Arunabh@vayuh.ai's Account" (`1b35b28518577975c68c10385f71a1f6`), not Eric's own (R2 is disabled there). The laptop's `~/.aws` profile `r2` holds an Object Read & Write token for `vayuh`; copy it to the VM with the README's `scp ~/.aws` step. The `cf` CLI exists only on Derecho; set `CLOUDFLARE_ACCOUNT_ID=1b35b28518577975c68c10385f71a1f6` to reach `vayuh`. `cf r2 objects put` caps objects at 300 MB, so bulk data goes through `aws --profile r2`.
- **End-of-run rule:** an idle 200 GiB disk costs ~$14/month. Before deleting the GPU VM, stream outputs from it straight to R2, then delete VM and data disk together (prefer a managed data disk so it dies with the VM). Use the split upload below, with `export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required` set.

- **Agents can't create Nebius VMs** (Claude Code auto mode blocks paid launches, and attaching a data disk read-write). Write the command and have Eric run it. Deleting a VM or disk Eric asked to delete went through.
- **R2 multipart gotcha:** one 35 GB streamed `aws s3 cp -` failed at `CompleteMultipartUpload` with `InvalidPart` after 24 min. Splitting the stream into 2 GiB pieces, each staged on local disk and uploaded with retries, worked first time. For anything over ~5 GB, use the split form:

  ```bash
  aws configure set profile.r2.s3.multipart_chunksize 64MB
  sudo tar -C <data dir> -cf - . | zstd -T0 -3 | tee >(sha256sum > up.sha256) \
    | split -b 2G -d -a 3 --filter='cat > $FILE && aws --profile r2 s3 cp $FILE s3://vayuh/eric/<path>.tar.zst.d/$FILE && rm $FILE' - part.
  ```

- **2026-10-23 cleanup, done:** the leftover `wbench-data` disk (200 GiB, 35 GB used: envs, micromamba, FCN3 caches, `wbench_bundle.tar`, `runs/`) is backed up to `s3://vayuh/eric/wbench/wbench-data.tar.zst.d/` (11 pieces, 21.0 GiB, stream sha256 `63d5f5294dd3bc601d8d13b60dc46b02f1cbe8db0632e7c13c175f598dadf78c`). Verified by streaming it back: sha256 matches and `tar -t` lists 160,946 entries, the same as `find` on the disk. The disk and the temporary VM are deleted; no instances or disks remain in any project. Still in Nebius: bucket `s3://wbench-bundle` (the 4.1 GB bundle). Restore:

  ```bash
  aws --profile r2 s3 cp --recursive s3://vayuh/eric/wbench/wbench-data.tar.zst.d/ parts/
  cat parts/part.* | zstd -d | tar -xf - -C <dest>
  ```

## Verified (2026-10-15, Derecho login node)

- `wbench/tests` in the Derecho `fcn3` env: **24 passed** (~5 min, mostly imports). This
  includes the per-walker noise tests against makani's real `DiffusionNoiseS2` on a tiny
  grid: init draw identical to earth2studio's, streams independent of batch, clones differ,
  AR(1) memory carried (corr ~phi).
- `wbench/tests` in `my-env` (GenCast): **19 passed, 5 skipped** (the torch tests skip
  there by design).
- `python -m wbench.gencast_bench --config {gencast_0p25,gencast_1p0} --check`: both build
  a 6-step segment batch from the PNW init. 176 + 504 + 24 = 704 channels, matching
  aires/HANDOFF.md. The 1.0 deg grid override works.
- `bash -n` on all four shell scripts: OK. No shellcheck on this box.
- `WB_DRY=1 WB_NGPU=8 bash wbench/vm/run_all.sh`: all phases, 50 steps scheduled onto 8
  slots with NUMA CPU slices. A second run with the same `WB_RUN_ID` skipped all 10
  finished solo steps. The dry tree was deleted afterwards.

Running the fcn3-env tests needs pytest, which that env lacks. It was staged outside the
env, in a session scratch dir that will be gone:

```bash
/glade/derecho/scratch/exu/conda-envs/fcn3/bin/python -m pip install -q --target /glade/derecho/scratch/exu/tmp/pytest_pkgs pytest
cd /glade/derecho/scratch/exu/S2S_ExtremeWeather
PYTHONPATH=/glade/derecho/scratch/exu/tmp/pytest_pkgs:. /glade/derecho/scratch/exu/conda-envs/fcn3/bin/python -m pytest -q wbench/tests -p no:cacheprovider
PYTHONPATH=. /glade/work/exu/conda-envs/my-env/bin/python -m pytest -q wbench/tests -p no:cacheprovider
```

## NOT verified - what could still be wrong on the GPU

Ordered by how much a failure would cost on the VM:

1. **`NativeFCN3.step` vs earth2studio.** It re-implements
   `makani ModelPackage.forward` (normalize, zenith cache, `net(xn, update_state=False)`,
   denormalize) inside earth2studio's autocast, with the AR(1) noise advanced per walker
   through makani's own `update()`. `checks.native_matches_e2s` is the test (bitwise at B=1
   over 4 steps). If it FAILs, the native speed numbers are not FCN3's. Inference-tensor
   rules matter here: `release()` must run before any earth2studio call after native use,
   or e2s's `noise.reset()` hits an in-place op on an inference tensor.
2. **The earth2studio walker path** (`fcn3_bench.E2SRoller`). This is
   `px.create_iterator(x[:, None, None], coords)` with a leading `ensemble` dim, assuming
   `batch_func` folds it into the batch the way `run.ensemble` does. Untested.
3. **The GenCast predictor timing wrapper** (`gencast_bench._timed`).
   `jax.block_until_ready` on an xarray Dataset relies on graphcast's pytree registration,
   with a fallback to `tree_leaves`. `compile_s` = first forward minus steady median.
4. **`setup_vm.sh`** has never run. Fragile points: the
   `nvidia/label/cuda-12.8.1` channel for `cuda-toolkit` (override with `WB_CUDA_LABEL`),
   earth2studio pulling a torch-harmonics wheel (step `th` force-reinstalls the source
   build afterwards and asserts the DISCO kernel), and makani installed from git before
   earth2studio.
5. **`report.py` on real records.** Only tested on synthetic records
   (`tests/test_report.py`). The projection itself is pinned: at a3mega rates N=64 gives
   38.8 walk + 25.1 score H100-h, against the pilot's ~36 + ~25.

## Decisions taken, worth knowing when reading results

- **JAX 0.6.2 on the VM** (Derecho `my-env`'s known-good set,
  `vm/requirements-gencast.txt`). a3mega production ran jax 0.10.2, so a GenCast Nebius
  vs a3mega difference may partly be the JAX version. The report records versions per
  result.
- **GenCast 1.0 deg runs fp32**, as the xres 1p0 member arrays ran (`pbs/xres_member_1p0.pbs`
  sets no `XRES_BF16`). GenCast 0.25 runs bf16, as production does. `gencast_bench`
  sets `XRES_SERIAL_INFER/XRES_BF16/XRES_INFER_GPUS` itself from the config.
- **The 1.0 deg walker state** is the xres stride-4 init (`runs/xres/1p0/week3/inputs/`),
  and `aconfig.STATE_NLAT/NLON` is rebound to 181x360 in that process only.
- **The FCN3 native walker persists the noise state** with the field (netCDF zlib4 like the
  GenCast walker, plus an `npz` raw variant for comparison). Seeds are
  `segment_seed(base, walker, step)` (blake2b), one `torch.Generator` per walker. The
  e2s walker cannot persist noise (re-drawn per iterator); its state files say
  `noise_persisted=0`.
- **Score cost in layer 3** uses the measured `fcn3_e2s` node8 compute per member-step
  (falls back to a3mega's 1.40 s). It is the same for both walkers and reported separately.
- **No persistent JAX compile cache** on the VM, so every GenCast process pays a cold
  compile, as production 0.25 does (its executable is over the 2 GiB cache limit).
- **The phase-2 batched FCN3 wave runs only if** the sweep's best B beats B=1 by
  `BATCH_GAIN_MIN` (10%) in walker-steps/s.

## Inputs the bundle carries (all exist on Derecho, checked 2026-10-15)

- `runs/models/params/GenCast {0p25deg,1p0deg} <2019.npz` (220 MB each), `runs/models/stats/*`
- `runs/xres/0p25/week3/inputs/PNW_HeatDome_2021_inputs.nc` (706 MB, init 2021-06-07 00Z)
- `runs/xres/1p0/week3/inputs/PNW_HeatDome_2021_inputs.nc` (44 MB)
- `runs/fcn3/week3/ic/PNW_HeatDome_2021_ic.nc` (299 MB, same instant)
- optional `--with-fcn3-weights`: `runs/fcn3/.cache/e2s/fcn3/*` (2.85 GB)

## Next steps

1. **Finish the GPU-path review (~20 min).** Re-read the rest of `fcn3_walker.py`, then
   `fcn3_bench.py` and `checks.py`, against the five risks above.
2. **Decide on a pre-Nebius GPU smoke test (needs the user's OK: it spends allocation).**
   Casper has 2 nodes with `gpu_type=h100` (80 GB), so `checks.py` plus a 1-segment
   `fcn3_bench` and `gencast_bench` would catch risks 1-3 before the VM is booked. Caveat:
   the Derecho `fcn3` env's torch-harmonics may not carry sm_90 kernels. Job name must be
   `Vayuh-s2s`, account `UCSC0009`, queue `casper@casper-pbs`.
3. **Build the bundle (~3 min):** `bash wbench/vm/make_bundle.sh`.
4. **Follow `wbench/vm/README.md` steps 2-6** on the VM (~1 h setup + ~1.5 h run). A
   `WB_QUICK=1` run first (~35 min) is the cheap way to shake out VM-side problems.
5. **Optional:** add a short `wbench/` pointer to the root `CLAUDE.md` stack table, and
   commit on a branch when the user asks.

## Files

    wbench/__init__.py        design: the 3 layers, 4 configs, why two FCN3 paths, why checks
    wbench/wconfig.py         configs, sizing knobs (WB_*), paths, a3mega reference rates, case schedule
    wbench/timing.py          Phases timer, summarize, atomic JSON results, host_info (stdlib)
    wbench/gencast_bench.py   GenCast layers 1+2, composed from aires.walker; --check for the login node
    wbench/fcn3_walker.py     NativeFCN3 (batched, per-walker noise), state/diag I/O, seeds, CONUS crop
    wbench/fcn3_bench.py      FCN3 --mode sweep|segments, --path native|e2s, --io nc|npz
    wbench/checks.py          7 FCN3 walker checks -> one checks JSON
    wbench/report.py          layers 1-3 -> REPORT.md, results.json, results.csv; --best-batch
    wbench/tests/             test_wconfig, test_report, test_fcn3_walker (CPU)
    wbench/vm/                README.md runbook, make_bundle.sh, setup_vm.sh, env.sh, run_all.sh,
                              cpu_slices.py, requirements-gencast.txt, constraints-fcn3.txt

## Deterministic AI+RES scorers on Nebius: Pangu24 and SFNO (built 2026-10-23, NOT RUN)

**The question.** Can ONE deterministic forecast per walker replace FCN3's M-member score?
`aires/score_noise.py` (aires/HANDOFF.md "What M buys") says what that would be worth: a
scorer at one member's cost that tracks the conditional mean as well as FCN3's ensemble
mean gives N_eq 418 at 3 sd against M=6's 368, ~3x direct sampling per H100-h; one only as
good as FCN3 M=1 gives 224. Those are surrogate numbers about a hypothetical scorer. This
run puts two real ones on the same 64 cases Gate 3 scored (16 walkers x t_k = 6/9/12/15 d,
PNW_HeatDome_2021) and times them on an H100.

It also measures `frac` - the share of FCN3's score error that is model error, the number
that halves or keeps every N_eq above. One scorer's rho cannot pin it; two scorers with
independent errors can: `v = rho_A rho_B / corr(theta_A, theta_B)`. With FCN3, Pangu and
SFNO there are three such pairs, so the independence assumption is checked, not assumed:
an error two scorers share biases that pair's v DOWN, to v^2 / (v + shared variance), so
the pairs disagree (`wbench/tests/test_det_score.py` pins the direction).

**Code:** `wbench/det_score.py` (stages `sp-check`, `ic`, `infer`, `reduce`),
`wbench/vm/run_det.sh`, `setup_vm.sh` step `det`, `make_bundle.sh --det`.

| piece | state |
|---|---|
| ICs (64, FCN3's 72 adapter channels + `sp`) | **built on Derecho** (`det_score --stage ic`), `runs/aires/PNW_HeatDome_2021/det/ic/`, 199 MB each |
| `sp` for SFNO (GenCast has none) | **validated against ERA5** (`--stage sp-check`, 2021-06-13 00Z): integrating down from the lowest pressure level above ground gives RMSE 0.40 hPa over land, 0.99 hPa above 1500 m, p99 2.8 hPa. The msl-only formula was 4.0 / 13.9 / 35 hPa, so it is used only where the ground is below the 1000 hPa surface |
| tests | `wbench/tests/test_det_score.py`, 7 CPU tests (sp, the v estimator incl. the shared-error bias, IC source, shard balance, cost). All of `wbench/tests` passes in `my-env` on Derecho |
| reduce | end-to-end tested on Derecho with a stand-in model (FCN3 member 0); see "Verified" below |
| GPU path (`--stage infer` without `--dry`) | **never run.** Pangu24/SFNO load, `earth2studio.run.deterministic` and the zarr -> cube assembly are untested until the VM |
| Nebius run | **not started** |

### Verified (2026-10-23, Derecho login node, no GPU)

- **ICs:** 64 of 64, (1, 73, 721, 1440), all finite, 199 MB each, 12 GB. `sp` exceeds `msl`
  only on 5759 cells, all at or below 1 m height (down to -141 m, the Caspian and such):
  ground below sea level, where it must.
- **`--dry` for both models in the Derecho `fcn3` env** (earth2studio 0.17.0a0, the VM's
  commit): Pangu24 takes 69 of the 73 IC channels, SFNO all 73; step counts 15 x 24 h
  (t_k = 6 d) down to 6 x 24 h, and 60 x 6 h down to 24 x 6 h.
- **Reduce, end to end, in the `fcn3` env** with a stand-in model (FCN3 member 0 of each
  Gate 3 cube, t2m only, written where a det cube goes): its Pearson r equals member 0's
  to 6 decimals at every lead, and the FCN3 reference columns reproduce Gate 3 (M=6 rho_s
  0.61 / 0.80 / 0.96 / 0.99, M=1 0.67 at 9 d). The stand-in's v is low (0.52 at 6 d vs
  frac=0's 0.59) because member 0 is inside FCN3's M=6 mean - a shared error, biasing v
  down as it should. Stand-in artifacts deleted.

### Run it (~1.5 h, ~$3-4 on one preemptible H100)

Agents cannot create Nebius resources, so every `nebius` command is Eric's.

1. **Bundle (Derecho login node, ~10 min, needs the 64 ICs).**
   `bash wbench/vm/make_bundle.sh --det` -> `/glade/derecho/scratch/exu/wbench_bundle/wbench_det_bundle.tar`
   (~13 GB) + `.sha256`.
2. **Stage it** in `s3://wbench-bundle` as in `vm/README.md` step 2 (laptop relay; with 13 GB
   the laptop leg dominates). Pull on the VM in-region.
3. **VM: one H100, preemptible**, `default-project-eu-north1`, spot max-price policy
   `h100-max-2` (see "Nebius and R2 operations" above for the three preemptible flags). One
   GPU is enough - 64 forecasts - and keeps the timings comparable with `small1`'s solo
   rows. The run is cache-aware, so a preemption costs only the forecast in flight.
4. **Setup (~30 min):** `bash wbench/vm/setup_vm.sh sys verify mamba fcn3 det`. No GenCast
   env, no torch-harmonics build, no FCN3 weights: neither model needs them, and the
   reduce imports nothing from JAX. Step `det` asserts onnxruntime's CUDA provider and
   loads both models once on the GPU so the weights download now.
5. **Run (~30-45 min, unmeasured):** `WB_NGPU=1 bash wbench/vm/run_det.sh` in tmux. It runs
   `--dry` first (input contract + step counts), then Pangu24, then SFNO, then the reduce.
6. **Bring it home, then delete the VM and its disk:** `runs/aires/PNW_HeatDome_2021/det/`
   minus `ic/` (cubes are CONUS t2m only, small) to R2
   `vayuh/eric/wbench/det1/`. `det_reduce.json` and `logs/reduce.log` are the answer.

### What the answer looks like

`reduce.log` prints, per model and t_k: rho_s against the walkers' realized A_L next to
FCN3 M=1 and M=6 (the Gate 3 bar: rho_s >= 0.5 at 9 and 12 d; FCN3 M=1 gives 0.67 / 0.93),
the predictability v estimated from each pair, and N_eq at 3 / 3.5 sd with the measured
H100-h per case at frac = 0, 0.5 and the model's own estimate. Read it in this order:

1. **Does it pass Gate 3?** If rho_s at 9 d is below FCN3 M=1's 0.67, it is not a cheaper
   scorer, it is a worse one, whatever its speed.
2. **Is it cheaper than one FCN3 member?** FCN3 is 3.94 s per simulated day on an H100.
3. **Do the three v estimates agree?** If FCN3 x Pangu, FCN3 x SFNO and Pangu x SFNO differ
   by more than their bootstrap CIs, the independent-error assumption fails and `frac` is
   not measured - say so rather than averaging. The lowest pair is the one with the most
   shared error (Pangu x SFNO, both trained on ERA5 with similar losses, is the suspect).

### Things that could break on the VM

- **onnxruntime-gpu against the cu128 torch stack** (step `det`). If the CUDA provider is
  missing, Pangu runs on the CPU at minutes per step; the step fails loudly instead.
- **`deterministic()` output layout.** `assemble()` expects zarr `t2m` as
  (time, lead_time, lat, lon) with lat descending, as FCN3's `ensemble()` output has
  minus the ensemble dim. `aindex.check_window` will refuse a cube whose 7-day window does
  not hold 7 (Pangu24) or 25 (SFNO) frames, so a wrong layout fails, it does not mislead.
- **Pangu24's 7 daily frames** are a 00Z-only week mean. A_L is an anomaly against the
  00Z climatology, so the diurnal sampling is consistent, but it is not the 6-hourly mean
  FCN3 is scored on. Rank skill is what Gate 3 measures; note it if Pangu lands close.
- **Model memory.** Neither is expected near 80 GB; `timing_shard0.json` records torch peak
  (SFNO) and `nvidia-smi` used memory (both, since Pangu's onnx session is outside torch).
