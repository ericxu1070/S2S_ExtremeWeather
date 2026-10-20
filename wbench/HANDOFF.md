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
| Nebius run | **not started** |

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
