# Walker benchmark on a Nebius 8xH100 VM

FCN3 vs GenCast as the AI+RES walker: raw step speed, the cost of a 3-day walker segment
with checkpoint I/O (1 worker, then 8 per node), and H100-h per AI+RES case. What each
layer means is in `wbench/__init__.py`.

**Total: ~1 h of setup (unattended) + ~1.5 h of benchmark.**

## Steps

1. **Build the bundle on Derecho (~3 min, login node, no GPU).**

   ```bash
   cd /glade/derecho/scratch/exu/S2S_ExtremeWeather
   bash wbench/vm/make_bundle.sh
   ```

   Output: `/glade/derecho/scratch/exu/wbench_bundle/wbench_bundle.tar` (~1.5 GB) plus
   `.sha256`. Add `--with-fcn3-weights` (+2.7 GB) only if the VM cannot reach HuggingFace.
   The bucket copy (2026-10-20) was built with it: 4.1 GB, sha256 `a570ccc4...`.

2. **Stage it in the Nebius bucket, then pull it on the VM (~5-15 min).** The bucket is
   `s3://wbench-bundle` (eu-north1, project `default-project-eu-north1`); the S3 key is
   service account `wbench-s3`, saved on the laptop as `~/.aws` profile `nebius`
   (expires 2027-01-31). On the laptop (the first `ssh derecho` asks for Duo):

   ```bash
   rsync -a 'derecho:/glade/derecho/scratch/exu/wbench_bundle/wbench_bundle.tar*' ~/wbench_bundle/
   uvx --from awscli aws --profile nebius s3 cp ~/wbench_bundle/ s3://wbench-bundle/ --recursive
   scp ~/.aws/credentials ~/.aws/config <vm>:~/.aws/   # mkdir ~/.aws on the VM first
   ```

   On the VM (eu-north1, so the pull is in-region):

   ```bash
   pip install --user awscli
   aws --profile nebius s3 cp s3://wbench-bundle/ ~/ --recursive
   ```

3. **Unpack on the VM and start a tmux session (~1 min).** A dropped SSH connection
   would otherwise kill the run.

   ```bash
   cd ~ && sha256sum -c wbench_bundle.tar.sha256
   mkdir -p ~/wbench && tar -xf ~/wbench_bundle.tar -C ~/wbench
   cd ~/wbench/S2S_ExtremeWeather
   tmux new -s wb
   ```

4. **Set up the envs (~45 min, unattended).**

   ```bash
   bash wbench/vm/setup_vm.sh
   ```

   Done when the last line reads `setup complete - next: bash wbench/vm/run_all.sh`.
   Each step leaves a marker, so after a failure fix the cause and run the same command
   again. The log is `~/wbench/setup.log`.

5. **Run the benchmark (~1.5 h).**

   ```bash
   bash wbench/vm/run_all.sh
   ```

   Want a smoke run first? `WB_QUICK=1 bash wbench/vm/run_all.sh` takes ~35 min and
   writes a complete but noisier report under its own run id.

6. **Read the answer and bring it home (~1 min).** The report is
   `runs/wbench/<run id>/REPORT.md`; its first section is the answer. On the laptop:

   ```bash
   rsync -av <vm>:~/wbench/S2S_ExtremeWeather/runs/wbench/<run id>/ ./wbench_<run id>/
   ```

   That copies `REPORT.md`, `results.json`/`.csv`, one JSON per measurement in
   `results/`, every worker log, a 5 s GPU utilisation trace and a machine snapshot
   (`nvidia-smi -q`, topology, `lscpu`, both envs' `pip freeze`).

## What runs, and for how long

| phase | what | wall |
|---|---|---|
| preflight | GPUs idle, disk, both envs see a GPU, DISCO kernel, FCN3 pickle | ~1 min |
| solo | GenCast 0.25 x2, GenCast 1.0, FCN3 native sweep x2, FCN3 e2s sweep, walker checks, FCN3 segments (nc, npz, e2s), one config per GPU | ~25 min |
| node8 | 8 identical workers, one wave each: FCN3 native B=1, FCN3 native at the best sweep B (only if it beats B=1 by 10%), FCN3 e2s, GenCast 0.25, GenCast 1.0 | ~45 min |
| report | REPORT.md, results.json, results.csv | <1 min |

GenCast 0.25 deg dominates: ~5 min XLA compile solo, ~15 min with 8 compiling at once, and
~3.5 min per 3-day segment. Every FCN3 wave takes a few minutes.

## Knobs

| variable | default | effect |
|---|---|---|
| `WB_RUN_ID` | UTC timestamp | reuse it to resume: steps with a result JSON are skipped |
| `WB_QUICK` | unset | `1` = 1 timed segment, sweep B=1,2,4, 4 timed sweep steps |
| `WB_SEGMENTS` | 3 | timed segments per worker (after 1 warm-up segment) |
| `WB_SWEEP` | 1,2,3,4,6,8 | FCN3 batch sizes; the sweep stops at the first OOM |
| `WB_CASE_N` | 32,64 | walker counts the case projection prices |
| `WB_IGNORE_BUSY` | unset | `1` = run even if a GPU already holds >2 GB |
| `WB_ALLOW_FP32` | unset | `1` = time FCN3 without the DISCO kernel (slow path, labelled) |
| `WB_OUT` | `runs/wbench` | put run trees elsewhere, e.g. on a faster local disk |

Phases can be run alone, in order: `bash wbench/vm/run_all.sh solo node8 report`.

## If something fails

- **`DISCO CUDA kernel NOT available`.** torch-harmonics was built without its kernel.
  Rerun that step: `bash wbench/vm/setup_vm.sh th`.
- **`cuda-toolkit` will not install.** The NVIDIA channel label moved. Use a
  different 12.8 label: `WB_CUDA_LABEL=nvidia/label/cuda-12.8.0 bash wbench/vm/setup_vm.sh fcn3 th`.
- **The FCN3 weights do not download.** HuggingFace is blocked. Rebuild the bundle with
  `--with-fcn3-weights` (step 1) and copy it again.
- **`GPUs already in use`.** Another process holds memory. Check `nvidia-smi`. If it is
  not yours, do not kill it. Set `WB_IGNORE_BUSY=1` only if you accept contaminated
  timings.
- **One worker `FAILED`.** The line gives its log path. Fix the cause, then run the same
  command with the same `WB_RUN_ID`. Only the missing steps run.
- **GenCast 0.25 runs out of memory.** The checkpoint needs ~64 GB of the 80 GB, so
  something else is on that card.

## Files

    make_bundle.sh          Derecho: code + checkpoints + one event's inputs -> tar
    setup_vm.sh             VM: both envs, torch-harmonics kernel, FCN3 weights + pickle
    env.sh                  VM: put one env on PATH (gencast | fcn3), pin threads
    run_all.sh              VM: preflight -> solo -> node8 -> report, with GPU slot scheduling
    cpu_slices.py           NUMA-local CPU slice per GPU (from nvidia-smi topo)
    requirements-gencast.txt, constraints-fcn3.txt   the pinned env contents
    MANIFEST.sha256         written by make_bundle.sh, checked by setup_vm.sh
