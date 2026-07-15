"""Main DDP training loop for HRRR EDM.

Handles gradient accumulation with model.no_sync(), bf16 autocast,
EMA updates, checkpointing, W&B logging, and two-phase training.
"""

import logging
import os
import shutil
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

log = logging.getLogger(__name__)

from model.ema import EMAModel
from training.loss import EDMLoss
from training.sigma_sampler import EDMSigmaSampler
from training.variable_weighting import OcampoWeighting

# Sigma-stratified train logging. The EDM loss depends strongly on the sampled noise level
# (empirically sigma alone drives ~74% of the per-step log-loss variance), so a single
# unstratified number is dominated by which sigma was drawn, not by whether the model
# improved. These edges split the sampled sigma into 4 bands (the sampler's P_mean=-1.2 puts
# the median sigma near 0.3) so each band's mean loss is a stable read on progress at that
# noise scale. torch.bucketize(sigma, EDGES) -> 0..len(EDGES); names line up with the bands.
SIGMA_LOG_EDGES = (0.1, 0.5, 2.0)
SIGMA_LOG_BUCKET_NAMES = ("lo", "mlo", "mhi", "hi")  # <0.1, 0.1-0.5, 0.5-2, >2


def _window_averages(win_sum, win_n, bkt_sum, bkt_n) -> dict:
    """Turn summed logging-window accumulators into mean losses. Pure arithmetic (unit-tested).

    Inputs are already reduced across ranks (a plain SUM all-reduce). Dividing the summed
    loss by the summed micro-batch count yields the mean over accum_steps * log_every_n_steps
    * world_size samples, versus the one-rank-one-sample value the trainer used to log.

    win_sum: indexable of [loss, loss_prog, loss_diag] sums over the window.
    win_n:   scalar count of micro-batches summed into win_sum.
    bkt_sum / bkt_n: per-sigma-bucket loss sums and micro-batch counts (len == buckets).
    Returns {metric: float}; empty sigma buckets are omitted rather than logged as NaN.
    """
    out: dict = {}
    n = float(win_n)
    if n > 0:
        out["train/loss"] = float(win_sum[0]) / n
        out["train/loss_prog"] = float(win_sum[1]) / n
        out["train/loss_diag"] = float(win_sum[2]) / n
    for i, name in enumerate(SIGMA_LOG_BUCKET_NAMES):
        ni = float(bkt_n[i])
        if ni > 0:
            out[f"train/loss_sigma_{name}"] = float(bkt_sum[i]) / ni
    return out


class Trainer:
    """DDP-aware training loop for HRRR EDM.

    Args:
        model: EDMPreconditioning model wrapped in DDP.
        ema: EMAModel for validation/inference.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (can be None for dummy runs).
        loss_fn: EDMLoss instance.
        sigma_sampler: EDMSigmaSampler instance.
        variable_weighting: OcampoWeighting instance.
        cfg: OmegaConf config object.
    """

    def __init__(
        self,
        model: DDP,
        ema: EMAModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        loss_fn: EDMLoss,
        sigma_sampler: EDMSigmaSampler,
        variable_weighting: OcampoWeighting,
        cfg,
    ):
        self.model = model
        self.ema = ema
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.sigma_sampler = sigma_sampler
        self.variable_weighting = variable_weighting
        self.cfg = cfg

        self.device = next(model.parameters()).device
        self.local_rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main = self.local_rank == 0

        self.accum_steps = cfg.training.accumulation_steps
        self.grad_clip = cfg.training.grad_clip
        self.max_steps = cfg.training.get("max_steps", None)
        self.n_prog = cfg.data.n_prognostic
        self.n_diag = cfg.data.n_diagnostic

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr_max,
            weight_decay=cfg.training.weight_decay,
            fused=cfg.training.fused_optimizer,
        )

        # LR scheduler (cosine annealing)
        total_steps = self._estimate_total_steps()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=cfg.training.lr_min,
        )

        # Metrics sink: a wandb.Run or a JsonlMetricsLogger — same .log()/.finish()
        # interface either way, chosen by cfg.logging.backend in train().
        self.metrics_run = None
        self.global_step = 0
        self.start_epoch = 0

        # Checkpoint dir — explicit override (cfg.ckpt_dir) wins; else a project-local
        # default. A fresh run should set cfg.ckpt_dir to its own empty dir so it does
        # not auto-resume an incompatible checkpoint.
        _ckpt_override = cfg.get("ckpt_dir", None)
        self.ckpt_dir = Path(_ckpt_override) if _ckpt_override else (
            Path(__file__).resolve().parent.parent / "checkpoints"
        )
        self.latest_path = self.ckpt_dir / "latest.pt"
        self.stop_file = self.ckpt_dir / "STOP"
        self.save_every_n_steps = int(cfg.training.get("save_every_n_steps", 0) or 0)
        self.keep_last_n = int(cfg.training.get("keep_last_n_checkpoints", 3))
        self.stop_poll_every = int(cfg.training.get("stop_file_poll_every_n_micro_steps", 10))

        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            # A STOP left behind by the previous run would make this job exit instantly.
            if self.stop_file.exists():
                self.stop_file.unlink()
                log.info(f"Cleared stale stop file {self.stop_file}")
        if dist.is_initialized():
            dist.barrier()  # no rank may glob ckpt_dir before rank 0 has made it

        # Stop triggers, both deadline-free:
        #   - STOP sentinel file in ckpt_dir (the manual path — see slurm/stop.sh)
        #   - SIGUSR1 (`scancel -s USR1`) or SIGTERM (plain `scancel`, walltime kill)
        self._should_stop = False
        self._resumed = False
        signal.signal(signal.SIGTERM, self._handle_stop_signal)
        signal.signal(signal.SIGUSR1, self._handle_stop_signal)

    def _estimate_total_steps(self) -> int:
        steps_per_epoch = len(self.train_loader) // self.accum_steps
        return steps_per_epoch * self.cfg.training.n_epochs

    def _handle_stop_signal(self, signum, frame):
        """Request a graceful stop at the next micro-step boundary.

        Only sets a flag: torch.save() from inside a signal handler can deadlock on the
        allocator, and the handler can fire on any rank at any point in the step.
        """
        log.info(
            f"{signal.Signals(signum).name} received — checkpointing and exiting "
            f"at the next step boundary."
        )
        self._should_stop = True

    def _check_stop(self, micro_step: int) -> bool:
        """Collective stop check. Must be called by every rank at the same micro-steps.

        All ranks have to agree: if rank 0 broke out of the loop while the others sat in
        backward()'s all-reduce, the job would hang until the NCCL timeout instead of
        exiting. Rank 0 alone polls the STOP file (one NFS stat, not world_size of them,
        and only every stop_poll_every micro-steps) and the verdict is all-reduced.
        """
        if self.is_main and not self._should_stop and micro_step % self.stop_poll_every == 0:
            if self.stop_file.exists():
                log.info(f"Stop file {self.stop_file} found — checkpointing and exiting.")
                self._should_stop = True
                self.stop_file.unlink()  # consume it, so the next job starts normally

        if dist.is_initialized():
            flag = torch.tensor(
                [1.0 if self._should_stop else 0.0], device=self.device
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            self._should_stop = bool(flag.item() > 0)
        return self._should_stop

    def find_latest_checkpoint(self) -> str | None:
        """Resume pointer: latest.pt, else the newest archival snapshot."""
        if not self.ckpt_dir.exists():
            return None
        if self.latest_path.exists():
            return str(self.latest_path)
        # Fallback for dirs written before latest.pt existed. checkpoint_epochNNNN sorts
        # before checkpoint_stepNNNNNNNN, and step numbers are zero-padded, so plain
        # lexicographic order is chronological.
        ckpts = sorted(self.ckpt_dir.glob("checkpoint_*.pt"))
        return str(ckpts[-1]) if ckpts else None

    def train(self) -> None:
        """Main training loop."""
        # Auto-resume, unless train.py already loaded an explicit cfg.resume_from.
        if not self._resumed:
            latest_ckpt = self.find_latest_checkpoint()
            if latest_ckpt:
                self.load_checkpoint(latest_ckpt)
                log.info(f"Auto-resumed from {latest_ckpt}")

        # Rebase the LR schedule on resume: the checkpointed scheduler carries T_max from the
        # world size the run was FIRST launched with; after a scale change the cosine would end
        # far from lr_min (e.g. 3-node start -> T_max=48400, but 56-GPU epochs end at ~22k steps).
        # Rebuild a fresh cosine from the CURRENT lr over the ACTUAL remaining steps — smooth
        # (no LR jump) and lands at lr_min exactly at n_epochs.
        if self.start_epoch > 0 and self.cfg.training.get("rebase_lr_on_resume", True):
            steps_per_epoch = len(self.train_loader) // self.accum_steps
            remaining = max(1, (self.cfg.training.n_epochs - self.start_epoch) * steps_per_epoch)
            cur_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=remaining, eta_min=self.cfg.training.lr_min
            )
            if self.is_main:
                log.info(f"[lr-rebase] cosine rebuilt: {cur_lr:.3e} -> {self.cfg.training.lr_min:.0e} "
                         f"over {remaining} steps (epochs {self.start_epoch}..{self.cfg.training.n_epochs})")

        # Metrics sink. Logging must NEVER kill an expensive distributed training
        # run (bad API key, network blip, full disk) — every failure path below
        # degrades to "no metrics" with a warning and training continues.
        backend = str(self.cfg.logging.get("backend", "wandb"))
        if self.is_main and self.cfg.logging.enabled:
            if backend == "local":
                from training.metrics_logger import JsonlMetricsLogger

                self.metrics_run = JsonlMetricsLogger(
                    self.cfg.logging.metrics_path,
                    meta={
                        "resumed_at_step": self.global_step,
                        "start_epoch": self.start_epoch,
                        "world_size": self.world_size,
                    },
                )
                log.info(f"Logging metrics to {self.cfg.logging.metrics_path}")
            elif backend == "wandb":
                try:
                    import wandb

                    self.metrics_run = wandb.init(
                        project=self.cfg.logging.project,
                        entity=self.cfg.logging.get("entity", None),
                        tags=list(self.cfg.logging.tags),
                        config=dict(self.cfg),
                        resume="allow",
                    )
                except ImportError:
                    log.info("wandb not installed, skipping logging")
                except Exception as e:
                    log.warning(
                        f"wandb.init failed ({type(e).__name__}: {e}); continuing without W&B"
                    )
                    self.metrics_run = None
            else:
                log.warning(f"unknown logging.backend={backend!r}; metrics logging disabled")

        for epoch in range(self.start_epoch, self.cfg.training.n_epochs):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self._train_epoch(epoch)

            # _train_epoch bailed part-way through on a stop request. Checkpoint the last
            # completed optimizer step and leave. The epoch is not marked complete, so the
            # next job re-runs it from these exact weights (with a fresh shuffle).
            if self._should_stop:
                self._save_checkpoint(epoch, epoch_completed=False)
                if self.is_main:
                    log.info(
                        f"Graceful stop during epoch {epoch} at step {self.global_step}. "
                        f"Resubmit the job to resume."
                    )
                break

            if self.is_main:
                self._log_metrics(train_metrics, epoch, prefix="train")

            # Validation
            if (
                self.val_loader is not None
                and (epoch + 1) % self.cfg.training.val_every_n_epochs == 0
            ):
                val_metrics = self._validate(epoch)
                if self.is_main:
                    self._log_metrics(val_metrics, epoch, prefix="val")

                # Update variable weights
                self.variable_weighting.update(
                    epoch, val_metrics["per_variable_loss"]
                )

            # Early termination for debug runs. Checked before the epoch-end save because
            # hitting max_steps cuts the epoch short — recording it as complete would make
            # a resumed run skip the rest of it.
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self._save_checkpoint(epoch, epoch_completed=False)
                if self.is_main:
                    log.info(f"Reached max_steps={self.max_steps}, stopping.")
                break

            # Save checkpoint every epoch (critical for walltime-limited jobs)
            self._save_checkpoint(epoch, epoch_completed=True)

        if self.metrics_run is not None:
            self.metrics_run.finish()

    def _iter_batches(self):
        """Yield training batches, tolerating a DataLoader that dies during shutdown.

        A signal is delivered to every process in the job step, so the DataLoader worker
        processes can be torn down a moment before the main loop notices its own stop
        flag — which surfaces here as a worker crash mid-iteration. Once we are stopping
        that is expected, not an error: end the epoch and let the caller checkpoint.
        """
        it = iter(self.train_loader)
        while True:
            try:
                yield next(it)
            except StopIteration:
                return
            except Exception:
                if not self._should_stop:
                    raise
                log.warning("DataLoader died during shutdown — ending epoch early.")
                return

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_loss_prog = 0.0
        total_loss_diag = 0.0
        n_steps = 0
        B = 0
        t_start = time.perf_counter()

        # Logging window: EVERY micro-batch feeds these accumulators; at each logging
        # boundary they are all-reduced across ranks, averaged, and reset. This replaces the
        # old "log one rank's single micro-batch loss" with a mean over
        # accum_steps * log_every_n_steps * world_size samples — the smoothing you were
        # eyeballing in the plot, done at the source. GPU tensors so nothing syncs to host
        # until a logging boundary.
        edges = torch.tensor(SIGMA_LOG_EDGES, device=self.device)
        n_buckets = len(SIGMA_LOG_BUCKET_NAMES)
        win_sum = torch.zeros(3, device=self.device)   # loss, loss_prog, loss_diag
        win_n = torch.zeros((), device=self.device)    # micro-batch count
        bkt_sum = torch.zeros(n_buckets, device=self.device)
        bkt_n = torch.zeros(n_buckets, device=self.device)

        self.optimizer.zero_grad()

        for micro_step, batch in enumerate(self._iter_batches()):
            x_cond = batch["x_cond"].to(self.device, non_blocking=True)
            # NOTE (downscaler): target_prognostic holds the normalized HRRR *state*
            # (full field), NOT a T->T+6h tendency. The denoising math is identical.
            target_prognostic = batch["target_prognostic"].to(self.device, non_blocking=True)
            target_diagnostic = batch["target_diagnostic"].to(self.device, non_blocking=True)

            # Concatenate target: (B, n_prog + n_diag, H, W)
            target = torch.cat([target_prognostic, target_diagnostic], dim=1)
            B = target.shape[0]

            assert target.shape[1] == self.n_prog + self.n_diag, (
                f"Target channels {target.shape[1]} != "
                f"n_prog({self.n_prog}) + n_diag({self.n_diag})"
            )

            # Sample noise
            sigma = self.sigma_sampler(B, self.device)
            noise = torch.randn_like(target)
            x_noisy = target + sigma[:, None, None, None] * noise

            # Determine sync context for gradient accumulation
            is_accum_step = (micro_step + 1) % self.accum_steps != 0
            ctx = self.model.no_sync if (is_accum_step and dist.is_initialized()) else _nullcontext

            with ctx():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    D_out = self.model(x_noisy, sigma, x_cond)

                    var_weights = self.variable_weighting.get_weights(
                        epoch, self.device
                    )
                    loss_dict = self.loss_fn(D_out, target, sigma, var_weights)
                    loss = loss_dict["loss"] / self.accum_steps

                loss.backward()

            # Feed the logging window on EVERY micro-batch (detached; no host sync here).
            # Bucketing by the micro-batch's mean sigma is exact at per_gpu_batch=1 (one
            # sigma per micro-batch, the current config) and an approximation above that.
            with torch.no_grad():
                win_sum[0] += loss_dict["loss"].detach()
                win_sum[1] += loss_dict["loss_prog"].detach()
                win_sum[2] += loss_dict["loss_diag"].detach()
                win_n += 1
                b = torch.bucketize(sigma.mean().detach().reshape(1), edges)
                bkt_sum.index_add_(0, b, loss_dict["loss"].detach().reshape(1))
                bkt_n.index_add_(0, b, torch.ones(1, device=self.device))

            # Optimizer step at accumulation boundary
            if not is_accum_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.ema.update(self.model.module if hasattr(self.model, "module") else self.model)
                self.scheduler.step()
                self.global_step += 1

                total_loss += loss_dict["loss"].item()
                total_loss_prog += loss_dict["loss_prog"].item()
                total_loss_diag += loss_dict["loss_diag"].item()
                n_steps += 1

                # Heartbeat to the console/slurm log — reliable progress signal (no W&B needed,
                # no NFS/ssh/power guessing). grep "[hb]" in the slurm log to watch live steps.
                if self.is_main and self.global_step % 10 == 0:
                    log.info(f"[hb] step={self.global_step} "
                             f"loss={loss_dict['loss'].item():.4f} "
                             f"peak_gb={torch.cuda.max_memory_allocated(self.device)/1e9:.1f}")

                # Smoothed / rank-reduced / sigma-stratified train metrics. The all-reduce
                # runs on EVERY rank (like _check_stop) and must stay OUTSIDE the rank-0
                # `metrics_run` guard: if only rank 0 entered the collective the other ranks
                # would never reach it and NCCL would hang until timeout. Only rank 0 then
                # writes the reduced values. The window is reset on all ranks afterwards.
                if self.global_step % self.cfg.logging.log_every_n_steps == 0:
                    packed = torch.cat([win_sum, win_n.reshape(1), bkt_sum, bkt_n])
                    if dist.is_initialized():
                        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                    if self.metrics_run is not None:
                        r_sum, r_n = packed[:3], packed[3]
                        r_bsum, r_bn = packed[4:4 + n_buckets], packed[4 + n_buckets:]
                        log_dict = _window_averages(r_sum, r_n, r_bsum, r_bn)
                        log_dict.update({
                            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/sigma_mean": sigma.mean().item(),
                            "train/peak_mem_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
                        })
                        self.metrics_run.log(log_dict, step=self.global_step)
                    win_sum.zero_(); win_n.zero_(); bkt_sum.zero_(); bkt_n.zero_()

                # Periodic checkpoint. An epoch on the real dataset is far too coarse to
                # be the unit of durability for a job that gets killed by hand.
                if (
                    self.save_every_n_steps
                    and self.global_step % self.save_every_n_steps == 0
                ):
                    self._save_checkpoint(epoch, epoch_completed=False)

            # Stop check on EVERY micro-step, not just optimizer-step boundaries: one
            # optimizer step is `accumulation_steps` forward/backward passes, which at full
            # 1059x1799 resolution can outlast scancel's 30 s KillWait on its own.
            # Abandoning a partial accumulation is safe — the gradient is dropped and the
            # weights are still exactly those of the last completed optimizer step.
            if self._check_stop(micro_step):
                self.optimizer.zero_grad(set_to_none=True)
                break

            # Early termination
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

        elapsed = time.perf_counter() - t_start
        n_steps = max(n_steps, 1)
        samples_total = n_steps * self.accum_steps * B * self.world_size

        return {
            "loss": total_loss / n_steps,
            "loss_prog": total_loss_prog / n_steps,
            "loss_diag": total_loss_diag / n_steps,
            "throughput": samples_total / elapsed,
            "epoch": epoch,
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        model_to_eval = self.model.module if hasattr(self.model, "module") else self.model

        with self.ema.ema_scope(model_to_eval):
            model_to_eval.eval()
            total_loss = 0.0
            per_var_losses = []
            n_steps = 0

            for batch in self.val_loader:
                x_cond = batch["x_cond"].to(self.device, non_blocking=True)
                target_prognostic = batch["target_prognostic"].to(self.device, non_blocking=True)
                target_diagnostic = batch["target_diagnostic"].to(self.device, non_blocking=True)
                target = torch.cat([target_prognostic, target_diagnostic], dim=1)
                B = target.shape[0]

                sigma = self.sigma_sampler(B, self.device)
                noise = torch.randn_like(target)
                x_noisy = target + sigma[:, None, None, None] * noise

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    D_out = model_to_eval(x_noisy, sigma, x_cond)
                    loss_dict = self.loss_fn(D_out, target, sigma)

                total_loss += loss_dict["loss"].item()
                per_var_losses.append(loss_dict["per_variable_loss"])
                n_steps += 1

        n_steps = max(n_steps, 1)
        return {
            "loss": total_loss / n_steps,
            "per_variable_loss": torch.stack(per_var_losses).mean(dim=0),
        }

    def _log_metrics(self, metrics: dict, epoch: int, prefix: str) -> None:
        loss_str = f"{prefix}/loss={metrics['loss']:.6f}"
        if "throughput" in metrics:
            loss_str += f" throughput={metrics['throughput']:.1f} samp/s"
        log.info(f"[Epoch {epoch}] {loss_str}")

        if self.metrics_run is not None:
            log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()
                        if isinstance(v, (int, float))}
            self.metrics_run.log(log_dict, step=self.global_step)

    def _save_checkpoint(self, epoch: int, epoch_completed: bool) -> None:
        """Write a checkpoint (rank 0) and hold the other ranks until it lands.

        `epoch_completed` distinguishes a checkpoint taken at the end of an epoch from one
        taken part-way through it, which is what load_checkpoint() keys off to decide
        whether to resume at this epoch or the next.
        """
        if self.is_main:
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            ckpt = {
                "epoch": epoch,
                "epoch_completed": epoch_completed,
                "global_step": self.global_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "ema_state_dict": self.ema.state_dict(),
                "variable_weighting": self.variable_weighting.state_dict(),
            }
            snapshot = self.ckpt_dir / f"checkpoint_step{self.global_step:08d}.pt"
            self._atomic_save(ckpt, snapshot)
            self._point_latest_at(snapshot)
            log.info(f"Saved checkpoint: {snapshot} (epoch_completed={epoch_completed})")

            # Retain a short history. latest.pt is a hardlink, so it keeps the newest
            # snapshot's data alive even once that snapshot is pruned.
            snapshots = sorted(self.ckpt_dir.glob("checkpoint_*.pt"))
            for old in snapshots[: -self.keep_last_n]:
                old.unlink()
                log.info(f"Removed old checkpoint: {old}")

        # Hold the other ranks until the write lands: without this they can reach
        # destroy_process_group() while rank 0 is still saving, which aborts it mid-write.
        # The checkpoint is already on disk by this point, so a failed barrier (a rank that
        # died when the stop signal raced its DataLoader teardown) costs nothing — never
        # let it take down an otherwise clean shutdown.
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                log.warning(f"post-save barrier failed ({type(e).__name__}) — exiting anyway")

    def _atomic_save(self, obj: dict, path: Path) -> None:
        """Save via a temp file + rename, so a kill mid-write cannot corrupt `path`.

        torch.save() straight to the destination leaves a truncated file if the process
        dies part-way, and auto-resume would then load it. os.replace() is atomic, so the
        previous checkpoint survives intact until the new one is fully on disk.
        """
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def _point_latest_at(self, snapshot: Path) -> None:
        """Atomically repoint latest.pt at `snapshot` (hardlink — no second copy)."""
        tmp = self.ckpt_dir / f"latest.pt.tmp{os.getpid()}"
        tmp.unlink(missing_ok=True)
        try:
            os.link(snapshot, tmp)
        except OSError:
            shutil.copyfile(snapshot, tmp)  # filesystem without hardlinks
        os.replace(tmp, self.latest_path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.ema.load_state_dict(ckpt["ema_state_dict"])
        # Older checkpoints predate persisted variable weights; they resume uniform.
        if "variable_weighting" in ckpt:
            self.variable_weighting.load_state_dict(ckpt["variable_weighting"])
        self.global_step = ckpt["global_step"]

        # An interrupted epoch is re-run from the top with a fresh shuffle. No training
        # progress is lost — the weights, optimizer moments and EMA are exactly those of
        # the last completed optimizer step; only the within-epoch sample coverage is
        # resampled, which is immaterial over hundreds of epochs. (Checkpoints written
        # before this field existed were all epoch-end saves, hence the True default.)
        epoch_completed = ckpt.get("epoch_completed", True)
        self.start_epoch = ckpt["epoch"] + 1 if epoch_completed else ckpt["epoch"]
        self._resumed = True

        where = "after" if epoch_completed else "during"
        log.info(
            f"Loaded checkpoint from {where} epoch {ckpt['epoch']} "
            f"(step {self.global_step}) — resuming at epoch {self.start_epoch}"
        )


class _nullcontext:
    """Minimal no-op context manager for non-DDP runs."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
