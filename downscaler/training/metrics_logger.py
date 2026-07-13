"""Local metrics sink: newline-delimited JSON inside the repo, no external service.

Drop-in for the subset of the wandb.Run interface the trainer uses
(.log(dict, step=...) and .finish()), so the trainer treats both sinks
identically. One JSON object per line; the file is append-only across
stop/resume cycles, so a resumed run keeps writing to the same history.

Two consequences of append-across-resumes to know about:
  - a resume re-runs up to `save_every_n_steps` optimizer steps, so a step
    number can appear twice — consumers should keep the LAST occurrence
    (scripts/plot_metrics.py does);
  - a `run_start` line is written on every (re)start, which doubles as a
    record of when the job was stopped and resumed.

Like the W&B path, logging must never kill a training run: any I/O failure
disables the sink with a warning and training continues.
"""

import json
import logging
import math
import os
import time

log = logging.getLogger(__name__)


def _clean(v):
    """Strict JSON has no NaN/Infinity (e.g. sigma_std of a batch of 1) — emit null."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class JsonlMetricsLogger:
    """Append metrics as JSON lines to a file (rank 0 only, by construction)."""

    def __init__(self, path: str, meta: dict | None = None):
        self.path = path
        self._dead = False
        self._f = None
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._f = open(path, "a", buffering=1)  # line-buffered
        except Exception as e:
            log.warning(f"cannot open metrics file {path} ({e}); metrics logging disabled")
            self._dead = True
        self._write({"event": "run_start", "time": round(time.time(), 3), **(meta or {})})

    def _write(self, obj: dict) -> None:
        if self._dead:
            return
        try:
            obj = {k: _clean(v) for k, v in obj.items()}
            self._f.write(json.dumps(obj, default=float) + "\n")
        except Exception as e:
            log.warning(f"metrics logging failed ({e}); disabled for the rest of this run")
            self._dead = True

    def log(self, metrics: dict, step: int | None = None) -> None:
        row = {"time": round(time.time(), 3)}
        if step is not None:
            row["step"] = step
        row.update(metrics)
        self._write(row)

    def finish(self) -> None:
        try:
            if self._f is not None:
                self._f.close()
        except Exception:
            pass
