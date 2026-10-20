"""Timers, summaries and result records - stdlib only, so both envs and the report share it.

A result is one JSON file per (config, phase, worker, kind). Writes are atomic (tmp +
rename) because eight workers write into one directory at once, and a half-written JSON
that the report then fails to parse is a lost measurement.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1


class Phases:
    """Named wall-clock intervals, accumulated. ``with ph("roll"): ...``."""

    def __init__(self):
        self.t: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0

    def as_dict(self, ndigits: int = 4) -> dict:
        return {k: round(v, ndigits) for k, v in self.t.items()}


def summarize(xs) -> dict:
    """n / median / mean / min / max / p10 / p90 of a sample; empty-safe."""
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    if n == 0:
        return dict(n=0)

    def q(p: float) -> float:
        if n == 1:
            return xs[0]
        k = p * (n - 1)
        lo, hi = int(k), min(int(k) + 1, n - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return dict(n=n, median=round(statistics.median(xs), 5),
                mean=round(statistics.fmean(xs), 5), min=round(xs[0], 5),
                max=round(xs[-1], 5), p10=round(q(0.10), 5), p90=round(q(0.90), 5))


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gpu_query() -> list[dict]:
    """nvidia-smi's view of the visible GPUs; [] where there is none (login node)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,"
             "driver_version,clocks.max.sm,power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except Exception:
        return []
    keys = ("index", "name", "mem_total_mib", "mem_used_mib", "driver", "sm_clock_max_mhz",
            "power_limit_w")
    return [dict(zip(keys, (c.strip() for c in line.split(","))))
            for line in out.strip().splitlines() if line.strip()]


def host_info() -> dict:
    """Where a number was measured. Two machines' numbers are only comparable when this
    says what differed (GPU SKU, driver, CPU, the env knobs that change the model's path)."""
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        mem_kb = int(next(line.split()[1] for line in
                          Path("/proc/meminfo").read_text().splitlines()
                          if line.startswith("MemTotal")))
    except (OSError, StopIteration, ValueError):
        mem_kb = 0
    try:
        aff = len(os.sched_getaffinity(0))
    except AttributeError:
        aff = os.cpu_count()
    knobs = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "XRES_SERIAL_INFER", "XRES_BF16",
             "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
             "XLA_FLAGS", "WB_RUN_ID", "WB_PHASE", "WB_CONCURRENCY", "WB_QUICK")
    return dict(hostname=socket.gethostname(), cpu_model=cpu, cpus_affinity=aff,
                cpus_node=os.cpu_count(), mem_total_gb=round(mem_kb / 1e6, 1),
                gpus=_gpu_query(), python=platform.python_version(),
                env={k: os.environ[k] for k in knobs if k in os.environ})


def result_name(config: str, phase: str, worker: int, kind: str) -> str:
    return f"{config}__{phase}__w{worker:02d}__{kind}.json"


def write_result(results_dir: Path, rec: dict) -> Path:
    """Atomically write one result record; returns its path."""
    for k in ("config", "phase", "worker", "kind"):
        if k not in rec:
            raise KeyError(f"result record needs {k!r}")
    rec = dict(schema=SCHEMA, written=now_utc(), **rec)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    p = results_dir / result_name(rec["config"], rec["phase"], int(rec["worker"]),
                                  rec["kind"])
    tmp = p.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(rec, indent=2, default=str))
    os.replace(tmp, p)
    print(f"[wbench] wrote {p}", flush=True)
    return p


def read_results(results_dir: Path) -> list[dict]:
    out = []
    for p in sorted(Path(results_dir).glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[wbench] skipping unreadable {p.name}: {e}")
    return out


def env_tag() -> dict:
    """The run-level identifiers run_all.sh exports to every worker."""
    return dict(run_id=os.environ.get("WB_RUN_ID", "adhoc"),
                phase=os.environ.get("WB_PHASE", "adhoc"),
                concurrency=int(os.environ.get("WB_CONCURRENCY", 1)))
