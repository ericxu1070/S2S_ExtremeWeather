#!/usr/bin/env python3
"""Print one CPU list per GPU, NUMA-local where nvidia-smi says which cores are local.

    $ python3 cpu_slices.py 8
    0 0-11
    1 12-23
    ...

Eight workers that each spawn one thread per core oversubscribe a node eightfold (a3mega
job 700: 1856 threads on 208 cores, GPUs idle 80+ min), so every worker is pinned to its
own slice. ``nvidia-smi topo -m`` gives each GPU's NUMA-local cores; the GPUs that share a
NUMA node split its cores evenly, in index order. Without topology (or on a parse
failure) the node's cores are split evenly instead. Stdlib only: runs on the system python.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys


def expand(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def compress(cpus: list[int]) -> str:
    cpus = sorted(cpus)
    runs, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:]:
        if c != prev + 1:
            runs.append((start, prev))
            start = c
        prev = c
    runs.append((start, prev))
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs)


def topo_affinity(n_gpu: int) -> dict[int, list[int]] | None:
    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True,
                             timeout=30, check=True).stdout
    except Exception:
        return None
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in out.splitlines()]
    head = next((l for l in lines if "CPU Affinity" in l), None)
    if head is None:
        return None
    cols = re.split(r"\t+", head.strip())
    try:
        ci = cols.index("CPU Affinity")
    except ValueError:
        return None
    aff = {}
    for l in lines:
        m = re.match(r"^GPU(\d+)\t", l)
        if not m:
            continue
        cells = re.split(r"\t+", l.strip())
        # the row has a leading "GPUk" label the header does not
        if len(cells) > ci + 1 and re.match(r"^[\d,\-]+$", cells[ci + 1]):
            aff[int(m.group(1))] = expand(cells[ci + 1])
    if len(aff) < n_gpu or any(not v for v in aff.values()):
        return None
    allowed = set(os.sched_getaffinity(0))
    return {g: [c for c in aff[g] if c in allowed] for g in range(n_gpu)}


def slices(n_gpu: int) -> dict[int, list[int]]:
    aff = topo_affinity(n_gpu)
    out = {}
    if aff:
        groups = {}
        for g in range(n_gpu):
            groups.setdefault(tuple(aff[g]), []).append(g)
        for cpus, gpus in groups.items():
            k = len(cpus) // len(gpus)
            if k < 1:
                aff = None
                break
            for i, g in enumerate(gpus):
                out[g] = list(cpus[i * k:(i + 1) * k])
    if not aff:
        cpus = sorted(os.sched_getaffinity(0))
        k = max(len(cpus) // n_gpu, 1)
        out = {g: cpus[(g * k) % len(cpus):(g * k) % len(cpus) + k] for g in range(n_gpu)}
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for g, cpus in sorted(slices(n).items()):
        print(g, compress(cpus))
