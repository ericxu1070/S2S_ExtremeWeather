"""Build-once-load cache for the FCN3 model, shared by the fcn3 six-event driver
(`fcn3/run_fcn3.py`) and the p90 90-case driver (`p90/run_p90.py`).

Constructing FCN3 recomputes the torch-harmonics DISCO geometry tensors on CPU (~0.5-2 min
in isolation). Running one build per worker CONCURRENTLY contends on memory bandwidth --
a3mega job 723: 4 builds still unfinished at 17 min with every GPU idle. So the first
worker to take a node lock builds the model and pickles the WHOLE module to NFS; every
other worker (this job and later ones, across BOTH drivers -- they share one cache dir and
one version tag) just torch.loads it in seconds. Builds never overlap: if pickling is
impossible, workers still build one at a time under the same lock.

The pickle stores the whole nn.Module, not state_dict(), because the DISCO tensors are
registered persistent=False and would otherwise be dropped (verified: buffers + forward
output identical after a save/load round-trip). Loading uses weights_only=False, so only
ever load a file THIS code wrote -- never an untrusted download.

Only earth2studio / torch are imported, lazily, so this module is safe to import from a
compare stage running in a different (JAX-free) conda env.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path


def fcn3_variables() -> list[str]:
    from earth2studio.models.px.fcn3 import VARIABLES
    return list(VARIABLES)


def model_cache_tag() -> str:
    """Version key: any library whose internals a pickled nn.Module depends on, plus the
    FCN3 variable set. A mismatch (env upgrade) misses the cache and rebuilds rather than
    loading an incompatible pickle. Identical across both drivers so they share one file."""
    import earth2studio
    import torch
    import torch_harmonics
    parts = [torch.__version__, torch_harmonics.__version__,
             getattr(earth2studio, "__version__", "?"), ",".join(fcn3_variables())]
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
    return f"{torch.__version__}_th{torch_harmonics.__version__}_{h}"


def model_cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / f"fcn3_model_{model_cache_tag()}.pt"


def build_lock_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "build.lock"


# --- node-local lock (PID-based; a dead holder's lock is stealable) -------------------- #
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _try_claim(path: Path) -> bool:
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(path.read_text().strip() or "0")
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    return False


def _build_model():
    from earth2studio.models.px import FCN3
    print("[model] building FCN3 (CPU-bound DISCO precompute) ...", flush=True)
    t = time.perf_counter()
    model = FCN3.load_model(FCN3.load_default_package())
    print(f"[model] built in {(time.perf_counter() - t) / 60:.2f} min", flush=True)
    return model


def _try_load_pickle(path: Path):
    import torch
    if not path.exists():
        return None
    try:
        t = time.perf_counter()
        model = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[model] loaded cache {path.name} in {time.perf_counter() - t:.1f} s",
              flush=True)
        return model
    except Exception as e:
        print(f"[model] cache {path.name} unusable ({type(e).__name__}: {e}); rebuilding",
              flush=True)
        return None


def _save_pickle(model, path: Path) -> None:
    import torch
    tmp = path.parent / f"{path.name}.tmp.{os.getpid()}"   # name has dots -> not with_suffix
    try:
        torch.save(model, tmp)
        os.replace(tmp, path)                              # atomic publish on NFS
        print(f"[model] saved cache -> {path.name} ({path.stat().st_size / 1e9:.2f} GB)",
              flush=True)
    except Exception as e:
        print(f"[model] could not pickle ({type(e).__name__}: {e}); peers will build",
              flush=True)
        try:
            tmp.unlink()
        except OSError:
            pass


def get_model(cache_dir: Path, disabled: bool = False, poll_s: int = 10,
              timeout_s: int = 2400):
    """Return an FCN3 model, building it AT MOST once per node under a lock. Fast path is a
    pickle load (no lock); the lock only bites on a cold cache so builds never overlap.

    disabled=True builds locally and skips the pickle (set via FCN3_MODEL_CACHE=0)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = model_cache_path(cache_dir)
    lock = build_lock_path(cache_dir)

    m = _try_load_pickle(cache)                            # 1. fast path
    if m is not None:
        return m
    if disabled:
        return _build_model()

    waited = 0
    while True:
        if _try_claim(lock):                               # 2. we build; peers wait
            try:
                m = _try_load_pickle(cache)                # someone may have built meanwhile
                if m is None:
                    m = _build_model()
                    _save_pickle(m, cache)
                return m
            finally:
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
        time.sleep(poll_s)                                 # 3. peer holds lock -> poll pickle
        waited += poll_s
        m = _try_load_pickle(cache)
        if m is not None:
            return m
        if waited >= timeout_s:                            # holder truly stuck/dead
            print(f"[model] waited {timeout_s // 60} min for a peer build; building locally",
                  flush=True)
            return _build_model()
