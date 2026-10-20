#!/usr/bin/env python
"""Layers 1+2 for the FCN3 walker, through either code path.

    --mode sweep      layer 1. Model resident, B walkers stepped together, B swept until
                      the card runs out of memory. Reports seconds per step, walker-steps
                      per second and peak memory per B - the number that says whether
                      batching walkers buys anything.
    --mode segments   layer 2. B walkers per process, each segment exactly what a walker
                      pays: read B parent states, roll 3 d (12 x 6 h), write B restart
                      states (field + noise) and B CONUS diagnostics.

    --path native     ``fcn3_walker.NativeFCN3``: makani directly, one forward per step
                      for the whole batch, noise persisted across segments.
    --path e2s        earth2studio's iterator, exactly as the scorer drives the model: a
                      new iterator (and so freshly drawn noise) every segment, and its
                      batch loop is serial.

Env: ``fcn3``.

    python -m wbench.fcn3_bench --path native --mode sweep --run runs/wbench/<id>
    python -m wbench.fcn3_bench --path native --mode segments --batch 1 --run ... --worker 3
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from fcn3 import fevents as F

# earth2studio / huggingface_hub read these at import time -> set before any e2s import.
os.environ.setdefault("EARTH2STUDIO_CACHE", str(F.FCN3_CACHE / "e2s"))
os.environ.setdefault("HF_HOME", str(F.FCN3_CACHE / "hf"))

from wbench import fcn3_walker as FW                                   # noqa: E402
from wbench import timing as T                                         # noqa: E402
from wbench import wconfig as W                                        # noqa: E402

CONFIG = {"native": "fcn3_native", "e2s": "fcn3_e2s"}


# --------------------------------------------------------------------------- #
# Shared setup
# --------------------------------------------------------------------------- #
def _threads() -> None:
    import torch
    n = os.environ.get("OMP_NUM_THREADS")
    if n:
        torch.set_num_threads(int(n))


def _setup(ph: T.Phases):
    """Model on the GPU, the IC on host. Load is timed - it is a per-leg cost."""
    with ph("import"):
        import torch
        _threads()
    if not torch.cuda.is_available():
        raise SystemExit("[fcn3_bench] no CUDA device visible")
    FW.require_disco_kernel()
    with ph("load"):
        px = FW.load_px("cuda")
        torch.cuda.synchronize()
    ic, t0 = FW.read_ic(W.fcn3_ic_path(), px.variables)
    return torch, px, ic, t0


def _mem(torch) -> dict:
    return dict(peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2),
                total_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))


class E2SRoller:
    """earth2studio's iterator as a walker: one iterator per segment, as the scorer does."""

    def __init__(self, px):
        import torch
        self.torch, self.px = torch, px
        ic = px.input_coords()
        self.lat, self.lon = ic["lat"], ic["lon"]
        self.lat_sl, self.lon_sl = FW.conus_slices(self.lat, self.lon)
        self.variables = list(px.variables)

    def roll(self, x, t0, seed: int, n_steps: int, crop: bool = True):
        """``x`` (B, 72, H, W) on device. Same return contract as ``NativeFCN3.roll``
        minus the noise, which earth2studio keeps internal and re-draws per iterator."""
        torch = self.torch
        B = x.shape[0]
        coords = OrderedDict(ensemble=np.arange(B), time=np.array([np.datetime64(t0)]),
                             lead_time=np.array([np.timedelta64(0, "h")]),
                             variable=np.array(self.variables), lat=self.lat, lon=self.lon)
        self.px.set_rng(seed=int(seed), reset=True)
        it = self.px.create_iterator(x[:, None, None].contiguous(), coords)
        next(it)                                                       # the IC itself
        crops, times, xs = [], [], None
        t = pd.Timestamp(t0)
        for _ in range(n_steps):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            xs, _ = next(it)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            t = t + pd.Timedelta(hours=FW.STEP_H)
            if crop:
                crops.append(xs[:, 0, 0, :, self.lat_sl, self.lon_sl].float().cpu().numpy())
            times.append((t2 - t1, time.perf_counter() - t1))
        return xs[:, 0, 0].clone(), t, crops, times


# --------------------------------------------------------------------------- #
# Layer 1: the batch sweep
# --------------------------------------------------------------------------- #
def sweep(path: str, run: Path, worker: int, batches) -> int:
    config = CONFIG[path]
    tag = T.env_tag()
    ph = T.Phases()
    torch, px, ic, t0 = _setup(ph)
    nat = FW.NativeFCN3(px) if path == "native" else None
    e2s = E2SRoller(px) if path == "e2s" else None
    n_warm, n_timed = W.SWEEP_WARM_STEPS, W.SWEEP_TIMED_STEPS
    print(f"[fcn3_bench] sweep {path}: B in {list(batches)}, {n_warm} warm + {n_timed} "
          f"timed steps each", flush=True)

    rows, first_step_s = [], None
    for B in batches:
        x = noise = None
        try:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.as_tensor(ic).to("cuda")[None].repeat(B, 1, 1, 1)
            seeds = [FW.segment_seed(W.BASE_SEED, w, 0) for w in range(B)]
            if nat is not None:
                gens = nat.generators(seeds)
                noise = nat.init_noise(gens)
                x, noise, t, _, warm = nat.roll(x, t0, noise, gens, n_warm, crop=False)
                x, noise, t, _, st = nat.roll(x, t, noise, gens, n_timed, crop=False)
            else:
                x, t, _, warm = e2s.roll(x, t0, seeds[0], n_warm, crop=False)
                x, t, _, st = e2s.roll(x, t, seeds[0], n_timed, crop=False)
            if first_step_s is None:
                first_step_s = warm[0][1]
            step = [s[0] for s in st]
            med = statistics.median(step)
            rows.append(dict(batch=B, oom=False, step_s=T.summarize(step),
                             walker_steps_per_s=round(B / med, 4),
                             s_per_walker_sim_day=round(med / B * (24 / FW.STEP_H), 4),
                             finite=bool(torch.isfinite(x).all().item()), **_mem(torch)))
            print(f"[fcn3_bench]   B={B}: {med:.3f} s/step, {B / med:.3f} walker-steps/s, "
                  f"peak {rows[-1]['peak_alloc_gb']} GB", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            rows.append(dict(batch=B, oom=True, error=str(e).splitlines()[0][:300],
                             **_mem(torch)))
            print(f"[fcn3_bench]   B={B}: OOM - stopping the sweep", flush=True)
            break
        finally:
            del x, noise
            gc.collect(); torch.cuda.empty_cache()

    ok = [r for r in rows if not r["oom"]]
    b1 = next((r for r in ok if r["batch"] == 1), None)
    best = max(ok, key=lambda r: r["walker_steps_per_s"]) if ok else None
    gain = (best["walker_steps_per_s"] / b1["walker_steps_per_s"]) if (best and b1) else None
    rec = dict(config=config, phase=tag["phase"], worker=worker, kind="sweep",
               run_id=tag["run_id"], concurrency=tag["concurrency"], model="fcn3",
               path=path, step_h=FW.STEP_H, torch=torch.__version__,
               device=torch.cuda.get_device_name(0), disco_kernel=FW.disco_kernel_ok(),
               startup=dict(import_s=round(ph.t["import"], 2), load_s=round(ph.t["load"], 2),
                            first_step_s=round(first_step_s, 3) if first_step_s else None),
               warm_steps=n_warm, timed_steps=n_timed, rows=rows,
               best_batch=best["batch"] if best else None,
               best_gain_vs_b1=round(gain, 4) if gain else None,
               host=T.host_info())
    T.write_result(W.results_dir(run), rec)
    return 0 if ok and all(r["finite"] for r in ok) else 3


# --------------------------------------------------------------------------- #
# Layer 2: walker segments
# --------------------------------------------------------------------------- #
def segments(path: str, run: Path, worker: int, batch: int, n_seg: int, io: str,
             keep_states: bool) -> int:
    config = CONFIG[path]
    tag = T.env_tag()
    ph = T.Phases()
    torch, px, ic, t0 = _setup(ph)
    nat = FW.NativeFCN3(px) if path == "native" else None
    e2s = E2SRoller(px) if path == "e2s" else None
    lats = nat.lats if nat else e2s.lat
    lons = nat.lons if nat else e2s.lon
    lat_sl, lon_sl = FW.conus_slices(lats, lons)
    variables = list(px.variables)
    n_steps = W.CONFIGS[config].seg_steps
    walkers = [worker * batch + b for b in range(batch)]
    ext = ".npz" if io == "npz" else ".nc"
    sdir = W.scratch_dir(run) / f"{config}__{tag['phase']}__w{worker:02d}__b{batch}__{io}"
    if sdir.exists():
        shutil.rmtree(sdir)
    print(f"[fcn3_bench] segments {path}: worker {worker}, walkers {walkers}, "
          f"{n_seg} timed x {n_steps} steps, io={io}", flush=True)

    segs, prev = [], None
    t_valid = t0
    x_host = np.repeat(ic[None], batch, axis=0)
    noise_host = None
    for s in range(n_seg + 1):                                  # s = 0 warms the kernels
        sp = T.Phases()
        if prev is not None:
            with sp("read_state"):
                st = [FW.read_state(prev / f"w{w:04d}{ext}") for w in walkers]
            x_host = np.stack([r["x"] for r in st])
            noise_host = np.stack([r["noise"] for r in st]) if nat else None
            t_valid = st[0]["valid_time"]
        with sp("h2d"):
            x = torch.as_tensor(x_host).to("cuda")
            torch.cuda.synchronize()
        seeds = [FW.segment_seed(W.BASE_SEED, w, s) for w in walkers]
        with sp("roll"):
            if nat is not None:
                gens = nat.generators(seeds)
                noise = (nat.init_noise(gens) if noise_host is None
                         else torch.as_tensor(noise_host).to("cuda"))
                x, noise, t_end, crops, st_t = nat.roll(x, t_valid, noise, gens, n_steps)
            else:
                x, t_end, crops, st_t = e2s.roll(x, t_valid, seeds[0], n_steps)
                noise = None
        with sp("d2h_state"):
            x_host = x.float().cpu().numpy()
            noise_host = noise.float().cpu().numpy() if noise is not None else None
        here = sdir / f"seg{s:02d}"
        # earth2studio keeps its noise private, so its state file is the field alone
        # (noise_persisted=0) - the e2s walker's restart is not exact, and its file says so.
        with sp("write_state"):
            for b, w in enumerate(walkers):
                FW.write_state(here / f"w{w:04d}{ext}", x_host[b],
                               noise_host[b] if noise_host is not None else None,
                               t_end, variables, lats, lons,
                               attrs=dict(walker=w, step=s, path=path), fmt=io)
        crop_arr = np.stack(crops, axis=1)                       # (B, steps, 72, lc, lo)
        vt = [t_valid + pd.Timedelta(hours=FW.STEP_H * (k + 1)) for k in range(n_steps)]
        with sp("write_diag"):
            for b, w in enumerate(walkers):
                FW.write_diag(here / f"w{w:04d}_diag.nc", crop_arr[b], vt, variables,
                              lats[lat_sl], lons[lon_sl], attrs=dict(walker=w, step=s))
        t2m = x_host[:, variables.index("t2m")]
        sb = [(here / f"w{w:04d}{ext}").stat().st_size for w in walkers]
        db = [(here / f"w{w:04d}_diag.nc").stat().st_size for w in walkers]
        segs.append(dict(seg=s, warm=(s == 0), steps=n_steps, batch=batch,
                         phases=sp.as_dict(), wall_s=round(sum(sp.t.values()), 4),
                         step_compute_s=[round(a, 4) for a, _ in st_t],
                         step_total_s=[round(b, 4) for _, b in st_t],
                         state_bytes_per_walker=int(np.mean(sb)),
                         diag_bytes_per_walker=int(np.mean(db)),
                         finite=bool(np.isfinite(x_host).all()),
                         t2m_global_mean_k=round(float(t2m.mean()), 3),
                         valid_end=str(t_end)))
        r = segs[-1]
        print(f"[fcn3_bench] seg {s}{' (warm)' if s == 0 else ''}: wall {r['wall_s']:.1f} s "
              f"for {batch} walker(s) = {r['phases']}", flush=True)
        if prev is not None and not keep_states:
            shutil.rmtree(prev, ignore_errors=True)
        prev, t_valid = here, t_end
        del x, noise

    timed = [r for r in segs if not r["warm"]]
    comp = [c for r in timed for c in r["step_compute_s"]]
    roll_ps = [r["phases"]["roll"] / r["steps"] for r in timed]
    wall = [r["wall_s"] for r in timed]
    spd = 24 / FW.STEP_H
    rec = dict(
        config=config, phase=tag["phase"], worker=worker, kind="segments",
        run_id=tag["run_id"], concurrency=tag["concurrency"], model="fcn3", path=path,
        step_h=FW.STEP_H, seg_steps=n_steps, seg_days=W.SEG_DAYS, batch=batch,
        walkers=walkers, io=("npz-raw" if io == "npz" else "nc-zlib4"),
        noise_persisted=nat is not None, torch=torch.__version__,
        device=torch.cuda.get_device_name(0), disco_kernel=FW.disco_kernel_ok(),
        startup=dict(import_s=round(ph.t["import"], 2), load_s=round(ph.t["load"], 2),
                     first_step_s=round(segs[0]["step_total_s"][0], 3)),
        compute_s_per_step=T.summarize(comp),
        roll_s_per_step=T.summarize(roll_ps),
        segment_wall_s=T.summarize(wall),
        s_per_sim_day=dict(   # per WALKER: a batch step advances B walkers by 6 h
            compute=round(statistics.median(comp) / batch * spd, 3) if comp else None,
            roll=round(statistics.median(roll_ps) / batch * spd, 3) if roll_ps else None,
            segment=round(statistics.median(wall) / batch / W.SEG_DAYS, 3) if wall else None),
        all_finite=all(r["finite"] for r in segs), segments=segs, **_mem(torch),
        host=T.host_info())
    T.write_result(W.results_dir(run), dict(rec, kind=f"segments_b{batch}_{io}"))
    if not keep_states:
        shutil.rmtree(sdir, ignore_errors=True)
    return 0 if rec["all_finite"] else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", choices=sorted(CONFIG), required=True)
    ap.add_argument("--mode", choices=["sweep", "segments"], required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--batches", default=",".join(map(str, W.SWEEP_BATCHES)),
                    help="sweep: batch sizes, ascending; stops at the first OOM")
    ap.add_argument("--batch", type=int, default=1, help="segments: walkers per process")
    ap.add_argument("--segments", type=int, default=W.TIMED_SEGMENTS)
    ap.add_argument("--io", choices=["nc", "npz"], default="nc")
    ap.add_argument("--keep-states", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "sweep":
        return sweep(a.path, a.run, a.worker, [int(b) for b in a.batches.split(",")])
    return segments(a.path, a.run, a.worker, a.batch, a.segments, a.io, a.keep_states)


if __name__ == "__main__":
    sys.exit(main())
