#!/usr/bin/env python
"""Layers 1+2 for the GenCast walker: per-step compute, per-step walker cost, per-segment cost.

One process, one GPU, one walker. It rolls a warm segment (pays the XLA compile) and then
``--segments`` timed ones, each exactly what a production leg does per walker:

    read the parent state  ->  roll 3 d (6 x 12 h)  ->  write the restart state  ->  write
    the CONUS diagnostics

composed from ``aires.walker``'s own functions in ``run_segment``'s order - ``read_state``,
``roll_segment``, ``write_state``, ``_host_materialize`` + ``_write_nc`` - so each phase
can be timed without a second implementation of any of them. The predictor is wrapped to
record its own wall time per call (``block_until_ready``), which splits a step into model
compute and the rollout's host-side work (the global frame's device->host copy, the CONUS
crop, xarray bookkeeping). That split is what tells you whether a faster GPU would help.

Env: the GenCast env. ``wbench/vm/env.sh gencast`` sets the XLA knobs; the serial/bf16
switches ``gencast_s2s.model.load_gencast`` reads are set HERE from the config, so the
precision a result claims is the precision that ran.

    python -m wbench.gencast_bench --config gencast_0p25 --run runs/wbench/<id> --worker 0
    python -m wbench.gencast_bench --config gencast_1p0 --check      # login node, no GPU
"""
from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wbench import timing as T
from wbench import wconfig as W


def _configure(config: str) -> W.ModelConfig:
    """Set the switches ``load_gencast`` reads, and widen the grid check for 1.0 deg.

    ``aires.walker.check_state`` asserts ``aconfig.STATE_NLAT/NLON`` (the 0.25 deg grid)
    on every read and write. The 1.0 deg config is the one place a walker state is
    legitimately 181x360, so it rebinds those two module attributes for THIS process;
    every other assertion in ``check_state`` (2 frames, 12 h apart, all 14 variables)
    still applies unchanged.
    """
    if config not in W.GENCAST_RES:
        raise SystemExit(f"--config must be one of {tuple(W.GENCAST_RES)}, got {config!r}")
    cfg = W.CONFIGS[config]
    os.environ["XRES_SERIAL_INFER"] = "1"       # one member per process, no pmap
    os.environ["XRES_INFER_GPUS"] = "1"
    os.environ["XRES_BF16"] = "1" if cfg.precision == "bf16" else "0"
    os.environ["AIRES_RES"] = W.GENCAST_RES[config]
    if config == "gencast_1p0":
        from aires import aconfig as A
        A.STATE_NLAT, A.STATE_NLON = W.GRID_1P0
    return cfg


def _block(out):
    import jax
    try:
        return jax.block_until_ready(out)
    except Exception:                           # a container jax cannot flatten
        return jax.block_until_ready(jax.tree_util.tree_leaves(out))


def _timed(fn, sink: list):
    """The predictor, recording its own wall time per call.

    Blocking here costs no overlap: the rollout's next step consumes this step's output,
    and ``roll_segment`` copies every frame to host straight after the yield anyway."""
    def wrapped(*a, **kw):
        t0 = time.perf_counter()
        out = _block(fn(*a, **kw))
        sink.append(time.perf_counter() - t0)
        return out
    return wrapped


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def check(config: str) -> int:
    """Login-node check: the init state reads under this config's grid and a segment's
    batch builds - no model, no GPU."""
    cfg = _configure(config)
    from aires import walker as Wk
    p = W.gencast_init_path(W.GENCAST_RES[config])
    state = Wk.read_state(p)
    print(f"[check] {config}: {p.name} {state.sizes['lat']}x{state.sizes['lon']}, valid "
          f"{Wk.valid_time(state)}")
    Wk.eval_arrays({"task_config": Wk._task_config_only(W.GENCAST_RES[config])}, state,
                   cfg.seg_steps)
    print(f"[check] OK - a {cfg.seg_steps}-step segment builds for {config}")
    return 0


def run(config: str, run: Path, worker: int, segments: int, keep_states: bool) -> int:
    cfg = _configure(config)
    tag = T.env_tag()
    t_start = time.perf_counter()
    ph = T.Phases()
    with ph("import"):
        import jax
        from aires import walker as Wk
    dev = jax.devices()[0]
    print(f"[gencast_bench] {config} worker {worker} on {dev.device_kind} "
          f"(jax {jax.__version__}), {segments} timed segment(s) of {cfg.seg_steps} steps",
          flush=True)

    with ph("load"):
        bundle = Wk.load_bundle(n_members=1, res=W.GENCAST_RES[config])
    fwd: list[float] = []
    bundle = dict(bundle, forward=_timed(bundle["forward"], fwd))

    init = Wk.read_state(W.gencast_init_path(W.GENCAST_RES[config]))
    sdir = W.scratch_dir(run) / f"{config}__{tag['phase']}__w{worker:02d}"
    if sdir.exists():
        shutil.rmtree(sdir)

    segs, prev, state = [], None, init
    for s in range(segments + 1):                      # s = 0 is the warm (compile) one
        sp = T.Phases()
        if prev is not None:
            with sp("read_state"):
                state = Wk.read_state(prev / "state.nc")
        i0 = len(fwd)
        key = Wk.segment_key(worker, s, W.BASE_SEED)
        with sp("roll"):
            nxt, cube = Wk.roll_segment(bundle, state, cfg.seg_steps, key,
                                        label=f" w{worker:02d} s{s}")
        here = sdir / f"seg{s:02d}"
        with sp("write_state"):
            Wk.write_state(nxt, here / "state.nc")
        with sp("write_diag"):
            Wk._write_nc(Wk._host_materialize(cube), here / "diag.nc")
        t2m = np.asarray(nxt["2m_temperature"].values)
        segs.append(dict(
            seg=s, warm=(s == 0), steps=cfg.seg_steps, phases=sp.as_dict(),
            wall_s=round(sum(sp.t.values()), 4),
            forward_s=[round(x, 4) for x in fwd[i0:]],
            state_bytes=(here / "state.nc").stat().st_size,
            diag_bytes=(here / "diag.nc").stat().st_size,
            finite=bool(np.isfinite(t2m).all()),
            t2m_global_mean_k=round(float(np.nanmean(t2m[..., -1, :, :])), 3),
            valid_end=str(Wk.valid_time(nxt))))
        r = segs[-1]
        print(f"[gencast_bench] seg {s}{' (warm)' if s == 0 else ''}: wall {r['wall_s']:.1f} s "
              f"= {r['phases']}; forward median {statistics.median(r['forward_s']):.2f} s",
              flush=True)
        if prev is not None and not keep_states:
            shutil.rmtree(prev, ignore_errors=True)
        prev, state = here, None
        del nxt, cube

    st = dev.memory_stats() or {}
    timed = [r for r in segs if not r["warm"]]
    fwd_timed = [x for r in timed for x in r["forward_s"]]
    roll_per_step = [r["phases"]["roll"] / r["steps"] for r in timed]
    seg_wall = [r["wall_s"] for r in timed]
    steady = statistics.median(fwd_timed) if fwd_timed else float("nan")
    rec = dict(
        config=config, phase=tag["phase"], worker=worker, kind="segments",
        run_id=tag["run_id"], concurrency=tag["concurrency"], model="gencast",
        res=W.GENCAST_RES[config], precision=cfg.precision, step_h=cfg.step_h,
        seg_steps=cfg.seg_steps, seg_days=W.SEG_DAYS, batch=1,
        io="nc-zlib4", jax=jax.__version__, device=dev.device_kind,
        startup=dict(import_s=round(ph.t["import"], 2), load_s=round(ph.t["load"], 2),
                     first_forward_s=round(fwd[0], 2),
                     compile_s=round(fwd[0] - steady, 2)),
        peak_gpu_gb=(round(st["peak_bytes_in_use"] / 1e9, 2)
                     if "peak_bytes_in_use" in st else None),
        gpu_limit_gb=(round(st["bytes_limit"] / 1e9, 2) if "bytes_limit" in st else None),
        compute_s_per_step=T.summarize(fwd_timed),
        roll_s_per_step=T.summarize(roll_per_step),
        segment_wall_s=T.summarize(seg_wall),
        s_per_sim_day=dict(
            compute=round(steady * cfg.steps_per_day, 3),
            roll=round(statistics.median(roll_per_step) * cfg.steps_per_day, 3)
            if roll_per_step else None,
            segment=round(statistics.median(seg_wall) / W.SEG_DAYS, 3)
            if seg_wall else None),
        all_finite=all(r["finite"] for r in segs),
        segments=segs, scratch_bytes=_dir_bytes(sdir),
        total_wall_s=round(time.perf_counter() - t_start, 2), host=T.host_info())
    T.write_result(W.results_dir(run), rec)
    if not keep_states:
        shutil.rmtree(sdir, ignore_errors=True)
    return 0 if rec["all_finite"] else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=sorted(W.GENCAST_RES))
    ap.add_argument("--run", type=Path, help="run directory (runs/wbench/<id>)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--segments", type=int, default=W.TIMED_SEGMENTS)
    ap.add_argument("--keep-states", action="store_true")
    ap.add_argument("--check", action="store_true", help="login node: build a batch, no GPU")
    a = ap.parse_args(argv)
    if a.check:
        return check(a.config)
    if a.run is None:
        ap.error("--run is required unless --check")
    return run(a.config, a.run, a.worker, a.segments, a.keep_states)


if __name__ == "__main__":
    sys.exit(main())
