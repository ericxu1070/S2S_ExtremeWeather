#!/usr/bin/env python
"""Can FCN3 actually be a walker? Pass/fail checks on the native walker, one GPU, ~5 min.

A speed number for a walker that cannot restart, cannot clone, or computes something other
than FCN3 is worthless, so these run alongside the benchmark and the report puts them
first. Each check returns PASS, WARN (works, with a caveat the report states) or FAIL.

    native_matches_e2s   at B=1 the native path is bit-identical to earth2studio, step by
                         step - the native speed numbers are FCN3's, not another model's
    restart_exact        rolling 2 segments in memory == rolling 1, writing the state +
                         noise to disk (netCDF and npz), reading it back, rolling the 2nd
    noise_renewal        informational: what restarting WITHOUT the noise state (the
                         earth2studio walker) does to the trajectory, against clone spread
    determinism          same seeds twice -> the same bits
    clone_divergence     two clones of one parent separate at step 1 and keep separating
    batch_invariance     a walker's trajectory does not depend on which batch it rode in
    state_integrity      a written state is float32, finite and physically plausible

Env: ``fcn3``.

    python -m wbench.checks --run runs/wbench/<id>
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from fcn3 import fevents as F

os.environ.setdefault("EARTH2STUDIO_CACHE", str(F.FCN3_CACHE / "e2s"))
os.environ.setdefault("HF_HOME", str(F.FCN3_CACHE / "hf"))

from wbench import fcn3_walker as FW                                   # noqa: E402
from wbench import timing as T                                         # noqa: E402
from wbench import wconfig as W                                        # noqa: E402

SEG = 3                       # steps per segment in the checks (18 h; enough to see it)
# Physical bounds for a 3-day FCN3 state; loose on purpose - a NaN or a 1000 K column is
# what they are for, not climatology.
BOUNDS = {"t2m": (170.0, 340.0), "z500": (4.4e4, 6.1e4), "tcwv": (0.0, 120.0),
          "msl": (8.5e4, 1.1e5)}


class Ctx:
    def __init__(self, px, ic, t0):
        import torch
        self.torch, self.px = torch, px
        self.nat = FW.NativeFCN3(px)
        self.ic, self.t0 = ic, t0
        self.vars = list(px.variables)
        self.it2m = self.vars.index("t2m")
        lat = np.deg2rad(self.nat.lats)
        self.w = torch.as_tensor(np.cos(lat)[:, None], dtype=torch.float64, device="cuda")

    def x0(self, B=1):
        return self.torch.as_tensor(self.ic).to("cuda")[None].repeat(B, 1, 1, 1)

    def seeds(self, walkers, step):
        return [FW.segment_seed(W.BASE_SEED, w, step) for w in walkers]

    def rms_t2m(self, a, b) -> float:
        """Area-weighted global RMS difference of T2m (K) between two (72, H, W) fields."""
        d = (a[self.it2m] - b[self.it2m]).double()
        return float(((d ** 2 * self.w).sum() / (self.w.sum() * d.shape[-1])).sqrt())

    def roll_native(self, x, t, noise, walkers, step, n=SEG):
        gens = self.nat.generators(self.seeds(walkers, step))
        if noise is None:
            noise = self.nat.init_noise(gens)
        x, noise, t, _, _ = self.nat.roll(x, t, noise, gens, n, crop=False)
        return x, noise, t


def _bitwise(a, b) -> bool:
    import torch
    return bool(torch.equal(a, b))


def _maxabs(a, b) -> float:
    return float((a.float() - b.float()).abs().max().item())


def check_native_matches_e2s(c: Ctx, n=4) -> dict:
    torch = c.torch
    seed = c.seeds([0], 0)[0]
    c.nat.release()
    coords = OrderedDict(ensemble=np.arange(1), time=np.array([np.datetime64(c.t0)]),
                         lead_time=np.array([np.timedelta64(0, "h")]),
                         variable=np.array(c.vars), lat=c.px.input_coords()["lat"],
                         lon=c.px.input_coords()["lon"])
    c.px.set_rng(seed=seed, reset=True)
    it = c.px.create_iterator(c.x0()[:, None, None].contiguous(), coords)
    next(it)
    e2s = [next(it)[0][0, 0, 0].clone() for _ in range(n)]
    c.nat.release()

    gens = c.nat.generators([seed])
    noise = c.nat.init_noise(gens)
    x, t, nat = c.x0(), pd.Timestamp(c.t0), []
    for _ in range(n):
        x, noise = c.nat.step(x, t, noise, gens)
        t += pd.Timedelta(hours=FW.STEP_H)
        nat.append(x[0].clone())
    exact = [_bitwise(a, b) for a, b in zip(e2s, nat)]
    maxabs = [_maxabs(a, b) for a, b in zip(e2s, nat)]
    rms = [c.rms_t2m(a, b) for a, b in zip(e2s, nat)]
    if all(exact):
        v = "PASS"
    elif rms[0] < 1e-3:
        v = "WARN"
    else:
        v = "FAIL"
    return dict(name="native_matches_e2s", verdict=v, steps=n, bitwise=exact,
                max_abs_diff=maxabs, t2m_rms_k=rms,
                note="WARN = same model to rounding (different kernel order), not bitwise")


def check_restart(c: Ctx, tmp: Path) -> list[dict]:
    """restart_exact (nc + npz), determinism and noise_renewal share one parent run."""
    torch = c.torch
    # A: two segments, in memory.
    x1, n1, t1 = c.roll_native(c.x0(), c.t0, None, [0], 0)
    xa, na, ta = c.roll_native(x1.clone(), t1, n1.clone(), [0], 1)
    out = []

    # Determinism: segment 1 again, from scratch.
    x1b, n1b, _ = c.roll_native(c.x0(), c.t0, None, [0], 0)
    det = _bitwise(x1, x1b) and _bitwise(n1, n1b)
    out.append(dict(name="determinism", verdict="PASS" if det else "FAIL",
                    bitwise=det, max_abs_diff=_maxabs(x1, x1b)))

    # B: segment 1 -> disk -> segment 2, once per format.
    for fmt in ("nc", "npz"):
        p = tmp / f"restart.{fmt}"
        t_w = time.perf_counter()
        FW.write_state(p, x1[0].cpu().numpy(), n1[0].cpu().numpy(), t1, c.vars,
                       c.nat.lats, c.nat.lons, fmt=fmt)
        r = FW.read_state(p)
        io_s = time.perf_counter() - t_w
        xr_ = torch.as_tensor(r["x"]).to("cuda")[None]
        nr_ = torch.as_tensor(r["noise"]).to("cuda")[None]
        roundtrip = _bitwise(xr_, x1) and _bitwise(nr_, n1)
        xb, nb, tb = c.roll_native(xr_, r["valid_time"], nr_, [0], 1)
        exact = _bitwise(xa, xb) and _bitwise(na, nb) and tb == ta
        out.append(dict(name=f"restart_exact_{fmt}", verdict="PASS" if exact else "FAIL",
                        roundtrip_bitwise=roundtrip, continuation_bitwise=exact,
                        max_abs_diff=_maxabs(xa, xb), t2m_rms_k=c.rms_t2m(xa[0], xb[0]),
                        state_bytes=p.stat().st_size, write_read_s=round(io_s, 2)))

    # C: segment 1's field, but noise re-drawn from scratch - the earth2studio restart.
    xc, _, _ = c.roll_native(x1.clone(), t1, None, [0], 1)
    renew = c.rms_t2m(xa[0], xc[0])
    out.append(dict(name="noise_renewal", verdict="INFO", t2m_rms_k_after_segment=renew,
                    note="restart without the persisted noise state vs with it, same seeds;"
                         " compare with clone_divergence at the same lead"))
    return out


def check_clones(c: Ctx, n=2 * SEG) -> dict:
    """Two clones of one parent, in one batch of 2, separate on their own."""
    torch = c.torch
    x1, n1, t1 = c.roll_native(c.x0(), c.t0, None, [0], 0)
    x = x1.repeat(2, 1, 1, 1)
    noise = n1.repeat(2, *([1] * (n1.ndim - 1)))
    gens = c.nat.generators(c.seeds([10, 11], 1))          # two NEW walker indices
    rms, t = [], t1
    for _ in range(n):
        x, noise = c.nat.step(x, t, noise, gens)
        t += pd.Timedelta(hours=FW.STEP_H)
        rms.append(c.rms_t2m(x[0], x[1]))
    grows = rms[0] > 0 and rms[-1] > rms[0]
    return dict(name="clone_divergence", verdict="PASS" if grows else "FAIL",
                t2m_rms_k_per_step=[round(r, 5) for r in rms], step_h=FW.STEP_H)


def check_batch_invariance(c: Ctx, clone_rms_at_seg: float | None) -> dict:
    torch = c.torch
    out = dict(name="batch_invariance")
    for B in (4, 2):
        try:
            xb, nb, _ = c.roll_native(c.x0(B), c.t0, None, list(range(B)), 0)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            xb = None
    if xb is None:
        return dict(out, verdict="SKIP", note="B=2 does not fit on this card")
    xs, ns, _ = c.roll_native(c.x0(1), c.t0, None, [0], 0)
    exact = _bitwise(xb[0:1], xs) and _bitwise(nb[0:1], ns)
    noise_exact = _bitwise(nb[0:1], ns)
    rms = c.rms_t2m(xb[0], xs[0])
    if exact:
        v = "PASS"
    elif noise_exact and clone_rms_at_seg and rms < 0.1 * clone_rms_at_seg:
        v = "WARN"
    else:
        v = "FAIL"
    return dict(out, verdict=v, batch=B, steps=SEG, bitwise=exact,
                noise_bitwise=noise_exact, t2m_rms_k=rms, max_abs_diff=_maxabs(xb[0:1], xs),
                clone_t2m_rms_k_same_lead=clone_rms_at_seg,
                note="WARN = the noise streams are batch-independent but batched kernels "
                     "round differently; negligible against clone spread")


def check_state_integrity(c: Ctx, tmp: Path) -> dict:
    x, n, t = c.roll_native(c.x0(), c.t0, None, [0], 0, n=12)       # one full 3-d segment
    p = tmp / "integrity.nc"
    FW.write_state(p, x[0].cpu().numpy(), n[0].cpu().numpy(), t, c.vars, c.nat.lats,
                   c.nat.lons)
    r = FW.read_state(p)
    problems = []
    for k in ("x", "noise"):
        if r[k].dtype != np.float32:
            problems.append(f"{k} is {r[k].dtype}")
        if not np.isfinite(r[k]).all():
            problems.append(f"{k} has non-finite values")
    ranges = {}
    for v, (lo, hi) in BOUNDS.items():
        if v not in r["variables"]:
            continue
        a = r["x"][r["variables"].index(v)]
        ranges[v] = [float(a.min()), float(a.max())]
        if a.min() < lo or a.max() > hi:
            problems.append(f"{v} outside [{lo}, {hi}]: {ranges[v]}")
    if r["valid_time"] != pd.Timestamp(c.t0) + pd.Timedelta(days=W.SEG_DAYS):
        problems.append(f"valid_time {r['valid_time']} != init + {W.SEG_DAYS} d")
    return dict(name="state_integrity", verdict="FAIL" if problems else "PASS",
                problems=problems, ranges=ranges, noise_shape=list(r["noise"].shape),
                state_bytes=p.stat().st_size)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--worker", type=int, default=0)
    a = ap.parse_args(argv)
    tag = T.env_tag()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("[checks] no CUDA device visible")
    FW.require_disco_kernel()
    t_start = time.perf_counter()
    px = FW.load_px("cuda")
    ic, t0 = FW.read_ic(W.fcn3_ic_path(), px.variables)
    c = Ctx(px, ic, t0)

    results = []

    def run(fn, *args):
        t = time.perf_counter()
        try:
            r = fn(*args)
        except Exception as e:                       # one broken check must not hide the rest
            import traceback
            traceback.print_exc()
            r = dict(name=fn.__name__.removeprefix("check_"), verdict="ERROR",
                     error=f"{type(e).__name__}: {e}"[:500])
        for one in (r if isinstance(r, list) else [r]):
            one["seconds"] = round(time.perf_counter() - t, 1)
            results.append(one)
            print(f"[checks] {one['name']}: {one['verdict']}", flush=True)
        return r

    sdir = W.scratch_dir(a.run)
    sdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=sdir) as td:
        tmp = Path(td)
        run(check_native_matches_e2s, c)
        run(check_restart, c, tmp)
        clones = run(check_clones, c)
        at_seg = (clones.get("t2m_rms_k_per_step") or [None] * SEG)[SEG - 1] \
            if isinstance(clones, dict) else None
        run(check_batch_invariance, c, at_seg)
        run(check_state_integrity, c, tmp)

    verdicts = [r["verdict"] for r in results]
    rec = dict(config="fcn3_native", phase=tag["phase"], worker=a.worker, kind="checks",
               run_id=tag["run_id"], device=torch.cuda.get_device_name(0),
               torch=torch.__version__, disco_kernel=FW.disco_kernel_ok(),
               checks=results, n_fail=sum(v in ("FAIL", "ERROR") for v in verdicts),
               n_warn=verdicts.count("WARN"),
               total_wall_s=round(time.perf_counter() - t_start, 1), host=T.host_info())
    T.write_result(W.results_dir(a.run), rec)
    return 0 if rec["n_fail"] == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
