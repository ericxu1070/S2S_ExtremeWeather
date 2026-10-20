"""FCN3 as an AI+RES walker: batched, restartable, with its AR(1) noise state on disk.

What a walker needs that the scorer never did
---------------------------------------------
The scorer rolls FCN3 from an IC to the peak and stops; nothing is ever restarted. A
walker is stopped every 3 d, written to disk, possibly cloned, and continued - so three
things the earth2studio path does not give have to be built here:

1. **Batching that batches.** ``earth2studio.models.px.fcn3.FCN3._forward`` loops over
   the batch in Python (``for j in batch: for i in time: self.model(x[j, i:i+1])``), so
   its ``batch_size`` is a loop count, not a tensor dimension. makani's network and its
   noise state both carry a real batch dimension; :class:`NativeFCN3` uses it.
2. **A noise state that survives a restart.** FCN3's stochasticity is makani's
   ``DiffusionNoiseS2``: an AR(1) process in spherical-harmonic space,
   ``state <- phi * state + eta``. earth2studio re-draws it from the stationary
   distribution whenever an iterator is created, so a walker restarted through it starts
   each segment with fresh, uncorrelated noise. Here the noise state is written next to
   the atmospheric state and read back, which is what makes a restart exact.
3. **Per-walker noise streams.** makani draws a whole batch's noise from ONE generator,
   so walker b's noise would depend on which batch it happened to share. Here every
   walker advances its own slice with its own ``torch.Generator``, seeded by
   ``segment_seed(base, walker, step)`` - the same (walker, step) folding the GenCast
   walker uses, so a clone (a new walker index pointed at a parent's files) draws noise
   no sibling has used, and a walker's trajectory does not depend on the batch.

The per-walker noise update goes through makani's own ``DiffusionNoiseS2.update`` (one
call per walker on a rebound 1-element state), and the forward pass mirrors
``makani.models.model_package.ModelPackage.forward`` line for line, inside the same
autocast earth2studio uses. ``wbench.checks`` pins the result: at B=1 it is bit-identical
to earth2studio.

Env: ``fcn3``. Imports torch lazily so the I/O helpers are usable from a CPU test.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fcn3 import fevents as F

STEP_H = F.STEP_H                                   # 6 h
NOISE_DIMS = ("noise_t", "noise_c", "noise_l", "noise_m", "noise_ri")


# --------------------------------------------------------------------------- #
# Seeds and model
# --------------------------------------------------------------------------- #
def segment_seed(base: int, walker: int, step: int) -> int:
    """A 63-bit seed unique per (base, walker, step); stable across machines and runs."""
    h = hashlib.blake2b(f"{int(base)}:{int(step)}:{int(walker)}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little") & (2 ** 63 - 1)


def disco_kernel_ok() -> bool:
    """Whether earth2studio will run FCN3 in bf16 on the DISCO CUDA kernel. Without it the
    forward silently drops to fp32 on a slow path (see ``fcn3/run_fcn3.py``), and every
    FCN3 number here would be the wrong number."""
    from earth2studio.models.px import fcn3 as m
    return bool(m._cuda_extension_available)


def require_disco_kernel() -> None:
    if disco_kernel_ok():
        return
    if os.environ.get("WB_ALLOW_FP32") == "1":
        print("[fcn3] WARNING: DISCO CUDA kernel missing; running fp32 by request "
              "(WB_ALLOW_FP32=1). These timings are NOT the production path.", flush=True)
        return
    raise SystemExit("[fcn3] torch-harmonics DISCO CUDA kernel missing -> FCN3 would run "
                     "fp32 on a slow path. Rebuild it (wbench/vm/setup_vm.sh step "
                     "'torch-harmonics') or set WB_ALLOW_FP32=1 to time the slow path.")


def load_px(device: str = "cuda"):
    """The earth2studio FCN3 model, built at most once per machine (``fcn3.model_cache``)."""
    from fcn3.model_cache import get_model
    px = get_model(F.model_cache_dir(),
                   disabled=os.environ.get("FCN3_MODEL_CACHE", "1") == "0")
    return px.to(device).eval()


def read_ic(path: Path, variables) -> tuple[np.ndarray, pd.Timestamp]:
    """(72, 721, 1440) float32 in the model's channel order, and its valid time."""
    import xarray as xr
    da = xr.open_dataset(path)["ic"].load()
    if da.sizes.get("time") != 1:
        raise ValueError(f"{path}: want one IC frame, got time={da.sizes.get('time')}")
    lat = da["lat"].values
    if not (lat[0] == 90.0 and lat[-1] == -90.0):
        raise ValueError(f"{path}: FCN3 lat must descend 90 -> -90, got {lat[0]}..{lat[-1]}")
    x = da.isel(time=0).sel(variable=list(variables)).values.astype(np.float32)
    return x, pd.Timestamp(da["time"].values[0])


def conus_slices(lats: np.ndarray, lons: np.ndarray) -> tuple[slice, slice]:
    """Index slices of the CONUS box on FCN3's grid (lat descending)."""
    li = np.where((lats >= F.LAT_LO) & (lats <= F.LAT_HI))[0]
    lo = np.where((lons >= F.LON_LO) & (lons <= F.LON_HI))[0]
    return slice(int(li[0]), int(li[-1]) + 1), slice(int(lo[0]), int(lo[-1]) + 1)


# --------------------------------------------------------------------------- #
# The batched walker
# --------------------------------------------------------------------------- #
class NativeFCN3:
    """B FCN3 walkers stepped in one forward pass, each with its own noise stream."""

    def __init__(self, px):
        import torch
        from earth2studio.models.px import fcn3 as e2s_fcn3

        self.torch = torch
        self.px = px
        self.pkg = px.model                          # makani ModelPackage
        self.net = self.pkg.model                    # Single/MultiStepWrapper
        self.pre = self.net.preprocessor
        self.noise = getattr(self.pre, "input_noise", None)
        if self.noise is None or not self.noise.is_stateful():
            raise RuntimeError("FCN3 has no stateful input noise - the walker contract "
                               "(persisted AR(1) noise) does not apply to this checkpoint")
        if not getattr(self.pkg, "add_zenith", False):
            raise RuntimeError("expected FCN3 to condition on the zenith angle")
        if int(self.pkg.timestep) != STEP_H:
            raise RuntimeError(f"model step {self.pkg.timestep} h != {STEP_H} h")
        self.device = self.pkg.in_bias.device
        self.amp_dtype = (torch.bfloat16 if e2s_fcn3._cuda_extension_available
                          else torch.float32)
        self.variables = list(px.variables)
        self.lats = np.asarray(self.pkg.lats)
        self.lons = np.asarray(self.pkg.lons)
        self.lat_sl, self.lon_sl = conus_slices(self.lats, self.lons)
        self.noise_shape = tuple(self.noise.state.shape[1:])
        # makani draws from rng_gpu when the state is on CUDA and rng_cpu otherwise, so
        # the per-walker generator has to replace whichever one this device uses.
        self._gen_attr = "rng_gpu" if self.device.type == "cuda" else "rng_cpu"
        self._orig_gen = getattr(self.noise, self._gen_attr, None)

    # -- noise ------------------------------------------------------------- #
    def generators(self, seeds) -> list:
        return [self.torch.Generator(device=self.device).manual_seed(int(s)) for s in seeds]

    def _bind(self, state, gen) -> None:
        # ``state`` is a registered buffer: assigning a tensor rebinds the buffer (nn.Module
        # __setattr__), which is how makani's own update() swaps a resized state in.
        self.noise.state = state
        if gen is not None:
            setattr(self.noise, self._gen_attr, gen)

    def init_noise(self, gens):
        """A fresh stationary noise state per walker - earth2studio's per-member draw."""
        torch = self.torch
        with torch.inference_mode():
            out = []
            for g in gens:
                self._bind(torch.zeros((1,) + self.noise_shape, dtype=torch.float32,
                                       device=self.device), g)
                self.noise.update(replace_state=True)
                out.append(self.noise.state.clone())
            return torch.cat(out, dim=0)

    def advance_noise(self, noise, gens):
        """One AR(1) step per walker, each from its own generator."""
        torch = self.torch
        with torch.inference_mode():
            new = torch.empty_like(noise)
            for b, g in enumerate(gens):
                self._bind(noise[b:b + 1].clone(), g)
                self.noise.update(replace_state=False)
                new[b:b + 1].copy_(self.noise.state)
            return new

    def release(self) -> None:
        """Hand the module back as a batch-1, earth2studio-usable model."""
        self._bind(self.torch.zeros((1,) + self.noise_shape, dtype=self.torch.float32,
                                    device=self.device), self._orig_gen)

    # -- forward ----------------------------------------------------------- #
    def _zenith(self, valid_time, batch: int):
        from makani.third_party.climt.zenith_angle import cos_zenith_angle
        # Exactly the timestamp earth2studio hands makani: naive UTC -> "+00:00".
        t = [datetime.fromisoformat(pd.Timestamp(valid_time).to_pydatetime().isoformat()
                                    + "+00:00")]
        cosz = cos_zenith_angle(t, self.pkg.lon_grid, self.pkg.lat_grid).astype(np.float32)
        z = self.torch.as_tensor(cosz).to(device=self.device)
        while z.ndim != 4:
            z = z[None]
        # A real copy, not expand(): makani copy_()s into the cached tensor next step,
        # and copy_ into an expanded (aliased) tensor is an error.
        return z.repeat(batch, 1, 1, 1) if batch > 1 else z

    def step(self, x, valid_time, noise, gens):
        """One 6 h step for B walkers. ``x`` (B, 72, 721, 1440) float32 valid at
        ``valid_time``; returns (x at valid_time + 6 h, advanced noise)."""
        torch = self.torch
        noise = self.advance_noise(noise, gens)
        with torch.inference_mode():
            self._bind(noise, None)                     # the batched state the net reads
            z = self._zenith(valid_time, x.shape[0])
            # ModelPackage.forward, inside earth2studio's autocast - the net is called
            # with update_state=False because the noise was advanced per walker above.
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                xn = (x - self.pkg.in_bias) / self.pkg.in_scale
                self.pre.cache_unpredicted_features(None, None, xz=z, yz=None)
                yn = self.net(xn, update_state=False, replace_state=False)
                y = yn * self.pkg.out_scale + self.pkg.out_bias
            return y.float(), noise

    def roll(self, x, t0, noise, gens, n_steps: int, crop: bool = True):
        """``n_steps`` steps. Returns (x, noise, valid_time, crops, step_times).

        ``crops`` is a list of host (B, 72, nlat_c, nlon_c) arrays, one per step - the
        CONUS diagnostics a walker writes. ``step_times`` are (compute_s, total_s) per
        step: compute is synchronised before the crop's device->host copy, total after.
        """
        import time
        torch = self.torch
        t = pd.Timestamp(t0)
        crops, times = [], []
        for _ in range(n_steps):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            x, noise = self.step(x, t, noise, gens)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            t = t + pd.Timedelta(hours=STEP_H)
            if crop:
                crops.append(x[:, :, self.lat_sl, self.lon_sl].float().cpu().numpy())
            times.append((t2 - t1, time.perf_counter() - t1))
        return x, noise, t, crops, times


# --------------------------------------------------------------------------- #
# State and diagnostics I/O
# --------------------------------------------------------------------------- #
def _atomic(path: Path, write) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write_state(path: Path, x: np.ndarray, noise: np.ndarray | None, valid_time,
                variables, lats, lons, attrs: dict | None = None, fmt: str = "nc") -> Path:
    """One walker's restart state: the (72, H, W) field AND its (noise_t, c, l, m, 2)
    noise state, float32 always (float16 would overflow geopotential, as the GenCast
    walker found). ``nc`` is netCDF4 + zlib level 4, the GenCast walker's format; ``npz``
    is raw, the fast alternative the benchmark prices next to it.

    ``noise=None`` writes the field alone and records ``noise_persisted=0``: that is the
    earth2studio walker, whose noise state is private and re-drawn every segment, so a
    reader can tell an exact-restart state from one that is not."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"state {x.shape}: want (variable, lat, lon)")
    if noise is not None:
        noise = np.asarray(noise, dtype=np.float32)
        if noise.ndim != len(NOISE_DIMS):
            raise ValueError(f"noise {noise.shape}: want {len(NOISE_DIMS)}-D {NOISE_DIMS}")
    meta = dict(attrs or {}, valid_time=str(pd.Timestamp(valid_time)), step_h=STEP_H,
                model="FourCastNet3", noise_persisted=int(noise is not None))
    if fmt == "npz":
        extra = {} if noise is None else {"noise": noise}

        def w(tmp):
            with open(tmp, "wb") as f:
                np.savez(f, x=x, variables=np.asarray(list(variables)),
                         lat=np.asarray(lats), lon=np.asarray(lons),
                         meta=np.asarray(json.dumps(meta)), **extra)
        return _atomic(path, w)
    if fmt != "nc":
        raise ValueError(f"fmt must be 'nc' or 'npz', got {fmt!r}")
    import xarray as xr
    data = {"state": (("variable", "lat", "lon"), x)}
    if noise is not None:
        data["noise"] = (NOISE_DIMS, noise)
    ds = xr.Dataset(
        data,
        coords={"variable": np.asarray(list(variables)), "lat": np.asarray(lats),
                "lon": np.asarray(lons)},
        attrs={k: (json.dumps(v) if isinstance(v, (dict, list, tuple)) else v)
               for k, v in meta.items()})
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    return _atomic(path, lambda tmp: ds.to_netcdf(tmp, encoding=enc))


def read_state(path: Path) -> dict:
    """Inverse of :func:`write_state`; format from the suffix."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            out = dict(x=z["x"], noise=z["noise"] if "noise" in z.files else None,
                       variables=[str(v) for v in z["variables"]],
                       lat=z["lat"], lon=z["lon"], attrs=meta)
    else:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        out = dict(x=ds["state"].values,
                   noise=ds["noise"].values if "noise" in ds else None,
                   variables=[str(v) for v in ds["variable"].values],
                   lat=ds["lat"].values, lon=ds["lon"].values, attrs=dict(ds.attrs))
    out["valid_time"] = pd.Timestamp(out["attrs"]["valid_time"])
    for k in ("x", "noise"):
        if out[k] is not None and out[k].dtype != np.float32:
            raise ValueError(f"{path}: {k} is {out[k].dtype}, a walker state is float32")
    return out


def write_diag(path: Path, crops: np.ndarray, times, variables, lats, lons,
               attrs: dict | None = None) -> Path:
    """(steps, 72, nlat_c, nlon_c) CONUS crops for one walker-segment, zlib netCDF."""
    import xarray as xr
    ds = xr.Dataset({"diag": (("time", "variable", "lat", "lon"),
                              np.asarray(crops, dtype=np.float32))},
                    coords={"time": pd.DatetimeIndex(times).values,
                            "variable": np.asarray(list(variables)),
                            "lat": np.asarray(lats), "lon": np.asarray(lons)},
                    attrs=dict(attrs or {}))
    return _atomic(path, lambda tmp: ds.to_netcdf(
        tmp, encoding={"diag": {"zlib": True, "complevel": 4}}))
