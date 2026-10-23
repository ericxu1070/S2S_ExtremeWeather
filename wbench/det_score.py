#!/usr/bin/env python
"""Deterministic AI+RES scorers: Pangu24 and SFNO on the 16 Gate 3 walkers.

The question (aires/HANDOFF.md "What M buys"): would ONE deterministic forecast per walker
score as well as FCN3's M-member mean, at one member's cost or less? `aires/score_noise.py`
says what that is worth in direct-sampling members - IF the deterministic model tracks the
conditional mean as well as FCN3's ensemble mean does. This measures it, on the same 64
(walker, t_k) cases Gate 3 scored, and measures each model's cost on an H100.

It also pins the one number `score_noise.py` cannot: `frac`, how much of FCN3's score
error is model error. One model's rho curve cannot (it brings its own unknown error), but
the cross-correlation of two scorers can, if their errors are independent:

    v_k = rho_A * rho_B / corr(theta_A, theta_B)      (v = Var(m_k), the predictability)

FCN3 x Pangu, FCN3 x SFNO and Pangu x SFNO give three estimates of the same v - a check of
the independence assumption: an error two scorers share raises their cross-correlation
and lowers their skill, so it biases that pair's v DOWN (v^2 / (v + shared variance)).

Stages (CPU work never runs on the GPU box):

    PYTHONPATH=. python -m wbench.det_score --stage sp-check        # Derecho CPU: sp vs ERA5
    PYTHONPATH=. python -m wbench.det_score --stage ic              # Derecho CPU: 64 ICs
    PYTHONPATH=. python -m wbench.det_score --stage infer --model pangu24 --dry   # no weights
    PYTHONPATH=. python -m wbench.det_score --stage infer --model sfno --shard 0 --nshards 8
    PYTHONPATH=. python -m wbench.det_score --stage reduce          # GenCast env, CPU

Inputs per case are FCN3's adapter IC (aires/adapter.py: 69 bit-identical channels plus
calibrated u100m/v100m/tcwv) plus surface pressure, which SFNO needs and GenCast does not
carry: `sp` is derived hydrostatically from msl, t2m and the surface geopotential, and
`--stage sp-check` measures that against ERA5. Pangu24 needs only the bit-identical
channels. Pangu24 steps 24 h, so its 7-day mean is 7 daily 00Z frames; aindex.check_window
infers the step from the cube, so that is checked, not assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from fcn3 import fevents as F

os.environ.setdefault("EARTH2STUDIO_CACHE", str(F.FCN3_CACHE / "e2s"))
os.environ.setdefault("HF_HOME", str(F.FCN3_CACHE / "hf"))

EVENT = "PNW_HeatDome_2021"
N_WALKERS = 16
LEADS = (6.0, 9.0, 12.0, 15.0)        # Gate 3's t_k with C_k > 0 (3 d buys no score)
# e2s class, step hours, e2s module (whose VARIABLES list is the input contract)
MODELS = {"pangu24": ("Pangu24", 24, "pangu"), "sfno": ("SFNO", 6, "sfno")}
IC_VARS = A.FCN3_VARIABLES + ("sp",)

R_D, G, LAPSE = 287.05, 9.80665, 0.0065   # J/kg/K, m/s2, K/m (ICAO standard lapse)
FCN3_S_PER_SIMDAY = 3.94                  # one FCN3 member, H100 (wbench small1)


def det_dir(event: str) -> Path:
    return A.event_dir(event) / "det"


def ic_path(event: str, walker: int, lead: float) -> Path:
    return det_dir(event) / "ic" / f"w{walker:02d}_lead{lead:02.0f}_ic.nc"


def cube_path(event: str, model: str, walker: int, lead: float) -> Path:
    return det_dir(event) / model / f"w{walker:02d}_lead{lead:02.0f}_cube.nc"


def members_csv(event: str) -> Path:
    return det_dir(event) / "fcn3_members.csv"


def cases(walkers: int = N_WALKERS):
    """Lead-major, so ``i % nshards`` hands every shard every lead: 4 leads and 8 shards
    are not coprime, and walker-major order would give one GPU all the 15 d forecasts."""
    return [(w, l) for l in LEADS for w in range(walkers)]


def derive_sp(msl, t2m, zs, z_lev=None, t_lev=None, p_lev_hpa=None):
    """Surface pressure, hydrostatically. Units: Pa, K, m2 s-2; levels in hPa.

    From msl alone: sp = msl * exp(-zs / (R_d Tbar)), Tbar = t2m warmed by the standard
    lapse rate over half the surface height. Exact over the ocean, but msl over high ground
    is itself a fiction (ERA5's reduction is not this one), and the error grows with height:
    14 hPa RMSE above 1500 m against ERA5 (`--stage sp-check`).

    With the pressure levels (z_lev, t_lev: (level, ...)), integrate down instead from the
    lowest level that is still above the ground, through at most one layer, with that
    level's own temperature lapsed halfway down: sp = p_a exp((Z_a - zs) / (R_d Tbar)).
    Columns whose ground is below the 1000 hPa surface keep the msl form, where it is good.
    """
    tbar = t2m + LAPSE * (zs / G) / 2.0
    sp = msl * np.exp(-zs / (R_D * tbar))
    if z_lev is None:
        return sp
    p = np.asarray(p_lev_hpa, dtype="float64")
    order = np.argsort(-p)                              # 1000 hPa first, upward
    p, z, t = p[order] * 100.0, np.asarray(z_lev)[order], np.asarray(t_lev)[order]
    above = z > zs[None]
    k = np.argmax(above, axis=0)                        # lowest level above the ground
    za = np.take_along_axis(z, k[None], 0)[0]
    ta = np.take_along_axis(t, k[None], 0)[0]
    tbar_l = ta + LAPSE * ((za - zs) / G) / 2.0
    sp_lev = p[k] * np.exp((za - zs) / (R_D * tbar_l))
    use = (k > 0) & above.any(axis=0)                  # ground above the 1000 hPa surface
    return np.where(use, sp_lev, sp)


class ICFile:
    """earth2studio DataSource over one cached IC (the same Protocol GenCastICSource uses)."""

    def __init__(self, path: Path):
        # eager: a lazy read would land inside the timed deterministic() call (~22 s per IC)
        self.da = xr.open_dataarray(path).load()

    def __call__(self, time, variable) -> xr.DataArray:
        t = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        return self.da.sel(time=t, variable=list(np.atleast_1d(variable)))


# --------------------------------------------------------------------------- #
# sp-check (Derecho CPU): the derivation against ERA5 itself
# --------------------------------------------------------------------------- #
RDA = Path("/glade/campaign/collections/rda/data/d633000")


def stage_sp_check(when: str = "2021-06-13T00") -> int:
    t = pd.Timestamp(when)
    sfc = RDA / "e5.oper.an.sfc" / t.strftime("%Y%m")
    span = f"{t:%Y%m}0100_{t:%Y%m}{t.days_in_month:02d}23"
    rd = lambda code, var: xr.open_dataset(sfc / f"e5.oper.an.sfc.{code}.ll025sc.{span}.nc")[var].sel(time=t)
    sp, msl, t2m = rd("128_134_sp", "SP"), rd("128_151_msl", "MSL"), rd("128_167_2t", "VAR_2T")
    zs = xr.open_dataset(RDA / "e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc."
                         "1979010100_1979010100.nc")["Z"].isel(time=0)
    pl = RDA / "e5.oper.an.pl" / t.strftime("%Y%m")
    day = f"{t:%Y%m%d}00_{t:%Y%m%d}23"
    rdp = lambda code, var: xr.open_dataset(pl / f"e5.oper.an.pl.{code}.ll025sc.{day}.nc")[var] \
        .sel(time=t, level=list(A.P13)).values
    z_lev, t_lev = rdp("128_129_z", "Z"), rdp("128_130_t", "T")
    w = np.cos(np.deg2rad(sp["latitude"].values))[:, None] * np.ones(zs.shape)
    land = zs.values > 1.0
    high = zs.values / G > 1500
    out = dict(when=str(t))
    for name, est in (("msl", derive_sp(msl.values, t2m.values, zs.values)),
                      ("levels", derive_sp(msl.values, t2m.values, zs.values, z_lev, t_lev, A.P13))):
        err = (est - sp.values) / 100.0                                   # hPa
        rms = lambda m: float(np.sqrt(np.sum(w[m] * err[m] ** 2) / np.sum(w[m])))
        out[name] = dict(rmse_hpa=rms(np.ones_like(land)), rmse_land_hpa=rms(land),
                         bias_land_hpa=float(np.sum(w[land] * err[land]) / np.sum(w[land])),
                         p99_abs_land_hpa=float(np.percentile(np.abs(err[land]), 99)),
                         max_abs_hpa=float(np.abs(err).max()), rmse_above_1500m_hpa=rms(high))
    print(json.dumps(out, indent=2))
    p = det_dir(EVENT) / "sp_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"  wrote {p}")
    return 0


# --------------------------------------------------------------------------- #
# ic (Derecho CPU): adapter IC + sp, and FCN3's per-member Gate 3 values
# --------------------------------------------------------------------------- #
def stage_ic(event: str) -> int:
    from aires import adapter as AD
    from aires import gate3 as G3
    from aires import gate3_msub as MS
    from aires import walker as W

    ev = G3._event(event)
    calib = AD.require_calibration()
    for w, l in cases():
        out = ic_path(event, w, l)
        if out.exists():
            continue
        state = W.read_state(A.state_path(event, w, A.segment_for_lead(l)))
        ic = AD.gencast_to_fcn3(state, calib=calib)
        zs = AD._orient(state["geopotential_at_surface"]).values
        st = state.isel(time=-1) if "time" in state["geopotential"].dims else state
        st = st.isel(batch=0) if "batch" in st.dims else st
        z_lev = AD._orient(st["geopotential"]).transpose("level", "lat", "lon")
        t_lev = AD._orient(st["temperature"]).transpose("level", "lat", "lon")
        sp = derive_sp(ic.sel(variable="msl").values, ic.sel(variable="t2m").values, zs,
                       z_lev.values, t_lev.values, z_lev["level"].values)
        state.close()
        full = xr.concat([ic, xr.DataArray(sp[:, None], dims=("time", "variable", "lat", "lon"),
                                           coords={"time": ic["time"], "variable": ["sp"],
                                                   "lat": ic["lat"], "lon": ic["lon"]})],
                         dim="variable").astype("float32")
        full.name = "ic"
        full.attrs.update(ic.attrs, sp="derived: wbench.det_score.derive_sp", walker=w,
                          score_lead_days=l, event=event)
        if not np.isfinite(full.values).all():
            raise ValueError(f"w{w:02d}@{l:g}d: non-finite IC channels")
        assert tuple(full["variable"].values) == IC_VARS
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".nc.tmp")
        full.to_netcdf(tmp, encoding={"ic": {"zlib": True, "complevel": 4}})
        os.replace(tmp, out)
        print(f"[ic] w{w:02d}@{l:g}d valid {pd.Timestamp(full.time.values[0])} "
              f"-> {out.name} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)
    res = json.loads(G3.result_path(ev).read_text())
    if not members_csv(event).exists():
        MS.members(ev, res).to_csv(members_csv(event), index=False)
        print(f"[ic] wrote {members_csv(event)}")
    return 0


# --------------------------------------------------------------------------- #
# infer (GPU, fcn3 env + the pangu/sfno extras)
# --------------------------------------------------------------------------- #
def _gpu_mem_gb() -> float:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                            "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                           capture_output=True, text=True, timeout=30)
        return float(q.stdout.strip().splitlines()[0]) / 1024
    except Exception:          # telemetry only; never fail a forecast over it
        return float("nan")


def nsteps(peak, valid, step_h: int) -> int:
    n, rem = divmod(int((pd.Timestamp(peak) - pd.Timestamp(valid)).total_seconds()), step_h * 3600)
    if rem or n <= 0:
        raise ValueError(f"{valid} is not a whole number of {step_h} h steps before {peak}")
    return n


def assemble(zarr: Path, out: Path, *, init, model: str, walker: int, lead: float,
             event: str, step_h: int) -> None:
    from fcn3.run_fcn3 import _crop_conus

    ds = xr.open_zarr(zarr)
    da = ds["t2m"].isel(time=0).drop_vars("time", errors="ignore")
    valid = np.datetime64(pd.Timestamp(init)) + da["lead_time"].values
    da = da.rename({"lead_time": "time"}).assign_coords(time=("time", valid))
    da = _crop_conus(da.expand_dims(member=[0]))
    gname, _ = F.FCN3_TO_GENCAST["t2m"]
    cube = xr.Dataset({gname: da})
    cube.attrs.update(event=event, model=model, role="aires_det_score_forecast", walker=walker,
                      score_lead_days=lead, init=str(pd.Timestamp(init)), members=1,
                      step_h=step_h, nsteps=int(cube.sizes["time"]) - 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".nc.tmp")
    cube.to_netcdf(tmp, encoding={gname: {"zlib": True, "complevel": 4}})
    os.replace(tmp, out)
    ds.close()


def stage_infer(event: str, model: str, shard: int, nshards: int, dry: bool) -> int:
    import importlib
    from aires import gate3 as G3

    ev = G3._event(event)
    cls_name, step_h, module = MODELS[model]
    mod = importlib.import_module(f"earth2studio.models.px.{module}")
    need = list(mod.VARIABLES)
    missing = [v for v in need if v not in IC_VARS]
    if missing:
        raise SystemExit(f"{model} needs {missing}, which the ICs do not carry")
    mine = [c for i, c in enumerate(cases()) if i % nshards == shard]
    todo = [c for c in mine if not cube_path(event, model, *c).exists()]
    absent = [c for c in todo if not ic_path(event, *c).exists()]
    if absent:
        raise SystemExit(f"{len(absent)} IC(s) missing, e.g. {ic_path(event, *absent[0])} - "
                         f"run --stage ic on Derecho and bundle with make_bundle.sh --with-det")
    print(f"[det] {model}: host={socket.gethostname()} shard {shard}/{nshards}: {len(mine)} "
          f"case(s), {len(mine) - len(todo)} cached; needs {len(need)} of the IC's {len(IC_VARS)} "
          f"channels; step {step_h} h", flush=True)
    for w, l in todo:
        with xr.open_dataarray(ic_path(event, w, l)) as ic:
            valid = pd.Timestamp(ic.time.values[0])
        n = nsteps(ev.peak, valid, step_h)
        print(f"[det]   w{w:02d}@{l:g}d: {n} x {step_h} h from {valid} -> peak {ev.peak}")
    if dry or not todo:
        return 0

    import torch
    from earth2studio.io import ZarrBackend
    from earth2studio.run import deterministic
    import earth2studio.models.px as PX

    cls = getattr(PX, cls_name)
    t0 = time.perf_counter()
    m = cls.load_model(cls.load_default_package()).to("cuda")
    load_s = time.perf_counter() - t0
    runs = []
    for w, l in todo:
        src = ICFile(ic_path(event, w, l))
        valid = pd.Timestamp(src.da.time.values[0])
        n = nsteps(ev.peak, valid, step_h)
        zarr = cube_path(event, model, w, l).with_suffix(".zarr")
        shutil.rmtree(zarr, ignore_errors=True)
        io = ZarrBackend(file_name=str(zarr), chunks={"time": 1, "lead_time": 1},
                         backend_kwargs={"overwrite": True})
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        deterministic([valid.strftime("%Y-%m-%dT%H:%M:%S")], n, m, src, io,
                      output_coords={"variable": np.array(["t2m"])}, device="cuda", verbose=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        assemble(zarr, cube_path(event, model, w, l), init=valid, model=model, walker=w,
                 lead=l, event=event, step_h=step_h)
        shutil.rmtree(zarr, ignore_errors=True)
        runs.append(dict(walker=w, lead=l, nsteps=n, seconds=dt, s_per_step=dt / n,
                         s_per_simday=dt / (n * step_h / 24), gpu_mem_gb=_gpu_mem_gb()))
        print(f"[det] {model} w{w:02d}@{l:g}d: {dt:.1f} s ({dt / n:.2f} s/step, "
              f"{runs[-1]['s_per_simday']:.2f} s/sim-day)", flush=True)
    rec = dict(model=model, host=socket.gethostname(), gpu=torch.cuda.get_device_name(0),
               load_s=load_s, torch_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
               runs=runs)
    p = det_dir(event) / model / f"timing_shard{shard}.json"
    p.write_text(json.dumps(rec, indent=2))
    print(f"[det] wrote {p}")
    return 0


# --------------------------------------------------------------------------- #
# reduce (CPU, GenCast env: aindex needs xres)
# --------------------------------------------------------------------------- #
def case_h100h(s_per_simday: float, n_walkers: int = 64) -> float:
    """H100-h of one N=64 AI+RES case scored by ONE forecast per walker per t_k.

    The walk (2.70 H100-h, FCN3 walker, wbench small1) plus n_walkers x 42 simulated days
    of score forecast (15 + 12 + 9 + 6 d from t_k = 6..15 d to the 21 d peak). One FCN3
    member at 3.94 s/sim-day gives 2.94, the same as score_noise.case_cost's M=1.
    """
    if not np.isfinite(s_per_simday):
        return float("nan")
    return 2.70 + s_per_simday * 42 * n_walkers / 3600


def v_from_pair(rho_a, rho_b, r_ab):
    """Predictability v from two scorers with independent errors: rho_a rho_b / r_ab."""
    return rho_a * rho_b / r_ab


def _boot(fn, n: int, reps: int = 5000, seed: int = 12345):
    rng = np.random.default_rng(seed)
    out = np.array([fn(rng.integers(0, n, n)) for _ in range(reps)])
    out = out[np.isfinite(out)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def stage_reduce(event: str, models, repeats: int) -> int:
    from multiprocessing import Pool
    from aires import aindex as AI
    from aires import gate3 as G3
    from aires import score_noise as SN

    ev = G3._event(event)
    # same filename, different grid on two machines once (CLAUDE.md "The climatology trap"):
    # a 2 deg file here would make every A_L a silently wrong anomaly
    from gencast_s2s import config as C
    with xr.open_dataset(C.CLIM_FILE) as clim:
        assert (clim.sizes.get("lat"), clim.sizes.get("lon")) == (105, 237), \
            f"{C.CLIM_FILE} is {dict(clim.sizes)}, not the 0.25 deg CONUS grid (lat=105, lon=237)"
    res = json.loads(G3.result_path(ev).read_text())
    walkers = res["walkers"]
    real = np.asarray(res["realized"]["box"], dtype=float)
    df = pd.read_csv(members_csv(event))
    fitted = SN.fit(SN.subset_rho(df))
    th6 = {l: df[df.lead_days == l].groupby("walker")["box"].mean().loc[walkers].to_numpy()
           for l in LEADS}

    theta = {}
    for model in models:
        paths = {(w, l): cube_path(event, model, w, l) for w in walkers for l in LEADS}
        if not all(p.exists() for p in paths.values()):
            print(f"[reduce] {model}: {sum(not p.exists() for p in paths.values())} cube(s) "
                  f"missing - skipped")
            continue
        theta[model] = {}
        for l in LEADS:
            v = []
            for w in walkers:
                with xr.open_dataset(paths[(w, l)]) as cube:
                    v.append(float(AI.indices(cube, ev.name, ev.peak, ev.metric,
                                              where=f"{model} w{w:02d}@{l:g}d")["box"].mean()))
            theta[model][l] = np.asarray(v)
    if not theta:
        print("[reduce] nothing to reduce", file=sys.stderr)
        return 1

    pear = lambda x, y: float(np.corrcoef(x, y)[0, 1])
    n = len(walkers)
    out = dict(event=event, n_walkers=n, leads=list(LEADS), models={}, pairs={})
    for model, th in theta.items():
        rows = {}
        for l in LEADS:
            f = fitted[l]
            r6, rd, r6d = pear(th6[l], real), pear(th[l], real), pear(th6[l], th[l])
            row = dict(spearman=G3._spearman(th[l], real)[0], pearson=rd,
                       spearman_ci=G3._bootstrap_rho(th[l], real), corr_with_fcn3_m6=r6d)
            if f["a"] is not None and r6d > 0:
                v = v_from_pair(r6, rd, r6d)
                lo = 1.0 / f["a"]
                row.update(v=v, frac=(v - lo) / (0.999 - lo), v_ci=_boot(
                    lambda i: v_from_pair(pear(th6[l][i], real[i]), pear(th[l][i], real[i]),
                                          pear(th6[l][i], th[l][i])), n))
            rows[str(l)] = row
        out["models"][model] = rows
    if len(theta) == 2:
        (a, ta), (b, tb) = theta.items()
        out["pairs"][f"{a}x{b}"] = {str(l): dict(v=v_from_pair(pear(ta[l], real), pear(tb[l], real),
                                                              pear(ta[l], tb[l])))
                                    for l in LEADS}

    # N_eq of each scorer on the fitted atmosphere, at frac = 0, 0.5 and its own estimate
    timing = {}
    for model in theta:
        recs = [json.loads(p.read_text()) for p in sorted((det_dir(event) / model).glob("timing_shard*.json"))]
        sd = [r["s_per_simday"] for rec in recs for r in rec["runs"]]
        timing[model] = dict(s_per_simday=float(np.median(sd)) if sd else float("nan"),
                             gpu=recs[0]["gpu"] if recs else None,
                             load_s=float(np.median([r["load_s"] for r in recs])) if recs else None)
    leads5 = sorted(fitted)
    jobs, meta = [], []
    for model, rows in out["models"].items():
        fr = [rows[str(l)]["frac"] for l in (6.0, 9.0) if "frac" in rows[str(l)]]
        own = float(np.clip(np.mean(fr), 0.0, 1.0)) if fr else None
        for label, frac in (("frac=0", 0.0), ("frac=0.5", 0.5), ("own frac", own)):
            if frac is None:
                continue
            sc = SN.scenario(fitted, frac)
            v = np.array([sc[l]["v"] for l in leads5])
            nv = []
            for l in leads5:
                if l not in LEADS or sc[l]["v"] <= 0:
                    nv.append(1.0)
                    continue
                rd = max(rows[str(l)]["pearson"], 1e-3)
                nv.append(max(sc[l]["v"] / rd ** 2 - sc[l]["v"], 0.0))
            for seed in SN.SEEDS:
                jobs.append((v, np.array(nv), seed, repeats))
                meta.append((model, label, frac))
    with Pool(min(6, len(jobs))) as pool:
        neq = pool.map(SN._job, jobs)
    agg = {}
    for (model, label, frac), o in zip(meta, neq):
        agg.setdefault((model, label, frac), []).append(o["n_eq"])
    out["n_eq"] = []
    for (model, label, frac), v in agg.items():
        v = np.asarray(v)
        simday = timing[model]["s_per_simday"]
        h100h = case_h100h(simday)
        out["n_eq"].append(dict(model=model, frame=label, frac=frac, case_h100h=h100h,
                                **{f"N_eq_{t:g}sd": float(v[:, i].mean())
                                   for i, t in enumerate(SN.THRESHOLDS)}))
    out["timing"] = timing
    p = det_dir(event) / "det_reduce.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"\n{event}: deterministic scorers on {n} Gate 3 walkers. rho_s = Spearman (the "
          f"Gate 3 statistic), r = Pearson (what v and N_eq use); FCN3 M=1 is the member mean")
    print(f"  {'lead':>5}  " + "  ".join(f"{m:>26s}" for m in theta)
          + f"  {'FCN3 M=1':>16s}  {'FCN3 M=6':>16s}")
    for l in LEADS:
        cells = []
        for m in theta:
            r = out["models"][m][str(l)]
            cells.append(f"{r['spearman']:+.2f} / {r['pearson']:+.2f}  v {r.get('v', float('nan')):.2f}")
        mem = [df[(df.lead_days == l) & (df.member == j)].set_index("walker").loc[walkers, "box"]
               for j in range(int(df.member.max()) + 1)]
        s1 = np.mean([G3._spearman(x, real)[0] for x in mem])
        r1 = np.mean([pear(x, real) for x in mem])
        print(f"  {l:4.0f}d  " + "  ".join(f"{c:>26s}" for c in cells)
              + f"  {s1:+.2f} / {r1:+.2f}     {G3._spearman(th6[l], real)[0]:+.2f} / "
                f"{pear(th6[l], real):+.2f}")
    print("  (cells: rho_s / r, then v from that model x FCN3 M=6)")
    for r in out["n_eq"]:
        ds = SN.DS_PER_H100H
        eff = r["N_eq_3sd"] / r["case_h100h"] / ds if np.isfinite(r["case_h100h"]) else float("nan")
        print(f"  {r['model']:8s} {r['frame']:9s} (frac {r['frac']:.2f})  N_eq 3 sd {r['N_eq_3sd']:5.0f}  "
              f"3.5 sd {r['N_eq_3.5sd']:5.0f}  case {r['case_h100h']:.1f} H100-h  {eff:.2f}x DS")
    for k, d in out["pairs"].items():
        print(f"  pair {k}: v = " + " / ".join(f"{d[str(l)]['v']:.2f}" for l in LEADS))
    print(f"\n  wrote {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=["sp-check", "ic", "infer", "reduce"])
    ap.add_argument("--event", default=EVENT)
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--models", default=",".join(MODELS), help="reduce: which to include")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="infer: plan and input check, no weights")
    ap.add_argument("--repeats", type=int, default=30000, help="reduce: replay repeats per seed")
    a = ap.parse_args(argv)
    if a.stage == "sp-check":
        return stage_sp_check()
    if a.stage == "ic":
        return stage_ic(a.event)
    if a.stage == "infer":
        if a.model is None:
            ap.error("--stage infer needs --model")
        return stage_infer(a.event, a.model, a.shard, a.nshards, a.dry)
    return stage_reduce(a.event, a.models.split(","), a.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
