#!/usr/bin/env python
"""Gate 2 - forecast equivalence: does an adapter-built IC forecast like a native one?

Gate 1 showed the adapter reproduces 69 of FCN3's 72 channels bit-exactly and derives the
other three to within 0.63 m/s / 4.6%. Gate 2 asks the question that actually matters for
AI+RES: **does that residual change the forecast?** If FCN3 rolled from an adapter IC
lands in the same place as FCN3 rolled from ERA5 directly, the walker->scorer handoff is
sound and the score a walker receives is a property of the walker's state, not of the
adapter.

Design
------
Two ensembles from the SAME ERA5 fields at the same init, differing only in how the 72
channels were assembled:

    native   runs/fcn3/week3/ic/<event>_ic.nc          ERA5 -> all 72 channels (ARCO)
    adapter  runs/xres/0p25/week3/inputs/<event>_inputs.nc
                                                       ERA5 -> GenCast names -> adapter

The native ensemble is **already on disk** (``runs/fcn3/week3/cache/<event>_cube.nc``,
24 members) from the FCN3-vs-GenCast experiment, so only the adapter side costs GPU time.
Members are matched **seed for seed**: this driver reuses ``fcn3/run_fcn3.py``'s exact
sharding and seeding (``seed_for(ev) + shard``, 3 members per shard over 8 shards), so
adapter member i and native member i share an internal-noise stream and their difference
is attributable to the IC alone.

Why M=24 and not the M=6 of the plan
------------------------------------
aires.md specifies M=6. The cached 24-member ensemble makes the noise floor measurable
rather than assumed, and it is fatal to M=6: the per-member spread of A_L for this event
is sigma = 0.62 K, so two 6-member ensemble means differ by 0.36 K (1 s.e.) from sampling
noise alone - **larger than the 0.25 K pass threshold**. A perfect adapter would fail
Gate 2 about half the time at M=6. The native side is free and the adapter side is ~5 min
of node wall, so both run at M=24 (s.e. 0.18 K), which puts the specified threshold above
the noise floor instead of below it. The pass criteria themselves are unchanged.

The three criteria
------------------
``G2a`` and ``G2b`` are aires.md's. ``G2c`` is added because the first two, taken alone,
cannot separate "the adapter perturbed the forecast" from "day-21 chaos" - at a 21 d lead
any perturbation whatsoever decorrelates individual members, so a rank-correlation test
run there measures the atmosphere's Lyapunov exponent, not the adapter.

    G2a  |mean A_L(adapter) - mean A_L(native)| < 0.25 K       (paired, n members)
    G2b  paired-member Spearman rank correlation > 0.95 out to at least 7 d lead
    G2c  the paired adapter-vs-native divergence never exceeds the ensemble's OWN
         internal-noise divergence, at any lead.

G2b is a HORIZON, not an endpoint (aires.md, amended 2026-08-18 after the first run).
Read at the 21 d verification window only - as aires.md originally specified - it cannot
discriminate: any IC difference whatsoever, the adapter's or a plain re-seed, has by then
grown to the internal-noise level and shuffled the members into an unrelated order, so a
perfect adapter scores ~0 there too. The endpoint value is still reported, as a
diagnostic; the pass/fail is taken from the lead at which rho_s falls through 0.95.

G2c is the well-posed form of "forecast equivalence". Both ensembles start from an
identical IC in 69 channels and are rolled with matched noise streams, so two divergence
curves can be compared directly:

    d_paired(t)    = RMS over members of (adapter_i(t) - native_i(t))   <- the adapter
    d_internal(t)  = sqrt(2) * ensemble spread of the native run        <- FCN3's own noise

(The sqrt(2) is exact: the RMS difference over all distinct member pairs of an ensemble is
sqrt(2) times its standard deviation.) If ``d_paired <= d_internal`` everywhere, swapping
a native IC for an adapter IC moves the forecast less than re-seeding the model does -
i.e. the adapter is indistinguishable from a different draw of the same ensemble. That is
a stronger statement than either threshold, and it stays meaningful at any lead.

Stages (two conda envs, as everywhere else in this repo)
-------------------------------------------------------
    prep   login node, either env : build + cache the adapter IC. NO GPU, no network.
    infer  GPU, ``fcn3`` env      : roll this shard's members from the adapter IC.
    assemble  CPU, ``fcn3`` env   : merge shard cubes.
    score  CPU, ``moe`` env       : A_L, the three criteria, CSV/JSON/figure.

``score`` needs ``xres.xmetrics`` and the 1990-2019 climatology, which live in the GenCast
env; the other stages do not.

    PYTHONPATH=. python -m aires.gate2 --stage prep            # login node, moe
    sbatch slurm/aires_gate2.slurm                             # infer + assemble
    PYTHONPATH=. python -m aires.gate2 --stage score           # login node, moe
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from fcn3 import fevents as F

# earth2studio / huggingface_hub read these at import time -> set before any e2s import.
os.environ.setdefault("EARTH2STUDIO_CACHE", str(F.FCN3_CACHE / "e2s"))
os.environ.setdefault("HF_HOME", str(F.FCN3_CACHE / "hf"))

DEFAULT_EVENT = "PNW_HeatDome_2021"

# Gate 2 pass criteria (aires.md; G2c added here - see the module docstring).
GATE2 = {
    "abs_delta_mean_AL_max": 0.25,   # K
    "spearman_min": 0.95,
    "rank_horizon_min_days": 7.0,    # G2b: rank-equivalence must reach at least this lead
    "require_below_internal_noise": True,
}


# --------------------------------------------------------------------------- #
# Paths. A separate tree so nothing in runs/fcn3/ is touched.
# --------------------------------------------------------------------------- #
GATE2_ROOT = A.AIRES_ROOT / "gate2"


def adapter_ic_path(ev: F.Event) -> Path:
    return GATE2_ROOT / "ic" / f"{ev.name}_adapter_ic.nc"


def adapter_cube_path(ev: F.Event) -> Path:
    return GATE2_ROOT / "cache" / f"{ev.name}_adapter_cube.nc"


def shard_cube_path(ev: F.Event, shard: int, nshards: int) -> Path:
    return GATE2_ROOT / "shards" / f"{ev.name}_{F.shard_tag(shard, nshards)}_cube.nc"


def shard_zarr_path(ev: F.Event, shard: int, nshards: int) -> Path:
    return GATE2_ROOT / "shards" / f"{ev.name}_{F.shard_tag(shard, nshards)}.zarr"


def result_path(ev: F.Event) -> Path:
    return GATE2_ROOT / "scores" / f"{ev.name}_gate2.json"


def ensure_dirs() -> None:
    for d in (GATE2_ROOT / "ic", GATE2_ROOT / "cache", GATE2_ROOT / "shards",
              GATE2_ROOT / "scores", A.fig_dir()):
        d.mkdir(parents=True, exist_ok=True)


def _event(name: str) -> F.Event:
    if name not in F.EVENTS:
        raise SystemExit(f"unknown event {name!r}; known: {', '.join(F.ORDER)}")
    return F.EVENTS[name]


# --------------------------------------------------------------------------- #
# prep - build the adapter IC (login node; no GPU, no network)
# --------------------------------------------------------------------------- #
def stage_prep(ev: F.Event, force: bool = False) -> int:
    from aires import adapter as AD

    ensure_dirs()
    out = adapter_ic_path(ev)
    src = F.gencast_inputs_path(ev)
    native = F.ic_path(ev)

    if not src.exists():
        print(f"[prep] ERROR: no GenCast input frames at {src}", file=sys.stderr)
        return 1
    if not native.exists():
        print(f"[prep] ERROR: no native FCN3 IC at {native}\n"
              f"       run `python fcn3/run_fcn3.py --stage prep` first", file=sys.stderr)
        return 1
    if out.exists() and not force:
        print(f"[prep] {ev.name}: adapter IC cached -> {out}")
        return 0

    cal = AD.require_calibration()
    print(f"[prep] {ev.name}: calibration = {cal.attrs.get('provenance')}")
    with xr.open_dataset(src) as gc:
        ic = AD.gencast_to_fcn3(gc, calib=cal)

    # The whole point is that both ICs describe the SAME instant from the SAME analysis.
    with xr.open_dataset(native) as nds:
        nic = nds["ic"].load()
    if ic["time"].values[0] != nic["time"].values[0]:
        print(f"[prep] ERROR: adapter IC is valid at {ic['time'].values[0]} but the native "
              f"IC is at {nic['time'].values[0]}", file=sys.stderr)
        return 1
    A.verify_against_ic(native)

    # Record the channel-by-channel difference; this is Gate 1 restated on the exact
    # frame Gate 2 will roll, so a regression shows up here rather than in a GPU job.
    a, b = ic.isel(time=0), nic.isel(time=0)
    w = np.cos(np.deg2rad(ic["lat"].values))[:, None]
    exact, derived = [], {}
    for name in A.FCN3_VARIABLES:
        d = a.sel(variable=name).values - b.sel(variable=name).values
        mx = float(np.abs(d).max())
        if name in A.DERIVED_VARS:
            derived[name] = dict(
                rmse=float(np.sqrt((w * d**2).sum() / (w.sum() * d.shape[1]))),
                bias=float((w * d).sum() / (w.sum() * d.shape[1])),
                absmax=mx,
            )
        elif mx == 0.0:
            exact.append(name)
        else:
            derived[name] = dict(absmax=mx, note="UNEXPECTED: renamed channel differs")

    print(f"[prep] {ev.name}: {len(exact)}/{len(A.FCN3_VARIABLES)} channels bit-identical "
          f"to the native IC")
    for k, v in derived.items():
        if "rmse" in v:
            print(f"[prep]   {k:6s} rmse {v['rmse']:.4f}  bias {v['bias']:+.4f}  "
                  f"absmax {v['absmax']:.3f}")
        else:
            print(f"[prep]   {k:6s} {v}")
    if len(exact) != len(A.FCN3_VARIABLES) - len(A.DERIVED_VARS):
        print("[prep] ERROR: a renamed channel is not bit-identical - the adapter or the "
              "IC has drifted; Gate 2 would not be a controlled comparison.", file=sys.stderr)
        return 1

    tmp = out.with_suffix(".nc.tmp")
    ds = ic.to_dataset(name="ic")
    ds.attrs.update(event=ev.name, init=ev.init.isoformat(), source="aires.adapter",
                    gencast_inputs=str(src), native_ic=str(native),
                    calibration=str(cal.attrs.get("provenance")),
                    n_bit_identical_channels=len(exact))
    ds.to_netcdf(tmp)
    os.replace(tmp, out)
    print(f"[prep] wrote {out} ({out.stat().st_size/1e6:.0f} MB)")

    (GATE2_ROOT / "scores").mkdir(parents=True, exist_ok=True)
    ic_report = GATE2_ROOT / "scores" / f"{ev.name}_ic_diff.json"
    ic_report.write_text(json.dumps(
        {"event": ev.name, "init": ev.init.isoformat(),
         "n_bit_identical": len(exact), "n_channels": len(A.FCN3_VARIABLES),
         "derived": derived}, indent=2))
    print(f"[prep] wrote {ic_report}")
    return 0


# --------------------------------------------------------------------------- #
# infer - roll the adapter ensemble (GPU, fcn3 env)
# --------------------------------------------------------------------------- #
def stage_infer(ev: F.Event, shard: int, nshards: int, members: int,
                batch_size: int, keep_zarr: bool) -> int:
    """Roll this shard's members from the ADAPTER IC, seeds matched to the native run.

    Every seeding and sharding decision is imported from ``fcn3/run_fcn3.py`` rather than
    restated, so adapter member i cannot silently drift onto a different noise stream than
    native member i.
    """
    import torch
    from earth2studio.io import ZarrBackend
    from earth2studio.perturbation import Zero
    from earth2studio.run import ensemble

    from fcn3.run_fcn3 import (LocalIC, _assemble_shard_cube, _get_model,
                               _shard_members, check_disco_kernel)

    ensure_dirs()
    check_disco_kernel(strict=os.environ.get("FCN3_ALLOW_FP32") != "1")
    A.verify_against_package()          # our 72-channel mirror vs the installed package

    count, offset = _shard_members(members, nshards, shard)
    if count == 0:
        print(f"[infer] shard {shard}/{nshards}: no members for total={members}")
        return 0

    cpath = shard_cube_path(ev, shard, nshards)
    if cpath.exists():
        print(f"[infer] {ev.name} shard {shard}: cached, skip")
        return 0

    icp = adapter_ic_path(ev)
    if not icp.exists():
        print(f"[infer] ERROR: no adapter IC at {icp}\n"
              f"        run `python -m aires.gate2 --stage prep` on the login node",
              file=sys.stderr)
        return 1

    _omp = os.environ.get("OMP_NUM_THREADS")
    if _omp:
        torch.set_num_threads(int(_omp))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[infer] host={socket.gethostname()} shard={shard}/{nshards}: members "
          f"{offset}..{offset+count-1} of {members}; device={dev}")

    model = _get_model()
    seed = F.seed_for(ev) + shard          # IDENTICAL to the native run's stream
    model.set_rng(seed=seed, reset=True)
    data = LocalIC(icp)

    zpath = shard_zarr_path(ev, shard, nshards)
    if zpath.exists():
        shutil.rmtree(zpath)
    io = ZarrBackend(file_name=str(zpath),
                     chunks={"ensemble": 1, "time": 1, "lead_time": 1},
                     backend_kwargs={"overwrite": True})
    out_vars = F.fcn3_vars(ev)

    print(f"[infer] {ev.name}: rolling {count} member(s) x {F.NSTEPS} steps from the "
          f"ADAPTER IC (init {ev.init.isoformat()}, vars={out_vars}, seed={seed}) ...")
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    io = ensemble([ev.init.strftime("%Y-%m-%dT%H:%M:%S")], F.NSTEPS, count, model, data,
                  io, Zero(), batch_size=batch_size,
                  output_coords={"variable": np.array(out_vars)}, device=dev, verbose=True)
    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[infer] {ev.name}: rollout {elapsed:.1f} s ({elapsed/60:.2f} min)")

    _assemble_shard_cube(ev, zpath, cpath, count, offset, seed)
    if not keep_zarr and zpath.exists():
        shutil.rmtree(zpath)
    return 0


def stage_assemble(ev: F.Event, members: int, nshards: int) -> int:
    from fcn3.run_fcn3 import _shard_members

    ensure_dirs()
    out = adapter_cube_path(ev)
    if out.exists():
        print(f"[assemble] {ev.name}: cube cached -> {out}")
        return 0
    parts, missing = [], []
    for s in range(nshards):
        count, _ = _shard_members(members, nshards, s)
        if count == 0:
            continue
        p = shard_cube_path(ev, s, nshards)
        (parts.append(xr.open_dataset(p)) if p.exists() else missing.append(s))
    if missing:
        for p in parts:
            p.close()
        print(f"[assemble] {ev.name}: shard cube(s) missing: {missing}", file=sys.stderr)
        return 1
    full = xr.concat(parts, dim="member").sortby("member")
    full = full.assign_coords(member=("member", np.arange(full.sizes["member"])))
    full.attrs.update(event=ev.name, init=ev.init.isoformat(), peak=ev.peak,
                      model="FourCastNet3", ic_source="aires.adapter",
                      members=int(full.sizes["member"]), nsteps=int(F.NSTEPS),
                      weeks=int(F.WEEKS), parallel_shards=int(nshards))
    tmp = out.with_suffix(".nc.tmp")
    full.to_netcdf(tmp)
    os.replace(tmp, out)
    for p in parts:
        p.close()
    print(f"[assemble] {ev.name}: wrote {out} {dict(full.sizes)}")
    return 0


# --------------------------------------------------------------------------- #
# score - the three criteria (CPU, moe env)
# --------------------------------------------------------------------------- #
def _area_mean(da: xr.DataArray) -> xr.DataArray:
    """Unboxed cos-lat area mean over the cube's full (CONUS) extent.

    Delegates to ``aires.aindex`` so the divergence curves below and the observable are
    reduced by one implementation. ``box=None`` is the CONUS reduction Gate 2 was scored
    with before the per-event boxes existed; see ``_AL``.
    """
    from aires import aindex as AI
    return AI.area_mean(da, box=None)


def _AL(cube: xr.Dataset, ev: F.Event) -> np.ndarray:
    """The scalar observable, per member: cos-lat-weighted CONUS mean of the metric field.

    Gate 2 is scored on the **CONUS** mean, not on the per-event box that
    ``aires/aindex.py`` now defines and that Gate 3 onwards uses. That is deliberate and
    not laziness: the published Gate 2 numbers (G2a 0.083 K, horizon 8.5 d, G2c 0.990)
    were measured this way, and Gate 2 asks whether the adapter perturbs the forecast at
    all - a question about the whole field, for which the widest average is the right
    reduction. The box would only make the same test noisier.
    """
    from aires import aindex as AI
    return AI.a_index(cube, ev.peak, ev.metric, box=None,
                      where=f"gate2 cube {ev.name}").values


def _divergence_curves(nat: xr.Dataset, ada: xr.Dataset, var: str = "2m_temperature"):
    """Paired adapter-vs-native divergence, and the ensemble's own internal-noise level.

    ``d_internal`` uses the identity that the RMS difference over all distinct member
    pairs of an ensemble equals sqrt(2) times its standard deviation - exact, and far
    cheaper than looping over the 276 pairs.
    """
    n, a = nat[var], ada[var]
    m = min(n.sizes["member"], a.sizes["member"])
    n, a = n.isel(member=slice(0, m)), a.isel(member=slice(0, m))
    w = np.cos(np.deg2rad(n["lat"]))
    d_paired = np.sqrt(_area_mean(((a - n) ** 2).mean("member")))
    d_internal = np.sqrt(2.0) * np.sqrt(_area_mean(n.var("member", ddof=1)))
    return (pd.DatetimeIndex(n["time"].values),
            d_paired.values.astype(float), d_internal.values.astype(float))


def _rank_corr_by_lead(nat: xr.Dataset, ada: xr.Dataset, var: str = "2m_temperature"):
    """Paired-member Spearman correlation as a function of lead.

    This is G2b. aires.md originally read it at the 21 d verification window only, where
    it cannot discriminate: a perturbation that has grown to the internal-noise level has,
    by definition, shuffled the members into an unrelated order, so ANY IC difference -
    the adapter's or a re-seed's - scores ~0 there. Resolving it by lead turns that from
    an assertion into a measurement: the correlation starts at 1, and the lead at which it
    falls through 0.95 is the model's own predictability horizon for this quantity, not a
    property of the adapter. G2b is now that horizon, and must reach >= 7 d.
    """
    from scipy.stats import spearmanr

    n, a = nat[var], ada[var]
    m = min(n.sizes["member"], a.sizes["member"])
    nm = _area_mean(n.isel(member=slice(0, m))).values      # (member, time)
    am = _area_mean(a.isel(member=slice(0, m))).values
    out = np.full(nm.shape[1], np.nan)
    for j in range(nm.shape[1]):
        if np.ptp(nm[:, j]) > 0 and np.ptp(am[:, j]) > 0:
            out[j] = spearmanr(nm[:, j], am[:, j]).statistic
    return out


def stage_score(ev: F.Event, plot: bool = True) -> int:
    from scipy.stats import spearmanr

    ensure_dirs()
    npath, apath = F.cube_path(ev), adapter_cube_path(ev)
    for p, what in ((npath, "native"), (apath, "adapter")):
        if not p.exists():
            print(f"[score] ERROR: no {what} cube at {p}", file=sys.stderr)
            return 1

    nat, ada = xr.open_dataset(npath), xr.open_dataset(apath)
    m = min(nat.sizes["member"], ada.sizes["member"])
    nat_m, ada_m = nat.isel(member=slice(0, m)), ada.isel(member=slice(0, m))
    print(f"[score] {ev.name}: {m} matched member(s), metric {ev.metric}")

    al_n, al_a = _AL(nat_m, ev), _AL(ada_m, ev)
    d = al_a - al_n
    delta_mean = float(d.mean())
    se_paired = float(d.std(ddof=1) / np.sqrt(m))
    sigma = float(al_n.std(ddof=1))
    rho = float(spearmanr(al_n, al_a).statistic)
    pear = float(np.corrcoef(al_n, al_a)[0, 1])

    times, d_paired, d_internal = _divergence_curves(nat_m, ada_m)
    rho_lead = _rank_corr_by_lead(nat_m, ada_m)
    lead_d = (times - pd.Timestamp(ev.init)).total_seconds().values / 86400.0
    ratio = np.divide(d_paired, d_internal, out=np.full_like(d_paired, np.nan),
                      where=d_internal > 0)
    worst_i = int(np.nanargmax(ratio))
    max_ratio = float(ratio[worst_i])

    # Last lead at which the two ensembles are still rank-equivalent by G2b's own
    # threshold - i.e. how far out the criterion is capable of discriminating at all.
    ok_lead = [lead_d[j] for j in range(len(rho_lead))
               if np.isfinite(rho_lead[j]) and rho_lead[j] > GATE2["spearman_min"]]
    horizon = float(max(ok_lead)) if ok_lead else float("nan")

    g2a = abs(delta_mean) < GATE2["abs_delta_mean_AL_max"]
    g2b = np.isfinite(horizon) and horizon >= GATE2["rank_horizon_min_days"]
    g2c = max_ratio <= 1.0

    print(f"\n  per-member A_L ({ev.metric}, cos-lat CONUS mean)")
    print(f"    native  mean {al_n.mean():+7.3f}   sd {sigma:.3f}")
    print(f"    adapter mean {al_a.mean():+7.3f}   sd {al_a.std(ddof=1):.3f}")
    print(f"    paired difference: mean {delta_mean:+.3f}  sd {d.std(ddof=1):.3f}  "
          f"s.e. {se_paired:.3f}")
    print(f"    decorrelated reference (sd of a difference of two independent draws): "
          f"{sigma*np.sqrt(2):.3f}")

    print(f"\n  GATE 2  (n = {m} matched members)")
    print(f"    G2a |delta mean A_L|      {abs(delta_mean):7.4f}    <  "
          f"{GATE2['abs_delta_mean_AL_max']:.2f}      {'PASS' if g2a else 'FAIL'}")
    print(f"    G2b rank-equiv horizon   {horizon:7.2f} d  >= "
          f"{GATE2['rank_horizon_min_days']:.1f} d     {'PASS' if g2b else 'FAIL'}"
          f"   (rho_s > {GATE2['spearman_min']:.2f}; at the 21 d window it is {rho:+.3f},"
          f"\n                                                    which no IC difference,"
          f" however small, could beat)")
    print(f"    G2c max d_paired/d_noise  {max_ratio:7.4f}    <= 1.00      "
          f"{'PASS' if g2c else 'FAIL'}   (at lead {lead_d[worst_i]:.2f} d)")
    print(f"\n    (Pearson r {pear:+.4f}; paired s.e. on delta mean A_L is {se_paired:.3f} K, "
          f"so G2a resolves\n     differences down to ~{2*se_paired:.2f} K and no finer.)")
    ok = g2a and g2b and g2c
    print(f"\n  GATE 2: {'PASS' if ok else 'FAIL'}")

    print(f"\n  paired rank correlation by lead (G2b's statistic, resolved)")
    for j in range(0, len(lead_d), max(1, len(lead_d) // 8)):
        print(f"    lead {lead_d[j]:5.2f} d   rho_s {rho_lead[j]:+.3f}")
    print(f"    lead {lead_d[-1]:5.2f} d   rho_s {rho_lead[-1]:+.3f}")
    print(f"    -> rank-equivalent (rho_s > {GATE2['spearman_min']:.2f}) out to "
          f"{horizon:.2f} d; beyond that the adapter's perturbation has grown into the "
          f"model's\n       own noise and no IC difference, however small, could score "
          f"otherwise.")

    res = {
        "event": ev.name, "metric": ev.metric, "init": ev.init.isoformat(),
        "peak": ev.peak, "n_members": m,
        "AL_native": al_n.tolist(), "AL_adapter": al_a.tolist(),
        "mean_AL_native": float(al_n.mean()), "mean_AL_adapter": float(al_a.mean()),
        "delta_mean_AL": delta_mean, "se_delta_mean_AL": se_paired,
        "sd_AL_native": sigma,
        "decorrelated_reference_sd": sigma * float(np.sqrt(2)),
        "spearman": rho, "pearson": pear,
        "lead_days": lead_d.tolist(),
        "d_paired": d_paired.tolist(), "d_internal": d_internal.tolist(),
        "rho_by_lead": [None if not np.isfinite(x) else float(x) for x in rho_lead],
        "rank_equivalence_horizon_days": horizon,
        "max_ratio_paired_over_internal": max_ratio,
        "max_ratio_lead_days": float(lead_d[worst_i]),
        "G2a_delta_mean_AL": bool(g2a), "G2b_rank_horizon": bool(g2b),
        "rank_horizon_min_days": GATE2["rank_horizon_min_days"],
        "G2c_below_internal_noise": bool(g2c), "gate2_pass": bool(ok),
    }
    rp = result_path(ev)
    rp.write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {rp}")

    csv = GATE2_ROOT / "scores" / f"{ev.name}_members.csv"
    pd.DataFrame({"member": np.arange(m), "AL_native": al_n, "AL_adapter": al_a,
                  "diff": d}).to_csv(csv, index=False)
    print(f"  wrote {csv}")

    if plot:
        _plot(ev, res)
    nat.close(); ada.close()
    return 0 if ok else 1


def _plot(ev: F.Event, res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    al_n = np.array(res["AL_native"]); al_a = np.array(res["AL_adapter"])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    lo = min(al_n.min(), al_a.min()); hi = max(al_n.max(), al_a.max())
    pad = 0.08 * (hi - lo)
    ax[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", lw=1, zorder=1)
    ax[0].scatter(al_n, al_a, s=34, color="#1f77b4", zorder=3)
    ax[0].set_xlabel("native IC   $A_L$  (K)")
    ax[0].set_ylabel("adapter IC   $A_L$  (K)")
    ax[0].set_title(f"paired members  (n={res['n_members']})\n"
                    rf"$\Delta\overline{{A_L}}$ = {res['delta_mean_AL']:+.3f} K, "
                    rf"$\rho_s$ = {res['spearman']:.3f}")
    ax[0].set_xlim(lo - pad, hi + pad); ax[0].set_ylim(lo - pad, hi + pad)
    ax[0].grid(alpha=.3)

    lead = np.array(res["lead_days"])
    ax[1].plot(lead, res["d_internal"], color="#d62728", lw=2,
               label=r"FCN3's own ensemble noise  ($\sqrt{2}\,\sigma$)")
    ax[1].plot(lead, res["d_paired"], color="#1f77b4", lw=2,
               label="adapter IC vs native IC (matched seeds)")
    ax[1].set_xlabel("lead (days)")
    ax[1].set_ylabel("CONUS T2m RMS difference (K)")
    ax[1].set_title("G2c: does the adapter move the forecast\n"
                    "less than re-seeding the model does?")
    ax[1].legend(loc="upper left", fontsize=9)
    ax[1].grid(alpha=.3)

    rho = np.array([np.nan if v is None else v for v in res["rho_by_lead"]])
    ax[2].axhline(0.95, color="0.6", lw=1, ls="--")
    ax[2].text(lead[-1], 0.90, r"$\rho_s = 0.95$  ", fontsize=8, color="0.4",
               va="top", ha="right")
    ax[2].axhline(0.0, color="0.85", lw=1)
    ax[2].plot(lead, rho, color="#2ca02c", lw=2)
    h = res["rank_equivalence_horizon_days"]
    if np.isfinite(h):
        ax[2].axvline(h, color="#2ca02c", lw=1, ls=":")
        ax[2].text(h, -0.75, f"  {h:.1f} d", fontsize=9, color="#2ca02c")
    ax[2].set_xlabel("lead (days)")
    ax[2].set_ylabel(r"paired Spearman $\rho_s$")
    ax[2].set_ylim(-1.05, 1.05)
    ax[2].set_title("G2b: how far out are the two ensembles\n"
                    "still member-for-member rank-equivalent?")
    ax[2].grid(alpha=.3)

    verdict = "PASS" if res["gate2_pass"] else "FAIL"
    fig.suptitle(f"Gate 2 - forecast equivalence, {ev.name} "
                 f"(init {res['init'][:10]}, {F.LEAD_DAYS} d lead): {verdict}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = A.fig_dir() / f"gate2_{ev.name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["prep", "infer", "assemble", "score"])
    ap.add_argument("--event", default=os.environ.get("AIRES_GATE2_EVENT", DEFAULT_EVENT))
    ap.add_argument("--members", type=int,
                    default=int(os.environ.get("AIRES_GATE2_MEMBERS", F.N_MEMBERS)))
    ap.add_argument("--shard", type=int, default=int(os.environ.get("FCN3_SHARD", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("FCN3_NSHARDS", 8)))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("FCN3_BATCH", 1)))
    ap.add_argument("--keep-zarr", action="store_true")
    ap.add_argument("--force", action="store_true", help="prep: rebuild a cached IC")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    ev = _event(a.event)

    if a.stage == "prep":
        return stage_prep(ev, force=a.force)
    if a.stage == "infer":
        return stage_infer(ev, a.shard, a.nshards, a.members, a.batch_size, a.keep_zarr)
    if a.stage == "assemble":
        return stage_assemble(ev, a.members, a.nshards)
    return stage_score(ev, plot=not a.no_plot)


if __name__ == "__main__":
    sys.exit(main())
