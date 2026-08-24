#!/usr/bin/env python
"""The AI+RES probability DISTRIBUTION, drawn the way this repo draws PDFs.

`aires/aplots.py` reduces the pilot to an exceedance curve of the scalar `A_L`. This
module asks the other half of the question: what does the whole verification-week T2m
anomaly FIELD look like as a probability distribution, and does the importance-weighted
AI+RES population put mass where ERA5 actually was?

    P(a grid point in the region has anomaly x)   for x in 100 shared bins

Six populations are drawn on one log-y axis, in the house `xres/xcombined.py`
`PDF_histogram` recipe (counts/N per bin, 100 bins, log y, ylim 1e-6..1):

    ERA5 truth                        the realized field, 1 map
    AI+RES weighted                   the 64 resampled walkers, each map weighted e^{-V_K}
    AI+RES unweighted                 the same 64 maps with the RES tilt left in
    GenCast direct (xres, 24)         the like-for-like direct ensemble
    GenCast free-running walkers (16)  Gate 3's un-resampled walkers
    FCN3 direct (24)                  a different model, carried for context

The weighted estimator
----------------------
`weighted_pdf` is the exact weighted generalization of `PDF_histogram`. A walker
contributes its G grid points with its own weight w_i, and the result is
SELF-NORMALIZED,

    p_b = (sum_i w_i n_ib) / (G * sum_i w_i)        sum_b p_b = 1

so the curve is a probability distribution and is directly comparable, bin for bin, with
the unweighted curves. The DMC estimator's own scale factor is `Z * mean(w)` -- the
`normalization_check` in `res_result.json`, 0.5684 for this pilot -- and applying it would
shift the whole weighted curve down by a constant 0.246 decades. That factor is REPORTED
on the figure and is available behind `--z-scaled`; it is not the default because a curve
whose mass is 0.568 invites the reader to compare its height against curves whose mass is
1, and the comparison this figure exists for is a shape comparison.

Read the weighted curve honestly
--------------------------------
The weight ESS is 5.7 of 64 walkers, and the single heaviest weight belongs to the
population's COLDEST member (resampling tilted towards heat, so the estimator has to undo
that tilt hardest on the walkers it least selected). The weighted curve's mode and left
flank are therefore a small-sample artifact of a handful of walkers. The right tail --
whether AI+RES reaches the observed one where direct sampling does not -- is what the
figure is for, and it is the only claim it supports. An unbiased forecast PDF should not
EQUAL a single realized field either; "reaches the observed tail" is the supportable
phrasing, not "matches ERA5".

Stages
------
    build   ~17 s. Rebuilds every walker's verification field from the genealogy (the
            same read `aires/run_aires.py --stage compare` does, restricted to the
            segments that actually cover the window and to `2m_temperature`), reduces the
            baselines, and writes ONE intermediate cube. Prints gate lines G1/G4/G13 and
            exits nonzero on a violation.
    figs    instant. Reads only the intermediate.

    PYTHONPATH=. python -m aires.apdfs --stage all --event PNW_HeatDome_2021 --tag pilot
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from aires import aplots as AP
from aires import run_aires as R
from fcn3 import fevents as F

FIGURES = ("pdf_conus", "pdf_box", "anom_maps")
STAGES = ("build", "figs", "all")

# House PDF style, `xres/xcombined.py`. Mirrored rather than imported: importing
# `xres.xcombined` drags in the whole plotting/HRRR stack (~28 s) and this module has to
# be cheap to import. `aires/tests/test_apdfs.py` asserts the two agree.
NBINS = 100
YMIN = 1e-6

# The verification window is 13 frames at a 12 h step. Never a "plausible" 2-frame mean.
WINDOW_FRAMES = 13
GATE_TOL = 1e-6

# The tail the figure tabulates: a per-grid-point exceedance, not an A_L exceedance.
TAIL_THRESHOLD = 9.0

# Which populations define the SHARED bin range (the house mean +/- 4 sigma rule, clipped
# to the data range). The three headline curves only: ERA5, the AI+RES population and the
# like-for-like direct GenCast ensemble. The Gate 3 walkers and the FCN3 ensemble are
# context curves drawn ON that range -- letting them widen it would move the bin edges,
# and with them every published per-bin probability, every time a context curve is added
# or dropped.
BIN_RANGE_SOURCES = ("era5", "aires_unweighted", "gencast_xres")

# Colours. Truth/forecast reds from `xres/xconfig.py::RES_SPECS['0p25']`, the AI+RES blue
# and the two direct-sampling colours from `aires/aplots.py`.
TRUTH_COLOR = "#99000d"
GENCAST_COLOR = "#f4562b"
RES_COLOR = AP.RES_COLOR                      # "#1f77b4"
OBS_COLOR = AP.OBS_COLOR                      # "#d62728"
G3_COLOR = AP.DS_STYLE["gencast_walkers"][0]  # "#2ca02c"
FCN3_COLOR = AP.DS_STYLE["fcn3"][0]           # "#9467bd"
HRRR_COLOR = "#4a1486"


# --------------------------------------------------------------------------- #
# Paths and small helpers
# --------------------------------------------------------------------------- #
def pdf_dir(ctx: R.Ctx) -> Path:
    return A.res_dir(ctx.event, ctx.tag) / "pdf"


def fields_path(ctx: R.Ctx) -> Path:
    return pdf_dir(ctx) / "pdf_fields.nc"


def compare_path(ctx: R.Ctx) -> Path:
    return A.res_dir(ctx.event, ctx.tag) / "compare.json"


def res_result_path(ctx: R.Ctx) -> Path:
    return A.res_dir(ctx.event, ctx.tag) / "res_result.json"


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "-C", str(A.ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(A.ROOT))
    except Exception:
        return str(p)


def _ess(w) -> float:
    """Kish effective sample size of a weight vector."""
    w = np.asarray(w, dtype="float64")
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def _clean(da: xr.DataArray, dim: str | None = None, name: str | None = None):
    """Strip every coordinate but lat/lon, and renumber ``dim`` from 0.

    Two cubes that both carry a ``member`` coordinate but number it differently would be
    ALIGNED when they meet in one Dataset - the union of the two indexes, padded with
    NaN - rather than stacked. That is a silent hole in a curve, so the member axis is
    positional here and nothing else survives the trip.
    """
    keep = {"lat", "lon"}
    drop = [c for c in da.coords if c not in keep and c != dim]
    if drop:
        da = da.drop_vars(drop)
    if dim is not None:
        da = da.assign_coords({dim: np.arange(da.sizes[dim], dtype="int32")})
    if name is not None:
        da = da.rename(name)
    da.attrs.clear()
    return da


def _region(event: str, key: str):
    from aires import aindex as AI
    return AI.CONUS if key == "conus" else AI.box_for(event)


# --------------------------------------------------------------------------- #
# The weighted PDF core (pure; tested without matplotlib)
# --------------------------------------------------------------------------- #
def weighted_pdf(x, weights=None, xmin=None, xmax=None, nbins: int = NBINS,
                 mode: str = "selfnorm", z_factor: float | None = None):
    """Binned PDF of ``x``, optionally weighting each leading-axis slab of ``x``.

    ``weights=None`` is exactly ``xres.xcombined.PDF_histogram``: every finite value
    counts once and the histogram is divided by the number of finite values.

    With ``weights`` (one per slab along axis 0, e.g. one per walker), slab ``i``
    contributes its ``G`` points with weight ``w_i`` and the result is self-normalized,

        p_b = (sum_i w_i n_ib) / (G * sum_i w_i)

    which sums to 1 whenever the bin range covers the data - the exact weighted
    generalization of the unweighted recipe (``w_i = 1`` gives it back identically).

    ``mode='z'`` multiplies by ``z_factor`` (the DMC ``normalization_check``), turning the
    self-normalized curve into the estimator's own un-normalized one. It is a constant
    rescaling, never a reshaping.

    Returns ``(bin_centers, density)``. Empty bins are ``0.0``, not NaN - the plotting
    layer decides whether to break the curve there, and the identity above has to hold
    bin for bin against ``PDF_histogram``.
    """
    if mode not in ("selfnorm", "z"):
        raise ValueError(f"mode must be 'selfnorm' or 'z', got {mode!r}")
    if mode == "z" and z_factor is None:
        raise ValueError("mode='z' needs z_factor (the DMC normalization_check)")

    x = np.asarray(x, dtype="float64")
    flat = x.ravel()

    if weights is None:
        vals = flat[np.isfinite(flat)]
        wr = None
        denom = float(vals.shape[0])
    else:
        w = np.asarray(weights, dtype="float64")
        if x.ndim < 2 or x.shape[0] != w.size:
            raise ValueError(
                f"weights has {w.size} entries but x has shape {x.shape}; x must be "
                "(n_weighted_slabs, ...) so every slab gets exactly one weight")
        g = int(np.prod(x.shape[1:]))
        ok = np.isfinite(flat)
        vals = flat[ok]
        wr = np.repeat(w, g)[ok]
        denom = float(g) * float(np.sum(w))

    if vals.size == 0:
        raise ValueError("no finite values to histogram")
    if denom <= 0:
        raise ValueError(f"non-positive normalization {denom}")
    if xmin is None:
        xmin = float(vals.mean() - 4 * vals.std())
    if xmax is None:
        xmax = float(vals.mean() + 4 * vals.std())

    hist, edges = np.histogram(vals, range=(float(xmin), float(xmax)), bins=nbins,
                               weights=wr)
    density = hist / denom
    if mode == "z":
        density = density * float(z_factor)
    points = (edges[0:-1] + edges[1:]) * 0.5
    return points, density


def shared_range(pools, nsigma: float = 4.0) -> tuple[float, float]:
    """The house shared bin range: mean +/- ``nsigma`` sigma of the concatenated pool,
    clipped to the pool's own extent (`xres/xcombined.py:158-162`)."""
    allv = np.concatenate([np.asarray(p, dtype="float64").ravel() for p in pools])
    allv = allv[np.isfinite(allv)]
    mu, sd = allv.mean(), allv.std()
    return (float(max(allv.min(), mu - nsigma * sd)),
            float(min(allv.max(), mu + nsigma * sd)))


def tail_probability(points, density, threshold: float = TAIL_THRESHOLD) -> float:
    """Mass of the binned curve at or above ``threshold``, by bin centre."""
    p = np.asarray(points, dtype="float64")
    d = np.asarray(density, dtype="float64")
    return float(np.nansum(d[p >= float(threshold)]))


# --------------------------------------------------------------------------- #
# stage: build
# --------------------------------------------------------------------------- #
def read_lineage(ctx: R.Ctx, d: dict) -> list[dict[int, int]]:
    """``[{segment: slot}, ...]`` per final walker, from ``compare.json``, cross-checked.

    `compare.json` is the artifact the rest of Phase 4 is scored against, so it is the
    primary. `run_aires.slot_lineage` walking `steps/leg*/walk.json` is the independent
    derivation. They must agree: a slot is a container, not a lineage, and reading the
    wrong slot silently returns a different walker's trajectory.
    """
    lin = d["realized"]["lineage"]
    if len(lin) != ctx.n_walkers:
        raise RuntimeError(f"compare.json holds {len(lin)} lineages for "
                           f"{ctx.n_walkers} walkers")
    par = R.parents_by_leg(ctx)
    out, bad = [], []
    for w in range(ctx.n_walkers):
        ref = {int(k): int(v) for k, v in lin[w].items()}
        got = {int(k): int(v) for k, v in R.slot_lineage(ctx, par, w).items()}
        if ref != got:
            bad.append((w, ref, got))
        out.append(ref)
    if bad:
        w, ref, got = bad[0]
        raise RuntimeError(
            f"lineage disagreement on {len(bad)}/{ctx.n_walkers} walkers. First is "
            f"w{w:02d}:\n  compare.json : {ref}\n  slot_lineage : {got}\n"
            "One of the two is reading a slot that never held this walker's ancestry, "
            "so every field built from it would be some other trajectory.")
    return out


def covering_segments(peak, segments, path_of) -> list[int]:
    """The segments whose frames actually fall in [peak-6d, peak].

    DERIVED from each segment's time coverage rather than hardcoded: a different horizon,
    segment length or event peak moves the window, and a hardcoded 5/6/7 would then read
    three wrong files and still produce a perfectly plausible 13-frame mean of something
    else. For this pilot it returns [5, 6, 7]; segment 5 contributes only its last frame.
    """
    peak = pd.Timestamp(peak)
    lo = peak - pd.Timedelta(days=6)
    out = []
    for s in segments:
        p = path_of(int(s))
        if not Path(p).exists():
            continue
        with xr.open_dataset(p) as ds:
            t = pd.DatetimeIndex(ds["time"].values)
        if bool(((t >= lo) & (t <= peak)).any()):
            out.append(int(s))
    if not out:
        raise RuntimeError(f"no segment covers [{lo}, {peak}]")
    return out


def _t2m_only(path) -> xr.DataArray:
    """One segment's ``2m_temperature``, and nothing else off the disk (~600 kB).

    ``lead_h`` is dropped for the reason `run_aires.concat_diag` drops it: it restarts at
    0 in every segment, so a stitched trajectory cannot carry it.
    """
    with xr.open_dataset(path) as ds:
        da = ds["2m_temperature"]
        da = da.drop_vars([c for c in ("lead_h",) if c in da.coords])
        return da.load()


def trajectory_field(peak, metric: str, paths, *, where: str):
    """One walker's (lat, lon) verification field, plus the window frames it used."""
    from aires import aindex as AI

    traj = xr.concat([_t2m_only(p) for p in paths], dim="time",
                     coords="minimal", compat="override").sortby("time")
    cube = traj.to_dataset(name="2m_temperature")
    got = AI.check_window(cube, peak, metric, where=where)
    if got != WINDOW_FRAMES:
        raise RuntimeError(f"{where}: the verification window holds {got} frames, "
                           f"want exactly {WINDOW_FRAMES}")
    times = pd.DatetimeIndex(AI.window_times(cube, peak, metric))
    want = pd.date_range(pd.Timestamp(peak) - pd.Timedelta(days=6), pd.Timestamp(peak),
                         freq="12h")
    if not times.equals(want):
        raise RuntimeError(f"{where}: window times {list(times)} != {list(want)}")
    return _clean(AI.field(cube, peak, metric)), times


def walker_fields(ctx: R.Ctx, lineage, segs):
    """The AI+RES population's verification fields, one per FINAL walker."""
    fields, times = [], None
    for w in range(ctx.n_walkers):
        f, t = trajectory_field(ctx.ev.peak, ctx.ev.metric,
                                [ctx.diag_path(lineage[w][s], s) for s in segs],
                                where=f"w{w:02d} trajectory")
        times = t if times is None else times
        fields.append(f)
    out = xr.concat(fields, dim="walker")
    return _clean(out, dim="walker", name="aires"), times


def gate3_fields(ctx: R.Ctx):
    """The 16 free-running Gate 3 walkers, reduced identically. slot == walker there.

    Their axis is ``gate3_walker`` rather than ``member`` because there are 16 of them
    against the direct ensembles' 24; one ``member`` dimension cannot hold both.
    """
    all_segs = [k for (k, _l, _n) in A.gate3_segments()]
    have = [w for w in range(A.GATE3_N_WALKERS)
            if all(A.diag_path(ctx.event, w, k).exists() for k in all_segs)]
    if not have:
        return None
    segs = covering_segments(ctx.ev.peak, all_segs,
                             lambda s: A.diag_path(ctx.event, have[0], s))
    fields = []
    for w in have:
        f, _ = trajectory_field(ctx.ev.peak, ctx.ev.metric,
                                [A.diag_path(ctx.event, w, s) for s in segs],
                                where=f"gate3 w{w:02d}")
        fields.append(f)
    out = xr.concat(fields, dim="gate3_walker")
    return _clean(out, dim="gate3_walker", name="gencast_walkers")


def _truth_field(path, name: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        return _clean(ds[list(ds.data_vars)[0]].load(), name=name)


def _member_field(ctx: R.Ctx, path, where: str, name: str) -> xr.DataArray:
    from aires import aindex as AI

    with xr.open_dataset(path) as cube:
        AI.check_window(cube, ctx.ev.peak, ctx.ev.metric, where=where)
        da = AI.field(cube, ctx.ev.peak, ctx.ev.metric).load()
    return _clean(da, dim="member", name=name)


def xres_member_field(ctx: R.Ctx):
    """The 24-member direct GenCast ensemble's verification field.

    Prefers the cached 2 MB `verif/<event>_<metric>_members.nc` over the 5.8 GB cube it
    was reduced from. They are bit-identical, and gate G13 below proves it on every run
    rather than assuming it.
    """
    verif = (A.RUNS / "xres" / A.WALKER_RES / f"week{A.WALKER_WEEKS}" / "verif" /
             f"{ctx.event}_{ctx.ev.metric}_members.nc")
    if verif.exists():
        with xr.open_dataset(verif) as ds:
            return _clean(ds[ctx.ev.metric].load(), dim="member", name="gencast_xres")
    cube = F.gencast_cube_path(ctx.ev)
    if cube.exists():
        return _member_field(ctx, cube, "gencast_xres", "gencast_xres")
    return None


def _max_abs(label: str, got, ref) -> float:
    got = np.asarray(got, dtype="float64")
    ref = np.asarray(ref, dtype="float64")
    if got.shape != ref.shape:
        raise RuntimeError(f"{label}: shape {got.shape} vs reference {ref.shape}")
    return float(np.max(np.abs(got - ref)))


def build(ctx: R.Ctx, *, force: bool = False) -> Path:
    """Rebuild every field the figures need and cache them in one netCDF."""
    from aires import aindex as AI
    from aires import dmc

    cp, rp = compare_path(ctx), res_result_path(ctx)
    for p in (cp, rp):
        if not p.exists():
            raise SystemExit(
                f"no {p}\n  Run the reductions first:\n"
                f"    PYTHONPATH=. python -m aires.run_aires --stage ds --event "
                f"{ctx.event}\n"
                f"    PYTHONPATH=. python -m aires.run_aires --stage compare --event "
                f"{ctx.event} --tag {ctx.tag}")

    out_path = fields_path(ctx)
    if out_path.exists() and not force:
        if out_path.stat().st_mtime >= cp.stat().st_mtime:
            print(f"[apdfs] build: {_rel(out_path)} is up to date "
                  f"({out_path.stat().st_size / 1e6:.1f} MB); --force to rebuild")
            return out_path
        print(f"[apdfs] build: {_rel(out_path)} predates compare.json; rebuilding")

    d = R.read_json(cp)
    rec = R.read_json(rp)
    ctx.n_walkers = int(d["config"]["n_walkers"])
    result = dmc.DMCResult.from_dict(rec["dmc"])
    weights = np.asarray(result.weights, dtype="float64")
    norm = float(result.normalization_check())

    box = AI.box_for(ctx.event)
    print(f"[apdfs] build: {ctx.event} tag={ctx.tag} metric={ctx.ev.metric} "
          f"box={box.name} ({box.label})")

    # --- G4: the weights, before anything is read off disk --------------------- #
    g4 = _max_abs("weights", weights, d["weights"])
    ok_g4 = g4 < 1e-12
    print(f"[apdfs] gate G4  max |w - compare.json| {g4:.2e}   "
          f"sum w = {weights.sum():.6f}   Z*mean(w) = {norm:.16f}   "
          f"ESS = {_ess(weights):.2f}   {'PASS' if ok_g4 else 'FAIL'}")

    # --- the walkers ----------------------------------------------------------- #
    lineage = read_lineage(ctx, d)
    segs = covering_segments(ctx.ev.peak, ctx.segments,
                             lambda s: ctx.diag_path(lineage[0][s], s))
    print(f"[apdfs] window segments {segs} of {ctx.segments} (derived from time "
          f"coverage; 2m_temperature only)")

    aires_f, times = walker_fields(ctx, lineage, segs)
    a_box = AI.area_mean(aires_f, box).values.astype("float64")
    a_conus = AI.area_mean(aires_f, AI.CONUS).values.astype("float64")
    print(f"[apdfs] build: {ctx.n_walkers}/{ctx.n_walkers} walkers "
          f"({WINDOW_FRAMES} frames each, {times[0]} .. {times[-1]})")

    # --- G1: does the rebuild reproduce compare.json's own reduction? ---------- #
    g1b = _max_abs("A_L box", a_box, d["realized"]["box"])
    g1c = _max_abs("A_L conus", a_conus, d["realized"]["conus"])
    ok_g1 = max(g1b, g1c) < GATE_TOL
    print(f"[apdfs] gate G1  max |A_L - compare.json|  box {g1b:.2e}  "
          f"conus {g1c:.2e}   (tol {GATE_TOL:g})   {'PASS' if ok_g1 else 'FAIL'}")

    # --- baselines ------------------------------------------------------------- #
    era5 = _truth_field(F.era5_truth_path(ctx.ev), "era5")
    hrrr_path = A.RUNS / "observations" / f"{ctx.event}_hrrr_verif_{ctx.ev.metric}.nc"
    hrrr = _truth_field(hrrr_path, "hrrr") if hrrr_path.exists() else None
    gencast_xres = xres_member_field(ctx)
    gencast_walkers = gate3_fields(ctx)
    fcn3 = (_member_field(ctx, F.cube_path(ctx.ev), "fcn3", "fcn3")
            if F.cube_path(ctx.ev).exists() else None)
    if gencast_xres is None:
        raise SystemExit("no direct GenCast ensemble on disk; nothing to compare against")
    if fcn3 is not None and fcn3.sizes["member"] != gencast_xres.sizes["member"]:
        fcn3 = fcn3.rename(member="fcn3_member")

    # --- G13: the baselines must reduce to compare.json's own ds numbers -------- #
    ds_ref = d.get("ds", {})
    worst = 0.0
    for key, da in (("gencast_xres", gencast_xres),
                    ("gencast_walkers", gencast_walkers), ("fcn3", fcn3)):
        if da is None or key not in ds_ref:
            continue
        worst = max(worst, _max_abs(f"{key} box", AI.area_mean(da, box).values,
                                    ds_ref[key]["box"]))
    ok_g13 = worst < GATE_TOL
    print(f"[apdfs] gate G13 max |direct A_L - compare.json ds| {worst:.2e}   "
          f"{'PASS' if ok_g13 else 'FAIL'}")

    # --- G5: one grid, compared as coordinate ARRAYS and not as shapes --------- #
    lat, lon = era5["lat"].values, era5["lon"].values
    for name, da in (("aires", aires_f), ("gencast_xres", gencast_xres),
                     ("gencast_walkers", gencast_walkers), ("fcn3", fcn3),
                     ("hrrr", hrrr)):
        if da is None:
            continue
        if not (np.array_equal(da["lat"].values, lat)
                and np.array_equal(da["lon"].values, lon)):
            raise RuntimeError(
                f"{name} is not on the ERA5 grid ({lat.size}x{lon.size}, "
                f"{lat[0]}..{lat[-1]} N). Pooling PDFs across two grids pools two "
                "different area weightings, silently.")

    # --- assemble -------------------------------------------------------------- #
    data = dict(
        aires=aires_f.astype("float32"),
        weight=xr.DataArray(weights, dims="walker"),
        a_box=xr.DataArray(a_box, dims="walker"),
        a_conus=xr.DataArray(a_conus, dims="walker"),
        lineage=xr.DataArray(
            np.array([[lineage[w][s] for s in ctx.segments]
                      for w in range(ctx.n_walkers)], dtype="int16"),
            dims=("walker", "segment"),
            coords={"segment": np.asarray(ctx.segments, dtype="int16")}),
        era5=era5.astype("float32"),
        gencast_xres=gencast_xres.astype("float32"),
    )
    for name, da in (("hrrr", hrrr), ("gencast_walkers", gencast_walkers),
                     ("fcn3", fcn3)):
        if da is not None:
            data[name] = da.astype("float32")
    ds = xr.Dataset(data)
    for v in ds.data_vars:
        ds[v].attrs.clear()

    # --- G6: a NaN in a stored field is a hole in every curve downstream -------- #
    for v in ds.data_vars:
        n_nan = int(np.count_nonzero(~np.isfinite(ds[v].values)))
        if n_nan:
            raise RuntimeError(f"{v} holds {n_nan} non-finite values")

    obs = (R.read_json(R.ds_path(ctx)).get("observed", {})
           if R.ds_path(ctx).exists() else {})
    ds.attrs = dict(
        event=ctx.event, tag=ctx.tag, metric=ctx.ev.metric,
        n_walkers=int(ctx.n_walkers),
        log_Z=float(result.log_Z), normalization_check=norm,
        sum_weights=float(weights.sum()), ess_weights=_ess(weights),
        observed_box=float(obs.get("box", d.get("observed", float("nan")))),
        observed_conus=float(obs.get("conus", float("nan"))),
        peak=str(ctx.ev.peak), init=str(ctx.ev.init),
        window_start=str(times[0]), window_end=str(times[-1]),
        window_frames=int(times.size), window_step_h=int(A.WALKER_STEP_H),
        window_segments=np.asarray(segs, dtype="int32"),
        box_name=box.name, box_label=box.label,
        box_lat=np.asarray(box.lat, dtype="float64"),
        box_lon=np.asarray(box.lon, dtype="float64"),
        conus_lat=np.asarray(AI.CONUS.lat, dtype="float64"),
        conus_lon=np.asarray(AI.CONUS.lon, dtype="float64"),
        gate_g1_max_abs=float(max(g1b, g1c)), gate_g4_max_abs=float(g4),
        gate_g13_max_abs=float(worst),
        git_rev=_git_rev(),
        source_compare_json=str(cp),
        source_compare_mtime=float(cp.stat().st_mtime),
        source_res_result_json=str(rp),
        created=str(pd.Timestamp.utcnow()),
        note=("Verification-week (the 7 days ending at the peak) T2m anomaly fields. "
              "`aires` is the resampled AI+RES population and `weight` its DMC "
              "importance weights exp(-V_K); `lineage` records which slot held each "
              "walker's ancestry at each segment. FCN3's window is 25 six-hourly frames "
              "against the walkers' 13 twelve-hourly ones - the established repo "
              "convention, kept so these numbers stay bit-identical to compare.json."),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f".nc.tmp.{os.getpid()}")
    ds.to_netcdf(tmp)
    os.replace(tmp, out_path)
    print(f"[apdfs] wrote {_rel(out_path)} "
          f"({out_path.stat().st_size / 1e6:.1f} MB; vars "
          f"{', '.join(sorted(ds.data_vars))})")

    # A weighted probability `compare.json`'s own curve already carries, printed so the
    # two can be lined up without opening either file.
    o = ds.attrs["observed_box"]
    if np.isfinite(o):
        p_self = float(np.sum(weights * (a_box >= o)) / np.sum(weights))
        print(f"[apdfs] weighted P(A_L >= observed {o:+.5f}) = {p_self:.6f} "
              f"self-normalized, {p_self * norm:.6f} Z-scaled")

    if not (ok_g1 and ok_g4 and ok_g13):
        raise SystemExit("[apdfs] BUILD FAILED a verification gate (G1/G4/G13 above)")
    return out_path


# --------------------------------------------------------------------------- #
# stage: figs
# --------------------------------------------------------------------------- #
def load_fields(ctx: R.Ctx) -> xr.Dataset:
    p = fields_path(ctx)
    if not p.exists():
        raise SystemExit(f"no {p}\n  Build it first:\n"
                         f"    PYTHONPATH=. python -m aires.apdfs --stage build "
                         f"--event {ctx.event} --tag {ctx.tag}")
    with xr.open_dataset(p) as h:
        ds = h.load()
    cp = compare_path(ctx)
    stale = float(ds.attrs.get("source_compare_mtime", 0.0))
    if cp.exists() and stale and cp.stat().st_mtime > stale:
        print(f"  WARNING: {_rel(cp)} is NEWER than the intermediate these figures are "
              f"drawn from. Re-run --stage build --force.", file=sys.stderr)
    return ds


def curve_entries(ds: xr.Dataset, region, *, with_hrrr: bool = False) -> list[dict]:
    """Every population the PDF figure draws, in draw order, cropped to ``region``.

    ``values`` keeps the leading member/walker axis so the weighted estimator can give a
    whole map one weight; ``weights`` is None for every unweighted curve.
    """
    from aires import aindex as AI

    def cropped(name):
        return None if name not in ds else AI.crop(ds[name], region).values

    era5 = cropped("era5")
    aires = cropped("aires")
    w = np.asarray(ds["weight"].values, dtype="float64")
    ess = float(ds.attrs["ess_weights"])
    n_w = int(ds.attrs["n_walkers"])

    out = [dict(key="era5", tag="ERA5 truth", values=era5, weights=None,
                label=f"ERA5 truth  (1 field, n={era5.size:,} pts)",
                style=dict(color=TRUTH_COLOR, ls="-", lw=2.8, zorder=6))]
    if with_hrrr and "hrrr" in ds:
        h = cropped("hrrr")
        out.append(dict(key="hrrr", tag="HRRR truth", values=h, weights=None,
                        label=f"HRRR truth  (1 field, n={h.size:,} pts)",
                        style=dict(color=HRRR_COLOR, ls="-", lw=1.8, zorder=5)))
    out.append(dict(
        key="aires_weighted", tag="AI+RES weighted", values=aires, weights=w,
        label=(f"AI+RES weighted by $e^{{-V_K}}$  (N={n_w}, ESS {ess:.1f}, "
               f"n={aires.size:,} pts)"),
        style=dict(color=RES_COLOR, ls="-", lw=2.8, zorder=7)))
    out.append(dict(
        key="aires_unweighted", tag="AI+RES unweighted", values=aires, weights=None,
        label=f"AI+RES resampled population, unweighted  (N={n_w}, "
              f"n={aires.size:,} pts)",
        style=dict(color=RES_COLOR, ls=(0, (1, 1.4)), lw=2.2, zorder=4)))
    for key, tag, color, ls, name in (
        ("gencast_xres", "GenCast direct", GENCAST_COLOR, (0, (5, 2)),
         "GenCast direct (xres ensemble)"),
        ("gencast_walkers", "Gate-3 walkers", G3_COLOR, (0, (6, 2, 1, 2)),
         "GenCast direct (free-running Gate 3 walkers)"),
        ("fcn3", "FCN3 direct", FCN3_COLOR, (0, (4, 2)), "FCN3 direct"),
    ):
        v = cropped(key)
        if v is None:
            continue
        out.append(dict(key=key, tag=tag, values=v, weights=None,
                        label=f"{name}  ({v.shape[0]} members, n={v.size:,} pts)",
                        style=dict(color=color, ls=ls, lw=2.0, zorder=3)))
    return out


def _inset_map(fig, ax, ds: xr.Dataset, region):
    """A small CONUS locator: the ERA5 anomaly, with the scored region dashed.

    Placed with ``fig.add_axes`` and not ``ax.inset_axes``: matplotlib's
    ``add_child_axes`` repoints the child's ``.axes`` at the PARENT, and cartopy resolves
    ``transform=`` through exactly that attribute, so an inset GeoAxes raises
    "Axes should be an instance of GeoAxes" on its first ``pcolormesh``. Call this AFTER
    ``tight_layout``, since the position is read off the host axes.
    """
    import cartopy.crs as ccrs
    from gencast_s2s import config as GC
    from gencast_s2s.plotting import _to_180, _add_conus_features

    pos = ax.get_position()
    ia = fig.add_axes([pos.x0 + 0.020 * pos.width, pos.y0 + 0.022 * pos.height,
                       0.28 * pos.width, 0.29 * pos.height],
                      projection=ccrs.PlateCarree())
    ia.set_extent(GC.CONUS_EXTENT, ccrs.PlateCarree())
    dd = _to_180(ds["era5"])
    v = float(np.nanpercentile(np.abs(dd.values), 99.5)) or 1.0
    ia.pcolormesh(dd.lon, dd.lat, dd.values, transform=ccrs.PlateCarree(),
                  cmap="RdBu_r", vmin=-v, vmax=v, shading="auto", alpha=0.9)
    _add_conus_features(ia)
    lo0, lo1 = float(region.lon[0]), float(region.lon[1])
    lo0 = lo0 - 360 if lo0 > 180 else lo0
    lo1 = lo1 - 360 if lo1 > 180 else lo1
    ia.plot([lo0, lo1, lo1, lo0, lo0],
            [region.lat[0], region.lat[0], region.lat[1], region.lat[1], region.lat[0]],
            transform=ccrs.PlateCarree(), color="#111111", lw=1.7, ls="--", zorder=6)
    ia.set_title(f"dashed: scored region {region.name} ({region.label});  "
                 f"shading: ERA5", fontsize=6.2, pad=2.0)
    return ia


def plot_pdf(ctx: R.Ctx, ds: xr.Dataset, region_key: str, *, z_scaled: bool = False,
             with_hrrr: bool = False) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    region = _region(ctx.event, region_key)
    entries = curve_entries(ds, region, with_hrrr=with_hrrr)
    by_key = {e["key"]: e for e in entries}
    znorm = float(ds.attrs["normalization_check"])

    xmin, xmax = shared_range([by_key[k]["values"] for k in BIN_RANGE_SOURCES
                               if k in by_key])

    fig, ax = plt.subplots(figsize=(11.6, 7.6))
    rows = []
    for e in entries:
        mode = "z" if (z_scaled and e["weights"] is not None) else "selfnorm"
        pts, dens = weighted_pdf(e["values"], e["weights"], xmin=xmin, xmax=xmax,
                                 nbins=NBINS, mode=mode, z_factor=znorm)
        st = dict(e["style"])
        if e["key"] == "aires_weighted":
            st["path_effects"] = [pe.Stroke(linewidth=st["lw"] + 1.8,
                                            foreground="white"), pe.Normal()]
        # An empty bin BREAKS the curve instead of crawling along the floor: a gap is an
        # honest "nothing sampled here"; a line at 1e-6 is a probability nobody measured.
        y = np.where(dens > 0, np.clip(dens, YMIN, None), np.nan)
        ax.plot(pts, y, label=e["label"], **st)
        rows.append((e["tag"], tail_probability(pts, dens), float(np.nansum(dens))))

    # Echoed so the run log carries the same numbers the figure does: the shared bin
    # range, each curve's mass inside it (a curve well under 1 is a clipped range) and
    # the tail the table shows.
    print(f"  {region_key}: {NBINS} shared bins over [{xmin:+.4f}, {xmax:+.4f}] K")
    for tag, p, m in rows:
        print(f"    {tag:22s} mass {m:.5f}   "
              f"P(point >= {TAIL_THRESHOLD:g} K) {p:.5f}")

    obs = float(ds.attrs[f"observed_{region_key}"])
    if np.isfinite(obs):
        ax.axvline(obs, color=OBS_COLOR, lw=1.5, ls="--", zorder=8)
        ax.text(obs, 0.02, f"  ERA5 observed $A_L$ = {obs:+.2f} K", color=OBS_COLOR,
                fontsize=8.5, transform=ax.get_xaxis_transform(), va="bottom",
                ha="left", rotation=90)

    ax.axvline(0, color="grey", lw=0.8, alpha=0.6, zorder=1)
    ax.grid(True, which="both", ls="--", color="0.85", lw=0.5)
    ax.set_yscale("log")
    ax.set_ylim(YMIN, 1.0)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("verification-week mean T2m anomaly (K)")
    ax.set_ylabel("PDF (probability per bin)")
    ax.set_title(
        f"{ctx.ev.label} ({ctx.tag}): spatial PDF of the verification-week T2m anomaly "
        f"over {region.name}\n{ds.attrs['window_start']} .. {ds.attrs['window_end']} "
        f"({ds.attrs['window_frames']} frames at {ds.attrs['window_step_h']} h);  "
        f"{NBINS} shared bins over [{xmin:+.2f}, {xmax:+.2f}] K", fontsize=10.5)
    ax.legend(fontsize=7.6, loc="upper left", framealpha=0.94)

    # The tail the figure is actually about - computed from the curves that were drawn,
    # never hand-typed.
    ref = dict((t, p) for t, p, _m in rows).get("ERA5 truth", float("nan"))
    lines = [f"P(point >= {TAIL_THRESHOLD:g} K)         p    x ERA5",
             "-" * 40]
    for tag, p, _m in rows:
        r = (p / ref) if ref else float("nan")
        lines.append(f"{tag:22s} {p:7.4f}   {r:6.2f}")
    ax.text(0.985, 0.975, "\n".join(lines), transform=ax.transAxes, ha="right",
            va="top", fontsize=7.0, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.93),
            zorder=9)

    scaled = ("The AI+RES weighted curve is Z-SCALED here (--z-scaled): "
              r"$p_b = Z\,\overline{w}\;\sum_i w_i n_{ib} / (G \sum_i w_i)$" "\n"
              f"so its total mass is Z*mean(w) = {znorm:.4f}, not 1.")
    plain = ("AI+RES weighted curve is SELF-NORMALIZED:  "
             r"$p_b = \sum_i w_i n_{ib} / (G \sum_i w_i)$" ",  total mass 1.\n"
             f"The DMC estimator's own factor Z*mean(w) = {znorm:.4f} would shift the "
             f"whole curve down by {abs(np.log10(znorm)):.3f} decades (not applied).")
    fig.text(0.5, 0.082, scaled if z_scaled else plain, ha="center", va="bottom",
             ma="center", fontsize=8.2, color="0.25")
    fig.text(0.5, 0.008,
             f"Weight ESS is {ds.attrs['ess_weights']:.1f} of "
             f"{int(ds.attrs['n_walkers'])} walkers and the heaviest weight belongs to "
             f"the population's coldest member, so the weighted curve's mode\n"
             f"and left flank are a small-sample artifact - the right tail is what this "
             f"figure is for.  FCN3's window is 25 six-hourly frames against the\n"
             f"walkers' 13 twelve-hourly ones, the established repo convention, kept so "
             f"these numbers stay bit-identical to compare.json.",
             ha="center", va="bottom", ma="center", fontsize=7.6, style="italic",
             color="0.35")
    fig.tight_layout(rect=[0, 0.145, 1, 1])
    _inset_map(fig, ax, ds, region)          # after tight_layout: reads ax's position
    return AP._save(fig, AP._fig_path(ctx, f"pdf_{region_key}"))


def plot_anom_maps(ctx: R.Ctx, ds: xr.Dataset) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    from aires import adaptest as AT
    from aires import aindex as AI

    w = xr.DataArray(np.asarray(ds["weight"].values, dtype="float64"), dims="walker")
    comp = (ds["aires"] * w).sum("walker") / w.sum()
    gc = ds["gencast_xres"].mean("member")
    era5 = ds["era5"]
    lim = AT._sym(era5.values, comp.values, gc.values)
    region = AI.box_for(ctx.event)
    box = None if region is AI.CONUS else region
    n_w = int(ds.attrs["n_walkers"])
    ess = float(ds.attrs["ess_weights"])

    fig = plt.figure(figsize=(15.0, 4.9))
    panels = [
        (era5, "ERA5 truth - the field that actually happened"),
        (comp, f"AI+RES weighted composite  $\\sum_i w_i f_i / \\sum_i w_i$  "
               f"(N={n_w}, ESS {ess:.1f})"),
        (gc, f"GenCast direct - {ds.sizes['member']}-member ensemble mean"),
    ]
    for i, (da, title) in enumerate(panels, start=1):
        AT._draw(fig, (1, 3, i), da, title, cmap="RdBu_r", vmin=-lim, vmax=lim,
                 label="verification-week mean T2m anomaly (K)", ccrs=ccrs, box=box)

    fig.suptitle(
        f"AI+RES pilot - {ctx.event} ({ctx.tag}): the three headline distributions as "
        f"maps, on one shared symmetric scale (+/-{lim:.1f} K, robust 99.5th pct).\n"
        f"Dashed box = the scored region, {region.name} ({region.label}). The composite "
        f"is a weighted MEAN whose ESS is {ess:.1f} of {n_w} walkers - read it as where "
        f"the weight sits, not as a forecast.", fontsize=9.6, y=0.99, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    return AP._save(fig, AP._fig_path(ctx, "anom_maps"))


def figs(ctx: R.Ctx, want, *, z_scaled: bool = False, with_hrrr: bool = False) -> None:
    ds = load_fields(ctx)
    print(f"[apdfs] figs: {ctx.event} tag={ctx.tag}: {', '.join(want)}")
    if "pdf_conus" in want:
        plot_pdf(ctx, ds, "conus", z_scaled=z_scaled, with_hrrr=with_hrrr)
    if "pdf_box" in want:
        plot_pdf(ctx, ds, "box", z_scaled=z_scaled, with_hrrr=with_hrrr)
    if "anom_maps" in want:
        plot_anom_maps(ctx, ds)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=STAGES)
    ap.add_argument("--event", default=os.environ.get("AIRES_EVENT", "PNW_HeatDome_2021"))
    ap.add_argument("--tag", default=A.RES_TAG)
    ap.add_argument("--only", default=",".join(FIGURES),
                    help=f"comma-separated subset of {FIGURES}")
    ap.add_argument("--force", action="store_true",
                    help="rebuild the intermediate even if it is up to date")
    ap.add_argument("--z-scaled", action="store_true",
                    help="draw the weighted curve on the DMC estimator's own scale "
                         "(x normalization_check) instead of self-normalized")
    ap.add_argument("--with-hrrr", action="store_true",
                    help="add the HRRR truth curve where the overlay exists")
    a = ap.parse_args(argv)

    ctx = R.Ctx(ev=R._event(a.event), tag=a.tag, n_walkers=A.RES_N_WALKERS,
                members=A.RES_M_MEMBERS, C=A.RES_C, leads=A.RES_LEAD_DAYS,
                horizon_days=A.RES_HORIZON_DAYS, dmc_seed=A.RES_DMC_SEED,
                base_seed=A.WALKER_BASE_SEED)

    want = [x.strip() for x in a.only.split(",") if x.strip()]
    bad = [x for x in want if x not in FIGURES]
    if bad:
        raise SystemExit(f"unknown figure(s) {bad}; want a subset of {list(FIGURES)}")

    if a.stage in ("build", "all"):
        build(ctx, force=a.force)
    if a.stage in ("figs", "all"):
        figs(ctx, want, z_scaled=a.z_scaled, with_hrrr=a.with_hrrr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
