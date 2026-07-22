#!/usr/bin/env python
"""Build a tiny, fully-synthetic sandbox for an END-TO-END smoke of the p90 compare stage.

The offline compare pipeline (``p90.pmetrics.compute_all`` + ``p90.pcompare.make_all``) is
pure CPU: it reads per-case forecast cubes (CUBE SCHEMA, see ``pconfig``), the ERA5 truth /
persistence anomaly files, and the 1990-2019 T2m climatology, and writes scores + figures.
This script fabricates all of those inputs on a coarse 2.0 deg "sandbox grid" so the whole
chain can be exercised on a login node with no network, no GPU, and no reads under the real
``runs/`` tree.

Everything is keyed off ``GENCAST_RUNS`` (point it at ``<SB>/runs`` before running), so all
paths flow through ``pconfig`` exactly as the real stages see them. Rerunnable: it overwrites
its own outputs. Selects the first 2 event cases + the first 1 control case from
``p90/cases.csv`` and, for each, writes:

  * ``cube_path("fcn3", case)``   - 8-member, 88-frame cube on the sandbox grid, values
        280 + N(0,2); EVENT cases get a +3 K bump on the last 28 frames (the whole
        verification window) so p_event and the event/control contrast are nontrivial.
  * ``truth_path(case)``          - week-mean T2m anomaly truth, N(0,1.5) + 2.5 K for events.
  * ``persist_path(case)``        - persistence-baseline anomaly, same recipe as truth.

Plus one shared climatology at ``C.CLIM_FILE`` (= ``<SB>/runs/models/...``): a constant
280.0 K field on (hour, dayofyear, lat, lon) so every anomaly is just "value - 280".

Run (from repo root):
  GENCAST_RUNS=<SB>/runs P90_N_MEMBERS=8 PYTHONPATH=. python p90/tests/make_smoke_sandbox.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr

from gencast_s2s import config as C
from p90 import pconfig as P

# Coarse sandbox grid (the real grid is 0.25 deg 105x237; here 2.0 deg keeps files tiny).
LAT = np.arange(24.0, 50.01, 2.0)
LON = np.arange(235.0, 294.01, 2.0)
BUMP_EVENT_K = 3.0            # forecast bump on the verification window for event cases
BUMP_TRUTH_K = 2.5           # truth-field bump for event cases
BUMP_FRAMES = 28             # last 28 of the 88 frames == the verification window


def _build_clim() -> None:
    """Constant 280.0 K climatology on (hour, dayofyear, lat, lon), sandbox grid."""
    hours = np.array([0, 6, 12, 18])
    doy = np.arange(1, 367)
    data = np.full((len(hours), len(doy), len(LAT), len(LON)), 280.0, dtype="float32")
    da = xr.DataArray(
        data, dims=("hour", "dayofyear", "lat", "lon"),
        coords={"hour": hours, "dayofyear": doy, "lat": LAT, "lon": LON},
        name="2m_temperature",
    )
    C.CLIM_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = C.CLIM_FILE.with_suffix(".nc.tmp")
    da.to_dataset(name="2m_temperature").to_netcdf(tmp)
    os.replace(tmp, C.CLIM_FILE)
    print(f"[smoke] wrote clim {C.CLIM_FILE}  {dict(da.sizes)}")


def _build_cube(row: pd.Series) -> None:
    name = row["case"]
    kind = str(row["kind"])
    idx = int(row["idx"])
    init = pd.Timestamp(row["init"])
    peak = pd.Timestamp(row["peak"])
    M = P.N_MEMBERS
    times = pd.date_range(init, periods=P.NSTEPS + 1, freq=f"{P.STEP_H}h")   # 88 frames
    rng = np.random.default_rng(P.seed_for(idx))
    data = 280.0 + rng.normal(0.0, 2.0, size=(M, len(times), len(LAT), len(LON)))
    if kind == "event":
        data[:, -BUMP_FRAMES:, :, :] += BUMP_EVENT_K
    data = data.astype("float32")
    cube = xr.Dataset(
        {"2m_temperature": (("member", "time", "lat", "lon"), data)},
        coords={"member": np.arange(M), "time": times, "lat": LAT, "lon": LON},
    )
    cube.attrs.update(
        case=name, kind=kind, init=init.isoformat(), peak=peak.isoformat(),
        model="fcn3", members=int(M), nsteps=int(P.NSTEPS), seed=int(P.seed_for(idx)),
    )
    cpath = P.cube_path("fcn3", name)
    tmp = cpath.with_suffix(".nc.tmp")
    cube.to_netcdf(tmp)
    os.replace(tmp, cpath)
    print(f"[smoke] wrote cube {cpath}  {dict(cube.sizes)} kind={kind}")


def _build_truth(row: pd.Series) -> None:
    name = row["case"]
    kind = str(row["kind"])
    idx = int(row["idx"])
    init = pd.Timestamp(row["init"])
    peak = pd.Timestamp(row["peak"])
    for path, sub_seed in ((P.truth_path(name), 0), (P.persist_path(name), 1)):
        rng = np.random.default_rng(P.seed_for(idx) + 10_000 + sub_seed)
        anom = rng.normal(0.0, 1.5, size=(len(LAT), len(LON)))
        if kind == "event":
            anom += BUMP_TRUTH_K
        da = xr.DataArray(
            anom.astype("float32"), dims=("lat", "lon"),
            coords={"lat": LAT, "lon": LON}, name="t2m_anom",
        )
        ds = da.to_dataset(name="t2m_anom")
        ds.attrs.update(case=name, peak=peak.isoformat(), init=init.isoformat())
        tmp = path.with_suffix(".nc.tmp")
        ds.to_netcdf(tmp)
        os.replace(tmp, path)
        print(f"[smoke] wrote {path.name}  {dict(da.sizes)}")


def _guard_sandboxed() -> None:
    """Refuse to write anywhere near the real data tree.

    Everything below keys off pconfig paths, so running without GENCAST_RUNS (or with it
    pointing at the repo's real runs/) would plant synthetic cubes/truth in the REAL
    experiment tree - and the cache-aware stages would then silently skip those cases.
    """
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(f"make_smoke_sandbox.py takes no arguments (got {sys.argv[1:]}); "
                         "point GENCAST_RUNS at the sandbox instead - see the docstring")
    env = os.environ.get("GENCAST_RUNS")
    if not env:
        raise SystemExit("GENCAST_RUNS is not set - refusing to write into the real runs/ "
                         "tree. Run: GENCAST_RUNS=<SB>/runs PYTHONPATH=. python "
                         "p90/tests/make_smoke_sandbox.py")
    real = (C.ROOT / "runs").resolve()
    if C.RUNS.resolve() == real or real in C.RUNS.resolve().parents:
        raise SystemExit(f"GENCAST_RUNS={env} resolves inside the real runs/ tree ({real}) - "
                         "refusing; use a scratch directory")


def main() -> None:
    _guard_sandboxed()
    P.ensure_dirs("fcn3")
    _build_clim()

    df = P.cases()   # unfiltered here (run WITHOUT P90_CASES_SEL so we can pick our own set)
    events = df[df["kind"] == "event"].head(2)
    controls = df[df["kind"] == "control"].head(1)
    picked = pd.concat([events, controls])
    names = list(picked["case"])
    print(f"[smoke] sandbox root GENCAST_RUNS={C.RUNS}")
    print(f"[smoke] picked {len(names)} cases: {', '.join(names)}")

    for _, row in picked.iterrows():
        _build_cube(row)
        _build_truth(row)

    print(f"[smoke] DONE. Run compare with:\n"
          f"  P90_CASES_SEL={','.join(names)}")


if __name__ == "__main__":
    main()
