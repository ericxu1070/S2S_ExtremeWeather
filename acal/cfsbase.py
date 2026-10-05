#!/usr/bin/env python
"""CFSv2 operational baseline for the calibration campaign. LOGIN NODE (internet, CPU).

The question: on the same 42 cases, at the same 21 d lead, with the same CONUS `A_L`
reduction, does AI+RES put more probability on the observed tail than NCEP's operational
CFSv2? `acal/CFS_PLAN.md` is the plan; this module is the code.

The ensemble
------------
16 trailing 6-hourly cycles ending ON the AI+RES init (leads 21.0-24.75 d), built with
`aires.cfs.build` - the same fetch, regrid and `aires.aindex` reduction that produced the
aires CFS table. No member is at a shorter lead than the walkers. The 4 cycles nearest
the init (the aires convention) are the last 4 members and are reported as a subset.

Cubes go to `runs/acal/cfs/<case>_cfs16.nc`, NOT `aconfig.cfs_cube_path`: that path does
not encode the cycle count, and a 16-member cube there would replace the 4-member one.

The anomaly is CFSv2 minus the ERA5 1990-2019 climatology, uncorrected, exactly as the
walkers are scored. See the `aires/cfs.py` docstring before quoting a number.

    python -m acal.cfsbase --stage build              # all 42 cases (~45 min, ~6 GB)
    python -m acal.cfsbase --stage build --case e02_c4_20210218
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import aprep, ccfg
from aires import cfs

N_CYCLES = 16
N_SUBSET = 4                             # the aires convention: init-18h .. init
MIN_MEMBERS = 12                         # fewer is flagged, not dropped
CFS_ROOT = ccfg.ACAL_ROOT / "cfs"
BUILD_CSV = CFS_ROOT / "build.csv"


def cube_path(eid: str) -> Path:
    return CFS_ROOT / f"{eid}_cfs{N_CYCLES}.nc"


def json_path(eid: str) -> Path:
    return CFS_ROOT / f"{eid}_cfs{N_CYCLES}.json"


def build_case(row, force: bool = False, keep_grib: bool = False) -> dict:
    """Build (or reuse) one case's cube and return its per-member CONUS A_L record."""
    eid = row.episode_id
    work = CFS_ROOT / ".grib" / eid
    out = cfs.build(eid, n_cycles=N_CYCLES, mode="trailing", force=force,
                    workdir=work, out=cube_path(eid))
    with xr.open_dataset(out) as cube:
        d = cfs.indices(eid, cube)
        attrs = dict(cube.attrs)
    # acal cases are CONUS-selected, so aindex.box_for falls back to CONUS and the "box"
    # and "conus" reductions must be the same numbers. If they ever differ, the case was
    # scored on some other box and nothing downstream would mean what it says.
    if not np.allclose(d["box"], d["conus"], atol=1e-6):
        raise SystemExit(f"[cfsbase] {eid}: box A_L != CONUS A_L - not a CONUS case?")
    if not keep_grib:
        shutil.rmtree(work, ignore_errors=True)

    al = np.asarray(d["conus"], dtype=float)
    s = 1.0 if row.family == "heat" else -1.0
    obs = float(row.a_l_conus)
    rec = dict(
        episode_id=eid, family=row.family, obs=obs, sign=s,
        n_members=int(al.size), cycles=d["cycles"],
        member_lead_days=d["member_lead_days"], al=al.tolist(),
        al_mean=float(al.mean()), al_std=float(al.std(ddof=1)),
        al_sub_mean=float(al[-N_SUBSET:].mean()),
        n_reach=int(np.sum(s * al >= s * obs)),
        n_reach_sub=int(np.sum(s * al[-N_SUBSET:] >= s * obs)),
        init_anom=d["init_anom"],
        cycles_skipped=attrs.get("cycles_skipped", "none"),
        archives=attrs.get("archives", ""),
    )
    json_path(eid).write_text(json.dumps(rec, indent=2))
    flag = "" if al.size >= MIN_MEMBERS else f"  FLAG: only {al.size} members"
    print(f"[cfsbase] {eid}: obs {obs:+.2f}  CFS mean {rec['al_mean']:+.2f} "
          f"(sd {rec['al_std']:.2f})  reach {rec['n_reach']}/{al.size}{flag}")
    return rec


def build(cases: list[str] | None = None, force: bool = False,
          keep_grib: bool = False) -> pd.DataFrame:
    CFS_ROOT.mkdir(parents=True, exist_ok=True)
    df = aprep.episodes()
    if cases:
        df = df[df.episode_id.isin(cases)]
        if df.empty:
            raise SystemExit(f"[cfsbase] no such case(s): {cases}")
    recs = [build_case(r, force, keep_grib) for r in df.itertuples()]
    # Merge with any rows already on disk, so a --case rebuild does not truncate the table.
    tab = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("cycles", "member_lead_days", "al")} for r in recs])
    if BUILD_CSV.exists() and cases:
        old = pd.read_csv(BUILD_CSV)
        tab = pd.concat([old[~old.episode_id.isin(tab.episode_id)], tab])
    order = {e: i for i, e in enumerate(aprep.episodes().episode_id)}
    tab = tab.sort_values("episode_id", key=lambda c: c.map(order))
    tab.to_csv(BUILD_CSV, index=False)
    print(f"[cfsbase] wrote {BUILD_CSV} ({len(tab)} cases)")
    short = tab[tab.n_members < MIN_MEMBERS]
    if len(short):
        print(f"[cfsbase] {len(short)} case(s) under {MIN_MEMBERS} members: "
              f"{', '.join(short.episode_id)}")
    return tab


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("build",), default="build")
    p.add_argument("--case", action="append", help="episode id; repeatable")
    p.add_argument("--force", action="store_true", help="re-download and rebuild")
    p.add_argument("--keep-grib", action="store_true")
    a = p.parse_args(argv)
    if a.stage == "build":
        build(a.case, a.force, a.keep_grib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
