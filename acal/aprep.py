#!/usr/bin/env python
"""Build the calibration cases' GenCast init frames. LOGIN NODE ONLY (needs internet).

`aires.walker.initial_state()` resolves a walker's step-0 state through
`aconfig.gencast_inputs_path(event)` -> `runs/xres/<res>/week<N>/inputs/<event>_inputs.nc`,
which is keyed by a NAMED event from `gencast_s2s.config`. A calibration case is an
arbitrary (box, peak) pair, so nothing in that path exists for it. This module closes the
gap without touching `aires/` or `xres/`: it builds each case's init with the same
primitive xres uses (`gencast_s2s.data.build_raw_inputs`, which already takes an arbitrary
peak and lead), writes the bytes into acal's OWN tree, and leaves a symlink where
`gencast_inputs_path` will find it.

    runs/acal/inputs/<case_id>_inputs.nc            <- the 673 MB of real data
    runs/xres/0p25/week3/inputs/<case_id>_inputs.nc -> symlink to the above

The symlink matters. `acal/__init__.py` promises this package "writes to its own tree",
and dropping 28 GB of calibration inits into the frozen xres experiment's input directory
would break that; a link costs nothing, keeps the bytes under `runs/acal/`, and makes
`rm -rf runs/acal` a complete uninstall (the dangling links are then obvious, not silent).

Why one subprocess per case
---------------------------
This login node has 15 GB of RAM and ~10 GB free. One 0.25 deg global 2-frame init is
673 MB on disk but costs several GB to assemble, and xarray/gcsfs do not hand it all back
promptly -- which is why `run_xres.py` defaults to `XRES_PREP_ISOLATE=1`. Building 42 in a
single process is how you get an OOM kill 30 cases in, with no partial credit. Each case
therefore runs in its own interpreter and the driver only sequences them.

    python -m acal.aprep                 # every case in cases (42)
    python -m acal.aprep --rung 3        # only |A_L| >= 3 K (12)
    python -m acal.aprep --case e14_h4_20230104
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gencast_s2s.data as D
from acal import ccfg
from aires import aconfig as A

EPISODES = ccfg.ACAL_ROOT / "catalog" / "conus_episodes_21d_2021_2025.csv"
INPUTS = ccfg.ACAL_ROOT / "inputs"


def episodes(rung: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(EPISODES)
    if rung is not None:
        df = df[df.a_l_conus.abs() >= rung].reset_index(drop=True)
    return df


def init_path(case_id: str) -> Path:
    return INPUTS / f"{case_id}_inputs.nc"


def link_path(case_id: str) -> Path:
    """Where `aires.walker.initial_state` will look for it."""
    return A.gencast_inputs_path(case_id)


def _link(case_id: str) -> None:
    src, dst = init_path(case_id), link_path(case_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        raise SystemExit(f"{dst} exists and is NOT a symlink -- refusing to clobber "
                         f"a real xres input file")
    dst.symlink_to(src)


def build_one(case_id: str, peak: str, *, lead_days: int = ccfg.LEAD_DAYS) -> Path:
    """One case's 2-frame global init. Idempotent; cached by file existence."""
    out = init_path(case_id)
    if out.exists():
        print(f"[prep] cached: {out.name}")
        _link(case_id)
        return out
    INPUTS.mkdir(parents=True, exist_ok=True)
    ds = D.build_raw_inputs(pd.Timestamp(peak), lead_days=lead_days, model="gencast",
                            statics=D.load_statics(), verbose=True,
                            res=0.25 if A.WALKER_RES == "0p25" else 1.0)
    D._atomic_to_netcdf(ds, out)
    _link(case_id)
    print(f"[prep] {case_id}: init=peak-{lead_days}d -> {out.name} "
          f"({out.stat().st_size / 2**20:.0f} MB)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", help="build exactly this case id, in THIS process")
    ap.add_argument("--rung", type=float, default=None, help="restrict to |A_L| >= this")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    df = episodes(a.rung)
    if a.case:                                  # the isolated-child path
        row = df[df.episode_id == a.case]
        if row.empty:
            raise SystemExit(f"no such case: {a.case}")
        build_one(a.case, row.iloc[0].peak)
        return 0

    todo = [r for r in df.itertuples() if not init_path(r.episode_id).exists()]
    have = len(df) - len(todo)
    print(f"[prep] {len(df)} cases, {have} cached, {len(todo)} to build "
          f"(~{len(todo) * 0.657:.0f} GB, ~{len(todo) * 3 / 60:.1f} h)")
    if a.dry_run:
        for r in todo:
            print(f"  {r.episode_id}  peak={r.peak}  init={r.init}")
        return 0

    t0 = time.time()
    failed = []
    for i, r in enumerate(todo, 1):
        print(f"\n[prep] ---- {i}/{len(todo)}  {r.episode_id}  peak={r.peak} "
              f"init={r.init}  ({(time.time() - t0) / 60:.0f} min elapsed) ----",
              flush=True)
        p = subprocess.run([sys.executable, "-m", "acal.aprep", "--case", r.episode_id],
                           cwd=str(Path(__file__).resolve().parents[1]))
        if p.returncode != 0:
            print(f"[prep] FAILED {r.episode_id} (rc={p.returncode})")
            failed.append(r.episode_id)
        else:
            _link(r.episode_id)

    ok = sum(1 for r in df.itertuples() if init_path(r.episode_id).exists())
    print(f"\n[prep] done: {ok}/{len(df)} inits present, {len(failed)} failed, "
          f"{(time.time() - t0) / 60:.0f} min")
    if failed:
        print("[prep] failed: " + " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
