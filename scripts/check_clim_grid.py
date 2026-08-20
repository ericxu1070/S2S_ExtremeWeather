#!/usr/bin/env python
"""Assert the T2m climatology is on the 0.25 degree CONUS grid, by GRID not by filename.

Two files with the name ``clim_1990_2019_t2m_conus.nc`` have existed in this project on
the same path but different grids: 105x237 (0.25 deg, 145.7 MB) on a3mega and 14x30
(~2 deg, 2.5 MB) on Derecho. Cubes are lat=105, lon=237 and every anomaly downstream
subtracts a climatology sampled on that grid, so the wrong file yields either a shape
hard-fail or -- worse -- silently wrong anomalies. Count-based sync verification reported
"ok" throughout the one time this bit (CLAUDE.md, 2026-08-17).

Split out of scripts/adaptest_prep.sh because the check there was an ``ncdump`` pipeline
and ncdump is not installed in the ``moe`` env on a3mega: under ``set -euo pipefail`` it
exited non-zero with empty output and killed the calling script with no message.
"""
import sys

import xarray as xr

EXPECT = (105, 237)


def main(path: str) -> int:
    with xr.open_dataset(path) as ds:
        got = (ds.sizes.get("lat"), ds.sizes.get("lon"))
    print(f"climatology grid: lat={got[0]} lon={got[1]}")
    if got != EXPECT:
        print(f"ERROR: {path} is not on the 0.25 degree grid "
              f"(expected lat={EXPECT[0]} lon={EXPECT[1]}).\n"
              f"  Every anomaly downstream would be silently wrong. Fix this first.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "runs/models/clim_1990_2019_t2m_conus.nc"))
