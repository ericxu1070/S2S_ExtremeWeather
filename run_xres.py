#!/usr/bin/env python
"""Cross-resolution GenCast S2S experiment driver (0.25deg vs 1.0deg, week-2 only).

Stages:
    prep     download both checkpoints + stats/statics/clim; build ERA5 + HRRR observed
             truth (all metrics); build per-resolution init frames.            [CPU]
    infer    roll one resolution's ensemble, CACHE THE FULL output cube per event, derive
             the per-member verification metrics.  Requires --res.             [8x GPU node]
    compare  build the 6-panel comparison maps + the 3 combined PDFs from BOTH
             resolutions' verification fields + ERA5/HRRR truth.               [CPU]

Examples
--------
    # 0) login node (internet, CPU): prep everything for both resolutions
    python run_xres.py --stage prep

    # 1) one GPU node per resolution (submitted via slurm/)
    python run_xres.py --stage infer --res 0p25
    python run_xres.py --stage infer --res 1p0

    # 2) after both infer jobs finish (login node, CPU)
    python run_xres.py --stage compare
"""
from __future__ import annotations

import argparse

from xres import xconfig as X


def do_prep(res_list, weeks):
    from xres import xdata, xhrrr
    xdata.download_models()
    xdata.build_era5_truth()
    xhrrr.build_hrrr_truth()
    for res in res_list:
        xdata.build_inputs(res, weeks)


def do_infer(res, weeks, members):
    from xres import xinference
    xinference.run_all(res, weeks, n_members=members)


def do_compare(weeks):
    from xres import xplotting, xcombined
    xplotting.make_maps(weeks)
    xcombined.make_all(weeks)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["prep", "infer", "compare", "all"], default="all")
    p.add_argument("--res", choices=list(X.RES_SPECS), default=None,
                   help="resolution for infer (required for infer); prep defaults to both")
    p.add_argument("--weeks", type=int, default=X.WEEKS)
    p.add_argument("--members", type=int, default=None,
                   help="override ensemble size (default per-resolution from xconfig)")
    args = p.parse_args(argv)

    if args.stage in ("prep", "all"):
        res_list = [args.res] if args.res else list(X.RES_ORDER)
        do_prep(res_list, args.weeks)

    if args.stage in ("infer", "all"):
        if not args.res:
            p.error("--res is required for the infer stage")
        do_infer(args.res, args.weeks, args.members)

    if args.stage in ("compare", "all"):
        do_compare(args.weeks)


if __name__ == "__main__":
    main()
