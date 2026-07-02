#!/usr/bin/env python
"""GenCast S2S extreme-weather experiment driver.

One experiment = one forecast horizon (week2 = 14d, week3 = 21d, week4 = 28d). Each runs
the same code; only the initialization date / lead changes. Stages:

    prep   download checkpoint+stats, build init frames + observed truth into runs/ folders
    infer  run the 24-member GenCast ensemble on all 8 GPUs, save week-mean T2m anomaly
    plot   PDFs + cartopy CONUS maps (obs|mean|error) + CRPS / rank histograms
    all    prep -> infer -> plot   (default)

Examples
--------
    # one-time, on a node with internet (login node):
    python run_experiment.py --weeks 2 --stage prep
    # on the 8xH100 node (submitted via slurm/gencast_week2.slurm):
    python run_experiment.py --weeks 2 --stage infer
    python run_experiment.py --weeks 2 --stage plot
"""
from __future__ import annotations

import argparse

from gencast_s2s import config as C


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weeks", type=int, required=True, choices=C.VALID_WEEKS,
                   help="forecast horizon: 2 (14d), 3 (21d), or 4 (28d)")
    p.add_argument("--stage", choices=["prep", "infer", "plot", "combined", "all"],
                   default="all")
    p.add_argument("--members", type=int, default=None,
                   help=f"ensemble size (default {C.N_MEMBERS}; rounded up to a multiple of #GPUs)")
    args = p.parse_args(argv)

    C.ensure_dirs(args.weeks)

    if args.stage in ("prep", "all"):
        from gencast_s2s import download
        download.prepare_all(args.weeks)

    if args.stage in ("infer", "all"):
        from gencast_s2s import inference
        inference.run_all(args.weeks, n_members=args.members)

    if args.stage in ("plot", "all"):
        from gencast_s2s import plotting
        plotting.make_all(args.weeks)

    # The combined-PDF root figures overlay all horizons, so they read every week's
    # forecasts rather than just --weeks. Run after the per-week plots exist:
    #   python run_experiment.py --weeks 4 --stage combined
    if args.stage in ("combined", "all"):
        from gencast_s2s import combined_pdfs
        combined_pdfs.make_all()


if __name__ == "__main__":
    main()
