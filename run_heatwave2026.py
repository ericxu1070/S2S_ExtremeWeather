#!/usr/bin/env python
"""Driver for the June-July 2026 heat-wave GenCast forecast (Vayuh report comparison).

Stages (see gencast_s2s/heatwave2026.py for the science):
    prep    login node + internet: init frames + observed daily truth
    infer   GPU (PBS): per-init daily-mean tmp2m ensemble cubes
    plot    login/CPU: Observed | Vayuh | GenCast figures + combined PDF

Examples
--------
    python run_heatwave2026.py --stage prep       # login node
    python run_heatwave2026.py --stage infer       # inside the PBS GPU job
    python run_heatwave2026.py --stage plot        # login node
"""
from __future__ import annotations

import argparse


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["prep", "infer", "plot"], required=True)
    p.add_argument("--members", type=int, default=None,
                   help="ensemble size (default HW2026_N_MEMBERS or 16)")
    args = p.parse_args(argv)

    from gencast_s2s import heatwave2026 as H

    if args.stage == "prep":
        H.prepare_all()
    elif args.stage == "infer":
        H.run_infer(n_members=args.members)
    elif args.stage == "plot":
        from gencast_s2s import heatwave2026_plot as P
        P.make_all()


if __name__ == "__main__":
    main()
