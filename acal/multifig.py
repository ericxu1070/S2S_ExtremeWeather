#!/usr/bin/env python
"""Multi-model versions of the AI+RES-vs-CFSv2 figures. CPU ONLY, LOGIN NODE.

The CFSv2-only figures (`acal.reach`, `acal.sidebyside`, `cfsbase.fig_paired` /
`cfsbase.fig_scorecard`) put AI+RES next to ONE baseline. This module and its four stage
modules put AI+RES next to every baseline the board scores, in ONE figure per type:

    stage     module               replaces (CFSv2-only)
    reach     acal.multi_reach     reach/prob_4K.png, reach/reach_4K.png, reach/reach_<case>.png
    csi       acal.multi_csi       sidebyside/csi_{heat,cold}_{2,3,4}K.png
    members   acal.multi_members   sidebyside/closest_member_summary.png, sidebyside/members/*
    paired    acal.multi_paired    acal_cfs_paired.png, acal_cfs_scorecard.png

Conventions every stage follows
-------------------------------
* Baselines are `models()`: the board sources with data (`s2sbase.BOARD_SOURCES` filtered by
  `s2sbase.has_data`), in board order. Nothing hard-codes a source list.
* CFSv2 is `cfs13` (the 13 00Z/12Z frames, the board's CFSv2 row), not the 25-frame `cfs`.
* Raw AND bias-corrected baselines appear in the same figure. A source keeps one hue:
  bias-corrected = `color(s)`, raw = `color(s, raw=True)` (the same hue darkened by
  `s2sbase.shade`). Markers: corrected squares, raw triangles, AI+RES circles.
* Each baseline is paired with AI+RES and with the truth on ITS window (`Source.obs_window`):
  13 frames for instant sources (CFSv2, GEFSv12), 12 frames for daily-mean sources (GEPS,
  EC46), with AI+RES re-reduced on the same 12 frames. Inputs are the per-source tables
  under runs/acal/analysis/s2s/era5/ (read-only here); anything new is cached under
  `DATA_DIR`.
* Truth is ERA5 (the CFSv2-only figures were ERA5 only; HRRR agrees to r = 0.9995).
* Every figure footer carries `s2sbase.SELECTION_NOTE` and `s2sbase.TILT_NOTE`.

    python -m acal.multifig --stage reach|csi|members|paired|all
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import ccfg
from acal import s2sbase as S2

TRUTH = "era5"
FIG_DIR = ccfg.FIG_ROOT / "s2s" / TRUTH / "multi"
DATA_DIR = AN.OUT / "multi"
TABLES = S2.ANALYSIS / TRUTH                       # per-source tables, read-only here
SHORT = {"cfs13": "CFSv2", "ec46": "EC46", "gefs": "GEFSv12", "geps": "GEPS"}
MARKER = {"aires": "o", "corr": "s", "raw": "^"}
FOOT = f"{S2.SELECTION_NOTE}. {S2.TILT_NOTE}"
STAGES = {"reach": "acal.multi_reach", "csi": "acal.multi_csi",
          "members": "acal.multi_members", "paired": "acal.multi_paired"}


def excluded() -> set[str]:
    """Sources left out on purpose: `ACAL_MULTI_EXCLUDE=ec46` (comma list), e.g. a baseline
    whose scores are still being rebuilt. `--exclude` on the CLI sets the same variable, so
    every stage module sees it through `models()`."""
    return {s.strip() for s in os.environ.get("ACAL_MULTI_EXCLUDE", "").split(",") if s.strip()}


def models() -> list[str]:
    """Board baselines with data, board order (cfs13, ec46, gefs, geps), minus `excluded()`."""
    return [s for s in S2.BOARD_SOURCES if S2.has_data(s) and s not in excluded()]


def by_window(window: str, sep: str = ", ") -> str:
    """Short names of the active baselines verified on `window` ('13f' or '12f'), for figure
    text that must not name a baseline the figure leaves out."""
    return sep.join(short(s) for s in models() if S2.get(s).obs_window == window)


def label(source: str) -> str:
    """Full board label ('ECMWF IFS (EC46)'); `short` for legends and row labels."""
    return S2.style(source)["label"]


def short(source: str) -> str:
    return SHORT.get(source, label(source))


def color(source: str, raw: bool = False) -> str:
    c = S2.style(source)["color"]
    return S2.shade(c) if raw and source != "aires" else c


def table(source: str, kind: str) -> Path:
    """runs/acal/analysis/s2s/era5/<source>_<kind>.csv (kind: scorecard, paired, prob_4K)."""
    return TABLES / f"{source}_{kind}.csv"


def save(fig, name: str, dpi: int = 140) -> Path:
    """Write figures/acal/s2s/era5/multi/<name> atomically (tmp file, then rename)."""
    import matplotlib.pyplot as plt
    p = FIG_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.png")
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", facecolor="white")
    tmp.replace(p)
    plt.close(fig)
    print(f"  wrote {p} ({p.stat().st_size / 1e3:.0f} kB)", flush=True)
    return p


def run(stage: str) -> list[Path]:
    """A stage with its own CLI gets an empty argv (its defaults), never this module's flags."""
    main_ = importlib.import_module(STAGES[stage]).main
    return main_([]) if "argv" in inspect.signature(main_).parameters else main_()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    p.add_argument("--exclude", default=None,
                   help="comma list of baselines to leave out (sets ACAL_MULTI_EXCLUDE)")
    a = p.parse_args(argv)
    if a.exclude is not None:
        os.environ["ACAL_MULTI_EXCLUDE"] = a.exclude
    print(f"[multifig] baselines: {', '.join(models())}", flush=True)
    for s in (STAGES if a.stage == "all" else (a.stage,)):
        run(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
