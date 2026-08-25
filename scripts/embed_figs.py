#!/usr/bin/env python
"""Inline the pilot PNGs into docs/aires_pilot.html as data URIs.

The Artifact CSP blocks every external host, so a published page can only carry images it
embeds. This rewrites the FIG_* placeholders in place from a source template and writes a
self-contained copy - the template keeps the placeholders so the substitution is
repeatable after a figure is regenerated.

    python scripts/embed_figs.py docs/aires_pilot.html <out.html>
"""
import base64
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIGS = {
    "FIG_TRAJECTORY":  "figures/aires/PNW_HeatDome_2021/aires_trajectory_PNW_HeatDome_2021_pilot.png",
    "FIG_EXCEEDANCE":  "figures/aires/PNW_HeatDome_2021/aires_exceedance_PNW_HeatDome_2021_pilot.png",
    "FIG_DIAGNOSTICS": "figures/aires/PNW_HeatDome_2021/aires_diagnostics_PNW_HeatDome_2021_pilot.png",
    "FIG_GENEALOGY":   "figures/aires/PNW_HeatDome_2021/aires_genealogy_PNW_HeatDome_2021_pilot.png",
    "FIG_COMPOSITE":   "figures/aires/PNW_HeatDome_2021/aires_composite_PNW_HeatDome_2021_pilot.png",
    "FIG_MAP_COMPARE": "figures/aires/PNW_HeatDome_2021/aires_map_compare_PNW_HeatDome_2021_pilot.png",
    "FIG_MAP_ERROR":   "figures/aires/PNW_HeatDome_2021/aires_map_error_PNW_HeatDome_2021_pilot.png",
    "FIG_MAP_WALK":    "figures/aires/PNW_HeatDome_2021/aires_map_walk_PNW_HeatDome_2021_pilot.png",
    "FIG_MAP_SPREAD":  "figures/aires/PNW_HeatDome_2021/aires_map_spread_PNW_HeatDome_2021_pilot.png",
}


def main(argv):
    src, out = Path(argv[1]), Path(argv[2])
    html = src.read_text()
    total = 0
    for token, rel in FIGS.items():
        p = REPO / rel
        if not p.exists():
            raise SystemExit(f"missing {p}")
        if token not in html:
            raise SystemExit(f"{src}: no placeholder {token!r} to fill")
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        html = html.replace(token, f"data:image/png;base64,{b64}")
        total += len(b64)
        print(f"  {token:16s} {p.stat().st_size/1e3:7.0f} kB -> {len(b64)/1e3:7.0f} kB b64")
    out.write_text(html)
    print(f"-> {out} ({out.stat().st_size/1e6:.2f} MB, limit 16 MB)")
    if out.stat().st_size > 15e6:
        raise SystemExit("too close to the 16 MB artifact ceiling")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
