#!/usr/bin/env python
"""Inline the event-slate PNGs into the slate page as WebP data URIs.

The Artifact CSP blocks every external host, so a published page can only carry images it
embeds, and the rendered page must stay under 16 MB. Thirty-one matplotlib figures do not
fit as PNG, so each is re-encoded to WebP - lossless and lossy are both tried and the
smaller wins, which for line-art panels is usually lossless.

The template keeps `src="DATA"` placeholders and is matched on the img's alt text, so the
substitution is repeatable after any figure is regenerated and a renamed figure fails
loudly instead of silently landing under the wrong caption.

    python scripts/embed_slate_figs.py <template.html> <out.html>
"""
import base64
import io
import re
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "figures" / "aires"

EVENTS = {
    "Southwest Heat Wave": "Southwest_HeatWave_2020",
    "California Heat Wave": "California_HeatWave_2022",
    "Winter Storm Elliott": "WinterStorm_Elliott_2022",
    "Winter Storm Uri": "WinterStorm_Uri_2021",
    "South-Central Heat Dome": "SCentral_HeatDome_2023",
    "Pacific Northwest Heat Dome": "PNW_HeatDome_2021",
    "The Nationwide Warm Spell": "p90_20251224",
    "The Normal-Day Test": "p90_20240802",
    "The Below-Median Day": "p90_20231107",
}

ALT_TO_FIG = {
    "Every AI+RES walker across all ten runs": FIGS / "aires_walkers.png",
    "Persistence-scored control trajectories":
        FIGS / "PNW_HeatDome_2021/aires_trajectory_PNW_HeatDome_2021_persist.png",
    "Per-walker anomaly and probability for the persistence control":
        FIGS / "PNW_HeatDome_2021/aires_walkers_PNW_HeatDome_2021_persist.png",
}
for name, key in EVENTS.items():
    ALT_TO_FIG[f"AI+RES map comparison for {name}"] = FIGS / key / f"aires_map_compare_{key}_pilot.png"
    ALT_TO_FIG[f"AI+RES walker trajectories for {name}"] = FIGS / key / f"aires_trajectory_{key}_pilot.png"
    ALT_TO_FIG[f"Per-walker anomaly and probability for {name}"] = FIGS / key / f"aires_walkers_{key}_pilot.png"

LIMIT = 16e6


def to_webp(path: Path) -> bytes:
    """Smaller of lossless and quality-93 lossy. Line art usually wins lossless."""
    im = Image.open(path).convert("RGB")
    out = []
    for kw in ({"lossless": True, "method": 6}, {"quality": 93, "method": 6}):
        buf = io.BytesIO()
        im.save(buf, "WEBP", **kw)
        out.append(buf.getvalue())
    return min(out, key=len)


def main(argv):
    src, out = Path(argv[1]), Path(argv[2])
    html = src.read_text(encoding="utf-8")
    alts = re.findall(r'<img src="DATA" alt="([^"]*)"', html)
    if not alts:
        raise SystemExit(f"{src}: no `src=\"DATA\"` placeholders to fill")
    unknown = [a for a in alts if a not in ALT_TO_FIG]
    if unknown:
        raise SystemExit("no figure registered for alt text: " + "; ".join(unknown))

    for alt in alts:
        p = ALT_TO_FIG[alt]
        if not p.exists():
            raise SystemExit(f"missing {p}")
        webp = to_webp(p)
        b64 = base64.b64encode(webp).decode("ascii")
        html = html.replace(f'<img src="DATA" alt="{alt}"',
                            f'<img src="data:image/webp;base64,{b64}" alt="{alt}"', 1)
        print(f"  {p.relative_to(FIGS)!s:62s} {p.stat().st_size/1e3:6.0f} kB png"
              f" -> {len(webp)/1e3:6.0f} kB webp")

    out.write_text(html, encoding="utf-8")
    size = out.stat().st_size
    print(f"-> {out} ({size/1e6:.2f} MB of a {LIMIT/1e6:.0f} MB ceiling, {len(alts)} figures)")
    if size > 0.94 * LIMIT:
        raise SystemExit("too close to the artifact ceiling")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
