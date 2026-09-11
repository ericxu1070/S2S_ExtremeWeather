#!/usr/bin/env python
"""Inline every insights-deck PNG into the page as a WebP data URI.

The Artifact CSP blocks every external host, so a published page can only carry images it
embeds, and the rendered page must stay under 16 MB. Forty-odd matplotlib figures do not
fit as PNG, so each is re-encoded to WebP - lossless and lossy are both tried and the
smaller wins, which for line-art panels is usually lossless and for the cartopy map
composites is usually lossy. Same policy as `scripts/embed_slate_figs.py`; the deck and
the slate should not disagree about how a figure is compressed.

The template keeps `src="DATA"` placeholders and is matched on the img's alt text, so the
substitution is repeatable after any figure is regenerated and a renamed figure fails
loudly instead of silently landing under the wrong caption. Every registered figure must
also be USED: an alt text with no placeholder is as much a mistake as a placeholder with
no figure, because it means a panel the deck promised is not on the page.

Three of the figures are new and are drawn by `aires/ainsights.py`; everything else is
already on disk under `figures/aires/` and `figures/astab/` and is collected, not
regenerated.

    python scripts/embed_insights.py                       # template -> docs/aires_insights.html
    python scripts/embed_insights.py <template> <out.html>
"""
import base64
import io
import re
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "figures" / "aires"
ASTAB = REPO / "figures" / "astab"

TEMPLATE = REPO / "docs" / "aires_insights_template.html"
OUT = REPO / "docs" / "aires_insights.html"

# Display name -> run directory. Shared with `scripts/embed_slate_figs.py` so one event
# is called the same thing in both decks.
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

# The three events whose registered box already IS CONUS, so `aires_pdf_box_*` was never
# written for them and their CONUS spatial PDF is the box one. This is the section-3.5
# fact showing up in the filenames.
CONUS_BOXED = ("p90_20251224", "p90_20240802", "p90_20231107")

PNW = "PNW_HeatDome_2021"

ALT_TO_FIG = {
    # --- new, from aires/ainsights.py ------------------------------------------ #
    "The AI+RES slate and its baseline coverage": FIGS / "aires_insights_slate.png",
    "The severity ladder": FIGS / "aires_insights_ladder.png",
    "The two calibration anchors": FIGS / "aires_insights_anchors.png",
    # --- cross-event, already on disk ------------------------------------------ #
    "Every AI+RES walker across all ten runs": FIGS / "aires_walkers.png",
    "The walker population as a density": FIGS / "aires_kde.png",
    # --- the control ----------------------------------------------------------- #
    "Persistence-scored control exceedance curve":
        FIGS / PNW / f"aires_exceedance_{PNW}_persist.png",
    "Per-walker anomaly and probability for the persistence control":
        FIGS / PNW / f"aires_walkers_{PNW}_persist.png",
    "Run health for the persistence control":
        FIGS / PNW / f"aires_diagnostics_{PNW}_persist.png",
    # --- the PNW map exemplars ------------------------------------------------- #
    "Per-member error maps for the Pacific Northwest Heat Dome":
        FIGS / PNW / f"aires_map_error_{PNW}_pilot.png",
    "Walker composite maps for the Pacific Northwest Heat Dome":
        FIGS / PNW / f"aires_map_walk_{PNW}_pilot.png",
    "Ensemble spread maps for the Pacific Northwest Heat Dome":
        FIGS / PNW / f"aires_map_spread_{PNW}_pilot.png",
    # --- the stability sweep --------------------------------------------------- #
    "Lead-time stability of the walker and the scorer": ASTAB / "stability.png",
    "Week-8 GenCast ensemble divergence":
        ASTAB / "ensemble_PNW_HeatDome_2021_wk08.png",
    # --- the two median-region runs, per walker -------------------------------- #
    # Only these two of the nine get their own per-walker detail figure. The pinning is
    # not legible in a summary - the observation is off the left edge of the population -
    # and the cross-event `aires_walkers.png` panel is too small to show it. Every other
    # event's per-walker panel is in that figure and is not repeated here, which is also
    # what keeps the page inside the 16 MB ceiling.
    "Per-walker anomaly and probability for The Normal-Day Test":
        FIGS / "p90_20240802" / "aires_walkers_p90_20240802_pilot.png",
    "Per-walker anomaly and probability for The Below-Median Day":
        FIGS / "p90_20231107" / "aires_walkers_p90_20231107_pilot.png",
}

for name, key in EVENTS.items():
    ALT_TO_FIG[f"Exceedance curve for {name}"] = \
        FIGS / key / f"aires_exceedance_{key}_pilot.png"
    ALT_TO_FIG[f"Run health for {name}"] = \
        FIGS / key / f"aires_diagnostics_{key}_pilot.png"
    ALT_TO_FIG[f"CFSv2 and walker trajectories for {name}"] = \
        FIGS / key / f"aires_trajectory_{key}_pilot.png"
    ALT_TO_FIG[f"AI+RES map comparison for {name}"] = \
        FIGS / key / f"aires_map_compare_{key}_pilot.png"
    # The spatial PDF: the box panel where the box is an event window, the CONUS panel
    # where the box already IS CONUS and no box figure exists.
    ALT_TO_FIG[f"Spatial tail probabilities for {name}"] = (
        FIGS / key / (f"aires_pdf_conus_{key}_pilot.png" if key in CONUS_BOXED
                      else f"aires_pdf_box_{key}_pilot.png"))

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
    src = Path(argv[1]) if len(argv) > 1 else TEMPLATE
    out = Path(argv[2]) if len(argv) > 2 else OUT
    if not src.exists():
        raise SystemExit(f"missing template {src}")
    html = src.read_text(encoding="utf-8")

    alts = re.findall(r'<img src="DATA" alt="([^"]*)"', html)
    if not alts:
        raise SystemExit(f'{src}: no `src="DATA"` placeholders to fill')
    unknown = [a for a in alts if a not in ALT_TO_FIG]
    if unknown:
        raise SystemExit("no figure registered for alt text: " + "; ".join(unknown))
    duplicated = sorted({a for a in alts if alts.count(a) > 1})
    if duplicated:
        raise SystemExit("the same figure would be embedded twice: "
                         + "; ".join(duplicated))

    # A registered figure that the template never asks for is a panel the deck promised
    # and does not show. Loud, not silent.
    unused = sorted(set(ALT_TO_FIG) - set(alts))
    if unused:
        raise SystemExit("registered but never placed in the page: " + "; ".join(unused))

    cache: dict[Path, bytes] = {}
    for alt in alts:
        p = ALT_TO_FIG[alt]
        if not p.exists():
            raise SystemExit(f"missing {p}  (for {alt!r})")
        if p not in cache:
            cache[p] = to_webp(p)
        b64 = base64.b64encode(cache[p]).decode("ascii")
        html = html.replace(f'<img src="DATA" alt="{alt}"',
                            f'<img src="data:image/webp;base64,{b64}" alt="{alt}"', 1)
        rel = p.relative_to(REPO)
        print(f"  {str(rel):72s} {p.stat().st_size / 1e3:6.0f} kB png"
              f" -> {len(cache[p]) / 1e3:6.0f} kB webp")

    # Every image must be INLINE. Anything else - a remote host the CSP drops, a relative
    # path that resolves to nothing once the page is published - renders as a hole with no
    # error, so it is checked here rather than trusted.
    loose = [s for s in re.findall(r'<img[^>]+src="([^"]*)"', html)
             if not s.startswith("data:")]
    if loose:
        raise SystemExit("image src that is not a data URI survived embedding: "
                         + "; ".join(loose))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size = out.stat().st_size
    print(f"-> {out} ({size / 1e6:.2f} MB of a {LIMIT / 1e6:.0f} MB ceiling, "
          f"{len(alts)} figures)")
    if size > 0.94 * LIMIT:
        raise SystemExit("too close to the artifact ceiling")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
