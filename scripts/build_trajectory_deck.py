#!/usr/bin/env python
"""One page per event: the trajectory figure, and the three numbers that read it.

`figures/aires/aires_event_slate.pdf` is the argued version of the experiment, 35 pages
of prose around 30 figures. This is the other extreme -- nine pages, one per event, each
carrying the trajectory panel at the largest size a page allows and nothing but the event
name, the observed anomaly, the deepest anomaly the run reached, and the probability the
run assigns to the observation. It is the artifact for a reader who wants to see the nine
runs side by side without reading an argument about them.

Vector all the way through
--------------------------
The trajectory PDFs are merged as PAGES, not rasterized and re-embedded. Each panel
carries ~450 grey killed branches, and `aires/aplots.py::_save` writes the vector copy for
exactly this reason: a raster of that panel turns to mush the moment it is printed. So the
header is drawn to its own one-page PDF and the trajectory page is stamped onto it with
`pypdf`, which keeps every branch a path.

Where the numbers come from
---------------------------
`docs/aires_event_slate.html`, parsed -- not `runs/aires/*/res/*/compare.json`. The
reductions for eight of the nine events live on a3mega; the slate is the committed record
of them and is on both boxes. Section ORDER in that file is the severity ladder (how far
the observation fell outside GenCast's own week-3 spread), and the deck keeps it: rung 1,
the Southwest heat wave, is the case the run does not reach.

Parsing is keyed on the `record` block's own labels, so a renamed field fails loudly here
rather than silently dropping a number onto the page.

`pypdf` is the one dependency this adds and is not in the `my-env` / `moe` base:

    pip install pypdf
    python scripts/build_trajectory_deck.py
"""
from __future__ import annotations

import html
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Type 42 keeps the header text as embedded TrueType, so the page is searchable and the
# numbers can be copied out. The default Type 3 would draw them as uncopyable outlines.
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter, Transformation

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "figures" / "aires"
SLATE = REPO / "docs" / "aires_event_slate.html"
OUT = FIGS / "aires_trajectory_deck.pdf"

# 16:9. The trajectory figure is 2.5:1, so on a Letter page it would be sized by the page's
# width and leave the lower third empty; this shape is filled by it almost exactly.
PAGE_W, PAGE_H = 13.333 * 72, 7.5 * 72
MARGIN = 0.45 * 72
BOTTOM = 16.0
# Baselines, in points from the TOP of the page. They are written out rather than derived
# from a single header height because the rule has to clear the notes row: computing it
# from the band height instead put it straight through the middle of the numbers.
Y_TITLE, Y_WHEN = 40.0, 61.0
Y_LABEL, Y_VALUE, Y_NOTE = 93.0, 120.0, 136.0
Y_RULE = 151.0
HEADER_H = 158.0              # where the trajectory band starts

OBS_COLOR = "#d62728"         # the observation, as `aires/aplots.py` draws it
RES_COLOR = "#1f77b4"         # the AI+RES population, likewise
INK = "#1a1a1a"
MUTED = "#6b6b6b"

# The `record` cells this deck reads. Keys are matched with whitespace removed, because the
# label carries markup (`A<sub>L</sub>`) that unescapes with no space around the subscript.
K_OBSERVED = "observedAL"
K_RES = "AI+RES"
K_P = "weightedP(ALbeyondobserved)"


def _text(x: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", x)).strip()


def read_slate(path: Path = SLATE) -> list[dict]:
    """The nine pilot sections, in the file's own order, which is the severity ladder."""
    s = path.read_text()
    secs = re.findall(r'<section id="([^"]+)-pilot">(.*?)</section>', s, re.S)
    if len(secs) != 9:
        raise SystemExit(f"{path}: expected 9 pilot sections, found {len(secs)}")
    out = []
    for key, body in secs:
        cells = re.findall(
            r'<div class="cell"><span class="k">(.*?)</span>'
            r'<span class="v[^"]*">(.*?)</span>'
            r'(?:<span class="note">(.*?)</span>)?</div>', body, re.S)
        rec = {_text(k).replace(" ", ""): (_text(v), _text(n or "")) for k, v, n in cells}
        missing = [k for k in (K_OBSERVED, K_RES, K_P) if k not in rec]
        if missing:
            raise SystemExit(
                f"{key}: the slate's record block has no {missing}. Its labels changed - "
                f"fix the K_* constants in this script rather than dropping the field.\n"
                f"  present: {sorted(rec)}")
        res_v, res_note = rec[K_RES]
        m = re.search(r"most extreme\s*(\S+\s*K)", res_note)
        if not m:
            raise SystemExit(f"{key}: no 'most extreme ... K' in the AI+RES note: {res_note!r}")
        out.append(dict(
            key=key,
            title=_text(re.search(r"<h2>(.*?)</h2>", body, re.S).group(1)),
            when=_text(re.search(r'<span class="when">(.*?)</span>', body, re.S).group(1)),
            rung=_text(re.search(r'<span class="rungtag">(.*?)</span>', body, re.S).group(1)),
            observed=rec[K_OBSERVED][0],
            reached=m.group(1).replace("  ", " "),
            n_reach=res_v,
            p=rec[K_P][0],
        ))
    return out


def header_page(ev: dict, path: Path) -> None:
    """The page furniture: title, date, and the three numbers, as one vector PDF page."""
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
    fig.patch.set_facecolor("white")

    def T(x, y, s, **kw):
        fig.text(x / PAGE_W, 1.0 - y / PAGE_H, s, **kw)

    T(MARGIN, Y_TITLE, ev["title"], fontsize=24, color=INK,
      fontweight="semibold", va="baseline")
    T(MARGIN, Y_WHEN, f"{ev['when']}   ·   {ev['rung']}", fontsize=11.5,
      color=MUTED, va="baseline")

    # The three numbers, on one row, each under its own label. Colour carries the same
    # meaning it does inside the panel below: red is the observation, blue is the run.
    cols = [
        ("observed $A_L$", ev["observed"], OBS_COLOR, "ERA5, verification week"),
        ("deepest walker", ev["reached"], RES_COLOR, f"{ev['n_reach']} walkers past the observation"),
        (r"$P(A_L$ beyond observed$)$", ev["p"], RES_COLOR, "sum of the walker probabilities"),
    ]
    x = MARGIN
    step = (PAGE_W - 2 * MARGIN) / 3.0
    for label, value, colour, note in cols:
        T(x, Y_LABEL, label, fontsize=10.5, color=MUTED, va="baseline")
        T(x, Y_VALUE, value, fontsize=21, color=colour, fontweight="semibold",
          va="baseline")
        T(x, Y_NOTE, note, fontsize=9, color=MUTED, va="baseline")
        x += step

    y = 1.0 - Y_RULE / PAGE_H
    fig.add_artist(plt.Line2D([MARGIN / PAGE_W, 1 - MARGIN / PAGE_W], [y, y],
                              color="#cfcfcf", lw=0.8, transform=fig.transFigure))
    fig.savefig(path, format="pdf", facecolor="white")
    plt.close(fig)


def build(out: Path = OUT) -> Path:
    events = read_slate()
    writer = PdfWriter()
    with tempfile.TemporaryDirectory() as tmp:
        for i, ev in enumerate(events, 1):
            traj = FIGS / ev["key"] / f"aires_trajectory_{ev['key']}_pilot.pdf"
            if not traj.exists():
                raise SystemExit(
                    f"{ev['key']}: no {traj}\n  Rebuild it with:\n"
                    f"    PYTHONPATH=. python -m aires.aplots --event {ev['key']} "
                    f"--tag pilot --figures trajectory")

            hp = Path(tmp) / f"{i:02d}_header.pdf"
            header_page(ev, hp)
            page = PdfReader(hp).pages[0]

            src = PdfReader(traj).pages[0]
            tw, th = float(src.mediabox.width), float(src.mediabox.height)
            band_h = PAGE_H - HEADER_H - BOTTOM
            s = min((PAGE_W - 2 * MARGIN) / tw, band_h / th)
            tx = (PAGE_W - tw * s) / 2.0 - float(src.mediabox.left) * s
            ty = BOTTOM + (band_h - th * s) / 2.0 - float(src.mediabox.bottom) * s
            page.merge_transformed_page(src, Transformation().scale(s).translate(tx, ty))
            writer.add_page(page)
            print(f"  {i}. {ev['title']:32s} obs {ev['observed']:>9s}  "
                  f"reached {ev['reached']:>9s}  P {ev['p']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        writer.write(fh)
    print(f"-> {out} ({len(events)} pages, {out.stat().st_size / 1e6:.2f} MB)")
    return out


if __name__ == "__main__":
    sys.exit(0 if build() else 0)
