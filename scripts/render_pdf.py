"""Render a self-contained HTML page to PDF with headless Chromium, using its print CSS.

Images marked loading="lazy" are never fetched in a headless print pass, so they are
promoted to eager and awaited before the PDF is taken.
"""
import sys, pathlib
from playwright.sync_api import sync_playwright

src, out = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 1600})
    pg.goto(src.as_uri(), wait_until="load", timeout=180_000)
    n = pg.evaluate("""async () => {
      const imgs = [...document.images];
      imgs.forEach(i => { i.loading = 'eager'; if (!i.complete) i.src = i.src; });
      await Promise.all(imgs.map(i => i.complete ? i.decode().catch(() => {})
        : new Promise(r => { i.onload = i.onerror = r; }).then(() => i.decode().catch(() => {}))));
      return imgs.filter(i => i.naturalWidth > 0).length + '/' + imgs.length;
    }""")
    print(f"  images loaded: {n}")
    try:
        pg.wait_for_load_state("networkidle", timeout=20_000)   # Google Fonts
    except Exception:
        print("  (fonts: network idle timed out, falling back to local stacks)")
    pg.emulate_media(media="print", color_scheme="light")
    pg.wait_for_timeout(1500)
    pg.pdf(path=str(out), print_background=True, prefer_css_page_size=True,
           format="Letter", margin={"top": "14mm", "bottom": "14mm", "left": "14mm", "right": "14mm"})
    b.close()
print(f"-> {out} ({out.stat().st_size/1e6:.2f} MB)")

# Usage on Derecho (no chromium/latex/weasyprint on this box):
#   python -m venv $SCRATCH/pdfvenv && $SCRATCH/pdfvenv/bin/pip install playwright
#   PLAYWRIGHT_BROWSERS_PATH=$SCRATCH/pw-browsers $SCRATCH/pdfvenv/bin/playwright install chromium
#   python scripts/embed_figs.py docs/aires_pilot.html /tmp/embedded.html
#   PLAYWRIGHT_BROWSERS_PATH=$SCRATCH/pw-browsers \
#     $SCRATCH/pdfvenv/bin/python scripts/render_pdf.py /tmp/embedded.html docs/aires_pilot.pdf
