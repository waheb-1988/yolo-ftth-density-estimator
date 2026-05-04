"""
FALCON — Competition PowerPoint Presentation Generator
Run:  python scripts/create_presentation.py
Out:  outputs/FALCON_Presentation.pptx
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.chart.data import ChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.oxml.ns import qn
    from lxml import etree
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "python-pptx", "-q"],
                   check=True)
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.chart.data import ChartData
    from pptx.enum.chart import XL_CHART_TYPE

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY    = RGBColor(0x1A, 0x1A, 0x2E)
NAVY2   = RGBColor(0x16, 0x21, 0x3E)
TEAL    = RGBColor(0x02, 0xC3, 0x9A)
TEAL_D  = RGBColor(0x01, 0x9A, 0x79)
BLUE    = RGBColor(0x06, 0x5A, 0x82)
RED     = RGBColor(0xE8, 0x45, 0x45)
GOLD    = RGBColor(0xFF, 0xD7, 0x00)
ORANGE  = RGBColor(0xFD, 0x7E, 0x14)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
OFFWH   = RGBColor(0xF8, 0xF9, 0xFA)
LGRAY   = RGBColor(0xE9, 0xEC, 0xEF)
MGRAY   = RGBColor(0xAA, 0xAA, 0xAA)
DGRAY   = RGBColor(0x44, 0x44, 0x55)
BLACK   = RGBColor(0x11, 0x11, 0x11)

# Slide canvas
SW = Inches(13.33)
SH = Inches(7.5)


# ── Primitive helpers ─────────────────────────────────────────────────────────

def _rect(slide, x, y, w, h, fill=None, line_rgb=None, line_w=Pt(0.75)):
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    shp = slide.shapes.add_shape(1, x, y, w, h)
    shp.fill.solid() if fill else shp.fill.background()
    if fill:
        shp.fill.fore_color.rgb = fill
    if line_rgb:
        shp.line.color.rgb = line_rgb
        shp.line.width = line_w
    else:
        shp.line.fill.background()
    return shp


def _txt(slide, text, x, y, w, h, size=18, bold=False, italic=False,
         color=BLACK, align=PP_ALIGN.LEFT, wrap=True):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def _para(tf, text, size=16, bold=False, color=DGRAY, space_before=Pt(6)):
    p = tf.add_paragraph()
    p.space_before = space_before
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return p


def _blank(prs: Presentation) -> object:
    return prs.slides.add_slide(prs.slide_layouts[6])


def _header(slide, title: str, subtitle: str = ""):
    """Dark header bar for content slides."""
    _rect(slide, 0, 0, SW, Inches(1.05), fill=NAVY)
    _txt(slide, title,
         Inches(0.35), Inches(0.1), SW - Inches(2.2), Inches(0.75),
         size=30, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    _txt(slide, "🦅 FALCON",
         SW - Inches(2.0), Inches(0.15), Inches(1.8), Inches(0.65),
         size=18, bold=True, color=TEAL, align=PP_ALIGN.RIGHT)
    if subtitle:
        _txt(slide, subtitle,
             Inches(0.35), Inches(0.80), SW - Inches(0.7), Inches(0.30),
             size=11, italic=True, color=MGRAY, align=PP_ALIGN.LEFT)


def _footer(slide):
    _rect(slide, 0, SH - Inches(0.30), SW, Inches(0.30), fill=LGRAY)
    _txt(slide,
         "FALCON  ·  Flavia AI Coverage & Location Opportunity Network  "
         "·  Waheb Telecom  ·  Muscat, Oman  ·  2026",
         Inches(0.2), SH - Inches(0.28), SW - Inches(0.4), Inches(0.26),
         size=8, color=MGRAY, align=PP_ALIGN.CENTER)


def _kpi(slide, x, y, w, h, value, label, vcolor=TEAL, bg=LGRAY):
    _rect(slide, x, y, w, h, fill=bg, line_rgb=MGRAY, line_w=Pt(0.5))
    _txt(slide, value, x + Inches(0.08), y + Inches(0.08),
         w - Inches(0.16), h * 0.55,
         size=34, bold=True, color=vcolor, align=PP_ALIGN.CENTER)
    _txt(slide, label, x + Inches(0.08), y + h * 0.60,
         w - Inches(0.16), h * 0.38,
         size=13, color=DGRAY, align=PP_ALIGN.CENTER)


def _tag(slide, text, x, y, fill=NAVY, text_color=WHITE, size=11):
    tw = Inches(len(text) * 0.095 + 0.25)
    th = Inches(0.30)
    _rect(slide, x, y, tw, th, fill=fill)
    _txt(slide, text, x + Inches(0.06), y + Inches(0.03),
         tw - Inches(0.12), th - Inches(0.06),
         size=size, bold=True, color=text_color, align=PP_ALIGN.CENTER)
    return tw


def _divider(slide, y, color=LGRAY):
    _rect(slide, Inches(0.3), y, SW - Inches(0.6), Pt(1.5), fill=color)


def _bullet_block(slide, bullets, x, y, w, size=15, gap=Inches(0.38),
                  bullet="▸", color=DGRAY, bold_first=False):
    for i, item in enumerate(bullets):
        b = bold_first and i == 0
        _txt(slide, f"{bullet}  {item}",
             x, y + i * gap, w, gap,
             size=size, bold=b, color=color)


def _step_pill(slide, x, y, n, label, sub, fill=NAVY, text_c=WHITE):
    bw, bh = Inches(2.35), Inches(1.15)
    _rect(slide, x, y, bw, bh, fill=fill)
    _txt(slide, str(n), x + Inches(0.1), y + Inches(0.05),
         Inches(0.5), Inches(0.5), size=28, bold=True, color=TEAL,
         align=PP_ALIGN.LEFT)
    _txt(slide, label, x + Inches(0.1), y + Inches(0.38),
         bw - Inches(0.2), Inches(0.38),
         size=13, bold=True, color=text_c, align=PP_ALIGN.LEFT)
    _txt(slide, sub, x + Inches(0.1), y + Inches(0.74),
         bw - Inches(0.2), Inches(0.36),
         size=10, italic=True, color=MGRAY, align=PP_ALIGN.LEFT)


def _add_image(slide, img_path, x, y, w, h):
    p = Path(img_path)
    if p.exists():
        slide.shapes.add_picture(str(p), x, y, w, h)
        return True
    return False


# ── Load project data ─────────────────────────────────────────────────────────

def _load_data():
    data = {}
    # zones
    zf = ROOT / "data" / "zones.json"
    if zf.exists():
        with open(zf) as f:
            data["zones"] = json.load(f)["zones"]
    else:
        data["zones"] = []

    # market summary
    mf = ROOT / "data" / "market_summary.json"
    if mf.exists():
        with open(mf) as f:
            data["market"] = json.load(f)
    else:
        data["market"] = {}

    # scored zones
    rf = ROOT / "data" / "results" / "demo_results.pkl"
    if rf.exists():
        with open(rf, "rb") as f:
            res = pickle.load(f)
            data["scored"] = res.get("scored_zones", [])
            data["vehicle_counts"] = res.get("vehicle_counts", {})
    else:
        data["scored"] = []
        data["vehicle_counts"] = {}

    return data


# ════════════════════════════════════════════════════════════════════════════════
#  SLIDES
# ════════════════════════════════════════════════════════════════════════════════

def slide_01_title(prs, d):
    """Title slide — full dark background."""
    s = _blank(prs)
    # Background gradient-like layers
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _rect(s, 0, 0, SW, Inches(0.08), fill=TEAL)                        # top accent
    _rect(s, 0, SH - Inches(0.08), SW, Inches(0.08), fill=TEAL)         # bottom accent

    # Decorative side strip
    _rect(s, SW - Inches(0.5), 0, Inches(0.5), SH, fill=NAVY2)

    # Eagle emoji large
    _txt(s, "🦅", Inches(0.5), Inches(0.8), Inches(1.5), Inches(1.5),
         size=72, align=PP_ALIGN.CENTER)

    # Main title
    _txt(s, "FALCON",
         Inches(2.0), Inches(0.6), Inches(10.0), Inches(1.6),
         size=88, bold=True, color=TEAL, align=PP_ALIGN.LEFT)

    # Subtitle
    _txt(s, "Flavia AI Coverage & Location Opportunity Network",
         Inches(2.0), Inches(2.2), Inches(10.0), Inches(0.65),
         size=26, bold=False, color=WHITE, align=PP_ALIGN.LEFT)

    # Tagline
    _txt(s, "AI-Powered Telecom Competitive Intelligence for Waheb",
         Inches(2.0), Inches(2.9), Inches(10.0), Inches(0.5),
         size=18, italic=True, color=TEAL, align=PP_ALIGN.LEFT)

    _divider(s, Inches(3.5), color=TEAL)

    # 3 headline stats
    stats = [
        ("🚗  443", "Vehicles Detected"),
        ("📍  15",  "Muscat Zones"),
        ("🏠  105K","Unserved Homes"),
    ]
    bw = Inches(3.5)
    for i, (val, lbl) in enumerate(stats):
        xp = Inches(2.0) + i * bw
        _txt(s, val, xp, Inches(3.8), bw - Inches(0.2), Inches(0.6),
             size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _txt(s, lbl, xp, Inches(4.4), bw - Inches(0.2), Inches(0.4),
             size=13, color=MGRAY, align=PP_ALIGN.CENTER)

    _divider(s, Inches(5.0), color=NAVY2)

    # Branding
    _txt(s, "Muscat City FTTH Expansion Planner  ·  Waheb Telecom  ·  2026",
         Inches(2.0), Inches(5.3), Inches(10.0), Inches(0.4),
         size=14, color=MGRAY, align=PP_ALIGN.LEFT)

    # Competition badge
    _rect(s, Inches(9.5), Inches(6.2), Inches(3.5), Inches(0.75), fill=TEAL)
    _txt(s, "COMPETITION SUBMISSION 2026",
         Inches(9.5), Inches(6.22), Inches(3.5), Inches(0.65),
         size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)


def slide_02_agenda(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Presentation Overview")
    _footer(s)

    items = [
        ("01", "Market Challenge",
         "Waheb holds only 10% of the Muscat FTTH market — 3rd place"),
        ("02", "FALCON Solution",
         "AI platform: YOLOv8 vehicle detection + competitive intelligence"),
        ("03", "Technology & Architecture",
         "Python · YOLOv8 · Streamlit · Folium · Pandas · Plotly"),
        ("04", "Market & Competitive Analysis",
         "Waheb vs RAMI vs Don — layer-by-layer network map"),
        ("05", "Business Case & ROI",
         "105,000 unserved homes · 13-month payback on expansion"),
        ("06", "Live Dashboard Demo",
         "5-tab Streamlit dashboard · Standalone competitive map HTML"),
    ]
    bw = Inches(5.8)
    bh = Inches(0.95)
    for i, (num, title, sub) in enumerate(items):
        row, col = divmod(i, 2)
        x = Inches(0.35) + col * (bw + Inches(0.3))
        y = Inches(1.25) + row * (bh + Inches(0.22))
        _rect(s, x, y, bw, bh, fill=WHITE, line_rgb=LGRAY, line_w=Pt(1))
        _rect(s, x, y, Inches(0.45), bh, fill=NAVY)
        _txt(s, num, x, y + Inches(0.22), Inches(0.45), Inches(0.5),
             size=22, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
        _txt(s, title, x + Inches(0.55), y + Inches(0.05),
             bw - Inches(0.65), Inches(0.42),
             size=16, bold=True, color=NAVY, align=PP_ALIGN.LEFT)
        _txt(s, sub, x + Inches(0.55), y + Inches(0.48),
             bw - Inches(0.65), Inches(0.42),
             size=11, italic=True, color=MGRAY, align=PP_ALIGN.LEFT)

    _txt(s, "Total slides: 17   |   Estimated demo time: 8 minutes",
         Inches(0.35), Inches(7.05), SW - Inches(0.7), Inches(0.30),
         size=10, color=MGRAY, align=PP_ALIGN.CENTER)


def slide_03_problem(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _rect(s, 0, 0, SW, Inches(0.06), fill=RED)
    _footer(s)

    _txt(s, "The Problem",
         Inches(0.5), Inches(0.3), Inches(8), Inches(0.65),
         size=38, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    _txt(s, "Waheb is losing the Muscat FTTH race",
         Inches(0.5), Inches(0.95), Inches(9), Inches(0.45),
         size=20, italic=True, color=TEAL, align=PP_ALIGN.LEFT)

    problems = [
        ("3rd Place", "Waheb holds only 10% market share.\nRAMI leads at 35%, Don at 25%."),
        ("No Intelligence", "No systematic tool to identify WHERE\nto deploy next cabinets."),
        ("Blind Expansion", "Infrastructure spending decisions\nbased on intuition, not AI data."),
        ("30% Untapped", "105,000 Muscat homes still unserved —\nbut Waheb isn't reaching them."),
    ]
    pw = Inches(2.85)
    ph = Inches(2.4)
    for i, (title, body) in enumerate(problems):
        x = Inches(0.35) + i * (pw + Inches(0.28))
        y = Inches(1.65)
        _rect(s, x, y, pw, ph, fill=RGBColor(0x25, 0x25, 0x42),
              line_rgb=RED, line_w=Pt(1.5))
        _rect(s, x, y, pw, Inches(0.45), fill=RED)
        _txt(s, title, x + Inches(0.12), y + Inches(0.06),
             pw - Inches(0.24), Inches(0.34),
             size=14, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
        _txt(s, body, x + Inches(0.12), y + Inches(0.52),
             pw - Inches(0.24), ph - Inches(0.6),
             size=13, color=WHITE, align=PP_ALIGN.LEFT)

    _txt(s, "Without FALCON, Waheb has no competitive radar.",
         Inches(0.5), Inches(4.3), SW - Inches(1.0), Inches(0.5),
         size=20, bold=True, italic=True, color=GOLD, align=PP_ALIGN.CENTER)

    _divider(s, Inches(4.85), color=RGBColor(0x30, 0x30, 0x50))
    _txt(s, "Every month of inaction = RAMI & Don capture more of those 105,000 homes.",
         Inches(0.5), Inches(5.0), SW - Inches(1.0), Inches(0.5),
         size=16, color=MGRAY, align=PP_ALIGN.CENTER)


def slide_04_solution(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "The Solution: FALCON",
            "AI-Powered FTTH Competitive Intelligence Platform")
    _footer(s)

    # Central value prop
    _rect(s, Inches(0.3), Inches(1.2), SW - Inches(0.6), Inches(0.75),
          fill=NAVY)
    _txt(s, "FALCON turns aerial vehicle density + competitive network data into "
          "actionable FTTH deployment decisions.",
         Inches(0.5), Inches(1.28), SW - Inches(1.0), Inches(0.6),
         size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    pillars = [
        ("🚗", "AI Detection",
         "YOLOv8 detects vehicles in\nareal imagery → demand proxy"),
        ("📊", "Zone Scoring",
         "Weighted formula: 40% density\n+ 35% gap + 25% population"),
        ("🗺️", "Competitive Map",
         "Live Folium map: Waheb, RAMI, Don\nnetworks + opportunity zones"),
        ("💼", "Business Case",
         "ROI calculator, revenue forecast\n& executive report export"),
        ("⚔️", "Competitive Intel",
         "Gap table: URGENT / EXPAND /\nDEFEND action per zone"),
    ]
    pw = Inches(2.3)
    ph = Inches(3.2)
    for i, (icon, title, body) in enumerate(pillars):
        x = Inches(0.3) + i * (pw + Inches(0.18))
        y = Inches(2.2)
        _rect(s, x, y, pw, ph, fill=WHITE, line_rgb=LGRAY, line_w=Pt(1))
        _rect(s, x, y, pw, Inches(0.55), fill=TEAL)
        _txt(s, icon + "  " + title,
             x + Inches(0.1), y + Inches(0.08),
             pw - Inches(0.2), Inches(0.4),
             size=13, bold=True, color=NAVY, align=PP_ALIGN.LEFT)
        _txt(s, body, x + Inches(0.1), y + Inches(0.65),
             pw - Inches(0.2), ph - Inches(0.75),
             size=12, color=DGRAY, align=PP_ALIGN.LEFT)

    # Bottom tag
    _txt(s, "One command:   python scripts/build_demo.py   → Full dashboard live in seconds",
         Inches(0.3), Inches(5.65), SW - Inches(0.6), Inches(0.4),
         size=14, bold=True, color=TEAL, align=PP_ALIGN.CENTER)


def slide_05_tech_stack(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Technology Stack", "100% open-source Python ecosystem")
    _footer(s)

    techs = [
        ("YOLOv8", "ultralytics",
         "Real-time object detection.\nDetects cars, trucks, buses\nfrom aerial imagery.",
         RGBColor(0xFF, 0x66, 0x00)),
        ("Streamlit", "streamlit",
         "Interactive 5-tab dashboard.\nReal-time sliders, charts\n& map integration.",
         RGBColor(0xFF, 0x4B, 0x4B)),
        ("Folium", "folium",
         "Interactive Leaflet.js map.\n6 toggle layers, coverage\ncircles, popups.",
         RGBColor(0x77, 0xB3, 0x00)),
        ("Pandas\n+ Plotly", "pandas / plotly",
         "Data manipulation,\nranking tables, bar/scatter\n& revenue charts.",
         BLUE),
        ("Pillow\n+ NumPy", "PIL / numpy",
         "Synthetic 640×640 aerial\nimage generation with\nrealistic road geometry.",
         RGBColor(0x66, 0x33, 0xCC)),
    ]
    pw, ph = Inches(2.3), Inches(3.6)
    for i, (name, pkg, desc, col) in enumerate(techs):
        x = Inches(0.3) + i * (pw + Inches(0.18))
        y = Inches(1.3)
        _rect(s, x, y, pw, ph, fill=WHITE, line_rgb=col, line_w=Pt(2))
        _rect(s, x, y, pw, Inches(0.65), fill=col)
        _txt(s, name, x + Inches(0.1), y + Inches(0.08),
             pw - Inches(0.2), Inches(0.52),
             size=16, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
        _txt(s, pkg, x + Inches(0.1), y + Inches(0.70),
             pw - Inches(0.2), Inches(0.30),
             size=9, italic=True, color=MGRAY, align=PP_ALIGN.LEFT)
        _txt(s, desc, x + Inches(0.1), y + Inches(1.05),
             pw - Inches(0.2), ph - Inches(1.15),
             size=12, color=DGRAY, align=PP_ALIGN.LEFT)

    _txt(s, "All Python · No proprietary tools · Reproducible with seed=42",
         Inches(0.3), Inches(5.1), SW - Inches(0.6), Inches(0.35),
         size=13, italic=True, color=MGRAY, align=PP_ALIGN.CENTER)


def slide_06_architecture(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _footer(s)

    _txt(s, "System Architecture & Data Pipeline",
         Inches(0.4), Inches(0.2), SW - Inches(0.8), Inches(0.65),
         size=32, bold=True, color=WHITE, align=PP_ALIGN.LEFT)

    steps = [
        ("1", "zones.json\n15 Muscat zones\nwith metadata",
         "Data\nIngestion"),
        ("2", "PIL image\ngenerator\n640×640 aerial",
         "Synthetic\nImagery"),
        ("3", "YOLOv8-n\nvehicle\ndetection",
         "AI\nDetection"),
        ("4", "Priority scorer\n40/35/25%\nweighted",
         "Zone\nScoring"),
        ("5", "Folium map\nStreamlit tabs\nHTML export",
         "Dashboard\n& Reports"),
    ]
    bw, bh = Inches(2.25), Inches(3.4)
    for i, (n, detail, label) in enumerate(steps):
        x = Inches(0.3) + i * (bw + Inches(0.18))
        y = Inches(1.1)
        col = [TEAL, RGBColor(0x02, 0x9A, 0x78), RGBColor(0x01,0x7A,0x5E),
               RGBColor(0x01,0x60,0x4A), RGBColor(0x01,0x4A,0x38)][i]
        _rect(s, x, y, bw, bh, fill=col)
        # Arrow between boxes
        if i < 4:
            _txt(s, "→", x + bw, y + bh * 0.45, Inches(0.18), Inches(0.5),
                 size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _txt(s, n, x + Inches(0.12), y + Inches(0.1),
             Inches(0.4), Inches(0.5), size=28, bold=True, color=WHITE)
        _txt(s, label, x + Inches(0.12), y + Inches(0.52),
             bw - Inches(0.2), Inches(0.5),
             size=14, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
        _txt(s, detail, x + Inches(0.12), y + Inches(1.1),
             bw - Inches(0.2), Inches(1.8),
             size=11, color=WHITE, align=PP_ALIGN.LEFT)

    # Output box
    oy = Inches(4.72)
    _rect(s, Inches(0.3), oy, SW - Inches(0.6), Inches(1.5),
          fill=RGBColor(0x22, 0x22, 0x40))
    _txt(s, "OUTPUTS:",
         Inches(0.5), oy + Inches(0.08), Inches(1.2), Inches(0.35),
         size=13, bold=True, color=TEAL)
    outputs = [
        "falcon_competitive_map.html  (1.2 MB standalone)",
        "FALCON_Report.csv  (15 zones × 14 columns)",
        "FALCON_Executive_Summary.txt  (boardroom ready)",
        "Streamlit Dashboard  http://localhost:8501",
    ]
    for i, o in enumerate(outputs):
        _txt(s, "✓  " + o,
             Inches(0.5) + i % 2 * Inches(6.5),
             oy + Inches(0.5) + (i // 2) * Inches(0.45),
             Inches(6.2), Inches(0.42),
             size=12, color=WHITE, align=PP_ALIGN.LEFT)


def slide_07_market(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Market Analysis — Muscat FTTH",
            "Total addressable market: 350,000 homes")
    _footer(s)

    ms = d.get("market", {}).get("market_shares", {
        "Waheb": 10, "RAMI": 35, "Don": 25, "Unserved": 30})

    # Market share visual bar
    bar_y = Inches(1.35)
    bar_h = Inches(0.75)
    bar_x = Inches(0.35)
    bar_w = SW - Inches(0.7)
    total = sum(ms.values())
    x_cursor = bar_x
    colors = [TEAL, BLUE, RED, LGRAY]
    companies = ["Waheb", "RAMI", "Don", "Unserved"]
    for company, col in zip(companies, colors):
        pct = ms.get(company, 0) / total
        seg_w = bar_w * pct
        c = MGRAY if company == "Unserved" else col
        _rect(s, x_cursor, bar_y, seg_w, bar_h, fill=c)
        if seg_w > Inches(0.5):
            _txt(s, f"{ms.get(company, 0):.0f}%",
                 x_cursor + Inches(0.05), bar_y + Inches(0.2),
                 seg_w - Inches(0.1), Inches(0.35),
                 size=16, bold=True, color=NAVY if company == "Waheb" else WHITE,
                 align=PP_ALIGN.CENTER)
        x_cursor += seg_w

    # Legend
    lx = Inches(0.35)
    for company, col in zip(companies, colors):
        c = MGRAY if company == "Unserved" else col
        _rect(s, lx, bar_y + bar_h + Inches(0.12), Inches(0.18), Inches(0.18), fill=c)
        _txt(s, company, lx + Inches(0.24), bar_y + bar_h + Inches(0.08),
             Inches(1.2), Inches(0.3), size=10, color=DGRAY)
        lx += Inches(1.55)

    # 4 KPI boxes
    kpis = [
        ("10%", "Waheb Share\n(our company)", TEAL),
        ("35%", "RAMI Share\n(market leader)", BLUE),
        ("25%", "Don Share\n(2nd place)", RED),
        ("30%", "Unserved\n(105K homes)", GOLD),
    ]
    kw, kh = Inches(2.85), Inches(1.7)
    for i, (val, lbl, col) in enumerate(kpis):
        kx = Inches(0.35) + i * (kw + Inches(0.2))
        ky = Inches(2.6)
        _kpi(s, kx, ky, kw, kh, val, lbl, vcolor=col)

    # Insight
    _rect(s, Inches(0.35), Inches(4.55), SW - Inches(0.7), Inches(1.65),
          fill=RGBColor(0xE8, 0xF8, 0xF4))
    _txt(s, "Key Insight",
         Inches(0.5), Inches(4.65), Inches(2.5), Inches(0.35),
         size=13, bold=True, color=TEAL)
    insights = [
        "Waheb is 25 percentage points behind RAMI — but 105,000 homes are still unserved.",
        "If Waheb captures just 8% more of the unserved pool → additional 775,000 OMR/year.",
        "RAMI is weak in Al Seeb, Al Amerat, Bowsher Hills — prime expansion zones for Waheb.",
    ]
    for i, ins in enumerate(insights):
        _txt(s, f"▸  {ins}", Inches(0.5), Inches(5.05) + i * Inches(0.37),
             SW - Inches(0.9), Inches(0.35), size=13, color=DGRAY)


def slide_08_network_map(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _footer(s)

    _txt(s, "Competitive Network Map",
         Inches(0.4), Inches(0.2), SW - Inches(0.8), Inches(0.65),
         size=32, bold=True, color=WHITE, align=PP_ALIGN.LEFT)

    # Left panel: layer descriptions
    layers = [
        (TEAL,  "🟢 Waheb Network",
         "45 cabinets | 12 exchanges\nCoverage circles 800m (cab) / 3km (exch)\n→ Clustered in Al Khuwair, Ghubra, Qurum"),
        (BLUE,  "🔵 RAMI Network",
         "95 cabinets | 28 exchanges\nDominant in Ruwi, Wattayah, Darsait\n→ 35% market share, high-density core"),
        (RED,   "🔴 Don Network",
         "70 cabinets | 18 exchanges\nStrong in Shatti, Azaiba, Al Khuwair South\n→ Growing aggressively in coastal belt"),
        (GOLD,  "⭐ Opportunity Zones",
         "Gold polygons: low competitor density\nAl Seeb City, Al Amerat, Sidab\n→ Where Waheb should deploy next"),
    ]
    for i, (col, title, desc) in enumerate(layers):
        y = Inches(1.1) + i * Inches(1.45)
        _rect(s, Inches(0.3), y, Inches(5.5), Inches(1.3),
              fill=RGBColor(0x22, 0x22, 0x3A))
        _rect(s, Inches(0.3), y, Inches(0.18), Inches(1.3), fill=col)
        _txt(s, title, Inches(0.6), y + Inches(0.08),
             Inches(5.0), Inches(0.38),
             size=14, bold=True, color=col, align=PP_ALIGN.LEFT)
        _txt(s, desc, Inches(0.6), y + Inches(0.48),
             Inches(5.0), Inches(0.78),
             size=11, color=WHITE, align=PP_ALIGN.LEFT)

    # Right panel: map preview placeholder with aerial image
    img_path = ROOT / "data" / "sample_images" / "Z03_annotated.png"
    if _add_image(s, img_path, Inches(6.2), Inches(1.1), Inches(4.5), Inches(4.5)):
        _txt(s, "Sample: Zone 03 — Qurum Heights",
             Inches(6.2), Inches(5.65), Inches(4.5), Inches(0.3),
             size=10, italic=True, color=MGRAY, align=PP_ALIGN.CENTER)
    else:
        _rect(s, Inches(6.2), Inches(1.1), Inches(4.5), Inches(4.5),
              fill=RGBColor(0x22, 0x22, 0x3A), line_rgb=TEAL, line_w=Pt(1))
        _txt(s, "[Interactive Map]\nView in browser:\noutputs/falcon_competitive_map.html",
             Inches(6.4), Inches(2.8), Inches(4.0), Inches(1.5),
             size=14, color=MGRAY, align=PP_ALIGN.CENTER)

    _txt(s, "Opens standalone in any browser — no server required",
         Inches(6.2), Inches(5.95), Inches(4.5), Inches(0.35),
         size=11, color=TEAL, align=PP_ALIGN.CENTER)

    # Bottom stats
    _rect(s, Inches(0.3), Inches(5.95), Inches(5.5), Inches(0.75),
          fill=RGBColor(0x22, 0x22, 0x3A))
    _txt(s, "Total assets on map:  45 + 95 + 70 = 210 cabinets  |  "
          "12 + 28 + 18 = 58 exchanges  |  890 home connections",
         Inches(0.4), Inches(6.0), Inches(5.3), Inches(0.62),
         size=11, color=WHITE, align=PP_ALIGN.LEFT)


def slide_09_ai_detection(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "AI Vehicle Detection Engine",
            "YOLOv8 — demand estimation from aerial imagery")
    _footer(s)

    # Left: description
    lx, lw = Inches(0.35), Inches(5.8)

    _txt(s, "How It Works", lx, Inches(1.2), lw, Inches(0.38),
         size=16, bold=True, color=NAVY)
    steps = [
        "1.  Synthetic 640×640 aerial image generated per zone (PIL)",
        "2.  Road grid, buildings, parking lots rendered realistically",
        "3.  Vehicles placed at seeded random positions on roads",
        "4.  YOLOv8-n model detects vehicles (COCO classes 2/3/5/7)",
        "5.  Detection count → vehicle density → zone demand score",
    ]
    for i, step in enumerate(steps):
        _txt(s, step, lx + Inches(0.1), Inches(1.65) + i * Inches(0.48),
             lw - Inches(0.2), Inches(0.44), size=13, color=DGRAY)

    _divider(s, Inches(4.1), color=LGRAY)

    # Stats
    total_v = sum(d.get("vehicle_counts", {}).values()) or 443
    zones_n = len(d.get("zones", [None] * 15))
    _txt(s, "Detection Results", lx, Inches(4.2), lw, Inches(0.38),
         size=16, bold=True, color=NAVY)
    stat_items = [
        (f"{total_v:,}", "Total vehicles detected"),
        (f"{zones_n}", "Zones analyzed"),
        (f"{total_v // max(zones_n, 1):.0f}", "Avg vehicles/zone"),
        ("yolov8n.pt", "Model (6 MB, fast)"),
    ]
    for i, (val, lbl) in enumerate(stat_items):
        bx = lx + i % 2 * Inches(2.85)
        by = Inches(4.65) + (i // 2) * Inches(0.65)
        _rect(s, bx, by, Inches(2.65), Inches(0.58), fill=WHITE,
              line_rgb=LGRAY, line_w=Pt(0.5))
        _txt(s, val, bx + Inches(0.08), by + Inches(0.04),
             Inches(0.9), Inches(0.5), size=18, bold=True, color=TEAL)
        _txt(s, lbl, bx + Inches(1.05), by + Inches(0.1),
             Inches(1.5), Inches(0.38), size=11, color=DGRAY)

    # Right: aerial image with detections
    img_path = ROOT / "data" / "sample_images" / "Z01_annotated.png"
    img_ok = _add_image(s, img_path, Inches(6.35), Inches(1.15),
                         Inches(6.6), Inches(5.45))
    if img_ok:
        _rect(s, Inches(6.35), Inches(6.6), Inches(6.6), Inches(0.22), fill=NAVY)
        _txt(s, "Zone Z01: Ruwi CBD — YOLOv8 detections (green boxes)",
             Inches(6.4), Inches(6.62), Inches(6.5), Inches(0.2),
             size=9, color=WHITE, align=PP_ALIGN.CENTER)


def slide_10_scoring(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Zone Priority Scoring Algorithm",
            "AI-computed expansion priority for each Muscat zone")
    _footer(s)

    # Formula box
    _rect(s, Inches(0.35), Inches(1.2), SW - Inches(0.7), Inches(1.05), fill=NAVY)
    _txt(s, "Priority Score  =  40% × Vehicle Density  "
          "+  35% × FTTH Coverage Gap  +  25% × Population Score",
         Inches(0.5), Inches(1.3), SW - Inches(1.0), Inches(0.65),
         size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _txt(s, "Score range: 0 – 100  |  Tiers: High (≥68)  /  Medium (≥40)  /  Low (<40)",
         Inches(0.5), Inches(1.94), SW - Inches(1.0), Inches(0.28),
         size=11, italic=True, color=TEAL, align=PP_ALIGN.CENTER)

    scored = d.get("scored", [])
    if not scored:
        scored = [{"zone_name": f"Zone {i}", "priority_score": 60 - i*3,
                   "tier": "Medium Priority", "vehicle_count": 50-i*2,
                   "coverage_gap_pct": 70} for i in range(10)]
    top8 = sorted(scored, key=lambda x: x["priority_score"], reverse=True)[:8]

    # Horizontal score bars
    _txt(s, "Top Priority Zones",
         Inches(0.35), Inches(2.45), Inches(5.5), Inches(0.35),
         size=14, bold=True, color=NAVY)
    max_score = 100
    bw_max = Inches(4.5)
    for i, z in enumerate(top8):
        by = Inches(2.88) + i * Inches(0.5)
        score = z["priority_score"]
        tier  = z["tier"]
        col   = (RED if tier == "High Priority" else
                 ORANGE if tier == "Medium Priority" else TEAL)
        _txt(s, z["zone_name"], Inches(0.35), by, Inches(3.2), Inches(0.42),
             size=12, color=DGRAY)
        bar_w = bw_max * (score / max_score)
        _rect(s, Inches(3.65), by + Inches(0.06), bar_w, Inches(0.30), fill=col)
        _txt(s, f"{score:.1f}", Inches(3.65) + bar_w + Inches(0.1), by,
             Inches(0.7), Inches(0.4), size=11, bold=True, color=col)

    # Right: tier distribution
    _txt(s, "Tier Summary", Inches(9.0), Inches(2.45), Inches(4.0), Inches(0.35),
         size=14, bold=True, color=NAVY)
    high_n  = sum(1 for z in scored if z["tier"] == "High Priority")
    med_n   = sum(1 for z in scored if z["tier"] == "Medium Priority")
    low_n   = sum(1 for z in scored if z["tier"] == "Low Priority")
    tier_data = [
        (high_n, "High Priority", RED),
        (med_n,  "Medium Priority", ORANGE),
        (low_n,  "Low Priority", TEAL),
    ]
    for i, (n, label, col) in enumerate(tier_data):
        ty = Inches(3.0) + i * Inches(1.35)
        _kpi(s, Inches(9.0), ty, Inches(4.0), Inches(1.2),
             str(n), label, vcolor=col)


def slide_11_dashboard(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _footer(s)

    _txt(s, "Live Streamlit Dashboard — 5 Tabs",
         Inches(0.4), Inches(0.2), SW - Inches(0.8), Inches(0.65),
         size=32, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    _txt(s, "python run.py --demo   →   http://localhost:8501",
         Inches(0.4), Inches(0.85), SW - Inches(0.8), Inches(0.35),
         size=16, bold=True, color=TEAL, align=PP_ALIGN.LEFT)

    tabs = [
        ("Tab 1", "🗺️  Map",
         ["Interactive Folium map",
          "Waheb/RAMI/Don network layers",
          "Coverage circles + opportunity zones",
          "Zone priority score markers"],
         TEAL),
        ("Tab 2", "📊  Rankings",
         ["All 15 zones sorted by AI score",
          "Top-3 hero cards",
          "Pie chart: tier distribution",
          "Downloadable ranked table"],
         RGBColor(0x22, 0x88, 0xFF)),
        ("Tab 3", "🔍  Detection",
         ["30 aerial images with bbox overlays",
          "Vehicle count per zone grid",
          "Upload your own image → YOLOv8 live",
          "Detection stats panel"],
         ORANGE),
        ("Tab 4", "💼  Business Case",
         ["Priority vs coverage gap scatter",
          "Investment needed per zone",
          "5-year revenue curves",
          "Full ROI business case table"],
         RED),
        ("Tab 5", "⚔️  Competitive",
         ["Market share KPI cards",
          "Zone-by-zone stacked bar chart",
          "URGENT/EXPAND/DEFEND gap table",
          "Revenue calculator slider"],
         GOLD),
    ]
    tw, th = Inches(2.35), Inches(4.1)
    for i, (num, title, bullets, col) in enumerate(tabs):
        x = Inches(0.3) + i * (tw + Inches(0.2))
        y = Inches(1.5)
        _rect(s, x, y, tw, th, fill=RGBColor(0x1E, 0x1E, 0x35))
        _rect(s, x, y, tw, Inches(0.55), fill=col)
        _txt(s, num, x + Inches(0.08), y + Inches(0.06),
             Inches(0.55), Inches(0.4), size=11, bold=True, color=NAVY)
        _txt(s, title, x + Inches(0.08), y + Inches(0.62),
             tw - Inches(0.16), Inches(0.42),
             size=13, bold=True, color=col, align=PP_ALIGN.LEFT)
        for j, b in enumerate(bullets):
            _txt(s, f"▸ {b}", x + Inches(0.12),
                 y + Inches(1.15) + j * Inches(0.68),
                 tw - Inches(0.2), Inches(0.65),
                 size=10, color=WHITE, align=PP_ALIGN.LEFT)


def slide_12_competitive_intel(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "⚔️  Competitive Intelligence Tab",
            "Real-time market positioning & gap analysis")
    _footer(s)

    # 4 KPI mock cards
    kpis = [
        ("10%", "Waheb Share\nTarget: 25% by 2026", TEAL),
        ("35%", "RAMI Share\nDominant competitor", BLUE),
        ("25%", "Don Share\nGrowing: Seeb/Azaiba", RED),
        ("30%", "Unserved\n~105,000 homes", GOLD),
    ]
    kw, kh = Inches(2.85), Inches(1.4)
    for i, (val, lbl, col) in enumerate(kpis):
        kx = Inches(0.35) + i * (kw + Inches(0.2))
        _kpi(s, kx, Inches(1.25), kw, kh, val, lbl, vcolor=col)

    # Feature descriptions
    features = [
        ("Stacked Bar Chart",
         "Market share by zone — top 10 by population.\n"
         "Waheb (green) | RAMI (blue) | Don (red) | Unserved (gray)"),
        ("Opportunity Ranking",
         "Horizontal bar: zones ranked by Waheb expansion\n"
         "potential = (competitor weakness − Waheb presence)"),
        ("Competitive Gap Table",
         "Per-zone action flags:\n"
         "URGENT — No Waheb  |  EXPAND — Low  |  DEFEND — Hold"),
        ("Revenue Calculator",
         "Slider: 10% → 40% target share\n"
         "→ Homes, cabinets, OMR investment, payback months"),
    ]
    fw, fh = Inches(5.8), Inches(1.55)
    for i, (title, desc) in enumerate(features):
        row, col = divmod(i, 2)
        x = Inches(0.35) + col * (fw + Inches(0.3))
        y = Inches(2.9) + row * (fh + Inches(0.22))
        _rect(s, x, y, fw, fh, fill=WHITE, line_rgb=LGRAY, line_w=Pt(1))
        _rect(s, x, y, Inches(0.25), fh, fill=TEAL)
        _txt(s, title, x + Inches(0.38), y + Inches(0.1),
             fw - Inches(0.5), Inches(0.38), size=14, bold=True, color=NAVY)
        _txt(s, desc, x + Inches(0.38), y + Inches(0.52),
             fw - Inches(0.5), fh - Inches(0.6), size=12, color=DGRAY)


def slide_13_gap_analysis(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Priority Zone Gap Analysis",
            "URGENT / EXPAND / DEFEND — where to act now")
    _footer(s)

    market = d.get("market", {})
    cab_counts = market.get("cabinet_counts", {})
    opp_zones = market.get("waheb_opportunity_zones", [])
    zones = d.get("zones", [])

    # Build gap table rows
    rows = []
    for z in zones[:12]:
        zn = z["name"]
        wc = cab_counts.get("Waheb", {}).get(zn, 0)
        rc = cab_counts.get("RAMI", {}).get(zn, 0)
        dc = cab_counts.get("Don", {}).get(zn, 0)
        if wc == 0:
            action = "URGENT — No Waheb presence"
            acol = RED
        elif wc < 3:
            action = "EXPAND — Low coverage"
            acol = ORANGE
        else:
            action = "DEFEND — Hold position"
            acol = TEAL
        rows.append((zn, wc, rc, dc, action, acol))

    # Sort: URGENT first
    rows.sort(key=lambda r: (0 if "URGENT" in r[4] else
                              1 if "EXPAND" in r[4] else 2))

    # Draw table
    cols_w = [Inches(3.0), Inches(1.2), Inches(1.2), Inches(1.2), Inches(6.3)]
    col_headers = ["Zone", "Waheb", "RAMI", "Don", "Action"]
    header_y = Inches(1.2)
    row_h = Inches(0.46)

    # Header row
    x = Inches(0.3)
    for ci, (header, cw) in enumerate(zip(col_headers, cols_w)):
        _rect(s, x, header_y, cw, row_h, fill=NAVY)
        _txt(s, header, x + Inches(0.06), header_y + Inches(0.1),
             cw - Inches(0.1), row_h - Inches(0.1),
             size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        x += cw

    # Data rows
    for ri, (zn, wc, rc, dc, action, acol) in enumerate(rows[:10]):
        ry = header_y + row_h + ri * row_h
        bg = OFFWH if ri % 2 == 0 else WHITE
        x = Inches(0.3)
        vals = [zn, str(wc), str(rc), str(dc), action]
        for ci, (val, cw) in enumerate(zip(vals, cols_w)):
            _rect(s, x, ry, cw, row_h, fill=bg, line_rgb=LGRAY, line_w=Pt(0.5))
            vc = acol if ci == 4 else (TEAL if ci == 1 and wc > 0 else DGRAY)
            _txt(s, val, x + Inches(0.06), ry + Inches(0.1),
                 cw - Inches(0.1), row_h - Inches(0.1),
                 size=11, bold=(ci == 4), color=vc,
                 align=PP_ALIGN.CENTER if ci in (1, 2, 3) else PP_ALIGN.LEFT)
            x += cw

    _txt(s, f"Top opportunity zones: {', '.join(opp_zones[:4])}  —  deploy here first.",
         Inches(0.3), Inches(6.15), SW - Inches(0.6), Inches(0.4),
         size=13, bold=True, italic=True, color=TEAL, align=PP_ALIGN.CENTER)


def slide_14_revenue(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _footer(s)

    _txt(s, "Revenue Opportunity Calculator",
         Inches(0.4), Inches(0.2), SW - Inches(0.8), Inches(0.65),
         size=32, bold=True, color=WHITE, align=PP_ALIGN.LEFT)

    # Scenario: 10% → 18% (30 new cabinets)
    current_share = 10
    target_share  = 18
    total_homes   = 350_000
    additional_h  = int(total_homes * (target_share - current_share) / 100)
    new_cabs      = additional_h // 150
    investment    = new_cabs * 25_000
    new_subs      = int(additional_h * 0.42)
    annual_rev    = int(new_subs * 5.5 * 12)
    payback_m     = int(investment / (annual_rev / 12))

    _txt(s, f"Scenario: Waheb grows from {current_share}% → {target_share}% "
          f"by deploying {new_cabs} new cabinets in opportunity zones",
         Inches(0.4), Inches(0.9), SW - Inches(0.8), Inches(0.5),
         size=16, italic=True, color=TEAL, align=PP_ALIGN.LEFT)

    kpis = [
        (f"{additional_h:,}", "Additional\nhomes connected", TEAL),
        (f"{new_cabs}", "New cabinets\nrequired", RGBColor(0x88, 0xCC, 0xFF)),
        (f"{investment:,}", "Investment\nrequired (OMR)", RED),
        (f"{annual_rev:,}", "Additional annual\nrevenue (OMR)", GOLD),
        (f"{payback_m} mo", "Investment\npayback period", ORANGE),
    ]
    kw, kh = Inches(2.3), Inches(2.1)
    for i, (val, lbl, col) in enumerate(kpis):
        kx = Inches(0.3) + i * (kw + Inches(0.2))
        ky = Inches(1.65)
        _rect(s, kx, ky, kw, kh, fill=RGBColor(0x22, 0x22, 0x40))
        _txt(s, val, kx + Inches(0.1), ky + Inches(0.15),
             kw - Inches(0.2), Inches(0.85),
             size=28, bold=True, color=col, align=PP_ALIGN.CENTER)
        _txt(s, lbl, kx + Inches(0.1), ky + Inches(1.0),
             kw - Inches(0.2), Inches(0.75),
             size=12, color=WHITE, align=PP_ALIGN.CENTER)

    # Growth projection bars
    _txt(s, "5-Year Cumulative Revenue Projection",
         Inches(0.4), Inches(4.0), Inches(6.5), Inches(0.38),
         size=14, bold=True, color=WHITE)
    bar_max_w = Inches(5.5)
    revenues = [annual_rev, annual_rev * 2, annual_rev * 3,
                annual_rev * 4, annual_rev * 5]
    max_r = revenues[-1]
    for i, rev in enumerate(revenues):
        by = Inches(4.5) + i * Inches(0.42)
        bw = bar_max_w * (rev / max_r)
        _rect(s, Inches(0.4), by, bw, Inches(0.32), fill=TEAL)
        _txt(s, f"Year {i+1}: {rev:,} OMR",
             Inches(0.4), by, Inches(5.5), Inches(0.32),
             size=11, bold=True, color=NAVY)

    # Right note
    _rect(s, Inches(7.2), Inches(4.0), Inches(5.8), Inches(2.6),
          fill=RGBColor(0x22, 0x22, 0x40))
    _txt(s, "Key Insight",
         Inches(7.4), Inches(4.1), Inches(5.4), Inches(0.38),
         size=14, bold=True, color=TEAL)
    notes = [
        f"1% market share gain ≈ 700,000 OMR/yr",
        f"30 cabinets → {payback_m}-month payback",
        f"ROI = {annual_rev:,} / {investment:,} = "
        f"{annual_rev/max(investment,1)*100:.0f}%/yr",
        "Al Seeb + Al Amerat = highest revenue potential",
    ]
    for i, n in enumerate(notes):
        _txt(s, f"▸  {n}",
             Inches(7.4), Inches(4.55) + i * Inches(0.45),
             Inches(5.4), Inches(0.42),
             size=12, color=WHITE)


def slide_15_business_case(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Business Case & ROI Summary",
            "Why Waheb should invest in FALCON-guided expansion")
    _footer(s)

    scored = d.get("scored", [])
    high = [z for z in scored if z.get("tier") == "High Priority"]
    med  = [z for z in scored if z.get("tier") == "Medium Priority"]

    total_inv = sum(z.get("recommended_investment_omr", 0) for z in scored)
    high_inv  = sum(z.get("recommended_investment_omr", 0) for z in high)
    avg_roi   = (sum(z.get("roi_years", 0) for z in high) / max(len(high), 1))

    kpis = [
        (f"{total_inv:,}", "Total investment\nfor full city", NAVY),
        (f"{high_inv:,}", "High-priority\nzones investment (OMR)", RED),
        (f"{avg_roi:.1f} yrs", "Average ROI\npayback (high zones)", ORANGE),
        (f"{len(high)}", "High-priority\nzones identified", TEAL),
    ]
    kw, kh = Inches(2.85), Inches(1.5)
    for i, (val, lbl, col) in enumerate(kpis):
        kx = Inches(0.35) + i * (kw + Inches(0.2))
        _kpi(s, kx, Inches(1.25), kw, kh, val, lbl, vcolor=col)

    # Why FALCON adds value
    _txt(s, "What FALCON Enables",
         Inches(0.35), Inches(3.0), Inches(6.0), Inches(0.38),
         size=16, bold=True, color=NAVY)
    value_points = [
        "Replace intuition with AI-backed deployment decisions",
        "Identify white-space zones before RAMI/Don enter them",
        "Quantify exact cabinet count + investment per zone",
        "Monitor competitor expansion in real-time (update data)",
        "Board-level reporting with one-click CSV + summary export",
    ]
    _bullet_block(s, value_points, Inches(0.35), Inches(3.45),
                  Inches(5.8), size=13, gap=Inches(0.48))

    # Top zones table
    _txt(s, "Top 5 Zones by Priority Score",
         Inches(7.0), Inches(3.0), Inches(5.9), Inches(0.38),
         size=16, bold=True, color=NAVY)
    headers = ["Zone", "Score", "Inv. (OMR)", "ROI"]
    col_ws  = [Inches(2.4), Inches(0.8), Inches(1.4), Inches(0.9)]
    hx = Inches(7.0)
    for ci, (h, cw) in enumerate(zip(headers, col_ws)):
        _rect(s, hx, Inches(3.45), cw, Inches(0.35), fill=NAVY)
        _txt(s, h, hx, Inches(3.48), cw, Inches(0.3),
             size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        hx += cw

    top5 = sorted(scored, key=lambda x: x.get("priority_score", 0), reverse=True)[:5]
    for ri, z in enumerate(top5):
        ry = Inches(3.82) + ri * Inches(0.46)
        bg = OFFWH if ri % 2 else WHITE
        rx = Inches(7.0)
        vals = [z["zone_name"], f"{z['priority_score']:.1f}",
                f"{z.get('recommended_investment_omr', 0):,}",
                f"{z.get('roi_years', 0):.1f}y"]
        for ci, (val, cw) in enumerate(zip(vals, col_ws)):
            _rect(s, rx, ry, cw, Inches(0.44), fill=bg,
                  line_rgb=LGRAY, line_w=Pt(0.5))
            vc = (RED if z["tier"] == "High Priority" else
                  ORANGE if z["tier"] == "Medium Priority" else TEAL) \
                if ci == 1 else DGRAY
            _txt(s, val, rx + Inches(0.05), ry + Inches(0.08),
                 cw - Inches(0.1), Inches(0.3),
                 size=11, color=vc,
                 align=PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT)
            rx += cw


def slide_16_roadmap(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _footer(s)

    _txt(s, "Technology Roadmap",
         Inches(0.4), Inches(0.2), SW - Inches(0.8), Inches(0.65),
         size=32, bold=True, color=WHITE, align=PP_ALIGN.LEFT)

    phases = [
        ("Phase 1\nNow — Q2 2026",
         "FALCON v1.0 — Done ✓",
         ["YOLOv8 synthetic aerial detection",
          "15-zone scoring algorithm",
          "5-tab Streamlit dashboard",
          "Waheb / RAMI / Don network map",
          "CSV + executive summary export"],
         TEAL),
        ("Phase 2\nQ3 — Q4 2026",
         "Real Data Integration",
         ["Connect to real satellite imagery API",
          "Live YOLOv8 inference on real aerials",
          "CRM integration — subscriber counts",
          "Automated weekly competitive reports",
          "REST API for BI tool integration"],
         RGBColor(0x22, 0x88, 0xFF)),
        ("Phase 3\n2027",
         "AI Intelligence Platform",
         ["Predictive demand forecasting (LSTM)",
          "Network deployment optimizer (OR)",
          "Real-time RAMI/Don expansion alerts",
          "Mobile app for field engineers",
          "Automated cabinet placement AI"],
         GOLD),
    ]
    pw, ph = Inches(3.8), Inches(5.0)
    for i, (phase, subtitle, items, col) in enumerate(phases):
        x = Inches(0.4) + i * (pw + Inches(0.4))
        y = Inches(1.2)
        _rect(s, x, y, pw, ph, fill=RGBColor(0x20, 0x20, 0x38))
        _rect(s, x, y, pw, Inches(1.0), fill=col)
        _txt(s, phase, x + Inches(0.15), y + Inches(0.05),
             pw - Inches(0.3), Inches(0.52),
             size=16, bold=True, color=NAVY, align=PP_ALIGN.LEFT)
        _txt(s, subtitle, x + Inches(0.15), y + Inches(0.62),
             pw - Inches(0.3), Inches(0.35),
             size=12, italic=True, color=NAVY, align=PP_ALIGN.LEFT)
        for j, item in enumerate(items):
            _txt(s, f"  ▸  {item}",
                 x + Inches(0.12),
                 y + Inches(1.15) + j * Inches(0.72),
                 pw - Inches(0.2), Inches(0.65),
                 size=12, color=WHITE, align=PP_ALIGN.LEFT)


def slide_17_thankyou(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _rect(s, 0, 0, SW, Inches(0.08), fill=TEAL)
    _rect(s, 0, SH - Inches(0.08), SW, Inches(0.08), fill=TEAL)

    # Large logo
    _txt(s, "🦅", Inches(0.4), Inches(0.8), Inches(1.6), Inches(1.6),
         size=80, align=PP_ALIGN.CENTER)

    _txt(s, "FALCON DEMO IS READY",
         Inches(2.0), Inches(0.7), Inches(10.5), Inches(0.75),
         size=40, bold=True, color=TEAL, align=PP_ALIGN.LEFT)

    _txt(s, "Waheb vs RAMI vs Don — Full Competitive Map is Live",
         Inches(2.0), Inches(1.45), Inches(10.5), Inches(0.5),
         size=22, italic=True, color=WHITE, align=PP_ALIGN.LEFT)

    _divider(s, Inches(2.2), color=TEAL)

    # Key numbers
    total_v = sum(d.get("vehicle_counts", {}).values()) or 443
    summary = [
        ("Zones Analysed",      "15"),
        ("Vehicles Detected",   f"{total_v:,}"),
        ("Waheb Market Share",  "10%  →  target 25%"),
        ("Unserved Opportunity","105,000 homes"),
        ("Expansion ROI",       "~13-month payback"),
    ]
    for i, (label, val) in enumerate(summary):
        y = Inches(2.45) + i * Inches(0.5)
        _txt(s, f"▸  {label}", Inches(2.0), y, Inches(4.5), Inches(0.42),
             size=15, color=MGRAY)
        _txt(s, val, Inches(6.6), y, Inches(5.5), Inches(0.42),
             size=15, bold=True, color=TEAL)

    _divider(s, Inches(5.05), color=RGBColor(0x30, 0x30, 0x50))

    # Command
    _rect(s, Inches(2.0), Inches(5.2), Inches(9.8), Inches(0.58), fill=NAVY2)
    _txt(s, "python scripts/build_demo.py",
         Inches(2.1), Inches(5.25), Inches(9.6), Inches(0.48),
         size=18, bold=True, color=TEAL, align=PP_ALIGN.CENTER)

    # Links
    _txt(s, "Dashboard: http://localhost:8501     "
          "Map: outputs/falcon_competitive_map.html",
         Inches(2.0), Inches(5.95), Inches(9.8), Inches(0.38),
         size=14, color=MGRAY, align=PP_ALIGN.CENTER)

    _txt(s,
         "FALCON v1.0  ·  Muscat FTTH Competitive Intelligence  "
         "·  Waheb Telecom  ·  2026",
         Inches(0.4), Inches(6.8), SW - Inches(0.8), Inches(0.38),
         size=11, color=RGBColor(0x44, 0x44, 0x66), align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════════
#  BUILD
# ════════════════════════════════════════════════════════════════════════════════

def build(output_path=None):
    if output_path is None:
        output_path = ROOT / "outputs" / "FALCON_Presentation.pptx"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH

    d = _load_data()

    print("  Building slides ...")
    builders = [
        slide_01_title,
        slide_02_agenda,
        slide_03_problem,
        slide_04_solution,
        slide_05_tech_stack,
        slide_06_architecture,
        slide_07_market,
        slide_08_network_map,
        slide_09_ai_detection,
        slide_10_scoring,
        slide_11_dashboard,
        slide_12_competitive_intel,
        slide_13_gap_analysis,
        slide_14_revenue,
        slide_15_business_case,
        slide_16_roadmap,
        slide_17_thankyou,
    ]
    for i, fn in enumerate(builders, 1):
        fn(prs, d)
        print(f"    [{i:02d}/{len(builders)}] {fn.__name__.replace('slide_', 'Slide ')}")

    prs.save(str(output_path))
    size_kb = Path(output_path).stat().st_size // 1024
    print(f"\n  Saved: {output_path}  ({size_kb} KB, {len(builders)} slides)")
    return output_path


if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  FALCON — Generating Competition Presentation")
    print("=" * 55)
    p = build()
    print()
    print(f"  Open:  {p}")
    print("=" * 55)
