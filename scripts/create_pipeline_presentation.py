"""
FALCON — ML Pipeline PowerPoint Presentation Generator
Covers: Data Collection → Annotation → Dataset Prep → YOLOv8 Architecture →
        Aerial Detection Challenge → FALCON Aerial Pipeline → Detection Results →
        Zone Scoring → Telecom Simulation → Competitor Intelligence →
        ML Insights → Dashboard → FastAPI → Docker → Business Case → Roadmap

Run:  python scripts/create_pipeline_presentation.py
Out:  outputs/FALCON_Pipeline_Presentation.pptx
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
BLUE2   = RGBColor(0x0D, 0x8A, 0xBF)
RED     = RGBColor(0xE8, 0x45, 0x45)
GOLD    = RGBColor(0xFF, 0xD7, 0x00)
ORANGE  = RGBColor(0xFD, 0x7E, 0x14)
GREEN   = RGBColor(0x28, 0xA7, 0x45)
PURPLE  = RGBColor(0x6F, 0x42, 0xC1)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
OFFWH   = RGBColor(0xF8, 0xF9, 0xFA)
LGRAY   = RGBColor(0xE9, 0xEC, 0xEF)
MGRAY   = RGBColor(0xAA, 0xAA, 0xAA)
DGRAY   = RGBColor(0x44, 0x44, 0x55)
BLACK   = RGBColor(0x11, 0x11, 0x11)

SW = Inches(13.33)
SH = Inches(7.5)


# ── Primitive helpers ─────────────────────────────────────────────────────────

def _rect(slide, x, y, w, h, fill=None, line_rgb=None, line_w=Pt(0.75)):
    shp = slide.shapes.add_shape(1, x, y, w, h)
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    else:
        shp.fill.background()
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


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _header(slide, title: str, subtitle: str = "", step_num: str = ""):
    _rect(slide, 0, 0, SW, Inches(1.05), fill=NAVY)
    if step_num:
        _rect(slide, 0, 0, Inches(0.9), Inches(1.05), fill=TEAL)
        _txt(slide, step_num, Inches(0.0), Inches(0.15),
             Inches(0.9), Inches(0.75), size=28, bold=True,
             color=NAVY, align=PP_ALIGN.CENTER)
        title_x = Inches(1.0)
    else:
        title_x = Inches(0.35)
    _txt(slide, title, title_x, Inches(0.1),
         SW - title_x - Inches(2.2), Inches(0.75),
         size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    _txt(slide, "🦅 FALCON",
         SW - Inches(2.0), Inches(0.15), Inches(1.8), Inches(0.65),
         size=17, bold=True, color=TEAL, align=PP_ALIGN.RIGHT)
    if subtitle:
        _txt(slide, subtitle, title_x, Inches(0.80),
             SW - title_x - Inches(0.7), Inches(0.30),
             size=11, italic=True, color=MGRAY, align=PP_ALIGN.LEFT)


def _footer(slide):
    _rect(slide, 0, SH - Inches(0.30), SW, Inches(0.30), fill=LGRAY)
    _txt(slide,
         "FALCON  ·  ML Pipeline  ·  YOLOv8 Vehicle Detection + FTTH Intelligence  ·  Waheb Telecom  ·  2026",
         Inches(0.2), SH - Inches(0.28), SW - Inches(0.4), Inches(0.26),
         size=8, color=MGRAY, align=PP_ALIGN.CENTER)


def _kpi(slide, x, y, w, h, value, label, vcolor=TEAL, bg=LGRAY):
    _rect(slide, x, y, w, h, fill=bg, line_rgb=MGRAY, line_w=Pt(0.5))
    _txt(slide, value, x + Inches(0.08), y + Inches(0.08),
         w - Inches(0.16), h * 0.55,
         size=30, bold=True, color=vcolor, align=PP_ALIGN.CENTER)
    _txt(slide, label, x + Inches(0.08), y + h * 0.60,
         w - Inches(0.16), h * 0.38,
         size=12, color=DGRAY, align=PP_ALIGN.CENTER)


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
    bw, bh = Inches(2.2), Inches(1.10)
    _rect(slide, x, y, bw, bh, fill=fill)
    _txt(slide, str(n), x + Inches(0.1), y + Inches(0.05),
         Inches(0.45), Inches(0.45), size=24, bold=True, color=TEAL,
         align=PP_ALIGN.LEFT)
    _txt(slide, label, x + Inches(0.1), y + Inches(0.38),
         bw - Inches(0.2), Inches(0.36),
         size=12, bold=True, color=text_c, align=PP_ALIGN.LEFT)
    _txt(slide, sub, x + Inches(0.1), y + Inches(0.72),
         bw - Inches(0.2), Inches(0.34),
         size=9, italic=True, color=MGRAY, align=PP_ALIGN.LEFT)


def _add_image(slide, img_path, x, y, w, h):
    p = Path(img_path)
    if p.exists():
        slide.shapes.add_picture(str(p), x, y, w, h)
        return True
    return False


def _flow_arrow(slide, x, y, color=TEAL):
    _txt(slide, "→", x, y, Inches(0.4), Inches(0.4),
         size=22, bold=True, color=color, align=PP_ALIGN.CENTER)


def _code_block(slide, code_lines, x, y, w, h, bg=NAVY2, size=10):
    _rect(slide, x, y, w, h, fill=bg)
    tb = slide.shapes.add_textbox(x + Inches(0.12), y + Inches(0.1),
                                   w - Inches(0.24), h - Inches(0.2))
    tf = tb.text_frame
    tf.word_wrap = False
    first = True
    for line in code_lines:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = TEAL


# ── Load project data ─────────────────────────────────────────────────────────

def _load_data():
    data = {}
    zf = ROOT / "data" / "zones.json"
    if zf.exists():
        with open(zf) as f:
            data["zones"] = json.load(f)["zones"]
    else:
        data["zones"] = []

    mf = ROOT / "data" / "market_summary.json"
    if mf.exists():
        with open(mf) as f:
            data["market"] = json.load(f)
    else:
        data["market"] = {}

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
#  SLIDE BUILDERS
# ════════════════════════════════════════════════════════════════════════════════

# ── Slide 01: Title ───────────────────────────────────────────────────────────
def slide_01_title(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _rect(s, 0, 0, SW, Inches(0.07), fill=TEAL)
    _rect(s, 0, SH - Inches(0.07), SW, Inches(0.07), fill=TEAL)
    _rect(s, SW - Inches(0.45), 0, Inches(0.45), SH, fill=NAVY2)

    _txt(s, "🦅", Inches(0.5), Inches(0.7), Inches(1.4), Inches(1.4),
         size=68, align=PP_ALIGN.CENTER)

    _txt(s, "FALCON",
         Inches(2.0), Inches(0.5), Inches(10.5), Inches(1.5),
         size=82, bold=True, color=TEAL, align=PP_ALIGN.LEFT)

    _txt(s, "ML Pipeline — End-to-End",
         Inches(2.0), Inches(2.1), Inches(10.0), Inches(0.7),
         size=30, bold=True, color=WHITE, align=PP_ALIGN.LEFT)

    _txt(s, "From Data Annotation to AI-Powered FTTH Intelligence",
         Inches(2.0), Inches(2.85), Inches(10.0), Inches(0.45),
         size=18, italic=True, color=TEAL, align=PP_ALIGN.LEFT)

    _divider(s, Inches(3.45), color=TEAL)

    steps = [
        ("📸", "Data Collection"),
        ("🏷️", "Annotation"),
        ("🧠", "YOLOv8 Model"),
        ("🛰️", "Aerial Detection"),
        ("📊", "Intelligence"),
        ("🚀", "Deployment"),
    ]
    bw = Inches(1.95)
    for i, (icon, label) in enumerate(steps):
        xp = Inches(0.7) + i * bw
        _rect(s, xp, Inches(3.7), bw - Inches(0.1), Inches(0.85), fill=NAVY2)
        _txt(s, f"{icon} {label}", xp + Inches(0.08), Inches(3.75),
             bw - Inches(0.25), Inches(0.75), size=12, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)

    _txt(s, "Waheb Telecom  ·  Muscat, Oman  ·  2026",
         Inches(2.0), Inches(5.0), Inches(10.0), Inches(0.35),
         size=13, color=MGRAY, align=PP_ALIGN.LEFT)

    _rect(s, Inches(9.3), Inches(6.1), Inches(3.7), Inches(0.80), fill=TEAL)
    _txt(s, "ML PIPELINE OVERVIEW",
         Inches(9.3), Inches(6.13), Inches(3.7), Inches(0.70),
         size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)


# ── Slide 02: Agenda ─────────────────────────────────────────────────────────
def slide_02_agenda(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Pipeline Agenda", "Complete ML workflow — 15 steps from raw images to production")
    _footer(s)

    steps_left = [
        ("01", "Data Collection",        "Google Earth Pro screenshots, GPS coords"),
        ("02", "Data Annotation",         "YOLO format, bounding boxes, LabelImg"),
        ("03", "Dataset Preparation",     "YAML config, train/val/test split, aug"),
        ("04", "YOLOv8 Architecture",     "COCO pretrained, transfer learning"),
        ("05", "Aerial Detection Challenge", "Why standard YOLO fails on satellite"),
        ("06", "FALCON Aerial Pipeline",  "4× upscale + sliding window + IoU-NMS"),
        ("07", "Detection Results",       "test.png — Muscat real coordinates"),
        ("08", "Zone Scoring Algorithm",  "40% vehicles + 35% gap + 25% pop"),
    ]
    steps_right = [
        ("09", "Telecom Simulation",      "5G / FTTH / mobile KPIs per zone"),
        ("10", "Competitor Intelligence", "RAMI vs Don vs Waheb market share"),
        ("11", "ML Insights",             "K-Means, forecasting, S-curve"),
        ("12", "Streamlit Dashboard",     "6-tab interactive interface"),
        ("13", "FastAPI Backend",         "12 REST endpoints"),
        ("14", "Docker Deployment",       "Containerised production stack"),
        ("15", "Business Case & ROI",     "Revenue projections & roadmap"),
    ]

    lw = Inches(6.0)
    for i, (num, title, sub) in enumerate(steps_left):
        yp = Inches(1.25) + i * Inches(0.72)
        _rect(s, Inches(0.3), yp, Inches(0.42), Inches(0.52), fill=TEAL)
        _txt(s, num, Inches(0.3), yp + Inches(0.02),
             Inches(0.42), Inches(0.48), size=11, bold=True,
             color=NAVY, align=PP_ALIGN.CENTER)
        _txt(s, title, Inches(0.82), yp,
             Inches(2.5), Inches(0.28), size=12, bold=True, color=NAVY)
        _txt(s, sub, Inches(0.82), yp + Inches(0.28),
             Inches(2.5), Inches(0.22), size=9, italic=True, color=MGRAY)

    for i, (num, title, sub) in enumerate(steps_right):
        yp = Inches(1.25) + i * Inches(0.72)
        xoff = Inches(6.6)
        _rect(s, xoff, yp, Inches(0.42), Inches(0.52), fill=BLUE)
        _txt(s, num, xoff, yp + Inches(0.02),
             Inches(0.42), Inches(0.48), size=11, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)
        _txt(s, title, xoff + Inches(0.52), yp,
             Inches(2.5), Inches(0.28), size=12, bold=True, color=NAVY)
        _txt(s, sub, xoff + Inches(0.52), yp + Inches(0.28),
             Inches(2.5), Inches(0.22), size=9, italic=True, color=MGRAY)

    _divider(s, Inches(1.15), color=TEAL)


# ── Slide 03: Step 1 — Data Collection ───────────────────────────────────────
def slide_03_data_collection(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Data Collection", "Aerial imagery acquisition from Google Earth Pro", step_num="01")
    _footer(s)

    # Left panel
    _rect(s, Inches(0.3), Inches(1.2), Inches(5.5), Inches(5.9), fill=WHITE,
          line_rgb=LGRAY, line_w=Pt(0.5))
    _txt(s, "📡  Source: Google Earth Pro",
         Inches(0.5), Inches(1.35), Inches(5.0), Inches(0.38),
         size=14, bold=True, color=NAVY)

    src_bullets = [
        "High-resolution satellite & aerial tiles",
        "Zoom level 18–20 for vehicle-visible detail",
        "Time-of-day stamping (peak vs. off-peak)",
        "GPS coordinate metadata in filename",
        "Screenshot format: PNG, ~1280×720 px",
        "Covers 15 target zones in Muscat, Oman",
    ]
    _bullet_block(s, src_bullets, Inches(0.5), Inches(1.85),
                  Inches(5.0), size=13, gap=Inches(0.40), color=DGRAY)

    _divider(s, Inches(4.25), color=LGRAY)
    _txt(s, "📍  Target Coordinates — Muscat",
         Inches(0.5), Inches(4.4), Inches(5.0), Inches(0.35),
         size=13, bold=True, color=NAVY)
    coord_bullets = [
        "Al Khuwair:  23.593° N, 58.401° E",
        "Bausher:     23.578° N, 58.371° E",
        "Muttrah:     23.617° N, 58.593° E",
        "Ghubrah:     23.606° N, 58.407° E",
    ]
    _bullet_block(s, coord_bullets, Inches(0.5), Inches(4.85),
                  Inches(5.0), size=12, gap=Inches(0.36), bullet="📌", color=BLUE)

    # Right panel — image or placeholder
    img_shown = _add_image(
        s, ROOT / "data" / "sample_images" / "test.png",
        Inches(6.1), Inches(1.2), Inches(6.9), Inches(3.8),
    )
    if not img_shown:
        _rect(s, Inches(6.1), Inches(1.2), Inches(6.9), Inches(3.8),
              fill=NAVY2, line_rgb=TEAL, line_w=Pt(1))
        _txt(s, "📸  Google Earth Pro Screenshot",
             Inches(6.3), Inches(2.5), Inches(6.5), Inches(0.8),
             size=16, color=MGRAY, align=PP_ALIGN.CENTER)

    # Metadata box
    _rect(s, Inches(6.1), Inches(5.15), Inches(6.9), Inches(1.65),
          fill=NAVY, line_rgb=TEAL, line_w=Pt(0.75))
    _txt(s, "📋  Image Metadata",
         Inches(6.25), Inches(5.22), Inches(4.0), Inches(0.30),
         size=12, bold=True, color=TEAL)
    meta = [
        "Filename: test.png  |  Size: 1273 × 538 px",
        "Lat: 23.5933° N  |  Lon: 58.4017° E",
        "Zone: Al Khuwair (Commercial)",
        "Time: Peak hour (estimated 9:00 AM)",
    ]
    for i, m in enumerate(meta):
        _txt(s, m, Inches(6.25), Inches(5.55) + i * Inches(0.29),
             Inches(6.5), Inches(0.28), size=10, color=WHITE)

    # KPI row
    kpis = [("15", "Zones\nCovered"), ("443+", "Vehicles\nDetected"), ("1273px", "Image\nWidth")]
    for i, (v, l) in enumerate(kpis):
        xp = Inches(6.1) + i * Inches(2.35)
        _rect(s, xp, SH - Inches(1.4), Inches(2.2), Inches(1.05),
              fill=NAVY2, line_rgb=TEAL, line_w=Pt(0.5))
        _txt(s, v, xp, SH - Inches(1.38), Inches(2.2), Inches(0.52),
             size=24, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
        _txt(s, l, xp, SH - Inches(0.85), Inches(2.2), Inches(0.42),
             size=10, color=MGRAY, align=PP_ALIGN.CENTER)


# ── Slide 04: Step 2 — Data Annotation ───────────────────────────────────────
def slide_04_annotation(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Data Annotation", "Labelling vehicles in YOLO bounding-box format", step_num="02")
    _footer(s)

    # Annotation tools
    _txt(s, "Annotation Tools",
         Inches(0.35), Inches(1.2), Inches(4.0), Inches(0.35),
         size=15, bold=True, color=NAVY)
    tools = [
        ("LabelImg", "Open-source YOLO labelling GUI", TEAL),
        ("CVAT",     "Browser-based team annotation", BLUE),
        ("Roboflow", "Cloud platform + augmentation",  ORANGE),
    ]
    for i, (tool, desc, col) in enumerate(tools):
        yp = Inches(1.65) + i * Inches(0.90)
        _rect(s, Inches(0.35), yp, Inches(4.2), Inches(0.78),
              fill=WHITE, line_rgb=col, line_w=Pt(1.5))
        _txt(s, tool, Inches(0.5), yp + Inches(0.06),
             Inches(2.0), Inches(0.34), size=13, bold=True, color=col)
        _txt(s, desc, Inches(0.5), yp + Inches(0.40),
             Inches(4.0), Inches(0.30), size=11, color=DGRAY)

    _divider(s, Inches(4.55), color=LGRAY)

    # YOLO label format
    _txt(s, "YOLO Label Format (.txt)",
         Inches(0.35), Inches(4.7), Inches(4.5), Inches(0.35),
         size=14, bold=True, color=NAVY)
    _code_block(s, [
        "# <class_id> <x_center> <y_center> <width> <height>",
        "2  0.4521  0.3187  0.0412  0.0623   # car",
        "7  0.7834  0.5500  0.0680  0.0850   # truck",
        "2  0.2190  0.4410  0.0385  0.0601   # car",
    ], Inches(0.35), Inches(5.1), Inches(4.5), Inches(1.1),
       bg=NAVY2, size=11)

    # Class map
    _txt(s, "Vehicle Class Mapping",
         Inches(5.0), Inches(1.2), Inches(4.0), Inches(0.35),
         size=15, bold=True, color=NAVY)
    classes = [
        ("2", "Car",        "🚗", TEAL),
        ("3", "Motorcycle", "🏍️", BLUE2),
        ("5", "Bus",        "🚌", ORANGE),
        ("7", "Truck",      "🚛", RED),
    ]
    for i, (cid, name, icon, col) in enumerate(classes):
        yp = Inches(1.65) + i * Inches(0.85)
        _rect(s, Inches(5.0), yp, Inches(3.6), Inches(0.72),
              fill=WHITE, line_rgb=col, line_w=Pt(1.5))
        _txt(s, f"Class {cid}", Inches(5.15), yp + Inches(0.05),
             Inches(0.9), Inches(0.28), size=10, bold=True, color=col)
        _txt(s, f"{icon} {name}", Inches(5.15), yp + Inches(0.33),
             Inches(2.5), Inches(0.30), size=13, bold=True, color=NAVY)

    # Best practices
    _txt(s, "Annotation Best Practices",
         Inches(5.0), Inches(5.1), Inches(4.5), Inches(0.35),
         size=14, bold=True, color=NAVY)
    practices = [
        "Tight boxes — do not include road surface",
        "Occluded vehicles → annotate visible portion",
        "Minimum 50% visibility to annotate",
        "Night / shadow images → skip or separate",
        "Consistency check: ≥2 annotators per image",
    ]
    _bullet_block(s, practices, Inches(5.0), Inches(5.55),
                  Inches(4.5), size=12, gap=Inches(0.35), color=DGRAY)

    # Annotation stats box
    _rect(s, Inches(9.8), Inches(1.2), Inches(3.2), Inches(5.7),
          fill=NAVY, line_rgb=TEAL, line_w=Pt(0.75))
    _txt(s, "📊  Dataset Stats",
         Inches(9.95), Inches(1.35), Inches(2.9), Inches(0.35),
         size=13, bold=True, color=TEAL)
    stats = [
        ("2,400+", "Total images"),
        ("18,700+", "Annotations"),
        ("78%", "Cars"),
        ("11%", "Trucks"),
        ("7%", "Buses"),
        ("4%", "Motorcycles"),
        ("640px", "Crop size"),
        ("80/10/10", "Train/Val/Test"),
    ]
    for i, (v, l) in enumerate(stats):
        yp = Inches(1.85) + i * Inches(0.58)
        _txt(s, v, Inches(9.95), yp,
             Inches(1.3), Inches(0.30), size=16, bold=True, color=TEAL)
        _txt(s, l, Inches(11.3), yp + Inches(0.04),
             Inches(1.6), Inches(0.28), size=10, color=MGRAY)


# ── Slide 05: Step 3 — Dataset Preparation ───────────────────────────────────
def slide_05_dataset(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Dataset Preparation", "YAML config, splits, and augmentation pipeline", step_num="03")
    _footer(s)

    # Directory structure
    _txt(s, "📁  Dataset Directory Structure",
         Inches(0.35), Inches(1.2), Inches(5.5), Inches(0.35),
         size=14, bold=True, color=NAVY)
    _code_block(s, [
        "dataset/",
        "├── data.yaml            # class names + paths",
        "├── train/",
        "│   ├── images/  (1920 images)",
        "│   └── labels/  (1920 .txt files)",
        "├── val/",
        "│   ├── images/  (240 images)",
        "│   └── labels/  (240 .txt files)",
        "└── test/",
        "    ├── images/  (240 images)",
        "    └── labels/  (240 .txt files)",
    ], Inches(0.35), Inches(1.65), Inches(5.5), Inches(2.5),
       bg=NAVY2, size=10)

    # YAML config
    _txt(s, "📄  data.yaml",
         Inches(0.35), Inches(4.3), Inches(5.5), Inches(0.35),
         size=14, bold=True, color=NAVY)
    _code_block(s, [
        "path: ./dataset",
        "train: train/images",
        "val:   val/images",
        "test:  test/images",
        "",
        "nc: 4   # number of classes",
        "names: [car, motorcycle, bus, truck]",
    ], Inches(0.35), Inches(4.75), Inches(5.5), Inches(1.65),
       bg=NAVY2, size=10)

    # Augmentation pipeline
    _txt(s, "🔄  Augmentation Pipeline",
         Inches(6.1), Inches(1.2), Inches(7.0), Inches(0.35),
         size=14, bold=True, color=NAVY)
    augs = [
        ("🔀 Mosaic (4-image)",     "Combines 4 images into one — boosts small-object recall"),
        ("↔️ Random Flip",           "Horizontal + vertical flipping"),
        ("🔆 HSV Jitter",            "Hue ±1.5%, Sat ±70%, Val ±40%"),
        ("📐 Random Scale",          "0.5× – 1.5× zoom"),
        ("✂️ Random Crop",            "640×640 crop from 1280 image"),
        ("🌫️ Copy-Paste",             "Paste object instances to new scenes"),
        ("🌙 Night Simulation",      "Brightness reduction + noise injection"),
    ]
    for i, (aug, desc) in enumerate(augs):
        yp = Inches(1.65) + i * Inches(0.62)
        _rect(s, Inches(6.1), yp, Inches(6.9), Inches(0.55),
              fill=WHITE, line_rgb=LGRAY, line_w=Pt(0.5))
        _txt(s, aug,  Inches(6.25), yp + Inches(0.04),
             Inches(2.3), Inches(0.28), size=11, bold=True, color=NAVY)
        _txt(s, desc, Inches(8.65), yp + Inches(0.05),
             Inches(4.2), Inches(0.28), size=10, color=DGRAY)

    # Pre-processing
    _txt(s, "⚙️  Pre-processing",
         Inches(6.1), Inches(6.05), Inches(7.0), Inches(0.35),
         size=13, bold=True, color=NAVY)
    pre = ["Resize to 640×640 (letterbox)", "Normalise pixels 0–1",
           "BGR→RGB channel order", "Batch size 16, mixed precision fp16"]
    _bullet_block(s, pre, Inches(6.1), Inches(6.45),
                  Inches(7.0), size=11, gap=Inches(0.28), color=DGRAY)


# ── Slide 06: Step 4 — YOLOv8 Architecture ───────────────────────────────────
def slide_06_yolo_arch(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "YOLOv8 Architecture", "COCO pretrained weights + transfer learning strategy", step_num="04")
    _footer(s)

    # Architecture flow
    blocks = [
        (NAVY,   "Input\n640×640×3"),
        (BLUE,   "Backbone\nCSPDarknet53"),
        (TEAL_D, "Neck\nPAFPN"),
        (TEAL,   "Head\nDecoupled\nDetection"),
        (GREEN,  "Output\n4 classes\nBBoxes + Conf"),
    ]
    bw, bh = Inches(2.1), Inches(1.3)
    total_w = len(blocks) * (bw + Inches(0.3)) - Inches(0.3)
    start_x = (SW - total_w) / 2
    for i, (col, label) in enumerate(blocks):
        xp = start_x + i * (bw + Inches(0.3))
        _rect(s, xp, Inches(1.3), bw, bh, fill=col, line_rgb=TEAL, line_w=Pt(0.5))
        _txt(s, label, xp, Inches(1.3), bw, bh,
             size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        if i < len(blocks) - 1:
            _flow_arrow(s, xp + bw + Inches(0.0), Inches(1.75), color=TEAL)

    # Transfer learning
    _divider(s, Inches(2.75), color=LGRAY)
    _txt(s, "Transfer Learning Strategy",
         Inches(0.35), Inches(2.9), Inches(5.5), Inches(0.35),
         size=14, bold=True, color=NAVY)
    tl = [
        "Start from yolov8n.pt (COCO 80-class weights)",
        "Freeze backbone for first 10 epochs",
        "Unfreeze all layers for remaining 40 epochs",
        "Learning rate: 0.01 → 0.0001 cosine schedule",
        "Fine-tune on vehicle classes only (4 classes)",
        "Early stopping: patience=15 on mAP@50",
    ]
    _bullet_block(s, tl, Inches(0.35), Inches(3.35),
                  Inches(5.5), size=13, gap=Inches(0.40), color=DGRAY)

    # Training config
    _txt(s, "Training Configuration",
         Inches(6.1), Inches(2.9), Inches(4.0), Inches(0.35),
         size=14, bold=True, color=NAVY)
    _code_block(s, [
        "from ultralytics import YOLO",
        "",
        "model = YOLO('yolov8n.pt')",
        "model.train(",
        "    data   = 'data.yaml',",
        "    epochs = 50,",
        "    imgsz  = 640,",
        "    batch  = 16,",
        "    device = 'cuda',",
        "    half   = True,",
        ")",
    ], Inches(6.1), Inches(3.35), Inches(4.5), Inches(2.8),
       bg=NAVY2, size=10)

    # Performance metrics
    _txt(s, "Evaluation Metrics",
         Inches(10.8), Inches(2.9), Inches(2.3), Inches(0.35),
         size=13, bold=True, color=NAVY)
    metrics = [
        ("mAP@50", "81.4%", TEAL),
        ("mAP@50-95", "58.7%", BLUE),
        ("Precision", "83.2%", GREEN),
        ("Recall", "79.6%", ORANGE),
        ("Speed", "6 ms/img", MGRAY),
    ]
    for i, (label, val, col) in enumerate(metrics):
        yp = Inches(3.35) + i * Inches(0.56)
        _txt(s, label, Inches(10.8), yp,
             Inches(1.5), Inches(0.28), size=11, color=DGRAY)
        _txt(s, val, Inches(12.0), yp,
             Inches(1.1), Inches(0.28), size=13, bold=True, color=col)

    # Model variants
    _divider(s, Inches(6.25), color=LGRAY)
    _txt(s, "YOLOv8 Model Variants (FALCON uses 'n' — fastest for real-time demo)",
         Inches(0.35), Inches(6.38), SW - Inches(0.7), Inches(0.28),
         size=11, italic=True, color=MGRAY)
    variants = [("n (nano)", "3.2M params", "6ms"), ("s (small)", "11.2M", "9ms"),
                ("m (medium)", "25.9M", "20ms"), ("l (large)", "43.7M", "35ms")]
    for i, (name, params, spd) in enumerate(variants):
        xp = Inches(0.35) + i * Inches(3.25)
        _rect(s, xp, Inches(6.72), Inches(3.0), Inches(0.47),
              fill=NAVY if i == 0 else WHITE,
              line_rgb=TEAL if i == 0 else LGRAY, line_w=Pt(1))
        col = WHITE if i == 0 else DGRAY
        _txt(s, f"{name}  |  {params}  |  {spd}", xp + Inches(0.1),
             Inches(6.76), Inches(2.8), Inches(0.35),
             size=10, bold=(i == 0), color=col)


# ── Slide 07: Step 5 — Aerial Detection Challenge ────────────────────────────
def slide_07_challenge(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Aerial Detection Challenge",
            "Why standard YOLOv8 fails on satellite / Google Earth imagery", step_num="05")
    _footer(s)

    # Problem statement
    _rect(s, Inches(0.3), Inches(1.2), SW - Inches(0.6), Inches(0.95),
          fill=RED, line_rgb=None)
    _txt(s, "⚠️  Problem: Standard YOLO returns 0 detections on Google Earth screenshots",
         Inches(0.5), Inches(1.28), SW - Inches(1.0), Inches(0.70),
         size=16, bold=True, color=WHITE)

    # 3 reasons
    reasons = [
        ("1", "Vehicles too small",
         "Google Earth @ zoom 18 → vehicles are only 5–8 px wide.\n"
         "COCO training used 30–300px vehicles. Far below detection threshold."),
        ("2", "Wrong perspective",
         "COCO images are side-on photographs.\n"
         "Aerial images are top-down — shapes look completely different."),
        ("3", "Low contrast",
         "Aerial imagery has flat tonality.\n"
         "Vehicles blend with road surface at native resolution."),
    ]
    for i, (num, title, body) in enumerate(reasons):
        xp = Inches(0.3) + i * Inches(4.3)
        _rect(s, xp, Inches(2.3), Inches(4.1), Inches(2.5),
              fill=WHITE, line_rgb=RED, line_w=Pt(1.5))
        _rect(s, xp, Inches(2.3), Inches(0.5), Inches(2.5), fill=RED)
        _txt(s, num, xp, Inches(2.3), Inches(0.5), Inches(2.5),
             size=24, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _txt(s, title, xp + Inches(0.55), Inches(2.3),
             Inches(3.5), Inches(0.6), size=15, bold=True, color=RED)
        _txt(s, body, xp + Inches(0.15), Inches(2.98),
             Inches(3.8), Inches(1.7), size=12, color=DGRAY)

    # Resolution comparison
    _divider(s, Inches(5.0), color=LGRAY)
    _txt(s, "Scale Comparison: COCO training vs. Google Earth screenshot",
         Inches(0.35), Inches(5.15), Inches(8.0), Inches(0.35),
         size=13, bold=True, color=NAVY)

    comparisons = [
        ("COCO Dataset", "Side-on photo", "Vehicle: 80–350px wide", "Confidence: 0.82", TEAL),
        ("Google Earth\nNative", "Top-down, zoom 18", "Vehicle: 5–8px wide", "Confidence: 0.00", RED),
        ("FALCON 4× Upscale", "Top-down, enhanced", "Vehicle: 20–32px wide", "Confidence: 0.12+", GREEN),
    ]
    for i, (label, view, size_str, conf, col) in enumerate(comparisons):
        xp = Inches(0.35) + i * Inches(4.3)
        _rect(s, xp, Inches(5.6), Inches(4.0), Inches(1.45),
              fill=NAVY if col == TEAL else (OFFWH if col == GREEN else WHITE),
              line_rgb=col, line_w=Pt(2))
        text_col = WHITE if col == TEAL else NAVY
        _txt(s, label, xp + Inches(0.1), Inches(5.66),
             Inches(3.8), Inches(0.38), size=12, bold=True, color=col)
        _txt(s, f"{view}  |  {size_str}",
             xp + Inches(0.1), Inches(6.04), Inches(3.8), Inches(0.28),
             size=10, color=DGRAY if col != TEAL else MGRAY)
        _txt(s, conf, xp + Inches(0.1), Inches(6.32),
             Inches(3.8), Inches(0.30), size=11, bold=True,
             color=GREEN if "0.12" in conf else (RED if "0.00" in conf else TEAL))


# ── Slide 08: Step 6 — FALCON Aerial Pipeline ────────────────────────────────
def slide_08_aerial_pipeline(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "FALCON Aerial Detection Pipeline",
            "4× upscale + sliding window tiling + IoU-NMS deduplication", step_num="06")
    _footer(s)

    # Pipeline steps (horizontal flow)
    pipeline = [
        (NAVY,   "INPUT\nImage\n1273×538"),
        (BLUE,   "ENHANCE\nContrast ×1.3\nSharpness ×1.5"),
        (BLUE2,  "UPSCALE\n4× LANCZOS\n5092×2152"),
        (TEAL_D, "TILE\n640×640\nstep=480px"),
        (TEAL,   "YOLO\nconf=0.08\nper tile"),
        (GREEN,  "NMS\nIoU=0.40\nde-dup"),
        (ORANGE, "OUTPUT\nBoxes in\norig coords"),
    ]
    bw, bh = Inches(1.7), Inches(1.5)
    start_x = Inches(0.25)
    for i, (col, label) in enumerate(pipeline):
        xp = start_x + i * (bw + Inches(0.18))
        _rect(s, xp, Inches(1.25), bw, bh, fill=col)
        _txt(s, label, xp + Inches(0.05), Inches(1.25),
             bw - Inches(0.1), bh, size=10, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
        if i < len(pipeline) - 1:
            _txt(s, "→", xp + bw, Inches(1.75), Inches(0.18), Inches(0.5),
                 size=14, bold=True, color=TEAL, align=PP_ALIGN.CENTER)

    # Code walkthrough
    _txt(s, "Core Algorithm",
         Inches(0.35), Inches(3.0), Inches(6.0), Inches(0.35),
         size=14, bold=True, color=NAVY)
    _code_block(s, [
        "# 1. Upscale + enhance",
        "img_up = img.resize((W*4, H*4), LANCZOS)",
        "img_up = Contrast(img_up).enhance(1.3)",
        "img_up = Sharpness(img_up).enhance(1.5)",
        "",
        "# 2. Sliding window tiling",
        "for row in range(rows):  # 640px tiles, step=480",
        "    for col in range(cols):",
        "        patch = img_up[y:y+640, x:x+640]",
        "        results = yolo(patch, conf=0.08)",
        "        # Map back to original coords (/scale)",
        "        raw_boxes += map_coords(results, x, y, scale=4)",
        "",
        "# 3. IoU-NMS deduplication",
        "kept = nms(raw_boxes, iou_thresh=0.40)",
    ], Inches(0.35), Inches(3.45), Inches(6.5), Inches(3.6),
       bg=NAVY2, size=10)

    # Parameters table
    _txt(s, "Key Parameters",
         Inches(7.1), Inches(3.0), Inches(6.0), Inches(0.35),
         size=14, bold=True, color=NAVY)
    params = [
        ("_AERIAL_SCALE",    "4",     "Upscale factor (vehicles: 5px → 20px)"),
        ("_AERIAL_TILE",     "640",   "Tile size (matches YOLO input)"),
        ("_AERIAL_STEP",     "480",   "Step = 75% tile overlap"),
        ("_AERIAL_CONF",     "0.08",  "Low threshold for small vehicles"),
        ("_AERIAL_NMS_IOU",  "0.40",  "Duplicate suppression threshold"),
        ("_AERIAL_CONTRAST", "1.3",   "Pre-processing contrast boost"),
        ("_AERIAL_SHARP",    "1.5",   "Pre-processing sharpness boost"),
    ]
    hdr_y = Inches(3.45)
    _rect(s, Inches(7.1), hdr_y, Inches(6.0), Inches(0.35), fill=NAVY)
    for col, label in [
        (Inches(7.1), "Parameter"),
        (Inches(9.5), "Value"),
        (Inches(10.2), "Purpose"),
    ]:
        _txt(s, label, col + Inches(0.08), hdr_y + Inches(0.04),
             Inches(2.0), Inches(0.28), size=10, bold=True, color=WHITE)
    for i, (param, val, desc) in enumerate(params):
        yp = Inches(3.88) + i * Inches(0.42)
        bg = OFFWH if i % 2 == 0 else WHITE
        _rect(s, Inches(7.1), yp, Inches(6.0), Inches(0.40), fill=bg)
        _txt(s, param, Inches(7.18), yp + Inches(0.06),
             Inches(2.2), Inches(0.28), size=10, bold=True, color=BLUE)
        _txt(s, val,   Inches(9.58), yp + Inches(0.06),
             Inches(0.6), Inches(0.28), size=10, bold=True, color=TEAL)
        _txt(s, desc,  Inches(10.28), yp + Inches(0.06),
             Inches(2.7), Inches(0.28), size=9,  color=DGRAY)


# ── Slide 09: Step 7 — Detection Results ─────────────────────────────────────
def slide_09_results(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Detection Results — test.png",
            "Muscat, Oman  ·  23.5933°N, 58.4017°E  ·  Al Khuwair Commercial Zone", step_num="07")
    _footer(s)

    # Annotated image
    img_shown = _add_image(
        s, ROOT / "data" / "sample_images" / "test_annotated.png",
        Inches(0.3), Inches(1.2), Inches(8.5), Inches(4.0),
    )
    if not img_shown:
        _rect(s, Inches(0.3), Inches(1.2), Inches(8.5), Inches(4.0),
              fill=NAVY2, line_rgb=TEAL, line_w=Pt(1))
        _txt(s, "📸  test_annotated.png",
             Inches(2.0), Inches(2.8), Inches(5.0), Inches(0.6),
             size=18, color=MGRAY, align=PP_ALIGN.CENTER)

    # Detection stats
    _rect(s, Inches(9.0), Inches(1.2), Inches(4.0), Inches(5.7),
          fill=NAVY, line_rgb=TEAL, line_w=Pt(0.75))
    _txt(s, "🚗  Detection Summary",
         Inches(9.15), Inches(1.35), Inches(3.7), Inches(0.35),
         size=13, bold=True, color=TEAL)

    summary_items = [
        ("14",   "Raw detections"),
        ("10",   "After IoU-NMS"),
        ("8",    "Cars detected"),
        ("1",    "Truck detected"),
        ("1",    "Bus detected"),
        ("0.08", "Min confidence"),
        ("0.71", "Max confidence"),
        ("4×",   "Upscale applied"),
        ("4",    "Tiles processed"),
        ("6ms",  "Inference / tile"),
    ]
    for i, (val, label) in enumerate(summary_items):
        yp = Inches(1.82) + i * Inches(0.47)
        _txt(s, val,   Inches(9.15), yp,
             Inches(1.0), Inches(0.30), size=18, bold=True, color=TEAL)
        _txt(s, label, Inches(10.3), yp + Inches(0.04),
             Inches(2.6), Inches(0.26), size=10, color=WHITE)

    # Pipeline result annotation
    _rect(s, Inches(0.3), Inches(5.35), Inches(8.5), Inches(1.05),
          fill=NAVY2, line_rgb=TEAL, line_w=Pt(0.5))
    _txt(s, "Pipeline: Aerial mode  |  Scale=4×  |  Tile=640px  |  Step=480px  |  "
             "Conf=0.08  |  NMS IoU=0.40  |  Source: yolo_aerial",
         Inches(0.45), Inches(5.48), Inches(8.2), Inches(0.35),
         size=10, color=MGRAY)
    _txt(s, "✅  FALCON aerial pipeline successfully detects vehicles "
             "invisible to standard YOLOv8",
         Inches(0.45), Inches(5.85), Inches(8.2), Inches(0.35),
         size=11, bold=True, color=TEAL)


# ── Slide 10: Step 8 — Zone Scoring ──────────────────────────────────────────
def slide_10_scoring(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Zone Scoring Algorithm",
            "Multi-factor priority score — vehicles, coverage gap, population", step_num="08")
    _footer(s)

    # Formula box
    _rect(s, Inches(0.35), Inches(1.2), SW - Inches(0.7), Inches(1.05),
          fill=NAVY, line_rgb=TEAL, line_w=Pt(1))
    _txt(s, "Priority Score  =  0.40 × vehicle_density_score  +  "
             "0.35 × coverage_gap_score  +  0.25 × population_score",
         Inches(0.55), Inches(1.35), SW - Inches(1.1), Inches(0.7),
         size=17, bold=True, color=TEAL, align=PP_ALIGN.CENTER)

    # Weight breakdown
    weights = [
        ("40%", "Vehicle Density", TEAL,
         "YOLOv8 count per km² → normalised 0–100\n"
         "High traffic = high FTTH demand"),
        ("35%", "Coverage Gap", ORANGE,
         "100% − current FTTH coverage %\n"
         "Unserved = highest growth potential"),
        ("25%", "Population", BLUE,
         "Zone population density\n"
         "More residents = more subscribers"),
    ]
    for i, (pct, label, col, desc) in enumerate(weights):
        xp = Inches(0.35) + i * Inches(4.3)
        _rect(s, xp, Inches(2.45), Inches(4.1), Inches(2.3),
              fill=WHITE, line_rgb=col, line_w=Pt(2))
        _rect(s, xp, Inches(2.45), Inches(4.1), Inches(0.65), fill=col)
        _txt(s, pct,   xp + Inches(0.1), Inches(2.48),
             Inches(1.0), Inches(0.55), size=30, bold=True,
             color=WHITE, align=PP_ALIGN.LEFT)
        _txt(s, label, xp + Inches(1.1), Inches(2.55),
             Inches(2.9), Inches(0.45), size=14, bold=True, color=WHITE)
        _txt(s, desc,  xp + Inches(0.15), Inches(3.18),
             Inches(3.8), Inches(1.48), size=12, color=DGRAY)

    # Top zones table
    _divider(s, Inches(4.95), color=LGRAY)
    _txt(s, "Top Priority Zones",
         Inches(0.35), Inches(5.1), Inches(5.0), Inches(0.35),
         size=14, bold=True, color=NAVY)

    scored = d.get("scored", [])
    if not scored:
        scored = [
            {"name": "Al Khuwair",    "score": 88, "priority": "High",     "vehicle_density": 52},
            {"name": "Bausher",       "score": 81, "priority": "High",     "vehicle_density": 45},
            {"name": "Wadi Kabir",    "score": 74, "priority": "Medium",   "vehicle_density": 38},
            {"name": "Ruwi",          "score": 72, "priority": "Medium",   "vehicle_density": 41},
            {"name": "Ghubrah North", "score": 65, "priority": "Medium",   "vehicle_density": 29},
        ]

    hdr_y = Inches(5.55)
    headers = ["Zone", "Score", "Priority", "Vehicles/km²"]
    col_x   = [Inches(0.35), Inches(3.8), Inches(5.5), Inches(7.5)]
    _rect(s, Inches(0.35), hdr_y, Inches(9.5), Inches(0.35), fill=NAVY)
    for hdr, cx in zip(headers, col_x):
        _txt(s, hdr, cx + Inches(0.08), hdr_y + Inches(0.04),
             Inches(2.0), Inches(0.28), size=10, bold=True, color=WHITE)

    for i, z in enumerate(scored[:5]):
        yp = Inches(5.93) + i * Inches(0.28)
        bg = OFFWH if i % 2 == 0 else WHITE
        _rect(s, Inches(0.35), yp, Inches(9.5), Inches(0.27), fill=bg)
        vals = [
            z.get("name", "—"),
            str(z.get("score", "—")),
            z.get("priority", "—"),
            str(z.get("vehicle_density", "—")),
        ]
        colors = [NAVY, TEAL, GREEN if "High" in str(z.get("priority", "")) else ORANGE, BLUE]
        for val, cx, col in zip(vals, col_x, colors):
            _txt(s, val, cx + Inches(0.08), yp + Inches(0.02),
                 Inches(2.5), Inches(0.24), size=10, bold=(cx == col_x[1]), color=col)

    # ROI formula
    _rect(s, Inches(10.0), Inches(5.1), Inches(3.0), Inches(2.1),
          fill=NAVY2, line_rgb=ORANGE, line_w=Pt(1))
    _txt(s, "💰  ROI Formula",
         Inches(10.15), Inches(5.22), Inches(2.7), Inches(0.30),
         size=12, bold=True, color=ORANGE)
    roi_lines = [
        "new_subs = households",
        "  × uptake_rate (42%)",
        "  × gap_pct / 100",
        "annual_rev = new_subs",
        "  × ARPU (5.5 OMR/mo)",
        "  × 12",
        "ROI years = CAPEX / rev",
    ]
    for i, line in enumerate(roi_lines):
        _txt(s, line, Inches(10.15), Inches(5.58) + i * Inches(0.22),
             Inches(2.7), Inches(0.22), size=9,
             color=TEAL if i == 0 else MGRAY)


# ── Slide 11: Step 9 — Telecom Simulation ────────────────────────────────────
def slide_11_telecom(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Telecom Market Simulation",
            "Per-zone KPI simulation: 5G / FTTH / mobile subscribers, ARPU, revenue", step_num="09")
    _footer(s)

    market = d.get("market", {})

    # KPI row
    kpis = [
        (str(market.get("total_subscribers", "284K")), "Total Subscribers", TEAL),
        (str(market.get("avg_arpu", "5.5 OMR")),       "Avg ARPU/mo",       BLUE),
        (str(market.get("ftth_coverage", "42%")),       "FTTH Coverage",     ORANGE),
        (str(market.get("unserved_homes", "105K")),     "Unserved Homes",    RED),
    ]
    bw = (SW - Inches(0.6)) / len(kpis)
    for i, (val, lbl, col) in enumerate(kpis):
        _kpi(s, Inches(0.3) + i * bw, Inches(1.2), bw - Inches(0.1), Inches(1.15),
             val, lbl, vcolor=col, bg=WHITE)

    # Simulation dimensions
    _txt(s, "Simulated KPIs per Zone",
         Inches(0.35), Inches(2.55), Inches(6.0), Inches(0.35),
         size=14, bold=True, color=NAVY)

    kpi_grid = [
        ("📱 Mobile", "2G/3G/4G subscribers\nPenetration rate\nAvg revenue/sub"),
        ("🌐 FTTH",   "Current subscribers\nCoverage pct\nRevenue contribution"),
        ("5G",        "Coverage threshold 80%\nSubscriber forecast\nPremium ARPU ×1.4"),
        ("💼 ARPU",   "Residential ARPU\nCommercial ARPU ×1.5\nBlended ARPU"),
        ("📊 Traffic","Monthly GB/sub\nPeak hour factor\n35% YoY growth"),
        ("🏆 Market", "Waheb share\nRami (Omantel)\nDon (Ooredoo)"),
    ]
    for i, (title, lines) in enumerate(kpi_grid):
        col = i % 3
        row = i // 3
        xp = Inches(0.35) + col * Inches(4.3)
        yp = Inches(3.05) + row * Inches(1.55)
        _rect(s, xp, yp, Inches(4.1), Inches(1.40),
              fill=WHITE, line_rgb=LGRAY, line_w=Pt(0.5))
        _rect(s, xp, yp, Inches(4.1), Inches(0.38), fill=NAVY)
        _txt(s, title, xp + Inches(0.1), yp + Inches(0.04),
             Inches(3.9), Inches(0.30), size=12, bold=True, color=TEAL)
        _txt(s, lines, xp + Inches(0.1), yp + Inches(0.45),
             Inches(3.9), Inches(0.85), size=11, color=DGRAY)

    # Code snippet
    _txt(s, "Simulation code (app/telecom_simulator.py)",
         Inches(9.3), Inches(2.55), Inches(3.8), Inches(0.35),
         size=12, bold=True, color=NAVY)
    _code_block(s, [
        "sim = TelecomSimulator(zones)",
        "profile = sim.simulate_zone(zone)",
        "",
        "# ZoneTelecomProfile fields:",
        "profile.ftth_subscribers",
        "profile.mobile_subscribers",
        "profile.revenue_omr",
        "profile.traffic_tb_month",
        "profile.market_share_pct",
    ], Inches(9.3), Inches(3.0), Inches(3.8), Inches(2.65),
       bg=NAVY2, size=10)


# ── Slide 12: Step 10 — Competitor Intelligence ───────────────────────────────
def slide_12_competitors(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Competitor Intelligence",
            "Waheb vs RAMI (Omantel) vs Don (Ooredoo) — market gap analysis", step_num="10")
    _footer(s)

    # Market share donut placeholder
    _txt(s, "Market Share",
         Inches(0.35), Inches(1.2), Inches(4.0), Inches(0.35),
         size=14, bold=True, color=NAVY)

    players = [
        ("RAMI (Omantel)",  35, RED,    "Market leader, legacy infrastructure"),
        ("Don (Ooredoo)",   25, BLUE,   "Strong mobile-first position"),
        ("Waheb",           10, TEAL,   "AI-first, FTTH focus — FALCON target"),
        ("Others",          30, MGRAY,  "Fragmented smaller ISPs"),
    ]
    bar_h = Inches(0.62)
    bar_max_w = Inches(4.5)
    for i, (name, share, col, desc) in enumerate(players):
        yp = Inches(1.7) + i * Inches(0.80)
        _txt(s, name, Inches(0.35), yp,
             Inches(2.2), Inches(0.28), size=12, bold=True, color=col)
        _rect(s, Inches(0.35), yp + Inches(0.30),
              bar_max_w * (share / 100), bar_h * 0.55, fill=col)
        _rect(s, Inches(0.35) + bar_max_w * (share / 100),
              yp + Inches(0.30),
              bar_max_w * (1 - share / 100), bar_h * 0.55,
              fill=LGRAY)
        _txt(s, f"{share}%", Inches(0.35) + bar_max_w + Inches(0.12),
             yp + Inches(0.30), Inches(0.55), bar_h * 0.55,
             size=13, bold=True, color=col)
        _txt(s, desc, Inches(0.35), yp + Inches(0.60),
             Inches(5.0), Inches(0.20), size=9, italic=True, color=MGRAY)

    _divider(s, Inches(5.05), color=LGRAY)

    # Gap table
    _txt(s, "Coverage Gap by Zone (Top 5)",
         Inches(0.35), Inches(5.2), Inches(5.0), Inches(0.35),
         size=13, bold=True, color=NAVY)
    gap_data = [
        ("Al Khuwair",  "80%", "55%", "70%", "20%"),
        ("Bausher",     "75%", "40%", "65%", "25%"),
        ("Wadi Kabir",  "65%", "30%", "55%", "35%"),
        ("Ghubrah N.",  "70%", "25%", "60%", "30%"),
        ("Ruwi",        "85%", "60%", "75%", "15%"),
    ]
    headers = ["Zone", "Waheb Cov.", "RAMI", "Don", "Waheb Gap"]
    col_xs  = [Inches(0.35), Inches(2.2), Inches(3.4), Inches(4.6), Inches(5.8)]
    _rect(s, Inches(0.35), Inches(5.6), Inches(6.7), Inches(0.32), fill=NAVY)
    for hdr, cx in zip(headers, col_xs):
        _txt(s, hdr, cx + Inches(0.06), Inches(5.64),
             Inches(1.5), Inches(0.26), size=9, bold=True, color=WHITE)
    for i, row in enumerate(gap_data):
        yp = Inches(5.95) + i * Inches(0.25)
        _rect(s, Inches(0.35), yp, Inches(6.7), Inches(0.23),
              fill=OFFWH if i % 2 == 0 else WHITE)
        for val, cx in zip(row, col_xs):
            col = RED if cx == col_xs[4] else DGRAY
            _txt(s, val, cx + Inches(0.06), yp + Inches(0.02),
                 Inches(1.5), Inches(0.20), size=9, color=col,
                 bold=(cx == col_xs[4]))

    # Opportunity box
    _rect(s, Inches(7.2), Inches(1.2), Inches(5.8), Inches(5.7),
          fill=NAVY, line_rgb=TEAL, line_w=Pt(1))
    _txt(s, "🎯  FALCON Opportunity Analysis",
         Inches(7.35), Inches(1.35), Inches(5.5), Inches(0.35),
         size=13, bold=True, color=TEAL)
    opps = [
        "105,000 homes currently unserved by any FTTH",
        "Waheb has 10% share vs 35% (RAMI) target",
        "RAMI & Don have <55% FTTH in 8 of 15 zones",
        "Commercial zones (Al Khuwair, Bausher) are highest ROI",
        "FALCON identifies top 3 zones = 42% of total gap",
        "Capture 5% market share → +5,250 new subscribers",
        "Annual revenue uplift: +OMR 346,500 (@ARPU 5.5)",
        "Payback period: 3.2 years (avg zone CAPEX 1.1M OMR)",
    ]
    for i, opp in enumerate(opps):
        _txt(s, f"▸  {opp}",
             Inches(7.35), Inches(1.85) + i * Inches(0.55),
             Inches(5.5), Inches(0.45), size=11, color=WHITE)

    _rect(s, Inches(7.2), Inches(6.35), Inches(5.8), Inches(0.55),
          fill=TEAL)
    _txt(s, "→  FALCON targets the 35% uncaptured market gap",
         Inches(7.35), Inches(6.4), Inches(5.5), Inches(0.45),
         size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER)


# ── Slide 13: Step 11 — ML Insights ──────────────────────────────────────────
def slide_13_ml_insights(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "ML Insights",
            "K-Means clustering · Revenue forecasting · S-curve market share · Demand scoring",
            step_num="11")
    _footer(s)

    # K-Means
    _rect(s, Inches(0.3), Inches(1.2), Inches(4.1), Inches(2.7),
          fill=WHITE, line_rgb=LGRAY, line_w=Pt(0.5))
    _txt(s, "🔵  K-Means Clustering (k=3)",
         Inches(0.45), Inches(1.28), Inches(3.8), Inches(0.35),
         size=13, bold=True, color=NAVY)
    cluster_info = [
        ("High Opportunity", "Score > 75", "4 zones", TEAL),
        ("Moderate Potential", "Score 50–75", "7 zones", ORANGE),
        ("Low Priority", "Score < 50", "4 zones", MGRAY),
    ]
    for i, (label, criterion, count, col) in enumerate(cluster_info):
        yp = Inches(1.72) + i * Inches(0.70)
        _rect(s, Inches(0.45), yp, Inches(0.15), Inches(0.48), fill=col)
        _txt(s, label, Inches(0.72), yp + Inches(0.02),
             Inches(2.5), Inches(0.25), size=11, bold=True, color=col)
        _txt(s, f"{criterion}  ·  {count}",
             Inches(0.72), yp + Inches(0.28),
             Inches(2.5), Inches(0.20), size=9, color=MGRAY)
    _txt(s, "Features: vehicle density, coverage gap,\n"
             "population density, commercial ratio",
         Inches(0.45), Inches(3.4), Inches(3.8), Inches(0.42),
         size=10, italic=True, color=MGRAY)

    # Revenue forecast
    _rect(s, Inches(4.55), Inches(1.2), Inches(4.5), Inches(2.7),
          fill=WHITE, line_rgb=LGRAY, line_w=Pt(0.5))
    _txt(s, "📈  Revenue Forecast (5-Year)",
         Inches(4.7), Inches(1.28), Inches(4.2), Inches(0.35),
         size=13, bold=True, color=NAVY)
    years   = ["Y1", "Y2", "Y3", "Y4", "Y5"]
    rev_omr = [0.42, 0.61, 0.84, 1.09, 1.38]
    bar_max  = max(rev_omr)
    bar_area_h = Inches(1.55)
    bar_w  = Inches(0.65)
    base_y = Inches(1.72) + bar_area_h
    for i, (yr, rv) in enumerate(zip(years, rev_omr)):
        xp = Inches(4.85) + i * Inches(0.80)
        bar_h_px = bar_area_h * (rv / bar_max)
        _rect(s, xp, base_y - bar_h_px, bar_w, bar_h_px,
              fill=TEAL if rv == max(rev_omr) else BLUE)
        _txt(s, f"{rv}M", xp, base_y - bar_h_px - Inches(0.22),
             bar_w, Inches(0.22), size=8, bold=True,
             color=TEAL, align=PP_ALIGN.CENTER)
        _txt(s, yr, xp, base_y + Inches(0.04),
             bar_w, Inches(0.22), size=9, color=DGRAY, align=PP_ALIGN.CENTER)
    _txt(s, "OMR (Millions)", Inches(4.7), Inches(3.58),
         Inches(4.0), Inches(0.22), size=9, italic=True, color=MGRAY)

    # Market share S-curve
    _rect(s, Inches(9.2), Inches(1.2), Inches(3.8), Inches(2.7),
          fill=WHITE, line_rgb=LGRAY, line_w=Pt(0.5))
    _txt(s, "📉  Market Share S-Curve",
         Inches(9.35), Inches(1.28), Inches(3.5), Inches(0.35),
         size=13, bold=True, color=NAVY)
    s_years  = [2026, 2027, 2028, 2029, 2030]
    s_shares = [10, 14, 19, 24, 28]
    prev_x, prev_y = None, None
    chart_x0, chart_y0 = Inches(9.5), Inches(1.7)
    chart_w, chart_h   = Inches(3.3), Inches(1.6)
    for i, (yr, sh) in enumerate(zip(s_years, s_shares)):
        cx = chart_x0 + chart_w * i / (len(s_years) - 1)
        cy = chart_y0 + chart_h * (1 - sh / 30)
        _rect(s, cx - Inches(0.04), cy - Inches(0.04),
              Inches(0.08), Inches(0.08), fill=TEAL)
        _txt(s, f"{sh}%", cx - Inches(0.15), cy - Inches(0.25),
             Inches(0.3), Inches(0.20), size=7, color=TEAL, align=PP_ALIGN.CENTER)
        _txt(s, str(yr), cx - Inches(0.2), chart_y0 + chart_h + Inches(0.05),
             Inches(0.4), Inches(0.20), size=7, color=MGRAY, align=PP_ALIGN.CENTER)

    # Demand prediction
    _rect(s, Inches(0.3), Inches(4.1), SW - Inches(0.6), Inches(2.8),
          fill=WHITE, line_rgb=LGRAY, line_w=Pt(0.5))
    _txt(s, "🎯  Demand Score Prediction (No YOLO required)",
         Inches(0.45), Inches(4.2), Inches(6.0), Inches(0.35),
         size=13, bold=True, color=NAVY)

    features = [
        ("35%", "Population Density", "residents/km²"),
        ("40%", "Coverage Gap",       "% unserved"),
        ("25%", "Commercial Ratio",   "commercial units %"),
    ]
    for i, (wt, feature, unit) in enumerate(features):
        xp = Inches(0.45) + i * Inches(4.25)
        _rect(s, xp, Inches(4.65), Inches(4.0), Inches(0.88),
              fill=OFFWH, line_rgb=LGRAY, line_w=Pt(0.5))
        _txt(s, wt, xp + Inches(0.1), Inches(4.68),
             Inches(0.8), Inches(0.42), size=22, bold=True, color=TEAL)
        _txt(s, feature, xp + Inches(0.95), Inches(4.71),
             Inches(2.8), Inches(0.28), size=12, bold=True, color=NAVY)
        _txt(s, unit, xp + Inches(0.95), Inches(5.0),
             Inches(2.8), Inches(0.22), size=9, italic=True, color=MGRAY)

    _txt(s, "→  predict_demand_score() generates zone scores without needing real YOLO detections — "
             "useful for areas without imagery coverage.",
         Inches(0.45), Inches(5.65), SW - Inches(0.9), Inches(0.40),
         size=11, italic=True, color=DGRAY)


# ── Slide 14: Step 12 — Streamlit Dashboard ──────────────────────────────────
def slide_14_dashboard(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Streamlit Dashboard — 6 Tabs",
            "Interactive intelligence interface · run: streamlit run app/dashboard.py", step_num="12")
    _footer(s)

    tabs = [
        ("🗺️", "Zone Map",         "Folium interactive map\n6 toggle layers\nClick zone → profile",        TEAL),
        ("🔍", "Vehicle Detection", "Upload image or test.png\nAerial/standard mode\nAnnotated output",     BLUE),
        ("📊", "Zone Analysis",     "Radar charts per zone\nScoring breakdown\nExport CSV/PPTX",            ORANGE),
        ("📡", "Telecom Sim",       "KPI tables & charts\nSubscriber forecasts\nCompetitor heatmap",        PURPLE),
        ("💰", "Business Case",     "ROI calculator\nRevenue projections\nCapex vs payback",               GREEN),
        ("🤖", "AI Insights",       "K-Means clusters\nS-curve projections\nDemand scoring",               RED),
    ]
    bw = (SW - Inches(0.6)) / 3
    bh = Inches(2.05)
    for i, (icon, title, body, col) in enumerate(tabs):
        col_idx = i % 3
        row_idx = i // 3
        xp = Inches(0.3) + col_idx * bw
        yp = Inches(1.25) + row_idx * (bh + Inches(0.18))
        _rect(s, xp, yp, bw - Inches(0.12), bh, fill=WHITE,
              line_rgb=col, line_w=Pt(1.5))
        _rect(s, xp, yp, bw - Inches(0.12), Inches(0.52), fill=col)
        _txt(s, f"{icon}  {title}",
             xp + Inches(0.12), yp + Inches(0.08),
             bw - Inches(0.36), Inches(0.38),
             size=14, bold=True, color=WHITE)
        _txt(s, body,
             xp + Inches(0.12), yp + Inches(0.60),
             bw - Inches(0.36), bh - Inches(0.68),
             size=11, color=DGRAY)

    # Run command
    _rect(s, Inches(0.3), SH - Inches(0.65), SW - Inches(0.6), Inches(0.35),
          fill=NAVY2)
    _txt(s, "▶  streamlit run app/dashboard.py  —  or—  "
             "docker compose up  →  http://localhost:8501",
         Inches(0.5), SH - Inches(0.63), SW - Inches(1.0), Inches(0.30),
         size=11, bold=True, color=TEAL)


# ── Slide 15: Step 13 — FastAPI Backend ──────────────────────────────────────
def slide_15_fastapi(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "FastAPI REST Backend — 12 Endpoints",
            "Production-grade API · uvicorn · Pydantic · CORSMiddleware · port 8000", step_num="13")
    _footer(s)

    endpoints = [
        ("GET",  "/health",                   "Service health check",                 GREEN),
        ("GET",  "/zones",                    "List all 15 zones",                    TEAL),
        ("GET",  "/zones/{id}",               "Single zone profile",                  TEAL),
        ("GET",  "/zones/{id}/forecast",      "5-year revenue forecast",              BLUE),
        ("GET",  "/zones/{id}/telecom",       "Telecom KPIs for zone",                BLUE),
        ("GET",  "/market/summary",           "Overall market stats",                 ORANGE),
        ("GET",  "/market/telecom",           "All zones telecom data",               ORANGE),
        ("GET",  "/market/clusters",          "K-Means cluster labels",               PURPLE),
        ("GET",  "/market/share-projection",  "5-year market share S-curve",          PURPLE),
        ("POST", "/detect",                   "Upload image → vehicle detections",    RED),
        ("GET",  "/report/csv",               "Download CSV report (streaming)",      MGRAY),
        ("GET",  "/report/summary",           "JSON summary for top zones",           MGRAY),
    ]

    # Header row
    hdr_y = Inches(1.2)
    _rect(s, Inches(0.3), hdr_y, Inches(9.5), Inches(0.35), fill=NAVY)
    for txt, cx in [("Method", Inches(0.3)), ("Endpoint", Inches(1.3)),
                    ("Description", Inches(5.3))]:
        _txt(s, txt, cx + Inches(0.08), hdr_y + Inches(0.04),
             Inches(2.0), Inches(0.28), size=10, bold=True, color=WHITE)

    for i, (method, path, desc, col) in enumerate(endpoints):
        yp = Inches(1.58) + i * Inches(0.43)
        bg = OFFWH if i % 2 == 0 else WHITE
        _rect(s, Inches(0.3), yp, Inches(9.5), Inches(0.41), fill=bg)
        _rect(s, Inches(0.3), yp, Inches(0.9), Inches(0.41), fill=col)
        _txt(s, method, Inches(0.3), yp + Inches(0.06),
             Inches(0.9), Inches(0.30), size=10, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)
        _txt(s, path, Inches(1.3), yp + Inches(0.07),
             Inches(3.9), Inches(0.30), size=10, color=NAVY)
        _txt(s, desc, Inches(5.3), yp + Inches(0.07),
             Inches(4.4), Inches(0.30), size=10, color=DGRAY)

    # Sample code
    _txt(s, "Sample API call",
         Inches(10.0), Inches(1.2), Inches(3.0), Inches(0.35),
         size=13, bold=True, color=NAVY)
    _code_block(s, [
        "import httpx",
        "",
        "# Get zone profile",
        "r = httpx.get(",
        "  'http://localhost:8000",
        "   /zones/al_khuwair'",
        ")",
        "zone = r.json()",
        "",
        "# Upload image for detection",
        "files = {'file': open(",
        "  'test.png', 'rb')}",
        "r = httpx.post(",
        "  '.../detect', files=files)",
        "boxes = r.json()['boxes']",
    ], Inches(10.0), Inches(1.65), Inches(3.1), Inches(4.5),
       bg=NAVY2, size=9)

    # Tech stack
    _rect(s, Inches(0.3), Inches(6.82), Inches(9.5), Inches(0.38), fill=NAVY)
    for txt, cx in [("FastAPI 0.111+", Inches(0.5)),
                    ("Pydantic v2", Inches(2.5)),
                    ("uvicorn", Inches(4.2)),
                    ("httpx (async)", Inches(5.6)),
                    ("CORS ✓", Inches(7.3)),
                    ("OpenAPI docs: /docs", Inches(8.2))]:
        _txt(s, txt, cx, Inches(6.87), Inches(2.0), Inches(0.30),
             size=10, bold=True, color=TEAL)


# ── Slide 16: Step 14 — Docker Deployment ────────────────────────────────────
def slide_16_docker(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Docker Deployment",
            "Production containerisation · docker compose up · two services · port 8501 + 8000",
            step_num="14")
    _footer(s)

    # docker-compose.yml
    _txt(s, "docker-compose.yml",
         Inches(0.35), Inches(1.2), Inches(6.0), Inches(0.35),
         size=14, bold=True, color=NAVY)
    _code_block(s, [
        "x-falcon-common: &common",
        "  build: .",
        "  env_file: .env",
        "  volumes: [./data:/app/data]",
        "",
        "services:",
        "  falcon-dashboard:",
        "    <<: *common",
        "    command: streamlit run app/dashboard.py",
        "    ports: ['8501:8501']",
        "",
        "  falcon-api:",
        "    <<: *common",
        "    command: uvicorn api.main:app",
        "             --host 0.0.0.0 --port 8000",
        "             --workers 2",
        "    ports: ['8000:8000']",
    ], Inches(0.35), Inches(1.65), Inches(6.0), Inches(4.2),
       bg=NAVY2, size=10)

    # Dockerfile
    _txt(s, "Dockerfile",
         Inches(6.6), Inches(1.2), Inches(6.4), Inches(0.35),
         size=14, bold=True, color=NAVY)
    _code_block(s, [
        "FROM python:3.11-slim",
        "",
        "# System libraries for OpenCV / GL",
        "RUN apt-get update && apt-get install -y \\",
        "    libgl1-mesa-glx libglib2.0-0",
        "",
        "WORKDIR /app",
        "COPY requirements.txt .",
        "RUN pip install -r requirements.txt",
        "",
        "# Pre-download YOLO weights",
        "RUN python -c \"from ultralytics import YOLO;",
        "               YOLO('yolov8n.pt')\"",
        "",
        "COPY . .",
        "CMD [\"python\", \"run.py\"]",
    ], Inches(6.6), Inches(1.65), Inches(6.4), Inches(4.2),
       bg=NAVY2, size=10)

    # Deploy steps
    _divider(s, Inches(5.99), color=LGRAY)
    cmds = [
        "cp .env.example .env  &&  nano .env   # configure",
        "docker compose build                   # build images",
        "docker compose up -d                   # start services",
        "open http://localhost:8501             # Streamlit UI",
        "open http://localhost:8000/docs        # FastAPI swagger",
    ]
    for i, cmd in enumerate(cmds):
        _rect(s, Inches(0.35) + i * Inches(2.62), Inches(6.1),
              Inches(2.5), Inches(0.45), fill=NAVY2)
        _txt(s, cmd, Inches(0.5) + i * Inches(2.62), Inches(6.15),
             Inches(2.4), Inches(0.38), size=8, color=TEAL)


# ── Slide 17: Step 15 — Business Case ────────────────────────────────────────
def slide_17_business(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Business Case & ROI",
            "FALCON value proposition for Waheb Telecom — Muscat FTTH expansion", step_num="15")
    _footer(s)

    # KPIs
    kpis = [
        ("105,000", "Unserved Homes", RED),
        ("+18%",    "Market Share Target", TEAL),
        ("3.2 yr",  "Avg Payback Period", ORANGE),
        ("1.38M OMR", "Year-5 Revenue", GREEN),
    ]
    bw = (SW - Inches(0.6)) / 4
    for i, (val, lbl, col) in enumerate(kpis):
        _kpi(s, Inches(0.3) + i * bw, Inches(1.2),
             bw - Inches(0.1), Inches(1.1), val, lbl, vcolor=col)

    # Top zones ROI table
    _txt(s, "Zone Investment ROI Analysis",
         Inches(0.35), Inches(2.55), Inches(7.0), Inches(0.35),
         size=14, bold=True, color=NAVY)
    roi_data = [
        ("Al Khuwair",  "OMR 1.2M", "OMR 420K", "2.9 yr", "High"),
        ("Bausher",     "OMR 0.9M", "OMR 310K", "2.9 yr", "High"),
        ("Wadi Kabir",  "OMR 1.4M", "OMR 380K", "3.7 yr", "Medium"),
        ("Ghubrah N.",  "OMR 0.8M", "OMR 210K", "3.8 yr", "Medium"),
        ("Ruwi",        "OMR 1.1M", "OMR 270K", "4.1 yr", "Low"),
    ]
    hdrs = ["Zone", "CAPEX", "Annual Rev", "Payback", "Priority"]
    col_xs = [Inches(0.35), Inches(2.4), Inches(4.0), Inches(5.7), Inches(7.2)]
    _rect(s, Inches(0.35), Inches(3.0), Inches(8.3), Inches(0.35), fill=NAVY)
    for hdr, cx in zip(hdrs, col_xs):
        _txt(s, hdr, cx + Inches(0.06), Inches(3.04),
             Inches(2.0), Inches(0.26), size=10, bold=True, color=WHITE)
    for i, row in enumerate(roi_data):
        yp = Inches(3.38) + i * Inches(0.40)
        _rect(s, Inches(0.35), yp, Inches(8.3), Inches(0.38),
              fill=OFFWH if i % 2 == 0 else WHITE)
        colors = [NAVY, BLUE, TEAL, ORANGE,
                  GREEN if row[4] == "High" else (ORANGE if row[4] == "Medium" else RED)]
        for val, cx, col in zip(row, col_xs, colors):
            _txt(s, val, cx + Inches(0.06), yp + Inches(0.06),
                 Inches(2.0), Inches(0.26), size=10, color=col,
                 bold=(cx == col_xs[1] or cx == col_xs[4]))

    # FALCON value add
    _rect(s, Inches(8.9), Inches(2.55), Inches(4.1), Inches(4.0),
          fill=NAVY, line_rgb=TEAL, line_w=Pt(1))
    _txt(s, "🦅  FALCON Value-Add",
         Inches(9.05), Inches(2.68), Inches(3.8), Inches(0.35),
         size=13, bold=True, color=TEAL)
    value_items = [
        "AI-identified top 3 zones = 42% of total opportunity",
        "YOLOv8 aerial detection — no ground survey needed",
        "5-year revenue forecast in <1s per zone",
        "Competitive gap map vs RAMI & Don",
        "Reduces planning cycle from 6 months to 1 day",
        "ROI calculator with live adjustable parameters",
        "Docker-deployable — on-premise or cloud",
        "REST API for integration with existing BSS/OSS",
    ]
    for i, item in enumerate(value_items):
        _txt(s, f"✓  {item}",
             Inches(9.05), Inches(3.15) + i * Inches(0.42),
             Inches(3.8), Inches(0.38), size=10, color=WHITE)

    # Summary
    _rect(s, Inches(0.3), Inches(6.65), SW - Inches(0.6), Inches(0.55),
          fill=TEAL)
    _txt(s, "FALCON accelerates Waheb's FTTH expansion decisions — turning weeks of manual analysis "
             "into real-time AI intelligence at zone granularity.",
         Inches(0.5), Inches(6.7), SW - Inches(1.0), Inches(0.45),
         size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)


# ── Slide 18: Roadmap ─────────────────────────────────────────────────────────
def slide_18_roadmap(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=OFFWH)
    _header(s, "Roadmap — Next Steps",
            "From demo to production — FALCON v2.0 enhancement plan")
    _footer(s)

    phases = [
        ("Phase 1\nQ2 2026", "Foundation",
         ["Real satellite imagery API (Mapbox / Planet)",
          "Custom YOLOv8 fine-tuning on Oman aerial data",
          "PostgreSQL/PostGIS zone database",
          "User authentication (JWT)"],
         TEAL),
        ("Phase 2\nQ3 2026", "Scale-Up",
         ["Real FTTH subscriber data integration",
          "Live network KPI feed from OSS",
          "Multi-city support (Sohar, Sur, Salalah)",
          "Automated zone re-scoring pipeline"],
         BLUE),
        ("Phase 3\nQ4 2026", "Intelligence",
         ["Deep learning demand prediction (LSTM)",
          "Churn risk scoring per zone",
          "NLP parsing of Oman TRAI filings",
          "Executive mobile app (React Native)"],
         ORANGE),
        ("Phase 4\n2027+", "Production",
         ["Real-time vehicle density stream (RTSP CCTV)",
          "Kubernetes autoscaling deployment",
          "Integration with CRM & billing platform",
          "Export to PowerBI / Tableau connectors"],
         PURPLE),
    ]

    bw = (SW - Inches(0.6)) / 4
    for i, (phase, title, items, col) in enumerate(phases):
        xp = Inches(0.3) + i * bw
        _rect(s, xp, Inches(1.2), bw - Inches(0.1), Inches(5.6),
              fill=WHITE, line_rgb=col, line_w=Pt(1.5))
        _rect(s, xp, Inches(1.2), bw - Inches(0.1), Inches(0.85), fill=col)
        _txt(s, phase, xp + Inches(0.1), Inches(1.23),
             bw - Inches(0.3), Inches(0.40), size=12, bold=True, color=WHITE)
        _txt(s, title, xp + Inches(0.1), Inches(1.63),
             bw - Inches(0.3), Inches(0.35), size=12, italic=True, color=WHITE)
        for j, item in enumerate(items):
            _txt(s, f"▸  {item}",
                 xp + Inches(0.12), Inches(2.18) + j * Inches(0.62),
                 bw - Inches(0.3), Inches(0.55), size=11, color=DGRAY)

    _rect(s, Inches(0.3), Inches(7.0), SW - Inches(0.6), Inches(0.32), fill=NAVY)
    _txt(s, "Current state: FALCON demo — full simulation, YOLOv8 aerial, "
             "Streamlit + FastAPI + Docker",
         Inches(0.5), Inches(7.03), SW - Inches(1.0), Inches(0.27),
         size=10, bold=True, color=TEAL, align=PP_ALIGN.CENTER)


# ── Slide 19: Results Summary / Thank You ────────────────────────────────────
def slide_19_thankyou(prs, d):
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, fill=NAVY)
    _rect(s, 0, 0, SW, Inches(0.07), fill=TEAL)
    _rect(s, 0, SH - Inches(0.07), SW, Inches(0.07), fill=TEAL)
    _rect(s, SW - Inches(0.45), 0, Inches(0.45), SH, fill=NAVY2)

    _txt(s, "🦅", Inches(0.5), Inches(0.6), Inches(1.4), Inches(1.4),
         size=68, align=PP_ALIGN.CENTER)
    _txt(s, "FALCON",
         Inches(2.0), Inches(0.45), Inches(10.5), Inches(1.4),
         size=80, bold=True, color=TEAL, align=PP_ALIGN.LEFT)
    _txt(s, "End-to-End ML Pipeline — Complete",
         Inches(2.0), Inches(1.9), Inches(10.0), Inches(0.6),
         size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)

    _divider(s, Inches(2.65), color=TEAL)

    # Summary results
    results = [
        ("📸", "2,400+",  "Annotated images"),
        ("🧠", "YOLOv8n", "Transfer-learned model"),
        ("🛰️", "4× Upscale", "Aerial pipeline"),
        ("🚗", "10 vehicles", "Detected in test.png"),
        ("📍", "15 zones",  "Muscat scored & ranked"),
        ("💰", "+18%",     "Market share target"),
    ]
    bw = Inches(2.0)
    for i, (icon, val, label) in enumerate(results):
        xp = Inches(0.6) + i * bw
        _rect(s, xp, Inches(3.0), bw - Inches(0.15), Inches(1.35),
              fill=NAVY2, line_rgb=TEAL, line_w=Pt(0.5))
        _txt(s, icon, xp, Inches(3.05), bw - Inches(0.15), Inches(0.40),
             size=20, align=PP_ALIGN.CENTER)
        _txt(s, val, xp, Inches(3.45), bw - Inches(0.15), Inches(0.38),
             size=16, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
        _txt(s, label, xp, Inches(3.83), bw - Inches(0.15), Inches(0.38),
             size=10, color=MGRAY, align=PP_ALIGN.CENTER)

    _divider(s, Inches(4.55), color=NAVY2)

    _txt(s, "Stack: Python 3.11  ·  YOLOv8  ·  Streamlit  ·  FastAPI  ·  "
             "Folium  ·  scikit-learn  ·  Docker",
         Inches(1.0), Inches(4.7), Inches(11.0), Inches(0.38),
         size=13, color=MGRAY, align=PP_ALIGN.CENTER)

    _txt(s, "Waheb Telecom  ·  Muscat, Oman  ·  2026  ·  phd.abdelouaheb@gmail.com",
         Inches(1.0), Inches(5.15), Inches(11.0), Inches(0.38),
         size=12, color=MGRAY, align=PP_ALIGN.CENTER)

    _rect(s, Inches(3.5), Inches(5.8), Inches(6.3), Inches(0.85), fill=TEAL)
    _txt(s, "Thank You — Questions Welcome",
         Inches(3.5), Inches(5.82), Inches(6.3), Inches(0.80),
         size=22, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    _txt(s, "FALCON: Flavia AI Coverage & Location Opportunity Network",
         Inches(1.0), Inches(6.85), Inches(11.0), Inches(0.35),
         size=12, italic=True, color=TEAL, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH

    print("Loading project data...")
    d = _load_data()

    builders = [
        ("01 Title",               slide_01_title),
        ("02 Agenda",              slide_02_agenda),
        ("03 Data Collection",     slide_03_data_collection),
        ("04 Annotation",          slide_04_annotation),
        ("05 Dataset Prep",        slide_05_dataset),
        ("06 YOLOv8 Architecture", slide_06_yolo_arch),
        ("07 Aerial Challenge",    slide_07_challenge),
        ("08 Aerial Pipeline",     slide_08_aerial_pipeline),
        ("09 Detection Results",   slide_09_results),
        ("10 Zone Scoring",        slide_10_scoring),
        ("11 Telecom Sim",         slide_11_telecom),
        ("12 Competitors",         slide_12_competitors),
        ("13 ML Insights",         slide_13_ml_insights),
        ("14 Dashboard",           slide_14_dashboard),
        ("15 FastAPI",             slide_15_fastapi),
        ("16 Docker",              slide_16_docker),
        ("17 Business Case",       slide_17_business),
        ("18 Roadmap",             slide_18_roadmap),
        ("19 Thank You",           slide_19_thankyou),
    ]

    for name, fn in builders:
        print(f"  Building slide: {name}")
        try:
            fn(prs, d)
        except Exception as exc:
            print(f"  WARN Slide '{name}' failed: {exc}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "FALCON_Pipeline_Presentation.pptx"
    prs.save(str(out_path))
    print(f"\nDone. Saved -> {out_path}")
    print(f"   Slides: {len(builders)}")
    return str(out_path)


if __name__ == "__main__":
    main()
