"""
Ooredoo Falcon — 18-slide PowerPoint presentation generator.
Run standalone: python scripts/generate_pptx.py
Output: outputs/ooredoo_falcon_presentation.pptx
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Colours ────────────────────────────────────────────────────────────────────
OO_RED       = "E10A0A"   # Ooredoo red
OO_NAVY      = "1A1A2E"   # Dark navy
OO_WHITE     = "FFFFFF"
OO_LIGHT     = "CCCCCC"
OO_MID_NAVY  = "16213E"

# ── Output path ────────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUTS_DIR / "ooredoo_falcon_presentation.pptx"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rgb(hex_str: str) -> RGBColor:
    return RGBColor.from_string(hex_str.lstrip("#"))


def set_slide_background(slide, hex_color: str = OO_NAVY) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = _rgb(hex_color)


def add_text_box(
    slide,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    font_size: int = 18,
    bold: bool = False,
    color: str = OO_WHITE,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    italic: bool = False,
) -> None:
    """Add a text box at the given inch coordinates."""
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = _rgb(color)


def add_multiline_text_box(
    slide,
    lines: list[str],
    left: float,
    top: float,
    width: float,
    height: float,
    font_size: int = 16,
    bold: bool = False,
    color: str = OO_LIGHT,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    line_spacing_pt: float | None = None,
) -> None:
    """Add a text box where each item in `lines` becomes its own paragraph."""
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.color.rgb = _rgb(color)
        if line_spacing_pt:
            from pptx.util import Pt as _Pt
            p.space_after = _Pt(line_spacing_pt)


def add_red_accent_line(slide, prs) -> None:
    """Add a thin red bar at the top of the slide."""
    accent = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Emu(0),
        Emu(0),
        prs.slide_width,
        Pt(7),
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = _rgb(OO_RED)
    accent.line.fill.background()


def add_navy_box(slide, left, top, width, height, color=OO_MID_NAVY) -> None:
    """Add a filled navy rectangle (decorative block)."""
    box = slide.shapes.add_shape(
        1,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.fill.solid()
    box.fill.fore_color.rgb = _rgb(color)
    box.line.fill.background()


def add_two_column_text(
    slide,
    left_title: str,
    right_title: str,
    left_lines: list[str],
    right_lines: list[str],
    top: float = 1.6,
    font_size: int = 15,
) -> None:
    """Render two side-by-side blocks of bullet text."""
    # Left column header
    add_navy_box(slide, 0.4, top, 4.5, 0.4, OO_RED)
    add_text_box(slide, left_title, 0.5, top + 0.02, 4.3, 0.4,
                 font_size=14, bold=True, color=OO_WHITE)
    add_multiline_text_box(slide, left_lines, 0.4, top + 0.45, 4.5,
                           3.5, font_size=font_size, color=OO_LIGHT)

    # Right column header
    add_navy_box(slide, 5.2, top, 4.5, 0.4, OO_RED)
    add_text_box(slide, right_title, 5.3, top + 0.02, 4.3, 0.4,
                 font_size=14, bold=True, color=OO_WHITE)
    add_multiline_text_box(slide, right_lines, 5.2, top + 0.45, 4.5,
                           3.5, font_size=font_size, color=OO_LIGHT)


# ── Slide builders ─────────────────────────────────────────────────────────────

def slide_01_title(prs: Presentation) -> None:
    """Slide 1 — Title slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)

    # Large red eagle emoji block (decorative)
    add_navy_box(slide, 3.8, 1.0, 2.4, 2.4, "2C0A0A")
    add_text_box(slide, "🦅", 4.15, 1.1, 1.7, 2.0,
                 font_size=72, bold=False, color=OO_WHITE, align=PP_ALIGN.CENTER)

    add_text_box(slide, "Ooredoo Falcon", 1.0, 3.6, 8.0, 1.2,
                 font_size=48, bold=True, color=OO_WHITE, align=PP_ALIGN.CENTER)

    add_text_box(
        slide,
        "AI-Powered FTTH Market Intelligence & Expansion Platform",
        1.0, 4.7, 8.0, 0.8,
        font_size=20, bold=False, color=OO_LIGHT, align=PP_ALIGN.CENTER,
    )

    # Red divider line
    add_navy_box(slide, 3.5, 5.55, 3.0, 0.05, OO_RED)

    add_text_box(
        slide,
        "Ooredoo Oman  ·  Muscat  ·  2026  |  CONFIDENTIAL",
        1.0, 5.7, 8.0, 0.5,
        font_size=13, bold=False, color="AAAAAA", align=PP_ALIGN.CENTER,
    )
    print("  Slide 1 — Title")


def slide_02_challenge(prs: Presentation) -> None:
    """Slide 2 — Business Challenge."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Business Challenge", 0.5, 0.2, 9.0, 0.8,
                 font_size=34, bold=True, color=OO_WHITE)

    bullets = [
        "◆  Rapid urbanization in Muscat creates surging demand for high-speed broadband",
        "◆  Manual FTTH planning is slow, costly, and error-prone — weeks per zone",
        "◆  Competitive pressure from Omantel (40%) and Awasr (30%) erodes Ooredoo market share",
        "◆  Coverage gap analysis relies on outdated surveys with no real-time intelligence",
        "◆  Revenue opportunities are missed due to lack of data-driven zone prioritization",
    ]
    add_multiline_text_box(slide, bullets, 0.5, 1.2, 9.2, 5.0,
                           font_size=18, color=OO_LIGHT, line_spacing_pt=8)
    print("  Slide 2 — Business Challenge")


def slide_03_why_ftth(prs: Presentation) -> None:
    """Slide 3 — Why FTTH Matters."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Why FTTH Matters for Ooredoo Oman", 0.5, 0.2, 9.2, 0.8,
                 font_size=30, bold=True, color=OO_WHITE)

    bullets = [
        "◆  FTTH delivers 100x faster speeds vs. legacy DSL — unmatched customer experience",
        "◆  Oman Vision 2040 mandates nationwide high-speed connectivity for smart cities",
        "◆  FTTH customers generate 3x higher ARPU than mobile-only subscribers",
        "◆  Fiber infrastructure lasts 20+ years — lowest total cost of ownership",
        "◆  350,000+ homes in Muscat represent a massive untapped revenue opportunity",
        "◆  First-mover advantage in underserved zones creates durable competitive moats",
    ]
    add_multiline_text_box(slide, bullets, 0.5, 1.2, 9.2, 5.0,
                           font_size=18, color=OO_LIGHT, line_spacing_pt=8)
    print("  Slide 3 — Why FTTH Matters")


def slide_04_current_process(prs: Presentation) -> None:
    """Slide 4 — Current Manual Process vs Falcon."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Current Manual Planning Process", 0.5, 0.2, 9.2, 0.8,
                 font_size=30, bold=True, color=OO_WHITE)

    old_way = [
        "✗  Field surveys (2-3 weeks per zone)",
        "✗  Manual data entry and spreadsheets",
        "✗  No competitive intelligence",
        "✗  Intuition-based decisions",
        "✗  Slow ROI analysis",
        "✗  No satellite or aerial insights",
    ]
    falcon_way = [
        "✓  Satellite + AI analysis (hours, not weeks)",
        "✓  Automated scoring dashboard",
        "✓  Real-time competitor mapping",
        "✓  Data-driven zone prioritization",
        "✓  Instant ROI and investment calc",
        "✓  YOLOv8 computer vision detection",
    ]
    add_two_column_text(
        slide,
        "Old Way — Manual Planning",
        "Falcon Way — AI Intelligence",
        old_way, falcon_way,
        top=1.3, font_size=15,
    )
    print("  Slide 4 — Current Process")


def slide_05_solution(prs: Presentation) -> None:
    """Slide 5 — Ooredoo Falcon Solution."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Introducing Ooredoo Falcon", 0.5, 0.2, 9.2, 0.8,
                 font_size=34, bold=True, color=OO_WHITE)

    add_text_box(
        slide,
        "An end-to-end AI platform combining satellite intelligence, computer vision, "
        "and large language models to transform Ooredoo FTTH expansion decisions.",
        0.5, 1.1, 9.2, 0.8,
        font_size=16, color=OO_LIGHT,
    )

    capabilities = [
        ("🔬  Computer Vision",
         "YOLOv8 detects buildings, vehicles & infrastructure from satellite/aerial images"),
        ("📊  Market Intelligence",
         "AI-scored zone rankings with investment, ROI, and competitive gap analysis"),
        ("🤖  LLM Business Analysis",
         "AI-generated executive briefs, marketing strategies & deployment recommendations"),
        ("📋  Automated Reporting",
         "One-click PowerPoint presentations, CSV exports & executive summaries"),
    ]

    y = 2.1
    for icon_title, desc in capabilities:
        add_navy_box(slide, 0.4, y, 9.4, 0.55, OO_MID_NAVY)
        add_text_box(slide, icon_title, 0.6, y + 0.04, 3.5, 0.5,
                     font_size=15, bold=True, color=OO_WHITE)
        add_text_box(slide, desc, 4.1, y + 0.04, 5.7, 0.5,
                     font_size=14, color=OO_LIGHT)
        y += 0.75
    print("  Slide 5 — Solution")


def slide_06_architecture(prs: Presentation) -> None:
    """Slide 6 — 8-Step Architecture."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Platform Architecture — 8-Step Intelligence Pipeline",
                 0.4, 0.2, 9.5, 0.8, font_size=26, bold=True, color=OO_WHITE)

    steps = [
        ("Step 1", "Interactive Map",          "Define target area on Muscat city map"),
        ("Step 2", "Zone Area Selection",       "Select specific zones for deep analysis"),
        ("Step 3", "Satellite Image Capture",   "Aerial images via satellite tile provider"),
        ("Step 4", "Computer Vision (YOLOv8)", "Detect buildings, vehicles & land use"),
        ("Step 5", "Market Potential Engine",   "Buildings × households × share = subscribers"),
        ("Step 6", "LLM Business Analysis",     "AI executive insights & risk assessment"),
        ("Step 7", "Marketing Strategy Gen.",   "AI go-to-market plan per zone"),
        ("Step 8", "Executive Report & Export", "PowerPoint + CSV + executive summary"),
    ]

    colors = [OO_RED, "B00000", OO_RED, "B00000", OO_RED, "B00000", OO_RED, "B00000"]
    y = 1.15
    for i, (num, title, desc) in enumerate(steps):
        add_navy_box(slide, 0.4, y, 1.0, 0.44, colors[i])
        add_text_box(slide, num, 0.45, y + 0.04, 0.9, 0.38,
                     font_size=11, bold=True, color=OO_WHITE, align=PP_ALIGN.CENTER)
        add_text_box(slide, title, 1.55, y + 0.02, 3.5, 0.4,
                     font_size=14, bold=True, color=OO_WHITE)
        add_text_box(slide, desc, 5.1, y + 0.04, 4.7, 0.38,
                     font_size=13, color=OO_LIGHT)
        y += 0.57
    print("  Slide 6 — Architecture")


def slide_07_image_collection(prs: Presentation) -> None:
    """Slide 7 — Satellite Image Collection."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Step 1: Satellite Image Collection", 0.5, 0.2, 9.2, 0.8,
                 font_size=30, bold=True, color=OO_WHITE)

    add_text_box(
        slide,
        "Aerial Intelligence — Capturing Ground Truth from Space",
        0.5, 1.1, 9.2, 0.6,
        font_size=18, bold=True, color=OO_RED,
    )

    bullets = [
        "◆  Satellite tile imagery acquired for each target zone across Muscat Governorate",
        "◆  Multiple zoom levels (16–18) to capture both building-level and street-level detail",
        "◆  OpenStreetMap tiles provide baseline reference; real satellite data via provider API",
        "◆  Image pre-processing: contrast enhancement, cloud masking, tile stitching",
        "◆  4x upscaling applied for aerial screenshots to improve small-object detection",
        "◆  Typical image size: 1024×1024 px covering 250m × 250m real-world area",
    ]
    add_multiline_text_box(slide, bullets, 0.5, 1.8, 9.2, 4.5,
                           font_size=17, color=OO_LIGHT, line_spacing_pt=8)
    print("  Slide 7 — Image Collection")


def slide_08_yolo_detection(prs: Presentation) -> None:
    """Slide 8 — YOLOv8 AI Detection."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Step 2: YOLOv8 AI Object Detection", 0.5, 0.2, 9.2, 0.8,
                 font_size=30, bold=True, color=OO_WHITE)

    add_text_box(
        slide,
        "What the model detects from aerial imagery:",
        0.5, 1.15, 9.2, 0.5,
        font_size=17, bold=True, color=OO_LIGHT,
    )

    detections = [
        ("🏠  Buildings",  "Rooftops, residential blocks — FTTH candidate units"),
        ("🚗  Vehicles",   "Cars, trucks — traffic density & economic activity signal"),
        ("🛣️  Roads",      "Street network topology — fiber route planning backbone"),
        ("🌳  Land Use",   "Vegetation, open land — distinguishes deployable from non-deployable areas"),
    ]
    y = 1.8
    for icon_label, desc in detections:
        add_navy_box(slide, 0.4, y, 9.3, 0.6, OO_MID_NAVY)
        add_text_box(slide, icon_label, 0.6, y + 0.08, 2.8, 0.48,
                     font_size=15, bold=True, color=OO_WHITE)
        add_text_box(slide, desc, 3.5, y + 0.1, 6.1, 0.42,
                     font_size=14, color=OO_LIGHT)
        y += 0.75

    add_text_box(
        slide,
        "Model: YOLOv8n (custom fine-tuned on FTTH aerial dataset)  |  "
        "Confidence threshold: 0.20–0.35  |  Detection latency: < 2 sec/image",
        0.5, 5.5, 9.2, 0.7,
        font_size=13, color="888888",
    )
    print("  Slide 8 — YOLO Detection")


def slide_09_building_classification(prs: Presentation) -> None:
    """Slide 9 — Building Classification."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Step 3: Building Classification & Density Scoring",
                 0.5, 0.2, 9.2, 0.8, font_size=26, bold=True, color=OO_WHITE)

    types = [
        ("Residential Villas",    "Single-family dwellings — 3–5 households/unit — standard FTTH drop"),
        ("Apartment Blocks",      "Multi-dwelling units — 8–24 households/unit — highest FTTH ROI"),
        ("Commercial Buildings",  "SME offices & retail — premium ARPU, SLA-grade FTTH required"),
        ("Mixed-use Complexes",   "Retail + residential — bundled FTTH + business service opportunity"),
    ]

    y = 1.3
    for i, (btype, desc) in enumerate(types):
        clr = OO_RED if i % 2 == 0 else OO_MID_NAVY
        add_navy_box(slide, 0.4, y, 9.3, 0.65, clr)
        add_text_box(slide, btype, 0.6, y + 0.1, 3.2, 0.5,
                     font_size=15, bold=True, color=OO_WHITE)
        add_text_box(slide, desc, 3.8, y + 0.12, 5.8, 0.45,
                     font_size=14, color=OO_LIGHT)
        y += 0.82

    add_text_box(
        slide,
        "Density Scoring Formula:",
        0.5, 5.1, 9.2, 0.4,
        font_size=15, bold=True, color=OO_WHITE,
    )
    add_text_box(
        slide,
        "Priority Score = (Vehicle Density × 0.40) + (Coverage Gap × 0.35) + (Population Density × 0.25)",
        0.5, 5.5, 9.2, 0.6,
        font_size=14, color=OO_LIGHT,
    )
    print("  Slide 9 — Building Classification")


def slide_10_market_potential(prs: Presentation) -> None:
    """Slide 10 — Market Potential Estimation."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Step 4: Market Potential Estimation Engine",
                 0.5, 0.2, 9.2, 0.8, font_size=28, bold=True, color=OO_WHITE)

    add_text_box(
        slide, "Core Formula:", 0.5, 1.15, 9.2, 0.5,
        font_size=16, bold=True, color=OO_RED,
    )
    add_text_box(
        slide,
        "Buildings Detected  ×  Households per Building (3.5)  ×  Market Share (20%)  =  Subscribers",
        0.5, 1.6, 9.2, 0.7,
        font_size=15, color=OO_LIGHT,
    )

    add_navy_box(slide, 0.4, 2.4, 9.3, 0.05, OO_RED)

    metrics = [
        ("350,000+", "Total Muscat Homes",   "Full addressable market"),
        ("70,000",   "Ooredoo Target (20%)", "At current market share"),
        ("5.5 OMR",  "ARPU / Month",         "Revenue per subscriber"),
        ("42%",      "FTTH Uptake Rate",     "Conversion assumption"),
    ]
    y = 2.6
    col_w = 2.3
    x_pos = [0.4, 2.75, 5.1, 7.45]
    for (val, label, sub), x in zip(metrics, x_pos):
        add_navy_box(slide, x, y, col_w - 0.05, 1.6, OO_MID_NAVY)
        add_text_box(slide, val, x + 0.1, y + 0.1, col_w - 0.25, 0.7,
                     font_size=26, bold=True, color=OO_RED, align=PP_ALIGN.CENTER)
        add_text_box(slide, label, x + 0.05, y + 0.75, col_w - 0.15, 0.45,
                     font_size=13, bold=True, color=OO_WHITE, align=PP_ALIGN.CENTER)
        add_text_box(slide, sub, x + 0.05, y + 1.15, col_w - 0.15, 0.35,
                     font_size=11, color="AAAAAA", align=PP_ALIGN.CENTER)

    bullets = [
        "◆  Each zone scored independently using population census + building detection data",
        "◆  Unserved market (10%) identified as priority greenfield — no competitive overlap",
        "◆  ROI calculated: Investment ÷ (Annual Revenue) = Payback years per zone",
    ]
    add_multiline_text_box(slide, bullets, 0.5, 4.4, 9.2, 2.0,
                           font_size=15, color=OO_LIGHT, line_spacing_pt=6)
    print("  Slide 10 — Market Potential")


def slide_11_market_share(prs: Presentation) -> None:
    """Slide 11 — Market Share Analysis."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Oman FTTH Market Share Analysis", 0.5, 0.2, 9.2, 0.8,
                 font_size=30, bold=True, color=OO_WHITE)

    operators = [
        ("Omantel",  "40%", "065A82", "Dominant incumbent — legacy DSL converting to fiber"),
        ("Awasr",    "30%", "00A651", "Aggressive pricing — fastest growing challenger"),
        ("Ooredoo",  "20%", OO_RED,   "Strong brand, mobile bundle advantage, expansion upside"),
        ("Others",   "10%", "FFA500", "Unserved greenfield — addressable by first-mover"),
    ]

    # Visual bar representation
    y = 1.3
    bar_total_width = 9.1
    shares = [40, 30, 20, 10]
    colors = ["065A82", "00A651", OO_RED, "FFA500"]
    x = 0.4
    add_text_box(slide, "Market Share Visual:", 0.4, y - 0.3, 4.0, 0.3,
                 font_size=13, bold=True, color="AAAAAA")
    for share, color in zip(shares, colors):
        w = bar_total_width * share / 100
        add_navy_box(slide, x, y, w - 0.02, 0.45, color)
        x += w

    y_detail = 2.05
    for name, pct, color, desc in operators:
        add_navy_box(slide, 0.4, y_detail, 0.45, 0.62, color)
        add_text_box(slide, pct, 1.0, y_detail + 0.05, 1.1, 0.5,
                     font_size=22, bold=True, color=OO_WHITE)
        add_text_box(slide, name, 2.1, y_detail + 0.0, 2.0, 0.3,
                     font_size=15, bold=True, color=OO_WHITE)
        add_text_box(slide, desc, 2.1, y_detail + 0.3, 7.3, 0.3,
                     font_size=13, color=OO_LIGHT)
        y_detail += 0.88

    add_text_box(
        slide,
        "Source: Ooredoo Falcon AI Market Analysis — Muscat, Oman 2026",
        0.4, 5.65, 9.2, 0.4,
        font_size=12, color="666666", italic=True,
    )
    print("  Slide 11 — Market Share")


def slide_12_market_opportunity(prs: Presentation) -> None:
    """Slide 12 — Market Opportunity."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Market Opportunity — Muscat FTTH", 0.5, 0.2, 9.2, 0.8,
                 font_size=30, bold=True, color=OO_WHITE)

    numbers = [
        ("350,000", "Total Homes in Muscat",    "Full addressable market",      OO_WHITE),
        ("140,000", "Omantel Homes (40%)",       "Largest single operator share", "065A82"),
        ("105,000", "Awasr Homes (30%)",         "Growing challenger footprint",  "00A651"),
        ("70,000",  "Ooredoo Homes (20%)",       "Current Ooredoo subscriber base", OO_RED),
        ("35,000",  "Unserved Others (10%)",     "Greenfield — first-mover wins", "FFA500"),
    ]

    y = 1.3
    for val, label, sub, color in numbers:
        add_navy_box(slide, 0.4, y, 9.3, 0.68, OO_MID_NAVY)
        add_text_box(slide, val, 0.55, y + 0.07, 1.9, 0.55,
                     font_size=24, bold=True, color=color, align=PP_ALIGN.RIGHT)
        add_text_box(slide, label, 2.55, y + 0.06, 4.2, 0.32,
                     font_size=14, bold=True, color=OO_WHITE)
        add_text_box(slide, sub, 2.55, y + 0.36, 6.8, 0.28,
                     font_size=12, color="AAAAAA")
        y += 0.84
    print("  Slide 12 — Market Opportunity")


def slide_13_ai_intelligence(prs: Presentation) -> None:
    """Slide 13 — AI Business Intelligence."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Step 5: AI Business Intelligence Engine",
                 0.5, 0.2, 9.2, 0.8, font_size=28, bold=True, color=OO_WHITE)

    add_text_box(
        slide,
        "Large Language Model (LLM) analysis generates executive-grade insights per zone:",
        0.5, 1.1, 9.2, 0.55, font_size=16, color=OO_LIGHT,
    )

    outputs = [
        ("📍  Area Overview",           "Zone demographics, location, infrastructure characteristics"),
        ("⚔️  Competitive Assessment",  "Per-zone market share, gaps vs Omantel & Awasr"),
        ("💪  Key Strengths",           "Density signals, coverage gap, first-mover indicators"),
        ("📋  Investment Recommendation","Tier-based advice: immediate / 6-month / Phase 2"),
        ("💰  Revenue Forecast",        "3-year projection at 8% CAGR per zone"),
        ("📣  Marketing Opportunities", "Channel mix, campaign message, customer segments"),
    ]

    y = 1.8
    for icon_label, desc in outputs:
        add_navy_box(slide, 0.4, y, 9.3, 0.55, OO_MID_NAVY)
        add_text_box(slide, icon_label, 0.6, y + 0.09, 3.0, 0.42,
                     font_size=14, bold=True, color=OO_WHITE)
        add_text_box(slide, desc, 3.6, y + 0.1, 6.0, 0.38,
                     font_size=13, color=OO_LIGHT)
        y += 0.67

    add_text_box(
        slide,
        "Supports OpenAI GPT-4 & Anthropic Claude via environment variable keys",
        0.5, 5.7, 9.2, 0.4, font_size=12, color="888888", italic=True,
    )
    print("  Slide 13 — AI Intelligence")


def slide_14_marketing_strategy(prs: Presentation) -> None:
    """Slide 14 — Marketing Strategy."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Step 6: AI Marketing Strategy Generator",
                 0.5, 0.2, 9.2, 0.8, font_size=28, bold=True, color=OO_WHITE)

    channels = [
        ("📱  SMS Campaigns",    "Targeted bulk SMS to existing Ooredoo mobile subscribers in zone"),
        ("💻  Digital Marketing","Geo-fenced Google/Meta ads optimized for Muscat FTTH intent"),
        ("🚪  Door-to-Door",     "Residential canvassing in high-density residential blocks"),
        ("🤝  Sales Teams",      "Dedicated field agents assigned per zone based on priority score"),
        ("📲  Social Media",     "Instagram/X brand campaigns with fiber speed testimonials"),
        ("🏪  Retail Stores",    "Ooredoo stores with dedicated FTTH promotion stands & QR codes"),
    ]

    packages = [
        ("Starter",   "10 Mbps",  "7 OMR/mo"),
        ("Home Plus", "50 Mbps",  "12 OMR/mo"),
        ("Premium",   "100 Mbps", "18 OMR/mo"),
    ]

    y = 1.25
    for icon_label, desc in channels[:4]:
        add_navy_box(slide, 0.4, y, 9.3, 0.5, OO_MID_NAVY)
        add_text_box(slide, icon_label, 0.6, y + 0.08, 2.8, 0.4,
                     font_size=14, bold=True, color=OO_WHITE)
        add_text_box(slide, desc, 3.5, y + 0.09, 6.1, 0.36,
                     font_size=13, color=OO_LIGHT)
        y += 0.62

    add_text_box(slide, "Recommended FTTH Packages:", 0.5, 3.85, 4.0, 0.4,
                 font_size=14, bold=True, color=OO_RED)
    x = 0.4
    for pkg_name, speed, price in packages:
        add_navy_box(slide, x, 4.3, 2.95, 1.1, OO_MID_NAVY)
        add_text_box(slide, pkg_name, x + 0.1, 4.35, 2.7, 0.35,
                     font_size=14, bold=True, color=OO_WHITE, align=PP_ALIGN.CENTER)
        add_text_box(slide, speed, x + 0.1, 4.66, 2.7, 0.38,
                     font_size=18, bold=True, color=OO_RED, align=PP_ALIGN.CENTER)
        add_text_box(slide, price, x + 0.1, 5.0, 2.7, 0.32,
                     font_size=13, color=OO_LIGHT, align=PP_ALIGN.CENTER)
        x += 3.1
    print("  Slide 14 — Marketing Strategy")


def slide_15_operational_benefits(prs: Presentation) -> None:
    """Slide 15 — Operational Benefits by Stakeholder."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Operational Benefits by Stakeholder",
                 0.5, 0.2, 9.2, 0.8, font_size=28, bold=True, color=OO_WHITE)

    stakeholders = [
        ("📣  Marketing",         "Identify high-value zones for targeted FTTH campaigns; "
                                   "geo-fenced digital ads powered by AI priority scores"),
        ("⚙️  Operations",        "Prioritize fiber rollout with AI-scored deployment maps; "
                                   "optimize cabinet placement & reduce infrastructure waste"),
        ("💼  Sales",             "Equip field teams with zone-level intelligence; "
                                   "convert door-to-door campaigns with data-driven targeting"),
        ("📈  Management",        "5-year revenue forecasts & ROI analysis; "
                                   "executive summaries exportable in one click for board reporting"),
        ("🌐  Network Planning",  "AI-recommended OLT/splitter placement; "
                                   "satellite-derived fiber route optimization across Muscat"),
    ]

    y = 1.3
    for icon_label, desc in stakeholders:
        add_navy_box(slide, 0.4, y, 9.3, 0.72, OO_MID_NAVY)
        add_text_box(slide, icon_label, 0.6, y + 0.12, 2.6, 0.55,
                     font_size=15, bold=True, color=OO_WHITE)
        add_text_box(slide, desc, 3.2, y + 0.12, 6.4, 0.55,
                     font_size=13, color=OO_LIGHT)
        y += 0.87
    print("  Slide 15 — Operational Benefits")


def slide_16_business_impact(prs: Presentation) -> None:
    """Slide 16 — Measurable Business Impact."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Measurable Business Impact",
                 0.5, 0.2, 9.2, 0.8, font_size=30, bold=True, color=OO_WHITE)

    kpis = [
        ("80%",     "Faster FTTH Planning",    "From 2-3 weeks to hours per zone"),
        ("40%",     "Cost Reduction",          "Fewer wasted field surveys & manual hours"),
        ("3×",      "Campaign Conversion",     "AI-targeted vs. blanket marketing"),
        ("25%",     "Better ROI",              "Investment directed to highest-score zones first"),
        ("Real-time","Zone Prioritization",    "Live competitive and coverage intelligence"),
        ("↑ FTTH",  "Penetration Rate",        "Data-driven targeting lifts subscriber conversion"),
    ]

    positions = [
        (0.4, 1.4), (3.4, 1.4), (6.4, 1.4),
        (0.4, 3.4), (3.4, 3.4), (6.4, 3.4),
    ]
    for (val, label, sub), (x, y) in zip(kpis, positions):
        add_navy_box(slide, x, y, 2.8, 1.7, OO_MID_NAVY)
        add_text_box(slide, val, x + 0.1, y + 0.12, 2.6, 0.7,
                     font_size=30, bold=True, color=OO_RED, align=PP_ALIGN.CENTER)
        add_text_box(slide, label, x + 0.05, y + 0.78, 2.7, 0.35,
                     font_size=12, bold=True, color=OO_WHITE, align=PP_ALIGN.CENTER)
        add_text_box(slide, sub, x + 0.05, y + 1.10, 2.7, 0.45,
                     font_size=11, color="AAAAAA", align=PP_ALIGN.CENTER)
    print("  Slide 16 — Business Impact")


def slide_17_roadmap(prs: Presentation) -> None:
    """Slide 17 — Future Roadmap."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)
    add_text_box(slide, "Future Roadmap — Ooredoo Falcon 3.0",
                 0.5, 0.2, 9.2, 0.8, font_size=30, bold=True, color=OO_WHITE)

    features = [
        ("Q1 2026", "Real-time 5G Coverage Heatmaps",     "Live network KPI overlays on zone map"),
        ("Q1 2026", "Automated Permit Integration",        "API to Muscat Municipality approval workflow"),
        ("Q2 2026", "Drone Imagery Pipeline",              "Own-fleet UAV data replaces satellite tiles"),
        ("Q2 2026", "Predictive Churn Model",              "ML identifies at-risk FTTH subscribers"),
        ("Q3 2026", "Multi-city Expansion Module",         "Extend platform to Salalah, Nizwa, Sohar"),
        ("Q3 2026", "AR Field Technician App",             "AR overlay showing fiber routes in real-world"),
        ("Q4 2026", "Enterprise API & White-label",        "License Falcon Intelligence to other telcos"),
    ]

    y = 1.25
    for quarter, title, desc in features:
        add_navy_box(slide, 0.4, y, 1.5, 0.5, OO_RED)
        add_text_box(slide, quarter, 0.45, y + 0.08, 1.4, 0.36,
                     font_size=12, bold=True, color=OO_WHITE, align=PP_ALIGN.CENTER)
        add_text_box(slide, title, 2.1, y + 0.04, 3.8, 0.42,
                     font_size=14, bold=True, color=OO_WHITE)
        add_text_box(slide, desc, 6.0, y + 0.07, 3.7, 0.38,
                     font_size=13, color=OO_LIGHT)
        y += 0.65
    print("  Slide 17 — Roadmap")


def slide_18_thank_you(prs: Presentation) -> None:
    """Slide 18 — Thank You / Vision."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, OO_NAVY)
    add_red_accent_line(slide, prs)

    add_text_box(slide, "🦅", 4.35, 0.5, 1.3, 1.2,
                 font_size=52, color=OO_WHITE, align=PP_ALIGN.CENTER)

    add_text_box(
        slide,
        "Empowering Smarter FTTH Expansion",
        0.8, 1.75, 8.6, 0.85,
        font_size=32, bold=True, color=OO_WHITE, align=PP_ALIGN.CENTER,
    )

    # Red divider
    add_navy_box(slide, 3.0, 2.7, 4.0, 0.06, OO_RED)

    add_text_box(
        slide,
        '"Ooredoo Falcon empowers Ooredoo Oman to make faster, smarter, and AI-driven '
        "FTTH expansion decisions by combining geospatial intelligence, computer vision, "
        "large language models, and marketing automation into a single enterprise platform.",
        0.8, 2.9, 8.6, 2.0,
        font_size=16, color=OO_LIGHT, align=PP_ALIGN.CENTER, italic=True,
    )

    add_navy_box(slide, 0.4, 5.15, 9.3, 0.05, OO_RED)

    add_text_box(
        slide,
        "Ooredoo Oman  ·  2026  |  AI-Powered FTTH Intelligence  |  CONFIDENTIAL",
        0.5, 5.3, 9.2, 0.5,
        font_size=13, color="888888", align=PP_ALIGN.CENTER,
    )
    print("  Slide 18 — Thank You")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("Ooredoo Falcon — Generating 18-slide PowerPoint presentation...")

    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(6.5)

    builders = [
        slide_01_title,
        slide_02_challenge,
        slide_03_why_ftth,
        slide_04_current_process,
        slide_05_solution,
        slide_06_architecture,
        slide_07_image_collection,
        slide_08_yolo_detection,
        slide_09_building_classification,
        slide_10_market_potential,
        slide_11_market_share,
        slide_12_market_opportunity,
        slide_13_ai_intelligence,
        slide_14_marketing_strategy,
        slide_15_operational_benefits,
        slide_16_business_impact,
        slide_17_roadmap,
        slide_18_thank_you,
    ]

    for fn in builders:
        fn(prs)

    prs.save(str(OUTPUT_FILE))
    print(f"\nPresentation saved: {OUTPUT_FILE}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
