"""
FALCON — Business Impact & Marketing Intelligence Presentation
5-slide executive deck: leads, campaigns, ROI
Run:  python scripts/create_business_impact_ppt.py
Out:  outputs/FALCON_Business_Impact.pptx
"""
from __future__ import annotations

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
    subprocess.run([sys.executable, "-m", "pip", "install", "python-pptx", "-q"], check=True)
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.chart.data import ChartData
    from pptx.enum.chart import XL_CHART_TYPE

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY    = RGBColor(0x0D, 0x1B, 0x2A)
NAVY2   = RGBColor(0x14, 0x26, 0x3D)
TEAL    = RGBColor(0x00, 0xC9, 0xA7)
TEAL_D  = RGBColor(0x00, 0x9E, 0x84)
ACCENT  = RGBColor(0xFF, 0xC3, 0x00)
BLUE    = RGBColor(0x1A, 0x6E, 0xB5)
RED     = RGBColor(0xE8, 0x3A, 0x3A)
ORANGE  = RGBColor(0xFD, 0x7E, 0x14)
GREEN   = RGBColor(0x28, 0xA7, 0x45)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
OFFWH   = RGBColor(0xF4, 0xF6, 0xF9)
LGRAY   = RGBColor(0xE8, 0xEC, 0xF0)
MGRAY   = RGBColor(0x99, 0xA5, 0xB2)
DGRAY   = RGBColor(0x44, 0x4F, 0x5A)
BLACK   = RGBColor(0x11, 0x15, 0x1A)

SW = Inches(13.33)
SH = Inches(7.5)


# ── Drawing primitives ─────────────────────────────────────────────────────────

def _rect(slide, x, y, w, h, fill=None, line_rgb=None, line_w=Pt(0.75), radius=0):
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


def _para(tf, text, size=16, bold=False, color=DGRAY, space_before=Pt(5)):
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


def _header(slide, title: str, subtitle: str = ""):
    _rect(slide, 0, 0, SW, Inches(1.1), fill=NAVY)
    _txt(slide, title,
         Inches(0.4), Inches(0.08), SW - Inches(2.5), Inches(0.72),
         size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    _txt(slide, "FALCON  |  Waheb Telecom",
         SW - Inches(2.4), Inches(0.18), Inches(2.2), Inches(0.60),
         size=13, bold=True, color=TEAL, align=PP_ALIGN.RIGHT)
    if subtitle:
        _txt(slide, subtitle,
             Inches(0.4), Inches(0.80), SW - Inches(0.8), Inches(0.28),
             size=10.5, italic=True, color=MGRAY, align=PP_ALIGN.LEFT)


def _footer(slide, page_num: int):
    _rect(slide, 0, SH - Inches(0.32), SW, Inches(0.32), fill=NAVY)
    _txt(slide,
         "FALCON  ·  AI-Powered FTTH Expansion Intelligence  ·  Waheb Telecom  ·  Muscat, Oman  ·  2026",
         Inches(0.3), SH - Inches(0.30), SW - Inches(1.0), Inches(0.28),
         size=8, color=MGRAY, align=PP_ALIGN.LEFT)
    _txt(slide, str(page_num),
         SW - Inches(0.5), SH - Inches(0.30), Inches(0.35), Inches(0.28),
         size=9, bold=True, color=TEAL, align=PP_ALIGN.RIGHT)


def _kpi_card(slide, x, y, w, h, value, label, sub="", vcolor=TEAL, bg=NAVY2,
               border=TEAL):
    _rect(slide, x, y, w, h, fill=bg, line_rgb=border, line_w=Pt(1.5))
    _txt(slide, value, x + Inches(0.1), y + Inches(0.1),
         w - Inches(0.2), h * 0.50,
         size=36, bold=True, color=vcolor, align=PP_ALIGN.CENTER)
    _txt(slide, label, x + Inches(0.08), y + h * 0.56,
         w - Inches(0.16), h * 0.28,
         size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    if sub:
        _txt(slide, sub, x + Inches(0.08), y + h * 0.82,
             w - Inches(0.16), h * 0.18,
             size=9.5, italic=True, color=MGRAY, align=PP_ALIGN.CENTER)


def _pill(slide, text, x, y, fill=TEAL_D, text_color=WHITE, size=10.5):
    tw = max(Inches(1.2), Inches(len(text) * 0.092 + 0.28))
    _rect(slide, x, y, tw, Inches(0.30), fill=fill)
    _txt(slide, text, x + Inches(0.06), y + Inches(0.03), tw - Inches(0.12),
         Inches(0.26), size=size, bold=True, color=text_color,
         align=PP_ALIGN.CENTER)
    return tw


def _divider(slide, y, color=TEAL, thickness=Pt(1.5)):
    shp = slide.shapes.add_shape(1, Inches(0.4), y, SW - Inches(0.8), thickness)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()


def _bar_chart(slide, x, y, w, h, cats, series_data, title=""):
    from pptx.chart.data import ChartData
    from pptx.enum.chart import XL_CHART_TYPE
    cd = ChartData()
    cd.categories = cats
    for name, vals in series_data:
        cd.add_series(name, vals)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, w, h, cd).chart
    chart.has_legend = len(series_data) > 1
    chart.has_title = bool(title)
    if title:
        chart.chart_title.text_frame.text = title
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = DGRAY
    plot = chart.plots[0]
    for i, (_, vals) in enumerate(series_data):
        colors = [TEAL, BLUE, ORANGE, RED, ACCENT]
        ser = plot.series[i]
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = colors[i % len(colors)]
    chart.value_axis.has_gridlines = True
    chart.category_axis.tick_labels.font.size = Pt(9)
    chart.value_axis.tick_labels.font.size = Pt(9)
    return chart


def _donut_chart(slide, x, y, w, h, cats, vals, title=""):
    cd = ChartData()
    cd.categories = cats
    cd.add_series("Share", vals)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.DOUGHNUT, x, y, w, h, cd).chart
    chart.has_legend = True
    chart.has_title = bool(title)
    if title:
        chart.chart_title.text_frame.text = title
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        chart.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = DGRAY
    colors = [BLUE, RED, ORANGE, GREEN]
    for i, c in enumerate(colors[:len(cats)]):
        pt = chart.plots[0].series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = c
    chart.legend.font.size = Pt(9)
    return chart


# ══════════════════════════════════════════════════════════════════════════════
#  SLIDE 1 — Title & Business Challenge
# ══════════════════════════════════════════════════════════════════════════════

def slide_01_title(prs):
    slide = _blank(prs)

    # Full-bleed background
    _rect(slide, 0, 0, SW, SH, fill=NAVY)

    # Left accent strip
    _rect(slide, 0, 0, Inches(0.18), SH, fill=TEAL)

    # Right decorative panel
    _rect(slide, SW - Inches(4.2), 0, Inches(4.2), SH, fill=NAVY2)
    _rect(slide, SW - Inches(4.2), 0, Inches(0.04), SH, fill=TEAL)

    # Geometric accent circles (decorative)
    for cx, cy, cr, col in [
        (SW - Inches(2.1), Inches(1.5),  Inches(2.8), RGBColor(0x00, 0x40, 0x60)),
        (SW - Inches(1.0), Inches(3.5),  Inches(1.8), RGBColor(0x00, 0x55, 0x78)),
        (SW - Inches(3.5), Inches(5.5),  Inches(1.4), RGBColor(0x00, 0x30, 0x50)),
    ]:
        shp = slide.shapes.add_shape(9, cx - cr / 2, cy - cr / 2, cr, cr)
        shp.fill.solid(); shp.fill.fore_color.rgb = col
        shp.line.fill.background()

    # Logo / brand
    _txt(slide, "🦅", Inches(0.5), Inches(0.5), Inches(0.9), Inches(0.9),
         size=40, color=TEAL, align=PP_ALIGN.CENTER)
    _txt(slide, "FALCON", Inches(1.3), Inches(0.52), Inches(2.4), Inches(0.7),
         size=32, bold=True, color=TEAL, align=PP_ALIGN.LEFT)
    _txt(slide, "AI Coverage & Location Opportunity Network",
         Inches(1.3), Inches(1.10), Inches(5.8), Inches(0.4),
         size=12, italic=True, color=MGRAY, align=PP_ALIGN.LEFT)

    # Main title
    _txt(slide, "Turning Aerial Intelligence\ninto Revenue Growth",
         Inches(0.5), Inches(1.85), Inches(8.5), Inches(2.0),
         size=44, bold=True, color=WHITE, align=PP_ALIGN.LEFT)

    # Subtitle line
    _divider(slide, Inches(3.95), color=TEAL, thickness=Pt(2))
    _txt(slide,
         "How FALCON identifies high-value leads, optimises marketing spend,\n"
         "and unlocks 105,000 unserved homes across Muscat",
         Inches(0.5), Inches(4.10), Inches(8.2), Inches(0.9),
         size=16, italic=True, color=OFFWH, align=PP_ALIGN.LEFT)

    # Stat pills
    for i, (val, lbl, col) in enumerate([
        ("350K", "Total Homes in Market", TEAL),
        ("30%",  "Market Unserved Today",  ACCENT),
        ("776K OMR", "Addressable Revenue",  ORANGE),
    ]):
        bx = Inches(0.5 + i * 2.72)
        by = Inches(5.2)
        _rect(slide, bx, by, Inches(2.5), Inches(1.5), fill=NAVY2,
              line_rgb=col, line_w=Pt(1.5))
        _txt(slide, val, bx, by + Inches(0.15), Inches(2.5), Inches(0.75),
             size=30, bold=True, color=col, align=PP_ALIGN.CENTER)
        _txt(slide, lbl, bx, by + Inches(0.85), Inches(2.5), Inches(0.55),
             size=10, color=MGRAY, align=PP_ALIGN.CENTER)

    # Right panel content
    _txt(slide, "Business Impact Deck",
         SW - Inches(4.0), Inches(1.0), Inches(3.8), Inches(0.45),
         size=14, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    _divider(slide, Inches(1.5), color=TEAL_D, thickness=Pt(0.75))

    for i, line in enumerate([
        "01  Market Opportunity",
        "02  Business Impact",
        "03  Lead Generation Engine",
        "04  Campaign Optimisation",
        "05  Growth Roadmap",
    ]):
        _txt(slide, line,
             SW - Inches(4.0), Inches(1.75 + i * 0.72), Inches(3.6), Inches(0.55),
             size=13.5, bold=(i == 0), color=WHITE if i > 0 else ACCENT,
             align=PP_ALIGN.LEFT)

    _txt(slide, "Waheb Telecom  ·  Muscat, Oman  ·  2026",
         Inches(0.5), SH - Inches(0.55), Inches(6.0), Inches(0.4),
         size=10, italic=True, color=MGRAY, align=PP_ALIGN.LEFT)

    return slide


# ══════════════════════════════════════════════════════════════════════════════
#  SLIDE 2 — Market Opportunity & FALCON's Business Impact
# ══════════════════════════════════════════════════════════════════════════════

def slide_02_market(prs):
    slide = _blank(prs)
    _rect(slide, 0, 0, SW, SH, fill=OFFWH)
    _header(slide, "The Market Opportunity",
            "Muscat FTTH landscape — where Waheb stands and where the growth is hidden")
    _footer(slide, 2)

    # ── KPI row ───────────────────────────────────────────────────────────────
    kpis = [
        ("10%",       "Waheb's\nCurrent Share",   "3rd place today",  BLUE,   NAVY),
        ("35%",       "RAMI\n(Omantel)",           "Dominant player",  RED,    NAVY2),
        ("25%",       "Don\n(Ooredoo)",            "Fast-growing",     ORANGE, NAVY2),
        ("30%",       "Unserved\nHomes",           "= 105,000 homes",  TEAL,   NAVY),
    ]
    for i, (val, lbl, sub, vc, bg) in enumerate(kpis):
        _kpi_card(slide,
                  x=Inches(0.25 + i * 3.27), y=Inches(1.25),
                  w=Inches(3.08), h=Inches(1.75),
                  value=val, label=lbl, sub=sub, vcolor=vc, bg=bg, border=vc)

    # ── Market share donut ────────────────────────────────────────────────────
    _donut_chart(slide,
                 x=Inches(0.25), y=Inches(3.15),
                 w=Inches(4.5), h=Inches(3.85),
                 cats=["Waheb", "RAMI", "Don", "Unserved"],
                 vals=[10, 35, 25, 30],
                 title="Current Market Share — Muscat FTTH")

    # ── Revenue opportunity panel ─────────────────────────────────────────────
    _rect(slide, Inches(4.95), Inches(3.15), Inches(8.1), Inches(3.85),
          fill=WHITE, line_rgb=LGRAY, line_w=Pt(0.75))

    _txt(slide, "Revenue Opportunity at a Glance",
         Inches(5.1), Inches(3.25), Inches(7.8), Inches(0.45),
         size=14, bold=True, color=NAVY, align=PP_ALIGN.LEFT)
    _divider(slide, Inches(3.78), color=TEAL, thickness=Pt(1.2))

    rows = [
        ("Addressable Market (unserved 30%)",         "105,000 homes",     TEAL),
        ("Est. FTTH adoption rate",                   "42%  →  44,100 subs", BLUE),
        ("ARPU (monthly)",                            "OMR 5.50 / home",   DGRAY),
        ("Annual revenue if Waheb captures 8% more",  "OMR 776,160 / yr",  GREEN),
        ("Required investment (30 cabinets)",         "OMR 4,650,000",     ORANGE),
        ("Payback period",                            "≈ 71 months",       MGRAY),
        ("Each 1% market-share gain",                 "≈ OMR 700,000 / yr",ACCENT),
    ]
    for i, (lbl, val, vc) in enumerate(rows):
        by = Inches(3.92 + i * 0.42)
        bg = OFFWH if i % 2 == 0 else WHITE
        _rect(slide, Inches(5.0), by, Inches(8.0), Inches(0.40), fill=bg)
        _txt(slide, lbl, Inches(5.1), by + Inches(0.06),
             Inches(5.0), Inches(0.30), size=11, color=DGRAY)
        _txt(slide, val, Inches(10.1), by + Inches(0.06),
             Inches(2.75), Inches(0.30), size=11, bold=True, color=vc,
             align=PP_ALIGN.RIGHT)

    # ── FALCON insight callout ────────────────────────────────────────────────
    _rect(slide, Inches(4.95), Inches(6.75), Inches(8.1), Inches(0.37),
          fill=TEAL, line_rgb=TEAL_D, line_w=Pt(1))
    _txt(slide,
         "FALCON pinpoints exactly which zones to target — eliminating guesswork and cutting campaign waste by up to 60%",
         Inches(5.05), Inches(6.78), Inches(7.9), Inches(0.32),
         size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    return slide


# ══════════════════════════════════════════════════════════════════════════════
#  SLIDE 3 — Business Impact of FALCON
# ══════════════════════════════════════════════════════════════════════════════

def slide_03_impact(prs):
    slide = _blank(prs)
    _rect(slide, 0, 0, SW, SH, fill=OFFWH)
    _header(slide, "Business Impact of FALCON",
            "Replacing manual guesswork with AI-driven precision across the full FTTH value chain")
    _footer(slide, 3)

    # ── Before / After banner ─────────────────────────────────────────────────
    _rect(slide, Inches(0.25), Inches(1.2), Inches(6.2), Inches(0.42), fill=RED)
    _txt(slide, "BEFORE  — Traditional Approach",
         Inches(0.35), Inches(1.22), Inches(6.0), Inches(0.38),
         size=13, bold=True, color=WHITE)

    _rect(slide, Inches(6.85), Inches(1.2), Inches(6.2), Inches(0.42), fill=TEAL_D)
    _txt(slide, "AFTER  — FALCON AI Approach",
         Inches(6.95), Inches(1.22), Inches(6.0), Inches(0.38),
         size=13, bold=True, color=WHITE)

    vs_items = [
        ("Manual survey teams\n(2–3 weeks per zone)",     "YOLOv8 aerial scan\n(seconds per zone)"),
        ("Gut-feel zone prioritisation",                   "AI score (density + gap + population)"),
        ("Uniform marketing spend\nacross all zones",     "Budget allocated to top-tier zones only"),
        ("Reactive competitive intel",                     "Real-time Waheb / RAMI / Don gap map"),
        ("12-month deployment cycles",                     "30-day sprint on ranked opportunities"),
        ("~40% wrong-zone investment",                     "Targeted ROI: 13-month payback confirmed"),
    ]

    for i, (before, after) in enumerate(vs_items):
        by = Inches(1.75 + i * 0.80)
        bg = OFFWH if i % 2 == 0 else WHITE
        _rect(slide, Inches(0.25), by, Inches(6.2), Inches(0.74), fill=bg,
              line_rgb=LGRAY, line_w=Pt(0.5))
        _txt(slide, before, Inches(0.4), by + Inches(0.08),
             Inches(5.9), Inches(0.60), size=11, color=DGRAY)

        _rect(slide, Inches(6.85), by, Inches(6.2), Inches(0.74), fill=NAVY2,
              line_rgb=TEAL_D, line_w=Pt(0.5))
        _txt(slide, after, Inches(7.0), by + Inches(0.08),
             Inches(5.9), Inches(0.60), size=11, bold=True, color=WHITE)

    # VS separator
    _rect(slide, Inches(6.55), Inches(1.15), Inches(0.5), SH - Inches(1.55), fill=ACCENT)
    _txt(slide, "VS", Inches(6.55), Inches(4.0), Inches(0.5), Inches(0.45),
         size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    # ── Impact KPIs ───────────────────────────────────────────────────────────
    _rect(slide, 0, Inches(6.7), SW, Inches(0.5), fill=NAVY)
    impact_kpis = [
        ("60%", "Less Campaign Waste"),
        ("8×",  "Faster Zone Analysis"),
        ("42%", "Lead Conversion Lift"),
        ("OMR 776K", "Additional Annual Revenue"),
        ("71 mo", "Clear Payback Timeline"),
    ]
    for i, (val, lbl) in enumerate(impact_kpis):
        bx = Inches(0.18 + i * 2.62)
        _txt(slide, val, bx, Inches(6.72), Inches(2.4), Inches(0.26),
             size=14, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
        _txt(slide, lbl, bx, Inches(6.96), Inches(2.4), Inches(0.20),
             size=8.5, color=MGRAY, align=PP_ALIGN.CENTER)

    return slide


# ══════════════════════════════════════════════════════════════════════════════
#  SLIDE 4 — Lead Generation Engine
# ══════════════════════════════════════════════════════════════════════════════

def slide_04_leads(prs):
    slide = _blank(prs)
    _rect(slide, 0, 0, SW, SH, fill=OFFWH)
    _header(slide, "Lead Generation Engine",
            "FALCON converts satellite imagery into a ranked pipeline of sales-ready zones")
    _footer(slide, 4)

    # ── Scoring formula banner ────────────────────────────────────────────────
    _rect(slide, Inches(0.25), Inches(1.22), SW - Inches(0.5), Inches(0.68),
          fill=NAVY, line_rgb=TEAL, line_w=Pt(1.5))
    _txt(slide,
         "AI Priority Score  =  40% × Vehicle Density   +   35% × Coverage Gap   +   25% × Population",
         Inches(0.4), Inches(1.30), SW - Inches(0.8), Inches(0.55),
         size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # ── Lead tier columns ─────────────────────────────────────────────────────
    tier_data = {
        "HIGH PRIORITY": {
            "color": TEAL, "bg": NAVY2, "score": "Score ≥ 68",
            "zones": [
                ("Al Seeb City",     "Score 84", "0 competitor cabinets",  "OMR 185K / yr"),
                ("Al Amerat",        "Score 79", "0 competitor cabinets",  "OMR 150K / yr"),
                ("Bawshar North",    "Score 74", "Low RAMI presence",       "OMR 120K / yr"),
            ],
        },
        "MEDIUM PRIORITY": {
            "color": ACCENT, "bg": RGBColor(0x2A, 0x24, 0x00), "score": "Score 40–68",
            "zones": [
                ("Sidab",            "Score 62", "4 RAMI cabinets",        "OMR 31K / yr"),
                ("Al Khuwair South", "Score 57", "Mixed competition",       "OMR 68K / yr"),
                ("Madinat Sultan",   "Score 51", "Don coverage gap",        "OMR 55K / yr"),
            ],
        },
        "WATCH LIST": {
            "color": ORANGE, "bg": RGBColor(0x2A, 0x18, 0x00), "score": "Score < 40",
            "zones": [
                ("Ruwi",             "Score 38", "RAMI dominant",           "Monitor only"),
                ("Mutrah",           "Score 31", "Saturated",               "Defensive play"),
                ("Wattayah",         "Score 29", "RAMI + Don overlap",      "Low priority"),
            ],
        },
    }

    col_w = Inches(4.25)
    for ci, (tier, td) in enumerate(tier_data.items()):
        bx = Inches(0.25 + ci * 4.36)
        by = Inches(2.03)

        # Column header
        _rect(slide, bx, by, col_w, Inches(0.50), fill=td["color"])
        _txt(slide, tier, bx + Inches(0.1), by + Inches(0.06),
             col_w - Inches(0.2), Inches(0.38),
             size=11, bold=True, color=NAVY if td["color"] == ACCENT else WHITE,
             align=PP_ALIGN.CENTER)
        _txt(slide, td["score"],
             bx + Inches(0.1), by + Inches(0.36),
             col_w - Inches(0.2), Inches(0.20),
             size=8.5, italic=True,
             color=NAVY if td["color"] == ACCENT else RGBColor(0xCC, 0xFF, 0xF0),
             align=PP_ALIGN.CENTER)

        for zi, (zone, score, intel, rev) in enumerate(td["zones"]):
            zy = by + Inches(0.55 + zi * 1.32)
            _rect(slide, bx, zy, col_w, Inches(1.25), fill=td["bg"],
                  line_rgb=td["color"], line_w=Pt(1.0))
            _txt(slide, zone, bx + Inches(0.12), zy + Inches(0.08),
                 col_w - Inches(0.24), Inches(0.38),
                 size=13, bold=True, color=td["color"])
            _txt(slide, score, bx + Inches(0.12), zy + Inches(0.44),
                 col_w - Inches(0.24), Inches(0.26),
                 size=10, bold=True, color=WHITE)
            _txt(slide, intel, bx + Inches(0.12), zy + Inches(0.68),
                 col_w - Inches(0.24), Inches(0.26),
                 size=9.5, italic=True, color=MGRAY)
            _txt(slide, rev, bx + Inches(0.12), zy + Inches(0.93),
                 col_w - Inches(0.24), Inches(0.26),
                 size=10, bold=True, color=ACCENT)

    # ── Bottom insight strip ──────────────────────────────────────────────────
    _rect(slide, 0, Inches(6.65), SW, Inches(0.53), fill=TEAL_D)
    _txt(slide,
         "FALCON automatically refreshes scores when new aerial imagery is ingested  "
         "|  Sales team receives ranked lead list every 24 hours  "
         "|  Each lead includes revenue estimate + competitive risk rating",
         Inches(0.3), Inches(6.68), SW - Inches(0.6), Inches(0.47),
         size=10, bold=False, color=WHITE, align=PP_ALIGN.CENTER)

    return slide


# ══════════════════════════════════════════════════════════════════════════════
#  SLIDE 5 — Marketing Campaign Optimisation & Growth Roadmap
# ══════════════════════════════════════════════════════════════════════════════

def slide_05_roadmap(prs):
    slide = _blank(prs)
    _rect(slide, 0, 0, SW, SH, fill=NAVY)
    _header(slide, "Campaign Optimisation & Growth Roadmap",
            "Data-driven playbook: from targeted acquisition to 18% market share")
    _footer(slide, 5)

    # Override header bg (already set by _header)

    # ── Left: Campaign strategy ───────────────────────────────────────────────
    _rect(slide, Inches(0.25), Inches(1.22), Inches(5.8), Inches(5.4),
          fill=NAVY2, line_rgb=TEAL_D, line_w=Pt(1))

    _txt(slide, "Campaign Optimisation Playbook",
         Inches(0.4), Inches(1.32), Inches(5.5), Inches(0.40),
         size=13, bold=True, color=TEAL)
    _divider(slide, Inches(1.78), color=TEAL_D, thickness=Pt(0.75))

    campaigns = [
        ("HYPER-LOCAL TARGETING", TEAL,
         "Serve digital ads only within High-Priority zone\n"
         "boundaries detected by FALCON — eliminate 60% of\n"
         "wasted impressions outside addressable homes."),
        ("COMPETITIVE DISPLACEMENT", ORANGE,
         "When FALCON detects RAMI / Don cabinet deployment,\n"
         "trigger 48-hour counter-campaign in adjacent blocks.\n"
         "Respond before competitor locks-in subscribers."),
        ("LIFECYCLE UPSELL", ACCENT,
         "K-Means cluster segments (Premium / Mid / Value)\n"
         "drive personalised bundle offers — 5G + FTTH cross-sell\n"
         "boosts ARPU from OMR 5.50 to OMR 8.20 target."),
        ("CHURN FIREWALL", BLUE,
         "Demand-score drop alerts signal risk zones; proactivly\n"
         "deploy retention offers before subscribers switch to\n"
         "RAMI or Don — projected churn reduction: 18%."),
    ]

    for i, (title, col, desc) in enumerate(campaigns):
        by = Inches(1.90 + i * 1.18)
        _rect(slide, Inches(0.3), by, Inches(0.22), Inches(0.90), fill=col)
        _txt(slide, title, Inches(0.62), by + Inches(0.02),
             Inches(5.2), Inches(0.30), size=10.5, bold=True, color=col)
        _txt(slide, desc, Inches(0.62), by + Inches(0.28),
             Inches(5.2), Inches(0.65), size=9.5, color=OFFWH, wrap=True)

    # ── Right: Growth roadmap timeline ────────────────────────────────────────
    _rect(slide, Inches(6.35), Inches(1.22), Inches(6.75), Inches(5.4),
          fill=NAVY2, line_rgb=TEAL_D, line_w=Pt(1))

    _txt(slide, "12-Month Growth Roadmap",
         Inches(6.5), Inches(1.32), Inches(6.4), Inches(0.40),
         size=13, bold=True, color=TEAL)
    _divider(slide, Inches(1.78), color=TEAL_D, thickness=Pt(0.75))

    milestones = [
        ("M 1–2",   TEAL,   "Deploy 30 cabinets in Al Seeb & Al Amerat.\n"
                             "Launch hyper-local digital acquisition campaigns."),
        ("M 3–4",   BLUE,   "FALCON re-scores all 15 zones with fresh data.\n"
                             "Trigger competitor-displacement campaigns in Sidab."),
        ("M 5–6",   ACCENT, "K-Means upsell bundles live for 11,760 new subs.\n"
                             "Revenue run-rate reaches OMR 64,680 / month."),
        ("M 7–9",   ORANGE, "Share target 14% — churn firewall active.\n"
                             "Add 10 more cabinets in Medium-Priority zones."),
        ("M 10–12", GREEN,  "18% market share achieved  (+8 pts from today).\n"
                             "Annual incremental revenue: OMR 776,160."),
    ]

    for i, (period, col, desc) in enumerate(milestones):
        by = Inches(1.90 + i * 0.88)
        _rect(slide, Inches(6.4), by + Inches(0.10), Inches(0.90), Inches(0.35), fill=col)
        _txt(slide, period, Inches(6.42), by + Inches(0.11),
             Inches(0.88), Inches(0.33), size=10, bold=True,
             color=NAVY if col == ACCENT else WHITE, align=PP_ALIGN.CENTER)
        _txt(slide, desc, Inches(7.42), by + Inches(0.06),
             Inches(5.45), Inches(0.74), size=9.5, color=OFFWH, wrap=True)
        if i < len(milestones) - 1:
            shp = slide.shapes.add_shape(1, Inches(6.82), by + Inches(0.46),
                                         Pt(3), Inches(0.42))
            shp.fill.solid(); shp.fill.fore_color.rgb = col
            shp.line.fill.background()

    # ── Bottom CTA bar ────────────────────────────────────────────────────────
    _rect(slide, 0, Inches(6.72), SW, Inches(0.56), fill=TEAL)
    _txt(slide,
         "Ready to act?  →  Run FALCON today:  python scripts/build_demo.py   "
         "|   Dashboard: localhost:8501   |   API: localhost:8000/docs",
         Inches(0.3), Inches(6.76), SW - Inches(0.6), Inches(0.48),
         size=11.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    return slide


# ══════════════════════════════════════════════════════════════════════════════
#  Main build
# ══════════════════════════════════════════════════════════════════════════════

def build():
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH

    print("Building slide 1 — Title & Vision")
    slide_01_title(prs)

    print("Building slide 2 — Market Opportunity")
    slide_02_market(prs)

    print("Building slide 3 — Business Impact")
    slide_03_impact(prs)

    print("Building slide 4 — Lead Generation Engine")
    slide_04_leads(prs)

    print("Building slide 5 — Campaign Optimisation & Roadmap")
    slide_05_roadmap(prs)

    out = ROOT / "outputs" / "FALCON_Business_Impact.pptx"
    out.parent.mkdir(exist_ok=True)
    prs.save(str(out))
    print(f"\nSaved -> {out}")
    return out


if __name__ == "__main__":
    build()
