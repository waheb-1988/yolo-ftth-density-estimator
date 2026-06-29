"""
FALCON -- Computer Vision Component: 2-Slide Deck
Slide A: Full Detection Pipeline (acquire -> preprocess -> tile -> infer -> NMS -> output)
Slide B: YOLOv8 Architecture, Dataset & Intelligent Scoring Engine
Run:  python scripts/create_cv_slides.py
Out:  outputs/FALCON_CV_Slides.pptx
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
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "python-pptx", "-q"], check=True)
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY    = RGBColor(0x09, 0x12, 0x1F)
NAVY2   = RGBColor(0x11, 0x1F, 0x35)
NAVY3   = RGBColor(0x18, 0x2B, 0x48)
TEAL    = RGBColor(0x00, 0xE5, 0xBB)
TEAL_D  = RGBColor(0x00, 0xB0, 0x90)
BLUE    = RGBColor(0x1E, 0x90, 0xFF)
BLUE_D  = RGBColor(0x10, 0x60, 0xC0)
ORANGE  = RGBColor(0xFF, 0x80, 0x00)
ORANGE_D= RGBColor(0xCC, 0x60, 0x00)
GOLD    = RGBColor(0xFF, 0xD0, 0x00)
GOLD_D  = RGBColor(0xCC, 0xA0, 0x00)
GREEN   = RGBColor(0x00, 0xE0, 0x70)
GREEN_D = RGBColor(0x00, 0xA8, 0x52)
RED     = RGBColor(0xFF, 0x45, 0x55)
RED_D   = RGBColor(0xCC, 0x22, 0x33)
PURPLE  = RGBColor(0xAA, 0x55, 0xFF)
PURPLE_D= RGBColor(0x77, 0x22, 0xCC)
CYAN    = RGBColor(0x00, 0xFF, 0xFF)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
OFFWH   = RGBColor(0xE8, 0xF0, 0xF8)
LGRAY   = RGBColor(0xCC, 0xD6, 0xE0)
MGRAY   = RGBColor(0x88, 0x98, 0xAA)
DGRAY   = RGBColor(0x44, 0x55, 0x66)
BLACK   = RGBColor(0x05, 0x08, 0x10)

SW = Inches(13.33)
SH = Inches(7.5)

STEP_COLORS = [BLUE, TEAL, ORANGE, GOLD, RED, GREEN]
STEP_DARK   = [BLUE_D, TEAL_D, ORANGE_D, GOLD_D, RED_D, GREEN_D]
# Dark glow backdrop: one-quarter brightness version of each step color
STEP_GLOW   = [
    RGBColor(0x07, 0x24, 0x3F),   # BLUE glow
    RGBColor(0x00, 0x39, 0x2E),   # TEAL glow
    RGBColor(0x3F, 0x20, 0x00),   # ORANGE glow
    RGBColor(0x3F, 0x34, 0x00),   # GOLD glow
    RGBColor(0x3F, 0x11, 0x15),   # RED glow
    RGBColor(0x00, 0x38, 0x1C),   # GREEN glow
]


# ── Drawing primitives ─────────────────────────────────────────────────────────

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


def _circle(slide, cx, cy, r, fill, line_rgb=None, line_w=Pt(0)):
    shp = slide.shapes.add_shape(9, cx - r, cy - r, r * 2, r * 2)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line_rgb:
        shp.line.color.rgb = line_rgb
        shp.line.width = line_w
    else:
        shp.line.fill.background()
    return shp


def _arrow_right(slide, x, y, w, h, fill, line_rgb=None):
    shp = slide.shapes.add_shape(13, x, y, w, h)  # MSO RIGHT_ARROW = 13
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line_rgb:
        shp.line.color.rgb = line_rgb
        shp.line.width = Pt(0.5)
    else:
        shp.line.fill.background()
    return shp


def _arrow_down(slide, x, y, w, h, fill):
    shp = slide.shapes.add_shape(36, x, y, w, h)  # MSO DOWN_ARROW = 36
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    return shp


def _txt(slide, text, x, y, w, h, size=14, bold=False, italic=False,
         color=WHITE, align=PP_ALIGN.LEFT, wrap=True):
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


def _txt_multi(slide, lines, x, y, w, h, sizes=None, bolds=None,
               colors=None, align=PP_ALIGN.LEFT, spacing=Pt(4)):
    """Multiple paragraphs in one textbox. lines = list of str."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    sizes  = sizes  or [12] * len(lines)
    bolds  = bolds  or [False] * len(lines)
    colors = colors or [WHITE] * len(lines)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before = spacing if i > 0 else Pt(0)
        run = p.add_run()
        run.text = line
        run.font.size = Pt(sizes[i])
        run.font.bold = bolds[i]
        run.font.color.rgb = colors[i]
    return tb


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _header(slide, title, subtitle=""):
    # Gradient-simulated dark header
    _rect(slide, 0, 0, SW, Inches(1.15), fill=NAVY)
    _rect(slide, 0, 0, SW, Inches(0.06), fill=TEAL)        # top glow strip
    _rect(slide, 0, 0, Inches(0.12), Inches(1.15), fill=TEAL)  # left accent
    _txt(slide, title, Inches(0.28), Inches(0.08), SW - Inches(3.0), Inches(0.70),
         size=26, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    _txt(slide, "FALCON  |  Computer Vision Module",
         SW - Inches(2.85), Inches(0.14), Inches(2.7), Inches(0.52),
         size=11, bold=True, color=TEAL, align=PP_ALIGN.RIGHT)
    if subtitle:
        _txt(slide, subtitle, Inches(0.28), Inches(0.80),
             SW - Inches(0.6), Inches(0.30),
             size=10, italic=True, color=MGRAY, align=PP_ALIGN.LEFT)


def _footer(slide, page):
    _rect(slide, 0, SH - Inches(0.30), SW, Inches(0.30), fill=NAVY)
    _txt(slide, "FALCON  |  YOLOv8  |  Ultralytics  |  Waheb Telecom  |  Muscat, Oman  |  2026",
         Inches(0.3), SH - Inches(0.28), SW - Inches(1.2), Inches(0.26),
         size=8, color=MGRAY, align=PP_ALIGN.LEFT)
    _txt(slide, str(page),
         SW - Inches(0.5), SH - Inches(0.28), Inches(0.35), Inches(0.26),
         size=9, bold=True, color=TEAL, align=PP_ALIGN.RIGHT)


def _badge(slide, text, x, y, fill, text_color=NAVY, size=9.5):
    tw = max(Inches(0.9), Inches(len(text) * 0.086 + 0.24))
    _rect(slide, x, y, tw, Inches(0.26), fill=fill)
    _txt(slide, text, x + Inches(0.05), y + Inches(0.03),
         tw - Inches(0.10), Inches(0.22), size=size, bold=True,
         color=text_color, align=PP_ALIGN.CENTER)
    return tw


def _glow_rect(slide, x, y, w, h, fill, glow_color, border_color):
    """Dark card with colored top accent bar + subtle border."""
    _rect(slide, x, y, w, h, fill=fill, line_rgb=border_color, line_w=Pt(1.2))
    _rect(slide, x, y, w, Inches(0.065), fill=glow_color)   # top accent strip
    return


# ── Mini aerial image mock ────────────────────────────────────────────────────

def _draw_aerial_mock(slide, x, y, w, h):
    """Draws a top-down urban grid with tiny vehicles + cyan bounding boxes."""
    sand = RGBColor(0xB8, 0xAA, 0x8C)
    road = RGBColor(0x28, 0x28, 0x28)
    side = RGBColor(0x60, 0x60, 0x58)
    b1   = RGBColor(0x88, 0x84, 0x78)
    b2   = RGBColor(0x70, 0x6C, 0x60)
    b3   = RGBColor(0x9A, 0x96, 0x88)

    _rect(slide, x, y, w, h, fill=sand)

    # Horizontal road
    ry = y + h * 0.44
    rh = Inches(0.12)
    _rect(slide, x, ry - rh/2, w, rh, fill=road)
    _rect(slide, x, ry - rh/2, w, Inches(0.018), fill=side)
    _rect(slide, x, ry + rh/2 - Inches(0.018), w, Inches(0.018), fill=side)

    # Vertical road
    rx = x + w * 0.54
    rw = Inches(0.09)
    _rect(slide, rx - rw/2, y, rw, h, fill=road)
    _rect(slide, rx - rw/2, y, Inches(0.014), h, fill=side)
    _rect(slide, rx + rw/2 - Inches(0.014), y, Inches(0.014), h, fill=side)

    # Building blocks
    blocks = [
        (x + Inches(0.06), y + Inches(0.08), Inches(0.28), Inches(0.38), b1),
        (x + Inches(0.40), y + Inches(0.06), Inches(0.18), Inches(0.32), b2),
        (x + Inches(0.06), y + Inches(0.68), Inches(0.32), Inches(0.34), b3),
        (x + Inches(0.64), y + Inches(0.08), Inches(0.26), Inches(0.26), b2),
        (x + Inches(0.68), y + Inches(0.64), Inches(0.22), Inches(0.34), b1),
        (x + Inches(0.40), y + Inches(0.70), Inches(0.16), Inches(0.28), b3),
    ]
    for bx, by, bw, bh, bc in blocks:
        _rect(slide, bx, by, bw, bh, fill=bc)
        # Shadow hint
        shadow = RGBColor(max(0, bc[0]-30), max(0, bc[1]-30), max(0, bc[2]-30))
        _rect(slide, bx + bw, by + Inches(0.015), Inches(0.018), bh, fill=shadow)
        _rect(slide, bx + Inches(0.015), by + bh, bw, Inches(0.018), fill=shadow)

    # Vehicles (tiny colored rectangles)
    vehicles = [
        (x + Inches(0.13), ry - Inches(0.065), Inches(0.09), Inches(0.045), RGBColor(0xE0,0x20,0x20)),
        (x + Inches(0.28), ry + Inches(0.030), Inches(0.09), Inches(0.045), RGBColor(0xE8,0xE8,0xE8)),
        (x + Inches(0.42), ry - Inches(0.065), Inches(0.09), Inches(0.045), RGBColor(0x30,0x60,0xE0)),
        (rx + Inches(0.06), y + Inches(0.52), Inches(0.045), Inches(0.09),  RGBColor(0x20,0xC0,0x40)),
        (rx - Inches(0.10), y + Inches(0.68), Inches(0.045), Inches(0.09),  RGBColor(0xE0,0xA0,0x20)),
    ]
    for vx, vy, vw, vh, vc in vehicles:
        _rect(slide, vx, vy, vw, vh, fill=vc)

    # Bounding boxes (cyan) over vehicles
    pad = Inches(0.018)
    for vx, vy, vw, vh, _ in vehicles:
        _rect(slide, vx - pad, vy - pad, vw + pad*2, vh + pad*2,
              fill=None, line_rgb=CYAN, line_w=Pt(1.5))

    # Label tags on first 3
    for i, (vx, vy, vw, vh, _) in enumerate(vehicles[:3]):
        tc = NAVY if i == 0 else NAVY
        _rect(slide, vx - pad, vy - pad - Inches(0.10),
              Inches(0.28), Inches(0.10), fill=CYAN)
        _txt(slide, f"car {0.81 + i*0.05:.2f}",
             vx - pad + Inches(0.01), vy - pad - Inches(0.09),
             Inches(0.26), Inches(0.09), size=6.5, bold=True, color=NAVY,
             align=PP_ALIGN.LEFT)

    # Image border
    _rect(slide, x, y, w, h, fill=None, line_rgb=TEAL, line_w=Pt(1.5))


# ══════════════════════════════════════════════════════════════════════════════
#  SLIDE A — Computer Vision Detection Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def slide_a_pipeline(prs):
    slide = _blank(prs)
    _rect(slide, 0, 0, SW, SH, fill=NAVY)

    # ── Decorative background grid dots ──────────────────────────────────────
    for gx in range(0, 14):
        for gy in range(0, 8):
            _circle(slide, Inches(gx * 0.95), Inches(gy * 0.95),
                    Pt(2.5), RGBColor(0x18, 0x28, 0x40))

    _header(slide, "Computer Vision Pipeline",
            "From raw satellite imagery to ranked demand intelligence -- step by step")
    _footer(slide, 1)

    # ══ Pipeline steps data ════════════════════════════════════════════════════
    steps = [
        {
            "num": "01",
            "name": "ACQUIRE",
            "icon": "[IMG]",
            "color": BLUE, "dark": BLUE_D,
            "params": ["640x640 px", "PNG / JPG", "Synthetic or OSM"],
            "tech": "PIL.Image",
        },
        {
            "num": "02",
            "name": "PREPROCESS",
            "icon": "[PRE]",
            "color": TEAL, "dark": TEAL_D,
            "params": ["4x Upscale", "Contrast x1.3", "Sharpness x1.5"],
            "tech": "PIL.ImageEnhance",
        },
        {
            "num": "03",
            "name": "TILE",
            "icon": "[TIL]",
            "color": ORANGE, "dark": ORANGE_D,
            "params": ["640x640 windows", "Step: 480px", "75% overlap"],
            "tech": "NumPy slicing",
        },
        {
            "num": "04",
            "name": "INFER",
            "icon": "[AI]",
            "color": GOLD, "dark": GOLD_D,
            "params": ["YOLOv8-nano", "conf >= 0.08", "per-tile detection"],
            "tech": "Ultralytics",
        },
        {
            "num": "05",
            "name": "NMS FILTER",
            "icon": "[NMS]",
            "color": RED, "dark": RED_D,
            "params": ["IoU <= 0.40", "Sort by conf", "Dedup bboxes"],
            "tech": "Custom IoU-NMS",
        },
        {
            "num": "06",
            "name": "OUTPUT",
            "icon": "[OUT]",
            "color": GREEN, "dark": GREEN_D,
            "params": ["vehicle_count", "bboxes list", "conf scores"],
            "tech": "DetectionResult",
        },
    ]

    # ── Layout constants ───────────────────────────────────────────────────────
    N        = len(steps)
    TOTAL_W  = SW - Inches(0.5)   # usable width
    ARW      = Inches(0.36)       # arrow width
    BOX_W    = (TOTAL_W - ARW * (N - 1)) / N   # each step box width
    BOX_H    = Inches(2.70)
    BY       = Inches(1.22)       # top of boxes
    BX_START = Inches(0.25)

    for i, step in enumerate(steps):
        bx = BX_START + i * (BOX_W + ARW)
        col   = step["color"]
        dark  = step["dark"]

        # Main card (dark bg + colored top strip)
        _glow_rect(slide, bx, BY, BOX_W, BOX_H, NAVY2, col, col)

        # Glow aura behind the card using precomputed dark color
        glow_c = STEP_GLOW[i]
        _rect(slide, bx - Inches(0.03), BY - Inches(0.03),
              BOX_W + Inches(0.06), BOX_H + Inches(0.06),
              fill=glow_c, line_rgb=col, line_w=Pt(0.4))
        # Re-draw actual card on top
        _glow_rect(slide, bx, BY, BOX_W, BOX_H, NAVY2, col, col)

        # Step number circle
        cx = bx + BOX_W / 2
        cy = BY + Inches(0.44)
        _circle(slide, cx, cy, Inches(0.30), dark)
        _circle(slide, cx, cy, Inches(0.24), col)
        _txt(slide, step["num"], cx - Inches(0.24), cy - Inches(0.18),
             Inches(0.48), Inches(0.36), size=12, bold=True,
             color=NAVY if col == GOLD else WHITE, align=PP_ALIGN.CENTER)

        # Step name
        _txt(slide, step["name"],
             bx + Inches(0.07), BY + Inches(0.82),
             BOX_W - Inches(0.14), Inches(0.36),
             size=12.5, bold=True, color=col, align=PP_ALIGN.CENTER)

        # Parameters (3 lines)
        for pi, param in enumerate(step["params"]):
            py = BY + Inches(1.24 + pi * 0.36)
            # bullet dot
            _circle(slide, bx + Inches(0.18), py + Inches(0.12),
                    Pt(4), col)
            _txt(slide, param,
                 bx + Inches(0.28), py,
                 BOX_W - Inches(0.36), Inches(0.32),
                 size=10, color=OFFWH)

        # Tech badge at bottom of card
        badge_w = BOX_W - Inches(0.22)
        _rect(slide, bx + Inches(0.11), BY + BOX_H - Inches(0.38),
              badge_w, Inches(0.28), fill=dark)
        _txt(slide, step["tech"],
             bx + Inches(0.11), BY + BOX_H - Inches(0.37),
             badge_w, Inches(0.26), size=9, bold=True,
             color=NAVY if col == GOLD else WHITE, align=PP_ALIGN.CENTER)

        # Arrow to next step
        if i < N - 1:
            ax = bx + BOX_W + Inches(0.04)
            ay = BY + BOX_H / 2 - Inches(0.13)
            _arrow_right(slide, ax, ay, ARW - Inches(0.08), Inches(0.26),
                         fill=step["color"])

    # ══ Lower section: two mode panels + aerial mock ═══════════════════════════
    PANEL_Y  = BY + BOX_H + Inches(0.20)
    PANEL_H  = SH - PANEL_Y - Inches(0.35)

    # ── Mode A: Standard single-pass ──────────────────────────────────────────
    _glow_rect(slide, Inches(0.25), PANEL_Y, Inches(4.05), PANEL_H, NAVY3, BLUE, BLUE)
    _txt(slide, "STANDARD MODE",
         Inches(0.35), PANEL_Y + Inches(0.12), Inches(3.85), Inches(0.30),
         size=11, bold=True, color=BLUE, align=PP_ALIGN.LEFT)
    std_lines = [
        ("Single-pass inference on full image",   10, False, OFFWH),
        ("Input:   640 x 640 px",                  9.5, False, LGRAY),
        ("Conf threshold:   0.25",                 9.5, False, LGRAY),
        ("Best for:   synthetic / demo imagery",   9.5, True,  BLUE),
        ("Speed:   <50 ms / image",                9.5, False, GREEN),
    ]
    for li, (txt, sz, bd, col) in enumerate(std_lines):
        _txt(slide, txt, Inches(0.42), PANEL_Y + Inches(0.50 + li * 0.24),
             Inches(3.75), Inches(0.22), size=sz, bold=bd, color=col)

    # ── Mode B: Aerial sliding-window ─────────────────────────────────────────
    _glow_rect(slide, Inches(4.55), PANEL_Y, Inches(5.15), PANEL_H, NAVY3, ORANGE, ORANGE)
    _txt(slide, "AERIAL MODE  (Real Satellite)",
         Inches(4.65), PANEL_Y + Inches(0.12), Inches(4.95), Inches(0.30),
         size=11, bold=True, color=ORANGE, align=PP_ALIGN.LEFT)
    aer_lines = [
        ("4x LANCZOS upscale before tiling",  10, False, OFFWH),
        ("Tile size:   640 x 640 px",          9.5, False, LGRAY),
        ("Overlap:    75%  (step = 480 px)",   9.5, False, LGRAY),
        ("Conf threshold:   0.08  (low)",       9.5, True,  ORANGE),
        ("IoU-NMS:   0.40  deduplication",     9.5, False, LGRAY),
        ("Best for:   Google Earth / Maxar",    9.5, True,  ORANGE),
        ("Speed:   ~1-2 s / 5000x5000 tile",   9.5, False, GREEN),
    ]
    for li, (txt, sz, bd, col) in enumerate(aer_lines):
        _txt(slide, txt, Inches(4.65), PANEL_Y + Inches(0.50 + li * 0.24),
             Inches(4.95), Inches(0.22), size=sz, bold=bd, color=col)

    # ── Live aerial mock image ─────────────────────────────────────────────────
    MOCK_W = Inches(2.82)
    MOCK_H = PANEL_H - Inches(0.08)
    MOCK_X = Inches(10.0)
    MOCK_Y = PANEL_Y + Inches(0.04)

    # Frame / label
    _glow_rect(slide, MOCK_X - Inches(0.12), PANEL_Y, MOCK_W + Inches(0.24),
               PANEL_H, NAVY3, TEAL, TEAL)
    _txt(slide, "YOLO DETECTION OUTPUT",
         MOCK_X - Inches(0.04), PANEL_Y + Inches(0.08),
         MOCK_W + Inches(0.08), Inches(0.22),
         size=8.5, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    _draw_aerial_mock(slide, MOCK_X, PANEL_Y + Inches(0.34), MOCK_W, MOCK_H - Inches(0.38))

    # ── Detected classes legend bar ───────────────────────────────────────────
    # (sits between mode panels, bottom of slide)
    _txt(slide, "DETECTED CLASSES :",
         Inches(4.62), PANEL_Y + PANEL_H - Inches(0.24),
         Inches(1.6), Inches(0.22), size=8, bold=True, color=MGRAY)

    class_badges = [
        ("car  [COCO:2]",     CYAN,   NAVY),
        ("truck  [COCO:7]",   BLUE,   WHITE),
        ("bus  [COCO:5]",     ORANGE, NAVY),
        ("motorcycle  [COCO:3]", RED, WHITE),
        ("building  [custom]", GOLD, NAVY),
    ]
    bx = Inches(6.22)
    for label, bc, tc in class_badges:
        w = _badge(slide, label, bx, PANEL_Y + PANEL_H - Inches(0.26),
                   fill=bc, text_color=tc, size=8.5)
        bx += w + Inches(0.10)

    return slide


# ══════════════════════════════════════════════════════════════════════════════
#  SLIDE B — YOLOv8 Architecture + Intelligent Scoring Engine
# ══════════════════════════════════════════════════════════════════════════════

def slide_b_architecture(prs):
    slide = _blank(prs)
    _rect(slide, 0, 0, SW, SH, fill=NAVY)

    # Decorative background: subtle hexagon-like dots
    for gx in range(0, 14):
        for gy in range(0, 8):
            _circle(slide, Inches(gx * 0.95 + 0.47), Inches(gy * 0.95 + 0.47),
                    Pt(2.5), RGBColor(0x16, 0x26, 0x3E))

    _header(slide, "YOLOv8 Architecture & Intelligent Scoring Engine",
            "Model specs, training dataset, class map, and how detections translate into business intelligence")
    _footer(slide, 2)

    # ══ Three-column layout ════════════════════════════════════════════════════
    COL1_X  = Inches(0.22)
    COL1_W  = Inches(4.18)
    COL2_X  = Inches(4.60)
    COL2_W  = Inches(5.38)
    COL3_X  = Inches(10.18)
    COL3_W  = Inches(2.98)
    COL_Y   = Inches(1.22)
    COL_H   = SH - Inches(1.60)

    # ══════════════════════════════
    #  COLUMN 1 — Model & Dataset
    # ══════════════════════════════
    _glow_rect(slide, COL1_X, COL_Y, COL1_W, COL_H, NAVY2, BLUE, BLUE)

    # Section title
    _txt(slide, "MODEL & DATASET",
         COL1_X + Inches(0.12), COL_Y + Inches(0.10), COL1_W - Inches(0.24), Inches(0.35),
         size=12, bold=True, color=BLUE, align=PP_ALIGN.LEFT)

    _rect(slide, COL1_X + Inches(0.12), COL_Y + Inches(0.48),
          COL1_W - Inches(0.24), Inches(0.02), fill=BLUE)

    # YOLOv8-nano chip
    _rect(slide, COL1_X + Inches(0.12), COL_Y + Inches(0.60),
          COL1_W - Inches(0.24), Inches(0.88), fill=NAVY3,
          line_rgb=BLUE, line_w=Pt(1))
    _txt(slide, "YOLOv8-nano",
         COL1_X + Inches(0.22), COL_Y + Inches(0.64),
         COL1_W - Inches(0.44), Inches(0.32),
         size=16, bold=True, color=BLUE)
    _txt(slide, "3.2M params  |  8.7 GFLOPs  |  <50ms latency",
         COL1_X + Inches(0.22), COL_Y + Inches(0.94),
         COL1_W - Inches(0.44), Inches(0.22),
         size=9, color=LGRAY)
    _txt(slide, "Backbone: CSPDarknet  |  Neck: PAN-FPN",
         COL1_X + Inches(0.22), COL_Y + Inches(1.14),
         COL1_W - Inches(0.44), Inches(0.22),
         size=9, italic=True, color=MGRAY)

    # Weight modes
    _txt(slide, "WEIGHTS",
         COL1_X + Inches(0.12), COL_Y + Inches(1.62),
         COL1_W - Inches(0.24), Inches(0.28), size=9.5, bold=True, color=MGRAY)
    for idx, (label, desc, col) in enumerate([
        ("COCO pre-trained", "yolov8n.pt  |  80 classes", TEAL),
        ("Custom FTTH",      "ftth_best.pt  |  2 classes", ORANGE),
    ]):
        wy = COL_Y + Inches(1.95 + idx * 0.58)
        _rect(slide, COL1_X + Inches(0.12), wy,
              COL1_W - Inches(0.24), Inches(0.50), fill=NAVY3,
              line_rgb=col, line_w=Pt(1.0))
        _rect(slide, COL1_X + Inches(0.12), wy, Inches(0.06), Inches(0.50), fill=col)
        _txt(slide, label, COL1_X + Inches(0.26), wy + Inches(0.04),
             COL1_W - Inches(0.50), Inches(0.22), size=10, bold=True, color=col)
        _txt(slide, desc, COL1_X + Inches(0.26), wy + Inches(0.25),
             COL1_W - Inches(0.50), Inches(0.20), size=8.5, color=LGRAY)

    # Dataset stats
    _txt(slide, "TRAINING DATASET  (Roboflow)",
         COL1_X + Inches(0.12), COL_Y + Inches(3.20),
         COL1_W - Inches(0.24), Inches(0.28), size=9.5, bold=True, color=MGRAY)

    ds_rows = [
        ("Split", "Images", "Labels"),
        ("Train", "31",     "31"),
        ("Valid", "6",      "6"),
        ("Test",  "2",      "2"),
    ]
    for ri, (s, im, lb) in enumerate(ds_rows):
        ry2 = COL_Y + Inches(3.54 + ri * 0.28)
        is_hdr = ri == 0
        bg = NAVY3 if ri % 2 == 0 and not is_hdr else (NAVY if not is_hdr else BLUE_D)
        _rect(slide, COL1_X + Inches(0.12), ry2, COL1_W - Inches(0.24), Inches(0.26), fill=bg)
        col3 = GOLD if is_hdr else WHITE
        for ci, val in enumerate([s, im, lb]):
            _txt(slide, val, COL1_X + Inches(0.18 + ci * 1.28), ry2 + Inches(0.04),
                 Inches(1.20), Inches(0.20), size=9, bold=is_hdr, color=col3)

    # Class map
    _txt(slide, "CLASS MAP",
         COL1_X + Inches(0.12), COL_Y + Inches(4.72),
         COL1_W - Inches(0.24), Inches(0.28), size=9.5, bold=True, color=MGRAY)
    for ci2, (cls_id, label, badge_col) in enumerate([
        ("0", "building", GOLD),
        ("1", "car",      CYAN),
    ]):
        cy2 = COL_Y + Inches(5.04 + ci2 * 0.34)
        _rect(slide, COL1_X + Inches(0.12), cy2,
              Inches(0.30), Inches(0.26), fill=badge_col)
        _txt(slide, cls_id, COL1_X + Inches(0.12), cy2 + Inches(0.03),
             Inches(0.30), Inches(0.22), size=10, bold=True,
             color=NAVY, align=PP_ALIGN.CENTER)
        _txt(slide, label, COL1_X + Inches(0.50), cy2 + Inches(0.04),
             Inches(1.5), Inches(0.22), size=10, bold=True, color=badge_col)
        _txt(slide, "(custom FTTH model)" if cls_id == "0" else "(COCO class 2)",
             COL1_X + Inches(1.32), cy2 + Inches(0.04),
             Inches(2.5), Inches(0.22), size=8.5, italic=True, color=MGRAY)

    # Training config badge
    _rect(slide, COL1_X + Inches(0.12), COL_Y + COL_H - Inches(0.45),
          COL1_W - Inches(0.24), Inches(0.36), fill=BLUE_D)
    _txt(slide, "Fine-tune:  100 epochs  |  Batch 16  |  Img 640  |  AdamW",
         COL1_X + Inches(0.18), COL_Y + COL_H - Inches(0.43),
         COL1_W - Inches(0.36), Inches(0.32),
         size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # ══════════════════════════════
    #  COLUMN 2 — Scoring Engine
    # ══════════════════════════════
    _glow_rect(slide, COL2_X, COL_Y, COL2_W, COL_H, NAVY2, TEAL, TEAL)

    _txt(slide, "INTELLIGENT SCORING ENGINE",
         COL2_X + Inches(0.14), COL_Y + Inches(0.10),
         COL2_W - Inches(0.28), Inches(0.35),
         size=12, bold=True, color=TEAL, align=PP_ALIGN.LEFT)
    _rect(slide, COL2_X + Inches(0.14), COL_Y + Inches(0.48),
          COL2_W - Inches(0.28), Inches(0.02), fill=TEAL)

    # Formula title
    _txt(slide, "Priority Score  =  0.40 x Vehicle Density   +   0.35 x Coverage Gap   +   0.25 x Population",
         COL2_X + Inches(0.14), COL_Y + Inches(0.56),
         COL2_W - Inches(0.28), Inches(0.40),
         size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # ── 3 Weighted factor bars ─────────────────────────────────────────────────
    factors = [
        ("Vehicle Density",  0.40, TEAL,   "vehicle_count / area_km2  /  50  (capped at 1.0)",  "Primary demand signal from YOLO"),
        ("Coverage Gap",     0.35, ORANGE, "(100 - ftth_coverage_pct) / 100",                   "Untapped market opportunity"),
        ("Population Score", 0.25, BLUE,   "population / 100,000  (capped at 1.0)",              "Total addressable household pool"),
    ]
    for fi, (name, weight, col, formula, note) in enumerate(factors):
        fy = COL_Y + Inches(1.10 + fi * 1.32)
        # Factor card background
        _glow_rect(slide, COL2_X + Inches(0.14), fy,
                   COL2_W - Inches(0.28), Inches(1.20), NAVY3, col, col)

        # Weight badge (big)
        _rect(slide, COL2_X + Inches(0.14), fy + Inches(0.065),
              Inches(0.90), Inches(1.07), fill=col)
        _txt(slide, f"{int(weight*100)}%",
             COL2_X + Inches(0.14), fy + Inches(0.22),
             Inches(0.90), Inches(0.50),
             size=28, bold=True,
             color=NAVY if col in (GOLD, TEAL) else WHITE,
             align=PP_ALIGN.CENTER)
        _txt(slide, "weight",
             COL2_X + Inches(0.14), fy + Inches(0.70),
             Inches(0.90), Inches(0.22),
             size=8, color=NAVY if col in (GOLD, TEAL) else WHITE,
             align=PP_ALIGN.CENTER)

        # Factor name
        _txt(slide, name, COL2_X + Inches(1.16), fy + Inches(0.10),
             COL2_W - Inches(1.50), Inches(0.30), size=13, bold=True, color=col)

        # Weight progress bar
        BAR_W = COL2_W - Inches(1.50)
        _rect(slide, COL2_X + Inches(1.16), fy + Inches(0.44),
              BAR_W, Inches(0.12), fill=NAVY)
        _rect(slide, COL2_X + Inches(1.16), fy + Inches(0.44),
              BAR_W * weight, Inches(0.12), fill=col)

        # Formula + note
        _txt(slide, formula, COL2_X + Inches(1.16), fy + Inches(0.62),
             COL2_W - Inches(1.52), Inches(0.24), size=9, color=LGRAY)
        _txt(slide, note, COL2_X + Inches(1.16), fy + Inches(0.84),
             COL2_W - Inches(1.52), Inches(0.24), size=8.5, italic=True, color=MGRAY)

    # ── Tier thresholds ────────────────────────────────────────────────────────
    _txt(slide, "PRIORITY TIERS",
         COL2_X + Inches(0.14), COL_Y + Inches(5.08),
         COL2_W - Inches(0.28), Inches(0.28), size=9.5, bold=True, color=MGRAY)

    tiers = [
        ("HIGH PRIORITY",   "Score >= 68",  GREEN,  "Deploy now  |  ROI < 24 mo"),
        ("MEDIUM PRIORITY", "Score 40-68",  ORANGE, "Pipeline  |  Monitor quarterly"),
        ("LOW PRIORITY",    "Score < 40",   RED,    "Watch list  |  Competitor-heavy"),
    ]
    for ti, (tier, rng, col, action) in enumerate(tiers):
        tx = COL2_X + Inches(0.14 + ti * 1.76)
        ty = COL_Y + Inches(5.42)
        _rect(slide, tx, ty, Inches(1.65), Inches(0.60), fill=NAVY3,
              line_rgb=col, line_w=Pt(1.2))
        _rect(slide, tx, ty, Inches(1.65), Inches(0.10), fill=col)
        _txt(slide, tier, tx + Inches(0.06), ty + Inches(0.14),
             Inches(1.55), Inches(0.20), size=8, bold=True, color=col)
        _txt(slide, rng, tx + Inches(0.06), ty + Inches(0.32),
             Inches(1.55), Inches(0.18), size=9.5, bold=True, color=WHITE)
        _txt(slide, action, tx + Inches(0.06), ty + Inches(0.48),
             Inches(1.55), Inches(0.16), size=7.5, italic=True, color=MGRAY)

    # ══════════════════════════════
    #  COLUMN 3 — Data Flow
    # ══════════════════════════════
    _glow_rect(slide, COL3_X, COL_Y, COL3_W, COL_H, NAVY2, PURPLE, PURPLE)

    _txt(slide, "DATA FLOW",
         COL3_X + Inches(0.12), COL_Y + Inches(0.10),
         COL3_W - Inches(0.24), Inches(0.35),
         size=12, bold=True, color=PURPLE, align=PP_ALIGN.LEFT)
    _rect(slide, COL3_X + Inches(0.12), COL_Y + Inches(0.48),
          COL3_W - Inches(0.24), Inches(0.02), fill=PURPLE)

    flow_nodes = [
        ("Aerial Image",         BLUE,   "[/api/v1/detect]"),
        ("YOLO Inference",       TEAL,   "VehicleDetector"),
        ("vehicle_count",        ORANGE, "per km2"),
        ("ZoneScorer",           GOLD,   "priority_score"),
        ("Tier Assignment",      GREEN,  "High / Med / Low"),
        ("Revenue Estimate",     RED,    "OMR / year"),
        ("REST API + Dashboard", PURPLE, "Streamlit + FastAPI"),
    ]

    node_h = Inches(0.54)
    node_gap = Inches(0.12)
    avail = COL_H - Inches(0.70)
    total_items = len(flow_nodes) * node_h + (len(flow_nodes) - 1) * node_gap
    ny_start = COL_Y + Inches(0.62) + (avail - total_items) / 2

    for ni, (label, col, sub) in enumerate(flow_nodes):
        ny = ny_start + ni * (node_h + node_gap)
        nx = COL3_X + Inches(0.12)
        nw = COL3_W - Inches(0.24)

        _glow_rect(slide, nx, ny, nw, node_h, NAVY3, col, col)

        # Left color bar
        _rect(slide, nx, ny + Inches(0.065), Inches(0.06),
              node_h - Inches(0.065), fill=col)

        _txt(slide, label, nx + Inches(0.14), ny + Inches(0.06),
             nw - Inches(0.20), Inches(0.26), size=10, bold=True, color=col)
        _txt(slide, sub, nx + Inches(0.14), ny + Inches(0.30),
             nw - Inches(0.20), Inches(0.20), size=8.5, italic=True, color=MGRAY)

        # Connector arrow to next (downward)
        if ni < len(flow_nodes) - 1:
            arrow_cx = COL3_X + COL3_W / 2
            ay = ny + node_h + Inches(0.015)
            aw = Inches(0.20)
            _arrow_down(slide, arrow_cx - aw / 2, ay,
                        aw, node_gap - Inches(0.03), fill=col)

    return slide


# ══════════════════════════════════════════════════════════════════════════════
#  Build
# ══════════════════════════════════════════════════════════════════════════════

def build():
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH

    print("Building Slide A: Computer Vision Pipeline")
    slide_a_pipeline(prs)

    print("Building Slide B: YOLOv8 Architecture & Scoring Engine")
    slide_b_architecture(prs)

    out = ROOT / "outputs" / "FALCON_CV_Slides.pptx"
    out.parent.mkdir(exist_ok=True)
    prs.save(str(out))
    print(f"Saved -> {out}")
    return out


if __name__ == "__main__":
    build()
