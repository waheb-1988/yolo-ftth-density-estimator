"""
Ooredoo Falcon — Logo PNG Generator
Renders the logo programmatically using Pillow.
Produces two artefacts:
  outputs/ooredoo_falcon_logo.png      (1280 × 380  — horizontal banner)
  outputs/ooredoo_falcon_icon.png      ( 400 × 400  — square icon / favicon)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    print("Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

from app.config import OUTPUTS_DIR

# ── Brand palette ─────────────────────────────────────────────────────────────
OO_RED       = (225, 10,  10)
OO_RED_DARK  = (139,  0,   0)
OO_RED_GLOW  = (225, 10,  10, 50)
OO_NAVY      = ( 13, 13,  26)
OO_NAVY_2    = ( 30, 16,  50)
OO_WHITE     = (255, 255, 255)
OO_GRAY      = (136, 146, 164)
OO_GRAY_DIM  = ( 60,  68,  88)
TRANSPARENT  = (0, 0, 0, 0)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(len(c1)))


def draw_gradient_rect(img: Image.Image, x0: int, y0: int, x1: int, y1: int,
                       c_top: tuple, c_bot: tuple) -> None:
    """Fill a rectangle with a vertical gradient."""
    draw = ImageDraw.Draw(img)
    h = y1 - y0
    for y in range(y0, y1):
        t = (y - y0) / max(h - 1, 1)
        draw.line([(x0, y), (x1, y)], fill=lerp_color(c_top, c_bot, t))


def draw_diagonal_gradient(img: Image.Image,
                            c_tl: tuple, c_br: tuple) -> None:
    """Diagonal top-left → bottom-right gradient."""
    W, H = img.size
    draw = ImageDraw.Draw(img)
    for y in range(H):
        for x in range(W):
            t = (x / W + y / H) / 2
            draw.point((x, y), fill=lerp_color(c_tl, c_br, t))


def radial_overlay(img: Image.Image, cx: int, cy: int,
                   radius: int, color: tuple, alpha_center: int = 50) -> None:
    """Soft radial glow overlay."""
    W, H = img.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for r in range(radius, 0, -1):
        a = int(alpha_center * (1 - r / radius) ** 2)
        draw.ellipse(
            [(cx - r, cy - r), (cx + r, cy + r)],
            fill=(*color[:3], a)
        )
    img.paste(overlay, mask=overlay)


def draw_glow_circle(draw: ImageDraw.Draw, cx: float, cy: float,
                     r: float, color: tuple, glow_r: float | None = None,
                     center_dot: tuple = OO_WHITE) -> None:
    """Draw a glowing network node."""
    if glow_r is None:
        glow_r = r * 2.5
    # Soft outer ring
    for i in range(int(glow_r), int(r), -1):
        a = int(180 * ((i - r) / (glow_r - r + 1)) ** 2)
        draw.ellipse(
            [(cx - i, cy - i), (cx + i, cy + i)],
            fill=(*color[:3], min(a, 120)),
        )
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=color)
    dot_r = r * 0.42
    draw.ellipse(
        [(cx - dot_r, cy - dot_r), (cx + dot_r, cy + dot_r)],
        fill=center_dot,
    )


def draw_fiber_line(draw: ImageDraw.Draw,
                    x0: float, y0: float, x1: float, y1: float,
                    color: tuple, width: int = 2, alpha: int = 120,
                    dashed: bool = False) -> None:
    """Draw a fiber-optic line (solid or dashed)."""
    c = (*color[:3], alpha)
    if not dashed:
        draw.line([(x0, y0), (x1, y1)], fill=c, width=width)
        return
    # Dashed: split into segments
    length = math.hypot(x1 - x0, y1 - y0)
    steps = max(int(length / 14), 1)
    for i in range(steps):
        t0, t1 = i / steps, (i + 0.45) / steps
        draw.line(
            [(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
             (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)],
            fill=c, width=max(width - 1, 1),
        )


def draw_arc_path(draw: ImageDraw.Draw,
                  cx: float, cy_base: float, rx: float,
                  color: tuple, width: int = 3, alpha: int = 200,
                  open_deg: float = 0, close_deg: float = 180) -> None:
    """Approximate a semi-ellipse arc with many short line segments."""
    n = 64
    c = (*color[:3], alpha)
    pts = []
    for i in range(n + 1):
        t = open_deg + (close_deg - open_deg) * i / n
        rad = math.radians(t)
        pts.append((cx + rx * math.cos(rad), cy_base - (rx * 0.35) * math.sin(rad)))
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=c, width=width)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a system font, fallback to PIL default."""
    candidates = (
        ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf",
         "LiberationSans-Bold.ttf"]
        if bold else
        ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
         "LiberationSans-Regular.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


# ── Falcon icon drawing ───────────────────────────────────────────────────────

def draw_falcon_icon(draw: ImageDraw.Draw,
                     ox: float, oy: float, scale: float = 1.0) -> None:
    """
    Draw the geometric falcon network icon centred at (ox, oy).
    scale = 1.0 → fits inside ~190×200 px bounding box.
    All coordinates are relative to (ox, oy); the hub sits at (0, 0).
    """
    S = scale
    hub = (ox, oy)

    # Key structural points (relative to hub)
    nodes = {
        "hub":       (0,    0),
        "neck":      (0,   -27 * S),
        "head_c":    (22 * S, -57 * S),   # head center
        "lw_tip":    (-80 * S, -45 * S),  # left wing tip
        "rw_tip":    ( 74 * S,  23 * S),  # right wing tip
        "lw_inner":  (-56 * S, -14 * S),
        "rw_inner":  ( 50 * S,   8 * S),
        "tail_base": (0,    57 * S),
        "tail_l":    (-30 * S,  76 * S),
        "tail_r":    ( 30 * S,  76 * S),
        "tail_m":    (0,    79 * S),
    }
    abs_nodes = {k: (ox + v[0], oy + v[1]) for k, v in nodes.items()}
    N = abs_nodes

    # ── Fibre lines (drawn first, background layer) ──────────────────────────
    draw_fiber_line(draw, *N["hub"], *N["lw_tip"],   OO_RED, int(2*S), 130)
    draw_fiber_line(draw, *N["neck"],*N["lw_inner"], OO_RED, int(1*S),  65)
    draw_fiber_line(draw, *N["hub"], *N["rw_tip"],   OO_RED, int(2*S), 130)
    draw_fiber_line(draw, *N["neck"],*N["rw_inner"], OO_RED, int(1*S),  65)
    draw_fiber_line(draw, *N["neck"],*N["tail_base"],OO_RED, int(2*S), 100)
    draw_fiber_line(draw, *N["neck"],*N["head_c"],   OO_RED, int(2*S), 110)
    draw_fiber_line(draw, *N["tail_base"],*N["tail_l"],  OO_RED, int(2*S), 110)
    draw_fiber_line(draw, *N["tail_base"],*N["tail_r"],  OO_RED, int(2*S), 110)
    draw_fiber_line(draw, *N["tail_base"],*N["tail_m"],  OO_RED, int(1*S),  80)

    # Cross-link dashed fibres
    draw_fiber_line(draw, *N["lw_tip"], *N["rw_tip"], OO_RED, 1, 22, dashed=True)
    draw_fiber_line(draw, *N["lw_tip"], *N["tail_l"], OO_RED, 1, 22, dashed=True)
    draw_fiber_line(draw, *N["head_c"],*N["tail_r"],  OO_RED, 1, 22, dashed=True)

    # ── Wing surface fills (semi-transparent polygons) ────────────────────────
    # Left wing
    wing_l = [N["hub"], N["lw_tip"], N["lw_inner"]]
    wing_l_flat = [c for pt in wing_l for c in pt]
    draw.polygon(wing_l_flat, fill=(*OO_RED, 20), outline=(*OO_RED, 88))
    # Right wing
    wing_r = [N["hub"], N["rw_tip"], N["rw_inner"]]
    wing_r_flat = [c for pt in wing_r for c in pt]
    draw.polygon(wing_r_flat, fill=(*OO_RED, 20), outline=(*OO_RED, 88))

    # ── Body outline (ellipse) ────────────────────────────────────────────────
    bx, by   = N["neck"]
    body_rx  = int(32 * S)
    body_ry  = int(50 * S)
    for thickness in range(4, 0, -1):
        a = 60 + thickness * 15
        draw.ellipse(
            [(bx - body_rx - thickness, by - 2 - thickness),
             (bx + body_rx + thickness, by + 2 * body_ry + thickness)],
            fill=(*OO_RED, 6), outline=(*OO_RED, a),
        )

    # ── Head ─────────────────────────────────────────────────────────────────
    hx, hy = N["head_c"]
    hr, hrv = int(18 * S), int(21 * S)
    draw.ellipse([(hx - hr, hy - hrv), (hx + hr, hy + hrv)],
                 fill=OO_NAVY, outline=OO_RED)

    # Crest feathers
    draw_fiber_line(draw, hx - 5*S, hy - hrv + 2,
                    hx - 10*S, hy - hrv - 14*S, OO_RED, 2, 140)
    draw_fiber_line(draw, hx + 2*S, hy - hrv,
                    hx + 2*S,  hy - hrv - 12*S, OO_RED, 1, 100)

    # Beak (hooked triangle)
    bk_pts = [
        (hx + hr, hy - 2*S),
        (hx + hr + 18*S, hy + 4*S),
        (hx + hr,        hy + 9*S),
    ]
    draw.polygon([c for pt in bk_pts for c in pt], fill=OO_RED)

    # Eye
    ex, ey = hx + 4*S, hy - 4*S
    draw.ellipse([(ex - 6*S, ey - 6*S), (ex + 6*S, ey + 6*S)], fill=OO_NAVY)
    draw.ellipse([(ex - 3.5*S, ey - 3.5*S), (ex + 3.5*S, ey + 3.5*S)],
                 fill=OO_WHITE)
    draw.ellipse([(ex - 1.5*S, ey - 1.5*S), (ex + 1.5*S, ey + 1.5*S)],
                 fill=OO_RED)

    # ── Network nodes ─────────────────────────────────────────────────────────
    draw_glow_circle(draw, *N["hub"],      5.5*S, OO_RED)
    draw_glow_circle(draw, *N["neck"],     3.5*S, OO_RED)
    draw_glow_circle(draw, *N["lw_tip"],   4.0*S, OO_RED)
    draw_glow_circle(draw, *N["rw_tip"],   4.0*S, OO_RED)
    draw_glow_circle(draw, *N["tail_base"],3.5*S, OO_RED)
    draw_glow_circle(draw, *N["lw_inner"], 2.5*S, OO_RED,
                     center_dot=(*OO_WHITE, 160))
    draw_glow_circle(draw, *N["rw_inner"], 2.5*S, OO_RED,
                     center_dot=(*OO_WHITE, 160))
    draw_glow_circle(draw, *N["tail_l"],   2.0*S, OO_RED,
                     center_dot=(*OO_WHITE, 100))
    draw_glow_circle(draw, *N["tail_r"],   2.0*S, OO_RED,
                     center_dot=(*OO_WHITE, 100))


# ── Horizontal banner logo ────────────────────────────────────────────────────

def generate_horizontal_logo(out_path: Path,
                              W: int = 1280, H: int = 380) -> None:
    img = Image.new("RGBA", (W, H), OO_NAVY)
    draw = ImageDraw.Draw(img, "RGBA")

    # Diagonal background gradient
    draw_diagonal_gradient(img, OO_NAVY, OO_NAVY_2)

    # Red accent glow (top-right)
    radial_overlay(img, int(W * 0.92), int(H * 0.1), int(H * 0.7),
                   OO_RED, alpha_center=28)
    # Subtle glow bottom-left
    radial_overlay(img, int(W * 0.03), int(H * 0.9), int(H * 0.5),
                   OO_RED, alpha_center=14)

    # Dot-grid overlay
    for gx in range(0, W, 30):
        for gy in range(0, H, 30):
            draw.ellipse([(gx - 1, gy - 1), (gx + 1, gy + 1)],
                         fill=(255, 255, 255, 11))

    # ── Falcon icon (left section) ──
    draw_falcon_icon(draw, ox=210, oy=H // 2, scale=1.7)

    # ── Vertical divider ──
    div_x = 420
    for y in range(H):
        t = abs(y - H / 2) / (H / 2)
        a = int(130 * max(0, 1 - t ** 1.2))
        draw.point((div_x, y), fill=(*OO_RED, a))

    # ── Typography ──
    f_oo   = get_font(44, bold=True)
    f_fal  = get_font(88, bold=True)
    f_tag  = get_font(22, bold=False)

    tx = 448
    draw.text((tx, 86),  "OOREDOO",                      font=f_oo,  fill=OO_RED)
    draw.text((tx - 4, 140), "FALCON",                   font=f_fal, fill=OO_WHITE)
    draw.text((tx, 238), "AI-POWERED FTTH INTELLIGENCE PLATFORM",
              font=f_tag, fill=OO_GRAY)

    # Thin rule under "OOREDOO"
    line_y = 132
    for dx in range(900):
        a = int(90 * (1 - abs(dx - 300) / 600)) if dx > 0 else 0
        draw.point((tx + dx, line_y), fill=(*OO_RED, max(a, 0)))

    # ── FTTH signal arcs (right section) ──
    sig_x, sig_y = W - 145, H // 2 + 60
    for i, (r, w, a) in enumerate([(108, 2, 40), (78, 3, 80),
                                    (50, 3, 130), (24, 4, 200)]):
        draw_arc_path(draw, sig_x, sig_y, r, OO_RED, w, a)

    draw_glow_circle(draw, sig_x, sig_y + 10, 8, OO_RED)
    # Vertical pole
    draw_fiber_line(draw, sig_x, sig_y + 18, sig_x, sig_y + 70, OO_RED, 2, 70)

    f_ftth = get_font(22, bold=True)
    f_oman = get_font(16, bold=False)
    draw.text((sig_x - 26, sig_y + 76), "FTTH", font=f_ftth, fill=OO_RED)
    draw.text((sig_x - 22, sig_y + 102), "OMAN", font=f_oman, fill=OO_GRAY_DIM)

    # ── Rounded border ──
    border_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(border_img)
    for th in range(3, 0, -1):
        bd.rounded_rectangle([(th, th), (W - th, H - th)],
                              radius=20, outline=(*OO_RED, 55 - th * 12))
    img = Image.alpha_composite(img, border_img)

    # Corner accent marks
    draw2 = ImageDraw.Draw(img, "RGBA")
    for (x, y, dx, dy) in [(2, 2, 1, 1), (W - 2, 2, -1, 1),
                            (2, H - 2, 1, -1), (W - 2, H - 2, -1, -1)]:
        draw2.line([(x, y), (x + dx * 30, y)],           fill=(*OO_RED, 200), width=3)
        draw2.line([(x, y), (x, y + dy * 30)],           fill=(*OO_RED, 200), width=3)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "PNG", dpi=(150, 150))
    print(f"[OK] Banner logo saved: {out_path}")


# ── Square icon logo ──────────────────────────────────────────────────────────

def generate_square_icon(out_path: Path, S: int = 400) -> None:
    img = Image.new("RGBA", (S, S), OO_NAVY)
    draw = ImageDraw.Draw(img, "RGBA")

    # Background gradient
    draw_diagonal_gradient(img, OO_NAVY, OO_NAVY_2)
    radial_overlay(img, int(S * 0.85), int(S * 0.1),
                   int(S * 0.55), OO_RED, alpha_center=30)

    # Dot grid
    for gx in range(0, S, 24):
        for gy in range(0, S, 24):
            draw.ellipse([(gx - 1, gy - 1), (gx + 1, gy + 1)],
                         fill=(255, 255, 255, 10))

    # Falcon centred
    draw_falcon_icon(draw, ox=S // 2 - 5, oy=S // 2, scale=1.55)

    # Small "OOREDOO FALCON" label at bottom
    f_sm = get_font(22, bold=True)
    label = "OOREDOO FALCON"
    bb = draw.textbbox((0, 0), label, font=f_sm)
    lw = bb[2] - bb[0]
    draw.text(((S - lw) // 2, S - 42), label, font=f_sm, fill=OO_GRAY_DIM)

    # Border
    border_img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(border_img)
    for th in range(3, 0, -1):
        bd.rounded_rectangle([(th, th), (S - th, S - th)],
                              radius=20, outline=(*OO_RED, 55 - th * 12))
    img = Image.alpha_composite(img, border_img)

    # Corner marks
    draw2 = ImageDraw.Draw(img, "RGBA")
    for (x, y, dx, dy) in [(2, 2, 1, 1), (S - 2, 2, -1, 1),
                            (2, S - 2, 1, -1), (S - 2, S - 2, -1, -1)]:
        draw2.line([(x, y), (x + dx * 24, y)],  fill=(*OO_RED, 200), width=3)
        draw2.line([(x, y), (x, y + dy * 24)],  fill=(*OO_RED, 200), width=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "PNG", dpi=(150, 150))
    print(f"[OK] Square icon saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    banner_path = OUTPUTS_DIR / "ooredoo_falcon_logo.png"
    icon_path   = OUTPUTS_DIR / "ooredoo_falcon_icon.png"

    print("Generating Ooredoo Falcon logos …")
    generate_horizontal_logo(banner_path)
    generate_square_icon(icon_path)
    print("\nDone. Files saved to outputs/")


if __name__ == "__main__":
    main()
