"""Preview renderer - a tiny software 3D "stage" so you can watch the motion
without opening MMD.

Not a game engine: painter's algorithm, a few polygons and thick lines.  It is
enough to judge timing, silhouette and whether the feet skate.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .synth import PoseState

W, H = 720, 720
SS = 2  # supersampling factor
CAM_POS = np.array([0.0, 11.0, 43.0])
CAM_TARGET = np.array([0.0, 9.4, 0.0])
FOV_DEG = 30.0

SKIN = (255, 220, 202)
SKIN_DARK = (226, 186, 170)
HAIR = (64, 214, 202)
HAIR_DARK = (38, 150, 146)
OUTFIT = (44, 48, 84)
OUTFIT_LIGHT = (66, 72, 118)
ACCENT = (255, 74, 138)
BOOT = (28, 30, 46)
WHITE = (245, 246, 255)
DIM = (140, 146, 178)


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
class Camera:
    def __init__(self, pos=CAM_POS, target=CAM_TARGET, fov=FOV_DEG, width=W * SS, height=H * SS):
        self.pos = np.asarray(pos, float)
        f = np.asarray(target, float) - self.pos
        self.f = f / np.linalg.norm(f)
        up = np.array([0.0, 1.0, 0.0])
        r = np.cross(self.f, up)
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)
        self.width = width
        self.height = height
        self.focal = (0.5 * height) / math.tan(math.radians(fov) / 2.0)

    def project(self, p) -> tuple[float, float, float]:
        v = np.asarray(p, float) - self.pos
        z = float(np.dot(v, self.f))
        if z < 0.05:
            z = 0.05
        x = float(np.dot(v, self.r))
        y = float(np.dot(v, self.u))
        return (self.width * 0.5 + self.focal * x / z, self.height * 0.5 - self.focal * y / z, z)

    def scale(self, p, units: float) -> float:
        _, _, z = self.project(p)
        return max(1.0, units * self.focal / z)


# --------------------------------------------------------------------------- #
# background / stage
# --------------------------------------------------------------------------- #
def _background(size) -> Image.Image:
    w, h = size
    top = np.array([10, 11, 22], float)
    mid = np.array([26, 18, 44], float)
    bot = np.array([12, 10, 20], float)
    rows = np.zeros((h, w, 3), np.float32)
    for y in range(h):
        t = y / max(1, h - 1)
        c = top + (mid - top) * (t / 0.55) if t < 0.55 else mid + (bot - mid) * ((t - 0.55) / 0.45)
        rows[y, :, :] = c
    img = Image.fromarray(rows.clip(0, 255).astype(np.uint8), "RGB").convert("RGBA")

    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = w // 2, int(h * 0.44)
    for i in range(9):
        rad = int(w * (0.10 + 0.045 * (8 - i)))
        a = 10 + i * 2
        gd.ellipse([cx - rad, cy - rad * 0.9, cx + rad, cy + rad * 0.9], fill=(120, 90, 220, a))
    glow = glow.filter(ImageFilter.GaussianBlur(w // 40))
    return Image.alpha_composite(img, glow)


def _draw_stage(draw: ImageDraw.ImageDraw, cam: Camera, size) -> None:
    w, h = size
    # floor lines
    for i in range(-8, 9, 2):
        p0 = cam.project((i, 0, -12))
        p1 = cam.project((i, 0, 12))
        col = (70, 62, 120, 90) if i % 4 else (110, 96, 180, 120)
        draw.line([p0[:2], p1[:2]], fill=col, width=2)
    for j in range(-6, 13, 2):
        p0 = cam.project((-12, 0, j))
        p1 = cam.project((12, 0, j))
        col = (70, 62, 120, 90) if j % 4 else (110, 96, 180, 120)
        draw.line([p0[:2], p1[:2]], fill=col, width=2)
    # horizon glow line
    y = int(h * 0.63)
    draw.line([(0, y), (w, y)], fill=(120, 90, 220, 60), width=3)


# --------------------------------------------------------------------------- #
# character
# --------------------------------------------------------------------------- #
def _pt(st: PoseState, name: str) -> np.ndarray:
    return st.pos[name]


def _capsule(d, cam, a, b, ra, rb, color):
    """Thick tapered limb segment with round caps."""
    pa = cam.project(a)
    pb = cam.project(b)
    wa = cam.scale(a, ra)
    wb = cam.scale(b, rb)
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    L = math.hypot(dx, dy) or 1e-6
    nx, ny = -dy / L, dx / L
    poly = [
        (pa[0] + nx * wa * 0.5, pa[1] + ny * wa * 0.5),
        (pb[0] + nx * wb * 0.5, pb[1] + ny * wb * 0.5),
        (pb[0] - nx * wb * 0.5, pb[1] - ny * wb * 0.5),
        (pa[0] - nx * wa * 0.5, pa[1] - ny * wa * 0.5),
    ]
    d.polygon(poly, fill=color)
    for p, w in ((pa, wa), (pb, wb)):
        r = w * 0.5
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)


def _disk(d, cam, p, r_units, color):
    q = cam.project(p)
    r = cam.scale(p, r_units)
    d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=color)


def _poly(d, cam, pts, color):
    d.polygon([cam.project(p)[:2] for p in pts], fill=color)


def draw_character(img: Image.Image, cam: Camera, st: PoseState, ground_shadow: bool = True) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    hip_l, hip_r = _pt(st, "leg.L"), _pt(st, "leg.R")
    knee_l, knee_r = _pt(st, "knee.L"), _pt(st, "knee.R")
    ank_l, ank_r = _pt(st, "ankle.L"), _pt(st, "ankle.R")
    toe_l, toe_r = _pt(st, "toe.L"), _pt(st, "toe.R")
    sh_l, sh_r = _pt(st, "shoulder.L"), _pt(st, "shoulder.R")
    el_l, el_r = _pt(st, "elbow.L"), _pt(st, "elbow.R")
    wr_l, wr_r = _pt(st, "wrist.L"), _pt(st, "wrist.R")
    head = _pt(st, "head")
    chest = _pt(st, "chest")
    neck = _pt(st, "neck")
    root = _pt(st, "root")

    if ground_shadow:
        for a, t in ((ank_l, toe_l), (ank_r, toe_r)):
            cx = (a[0] + t[0]) * 0.5
            cz = (a[2] + t[2]) * 0.5
            height = max(0.0, a[1] - 0.9)
            fade = max(0.12, 1.0 - height / 3.5)
            p = cam.project((cx, 0.02, cz))
            rx = cam.scale((cx, 0, cz), 2.0) * (1.0 + 0.5 * (1.0 - fade))
            ry = rx * 0.34
            d.ellipse([p[0] - rx, p[1] - ry, p[0] + rx, p[1] + ry], fill=(0, 0, 0, int(120 * fade)))

    # ---- legs ----------------------------------------------------------
    legs = [
        ("R", hip_r, knee_r, ank_r, toe_r, SKIN_DARK, SKIN_DARK),
        ("L", hip_l, knee_l, ank_l, toe_l, SKIN, SKIN),
    ]
    for _, hip, knee, ank, toe, skin, _s in legs:
        _capsule(d, cam, hip, knee, 1.25, 0.92, skin)
        _capsule(d, cam, knee, ank, 0.92, 0.62, skin)
        _capsule(d, cam, ank, toe, 0.62, 0.52, BOOT)
        _disk(d, cam, toe, 0.34, BOOT)
        _disk(d, cam, ank, 0.36, BOOT)

    # ---- hips + skirt --------------------------------------------------
    _capsule(d, cam, root + np.array([-0.55, 0.25, 0]), root + np.array([0.55, 0.25, 0]), 1.25, 1.25, SKIN)
    flare = 2.05
    skirt = [
        hip_l * 0.86 + root * 0.14 + np.array([0.28, 0.35, 0]),
        hip_r * 0.86 + root * 0.14 + np.array([-0.28, 0.35, 0]),
        np.array([-flare, root[1] - 2.25, 0.22]),
        np.array([flare, root[1] - 2.25, 0.22]),
    ]
    _poly(d, cam, skirt, OUTFIT_LIGHT + (255,))
    # hem trim
    _capsule(
        d,
        cam,
        np.array([-flare, root[1] - 2.2, 0.23]),
        np.array([flare, root[1] - 2.2, 0.23]),
        0.16,
        0.16,
        ACCENT + (210,),
    )

    # ---- torso ---------------------------------------------------------
    sh_l_v = sh_l + np.array([0.75, 0.22, 0])
    sh_r_v = sh_r + np.array([-0.75, 0.22, 0])
    waist_l = chest * 0.45 + root * 0.55 + np.array([1.22, 0.0, 0])
    waist_r = chest * 0.45 + root * 0.55 + np.array([-1.22, 0.0, 0])
    hip_l_v = hip_l + np.array([0.42, 0.28, 0])
    hip_r_v = hip_r + np.array([-0.42, 0.28, 0])
    torso = [sh_l_v, sh_r_v, waist_r, hip_r_v, hip_l_v, waist_l]
    _poly(d, cam, torso, OUTFIT + (255,))
    _capsule(d, cam, sh_l_v, sh_r_v, 1.05, 1.05, OUTFIT)

    # collar / accent
    c0 = chest * 0.6 + neck * 0.4
    _capsule(d, cam, c0 + np.array([-0.42, 0.05, 0.35]), c0 + np.array([0.42, 0.05, 0.35]), 0.34, 0.34, ACCENT)

    # ---- arms ----------------------------------------------------------
    # The arm bone origin sits inside the torso silhouette, so the upper arm is
    # drawn from a shoulder anchor that rides with the chest instead.
    from .mathx import quat_rotate

    chest_q = st.world["chest"]
    anchors = {
        "L": chest + quat_rotate(chest_q, np.array([1.82, 0.55, 0.0])),
        "R": chest + quat_rotate(chest_q, np.array([-1.82, 0.55, 0.0])),
    }
    _capsule(d, cam, anchors["L"], anchors["R"], 1.15, 1.15, OUTFIT)
    arms = [
        ("R", anchors["R"], el_r, wr_r, SKIN_DARK),
        ("L", anchors["L"], el_l, wr_l, SKIN),
    ]
    arms.sort(key=lambda a: -cam.project(a[3])[2])  # far arm first
    for _, sh, el, wr, skin in arms:
        _capsule(d, cam, sh, el, 0.78, 0.58, skin)
        _capsule(d, cam, el, wr, 0.58, 0.45, skin)
        _disk(d, cam, wr, 0.42, skin)
        _disk(d, cam, el, 0.30, skin)

    # ---- head ----------------------------------------------------------
    _capsule(d, cam, neck + np.array([0, -0.25, 0]), neck + np.array([0, 0.55, 0]), 0.62, 0.58, SKIN)
    hc = head + np.array([0.0, 1.15, 0.0])
    hp = cam.project(hc)
    hr = cam.scale(hc, 1.30)
    hair_r = 1.58

    for s in (-1, 1):  # twintails behind the head
        base = hc + np.array([s * 1.05, 0.25, -0.5])
        mid = hc + np.array([s * 1.72, -1.6, -0.75])
        tip = hc + np.array([s * 1.5, -4.0, -0.55])
        _capsule(d, cam, base, mid, 0.62, 0.50, HAIR_DARK)
        _capsule(d, cam, mid, tip, 0.50, 0.26, HAIR_DARK)
    _disk(d, cam, hc + np.array([0, 0.08, -0.12]), hair_r, HAIR)

    d.ellipse([hp[0] - hr, hp[1] - hr * 1.06, hp[0] + hr, hp[1] + hr * 1.06], fill=SKIN + (255,))
    d.pieslice(
        [hp[0] - hr * 1.03, hp[1] - hr * 1.2, hp[0] + hr * 1.03, hp[1] + hr * 0.62],
        180,
        360,
        fill=HAIR + (255,),
    )
    ey = hp[1] + hr * 0.12
    for s in (-1, 1):
        ex = hp[0] + s * hr * 0.40
        er = max(1.5, hr * 0.15)
        d.ellipse([ex - er, ey - er * 1.5, ex + er, ey + er * 1.5], fill=(24, 190, 182, 255))
        d.ellipse([ex - er * 0.42, ey - er * 1.05, ex + er * 0.42, ey + er * 1.05], fill=(16, 22, 34, 255))
        d.ellipse([ex - er * 0.42, ey - er * 1.05, ex, ey - er * 0.35], fill=(255, 255, 255, 210))
    d.arc(
        [hp[0] - hr * 0.22, ey + hr * 0.30, hp[0] + hr * 0.22, ey + hr * 0.62],
        20,
        160,
        fill=(180, 105, 115, 255),
        width=max(1, int(hr * 0.07)),
    )

    img.alpha_composite(layer)


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\consolab.ttf",
]
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                f = ImageFont.truetype(path, size)
                _font_cache[size] = f
                return f
            except OSError:
                continue
    f = ImageFont.load_default(size)
    _font_cache[size] = f
    return f


def draw_hud(img: Image.Image, info: dict) -> None:
    d = ImageDraw.Draw(img)
    w, h = img.size
    pad = int(w * 0.035)

    d.text((pad, pad), "MMDANCE", font=_font(int(h * 0.030)), fill=(235, 238, 255, 235))
    d.text((pad, pad + int(h * 0.036)), info.get("subtitle", ""), font=_font(int(h * 0.0175)), fill=DIM + (220,))

    sec = info.get("section", "")
    f = _font(int(h * 0.026))
    tw = d.textlength(sec.upper(), font=f)
    d.text((w - pad - tw, pad), sec.upper(), font=f, fill=ACCENT + (240,))

    # beat pips
    beat = info.get("beat", 0.0)
    bar_pos = beat % 4
    pip_r = int(h * 0.009)
    gap = int(h * 0.030)
    total = gap * 3
    x0 = w // 2 - total // 2
    y0 = h - int(h * 0.055)
    for i in range(4):
        cx = x0 + i * gap
        active = int(bar_pos) == i
        col = ACCENT if active else (70, 74, 108)
        d.ellipse([cx - pip_r, y0 - pip_r, cx + pip_r, y0 + pip_r], fill=col + (255,))

    # beat counter + move names
    d.text(
        (pad, h - int(h * 0.075)),
        f"BEAT {beat:5.1f}   BAR {int(beat // 4) + 1:03d}",
        font=_font(int(h * 0.019)),
        fill=DIM + (230,),
    )
    moves = info.get("moves", "")
    if moves:
        d.text((pad, h - int(h * 0.112)), moves, font=_font(int(h * 0.024)), fill=(255, 255, 255, 235))

    # energy bar
    e = float(info.get("energy", 0.0))
    bx = w - pad - int(w * 0.012)
    by0, by1 = int(h * 0.16), int(h * 0.52)
    d.rounded_rectangle([bx, by0, bx + int(w * 0.012), by1], radius=6, fill=(40, 42, 66, 200))
    top = by1 - (by1 - by0) * min(1.0, e)
    d.rounded_rectangle([bx, top, bx + int(w * 0.012), by1], radius=6, fill=ACCENT + (230,))


# --------------------------------------------------------------------------- #
# frame + sequence rendering
# --------------------------------------------------------------------------- #
def render_frame(st: PoseState, info: dict, size=(W, H), ss: int = SS) -> Image.Image:
    w, h = size
    cam = Camera(width=w * ss, height=h * ss)
    img = _background((w * ss, h * ss))
    stage = Image.new("RGBA", img.size, (0, 0, 0, 0))
    _draw_stage(ImageDraw.Draw(stage), cam, img.size)
    img.alpha_composite(stage)
    draw_character(img, cam, st)
    out = img.resize((w, h), Image.LANCZOS)
    draw_hud(out, info)
    return out


def render_sequence(
    states: list[PoseState],
    labels: list[dict],
    out_dir: str | Path,
    size=(W, H),
    step: int = 1,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(0, len(states), step):
        img = render_frame(states[i], labels[i], size=size)
        p = out_dir / f"frame_{i:05d}.png"
        img.save(p)
        paths.append(p)
    return paths


def encode_mp4(frame_dir: str | Path, out_path: str | Path, fps: int = 30, step: int = 1) -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    src = Path(frame_dir) / "frame_%05d.png"
    cmd = [
        ffmpeg, "-v", "error", "-y",
        "-framerate", str(fps),
        "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return str(out_path)


def make_gif(frames: list[Image.Image], out_path: str | Path, fps: int = 15, colors: int = 96, width: int = 480):
    if not frames:
        return None
    resized = []
    for f in frames:
        r = f.convert("RGB").resize((width, int(width * f.height / f.width)), Image.LANCZOS)
        resized.append(r)
    duration = int(1000 / fps)
    pal = resized[0].quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    quant = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in resized]
    quant[0].save(str(out_path), save_all=True, append_images=quant[1:], duration=duration, loop=0, optimize=True)
    return str(out_path)


def contact_sheet(images: list[Image.Image], titles: list[str], out_path: str | Path, cols: int = 3, tile: int = 320):
    rows = (len(images) + cols - 1) // cols
    pad = 10
    W_ = cols * tile + pad * (cols + 1)
    H_ = rows * (tile + 26) + pad * (rows + 1)
    sheet = Image.new("RGB", (W_, H_), (14, 14, 24))
    d = ImageDraw.Draw(sheet)
    for i, (im, t) in enumerate(zip(images, titles)):
        r, c = divmod(i, cols)
        x = pad + c * (tile + pad)
        y = pad + r * (tile + 26 + pad)
        sheet.paste(im.convert("RGB").resize((tile, tile), Image.LANCZOS), (x, y))
        d.text((x + 4, y + tile + 4), t, font=_font(15), fill=(210, 214, 235))
    sheet.save(out_path)
    return str(out_path)
