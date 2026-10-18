import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]


def font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def to_png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def upscale(im: Image.Image, min_side: int = 400) -> tuple[Image.Image, int]:
    """Small captchas lose detail in the model's own resize; returns the image and the integer factor used."""
    factor = max(1, math.ceil(min_side / max(im.size)))
    if factor == 1:
        return im, 1
    return im.resize((im.width * factor, im.height * factor), Image.LANCZOS), factor


def grid_marks(im: Image.Image, rows: int, columns: int) -> Image.Image:
    """Set-of-Mark style: draws tile borders and a number in each tile so the model answers with labels it can see."""
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    tw, th = im.width / columns, im.height / rows
    f = font(max(12, int(min(tw, th) / 5)))
    for i in range(1, columns):
        d.line([(i * tw, 0), (i * tw, im.height)], fill=(255, 0, 0), width=2)
    for j in range(1, rows):
        d.line([(0, j * th), (im.width, j * th)], fill=(255, 0, 0), width=2)
    for j in range(rows):
        for i in range(columns):
            label = str(j * columns + i + 1)
            x, y = i * tw + 4, j * th + 4
            box = d.textbbox((x, y), label, font=f)
            d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(255, 255, 255))
            d.text((x, y), label, fill=(200, 0, 0), font=f)
    return im


def coordinate_overlay(im: Image.Image, original_size: tuple[int, int], lines: int = 10) -> Image.Image:
    """Adds rulers labelled in the ORIGINAL image's pixels (even if `im` was upscaled) plus light grid lines."""
    im = im.convert("RGB")
    ow, oh = original_size
    sx, sy = im.width / ow, im.height / oh
    margin = 44
    out = Image.new("RGB", (im.width + margin, im.height + margin), (255, 255, 255))
    out.paste(im, (margin, margin))
    d = ImageDraw.Draw(out)
    f = font(14)
    step_x = max(1, round(ow / lines))
    step_y = max(1, round(oh / lines))
    for v in range(0, ow + 1, step_x):
        x = margin + v * sx
        d.line([(x, margin), (x, out.height)], fill=(0, 160, 255), width=1)
        d.text((x - 10, 4), str(v), fill=(0, 0, 0), font=f)
    for v in range(0, oh + 1, step_y):
        y = margin + v * sy
        d.line([(margin, y), (out.width, y)], fill=(0, 160, 255), width=1)
        d.text((2, y - 8), str(v), fill=(0, 0, 0), font=f)
    return out


def _corr(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Valid-mode cross-correlation via FFT."""
    H, W = a.shape
    h, w = k.shape
    fh, fw = H + h - 1, W + w - 1
    full = np.fft.irfft2(np.fft.rfft2(a, (fh, fw)) * np.fft.rfft2(k[::-1, ::-1], (fh, fw)), (fh, fw))
    return full[h - 1:H, w - 1:W]


def slide_ncc(piece: Image.Image, background: Image.Image) -> tuple[int, int, float]:
    """Finds the gap for a slider piece by masked normalized cross-correlation.

    Normalizing per window makes it insensitive to the darkening or lightening that marks the gap.
    Returns the gap's top-left corner and the correlation peak (1.0 = perfect match).
    """
    p = np.asarray(piece.convert("RGBA"), dtype=np.float64)
    bg = np.asarray(background.convert("L"), dtype=np.float64)
    if p.shape[0] > bg.shape[0] or p.shape[1] > bg.shape[1]:
        raise ValueError("piece is larger than the background")
    mask = (p[..., 3] > 128).astype(np.float64)
    n = mask.sum()
    if n < 16:
        mask, n = np.ones(mask.shape), float(mask.size)
    gray = p[..., :3] @ np.array([0.299, 0.587, 0.114])
    pz = (gray - (gray * mask).sum() / n) * mask
    pnorm = np.sqrt((pz ** 2).sum()) or 1e-9
    s1, s2 = _corr(bg, mask), _corr(bg ** 2, mask)
    ncc = _corr(bg, pz) / (np.sqrt(np.maximum(s2 - s1 ** 2 / n, 1e-9)) * pnorm)
    y, x = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    return int(x), int(y), float(ncc[y, x])
