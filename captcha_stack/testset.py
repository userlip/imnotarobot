"""Test captchas with known answers, for comparing models.

A case is {"id", "set", "task", "expected"}; `expected` is {"text"}, {"tiles"} or {"box": [x0, y0, x1, y1]} (any
click inside the box counts). Generated cases are deterministic per seed; downloaded sets live in
<app dir>/testsets/<name>/manifest.json in the same format.
"""

import base64
import io
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from . import config
from .imaging import font

FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
         "/System/Library/Fonts/Supplemental/Georgia.ttf", "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
         "/System/Library/Fonts/Supplemental/Times New Roman.ttf"]
ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
COLORS = {"red": (220, 30, 30), "green": (30, 160, 60), "blue": (40, 80, 220), "orange": (240, 140, 20),
          "purple": (140, 50, 170)}
SHAPES = ["circle", "square", "triangle", "star"]


def _b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _font(rnd: random.Random, size: int):
    from PIL import ImageFont

    for path in rnd.sample(FONTS, len(FONTS)):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return font(size)


def _noise(im: Image.Image, rnd: random.Random, lines: int, dots: int) -> None:
    d = ImageDraw.Draw(im)
    w, h = im.size
    for _ in range(lines):
        d.line([(rnd.randint(0, w), rnd.randint(0, h)) for _ in range(2)],
               fill=tuple(rnd.randint(60, 180) for _ in range(3)), width=rnd.randint(1, 2))
    for _ in range(dots):
        x, y = rnd.randint(0, w - 1), rnd.randint(0, h - 1)
        d.point((x, y), fill=tuple(rnd.randint(0, 255) for _ in range(3)))


def text_case(rnd: random.Random) -> tuple[dict, dict]:
    word = "".join(rnd.choice(ALPHABET) for _ in range(rnd.randint(5, 6)))
    im = Image.new("RGB", (200, 70), tuple(rnd.randint(215, 250) for _ in range(3)))
    x = 10
    for ch in word:
        glyph = Image.new("RGBA", (40, 50), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((4, 2), ch, font=_font(rnd, 34), fill=tuple(rnd.randint(0, 110) for _ in range(3)))
        glyph = glyph.rotate(rnd.uniform(-25, 25), resample=Image.BICUBIC)
        im.paste(glyph, (x, rnd.randint(4, 16)), glyph)
        x += rnd.randint(24, 30)
    _noise(im, rnd, lines=5, dots=300)
    im = im.filter(ImageFilter.GaussianBlur(0.5))
    return {"type": "ImageToTextTask", "body": _b64(im)}, {"text": word}


def math_case(rnd: random.Random) -> tuple[dict, dict]:
    a, b = rnd.randint(2, 30), rnd.randint(2, 20)
    op = rnd.choice(["+", "-", "x"])
    value = {"+": a + b, "-": a - b, "x": a * b}[op]
    im = Image.new("RGB", (200, 60), (245, 245, 240))
    ImageDraw.Draw(im).text((12, 10), f"{a} {op} {b} = ?", font=_font(rnd, 30), fill=(30, 30, 90))
    _noise(im, rnd, lines=3, dots=150)
    return {"type": "ImageToTextTask", "body": _b64(im), "math": True}, {"text": str(value)}


def _draw_shape(d: ImageDraw.ImageDraw, shape: str, cx: float, cy: float, r: float, color) -> None:
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        d.rectangle([cx - r * 0.85, cy - r * 0.85, cx + r * 0.85, cy + r * 0.85], fill=color)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r * 0.8), (cx + r, cy + r * 0.8)], fill=color)
    else:
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else r * 0.45
            a = math.pi / 2 + i * math.pi / 5
            pts.append((cx + rr * math.cos(a), cy - rr * math.sin(a)))
        d.polygon(pts, fill=color)


def click_case(rnd: random.Random) -> tuple[dict, dict]:
    w, h, r = 320, 200, 22
    im = Image.new("RGB", (w, h), (238, 236, 228))
    d = ImageDraw.Draw(im)
    combos = rnd.sample([(s, c) for s in SHAPES for c in COLORS], 6)
    spots: list[tuple[float, float]] = []
    placed = []
    for shape, color in combos:
        for _ in range(100):
            cx, cy = rnd.uniform(r + 4, w - r - 4), rnd.uniform(r + 4, h - r - 4)
            if all(math.dist((cx, cy), p) > 2.4 * r for p in spots):
                break
        spots.append((cx, cy))
        _draw_shape(d, shape, cx, cy, r, COLORS[color])
        placed.append((shape, color, cx, cy))
    shape, color, cx, cy = rnd.choice(placed)
    task = {"type": "CoordinatesTask", "body": _b64(im), "comment": f"click the {color} {shape}"}
    return task, {"box": [cx - r, cy - r, cx + r, cy + r]}


def grid_case(rnd: random.Random) -> tuple[dict, dict]:
    tile, n = 100, 3
    target = rnd.choice(SHAPES)
    im = Image.new("RGB", (tile * n, tile * n), (250, 250, 250))
    d = ImageDraw.Draw(im)
    tiles = set()
    for i in range(n * n):
        shape = target if rnd.random() < 0.35 else rnd.choice([s for s in SHAPES if s != target])
        if shape == target:
            tiles.add(i + 1)
        x0, y0 = (i % n) * tile, (i // n) * tile
        d.rectangle([x0, y0, x0 + tile - 1, y0 + tile - 1], fill=tuple(rnd.randint(200, 245) for _ in range(3)))
        _draw_shape(d, shape, x0 + tile / 2 + rnd.uniform(-12, 12), y0 + tile / 2 + rnd.uniform(-12, 12),
                    rnd.uniform(22, 32), rnd.choice(list(COLORS.values())))
    if not tiles:
        return grid_case(rnd)
    task = {"type": "GridTask", "body": _b64(im), "rows": n, "columns": n,
            "comment": f"select all images with a {target}"}
    return task, {"tiles": sorted(tiles)}


GENERATORS = {"text": text_case, "math": math_case, "click": click_case, "grid": grid_case}


def generated(per_kind: int = 5, seed: int = 7) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    cases = []
    for kind, gen in GENERATORS.items():
        for i in range(per_kind):
            task, expected = gen(rnd)
            cases.append({"id": f"gen-{kind}-{i}", "set": f"сгенерированные: {kind}", "task": task, "expected": expected})
    return cases


VENUS_CLICK_TYPES = {"icon_click", "text_click", "visual_reasoning", "oneclick_login"}


def from_venusbench(root: Path) -> list[dict[str, Any]]:
    """Converts VenusBench-CAPTCHA (inclusionAI, Hugging Face) into cases.

    Its drag tasks (sliders, rotation) need a drag action this solver does not return, so they are left out.
    """
    import re

    items = json.loads((root / "instruction" / "VenusBench-CAPTCHA.json").read_text())
    cases = []
    for i, item in enumerate(items):
        kind = item.get("captcha_type")
        if kind != "OCR" and kind not in VENUS_CLICK_TYPES:
            continue
        image = (root / "instruction" / item["images"][0]).resolve()
        if not image.exists():
            continue
        user = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        # the benchmark's instruction is part of the screenshot itself; the text message is only "<image>"
        instruction = re.sub(r"<image>", "", user if isinstance(user, str) else str(user)).strip()[:600]
        body = base64.b64encode(image.read_bytes()).decode()
        if kind == "OCR":
            m = re.search(r"Type\(content='([^']*)'\)", item["action_raw"])
            if not m:
                continue
            task = {"type": "ImageToTextTask", "body": body,
                    "comment": instruction or "this is a screenshot of a page with a CAPTCHA; type the code shown in the CAPTCHA"}
            expected: dict[str, Any] = {"text": m.group(1)}
        else:
            task = {"type": "CoordinatesTask", "body": body,
                    "comment": instruction or "this is a screenshot of a page with a CAPTCHA; read its instruction in the "
                                              "screenshot and click what it asks, in the order it asks"}
            expected = {"boxes": item["action_raw_rect"], "inorder": bool(item.get("inorder"))}
        cases.append({"id": f"venus-{i}", "set": f"VenusBench: {kind}", "task": task, "expected": expected})
    return cases


VENUS_URL = "https://huggingface.co/datasets/inclusionAI/VenusBench-CAPTCHA"
DOWNLOADABLE = {
    "venusbench": {
        "title": "VenusBench-CAPTCHA", "size_mb": 36, "cases": 129, "url": VENUS_URL,
        "about": "Настоящие капчи с сайтов и приложений: текст, клики по иконкам и словам, задачи на смысл",
        "license": "лицензия не указана — только для личной проверки",
        "files": ["instruction", "images/OCR", "images/icon_click", "images/text_click", "images/visual_reasoning",
                  "images/oneclick_login"],
    },
}


def count_cases(folder: Path) -> int:
    if (folder / "instruction" / "VenusBench-CAPTCHA.json").exists():
        return len(from_venusbench(folder))
    return 0


def download(name: str, dest: Path, progress=None, transport=None) -> int:
    """Fetches a test set file by file from Hugging Face; returns the number of usable cases."""
    import httpx

    info = DOWNLOADABLE[name]
    repo = info["url"].split("huggingface.co/datasets/")[1]
    with httpx.Client(timeout=60, follow_redirects=True, transport=transport) as client:
        files: list[str] = []
        for folder in info["files"]:
            r = client.get(f"https://huggingface.co/api/datasets/{repo}/tree/main/{folder}")
            r.raise_for_status()
            files += [x["path"] for x in r.json() if x.get("type") == "file"]
        partial = dest.with_name(dest.name + ".partial")
        for i, path in enumerate(files, 1):
            target = partial / path
            target.parent.mkdir(parents=True, exist_ok=True)
            r = client.get(f"https://huggingface.co/datasets/{repo}/resolve/main/{path}")
            r.raise_for_status()
            target.write_bytes(r.content)
            if progress:
                progress(i, len(files))
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    partial.rename(dest)
    return count_cases(dest)


def downloaded() -> list[dict[str, Any]]:
    root = Path(config.APP_DIR) / "testsets"
    cases: list[dict[str, Any]] = []
    for manifest in sorted(root.glob("*/manifest.json")):
        cases += json.loads(manifest.read_text())
    venus = root / "venusbench"
    if (venus / "instruction" / "VenusBench-CAPTCHA.json").exists():
        cases += from_venusbench(venus)
    return cases
