import asyncio
import contextlib
import io
from functools import cached_property
from typing import Any

from .. import imaging
from .base import Layer, Result, Unsolvable, decode_image, text_fits

# below this correlation peak the gap match is treated as a guess
NCC_TRUSTED = 0.8


class LocalLayer(Layer):
    """Models on this machine: ddddocr for text captchas, correlation matching for slider gaps."""

    name = "local"
    task_types = frozenset({"ImageToTextTask", "SlideMatchTask"})

    def __init__(self, min_confidence: float = 0.9):
        self.min_confidence = min_confidence

    @cached_property
    def _ocr(self):
        import ddddocr

        with contextlib.redirect_stdout(io.StringIO()):
            return ddddocr.DdddOcr(show_ad=False)

    @cached_property
    def _slider(self):
        import ddddocr

        with contextlib.redirect_stdout(io.StringIO()):
            return ddddocr.DdddOcr(det=False, ocr=False, show_ad=False)

    async def solve(self, task: dict[str, Any]) -> Result:
        return await asyncio.to_thread(self._solve_sync, task)

    def _solve_sync(self, task: dict[str, Any]) -> Result:
        if task["type"] == "ImageToTextTask":
            r = self._ocr.classification(decode_image(task["body"]), probability=True)
            text = r["text"] if task.get("case") else r["text"].lower()
            if not text_fits(text, task):
                raise Unsolvable(f"answer {text!r} breaks the task constraints")
            return Result({"text": text}, float(r["confidence"]), self.name)
        piece_bytes, bg_bytes = decode_image(task["target"]), decode_image(task["background"])
        piece, background = imaging.load(piece_bytes), imaging.load(bg_bytes)
        x, y, score = imaging.slide_ncc(piece, background)
        method = "ncc"
        if score < NCC_TRUSTED:
            # ddddocr's edge matcher reports the piece centre, not its corner
            r = self._slider.slide_match(piece_bytes, bg_bytes)
            x, y = int(r["target_x"] - piece.width / 2), int(r["target_y"] - piece.height / 2)
            score, method = min(score, 0.5), "ddddocr-edges"
        solution = {"x": x, "y": y, "centerX": x + piece.width // 2, "centerY": y + piece.height // 2}
        return Result(solution, score, self.name, meta={"method": method})
