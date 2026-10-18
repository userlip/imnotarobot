import asyncio
import contextlib
import io
from functools import cached_property
from typing import Any

from .base import Layer, Result, Unsolvable, decode_image, text_fits


class LocalLayer(Layer):
    """ddddocr models on this machine: text captchas and slider gaps."""

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
        r = self._slider.slide_match(
            decode_image(task["target"]), decode_image(task["background"]), simple_target=bool(task.get("simpleTarget"))
        )
        return Result({"x": int(r["target_x"]), "y": int(r["target_y"])}, float(r.get("confidence", 0.0)), self.name)
