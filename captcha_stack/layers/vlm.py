"""Vision-model layer.

How the task is put to the model:
- small images are upscaled first, so the model's own downscaling does not erase thin characters;
- grids get a red number in every tile (Set-of-Mark prompting) and the model must list each tile before answering;
- click tasks get a second copy of the image with rulers in original pixels, the overlay hcaptcha-challenger uses;
- the answer comes as JSON on the last line, after short reasoning.
Confidence is not the model's self-report (models overrate themselves): the same task is asked several times
and confidence is the share of answers that agree.
"""

import asyncio
import json
import math
import statistics
from collections import Counter
from typing import Any, Callable

import httpx

from .. import imaging
from ..providers import ModelSpec, ask
from .base import Layer, Result, Unsolvable, decode_image, text_fits

TEXT_PROMPT = (
    "The image is a captcha. Read its characters exactly as shown, nothing else. {hint}\n"
    'Reply with only this JSON: {{"text": "<characters>"}}'
)
GRID_PROMPT = (
    "The image is a captcha split into {rows} rows x {columns} columns of tiles. "
    "Each tile has a red number in its top-left corner.\n"
    "Task: {comment}\n"
    "First write one short line per tile: <number>: <what it shows> -> yes/no.\n"
    'Then, on the last line, the JSON {{"tiles": [<numbers of every yes tile>]}}'
)
COORD_PROMPT = (
    "The first image is a captcha. The second is the same picture with rulers: the numbers on the top and left "
    "edges are pixel coordinates of the original image.\n"
    "Task: {comment}\n"
    "Briefly say where each target is, using the ruler numbers. "
    'Then, on the last line, the JSON {{"points": [[x, y], ...]}} with pixel coordinates in click order.'
)


def _text_hint(task: dict[str, Any]) -> str:
    parts = []
    if task.get("comment"):
        parts.append(f"Instruction: {task['comment']}.")
    if task.get("math"):
        parts.append("It is an arithmetic problem: give the result only.")
    numeric = int(task.get("numeric") or 0)
    parts += {1: ["Digits only."], 2: ["Letters only."]}.get(numeric, [])
    if task.get("minLength") or task.get("maxLength"):
        parts.append(f"Length {task.get('minLength') or 1} to {task.get('maxLength') or 'any'}.")
    if task.get("case"):
        parts.append("Letter case matters.")
    return " ".join(parts)


def last_json(text: str) -> dict[str, Any]:
    """The last parseable {...} in the reply: the model may reason before answering."""
    end = text.rfind("}")
    while end != -1:
        start = text.rfind("{", 0, end)
        while start != -1:
            try:
                value = json.loads(text[start:end + 1])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
            start = text.rfind("{", 0, start)
        end = text.rfind("}", 0, end)
    raise Unsolvable(f"модель ответила не в нужном виде: {text[-150:]!r}")


class VlmLayer(Layer):
    task_types = frozenset({"ImageToTextTask", "GridTask", "CoordinatesTask"})

    def __init__(self, spec: ModelSpec, transport: httpx.AsyncBaseTransport | None = None):
        self.spec = spec
        self.name = f"vlm:{spec.name}"
        self.title = spec.model or spec.name
        self.min_confidence = spec.min_confidence
        self.transport = transport

    def available(self) -> bool:
        return self.spec.ready()

    async def solve(self, task: dict[str, Any]) -> Result:
        images, prompt, parse, same = self._prepare(task)
        votes = max(1, self.spec.votes)
        answers: list[Any] = []
        errors: list[str] = []
        cost = 0.0

        async def one() -> None:
            nonlocal cost
            try:
                reply, usage = await ask(self.spec, images, prompt, self.transport)
                cost += self.spec.cost(usage)
                answers.append(parse(last_json(reply)))
            except Exception as e:
                errors.append(str(e))

        asked = min(votes, 2)
        await asyncio.gather(*(one() for _ in range(asked)))
        if votes > 2 and not (len(answers) == 2 and same(answers[0], answers[1])):
            await asyncio.gather(*(one() for _ in range(votes - asked)))
            asked = votes
        if not answers:
            raise Unsolvable("; ".join(dict.fromkeys(errors)) or "нет ответа")
        final, support = _consensus(task["type"], answers, same)
        confidence = 1.0 if votes == 1 else support / asked
        return Result(final, confidence, self.name, cost=cost,
                      meta={"model": self.spec.model, "answers": answers, "errors": errors})

    def _prepare(self, task: dict[str, Any]) -> tuple[list[bytes], str, Callable, Callable]:
        t = task["type"]
        original = imaging.load(decode_image(task["body"]))
        big, _ = imaging.upscale(original)
        extra = [decode_image(task["imgInstructions"])] if task.get("imgInstructions") else []
        note = " The last image shows the instruction." if extra else ""
        comment = task.get("comment") or "follow the instruction shown on the image"
        if t == "ImageToTextTask":
            case = bool(task.get("case"))

            def parse(j):
                text = str(j.get("text", "")).strip()
                if not text_fits(text, task):
                    raise Unsolvable(f"ответ {text!r} не подходит под условия задачи")
                return text

            return ([imaging.to_png(big)] + extra, TEXT_PROMPT.format(hint=_text_hint(task)) + note, parse,
                     lambda a, b: a == b if case else a.lower() == b.lower())
        if t == "GridTask":
            rows, cols = int(task.get("rows") or 3), int(task.get("columns") or 3)

            def parse(j):
                tiles = sorted({int(x) for x in j.get("tiles", []) if 1 <= int(x) <= rows * cols})
                if not tiles and not task.get("canNoAnswer"):
                    raise Unsolvable("модель не выбрала ни одной клетки")
                return tiles

            marked = imaging.grid_marks(big, rows, cols)
            return ([imaging.to_png(marked)] + extra,
                    GRID_PROMPT.format(rows=rows, columns=cols, comment=comment) + note, parse,
                    lambda a, b: a == b)
        w, h = original.size
        radius = max(8.0, 0.04 * math.hypot(w, h))

        def parse(j):
            pts = [(float(x), float(y)) for x, y in j.get("points", []) if 0 <= float(x) <= w and 0 <= float(y) <= h]
            if not pts:
                raise Unsolvable("модель не указала точек внутри картинки")
            return pts

        def same(a, b):
            return len(a) == len(b) and all(math.dist(p, q) <= radius for p, q in zip(a, b))

        overlay = imaging.coordinate_overlay(big, original.size)
        return ([imaging.to_png(big), imaging.to_png(overlay)] + extra, COORD_PROMPT.format(comment=comment) + note,
                parse, same)


def _consensus(task_type: str, answers: list[Any], same: Callable) -> tuple[dict[str, Any], int]:
    """Picks the answer most others agree with; returns it in 2Captcha's solution shape and its support count."""
    if task_type == "ImageToTextTask":
        best = max(answers, key=lambda a: sum(same(a, b) for b in answers))
        return {"text": best}, sum(same(best, b) for b in answers)
    if task_type == "GridTask":
        counts = Counter(t for a in answers for t in a)
        majority = sorted(t for t, c in counts.items() if c * 2 > len(answers))
        best = majority or max(answers, key=lambda a: sum(same(a, b) for b in answers))
        return {"click": best}, sum(same(best, b) for b in answers)
    modal = Counter(len(a) for a in answers).most_common(1)[0][0]
    group = [a for a in answers if len(a) == modal]
    merged = [(statistics.median(p[i][0] for p in group), statistics.median(p[i][1] for p in group))
              for i in range(modal)]
    support = sum(same(merged, a) for a in answers)
    return {"coordinates": [{"x": round(x), "y": round(y)} for x, y in merged]}, support
