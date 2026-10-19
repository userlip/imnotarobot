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
import time
from collections import Counter
from typing import Any, Callable

import httpx

from .. import imaging
from ..providers import ModelSpec, ask
from .base import Layer, Result, Unavailable, Unsolvable, decode_image, text_fits
from ..i18n import tr

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


BOX_PROMPT = (
    "The image contains a CAPTCHA. Task: {comment}\n"
    "1. Write the instruction exactly as the CAPTCHA states it (it may be printed in the image).\n"
    "2. Name every target you must click, in the required order.\n"
    "3. Find each target in the image.\n"
    "On the last line give the JSON {{\"targets\": [{{\"label\": \"<what>\", \"box_2d\": [ymin, xmin, ymax, xmax]}}, ...]}} "
    "in click order, with box_2d normalized to 0-1000."
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
    raise Unsolvable(tr("модель ответила не в нужном виде: {text}", text=repr(text[-150:])))


# last failure per model layer, kept across rebuilds of the layer list: (unix time, message)
LAST_ERRORS: dict[str, tuple[float, str]] = {}


class VlmLayer(Layer):
    task_types = frozenset({"ImageToTextTask", "GridTask", "CoordinatesTask"})

    def __init__(self, spec: ModelSpec, transport: httpx.AsyncBaseTransport | None = None):
        self.spec = spec
        self.name = f"vlm:{spec.name}"
        self.title = spec.model or spec.name
        self.min_confidence = spec.min_confidence
        self.transport = transport
        self.upstream = f"{spec.base_url}|{spec.key or spec.name}"

    def available(self) -> bool:
        return self.spec.ready()

    accepts_prior = True

    async def solve(self, task: dict[str, Any], prior: list[dict[str, Any]] | None = None) -> Result:
        """`prior` holds answers of earlier layers that were unsure; with one vote, disagreeing with them earns a second ask."""
        images, prompt, parse, same = self._prepare(task)
        effort = self.spec.effort_clicks if task["type"] == "CoordinatesTask" else self.spec.effort_recognition
        votes = max(1, self.spec.votes)
        answers: list[Any] = []
        errors: list[str] = []
        cost = 0.0
        tokens = {"input": 0, "output": 0, "asks": 0}
        unreachable = 0

        async def one() -> None:
            nonlocal cost, unreachable
            try:
                reply, usage = await ask(self.spec, images, prompt, self.transport, effort)
            except Exception as e:
                errors.append(str(e))
                unreachable += 1
                return
            try:
                cost += self.spec.cost(usage)
                tokens["input"] += usage.get("input", 0)
                tokens["output"] += usage.get("output", 0)
                tokens["asks"] += 1
                answers.append(parse(last_json(reply)))
            except Exception as e:
                errors.append(str(e))

        asked = min(votes, 2)
        await asyncio.gather(*(one() for _ in range(asked)))
        if votes > 2 and not (len(answers) == 2 and same(answers[0], answers[1])):
            await asyncio.gather(*(one() for _ in range(votes - asked)))
            asked = votes
        if not answers and unreachable == asked:
            # one immediate retry; if the service is still silent the chain moves past it
            await one()
            asked += 1
        disputed = bool(votes == 1 and answers and prior
                        and _consensus(task["type"], answers, same)[0] not in prior)
        if disputed:
            await one()
            asked = len(answers)
        if errors:
            LAST_ERRORS[self.name] = (time.time(), errors[-1][:300])
        else:
            LAST_ERRORS.pop(self.name, None)
        if not answers:
            error = Unavailable if unreachable == asked else Unsolvable
            raise error("; ".join(dict.fromkeys(errors)) or tr("нет ответа"))
        final, support = _consensus(task["type"], answers, same)
        if disputed and support < len(answers):
            # the two asks split: hand back the one that matches the earlier layer, the chain accepts such a twin
            final = next((s for s in (_consensus(task["type"], [a], same)[0] for a in answers) if s in prior), final)
            confidence = 0.5
        elif votes == 1:
            confidence = 1.0
        else:
            confidence = support / asked
        return Result(final, confidence, self.name, cost=cost,
                      meta={"model": self.spec.model, "answers": answers, "errors": errors, "tokens": tokens})

    def _prepare(self, task: dict[str, Any]) -> tuple[list[bytes], str, Callable, Callable]:
        t = task["type"]
        original = imaging.load(decode_image(task["body"]))
        big = imaging.fit(imaging.upscale(original)[0])
        extra = [decode_image(task["imgInstructions"])] if task.get("imgInstructions") else []
        note = " The last image shows the instruction." if extra else ""
        comment = task.get("comment") or "follow the instruction shown on the image"
        if t == "ImageToTextTask":
            case = bool(task.get("case"))

            def parse(j):
                text = str(j.get("text", "")).strip()
                if not text_fits(text, task):
                    raise Unsolvable(tr("ответ {text} не подходит под условия задачи", text=repr(text)))
                return text

            return ([imaging.to_png(big)] + extra, TEXT_PROMPT.format(hint=_text_hint(task)) + note, parse,
                     lambda a, b: a == b if case else a.lower() == b.lower())
        if t == "GridTask":
            rows, cols = int(task.get("rows") or 3), int(task.get("columns") or 3)

            def parse(j):
                tiles = sorted({int(x) for x in j.get("tiles", []) if 1 <= int(x) <= rows * cols})
                if not tiles and not task.get("canNoAnswer"):
                    raise Unsolvable(tr("модель не выбрала ни одной клетки"))
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
                raise Unsolvable(tr("модель не указала точек внутри картинки"))
            return pts

        def same(a, b):
            return len(a) == len(b) and all(math.dist(p, q) <= radius for p, q in zip(a, b))

        if self.spec.click_mode == "boxes":
            def parse_boxes(j):
                pts = []
                for t in j.get("targets", []):
                    box = t.get("box_2d") if isinstance(t, dict) else None
                    if not box or len(box) != 4:
                        continue
                    ymin, xmin, ymax, xmax = (float(v) for v in box)
                    pts.append(((xmin + xmax) / 2 / 1000 * w, (ymin + ymax) / 2 / 1000 * h))
                if not pts:
                    raise Unsolvable(tr("модель не обвела ни одной цели"))
                return pts

            return ([imaging.to_png(big)] + extra, BOX_PROMPT.format(comment=comment) + note, parse_boxes, same)
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
