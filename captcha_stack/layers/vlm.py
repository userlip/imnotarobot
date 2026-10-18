import base64
import io
import json
import re
from typing import Any

import httpx
from PIL import Image

from .base import Layer, Result, Unsolvable, decode_image, image_size, text_fits

ANSWER_RULES = (
    "Reply with one JSON object and nothing else. "
    'Always include "confidence": a number from 0 to 1 for how sure you are.'
)

PROMPTS = {
    "ImageToTextTask": (
        "This image is a captcha. Read the characters exactly as shown. {hint}"
        'JSON: {{"text": "<characters>", "confidence": <0..1>}}'
    ),
    "GridTask": (
        "This captcha image is a grid of {rows} rows and {columns} columns. Task: {comment}. "
        "Number tiles left to right, top to bottom, starting at 1. List every tile that matches. "
        'JSON: {{"tiles": [<numbers>], "confidence": <0..1>}}'
    ),
    "CoordinatesTask": (
        "Captcha task: {comment}. Give the point(s) to click, in order. "
        "Use coordinates normalized to 0..1000, where 0,0 is the top-left corner and 1000,1000 the bottom-right. "
        'JSON: {{"points": [[x, y], ...], "confidence": <0..1>}}'
    ),
}


def _text_hint(task: dict[str, Any]) -> str:
    parts = []
    if task.get("comment"):
        parts.append(f"Instruction: {task['comment']}.")
    if task.get("math"):
        parts.append("It is an arithmetic problem: answer with the result only.")
    numeric = int(task.get("numeric") or 0)
    if numeric == 1:
        parts.append("Only digits.")
    elif numeric == 2:
        parts.append("Only letters.")
    if task.get("minLength") or task.get("maxLength"):
        parts.append(f"Length {task.get('minLength') or 1} to {task.get('maxLength') or 'any'}.")
    return " ".join(parts) + (" " if parts else "")


def parse_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise Unsolvable(f"model did not return JSON: {text[:200]!r}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise Unsolvable(f"model returned broken JSON: {e}") from e


class VlmLayer(Layer):
    """Any OpenAI-compatible vision model (Qwen-VL on DashScope by default) looking at the captcha image."""

    name = "vlm"
    task_types = frozenset(PROMPTS)

    def __init__(self, base_url: str, api_key: str, model: str, min_confidence: float = 0.7,
                 timeout: float = 60.0, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.min_confidence = min_confidence
        self.timeout = timeout
        self.transport = transport

    def available(self) -> bool:
        return bool(self.api_key)

    async def solve(self, task: dict[str, Any]) -> Result:
        image = decode_image(task["body"])
        t = task["type"]
        prompt = PROMPTS[t].format(
            hint=_text_hint(task), comment=task.get("comment") or "follow the instruction on the image",
            rows=task.get("rows", 3), columns=task.get("columns", 3),
        )
        content = [{"type": "image_url", "image_url": {"url": _data_url(image)}}]
        if task.get("imgInstructions"):
            content.append({"type": "image_url", "image_url": {"url": _data_url(decode_image(task["imgInstructions"]))}})
            prompt += " The second image shows the instruction."
        content.append({"type": "text", "text": prompt + " " + ANSWER_RULES})
        reply, usage = await self._chat(content)
        data = parse_json(reply)
        conf = float(data.get("confidence", 0.0))
        meta = {"model": self.model, "tokens": usage.get("total_tokens")}
        if t == "ImageToTextTask":
            text = str(data.get("text", "")).strip()
            if not text_fits(text, task):
                raise Unsolvable(f"answer {text!r} breaks the task constraints")
            return Result({"text": text}, conf, self.name, meta=meta)
        if t == "GridTask":
            total = int(task.get("rows", 3)) * int(task.get("columns", 3))
            tiles = sorted({int(x) for x in data.get("tiles", []) if 1 <= int(x) <= total})
            if not tiles and not task.get("canNoAnswer"):
                raise Unsolvable("model selected no tiles")
            return Result({"click": tiles}, conf, self.name, meta=meta)
        width, height = image_size(image)
        points = [
            {"x": round(float(x) / 1000 * width), "y": round(float(y) / 1000 * height)}
            for x, y in data.get("points", [])
            if 0 <= float(x) <= 1000 and 0 <= float(y) <= 1000
        ]
        if not points:
            raise Unsolvable("model returned no points")
        return Result({"coordinates": points}, conf, self.name, meta=meta)

    async def _chat(self, content: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        body = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": content}]}
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code != 200:
            raise Unsolvable(f"vision API HTTP {r.status_code}: {r.text[:200]}")
        j = r.json()
        return j["choices"][0]["message"]["content"], j.get("usage", {})


def _data_url(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as im:
        fmt = (im.format or "PNG").lower()
    return f"data:image/{fmt};base64,{base64.b64encode(data).decode()}"
