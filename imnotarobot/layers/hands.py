"""Hands for the vision model: it looks at a challenge, acts on it, looks again, until the captcha is passed.

One answer per screenshot is not enough for puzzles that change after each move (drag a piece, pick
several objects, press Next). Here the model chooses one action at a time and sees what it did.
Points use Gemini's native pointing format, [y, x] normalized to 0-1000: in our measurements that
was far more accurate than reading pixel rulers drawn on the picture.
"""

import json
import logging
from typing import Any, Awaitable, Callable

from ..providers import ModelSpec, ask
from .vlm import last_json

log = logging.getLogger(__name__)

PROMPT = """You are solving a visual CAPTCHA inside a web page. The image is the current screenshot of the challenge.
Task as shown: {instruction}
{hint}
Actions so far: {history}

Choose ONE next action. Points are [y, x] normalized to 0-1000 over this screenshot.
- {{"action": "click", "point": [y, x]}} — click a target, a tile, or a button such as Verify / Next / Submit
- {{"action": "drag", "from": [y, x], "to": [y, x]}} — drag a piece to its place
- {{"action": "wait"}} — the challenge is still loading or animating
- {{"action": "give_up"}} — the task cannot be solved from what you see
After you have picked every target, click the Verify / Next / Submit button.
Think briefly about what the task requires and where things are, then end with the JSON on the last line."""

Act = Callable[[dict[str, Any], tuple[float, float, float, float]], Awaitable[None]]


class Hands:
    def __init__(self, spec: ModelSpec, transport=None, max_steps: int = 10):
        self.spec = spec
        self.transport = transport
        self.max_steps = max_steps

    async def solve(self, shoot: Callable[[], Awaitable[tuple[bytes, tuple[float, float, float, float]]]],
                    act: Act, passed: Callable[[], Awaitable[bool]], instruction: Callable[[], Awaitable[str]],
                    hint: str = "") -> bool:
        """`shoot` returns a PNG and the page box (x, y, width, height) it covers; `act` performs a decoded action."""
        history: list[str] = []
        for step in range(self.max_steps):
            if await passed():
                return True
            png, box = await shoot()
            reply, _ = await ask(self.spec, [png], PROMPT.format(
                instruction=await instruction(), hint=hint, history="; ".join(history) or "none"),
                self.transport, "low")
            try:
                action = last_json(reply)
            except Exception:
                # a refusal or free text: the model is not going to act, the next step takes over
                log.info("hands step %d: no action, the model answered %r", step + 1, reply[:120])
                return False
            log.info("hands step %d: %s", step + 1, json.dumps(action))
            kind = action.get("action")
            if kind == "give_up":
                return False
            if kind in ("click", "drag"):
                await act(action, box)
                history.append(json.dumps(action))
        return await passed()


def to_page(point: list[float], box: tuple[float, float, float, float]) -> tuple[float, float]:
    """[y, x] in 0-1000 over the screenshot -> page coordinates."""
    y, x = point
    return box[0] + box[2] * x / 1000, box[1] + box[3] * y / 1000
