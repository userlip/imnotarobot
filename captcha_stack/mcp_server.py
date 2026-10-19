"""MCP server: lets agents solve captchas through the «Я не робот» (Ya Ne Robot) app running on this Mac."""

from typing import Literal

import functools

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .client import Client, NotRunning, image_b64

server = MCPServer(
    name="captcha-stack",
    instructions=(
        "Solves captchas through the user's own «Я не робот» app (local model, then vision models, then RuCaptcha workers). "
        "Use solve_image_captcha when you have the captcha picture (a screenshot or a file); use solve_token_captcha when "
        "the page has a reCAPTCHA, hCaptcha or Turnstile widget and you know its site key. If the site rejects an answer, "
        "call report_wrong with the task_id so that layer is not trusted for that picture again."
    ),
)

KINDS = {"text": "ImageToTextTask", "click": "CoordinatesTask", "grid": "GridTask"}
TOKEN_KINDS = {"recaptcha_v2": "RecaptchaV2TaskProxyless", "recaptcha_v3": "RecaptchaV3TaskProxyless",
               "hcaptcha": "HCaptchaTaskProxyless", "turnstile": "TurnstileTaskProxyless"}


def _client() -> Client:
    try:
        return Client()
    except NotRunning as e:
        raise ToolError(str(e)) from e


def explained(fn):
    """Failures reach the agent as readable text (the SDK hides the message of anything but ToolError)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"не получилось: {e}") from e
    return wrapper


def _result(r: dict) -> dict:
    return {"answer": r.get("solution"), "layer": r.get("layer"), "confidence": r.get("confidence"),
            "low_confidence": r.get("lowConfidence"), "task_id": r.get("taskId"), "cost": r.get("cost")}


@server.tool()
@explained
def solve_image_captcha(image: str, kind: Literal["text", "click", "grid"] = "text", instruction: str = "",
                        rows: int = 3, columns: int = 3) -> dict:
    """Solve a captcha picture. `image` is a file path, a data URL or base64. kind: text = type the characters;
    click = points to click (answer.coordinates in image pixels, in order); grid = tile numbers from 1, left to right,
    top to bottom (answer.click). `instruction` is the captcha's task text, e.g. "select all images with buses"."""
    task = {"type": KINDS[kind], "body": image_b64(image)}
    if instruction:
        task["comment"] = instruction
    if kind == "grid":
        task.update(rows=rows, columns=columns)
    return _result(_client().solve(task))


@server.tool()
@explained
def solve_token_captcha(kind: Literal["recaptcha_v2", "recaptcha_v3", "hcaptcha", "turnstile"], page_url: str,
                        site_key: str, action: str = "") -> dict:
    """Get a pass token for a checkbox captcha widget on page_url (solved by RuCaptcha workers, paid). Put the
    returned answer.gRecaptchaResponse or answer.token into the page's response field."""
    task = {"type": TOKEN_KINDS[kind], "websiteURL": page_url, "websiteKey": site_key}
    if action:
        task["pageAction"] = action
    return _result(_client().solve(task, timeout=300))


@server.tool()
@explained
def report_wrong(task_id: int, retry: bool = True) -> dict:
    """Tell the solver that the site rejected this answer; with retry the next layer solves it again."""
    c = _client()
    r = c.report_wrong(task_id, retry)
    if r.get("errorId"):
        raise ToolError(r.get("errorDescription") or r.get("errorCode"))
    return {"reported": True, "retry": _result(c.wait(r["taskId"])) if r.get("taskId") else None}


@server.tool()
@explained
def captcha_status() -> dict:
    """Which layers are ready, today's solved count and spend."""
    c = _client()
    s = c.stats()
    return {"address": c.address, "layers": s.get("layers"), "today": s.get("today")}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
