"""Checkbox captchas (reCAPTCHA v2, hCaptcha, Turnstile) solved in an invisible Chrome.

The page at websiteURL is replaced by a bare page with just the widget, so the site's own scripts and
protections never load while the widget still sees the right domain. Picture grids go to the vision models.
"""

import asyncio
import base64
import html
import io
import logging
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from PIL import Image

from .base import Layer, Result, Unavailable, Unsolvable
from ..i18n import tr

log = logging.getLogger(__name__)

KINDS = {
    "RecaptchaV2TaskProxyless": "recaptcha", "RecaptchaV2Task": "recaptcha",
    "HCaptchaTaskProxyless": "hcaptcha", "HCaptchaTask": "hcaptcha",
    "TurnstileTaskProxyless": "turnstile", "TurnstileTask": "turnstile",
    "RecaptchaV3TaskProxyless": "recaptcha3",
}

# hCaptcha canvas challenges come in families; the matching hint is added to the model's task
CANVAS_HINTS = [
    (("puzzle piece", "complete the image"),
     "The movable pieces are in the panel on the right; the main picture has a missing area. "
     "First point: the piece whose content fills the missing area. Second point: the centre of the missing area."),
    (("scale",),
     "First point: the animal in the right panel that balances the scale. Second point: the pan of the scale that needs it."),
    (("pair",),
     "One pattern on the picture has no partner. First point: the matching tile in the right panel. "
     "Second point: the free end next to the unpaired pattern."),
    (("shadow", "similar", "pattern that match", "matching"),
     "First point: the movable item. Second point: the place whose outline or shape matches it."),
]

CHROME_PATHS = [Path("/Applications/Google Chrome.app"), Path.home() / "Applications/Google Chrome.app"]

TOKEN_FIELD = {
    "recaptcha": "textarea[name=g-recaptcha-response]",
    "hcaptcha": "textarea[name=h-captcha-response]",
    "turnstile": "input[name=cf-turnstile-response]",
}

Solver = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def widget_page(kind: str, task: dict[str, Any]) -> str:
    key = html.escape(str(task.get("websiteKey") or ""), quote=True)
    if not key:
        raise Unsolvable(tr("не указан websiteKey"))
    if kind == "recaptcha":
        invisible = bool(task.get("isInvisible"))
        extra = ' data-size="invisible"' if invisible else ""
        if task.get("recaptchaDataSValue"):
            extra += f' data-s="{html.escape(str(task["recaptchaDataSValue"]), quote=True)}"'
        run = "<script>function go(){grecaptcha.execute()}</script>" if invisible else ""
        onload = "?onload=go" if invisible else ""
        return (f'<!doctype html><html><head><script src="https://www.google.com/recaptcha/api.js{onload}" async defer>'
                f'</script>{run}</head><body><div class="g-recaptcha" data-sitekey="{key}"{extra}></div></body></html>')
    if kind == "recaptcha3":
        return (f'<!doctype html><html><head><script src="https://www.google.com/recaptcha/api.js?render={key}"></script>'
                '</head><body></body></html>')
    if kind == "hcaptcha":
        return ('<!doctype html><html><head><script src="https://js.hcaptcha.com/1/api.js" async defer></script></head>'
                f'<body><div class="h-captcha" data-sitekey="{key}"></div></body></html>')
    attrs = "".join(f' data-{a}="{html.escape(str(task[k]), quote=True)}"'
                    for k, a in (("action", "action"), ("data", "cdata")) if task.get(k))
    return ('<!doctype html><html><head><script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer>'
            f'</script></head><body><div class="cf-turnstile" data-sitekey="{key}"{attrs}></div></body></html>')


def proxy_of(task: dict[str, Any]) -> dict[str, str] | None:
    if not task.get("proxyAddress"):
        return None
    p = {"server": f"{task.get('proxyType', 'http')}://{task['proxyAddress']}:{task.get('proxyPort', '')}"}
    if task.get("proxyLogin"):
        p.update(username=str(task["proxyLogin"]), password=str(task.get("proxyPassword", "")))
    return p


class BrowserLayer(Layer):
    name = "browser"
    title = tr("Браузер")
    task_types = frozenset(KINDS)

    def __init__(self, models: Solver | None, timeout: float = 150.0, rounds: int = 8, parallel: int = 2):
        self.models = models
        self.timeout = timeout
        self.rounds = rounds
        # each Chrome takes a few hundred MB
        self._slots = asyncio.Semaphore(parallel)

    def available(self) -> bool:
        return any(p.exists() for p in CHROME_PATHS)

    async def solve(self, task: dict[str, Any]) -> Result:
        kind = KINDS[task["type"]]
        url = str(task.get("websiteURL") or "").split("#")[0]
        if not url.startswith("http"):
            raise Unsolvable(tr("не указан websiteURL"))
        page_html = widget_page(kind, task)
        try:
            from patchright.async_api import async_playwright
        except ImportError as e:
            raise Unavailable(tr("в сборке нет браузерного модуля")) from e
        async with self._slots:
            try:
                async with async_playwright() as pw:
                    browser = await pw.chromium.launch(channel="chrome", headless=True, proxy=proxy_of(task))
                    try:
                        ctx = await browser.new_context(locale="en-US", user_agent=task.get("userAgent") or None)
                        page = await ctx.new_page()
                        await page.route(lambda u: u.split("#")[0] == url,
                                         lambda r: r.fulfill(status=200, content_type="text/html", body=page_html))
                        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        token = await asyncio.wait_for(self._run(page, kind, task), self.timeout)
                        agent = await page.evaluate("() => navigator.userAgent")
                    finally:
                        await browser.close()
            except asyncio.TimeoutError:
                raise Unsolvable(tr("браузер не получил ответ за {seconds} с", seconds=f"{self.timeout:.0f}")) from None
            except Unsolvable:
                raise
            except Exception as e:
                raise Unsolvable(tr("браузер: {error}", error=str(e).splitlines()[0][:200])) from e
        solution = {"token": token, "userAgent": agent}
        if kind != "turnstile":
            solution["gRecaptchaResponse"] = token
        if kind == "hcaptcha":
            solution["respKey"] = ""
        return Result(solution, 1.0, self.name)

    async def _token(self, page, kind: str) -> str:
        return await page.evaluate(
            "s => [...document.querySelectorAll(s)].map(e => e.value).find(v => v) || ''", TOKEN_FIELD[kind])

    async def _run(self, page, kind: str, task: dict[str, Any]) -> str:
        if kind == "recaptcha3":
            # the patched browser evaluates in an isolated world by default, where the page's grecaptcha is not visible
            for _ in range(60):
                if await page.evaluate("() => !!(window.grecaptcha && grecaptcha.execute)", isolated_context=False):
                    break
                await asyncio.sleep(0.5)
            return await page.evaluate(
                "([k, a]) => new Promise((ok, no) => grecaptcha.ready(() => grecaptcha.execute(k, {action: a}).then(ok, no)))",
                [task["websiteKey"], task.get("pageAction") or "verify"], isolated_context=False)
        if kind == "recaptcha" and not task.get("isInvisible"):
            await page.frame_locator('iframe[src*="/anchor"]').locator("#recaptcha-anchor").click(timeout=30000)
        elif kind == "hcaptcha":
            await page.frame_locator('iframe[src*="frame=checkbox"]').locator("#checkbox").click(timeout=30000)
        clicked_turnstile = False
        started = time.monotonic()
        rounds = 0
        while True:
            await asyncio.sleep(1)
            token = await self._token(page, kind)
            if token:
                return token
            if kind == "turnstile":
                if not clicked_turnstile and time.monotonic() - started > 4:
                    clicked_turnstile = True
                    frame = page.frame_locator('iframe[src*="challenges.cloudflare.com"]')
                    try:
                        await frame.locator("input[type=checkbox]").click(timeout=5000)
                    except Exception:
                        log.info("turnstile: no checkbox to click")
                continue
            if rounds >= self.rounds:
                raise Unsolvable(tr("не прошёл за {rounds} заданий с картинками", rounds=self.rounds))
            if kind == "recaptcha" and await self._recaptcha_round(page):
                rounds += 1
            elif kind == "hcaptcha" and await self._hcaptcha_round(page):
                rounds += 1

    async def _ask(self, image: bytes, comment: str, rows: int, columns: int) -> list[int]:
        if self.models is None:
            raise Unsolvable(tr("нет моделей для картинок"))
        task = {"type": "GridTask", "body": base64.b64encode(image).decode(), "comment": comment,
                "rows": rows, "columns": columns, "canNoAnswer": True}
        return list((await self.models(task)).get("click") or [])

    async def _recaptcha_round(self, page) -> bool:
        """Solves one picture challenge if it is showing; False when there is nothing to do yet."""
        frame_el = page.locator('iframe[src*="/bframe"]')
        if not await frame_el.count() or not await frame_el.first.is_visible():
            return False
        bf = page.frame_locator('iframe[src*="/bframe"]')
        if await bf.locator(".rc-doscaptcha-header").count():
            raise Unsolvable(tr("Google считает запросы автоматическими и не показывает задание"))
        table = bf.locator("#rc-imageselect-target table")
        if not await table.count():
            return False
        desc = bf.locator(".rc-imageselect-desc-wrapper").first
        comment = " ".join((await desc.inner_text()).split())
        size = 4 if "44" in (await table.first.get_attribute("class") or "") else 3
        dynamic = "none left" in comment.lower()
        for _ in range(6):
            tiles = await self._ask(await table.first.screenshot(), comment, size, size)
            cells = bf.locator("#rc-imageselect-target td")
            for t in tiles:
                await cells.nth(t - 1).click()
                await asyncio.sleep(0.3)
            if not dynamic or not tiles:
                break
            # replaced tiles fade in over a few seconds
            await asyncio.sleep(4.5)
        log.info("recaptcha round: %s, %dx%d", comment, size, size)
        await bf.locator("#recaptcha-verify-button").click()
        await asyncio.sleep(2)
        return True

    async def _hcaptcha_round(self, page) -> bool:
        frame_el = page.locator('iframe[src*="frame=challenge"]')
        if not await frame_el.count() or not await frame_el.first.is_visible():
            return False
        ch = page.frame_locator('iframe[src*="frame=challenge"]')
        cells, canvas = ch.locator(".task-grid .task"), ch.locator("canvas")
        for _ in range(16):
            if await cells.count() == 9 or await canvas.count():
                break
            await asyncio.sleep(0.5)
        else:
            return False
        comment = " ".join((await ch.locator(".prompt-text").first.inner_text()).split())
        log.debug("hcaptcha challenge: %s", comment)
        if await cells.count() == 9:
            tiles = await self._ask(await ch.locator(".task-grid").first.screenshot(), comment, 3, 3)
            for t in tiles:
                await cells.nth(t - 1).click()
                await asyncio.sleep(0.2)
            log.info("hcaptcha grid: %s -> %s", comment, tiles)
        else:
            await self._canvas_round(page, canvas.first, comment)
        await ch.locator(".button-submit").click()
        await asyncio.sleep(2)
        return True

    async def _canvas_round(self, page, canvas, comment: str) -> None:
        """Click and drag challenges drawn on a canvas: the models point, the browser clicks or drags."""
        await asyncio.sleep(1)
        box = await canvas.bounding_box()
        shot = await canvas.screenshot()
        low = comment.lower()
        drag = "drag" in low
        hint = comment
        if drag:
            family = next((h for keys, h in CANVAS_HINTS if any(k in low for k in keys)),
                          "First point: the centre of the item to move. Second point: where to drop it.")
            hint = f"{comment}. {family} Give exactly two targets."
        task = {"type": "CoordinatesTask", "body": base64.b64encode(shot).decode(), "comment": hint}
        if self.models is None:
            raise Unsolvable(tr("нет моделей для картинок"))
        points = (await self.models(task)).get("coordinates") or []
        # the screenshot may be at device scale; convert back to CSS pixels
        k = box["width"] / Image.open(io.BytesIO(shot)).width
        xy = [(box["x"] + p["x"] * k, box["y"] + p["y"] * k) for p in points]
        log.info("hcaptcha canvas: %s -> %s", comment, points)
        if drag:
            if len(xy) < 2:
                raise Unsolvable(tr("модель не указала, что и куда тащить"))
            await drag_like_a_hand(page, xy[0], xy[-1])
        else:
            for x, y in xy:
                await page.mouse.click(x, y)
                await asyncio.sleep(0.3)


async def drag_like_a_hand(page, start: tuple[float, float], end: tuple[float, float], steps: int = 28) -> None:
    """Eased curve with a slight arc; canvas challenges reject straight constant-speed drags."""
    (x0, y0), (x1, y1) = start, end
    cx, cy = (x0 + x1) / 2 + random.uniform(-30, 30), (y0 + y1) / 2 + random.uniform(-30, 30)
    await page.mouse.move(x0, y0)
    await asyncio.sleep(random.uniform(0.1, 0.25))
    await page.mouse.down()
    for i in range(1, steps + 1):
        t = i / steps
        t = t * t * (3 - 2 * t)
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1
        await page.mouse.move(x, y)
        await asyncio.sleep(random.uniform(0.008, 0.025))
    await page.mouse.move(x1, y1)
    await asyncio.sleep(random.uniform(0.1, 0.2))
    await page.mouse.up()


async def ask_models(layers: list[Layer], task: dict[str, Any]) -> dict[str, Any]:
    """Asks the vision layers in order and returns the first answer; one silent service skips its other models."""
    errors, silent = [], set()
    for layer in layers:
        if not layer.available() or (layer.upstream and layer.upstream in silent):
            continue
        try:
            return (await layer.solve(task)).solution
        except Unavailable as e:
            silent.add(layer.upstream)
            errors.append(str(e))
        except Exception as e:
            errors.append(str(e))
    raise Unsolvable("; ".join(errors) or tr("нет моделей для картинок"))
