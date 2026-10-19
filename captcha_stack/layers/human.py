import asyncio
import time
from typing import Any

import httpx

from .base import Layer, Result, Unsolvable
from ..i18n import tr

ERROR_TEXT = {code: tr(text) for code, text in {
    "ERROR_KEY_DOES_NOT_EXIST": "ключ не подходит",
    "ERROR_WRONG_USER_KEY": "ключ не подходит",
    "ERROR_ZERO_BALANCE": "на балансе нет денег",
    "ERROR_NO_SLOT_AVAILABLE": "сервис перегружен, повторите позже",
    "ERROR_CAPTCHA_UNSOLVABLE": "работники не смогли решить",
    "ERROR_IP_NOT_ALLOWED": "в настройках сервиса запрещён ваш IP",
    "ERROR_IP_BANNED": "ваш IP заблокирован сервисом",
    "ERROR_TOO_BIG_CAPTCHA_FILESIZE": "картинка слишком большая",
    "ERROR_ZERO_CAPTCHA_FILESIZE": "картинка пустая",
    "ERROR_WRONG_FILE_EXTENSION": "неподходящий формат картинки",
    "ERROR_IMAGE_TYPE_NOT_SUPPORTED": "неподходящий формат картинки",
    "ERROR_TASK_NOT_SUPPORTED": "сервис не решает такой вид капчи",
    "ERROR_BAD_PARAMETERS": "сервису не хватило данных о капче",
    "ERROR_PAGEURL": "не указан адрес страницы с капчей",
}.items()}

# Token captchas (reCAPTCHA, hCaptcha, Turnstile…) are only solvable here, so any task type is forwarded as is.
LOCAL_ONLY_TYPES = frozenset({"SlideMatchTask"})


class HumanLayer(Layer):
    """2Captcha / RuCaptcha API v2: workers plus their own AI, the last resort."""

    name = "human"
    title = tr("Люди")

    def __init__(self, base_url: str, api_key: str, timeout: float = 180.0, poll_interval: float = 5.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.transport = transport

    def available(self) -> bool:
        return bool(self.api_key)

    def supports(self, task_type: str) -> bool:
        return task_type not in LOCAL_ONLY_TYPES

    async def _post(self, client: httpx.AsyncClient, method: str, body: dict[str, Any]) -> dict[str, Any]:
        r = await client.post(f"{self.base_url}/{method}", json={"clientKey": self.api_key, **body})
        j = r.json()
        if j.get("errorId"):
            code = j.get("errorCode") or ""
            raise Unsolvable(f"{ERROR_TEXT.get(code, j.get('errorDescription') or tr('ошибка сервиса'))} ({code})")
        return j

    async def solve(self, task: dict[str, Any]) -> Result:
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            created = await self._post(client, "createTask", {"task": task})
            task_id = created["taskId"]
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(self.poll_interval)
                j = await self._post(client, "getTaskResult", {"taskId": task_id})
                if j.get("status") == "ready":
                    return Result(j["solution"], 1.0, self.name, cost=float(j.get("cost") or 0),
                                  meta={"remote_task_id": task_id})
        raise Unsolvable(tr("нет ответа от {url} за {seconds} с", url=self.base_url, seconds=f"{self.timeout:.0f}"))

    async def report_incorrect(self, result: Result) -> None:
        task_id = result.meta.get("remote_task_id")
        if task_id is None:
            return
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            await self._post(client, "reportIncorrect", {"taskId": task_id})

    async def balance(self) -> float:
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            return float((await self._post(client, "getBalance", {})).get("balance", 0))
