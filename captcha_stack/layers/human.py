import asyncio
import time
from typing import Any

import httpx

from .base import Layer, Result, Unsolvable

# Token captchas (reCAPTCHA, hCaptcha, Turnstile…) are only solvable here, so any task type is forwarded as is.
LOCAL_ONLY_TYPES = frozenset({"SlideMatchTask"})


class HumanLayer(Layer):
    """2Captcha / RuCaptcha API v2: workers plus their own AI, the last resort."""

    name = "human"

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
            raise Unsolvable(f"{method}: {j.get('errorCode')} {j.get('errorDescription', '')}".strip())
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
        raise Unsolvable(f"no answer from {self.base_url} in {self.timeout:.0f}s")

    async def report_incorrect(self, result: Result) -> None:
        task_id = result.meta.get("remote_task_id")
        if task_id is None:
            return
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            await self._post(client, "reportIncorrect", {"taskId": task_id})

    async def balance(self) -> float:
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            return float((await self._post(client, "getBalance", {})).get("balance", 0))
