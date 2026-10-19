"""Client for the running «Я не робот» app: used by the MCP server and by other programs on this Mac."""

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from . import config


class NotRunning(Exception):
    pass


def locate() -> tuple[str, str]:
    """(address, access key) of the solver running on this Mac."""
    runtime = Path(config.APP_DIR) / "runtime.json"
    try:
        info = json.loads(runtime.read_text())
        address = info["address"]
        os.kill(int(info["pid"]), 0)
    except (OSError, ValueError, KeyError) as e:
        raise NotRunning("«Я не робот» не запущено — откройте программу из «Программ»") from e
    cfg = json.loads((Path(config.APP_DIR) / "config.json").read_text())
    key = cfg.get("client_key", "") if cfg.get("require_client_key", True) else ""
    return address, key


class Client:
    def __init__(self, address: str | None = None, key: str | None = None, transport: httpx.BaseTransport | None = None):
        if address is None:
            address, found_key = locate()
            key = found_key if key is None else key
        self.address = address.rstrip("/")
        self.key = key or ""
        self.transport = transport

    def _post(self, method: str, body: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=30, transport=self.transport) as c:
            return c.post(f"{self.address}/{method}", json={"clientKey": self.key, **body}).json()

    def solve(self, task: dict[str, Any], timeout: float = 240) -> dict[str, Any]:
        created = self._post("createTask", {"task": task})
        if created.get("errorId"):
            raise RuntimeError(created.get("errorDescription") or created.get("errorCode"))
        return self.wait(created["taskId"], timeout)

    def wait(self, task_id: int, timeout: float = 240) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = self._post("getTaskResult", {"taskId": task_id})
            if r.get("status") == "ready":
                return {**r, "taskId": task_id}
            if r.get("errorId"):
                raise RuntimeError(r.get("errorDescription") or r.get("errorCode"))
            time.sleep(0.5)
        raise RuntimeError(f"нет ответа за {timeout:.0f} с")

    def report_wrong(self, task_id: int, retry: bool = False) -> dict[str, Any]:
        return self._post("reportIncorrect", {"taskId": task_id, "retry": retry})

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=5, transport=self.transport) as c:
            return c.get(f"{self.address}/health").json()

    def stats(self) -> dict[str, Any]:
        with httpx.Client(timeout=5, transport=self.transport) as c:
            return c.get(f"{self.address}/stats").json()


def image_b64(path_or_b64: str) -> str:
    """Accepts a file path, a data URL or bare base64."""
    p = Path(path_or_b64).expanduser()
    if len(path_or_b64) < 1024 and p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return path_or_b64.split(",", 1)[1] if path_or_b64.startswith("data:") else path_or_b64
