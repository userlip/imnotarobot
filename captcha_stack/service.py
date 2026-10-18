import asyncio
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .chain import Chain, State, today
from .config import Config
from .layers import HumanLayer, Layer, LocalLayer, Result, VlmLayer


def build_layers(cfg: Config) -> list[Layer]:
    out: list[Layer] = []
    for name in cfg.layers:
        if name == "local":
            out.append(LocalLayer(cfg.ocr_min_confidence))
        elif name == "vlm":
            out += [VlmLayer(m) for m in cfg.models]
        elif name.startswith("vlm:"):
            out += [VlmLayer(m) for m in cfg.models if m.name == name[4:]]
        elif name == "human":
            out.append(HumanLayer(cfg.human_base_url, cfg.human_api_key, cfg.human_timeout, cfg.human_poll_interval))
    return out


@dataclass
class Job:
    task: dict[str, Any]
    created: float = field(default_factory=time.monotonic)
    result: Result | None = None
    error: str = ""
    done: asyncio.Event = field(default_factory=asyncio.Event)


class Service:
    """Task bookkeeping shared by the HTTP APIs and the menu bar app."""

    def __init__(self, cfg: Config, layers: list[Layer] | None = None):
        self.cfg = cfg
        self.chain = Chain(layers if layers is not None else build_layers(cfg),
                           State(cfg.state_path, cfg.skip_after_reports), {"human": cfg.human_daily_limit_usd})
        self.jobs: dict[int, Job] = {}
        self._ids = itertools.count(int(time.time()))
        self._running: set[asyncio.Task] = set()
        self.paused = False
        self.recent: deque[dict[str, Any]] = deque(maxlen=100)

    def reconfigure(self, cfg: Config) -> None:
        """Swaps layers and limits; statistics and open tasks are kept."""
        self.cfg = cfg
        self.chain.layers = build_layers(cfg)
        self.chain.daily_limits = {"human": cfg.human_daily_limit_usd}
        self.chain.state.skip_after_reports = cfg.skip_after_reports

    def supports(self, task_type: str) -> bool:
        return any(layer.supports(task_type) for layer in self.chain.layers)

    def submit(self, task: dict[str, Any]) -> int:
        now = time.monotonic()
        for k in [k for k, j in self.jobs.items() if now - j.created > self.cfg.task_ttl]:
            del self.jobs[k]
        task_id = next(self._ids)
        job = Job(task)
        self.jobs[task_id] = job
        t = asyncio.create_task(self._run(job))
        self._running.add(t)
        t.add_done_callback(self._running.discard)
        return task_id

    async def _run(self, job: Job) -> None:
        started = time.time()
        try:
            if self.paused:
                raise RuntimeError("решатель на паузе")
            job.result = await self.chain.solve(job.task)
        except Exception as e:
            job.error = str(e)
        finally:
            job.done.set()
            r = job.result
            self.recent.appendleft({
                "at": started, "seconds": round(time.time() - started, 2), "type": job.task.get("type"),
                "layer": r.layer if r else "",
                "layerTitle": next((x.title for x in self.chain.layers if r and x.name == r.layer), r.layer if r else ""), "confidence": round(r.confidence, 2) if r else 0,
                "cost": r.cost if r else 0.0, "lowConfidence": bool(r and r.meta.get("low_confidence")),
                "error": job.error, "notes": (r.meta.get("tried", []) if r else []),
            })

    async def report(self, task_id: int, retry: bool = False) -> int | None:
        job = self.jobs.get(task_id)
        if job is None or job.result is None:
            raise KeyError(task_id)
        await self.chain.report_incorrect(job.task, job.result)
        return self.submit(job.task) if retry else None

    async def balance(self) -> float:
        human = self.chain.layer("human")
        if isinstance(human, HumanLayer) and human.available():
            return await human.balance()
        return 0.0

    def today(self) -> dict[str, dict[str, float]]:
        return self.chain.state.daily.get(today(), {})
