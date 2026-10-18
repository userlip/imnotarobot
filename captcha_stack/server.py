import asyncio
import itertools
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .chain import Chain
from .config import Config
from .layers import HumanLayer, Layer, LocalLayer, Result, VlmLayer


def build_layers(cfg: Config) -> list[Layer]:
    known = {
        "local": lambda: LocalLayer(cfg.ocr_min_confidence),
        "vlm": lambda: VlmLayer(cfg.vlm_base_url, cfg.vlm_api_key, cfg.vlm_model, cfg.vlm_min_confidence, cfg.vlm_timeout),
        "human": lambda: HumanLayer(cfg.human_base_url, cfg.human_api_key, cfg.human_timeout, cfg.human_poll_interval),
    }
    return [known[name]() for name in cfg.layers if name in known]


@dataclass
class Job:
    task: dict[str, Any]
    created: float = field(default_factory=time.monotonic)
    result: Result | None = None
    error: str = ""
    done: asyncio.Event = field(default_factory=asyncio.Event)


def _error(code: str, description: str, error_id: int = 1) -> JSONResponse:
    return JSONResponse({"errorId": error_id, "errorCode": code, "errorDescription": description})


def create_app(cfg: Config | None = None, layers: list[Layer] | None = None) -> FastAPI:
    cfg = cfg or Config.from_env()
    chain = Chain(layers if layers is not None else build_layers(cfg), cfg.skip_after_reports)
    jobs: dict[int, Job] = {}
    running: set[asyncio.Task] = set()
    ids = itertools.count(int(time.time()))
    app = FastAPI(title="captcha-stack")
    app.state.chain = chain
    app.state.jobs = jobs

    def expire() -> None:
        now = time.monotonic()
        for k in [k for k, j in jobs.items() if now - j.created > cfg.task_ttl]:
            del jobs[k]

    def start(job: Job) -> None:
        t = asyncio.create_task(run(job))
        running.add(t)
        t.add_done_callback(running.discard)

    async def run(job: Job) -> None:
        try:
            job.result = await chain.solve(job.task)
        except Exception as e:
            job.error = str(e)
        finally:
            job.done.set()

    async def body_of(request: Request) -> dict[str, Any] | JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            return _error("ERROR_BAD_PARAMETERS", "body is not JSON")
        if cfg.client_key and body.get("clientKey") != cfg.client_key:
            return _error("ERROR_KEY_DOES_NOT_EXIST", "wrong clientKey")
        return body

    @app.post("/createTask")
    async def create_task(request: Request):
        body = await body_of(request)
        if isinstance(body, JSONResponse):
            return body
        task = body.get("task") or {}
        if not task.get("type"):
            return _error("ERROR_BAD_PARAMETERS", "task.type is required")
        if not any(layer.supports(task["type"]) for layer in chain.layers):
            return _error("ERROR_TASK_NOT_SUPPORTED", f"no layer handles {task['type']}")
        expire()
        task_id = next(ids)
        job = Job(task)
        jobs[task_id] = job
        start(job)
        return {"errorId": 0, "taskId": task_id}

    @app.post("/getTaskResult")
    async def get_task_result(request: Request):
        body = await body_of(request)
        if isinstance(body, JSONResponse):
            return body
        job = jobs.get(body.get("taskId"))
        if job is None:
            return _error("ERROR_NO_SUCH_CAPCHA_ID", "unknown or expired taskId")
        if not job.done.is_set():
            return {"errorId": 0, "status": "processing"}
        if job.result is None:
            return _error("ERROR_CAPTCHA_UNSOLVABLE", job.error)
        r = job.result
        return {"errorId": 0, "status": "ready", "solution": r.solution, "cost": f"{r.cost:.5f}",
                "layer": r.layer, "confidence": round(r.confidence, 3), "notes": r.meta.get("tried", []),
                "lowConfidence": bool(r.meta.get("low_confidence"))}

    @app.post("/reportIncorrect")
    async def report_incorrect(request: Request):
        body = await body_of(request)
        if isinstance(body, JSONResponse):
            return body
        job = jobs.get(body.get("taskId"))
        if job is None or job.result is None:
            return _error("ERROR_NO_SUCH_CAPCHA_ID", "unknown taskId or no answer to report")
        await chain.report_incorrect(job.task, job.result)
        reply: dict[str, Any] = {"errorId": 0, "status": "success"}
        if body.get("retry"):
            task_id = next(ids)
            retry = Job(job.task)
            jobs[task_id] = retry
            start(retry)
            reply["taskId"] = task_id
        return reply

    @app.post("/reportCorrect")
    async def report_correct(request: Request):
        body = await body_of(request)
        return body if isinstance(body, JSONResponse) else {"errorId": 0, "status": "success"}

    @app.post("/getBalance")
    async def get_balance(request: Request):
        body = await body_of(request)
        if isinstance(body, JSONResponse):
            return body
        human = chain.layer("human")
        balance = 0.0
        if isinstance(human, HumanLayer) and human.available():
            try:
                balance = await human.balance()
            except Exception as e:
                return _error("ERROR_UPSTREAM", str(e))
        return {"errorId": 0, "balance": balance}

    @app.get("/stats")
    async def stats():
        return {"layers": [{"name": x.name, "available": x.available()} for x in chain.layers],
                "stats": chain.stats.as_dict(), "openTasks": len(jobs)}

    return app
