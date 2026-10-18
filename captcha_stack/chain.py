import datetime
import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .layers import Layer, Result, Unsolvable

log = logging.getLogger(__name__)


def kind_of(task: dict[str, Any]) -> str:
    """Groups captchas that should behave alike: same task type and instruction, so per-layer statistics carry over."""
    return f"{task.get('type')}|{(task.get('comment') or '').strip().lower()}"


def image_key(task: dict[str, Any]) -> str:
    raw = (task.get("body") or "") + (task.get("target") or "") + (task.get("background") or "")
    return hashlib.sha256(raw.encode()).hexdigest()[:16] if raw else ""


def today() -> str:
    return datetime.date.today().isoformat()


class State:
    """Per-layer statistics, rejected answers and daily spend; kept on disk so a restart does not forget them."""

    def __init__(self, path: str | None = None, skip_after_reports: int = 5, max_rejected: int = 20000):
        self.path = Path(path) if path else None
        self.skip_after_reports = skip_after_reports
        self.max_rejected = max_rejected
        self.stats: dict[str, dict[str, int]] = {}
        self.rejected: OrderedDict[str, list[str]] = OrderedDict()
        self.daily: dict[str, dict[str, dict[str, float]]] = {}
        if self.path and self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                self.stats = d.get("stats", {})
                self.rejected = OrderedDict(d.get("rejected", {}))
                self.daily = d.get("daily", {})
            except (ValueError, OSError) as e:
                log.warning("state file unreadable, starting fresh: %s", e)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"stats": self.stats, "rejected": self.rejected, "daily": self.daily}))
        tmp.replace(self.path)

    def _stat(self, kind: str, layer: str) -> dict[str, int]:
        return self.stats.setdefault(f"{kind}\t{layer}", {"answered": 0, "reported_bad": 0})

    def answered(self, kind: str, layer: str, cost: float) -> None:
        self._stat(kind, layer)["answered"] += 1
        day = self.daily.setdefault(today(), {}).setdefault(layer, {"solved": 0, "usd": 0.0})
        day["solved"] += 1
        day["usd"] += cost
        for old in sorted(self.daily)[:-31]:
            del self.daily[old]
        self.save()

    def bad(self, kind: str, layer: str, key: str) -> None:
        self._stat(kind, layer)["reported_bad"] += 1
        if key:
            layers = self.rejected.pop(key, [])
            self.rejected[key] = sorted(set(layers) | {layer})
            while len(self.rejected) > self.max_rejected:
                self.rejected.popitem(last=False)
        self.save()

    def unreliable(self, kind: str, layer: str) -> bool:
        d = self.stats.get(f"{kind}\t{layer}")
        if not d or d["reported_bad"] < self.skip_after_reports:
            return False
        return d["reported_bad"] * 2 >= d["answered"]

    def spent_today(self, layer: str) -> float:
        return self.daily.get(today(), {}).get(layer, {}).get("usd", 0.0)

    def as_list(self) -> list[dict[str, Any]]:
        out = []
        for k, v in sorted(self.stats.items()):
            kind, layer = k.split("\t")
            out.append({"kind": kind, "layer": layer, **v})
        return out


class Chain:
    """Tries layers cheapest first; moves on when a layer fails, is unsure, is over budget,
    or keeps getting reported for this kind of captcha."""

    def __init__(self, layers: list[Layer], state: State | None = None, daily_limits: dict[str, float] | None = None):
        self.layers = layers
        self.state = state or State()
        self.daily_limits = daily_limits or {}

    def layer(self, name: str) -> Layer | None:
        return next((x for x in self.layers if x.name == name), None)

    def _skip_reason(self, layer: Layer, kind: str, key: str) -> str:
        if not layer.available():
            return "нет ключа"
        if layer.name in self.state.rejected.get(key, ()):
            return "его ответ на эту картинку уже отвергли"
        if self.state.unreliable(kind, layer.name):
            return "пропущен: большинство его ответов на такие капчи отвергли"
        limit = self.daily_limits.get(layer.name, 0)
        if limit and self.state.spent_today(layer.name) >= limit:
            return f"исчерпан дневной лимит ${limit:g}"
        return ""

    async def solve(self, task: dict[str, Any]) -> Result:
        kind, key = kind_of(task), image_key(task)
        candidates: list[Result] = []
        notes: list[str] = []
        for layer in self.layers:
            if not layer.supports(task["type"]):
                continue
            reason = self._skip_reason(layer, kind, key)
            if reason:
                notes.append(f"{layer.title or layer.name}: {reason}")
                continue
            try:
                r = await layer.solve(task)
            except Exception as e:  # any layer failure means: try the next one
                notes.append(f"{layer.title or layer.name}: {e}")
                log.info("layer %s failed: %s", layer.name, e)
                continue
            self.state.answered(kind, layer.name, r.cost)
            r.meta["tried"] = notes.copy()
            if r.confidence >= layer.min_confidence:
                return r
            twin = next((c for c in candidates if c.solution == r.solution), None)
            if twin:
                r.meta["agreed_with"] = twin.layer
                return r
            notes.append(f"{layer.title or layer.name}: не уверен ({r.confidence:.2f})")
            candidates.append(r)
        if candidates:
            best = max(candidates, key=lambda c: c.confidence)
            best.meta["low_confidence"] = True
            best.meta["tried"] = notes
            return best
        raise Unsolvable("; ".join(notes) or f"ни один слой не умеет {task['type']}")

    async def report_incorrect(self, task: dict[str, Any], result: Result) -> None:
        self.state.bad(kind_of(task), result.layer, image_key(task))
        layer = self.layer(result.layer)
        if layer:
            try:
                await layer.report_incorrect(result)
            except Exception as e:
                log.warning("could not forward the report to %s: %s", result.layer, e)
