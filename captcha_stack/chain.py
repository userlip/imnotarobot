import hashlib
import logging
from collections import defaultdict
from typing import Any

from .layers import Layer, Result, Unsolvable

log = logging.getLogger(__name__)


def kind_of(task: dict[str, Any]) -> str:
    """Groups captchas that should behave alike: same task type and instruction, so per-layer statistics carry over."""
    return f"{task.get('type')}|{(task.get('comment') or '').strip().lower()}"


def image_key(task: dict[str, Any]) -> str:
    raw = (task.get("body") or "") + (task.get("target") or "") + (task.get("background") or "")
    return hashlib.sha256(raw.encode()).hexdigest()[:16] if raw else ""


class Stats:
    def __init__(self, skip_after_reports: int):
        self.skip_after_reports = skip_after_reports
        self.data: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"answered": 0, "reported_bad": 0})

    def answered(self, kind: str, layer: str) -> None:
        self.data[(kind, layer)]["answered"] += 1

    def bad(self, kind: str, layer: str) -> None:
        self.data[(kind, layer)]["reported_bad"] += 1

    def unreliable(self, kind: str, layer: str) -> bool:
        d = self.data.get((kind, layer))
        if not d or d["reported_bad"] < self.skip_after_reports:
            return False
        return d["reported_bad"] * 2 >= d["answered"]

    def as_dict(self) -> list[dict[str, Any]]:
        return [{"kind": k, "layer": layer, **v} for (k, layer), v in sorted(self.data.items())]


class Chain:
    """Tries layers cheapest first; moves on when a layer fails, is unsure, or keeps getting reported for this kind."""

    def __init__(self, layers: list[Layer], skip_after_reports: int = 5):
        self.layers = layers
        self.stats = Stats(skip_after_reports)
        self.rejected: dict[str, set[str]] = defaultdict(set)

    def layer(self, name: str) -> Layer | None:
        return next((x for x in self.layers if x.name == name), None)

    async def solve(self, task: dict[str, Any]) -> Result:
        kind, key = kind_of(task), image_key(task)
        candidates: list[Result] = []
        notes: list[str] = []
        for layer in self.layers:
            if not layer.supports(task["type"]):
                continue
            if not layer.available():
                notes.append(f"{layer.name}: not configured")
                continue
            if layer.name in self.rejected.get(key, ()):
                notes.append(f"{layer.name}: its answer for this image was reported wrong")
                continue
            if self.stats.unreliable(kind, layer.name):
                notes.append(f"{layer.name}: skipped, most of its answers for this kind were reported wrong")
                continue
            try:
                r = await layer.solve(task)
            except Exception as e:  # any layer failure means: try the next one
                notes.append(f"{layer.name}: {e}")
                log.info("layer %s failed: %s", layer.name, e)
                continue
            self.stats.answered(kind, layer.name)
            r.meta["tried"] = notes.copy()
            if r.confidence >= layer.min_confidence:
                return r
            twin = next((c for c in candidates if c.solution == r.solution), None)
            if twin:
                r.meta["agreed_with"] = twin.layer
                return r
            notes.append(f"{layer.name}: unsure ({r.confidence:.2f})")
            candidates.append(r)
        if candidates:
            best = max(candidates, key=lambda c: c.confidence)
            best.meta["low_confidence"] = True
            best.meta["tried"] = notes
            return best
        raise Unsolvable("; ".join(notes) or f"no layer handles {task['type']}")

    async def report_incorrect(self, task: dict[str, Any], result: Result) -> None:
        self.stats.bad(kind_of(task), result.layer)
        key = image_key(task)
        if key:
            self.rejected[key].add(result.layer)
        layer = self.layer(result.layer)
        if layer:
            try:
                await layer.report_incorrect(result)
            except Exception as e:
                log.warning("could not forward the report to %s: %s", result.layer, e)
