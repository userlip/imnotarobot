"""Runs vision models over test captchas with known answers and ranks them."""

import asyncio
import html
import itertools
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import httpx

from .layers import VlmLayer
from .providers import ModelSpec

# rough per-ask usage for one captcha image plus the prompt and a short reasoning
TOKENS_IN, TOKENS_OUT = 1500, 400


def _boxes(expected: dict[str, Any]) -> list[list[float]]:
    return expected.get("boxes") or ([expected["box"]] if "box" in expected else [])


def _inside(point: dict[str, float], box: list[float]) -> bool:
    return box[0] <= point["x"] <= box[2] and box[1] <= point["y"] <= box[3]


def score(case: dict[str, Any], solution: dict[str, Any]) -> bool:
    exp = case["expected"]
    if "text" in exp:
        got = str(solution.get("text", "")).strip()
        return got == exp["text"] if case["task"].get("case") else got.lower() == exp["text"].lower()
    if "tiles" in exp:
        return sorted(solution.get("click", [])) == sorted(exp["tiles"])
    boxes, points = _boxes(exp), solution.get("coordinates", [])
    if len(points) != len(boxes):
        return False
    if exp.get("inorder", True):
        return all(_inside(p, b) for p, b in zip(points, boxes))
    return any(all(_inside(p, b) for p, b in zip(points, perm)) for perm in itertools.permutations(boxes))


def estimate_usd(specs: list[ModelSpec], n_cases: int, votes: int) -> float | None:
    """None when a model's price is unknown."""
    total = 0.0
    for s in specs:
        if not (s.price_in or s.price_out):
            return None
        total += n_cases * votes * (TOKENS_IN * s.price_in + TOKENS_OUT * s.price_out) / 1e6
    return total


async def run(specs: list[ModelSpec], cases: list[dict[str, Any]], votes: int = 1, concurrency: int = 3,
              transport: httpx.AsyncBaseTransport | None = None,
              progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    gate = asyncio.Semaphore(concurrency)
    total, done = len(specs) * len(cases), 0
    results: dict[str, Any] = {}

    async def one(layer: VlmLayer, case: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        async with gate:
            t0 = time.monotonic()
            try:
                r = await layer.solve(case["task"])
                row = {"ok": score(case, r.solution), "cost": r.cost, "answer": r.solution, "error": ""}
            except Exception as e:
                row = {"ok": False, "cost": 0.0, "answer": None, "error": str(e)[:200]}
            row.update(id=case["id"], set=case["set"], seconds=time.monotonic() - t0)
            done += 1
            if progress:
                progress(done, total)
            return row

    for spec in specs:
        layer = VlmLayer(replace(spec, votes=votes), transport=transport)
        rows = await asyncio.gather(*(one(layer, c) for c in cases))
        by_set: dict[str, dict[str, int]] = {}
        for row in rows:
            s = by_set.setdefault(row["set"], {"n": 0, "ok": 0})
            s["n"] += 1
            s["ok"] += row["ok"]
        n = len(rows) or 1
        cost = sum(r["cost"] for r in rows)
        results[spec.name] = {
            "model": spec.model, "provider": spec.provider, "accuracy": sum(r["ok"] for r in rows) / n,
            "seconds": sum(r["seconds"] for r in rows) / n, "cost": cost,
            "usd_per_1000": cost / n * 1000 if (spec.price_in or spec.price_out) else None,
            "errors": sum(bool(r["error"]) for r in rows), "by_set": by_set, "rows": rows,
        }
    return results


def ranking(results: dict[str, Any]) -> list[str]:
    return sorted(results, key=lambda k: (-results[k]["accuracy"], results[k]["usd_per_1000"] or 0, results[k]["seconds"]))


def report_html(results: dict[str, Any], path: Path, votes: int) -> Path:
    sets = sorted({s for r in results.values() for s in r["by_set"]})
    e = html.escape
    head = "".join(f"<th>{e(s)}</th>" for s in sets)
    body = []
    for name in ranking(results):
        r = results[name]
        cells = "".join(
            f"<td>{r['by_set'][s]['ok']}/{r['by_set'][s]['n']}</td>" if s in r["by_set"] else "<td>—</td>" for s in sets)
        price = f"${r['usd_per_1000']:.2f}" if r["usd_per_1000"] is not None else "неизвестно"
        body.append(f"<tr><td><b>{e(name)}</b><br><small>{e(r['provider'])} / {e(r['model'])}</small></td>"
                    f"<td><b>{r['accuracy']:.0%}</b></td>{cells}<td>{r['seconds']:.1f} с</td><td>{price}</td>"
                    f"<td>{r['errors']}</td></tr>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>Сравнение моделей на капчах</title>
<style>
:root {{ --bg:#fafaf8; --fg:#1d1d1b; --line:#ddd; --mut:#666 }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ecebe6; --line:#333; --mut:#999 }} }}
body {{ font: 15px -apple-system, sans-serif; background: var(--bg); color: var(--fg); margin: 24px }}
table {{ border-collapse: collapse }} td, th {{ border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: left }}
small, .mut {{ color: var(--mut) }}
</style>
<h1>Сравнение моделей на капчах</h1>
<p class="mut">Лучшие сверху: точность, затем цена, затем скорость. Каждую капчу модель решала с {votes} попытк(ой/ами).
Цена за 1000 — по ценам поставщика, если он их публикует.</p>
<table><tr><th>Модель</th><th>Точность</th>{head}<th>Время</th><th>Цена за 1000</th><th>Ошибок API</th></tr>
{''.join(body)}</table>""")
    return path
