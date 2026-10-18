"""JSON API and page for the settings window.

Every /api/ui call must carry the X-UI-Token header. A web page in a browser cannot add a custom header to a
cross-origin request without a CORS preflight, which this server never approves, so other sites cannot change
settings or read state through the local port. Secrets are never returned, only their hints.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .control import HUMAN_TITLES, LAYER_TITLES

if TYPE_CHECKING:
    from .control import Controller

PAGE = Path(__file__).with_name("assets") / "ui.html"


def state(c: "Controller") -> dict[str, Any]:
    cfg = c.cfg
    layers = []
    known = {f"vlm:{m.name}": m for m in cfg.models}
    order = list(cfg.layers) + [i for i in ["local", *known, "human"] if i not in cfg.layers]
    built = {x.name: x for x in c.service.chain.layers}
    for layer_id in order:
        if layer_id == "local":
            title, ready, why = LAYER_TITLES["local"], True, ""
        elif layer_id == "human":
            title = f"Люди ({HUMAN_TITLES.get(cfg.human_provider, cfg.human_provider)})"
            ready, why = bool(cfg.human_api_key), "" if cfg.human_api_key else "нет ключа"
        elif layer_id in known:
            m = known[layer_id]
            title, ready, why = f"{m.model} · {m.key or m.provider}", m.ready(), "" if m.ready() else "нет ключа"
        else:
            continue
        layers.append({"id": layer_id, "title": title, "enabled": layer_id in cfg.layers, "ready": ready,
                       "why": why, "active": layer_id in built})
    day = c.service.today()
    return {
        "paused": c.service.paused, "address": c.address, "startAtLogin": c.starts_at_login(),
        "today": {"solved": int(sum(v.get("solved", 0) for v in day.values())),
                  "usd": sum(v.get("usd", 0.0) for v in day.values()), "byLayer": day},
        "layers": layers,
        "keys": [{"name": k.name, "provider": k.provider, "hint": k.hint, "baseUrl": k.base_url, "listable": k.listable,
                  "models": [{"id": mid, "enabled": on} for mid, on in c.key_models(k.name)]} for k in cfg.keys],
        "human": {"provider": cfg.human_provider, "hasKey": bool(cfg.human_api_key), "legacyKey": c.legacy_human_key,
                  "limit": cfg.human_daily_limit_usd, "status": c.human_status,
                  "spentToday": c.service.chain.state.spent_today("human")},
        "recent": list(c.service.recent)[:30],
        "bench": {k: v for k, v in c.bench.items() if k != "report"},
        "now": time.time(),
    }


def add_ui_routes(app: FastAPI, c: "Controller") -> None:
    def guard(request: Request) -> JSONResponse | None:
        if request.headers.get("x-ui-token") != c.ui_token:
            return JSONResponse({"ok": False, "error": "нет доступа"}, status_code=403)
        return None

    async def body(request: Request) -> dict[str, Any]:
        try:
            return await request.json()
        except ValueError:
            return {}

    def done(message: str, **extra) -> dict[str, Any]:
        return {"ok": True, "message": message, "state": state(c), **extra}

    def failed(e: Exception) -> JSONResponse:
        return JSONResponse({"ok": False, "error": str(e), "state": state(c)}, status_code=400)

    @app.get("/ui", response_class=HTMLResponse)
    def page():
        return FileResponse(PAGE)

    @app.get("/api/ui/state")
    def get_state(request: Request):
        return guard(request) or state(c)

    @app.post("/api/ui/{action}")
    async def act(action: str, request: Request):
        denied = guard(request)
        if denied:
            return denied
        b = await body(request)
        try:
            return await run_action(action, b)
        except Exception as e:
            return failed(e)

    async def run_action(action: str, b: dict[str, Any]) -> Any:
        from starlette.concurrency import run_in_threadpool

        if action == "add-key":
            entry = await run_in_threadpool(c.add_key, b.get("secret", ""), b.get("baseUrl", ""))
            n = len(c.key_models(entry.name))
            where = "посредник" if entry.provider == "custom" else "поставщик"
            hint = f"моделей с картинками: {n}" if entry.listable else "список моделей недоступен — впишите модель вручную"
            return done(f"Ключ добавлен: {where} {entry.name}, {hint}.", key=entry.name)
        if action == "refresh-key":
            entry = await run_in_threadpool(c.refresh_key, b["name"])
            return done(f"Список моделей {entry.name} обновлён: {len(c.key_models(entry.name))} с картинками.")
        if action == "remove-key":
            c.remove_key(b["name"])
            return done(f"Ключ {b['name']} удалён вместе с его моделями.")
        if action == "toggle-model":
            c.toggle_key_model(b["key"], b["model"])
            on = any(m.key == b["key"] and m.model == b["model"] for m in c.cfg.models)
            return done(f"{b['model']}: {'включена' if on else 'выключена'}.")
        if action == "manual-model":
            entry = c.key(b["key"])
            model = b.get("model", "").strip()
            if not model:
                raise ValueError("впишите название модели")
            if not any(mid == model for mid, *_ in entry.models):
                entry.models.append([model, True, 0.0, 0.0])
            c.toggle_key_model(b["key"], model)
            return done(f"{model} добавлена и включена.")
        if action == "toggle-layer":
            c.toggle_layer(b["id"])
            return done("Слои обновлены.")
        if action == "move-layer":
            c.move_layer(b["id"], int(b.get("step", 0)))
            return done("Порядок обновлён.")
        if action == "human":
            limit = b.get("limit")
            c.set_human(b.get("provider"), b.get("key"), float(limit) if limit not in (None, "") else None)
            if b.get("key") is not None or b.get("provider") is not None:
                st = await run_in_threadpool(c.check_human)
                if not st["ok"]:
                    return JSONResponse({"ok": False, "error": f"ключ сохранён, но сервис ответил: {st['error']}",
                                         "state": state(c)}, status_code=400)
                return done(f"Ключ работает, баланс ${st['balance']:.2f}.")
            return done("Настройки людей сохранены.")
        if action == "check-human":
            st = await run_in_threadpool(c.check_human)
            return done(f"Баланс ${st['balance']:.2f}." if st["ok"] else f"Ошибка: {st['error']}")
        if action == "pause":
            c.set_paused(bool(b.get("paused")))
            return done("Решатель на паузе." if c.service.paused else "Решатель работает.")
        if action == "login":
            c.set_start_at_login(bool(b.get("on")))
            return done("Будет запускаться при входе." if c.starts_at_login() else "Автозапуск выключен.")
        if action == "bench-estimate":
            specs = c.bench_specs(b.get("key"))
            runs, usd = c.bench_estimate(specs)
            return done("", models=len(specs), runs=runs, usd=usd)
        if action == "bench-start":
            c.start_bench(b.get("key"))
            return done("Сравнение запущено.")
        if action == "copy-address":
            import subprocess

            subprocess.run(["pbcopy"], input=c.address.encode(), check=False)
            return done(f"Адрес {c.address} скопирован.")
        if action == "use-best":
            c.use_best(b["name"])
            return done(f"{b['name']} теперь первая среди моделей.")
        raise ValueError(f"неизвестное действие {action}")
