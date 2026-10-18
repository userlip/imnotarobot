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

from .config import HUMAN_CURRENCY, HUMAN_TOPUP
from .control import HUMAN_TITLES, LAYER_TITLES, money
from .layers.vlm import LAST_ERRORS

if TYPE_CHECKING:
    from .control import Controller

PAGE = Path(__file__).with_name("assets") / "ui.html"

TYPE_TITLES = {
    "ImageToTextTask": "Текст на картинке", "GridTask": "Сетка картинок", "CoordinatesTask": "Клик по картинке",
    "SlideMatchTask": "Слайдер", "RotateTask": "Поворот", "RecaptchaV2TaskProxyless": "reCAPTCHA v2",
    "RecaptchaV2Task": "reCAPTCHA v2", "RecaptchaV3TaskProxyless": "reCAPTCHA v3", "HCaptchaTaskProxyless": "hCaptcha",
    "HCaptchaTask": "hCaptcha", "TurnstileTaskProxyless": "Cloudflare Turnstile", "TurnstileTask": "Cloudflare Turnstile",
    "GeeTestTaskProxyless": "GeeTest", "FunCaptchaTaskProxyless": "FunCaptcha",
}


LAYER_CAN = {"local": "текст на картинке, слайдеры · бесплатно, доли секунды",
             "human": "всё, включая reCAPTCHA, hCaptcha, Turnstile · платно, 10–60 секунд",
             "vlm": "текст, сетки картинок, клики · центы, 5–30 секунд"}


def type_title(task_type: str) -> str:
    return TYPE_TITLES.get(task_type, task_type)


def stats(c: "Controller") -> dict[str, Any]:
    titles = {x.name: x.title or x.name for x in c.service.chain.layers}
    rows = []
    for s in c.service.chain.state.as_list():
        task_type, _, comment = s["kind"].partition("|")
        rows.append({"type": type_title(task_type), "comment": comment, "layer": titles.get(s["layer"], s["layer"]),
                     "answered": s["answered"], "bad": s["reported_bad"]})
    days = sorted(c.service.chain.state.daily.items())[-7:]
    history = [{"day": d, "solved": int(sum(v.get("solved", 0) for v in layers.values())),
                "cloud": sum(v.get("spent", v.get("usd", 0.0)) for k, v in layers.items() if k.startswith("vlm:")),
                "human": layers.get("human", {}).get("spent", layers.get("human", {}).get("usd", 0.0))}
               for d, layers in days]
    return {"rows": sorted(rows, key=lambda r: -r["answered"])[:40], "days": history}


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
        can = LAYER_CAN["vlm" if layer_id.startswith("vlm:") else layer_id]
        layers.append({"id": layer_id, "title": title, "enabled": layer_id in cfg.layers, "ready": ready,
                       "why": why, "active": layer_id in built, "can": can})
    day = c.service.today()
    return {
        "paused": c.service.paused, "address": c.address, "startAtLogin": c.starts_at_login(),
        "today": {"solved": int(sum(v.get("solved", 0) for v in day.values())),
                  "usd": sum(v.get("usd", 0.0) for v in day.values()), "byLayer": day},
        "layers": layers,
        "keys": [{"name": k.name, "provider": k.provider, "hint": k.hint, "baseUrl": k.base_url, "listable": k.listable,
                  "models": [{"id": mid, "enabled": on} for mid, on in c.key_models(k.name)]} for k in cfg.keys],
        "human": {"provider": cfg.human_provider, "currency": HUMAN_CURRENCY.get(cfg.human_provider, "$"),
                  "hasKey": bool(cfg.human_api_key), "hint": cfg.human_key_hint, "legacyKey": c.legacy_human_key,
                  "limit": cfg.human_daily_limit, "lowBalance": cfg.human_low_balance, "status": c.human_status,
                  "spentToday": c.service.chain.state.spent_today("human"), "topup": HUMAN_TOPUP.get(cfg.human_provider)},
        "cloud": {"limit": cfg.cloud_daily_limit_usd, "spentToday": c.spend_today()["cloud_usd"],
                  "unpriced": [m.name for m in cfg.models if not (m.price_in or m.price_out)]},
        "recent": [{**r, "typeTitle": type_title(r.get("type", ""))} for r in list(c.service.recent)[:50]],
        "stats": stats(c),
        "bench": {k: v for k, v in c.bench.items() if k != "report"},
        "datasets": c.datasets(),
        "access": {"required": cfg.require_client_key, "key": cfg.client_key, "portNote": c.port_note},
        "detecting": c.detecting,
        "modelErrors": {name[4:]: {"at": at, "error": msg} for name, (at, msg) in LAST_ERRORS.items()},
        "modelTests": c.model_tests,
        "benchCases": len(c.bench_cases()),
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

    @app.get("/api/ui/journal/{entry_id}")
    def journal(entry_id: int, request: Request):
        denied = guard(request)
        if denied:
            return denied
        entry = c.service.journal_entry(entry_id)
        if entry is None:
            return JSONResponse({"ok": False, "error": "запись не найдена"}, status_code=404)
        task = entry["task"]
        images = {k: task[k] for k in ("body", "target", "background", "imgInstructions") if task.get(k)}
        details = {k: v for k, v in task.items() if k not in images}
        return {"ok": True, "summary": {**entry["summary"], "typeTitle": type_title(task.get("type", ""))},
                "images": images, "task": details, "solution": entry.get("solution"), "meta": entry.get("meta", {})}

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
        cur = HUMAN_CURRENCY.get(c.cfg.human_provider, "$")
        if action == "human-key":
            key = str(b.get("key") or "").strip()
            if not key:
                raise ValueError("вставьте ключ")
            c.set_human(key=key)
            st = await run_in_threadpool(c.check_human)
            if not st["ok"]:
                return JSONResponse({"ok": False, "error": f"ключ сохранён, но {c.cfg.human_provider} его не принял: {st['error']}",
                                     "state": state(c)}, status_code=400)
            return done(f"Ключ работает, баланс {money(st['balance'], HUMAN_CURRENCY.get(c.cfg.human_provider, '$'))}.")
        if action == "human-remove-key":
            c.set_human(key="")
            return done("Ключ людей удалён, слой людей выключен.")
        if action == "human-provider":
            c.set_human(provider=b["provider"])
            msg = f"Сервис: {HUMAN_TITLES.get(c.cfg.human_provider)}. Валюта: {HUMAN_CURRENCY.get(c.cfg.human_provider)}."
            if c.cfg.human_api_key:
                st = await run_in_threadpool(c.check_human)
                msg += " Ключ подходит." if st["ok"] else " Этот ключ к нему не подходит — вставьте ключ этого сервиса."
            return done(msg)
        if action == "limits":
            def number(name: str) -> float | None:
                v = b.get(name)
                if v in (None, ""):
                    return None
                try:
                    v = float(str(v).replace(",", "."))
                except ValueError:
                    raise ValueError("нужно число")
                if v < 0:
                    raise ValueError("число не может быть отрицательным")
                return v

            human, low, cloud = number("human"), number("low"), number("cloud")
            if human is not None:
                c.cfg.human_daily_limit = human
            if low is not None:
                c.cfg.human_low_balance = low
            if cloud is not None:
                c.cfg.cloud_daily_limit_usd = cloud
            c._apply()
            h = c.cfg.human_daily_limit
            parts = [f"люди — {'без лимита' if not h else money(h, HUMAN_CURRENCY.get(c.cfg.human_provider, '$')) + ' в день'}",
                     f"модели — {'без лимита' if not c.cfg.cloud_daily_limit_usd else money(c.cfg.cloud_daily_limit_usd, '$') + ' в день'}"]
            return done("Лимиты сохранены: " + ", ".join(parts) + ".")
        if action == "check-human":
            st = await run_in_threadpool(c.check_human)
            if not st["ok"]:
                return JSONResponse({"ok": False, "error": st["error"], "state": state(c)}, status_code=400)
            return done(f"Баланс {money(st['balance'], cur)}.")
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
        if action == "report-wrong":
            try:
                await c.service.report_entry(int(b["id"]))
            except KeyError:
                raise ValueError("у этой капчи нет ответа, сообщать нечего")
            return done("Записал: этот слой на такой картинке больше не используется. Если решали люди — сервису отправлена жалоба.")
        if action == "copy-key":
            import subprocess

            subprocess.run(["pbcopy"], input=c.cfg.client_key.encode(), check=False)
            return done("Ключ доступа скопирован.")
        if action == "copy-address":
            import subprocess

            subprocess.run(["pbcopy"], input=c.address.encode(), check=False)
            return done(f"Адрес {c.address} скопирован.")
        if action == "access":
            c.set_access(b.get("required"), bool(b.get("regenerate")))
            if b.get("regenerate"):
                return done("Новый ключ доступа создан. Старый больше не работает — обновите его в своих программах.")
            return done("Программы должны присылать ключ доступа." if c.cfg.require_client_key
                        else "Ключ доступа выключен: решатель примет запрос от любой программы на этом Маке.")
        if action == "check-connection":
            r = await run_in_threadpool(c.check_connection)
            if not r["ok"]:
                return JSONResponse({"ok": False, "error": f"программа получила бы ошибку: {r['error']}", "state": state(c)},
                                    status_code=400)
            verdict = "ответ верный" if r["correct"] else "ответ неверный"
            return done(f"Подключение работает: тестовую капчу решил слой «{r['layer']}» за {r['seconds']} с, {verdict}.")
        if action == "test-model":
            r = await run_in_threadpool(c.test_model, b["key"], b["model"])
            if r["error"] and not r["ok"]:
                return JSONResponse({"ok": False, "error": f"{b['model']}: {r['error']}", "state": state(c)}, status_code=400)
            return done(f"{b['model']}: решено {r['ok']} из {r['n']}, в среднем {r['seconds']} с на капчу.")
        if action == "bench-stop":
            c.stop_bench()
            return done("Останавливаю: дорешаю начатые капчи и покажу то, что успели.")
        if action == "dataset-download":
            c.download_dataset(b["name"])
            return done("Скачиваю набор — прогресс виден в разделе сравнения.")
        if action == "dataset-remove":
            c.remove_dataset(b["name"])
            return done("Набор удалён.")
        if action == "use-best":
            c.use_best(b["name"])
            return done(f"{b['name']} теперь первая среди моделей.")
        raise ValueError(f"неизвестное действие {action}")
