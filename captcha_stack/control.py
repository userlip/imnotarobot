"""Everything the menu bar does, without the GUI, so it can be tested."""

import asyncio
import json
import plistlib
import secrets
import sys
import threading
import time
from urllib.parse import urlparse
from pathlib import Path

import uvicorn

from . import config as config_mod
from . import bench, testset
from .config import HUMAN_CURRENCY, HUMAN_PRESETS, Config, KeyEntry
from .discovery import detect_sync
from .providers import PRESETS, ModelSpec
from .server import create_app
from .service import Service

LAUNCH_AGENT = Path.home() / "Library" / "LaunchAgents" / "app.captcha-stack.plist"
LAYER_TITLES = {"local": "Локальная модель", "human": "Люди"}
HUMAN_TITLES = {"rucaptcha": "RuCaptcha", "2captcha": "2Captcha"}


def money(value: float, currency: str) -> str:
    return f"{value:.2f} ₽" if currency == "₽" else f"${value:.2f}"


class Controller:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.load()
        self._explicit_layers()
        if self.cfg.require_client_key and not self.cfg.client_key:
            self.cfg.client_key = "cs-" + secrets.token_urlsafe(18)
            self.cfg.save()
        self.service = Service(self.cfg)
        self._server: uvicorn.Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.bench: dict = {}
        self.detecting = ""
        self.model_tests: dict[str, dict] = {}
        self.dataset_jobs: dict[str, dict] = {}
        self.port: int | None = None
        self.port_note = ""
        self.human_status: dict = {}
        self.legacy_human_key = not self.cfg.human_api_key and config_mod.legacy_secret_exists("human")
        self.ui_token = secrets.token_urlsafe(24)

    # --- server ---------------------------------------------------------------
    def free_port(self) -> int:
        """The configured port, or the next free one if another program holds it."""
        import socket

        for port in range(self.cfg.port, self.cfg.port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"порты {self.cfg.port}–{self.cfg.port + 19} заняты другими программами")

    def start_server(self) -> None:
        port = self.free_port()
        if port != self.cfg.port:
            self.port_note = f"Порт {self.cfg.port} занят другой программой — работаю на {port}. Направьте программы сюда."
        self.port = port
        app = create_app(service=self.service)
        from .ui import add_ui_routes

        add_ui_routes(app, self)
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))

        def run():
            self._loop = asyncio.new_event_loop()
            self._loop.run_until_complete(self._server.serve())

        threading.Thread(target=run, daemon=True, name="captcha-stack-server").start()

    def stop_server(self) -> None:
        if self._server:
            self._server.should_exit = True

    @property
    def address(self) -> str:
        return f"http://127.0.0.1:{self.port or self.cfg.port}"

    def _explicit_layers(self) -> None:
        """The menu switches models one by one, so the catch-all "vlm" layer is spelled out per model."""
        if "vlm" in self.cfg.layers:
            i = self.cfg.layers.index("vlm")
            self.cfg.layers[i:i + 1] = [f"vlm:{m.name}" for m in self.cfg.models]

    def _add_model_layer(self, name: str) -> None:
        if f"vlm:{name}" in self.cfg.layers:
            return
        at = self.cfg.layers.index("human") if "human" in self.cfg.layers else len(self.cfg.layers)
        self.cfg.layers.insert(at, f"vlm:{name}")

    def _apply(self) -> None:
        self.cfg.save()
        fresh = Config.load()
        fresh.state_path = self.cfg.state_path
        self.cfg = fresh
        self._explicit_layers()
        self.service.reconfigure(fresh)

    # --- what the menu shows --------------------------------------------------
    def today_line(self) -> str:
        day = self.service.today()
        solved = int(sum(v.get("solved", 0) for v in day.values()))
        return f"Сегодня: решено {solved} · {self.spend_text()}"

    def spend_today(self) -> dict[str, float]:
        """Today's spend split by currency: cloud models in dollars, workers in their service's currency."""
        state = self.service.chain.state
        return {"cloud_usd": state.spent_today_prefix("vlm:"), "human": state.spent_today("human")}

    def spend_text(self) -> str:
        s = self.spend_today()
        cur = HUMAN_CURRENCY.get(self.cfg.human_provider, "$")
        parts = [f"модели ${s['cloud_usd']:.3f}"]
        if self.cfg.human_api_key or s["human"]:
            parts.append(f"люди {money(s['human'], cur)}")
        return ", ".join(parts)

    def layer_items(self) -> list[tuple[str, str, bool]]:
        """(layer id, title, enabled) for every layer that can be switched."""
        items = [("local", LAYER_TITLES["local"], "local" in self.cfg.layers)]
        for m in self.cfg.models:
            items.append((f"vlm:{m.name}", f"Модель {m.name}", f"vlm:{m.name}" in self.cfg.layers))
        items.append(("human", f"{LAYER_TITLES['human']} ({self.cfg.human_provider})", "human" in self.cfg.layers))
        return items

    def model_lines(self) -> list[tuple[str, str]]:
        return [(m.name, f"{'✓' if m.ready() else '⚠ нет ключа'}  {m.name}: {m.provider} / {m.model or '?'}")
                for m in self.cfg.models]

    # --- actions ----------------------------------------------------------------
    def toggle_layer(self, layer_id: str) -> None:
        order = [i for i, _, _ in self.layer_items()]
        enabled = {i for i, _, on in self.layer_items() if on}
        enabled ^= {layer_id}
        self.cfg.layers = [i for i in order if i in enabled]
        self._apply()

    def add_model(self, name: str, provider: str, model: str, key: str, base_url: str = "") -> ModelSpec:
        name = name.strip() or provider
        if provider not in PRESETS:
            raise ValueError(f"неизвестный поставщик {provider!r}; есть: {', '.join(PRESETS)}")
        if any(m.name == name for m in self.cfg.models):
            raise ValueError(f"модель с именем {name!r} уже есть")
        spec = ModelSpec(name=name, provider=provider, model=model.strip(), base_url=base_url.strip())
        if not spec.model:
            raise ValueError("укажите название модели")
        if key.strip():
            config_mod.set_secret(f"model:{name}", key.strip())
        self.cfg.models.append(spec)
        self._add_model_layer(name)
        self._apply()
        return spec

    def remove_model(self, name: str) -> None:
        self.cfg.models = [m for m in self.cfg.models if m.name != name]
        self.cfg.layers = [x for x in self.cfg.layers if x != f"vlm:{name}"]
        config_mod.set_secret(f"model:{name}", "")
        self._apply()

    def set_model_key(self, name: str, key: str) -> None:
        config_mod.set_secret(f"model:{name}", key.strip())
        self._apply()

    def set_human(self, provider: str | None = None, key: str | None = None, daily_limit: float | None = None) -> None:
        if provider is not None:
            if provider not in HUMAN_PRESETS:
                raise ValueError(f"сервис {provider!r} не знаю; есть: {', '.join(HUMAN_PRESETS)}")
            self.cfg.human_provider = provider
            self.cfg.human_base_url = HUMAN_PRESETS[provider]
        if key is not None:
            key = key.strip()
            config_mod.set_secret("human", key)
            self.cfg.human_key_hint = f"{key[:4]}…{key[-4:]}" if key else ""
            self.legacy_human_key = False
            self.human_status = {}
        if daily_limit is not None:
            if daily_limit < 0:
                raise ValueError("лимит не может быть отрицательным")
            self.cfg.human_daily_limit = daily_limit
        self._apply()

    def set_paused(self, paused: bool) -> None:
        self.service.paused = paused

    def check_human(self) -> dict:
        """Asks the workers' service for the balance: proves the key works. Cached for the menu."""
        human = self.service.chain.layer("human")
        if not human or not human.available():
            self.human_status = {"ok": False, "error": "нет ключа"}
        else:
            try:
                self.human_status = {"ok": True, "balance": asyncio.run(self.service.balance())}
            except Exception as e:
                self.human_status = {"ok": False, "error": str(e)[:120]}
        return self.human_status

    def human_line(self) -> str:
        st = getattr(self, "human_status", None) or {}
        name = HUMAN_TITLES.get(self.cfg.human_provider, self.cfg.human_provider)
        if not self.cfg.human_api_key:
            return f"Люди ({name}): введите ключ ещё раз" if self.legacy_human_key else f"Люди ({name}): нет ключа"
        if st.get("ok"):
            cur = HUMAN_CURRENCY.get(self.cfg.human_provider, "$")
            low = " — пора пополнить" if st["balance"] < self.cfg.human_low_balance else ""
            return f"Люди ({name}): ✓ баланс {money(st['balance'], cur)}{low}"
        if st:
            return f"Люди ({name}): ⚠ {st.get('error', '')[:40]}"
        return f"Люди ({name}): ключ есть"

    def move_layer(self, layer_id: str, step: int) -> None:
        layers = self.cfg.layers
        if layer_id not in layers:
            return
        i = layers.index(layer_id)
        j = max(0, min(len(layers) - 1, i + step))
        layers.insert(j, layers.pop(i))
        self._apply()

    # --- keys --------------------------------------------------------------------
    def add_key(self, secret: str, base_url: str = "") -> KeyEntry:
        """Detects the provider, stores the secret in the Keychain and remembers the provider's model list."""
        secret = secret.strip()
        if not secret:
            raise ValueError("пустой ключ")
        self.detecting = "начинаю поиск поставщика"
        try:
            found = detect_sync(secret, base_url.strip(), progress=lambda text: setattr(self, "detecting", text))
        finally:
            self.detecting = ""
        taken = {k.name for k in self.cfg.keys}
        base_name = urlparse(found.base_url).hostname or "custom" if found.provider == "custom" else found.provider
        name = base_name if base_name not in taken else next(
            f"{base_name}-{i}" for i in range(2, 99) if f"{base_name}-{i}" not in taken)
        entry = KeyEntry(name=name, provider=found.provider, base_url=found.base_url,
                         hint=f"{secret[:5]}…{secret[-4:]}", listable=found.listable,
                         models=[[m.id, m.vision, m.price_in, m.price_out] for m in found.models])
        config_mod.set_secret(f"key:{name}", secret)
        self.cfg.keys.append(entry)
        self._apply()
        return entry

    def key(self, name: str) -> KeyEntry:
        entry = next((k for k in self.cfg.keys if k.name == name), None)
        if entry is None:
            raise ValueError(f"ключа {name!r} нет")
        return entry

    def refresh_key(self, name: str) -> KeyEntry:
        entry = self.key(name)
        found = detect_sync(config_mod.get_secret(f"key:{name}"), entry.base_url if entry.provider == "custom" else "")
        entry.models = [[m.id, m.vision, m.price_in, m.price_out] for m in found.models]
        self._apply()
        return entry

    def remove_key(self, name: str) -> None:
        self.cfg.layers = [x for x in self.cfg.layers
                           if not any(x == f"vlm:{m.name}" for m in self.cfg.models if m.key == name)]
        self.cfg.models = [m for m in self.cfg.models if m.key != name]
        self.cfg.keys = [k for k in self.cfg.keys if k.name != name]
        config_mod.set_secret(f"key:{name}", "")
        self._apply()

    def key_models(self, name: str, vision_only: bool = True) -> list[tuple[str, bool]]:
        """(model id, enabled) for the menu."""
        enabled = {m.model for m in self.cfg.models if m.key == name}
        return [(mid, mid in enabled) for mid, vision, *_ in self.key(name).models if vision or not vision_only]

    def toggle_key_model(self, key_name: str, model_id: str) -> None:
        entry = self.key(key_name)
        existing = next((m for m in self.cfg.models if m.key == key_name and m.model == model_id), None)
        if existing:
            self.cfg.models.remove(existing)
            self.cfg.layers = [x for x in self.cfg.layers if x != f"vlm:{existing.name}"]
        else:
            price = next(((p_in, p_out) for mid, _, p_in, p_out in entry.models if mid == model_id), (0.0, 0.0))
            spec = ModelSpec(name=f"{key_name}/{model_id}", provider=entry.provider, base_url=entry.base_url,
                             model=model_id, key=key_name, price_in=price[0], price_out=price[1])
            self.cfg.models.append(spec)
            self._add_model_layer(spec.name)
        self._apply()

    # --- comparing models ----------------------------------------------------------
    def bench_cases(self, per_kind: int = 5) -> list[dict]:
        return testset.generated(per_kind) + testset.downloaded()

    def bench_specs(self, key_name: str | None = None) -> list[ModelSpec]:
        """Every vision model of one key (enabled or not), or every enabled model."""
        if key_name is None:
            return [m for m in self.cfg.models if m.ready()]
        entry = self.key(key_name)
        secret = config_mod.get_secret(f"key:{key_name}")
        return [ModelSpec(name=f"{key_name}/{mid}", provider=entry.provider, base_url=entry.base_url, model=mid,
                          key=key_name, api_key=secret, price_in=p_in, price_out=p_out)
                for mid, vision, p_in, p_out in entry.models if vision]

    def bench_estimate(self, specs: list[ModelSpec], votes: int = 1) -> tuple[int, float | None]:
        n = len(self.bench_cases())
        return n * len(specs), bench.estimate_usd(specs, n, votes)

    def run_bench(self, specs: list[ModelSpec], votes: int = 1, progress=None, stop=None) -> Path:
        cases = self.bench_cases()
        results = asyncio.run(bench.run(specs, cases, votes=votes, progress=progress, stop=stop))
        if not results:
            raise RuntimeError("сравнение остановлено до первого ответа")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = Path(config_mod.APP_DIR) / "reports" / f"compare-{stamp}.html"
        (out.with_suffix(".json")).parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, default=str))
        return bench.report_html(results, out, votes)

    def start_bench(self, key_name: str | None) -> dict:
        """Runs the comparison in a background thread; progress and results land in self.bench."""
        if self.bench.get("running"):
            raise ValueError("сравнение уже идёт")
        specs = self.bench_specs(key_name)
        if not specs:
            raise ValueError("нет моделей с картинками для сравнения")
        runs, _ = self.bench_estimate(specs)
        state = {"running": True, "done": 0, "total": runs, "key": key_name, "started": time.time(), "stopping": False}
        self.bench = state

        def work():
            try:
                report = self.run_bench(specs, progress=lambda d, t: state.update(done=d, total=t),
                                        stop=lambda: state["stopping"])
                results = json.loads(report.with_suffix(".json").read_text())
                state["report"] = str(report)
                state["rows"] = [{"name": n, "model": results[n]["model"], "accuracy": results[n]["accuracy"],
                                  "seconds": results[n]["seconds"], "usd_per_1000": results[n]["usd_per_1000"],
                                  "errors": results[n]["errors"], "by_set": results[n]["by_set"]}
                                 for n in bench.ranking(results)]
            except Exception as e:
                state["error"] = str(e)[:300]
            state["running"] = False
            state["finished"] = time.time()

        threading.Thread(target=work, daemon=True, name="captcha-stack-bench").start()
        return state

    def test_model(self, key_name: str, model_id: str) -> dict:
        """One text, one click and one grid captcha with a single ask each: does this model work at all, how fast."""
        spec = next((s for s in self.bench_specs(key_name) if s.model == model_id), None)
        if spec is None:
            entry = self.key(key_name)
            spec = ModelSpec(name=f"{key_name}/{model_id}", provider=entry.provider, base_url=entry.base_url,
                             model=model_id, key=key_name, api_key=config_mod.get_secret(f"key:{key_name}"))
        cases = [c for c in testset.generated(1, seed=11) if not c["id"].startswith("gen-math")]
        rows = asyncio.run(bench.run([spec], cases, votes=1))[spec.name]["rows"]
        result = {"at": time.time(), "ok": sum(r["ok"] for r in rows), "n": len(rows),
                  "seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1),
                  "error": next((r["error"] for r in rows if r["error"]), ""),
                  "details": [{"set": r["set"].replace("сгенерированные: ", ""), "ok": r["ok"], "error": r["error"]} for r in rows]}
        self.model_tests[spec.name] = result
        return result

    def set_access(self, required: bool | None = None, regenerate: bool = False) -> None:
        if required is not None:
            self.cfg.require_client_key = required
        if regenerate or (self.cfg.require_client_key and not self.cfg.client_key):
            self.cfg.client_key = "cs-" + secrets.token_urlsafe(18)
        self._apply()

    def check_connection(self) -> dict:
        """Sends a generated text captcha through the app's own 2Captcha-style API, the way a program would."""
        import httpx

        case = next(c for c in testset.generated(1, seed=5) if c["task"]["type"] == "ImageToTextTask")
        started = time.time()
        with httpx.Client(base_url=self.address, timeout=10) as client:
            created = client.post("/createTask", json={"clientKey": self.cfg.client_key, "task": case["task"]}).json()
            if created.get("errorId"):
                return {"ok": False, "error": created.get("errorDescription") or created.get("errorCode")}
            while time.time() - started < 120:
                r = client.post("/getTaskResult", json={"clientKey": self.cfg.client_key, "taskId": created["taskId"]}).json()
                if r.get("status") != "processing":
                    break
                time.sleep(0.3)
        if r.get("errorId"):
            return {"ok": False, "error": r.get("errorDescription") or r.get("errorCode")}
        titles = {x.name: x.title or x.name for x in self.service.chain.layers}
        return {"ok": True, "seconds": round(time.time() - started, 2), "layer": titles.get(r.get("layer"), r.get("layer")),
                "correct": r.get("solution", {}).get("text") == case["expected"]["text"]}

    def alerts(self) -> list[tuple[str, str]]:
        """(id, text) of things worth a notification right now; the menu shows each id once until it clears."""
        out = []
        cur = HUMAN_CURRENCY.get(self.cfg.human_provider, "$")
        st = self.human_status or {}
        if st.get("ok") and st["balance"] < self.cfg.human_low_balance:
            out.append(("low-balance", f"На {HUMAN_TITLES.get(self.cfg.human_provider)} осталось {money(st['balance'], cur)} — пора пополнить."))
        limit = self.cfg.human_daily_limit
        if limit and self.service.chain.state.spent_today("human") >= limit:
            out.append(("human-limit", f"Люди: исчерпан дневной лимит {money(limit, cur)}. До завтра капчи им не отправляются."))
        cloud = self.cfg.cloud_daily_limit_usd
        if cloud and self.spend_today()["cloud_usd"] >= cloud:
            out.append(("cloud-limit", f"Облачные модели: исчерпан дневной лимит ${cloud:g}."))
        from .layers.vlm import LAST_ERRORS

        for name, (_, msg) in LAST_ERRORS.items():
            out.append((f"model-error:{name}:{msg[:40]}", f"Модель {name[4:]} ошибается: {msg[:120]}"))
        if self.bench.get("finished") and not self.bench.get("running"):
            best = (self.bench.get("rows") or [{}])[0].get("model")
            text = f"Сравнение моделей готово. Лучшая: {best}." if best else "Сравнение моделей закончилось."
            out.append((f"bench:{self.bench['finished']}", self.bench.get("error") or text))
        return out

    def stop_bench(self) -> None:
        if self.bench.get("running"):
            self.bench["stopping"] = True

    # --- test sets ------------------------------------------------------------------
    def datasets(self) -> list[dict]:
        root = Path(config_mod.APP_DIR) / "testsets"
        out = []
        for name, info in testset.DOWNLOADABLE.items():
            have = testset.count_cases(root / name)
            out.append({"name": name, **{k: v for k, v in info.items() if k != "files"}, "installed": have,
                        "job": self.dataset_jobs.get(name, {})})
        return out

    def download_dataset(self, name: str) -> None:
        if name not in testset.DOWNLOADABLE:
            raise ValueError(f"набора {name} нет")
        if self.dataset_jobs.get(name, {}).get("running"):
            raise ValueError("уже скачивается")
        job = {"running": True, "done": 0, "total": 0}
        self.dataset_jobs[name] = job

        def work():
            try:
                testset.download(name, Path(config_mod.APP_DIR) / "testsets" / name,
                                 progress=lambda d, t: job.update(done=d, total=t))
            except Exception as e:
                job["error"] = str(e)[:300]
            job["running"] = False

        threading.Thread(target=work, daemon=True, name=f"download-{name}").start()

    def remove_dataset(self, name: str) -> None:
        import shutil

        target = Path(config_mod.APP_DIR) / "testsets" / name
        if name in testset.DOWNLOADABLE and target.exists():
            shutil.rmtree(target)
        self.dataset_jobs.pop(name, None)

    def promote(self, model_names: list[str]) -> None:
        """Puts the given models first among the vision layers, in this order."""
        vlm = [f"vlm:{n}" for n in model_names if any(m.name == n for m in self.cfg.models)]
        others = [x for x in self.cfg.layers if x not in vlm]
        insert_at = next((i for i, x in enumerate(others) if x.startswith("vlm") or x == "human"), len(others))
        self.cfg.layers = others[:insert_at] + vlm + others[insert_at:]
        self._apply()

    def use_best(self, model_name: str) -> None:
        """Enables a compared model if needed and puts it first among the vision layers."""
        if not any(m.name == model_name for m in self.cfg.models):
            key_name, model_id = model_name.split("/", 1)
            self.toggle_key_model(key_name, model_id)
        self.promote([model_name])

    # --- start at login ---------------------------------------------------------
    @staticmethod
    def launch_agent_plist() -> dict:
        # inside the bundled app sys.executable is the app itself
        args = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "captcha_stack.menubar"]
        return {"Label": "app.captcha-stack", "ProgramArguments": args, "RunAtLoad": True, "ProcessType": "Interactive"}

    @staticmethod
    def starts_at_login() -> bool:
        return LAUNCH_AGENT.exists()

    def set_start_at_login(self, on: bool) -> None:
        if on:
            LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
            LAUNCH_AGENT.write_bytes(plistlib.dumps(self.launch_agent_plist()))
        elif LAUNCH_AGENT.exists():
            LAUNCH_AGENT.unlink()
