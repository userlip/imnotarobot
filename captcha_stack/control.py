"""Everything the menu bar does, without the GUI, so it can be tested."""

import asyncio
import plistlib
import sys
import threading
from pathlib import Path

import uvicorn

from . import config as config_mod
from .config import HUMAN_PRESETS, Config
from .providers import PRESETS, ModelSpec
from .server import create_app
from .service import Service

LAUNCH_AGENT = Path.home() / "Library" / "LaunchAgents" / "app.captcha-stack.plist"
LAYER_TITLES = {"local": "Локальная модель", "human": "Люди"}


class Controller:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.load()
        self.service = Service(self.cfg)
        self._server: uvicorn.Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- server ---------------------------------------------------------------
    def start_server(self) -> None:
        app = create_app(service=self.service)
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.cfg.port, log_level="warning"))

        def run():
            self._loop = asyncio.new_event_loop()
            self._loop.run_until_complete(self._server.serve())

        threading.Thread(target=run, daemon=True, name="captcha-stack-server").start()

    def stop_server(self) -> None:
        if self._server:
            self._server.should_exit = True

    @property
    def address(self) -> str:
        return f"http://127.0.0.1:{self.cfg.port}"

    def _apply(self) -> None:
        self.cfg.save()
        fresh = Config.load()
        fresh.state_path = self.cfg.state_path
        self.cfg = fresh
        self.service.reconfigure(fresh)

    # --- what the menu shows --------------------------------------------------
    def today_line(self) -> str:
        day = self.service.today()
        solved = int(sum(v.get("solved", 0) for v in day.values()))
        usd = sum(v.get("usd", 0.0) for v in day.values())
        return f"Сегодня: решено {solved} · ${usd:.3f}"

    def layer_items(self) -> list[tuple[str, str, bool]]:
        """(layer id, title, enabled) for every layer that can be switched."""
        items = [("local", LAYER_TITLES["local"], "local" in self.cfg.layers)]
        vlm_all = "vlm" in self.cfg.layers
        for m in self.cfg.models:
            items.append((f"vlm:{m.name}", f"Модель {m.name} ({m.provider})", vlm_all or f"vlm:{m.name}" in self.cfg.layers))
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
        if "vlm" not in self.cfg.layers:
            self.cfg.layers = [x for x in self.cfg.layers if x != "human"] + [f"vlm:{name}"] + \
                (["human"] if "human" in self.cfg.layers else [])
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
            config_mod.set_secret("human", key.strip())
        if daily_limit is not None:
            if daily_limit < 0:
                raise ValueError("лимит не может быть отрицательным")
            self.cfg.human_daily_limit_usd = daily_limit
        self._apply()

    def set_paused(self, paused: bool) -> None:
        self.service.paused = paused

    def balance_line(self) -> str:
        human = self.service.chain.layer("human")
        if not human or not human.available():
            return "Баланс: нет ключа"
        future = asyncio.run_coroutine_threadsafe(self.service.balance(), self._loop) if self._loop else None
        try:
            value = future.result(timeout=15) if future else asyncio.run(self.service.balance())
        except Exception as e:
            return f"Баланс: ошибка ({str(e)[:40]})"
        return f"Баланс: ${value:.2f}"

    # --- start at login ---------------------------------------------------------
    @staticmethod
    def launch_agent_plist() -> dict:
        return {"Label": "app.captcha-stack", "ProgramArguments": [sys.executable, "-m", "captcha_stack.menubar"],
                "RunAtLoad": True, "ProcessType": "Interactive"}

    @staticmethod
    def starts_at_login() -> bool:
        return LAUNCH_AGENT.exists()

    def set_start_at_login(self, on: bool) -> None:
        if on:
            LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
            LAUNCH_AGENT.write_bytes(plistlib.dumps(self.launch_agent_plist()))
        elif LAUNCH_AGENT.exists():
            LAUNCH_AGENT.unlink()
