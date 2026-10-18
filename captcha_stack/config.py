"""Settings live in a JSON file; API keys live in the macOS Keychain, never in the file."""

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .providers import ModelSpec

APP_DIR = Path(os.environ.get("CAPTCHA_STACK_HOME", Path.home() / "Library" / "Application Support" / "captcha-stack"))
KEYCHAIN_SERVICE = "captcha-stack"
HUMAN_PRESETS = {"2captcha": "https://api.2captcha.com", "rucaptcha": "https://api.rucaptcha.com"}


def get_secret(account: str) -> str:
    try:
        import keyring

        return keyring.get_password(KEYCHAIN_SERVICE, account) or ""
    except Exception:
        return ""


def set_secret(account: str, value: str) -> None:
    import keyring

    if value:
        keyring.set_password(KEYCHAIN_SERVICE, account, value)
    else:
        try:
            keyring.delete_password(KEYCHAIN_SERVICE, account)
        except keyring.errors.PasswordDeleteError:
            pass


def _dashscope_key() -> str:
    path = Path.home() / ".config" / "dashscope.env"
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        if line.startswith("DASHSCOPE_API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip()
    return ""


def model_key(spec: ModelSpec) -> str:
    return (get_secret(f"model:{spec.name}")
            or os.environ.get(f"{spec.name.upper().replace('-', '_')}_API_KEY", "")
            or (_dashscope_key() if spec.provider == "dashscope" else ""))


@dataclass
class Config:
    client_key: str = ""
    port: int = 8765
    layers: list[str] = field(default_factory=lambda: ["local", "vlm", "human"])
    ocr_min_confidence: float = 0.9
    models: list[ModelSpec] = field(default_factory=list)
    human_provider: str = "rucaptcha"
    human_base_url: str = ""
    human_api_key: str = ""
    human_timeout: float = 180.0
    human_poll_interval: float = 5.0
    # 0 = no limit
    human_daily_limit_usd: float = 1.0
    # a layer is skipped for a captcha kind once it has this many reports and fails at least half of them
    skip_after_reports: int = 5
    task_ttl: float = 600.0
    state_path: str = ""

    def __post_init__(self):
        self.human_base_url = (self.human_base_url or HUMAN_PRESETS.get(self.human_provider, "")).rstrip("/")
        self.state_path = self.state_path or str(APP_DIR / "state.json")

    @classmethod
    def load(cls, path: Path | None = None, with_secrets: bool = True) -> "Config":
        path = path or APP_DIR / "config.json"
        raw: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        names = {f.name for f in fields(cls)} - {"models", "human_api_key"}
        cfg = cls(**{k: v for k, v in raw.items() if k in names})
        if "models" in raw:
            cfg.models = [ModelSpec(**m) for m in raw["models"]]
        elif _dashscope_key():
            cfg.models = [ModelSpec(name="qwen", provider="dashscope")]
        if with_secrets:
            for m in cfg.models:
                m.api_key = m.api_key or model_key(m)
            cfg.human_api_key = get_secret("human") or os.environ.get("HUMAN_API_KEY", "")
        if os.environ.get("LAYERS"):
            cfg.layers = [x.strip() for x in os.environ["LAYERS"].split(",") if x.strip()]
        cfg.client_key = os.environ.get("CLIENT_KEY", cfg.client_key)
        return cfg

    def save(self, path: Path | None = None) -> None:
        path = path or APP_DIR / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("human_api_key")
        for m in data["models"]:
            m.pop("api_key", None)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(path)
