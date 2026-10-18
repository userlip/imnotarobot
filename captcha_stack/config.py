"""Settings live in a JSON file; API keys live in the macOS Keychain, never in the file."""

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .providers import ModelSpec

APP_DIR = Path(os.environ.get("CAPTCHA_STACK_HOME", Path.home() / "Library" / "Application Support" / "captcha-stack"))
KEYCHAIN_SERVICE = "captcha-stack"
HUMAN_PRESETS = {"2captcha": "https://api.2captcha.com", "rucaptcha": "https://api.rucaptcha.com"}


# Secrets go through Apple's own `security` tool. An item written by the app itself is tied to that build's code
# hash, so every rebuild lost access to it; an item written by /usr/bin/security stays readable by that tool.
# Trade-off: the value is briefly visible in the tool's argument list while it is being saved.
SECURITY = "/usr/bin/security"
SECRET_SERVICE = "Captcha Stack"
LEGACY_SERVICE = KEYCHAIN_SERVICE


def get_secret(account: str) -> str:
    r = subprocess.run([SECURITY, "find-generic-password", "-s", SECRET_SERVICE, "-a", account, "-w"],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def set_secret(account: str, value: str) -> None:
    if value:
        r = subprocess.run([SECURITY, "add-generic-password", "-U", "-s", SECRET_SERVICE, "-a", account,
                            "-l", f"Captcha Stack — {account}", "-w", value], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"Связка ключей не сохранила ключ: {r.stderr.strip()[:120]}")
    else:
        subprocess.run([SECURITY, "delete-generic-password", "-s", SECRET_SERVICE, "-a", account],
                       capture_output=True, check=False)


def legacy_secret_exists(account: str) -> bool:
    """A key saved by an older build that this build cannot read: the user has to enter it once more."""
    r = subprocess.run([SECURITY, "find-generic-password", "-s", LEGACY_SERVICE, "-a", account],
                       capture_output=True, check=False)
    return r.returncode == 0


def model_key(spec: ModelSpec) -> str:
    if spec.key:
        return get_secret(f"key:{spec.key}")
    return get_secret(f"model:{spec.name}") or os.environ.get(f"{spec.name.upper().replace('-', '_')}_API_KEY", "")


@dataclass
class KeyEntry:
    """An API key the user added; the secret itself is in the Keychain under key:<name>."""

    name: str
    provider: str
    base_url: str = ""
    hint: str = ""
    # model ids that accept images, as last listed by the provider: [id, vision, price_in, price_out]
    models: list[list] = field(default_factory=list)
    listable: bool = True


@dataclass
class Config:
    client_key: str = ""
    port: int = 8765
    layers: list[str] = field(default_factory=lambda: ["local", "vlm", "human"])
    ocr_min_confidence: float = 0.9
    keys: list[KeyEntry] = field(default_factory=list)
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
        names = {f.name for f in fields(cls)} - {"models", "keys", "human_api_key"}
        cfg = cls(**{k: v for k, v in raw.items() if k in names})
        cfg.keys = [KeyEntry(**k) for k in raw.get("keys", [])]
        cfg.models = [ModelSpec(**m) for m in raw.get("models", [])]
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
