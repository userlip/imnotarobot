import os
from dataclasses import dataclass, field
from pathlib import Path


def _dashscope_key() -> str:
    path = Path.home() / ".config" / "dashscope.env"
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        if line.startswith("DASHSCOPE_API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip()
    return ""


@dataclass
class Config:
    client_key: str = ""
    layers: list[str] = field(default_factory=lambda: ["local", "vlm", "human"])
    ocr_min_confidence: float = 0.9
    vlm_min_confidence: float = 0.7
    vlm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_api_key: str = ""
    vlm_model: str = "qwen3-vl-plus"
    vlm_timeout: float = 60.0
    human_base_url: str = "https://api.2captcha.com"
    human_api_key: str = ""
    human_timeout: float = 180.0
    human_poll_interval: float = 5.0
    # a layer is skipped for a captcha kind once it has this many reports and fails at least half of them
    skip_after_reports: int = 5
    task_ttl: float = 600.0

    @classmethod
    def from_env(cls) -> "Config":
        e = os.environ.get
        c = cls()
        c.client_key = e("CLIENT_KEY", c.client_key)
        if e("LAYERS"):
            c.layers = [x.strip() for x in e("LAYERS").split(",") if x.strip()]
        c.ocr_min_confidence = float(e("OCR_MIN_CONFIDENCE", c.ocr_min_confidence))
        c.vlm_min_confidence = float(e("VLM_MIN_CONFIDENCE", c.vlm_min_confidence))
        c.vlm_base_url = e("VLM_BASE_URL", c.vlm_base_url)
        c.vlm_api_key = e("VLM_API_KEY") or _dashscope_key()
        c.vlm_model = e("VLM_MODEL", c.vlm_model)
        c.human_base_url = e("HUMAN_BASE_URL", c.human_base_url)
        c.human_api_key = e("HUMAN_API_KEY", c.human_api_key)
        c.human_timeout = float(e("HUMAN_TIMEOUT", c.human_timeout))
        return c
