import base64
from dataclasses import dataclass, field
from typing import Any

import httpx

# protocol: "openai" = POST {base}/chat/completions; "anthropic" = POST {base}/messages
PRESETS: dict[str, dict[str, Any]] = {
    "dashscope": {"protocol": "openai", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                  "model": "qwen3-vl-plus", "price_in": 0.20, "price_out": 1.60},
    "openai": {"protocol": "openai", "base_url": "https://api.openai.com/v1", "model": ""},
    "gemini": {"protocol": "openai", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "model": "gemini-2.5-flash"},
    "anthropic": {"protocol": "anthropic", "base_url": "https://api.anthropic.com/v1", "model": ""},
    "openrouter": {"protocol": "openai", "base_url": "https://openrouter.ai/api/v1", "model": ""},
    "ollama": {"protocol": "openai", "base_url": "http://localhost:11434/v1", "model": "", "keyless": True},
    "custom": {"protocol": "openai", "base_url": "", "model": ""},
}


class ProviderError(Exception):
    pass


@dataclass
class ModelSpec:
    name: str
    provider: str = "custom"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    protocol: str = ""
    votes: int = 3
    min_confidence: float = 0.66
    timeout: float = 90.0
    # USD per 1M tokens; 0 means the cost is not tracked
    price_in: float = 0.0
    price_out: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        preset = PRESETS.get(self.provider, PRESETS["custom"])
        self.protocol = self.protocol or preset["protocol"]
        self.base_url = (self.base_url or preset["base_url"]).rstrip("/")
        self.model = self.model or preset["model"]
        self.price_in = self.price_in or preset.get("price_in", 0.0)
        self.price_out = self.price_out or preset.get("price_out", 0.0)

    @property
    def keyless(self) -> bool:
        return bool(PRESETS.get(self.provider, {}).get("keyless"))

    def ready(self) -> bool:
        return bool(self.model and self.base_url and (self.api_key or self.keyless))

    def cost(self, usage: dict[str, int]) -> float:
        return (usage.get("input", 0) * self.price_in + usage.get("output", 0) * self.price_out) / 1e6


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


async def ask(spec: ModelSpec, images: list[bytes], prompt: str,
              transport: httpx.AsyncBaseTransport | None = None) -> tuple[str, dict[str, int]]:
    """Sends PNG images plus a prompt; returns the reply text and {"input", "output"} token counts."""
    async with httpx.AsyncClient(timeout=spec.timeout, transport=transport) as client:
        if spec.protocol == "anthropic":
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(i)}}
                       for i in images] + [{"type": "text", "text": prompt}]
            r = await client.post(
                f"{spec.base_url}/messages",
                headers={"x-api-key": spec.api_key, "anthropic-version": "2023-06-01"},
                json={"model": spec.model, "max_tokens": 2048, "messages": [{"role": "user", "content": content}]},
            )
            j = _json_or_raise(r)
            text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
            u = j.get("usage", {})
            return text, {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0)}
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(i)}"}} for i in images]
        content.append({"type": "text", "text": prompt})
        headers = {"Authorization": f"Bearer {spec.api_key}"} if spec.api_key else {}
        r = await client.post(f"{spec.base_url}/chat/completions", headers=headers,
                              json={"model": spec.model, "messages": [{"role": "user", "content": content}],
                                    **spec.extra})
        j = _json_or_raise(r)
        text = j["choices"][0]["message"].get("content") or ""
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text)
        u = j.get("usage", {})
        return text, {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0)}


def _json_or_raise(r: httpx.Response) -> dict[str, Any]:
    if r.status_code != 200:
        raise ProviderError(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except ValueError as e:
        raise ProviderError(f"not JSON: {r.text[:200]}") from e
