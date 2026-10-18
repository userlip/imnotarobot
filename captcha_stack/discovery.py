"""From a bare API key to a provider and its vision-capable models."""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .providers import PRESETS

# prefixes are a hint only; every guess is confirmed by an authenticated request
PREFIX_HINTS = [("sk-ant-", "anthropic"), ("AIza", "gemini"), ("sk-or-", "openrouter"), ("xai-", "xai"), ("gsk_", "groq")]
# tried in this order when the prefix says nothing
PROBE_ORDER = ["openai", "dashscope", "dashscope-intl", "mistral", "openrouter", "xai", "groq", "anthropic", "gemini"]

# names of models that take images, for providers whose model list does not say so
VISION_PATTERNS = {
    "openai": r"^(gpt-4o|gpt-4\.1|gpt-5|o1(?!-mini)|o3|o4|chatgpt-4o)",
    "dashscope": r"(vl|omni|qvq)",
    "dashscope-intl": r"(vl|omni|qvq)",
    "gemini": r"^gemini",
    "anthropic": r"^claude",
    "xai": r"(vision|grok-4|grok-5)",
    "groq": r"(llama-4|vision)",
    "mistral": r"(pixtral|mistral-(medium|small|large)-)",
}
NOT_CHAT = r"(embed|tts|whisper|transcri|audio|realtime|dall-e|imagen|veo|moderation|search|image-gen|-image$)"


@dataclass
class ModelInfo:
    id: str
    vision: bool
    # USD per 1M tokens when the provider publishes it
    price_in: float = 0.0
    price_out: float = 0.0
    sure: bool = False


@dataclass
class Detected:
    provider: str
    base_url: str
    models: list[ModelInfo] = field(default_factory=list)
    # False when the provider has no model list (resellers): the model name must be typed in
    listable: bool = True


class DetectError(Exception):
    pass


def _headers(provider: str, key: str) -> dict[str, str]:
    if PRESETS[provider]["protocol"] == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {key}"}


def _parse_models(provider: str, j: dict[str, Any]) -> list[ModelInfo]:
    out = []
    pattern = VISION_PATTERNS.get(provider)
    for m in j.get("data", j.get("models", [])):
        mid = str(m.get("id") or m.get("name") or "").removeprefix("models/")
        if not mid or re.search(NOT_CHAT, mid, re.I):
            continue
        arch = m.get("architecture") or {}
        caps = m.get("capabilities") or {}
        if "input_modalities" in arch:
            info = ModelInfo(mid, "image" in arch["input_modalities"], sure=True)
        elif "vision" in caps:
            info = ModelInfo(mid, bool(caps["vision"]), sure=True)
        else:
            info = ModelInfo(mid, bool(pattern and re.search(pattern, mid, re.I)) if pattern else True)
        pricing = m.get("pricing") or {}
        try:
            info.price_in = float(pricing.get("prompt", 0)) * 1e6
            info.price_out = float(pricing.get("completion", 0)) * 1e6
        except (TypeError, ValueError):
            pass
        out.append(info)
    return sorted(out, key=lambda x: (not x.vision, x.id))


async def _try(client: httpx.AsyncClient, provider: str, key: str, base_url: str) -> Detected | None:
    preset = PRESETS[provider]
    base = (base_url or preset["base_url"]).rstrip("/")
    if provider == "openrouter":
        check = await client.get(f"{base}/key", headers=_headers(provider, key))
        if check.status_code != 200:
            return None
        r = await client.get(f"{base}/models")
    else:
        r = await client.get(f"{base}/models", headers=_headers(provider, key))
    if r.status_code == 200:
        try:
            return Detected(provider, base, _parse_models(provider, r.json()))
        except ValueError:
            return None
    return None


async def detect(key: str, base_url: str = "", transport: httpx.AsyncBaseTransport | None = None) -> Detected:
    """Finds who issued the key. With base_url (a reseller or self-hosted server) only that address is checked."""
    key = key.strip()
    async with httpx.AsyncClient(timeout=20, transport=transport) as client:
        if base_url:
            base = base_url.rstrip("/")
            r = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 200:
                return Detected("custom", base, _parse_models("custom", r.json()))
            if r.status_code in (401, 403):
                raise DetectError(f"{base} не принял ключ (HTTP {r.status_code})")
            probe = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={})
            if probe.status_code in (401, 403):
                raise DetectError(f"{base} не принял ключ (HTTP {probe.status_code})")
            return Detected("custom", base, [], listable=False)
        hinted = [p for prefix, p in PREFIX_HINTS if key.startswith(prefix)]
        order = hinted + [p for p in PROBE_ORDER if p not in hinted]
        for provider in order:
            try:
                found = await _try(client, provider, key, "")
            except httpx.HTTPError:
                continue
            if found:
                return found
    raise DetectError("ни один известный поставщик не принял этот ключ; если это посредник — укажите его адрес")


def detect_sync(key: str, base_url: str = "") -> Detected:
    return asyncio.run(detect(key, base_url))
