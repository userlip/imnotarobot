"""Checkbox captchas on their vendors' official demo pages: how often the browser step passes and how fast.

    python -m imnotarobot.demo_bench --runs 5
    python -m imnotarobot.demo_bench --only recaptcha-v2 hcaptcha --runs 10 --json results.json

Uses the vision models switched on in the app. reCAPTCHA v2 tokens are checked by Google's own demo
form; the others count as passed when the widget hands out a token.
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from typing import Any

import httpx

from .config import Config
from .service import build_layers

DEMOS: dict[str, dict[str, Any]] = {
    "recaptcha-v2": {"type": "RecaptchaV2TaskProxyless", "websiteURL": "https://www.google.com/recaptcha/api2/demo",
                     "websiteKey": "6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-"},
    "recaptcha-v3": {"type": "RecaptchaV3TaskProxyless", "websiteURL": "https://www.google.com/recaptcha/api2/demo",
                     "websiteKey": "6LcR_okUAAAAAPYrPe-HK_0RULO1aZM15ENyM-Mf", "pageAction": "homepage"},
    "hcaptcha": {"type": "HCaptchaTaskProxyless", "websiteURL": "https://accounts.hcaptcha.com/demo",
                 "websiteKey": "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"},
    # Cloudflare's documented test key that always passes: checks the plumbing, not the solving
    "turnstile-test": {"type": "TurnstileTaskProxyless", "websiteURL": "https://example.com/",
                       "websiteKey": "1x00000000000000000000AA"},
}


def google_accepts(token: str) -> bool:
    r = httpx.post(DEMOS["recaptcha-v2"]["websiteURL"], data={"g-recaptcha-response": token}, timeout=30)
    return "Verification Success" in r.text


async def run(names: list[str], runs: int) -> dict[str, Any]:
    browser = next(x for x in build_layers(Config.load()) if x.name == "browser")
    out: dict[str, Any] = {}
    for name in names:
        rows = []
        for i in range(runs):
            started = time.monotonic()
            try:
                token = (await browser.solve(DEMOS[name])).solution["token"]
                ok = google_accepts(token) if name == "recaptcha-v2" else bool(token)
                error = "" if ok else "token rejected"
            except Exception as e:
                ok, error = False, str(e)[:200]
            rows.append({"ok": ok, "seconds": round(time.monotonic() - started, 1), "error": error})
            print(f"{name} {i + 1}/{runs}: {'ok' if ok else 'FAIL'} {rows[-1]['seconds']} s {error}", flush=True)
        passed = [r["seconds"] for r in rows if r["ok"]]
        out[name] = {"passed": len(passed), "runs": runs,
                     "median_seconds": statistics.median(passed) if passed else None, "rows": rows}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="*", choices=list(DEMOS), default=list(DEMOS))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--json", help="also write the results to this file")
    p.add_argument("-v", action="store_true", help="show each picture round")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO if a.v else logging.WARNING)
    results = asyncio.run(run(a.only, a.runs))
    for name, r in results.items():
        median = f"{r['median_seconds']} s" if r["median_seconds"] is not None else "—"
        print(f"{name:15} passed {r['passed']}/{r['runs']}  median {median}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
