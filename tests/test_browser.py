import asyncio
import os

import pytest

from imnotarobot.config import Config, ModelSpec
from imnotarobot.layers import Unavailable, Unsolvable
from imnotarobot.layers.browser import BrowserLayer, ask_models, proxy_of, widget_page
from imnotarobot.service import build_layers
from helpers import FakeLayer


def test_widget_page_escapes_the_key_and_picks_the_script():
    page = widget_page("recaptcha", {"websiteKey": 'k"><script>'})
    assert "&quot;&gt;&lt;script&gt;" in page and "recaptcha/api.js" in page
    assert "js.hcaptcha.com" in widget_page("hcaptcha", {"websiteKey": "k"})
    assert 'data-action="login"' in widget_page("turnstile", {"websiteKey": "k", "action": "login"})
    assert "render=k" in widget_page("recaptcha3", {"websiteKey": "k"})
    assert "grecaptcha.execute()" in widget_page("recaptcha", {"websiteKey": "k", "isInvisible": True})
    with pytest.raises(Unsolvable):
        widget_page("hcaptcha", {})


def test_proxy_is_passed_to_the_browser():
    assert proxy_of({}) is None
    assert proxy_of({"proxyType": "socks5", "proxyAddress": "1.2.3.4", "proxyPort": 1080, "proxyLogin": "u",
                     "proxyPassword": "p"}) == {"server": "socks5://1.2.3.4:1080", "username": "u", "password": "p"}


def test_models_are_asked_in_order_and_a_silent_service_is_skipped():
    class Grid(FakeLayer):
        async def solve(self, task):
            r = await super().solve(task)
            r.solution = {"click": [int(r.solution["text"])]}
            return r

    low = Grid("low", [Unavailable("All accounts failed")], types=("GridTask",))
    high = Grid("high", [("2", 1.0)], types=("GridTask",))
    other = Grid("other", [("3", 1.0)], types=("GridTask",))
    low.upstream = high.upstream = "antigravity"
    assert asyncio.run(ask_models([low, high, other], {"type": "GridTask"})) == {"click": [3]}
    assert high.calls == 0


def test_browser_layer_gets_every_enabled_model():
    cfg = Config(layers=["local", "browser", "vlm:g", "human"],
                 models=[ModelSpec(name="g", provider="gemini", api_key="k")])
    layers = build_layers(cfg)
    browser = next(x for x in layers if x.name == "browser")
    assert browser.models is not None and browser.supports("RecaptchaV3TaskProxyless")
    assert not browser.supports("ImageToTextTask")


@pytest.mark.skipif(not os.environ.get("CAPTCHA_LIVE"), reason="CAPTCHA_LIVE=1 opens Chrome against official test keys")
@pytest.mark.parametrize("task", [
    {"type": "TurnstileTaskProxyless", "websiteURL": "https://example.com/", "websiteKey": "1x00000000000000000000AA"},
    {"type": "HCaptchaTaskProxyless", "websiteURL": "https://example.com/", "websiteKey": "10000000-ffff-ffff-ffff-000000000001"},
])
def test_live_test_keys_give_a_token(task):
    r = asyncio.run(BrowserLayer(None).solve(task))
    assert r.solution["token"]
