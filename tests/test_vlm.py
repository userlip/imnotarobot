import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from imnotarobot import imaging
from imnotarobot.layers import Unsolvable, VlmLayer
from imnotarobot.layers.vlm import last_json
from imnotarobot.providers import ModelSpec, ask
from helpers import blank_image


def run(coro):
    return asyncio.run(coro)


class FakeModel:
    """Answers OpenAI- or Anthropic-style requests with scripted replies and records what it was sent."""

    def __init__(self, replies, protocol="openai"):
        self.replies = list(replies)
        self.protocol = protocol
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if self.protocol == "anthropic":
            return httpx.Response(200, json={"content": [{"type": "text", "text": reply}],
                                             "usage": {"input_tokens": 1000, "output_tokens": 100}})
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})

    def layer(self, votes=3, **spec):
        s = ModelSpec(name="m", provider=spec.pop("provider", "dashscope"), api_key="k", votes=votes, **spec)
        return VlmLayer(s, transport=httpx.MockTransport(self.handler))


def images_sent(request) -> list[Image.Image]:
    out = []
    for part in request["body"]["messages"][0]["content"]:
        if part.get("type") == "image_url":
            out.append(Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))))
    return out


def test_two_agreeing_answers_stop_early_with_full_confidence():
    m = FakeModel(['I see "k7m2p". {"text": "K7M2P"}'])
    r = run(m.layer().solve({"type": "ImageToTextTask", "body": blank_image()}))
    assert (r.solution, r.confidence, len(m.requests)) == ({"text": "K7M2P"}, 1.0, 2)
    assert r.cost == pytest.approx(2 * (1000 * 0.20 + 100 * 1.60) / 1e6)


def test_disagreement_asks_a_third_time_and_takes_the_majority():
    m = FakeModel(['{"text": "abc"}', '{"text": "abd"}', '{"text": "abc"}'])
    r = run(m.layer().solve({"type": "ImageToTextTask", "body": blank_image()}))
    assert (r.solution, round(r.confidence, 2), len(m.requests)) == ({"text": "abc"}, 0.67, 3)


def test_three_different_answers_give_low_confidence():
    m = FakeModel(['{"text": "a1"}', '{"text": "b2"}', '{"text": "c3"}'])
    assert run(m.layer().solve({"type": "ImageToTextTask", "body": blank_image()})).confidence < 0.5


def test_grid_is_numbered_upscaled_and_majority_voted_per_tile():
    m = FakeModel(['1: bus -> yes\n{"tiles": [1, 4]}', '{"tiles": [1, 4, 9]}', '{"tiles": [1, 4]}'])
    r = run(m.layer().solve({"type": "GridTask", "body": blank_image((150, 150)), "comment": "select buses",
                             "rows": 3, "columns": 3}))
    assert r.solution == {"click": [1, 4]}
    sent = images_sent(m.requests[0])[0]
    assert sent.size == (450, 450)
    assert sent.getpixel((150, 300))[:3] == (255, 0, 0)
    prompt = m.requests[0]["body"]["messages"][0]["content"][-1]["text"]
    assert "3 rows x 3 columns" in prompt and "select buses" in prompt and "red number" in prompt


def test_coordinates_get_a_ruler_copy_and_close_points_are_merged():
    m = FakeModel(['{"points": [[100, 40]]}', '{"points": [[104, 42]]}'])
    r = run(m.layer(click_mode="rulers").solve({"type": "CoordinatesTask", "body": blank_image((200, 100)), "comment": "click the cat"}))
    assert r.solution == {"coordinates": [{"x": 102, "y": 41}]} and r.confidence == 1.0
    plain, ruler = images_sent(m.requests[0])
    assert ruler.width > plain.width and ruler.height > plain.height


def test_points_outside_the_image_are_dropped():
    m = FakeModel(['{"points": [[900, 900]]}'])
    with pytest.raises(Unsolvable, match="нет ответа|не указала точек"):
        run(m.layer(votes=1, click_mode="rulers").solve({"type": "CoordinatesTask", "body": blank_image((200, 100))}))


def test_failed_calls_count_against_confidence():
    m = FakeModel(['{"text": "abc"}', "sorry, I can't", '{"text": "abd"}'])
    r = run(m.layer().solve({"type": "ImageToTextTask", "body": blank_image()}))
    assert r.confidence <= 1 / 3 + 1e-9 and r.meta["errors"]


def test_single_vote_is_trusted_as_configured():
    m = FakeModel(['{"text": "abc"}'])
    assert run(m.layer(votes=1).solve({"type": "ImageToTextTask", "body": blank_image()})).confidence == 1.0


def test_anthropic_protocol_and_headers():
    m = FakeModel(['{"text": "abc"}'], protocol="anthropic")
    spec = ModelSpec(name="claude", provider="anthropic", model="claude-x", api_key="sk-ant")
    text, usage = run(ask(spec, [b"\x89PNG"], "hi", httpx.MockTransport(m.handler)))
    req = m.requests[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["x-api-key"] == "sk-ant" and req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["body"]["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    assert (text, usage) == ('{"text": "abc"}', {"input": 1000, "output": 100})


@pytest.mark.parametrize("provider,url", [
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions"),
    ("openai", "https://api.openai.com/v1/chat/completions"),
])
def test_openai_compatible_presets(provider, url):
    m = FakeModel(['{"text": "x"}'])
    spec = ModelSpec(name="p", provider=provider, model="some-model", api_key="k")
    run(ask(spec, [b"img"], "hi", httpx.MockTransport(m.handler)))
    assert m.requests[0]["url"] == url and m.requests[0]["headers"]["authorization"] == "Bearer k"


def test_model_needs_key_and_name_except_keyless_ollama():
    assert not ModelSpec(name="a", provider="openai", api_key="k").ready()
    assert ModelSpec(name="a", provider="openai", model="m", api_key="k").ready()
    assert ModelSpec(name="o", provider="ollama", model="qwen2.5vl").ready()


def test_last_json_skips_reasoning_and_braces_in_text():
    assert last_json('tile {1} has a bus. Answer: {"tiles": [1]}\n') == {"tiles": [1]}
    with pytest.raises(Unsolvable):
        last_json("no json here")


def test_upscale_only_small_images():
    small = Image.new("RGB", (100, 40))
    assert imaging.upscale(small)[0].size == (400, 160)
    assert imaging.upscale(Image.new("RGB", (800, 600)))[1] == 1


def test_failures_are_remembered_per_model_and_cleared_on_success():
    from imnotarobot.layers.vlm import LAST_ERRORS

    bad = FakeModel(["no json at all"])
    layer = bad.layer(votes=1)
    with pytest.raises(Unsolvable):
        run(layer.solve({"type": "ImageToTextTask", "body": blank_image()}))
    assert "не в нужном виде" in LAST_ERRORS[layer.name][1]
    good = FakeModel(['{"text": "abc"}'])
    layer.transport = httpx.MockTransport(good.handler)
    run(layer.solve({"type": "ImageToTextTask", "body": blank_image()}))
    assert layer.name not in LAST_ERRORS


@pytest.mark.parametrize("provider,protocol,effort,expected", [
    ("openai", "", "low", {"reasoning_effort": "low"}),
    ("gemini", "", "none", {"reasoning_effort": "none"}),
    ("openrouter", "", "none", {"reasoning": {"enabled": False}}),
    ("openrouter", "", "high", {"reasoning": {"effort": "high"}}),
    ("dashscope", "", "medium", {"enable_thinking": True, "thinking_budget": 4096}),
    ("dashscope", "", "none", {"enable_thinking": False}),
    ("anthropic", "", "low", {"thinking": {"type": "enabled", "budget_tokens": 1024}, "max_tokens": 3072}),
    ("anthropic", "", "none", {}),
    ("openai", "", "default", {}),
])
def test_effort_dialects(provider, protocol, effort, expected):
    from imnotarobot.providers import effort_fields

    assert effort_fields(ModelSpec(name=f"e-{provider}-{effort}", provider=provider, model="m"), effort) == expected


def test_effort_is_dropped_when_the_api_refuses_it():
    from imnotarobot.providers import NO_EFFORT

    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "Unrecognized request argument: reasoning_effort"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"text": "ok"}'}}]})

    spec = ModelSpec(name="refuses-effort", provider="openai", model="m", api_key="k")
    text, _ = run(ask(spec, [b"png"], "hi", httpx.MockTransport(handler), effort="low"))
    assert text == '{"text": "ok"}' and "refuses-effort" in NO_EFFORT
    run(ask(spec, [b"png"], "hi", httpx.MockTransport(handler), effort="low"))
    assert ["reasoning_effort" in b for b in sent] == [True, False, False]


def test_clicks_and_recognition_use_their_own_effort():
    m = FakeModel(['{"points": [[10, 10]]}', '{"text": "abc"}'])
    layer = m.layer(votes=1, effort_recognition="none", effort_clicks="high", click_mode="rulers")
    run(layer.solve({"type": "CoordinatesTask", "body": blank_image(), "comment": "x"}))
    run(layer.solve({"type": "ImageToTextTask", "body": blank_image()}))
    assert m.requests[0]["body"]["enable_thinking"] is True and m.requests[0]["body"]["thinking_budget"] == 16000
    assert m.requests[1]["body"]["enable_thinking"] is False


def test_box_mode_clicks_box_centres_in_order_and_sends_only_the_image():
    reply = ('Instruction: click the cat, then the dog.\n'
             '{"targets": [{"label": "cat", "box_2d": [100, 200, 300, 400]}, {"label": "dog", "box_2d": [500, 600, 700, 800]}]}')
    m = FakeModel([reply])
    r = run(m.layer(votes=2).solve({"type": "CoordinatesTask", "body": blank_image((200, 100)), "comment": ""}))
    assert r.solution == {"coordinates": [{"x": 60, "y": 20}, {"x": 140, "y": 60}]} and r.confidence == 1.0
    assert len(images_sent(m.requests[0])) == 1
    assert "box_2d" in m.requests[0]["body"]["messages"][0]["content"][-1]["text"]


def test_box_mode_without_targets_fails():
    m = FakeModel(['{"targets": []}'])
    with pytest.raises(Unsolvable, match="не обвела"):
        run(m.layer(votes=1).solve({"type": "CoordinatesTask", "body": blank_image((200, 100))}))


def test_one_ask_by_default():
    m = FakeModel(['{"text": "abc"}'])
    r = run(m.layer(votes=1).solve({"type": "ImageToTextTask", "body": blank_image()}))
    assert (r.solution, r.confidence, len(m.requests)) == ({"text": "abc"}, 1.0, 1)
    assert ModelSpec(name="x", provider="gemini").votes == 1


def test_matching_earlier_answer_needs_no_second_ask():
    m = FakeModel(['{"text": "abc"}'])
    r = run(m.layer(votes=1).solve({"type": "ImageToTextTask", "body": blank_image()}, prior=[{"text": "abc"}]))
    assert (r.confidence, len(m.requests)) == (1.0, 1)


def test_disagreeing_with_earlier_layer_asks_once_more():
    m = FakeModel(['{"text": "abd"}', '{"text": "abd"}'])
    r = run(m.layer(votes=1).solve({"type": "ImageToTextTask", "body": blank_image()}, prior=[{"text": "abc"}]))
    assert (r.solution, r.confidence, len(m.requests)) == ({"text": "abd"}, 1.0, 2)


def test_split_asks_side_with_the_earlier_layer():
    m = FakeModel(['{"text": "abd"}', '{"text": "abc"}'])
    r = run(m.layer(votes=1).solve({"type": "ImageToTextTask", "body": blank_image()}, prior=[{"text": "abc"}]))
    assert (r.solution, r.confidence, len(m.requests)) == ({"text": "abc"}, 0.5, 2)


def test_silent_service_gets_one_immediate_retry():
    calls = []

    def flaky(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="All accounts failed")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"text": "abc"}'}}]})

    s = ModelSpec(name="m", provider="custom", base_url="http://x/v1", api_key="k", votes=1)
    r = run(VlmLayer(s, transport=httpx.MockTransport(flaky)).solve({"type": "ImageToTextTask", "body": blank_image()}))
    assert r.solution == {"text": "abc"} and len(calls) == 2


def test_service_silent_twice_is_reported_unavailable():
    from imnotarobot.layers import Unavailable

    s = ModelSpec(name="m", provider="custom", base_url="http://x/v1", api_key="k", votes=1)
    layer = VlmLayer(s, transport=httpx.MockTransport(lambda r: httpx.Response(503, text="All accounts failed")))
    with pytest.raises(Unavailable):
        run(layer.solve({"type": "ImageToTextTask", "body": blank_image()}))
