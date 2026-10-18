import asyncio
import json

import httpx
import pytest

from captcha_stack.layers import HumanLayer, LocalLayer, Unsolvable, VlmLayer
from helpers import blank_image, text_image


def run(coro):
    return asyncio.run(coro)


def test_local_reads_a_clean_text_captcha():
    r = run(LocalLayer().solve({"type": "ImageToTextTask", "body": text_image("ab3k7")}))
    assert r.solution == {"text": "ab3k7"}
    assert r.confidence > 0.9


def test_local_rejects_an_answer_that_breaks_constraints():
    with pytest.raises(Unsolvable):
        run(LocalLayer().solve({"type": "ImageToTextTask", "body": text_image("ab3k7"), "numeric": 1}))


def vlm_with(reply: str, seen: list | None = None) -> VlmLayer:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}], "usage": {"total_tokens": 900}})

    return VlmLayer("https://vlm.test/v1", "k", "qwen3-vl-plus", transport=httpx.MockTransport(handler))


def test_vlm_converts_normalized_points_to_pixels():
    layer = vlm_with('```json\n{"points": [[500, 500], [1000, 0]], "confidence": 0.8}\n```')
    r = run(layer.solve({"type": "CoordinatesTask", "body": blank_image((200, 100)), "comment": "click the cat"}))
    assert r.solution == {"coordinates": [{"x": 100, "y": 50}, {"x": 200, "y": 0}]}
    assert (r.confidence, r.meta["tokens"]) == (0.8, 900)


def test_vlm_grid_drops_tiles_outside_the_grid_and_sends_the_task():
    seen: list = []
    layer = vlm_with('{"tiles": [3, 1, 3, 17], "confidence": 0.9}', seen)
    r = run(layer.solve({"type": "GridTask", "body": blank_image(), "comment": "select buses", "rows": 4, "columns": 4}))
    assert r.solution == {"click": [1, 3]}
    prompt = seen[0]["messages"][0]["content"][-1]["text"]
    assert "4 rows and 4 columns" in prompt and "select buses" in prompt
    assert seen[0]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_vlm_text_answer_is_checked_against_constraints():
    with pytest.raises(Unsolvable):
        run(vlm_with('{"text": "abc", "confidence": 0.9}').solve({"type": "ImageToTextTask", "body": blank_image(), "numeric": 1}))


def test_vlm_without_json_is_a_failure():
    with pytest.raises(Unsolvable, match="did not return JSON"):
        run(vlm_with("I cannot help with that.").solve({"type": "ImageToTextTask", "body": blank_image()}))


def test_vlm_is_unavailable_without_a_key():
    assert not VlmLayer("https://x", "", "m").available()


def test_human_creates_polls_and_reports():
    calls: list[tuple[str, dict]] = []
    polls = iter(["processing", "ready"])

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = request.url.path.strip("/")
        calls.append((method, body))
        if method == "createTask":
            return httpx.Response(200, json={"errorId": 0, "taskId": 42})
        if method == "getTaskResult":
            status = next(polls)
            extra = {"solution": {"text": "w0rd"}, "cost": "0.00025"} if status == "ready" else {}
            return httpx.Response(200, json={"errorId": 0, "status": status, **extra})
        return httpx.Response(200, json={"errorId": 0, "status": "success"})

    layer = HumanLayer("https://api.2captcha.test", "key", poll_interval=0, transport=httpx.MockTransport(handler))
    task = {"type": "ImageToTextTask", "body": "eA=="}
    r = run(layer.solve(task))
    assert (r.solution, r.cost, r.confidence) == ({"text": "w0rd"}, 0.00025, 1.0)
    assert calls[0] == ("createTask", {"clientKey": "key", "task": task})
    run(layer.report_incorrect(r))
    assert calls[-1] == ("reportIncorrect", {"clientKey": "key", "taskId": 42})


def test_human_error_is_raised():
    handler = lambda r: httpx.Response(200, json={"errorId": 1, "errorCode": "ERROR_ZERO_BALANCE"})
    layer = HumanLayer("https://x.test", "key", poll_interval=0, transport=httpx.MockTransport(handler))
    with pytest.raises(Unsolvable, match="ERROR_ZERO_BALANCE"):
        run(layer.solve({"type": "ImageToTextTask", "body": "eA=="}))


def test_human_takes_token_captchas_but_not_local_only_ones():
    layer = HumanLayer("https://x", "k")
    assert layer.supports("HCaptchaTaskProxyless") and not layer.supports("SlideMatchTask")
