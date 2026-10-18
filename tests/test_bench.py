import asyncio
import json

import httpx
import pytest

from captcha_stack import bench, testset
from captcha_stack.providers import ModelSpec


def test_generated_cases_are_deterministic_and_cover_every_kind():
    a, b = testset.generated(2), testset.generated(2)
    assert [c["expected"] for c in a] == [c["expected"] for c in b]
    assert {c["set"] for c in a} == {f"сгенерированные: {k}" for k in testset.GENERATORS}


@pytest.mark.parametrize("case,solution,ok", [
    ({"task": {}, "expected": {"text": "Ab3"}}, {"text": "ab3"}, True),
    ({"task": {"case": True}, "expected": {"text": "Ab3"}}, {"text": "ab3"}, False),
    ({"task": {}, "expected": {"tiles": [1, 4]}}, {"click": [4, 1]}, True),
    ({"task": {}, "expected": {"box": [0, 0, 10, 10]}}, {"coordinates": [{"x": 5, "y": 5}]}, True),
    ({"task": {}, "expected": {"boxes": [[0, 0, 10, 10], [20, 0, 30, 10]], "inorder": True}},
     {"coordinates": [{"x": 25, "y": 5}, {"x": 5, "y": 5}]}, False),
    ({"task": {}, "expected": {"boxes": [[0, 0, 10, 10], [20, 0, 30, 10]], "inorder": False}},
     {"coordinates": [{"x": 25, "y": 5}, {"x": 5, "y": 5}]}, True),
    ({"task": {}, "expected": {"box": [0, 0, 10, 10]}}, {"coordinates": [{"x": 5, "y": 5}, {"x": 6, "y": 6}]}, False),
])
def test_scoring(case, solution, ok):
    assert bench.score(case, solution) is ok


def test_run_ranks_the_model_that_answers_right_first(tmp_path):
    cases = [c for c in testset.generated(2) if c["task"]["type"] == "ImageToTextTask"]
    # concurrency=1 and one vote: the semaphore is FIFO, so requests arrive in case order
    good_answers = iter(c["expected"]["text"] for c in cases)

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        text = next(good_answers) if model == "good" else "zzzzz"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"text": text})}}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})

    specs = [ModelSpec(name="bad", provider="openai", model="bad", api_key="k", price_in=1, price_out=1),
             ModelSpec(name="good", provider="openai", model="good", api_key="k", price_in=1, price_out=1)]
    seen = []
    results = asyncio.run(bench.run(specs, cases, concurrency=1, transport=httpx.MockTransport(handler),
                                    progress=lambda d, t: seen.append((d, t))))
    assert bench.ranking(results) == ["good", "bad"]
    assert results["good"]["accuracy"] == 1.0 and results["bad"]["accuracy"] == 0.0
    assert results["good"]["usd_per_1000"] == pytest.approx(1.1)
    assert seen[-1] == (2 * len(cases), 2 * len(cases))
    page = bench.report_html(results, tmp_path / "r.html", votes=1).read_text()
    assert page.index("good") < page.index("bad") and "100%" in page


def test_estimate_is_unknown_without_prices():
    priced = ModelSpec(name="a", provider="openai", model="m", price_in=1.0, price_out=2.0)
    assert bench.estimate_usd([priced], 10, 1) == pytest.approx(10 * (1500 * 1 + 400 * 2) / 1e6)
    assert bench.estimate_usd([priced, ModelSpec(name="b", provider="openai", model="m")], 10, 1) is None


def test_venusbench_conversion(tmp_path):
    root = tmp_path / "venusbench"
    (root / "instruction").mkdir(parents=True)
    (root / "images" / "OCR").mkdir(parents=True)
    (root / "images" / "icon_click").mkdir(parents=True)
    (root / "images" / "OCR" / "1.png").write_bytes(b"png")
    (root / "images" / "icon_click" / "2.png").write_bytes(b"png")
    items = [
        {"images": ["../images/OCR/1.png"], "captcha_type": "OCR", "messages": [{"role": "user", "content": "<image>Enter the code"}],
         "action_raw": "Click(box=(1, 2)),Type(content='1712')", "action_raw_rect": [[0, 0, 5, 5]], "inorder": True},
        {"images": ["../images/icon_click/2.png"], "captcha_type": "icon_click", "messages": [{"role": "user", "content": "Click the cat"}],
         "action_raw": "Click(box=(3, 3))", "action_raw_rect": [[1, 1, 9, 9]], "inorder": False},
        {"images": ["../images/x.png"], "captcha_type": "slider_puzzle", "messages": [], "action_raw": "", "action_raw_rect": []},
    ]
    (root / "instruction" / "VenusBench-CAPTCHA.json").write_text(json.dumps(items))
    cases = testset.from_venusbench(root)
    assert [(c["task"]["type"], c["expected"]) for c in cases] == [
        ("ImageToTextTask", {"text": "1712"}), ("CoordinatesTask", {"boxes": [[1, 1, 9, 9]], "inorder": False})]
    assert cases[0]["task"]["comment"] == "Enter the code"
