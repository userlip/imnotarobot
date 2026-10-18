import asyncio

import pytest

from captcha_stack.chain import Chain
from captcha_stack.layers import Unsolvable
from helpers import FakeLayer

TASK = {"type": "ImageToTextTask", "body": "aGVsbG8="}


def solve(chain, task=TASK):
    return asyncio.run(chain.solve(task))


def test_confident_first_layer_wins_and_later_layers_are_not_called():
    local, vlm = FakeLayer("local", [("abc", 0.95)]), FakeLayer("vlm", [("xyz", 0.99)])
    r = solve(Chain([local, vlm]))
    assert (r.layer, r.solution) == ("local", {"text": "abc"})
    assert vlm.calls == 0


def test_unsure_layer_escalates_to_next():
    local, vlm = FakeLayer("local", [("abc", 0.4)]), FakeLayer("vlm", [("abd", 0.9)])
    r = solve(Chain([local, vlm]))
    assert r.layer == "vlm"
    assert "local: unsure (0.40)" in r.meta["tried"]


def test_two_unsure_layers_that_agree_are_accepted():
    local, vlm, human = FakeLayer("local", [("abc", 0.4)]), FakeLayer("vlm", [("abc", 0.5)]), FakeLayer("human", [("zzz", 1.0)])
    r = solve(Chain([local, vlm, human]))
    assert (r.layer, r.meta["agreed_with"]) == ("vlm", "local")
    assert human.calls == 0


def test_failures_and_missing_config_fall_through_to_the_last_layer():
    local = FakeLayer("local", [Unsolvable("bad constraints")])
    vlm = FakeLayer("vlm", [("x", 1.0)], available=False)
    human = FakeLayer("human", [("final", 1.0)], min_confidence=1.0)
    r = solve(Chain([local, vlm, human]))
    assert r.layer == "human"
    assert r.meta["tried"] == ["local: bad constraints", "vlm: not configured"]


def test_best_unsure_answer_is_returned_flagged_when_nothing_else_is_left():
    r = solve(Chain([FakeLayer("local", [("abc", 0.3)]), FakeLayer("vlm", [("abd", 0.6)])]))
    assert (r.layer, r.meta["low_confidence"]) == ("vlm", True)


def test_all_layers_failing_raises_with_every_reason():
    chain = Chain([FakeLayer("local", [Unsolvable("no")]), FakeLayer("vlm", [Unsolvable("nope")])])
    with pytest.raises(Unsolvable, match="local: no; vlm: nope"):
        solve(chain)


def test_report_skips_that_layer_for_the_same_image_and_is_forwarded():
    local, vlm = FakeLayer("local", [("abc", 0.95)]), FakeLayer("vlm", [("abd", 0.9)])
    chain = Chain([local, vlm])
    first = solve(chain)
    asyncio.run(chain.report_incorrect(TASK, first))
    second = solve(chain)
    assert (first.layer, second.layer, local.reports) == ("local", "vlm", 1)
    other_image = solve(chain, {**TASK, "body": "b3RoZXI="})
    assert other_image.layer == "local"


def test_layer_is_skipped_for_a_kind_after_most_of_its_answers_were_reported():
    local, vlm = FakeLayer("local", [("abc", 0.95)]), FakeLayer("vlm", [("abd", 0.9)])
    chain = Chain([local, vlm], skip_after_reports=3)
    for i in range(3):
        task = {**TASK, "body": f"img{i}"}
        asyncio.run(chain.report_incorrect(task, solve(chain, task)))
    assert solve(chain, {**TASK, "body": "fresh"}).layer == "vlm"
    other_kind = {**TASK, "body": "fresh2", "comment": "type the red letters"}
    assert solve(chain, other_kind).layer == "local"
