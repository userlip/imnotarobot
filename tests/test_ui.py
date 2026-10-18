import time

import pytest
from fastapi.testclient import TestClient

import captcha_stack.config as config
import captcha_stack.control as control
from captcha_stack.config import Config
from captcha_stack.control import Controller
from captcha_stack.server import create_app
from captcha_stack.ui import add_ui_routes
from helpers import text_image


@pytest.fixture
def ui(tmp_path, monkeypatch):
    secrets: dict[str, str] = {}
    monkeypatch.setattr(config, "get_secret", lambda a: secrets.get(a, ""))
    monkeypatch.setattr(config, "set_secret", lambda a, v: secrets.__setitem__(a, v) if v else secrets.pop(a, None))
    monkeypatch.setattr(control, "LAUNCH_AGENT", tmp_path / "agent.plist")
    c = Controller(Config.load())
    app = create_app(service=c.service)
    add_ui_routes(app, c)
    with TestClient(app) as client:
        client.headers["X-UI-Token"] = c.ui_token
        yield c, client


def test_api_refuses_requests_without_the_window_token(ui):
    c, client = ui
    assert client.get("/api/ui/state", headers={"X-UI-Token": "wrong"}).status_code == 403
    assert client.post("/api/ui/pause", json={"paused": True}, headers={"X-UI-Token": ""}).status_code == 403
    assert not c.service.paused
    assert client.get("/ui").status_code == 200


def test_state_never_contains_secrets(ui, monkeypatch):
    c, client = ui
    from captcha_stack.discovery import Detected, ModelInfo

    monkeypatch.setattr(control, "detect_sync", lambda k, b="": Detected("custom", "https://reseller.example/v1",
                                                                          [ModelInfo("claude-x", True)]))
    r = client.post("/api/ui/add-key", json={"secret": "sk-reseller-supersecretvalue"}).json()
    assert r["ok"] and "посредник reseller.example" in r["message"] and "моделей с картинками: 1" in r["message"]
    assert "supersecretvalue" not in client.get("/api/ui/state").text
    r = client.post("/api/ui/toggle-model", json={"key": "reseller.example", "model": "claude-x"}).json()
    assert r["message"] == "claude-x: включена."
    layer = next(x for x in r["state"]["layers"] if x["id"] == "vlm:reseller.example/claude-x")
    assert layer["enabled"] and layer["ready"]


def test_bad_key_explains_why(ui, monkeypatch):
    _, client = ui
    from captcha_stack.discovery import DetectError

    def nobody(key, base=""):
        raise DetectError("ни один известный поставщик не принял этот ключ")

    monkeypatch.setattr(control, "detect_sync", nobody)
    r = client.post("/api/ui/add-key", json={"secret": "sk-nothing-at-all"})
    assert r.status_code == 400 and "не принял" in r.json()["error"]


def test_saving_a_human_key_checks_it_and_reports_balance(ui, monkeypatch):
    c, client = ui

    async def balance():
        return 3.2

    monkeypatch.setattr(c.service, "balance", balance)
    r = client.post("/api/ui/human", json={"provider": "rucaptcha", "key": "abc123", "limit": "2"}).json()
    assert r["ok"] and r["message"] == "Ключ работает, баланс $3.20."
    assert r["state"]["human"] == {"provider": "rucaptcha", "hasKey": True, "legacyKey": False, "limit": 2.0,
                                   "status": {"ok": True, "balance": 3.2}, "spentToday": 0.0}
    assert c.human_line() == "Люди (RuCaptcha): ✓ баланс $3.20"


def test_rejected_human_key_is_reported(ui, monkeypatch):
    c, client = ui

    async def balance():
        raise RuntimeError("getBalance: ERROR_KEY_DOES_NOT_EXIST")

    monkeypatch.setattr(c.service, "balance", balance)
    r = client.post("/api/ui/human", json={"key": "wrong"})
    assert r.status_code == 400 and "ERROR_KEY_DOES_NOT_EXIST" in r.json()["error"]
    assert c.human_line().startswith("Люди (RuCaptcha): ⚠")


def test_solved_captchas_show_up_in_recent(ui):
    c, client = ui
    task_id = client.post("/createTask", json={"task": {"type": "ImageToTextTask", "body": text_image("ab3k7")}}).json()["taskId"]
    for _ in range(200):
        if client.post("/getTaskResult", json={"taskId": task_id}).json().get("status") == "ready":
            break
        time.sleep(0.05)
    recent = client.get("/api/ui/state").json()["recent"]
    assert recent[0]["layer"] == "local" and recent[0]["layerTitle"] == "Локальная модель" and recent[0]["type"] == "ImageToTextTask" and not recent[0]["error"]


def test_layers_move_and_toggle(ui):
    c, client = ui
    client.post("/api/ui/move-layer", json={"id": "human", "step": -1})
    assert c.cfg.layers == ["human", "local"]
    r = client.post("/api/ui/toggle-layer", json={"id": "local"}).json()
    assert [(x["id"], x["enabled"]) for x in r["state"]["layers"]] == [("human", True), ("local", False)]


def test_key_left_by_an_older_build_asks_to_be_entered_again(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "get_secret", lambda a: "")
    monkeypatch.setattr(config, "legacy_secret_exists", lambda a: a == "human")
    monkeypatch.setattr(control, "LAUNCH_AGENT", tmp_path / "agent.plist")
    c = Controller(Config.load())
    assert c.human_line() == "Люди (RuCaptcha): введите ключ ещё раз"
    monkeypatch.setattr(config, "set_secret", lambda a, v: None)
    c.set_human(key="new")
    assert not c.legacy_human_key
