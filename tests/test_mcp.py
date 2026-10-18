import base64
import time

import httpx
import pytest

import captcha_stack.config as config
from captcha_stack import client as client_mod
from captcha_stack import mcp_server
from captcha_stack.config import Config
from captcha_stack.control import Controller
from helpers import text_image


@pytest.fixture
def running(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "set_secret", lambda a, v: None)
    cfg = Config.load()
    cfg.port = 18765
    cfg.layers = ["local"]
    c = Controller(cfg)
    c.start_server()
    for _ in range(100):
        try:
            if httpx.get(f"{c.address}/health", timeout=1).json()["ok"]:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield c
    c.stop_server()


def test_running_solver_is_found_and_solves_through_the_mcp_tool(running, tmp_path):
    address, key = client_mod.locate()
    assert address == running.address and key == running.cfg.client_key
    png = tmp_path / "c.png"
    png.write_bytes(base64.b64decode(text_image("k7m2p")))
    r = mcp_server.solve_image_captcha(str(png), kind="text")
    assert r["answer"] == {"text": "k7m2p"} and r["layer"] == "local" and r["task_id"]
    status = mcp_server.captcha_status()
    assert status["address"] == running.address and status["layers"][0]["name"] == "local"
    reported = mcp_server.report_wrong(r["task_id"], retry=False)
    assert reported == {"reported": True, "retry": None}


def test_not_running_is_explained(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="не запущен"):
        mcp_server.captcha_status()


def test_wrong_access_key_is_refused(running):
    c = client_mod.Client(running.address, key="nope")
    with pytest.raises(RuntimeError, match="clientKey"):
        c.solve({"type": "ImageToTextTask", "body": text_image("ab")})
