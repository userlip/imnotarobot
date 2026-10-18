import plistlib

import pytest

import captcha_stack.config as config
import captcha_stack.control as control
from captcha_stack.config import Config
from captcha_stack.control import Controller


@pytest.fixture
def ctl(tmp_path, monkeypatch):
    secrets: dict[str, str] = {}
    monkeypatch.setattr(config, "get_secret", lambda a: secrets.get(a, ""))
    monkeypatch.setattr(config, "set_secret", lambda a, v: secrets.__setitem__(a, v) if v else secrets.pop(a, None))
    monkeypatch.setattr(config, "_dashscope_key", lambda: "")
    monkeypatch.setattr(control, "LAUNCH_AGENT", tmp_path / "LaunchAgents" / "app.captcha-stack.plist")
    c = Controller(Config.load())
    c.secrets = secrets
    return c


def test_adding_a_model_stores_the_key_in_keychain_not_in_the_file(ctl, tmp_path):
    ctl.add_model("gem", "gemini", "gemini-2.5-flash", "AIza-secret")
    saved = (tmp_path / "config.json").read_text()
    assert "AIza-secret" not in saved and '"gemini-2.5-flash"' in saved
    assert ctl.secrets["model:gem"] == "AIza-secret"
    assert [layer.name for layer in ctl.service.chain.layers] == ["local", "vlm:gem", "human"]
    assert ctl.service.chain.layer("vlm:gem").available()


def test_model_lines_show_missing_keys(ctl):
    ctl.add_model("oa", "openai", "gpt-x", "")
    assert ctl.model_lines() == [("oa", "⚠ нет ключа  oa: openai / gpt-x")]


def test_bad_input_is_refused(ctl):
    with pytest.raises(ValueError, match="неизвестный поставщик"):
        ctl.add_model("x", "nope", "m", "k")
    with pytest.raises(ValueError, match="название модели"):
        ctl.add_model("x", "openai", "", "k")
    with pytest.raises(ValueError, match="отрицательным"):
        ctl.set_human(daily_limit=-1)


def test_toggling_layers_keeps_order_and_survives_reload(ctl):
    ctl.add_model("q", "dashscope", "qwen3-vl-plus", "k")
    ctl.toggle_layer("local")
    assert ctl.cfg.layers == ["vlm:q", "human"]
    ctl.toggle_layer("local")
    assert ctl.cfg.layers == ["local", "vlm:q", "human"]
    assert Config.load().layers == ["local", "vlm:q", "human"]


def test_human_settings(ctl):
    ctl.set_human("2captcha", "hkey", 2.5)
    assert ctl.cfg.human_base_url == "https://api.2captcha.com" and ctl.secrets["human"] == "hkey"
    assert ctl.service.chain.daily_limits == {"human": 2.5}
    assert ctl.service.chain.layer("human").available()


def test_remove_model_deletes_its_key(ctl):
    ctl.add_model("q", "dashscope", "qwen3-vl-plus", "k")
    ctl.remove_model("q")
    assert "model:q" not in ctl.secrets and ctl.cfg.models == []


def test_start_at_login_writes_a_launch_agent(ctl):
    ctl.set_start_at_login(True)
    data = plistlib.loads(control.LAUNCH_AGENT.read_bytes())
    assert data["ProgramArguments"][1:] == ["-m", "captcha_stack.menubar"] and data["RunAtLoad"]
    ctl.set_start_at_login(False)
    assert not ctl.starts_at_login()


def test_today_line_sums_layers(ctl):
    ctl.service.chain.state.answered("k", "local", 0.0)
    ctl.service.chain.state.answered("k", "human", 0.0012)
    assert ctl.today_line() == "Сегодня: решено 2 · $0.001"
