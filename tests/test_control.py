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


def fake_detect(monkeypatch, provider, models, listable=True, base="https://api.example/v1"):
    from captcha_stack.discovery import Detected, ModelInfo

    found = Detected(provider, base, [ModelInfo(m, v, 0.3, 2.5) for m, v in models], listable)
    monkeypatch.setattr(control, "detect_sync", lambda key, base_url="": found)


def test_add_key_lists_vision_models_and_hides_the_secret(ctl, monkeypatch, tmp_path):
    fake_detect(monkeypatch, "openrouter", [("google/gemini-x", True), ("deepseek/text", False)])
    entry = ctl.add_key("sk-or-v1-secretsecret")
    assert (entry.name, entry.hint) == ("openrouter", "sk-or…cret")
    assert ctl.secrets["key:openrouter"] == "sk-or-v1-secretsecret"
    assert "secretsecret" not in (tmp_path / "config.json").read_text()
    assert ctl.key_models("openrouter") == [("google/gemini-x", False)]


def test_toggle_model_from_key_adds_a_ready_layer_with_prices(ctl, monkeypatch):
    fake_detect(monkeypatch, "openrouter", [("google/gemini-x", True)])
    ctl.add_key("sk-or-v1-abc")
    ctl.toggle_key_model("openrouter", "google/gemini-x")
    spec = ctl.cfg.models[0]
    assert (spec.name, spec.key, spec.price_in, spec.ready()) == ("openrouter/google/gemini-x", "openrouter", 0.3, True)
    assert ctl.cfg.layers == ["local", "vlm:openrouter/google/gemini-x", "human"]
    ctl.toggle_key_model("openrouter", "google/gemini-x")
    assert ctl.cfg.models == [] and ctl.cfg.layers == ["local", "human"]


def test_second_key_of_same_provider_gets_its_own_name_and_removal_cleans_up(ctl, monkeypatch):
    fake_detect(monkeypatch, "openai", [("gpt-5", True)])
    ctl.add_key("sk-1111")
    ctl.add_key("sk-2222")
    assert [k.name for k in ctl.cfg.keys] == ["openai", "openai-2"]
    ctl.toggle_key_model("openai-2", "gpt-5")
    ctl.remove_key("openai-2")
    assert [k.name for k in ctl.cfg.keys] == ["openai"] and ctl.cfg.models == []
    assert "key:openai-2" not in ctl.secrets and ctl.cfg.layers == ["local", "human"]


def test_bench_specs_cover_every_vision_model_of_a_key(ctl, monkeypatch):
    fake_detect(monkeypatch, "openai", [("gpt-5", True), ("gpt-4o", True), ("tts-1", False)])
    ctl.add_key("sk-xyz")
    specs = ctl.bench_specs("openai")
    assert [s.model for s in specs] == ["gpt-5", "gpt-4o"] and all(s.api_key == "sk-xyz" for s in specs)
    runs, usd = ctl.bench_estimate(specs)
    assert runs == 2 * len(ctl.bench_cases()) and usd > 0


def test_use_best_enables_and_moves_it_first(ctl, monkeypatch):
    fake_detect(monkeypatch, "openai", [("gpt-5", True), ("gpt-4o", True)])
    ctl.add_key("sk-xyz")
    ctl.toggle_key_model("openai", "gpt-5")
    ctl.use_best("openai/gpt-4o")
    assert ctl.cfg.layers == ["local", "vlm:openai/gpt-4o", "vlm:openai/gpt-5", "human"]
