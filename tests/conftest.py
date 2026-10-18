import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keeps tests away from the real settings, state file and Keychain."""
    monkeypatch.setenv("CAPTCHA_STACK_HOME", str(tmp_path))
    import captcha_stack.config as config

    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setattr(config, "get_secret", lambda account: "")


@pytest.fixture(autouse=True)
def no_local_resellers(monkeypatch):
    import captcha_stack.discovery as discovery

    monkeypatch.setattr(discovery, "local_reseller_urls", lambda: [])


@pytest.fixture(autouse=True)
def no_real_keychain(monkeypatch):
    import captcha_stack.config as config

    monkeypatch.setattr(config, "legacy_secret_exists", lambda account: False)
