import shutil
import subprocess
import sys

import pytest

import imnotarobot.update as update


class Resp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def release(tag, assets=("ImNotARobot.dmg",)):
    return {"tag_name": tag, "html_url": f"https://github.com/x/releases/tag/{tag}",
            "assets": [{"name": n, "browser_download_url": f"https://dl/{n}", "size": 10,
                        "digest": "sha256:" + "a" * 64} for n in assets]}


def test_versions_compare_as_numbers():
    assert update.parse_version("v0.10.0") > update.parse_version("0.9.9")
    assert update.parse_version("0.4.1") == (0, 4, 1)


def test_latest_reports_only_a_newer_release_with_the_dmg(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp(release("v0.5.0")))
    info = update.latest("0.4.1")
    assert info["version"] == "0.5.0" and info["url"] == "https://dl/ImNotARobot.dmg" and info["sha256"] == "a" * 64
    assert update.latest("0.5.0") is None
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp(release("v0.6.0", assets=("other.zip",))))
    assert update.latest("0.4.1") is None


def test_running_from_source_has_nothing_to_update(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert update.running_app() is None


@pytest.mark.skipif(not shutil.which("hdiutil"), reason="needs macOS")
def test_install_swaps_the_app_and_trashes_the_old_one(tmp_path, monkeypatch):
    src = tmp_path / "src" / "ImNotARobot.app" / "Contents"
    src.mkdir(parents=True)
    (src / "version").write_text("new")
    dmg = tmp_path / "release.dmg"
    subprocess.run(["hdiutil", "create", "-quiet", "-srcfolder", str(tmp_path / "src"), "-format", "UDZO", str(dmg)], check=True)
    apps = tmp_path / "Applications"
    old = apps / "ImNotARobot.app" / "Contents"
    old.mkdir(parents=True)
    (old / "version").write_text("old")
    trashed = []
    monkeypatch.setattr(update, "trash", lambda p: (trashed.append(p.read_bytes() if p.is_file() else (p / "Contents/version").read_text()), shutil.rmtree(p)))
    monkeypatch.setattr(update, "download", lambda url, dest, sha, progress=None: shutil.copy(dmg, dest))
    update.install({"url": "x"}, apps / "ImNotARobot.app")
    assert (apps / "ImNotARobot.app/Contents/version").read_text() == "new"
    assert trashed == ["old"] and not (apps / ".ImNotARobot.app.new").exists()


def test_a_damaged_download_is_refused(tmp_path, monkeypatch):
    import httpx

    class Stream:
        headers = {"content-length": "3"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def raise_for_status(self):
            pass

        def iter_bytes(self, n):
            yield b"abc"

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: Stream())
    with pytest.raises(RuntimeError):
        update.download("u", tmp_path / "f", "0" * 64)
