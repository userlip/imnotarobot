"""New versions of the Mac app from GitHub Releases.

The app only looks for a release; nothing is installed until the person presses «Обновить». The download is
the same .dmg people get by hand, checked against the SHA-256 GitHub publishes for it, and the old copy of the
app goes to the Trash.
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .i18n import tr

REPO = "zigansh1n/imnotarobot"
ASSET = "ImNotARobot.dmg"
APP = "ImNotARobot.app"


def parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text)[:3])


def latest(current: str = __version__) -> dict[str, Any] | None:
    """The newest release when it is newer than `current` and carries the .dmg, else None."""
    import httpx

    r = httpx.get(f"https://api.github.com/repos/{REPO}/releases/latest", timeout=15,
                  headers={"Accept": "application/vnd.github+json"}, follow_redirects=True)
    r.raise_for_status()
    rel = r.json()
    version = rel.get("tag_name", "").lstrip("v")
    if parse_version(version) <= parse_version(current):
        return None
    asset = next((a for a in rel.get("assets", []) if a.get("name") == ASSET), None)
    if asset is None:
        return None
    digest = asset.get("digest") or ""
    return {"version": version, "url": asset["browser_download_url"], "size": asset.get("size", 0),
            "sha256": digest.removeprefix("sha256:") if digest.startswith("sha256:") else "", "page": rel.get("html_url", "")}


def running_app() -> Path | None:
    """The .app bundle this process runs from; None when it runs from source, which has nothing to replace."""
    if not getattr(sys, "frozen", False):
        return None
    app = Path(sys.executable).resolve().parents[2]
    return app if app.suffix == ".app" else None


def download(url: str, dest: Path, sha256: str = "", progress=None) -> None:
    import httpx

    h = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r, dest.open("wb") as f:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        for chunk in r.iter_bytes(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total)
    if sha256 and h.hexdigest() != sha256:
        raise RuntimeError(tr("скачанный файл повреждён, попробуйте ещё раз"))


def trash(path: Path) -> None:
    from Foundation import NSURL, NSFileManager

    ok, _, err = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(NSURL.fileURLWithPath_(str(path)), None, None)
    if not ok:
        raise RuntimeError(str(err))


def install(info: dict[str, Any], app: Path, progress=None) -> None:
    """Puts the release's app in place of `app`; the running copy keeps working until it quits."""
    if not app.parent.is_dir() or not os.access(app.parent, os.W_OK):
        raise PermissionError(tr("нет прав менять {folder} — скачайте новую версию вручную", folder=app.parent))
    with tempfile.TemporaryDirectory() as tmp:
        dmg, mnt = Path(tmp) / ASSET, Path(tmp) / "mnt"
        download(info["url"], dmg, info.get("sha256", ""), progress)
        subprocess.run(["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", str(mnt), str(dmg)],
                       check=True, capture_output=True)
        try:
            new = mnt / APP
            if not new.is_dir():
                raise RuntimeError(tr("в файле обновления нет приложения"))
            # copied next to the old one first: the swap below is then a rename on one disk
            staged = app.with_name(f".{app.name}.new")
            if staged.exists():
                trash(staged)
            subprocess.run(["ditto", str(new), str(staged)], check=True, capture_output=True)
        finally:
            subprocess.run(["hdiutil", "detach", "-force", str(mnt)], capture_output=True)
    trash(app)
    staged.rename(app)


def relaunch_later(app: Path) -> None:
    """Opens `app` again a moment after this process quits."""
    subprocess.Popen(["/bin/sh", "-c", 'while kill -0 "$1" 2>/dev/null; do sleep 0.5; done; open "$2"', "sh",
                      str(os.getpid()), str(app)], start_new_session=True)
