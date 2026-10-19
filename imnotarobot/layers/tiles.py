"""Local classifier for reCAPTCHA picture tiles: answers a grid in a fraction of a second when it is sure.

The model is the MIT-licensed reCAPTCHA classifier trained on 57 thousand tiles
(huggingface.co/DannyLuna/recaptcha-classification-57k, YOLO11x-cls exported to ONNX, 640 px input).
It is not shipped with the app; `download()` fetches it on the user's request.
"""

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config

URL = "https://huggingface.co/DannyLuna/recaptcha-classification-57k/resolve/main/recaptcha_classification_57k.onnx"
CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "hydrant", "motorcycle", "mountain",
           "other", "palm", "stair", "tractor", "traffic light"]
# how reCAPTCHA names the classes in its instructions
SYNONYMS = {"bicycle": ["bicycle"], "bridge": ["bridge"], "bus": ["bus"], "car": ["car"], "chimney": ["chimney"],
            "crosswalk": ["crosswalk"], "hydrant": ["hydrant"], "motorcycle": ["motorcycle"],
            "mountain": ["mountain", "hill"], "palm": ["palm"], "stair": ["stair"], "tractor": ["tractor"],
            "traffic light": ["traffic light"]}
SIZE = 640
SURE_YES, SURE_NO = 0.7, 0.15


def model_path() -> Path:
    return Path(config.APP_DIR) / "models" / "recaptcha_classification_57k.onnx"


def available() -> bool:
    return model_path().exists()


def target_class(instruction: str) -> str | None:
    text = instruction.lower()
    return next((c for c, words in SYNONYMS.items() if any(w in text for w in words)), None)


def split(grid_png: bytes, rows: int, columns: int) -> list[Image.Image]:
    im = Image.open(io.BytesIO(grid_png)).convert("RGB")
    w, h = im.width / columns, im.height / rows
    return [im.crop((round(c * w), round(r * h), round((c + 1) * w), round((r + 1) * h)))
            for r in range(rows) for c in range(columns)]


def prepare(tile: Image.Image) -> np.ndarray:
    """YOLO classification preprocessing: shortest side to 640, centre crop, RGB in 0..1, channels first."""
    k = SIZE / min(tile.size)
    im = tile.resize((max(SIZE, round(tile.width * k)), max(SIZE, round(tile.height * k))), Image.BILINEAR)
    left, top = (im.width - SIZE) // 2, (im.height - SIZE) // 2
    im = im.crop((left, top, left + SIZE, top + SIZE))
    return (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)


class TileClassifier:
    def __init__(self, session=None):
        self._session = session

    def _run(self, batch: np.ndarray) -> np.ndarray:
        if self._session is None:
            import onnxruntime

            self._session = onnxruntime.InferenceSession(str(model_path()), providers=["CPUExecutionProvider"])
        name = self._session.get_inputs()[0].name
        return np.asarray(self._session.run(None, {name: batch})[0])

    def answer(self, grid_png: bytes, rows: int, columns: int, instruction: str) -> list[int] | None:
        """Tile numbers from 1 that show the target, or None when the model is unsure about any tile."""
        target = target_class(instruction)
        if target is None:
            return None
        probs = self._run(np.stack([prepare(t) for t in split(grid_png, rows, columns)]))
        p = probs[:, CLASSES.index(target)]
        if np.any((p > SURE_NO) & (p < SURE_YES)):
            return None
        return [i + 1 for i in np.flatnonzero(p >= SURE_YES)]


def download(sha256: str | None = None) -> Path:
    import httpx

    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    digest = hashlib.sha256()
    with httpx.stream("GET", URL, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                digest.update(chunk)
    if sha256 and digest.hexdigest() != sha256:
        tmp.unlink()
        raise ValueError("the downloaded model does not match its checksum")
    tmp.replace(path)
    return path
