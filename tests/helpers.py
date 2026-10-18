import base64
import io

from PIL import Image, ImageDraw, ImageFont

from captcha_stack.layers import Layer, Result, Unsolvable

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def text_image(text: str, size=(160, 60)) -> str:
    img = Image.new("RGB", size, (240, 240, 240))
    ImageDraw.Draw(img).text((10, 8), text, fill=(20, 20, 20), font=ImageFont.truetype(FONT, 36))
    return png_b64(img)


def blank_image(size=(200, 100)) -> str:
    return png_b64(Image.new("RGB", size, (255, 255, 255)))


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeLayer(Layer):
    def __init__(self, name, answers, min_confidence=0.8, types=("ImageToTextTask",), available=True):
        self.name = name
        self.answers = list(answers)
        self.min_confidence = min_confidence
        self.task_types = frozenset(types)
        self._available = available
        self.calls = 0
        self.reports = 0

    def available(self):
        return self._available

    async def solve(self, task):
        self.calls += 1
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, Exception):
            raise a
        text, conf = a
        return Result({"text": text}, conf, self.name)

    async def report_incorrect(self, result):
        self.reports += 1


__all__ = ["FakeLayer", "Unsolvable", "blank_image", "png_b64", "text_image"]
