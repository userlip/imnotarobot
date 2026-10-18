# captcha-stack

Свой решатель капч с API как у 2Captcha (v2: `createTask` / `getTaskResult`). Капча проходит слои по очереди, от бесплатного к платному:

1. **local** — модели ddddocr на этом компьютере: текстовые капчи, слайдеры. Бесплатно, миллисекунды.
2. **vlm** — облачная модель, которая видит картинку (по умолчанию Qwen3-VL через DashScope; подходит любая OpenAI-совместимая): текст, сетки «выбери все автобусы», клики по картинке.
3. **human** — 2Captcha / RuCaptcha: люди и их ИИ. Сюда же уходят капчи-токены (reCAPTCHA, hCaptcha, Turnstile).

Следующий слой включается, если предыдущий не справился, не уверен в ответе или нарушил условия задачи. Два неуверенных слоя с одинаковым ответом — ответ принимается.

Отличие от найденных проектов — обратная связь:
- `reportIncorrect` запоминает, что слой ошибся на этой картинке; с `"retry": true` сразу решает заново следующим слоем.
- Если слой ошибается на большинстве капч одного вида (тип + инструкция), он пропускается для этого вида.
- Жалоба на ответ людей пересылается в 2Captcha (там возвращают деньги).

## Запуск

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -e ".[test]"
.venv/bin/python -m captcha_stack --port 8765
```

Настройки — переменные окружения:

| Переменная | По умолчанию | Что |
|---|---|---|
| `LAYERS` | `local,vlm,human` | порядок слоёв |
| `VLM_API_KEY` | первый ключ из `~/.config/dashscope.env` | ключ облачной модели |
| `VLM_BASE_URL` / `VLM_MODEL` | DashScope / `qwen3-vl-plus` | любая OpenAI-совместимая модель |
| `HUMAN_API_KEY` | — | ключ 2Captcha или RuCaptcha; без него слой выключен |
| `HUMAN_BASE_URL` | `https://api.2captcha.com` | для RuCaptcha — `https://api.rucaptcha.com` |
| `OCR_MIN_CONFIDENCE` / `VLM_MIN_CONFIDENCE` | 0.9 / 0.7 | ниже — ответ считается неуверенным |
| `CLIENT_KEY` | — | если задан, запросы без него отклоняются |

## Виды заданий

| Тип | local | vlm | human |
|---|---|---|---|
| `ImageToTextTask` (текст на картинке) | ✓ | ✓ | ✓ |
| `GridTask` (сетка, ответ — номера клеток с 1) | | ✓ | ✓ |
| `CoordinatesTask` (куда кликнуть, ответ — пиксели) | | ✓ | ✓ |
| `SlideMatchTask` (свой: `target` + `background`, ответ — `x`, `y`) | ✓ | | |
| Любой другой тип 2Captcha (токены, повороты…) | | | ✓ |

В ответе `getTaskResult` кроме полей 2Captcha есть `layer`, `confidence`, `notes` (что пробовали до этого) и `lowConfidence`. Статистика слоёв — `GET /stats`.

## Проверка

```bash
.venv/bin/python -m pytest -q
```
