# I'm Not a Robot · imnotarobot

**English** · [Русский](README.ru.md)

A captcha solver that lives in the macOS menu bar. Programs talk to it through the 2Captcha / RuCaptcha API (`createTask` / `getTaskResult`, `in.php` / `res.php`), agents through an MCP server. Every captcha climbs a ladder from free to paid:

1. **Local model**: ddddocr and piece matching on this Mac. Text captchas and sliders, free, milliseconds.
2. **Vision model** (Gemini Flash by default, any OpenAI-compatible endpoint works): text, "select all buses" grids, clicks on a picture. One request per captcha; a second one only when the answer disagrees with the local model. A silent service gets one immediate retry, then the next step takes over.
3. **Backup vision model** (Gemini High): when the first one is unsure or wrong.
4. **Browser**: reCAPTCHA v2/v3, hCaptcha and Turnstile checkboxes. An invisible Chrome ticks the box and hands picture grids and hCaptcha canvas tasks to the vision model.
5. **RuCaptcha / 2Captcha**: human workers, the last and only paid step.

The next step runs when the previous one failed, was unsure, or cannot do that kind of captcha. Two unsure steps that agree are accepted.

Feedback loop:
- `reportIncorrect` remembers that a step was wrong on that image; with `"retry": true` the next step solves it right away.
- A step that is wrong on most captchas of one kind (type + instruction) is skipped for that kind.
- Complaints about human answers are forwarded to RuCaptcha / 2Captcha, which refunds them.

Next on the roadmap: hands for the model, a toolset with which the vision model solves any captcha on its own in a look → act → check loop (see `OPEN.md`).

The app speaks the system language: English and Russian for now.

## Run

Build the app with `packaging/build.sh`; it installs into `~/Applications`. Keys, models, the order of steps and daily limits are set in its window. Keys live in the macOS Keychain.

Without the app:

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -e ".[test]"
.venv/bin/python -m captcha_stack --port 8765
```

| Variable | Default | What |
|---|---|---|
| `LAYERS` | `local,vlm,browser,human` | order of steps |
| `HUMAN_API_KEY` | — | 2Captcha or RuCaptcha key; without it the last step is off |
| `CLIENT_KEY` | — | requests without it are rejected |

## Task types

| Type | local | vision | browser | humans |
|---|---|---|---|---|
| `ImageToTextTask` (text on a picture) | ✓ | ✓ | | ✓ |
| `GridTask` (grid, answer is tile numbers from 1) | | ✓ | | ✓ |
| `CoordinatesTask` (where to click, answer in pixels) | | ✓ | | ✓ |
| `SlideMatchTask` (own: `target` + `background`, answer `x`, `y`) | ✓ | | | |
| `RecaptchaV2TaskProxyless`, `RecaptchaV3TaskProxyless`, `HCaptchaTaskProxyless`, `TurnstileTaskProxyless` | | | ✓ | ✓ |
| Any other 2Captcha type (rotate, GeeTest…) | | | | ✓ |

Besides the 2Captcha fields, `getTaskResult` returns `layer`, `confidence`, `notes` (what was tried before) and `lowConfidence`. Step statistics: `GET /stats`.

## Tests

```bash
.venv/bin/python -m pytest -q
```
