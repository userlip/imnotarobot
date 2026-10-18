"""Entry point of the bundled app. `--selftest` checks the bundle without any window or menu bar icon."""

import sys


def selftest() -> int:
    import asyncio
    import json
    import tempfile

    import keyring
    import rumps  # noqa: F401
    import uvicorn  # noqa: F401

    from captcha_stack import testset
    from captcha_stack.config import Config
    from captcha_stack.service import Service

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(layers=["local"], state_path=f"{tmp}/state.json")
        case = testset.generated(1, seed=3)[0]
        r = asyncio.run(Service(cfg).chain.solve(case["task"]))
    import WebKit  # noqa: F401

    from captcha_stack.menubar import ICON
    from captcha_stack.ui import PAGE

    print(json.dumps({"icon": ICON.exists(), "page": PAGE.exists(), "keyring": type(keyring.get_keyring()).__module__, "expected": case["expected"],
                      "solution": r.solution, "layer": r.layer, "confidence": round(r.confidence, 3)}))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    from captcha_stack.menubar import main

    main()
