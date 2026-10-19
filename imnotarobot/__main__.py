import argparse
import os
import signal
import threading

from .control import Controller


def main() -> None:
    p = argparse.ArgumentParser(prog="imnotarobot", description="I'm Not a Robot: a free-first captcha solver with a 2Captcha-compatible API")
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    a = p.parse_args()
    c = Controller()
    c.cfg.port = a.port
    for line in c.setup_from_env():
        print("configured from environment:", line, flush=True)
    c.start_server(host=a.host)
    key = c.cfg.client_key
    access = f"{key[:5]}… (CLIENT_KEY sets your own)" if c.cfg.require_client_key else "not required"
    print(f"I'm Not a Robot is listening on {a.host}:{c.port}; access key: {access}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    c.stop_server()


if __name__ == "__main__":
    main()
