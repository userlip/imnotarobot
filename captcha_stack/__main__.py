import argparse

import uvicorn

from .server import create_app

p = argparse.ArgumentParser(prog="captcha-stack", description="Layered captcha solver with a 2Captcha-style API")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8765)
a = p.parse_args()
uvicorn.run(create_app(), host=a.host, port=a.port)
