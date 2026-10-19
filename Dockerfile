# I'm Not a Robot as a server: the same API and MCP as the Mac app, without the window.
#   docker build -t imnotarobot .
#   docker run -p 8765:8765 -e GEMINI_API_KEY=... -e CLIENT_KEY=... -v imnotarobot:/data imnotarobot
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 IMNOTAROBOT_HOME=/data HOST=0.0.0.0 PORT=8765 PLAYWRIGHT_BROWSERS_PATH=/opt/browsers
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md LICENSE ./
COPY imnotarobot ./imnotarobot
RUN uv pip install --system --no-cache . \
 && patchright install --with-deps chromium \
 && apt-get install -y --no-install-recommends fonts-dejavu-core \
 && apt-get clean

VOLUME /data
EXPOSE 8765
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["python", "-m", "imnotarobot"]
