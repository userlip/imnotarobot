#!/bin/sh
# Builds "Я не робот.app" into dist/ and copies it to ~/Applications (older copies go to the Trash).
set -e
cd "$(dirname "$0")"
APP="Я не робот.app"
../.venv/bin/pyinstaller --noconfirm --distpath ../dist --workpath ../build CaptchaStack.spec
"../dist/$APP/Contents/MacOS/YaNeRobot" --selftest
mkdir -p "$HOME/Applications"
for old in "$HOME/Applications/$APP" "$HOME/Applications/Captcha Stack.app"; do
  if [ -e "$old" ]; then trash "$old"; fi
done
ditto "../dist/$APP" "$HOME/Applications/$APP"
echo "installed: $HOME/Applications/$APP"
