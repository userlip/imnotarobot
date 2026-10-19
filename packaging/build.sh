#!/bin/sh
# Builds "ImNotARobot.app" (shown as «Я не робот» on a Russian Mac) into dist/ and copies it to ~/Applications (older copies go to the Trash).
set -e
cd "$(dirname "$0")"
APP="ImNotARobot.app"
../.venv/bin/pyinstaller --noconfirm --distpath ../dist --workpath ../build ImNotARobot.spec
"../dist/$APP/Contents/MacOS/ImNotARobot" --selftest
mkdir -p "$HOME/Applications"
for old in "$HOME/Applications/$APP" "$HOME/Applications/Я не робот.app" "$HOME/Applications/Captcha Stack.app"; do
  if [ -e "$old" ]; then trash "$old"; fi
done
ditto "../dist/$APP" "$HOME/Applications/$APP"
echo "installed: $HOME/Applications/$APP"
