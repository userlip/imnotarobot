#!/bin/sh
# Builds "Captcha Stack.app" into dist/ and copies it to ~/Applications (an older copy goes to the Trash).
set -e
cd "$(dirname "$0")"
../.venv/bin/pyinstaller --noconfirm --distpath ../dist --workpath ../build CaptchaStack.spec
"../dist/Captcha Stack.app/Contents/MacOS/Captcha Stack" --selftest
mkdir -p "$HOME/Applications"
if [ -e "$HOME/Applications/Captcha Stack.app" ]; then trash "$HOME/Applications/Captcha Stack.app"; fi
ditto "../dist/Captcha Stack.app" "$HOME/Applications/Captcha Stack.app"
echo "installed: $HOME/Applications/Captcha Stack.app"
