#!/bin/sh
# Packs dist/ImNotARobot.app (built by build.sh) into dist/ImNotARobot.dmg with a link to Applications.
set -e
cd "$(dirname "$0")/.."
STAGE=build/dmg
if [ -e "$STAGE" ]; then trash "$STAGE"; fi
mkdir -p "$STAGE"
ditto dist/ImNotARobot.app "$STAGE/ImNotARobot.app"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "I'm Not a Robot" -srcfolder "$STAGE" -ov -format UDZO dist/ImNotARobot.dmg
