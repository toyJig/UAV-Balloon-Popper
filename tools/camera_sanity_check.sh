#!/usr/bin/env bash
# Task 4 - run ON THE PI with a monitor + keyboard attached (desktop
# session, not headless). Confirms Camera Module 3, the ribbon connection,
# and the libcamera stack all work, before any pipeline code is built.
#
# Usage: ./tools/camera_sanity_check.sh
set -euo pipefail

OUT_DIR="$HOME/lockon-camera-check"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

echo "== Camera detection =="
rpicam-hello --list-cameras || libcamera-hello --list-cameras

echo
echo "== Live preview (5s) - a window should appear on the monitor showing the live feed =="
rpicam-hello -t 5000 || libcamera-hello -t 5000

echo
echo "== Still capture =="
STILL="$OUT_DIR/still_$STAMP.jpg"
rpicam-still -o "$STILL" || libcamera-still -o "$STILL"
echo "Saved $STILL"

echo
echo "== 5s video clip (forward-facing framing check) =="
VID="$OUT_DIR/clip_$STAMP.h264"
rpicam-vid -t 5000 -o "$VID" || libcamera-vid -t 5000 -o "$VID"
echo "Saved $VID"

echo
echo "Done. Open $STILL and $VID on the desktop to confirm focus, framing, and exposure look sane"
echo "for a forward-facing outdoor shot."
