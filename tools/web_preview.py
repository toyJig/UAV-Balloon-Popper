#!/usr/bin/env python3
"""Headless camera preview + optional detector overlay, streamed over HTTP.

For a Pi with no monitor: run this over SSH and open the printed URL in a
browser ON YOUR LAPTOP to watch the live camera feed (and, optionally, the
detector's boxes). Needs only the camera and the Pi's own power - the
flight controller does NOT have to be connected for this.

Usage (on the Pi, over SSH):
    python3 tools/web_preview.py                      # raw camera, no detector
    python3 tools/web_preview.py --mode fallback      # color/motion boxes
    python3 tools/web_preview.py --mode ml     --model-path models/balloon_drone_v2/ncnn_640x480
    python3 tools/web_preview.py --mode hybrid --model-path models/balloon_drone_v2/ncnn_640x480

Then on your laptop browser go to  http://<pi-ip>:8000/  (the script prints
the exact address). Ctrl+C on the Pi to stop.

Endpoints:  /  (viewer page)   /stream.mjpg  (raw MJPEG)   /snapshot.jpg
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from lockon.vision.capture import CameraCapture
from lockon.vision.fallback_detector import ColorMotionDetector
from lockon.vision.hybrid import HybridDetector
from lockon.vision.mjpeg import FrameHub, serve
from lockon.vision.types import DetectionSource

COLOR_BY_SOURCE = {
    DetectionSource.ML: (0, 255, 0),          # green (BGR)
    DetectionSource.FALLBACK: (0, 165, 255),  # orange
}


def build_processor(mode: str, model_path, frame_size):
    if mode == "none":
        return None
    fallback = ColorMotionDetector()
    if mode == "fallback":
        return lambda frame: fallback.detect(frame)
    from lockon.vision.ml_detector import MLDetector  # imported here so --mode none/fallback need no model
    if not model_path:
        raise SystemExit(f"--mode {mode} requires --model-path")
    ml = MLDetector(model_path=model_path)
    if mode == "ml":
        return lambda frame: ml.detect(frame)
    hybrid = HybridDetector(ml, fallback, frame_size=frame_size)
    return lambda frame: [d for d in [hybrid.process(frame)] if d is not None]


def draw(bgr, detection) -> None:
    color = COLOR_BY_SOURCE.get(detection.source, (255, 255, 255))
    x, y, w, h = int(detection.x), int(detection.y), int(detection.w), int(detection.h)
    cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
    label = f"{detection.source.value} {detection.class_name} {detection.confidence:.2f}"
    cv2.putText(bgr, label, (x, max(12, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def capture_loop(cap: CameraCapture, process, hub: FrameHub, rotate180: bool, jpeg_quality: int) -> None:
    last = time.monotonic()
    fps = 0.0
    try:
        for frame in cap.frames():
            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if rotate180:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            if process is not None:
                try:
                    for det in process(frame):
                        draw(bgr, det)
                except Exception as exc:  # keep streaming even if a detector hiccups
                    cv2.putText(bgr, f"detector error: {exc}", (8, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(bgr, f"{fps:4.1f} fps  {bgr.shape[1]}x{bgr.shape[0]}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok:
                hub.publish(buf.tobytes())
            if hub.stopped:
                break
    finally:
        hub.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["none", "fallback", "ml", "hybrid"], default="none")
    ap.add_argument("--model-path", help="NCNN export folder, required for --mode ml/hybrid")
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--rotate180", action="store_true", help="if the camera is mounted upside down")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    args = ap.parse_args()

    w, h = (int(v) for v in args.resolution.split("x"))
    process = build_processor(args.mode, args.model_path, (w, h))
    hub = FrameHub()

    cap = CameraCapture(resolution=(w, h))
    cap.start()
    worker = threading.Thread(target=capture_loop,
                              args=(cap, process, hub, args.rotate180, args.quality), daemon=True)
    worker.start()

    server = serve(hub, args.http_port)
    print(f"  (mode={args.mode}; Ctrl+C here to stop)\n")
    try:
        while worker.is_alive():
            worker.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()
        server.shutdown()
        cap.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
