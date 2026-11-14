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
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from lockon.vision.capture import CameraCapture
from lockon.vision.fallback_detector import ColorMotionDetector
from lockon.vision.hybrid import HybridDetector
from lockon.vision.types import DetectionSource

COLOR_BY_SOURCE = {
    DetectionSource.ML: (0, 255, 0),          # green (BGR)
    DetectionSource.FALLBACK: (0, 165, 255),  # orange
}


class FrameHub:
    """Holds the most recent JPEG; readers wait for the next one."""

    def __init__(self) -> None:
        self._jpeg = b""
        self._seq = 0
        self._cv = threading.Condition()
        self.stopped = False

    def publish(self, jpeg: bytes) -> None:
        with self._cv:
            self._jpeg = jpeg
            self._seq += 1
            self._cv.notify_all()

    def latest(self) -> bytes:
        with self._cv:
            return self._jpeg

    def wait_next(self, last_seq: int, timeout: float = 5.0):
        with self._cv:
            self._cv.wait_for(lambda: self._seq != last_seq or self.stopped, timeout=timeout)
            return self._jpeg, self._seq

    def stop(self) -> None:
        with self._cv:
            self.stopped = True
            self._cv.notify_all()


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


PAGE = b"""<!doctype html><html><head><title>lock-on camera</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center}
img{max-width:100vw;max-height:95vh}</style></head>
<body><img src="/stream.mjpg"></body></html>"""


def make_handler(hub: FrameHub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/snapshot.jpg":
                jpeg = hub.latest()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seq = -1
                try:
                    while not hub.stopped:
                        jpeg, seq = hub.wait_next(seq)
                        if not jpeg:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


def local_ip() -> str:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout.split()
        if out:
            return out[0]
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<pi-ip>"


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

    server = ThreadingHTTPServer(("0.0.0.0", args.http_port), make_handler(hub))
    ip = local_ip()
    print(f"\n  Open this on your laptop browser:  http://{ip}:{args.http_port}/\n"
          f"  (mode={args.mode}; Ctrl+C here to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()
        server.shutdown()
        cap.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
