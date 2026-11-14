#!/usr/bin/env python3
"""Tasks 7/8/9 live demo - run ON THE PI with a monitor + keyboard attached
(same desktop session set up in Task 4). Shows the live camera feed with
the active detector's bounding box(es) drawn on it in real time, so you
can point the camera at a real balloon/drone and watch detection happen
instead of only reviewing recorded clips afterward.

Usage:
    python tools/live_overlay_demo.py --mode fallback
    python tools/live_overlay_demo.py --mode ml --model-path models/balloon_drone_v1/ncnn_640x480
    python tools/live_overlay_demo.py --mode hybrid --model-path models/balloon_drone_v1/ncnn_640x480

Press 'q' in the preview window to quit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from lockon.vision.capture import CameraCapture
from lockon.vision.fallback_detector import ColorMotionDetector
from lockon.vision.hybrid import HybridDetector
from lockon.vision.ml_detector import MLDetector
from lockon.vision.types import DetectionSource

COLOR_BY_SOURCE = {
    DetectionSource.ML: (0, 255, 0),       # green
    DetectionSource.FALLBACK: (0, 165, 255),  # orange
}


def draw_detection(frame, detection, label_extra=""):
    color = COLOR_BY_SOURCE.get(detection.source, (255, 255, 255))
    x, y, w, h = int(detection.x), int(detection.y), int(detection.w), int(detection.h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"{detection.source.value} {detection.class_name} {detection.confidence:.2f}{label_extra}"
    cv2.putText(frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fallback", "ml", "hybrid"], required=True)
    parser.add_argument("--model-path", help="required for --mode ml/hybrid")
    parser.add_argument("--resolution", default="640x480")
    args = parser.parse_args()

    w, h = (int(v) for v in args.resolution.split("x"))

    fallback = ColorMotionDetector()
    ml = MLDetector(model_path=args.model_path) if args.model_path else None

    if args.mode == "fallback":
        process = lambda frame: fallback.detect(frame)
    elif args.mode == "ml":
        if ml is None:
            raise SystemExit("--mode ml requires --model-path")
        process = lambda frame: ml.detect(frame)
    else:
        if ml is None:
            raise SystemExit("--mode hybrid requires --model-path")
        hybrid = HybridDetector(ml, fallback, frame_size=(w, h))
        process = lambda frame: [d for d in [hybrid.process(frame)] if d is not None]

    with CameraCapture(resolution=(w, h)) as cap:
        for frame in cap.frames():
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for detection in process(frame):
                draw_detection(bgr, detection)
            cv2.imshow("lock-on live overlay", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
