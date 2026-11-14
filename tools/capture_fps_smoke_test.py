#!/usr/bin/env python3
"""Task 6 demo - run ON THE PI. Saves a short frame sequence from the real
camera and reports the measured capture fps, validating the pipeline in
isolation before any detector is wired in.

Usage:
    python3 tools/capture_fps_smoke_test.py --resolution 640x480 --save-dir ~/lockon-capture-check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from lockon.vision.capture import CameraCapture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="640x480")
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--save-dir", default=str(Path.home() / "lockon-capture-check"))
    parser.add_argument("--save-count", type=int, default=10)
    parser.add_argument("--fps-sample-count", type=int, default=120)
    args = parser.parse_args()

    w, h = (int(v) for v in args.resolution.split("x"))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with CameraCapture(resolution=(w, h), framerate=args.framerate) as cap:
        print(f"Saving {args.save_count} sample frames to {save_dir} ...")
        for i, frame in enumerate(cap.frames(count=args.save_count)):
            np.save(save_dir / f"frame_{i:03d}.npy", frame)

        print(f"Measuring fps over {args.fps_sample_count} frames ...")
        fps = cap.measure_fps(num_frames=args.fps_sample_count)
        print(f"Measured capture fps: {fps:.1f}")


if __name__ == "__main__":
    main()
