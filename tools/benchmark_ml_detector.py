#!/usr/bin/env python3
"""Task 8 demo/benchmark - run ON THE PI (also works on a laptop with the
`ncnn` package) against an exported NCNN model and a held-out YOLO-format
image set. Reports per-class precision/recall at IoU >= 0.5 for the
confidence threshold MLDetector uses, and the time spent in
MLDetector.detect() (letterbox + NCNN forward + decode + NMS; image
loading from disk is excluded).

Expected layout:
    <held_out_dir>/images/*.jpg
    <held_out_dir>/labels/*.txt   (YOLO format: class cx cy w h, normalized 0-1)

Usage:
    python tools/benchmark_ml_detector.py --model-path models/balloon_drone_v1/ncnn_640x480 \\
        --held-out-dir datasets/combined/test --max-images 800
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from error_breakdown import load_gt, match  # noqa: E402  (pure helpers; no ultralytics needed)
from lockon.vision.ml_detector import MLDetector  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="NCNN export folder")
    parser.add_argument("--held-out-dir", required=True, type=Path)
    parser.add_argument("--conf-threshold", type=float, default=0.4)
    parser.add_argument("--max-images", type=int, default=0, help="random subset size (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    detector = MLDetector(model_path=args.model_path, conf_threshold=args.conf_threshold)
    names = detector.class_names
    print(f"model input {detector.input_size}, classes {names}, conf >= {args.conf_threshold}")

    paths = sorted(p for p in (args.held_out_dir / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"No images found in {args.held_out_dir / 'images'}")
    if args.max_images and len(paths) > args.max_images:
        random.Random(args.seed).shuffle(paths)
        paths = sorted(paths[: args.max_images])

    counts = defaultdict(int)
    times = []
    for i, img_path in enumerate(paths):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"warning: could not read {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gts = [(c, (x1, y1, x2, y2)) for c, x1, y1, x2, y2 in load_gt(args.held_out_dir / "labels" / (img_path.stem + ".txt"), w, h)]

        start = time.perf_counter()
        detections = detector.detect(rgb)
        if i >= 3:  # skip warm-up calls
            times.append(time.perf_counter() - start)

        preds = [(names.index(d.class_name), d.confidence, (d.x, d.y, d.x + d.w, d.y + d.h)) for d in detections]
        tp, fps, fns = match(preds, gts)
        for c in range(len(names)):
            counts[(c, "tp")] += tp[c]
            counts[(c, "fp")] += sum(1 for f in fps if f[0] == c)
            counts[(c, "fn")] += sum(1 for f in fns if f[0] == c)

    print(f"images evaluated: {len(paths)}")
    for c, name in enumerate(names):
        tp, fp, fn = counts[(c, "tp")], counts[(c, "fp")], counts[(c, "fn")]
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  {name:8s} precision={p:.3f} recall={r:.3f}  (tp={tp} fp={fp} fn={fn})")
    if times:
        ms = np.array(times) * 1000
        print(f"detect() time: mean {ms.mean():.1f} ms, p50 {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms "
              f"-> {1000 / ms.mean():.1f} fps for the detector alone")


if __name__ == "__main__":
    main()
