#!/usr/bin/env python3
"""Check that the Pi inference path (MLDetector + NcnnYoloBackend: our own
letterbox, NCNN forward pass, decode and NMS) gives the same detections as
Ultralytics' PyTorch inference of the same trained weights.

Catches export/preprocessing mistakes (wrong channel order, missing /255
normalization, wrong output layout, wrong un-letterboxing) on a laptop,
before anything goes to the Pi. Needs both `ultralytics` and `ncnn`.

For each image, every PyTorch detection is matched to the best NCNN
detection of the same class; reports how many matched at IoU >= 0.9 and
the confidence difference.

Usage:
    python tools/check_ncnn_parity.py --weights models/export_640/balloon_drone_v1.pt \\
        --ncnn models/export_640/balloon_drone_v1_ncnn_model --images datasets/combined/test/images --count 300
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from error_breakdown import iou  # noqa: E402
from lockon.vision.ml_detector import MLDetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--ncnn", required=True, help="Ultralytics NCNN export folder")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from ultralytics import YOLO

    detector = MLDetector(model_path=args.ncnn, conf_threshold=args.conf)
    size_w, size_h = detector.input_size
    torch_model = YOLO(args.weights)
    print(f"NCNN input size {detector.input_size}, classes {detector.class_names}")

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.count]

    n_torch = n_ncnn = matched = 0
    conf_diffs, ious = [], []
    worst = []
    for p in paths:
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ours = detector.detect(rgb)
        # Ultralytics expects BGR numpy input and imgsz as (h, w); rect=False
        # forces the same fixed-size letterbox our pipeline uses
        r = torch_model.predict(bgr, imgsz=(size_h, size_w), conf=args.conf, verbose=False, rect=False)[0]
        ref = [(detector.class_names[int(c)], float(s), tuple(map(float, b)))
               for c, s, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
        n_torch += len(ref)
        n_ncnn += len(ours)
        for name, conf, box in ref:
            best, best_d = 0.0, None
            for d in ours:
                if d.class_name != name:
                    continue
                s = iou(box, (d.x, d.y, d.x + d.w, d.y + d.h))
                if s > best:
                    best, best_d = s, d
            ious.append(best)
            if best >= 0.9:
                matched += 1
                conf_diffs.append(abs(best_d.confidence - conf))
            else:
                worst.append((best, p.name, name, conf))

    print(f"images: {len(paths)}  detections: pytorch={n_torch} ncnn={n_ncnn}")
    print(f"pytorch detections matched by ncnn at IoU>=0.9: {matched}/{n_torch} ({100.0 * matched / max(1, n_torch):.1f}%)")
    if conf_diffs:
        print(f"confidence |diff| on matches: mean={np.mean(conf_diffs):.4f} max={np.max(conf_diffs):.4f}")
    if ious:
        print(f"IoU with best same-class ncnn box: median={np.median(ious):.3f} min={np.min(ious):.3f}")
    for best, name, cls, conf in sorted(worst)[:8]:
        print(f"  unmatched: {name} {cls} conf={conf:.2f} best IoU={best:.2f}")


if __name__ == "__main__":
    main()
