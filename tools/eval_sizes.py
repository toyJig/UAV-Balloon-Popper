#!/usr/bin/env python3
"""Measure how detection accuracy changes with inference input size, for
picking the size the Pi runs at (smaller = faster on the Pi CPU, but small
or distant targets get harder to see).

Reports per-class precision/recall at a fixed confidence (the threshold
MLDetector uses) for:
  * the full split, and
  * "sparse" balloon images with 1..--sparse-max balloon boxes - the scenes
    closest to the real use case (a few balloons, not a dense party arch,
    where box granularity in the labels dominates the score).

Usage:
    python tools/eval_sizes.py --weights runs/lockon/balloon_drone_v1/weights/best.pt \\
        --split-dirs datasets/combined/valid datasets/combined/test --sizes 640 480 416 320
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_breakdown import NAMES, load_gt, match  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 480, 416, 320])
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--sparse-max", type=int, default=3)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    images = []
    for d in args.split_dirs:
        images += [(p, d) for p in sorted((d / "images").iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    print(f"{len(images)} images from {[str(d) for d in args.split_dirs]}, conf >= {args.conf}, IoU >= 0.5")
    print(f"{'imgsz':>5s} | {'drone P/R':>13s} | {'balloon P/R (all)':>18s} | {'balloon P/R (1-%d boxes)' % args.sparse_max:>22s} | sparse imgs")
    for size in args.sizes:
        totals = defaultdict(int)
        sparse = defaultdict(int)
        n_sparse = 0
        for start in range(0, len(images), 64):
            batch = images[start:start + 64]
            results = model.predict([str(p) for p, _ in batch], imgsz=size, conf=args.conf, verbose=False)
            for (img_path, d), r in zip(batch, results):
                h, w = r.orig_shape
                gts = [(c, (x1, y1, x2, y2)) for c, x1, y1, x2, y2 in load_gt(d / "labels" / (img_path.stem + ".txt"), w, h)]
                preds = [(int(c), float(s), tuple(map(float, b)))
                         for c, s, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
                tp, fps, fns = match(preds, gts)
                n_balloon_gt = sum(1 for g in gts if g[0] == 0)
                is_sparse = img_path.stem.startswith("balloon-") and 1 <= n_balloon_gt <= args.sparse_max
                n_sparse += is_sparse
                for c in NAMES:
                    for bucket in ([totals, sparse] if is_sparse else [totals]):
                        bucket[f"{c}_tp"] += tp[c]
                        bucket[f"{c}_fp"] += sum(1 for f in fps if f[0] == c)
                        bucket[f"{c}_fn"] += sum(1 for f in fns if f[0] == c)

        def pr(bucket, c):
            tp, fp, fn = bucket[f"{c}_tp"], bucket[f"{c}_fp"], bucket[f"{c}_fn"]
            p = tp / (tp + fp) if tp + fp else float("nan")
            r = tp / (tp + fn) if tp + fn else float("nan")
            return f"{p:.3f}/{r:.3f}"

        print(f"{size:5d} | {pr(totals, 1):>13s} | {pr(totals, 0):>18s} | {pr(sparse, 0):>22s} | {n_sparse}")


if __name__ == "__main__":
    main()
