#!/usr/bin/env python3
"""Summarize a YOLO split: images, boxes per class, empty images, and box
size (longest side in pixels) percentiles per class.

Usage:
    python tools/label_stats.py --split-dir datasets/synthetic_sky/train
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args()

    sizes = defaultdict(list)
    images = empty = 0
    for img_path in sorted((args.split_dir / "images").glob("*.jpg")):
        lbl = args.split_dir / "labels" / (img_path.stem + ".txt")
        lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
        images += 1
        if not lines:
            empty += 1
            continue
        h, w = cv2.imread(str(img_path)).shape[:2]
        for c, _, _, bw, bh in lines:
            sizes[int(c)].append(max(float(bw) * w, float(bh) * h))
    print(f"images={images} empty={empty}")
    for c, s in sorted(sizes.items()):
        p = np.percentile(s, [10, 25, 50, 75, 90]).round(1).tolist()
        print(f"class {c}: boxes={len(s)} longest-side px p10/25/50/75/90 = {p}")


if __name__ == "__main__":
    main()
