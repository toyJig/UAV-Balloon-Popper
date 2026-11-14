#!/usr/bin/env python3
"""Zoomed crops around each labeled box, to check label tightness on small
objects that are too small to judge on a full-frame contact sheet.

Usage:
    python tools/box_crops_sheet.py --split-dir datasets/sky_test_clip --out runs/sheets/clip_crops.jpg --count 40
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--tile", type=int, default=180)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--context", type=float, default=1.5, help="crop size as a multiple of the box size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cls", type=int, default=None, help="only boxes of this class id")
    args = parser.parse_args()

    labels = sorted((args.split_dir / "labels").glob("*.txt"))
    random.Random(args.seed).shuffle(labels)
    tiles = []
    for lbl in labels:
        lines = [l.split() for l in lbl.read_text().splitlines()
                 if len(l.split()) == 5 and (args.cls is None or int(l.split()[0]) == args.cls)]
        if not lines:
            continue
        img_path = next((args.split_dir / "images").glob(lbl.stem + ".*"))
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        _, cx, cy, bw, bh = (float(v) for v in lines[0])
        side = max(bw * w, bh * h) * (1 + args.context)
        x1, y1 = int(cx * w - side / 2), int(cy * h - side / 2)
        x2, y2 = int(cx * w + side / 2), int(cy * h + side / 2)
        pad = int(side)
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        crop = padded[y1 + pad:y2 + pad, x1 + pad:x2 + pad].copy()
        s = args.tile / crop.shape[1]
        crop = cv2.resize(crop, (args.tile, args.tile), interpolation=cv2.INTER_NEAREST)
        bx1, by1 = int(((cx - bw / 2) * w - x1) * s), int(((cy - bh / 2) * h - y1) * s)
        bx2, by2 = int(((cx + bw / 2) * w - x1) * s), int(((cy + bh / 2) * h - y1) * s)
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 255, 0), 1)
        cv2.putText(crop, f"{int(bw * w)}x{int(bh * h)}px", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(crop)
        if len(tiles) >= args.count:
            break
    while len(tiles) % args.cols:
        tiles.append(np.full((args.tile, args.tile, 3), 40, np.uint8))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack([np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
