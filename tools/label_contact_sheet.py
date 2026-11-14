#!/usr/bin/env python3
"""Draw YOLO boxes on a random sample of images and tile them into one
contact-sheet JPEG, to visually confirm labels line up with the objects
(catches wrong orientation, swapped x/y, wrong normalization, wrong class
mapping) before spending GPU hours on training.

Usage:
    python tools/label_contact_sheet.py --split-dir datasets/openimages_balloon/train --out runs/sheets/oi_train.jpg
    python tools/label_contact_sheet.py --split-dir datasets/combined/train --prefix balloon- --out runs/sheets/x.jpg
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}  # BGR: class 0 red, class 1 green
NAMES = {0: "balloon", 1: "drone"}


def render(img_path: Path, label_path: Path, tile: int, names: dict) -> np.ndarray:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            color = COLORS.get(c, (255, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, w // 200))
            cv2.putText(img, names.get(c, str(c)), (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, w / 900), color, 2)
    scale = tile / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.full((tile, tile, 3), 40, np.uint8)
    canvas[: img.shape[0], : img.shape[1]] = img
    cv2.putText(canvas, img_path.stem[:40], (4, tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--single-class-name", default=None,
                        help="name to show for class 0 when viewing a single-class source dataset")
    args = parser.parse_args()

    names = dict(NAMES)
    if args.single_class_name:
        names = {0: args.single_class_name}

    images = sorted(p for p in (args.split_dir / "images").iterdir() if p.name.startswith(args.prefix))
    random.Random(args.seed).shuffle(images)
    images = images[: args.count]
    tiles = [render(p, args.split_dir / "labels" / (p.stem + ".txt"), args.tile, names) for p in images]
    while len(tiles) % args.cols:
        tiles.append(np.full((args.tile, args.tile, 3), 40, np.uint8))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {args.out} ({len(images)} images)")


if __name__ == "__main__":
    main()
