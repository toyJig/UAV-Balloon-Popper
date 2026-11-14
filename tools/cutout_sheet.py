#!/usr/bin/env python3
"""Tile a random sample of RGBA balloon cutouts on a sky-blue background,
to check mask quality by eye.

Usage:
    python tools/cutout_sheet.py --cutouts datasets/balloon_cutouts --out runs/sheets/cutouts.jpg
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--tile", type=int, default=120)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.cutouts.glob("*.png"))
    random.Random(args.seed).shuffle(files)
    tiles = []
    for p in files[: args.count]:
        rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        h, w = rgba.shape[:2]
        s = (args.tile - 10) / max(h, w)
        rgba = cv2.resize(rgba, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        tile = np.zeros((args.tile, args.tile, 3), np.float32)
        tile[:] = (235, 190, 130)  # sky blue (BGR)
        y0, x0 = (args.tile - rgba.shape[0]) // 2, (args.tile - rgba.shape[1]) // 2
        a = rgba[..., 3:4].astype(np.float32) / 255
        region = tile[y0:y0 + rgba.shape[0], x0:x0 + rgba.shape[1]]
        tile[y0:y0 + rgba.shape[0], x0:x0 + rgba.shape[1]] = region * (1 - a) + rgba[..., :3] * a
        tiles.append(tile.astype(np.uint8))
    while len(tiles) % args.cols:
        tiles.append(np.full((args.tile, args.tile, 3), 40, np.uint8))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack([np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
