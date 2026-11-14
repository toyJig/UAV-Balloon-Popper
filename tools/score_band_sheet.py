#!/usr/bin/env python3
"""Render a labeled contact sheet of the images whose hot-air score
(from filter_hot_air_balloons.py's CSV) falls inside a band, sorted by
score, so a filtering threshold can be checked by eye.

Usage:
    python tools/score_band_sheet.py --dataset datasets/openimages_balloon \\
        --scores datasets/openimages_balloon_quarantine/hot_air_scores.csv --low 0.5 --high 0.7 --out runs/sheets/band.jpg
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_contact_sheet import render  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=1.01)
    parser.add_argument("--column", default="p_hot_air", help="score column to band/sort on")
    parser.add_argument("--flagged", choices=["0", "1"], default=None, help="only rows with this flagged value")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max", type=int, default=36)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--tile", type=int, default=300)
    args = parser.parse_args()

    with open(args.scores, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if args.low <= float(r[args.column]) < args.high
                and (args.flagged is None or r["flagged"] == args.flagged)]
    rows.sort(key=lambda r: float(r[args.column]))
    if len(rows) > args.max:  # spread the sample evenly across the band
        step = len(rows) / args.max
        rows = [rows[int(i * step)] for i in range(args.max)]

    tiles = []
    for r in rows:
        split_dir = args.dataset / r["split"]
        img = split_dir / "images" / r["image"]
        tile = render(img, split_dir / "labels" / (img.stem + ".txt"), args.tile, {0: "balloon"})
        label = f"s={float(r['p_hot_air']):.2f}"
        if "p_crop_max" in r:
            label += f" c={float(r['p_crop_max']):.2f}"
        cv2.putText(tile, label, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(tile)
    if not tiles:
        print("no images in band")
        return
    while len(tiles) % args.cols:
        tiles.append(np.full((args.tile, args.tile, 3), 40, np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"wrote {args.out} ({len(rows)} images in [{args.low}, {args.high}))")


if __name__ == "__main__":
    main()
