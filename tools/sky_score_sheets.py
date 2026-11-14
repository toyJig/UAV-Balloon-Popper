#!/usr/bin/env python3
"""Contact sheets of images from tools/score_sky_scenes.py's CSV, grouped
into CLIP sky-score bands, for choosing a sky threshold by eye.

Usage:
    python tools/sky_score_sheets.py --scores runs/sky_scores.csv --out-dir runs/sheets
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

BANDS = ((0.5, 1.01, "high"), (0.3, 0.5, "mid"), (0.15, 0.3, "low"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-sheet", type=int, default=40)
    parser.add_argument("--tile", type=int, default=225)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.scores)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for lo, hi, name in BANDS:
        band = [r for r in rows if lo <= float(r["clip_sky"]) < hi]
        step = max(1, len(band) // args.per_sheet)
        tiles = []
        for r in band[::step][: args.per_sheet]:
            p = Path(r["image"])
            t = render(p, p.parent.parent / "labels" / (p.stem + ".txt"), args.tile, {0: "balloon"})
            cv2.putText(t, f"{float(r['clip_sky']):.2f} ring={r['sky_ring_fraction']}", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            tiles.append(t)
        if not tiles:
            continue
        while len(tiles) % 8:
            tiles.append(np.full((args.tile, args.tile, 3), 40, np.uint8))
        sheet = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
        cv2.imwrite(str(args.out_dir / f"sky_{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"{name}: {len(band)} images in [{lo}, {hi})")


if __name__ == "__main__":
    main()
