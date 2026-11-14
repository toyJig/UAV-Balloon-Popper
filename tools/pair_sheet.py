#!/usr/bin/env python3
"""Show image pairs side by side (one pair per row) for duplicate checks.

Usage:
    python tools/pair_sheet.py --pairs a1.jpg b1.jpg a2.jpg b2.jpg --out runs/sheets/pairs.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height", type=int, default=260)
    args = parser.parse_args()
    if len(args.pairs) % 2:
        raise SystemExit("--pairs needs an even number of paths")
    rows = []
    for a, b in zip(args.pairs[0::2], args.pairs[1::2]):
        ims = []
        for p in (a, b):
            im = cv2.imread(p)
            ims.append(cv2.resize(im, (int(im.shape[1] * args.height / im.shape[0]), args.height)))
        row = np.hstack([ims[0], np.full((args.height, 10, 3), 255, np.uint8), ims[1]])
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((args.height, width - r.shape[1], 3), 40, np.uint8)]) for r in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
