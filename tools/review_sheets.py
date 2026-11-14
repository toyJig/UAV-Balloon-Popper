#!/usr/bin/env python3
"""Numbered contact sheets for picking images by eye. Writes an index CSV
(number -> image path) next to the sheets, so a hand-picked list of
numbers can be turned back into paths with --pick.

Usage:
    python tools/review_sheets.py --scores runs/sky_scores.csv --min-score 0.3 --out-dir runs/review_sky
    python tools/review_sheets.py --index runs/review_sky/index.csv --pick 0 3 7-12 --out runs/sky_picked.txt
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


def parse_picks(tokens):
    out = set()
    for t in tokens:
        if "-" in t:
            a, b = (int(v) for v in t.split("-"))
            out.update(range(a, b + 1))
        else:
            out.add(int(t))
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--per-sheet", type=int, default=48)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--tile", type=int, default=225)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--pick", nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.pick:
        rows = list(csv.DictReader(open(args.index)))
        by_num = {int(r["number"]): r["image"] for r in rows}
        picked = [by_num[n] for n in parse_picks(args.pick)]
        args.out.write_text("\n".join(picked) + "\n")
        print(f"wrote {len(picked)} paths to {args.out}")
        return

    rows = [r for r in csv.DictReader(open(args.scores)) if float(r["clip_sky"]) >= args.min_score]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "image", "split", "clip_sky"])
        for n, r in enumerate(rows):
            w.writerow([n, r["image"], r["split"], r["clip_sky"]])
    for s in range(0, len(rows), args.per_sheet):
        tiles = []
        for n in range(s, min(s + args.per_sheet, len(rows))):
            p = Path(rows[n]["image"])
            t = render(p, p.parent.parent / "labels" / (p.stem + ".txt"), args.tile, {0: "b"})
            cv2.rectangle(t, (0, 0), (60, 30), (0, 0, 0), -1)
            cv2.putText(t, str(n), (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            tiles.append(t)
        while len(tiles) % args.cols:
            tiles.append(np.full((args.tile, args.tile, 3), 40, np.uint8))
        sheet = np.vstack([np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)])
        out = args.out_dir / f"sheet_{s // args.per_sheet}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"wrote {out} ({s}-{min(s + args.per_sheet, len(rows)) - 1})")


if __name__ == "__main__":
    main()
