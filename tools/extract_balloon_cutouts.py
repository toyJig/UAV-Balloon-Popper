#!/usr/bin/env python3
"""Cut individual balloons out of Open Images TRAIN photos using the
dataset's human-verified segmentation masks, for pasting onto sky
backgrounds (tools/make_sky_composites.py).

Only images already in datasets/openimages_balloon/train are used (so all
the earlier filtering - no group boxes, no hot-air balloons, no drones -
still applies), minus anything listed in --exclude-list (the real sky test
images). Cutouts never come from valid/test images.

Quality filters per mask:
  * PredictedIoU >= --min-iou (Open Images' own mask quality estimate),
  * box not touching the image border (a cut-off balloon looks wrong in the sky),
  * box at least --min-side px on the shorter side in our 640px copy,
  * mask fills 62-92% of its box, box aspect 0.5-1.6,
  * one dominant connected component,
  * near-convex (solidity >= 0.93) and near-elliptical (IoU >= 0.85 with
    the fitted ellipse): drops balloons partly hidden behind other
    balloons, which would look like crescents in the sky, and balloon
    animals.

Output: RGBA PNGs (alpha = mask) cropped to the mask, plus index.csv.

Usage:
    python tools/extract_balloon_cutouts.py --out datasets/balloon_cutouts --exclude-list runs/sky_picked.txt
"""
from __future__ import annotations

import argparse
import csv
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BALLOON = "/m/01j51"
MIN_SOLIDITY = 0.93
MIN_ELLIPSE_IOU = 0.85


def balloon_shape(mask: np.ndarray) -> "tuple[bool, float, float]":
    """A whole, unoccluded balloon is close to convex and close to an
    ellipse. Occluded pieces (crescents, cut-off sides) and balloon animals
    are not. Returns (ok, solidity, IoU with the best-fit ellipse)."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = area / hull if hull > 0 else 0.0
    if len(c) < 5:
        return False, solidity, 0.0
    ellipse = np.zeros_like(m)
    cv2.ellipse(ellipse, cv2.fitEllipse(c), 1, -1)
    inter = np.logical_and(ellipse, m).sum()
    union = np.logical_or(ellipse, m).sum()
    e_iou = inter / union if union else 0.0
    return solidity >= MIN_SOLIDITY and e_iou >= MIN_ELLIPSE_IOU, solidity, e_iou


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-dir", type=Path, default=Path("raw_datasets/openimages/meta"))
    parser.add_argument("--masks-dir", type=Path, default=Path("raw_datasets/openimages/masks"))
    parser.add_argument("--images-dir", type=Path, default=Path("datasets/openimages_balloon/train/images"))
    parser.add_argument("--exclude-list", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.7)
    parser.add_argument("--min-side", type=int, default=24)
    args = parser.parse_args()

    have = {p.stem[3:]: p for p in args.images_dir.glob("oi_*.jpg")}
    excluded = set()
    if args.exclude_list:
        excluded = {Path(l).stem[3:] for l in args.exclude_list.read_text().split() if Path(l).stem.startswith("oi_")}
    rows = defaultdict(list)
    with open(args.meta_dir / "train-annotations-object-segmentation.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["LabelName"] == BALLOON and r["ImageID"] in have and r["ImageID"] not in excluded:
                rows[r["ImageID"][0]].append(r)
    total = sum(len(v) for v in rows.values())
    print(f"{total} candidate masks in {sum(len({r['ImageID'] for r in v}) for v in rows.values())} images")

    args.out.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(int)
    index = []
    for shard, shard_rows in sorted(rows.items()):
        with zipfile.ZipFile(args.masks_dir / f"train-masks-{shard}.zip") as z:
            for r in shard_rows:
                if float(r["PredictedIoU"] or 0) < args.min_iou:
                    stats["low_iou"] += 1
                    continue
                x1, x2, y1, y2 = (float(r[k]) for k in ("BoxXMin", "BoxXMax", "BoxYMin", "BoxYMax"))
                if x1 < 0.005 or y1 < 0.005 or x2 > 0.995 or y2 > 0.995:
                    stats["touches_border"] += 1
                    continue
                img = cv2.imread(str(have[r["ImageID"]]))
                h, w = img.shape[:2]
                if min((x2 - x1) * w, (y2 - y1) * h) < args.min_side:
                    stats["too_small"] += 1
                    continue
                mask = cv2.imdecode(np.frombuffer(z.read(r["MaskPath"]), np.uint8), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) > 127
                n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                if n < 2:
                    stats["empty_mask"] += 1
                    continue
                main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                if st[main, cv2.CC_STAT_AREA] < 0.85 * st[1:, cv2.CC_STAT_AREA].sum():
                    stats["fragmented"] += 1
                    continue
                mask = lab == main
                bx, by, bw, bh = st[main, :4]
                fill = mask.sum() / float(bw * bh)
                if not 0.62 <= fill <= 0.92 or not 0.5 <= bw / float(bh) <= 1.6:
                    stats["bad_fill_or_aspect"] += 1
                    continue
                shape_ok, solidity, ellipse_iou = balloon_shape(mask[by:by + bh, bx:bx + bw])
                if not shape_ok:
                    stats["not_balloon_shaped"] += 1  # occluded fragments, balloon animals, odd shapes
                    continue
                crop = img[by:by + bh, bx:bx + bw]
                alpha = (mask[by:by + bh, bx:bx + bw] * 255).astype(np.uint8)
                name = f"{r['ImageID']}_{r['BoxID']}.png"
                cv2.imwrite(str(args.out / name), np.dstack([crop, alpha]))
                index.append({"file": name, "image_id": r["ImageID"], "w": bw, "h": bh, "fill": f"{fill:.2f}",
                              "solidity": f"{solidity:.3f}", "ellipse_iou": f"{ellipse_iou:.3f}",
                              "predicted_iou": r["PredictedIoU"]})
                stats["kept"] += 1
    with open(args.out / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(index[0]))
        wr.writeheader()
        wr.writerows(index)
    print(dict(stats))
    print(f"wrote {len(index)} cutouts from {len({i['image_id'] for i in index})} images -> {args.out}")


if __name__ == "__main__":
    main()
