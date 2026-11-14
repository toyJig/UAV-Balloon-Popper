#!/usr/bin/env python3
"""Build a YOLO-format balloon dataset from LVIS v1 (annotations CC BY 4.0,
images are COCO 2017 images served from images.cocodataset.org). No
account needed.

LVIS is "federated": each category is only exhaustively labeled in some
images. Filtering choices, each to keep labels trustworthy:

* Only images with at least one "balloon" box.
* Drop the image if "balloon" is in its not_exhaustive_category_ids (some
  balloons in it were left unlabeled).
* Drop the image if it also has any other balloon-like category (e.g. a
  hot-air balloon category), so "balloon" means party/toy balloon only.
* Drop images with more than --max-boxes balloon boxes.

Splits: LVIS train -> our train. LVIS val -> split in half into our valid
and test. LVIS train/val are image-disjoint.

Usage:
    python tools/fetch_lvis_balloons.py --lvis-dir raw_datasets/lvis --out datasets/lvis_balloon [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def load_lvis(path_zip: Path) -> dict:
    with zipfile.ZipFile(path_zip) as z:
        return json.loads(z.read(z.namelist()[0]))


def select_images(data: dict, max_boxes: int):
    cats = data["categories"]
    balloon_like = [c for c in cats if "balloon" in c["name"]]
    balloon = next(c for c in balloon_like if c["name"] == "balloon")
    other_ids = {c["id"] for c in balloon_like if c["id"] != balloon["id"]}
    print("  balloon-like categories:", [(c["id"], c["name"], c["synset"]) for c in balloon_like])

    boxes = defaultdict(list)
    other_imgs = set()
    for a in data["annotations"]:
        if a["category_id"] == balloon["id"]:
            boxes[a["image_id"]].append(a["bbox"])
        elif a["category_id"] in other_ids:
            other_imgs.add(a["image_id"])

    images = {im["id"]: im for im in data["images"]}
    stats = {"candidates": len(boxes), "not_exhaustive": 0, "other_balloon_type": 0, "too_many_boxes": 0}
    kept = {}
    for image_id, bxs in boxes.items():
        im = images[image_id]
        if balloon["id"] in im.get("not_exhaustive_category_ids", []):
            stats["not_exhaustive"] += 1
        elif image_id in other_imgs:
            stats["other_balloon_type"] += 1
        elif len(bxs) > max_boxes:
            stats["too_many_boxes"] += 1
        else:
            kept[image_id] = (im, bxs)
    return kept, stats


def to_yolo_lines(im: dict, bxs) -> "list[str]":
    w_img, h_img = float(im["width"]), float(im["height"])
    lines = []
    for x, y, w, h in bxs:
        x1, y1 = max(0.0, x / w_img), max(0.0, y / h_img)
        x2, y2 = min(1.0, (x + w) / w_img), min(1.0, (y + h) / h_img)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return lines


def download_and_resize(url: str, dst: Path, max_side: int, timeout_s: float = 30.0) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            data = resp.read()
    except Exception:
        return False
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return False
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lvis-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-boxes", type=int, default=20)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    assignments = {"train": [], "valid": [], "test": []}

    print("[lvis train] loading ...", flush=True)
    kept, stats = select_images(load_lvis(args.lvis_dir / "lvis_v1_train.json.zip"), args.max_boxes)
    print(f"[lvis train] {stats} -> kept {len(kept)} images, {sum(len(b) for _, b in kept.values())} boxes", flush=True)
    assignments["train"] = list(kept.values())

    print("[lvis val] loading ...", flush=True)
    kept, stats = select_images(load_lvis(args.lvis_dir / "lvis_v1_val.json.zip"), args.max_boxes)
    print(f"[lvis val] {stats} -> kept {len(kept)} images, {sum(len(b) for _, b in kept.values())} boxes", flush=True)
    val_items = sorted(kept.values(), key=lambda t: t[0]["id"])
    rng.shuffle(val_items)
    half = len(val_items) // 2
    assignments["valid"], assignments["test"] = val_items[:half], val_items[half:]

    jobs = []
    for split, items in assignments.items():
        img_dir, lbl_dir = args.out / split / "images", args.out / split / "labels"
        if not args.dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
        for im, bxs in items:
            stem = f"lvis_{im['id']}"
            jobs.append((im["coco_url"], img_dir / f"{stem}.jpg", lbl_dir / f"{stem}.txt", to_yolo_lines(im, bxs)))
    print({k: len(v) for k, v in assignments.items()})
    if args.dry_run:
        print(f"dry run: would download {len(jobs)} images")
        return

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_and_resize, url, img, args.max_side): (img, lbl, lines)
                   for url, img, lbl, lines in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            img, lbl, lines = futures[fut]
            if fut.result():
                lbl.write_text("\n".join(lines) + "\n")
            else:
                failed += 1
                img.unlink(missing_ok=True)
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)} done, {failed} failed", flush=True)
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "names:\n  0: balloon\n"
    )
    print(f"DONE: {len(jobs) - failed} images written, {failed} failed downloads (dropped).")


if __name__ == "__main__":
    sys.exit(main())
