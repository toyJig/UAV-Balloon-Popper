#!/usr/bin/env python3
"""Build a YOLO-format balloon dataset (plus hard-negative background
images) from Google Open Images V7. No account needed: annotation CSVs
come from storage.googleapis.com/openimages and images from the public
open-images-dataset S3 bucket.

Filtering choices (each one is there to keep labels trustworthy when this
set is merged with the drone set):

* Only images with at least one "Balloon" (/m/01j51) box.
* Drop the whole image if ANY balloon box is IsGroupOf=1. Open Images uses
  one box around a cluster of 5+ touching instances; keeping the image but
  dropping that box would leave unlabeled balloons, which trains the model
  to call real balloons background.
* Drop the whole image if any balloon box is IsDepiction=1 (cartoons,
  prints, drawings of balloons).
* Drop images verified (image-level label, Confidence=1) as "Hot air
  balloon", "Hot air ballooning" or "Cluster ballooning". Open Images has
  no separate boxable hot-air-balloon class, so those would otherwise be
  labeled "balloon".
* Drop images with more than --max-boxes balloon boxes (dense decoration
  arches, far from the balloon-in-the-sky use case).
* Drop images whose Rotation metadata is non-zero, and decode with EXIF
  orientation ignored, so pixels match the annotated orientation exactly.

Hard negatives: images containing Bird, Kite, Aircraft, Fixed-wing
aircraft or Helicopter boxes and no balloon box/label, written with empty
label files so the model learns not to fire on other things in the sky.

Splits follow Open Images' own train/validation/test split, so there is no
image overlap between our train/valid/test.

Usage:
    python tools/fetch_openimages_balloons.py --meta-dir raw_datasets/openimages/meta --out datasets/openimages_balloon
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

BALLOON = "/m/01j51"
EXCLUDE_IMAGE_LABELS = {
    "/m/09j06",    # Hot air balloon
    "/m/02p81ht",  # Hot air ballooning
    "/m/084x1x",   # Cluster ballooning
}
# Image-level labels (not boxable in Open Images) for drone-like objects. Any
# image verified to contain one is dropped from BOTH the balloon set and the
# negatives: the drone would be unlabeled, which teaches "drone = background".
DRONE_LIKE_IMAGE_LABELS = {
    "/m/0g2bc",    # Drone
    "/m/053scx",   # Radio-controlled aircraft
    "/m/074l3b",   # Radio-controlled helicopter
    "/m/0l8fl",    # Model aircraft
    "/m/0h8mhmy",  # Toy airplane
}
NEGATIVE_BOX_LABELS = {
    "/m/015p6",  # Bird
    "/m/02zt3",  # Kite
    "/m/0k5j",   # Aircraft
    "/m/0cmf2",  # Fixed-wing aircraft
    "/m/09ct_",  # Helicopter
}

# our split name -> (Open Images split name, bbox csv, image-level csv, rotation csv)
SPLITS = {
    "train": ("train", "oidv6-train-annotations-bbox.csv", "train-imagelabels.csv", "train-images-rotation.csv"),
    "valid": ("validation", "validation-annotations-bbox.csv", "validation-imagelabels.csv", "validation-images-rotation.csv"),
    "test": ("test", "test-annotations-bbox.csv", "test-imagelabels.csv", "test-images-rotation.csv"),
}
S3_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"


def scan_boxes(bbox_csv: Path):
    """Return (balloon_boxes, rejected_balloon_images, negative_candidates).

    balloon_boxes: image_id -> list of (xmin, xmax, ymin, ymax)
    """
    balloon_boxes = defaultdict(list)
    rejected = set()
    negatives = set()
    with open(bbox_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        i_id, i_label = col["ImageID"], col["LabelName"]
        i_xmin, i_xmax, i_ymin, i_ymax = col["XMin"], col["XMax"], col["YMin"], col["YMax"]
        i_group, i_depict = col["IsGroupOf"], col["IsDepiction"]
        for row in reader:
            label = row[i_label]
            if label == BALLOON:
                image_id = row[i_id]
                if row[i_group] == "1" or row[i_depict] == "1":
                    rejected.add(image_id)
                balloon_boxes[image_id].append(
                    (float(row[i_xmin]), float(row[i_xmax]), float(row[i_ymin]), float(row[i_ymax]))
                )
            elif label in NEGATIVE_BOX_LABELS:
                negatives.add(row[i_id])
    return balloon_boxes, rejected, negatives


def scan_image_labels(imagelabels_csv: Path) -> "tuple[set, set, set]":
    """Return (images verified to show excluded balloon types,
    images with a positive Balloon label, images verified to show a drone-like object)."""
    excluded, balloon_positive, drone_like = set(), set(), set()
    with open(imagelabels_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        i_id, i_label, i_conf = col["ImageID"], col["LabelName"], col["Confidence"]
        for row in reader:
            if row[i_conf] not in ("1", "1.0"):
                continue
            label = row[i_label]
            if label in EXCLUDE_IMAGE_LABELS:
                excluded.add(row[i_id])
            elif label == BALLOON:
                balloon_positive.add(row[i_id])
            elif label in DRONE_LIKE_IMAGE_LABELS:
                drone_like.add(row[i_id])
    return excluded, balloon_positive, drone_like


def scan_rotation(rotation_csv: Path, wanted: set) -> "dict[str, dict]":
    """Return image_id -> metadata row for wanted images (rotation + attribution)."""
    out = {}
    with open(rotation_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ImageID"] in wanted:
                out[row["ImageID"]] = row
    return out


def is_unrotated(meta_row: dict) -> bool:
    value = (meta_row.get("Rotation") or "").strip()
    return value in ("", "0", "0.0")


def to_yolo_lines(boxes, class_id: int = 0) -> "list[str]":
    lines = []
    for xmin, xmax, ymin, ymax in boxes:
        xmin, xmax = max(0.0, min(1.0, xmin)), max(0.0, min(1.0, xmax))
        ymin, ymax = max(0.0, min(1.0, ymin)), max(0.0, min(1.0, ymax))
        w, h = xmax - xmin, ymax - ymin
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{class_id} {(xmin + xmax) / 2:.6f} {(ymin + ymax) / 2:.6f} {w:.6f} {h:.6f}")
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
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-boxes", type=int, default=20)
    parser.add_argument("--negative-ratio", type=float, default=1.5,
                        help="background images (birds/kites/aircraft/helicopters) per balloon image, per split")
    parser.add_argument("--train-to-valid-fraction", type=float, default=0.08)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="only print counts, download nothing")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    attribution_rows = []
    jobs = []  # (url, image_path, label_path, label_lines)

    for our_split, (oi_split, bbox_name, labels_name, rot_name) in SPLITS.items():
        print(f"[{our_split}] scanning boxes ...", flush=True)
        balloon_boxes, rejected, negative_candidates = scan_boxes(args.meta_dir / bbox_name)
        print(f"[{our_split}] scanning image-level labels ...", flush=True)
        excluded, balloon_positive, drone_like = scan_image_labels(args.meta_dir / labels_name)

        stats = {"balloon_candidates": len(balloon_boxes), "group_or_depiction": 0,
                 "hot_air_or_cluster": 0, "contains_drone": 0, "too_many_boxes": 0, "rotated": 0}
        kept = {}
        for image_id, boxes in balloon_boxes.items():
            if image_id in rejected:
                stats["group_or_depiction"] += 1
            elif image_id in excluded:
                stats["hot_air_or_cluster"] += 1
            elif image_id in drone_like:
                stats["contains_drone"] += 1
            elif len(boxes) > args.max_boxes:
                stats["too_many_boxes"] += 1
            else:
                kept[image_id] = boxes

        negatives = sorted(negative_candidates - set(balloon_boxes) - excluded - balloon_positive - drone_like)
        rng.shuffle(negatives)
        # oversample candidates a bit to survive the rotation filter
        negatives = negatives[: int(len(kept) * args.negative_ratio * 1.3) + 10]

        print(f"[{our_split}] reading rotation/attribution metadata ...", flush=True)
        meta = scan_rotation(args.meta_dir / rot_name, set(kept) | set(negatives))
        for image_id in list(kept):
            if image_id not in meta or not is_unrotated(meta[image_id]):
                stats["rotated"] += 1
                del kept[image_id]
        negatives = [i for i in negatives if i in meta and is_unrotated(meta[i])]
        negatives = negatives[: int(len(kept) * args.negative_ratio)]

        n_boxes = sum(len(b) for b in kept.values())
        print(f"[{our_split}] {stats} -> kept {len(kept)} balloon images ({n_boxes} boxes), "
              f"{len(negatives)} background images", flush=True)

        # Open Images' own validation/test splits only hold a few dozen
        # usable balloon images, too few to pick checkpoints with, so move a
        # fixed fraction of the train balloons (and matching negatives) into
        # our valid split. Images still never appear in two splits.
        destinations = {image_id: our_split for image_id in list(kept) + negatives}
        if our_split == "train" and args.train_to_valid_fraction > 0:
            for pool in (sorted(kept), sorted(negatives)):
                rng.shuffle(pool)
                for image_id in pool[: int(len(pool) * args.train_to_valid_fraction)]:
                    destinations[image_id] = "valid"

        for image_id in list(kept) + negatives:
            dest = destinations[image_id]
            img_dir, lbl_dir = args.out / dest / "images", args.out / dest / "labels"
            if not args.dry_run:
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
            is_negative = image_id not in kept
            stem = f"oineg_{image_id}" if is_negative else f"oi_{image_id}"
            lines = [] if is_negative else to_yolo_lines(kept[image_id])
            jobs.append((S3_URL.format(split=oi_split, image_id=image_id),
                         img_dir / f"{stem}.jpg", lbl_dir / f"{stem}.txt", lines))
            m = meta[image_id]
            attribution_rows.append({"split": dest, "ImageID": image_id, "License": m.get("License", ""),
                                     "Author": m.get("Author", ""), "OriginalURL": m.get("OriginalURL", "")})
        moved = sum(1 for d in destinations.values() if d != our_split)
        if moved:
            print(f"[{our_split}] moved {moved} images to valid", flush=True)

    if args.dry_run:
        print(f"dry run: would download {len(jobs)} images")
        return

    print(f"downloading {len(jobs)} images with {args.workers} workers ...", flush=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_and_resize, url, img, args.max_side): (img, lbl, lines)
                   for url, img, lbl, lines in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            img, lbl, lines = futures[fut]
            if fut.result():
                lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
            else:
                failed += 1
                img.unlink(missing_ok=True)
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)} done, {failed} failed", flush=True)

    with open(args.out / "attribution.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "ImageID", "License", "Author", "OriginalURL"])
        writer.writeheader()
        writer.writerows(attribution_rows)
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "names:\n  0: balloon\n"
    )
    print(f"DONE: {len(jobs) - failed} images written, {failed} failed downloads (dropped).")


if __name__ == "__main__":
    sys.exit(main())
