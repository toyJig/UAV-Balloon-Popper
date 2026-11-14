#!/usr/bin/env python3
"""Find near-duplicate images across the splits of a merged YOLO dataset
and remove them from valid/test, so evaluation scores aren't inflated by
images the model effectively trained on.

Why this is needed here: Open Images and COCO (which LVIS annotates) were
both sourced from Flickr, so the same photo can arrive from both sources
and land in train from one and in valid/test from the other.

Method: 256-bit difference hash (dHash) of each image, Hamming distance
between every valid/test image and every train image. Train images that
are intentional oversampling copies (names ending in _r<N>) are skipped.

* --report-only prints the distance histogram and writes side-by-side
  pictures of the closest pairs to --pairs-dir, for calibrating
  --max-distance by eye before deleting anything.
* Otherwise: any valid/test image within --max-distance of a train image
  is deleted (image + label); then any test image within --max-distance
  of a valid image is deleted too.

Usage:
    python tools/dedup_splits.py --dataset datasets/combined --prefix balloon- --report-only --pairs-dir runs/dedup_pairs
    python tools/dedup_splits.py --dataset datasets/combined --prefix balloon- --max-distance 24
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPEAT_RE = re.compile(r"_r\d+$")
HASH_SIDE = 16  # 16x16 = 256-bit hash


def dhash(path: Path) -> "np.ndarray | None":
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (HASH_SIDE + 1, HASH_SIDE), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).view(np.uint64)  # 4 x uint64


def list_images(dataset: Path, split: str, prefix: str) -> "list[Path]":
    d = dataset / split / "images"
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in IMAGE_SUFFIXES and p.name.startswith(prefix) and not REPEAT_RE.search(p.stem))


def hash_all(paths: "list[Path]", workers: int = 16) -> "tuple[list[Path], np.ndarray]":
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(dhash, paths))
    kept = [(p, h) for p, h in zip(paths, hashes) if h is not None]
    if not kept:
        return [], np.zeros((0, HASH_SIDE * HASH_SIDE // 64), dtype=np.uint64)
    return [p for p, _ in kept], np.stack([h for _, h in kept])


def nearest(query: np.ndarray, ref: np.ndarray, chunk: int = 256) -> "tuple[np.ndarray, np.ndarray]":
    """For each query hash, return (min Hamming distance to ref, index of that ref)."""
    if len(ref) == 0 or len(query) == 0:
        return np.full(len(query), 10**9), np.zeros(len(query), dtype=int)
    dists, idxs = [], []
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        d = np.bitwise_count(q[:, None, :] ^ ref[None, :, :]).sum(axis=2)
        idxs.append(d.argmin(axis=1))
        dists.append(d.min(axis=1))
    return np.concatenate(dists), np.concatenate(idxs)


def save_pair(a: Path, b: Path, dist: int, out: Path) -> None:
    ia, ib = cv2.imread(str(a)), cv2.imread(str(b))
    h = 320
    ia = cv2.resize(ia, (int(ia.shape[1] * h / ia.shape[0]), h))
    ib = cv2.resize(ib, (int(ib.shape[1] * h / ib.shape[0]), h))
    canvas = np.hstack([ia, np.full((h, 10, 3), 255, np.uint8), ib])
    cv2.putText(canvas, f"d={dist}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(str(out / f"d{dist:03d}_{a.stem}__{b.stem}.jpg"), canvas)


def delete_item(img: Path) -> None:
    label = img.parent.parent / "labels" / (img.stem + ".txt")
    img.unlink(missing_ok=True)
    label.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prefix", default="", help="only consider images whose filename starts with this")
    parser.add_argument("--max-distance", type=int, default=24)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--pairs-dir", type=Path, default=None)
    parser.add_argument("--pairs-to-save", type=int, default=40)
    args = parser.parse_args()

    train_paths, train_h = hash_all(list_images(args.dataset, "train", args.prefix))
    valid_paths, valid_h = hash_all(list_images(args.dataset, "valid", args.prefix))
    test_paths, test_h = hash_all(list_images(args.dataset, "test", args.prefix))
    print(f"hashed train={len(train_paths)} valid={len(valid_paths)} test={len(test_paths)}")

    # within-train duplicates (reported only: harmless for evaluation)
    if len(train_h) > 1:
        d_tt, _ = nearest(train_h, train_h)  # includes self-match at 0, so recompute excluding self
        self_free = []
        for start in range(0, len(train_h), 256):
            q = train_h[start:start + 256]
            d = np.bitwise_count(q[:, None, :] ^ train_h[None, :, :]).sum(axis=2)
            for k in range(len(q)):
                d[k, start + k] = 10**9
            self_free.append(d.min(axis=1))
        d_tt = np.concatenate(self_free)
        print(f"train images with a near-duplicate inside train (<= {args.max_distance}): {(d_tt <= args.max_distance).sum()}")

    results = []
    for split, paths, hashes in (("valid", valid_paths, valid_h), ("test", test_paths, test_h)):
        d, idx = nearest(hashes, train_h)
        results.append((split, paths, d, idx, train_paths))
        hist = np.histogram(d, bins=[0, 5, 10, 15, 20, 25, 30, 40, 60, 257])[0]
        print(f"{split} vs train: nearest-distance histogram [0,5,10,15,20,25,30,40,60,256] -> {hist.tolist()}")
    d_vt, idx_vt = nearest(test_h, valid_h)
    results.append(("test-vs-valid", test_paths, d_vt, idx_vt, valid_paths))

    if args.pairs_dir is not None:
        args.pairs_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        for split, paths, d, idx, ref_paths in results:
            for i in np.argsort(d)[: args.pairs_to_save]:
                candidates.append((int(d[i]), paths[i], ref_paths[idx[i]]))
        candidates.sort(key=lambda c: c[0])
        for dist, a, b in candidates[: args.pairs_to_save]:
            save_pair(a, b, dist, args.pairs_dir)
        print(f"wrote {min(len(candidates), args.pairs_to_save)} closest pairs to {args.pairs_dir}")

    for split, paths, d, idx, ref_paths in results:
        dupes = [paths[i] for i in np.nonzero(d <= args.max_distance)[0]]
        print(f"{split}: {len(dupes)} images within distance {args.max_distance}")
        if not args.report_only:
            for p in dupes:
                delete_item(p)
    if not args.report_only:
        print("duplicates removed from valid/test")


if __name__ == "__main__":
    main()
