#!/usr/bin/env python3
"""Turn the downloaded Seraphim drone dataset (Hugging Face
lgrzybowski/seraphim-drone-detection-dataset, CC BY 4.0) into a standard
YOLO dataset with train/valid/test splits and a data.yaml, ready for
tools/dataset_prep.py to merge with the balloon set.

* Extracts the batch_*.zip archives once (skipped if already extracted).
* train: a random subsample of Seraphim's train split (--train-count), so
  the ~75k drone images don't swamp the much smaller balloon set and the
  epoch time stays reasonable.
* valid/test: Seraphim's own test split, divided in half. Seraphim's
  train/test boundary is kept intact, so none of our validation or test
  images come from their training pool.
* Files are hardlinked, not copied (no extra disk space).

Usage:
    python tools/prepare_seraphim.py --raw raw_datasets/seraphim --out datasets/seraphim_subset --train-count 24000
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import zipfile
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_split(raw: Path, split: str, extracted: Path) -> None:
    for kind in ("images", "labels"):
        dst = extracted / split / kind
        marker = dst / ".extracted"
        if marker.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for zip_path in sorted((raw / split / kind).glob("batch_*.zip")):
            print(f"extracting {zip_path} ...", flush=True)
            with zipfile.ZipFile(zip_path) as z:
                for member in z.infolist():
                    if member.is_dir():
                        continue
                    # flatten any internal folders: we only want the files
                    target = dst / Path(member.filename).name
                    with z.open(member) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
        marker.write_text("ok")


def paired_stems(split_dir: Path) -> "list[tuple[Path, Path]]":
    labels = {p.stem: p for p in (split_dir / "labels").glob("*.txt")}
    pairs = []
    for img in (split_dir / "images").iterdir():
        if img.suffix.lower() in IMAGE_SUFFIXES and img.stem in labels:
            pairs.append((img, labels[img.stem]))
    pairs.sort(key=lambda p: p[0].name)
    return pairs


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split(pairs, out_split: Path) -> None:
    (out_split / "images").mkdir(parents=True, exist_ok=True)
    (out_split / "labels").mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        link(img, out_split / "images" / img.name)
        link(lbl, out_split / "labels" / lbl.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="snapshot_download folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    extracted = args.raw.parent / (args.raw.name + "_extracted")
    for split in ("train", "test"):
        extract_split(args.raw, split, extracted)

    rng = random.Random(args.seed)
    train_pairs = paired_stems(extracted / "train")
    test_pairs = paired_stems(extracted / "test")
    print(f"paired: train={len(train_pairs)} test={len(test_pairs)}")

    rng.shuffle(train_pairs)
    train_pairs = train_pairs[: args.train_count]
    rng.shuffle(test_pairs)
    half = len(test_pairs) // 2
    valid_pairs, test_pairs = test_pairs[:half], test_pairs[half:]

    write_split(train_pairs, args.out / "train")
    write_split(valid_pairs, args.out / "valid")
    write_split(test_pairs, args.out / "test")
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "names:\n  0: drone\n"
    )
    print(f"DONE: train={len(train_pairs)} valid={len(valid_pairs)} test={len(test_pairs)} -> {args.out}")


if __name__ == "__main__":
    main()
