#!/usr/bin/env python3
"""Move a hand-picked list of images (and their labels) out of their source
dataset splits into a separate test set, so they can never be trained on.

The list holds image paths like datasets/openimages_balloon/train/images/oi_x.jpg
(one per line, as written by tools/review_sheets.py --pick).

Usage:
    python tools/move_to_test_set.py --list runs/sky_picked.txt --out datasets/sky_test_real --name balloon
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="balloon", help="class 0 name for data.yaml")
    args = parser.parse_args()

    img_out, lbl_out = args.out / "test" / "images", args.out / "test" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    moved = missing = 0
    with open(args.out / "sources.txt", "a") as log:
        for line in args.list.read_text().split():
            img = Path(line)
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            if not img.exists():
                if (img_out / img.name).exists():
                    continue  # already moved by an earlier run
                missing += 1
                print(f"missing: {img}")
                continue
            shutil.move(str(img), img_out / img.name)
            if lbl.exists():
                shutil.move(str(lbl), lbl_out / lbl.name)
            else:
                (lbl_out / lbl.name).write_text("")
            log.write(f"{img}\n")
            moved += 1
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: test/images\nval: test/images\ntest: test/images\n"
        f"names:\n  0: {args.name}\n")
    print(f"moved {moved} images to {args.out} ({missing} missing)")


if __name__ == "__main__":
    main()
