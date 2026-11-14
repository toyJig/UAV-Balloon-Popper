#!/usr/bin/env python3
"""Task 8 - merge a public drone-detection dataset (YOLO format, e.g.
downloaded from Roboflow Universe) with your own labeled balloon photos
into one combined YOLO-format dataset ready for training.

Expected input layout for each source dataset (standard Roboflow YOLO export):
    <dataset>/
      train/images/*.jpg  train/labels/*.txt
      valid/images/*.jpg  valid/labels/*.txt
      test/images/*.jpg   test/labels/*.txt   (optional)
      data.yaml

Class ids are remapped to a single combined class list
(default: 0=balloon, 1=drone) - pass --drone-class-name / --balloon-class-name
if either source dataset uses different label text for the classes you
want merged in.

Both --drone-dataset and --balloon-dataset accept several folders. Output
files are prefixed with "<class>-<source folder name>_" so every image's
origin stays traceable.

Usage:
    python tools/dataset_prep.py \\
        --drone-dataset ./datasets/seraphim_subset \\
        --balloon-dataset ./datasets/openimages_balloon ./datasets/lvis_balloon \\
        --balloon-train-repeat 2 \\
        --out ./datasets/combined
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

COMBINED_CLASS_NAMES = ["balloon", "drone"]
SPLITS = ("train", "valid", "test")


def load_class_names(dataset_dir: Path) -> "list[str]":
    with open(dataset_dir / "data.yaml") as f:
        data = yaml.safe_load(f)
    return names_to_list(data["names"])


def names_to_list(names) -> "list[str]":
    """data.yaml `names` may be a list (['drone']) or an id->name mapping
    ({0: 'drone'}); both are valid Ultralytics syntax."""
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def normalize_box(values: "list[float]") -> "tuple[float, float, float, float] | None":
    """Return a clipped (cx, cy, w, h) box from either a YOLO box (4 values)
    or a YOLO polygon (x1 y1 x2 y2 ..., even count >= 6), or None if the
    result is degenerate. Some source datasets mix polygon rows into
    otherwise box-format label files."""
    if len(values) == 4:
        cx, cy, w, h = values
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, w, h)


def remap_label_file(src_path: Path, dst_path: Path, class_id_map: dict) -> int:
    """Write a remapped copy of a YOLO label file. Returns how many rows were
    dropped as malformed/degenerate (rows of unwanted classes are dropped
    silently and not counted)."""
    lines_out = []
    malformed = 0
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_id = int(float(parts[0]))
            if old_id not in class_id_map:
                continue  # class not wanted in the combined set - drop this box
            try:
                box = normalize_box([float(v) for v in parts[1:]])
            except ValueError:
                box = None
            if box is None:
                malformed += 1
                continue
            new_id = class_id_map[old_id]
            lines_out.append(f"{new_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
    dst_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return malformed


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink: no extra disk space, near-instant
    except OSError:
        shutil.copy2(src, dst)  # different drive / filesystem without hardlinks


def merge_source(dataset_dir: Path, class_id_map: dict, out_dir: Path, prefix: str, train_repeat: int = 1) -> None:
    """Merge one source dataset into out_dir. ``train_repeat`` > 1 writes
    each train image that many times (as hardlinks with distinct names) to
    oversample a minority class; valid/test are never repeated."""
    malformed = 0
    counts = {}
    for split in SPLITS:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.exists():
            continue
        out_images = out_dir / split / "images"
        out_labels = out_dir / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        repeats = train_repeat if split == "train" else 1

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            has_objects = label_path.exists() and label_path.read_text().strip() != ""
            # only oversample images that contain objects: repeating background
            # images would push their share well past the ~10% YOLO likes
            for r in range(repeats if has_objects else 1):
                suffix = "" if r == 0 else f"_r{r}"
                new_name = f"{prefix}_{img_path.stem}{suffix}{img_path.suffix}"
                _link_or_copy(img_path, out_images / new_name)
                new_label_path = out_labels / (Path(new_name).stem + ".txt")
                if label_path.exists():
                    dropped = remap_label_file(label_path, new_label_path, class_id_map)
                    if r == 0:
                        malformed += dropped
                else:
                    new_label_path.write_text("")  # background image, no objects
                counts[split] = counts.get(split, 0) + 1
    print(f"merged {prefix}: images per split {counts}, malformed label rows dropped: {malformed}")


def build_class_id_map(source_names: "list[str]", wanted_name: str, combined_id: int) -> dict:
    mapping = {}
    for i, name in enumerate(source_names):
        if name.lower() == wanted_name.lower():
            mapping[i] = combined_id
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--balloon-dataset", type=Path, nargs="+", required=True)
    # (see also --mixed-dataset below)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drone-class-name", default="drone")
    parser.add_argument("--balloon-class-name", default="balloon")
    parser.add_argument("--balloon-train-repeat", type=int, default=1,
                        help="oversample balloon train images this many times (hardlinked duplicates)")
    parser.add_argument("--mixed-dataset", type=Path, nargs="*", default=[],
                        help="sources that already contain both classes; mapped by class name")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; delete it first so stale files from an earlier merge can't leak in")

    sources = [(d, "drone", args.drone_class_name, 1) for d in args.drone_dataset]
    sources += [(d, "balloon", args.balloon_class_name, args.balloon_train_repeat) for d in args.balloon_dataset]
    for dataset_dir, kind, wanted_name, repeat in sources:
        names = load_class_names(dataset_dir)
        class_map = build_class_id_map(names, wanted_name, COMBINED_CLASS_NAMES.index(kind))
        if not class_map:
            raise SystemExit(f"No class named '{wanted_name}' found in {dataset_dir}/data.yaml ({names})")
        merge_source(dataset_dir, class_map, args.out, prefix=f"{kind}-{dataset_dir.name}", train_repeat=repeat)
    for dataset_dir in args.mixed_dataset:
        names = load_class_names(dataset_dir)
        class_map = {}
        for wanted in COMBINED_CLASS_NAMES:
            class_map.update(build_class_id_map(names, wanted, COMBINED_CLASS_NAMES.index(wanted)))
        if not class_map:
            raise SystemExit(f"No known class names in {dataset_dir}/data.yaml ({names})")
        merge_source(dataset_dir, class_map, args.out, prefix=f"mixed-{dataset_dir.name}")

    data_yaml = {
        "path": str(args.out.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(COMBINED_CLASS_NAMES)},
    }
    with open(args.out / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"Combined dataset written to {args.out} with classes {COMBINED_CLASS_NAMES}")


if __name__ == "__main__":
    main()
