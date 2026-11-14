#!/usr/bin/env python3
"""Task 8 - export trained YOLO weights (best.pt) to NCNN for CPU-only
inference on the Pi 5, and package each export as a clean folder
(model.ncnn.param, model.ncnn.bin, metadata.yaml) ready to copy over.

NCNN exports only accept the exact input size they were exported at, so
export one folder per size you want to try. Sizes are WIDTHxHEIGHT;
match the camera's aspect ratio (640x480 for the default 640x480 capture)
so no pixels are wasted on letterbox padding.

Run where ultralytics is installed (the laptop). Afterwards, check the
export against PyTorch with tools/check_ncnn_parity.py.

Usage:
    python tools/export_ncnn.py --weights runs/lockon/balloon_drone_v1/weights/best.pt \\
        --out models/balloon_drone_v1 --sizes 640x480 416x320
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def _write_metadata(path: Path, weights: Path, w: int, h: int, names) -> None:
    """Minimal Ultralytics-compatible metadata.yaml (the Pi backend only
    needs imgsz and names). ``names`` is a {index: name} dict."""
    lines = [
        f"description: YOLO11n NCNN export of {weights.name}",
        "author: Ultralytics",
        "task: detect",
        "stride: 32",
        "batch: 1",
        "channels: 3",
        "imgsz:",
        f"- {h}",
        f"- {w}",
        "names:",
    ]
    lines += [f"  {int(k)}: {v}" for k, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    path.write_text("\n".join(lines) + "\n")


def parse_size(text: str) -> "tuple[int, int]":
    if "x" in text.lower():
        w, h = (int(v) for v in text.lower().split("x"))
    else:
        w = h = int(text)
    if w % 32 or h % 32:
        raise argparse.ArgumentTypeError(f"{text}: width and height must be multiples of 32 (YOLO stride)")
    return w, h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=Path, help="path to trained best.pt")
    parser.add_argument("--out", required=True, type=Path, help="package folder, e.g. models/balloon_drone_v1")
    parser.add_argument("--sizes", nargs="+", type=parse_size, default=[(640, 480)])
    args = parser.parse_args()

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.weights, args.out / "best.pt")
    for w, h in args.sizes:
        # export into a scratch copy so nothing is written next to the training run
        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / "model.pt"
            shutil.copy2(args.weights, pt)
            exported = Path(YOLO(str(pt)).export(format="ncnn", imgsz=[h, w]))
            dst = args.out / f"ncnn_{w}x{h}"
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("model.ncnn.param", "model.ncnn.bin"):
                shutil.copy2(exported / name, dst / name)
            # the Pi backend reads imgsz and class names from metadata.yaml;
            # Ultralytics sometimes leaves the copied one empty, so write a
            # known-good one from the model's own names if needed
            src_meta = exported / "metadata.yaml"
            if src_meta.exists() and src_meta.stat().st_size > 0:
                shutil.copy2(src_meta, dst / "metadata.yaml")
            else:
                _write_metadata(dst / "metadata.yaml", args.weights, w, h, YOLO(str(pt)).names)
        print(f"packaged {dst}")
    print("Copy the package folder to the Pi and pass one ncnn_<WxH> folder as --model-path.")


if __name__ == "__main__":
    main()
