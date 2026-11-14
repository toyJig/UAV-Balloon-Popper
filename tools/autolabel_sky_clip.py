#!/usr/bin/env python3
"""Auto-label frames of a balloon filmed against the sky, by colour.

Works when the balloon is the only strongly coloured object in the sky
(true for the Balloons_vs_Sky clip, and usually for your own test footage
of a single-colour balloon). For each frame:

  1. HSV threshold for saturated balloon colours (default: red/orange,
     both ends of the hue circle),
  2. connected components, ignoring blobs smaller than --min-area,
  3. keep components inside the "sky" part of the frame: the area above
     the highest row where non-sky (low-saturation-blue-failing) pixels
     dominate. Blobs in buildings/trees below the skyline are dropped,
  4. merge kept blobs that are close together (a balloon bunch) into one
     box, and write the largest group as a YOLO label (class 0).

Frames where nothing is found get an empty label file and are listed, so
you can check them. ALWAYS review the output with
tools/label_contact_sheet.py before using it for evaluation.

Usage:
    python tools/autolabel_sky_clip.py --frames raw_datasets/balloons_vs_sky/frames \\
        --out datasets/sky_test_clip --every 2
    python tools/autolabel_sky_clip.py --video my_balloon_pass.mp4 --out datasets/my_clip_01 --every 3
    (for a blue balloon, e.g.: --hue 95-130)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def balloon_mask(bgr: np.ndarray, hue_ranges, min_sat: int, min_val: int) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in hue_ranges:
        mask |= cv2.inRange(hsv, (lo, min_sat, min_val), (hi, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def skyline_row(bgr: np.ndarray) -> int:
    """First row (from the top) where most pixels stop looking like sky
    (blue-dominant or bright-grey, low texture)."""
    small = cv2.resize(bgr, (160, int(160 * bgr.shape[0] / bgr.shape[1])))
    b, g, r = [small[..., i].astype(np.int16) for i in range(3)]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    sky = ((b > r + 15) | ((abs(b - r) < 20) & (gray > 170))) & (grad < 40)
    frac = sky.mean(axis=1)
    rows = np.nonzero(frac < 0.6)[0]
    row = int(rows[0]) if len(rows) else small.shape[0]
    return int(row * bgr.shape[0] / small.shape[0])


def find_balloon(bgr, hue_ranges, min_sat, min_val, min_area, merge_px):
    mask = balloon_mask(bgr, hue_ranges, min_sat, min_val)
    limit = skyline_row(bgr)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area and y + h <= limit + 5:
            boxes.append([x, y, x + w, y + h, area])
    if not boxes:
        return None
    # merge boxes closer than merge_px (balloon bunches), keep the biggest group
    groups = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        for g in groups:
            if b[0] - merge_px < g[2] and b[2] + merge_px > g[0] and b[1] - merge_px < g[3] and b[3] + merge_px > g[1]:
                g[0], g[1], g[2], g[3] = min(g[0], b[0]), min(g[1], b[1]), max(g[2], b[2]), max(g[3], b[3])
                g[4] += b[4]
                break
        else:
            groups.append(list(b))
    return max(groups, key=lambda g: g[4])[:4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=None, help="folder of frames")
    parser.add_argument("--video", type=Path, default=None, help="or a video file (frames are extracted)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--hue", nargs="+", default=["0-22", "165-180"], help="OpenCV hue ranges (0-180)")
    parser.add_argument("--min-sat", type=int, default=110)
    parser.add_argument("--min-val", type=int, default=90)
    parser.add_argument("--min-area", type=int, default=40)
    parser.add_argument("--merge-px", type=int, default=25)
    parser.add_argument("--prefix", default="skyclip_")
    args = parser.parse_args()

    hue_ranges = [tuple(int(v) for v in r.split("-")) for r in args.hue]
    img_out, lbl_out = args.out / "images", args.out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    if (args.frames is None) == (args.video is None):
        raise SystemExit("give exactly one of --frames or --video")
    if args.video is not None:
        args.frames = args.out / "_frames"
        args.frames.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(args.video))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(args.frames / f"{args.video.stem}_{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
        print(f"extracted {n} frames from {args.video}")
    frames = sorted(p for p in args.frames.iterdir() if p.suffix.lower() in {".jpg", ".png"})[:: args.every]
    missing = []
    for p in frames:
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        box = find_balloon(bgr, hue_ranges, args.min_sat, args.min_val, args.min_area, args.merge_px)
        stem = f"{args.prefix}{p.stem}"
        shutil.copy2(p, img_out / f"{stem}{p.suffix}")
        if box is None:
            missing.append(p.name)
            (lbl_out / f"{stem}.txt").write_text("")
            continue
        x1, y1, x2, y2 = box
        (lbl_out / f"{stem}.txt").write_text(
            f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
    print(f"labeled {len(frames) - len(missing)}/{len(frames)} frames -> {args.out}")
    if missing:
        print("no balloon found in:", ", ".join(missing[:30]) + (" ..." if len(missing) > 30 else ""))


if __name__ == "__main__":
    main()
