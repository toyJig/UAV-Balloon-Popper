#!/usr/bin/env python3
"""Break a trained detector's errors down by image source, so a weak
aggregate score can be traced to where it comes from (e.g. false balloon
detections on drone photos vs. missed balloons on balloon photos).

Matches predictions to ground truth per image and class at IoU >= 0.5
(greedy, highest confidence first), at a fixed confidence threshold - by
default 0.40, the same threshold MLDetector uses on the Pi.

Source is taken from the merged filename prefix written by dataset_prep.py,
e.g. "balloon-openimages_balloon_oi_..." -> "openimages/balloon",
"balloon-openimages_balloon_oineg_..." -> "openimages/background",
"drone-seraphim_subset_..." -> "seraphim/drone".

Optionally writes contact sheets of the worst images for one class.

Usage:
    python tools/error_breakdown.py --weights runs/lockon/balloon_drone_v1/weights/best.pt \\
        --split-dir datasets/combined/test --sheet-class balloon --sheet-dir runs/sheets/errors_test
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

NAMES = {0: "balloon", 1: "drone"}


def source_of(stem: str) -> str:
    if stem.startswith("drone-"):
        return "seraphim/drone"
    if "_oineg_" in stem:
        return "openimages/background"
    if "_oi_" in stem:
        return "openimages/balloon"
    if "_lvis_" in stem:
        return "lvis/balloon"
    return "other"


def load_gt(label_path: Path, w: int, h: int):
    out = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, iou_thr=0.5):
    """preds: list of (cls, conf, box); gts: list of (cls, box). Returns
    (tp per class, fp list of (cls, conf, box), fn list of (cls, box))."""
    tp = defaultdict(int)
    fps, used = [], set()
    for c, conf, box in sorted(preds, key=lambda p: -p[1]):
        best, best_j = 0.0, None
        for j, (gc, gbox) in enumerate(gts):
            if gc != c or j in used:
                continue
            s = iou(box, gbox)
            if s > best:
                best, best_j = s, j
        if best >= iou_thr:
            used.add(best_j)
            tp[c] += 1
        else:
            fps.append((c, conf, box))
    fns = [g for j, g in enumerate(gts) if j not in used]
    return tp, fps, fns


def draw(img, gts, fps, fns, tile=300):
    for c, box in gts:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 200, 0), 2)
    for c, box in fns:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 3)
    for c, conf, box in fps:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.putText(img, f"{NAMES[c]} {conf:.2f}", (int(box[0]), max(12, int(box[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    h, w = img.shape[:2]
    s = tile / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((tile, tile, 3), 40, np.uint8)
    canvas[: img.shape[0], : img.shape[1]] = img
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sheet-class", choices=list(NAMES.values()), default=None)
    parser.add_argument("--sheet-dir", type=Path, default=None)
    parser.add_argument("--sheet-count", type=int, default=36)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    images = sorted(p for p in (args.split_dir / "images").iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})

    stats = defaultdict(lambda: defaultdict(int))  # source -> counter
    per_image = []
    for start in range(0, len(images), 64):
        batch = images[start:start + 64]
        results = model.predict([str(p) for p in batch], imgsz=args.imgsz, conf=args.conf, verbose=False, stream=False)
        for img_path, r in zip(batch, results):
            h, w = r.orig_shape
            gts_raw = load_gt(args.split_dir / "labels" / (img_path.stem + ".txt"), w, h)
            gts = [(c, (x1, y1, x2, y2)) for c, x1, y1, x2, y2 in gts_raw]
            preds = [(int(c), float(s), tuple(map(float, b)))
                     for c, s, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
            tp, fps, fns = match(preds, gts)
            src = source_of(img_path.stem)
            s = stats[src]
            s["images"] += 1
            for c in NAMES:
                name = NAMES[c]
                s[f"{name}_gt"] += sum(1 for g in gts if g[0] == c)
                s[f"{name}_tp"] += tp[c]
                s[f"{name}_fp"] += sum(1 for f in fps if f[0] == c)
                s[f"{name}_fn"] += sum(1 for f in fns if f[0] == c)
            per_image.append((img_path, gts, fps, fns))

    print(f"conf >= {args.conf}, IoU >= 0.5, split {args.split_dir}")
    print(f"{'source':24s} {'imgs':>6s} | {'balloon gt/tp/fp/fn':>22s} | {'drone gt/tp/fp/fn':>22s}")
    totals = defaultdict(int)
    for src in sorted(stats):
        s = stats[src]
        for k, v in s.items():
            totals[k] += v
        print(f"{src:24s} {s['images']:6d} | "
              f"{s['balloon_gt']:5d}/{s['balloon_tp']:5d}/{s['balloon_fp']:5d}/{s['balloon_fn']:5d} | "
              f"{s['drone_gt']:5d}/{s['drone_tp']:5d}/{s['drone_fp']:5d}/{s['drone_fn']:5d}")
    for name in NAMES.values():
        tp, fp, fn = totals[f"{name}_tp"], totals[f"{name}_fp"], totals[f"{name}_fn"]
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        print(f"{name}: precision={p:.3f} recall={r:.3f} (tp={tp} fp={fp} fn={fn})")

    if args.sheet_class and args.sheet_dir:
        cid = next(k for k, v in NAMES.items() if v == args.sheet_class)
        args.sheet_dir.mkdir(parents=True, exist_ok=True)
        for kind in ("fp", "fn"):
            scored = []
            for img_path, gts, fps, fns in per_image:
                n = sum(1 for f in fps if f[0] == cid) if kind == "fp" else sum(1 for f in fns if f[0] == cid)
                if n:
                    scored.append((n, img_path, gts, fps, fns))
            scored.sort(key=lambda t: -t[0])
            tiles = [draw(cv2.imread(str(p)), g, f, n_) for _, p, g, f, n_ in scored[: args.sheet_count]]
            if not tiles:
                continue
            while len(tiles) % 6:
                tiles.append(np.full((300, 300, 3), 40, np.uint8))
            sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
            out = args.sheet_dir / f"{args.sheet_class}_{kind}.jpg"
            cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
            print(f"wrote {out} ({len(scored)} images with {args.sheet_class} {kind}; green=GT, red=false positive, magenta=missed)")


if __name__ == "__main__":
    main()
