#!/usr/bin/env python3
"""Pick a confidence threshold for lock-on: for each candidate threshold,
report how often the detector fires on things that are NOT targets
(birds, kites, aircraft, helicopters, empty sky) versus how many real
targets it finds.

Runs inference once at a low confidence, then re-scores at every
threshold. Reports, per threshold:
  * false-alarm rate: fraction of no-target images (Open Images
    background set: birds/kites/aircraft/helicopters) with ANY detection,
  * balloon precision/recall on sparse balloon images (1..3 balloons),
  * drone precision/recall on drone images.

Usage:
    python tools/threshold_sweep.py --weights models/balloon_drone_v1/best.pt \\
        --split-dirs datasets/combined/valid datasets/combined/test --imgsz 640
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_breakdown import load_gt, match  # noqa: E402

THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sparse-max", type=int, default=3)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    items = []  # (kind, preds, gts)
    for d in args.split_dirs:
        paths = sorted(p for p in (d / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for start in range(0, len(paths), 64):
            batch = paths[start:start + 64]
            for p, r in zip(batch, model.predict([str(x) for x in batch], imgsz=args.imgsz, conf=min(THRESHOLDS), verbose=False)):
                h, w = r.orig_shape
                gts = [(c, (x1, y1, x2, y2)) for c, x1, y1, x2, y2 in load_gt(d / "labels" / (p.stem + ".txt"), w, h)]
                preds = [(int(c), float(s), tuple(map(float, b)))
                         for c, s, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
                if "_oineg_" in p.stem:
                    kind = "background"
                elif p.stem.startswith("drone-"):
                    kind = "drone"
                else:
                    n = sum(1 for g in gts if g[0] == 0)
                    kind = "sparse_balloon" if 1 <= n <= args.sparse_max else "dense_balloon"
                items.append((kind, preds, gts))

    n_bg = sum(1 for k, _, _ in items if k == "background")
    n_sb = sum(1 for k, _, _ in items if k == "sparse_balloon")
    n_dr = sum(1 for k, _, _ in items if k == "drone")
    print(f"images: background={n_bg} sparse_balloon={n_sb} drone={n_dr}")
    print(f"{'conf':>5s} | {'false-alarm imgs':>16s} | {'balloon P/R (sparse)':>21s} | {'drone P/R':>13s}")
    for t in THRESHOLDS:
        fa = 0
        c = defaultdict(int)
        for kind, preds, gts in items:
            kept = [p for p in preds if p[1] >= t]
            if kind == "background":
                fa += bool(kept)
                continue
            if kind == "dense_balloon":
                continue
            tp, fps, fns = match(kept, gts)
            cls = 0 if kind == "sparse_balloon" else 1
            c["tp"if cls == 0 else "dtp"] += tp[cls]
            c["fp"if cls == 0 else "dfp"] += sum(1 for f in fps if f[0] == cls)
            c["fn"if cls == 0 else "dfn"] += sum(1 for f in fns if f[0] == cls)

        def pr(tp, fp, fn):
            return f"{tp / (tp + fp) if tp + fp else float('nan'):.3f}/{tp / (tp + fn) if tp + fn else float('nan'):.3f}"

        print(f"{t:5.2f} | {fa:5d} ({100.0 * fa / max(1, n_bg):5.1f}%) | {pr(c['tp'], c['fp'], c['fn']):>21s} | "
              f"{pr(c['dtp'], c['dfp'], c['dfn']):>13s}")


if __name__ == "__main__":
    main()
