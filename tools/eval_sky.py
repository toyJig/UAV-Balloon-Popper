#!/usr/bin/env python3
"""Evaluate detectors on REAL balloons-in-the-sky images, at the Pi's input
size (640x480 letterbox), and compare models side by side.

Two test sets, neither ever used for training:

* datasets/sky_test_real: 43 hand-picked Open Images / LVIS photos with
  balloons seen against the sky (143 boxes). Scored per box (IoU >= 0.5),
  plus "image hit rate": the share of images where at least one labeled
  balloon is found - what matters for locking on.
* datasets/sky_test_clip: 277 frames (every 2nd) of a phone video of one
  balloon bunch against blue sky (tau-adl/Balloons_vs_Sky test clip), one
  box around the bunch per frame. Scored per frame: hit = a balloon
  detection whose centre lies inside the bunch box (+25% margin); false
  alarm = any detection (either class) centred outside it.

Usage:
    python tools/eval_sky.py --weights models/balloon_drone_v1/best.pt runs/lockon/balloon_drone_v2/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_breakdown import load_gt, match  # noqa: E402

IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


def predict_all(model, paths, imgsz, conf):
    out = []
    for s in range(0, len(paths), 32):
        batch = paths[s:s + 32]
        for p, r in zip(batch, model.predict([str(x) for x in batch], imgsz=imgsz, conf=conf, rect=False, verbose=False)):
            preds = [(int(c), float(sc), tuple(map(float, b)))
                     for c, sc, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
            out.append((p, r.orig_shape, preds))
    return out


def eval_real(results, label_dir, conf):
    tp = fp = fn = hit_images = images = 0
    for p, (h, w), preds in results:
        gts = [(c, (x1, y1, x2, y2)) for c, x1, y1, x2, y2 in load_gt(label_dir / (p.stem + ".txt"), w, h)]
        kept = [x for x in preds if x[1] >= conf and x[0] == 0]
        t, fps, fns = match(kept, gts)
        tp += t[0]
        fp += len(fps)
        fn += len(fns)
        images += 1
        hit_images += t[0] > 0
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return f"P={p:.3f} R={r:.3f} image-hit={hit_images}/{images} ({100 * hit_images / max(1, images):.0f}%)"


def eval_clip(results, label_dir, conf, margin=0.25):
    hits = frames = false_alarms = drone_calls = 0
    for p, (h, w), preds in results:
        gts = load_gt(label_dir / (p.stem + ".txt"), w, h)
        if not gts:
            continue
        frames += 1
        _, x1, y1, x2, y2 = gts[0]
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        inside = lambda b: x1 - mx <= (b[0] + b[2]) / 2 <= x2 + mx and y1 - my <= (b[1] + b[3]) / 2 <= y2 + my
        kept = [x for x in preds if x[1] >= conf]
        hits += any(c == 0 and inside(b) for c, _, b in kept)
        drone_calls += any(c == 1 and inside(b) for c, _, b in kept)
        false_alarms += sum(1 for _, _, b in kept if not inside(b))
    return (f"frame-hit={hits}/{frames} ({100 * hits / max(1, frames):.0f}%) "
            f"called-drone={drone_calls} false-alarms={false_alarms} ({false_alarms / max(1, frames):.2f}/frame)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--real", type=Path, default=Path("datasets/sky_test_real/test"))
    parser.add_argument("--clip", type=Path, default=Path("datasets/sky_test_clip"))
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640], metavar=("H", "W"))
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.4, 0.55])
    args = parser.parse_args()

    from ultralytics import YOLO

    real_paths = sorted(p for p in (args.real / "images").iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    clip_paths = sorted(p for p in (args.clip / "images").iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    print(f"sky_test_real: {len(real_paths)} images | sky_test_clip: {len(clip_paths)} frames | input {args.imgsz[1]}x{args.imgsz[0]}")
    for wpath in args.weights:
        model = YOLO(wpath)
        lo = min(args.conf)
        real = predict_all(model, real_paths, tuple(args.imgsz), lo)
        clip = predict_all(model, clip_paths, tuple(args.imgsz), lo)
        print(f"\n== {wpath}")
        for c in args.conf:
            print(f"  conf {c:.2f} | real: {eval_real(real, args.real / 'labels', c)} | clip: {eval_clip(clip, args.clip / 'labels', c)}")


if __name__ == "__main__":
    main()
