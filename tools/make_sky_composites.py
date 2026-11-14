#!/usr/bin/env python3
"""Synthetic training images: real balloon cutouts (from Open Images masks,
tools/extract_balloon_cutouts.py) pasted into real sky photos, mostly
small and far away, as seen from a plane.

Backgrounds:
  * --sky-dir: Open Images photos of open sky with no flying objects
    (tools/fetch_openimages_sky.py). A share of these is written with no
    balloon at all, as "empty sky" negatives.
  * --drone-dir: Seraphim drone TRAIN images; their drone labels are kept
    and balloons are placed away from the drones, so both classes appear
    together on real airborne-camera frames.

Per pasted balloon:
  * size: longest side log-uniform between --min-px and --max-px at the
    640 px working size (most land under 40 px: far away),
  * place: preferably on smooth, sky-coloured pixels; sometimes anywhere
    (over trees/buildings),
  * appearance: random hue rotation (balloons come in every colour), mild
    saturation change, brightness pulled toward the local background, a
    little haze for small (far) balloons, soft 1 px alpha edge, optional
    motion blur along a random direction.
Then the whole image gets mild noise and JPEG re-compression so the pasted
pixels share the background's noise statistics (avoids the model learning
"sharp clean blob = balloon").

Labels: YOLO, class 0 = balloon (tight box of alpha > 0.5), class 1 = drone.
Output is a train-only YOLO dataset with data.yaml names {0: balloon, 1: drone}.

Usage:
    python tools/make_sky_composites.py --cutouts datasets/balloon_cutouts \\
        --sky-dir raw_datasets/openimages_sky/images --drone-dir datasets/seraphim_subset/train \\
        --drone-count 3000 --out datasets/synthetic_sky
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

WORK_SIDE = 640


def content_region(img: np.ndarray):
    """Bounding box (x1, y1, x2, y2) of the non-padding part of a
    letterboxed image (uniform bars at the edges)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = np.nonzero(gray.std(axis=1) > 2.0)[0]
    cols = np.nonzero(gray.std(axis=0) > 2.0)[0]
    if len(rows) < 32 or len(cols) < 32:
        return 0, 0, img.shape[1], img.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def sky_mask(img: np.ndarray) -> np.ndarray:
    """Rough per-pixel 'open sky' mask: smooth and blue-ish or bright grey."""
    b, g, r = [img[..., i].astype(np.int16) for i in range(3)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.blur(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1)), (9, 9))
    colour = (b > r + 8) | ((np.abs(b - r) < 25) & (gray > 140))
    mask = (colour & (grad < 25)).astype(np.uint8)
    # water is also smooth and blue: only accept sky-like pixels that are
    # connected to the top edge of the image region (sky continues upward)
    n, lab = cv2.connectedComponents(mask)
    top = set(np.unique(lab[0][mask[0] > 0]).tolist()) - {0}
    return np.isin(lab, list(top)).astype(np.uint8) if top else np.zeros_like(mask)


def load_cutouts(folder: Path):
    return sorted(folder.glob("*.png"))


def prepare_cutout(rgba: np.ndarray, target_long: float, rng: random.Random) -> np.ndarray:
    h, w = rgba.shape[:2]
    s = target_long / max(h, w)
    nw, nh = max(2, int(round(w * s))), max(2, int(round(h * s)))
    small = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = small[..., :3].astype(np.float32)
    alpha = small[..., 3].astype(np.float32) / 255.0

    if rng.random() < 0.7:  # hue rotation
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(0, 180)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.75, 1.2), 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if rng.random() < 0.2:  # random mirror
        rgb, alpha = rgb[:, ::-1], alpha[:, ::-1]
    if rng.random() < 0.5:  # small rotation (balloons tilt in wind)
        ang = rng.uniform(-25, 25)
        m = cv2.getRotationMatrix2D((nw / 2, nh / 2), ang, 1.0)
        rgb = cv2.warpAffine(np.ascontiguousarray(rgb), m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
        alpha = cv2.warpAffine(np.ascontiguousarray(alpha), m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    alpha = cv2.GaussianBlur(np.ascontiguousarray(alpha), (3, 3), 0.6 if target_long > 12 else 0.4)
    return np.dstack([rgb, alpha])


def motion_blur(rgba: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([3, 5, 7])
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    pad = k
    padded = cv2.copyMakeBorder(rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    premult = padded.copy()
    premult[..., :3] *= premult[..., 3:4]
    blurred = cv2.filter2D(premult, -1, kernel)
    a = blurred[..., 3:4]
    rgb = np.where(a > 1e-4, blurred[..., :3] / np.maximum(a, 1e-4), 0)
    return np.dstack([rgb, a[..., 0]])


def paste(bg: np.ndarray, piece: np.ndarray, x: int, y: int, rng: random.Random, long_side: float):
    """Alpha-blend piece onto bg at (x, y) in place; returns tight box or None."""
    h, w = piece.shape[:2]
    H, W = bg.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    p = piece[y1 - y:y2 - y, x1 - x:x2 - x]
    rgb, a = p[..., :3], p[..., 3:4]
    region = bg[y1:y2, x1:x2].astype(np.float32)
    local = region.reshape(-1, 3).mean(axis=0)
    # brightness toward local background (same sun), haze for far balloons
    target = 0.35 * (local.mean() - rgb.mean()) * rng.uniform(0.3, 1.0)
    rgb = rgb + target
    haze = float(np.clip(0.2 * (1 - long_side / 40.0), 0, 0.2)) * rng.uniform(0.3, 1.0)
    rgb = rgb * (1 - haze) + local * haze
    out = region * (1 - a) + np.clip(rgb, 0, 255) * a
    bg[y1:y2, x1:x2] = out.astype(np.uint8)
    ys, xs = np.nonzero(a[..., 0] > 0.5)
    if len(xs) == 0:
        return None
    return x1 + xs.min(), y1 + ys.min(), x1 + xs.max() + 1, y1 + ys.max() + 1


def overlaps(box, boxes, margin=4):
    for b in boxes:
        if box[0] - margin < b[2] and box[2] + margin > b[0] and box[1] - margin < b[3] and box[3] + margin > b[1]:
            return True
    return False


def read_yolo(label_path: Path, w: int, h: int):
    out = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def to_yolo(c, box, w, h):
    x1, y1, x2, y2 = box
    return f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def compose(bg: np.ndarray, existing, cutouts, rng: random.Random, args):
    H, W = bg.shape[:2]
    rx1, ry1, rx2, ry2 = content_region(bg)
    sky = sky_mask(bg)
    sky[:ry1], sky[ry2:], sky[:, :rx1], sky[:, rx2:] = 0, 0, 0, 0
    sky_pts = np.argwhere(sky > 0)
    n = rng.choices([1, 2, 3, 4], weights=[60, 22, 12, 6])[0]
    placed, lines = [], []
    taken = [b for _, b in existing]
    for _ in range(n):
        long_side = math.exp(rng.uniform(math.log(args.min_px), math.log(args.max_px)))
        piece = prepare_cutout(cv2.imread(str(rng.choice(cutouts)), cv2.IMREAD_UNCHANGED), long_side, rng)
        if rng.random() < args.motion_blur_p:
            piece = motion_blur(piece, rng)
        ph, pw = piece.shape[:2]
        for _attempt in range(25):
            if len(sky_pts) and rng.random() < 0.9:
                cy, cx = sky_pts[rng.randrange(len(sky_pts))]
            else:
                cx, cy = rng.uniform(rx1, rx2), rng.uniform(ry1, ry2)
            x, y = int(cx - pw / 2), int(cy - ph / 2)
            cand = (x, y, x + pw, y + ph)
            if x < rx1 or y < ry1 or x + pw > rx2 or y + ph > ry2 or overlaps(cand, taken + placed):
                continue
            box = paste(bg, piece, x, y, rng, long_side)
            if box is not None and min(box[2] - box[0], box[3] - box[1]) >= args.min_label_px:
                placed.append(box)
                lines.append(to_yolo(0, box, W, H))
            break
    return placed, lines


def finish(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.7:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1.0, 4.0), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.8))
    q = rng.randint(70, 95)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def fit(img: np.ndarray) -> np.ndarray:
    s = WORK_SIDE / max(img.shape[:2])
    if abs(s - 1) > 1e-3:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutouts", type=Path, required=True)
    parser.add_argument("--sky-dir", type=Path, required=True)
    parser.add_argument("--drone-dir", type=Path, default=None, help="YOLO split dir with images/ and labels/")
    parser.add_argument("--drone-count", type=int, default=3000)
    parser.add_argument("--sky-repeats", type=int, default=1, help="composites per sky background")
    parser.add_argument("--sky-count", type=int, default=3500, help="max sky backgrounds to use")
    parser.add_argument("--min-sky-fraction", type=float, default=0.35)
    parser.add_argument("--empty-fraction", type=float, default=0.15, help="sky backgrounds kept with no balloon")
    parser.add_argument("--min-px", type=float, default=7)
    parser.add_argument("--max-px", type=float, default=120)
    parser.add_argument("--min-drone-sky", type=float, default=0.3,
                        help="only use drone frames whose content is at least this fraction sky-like")
    parser.add_argument("--min-label-px", type=int, default=5)
    parser.add_argument("--motion-blur-p", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cutouts = load_cutouts(args.cutouts)
    img_out, lbl_out = args.out / "train" / "images", args.out / "train" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    sky_files = sorted(args.sky_dir.glob("*.jpg"))
    rng.shuffle(sky_files)
    kept_sky = []
    for p in sky_files:  # keep photos where open sky (connected to the top) covers enough of the frame
        if len(kept_sky) >= args.sky_count:
            break
        img = cv2.imread(str(p))
        if img is not None and sky_mask(fit(img)).mean() >= args.min_sky_fraction:
            kept_sky.append(p)
    print(f"sky backgrounds: {len(kept_sky)} with >= {args.min_sky_fraction:.0%} open sky")
    for p in kept_sky:
        for r in range(args.sky_repeats):
            jobs.append(("sky", p, None, r))
    if args.drone_dir is not None:
        drones = sorted((args.drone_dir / "images").glob("*.jpg"))
        rng.shuffle(drones)
        picked, scanned = [], 0
        for p in drones:  # keep frames that actually show a good amount of sky
            if len(picked) >= args.drone_count:
                break
            scanned += 1
            img = cv2.imread(str(p))
            if img is None:
                continue
            x1, y1, x2, y2 = content_region(img)
            if sky_mask(img[y1:y2, x1:x2]).mean() >= args.min_drone_sky:
                picked.append(p)
        print(f"drone backgrounds: {len(picked)} sky-heavy frames out of {scanned} scanned")
        jobs += [("drone", p, args.drone_dir / "labels" / (p.stem + ".txt"), 0) for p in picked]

    counts = {"sky_composite": 0, "sky_empty": 0, "drone_composite": 0, "balloons": 0}
    for i, (kind, path, label_path, rep) in enumerate(jobs):
        bg = cv2.imread(str(path))
        if bg is None:
            continue
        existing = []
        if kind == "drone":
            existing = [(c, b) for c, b in read_yolo(label_path, bg.shape[1], bg.shape[0])]
        bg = fit(bg) if kind == "sky" else bg
        H, W = bg.shape[:2]
        lines = [to_yolo(1 if kind == "drone" else c, b, W, H) for c, b in existing]
        if kind == "sky" and rng.random() < args.empty_fraction:
            name = f"synsky_empty_{path.stem}_{rep}"
            counts["sky_empty"] += 1
        else:
            placed, blines = compose(bg, existing, cutouts, rng, args)
            lines += blines
            counts["balloons"] += len(placed)
            counts["sky_composite" if kind == "sky" else "drone_composite"] += 1
            name = f"syn{kind}_{path.stem}_{rep}"
        cv2.imwrite(str(img_out / f"{name}.jpg"), finish(bg, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        (lbl_out / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(jobs)}", flush=True)
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: train/images\nval: train/images\nnames:\n  0: balloon\n  1: drone\n")
    print(counts)


if __name__ == "__main__":
    main()
