#!/usr/bin/env python3
"""Score how much each balloon image looks like "balloons in the sky" (the
use case) versus indoor / party / street scenes, using CLIP zero-shot on
the whole image plus a pixel check of the area around each balloon box.

Writes a CSV (image path, split, CLIP sky probability, fraction of boxes
whose surroundings are plain sky-like pixels) for picking a sky subset,
e.g. a real-world sky test set. Nothing is moved or deleted.

Usage:
    python tools/score_sky_scenes.py --datasets datasets/openimages_balloon datasets/lvis_balloon \\
        --prefixes oi_ lvis_ --out runs/sky_scores.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

MODEL_ID = "openai/clip-vit-large-patch14"
SKY_PROMPTS = [
    "a photo of balloons floating in the sky",
    "a photo of a balloon high up in the sky",
    "a photo of balloons against a blue sky",
    "a photo of a balloon against a cloudy sky",
    "a photo of a small balloon far away in the sky",
]
OTHER_PROMPTS = [
    "a photo of balloons indoors",
    "a photo of a party with balloon decorations",
    "a photo of people holding balloons",
    "a photo of balloons on a street",
    "a close-up photo of a balloon",
]


def ring_is_sky(img_bgr: np.ndarray, box, pad: float = 0.6) -> bool:
    """True when the ring of pixels around a box is smooth and bright/bluish,
    i.e. the balloon is seen against plain sky."""
    h, w = img_bgr.shape[:2]
    cx, cy, bw, bh = box
    x1, y1 = int((cx - bw / 2 - bw * pad) * w), int((cy - bh / 2 - bh * pad) * h)
    x2, y2 = int((cx + bw / 2 + bw * pad) * w), int((cy + bh / 2 + bh * pad) * h)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    ix1, iy1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    ix2, iy2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    region = img_bgr[y1:y2, x1:x2].astype(np.float32)
    if region.size == 0:
        return False
    mask = np.ones(region.shape[:2], bool)
    mask[max(0, iy1 - y1):max(0, iy2 - y1), max(0, ix1 - x1):max(0, ix2 - x1)] = False
    if mask.sum() < 50:
        return False
    gray = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    b, g, r = region[..., 0][mask], region[..., 1][mask], region[..., 2][mask]
    smooth = np.median(grad[mask]) < 12
    sky_colour = np.median(b) >= np.median(r) - 10 and np.median(gray[mask]) > 90
    return bool(smooth and sky_colour)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--prefixes", nargs="+", required=True, help="image name prefix per dataset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    with torch.no_grad():
        t = processor(text=SKY_PROMPTS + OTHER_PROMPTS, return_tensors="pt", padding=True).to(device)
        text = model.text_projection(model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output)
        text = text / text.norm(dim=-1, keepdim=True)

    items = []
    for ds, prefix in zip(args.datasets, args.prefixes):
        for split in ("train", "valid", "test"):
            d = ds / split / "images"
            if d.exists():
                items += [(p, split) for p in sorted(d.glob(f"{prefix}*.jpg"))]
    print(f"scoring {len(items)} images", flush=True)

    rows = []
    for start in range(0, len(items), args.batch):
        batch = items[start:start + args.batch]
        imgs = [cv2.imread(str(p)) for p, _ in batch]
        with torch.no_grad():
            px = processor(images=[cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs], return_tensors="pt").to(device)
            emb = model.visual_projection(model.vision_model(pixel_values=px["pixel_values"]).pooler_output)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            probs = (model.logit_scale.exp() * emb @ text.T).softmax(dim=-1)[:, :len(SKY_PROMPTS)].sum(-1).cpu().numpy()
        for (p, split), im, prob in zip(batch, imgs, probs):
            label = p.parent.parent / "labels" / (p.stem + ".txt")
            boxes = [tuple(map(float, l.split()[1:5])) for l in label.read_text().splitlines() if len(l.split()) == 5]
            sky_boxes = sum(ring_is_sky(im, b) for b in boxes)
            rows.append({"image": str(p), "split": split, "clip_sky": f"{prob:.3f}",
                         "boxes": len(boxes), "sky_ring_fraction": f"{sky_boxes / max(1, len(boxes)):.2f}"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -float(r["clip_sky"])))
    probs = np.array([float(r["clip_sky"]) for r in rows])
    print("clip_sky histogram [0,.2,.4,.5,.6,.8,1]:", np.histogram(probs, bins=[0, .2, .4, .5, .6, .8, 1.01])[0].tolist())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
