#!/usr/bin/env python3
"""Quarantine images whose "balloon" boxes are actually hot-air balloons
or airships, using CLIP zero-shot classification on each box crop.

Why: Open Images has no boxable hot-air-balloon class, so hot-air balloons
are boxed as "Balloon". Its image-level "Hot air balloon" label (already
filtered in fetch_openimages_balloons.py) is only present where a human
verified it, so many slip through. Training on them would teach the
detector that hot-air balloons are targets.

Rule: an image is quarantined if the whole-image scene score says "hot-air
balloon" (>= --scene-threshold) and at least one box crop agrees
(>= --crop-confirm), or if a single crop is near-certain on its own
(>= --crop-only-threshold). Quarantined image+label pairs are MOVED
(not deleted) to --quarantine, mirroring the split layout, so the decision
is reviewable and reversible.

Usage:
    python tools/filter_hot_air_balloons.py --dataset datasets/openimages_balloon --quarantine datasets/openimages_balloon_quarantine --report-only
    python tools/filter_hot_air_balloons.py --dataset datasets/openimages_balloon --quarantine datasets/openimages_balloon_quarantine
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

MODEL_ID = "openai/clip-vit-large-patch14"
REJECT_PROMPTS = [
    "a photo of a hot air balloon",
    "a photo of a hot air balloon flying in the sky",
    "a photo of an airship or blimp",
]
KEEP_PROMPTS = [
    "a photo of a party balloon",
    "a photo of a latex toy balloon",
    "a photo of a shiny foil balloon",
    "a photo of a bunch of party balloons",
    "a photo of a balloon on a string",
]
# Whole-image scene prompts. Crop scores alone over-flag busy party scenes
# (one blurry crop out of twenty is enough to trip a max), so the final rule
# combines both views - see main().
SCENE_REJECT_PROMPTS = [
    "a photo of hot air balloons",
    "a photo of a hot air balloon festival",
    "a photo of a hot air balloon being inflated",
    "a photo of an airship in the sky",
]
SCENE_KEEP_PROMPTS = [
    "a photo of party balloons",
    "a photo of people holding balloons",
    "a photo of balloon decorations at a party",
    "a photo of toy balloons floating in the sky",
    "a photo of a single balloon",
]
PAD_FRACTION = 0.15
MIN_CROP_PX = 24


def crop_boxes(img: np.ndarray, label_path: Path) -> "list[np.ndarray]":
    h, w = img.shape[:2]
    crops = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:])
        pad_w, pad_h = bw * PAD_FRACTION, bh * PAD_FRACTION
        x1 = int(max(0, (cx - bw / 2 - pad_w) * w))
        y1 = int(max(0, (cy - bh / 2 - pad_h) * h))
        x2 = int(min(w, (cx + bw / 2 + pad_w) * w))
        y2 = int(min(h, (cy + bh / 2 + pad_h) * h))
        if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
            continue  # too small for CLIP to judge reliably
        crops.append(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    return crops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--prefix", default="oi_", help="only check images whose name starts with this")
    parser.add_argument("--scene-threshold", type=float, default=0.5,
                        help="flag if the whole-image hot-air score reaches this...")
    parser.add_argument("--crop-confirm", type=float, default=0.3,
                        help="...and at least one box crop also scores this much (both views must agree)")
    parser.add_argument("--crop-only-threshold", type=float, default=0.86,
                        help="flag on crop score alone only when it is this certain")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    def embed_text(prompts):
        with torch.no_grad():
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            # Apply the projection heads explicitly: get_text_features/get_image_features
            # changed return types between transformers 4.x and 5.x.
            pooled = model.text_model(input_ids=text_inputs["input_ids"],
                                      attention_mask=text_inputs["attention_mask"]).pooler_output
            emb = model.text_projection(pooled)
            return emb / emb.norm(dim=-1, keepdim=True)

    def p_reject(images_rgb, text_emb, n_reject):
        with torch.no_grad():
            inputs = processor(images=images_rgb, return_tensors="pt").to(device)
            pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            img_emb = model.visual_projection(pooled)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            probs = (model.logit_scale.exp() * img_emb @ text_emb.T).softmax(dim=-1)
            return probs[:, :n_reject].sum(dim=-1).cpu().numpy()

    crop_text = embed_text(REJECT_PROMPTS + KEEP_PROMPTS)
    scene_text = embed_text(SCENE_REJECT_PROMPTS + SCENE_KEEP_PROMPTS)

    rows = []
    for split in ("train", "valid", "test"):
        img_dir = args.dataset / split / "images"
        if not img_dir.exists():
            continue
        items = []  # (image path, crop) per box
        scenes = []  # (image path, whole image)
        for img_path in sorted(img_dir.glob(f"{args.prefix}*")):
            label_path = args.dataset / split / "labels" / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            img = cv2.imread(str(img_path))
            scenes.append((img_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            for crop in crop_boxes(img, label_path):
                items.append((img_path, crop))

        crop_max = {p: 0.0 for p, _ in scenes}
        for start in range(0, len(items), args.batch):
            batch = items[start:start + args.batch]
            for (img_path, _), p in zip(batch, p_reject([c for _, c in batch], crop_text, len(REJECT_PROMPTS))):
                crop_max[img_path] = max(crop_max[img_path], float(p))
        scene_p = {}
        for start in range(0, len(scenes), args.batch):
            batch = scenes[start:start + args.batch]
            for (img_path, _), p in zip(batch, p_reject([s for _, s in batch], scene_text, len(SCENE_REJECT_PROMPTS))):
                scene_p[img_path] = float(p)

        def is_flagged(img_path):
            s, c = scene_p[img_path], crop_max[img_path]
            return (s >= args.scene_threshold and c >= args.crop_confirm) or c >= args.crop_only_threshold

        flagged = [p for p in scene_p if is_flagged(p)]
        print(f"[{split}] {len(items)} crops from {len(scene_p)} images -> {len(flagged)} images flagged")
        for img_path in scene_p:
            rows.append({"split": split, "image": img_path.name, "p_hot_air": f"{scene_p[img_path]:.3f}",
                         "p_crop_max": f"{crop_max[img_path]:.3f}", "flagged": int(is_flagged(img_path))})

        if not args.report_only:
            for img_path in flagged:
                label_path = args.dataset / split / "labels" / (img_path.stem + ".txt")
                q_img = args.quarantine / split / "images"
                q_lbl = args.quarantine / split / "labels"
                q_img.mkdir(parents=True, exist_ok=True)
                q_lbl.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), q_img / img_path.name)
                shutil.move(str(label_path), q_lbl / label_path.name)

    args.quarantine.mkdir(parents=True, exist_ok=True)
    with open(args.quarantine / "hot_air_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "p_hot_air", "p_crop_max", "flagged"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: -float(r["p_hot_air"])))
    print(f"scores written to {args.quarantine / 'hot_air_scores.csv'}"
          + ("" if args.report_only else "; flagged images moved to quarantine"))


if __name__ == "__main__":
    main()
