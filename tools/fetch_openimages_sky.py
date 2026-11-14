#!/usr/bin/env python3
"""Download Open Images TRAIN images that are mostly open sky and contain
no flying objects, as backgrounds for synthetic balloon composites.

Candidate filter (annotation-based):
  * human-verified image-level label "Sky" (Confidence 1),
  * NO positive image-level label or box for anything that flies or could
    be mistaken for a target: Balloon, Hot air balloon, Bird, Kite,
    Aircraft, Airplane, Fixed-wing aircraft, Helicopter, Drone, RC aircraft,
    Model aircraft, Toy airplane, Insect, Butterfly, Moths, Parachute,
    Paragliding, Rocket, Flag.

Then a CLIP (ViT-L/14) zero-shot check keeps images that look like open
sky (with or without horizon, clouds, treetops, buildings at the edge)
rather than portraits / interiors / street scenes that merely show some
sky.

Usage:
    python tools/fetch_openimages_sky.py --meta-dir raw_datasets/openimages/meta \\
        --out raw_datasets/openimages_sky --candidates 6000 --keep 3000
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SKY = "/m/01bqvp"
EXCLUDE = {
    "/m/01j51", "/m/09j06", "/m/02p81ht", "/m/084x1x",  # balloons, hot air, cluster ballooning
    "/m/015p6", "/m/02zt3",                               # bird, kite
    "/m/0k5j", "/m/05czz6l", "/m/0cmf2", "/m/09ct_",      # aircraft, airplane, fixed-wing, helicopter
    "/m/0g2bc", "/m/053scx", "/m/0l8fl", "/m/0h8mhmy", "/m/074l3b",  # drone, RC/model/toy aircraft, RC heli
    "/m/03vt0", "/m/0cyf8", "/m/0d_2m", "/m/01h44",       # insect, butterfly, moths and butterflies, bat
    "/m/0cyfs", "/m/0h1j7", "/m/09rvcxw",                 # parachute, paragliding, rocket
    "/m/03120", "/m/01jfsr", "/m/025_r8s",                # flag, lantern, sky lantern
}
S3 = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"
SKY_PROMPTS = ["a photo of the open sky", "a photo of a blue sky with clouds",
               "a photo of the sky above trees", "a photo of the sky above a city skyline",
               "a photo of a cloudy sky", "a photo of the sky at sunset"]
OTHER_PROMPTS = ["a photo of people", "a photo of an indoor room", "a photo of a street",
                 "a close-up photo of an object", "a photo of a building facade", "a photo of a landscape with no sky"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=6000)
    parser.add_argument("--keep", type=int, default=3000)
    parser.add_argument("--max-side", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("scanning image-level labels ...", flush=True)
    sky, excluded = set(), set()
    with open(args.meta_dir / "train-imagelabels.csv", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        c = {n: i for i, n in enumerate(next(r))}
        for row in r:
            if row[c["Confidence"]] not in ("1", "1.0"):
                continue
            if row[c["LabelName"]] == SKY:
                sky.add(row[c["ImageID"]])
            elif row[c["LabelName"]] in EXCLUDE:
                excluded.add(row[c["ImageID"]])
    print("scanning boxes ...", flush=True)
    with open(args.meta_dir / "oidv6-train-annotations-bbox.csv", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        c = {n: i for i, n in enumerate(next(r))}
        for row in r:
            if row[c["LabelName"]] in EXCLUDE:
                excluded.add(row[c["ImageID"]])
    # Only "boxable" images are on the public S3 bucket, and only boxable
    # images had every box drawn - which is what makes the exclusion list
    # above trustworthy. train-images-boxable-with-rotation.csv lists them.
    boxable, rotated = set(), set()
    with open(args.meta_dir / "train-images-rotation.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ImageID"] in sky:
                boxable.add(row["ImageID"])
                if (row.get("Rotation") or "").strip() not in ("", "0", "0.0"):
                    rotated.add(row["ImageID"])
    pool = sorted((sky & boxable) - excluded - rotated)
    print(f"sky={len(sky)} boxable={len(boxable)} excluded={len(boxable & excluded)} rotated={len(rotated)} "
          f"-> pool {len(pool)}", flush=True)
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.candidates]

    cand_dir = args.out / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    def fetch(image_id):
        dst = cand_dir / f"{image_id}.jpg"
        if dst.exists():
            return dst
        try:
            with urllib.request.urlopen(S3.format(image_id), timeout=30) as resp:
                data = resp.read()
        except Exception:
            return None
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            return None
        s = args.max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return dst

    print(f"downloading {len(pool)} candidates ...", flush=True)
    with ThreadPoolExecutor(32) as ex:
        paths = [p for p in ex.map(fetch, pool) if p is not None]
    print(f"downloaded {len(paths)}", flush=True)

    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    with torch.no_grad():
        t = proc(text=SKY_PROMPTS + OTHER_PROMPTS, return_tensors="pt", padding=True).to(device)
        text = model.text_projection(model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output)
        text = text / text.norm(dim=-1, keepdim=True)
    scores = []
    for s in range(0, len(paths), 48):
        batch = paths[s:s + 48]
        ims = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in batch]
        with torch.no_grad():
            px = proc(images=ims, return_tensors="pt").to(device)
            e = model.visual_projection(model.vision_model(pixel_values=px["pixel_values"]).pooler_output)
            e = e / e.norm(dim=-1, keepdim=True)
            pr = (model.logit_scale.exp() * e @ text.T).softmax(-1)[:, :len(SKY_PROMPTS)].sum(-1).cpu().numpy()
        scores += list(zip(batch, pr))
    scores.sort(key=lambda x: -x[1])
    keep = [(p, s) for p, s in scores if s >= 0.5][: args.keep]
    keep_dir = args.out / "images"
    keep_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "clip_sky", "kept"])
        kept_ids = {p.stem for p, _ in keep}
        for p, s in scores:
            w.writerow([p.stem, f"{s:.3f}", int(p.stem in kept_ids)])
    for p, _ in keep:
        dst = keep_dir / p.name
        if not dst.exists():
            dst.write_bytes(p.read_bytes())
    print(f"kept {len(keep)} sky backgrounds -> {keep_dir}")


if __name__ == "__main__":
    sys.exit(main())
