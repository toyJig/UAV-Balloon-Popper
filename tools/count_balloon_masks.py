#!/usr/bin/env python3
"""Count Open Images train balloon segmentation masks, and how many belong
to images in our filtered balloon train set (excluding the hand-picked sky
test images). Prints a summary only."""
import csv
from pathlib import Path

ours = {p.stem[3:] for p in Path("datasets/openimages_balloon/train/images").glob("oi_*.jpg")}
picked = {Path(l).stem[3:] for l in Path("runs/sky_picked.txt").read_text().split() if Path(l).stem.startswith("oi_")}
with open("raw_datasets/openimages/meta/train-annotations-object-segmentation.csv", newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["LabelName"] == "/m/01j51"]
print("balloon masks in train:", len(rows), "images:", len({r["ImageID"] for r in rows}))
inset = [r for r in rows if r["ImageID"] in ours and r["ImageID"] not in picked]
print("in our filtered train set (excluding sky test):", len(inset), "images:", len({r["ImageID"] for r in inset}))
print("columns:", list(rows[0].keys()))
