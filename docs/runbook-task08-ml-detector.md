# Task 8 Runbook - ML Detector (balloon/drone)

## Datasets used (all downloadable without an account)

| Class | Source | License | What we took |
|-------|--------|---------|--------------|
| drone | [Seraphim Drone Detection Dataset](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset) (Hugging Face, revision `5b1c834`) | CC BY 4.0 | Random 20,000 of its 75,134 train images. Its 8,349 test images were split in half for our valid/test. |
| balloon | [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) "Balloon" boxes | Annotations CC BY 4.0, images listed as CC BY 2.0 (per-image credit in `datasets/openimages_balloon/attribution.csv`) | Filtered images, see below |
| balloon | [LVIS v1](https://www.lvisdataset.org/) "balloon" boxes on COCO 2017 images | Annotations CC BY 4.0, COCO images under Flickr terms | Filtered images, see below |
| background | Open Images images with Bird, Kite, Aircraft, Fixed-wing aircraft or Helicopter boxes | same as Open Images | Empty-label images, so the model learns not to fire on other things in the sky |

Seraphim already merges 23 public drone sets (including the Kaggle and
Roboflow drone sets you'd otherwise download by hand) and removes
duplicates, so no manual Kaggle/Roboflow downloads were needed.

The Kaggle "V2 Balloon Detection Dataset" was not used. It contains 74
images, it is the Matterport Mask R-CNN balloon sample set, and its
photos come from Flickr. Open Images and COCO are also Flickr-sourced,
so it would add almost nothing while risking the same photo landing in
both train and test.

## Filtering, and why

Open Images balloon images (3,262 train candidates):
- Dropped the whole image if any balloon box is `IsGroupOf` (one box drawn
  around a cluster) or `IsDepiction` (drawings/prints): 1,270 images.
  Keeping the image but dropping only that box would leave unlabeled
  balloons, which teaches the model that balloons are background.
- Dropped images human-verified as "Hot air balloon", "Hot air ballooning"
  or "Cluster ballooning": 449 images.
- Dropped images verified to contain a drone, RC aircraft, model aircraft
  or toy airplane, because those would be unlabeled drones.
- Dropped images with more than 20 balloon boxes (dense decoration arches)
  and images whose rotation metadata is non-zero.
- Then a CLIP (ViT-L/14) zero-shot check on both the whole image and every
  box crop moved 125 more hot-air-balloon/airship images to
  `datasets/openimages_balloon_quarantine/` (scores in `hot_air_scores.csv`
  there). Those had slipped through because Open Images has no boxable
  hot-air-balloon class. The thresholds were checked by eye on contact
  sheets above and below the cut-off.
- Open Images' own validation/test splits only had 19 and 38 usable balloon
  images, so 8% of the train balloons (with matching backgrounds) moved to
  our valid split.

LVIS balloon images (241 candidates):
- Dropped images where "balloon" is listed as not exhaustively labeled,
  images with LVIS's separate hot-air-balloon category, and images with
  more than 20 balloons.
- CLIP flagged 9; after viewing them, only the 2 real hot-air balloon
  images were moved to `datasets/lvis_balloon_quarantine/`. The other 7
  were small party balloons the scorer misread.

Merge (`datasets/combined`, classes `0=balloon, 1=drone`):
- Balloon train images that contain balloons are included twice
  (hardlinked copies named `*_r1`) to offset the roughly 7:1 drone/balloon
  image imbalance. Background images are not repeated.
- Near-duplicate removal (256-bit dHash, Hamming distance <= 24, threshold
  set by viewing pairs at several distances): valid/test images that
  near-duplicate a train image are deleted, which removed 258 valid and
  281 test images. Nearly all of these were drone video frames: Seraphim's
  own split shares frames from the same videos.

Final counts:

| split | drone imgs | balloon imgs | background imgs |
|-------|-----------:|-------------:|----------------:|
| train | 20,000 | 2,818 (1,409 unique) | 2,021 |
| valid | 3,916 | 135 | 203 |
| test  | 3,881 | 47 | 57 |

The balloon test split is small (47 images). Treat balloon test numbers as
a rough signal; your own camera footage (below) is the real test.

## Rebuild from scratch

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-dev.txt -r requirements-train.txt

# drones (9 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('lgrzybowski/seraphim-drone-detection-dataset', repo_type='dataset', revision='5b1c8348bf67e6836e3dbc42b053a5a4bfdec925', local_dir='raw_datasets/seraphim')"
python tools/prepare_seraphim.py --raw raw_datasets/seraphim --out datasets/seraphim_subset --train-count 20000

# balloons: Open Images CSVs (~6 GB) into raw_datasets/openimages/meta first
#   (file list and URLs: see SPLITS in tools/fetch_openimages_balloons.py)
python tools/fetch_openimages_balloons.py --meta-dir raw_datasets/openimages/meta --out datasets/openimages_balloon
#   LVIS annotations: lvis_v1_train.json.zip + lvis_v1_val.json.zip from dl.fbaipublicfiles.com/LVIS/ into raw_datasets/lvis
python tools/fetch_lvis_balloons.py --lvis-dir raw_datasets/lvis --out datasets/lvis_balloon
python tools/filter_hot_air_balloons.py --dataset datasets/openimages_balloon --quarantine datasets/openimages_balloon_quarantine

# merge, dedup, train
python tools/dataset_prep.py --drone-dataset datasets/seraphim_subset --balloon-dataset datasets/openimages_balloon datasets/lvis_balloon --balloon-train-repeat 2 --out datasets/combined
python tools/dedup_splits.py --dataset datasets/combined --prefix balloon- --max-distance 24
python tools/dedup_splits.py --dataset datasets/combined --prefix drone- --max-distance 24
python tools/train_yolo.py --data datasets/combined/data.yaml --name balloon_drone_v1
```

`tools/label_contact_sheet.py` draws the boxes on a random sample so you
can confirm labels line up before training.

## Training run v1

- YOLO11n (COCO-pretrained), 640px, batch 32, 80 epochs on an RTX 4070
  Laptop GPU, about 3.5 minutes per epoch (4.6 hours total). Validation
  mAP50 reached 0.877 by epoch 35 and stayed between 0.874 and 0.877 from
  there, so more epochs on this data would not help.
- Windows note: each dataloader worker is a separate ~0.6-0.9 GB process,
  and Ultralytics gives the validation loader twice as many by default.
  With `--workers 8` the first attempt used ~18 GB and ran the 31 GB
  machine out of RAM. `tools/train_yolo.py` now defaults to 4 workers for
  both loaders plus `--cache disk`.
- The disk cache is 31.6 GB of `.npy` files inside
  `datasets/combined/train/images` and `valid/images`. Delete them
  (`*.npy` only) if you need the space; the next training run rebuilds
  them.
- Output: `runs/lockon/balloon_drone_v1/weights/best.pt` (last epoch's
  validation numbers are within 0.002 of the best), plus `results.csv` and
  plots in the same folder.

### Results

Validation picked the checkpoint, so the held-out **test** split is the
honest number. IoU >= 0.5:

| class | split | precision | recall | mAP50 | mAP50-95 |
|-------|-------|----------:|-------:|------:|---------:|
| drone | valid | 0.958 | 0.917 | 0.959 | 0.673 |
| drone | **test** | 0.978 | 0.902 | 0.966 | 0.685 |
| balloon | valid | 0.854 | 0.747 | 0.795 | 0.613 |
| balloon | **test** | 0.422 | 0.607 | 0.438 | 0.353 |

The balloon test score is low mostly because of how the test images are
labeled, not only because the model is weak there. Per-source breakdown at
confidence 0.40 (`tools/error_breakdown.py`):

- 157 of the 159 balloon false positives on test are in Open Images
  photos, almost all party scenes (balloon arches, ceilings full of
  balloons) where only some balloons are boxed. The model finds the
  unboxed ones, and they count as false positives. Only 17 test images
  have any balloon false positive at all.
- Many misses are the opposite labeling case: one box drawn around several
  touching balloons that the model splits into individual balloons. The
  rest are genuinely hard: small balloons in crowds, balloons partly cut
  off, and some balloon-shaped inflatables the labelers counted.
- The balloon test set is only 47 images (178 boxes). The 106 valid+test
  images with 1-3 balloons, the closest match to the real use case, give
  precision 0.50 and recall 0.77 at confidence 0.40.

On no-target images (birds, kites, aircraft, helicopters; 260 images in
valid+test), the detector fires on 12 (4.6%) at confidence 0.40.
Looking at them: distant jets and a helicopter called "drone" (at range
they look alike), round objects called "balloon" (an airliner nose, a
cockatoo's head). One "background" image is an actual Predator drone,
and one shows a partly visible red balloon - the background labels are
not perfect either.

| conf threshold | false-alarm images | balloon P/R (1-3 balloons) | drone P/R |
|---:|---:|---:|---:|
| 0.25 | 5.8% | 0.454 / 0.800 | 0.960 / 0.928 |
| 0.40 (default) | 4.6% | 0.500 / 0.772 | 0.975 / 0.901 |
| 0.50 | 3.8% | 0.534 / 0.744 | 0.981 / 0.873 |
| 0.60 | 2.7% | 0.578 / 0.722 | 0.987 / 0.826 |
| 0.70 | 0.8% | 0.646 / 0.689 | 0.994 / 0.712 |

(`tools/threshold_sweep.py`.) Keep 0.40 until you have your own footage,
then re-pick it on that footage.

### Input size

Accuracy vs. inference size on valid+test at confidence 0.40
(`tools/eval_sizes.py`, square inputs):

| imgsz | drone P/R | balloon P/R (1-3 balloons) |
|---:|---:|---:|
| 640 | 0.975 / 0.901 | 0.500 / 0.772 |
| 480 | 0.977 / 0.871 | 0.615 / 0.772 |
| 416 | 0.979 / 0.852 | 0.621 / 0.783 |
| 320 | 0.972 / 0.796 | 0.650 / 0.711 |

Drone recall drops steadily below 640 because many Seraphim drones are
tiny. The camera captures 640x480, so the exports use a 640x480 input
(no letterbox padding, 25% less compute than 640x640) as the default, and
416x320 as a faster fallback if the Pi can't keep up.

## Export and Pi package (done for v1)

```
python tools/export_ncnn.py --weights runs/lockon/balloon_drone_v1/weights/best.pt \
    --out models/balloon_drone_v1 --sizes 640x480 416x320
```

This produced `models/balloon_drone_v1/`:

```
best.pt                 PyTorch weights (for future fine-tuning)
ncnn_640x480/           default: model.ncnn.param, model.ncnn.bin, metadata.yaml
ncnn_416x320/           faster fallback
```

Pass an `ncnn_<WxH>` folder as `--model-path`. Input size and class
names are read from its `metadata.yaml`; NCNN exports only accept the
size they were exported at.

Checks already run on the laptop (the `ncnn` package also installs on
Windows):

- **Output layout**: verified against the real export. Blob names are
  `in0`/`out0` and are now read from the network instead of hardcoded; the
  output is `[6, N]` (box cx, cy, w, h + 2 class scores).
- **Parity with PyTorch** (`tools/check_ncnn_parity.py`, 300 test images):
  our full Pi path (letterbox, NCNN, decode, NMS) reproduced 392/392
  PyTorch detections at IoU >= 0.9 for 640x480 and 351/352 for 416x320,
  with confidence differences under 0.002.
- **Speed on the laptop CPU** (Ryzen 9 8945HS, 4 threads): `detect()`
  takes 42-51 ms per frame at 640x480 (20-24 fps), 25 ms at 416x416. The
  Pi 5 will be slower; published YOLO11n NCNN numbers for the Pi 5 are
  around 70 ms at 640x640. Measure it on the Pi:

```
python tools/benchmark_ml_detector.py --model-path models/balloon_drone_v1/ncnn_640x480 \
    --held-out-dir datasets/combined/test --max-images 500
```

Copy `models/balloon_drone_v1/` and `datasets/combined/test/` (images +
labels, ~4,000 images) to the Pi for that; `models/` and `datasets/` are
git-ignored, so `git clone` alone won't bring them.

### Fixes made while verifying the Pi path

- **Camera color order**: Picamera2's `"RGB888"` format actually delivers
  B,G,R byte order (libcamera names formats by packed-word order), which
  would have shown a red balloon to both detectors as blue.
  `src/lockon/vision/capture.py` now requests `"BGR888"`, which yields
  R,G,B arrays. Confirmed from Picamera2's own source (its PIL conversion
  maps `RGB888` to `BGR`). Check it on the Task 4 monitor setup: a red
  balloon should get a box in `live_overlay_demo.py --mode fallback`.
- **Tracker could hold a false lock forever**: after a single detection,
  the tracker kept "tracking" whatever patch it started on with no time
  limit, so one false positive could become a lock that never ends.
  `HybridDetector` now drops the lock after 10 frames without a detector
  re-confirming it (`max_bridge_frames`).

## v2: balloons against the sky (current deploy model)

v1 was trained mostly on ground-level balloon photos and did poorly on
balloons seen from a plane against open sky (the real use case). v2
fine-tunes v1 for the sky case.

What changed in the data:

- Synthetic sky composites (`datasets/synthetic_sky`, 7992 balloon boxes):
  balloon cutouts (`tools/extract_balloon_cutouts.py` from Open Images
  masks, filtered for solidity/ellipse shape) pasted onto open-sky photos
  and sky-heavy Seraphim frames (`tools/make_sky_composites.py`). Median
  box ~28 px, i.e. far-away balloons.
- Two held-out real sky tests, never trained on:
  `datasets/sky_test_real` (43 hand-picked sky photos) and
  `datasets/sky_test_clip` (277 frames of a real balloon-vs-sky clip).

Training (fine-tune from v1 `best.pt`, 16 epochs, early-stopped):

```
python tools/train_yolo.py --data datasets/combined_v2/data.yaml \
    --model models/balloon_drone_v1/best.pt --epochs 40 --patience 15 \
    --batch 32 --workers 4 --cache disk --name balloon_drone_v2
```

Results on the sky tests, at the Pi's 640x480 input (`tools/eval_sky.py`):

| model | conf | real recall | real image-hit | clip frame-hit | false alarms |
|-------|------|-------------|----------------|----------------|--------------|
| v1 | 0.40 | 0.888 | 42/43 | 56/277 (20%) | 0 |
| v2 | 0.40 | 0.986 | 43/43 (100%) | 167/277 (60%) | 0 |
| v2 | 0.25 | 0.986 | 43/43 (100%) | 201/277 (73%) | 0 |

v2 finds a balloon in every real sky photo and tracks the clip far more
often, with zero false alarms on either sky set. The low-altitude clip
hit-rate matters most: it is the closest thing to flight footage.

Background false-alarm sweep on the combined test set
(`tools/threshold_sweep.py`): at conf 0.25-0.40, only 2 of 57 background
images (3.5%) get any false box, and the drone class holds (P 0.956 /
R 0.925). So the threshold can be dropped for more balloon range without
new false locks.

**Deploy: v2 at conf 0.30.** Exported and parity-checked:

```
python tools/export_ncnn.py --weights runs/lockon/balloon_drone_v2/weights/best.pt \
    --out models/balloon_drone_v2 --sizes 640x480 416x320
python tools/check_ncnn_parity.py --weights models/balloon_drone_v2/best.pt \
    --ncnn models/balloon_drone_v2/ncnn_640x480 --images datasets/sky_test_real/test/images --count 43
```

Parity: 164/165 PyTorch detections reproduced by the full Pi path at
IoU >= 0.9, mean confidence diff 0.001. Use
`--model-path models/balloon_drone_v2/ncnn_640x480` on the Pi; set the
detector confidence to 0.30.

## Biggest remaining gap: your own balloon footage

Almost none of the public balloon images look like what the Pi camera will
see: a single party balloon against open sky, from a moving plane, often
small and motion-blurred. Before relying on the detector in flight:

1. Record Camera Module 3 clips of your actual test balloon at a range of
   distances and lighting conditions, ideally from the plane on a pass.
2. Label a few hundred frames (any YOLO-format annotation tool), spread
   across clips rather than consecutive frames.
3. Add them as another `--balloon-dataset` folder, splitting by clip (not by
   frame) into train/valid/test, and fine-tune from this run's `best.pt`.

4. Retrain or fine-tune, then re-export with `tools/export_ncnn.py` and
   re-run `tools/check_ncnn_parity.py` before copying to the Pi.

Your footage will also show whether the 0.40 confidence threshold and the
"lock on the first detection" behavior are right for real sky scenes.
With a ~5% per-image false-alarm rate on bird/aircraft photos, requiring
the same target in 2-3 consecutive frames before locking would cut
single-frame false locks a lot, at the cost of ~0.1-0.2 s extra
acquisition time. That is not implemented; decide after seeing real
footage.
