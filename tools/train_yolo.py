#!/usr/bin/env python3
"""Task 8 - fine-tune a YOLO11n detector on the combined dataset produced
by dataset_prep.py. Run on a laptop/desktop GPU, NOT on the Pi - training
is far too slow on the Pi 5's CPU; the Pi only ever runs inference on the
exported model (export_ncnn.py).

Defaults are sized for an 8 GB laptop GPU (RTX 4070 Laptop): YOLO11n at
640px, batch 32.

Usage:
    python tools/train_yolo.py --data datasets/combined/data.yaml
    python tools/train_yolo.py --data datasets/combined/data.yaml --resume runs/lockon/balloon_drone_v1/weights/last.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def make_trainer_class():
    """DetectionTrainer whose validation loader uses the same number of
    workers as the training loader, instead of Ultralytics' default of
    twice as many. Both loaders keep their workers alive between epochs,
    and on Windows each worker is a separate process, so the default
    roughly doubles the RAM the dataloaders hold for the whole run."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class EqualWorkersTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "val":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            saved = self.args.workers
            self.args.workers = max(1, saved // 2)  # the parent method doubles this for val
            try:
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = saved

    return EqualWorkersTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to data.yaml from dataset_prep.py")
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained base weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20, help="stop early after this many epochs without val improvement")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    # Each worker is a separate process on Windows (~0.6-0.9 GB each); the
    # train and val loaders each get this many (see make_trainer_class).
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    # disk: decoded images saved once as .npy next to each image (~0.9 MB each),
    # so workers skip JPEG decoding every epoch
    parser.add_argument("--cache", default="disk", choices=["false", "ram", "disk"])
    parser.add_argument("--project", default="runs/lockon")
    parser.add_argument("--name", default="balloon_drone_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None, help="path to last.pt to resume an interrupted run")
    # For continuing a finished run from its last.pt: a lower starting rate and
    # no warmup avoid the accuracy dip a full-rate restart causes.
    parser.add_argument("--optimizer", default=None, help="e.g. SGD (default: Ultralytics 'auto')")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    args = parser.parse_args()
    extra = {k: v for k, v in (("optimizer", args.optimizer), ("lr0", args.lr0),
                               ("warmup_epochs", args.warmup_epochs)) if v is not None}

    from ultralytics import YOLO  # imported lazily: only needed for training, not for inference on the Pi

    trainer_cls = make_trainer_class()
    if args.resume:
        YOLO(args.resume).train(resume=True, trainer=trainer_cls)
        return

    # Absolute project path: Ultralytics nests relative paths under runs/detect/.
    project = Path(args.project).resolve()
    model = YOLO(args.model)
    model.train(
        trainer=trainer_cls,
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=False if args.cache == "false" else args.cache,
        project=str(project),
        name=args.name,
        seed=args.seed,
        exist_ok=False,
        plots=True,
        **extra,
    )
    print(f"\nBest weights: {project / args.name / 'weights' / 'best.pt'}")
    print("Next: python tools/export_ncnn.py --weights <best.pt path>")


if __name__ == "__main__":
    main()
