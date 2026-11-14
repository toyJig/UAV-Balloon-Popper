#!/usr/bin/env python3
"""Check a held-out test folder for near-duplicates in a training folder
(same dHash method and threshold as tools/dedup_splits.py). Reports only.

Usage:
    python tools/check_leak.py --test datasets/sky_test_real/test/images --train datasets/combined_v2/train/images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_splits import IMAGE_SUFFIXES, REPEAT_RE, hash_all, nearest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--max-distance", type=int, default=24)
    parser.add_argument("--train-prefix", nargs="*", default=[])
    args = parser.parse_args()

    test = sorted(p for p in args.test.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    train = sorted(p for p in args.train.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                   and not REPEAT_RE.search(p.stem)
                   and (not args.train_prefix or any(p.name.startswith(x) for x in args.train_prefix)))
    tp, th = hash_all(test)
    rp, rh = hash_all(train)
    d, idx = nearest(th, rh)
    print(f"test={len(tp)} train={len(rp)}; nearest distance min/median: {int(d.min())}/{int(np.median(d))}")
    leaks = [(int(d[i]), tp[i].name, rp[idx[i]].name) for i in np.argsort(d) if d[i] <= args.max_distance]
    print(f"near-duplicates (<= {args.max_distance}): {len(leaks)}")
    for dist, a, b in leaks[:20]:
        print(f"  d={dist} {a} ~ {b}")


if __name__ == "__main__":
    main()
