"""Tests for the pure numpy helpers in ncnn_backend.py (box math + NMS).
These don't need the `ncnn` package itself, which is Pi-only - only the
NcnnYoloBackend class's __init__/infer methods (which import ncnn lazily)
require real hardware."""

import numpy as np

import pytest

from lockon.vision.ncnn_backend import _nms, _xywh_to_xyxy, decode_output, read_metadata, resolve_model_files


def test_decode_output_picks_best_class_and_suppresses_overlaps():
    # 3 candidate boxes, 2 classes: rows = cx, cy, w, h, score_balloon, score_drone
    output = np.array([
        [100.0, 102.0, 400.0],
        [100.0, 100.0, 300.0],
        [40.0, 40.0, 20.0],
        [40.0, 40.0, 20.0],
        [0.90, 0.60, 0.01],   # balloon scores
        [0.10, 0.20, 0.70],   # drone scores
    ])

    results = decode_output(output, score_floor=0.05)

    # box 1 overlaps box 0 heavily and scores lower -> suppressed; box 2 is a drone elsewhere
    assert len(results) == 2
    x1, y1, x2, y2, conf, cls = results[0]
    assert (x1, y1, x2, y2) == (80.0, 80.0, 120.0, 120.0)
    assert (round(conf, 2), cls) == (0.90, 0)
    assert results[1][5] == 1


def test_decode_output_accepts_a_leading_batch_dimension_and_empty_results():
    output = np.zeros((1, 6, 10))
    assert decode_output(output, score_floor=0.05) == []


def test_resolve_model_files_accepts_an_export_folder(tmp_path):
    for name in ("model.ncnn.param", "model.ncnn.bin"):
        (tmp_path / name).write_text("x")
    (tmp_path / "metadata.yaml").write_text("imgsz:\n- 416\n- 416\nnames:\n  0: balloon\n  1: drone\n")

    param, binary, meta = resolve_model_files(tmp_path)

    assert param.name == "model.ncnn.param" and binary.name == "model.ncnn.bin"
    assert read_metadata(meta) == ((416, 416), ["balloon", "drone"])


def test_resolve_model_files_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model_files(tmp_path / "nope")


def test_xywh_to_xyxy_conversion():
    boxes = np.array([[100.0, 100.0, 40.0, 20.0]])  # cx, cy, w, h
    xyxy = _xywh_to_xyxy(boxes)
    assert xyxy[0].tolist() == [80.0, 90.0, 120.0, 110.0]


def test_nms_keeps_highest_scoring_of_two_overlapping_boxes():
    boxes = np.array([
        [0.0, 0.0, 100.0, 100.0],
        [5.0, 5.0, 105.0, 105.0],  # heavily overlapping with the first
    ])
    scores = np.array([0.6, 0.9])

    keep = _nms(boxes, scores, iou_threshold=0.45)

    assert keep == [1]


def test_nms_keeps_both_non_overlapping_boxes():
    boxes = np.array([
        [0.0, 0.0, 50.0, 50.0],
        [500.0, 500.0, 550.0, 550.0],
    ])
    scores = np.array([0.8, 0.7])

    keep = _nms(boxes, scores, iou_threshold=0.45)

    assert set(keep) == {0, 1}
