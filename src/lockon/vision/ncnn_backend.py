"""Real NCNN backend for MLDetector. Needs the `ncnn` package (Pi:
requirements-pi.txt; it also installs on Windows/Linux x86 for testing).

Expects an Ultralytics NCNN export folder (`yolo export format=ncnn`),
e.g. `best_ncnn_model/` containing model.ncnn.param, model.ncnn.bin and
metadata.yaml. Verified against Ultralytics 8.4 YOLO11n exports:

* input/output blob names are read from the network (they are `in0` /
  `out0` in current exports - do not hardcode them),
* the single output is [4 + num_classes, num_boxes]: box cx, cy, w, h in
  input-pixel units, then per-class sigmoid scores (no objectness term),
* num_boxes is baked into a Reshape layer for the export's imgsz, so the
  model only accepts that exact square input size. The size and class
  names are read from metadata.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

NMS_IOU_THRESHOLD = 0.45
DEFAULT_SCORE_FLOOR = 0.05


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Class-agnostic NMS: one physical object should not come out as both
    a balloon and a drone."""
    order = scores.argsort()[::-1]
    keep: List[int] = []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0)
        order = rest[iou <= iou_threshold]
    return keep


def decode_output(output: np.ndarray, score_floor: float, iou_threshold: float = NMS_IOU_THRESHOLD
                  ) -> "list[tuple[float, float, float, float, float, int]]":
    """Turn the raw [4 + num_classes, num_boxes] output into NMS'd
    (x1, y1, x2, y2, confidence, class_id) tuples in input-pixel space."""
    output = np.asarray(output, dtype=np.float32)
    if output.ndim == 3:  # tolerate a leading batch dimension
        output = output[0]
    boxes_xywh = output[0:4, :].T
    class_scores = output[4:, :].T

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    mask = confidences >= score_floor
    if not mask.any():
        return []

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh[mask])
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    keep = _nms(boxes_xyxy, confidences, iou_threshold)
    results = []
    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i]
        results.append((float(x1), float(y1), float(x2), float(y2), float(confidences[i]), int(class_ids[i])))
    return results


def resolve_model_files(model_path: "str | Path") -> "tuple[Path, Path, Optional[Path]]":
    """Accept either an export folder or a path prefix to <prefix>.param /
    <prefix>.bin. Returns (param, bin, metadata.yaml or None)."""
    p = Path(model_path)
    if p.is_dir():
        param, binary, meta = p / "model.ncnn.param", p / "model.ncnn.bin", p / "metadata.yaml"
    else:
        param, binary = Path(f"{p}.param"), Path(f"{p}.bin")
        meta = p.parent / "metadata.yaml"
    for f in (param, binary):
        if not f.exists():
            raise FileNotFoundError(f"NCNN model file not found: {f}")
    return param, binary, (meta if meta.exists() else None)


def read_metadata(meta_path: Optional[Path]) -> "tuple[Optional[Tuple[int, int]], Optional[List[str]]]":
    """Return (input size as (w, h), class names) from an Ultralytics
    metadata.yaml, or (None, None) if absent."""
    if meta_path is None:
        return None, None
    import yaml

    data = yaml.safe_load(meta_path.read_text())
    size = None
    if data.get("imgsz"):
        h, w = data["imgsz"] if isinstance(data["imgsz"], (list, tuple)) else (data["imgsz"], data["imgsz"])
        size = (int(w), int(h))
    names = None
    if isinstance(data.get("names"), dict):
        names = [data["names"][k] for k in sorted(data["names"], key=int)]
    elif isinstance(data.get("names"), list):
        names = list(data["names"])
    return size, names


class NcnnYoloBackend:
    def __init__(
        self,
        model_path: "str | Path",
        input_size: Optional[Tuple[int, int]] = None,
        score_floor: float = DEFAULT_SCORE_FLOOR,
        num_threads: int = 4,
    ):
        import ncnn  # type: ignore

        param, binary, meta = resolve_model_files(model_path)
        meta_size, meta_names = read_metadata(meta)
        if input_size is not None and meta_size is not None and tuple(input_size) != tuple(meta_size):
            raise ValueError(
                f"input_size {input_size} does not match the size this model was exported at {meta_size}; "
                "NCNN exports only accept their export size. Re-export, or pass input_size=None."
            )
        self.input_size = tuple(input_size or meta_size or (640, 640))
        self.class_names: Optional[List[str]] = meta_names
        self.score_floor = score_floor

        self._net = ncnn.Net()
        self._net.opt.use_vulkan_compute = False
        self._net.opt.num_threads = num_threads
        self._net.load_param(str(param))
        self._net.load_model(str(binary))
        self._input_name = self._net.input_names()[0]
        self._output_name = self._net.output_names()[0]

    def infer(self, letterboxed_rgb: np.ndarray) -> "list[tuple[float, float, float, float, float, int]]":
        import ncnn  # type: ignore

        h, w = letterboxed_rgb.shape[:2]
        if (w, h) != self.input_size:
            raise ValueError(f"expected a {self.input_size} letterboxed frame, got {(w, h)}")
        blob = np.ascontiguousarray(letterboxed_rgb.transpose(2, 0, 1), dtype=np.float32) * (1.0 / 255.0)
        with self._net.create_extractor() as ex:
            ex.input(self._input_name, ncnn.Mat(blob))
            _, mat_out = ex.extract(self._output_name)
            output = np.array(mat_out)
        return decode_output(output, self.score_floor)
