"""Task 8 - ML detector wrapper (YOLOv8n/YOLOv11n exported to NCNN).

The exact raw-tensor decode for a given exported model (anchor-free head
layout, NMS) is owned by the backend (``NcnnYoloBackend``, built from the
real .param/.bin files on the Pi) rather than reimplemented here, since
that layout is specific to the exact model/export version actually trained
in Task 8 and shouldn't be guessed at. What *is* implemented and unit
tested here, independent of any real model, is everything this project
controls regardless of backend: letterbox preprocessing, unscaling boxes
back to original frame coordinates, confidence filtering, and wrapping
results into our common ``Detection`` type. See
docs/runbook-task08-ml-detector.md for the dataset/training/export
procedure.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .types import Detection, DetectionSource

DEFAULT_INPUT_SIZE = (640, 640)
# 0.30 chosen for the v2 model on sky footage: max balloon range with no
# new background false alarms (see docs/runbook-task08-ml-detector.md).
DEFAULT_CONF_THRESHOLD = 0.30
DEFAULT_CLASS_NAMES = ("balloon", "drone")
LETTERBOX_FILL = 114  # standard YOLO letterbox gray fill value


class YoloBackend(Protocol):
    """Contract for anything that can run the model. The real
    implementation (NcnnYoloBackend) loads the exported .param/.bin files
    and performs the architecture-specific output decode + NMS; a fake
    implementing just this method is used in tests."""

    def infer(self, letterboxed_rgb: np.ndarray) -> "Sequence[tuple[float, float, float, float, float, int]]":
        """Return detections as (x1, y1, x2, y2, confidence, class_id) in
        the pixel space of ``letterboxed_rgb`` (i.e. the square input
        tensor), already NMS'd. Confidence filtering against the caller's
        own threshold happens in MLDetector, not here."""
        ...


def letterbox(frame: np.ndarray, size: Tuple[int, int]) -> "tuple[np.ndarray, float, float, float]":
    """Resize+pad `frame` to a square `size` canvas, preserving aspect
    ratio (standard YOLO letterbox). Returns (canvas, scale, pad_x, pad_y)."""
    import cv2

    h, w = frame.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0
    canvas = np.full((target_h, target_w, 3), LETTERBOX_FILL, dtype=np.uint8)
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, scale, pad_x, pad_y


def unletterbox_box(
    box: "tuple[float, float, float, float]", scale: float, pad_x: float, pad_y: float
) -> "tuple[float, float, float, float]":
    """Map an (x1, y1, x2, y2) box from letterboxed-canvas coordinates back
    to the original frame's coordinates."""
    x1, y1, x2, y2 = box
    return (
        (x1 - pad_x) / scale,
        (y1 - pad_y) / scale,
        (x2 - pad_x) / scale,
        (y2 - pad_y) / scale,
    )


def _make_ncnn_backend(model_path: str, input_size: Tuple[int, int]) -> YoloBackend:
    try:
        import ncnn  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only hit off the Pi
        raise RuntimeError(
            "ncnn is not installed. Install it on the Pi per "
            "requirements-pi.txt, or pass an explicit `backend=` for "
            "development/tests off the Pi."
        ) from exc
    from .ncnn_backend import NcnnYoloBackend  # local import: only needed when ncnn is present

    return NcnnYoloBackend(model_path, input_size)


class MLDetector:
    """``model_path`` is an Ultralytics NCNN export folder (or a
    <prefix> of <prefix>.param/.bin). When ``input_size`` / ``class_names``
    are left as None they come from the model's metadata.yaml, falling back
    to 640x640 and (balloon, drone)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: Optional[YoloBackend] = None,
        input_size: Optional[Tuple[int, int]] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        class_names: Optional[Sequence[str]] = None,
    ):
        if backend is None and model_path is None:
            raise ValueError("MLDetector needs either model_path (real NCNN model) or backend (for tests)")
        self._backend = backend or _make_ncnn_backend(model_path, input_size)  # type: ignore[arg-type]
        self.input_size = tuple(input_size or getattr(self._backend, "input_size", None) or DEFAULT_INPUT_SIZE)
        self.conf_threshold = conf_threshold
        self.class_names = list(class_names or getattr(self._backend, "class_names", None) or DEFAULT_CLASS_NAMES)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        frame_h, frame_w = frame.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(frame, self.input_size)
        raw = self._backend.infer(canvas)

        detections: List[Detection] = []
        for x1, y1, x2, y2, conf, cls_id in raw:
            if conf < self.conf_threshold:
                continue
            ox1, oy1, ox2, oy2 = unletterbox_box((x1, y1, x2, y2), scale, pad_x, pad_y)
            # boxes may spill into the letterbox padding; keep them inside the frame
            ox1, ox2 = min(max(ox1, 0.0), frame_w), min(max(ox2, 0.0), frame_w)
            oy1, oy2 = min(max(oy1, 0.0), frame_h), min(max(oy2, 0.0), frame_h)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            class_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class_{cls_id}"
            detections.append(
                Detection(
                    x=ox1, y=oy1, w=max(0.0, ox2 - ox1), h=max(0.0, oy2 - oy1),
                    confidence=float(conf),
                    source=DetectionSource.ML,
                    class_name=class_name,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
