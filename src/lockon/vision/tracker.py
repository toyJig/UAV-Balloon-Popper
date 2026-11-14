"""Task 9 - lightweight single-object tracker used to bridge gaps between
detector passes (e.g. one blurred frame) without immediately dropping
lock. CSRT/KCF live in opencv-contrib, not the base opencv package, and
their exact API location has moved around across OpenCV versions (plain
`cv2.TrackerCSRT_create`, then `cv2.legacy.TrackerCSRT_create`) - this
module tries the known locations and raises an informative error only
when actually asked to construct a real one, so unit tests can inject a
fake backend instead and never need opencv-contrib at all.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]  # x, y, w, h


class TrackerBackend(Protocol):
    def init(self, frame: np.ndarray, bbox: BBox) -> bool: ...

    def update(self, frame: np.ndarray) -> "tuple[bool, BBox]": ...


def _make_cv2_tracker() -> TrackerBackend:
    import cv2

    for factory_name in ("TrackerCSRT_create", "TrackerKCF_create"):
        factory = getattr(cv2, factory_name, None)
        if factory is not None:
            return factory()
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None:
            factory = getattr(legacy, factory_name, None)
            if factory is not None:
                return factory()
    raise RuntimeError(
        "No CSRT/KCF tracker available in this OpenCV build. Install "
        "opencv-contrib-python (pip) instead of the base opencv package, "
        "see requirements-pi.txt."
    )


class BoxTracker:
    """Wraps a single-object tracker backend behind init()/update(), with
    the backend construction deferred to start() (not __init__) so a
    fresh backend is created for each new lock, matching how OpenCV
    trackers are meant to be used (one instance per track)."""

    def __init__(self, backend_factory: Optional[Callable[[], TrackerBackend]] = None):
        self._backend_factory = backend_factory or _make_cv2_tracker
        self._backend: Optional[TrackerBackend] = None

    @property
    def is_active(self) -> bool:
        return self._backend is not None

    def start(self, frame: np.ndarray, bbox: BBox) -> None:
        self._backend = self._backend_factory()
        self._backend.init(frame, bbox)

    def stop(self) -> None:
        self._backend = None

    def update(self, frame: np.ndarray) -> Optional[BBox]:
        if self._backend is None:
            return None
        ok, bbox = self._backend.update(frame)
        if not ok:
            self._backend = None
            return None
        return bbox
