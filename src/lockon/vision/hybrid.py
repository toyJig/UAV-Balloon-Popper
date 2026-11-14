"""Task 9 - hybrid target acquisition: ML detector primary, color/motion
fallback only once the ML detector has found nothing for several
consecutive frames, and a tracker bridging single-frame gaps so a brief
miss from both detectors doesn't instantly drop the lock.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np

from .selection import select_target
from .tracker import BoxTracker
from .types import Detection, DetectionSource

DEFAULT_MAX_MISSED_ML_FRAMES = 5
# Consecutive frames the tracker may stand in for the detectors before the
# lock is dropped: about 0.5-0.8 s at the 12-20 fps the detector runs at.
DEFAULT_MAX_BRIDGE_FRAMES = 10


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> "list[Detection]": ...


class HybridDetector:
    def __init__(
        self,
        ml_detector: Detector,
        fallback_detector: Detector,
        frame_size: Tuple[int, int],
        max_missed_ml_frames: int = DEFAULT_MAX_MISSED_ML_FRAMES,
        tracker: Optional[BoxTracker] = None,
        max_bridge_frames: int = DEFAULT_MAX_BRIDGE_FRAMES,
    ):
        self._ml = ml_detector
        self._fallback = fallback_detector
        self._frame_size = frame_size
        self._max_missed_ml_frames = max_missed_ml_frames
        self._max_bridge_frames = max_bridge_frames
        self._tracker = tracker if tracker is not None else BoxTracker()
        self._missed_ml_frames = 0
        self._bridged_frames = 0
        self._last_detection: Optional[Detection] = None

    @property
    def missed_ml_frames(self) -> int:
        return self._missed_ml_frames

    def reset(self) -> None:
        self._missed_ml_frames = 0
        self._bridged_frames = 0
        self._last_detection = None
        self._tracker.stop()

    def process(self, frame: np.ndarray) -> Optional[Detection]:
        ml_detections = self._ml.detect(frame)
        if ml_detections:
            self._missed_ml_frames = 0
            return self._lock_onto(select_target(ml_detections, self._frame_size), frame)

        self._missed_ml_frames += 1

        if self._missed_ml_frames >= self._max_missed_ml_frames:
            fallback_detections = self._fallback.detect(frame)
            target = select_target(fallback_detections, self._frame_size)
            if target is not None:
                return self._lock_onto(target, frame)

        return self._bridge_with_tracker(frame)

    def _lock_onto(self, target: Detection, frame: np.ndarray) -> Detection:
        self._last_detection = target
        self._bridged_frames = 0
        self._tracker.start(frame, (target.x, target.y, target.w, target.h))
        return target

    def _bridge_with_tracker(self, frame: np.ndarray) -> Optional[Detection]:
        if not self._tracker.is_active or self._last_detection is None:
            return None
        # A tracker will happily keep following whatever patch it was started
        # on (including empty sky after a one-frame false detection), so it may
        # only cover short detector gaps. Drop the lock once no detector has
        # re-confirmed the target for max_bridge_frames frames.
        if self._bridged_frames >= self._max_bridge_frames:
            self._tracker.stop()
            self._last_detection = None
            return None
        self._bridged_frames += 1
        bbox = self._tracker.update(frame)
        if bbox is None:
            self._last_detection = None
            return None
        x, y, w, h = bbox
        bridged = Detection(
            x=x, y=y, w=w, h=h,
            confidence=self._last_detection.confidence,
            source=self._last_detection.source,
            class_name=self._last_detection.class_name,
        )
        self._last_detection = bridged
        return bridged
