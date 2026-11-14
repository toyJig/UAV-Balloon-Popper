"""Task 7 - non-ML color/motion fallback detector.

Used only when the ML detector (Task 8) finds nothing for several
consecutive frames. Tuned for a single bright, saturated balloon against
open sky: an HSV color-threshold mask finds balloon-colored blobs, and a
frame-to-frame motion mask (absolute difference against the previous
frame) is used as a confidence bonus for blobs that are also moving -
useful for telling a moving drone/balloon apart from a static
similarly-colored patch of background, without requiring motion (a
stationary balloon against a static sky background must still be found).

Default HSV range targets a bright red/orange balloon; retune
`lower_hsv`/`upper_hsv` for your actual balloon color during Task 7
bench testing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .types import Detection, DetectionSource

DEFAULT_LOWER_HSV = (0, 120, 100)   # bright red/orange, low end (wraps hue=0)
DEFAULT_UPPER_HSV = (15, 255, 255)
DEFAULT_MIN_AREA_PX = 150
MOTION_DIFF_THRESHOLD = 25
MOTION_CONFIDENCE_BONUS = 0.25
BASE_CONFIDENCE = 0.5


class ColorMotionDetector:
    def __init__(
        self,
        lower_hsv: Tuple[int, int, int] = DEFAULT_LOWER_HSV,
        upper_hsv: Tuple[int, int, int] = DEFAULT_UPPER_HSV,
        min_area_px: int = DEFAULT_MIN_AREA_PX,
    ):
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.min_area_px = min_area_px
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        color_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        motion_mask = self._motion_mask(gray)
        self._prev_gray = gray

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)

            moving = bool(motion_mask[y:y + h, x:x + w].any())
            confidence = min(1.0, BASE_CONFIDENCE + (MOTION_CONFIDENCE_BONUS if moving else 0.0))

            detections.append(
                Detection(
                    x=float(x), y=float(y), w=float(w), h=float(h),
                    confidence=confidence,
                    source=DetectionSource.FALLBACK,
                    class_name="balloon",
                )
            )

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    def _motion_mask(self, gray: np.ndarray) -> np.ndarray:
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            return np.zeros_like(gray, dtype=bool)
        diff = cv2.absdiff(gray, self._prev_gray)
        return diff > MOTION_DIFF_THRESHOLD
