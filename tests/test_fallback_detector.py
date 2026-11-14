import cv2
import numpy as np

from lockon.vision.fallback_detector import ColorMotionDetector
from lockon.vision.types import DetectionSource

SKY_RGB = (130, 170, 230)
BALLOON_RGB = (230, 40, 30)  # bright red/orange, within default HSV range


def make_frame(center, radius=40, size=(480, 640)):
    h, w = size
    frame = np.full((h, w, 3), SKY_RGB, dtype=np.uint8)
    cv2.circle(frame, center, radius, BALLOON_RGB, thickness=-1)
    return frame


def test_detects_a_balloon_near_its_true_location():
    detector = ColorMotionDetector()
    frame = make_frame(center=(320, 240), radius=40)

    detections = detector.detect(frame)

    assert len(detections) == 1
    d = detections[0]
    cx, cy = d.center
    assert abs(cx - 320) < 10
    assert abs(cy - 240) < 10
    assert d.source == DetectionSource.FALLBACK


def test_no_detection_on_empty_sky():
    detector = ColorMotionDetector()
    frame = np.full((480, 640, 3), SKY_RGB, dtype=np.uint8)

    detections = detector.detect(frame)

    assert detections == []


def test_filters_out_specks_smaller_than_min_area():
    detector = ColorMotionDetector(min_area_px=500)
    frame = make_frame(center=(320, 240), radius=5)  # tiny speck, area ~78px

    detections = detector.detect(frame)

    assert detections == []


def test_moving_balloon_gets_a_confidence_bonus_over_a_static_one():
    static_detector = ColorMotionDetector()
    static_detector.detect(make_frame(center=(320, 240)))
    static_result = static_detector.detect(make_frame(center=(320, 240)))  # same spot, no motion

    moving_detector = ColorMotionDetector()
    moving_detector.detect(make_frame(center=(320, 240)))
    moving_result = moving_detector.detect(make_frame(center=(420, 240)))  # jumped -> motion

    assert static_result[0].confidence < moving_result[0].confidence


def test_largest_blob_is_returned_first_when_multiple_present():
    detector = ColorMotionDetector()
    frame = np.full((480, 640, 3), SKY_RGB, dtype=np.uint8)
    cv2.circle(frame, (150, 150), 20, BALLOON_RGB, thickness=-1)   # smaller
    cv2.circle(frame, (450, 300), 45, BALLOON_RGB, thickness=-1)   # larger

    detections = detector.detect(frame)

    assert len(detections) == 2
    assert detections[0].area > detections[1].area
    cx, cy = detections[0].center
    assert abs(cx - 450) < 10 and abs(cy - 300) < 10


def test_reset_clears_motion_history():
    detector = ColorMotionDetector()
    detector.detect(make_frame(center=(320, 240)))
    detector.reset()
    assert detector._prev_gray is None
