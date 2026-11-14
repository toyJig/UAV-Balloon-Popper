import numpy as np
import pytest

from lockon.vision.ml_detector import MLDetector, letterbox, unletterbox_box
from lockon.vision.types import DetectionSource


def test_letterbox_pads_a_wider_than_tall_frame_to_square():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 4:3, wider than tall
    canvas, scale, pad_x, pad_y = letterbox(frame, (640, 640))

    assert canvas.shape == (640, 640, 3)
    assert scale == pytest.approx(1.0)
    assert pad_x == pytest.approx(0.0)
    assert pad_y == pytest.approx(80.0)  # (640-480)/2


def test_unletterbox_box_inverts_the_padding_and_scale():
    box_in_canvas = (100.0, 130.0, 200.0, 230.0)
    original = unletterbox_box(box_in_canvas, scale=1.0, pad_x=0.0, pad_y=80.0)
    assert original == pytest.approx((100.0, 50.0, 200.0, 150.0))


def test_unletterbox_box_accounts_for_downscaling():
    # a 1280x960 frame letterboxed into 640x640 -> scale 0.5, pad_y=0, pad_x=0
    box_in_canvas = (100.0, 100.0, 300.0, 300.0)
    original = unletterbox_box(box_in_canvas, scale=0.5, pad_x=0.0, pad_y=0.0)
    assert original == pytest.approx((200.0, 200.0, 600.0, 600.0))


class FakeYoloBackend:
    def __init__(self, raw_detections):
        self.raw_detections = raw_detections
        self.last_input_shape = None

    def infer(self, letterboxed_rgb):
        self.last_input_shape = letterboxed_rgb.shape
        return self.raw_detections


def test_ml_detector_filters_by_confidence_threshold():
    backend = FakeYoloBackend([
        (100, 130, 200, 230, 0.9, 0),   # above threshold
        (300, 130, 400, 230, 0.1, 0),   # below threshold -> dropped
    ])
    detector = MLDetector(backend=backend, input_size=(640, 640), conf_threshold=0.4)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9)


def test_ml_detector_maps_class_id_to_class_name():
    backend = FakeYoloBackend([(100, 130, 200, 230, 0.8, 1)])
    detector = MLDetector(backend=backend, class_names=("balloon", "drone"))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert detections[0].class_name == "drone"
    assert detections[0].source == DetectionSource.ML


def test_ml_detector_unscales_boxes_back_to_original_frame_coordinates():
    # 480x640 frame -> scale 1.0, pad_y=80 as established above.
    backend = FakeYoloBackend([(100, 130, 200, 230, 0.8, 0)])
    detector = MLDetector(backend=backend, input_size=(640, 640))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    d = detections[0]
    assert (d.x, d.y, d.w, d.h) == pytest.approx((100.0, 50.0, 100.0, 100.0))


def test_ml_detector_sorts_by_confidence_descending():
    backend = FakeYoloBackend([
        (0, 0, 10, 10, 0.5, 0),
        (0, 0, 10, 10, 0.95, 0),
        (0, 0, 10, 10, 0.6, 0),
    ])
    detector = MLDetector(backend=backend, conf_threshold=0.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    confidences = [d.confidence for d in detections]
    assert confidences == sorted(confidences, reverse=True)


def test_ml_detector_clips_boxes_that_spill_into_the_letterbox_padding():
    # 480x640 frame -> pad_y=80; a box from canvas y=40..200 starts inside the padding
    backend = FakeYoloBackend([(100, 40, 200, 200, 0.8, 0)])
    detector = MLDetector(backend=backend, input_size=(640, 640))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    d = detector.detect(frame)[0]

    assert d.y == pytest.approx(0.0)
    assert d.y + d.h == pytest.approx(120.0)


def test_ml_detector_takes_input_size_and_class_names_from_the_backend():
    # 480x640 into 416x416 -> pad_y=52, so keep the box below the padding
    backend = FakeYoloBackend([(10, 100, 50, 150, 0.9, 0)])
    backend.input_size = (416, 416)
    backend.class_names = ["kite"]
    detector = MLDetector(backend=backend)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert detector.input_size == (416, 416)
    assert backend.last_input_shape == (416, 416, 3)
    assert detections[0].class_name == "kite"


def test_requires_either_model_path_or_backend():
    with pytest.raises(ValueError):
        MLDetector()
