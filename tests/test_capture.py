import numpy as np

from lockon.vision.capture import CameraCapture, StubDetector


class FakeCameraBackend:
    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution
        self.started = False
        self.stopped_count = 0
        self.frames_served = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped_count += 1
        self.started = False

    def capture_array(self):
        self.frames_served += 1
        w, h = self.resolution
        # Deterministic-but-distinct frame content so shape/dtype checks
        # are meaningful without needing a real camera.
        return np.full((h, w, 3), fill_value=self.frames_served % 256, dtype=np.uint8)


def test_start_is_idempotent():
    backend = FakeCameraBackend()
    cap = CameraCapture(backend=backend)
    cap.start()
    cap.start()
    assert backend.started is True


def test_read_one_returns_correct_shape_and_dtype():
    backend = FakeCameraBackend(resolution=(640, 480))
    cap = CameraCapture(resolution=(640, 480), backend=backend)
    frame = cap.read_one()
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_frames_generator_yields_requested_count():
    backend = FakeCameraBackend()
    cap = CameraCapture(backend=backend)
    frames = list(cap.frames(count=5))
    assert len(frames) == 5
    assert backend.frames_served == 5


def test_measure_fps_reports_a_positive_number():
    backend = FakeCameraBackend()
    cap = CameraCapture(backend=backend)
    fps = cap.measure_fps(num_frames=10)
    assert fps > 0
    assert backend.frames_served == 10


def test_context_manager_starts_and_stops():
    backend = FakeCameraBackend()
    with CameraCapture(backend=backend) as cap:
        assert backend.started is True
        cap.read_one()
    assert backend.stopped_count == 1


def test_stub_detector_always_returns_no_detections():
    detector = StubDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


# -- frame timestamps and camera model ------------------------------------
import time  # noqa: E402

import pytest  # noqa: E402

from lockon.vision.capture import exposure_time_from_metadata  # noqa: E402


class FakeRequest:
    def __init__(self, frame, metadata):
        self._frame, self._metadata = frame, metadata
        self.released = False

    def make_array(self, name):
        assert name == "main"
        return self._frame

    def get_metadata(self):
        return self._metadata

    def release(self):
        self.released = True


class FakeRequestBackend(FakeCameraBackend):
    def __init__(self, model="imx708", age_s=0.03, crop=(576, 0, 3456, 2592)):
        super().__init__()
        self.camera_properties = {"Model": model}
        self.age_s, self.crop = age_s, crop
        self.requests = []

    def capture_request(self):
        md = {"SensorTimestamp": int((time.monotonic() - self.age_s) * 1e9), "ExposureTime": 10000,
              "ScalerCrop": self.crop}
        req = FakeRequest(self.capture_array(), md)
        self.requests.append(req)
        return req


def test_exposure_time_uses_sensor_timestamp_plus_half_exposure():
    t = exposure_time_from_metadata({"SensorTimestamp": 10_000_000_000, "ExposureTime": 8000}, arrival_time=10.05)
    assert t == pytest.approx(10.004)


def test_exposure_time_falls_back_when_missing_or_implausible():
    assert exposure_time_from_metadata({}, 5.0) == 5.0
    assert exposure_time_from_metadata({"SensorTimestamp": 1_000_000_000}, 5.0) == 5.0   # 4 s old
    assert exposure_time_from_metadata({"SensorTimestamp": 6_000_000_000}, 5.0) == 5.0   # in the future


def test_read_timed_uses_requests_and_releases_them():
    backend = FakeRequestBackend(age_s=0.03)
    cap = CameraCapture(backend=backend)
    frame, t = cap.read_timed()
    assert frame.shape == (480, 640, 3)
    assert backend.requests[0].released
    assert 0.02 < time.monotonic() - t < 0.1


def test_read_timed_without_requests_uses_arrival_time():
    cap = CameraCapture(backend=FakeCameraBackend())
    frame, t = cap.read_timed()
    assert abs(time.monotonic() - t) < 0.05


def test_camera_model_picks_lens_and_crop():
    cap = CameraCapture(backend=FakeRequestBackend(model="imx708_wide"))
    cap.read_timed()
    assert cap.camera_model().f_px == pytest.approx(364, abs=1)
    cap = CameraCapture(backend=FakeRequestBackend(model="imx708", crop=(0, 0, 4608, 2592)))
    cap.read_timed()
    assert cap.camera_model().f_px == pytest.approx(470, abs=1)
