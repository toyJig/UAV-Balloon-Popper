"""Task 6 - camera capture pipeline.

Wraps Picamera2 behind a small interface so the rest of the pipeline (and
all unit tests) never touch the camera library directly - tests inject a
fake backend instead. Real camera access only happens on the Pi, where
picamera2 is installed via apt (see requirements-pi.txt).
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Mapping, Optional, Protocol, Tuple

import numpy as np

from .camera_model import CameraModel
from .types import Detection

log = logging.getLogger(__name__)

# A frame older than this on arrival means the timestamp isn't on our clock.
MAX_PLAUSIBLE_FRAME_AGE_S = 0.5


def exposure_time_from_metadata(metadata: Mapping, arrival_time: float) -> float:
    """Exposure time of a frame on the time.monotonic clock.

    libcamera's SensorTimestamp (ns, CLOCK_MONOTONIC on the Pi) marks the
    start of the first row's exposure; half the exposure time is added to
    get closer to the middle of it. Falls back to the arrival time when the
    timestamp is missing or implausible."""
    ts_ns = metadata.get("SensorTimestamp") if metadata else None
    if ts_ns is None:
        return arrival_time
    t = ts_ns / 1e9 + metadata.get("ExposureTime", 0) / 2e6
    if 0.0 <= arrival_time - t <= MAX_PLAUSIBLE_FRAME_AGE_S:
        return t
    return arrival_time


class CameraBackend(Protocol):
    """The subset of Picamera2's interface this module needs."""

    def start(self) -> None: ...

    def capture_array(self) -> np.ndarray: ...

    def stop(self) -> None: ...


def _make_picamera2_backend(resolution: Tuple[int, int], framerate: int) -> CameraBackend:
    try:
        from picamera2 import Picamera2  # type: ignore
    except ImportError as exc:  # pragma: no cover - only hit off the Pi
        raise RuntimeError(
            "picamera2 is not installed. This module must run on the "
            "Raspberry Pi with 'sudo apt install python3-picamera2' done "
            "first (see requirements-pi.txt). For development/tests off "
            "the Pi, pass an explicit `backend=` instead."
        ) from exc

    picam2 = Picamera2()
    # libcamera names formats by packed-word order, so Picamera2's "BGR888"
    # yields numpy arrays in R,G,B byte order (and "RGB888" yields B,G,R).
    # Everything downstream (detectors, overlay) expects RGB arrays.
    config = picam2.create_video_configuration(
        main={"size": resolution, "format": "BGR888"},
        controls={"FrameRate": framerate},
    )
    picam2.configure(config)
    return picam2


class CameraCapture:
    """Yields RGB frames as numpy arrays (H, W, 3) at roughly the requested
    framerate, measuring and reporting the actually-achieved fps."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        framerate: int = 30,
        backend: Optional[CameraBackend] = None,
    ):
        self.resolution = resolution
        self.framerate = framerate
        self._backend = backend or _make_picamera2_backend(resolution, framerate)
        self._started = False
        self.last_metadata: Mapping = {}

    def start(self) -> None:
        if not self._started:
            self._backend.start()
            self._started = True

    def stop(self) -> None:
        if self._started:
            self._backend.stop()
            self._started = False

    def read_one(self) -> np.ndarray:
        self.start()
        return self._backend.capture_array()

    def frames(self, count: Optional[int] = None) -> Iterator[np.ndarray]:
        """Yield frames forever (count=None) or exactly `count` frames."""
        self.start()
        n = 0
        while count is None or n < count:
            yield self._backend.capture_array()
            n += 1

    def read_timed(self) -> "tuple[np.ndarray, float]":
        """One frame plus its exposure time (time.monotonic clock)."""
        self.start()
        backend = self._backend
        if hasattr(backend, "capture_request"):
            request = backend.capture_request()
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
            finally:
                request.release()
            self.last_metadata = metadata
            return frame, exposure_time_from_metadata(metadata, time.monotonic())
        frame = backend.capture_array()
        return frame, time.monotonic()

    def timed_frames(self, count: Optional[int] = None) -> Iterator["tuple[np.ndarray, float]"]:
        n = 0
        while count is None or n < count:
            yield self.read_timed()
            n += 1

    def camera_model(self, mount_pitch_deg: float = 0.0) -> CameraModel:
        """Geometry for the attached Camera Module 3 (standard or wide lens,
        from the sensor's model name) and the sensor crop in use (from the
        last frame's ScalerCrop). Call after at least one read_timed()."""
        props = getattr(self._backend, "camera_properties", None) or {}
        model = str(props.get("Model", ""))
        if model and not model.startswith("imx708"):
            log.warning("camera %r is not a Camera Module 3 (imx708); using Module 3 optics anyway", model)
        crop = self.last_metadata.get("ScalerCrop") if self.last_metadata else None
        w, h = self.resolution
        return CameraModel.for_module3(width=w, height=h, wide="wide" in model,
                                       crop_width_sensor_px=crop[2] if crop else None,
                                       mount_pitch_deg=mount_pitch_deg)

    def measure_fps(self, num_frames: int = 60) -> float:
        self.start()
        start_time = time.monotonic()
        for _ in range(num_frames):
            self._backend.capture_array()
        elapsed = time.monotonic() - start_time
        return num_frames / elapsed if elapsed > 0 else float("inf")

    def __enter__(self) -> "CameraCapture":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()


class StubDetector:
    """Trivial detector used only to validate the capture pipeline in
    isolation (Task 6) before any real detection logic (Tasks 7/8) exists."""

    def detect(self, frame: np.ndarray) -> "list[Detection]":
        return []
