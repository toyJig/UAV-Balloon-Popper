import numpy as np

from lockon.vision.hybrid import HybridDetector
from lockon.vision.tracker import BoxTracker
from lockon.vision.types import Detection, DetectionSource

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FRAME_SIZE = (640, 480)


def det(source, x=300, y=220, w=40, h=40, conf=0.8):
    return Detection(x=x, y=y, w=w, h=h, confidence=conf, source=source)


class ScriptedDetector:
    """Returns a pre-programmed sequence of detection lists, one per call."""

    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.calls = 0

    def detect(self, frame):
        result = self._sequence[self.calls] if self.calls < len(self._sequence) else []
        self.calls += 1
        return result


class FakeTrackerBackend:
    """A tracker that always reports a fixed box, shifted slightly each
    update() call, until told to fail."""

    def __init__(self):
        self.fail_next = False
        self._bbox = None

    def init(self, frame, bbox):
        self._bbox = bbox
        return True

    def update(self, frame):
        if self.fail_next:
            return False, (0, 0, 0, 0)
        x, y, w, h = self._bbox
        self._bbox = (x + 1, y, w, h)  # drift by 1px so bridged frames are distinguishable
        return True, self._bbox


def make_hybrid(ml_sequence, fallback_sequence, max_missed_ml_frames=3, tracker_backend=None):
    ml = ScriptedDetector(ml_sequence)
    fallback = ScriptedDetector(fallback_sequence)
    tracker = BoxTracker(backend_factory=lambda: tracker_backend or FakeTrackerBackend())
    hybrid = HybridDetector(ml, fallback, FRAME_SIZE, max_missed_ml_frames=max_missed_ml_frames, tracker=tracker)
    return hybrid, ml, fallback


def test_ml_detection_is_used_directly_when_present():
    hybrid, ml, fallback = make_hybrid(ml_sequence=[[det(DetectionSource.ML)]], fallback_sequence=[[]])
    result = hybrid.process(FRAME)
    assert result.source == DetectionSource.ML
    assert hybrid.missed_ml_frames == 0


def test_falls_back_only_after_max_missed_ml_frames():
    ml_seq = [[], [], [], []]  # ML never finds anything
    # missed_ml_frames reaches 3 (the threshold) on the 3rd processed frame
    # (index 2), so fallback.detect() is first called there, then again on
    # frame index 3 - these two calls use fallback_seq[0] and [1].
    fallback_seq = [[], [det(DetectionSource.FALLBACK)]]
    hybrid, ml, fallback = make_hybrid(ml_seq, fallback_seq, max_missed_ml_frames=3)

    assert hybrid.process(FRAME) is None
    assert fallback.calls == 0  # not consulted yet (1 missed ML frame)

    assert hybrid.process(FRAME) is None
    assert fallback.calls == 0  # still not consulted (2 missed ML frames)

    assert hybrid.process(FRAME) is None  # threshold just hit; fallback tried but found nothing this frame
    assert fallback.calls == 1

    result = hybrid.process(FRAME)
    assert result.source == DetectionSource.FALLBACK
    assert fallback.calls == 2


def test_tracker_bridges_a_single_frame_miss_after_a_lock():
    ml_seq = [[det(DetectionSource.ML, x=300)], []]  # locks, then ML misses next frame
    fallback_seq = [[], []]
    hybrid, ml, fallback = make_hybrid(ml_seq, fallback_seq, max_missed_ml_frames=5)

    first = hybrid.process(FRAME)
    second = hybrid.process(FRAME)

    assert first.source == DetectionSource.ML
    assert second is not None
    assert second.source == DetectionSource.ML  # bridged box keeps the source it was last actually detected with
    assert second.x == first.x + 1  # drifted via the fake tracker, proving the tracker (not a stale detection) produced this


def test_returns_none_when_nothing_ever_locked():
    hybrid, ml, fallback = make_hybrid(ml_sequence=[[]], fallback_sequence=[[]], max_missed_ml_frames=5)
    assert hybrid.process(FRAME) is None


def test_lock_is_dropped_once_tracker_also_fails():
    ml_seq = [[det(DetectionSource.ML)], [], []]
    fallback_seq = [[], [], []]
    tracker_backend = FakeTrackerBackend()
    hybrid, ml, fallback = make_hybrid(ml_seq, fallback_seq, max_missed_ml_frames=10, tracker_backend=tracker_backend)

    hybrid.process(FRAME)  # lock
    hybrid.process(FRAME)  # bridged ok
    tracker_backend.fail_next = True
    result = hybrid.process(FRAME)  # tracker now fails too

    assert result is None


def test_tracker_alone_cannot_hold_a_lock_past_max_bridge_frames():
    # one detection (e.g. a single-frame false positive), then the detectors
    # never confirm it again while the tracker keeps reporting success
    ml_seq = [[det(DetectionSource.ML)]] + [[]] * 10
    hybrid, ml, fallback = make_hybrid(ml_seq, fallback_sequence=[[]] * 11, max_missed_ml_frames=100)
    hybrid._max_bridge_frames = 3

    results = [hybrid.process(FRAME) for _ in range(6)]

    assert results[0] is not None            # the detection itself
    assert all(r is not None for r in results[1:4])  # 3 bridged frames allowed
    assert results[4] is None and results[5] is None  # then the lock is dropped


def test_a_fresh_detection_resets_the_bridge_budget():
    d = det(DetectionSource.ML)
    ml_seq = [[d], [], [], [d], [], [], []]
    hybrid, ml, fallback = make_hybrid(ml_seq, fallback_sequence=[[]] * 7, max_missed_ml_frames=100)
    hybrid._max_bridge_frames = 2

    results = [hybrid.process(FRAME) for _ in range(7)]

    assert all(r is not None for r in results[:6])  # detect, 2 bridged, detect, 2 bridged
    assert results[6] is None


def test_reset_clears_state():
    hybrid, ml, fallback = make_hybrid(ml_sequence=[[det(DetectionSource.ML)]], fallback_sequence=[[]])
    hybrid.process(FRAME)
    hybrid.reset()
    assert hybrid.missed_ml_frames == 0
    assert hybrid._last_detection is None
    assert hybrid._tracker.is_active is False


def test_selects_among_multiple_ml_candidates():
    near_center = det(DetectionSource.ML, x=300, y=220, conf=0.5)
    at_edge = det(DetectionSource.ML, x=0, y=0, conf=0.55)
    hybrid, ml, fallback = make_hybrid(ml_sequence=[[near_center, at_edge]], fallback_sequence=[[]])

    result = hybrid.process(FRAME)

    assert result.x == near_center.x and result.y == near_center.y
