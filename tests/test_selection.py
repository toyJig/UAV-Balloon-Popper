from lockon.vision.selection import select_target
from lockon.vision.types import Detection, DetectionSource


def det(x, y, w, h, conf):
    return Detection(x=x, y=y, w=w, h=h, confidence=conf, source=DetectionSource.ML)


def test_returns_none_for_empty_list():
    assert select_target([], frame_size=(640, 480)) is None


def test_picks_the_only_candidate():
    d = det(300, 220, 40, 40, 0.9)
    assert select_target([d], frame_size=(640, 480)) is d


def test_prefers_centered_target_over_more_confident_edge_target():
    centered = det(300, 220, 40, 40, 0.5)   # near center (320, 240)
    edge = det(0, 0, 20, 20, 0.6)           # corner, slightly higher raw confidence

    result = select_target([centered, edge], frame_size=(640, 480))

    assert result is centered


def test_prefers_much_more_confident_target_even_if_less_centered():
    centered_low_conf = det(300, 220, 40, 40, 0.1)
    edge_high_conf = det(500, 400, 40, 40, 0.95)

    result = select_target([centered_low_conf, edge_high_conf], frame_size=(640, 480))

    assert result is edge_high_conf


def test_two_identical_candidates_returns_the_first():
    a = det(300, 220, 40, 40, 0.8)
    b = det(300, 220, 40, 40, 0.8)
    assert select_target([a, b], frame_size=(640, 480)) is a
