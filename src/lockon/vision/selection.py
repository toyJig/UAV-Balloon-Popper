"""Task 9 - target selection when multiple detections are present in one
frame. Pure function, no hardware/model dependency.

Score = confidence, weighted down by how far the detection's center is
from the frame center (closer to boresight wins ties and near-ties over
a merely-more-confident detection near the edge - useful since the plane
should generally commit to whatever it's already roughly pointed at
rather than snapping to a newly-appeared, higher-confidence target at the
edge of frame, which would fight the guidance law in Task 10).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .types import Detection


def select_target(detections: List[Detection], frame_size: Tuple[int, int]) -> Optional[Detection]:
    if not detections:
        return None

    frame_w, frame_h = frame_size
    center_x, center_y = frame_w / 2.0, frame_h / 2.0
    max_dist = math.hypot(center_x, center_y) or 1.0  # center-to-corner distance; avoid /0 on a 0x0 frame

    best: Optional[Detection] = None
    best_score = -1.0
    for d in detections:
        dx, dy = d.center
        dist = math.hypot(dx - center_x, dy - center_y)
        normalized_dist = min(1.0, dist / max_dist)
        score = d.confidence * (1.0 - normalized_dist)
        if score > best_score:
            best_score = score
            best = d
    return best
