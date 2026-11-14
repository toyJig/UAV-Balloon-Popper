"""Task 10 - pursuit guidance law: maps a target's pixel position to
roll/pitch commands, with a predictive lead term based on the target's
estimated screen-space velocity. Pure function/class, no hardware
dependency - not yet wired to the MSP client (that happens in Task 11).

Output is a normalized command in [-1, 1] per axis (positive roll = target
is to the right of boresight -> turn right; positive pitch = target is
*above* boresight -> climb toward it, negative = dive toward it below -
i.e. pitch_cmd uses the same "positive = climb" convention as
AltitudeFloorClamp's climb_cmd, since Task 11 feeds this value straight
into that clamp). The integration layer maps both axes to actual
microsecond stick values with the sign/scale confirmed on the bench in
Task 2.

Guidance is a simple PD controller on pixel error, computed not against
the target's raw current position but against a "lead point" projected
ahead along its estimated velocity - classic proportional-navigation-style
lead, appropriate for a target moving roughly at constant velocity over
the short time horizon of an attack run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

DEFAULT_KP = 1.0
DEFAULT_KD = 0.3
DEFAULT_LEAD_TIME_S = 0.3
DEFAULT_MAX_CMD = 1.0


@dataclass(frozen=True)
class GuidanceCommand:
    roll_cmd: float
    pitch_cmd: float
    lead_point: Tuple[float, float]


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class PursuitGuidance:
    def __init__(
        self,
        frame_size: Tuple[int, int],
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        lead_time_s: float = DEFAULT_LEAD_TIME_S,
        max_cmd: float = DEFAULT_MAX_CMD,
    ):
        self.frame_size = frame_size
        self.kp = kp
        self.kd = kd
        self.lead_time_s = lead_time_s
        self.max_cmd = max_cmd
        self._prev_center: Optional[Tuple[float, float]] = None
        self._prev_error: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._prev_center = None
        self._prev_error = None

    def update(self, target_center_px: Tuple[float, float], dt: float) -> GuidanceCommand:
        if dt <= 0:
            raise ValueError("dt must be > 0")

        frame_w, frame_h = self.frame_size
        boresight = (frame_w / 2.0, frame_h / 2.0)

        velocity = self._estimate_velocity(target_center_px, dt)
        lead_point = (
            target_center_px[0] + velocity[0] * self.lead_time_s,
            target_center_px[1] + velocity[1] * self.lead_time_s,
        )

        error_x = (lead_point[0] - boresight[0]) / (frame_w / 2.0)
        # Image y grows downward, but pitch_cmd uses "positive = climb"
        # (see class docstring), so this axis is inverted relative to x.
        error_y = (boresight[1] - lead_point[1]) / (frame_h / 2.0)

        d_error_x, d_error_y = self._derivative((error_x, error_y), dt)

        roll_cmd = _clip(self.kp * error_x + self.kd * d_error_x, self.max_cmd)
        pitch_cmd = _clip(self.kp * error_y + self.kd * d_error_y, self.max_cmd)

        self._prev_center = target_center_px
        self._prev_error = (error_x, error_y)

        return GuidanceCommand(roll_cmd=roll_cmd, pitch_cmd=pitch_cmd, lead_point=lead_point)

    def _estimate_velocity(self, center: Tuple[float, float], dt: float) -> Tuple[float, float]:
        if self._prev_center is None:
            return (0.0, 0.0)
        return (
            (center[0] - self._prev_center[0]) / dt,
            (center[1] - self._prev_center[1]) / dt,
        )

    def _derivative(self, error: Tuple[float, float], dt: float) -> Tuple[float, float]:
        if self._prev_error is None:
            return (0.0, 0.0)
        return (
            (error[0] - self._prev_error[0]) / dt,
            (error[1] - self._prev_error[1]) / dt,
        )
