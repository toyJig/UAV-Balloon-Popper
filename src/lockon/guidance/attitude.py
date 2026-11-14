"""Aircraft attitude from INAV, in standard aerospace conventions, and
gyro-based extrapolation to the moment a camera frame was exposed.

Conventions used everywhere in guidance code (body axes x forward, y right,
z down; level frame North-East-Down rotated so "north" = current heading):
    roll_deg   positive = right wing down (banked right)
    pitch_deg  positive = nose up
    yaw_deg    heading, positive = clockwise (turning right)
    p, q, r    body rates deg/s: roll right +, pitch nose-up +, yaw right +

INAV sends roll positive right-wing-down and pitch positive nose-DOWN in
MSP_ATTITUDE (its angle controller compares stick-derived targets, where
forward stick = nose down = positive, directly against attitude). Gyro
signs are converted the same way. ``InavSigns`` holds the conversions so a
bench check (tools/check_attitude.py) can confirm or correct them without
touching guidance code.

Why extrapolate: a frame is exposed some tens of milliseconds before the Pi
reads INAV's attitude. At the roll rates a fast plane manoeuvres with
(200 deg/s and more), 40 ms is 8 degrees of bank - enough to misplace the
target in the level frame. Using the gyro rates read at the same time, the
attitude is wound back to the exposure time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Attitude:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass(frozen=True)
class BodyRates:
    p_dps: float
    q_dps: float
    r_dps: float


@dataclass(frozen=True)
class InavSigns:
    roll: int = 1
    pitch: int = -1   # INAV pitch positive = nose down
    yaw: int = 1
    gyro_roll: int = 1
    gyro_pitch: int = -1
    gyro_yaw: int = 1


def from_inav(raw, signs: InavSigns = InavSigns()) -> Attitude:
    """``raw`` is a codec.RawAttitude."""
    return Attitude(
        roll_deg=signs.roll * raw.roll_decideg / 10.0,
        pitch_deg=signs.pitch * raw.pitch_decideg / 10.0,
        yaw_deg=(signs.yaw * raw.yaw_deg) % 360.0,
    )


def rates_from_inav(gyro_dps, signs: InavSigns = InavSigns()) -> BodyRates:
    gx, gy, gz = gyro_dps
    return BodyRates(signs.gyro_roll * gx, signs.gyro_pitch * gy, signs.gyro_yaw * gz)


def euler_rates(att: Attitude, rates: BodyRates) -> "tuple[float, float, float]":
    """d(roll, pitch, yaw)/dt in deg/s from body rates (standard kinematics)."""
    phi, theta = math.radians(att.roll_deg), math.radians(att.pitch_deg)
    p, q, r = rates.p_dps, rates.q_dps, rates.r_dps
    cos_t = math.cos(theta)
    if abs(cos_t) < 1e-3:  # near vertical: heading is undefined, don't extrapolate it
        cos_t = math.copysign(1e-3, cos_t)
    roll_dot = p + (q * math.sin(phi) + r * math.cos(phi)) * math.tan(theta)
    pitch_dot = q * math.cos(phi) - r * math.sin(phi)
    yaw_dot = (q * math.sin(phi) + r * math.cos(phi)) / cos_t
    return roll_dot, pitch_dot, yaw_dot


def extrapolate(att: Attitude, rates: BodyRates, dt_s: float) -> Attitude:
    """Attitude ``dt_s`` seconds later (negative = earlier), assuming
    constant body rates over that short interval."""
    rd, pd, yd = euler_rates(att, rates)
    pitch = max(-90.0, min(90.0, att.pitch_deg + pd * dt_s))
    roll = (att.roll_deg + rd * dt_s + 180.0) % 360.0 - 180.0
    return Attitude(roll, pitch, (att.yaw_deg + yd * dt_s) % 360.0)
