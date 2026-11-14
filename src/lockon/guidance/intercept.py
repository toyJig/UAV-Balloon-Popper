"""Intercept guidance: line of sight -> target bank and pitch angles for
INAV's angle controller. Replaces the pixel-proportional control law.

Inputs each frame: the target's line of sight in the level frame (after
removing roll/pitch with the camera model and the FC's attitude), and the
frame's exposure time. Two terms per axis:

* pursuit: steer the flight path toward the target's current direction,
* proportional navigation (PN): add N x the line-of-sight rotation rate.
  A stationary or slowly moving target seen from a fast plane has a LOS
  rate that grows as the plane gets close; PN removes it early, so the
  plane flies a near-straight collision course instead of curving in late.
  The LOS rate is measured in the level frame (inertial azimuth = heading +
  azimuth), so the plane's own turning - which INAV's gyro tracks in its
  attitude estimate - does not show up as target motion.

Lateral output: a turn rate, converted to bank with the coordinated-turn
relation  bank = atan(V * turn_rate / g). ``speed_mps`` is the assumed
airspeed (no pitot/GPS on board); errors in it only scale the lateral gain,
and the camera loop corrects what's left.

Vertical output: a pitch target = LOS elevation (+PN lead) + the angle of
attack the wing flies at, because INAV holds the NOSE angle while the
flight path sits AoA below it.

Speeds for this airframe (AWRAS 1.0 performance analysis, 1.2-1.3 kg):
stall ~10 m/s, cruise 14-16 m/s, max level 30-36 m/s at a 70 % throttle
cap. With the Pi added (~1.4 kg) and possibly more throttle, the defaults
assume a 30 m/s attack run; tools/intercept_sim.py checks 15-45 m/s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..vision.camera_model import LineOfSight

G = 9.81


@dataclass(frozen=True)
class GuidanceParams:
    speed_mps: float = 30.0
    nav_constant: float = 3.0       # PN gain N (3-4 is standard)
    pursuit_time_s: float = 1.0     # time constant for nulling heading error
    pitch_lead_s: float = 0.6       # vertical PN lead: pitch += N * el_rate * this / N
    aoa_deg: float = 3.0            # nose above flight path at attack speed
    max_bank_deg: float = 45.0      # INAV max_angle_inclination_rll
    max_pitch_deg: float = 30.0     # INAV max_angle_inclination_pit
    rate_alpha: float = 0.35        # low-pass weight for new LOS-rate samples (pixel noise)
    max_los_rate_dps: float = 90.0
    max_dt_s: float = 0.5           # longer gaps restart the rate estimate
    # extra time from computing a command to INAV flying it (MSP writes +
    # waiting for the next 10 Hz logic-condition tick, ~50 ms on average)
    command_delay_s: float = 0.05


@dataclass(frozen=True)
class GuidanceCommand:
    bank_deg: float
    pitch_deg: float
    az_deg: float
    el_deg: float
    az_rate_dps: float
    el_rate_dps: float
    turn_rate_dps: float


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def bank_for_turn_rate(turn_rate_dps: float, speed_mps: float) -> float:
    return math.degrees(math.atan(speed_mps * math.radians(turn_rate_dps) / G))


class SpeedEstimator:
    """Airspeed estimate from the FC's own sensors, no pitot or GPS.

    In a banked, level, coordinated turn  heading_rate = g tan(bank) / V,
    so V = g tan(bank) / heading_rate. INAV's attitude gives the bank and
    the gyro (via the Euler-rate kinematics) gives the heading rate. Only
    samples with a clear bank and turn in the same direction are used, and
    the estimate moves slowly (``tau_s``) and stays inside [min, max], so
    a manoeuvre that breaks the level-turn assumption can't swing it far.
    Until the first good sample it reports ``default_mps``.
    """

    def __init__(self, default_mps: float = 30.0, min_mps: float = 15.0, max_mps: float = 45.0,
                 min_bank_deg: float = 15.0, min_rate_dps: float = 4.0, max_climb_deg: float = 10.0,
                 tau_s: float = 2.0):
        self.default_mps = default_mps
        self.min_mps, self.max_mps = min_mps, max_mps
        self.min_bank_deg, self.min_rate_dps = min_bank_deg, min_rate_dps
        self.max_climb_deg = max_climb_deg
        self.tau_s = tau_s
        self.speed_mps = default_mps
        self.samples = 0

    def update(self, bank_deg: float, pitch_deg: float, heading_rate_dps: float, dt_s: float) -> float:
        usable = (abs(bank_deg) >= self.min_bank_deg and abs(bank_deg) <= 70.0
                  and abs(heading_rate_dps) >= self.min_rate_dps
                  and (bank_deg > 0) == (heading_rate_dps > 0)
                  and abs(pitch_deg) <= self.max_climb_deg + 5.0
                  and 0 < dt_s <= 0.5)
        if usable:
            raw = G * math.tan(math.radians(abs(bank_deg))) / math.radians(abs(heading_rate_dps))
            raw = max(self.min_mps, min(self.max_mps, raw))
            if self.samples == 0:
                self.speed_mps = raw
            else:
                self.speed_mps += min(1.0, dt_s / self.tau_s) * (raw - self.speed_mps)
            self.samples += 1
        return self.speed_mps


class InterceptGuidance:
    def __init__(self, params: GuidanceParams = GuidanceParams()):
        self.p = params
        self.speed_mps = params.speed_mps  # may be updated from a SpeedEstimator
        self.reset()

    def reset(self) -> None:
        self._last_t: Optional[float] = None
        self._last_az: Optional[float] = None
        self._last_el: Optional[float] = None
        self._az_rate = 0.0
        self._el_rate = 0.0

    def _update_rates(self, inertial_az: float, el: float, t: float) -> None:
        if self._last_t is not None:
            dt = t - self._last_t
            if 0 < dt <= self.p.max_dt_s:
                lim = self.p.max_los_rate_dps
                raw_az = max(-lim, min(lim, _wrap180(inertial_az - self._last_az) / dt))
                raw_el = max(-lim, min(lim, (el - self._last_el) / dt))
                a = self.p.rate_alpha
                self._az_rate += a * (raw_az - self._az_rate)
                self._el_rate += a * (raw_el - self._el_rate)
            elif dt > self.p.max_dt_s:
                self._az_rate = self._el_rate = 0.0
        self._last_t, self._last_az, self._last_el = t, inertial_az, el

    def update(self, los: LineOfSight, t: float, now: Optional[float] = None,
               heading_now_deg: Optional[float] = None) -> GuidanceCommand:
        """``los`` measured at frame time ``t``. If the current time and the
        heading read from the FC after inference are given, the line of
        sight is predicted forward to when INAV will fly this command
        (``now`` + command_delay_s) and the azimuth is taken relative to
        the CURRENT heading, so detector latency doesn't cause overshoot."""
        p = self.p
        self._update_rates(los.inertial_az_deg, los.el_deg, t)

        az, el = los.az_deg, los.el_deg
        if now is not None:
            horizon = max(0.0, now - t) + p.command_delay_s
            heading = los.heading_deg if heading_now_deg is None else heading_now_deg
            az = _wrap180(los.inertial_az_deg + self._az_rate * horizon - heading)
            el = los.el_deg + self._el_rate * horizon

        turn_rate = az / p.pursuit_time_s + p.nav_constant * self._az_rate
        bank = max(-p.max_bank_deg, min(p.max_bank_deg, bank_for_turn_rate(turn_rate, self.speed_mps)))

        pitch = el + p.pitch_lead_s * p.nav_constant * self._el_rate + p.aoa_deg
        pitch = max(-p.max_pitch_deg, min(p.max_pitch_deg, pitch))
        return GuidanceCommand(bank, pitch, az, el, self._az_rate, self._el_rate, turn_rate)
