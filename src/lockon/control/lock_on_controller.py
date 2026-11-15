"""Task 11 - wires the hybrid detector, intercept guidance and the
altitude guard into the flight controller, implementing the lock-on state
machine (state_machine.py).

Per frame while the lock-on switch is on and a target is detected:

    pixel (u, v)
      + INAV attitude (MSP_ATTITUDE) wound back to the frame's exposure time
        with the gyro rates (MSP_RAW_IMU)
      -> CameraModel.line_of_sight: azimuth/elevation in the level frame
      -> InterceptGuidance: bank and pitch targets (pursuit + PN)
      -> AltitudeGuard: dive limit / recovery climb above the 15 ft floor
      -> InavLockOnLink.write: INAV's angle controller flies the targets

Output goes through INAV logic conditions: the flight controller only
applies the Pi's targets while the pilot's switch is on and the Pi's
heartbeat is fresh. While nothing is locked the state is "off", so the
pilot's own sticks fly the plane.

Throttle is never touched here; it stays on the pilot's stick.

Every step must reach ``link.write`` (it carries the heartbeat), so the
loop must run at least every 300 ms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..guidance.altitude_clamp import AltitudeGuard
from ..guidance.attitude import Attitude, BodyRates, InavSigns, euler_rates, extrapolate, from_inav, rates_from_inav
from ..guidance.intercept import GuidanceCommand, InterceptGuidance, SpeedEstimator
from ..msp.client import MSPClient
from ..msp.inav_logic import STATE_ENGAGED, STATE_ENGAGED_LOCKED, STATE_OFF, InavLockOnLink
from ..vision.camera_model import CameraModel
from ..vision.hybrid import HybridDetector
from ..vision.types import Detection, DetectionSource
from .state_machine import LockState

LOCK_SWITCH_INDEX = 6  # receiver-order index of the lock-on switch: CH7
# INAV's HIGH operation is "> 1666 us"; use the same threshold so the Pi and
# the flight controller agree on when the switch is on.
SWITCH_ON_THRESHOLD_US = 1667

AUTO_LEVEL_DURATION_S = 1.0
# Attitude is wound back by at most this much; an older frame means the
# loop is badly behind and extrapolating further would add more error.
MAX_EXTRAPOLATION_S = 0.15


@dataclass(frozen=True)
class StepResult:
    state: LockState
    altitude_boosting: bool
    bank_deg: float = 0.0
    pitch_deg: float = 0.0
    attitude: Optional[Attitude] = None
    command: Optional[GuidanceCommand] = None
    altitude_ft: Optional[float] = None
    detection: Optional[Detection] = None


class LockOnController:
    def __init__(
        self,
        msp_client: MSPClient,
        link: InavLockOnLink,
        hybrid_detector: HybridDetector,
        camera: CameraModel,
        guidance: InterceptGuidance,
        altitude_guard: AltitudeGuard,
        lock_switch_index: int = LOCK_SWITCH_INDEX,
        signs: InavSigns = InavSigns(),
        speed_estimator: Optional[SpeedEstimator] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._msp = msp_client
        self._link = link
        self._hybrid = hybrid_detector
        self._camera = camera
        self._guidance = guidance
        self._guard = altitude_guard
        self._lock_switch_index = lock_switch_index
        self._signs = signs
        self._speed = speed_estimator if speed_estimator is not None else SpeedEstimator(guidance.p.speed_mps)
        self._clock = clock

        self._state = LockState.DISENGAGED
        self._auto_level_deadline: Optional[float] = None
        self._last_speed_t: Optional[float] = None

    @property
    def state(self) -> LockState:
        return self._state

    def step(self, frame: np.ndarray, frame_time: float, now: float) -> StepResult:
        """``frame_time``: when the frame was exposed, on the same clock as
        ``clock`` (time.monotonic). ``now``: current time on that clock."""
        pilot_channels = self._msp.read_rc()
        switch_on = pilot_channels[self._lock_switch_index] >= SWITCH_ON_THRESHOLD_US

        if not switch_on:
            self._disengage()
            self._link.write(STATE_OFF)
            return StepResult(state=self._state, altitude_boosting=False)

        detection = self._hybrid.process(frame)

        if detection is not None:
            return self._fly_toward(detection, frame_time, now)

        if self._state.is_locked:
            self._enter_auto_level(now)

        if self._state == LockState.AUTO_LEVELING:
            return self._auto_level(now)

        self._state = LockState.SEARCHING
        self._link.write(STATE_OFF)
        return StepResult(state=self._state, altitude_boosting=False)

    def shutdown(self) -> None:
        """Hand control back explicitly when the program exits."""
        self._disengage()
        self._link.release()

    # -- internal helpers --------------------------------------------
    def _disengage(self) -> None:
        self._state = LockState.DISENGAGED
        self._auto_level_deadline = None
        self._guidance.reset()
        self._guard.reset()
        self._hybrid.reset()

    def _enter_auto_level(self, now: float) -> None:
        self._state = LockState.AUTO_LEVELING
        self._auto_level_deadline = now + AUTO_LEVEL_DURATION_S
        self._guidance.reset()

    def _read_altitude(self):
        alt = self._msp.read_altitude()
        return alt.altitude_ft, alt.vario_cm_s / 30.48

    def _read_attitude(self, frame_time: float) -> "tuple[Attitude, Attitude, BodyRates]":
        """Returns (attitude now, attitude wound back to the frame time,
        body rates). The 'now' attitude heading is what the command will be
        flown relative to; the frame-time one removes tilt from the pixel."""
        raw = self._msp.read_attitude()
        gyro = self._msp.read_gyro_dps()
        read_time = self._clock()
        att_now = from_inav(raw, self._signs)
        rates = rates_from_inav(gyro, self._signs)
        dt = max(-MAX_EXTRAPOLATION_S, min(0.0, frame_time - read_time))
        return att_now, extrapolate(att_now, rates, dt), rates

    def _update_speed(self, att: Attitude, rates: BodyRates, now: float) -> None:
        _, _, yaw_rate = euler_rates(att, rates)
        dt = 0.0 if self._last_speed_t is None else now - self._last_speed_t
        self._last_speed_t = now
        self._guidance.speed_mps = self._speed.update(att.roll_deg, att.pitch_deg, yaw_rate, dt)

    def _guarded(self, pitch_deg: float, bank_deg: float):
        altitude_ft, climb_ftps = self._read_altitude()
        result = self._guard.update(altitude_ft, climb_ftps, pitch_deg, bank_deg,
                                    self._guidance.speed_mps)
        return result, altitude_ft

    def _auto_level(self, now: float) -> StepResult:
        # wings level, nose at the cruise angle of attack unless the floor
        # demands a climb; indicator off
        guard, altitude_ft = self._guarded(self._guidance.p.aoa_deg, 0.0)
        self._link.write(STATE_ENGAGED, roll_deg=guard.bank_deg, pitch_deg=guard.pitch_deg)

        if now >= self._auto_level_deadline:
            self._state = LockState.SEARCHING
            self._auto_level_deadline = None

        return StepResult(self._state, guard.boosting, guard.bank_deg, guard.pitch_deg,
                          altitude_ft=altitude_ft)

    def _fly_toward(self, detection, frame_time: float, now: float) -> StepResult:
        self._auto_level_deadline = None
        self._state = LockState.LOCKED_ML if detection.source == DetectionSource.ML else LockState.LOCKED_FALLBACK

        att_now, att, rates = self._read_attitude(frame_time)
        self._update_speed(att_now, rates, now)
        u, v = detection.center
        los = self._camera.line_of_sight(u, v, att)
        cmd = self._guidance.update(los, frame_time, now=now, heading_now_deg=att_now.yaw_deg)
        guard, altitude_ft = self._guarded(cmd.pitch_deg, cmd.bank_deg)

        self._link.write(STATE_ENGAGED_LOCKED, roll_deg=guard.bank_deg, pitch_deg=guard.pitch_deg)
        return StepResult(self._state, guard.boosting, guard.bank_deg, guard.pitch_deg, att, cmd, altitude_ft,
                          detection=detection)
