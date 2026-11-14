import numpy as np
import pytest

from fakes import FakeSerialPort

from lockon.control.lock_on_controller import LOCK_SWITCH_INDEX, LockOnController
from lockon.control.state_machine import LockState
from lockon.guidance.altitude_clamp import AltitudeGuard
from lockon.guidance.intercept import GuidanceParams, InterceptGuidance
from lockon.msp.client import CENTER_US, MSPClient
from lockon.msp.inav_logic import (
    STATE_ENGAGED, STATE_ENGAGED_LOCKED, STATE_OFF, InavLockOnLink, LockOnChainConfig, build_chain,
)
from lockon.vision.camera_model import CameraModel
from lockon.vision.types import Detection, DetectionSource

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
SWITCH_ON_US = 2000
SWITCH_OFF_US = 1000
SAFE_ALTITUDE_CM = 3000  # ~98 ft, well above the 15ft floor
CFG = LockOnChainConfig()
PARAMS = GuidanceParams()


class FakeHybrid:
    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.calls = 0
        self.reset_calls = 0

    def process(self, frame):
        result = self._sequence[self.calls] if self.calls < len(self._sequence) else None
        self.calls += 1
        return result

    def reset(self):
        self.reset_calls += 1


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def make_controller(hybrid_sequence, switch_us=SWITCH_ON_US):
    port = FakeSerialPort()
    port.rc_channels_to_report = [CENTER_US] * 12
    port.rc_channels_to_report[LOCK_SWITCH_INDEX] = switch_us
    port.altitude_cm_to_report = SAFE_ALTITUDE_CM
    port.load_chain(build_chain(CFG))
    client = MSPClient(port)
    link = InavLockOnLink(client, CFG)
    hybrid = FakeHybrid(hybrid_sequence)
    clock = FakeClock()
    controller = LockOnController(client, link, hybrid, CameraModel.for_module3(), InterceptGuidance(PARAMS),
                                  AltitudeGuard(), clock=clock)
    controller.clock = clock
    return controller, port, link, hybrid


def det(source=DetectionSource.ML, cx=320.0, cy=240.0, w=40, h=40, conf=0.9):
    """Detection centred on pixel (cx, cy)."""
    return Detection(x=cx - w / 2, y=cy - h / 2, w=w, h=h, confidence=conf, source=source)


def test_switch_off_is_disengaged_and_writes_state_off():
    controller, port, link, hybrid = make_controller([None], switch_us=SWITCH_OFF_US)

    result = controller.step(FRAME, frame_time=0.0, now=0.0)

    assert result.state == LockState.DISENGAGED
    assert link.last_written == (STATE_OFF, 0, 0)
    assert hybrid.calls == 0  # no vision work while disengaged


def test_switch_on_with_no_target_is_searching_and_does_not_engage():
    controller, port, link, hybrid = make_controller([None])

    result = controller.step(FRAME, frame_time=0.0, now=0.0)

    assert result.state == LockState.SEARCHING
    assert link.last_written[0] == STATE_OFF


def test_every_step_sends_a_heartbeat():
    controller, port, link, hybrid = make_controller([None, det(), None])
    for t in range(3):
        controller.step(FRAME, frame_time=t * 0.05, now=t * 0.05)
    heartbeats = [i for i, _ in port.lc_writes if i == 1]
    assert len(heartbeats) == 3


def test_target_at_center_with_level_plane_flies_straight_with_aoa_nose_up():
    controller, port, link, hybrid = make_controller([det()])

    result = controller.step(FRAME, frame_time=0.0, now=0.0)

    assert result.state == LockState.LOCKED_ML
    assert link.last_written == (STATE_ENGAGED_LOCKED, 0, round(PARAMS.aoa_deg))


def test_locked_target_right_of_center_banks_right_with_indicator_on():
    controller, port, link, hybrid = make_controller([det(cx=520)])

    controller.step(FRAME, frame_time=0.0, now=0.0)

    state, roll, pitch = link.last_written
    assert state == STATE_ENGAGED_LOCKED
    assert roll > 0


def test_target_left_of_center_banks_left():
    controller, port, link, hybrid = make_controller([det(cx=150)])
    controller.step(FRAME, frame_time=0.0, now=0.0)
    assert link.last_written[1] < 0


def test_target_above_center_pitches_nose_up():
    controller, port, link, hybrid = make_controller([det(cy=60)])

    controller.step(FRAME, frame_time=0.0, now=0.0)

    assert link.last_written[2] > PARAMS.aoa_deg  # positive = nose up in the link's convention


def test_nose_down_attitude_is_removed_from_the_line_of_sight():
    # plane pitched 10 deg nose down (INAV +100 decideg); target dead centre
    # in the image is therefore 10 deg BELOW the horizon
    controller, port, link, hybrid = make_controller([det()])
    port.attitude_to_report = (0, 100, 0)

    result = controller.step(FRAME, frame_time=0.0, now=0.0)

    assert result.attitude.pitch_deg == pytest.approx(-10.0)
    assert result.command.el_deg == pytest.approx(-10.0, abs=0.01)
    assert link.last_written[2] == round(-10.0 + PARAMS.aoa_deg)


def test_bank_is_removed_from_the_line_of_sight():
    # banked 30 deg right; a target straight ahead on the horizon appears
    # rotated in the image. Put the target where the horizon point lands.
    cam = CameraModel.for_module3()
    import math
    # level-frame direction az=10, el=0 seen from a 30 deg right bank:
    # body y = sin(az)cos(30), body z = -sin(az)sin(30)  (rotate level->body)
    az = math.radians(10)
    x, y_l = math.cos(az), math.sin(az)
    y_b = y_l * math.cos(math.radians(30))
    z_b = -y_l * math.sin(math.radians(30))
    u, v = cam.cx + cam.f_px * y_b / x, cam.cy + cam.f_px * z_b / x
    controller, port, link, hybrid = make_controller([det(cx=u, cy=v)])
    port.attitude_to_report = (300, 0, 0)

    result = controller.step(FRAME, frame_time=0.0, now=0.0)

    assert result.command.az_deg == pytest.approx(10.0, abs=0.01)
    assert result.command.el_deg == pytest.approx(0.0, abs=0.01)


def test_gyro_rate_winds_attitude_back_to_the_frame_time():
    controller, port, link, hybrid = make_controller([det()])
    port.attitude_to_report = (0, 0, 0)
    port.gyro_dps_to_report = (200, 0, 0)  # rolling right at 200 deg/s
    controller.clock.t = 1.10              # attitude read 100 ms after exposure

    result = controller.step(FRAME, frame_time=1.0, now=1.10)

    assert result.attitude.roll_deg == pytest.approx(-20.0)  # was 20 deg left at exposure


def test_extrapolation_is_limited_for_very_old_frames():
    controller, port, link, hybrid = make_controller([det()])
    port.gyro_dps_to_report = (200, 0, 0)
    controller.clock.t = 2.0

    result = controller.step(FRAME, frame_time=1.0, now=2.0)

    assert result.attitude.roll_deg == pytest.approx(-30.0)  # 0.15 s cap


def test_locked_on_fallback_target_reports_locked_fallback_state():
    controller, port, link, hybrid = make_controller([det(DetectionSource.FALLBACK)])

    result = controller.step(FRAME, frame_time=0.0, now=0.0)

    assert result.state == LockState.LOCKED_FALLBACK
    assert link.last_written[0] == STATE_ENGAGED_LOCKED


def test_lock_loss_auto_levels_with_indicator_off_then_returns_to_searching():
    controller, port, link, hybrid = make_controller([det(cx=520), None, None, None])

    assert controller.step(FRAME, frame_time=0.0, now=0.0).state == LockState.LOCKED_ML

    r2 = controller.step(FRAME, frame_time=0.1, now=0.1)  # lock just lost
    assert r2.state == LockState.AUTO_LEVELING
    # still engaged, wings level, nose at the cruise AoA, indicator off
    assert link.last_written == (STATE_ENGAGED, 0, round(PARAMS.aoa_deg))

    assert controller.step(FRAME, frame_time=0.5, now=0.5).state == LockState.AUTO_LEVELING
    assert controller.step(FRAME, frame_time=1.2, now=1.2).state == LockState.SEARCHING


def test_searching_after_auto_level_releases_control():
    controller, port, link, hybrid = make_controller([det(), None, None, None])
    controller.step(FRAME, frame_time=0.0, now=0.0)
    controller.step(FRAME, frame_time=0.1, now=0.1)
    controller.step(FRAME, frame_time=1.2, now=1.2)   # auto-level ends
    controller.step(FRAME, frame_time=1.3, now=1.3)   # searching

    assert link.last_written[0] == STATE_OFF


def test_switch_off_overrides_auto_leveling_immediately():
    controller, port, link, hybrid = make_controller([det(), None])
    controller.step(FRAME, frame_time=0.0, now=0.0)  # locked

    port.rc_channels_to_report[LOCK_SWITCH_INDEX] = SWITCH_OFF_US  # pilot kills it
    result = controller.step(FRAME, frame_time=0.1, now=0.1)

    assert result.state == LockState.DISENGAGED
    assert link.last_written[0] == STATE_OFF
    assert hybrid.reset_calls == 1


def test_below_floor_forces_recovery_climb_and_limits_bank():
    controller, port, link, hybrid = make_controller([det(cx=600, cy=460)])  # target low right: dive + hard bank
    port.altitude_cm_to_report = 300  # ~9.8 ft, below the 15ft floor

    result = controller.step(FRAME, frame_time=0.0, now=0.0)

    assert result.altitude_boosting is True
    state, roll, pitch = link.last_written
    assert pitch == 15
    assert 0 < roll <= 15


def test_fast_sink_above_floor_triggers_recovery_early():
    controller, port, link, hybrid = make_controller([det(cy=400)])
    port.altitude_cm_to_report = 900        # ~29.5 ft
    port.vario_cm_s_to_report = -400        # sinking 13 ft/s -> 1.5 s ahead: ~10 ft

    result = controller.step(FRAME, frame_time=0.0, now=0.0)

    assert result.altitude_boosting is True
    assert link.last_written[2] == 15


def test_dive_is_limited_near_the_floor():
    controller, port, link, hybrid = make_controller([det(cy=470)])  # target ~20 deg below
    port.altitude_cm_to_report = 760  # ~25 ft: 10 ft above the floor

    result = controller.step(FRAME, frame_time=0.0, now=0.0)

    assert result.altitude_boosting is False
    assert result.command.pitch_deg < -15
    assert link.last_written[2] > -5  # 3 m in 2 s at 30 m/s = ~3 deg dive allowed


def test_auto_level_also_climbs_when_below_floor():
    controller, port, link, hybrid = make_controller([det(), None])
    controller.step(FRAME, frame_time=0.0, now=0.0)
    port.altitude_cm_to_report = 300

    result = controller.step(FRAME, frame_time=0.1, now=0.1)

    assert result.state == LockState.AUTO_LEVELING
    assert result.altitude_boosting is True
    assert link.last_written == (STATE_ENGAGED, 0, 15)


def test_shutdown_releases():
    controller, port, link, hybrid = make_controller([det()])
    controller.step(FRAME, frame_time=0.0, now=0.0)

    controller.shutdown()

    assert link.last_written == (STATE_OFF, 0, 0)
    assert controller.state == LockState.DISENGAGED
