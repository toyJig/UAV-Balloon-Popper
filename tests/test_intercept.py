import math

import pytest

from lockon.guidance.intercept import GuidanceParams, InterceptGuidance, bank_for_turn_rate
from lockon.vision.camera_model import LineOfSight


def test_bank_for_turn_rate_coordinated_turn():
    # 30 m/s, 18 deg/s -> tan(bank) = 30 * 0.314 / 9.81 = 0.96 -> 43.8 deg
    assert bank_for_turn_rate(18, 30) == pytest.approx(math.degrees(math.atan(30 * math.radians(18) / 9.81)))
    assert bank_for_turn_rate(-18, 30) < 0


def test_target_dead_ahead_flies_straight_with_aoa():
    g = InterceptGuidance(GuidanceParams())
    cmd = g.update(LineOfSight(0, 0, 0), 0.0)
    assert cmd.bank_deg == 0
    assert cmd.pitch_deg == pytest.approx(3.0)


def test_pursuit_banks_toward_the_target_and_clamps():
    p = GuidanceParams()
    assert InterceptGuidance(p).update(LineOfSight(5, 0, 0), 0.0).bank_deg == pytest.approx(bank_for_turn_rate(5, 30))
    assert InterceptGuidance(p).update(LineOfSight(-60, 0, 0), 0.0).bank_deg == -p.max_bank_deg


def test_pitch_follows_elevation_and_clamps():
    p = GuidanceParams()
    assert InterceptGuidance(p).update(LineOfSight(0, -10, 0), 0.0).pitch_deg == pytest.approx(-7)
    assert InterceptGuidance(p).update(LineOfSight(0, 60, 0), 0.0).pitch_deg == p.max_pitch_deg


def test_pn_adds_lead_for_a_drifting_line_of_sight():
    p = GuidanceParams(rate_alpha=1.0)
    g = InterceptGuidance(p)
    g.update(LineOfSight(0, 0, 0), 0.0)
    cmd = g.update(LineOfSight(1, 0, 0), 0.1)  # LOS rotating right 10 deg/s
    assert cmd.az_rate_dps == pytest.approx(10)
    assert cmd.turn_rate_dps == pytest.approx(1 / p.pursuit_time_s + p.nav_constant * 10)


def test_own_turn_is_not_target_motion():
    # plane turns right 10 deg/s toward a fixed target: the azimuth shrinks
    # by exactly the heading change, so the inertial LOS rate is zero
    g = InterceptGuidance(GuidanceParams(rate_alpha=1.0))
    g.update(LineOfSight(20, 0, 100), 0.0)
    cmd = g.update(LineOfSight(19, 0, 101), 0.1)
    assert cmd.az_rate_dps == pytest.approx(0)


def test_heading_wrap_does_not_spike_the_rate():
    g = InterceptGuidance(GuidanceParams(rate_alpha=1.0))
    g.update(LineOfSight(0, 0, 359.5), 0.0)
    cmd = g.update(LineOfSight(0, 0, 0.5), 0.1)
    assert cmd.az_rate_dps == pytest.approx(10)


def test_long_gap_and_reset_restart_the_rate_estimate():
    g = InterceptGuidance(GuidanceParams(rate_alpha=1.0))
    g.update(LineOfSight(0, 0, 0), 0.0)
    g.update(LineOfSight(1, 0, 0), 0.1)
    assert g.update(LineOfSight(5, 0, 0), 2.0).az_rate_dps == 0
    g.reset()
    assert g.update(LineOfSight(10, 0, 0), 2.1).az_rate_dps == 0
