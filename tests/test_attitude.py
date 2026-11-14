import pytest

from lockon.guidance.attitude import Attitude, BodyRates, euler_rates, extrapolate, from_inav, rates_from_inav
from lockon.msp.codec import RawAttitude


def test_from_inav_converts_units_and_pitch_sign():
    att = from_inav(RawAttitude(roll_decideg=150, pitch_decideg=-100, yaw_deg=-90))
    assert att.roll_deg == 15.0
    assert att.pitch_deg == 10.0  # INAV -10 (nose up) -> +10 nose up
    assert att.yaw_deg == 270.0


def test_rates_from_inav_flips_pitch():
    r = rates_from_inav((10, 20, -30))
    assert (r.p_dps, r.q_dps, r.r_dps) == (10, -20, -30)


def test_level_euler_rates_equal_body_rates():
    assert euler_rates(Attitude(0, 0, 0), BodyRates(10, 20, 30)) == pytest.approx((10, 20, 30))


def test_banked_pitch_rate_turns_the_heading():
    # 90 deg right bank: a nose-up body pitch rate is a right turn
    rd, pd, yd = euler_rates(Attitude(90, 0, 0), BodyRates(0, 10, 0))
    assert yd == pytest.approx(10)
    assert pd == pytest.approx(0, abs=1e-9)


def test_extrapolate_backwards_and_wraps():
    att = extrapolate(Attitude(170, 0, 0), BodyRates(200, 0, 0), -0.1)
    assert att.roll_deg == pytest.approx(150)
    att = extrapolate(Attitude(0, 0, 5), BodyRates(0, 0, 100), -0.1)
    assert att.yaw_deg == pytest.approx(355)
    att = extrapolate(Attitude(170, 0, 0), BodyRates(200, 0, 0), 0.1)
    assert att.roll_deg == pytest.approx(-170)


def test_extrapolate_clamps_pitch():
    att = extrapolate(Attitude(0, 85, 0), BodyRates(0, 100, 0), 0.1)
    assert att.pitch_deg == 90
