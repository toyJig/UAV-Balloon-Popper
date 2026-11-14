import pytest

from lockon.guidance.altitude_clamp import AltitudeGuard

V = 30.0


def guard():
    return AltitudeGuard(floor_ft=15.0, margin_ft=5.0, lookahead_s=1.5, time_to_floor_s=2.0,
                         recovery_pitch_deg=15.0, recovery_max_bank_deg=15.0)


def test_passes_command_through_when_high_and_level():
    r = guard().update(altitude_ft=300.0, climb_rate_ftps=0.0, pitch_deg=-20.0, bank_deg=40.0, speed_mps=V)
    assert r.boosting is False
    assert (r.pitch_deg, r.bank_deg) == (-20.0, 40.0)


def test_below_floor_forces_recovery_pitch_and_limits_bank():
    r = guard().update(altitude_ft=14.9, climb_rate_ftps=0.0, pitch_deg=-20.0, bank_deg=-40.0, speed_mps=V)
    assert r.boosting is True
    assert r.pitch_deg == 15.0
    assert r.bank_deg == -15.0


def test_recovery_keeps_a_steeper_requested_climb():
    r = guard().update(altitude_ft=10.0, climb_rate_ftps=0.0, pitch_deg=25.0, bank_deg=0.0, speed_mps=V)
    assert r.pitch_deg == 25.0


def test_predicted_crossing_triggers_recovery_before_the_floor():
    # 25 ft, sinking 10 ft/s: 1.5 s ahead the plane would be at 10 ft
    r = guard().update(altitude_ft=25.0, climb_rate_ftps=-10.0, pitch_deg=-10.0, bank_deg=0.0, speed_mps=V)
    assert r.boosting is True


def test_climbing_does_not_count_toward_the_prediction():
    r = guard().update(altitude_ft=16.0, climb_rate_ftps=20.0, pitch_deg=0.0, bank_deg=0.0, speed_mps=V)
    assert r.boosting is False


def test_dive_limit_shrinks_toward_level_near_the_floor():
    g = guard()
    high = g.update(altitude_ft=200.0, climb_rate_ftps=0.0, pitch_deg=-30.0, bank_deg=0.0, speed_mps=V)
    near = g.update(altitude_ft=20.0, climb_rate_ftps=0.0, pitch_deg=-30.0, bank_deg=0.0, speed_mps=V)
    assert high.pitch_deg == -30.0  # 185 ft / 2 s = 28 m/s allowed sink: no limit at 30 m/s for -30 deg
    assert -3.0 < near.pitch_deg < 0.0
    # 5 ft above floor -> 0.76 m/s allowed sink at 30 m/s -> asin(0.0254) = 1.46 deg
    assert near.dive_limit_deg == pytest.approx(1.455, abs=0.01)


def test_dive_limit_is_larger_at_lower_speed():
    g = guard()
    slow = g.update(altitude_ft=40.0, climb_rate_ftps=0.0, pitch_deg=-30.0, bank_deg=0.0, speed_mps=15.0)
    fast = g.update(altitude_ft=40.0, climb_rate_ftps=0.0, pitch_deg=-30.0, bank_deg=0.0, speed_mps=45.0)
    assert slow.dive_limit_deg > fast.dive_limit_deg


def test_stays_boosting_through_the_hysteresis_band():
    g = guard()
    g.update(altitude_ft=10.0, climb_rate_ftps=0.0, pitch_deg=0.0, bank_deg=0.0, speed_mps=V)
    r = g.update(altitude_ft=18.0, climb_rate_ftps=5.0, pitch_deg=-5.0, bank_deg=0.0, speed_mps=V)
    assert r.boosting is True
    assert r.pitch_deg == 15.0


def test_releases_only_above_floor_plus_margin_and_not_descending():
    g = guard()
    g.update(altitude_ft=10.0, climb_rate_ftps=0.0, pitch_deg=0.0, bank_deg=0.0, speed_mps=V)
    assert g.update(20.0, -1.0, 0.0, 0.0, V).boosting is True   # high enough but still sinking
    assert g.update(20.0, 0.0, 0.0, 0.0, V).boosting is False


def test_does_not_chatter_at_the_floor():
    g = guard()
    results = [g.update(a, 0.0, 0.0, 0.0, V) for a in (15.0, 14.9, 15.1, 14.95, 15.05, 14.8)]
    first = next(i for i, r in enumerate(results) if r.boosting)
    assert all(r.boosting for r in results[first:])


def test_reset_clears_boosting_state():
    g = guard()
    g.update(5.0, 0.0, 0.0, 0.0, V)
    assert g.is_boosting is True
    g.reset()
    assert g.is_boosting is False


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        AltitudeGuard(margin_ft=-1.0)
    with pytest.raises(ValueError):
        AltitudeGuard(time_to_floor_s=0.0)
