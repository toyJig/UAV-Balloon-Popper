from dataclasses import replace

import pytest

from lockon.guidance.intercept import GuidanceParams, SpeedEstimator
from lockon.guidance.sim import SimConfig, project, simulate
from lockon.guidance.attitude import Attitude
from lockon.vision.camera_model import CameraModel

CASES = {
    "ahead": ((150.0, 0.0, -20.0), (0.0, 0.0, 0.0)),
    "off-axis": ((150.0, 25.0, -25.0), (0.0, 0.0, 0.0)),
    "high left": ((200.0, -40.0, -45.0), (0.0, 0.0, 0.0)),
    "drifting": ((180.0, 20.0, -30.0), (0.0, -3.0, -0.5)),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("speed", [15.0, 30.0, 45.0])
def test_nominal_intercept_pops_a_balloon(name, speed):
    pos, vel = CASES[name]
    worst = max(simulate(SimConfig(speed_mps=speed, target_pos=pos, target_vel=vel, seed=s)).miss_m
                for s in range(4))
    assert worst < 2.2  # a ~1 m wing pops a balloon well inside this


def test_clean_run_is_a_direct_hit():
    r = simulate(SimConfig(pixel_noise_px=0.0, attitude_noise_deg=0.0,
                           target_pos=(150.0, 20.0, -25.0)))
    assert r.miss_m < 0.6
    assert r.hit


def test_floor_is_respected_even_chasing_a_low_target():
    # target at ~16 ft; the plane must not dive through the 15 ft floor
    r = simulate(SimConfig(start_alt_m=20.0, target_pos=(150.0, 10.0, -5.0), floor_ft=15.0,
                           pixel_noise_px=0.0, attitude_noise_deg=0.0))
    assert r.min_alt_ft >= 15.0 - 0.5


def test_latency_prediction_helps_at_high_latency():
    cfg = SimConfig(latency_s=0.3, frame_rate_hz=10.0, target_pos=(180.0, 30.0, -25.0),
                    pixel_noise_px=0.0, attitude_noise_deg=0.0)
    r = simulate(cfg)
    assert r.miss_m < 2.0


def test_most_frames_see_the_target():
    # frames can be lost after the plane flies past the target; what matters
    # is that it is tracked almost the whole way in
    r = simulate(SimConfig(target_pos=(150.0, 10.0, -22.0)))
    assert r.frames_seen / r.frames_total > 0.85


def test_project_is_none_behind_and_out_of_frame():
    cam = CameraModel.for_module3()
    assert project(cam, (-100.0, 0.0, 0.0), Attitude(0, 0, 0)) is None  # behind
    assert project(cam, (100.0, 500.0, 0.0), Attitude(0, 0, 0)) is None  # off to the side


# -- speed estimator ------------------------------------------------------
def test_speed_estimator_recovers_true_speed_from_a_coordinated_turn():
    import math
    G = 9.81
    est = SpeedEstimator(default_mps=30.0)
    V = 22.0
    bank = 30.0
    rate = math.degrees(G * math.tan(math.radians(bank)) / V)  # true heading rate
    for _ in range(20):
        est.update(bank_deg=bank, pitch_deg=0.0, heading_rate_dps=rate, dt_s=0.1)
    assert est.speed_mps == pytest.approx(V, abs=0.5)


def test_speed_estimator_ignores_wings_level_and_clamps():
    est = SpeedEstimator(default_mps=30.0)
    assert est.update(2.0, 0.0, 0.5, 0.1) == 30.0   # barely banked: ignored
    assert est.samples == 0
    # absurd reading clamps into [min, max]
    est.update(60.0, 0.0, 1.0, 0.1)
    assert 15.0 <= est.speed_mps <= 45.0
