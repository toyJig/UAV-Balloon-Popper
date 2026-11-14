import pytest

from lockon.guidance.control_law import PursuitGuidance


FRAME_SIZE = (640, 480)  # center = (320, 240)


def test_stationary_target_at_center_gives_near_zero_command():
    guidance = PursuitGuidance(FRAME_SIZE, kp=1.0, kd=0.0, lead_time_s=0.0)
    cmd = guidance.update((320, 240), dt=0.05)
    assert cmd.roll_cmd == pytest.approx(0.0, abs=1e-6)
    assert cmd.pitch_cmd == pytest.approx(0.0, abs=1e-6)


def test_target_to_the_right_gives_positive_roll_command():
    guidance = PursuitGuidance(FRAME_SIZE, kp=1.0, kd=0.0, lead_time_s=0.0)
    cmd = guidance.update((620, 240), dt=0.05)  # far right, same height
    assert cmd.roll_cmd > 0
    assert cmd.pitch_cmd == pytest.approx(0.0, abs=1e-6)


def test_target_below_center_gives_negative_pitch_command_ie_dive():
    # pitch_cmd follows the altitude clamp's "positive = climb" convention
    # (Task 11 feeds this value straight into AltitudeFloorClamp), so a
    # target below boresight -> dive toward it -> negative pitch_cmd.
    guidance = PursuitGuidance(FRAME_SIZE, kp=1.0, kd=0.0, lead_time_s=0.0)
    cmd = guidance.update((320, 440), dt=0.05)  # near bottom of frame
    assert cmd.pitch_cmd < 0
    assert cmd.roll_cmd == pytest.approx(0.0, abs=1e-6)


def test_target_above_center_gives_positive_pitch_command_ie_climb():
    guidance = PursuitGuidance(FRAME_SIZE, kp=1.0, kd=0.0, lead_time_s=0.0)
    cmd = guidance.update((320, 40), dt=0.05)  # near top of frame
    assert cmd.pitch_cmd > 0
    assert cmd.roll_cmd == pytest.approx(0.0, abs=1e-6)


def test_moving_target_produces_a_lead_point_ahead_of_its_raw_position():
    guidance = PursuitGuidance(FRAME_SIZE, kp=1.0, kd=0.0, lead_time_s=0.5)
    guidance.update((320, 240), dt=0.1)             # first sample, no velocity yet
    cmd = guidance.update((340, 240), dt=0.1)        # moved 20px right in 0.1s -> 200px/s rightward

    # lead point should be further right than the target's current raw position
    assert cmd.lead_point[0] > 340


def test_command_is_clipped_to_max_cmd():
    guidance = PursuitGuidance(FRAME_SIZE, kp=10.0, kd=0.0, lead_time_s=0.0, max_cmd=1.0)
    cmd = guidance.update((640, 480), dt=0.05)  # extreme bottom-right corner, huge raw error
    assert cmd.roll_cmd == pytest.approx(1.0)       # far right -> max roll
    assert cmd.pitch_cmd == pytest.approx(-1.0)     # far below center -> max dive


def test_rejects_non_positive_dt():
    guidance = PursuitGuidance(FRAME_SIZE)
    with pytest.raises(ValueError):
        guidance.update((320, 240), dt=0.0)


def test_reset_clears_velocity_history_so_no_sudden_jump():
    guidance = PursuitGuidance(FRAME_SIZE, kp=1.0, kd=0.0, lead_time_s=0.5)
    guidance.update((100, 240), dt=0.1)
    guidance.reset()
    # After reset, a big position jump must not be treated as velocity
    # (no history), so the lead point should equal the raw target position.
    cmd = guidance.update((500, 240), dt=0.1)
    assert cmd.lead_point == (500, 240)


def test_derivative_term_dampens_a_rapidly_growing_error():
    with_damping = PursuitGuidance(FRAME_SIZE, kp=1.0, kd=0.5, lead_time_s=0.0)
    without_damping = PursuitGuidance(FRAME_SIZE, kp=1.0, kd=0.0, lead_time_s=0.0)

    for g in (with_damping, without_damping):
        g.update((320, 240), dt=0.1)  # at center first

    cmd_damped = with_damping.update((500, 240), dt=0.1)   # error growing fast
    cmd_undamped = without_damping.update((500, 240), dt=0.1)

    # Derivative term opposes a growing error, so the damped command should
    # be larger in magnitude than the undamped P-only command here (error
    # went from 0 to positive, so d_error is positive, kd term adds to kp term).
    assert cmd_damped.roll_cmd > cmd_undamped.roll_cmd
