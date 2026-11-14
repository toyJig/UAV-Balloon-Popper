import math

import pytest

from lockon.guidance.attitude import Attitude
from lockon.vision.camera_model import CameraModel

LEVEL = Attitude(0, 0, 0)


def test_module3_focal_lengths_at_640x480():
    assert CameraModel.for_module3().f_px == pytest.approx(627, abs=1)
    assert CameraModel.for_module3(wide=True).f_px == pytest.approx(364, abs=1)
    assert CameraModel.for_module3().hfov_deg == pytest.approx(54, abs=0.5)


def test_scaler_crop_width_is_used_when_given():
    full = CameraModel.for_module3(crop_width_sensor_px=4608)
    assert full.f_px == pytest.approx(4.74 / 0.0014 * 640 / 4608)


def test_centre_pixel_is_straight_ahead():
    los = CameraModel.for_module3().line_of_sight(320, 240, LEVEL)
    assert los.az_deg == pytest.approx(0)
    assert los.el_deg == pytest.approx(0)


def test_pixel_right_and_up_when_level():
    cam = CameraModel.for_module3()
    los = cam.line_of_sight(320 + cam.f_px, 240, LEVEL)  # 45 deg right
    assert los.az_deg == pytest.approx(45)
    los = cam.line_of_sight(320, 240 - cam.f_px * math.tan(math.radians(10)), LEVEL)
    assert los.el_deg == pytest.approx(10)


def test_nose_up_attitude_raises_elevation():
    los = CameraModel.for_module3().line_of_sight(320, 240, Attitude(0, 12, 0))
    assert los.el_deg == pytest.approx(12)
    assert los.az_deg == pytest.approx(0)


def test_right_bank_turns_an_image_right_offset_downward():
    # banked right 90 deg: image "right" points at the ground
    cam = CameraModel.for_module3()
    los = cam.line_of_sight(320 + cam.f_px, 240, Attitude(90, 0, 0))
    assert los.el_deg == pytest.approx(-45)
    assert los.az_deg == pytest.approx(0, abs=1e-9)


def test_mount_pitch_tilts_boresight_up():
    los = CameraModel.for_module3(mount_pitch_deg=5).line_of_sight(320, 240, LEVEL)
    assert los.el_deg == pytest.approx(5)


def test_inertial_azimuth_adds_heading():
    cam = CameraModel.for_module3()
    los = cam.line_of_sight(320 + cam.f_px, 240, Attitude(0, 0, 350))
    assert los.inertial_az_deg == pytest.approx(395)
