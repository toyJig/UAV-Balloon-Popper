"""Camera geometry: turns a pixel into a direction, and with the aircraft's
attitude into a line of sight (azimuth/elevation) in a level frame.

Raspberry Pi Camera Module 3 (Sony IMX708): 1.4 um pixels, 4608 x 2592
active array. Lens focal length 4.74 mm (standard, 66 deg horizontal FOV)
or 2.75 mm (wide, 102 deg). A 640x480 (4:3) capture is cropped from the
16:9 sensor; the actual crop is read from Picamera2's ScalerCrop metadata
when available, otherwise a centred 4:3 crop of the full height is assumed
(what libcamera does by default). The focal length in output pixels is

    f_px = (focal_mm / 0.0014 mm) * output_width / crop_width_sensor_px

which is ~627 px (standard) or ~364 px (wide) for 640x480.

Frames: camera x right, y down, z forward. Body x forward, y right, z down.
The camera is assumed to look along the body x axis, optionally tilted up
by ``mount_pitch_deg`` (measure it once on the airframe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ..guidance.attitude import Attitude

PIXEL_MM = 0.0014
SENSOR_W, SENSOR_H = 4608, 2592
FOCAL_MM = {"imx708": 4.74, "imx708_wide": 2.75}


@dataclass(frozen=True)
class LineOfSight:
    az_deg: float        # right of current heading, in the level frame
    el_deg: float        # above the horizon
    heading_deg: float   # aircraft heading the azimuth is relative to

    @property
    def inertial_az_deg(self) -> float:
        return self.heading_deg + self.az_deg


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    f_px: float
    cx: float
    cy: float
    mount_pitch_deg: float = 0.0

    @classmethod
    def for_module3(cls, width: int = 640, height: int = 480, wide: bool = False,
                    crop_width_sensor_px: Optional[float] = None, mount_pitch_deg: float = 0.0) -> "CameraModel":
        focal = FOCAL_MM["imx708_wide" if wide else "imx708"]
        if crop_width_sensor_px is None:
            # centred crop of the full sensor height to the output aspect ratio
            crop_width_sensor_px = min(SENSOR_W, SENSOR_H * width / height)
        f_px = focal / PIXEL_MM * width / crop_width_sensor_px
        return cls(width, height, f_px, width / 2.0, height / 2.0, mount_pitch_deg)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / 2.0 / self.f_px))

    def body_ray(self, u: float, v: float) -> Tuple[float, float, float]:
        """Unit vector (x fwd, y right, z down) in body axes for pixel (u, v)."""
        xc, yc = (u - self.cx) / self.f_px, (v - self.cy) / self.f_px
        fwd, right, down = 1.0, xc, yc
        a = math.radians(self.mount_pitch_deg)  # camera tilted nose-up by a
        fwd, down = fwd * math.cos(a) + down * math.sin(a), -fwd * math.sin(a) + down * math.cos(a)
        n = math.sqrt(fwd * fwd + right * right + down * down)
        return fwd / n, right / n, down / n

    def line_of_sight(self, u: float, v: float, att: Attitude) -> LineOfSight:
        """Pixel -> level-frame azimuth/elevation, removing the aircraft's
        roll and pitch (the image tilts with the plane; the target doesn't)."""
        x, y, z = self.body_ray(u, v)
        phi, theta = math.radians(att.roll_deg), math.radians(att.pitch_deg)
        # body -> level (heading-aligned NED): v = Ry(theta) Rx(phi) v_body
        y1 = y * math.cos(phi) - z * math.sin(phi)
        z1 = y * math.sin(phi) + z * math.cos(phi)
        x2 = x * math.cos(theta) + z1 * math.sin(theta)
        z2 = -x * math.sin(theta) + z1 * math.cos(theta)
        az = math.degrees(math.atan2(y1, x2))
        el = math.degrees(math.atan2(-z2, math.hypot(x2, y1)))
        return LineOfSight(az, el, att.yaw_deg)
