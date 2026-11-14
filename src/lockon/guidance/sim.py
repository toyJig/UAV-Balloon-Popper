"""Closed-loop intercept simulation: plane + camera + guidance + INAV link.

Checks the guidance against things the unit tests can't: speed, lag,
latency, the 10 Hz logic-condition update rate, pixel noise and attitude
noise, all at once. Not an aerodynamic model; just the parts that set the
miss distance:

* Point-mass plane at constant airspeed ``speed_mps`` (the guidance itself
  assumes ``GuidanceParams.speed_mps``; set them differently to test a
  wrong speed estimate). Coordinated turns: turn rate = g tan(bank) / V.
* INAV's angle mode as first-order lags with rate limits: bank follows the
  target with ``roll_tau_s``, nose pitch with ``pitch_tau_s``. The flight
  path sits ``aoa_deg`` below the nose.
* Frames at ``frame_rate_hz``; each command reaches the FC ``latency_s``
  after the frame was exposed (inference + MSP), and the FC picks up the
  latest command at its 10 Hz logic-condition tick.
* Detections are the target's true pixel position plus Gaussian noise;
  attitude passed to guidance is the true attitude at exposure plus noise
  (the controller's gyro extrapolation makes it near-true in practice).
* The 15 ft altitude guard runs in the loop.

Result: miss distance = closest approach between the plane and the target
centre (computed between integration steps, so a fast pass isn't missed).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..vision.camera_model import CameraModel
from .altitude_clamp import FT_PER_M, AltitudeGuard
from .attitude import Attitude
from .intercept import G, GuidanceParams, InterceptGuidance

Vec = Tuple[float, float, float]


@dataclass
class SimConfig:
    speed_mps: float = 30.0
    guidance: GuidanceParams = field(default_factory=GuidanceParams)
    # NED metres, plane starts at (0, 0, -start_alt_m) heading north, level
    start_alt_m: float = 20.0
    target_pos: Vec = (150.0, 25.0, -25.0)
    target_vel: Vec = (0.0, 0.0, 0.0)
    start_heading_deg: float = 0.0
    frame_rate_hz: float = 15.0
    latency_s: float = 0.15
    lc_period_s: float = 0.1
    roll_tau_s: float = 0.2
    max_roll_rate_dps: float = 250.0
    pitch_tau_s: float = 0.3
    max_pitch_rate_dps: float = 120.0
    aoa_deg: float = 3.0
    pixel_noise_px: float = 1.0
    attitude_noise_deg: float = 0.5
    camera: CameraModel = field(default_factory=CameraModel.for_module3)
    floor_ft: float = 15.0
    dt: float = 0.002
    t_max: float = 20.0
    seed: int = 0


@dataclass
class SimResult:
    miss_m: float
    t_closest_s: float
    frames_seen: int
    frames_total: int
    min_alt_ft: float
    hit: bool
    trace: List[dict]


def _rot_body_to_ned(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # R = Rz(yaw) Ry(pitch) Rx(roll); rows are NED components of body axes
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def project(cam: CameraModel, rel_ned: Vec, att: Attitude) -> Optional[Tuple[float, float]]:
    """Pixel of a NED offset seen from the plane, or None if out of view."""
    R = _rot_body_to_ned(math.radians(att.roll_deg), math.radians(att.pitch_deg), math.radians(att.yaw_deg))
    bx, by, bz = (sum(R[i][j] * rel_ned[i] for i in range(3)) for j in range(3))  # R^T r
    a = math.radians(cam.mount_pitch_deg)  # camera tilted up: rotate into camera axes
    fwd = bx * math.cos(a) - bz * math.sin(a)
    down = bx * math.sin(a) + bz * math.cos(a)
    if fwd <= 0.5:
        return None
    u, v = cam.cx + cam.f_px * by / fwd, cam.cy + cam.f_px * down / fwd
    if not (0 <= u < cam.width and 0 <= v < cam.height):
        return None
    return u, v


def _closest_on_segment(r0: Vec, r1: Vec) -> float:
    d = [b - a for a, b in zip(r0, r1)]
    dd = sum(x * x for x in d)
    s = 0.0 if dd == 0 else max(0.0, min(1.0, -sum(a * b for a, b in zip(r0, d)) / dd))
    return math.sqrt(sum((a + s * b) ** 2 for a, b in zip(r0, d)))


def simulate(cfg: SimConfig = SimConfig(), hit_radius_m: float = 1.0, keep_trace: bool = False) -> SimResult:
    rng = random.Random(cfg.seed)
    guidance = InterceptGuidance(cfg.guidance)
    guard = AltitudeGuard(floor_ft=cfg.floor_ft)
    cam = cfg.camera

    pos = [0.0, 0.0, -cfg.start_alt_m]
    tgt = list(cfg.target_pos)
    roll = 0.0
    pitch = cfg.aoa_deg
    yaw = cfg.start_heading_deg
    V = cfg.speed_mps

    infer_latency = max(0.0, cfg.latency_s - cfg.guidance.command_delay_s)
    captured: List[Tuple] = []                       # frames awaiting guidance (inference lag)
    pending: List[Tuple[float, float, float]] = []   # (time available at FC, bank, pitch)
    written = (0.0, cfg.aoa_deg)                     # latest command the Pi has written
    fc_target = (0.0, cfg.aoa_deg)                   # what INAV's angle controller holds
    next_frame, next_tick = 0.0, 0.0
    frames_seen = frames_total = 0
    best, t_best = float("inf"), 0.0
    min_alt_ft = float("inf")
    prev_rel: Optional[Vec] = None
    trace: List[dict] = []
    gamma_prev = math.radians(pitch - cfg.aoa_deg)

    t = 0.0
    while t < cfg.t_max:
        rel = tuple(b - a for a, b in zip(pos, tgt))
        if prev_rel is not None:
            d = _closest_on_segment(prev_rel, rel)
            if d < best:
                best, t_best = d, t
        prev_rel = rel
        alt_ft = -pos[2] * FT_PER_M
        min_alt_ft = min(min_alt_ft, alt_ft)
        if best < 30 and math.sqrt(sum(x * x for x in rel)) > best + 15:
            break  # passed the target

        if t >= next_frame:
            next_frame += 1.0 / cfg.frame_rate_hz
            frames_total += 1
            true_att = Attitude(roll, pitch, yaw % 360.0)
            px = project(cam, rel, true_att)
            n = cfg.attitude_noise_deg
            if px is not None:
                frames_seen += 1
                u = px[0] + rng.gauss(0, cfg.pixel_noise_px)
                v = px[1] + rng.gauss(0, cfg.pixel_noise_px)
                seen_att = Attitude(roll + rng.gauss(0, n), pitch + rng.gauss(0, n), (yaw + rng.gauss(0, n)) % 360.0)
                captured.append((t + infer_latency, t, u, v, seen_att))
            else:
                captured.append((t + infer_latency, t, None, None, None))

        # the Pi reads the FC attitude and runs guidance infer_latency after
        # exposure; the command then waits command_delay_s to reach the FC
        while captured and captured[0][0] <= t:
            t_run, exp_t, u, v, seen_att = captured.pop(0)
            climb_ftps = V * math.sin(gamma_prev) * FT_PER_M
            if u is not None:
                heading_now = (yaw + rng.gauss(0, n)) % 360.0
                cmd = guidance.update(cam.line_of_sight(u, v, seen_att), exp_t,
                                      now=t_run, heading_now_deg=heading_now)
                res = guard.update(alt_ft, climb_ftps, cmd.pitch_deg, cmd.bank_deg, cfg.guidance.speed_mps)
            else:
                guidance.reset()
                res = guard.update(alt_ft, climb_ftps, cfg.aoa_deg, 0.0, cfg.guidance.speed_mps)
            bank = max(-45, min(45, round(res.bank_deg)))  # link rounds, INAV clamps to its max angles
            pit = max(-30, min(30, round(res.pitch_deg)))
            pending.append((t_run + cfg.guidance.command_delay_s, bank, pit))
            if keep_trace:
                trace.append(dict(t=t, pos=tuple(pos), roll=roll, pitch=pitch, yaw=yaw, seen=u is not None,
                                  bank_cmd=bank, pitch_cmd=pit, alt_ft=alt_ft, boost=res.boosting))

        while pending and pending[0][0] <= t:
            _, b, p_ = pending.pop(0)
            written = (b, p_)
        if t >= next_tick:
            next_tick += cfg.lc_period_s
            fc_target = written

        # INAV angle mode as rate-limited first-order lags
        droll = (fc_target[0] - roll) / cfg.roll_tau_s
        droll = max(-cfg.max_roll_rate_dps, min(cfg.max_roll_rate_dps, droll))
        dpitch = (fc_target[1] - pitch) / cfg.pitch_tau_s
        dpitch = max(-cfg.max_pitch_rate_dps, min(cfg.max_pitch_rate_dps, dpitch))
        roll += droll * cfg.dt
        pitch += dpitch * cfg.dt
        gamma = math.radians(pitch - cfg.aoa_deg)
        # coordinated turn; the pitch axis turns with bank, so heading change
        # is g tan(bank) / V
        yaw += math.degrees(G * math.tan(math.radians(roll)) / V) * cfg.dt
        psi = math.radians(yaw)
        vel = (V * math.cos(gamma) * math.cos(psi), V * math.cos(gamma) * math.sin(psi), -V * math.sin(gamma))
        for i in range(3):
            pos[i] += vel[i] * cfg.dt
            tgt[i] += cfg.target_vel[i] * cfg.dt
        gamma_prev = gamma
        t += cfg.dt

    return SimResult(best, t_best, frames_seen, frames_total, min_alt_ft, best <= hit_radius_m, trace)
