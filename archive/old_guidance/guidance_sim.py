#!/usr/bin/env python3
"""Task 10 demo - simulates a target moving across the frame and prints
the guidance law's commanded output each step, so the predictive lead
and damping behavior can be sanity-checked before any camera/RC wiring
exists (that happens in Task 11). No hardware needed; runs anywhere.

Usage:
    python tools/guidance_sim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lockon.guidance.control_law import PursuitGuidance

FRAME_SIZE = (640, 480)
DT = 0.05  # 20 Hz simulated loop


def simulate_linear_crossing() -> None:
    """Target starts at the left edge and crosses to the right edge at a
    constant speed - the lead point should visibly run ahead of the raw
    target position once velocity has been established."""
    guidance = PursuitGuidance(FRAME_SIZE, kp=1.0, kd=0.3, lead_time_s=0.3)

    print(f"{'t':>5} {'target_x':>9} {'target_y':>9} {'lead_x':>8} {'lead_y':>8} {'roll_cmd':>9} {'pitch_cmd':>10}")
    x, y = 0.0, 240.0
    vx = 400.0  # px/s rightward
    t = 0.0
    for _ in range(40):
        cmd = guidance.update((x, y), dt=DT)
        print(f"{t:5.2f} {x:9.1f} {y:9.1f} {cmd.lead_point[0]:8.1f} {cmd.lead_point[1]:8.1f} "
              f"{cmd.roll_cmd:9.3f} {cmd.pitch_cmd:10.3f}")
        x += vx * DT
        t += DT


if __name__ == "__main__":
    simulate_linear_crossing()
