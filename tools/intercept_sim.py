#!/usr/bin/env python3
"""Closed-loop intercept simulation (no hardware): how close does the
guidance get the plane to a balloon/drone across speeds, latencies and
noise? See src/lockon/guidance/sim.py for what is modelled.

Usage:
    python tools/intercept_sim.py                  # scenario sweep table
    python tools/intercept_sim.py --speeds 15 30 45 --latency 0.25
    python tools/intercept_sim.py --assumed-speed 30 --wide

"miss" is the closest approach to the target centre in metres. A 30 cm
balloon is popped by a ~1 m wing when miss < ~0.6 m (half span + radius).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lockon.guidance.intercept import GuidanceParams  # noqa: E402
from lockon.guidance.sim import SimConfig, simulate  # noqa: E402
from lockon.vision.camera_model import CameraModel  # noqa: E402

# name: (target position NED m, target velocity NED m/s); plane starts at 20 m altitude
SCENARIOS = {
    "balloon ahead":        ((150.0, 0.0, -20.0), (0.0, 0.0, 0.0)),
    "balloon off-axis":     ((150.0, 25.0, -25.0), (0.0, 0.0, 0.0)),
    "balloon high left":    ((200.0, -40.0, -45.0), (0.0, 0.0, 0.0)),
    "balloon drifting":     ((180.0, 20.0, -30.0), (0.0, -3.0, -0.5)),   # wind drift + rising
    "drone crossing":       ((200.0, -30.0, -30.0), (0.0, 8.0, 0.0)),
    "drone receding":       ((120.0, 10.0, -25.0), (6.0, 2.0, 0.0)),
    "balloon low (floor)":  ((150.0, 10.0, -6.0), (0.0, 0.0, 0.0)),      # ~20 ft: floor guard active
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speeds", type=float, nargs="+", default=[15.0, 22.0, 30.0, 38.0, 45.0],
                    help="true airspeeds to simulate (m/s)")
    ap.add_argument("--assumed-speed", type=float, default=GuidanceParams().speed_mps,
                    help="speed the guidance assumes (GuidanceParams.speed_mps)")
    ap.add_argument("--latency", type=float, default=0.15, help="exposure -> command at FC (s)")
    ap.add_argument("--fps", type=float, default=15.0, help="detector frame rate")
    ap.add_argument("--pixel-noise", type=float, default=1.0)
    ap.add_argument("--attitude-noise", type=float, default=0.5)
    ap.add_argument("--wide", action="store_true", help="Camera Module 3 Wide optics")
    ap.add_argument("--seeds", type=int, default=5, help="noise seeds per case")
    args = ap.parse_args()

    base = SimConfig(
        guidance=replace(GuidanceParams(), speed_mps=args.assumed_speed),
        latency_s=args.latency, frame_rate_hz=args.fps, pixel_noise_px=args.pixel_noise,
        attitude_noise_deg=args.attitude_noise, camera=CameraModel.for_module3(wide=args.wide),
    )
    print(f"guidance assumes {args.assumed_speed} m/s, latency {args.latency}s, {args.fps} fps, "
          f"{'wide' if args.wide else 'standard'} lens; miss in metres (median / worst of {args.seeds} seeds)")
    header = f"{'scenario':22s}" + "".join(f"{f'{v:g} m/s':>16s}" for v in args.speeds)
    print(header)
    print("-" * len(header))
    for name, (pos, vel) in SCENARIOS.items():
        cells = []
        for v in args.speeds:
            results = [simulate(replace(base, speed_mps=v, target_pos=pos, target_vel=vel, seed=s))
                       for s in range(args.seeds)]
            misses = [r.miss_m for r in results]
            low = min(r.min_alt_ft for r in results)
            flag = "*" if low < base.floor_ft - 0.5 else " "
            cells.append(f"{statistics.median(misses):6.2f} /{max(misses):6.2f}{flag}")
        print(f"{name:22s}" + "".join(f"{c:>16s}" for c in cells))
    print("* = dipped more than 0.5 ft below the altitude floor in some run")


if __name__ == "__main__":
    main()
