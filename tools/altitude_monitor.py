#!/usr/bin/env python3
"""Task 5 bench demo - streams live baro altitude off the real FC and
prints the AltitudeFloorClamp's decision in real time.

Move/tilt the plane by hand near the 15 ft floor (or, more practically on a
bench, temporarily lower the configured floor with --floor-ft to a value
reachable by just lifting the nose/tail) and watch the printed boosting
state flip on/off correctly with hysteresis.

Usage:
    python tools/altitude_monitor.py --port COM5 --floor-ft 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import serial  # type: ignore

from lockon.msp.client import MSPClient
from lockon.guidance.altitude_clamp import FT_PER_M, AltitudeGuard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--floor-ft", type=float, default=15.0)
    parser.add_argument("--margin-ft", type=float, default=5.0)
    parser.add_argument("--speed-mps", type=float, default=30.0,
                        help="assumed airspeed, for the dive-angle limit")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    client = MSPClient(ser)
    guard = AltitudeGuard(floor_ft=args.floor_ft, margin_ft=args.margin_ft)

    print(f"Floor={args.floor_ft}ft, margin={args.margin_ft}ft, V={args.speed_mps}m/s. "
          "Pretend the plane is diving (pitch=-20). Ctrl+C to stop.")
    try:
        while True:
            alt = client.read_altitude()
            climb_ftps = alt.vario_cm_s / 30.48
            # feed a -20 deg dive so the dive limit and recovery are visible on the bench
            result = guard.update(alt.altitude_ft, climb_ftps, pitch_deg=-20.0, bank_deg=0.0,
                                  speed_mps=args.speed_mps)
            flag = "** RECOVERY CLIMB **" if result.boosting else "ok"
            print(f"altitude={alt.altitude_ft:6.1f} ft  vario={alt.vario_cm_s:+4d} cm/s  "
                  f"dive_limit={result.dive_limit_deg:4.1f} deg  pitch_out={result.pitch_deg:+5.1f} -> {flag}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
