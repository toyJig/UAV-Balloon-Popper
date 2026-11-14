#!/usr/bin/env python3
"""Bench check for the attitude/gyro sign conventions guidance relies on.

Guidance converts INAV's MSP_ATTITUDE and MSP_RAW_IMU into standard
aerospace signs (see src/lockon/guidance/attitude.py, InavSigns). Those
signs are assumed, not measured. This tool reads both at ~50 Hz while you
move the plane by hand and reports, per axis, whether the gyro rate matches
the rate of change of the attitude angle. If any axis prints MISMATCH, flip
the matching gyro_* sign in InavSigns and re-run.

It also prints a live attitude line so you can confirm the ATTITUDE signs
directly:
    roll > 0 when the RIGHT wing is down,
    pitch > 0 when the NOSE is UP,
    yaw increases when the nose turns to the RIGHT (clockwise seen from above).

Usage:
    python tools/check_attitude.py --port /dev/ttyAMA0
    python tools/check_attitude.py --port COM5 --seconds 20

The FC must be level and still for the first second (bias capture), then
roll, pitch and yaw it through moderate angles one axis at a time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import serial  # type: ignore

from lockon.guidance.attitude import InavSigns, euler_rates, from_inav, rates_from_inav
from lockon.msp.client import MSPClient


def _wrap(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--rate-threshold", type=float, default=20.0,
                    help="deg/s below which an axis is treated as 'not moving' for the correlation")
    args = ap.parse_args()

    client = MSPClient(serial.Serial(args.port, args.baud, timeout=0.05))
    signs = InavSigns()

    # correlation accumulators: sum(gyro_euler * attitude_delta_rate) per axis
    labels = ("roll", "pitch", "yaw")
    agree = [0, 0, 0]
    disagree = [0, 0, 0]

    prev = None
    prev_t = None
    t_end = time.monotonic() + args.seconds
    print("Move the plane one axis at a time. Watch the live line; summary prints at the end.\n")
    while time.monotonic() < t_end:
        att = from_inav(client.read_attitude(), signs)
        rates = rates_from_inav(client.read_gyro_dps(), signs)
        now = time.monotonic()
        euler = euler_rates(att, rates)  # expected d(roll,pitch,yaw)/dt from the gyro

        if prev is not None:
            dt = now - prev_t
            if dt > 0:
                measured = (_wrap(att.roll_deg - prev[0]) / dt,
                            _wrap(att.pitch_deg - prev[1]) / dt,
                            _wrap(att.yaw_deg - prev[2]) / dt)
                for i in range(3):
                    if abs(euler[i]) >= args.rate_threshold:
                        if euler[i] * measured[i] > 0:
                            agree[i] += 1
                        else:
                            disagree[i] += 1
        prev = (att.roll_deg, att.pitch_deg, att.yaw_deg)
        prev_t = now

        print(f"\rroll={att.roll_deg:+6.1f} pitch={att.pitch_deg:+6.1f} yaw={att.yaw_deg:5.1f}  "
              f"gyro euler r/p/y={euler[0]:+6.0f}/{euler[1]:+6.0f}/{euler[2]:+6.0f} dps   ", end="")
        time.sleep(0.02)

    print("\n\nSign check (gyro-derived rate vs measured attitude change):")
    ok = True
    for i, name in enumerate(labels):
        total = agree[i] + disagree[i]
        if total < 5:
            print(f"  {name:5s}: not enough motion - rotate this axis more and re-run")
            ok = False
        elif disagree[i] > agree[i]:
            print(f"  {name:5s}: MISMATCH ({agree[i]} agree / {disagree[i]} disagree) "
                  f"-> flip InavSigns.gyro_{name}")
            ok = False
        else:
            print(f"  {name:5s}: OK ({agree[i]} agree / {disagree[i]} disagree)")
    print("\nAll gyro signs confirmed." if ok else
          "\nFix the flagged axes in src/lockon/guidance/attitude.py (InavSigns) and re-run.")


if __name__ == "__main__":
    main()
