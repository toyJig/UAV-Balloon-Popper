#!/usr/bin/env python3
"""Tasks 2 and 3 bench test for the INAV logic-condition link. PROPS OFF.

Runs from the Pi (port /dev/ttyAMA0, wired to UART3) or from a laptop on
any MSP port of the flight controller. The INAV CLI must be CLOSED on every
port: INAV pauses logic conditions while the CLI is open.

Modes:
  --idle          heartbeat only, state "off" (Task 3). Your sticks must
                  behave exactly as without the Pi, switch on or off.
  --sweep         engaged + locked, sweeping the bank target +-20 deg and
                  the pitch target +-10 deg (Task 2). Surfaces should follow
                  only while the lock-on switch is on.
  --hold R P      engaged + locked at fixed targets (degrees; P positive =
                  nose up), for checking surface directions.
  --crash-after S with --sweep/--hold: stop sending after S seconds while
                  still polling, to simulate the Pi dying. "alive" and
                  "engaged" must drop to 0 within about half a second.

Prints the flight controller's own view every 0.5 s: switch, alive,
engaged, indicator (1 = true).

Usage:
    python3 tools/bench_logic_test.py --port /dev/ttyAMA0 --idle
    python3 tools/bench_logic_test.py --port /dev/ttyAMA0 --sweep
    python3 tools/bench_logic_test.py --port COM5 --hold 15 0
    python3 tools/bench_logic_test.py --port /dev/ttyAMA0 --sweep --crash-after 10
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import serial  # type: ignore  # noqa: E402

from lockon.msp.client import MSPClient, MSPTimeoutError  # noqa: E402
from lockon.msp.inav_logic import (  # noqa: E402
    STATE_ENGAGED_LOCKED, STATE_OFF, InavLockOnLink, LockOnChainConfig,
)

LOOP_HZ = 15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--idle", action="store_true")
    mode.add_argument("--sweep", action="store_true")
    mode.add_argument("--hold", nargs=2, type=float, metavar=("ROLL_DEG", "PITCH_DEG"))
    parser.add_argument("--crash-after", type=float, default=None)
    parser.add_argument("--switch-channel", type=int, default=7)
    parser.add_argument("--indicator-channel", type=int, default=12)
    args = parser.parse_args()

    cfg = LockOnChainConfig(switch_channel=args.switch_channel, indicator_channel=args.indicator_channel)
    client = MSPClient(serial.Serial(args.port, args.baud, timeout=0.05))
    link = InavLockOnLink(client, cfg)
    link.verify()
    print("Logic-condition chain verified on the flight controller. PROPS OFF. Ctrl+C to stop.")

    start = time.monotonic()
    last_print = -1.0
    crashed = False
    try:
        while True:
            t = time.monotonic() - start
            if args.crash_after is not None and t >= args.crash_after and not crashed:
                crashed = True
                print(f"[t={t:5.1f}s] simulating Pi crash: no more writes")
            if not crashed:
                if args.idle:
                    link.write(STATE_OFF)
                elif args.sweep:
                    link.write(STATE_ENGAGED_LOCKED,
                               roll_deg=20 * math.sin(2 * math.pi * t / 4.0),
                               pitch_deg=10 * math.sin(2 * math.pi * t / 6.0))
                else:
                    link.write(STATE_ENGAGED_LOCKED, roll_deg=args.hold[0], pitch_deg=args.hold[1])
            if t - last_print >= 0.5:
                last_print = t
                try:
                    s = link.read_status()
                    state, roll, pitch = link.last_written
                    print(f"[t={t:5.1f}s] FC: switch={s['switch']} alive={s['alive']} engaged={s['engaged']} "
                          f"indicator={s['indicator']} | Pi sent state={state} roll={roll:+d} pitch={pitch:+d}")
                except MSPTimeoutError as exc:
                    print(f"[t={t:5.1f}s] warning: {exc}")
            time.sleep(1.0 / LOOP_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            link.release()
        except Exception as exc:  # the FC still times the Pi out on its own
            print(f"release failed ({exc}); the flight controller will time the Pi out by itself")
        print("Stopped; state set to off.")


if __name__ == "__main__":
    main()
