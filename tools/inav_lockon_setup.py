#!/usr/bin/env python3
"""Task 2 - print the INAV CLI commands that set up the lock-on logic
conditions, the indicator mode and the Pi's MSP port. Paste the output into
the INAV CLI (it ends with `save`, which reboots the board).

Defaults match this plane: lock-on switch on CH7, indicator on CH12,
Pi on UART3, mode slot 2 (slots 0-1 hold ARM and ANGLE). INAV serial
identifiers: 0 = UART1, 1 = UART2, 2 = UART3, 3 = UART4, 4 = UART5, 5 = UART6.

Usage:
    python tools/inav_lockon_setup.py
    python tools/inav_lockon_setup.py --switch-channel 8 --indicator-channel 13 --uart 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lockon.msp.inav_logic import LockOnChainConfig, setup_cli_commands  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--switch-channel", type=int, default=7, help="radio channel of the lock-on switch")
    parser.add_argument("--indicator-channel", type=int, default=12, help="spare radio channel for the indicator")
    parser.add_argument("--uart", type=int, default=3, help="UART number the Pi is wired to (1-6)")
    parser.add_argument("--aux-slot", type=int, default=2, help="free `aux` slot for the indicator mode range")
    parser.add_argument("--base-index", type=int, default=0, help="first free `logic` slot")
    args = parser.parse_args()

    cfg = LockOnChainConfig(switch_channel=args.switch_channel, indicator_channel=args.indicator_channel,
                            aux_slot=args.aux_slot, base_index=args.base_index)
    print(f"# lock-on setup: switch CH{cfg.switch_channel}, indicator CH{cfg.indicator_channel} -> HORIZON,"
          f" Pi MSP on UART{args.uart}, logic {cfg.base_index}-{cfg.base_index + 13}")
    for line in setup_cli_commands(cfg, msp_uart_identifier=args.uart - 1):
        print(line)


if __name__ == "__main__":
    main()
