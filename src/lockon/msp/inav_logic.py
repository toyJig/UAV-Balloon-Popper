"""Lock-on control through INAV's programming framework (logic conditions).

Works on stock INAV 6.1.1, which does not include MSP RC override. A chain
of logic conditions (LCs) is entered once through the CLI and saved; at run
time the Pi only rewrites a few operand values in RAM over MSP
(MSP2_INAV_SET_LOGIC_CONDITIONS) and never saves them, so a reboot always
returns to the saved, disengaged values.

The chain (offsets from ``base_index``; INAV runs all LCs at 10 Hz):

    +0  SWITCH     HIGH(RC channel <switch_channel>)            pilot's lock-on switch
    +1  HEARTBEAT  DELTA(VALUE <hb>, VALUE 1)                   Pi changes <hb> every write
    +2             NOT(+1)
    +3             DELAY(+2, VALUE <timeout ms>)                no heartbeat change for timeout
    +4  ALIVE      NOT(+3)
    +5  STATE      ADD(VALUE <state>, VALUE 0)                  Pi writes 0 off, 1 engaged, 2 engaged+locked
    +6             GREATER_THAN(+5, VALUE 0)
    +7             AND(+0, +4)                                  switch on AND Pi alive
    +8  ENGAGED    AND(+7, +6)
    +9  ROLL       FLIGHT_AXIS_ANGLE_OVERRIDE(VALUE 0, VALUE <deg>)   active only if +8
    +10 PITCH      FLIGHT_AXIS_ANGLE_OVERRIDE(VALUE 1, VALUE <deg>)   active only if +8
    +11            EQUAL(+5, VALUE 2)
    +12            AND(+8, +11)
    +13 INDICATOR  RC_CHANNEL_OVERRIDE(VALUE <indicator_channel>, VALUE 2000)  active only if +12

plus one mode range: HORIZON on <indicator_channel> (1700-2100), so the
CRSF flight-mode text on the radio reads "HOR" while a target is locked.
While the angle overrides are active, roll/pitch follow the Pi's targets
whatever the flight mode, so Horizon has no effect on those axes.

Safety properties (from the INAV 6.1.1 source):
  * switch off  -> +0 false -> overrides stop within one LC tick (<= 100 ms),
    decided by the flight controller, not the Pi.
  * Pi dead     -> heartbeat stops changing -> +4 false after timeout + 100-200 ms
    (400-500 ms with the default 300 ms; checked in tests/test_inav_chain_emulation.py).
  * CLI open    -> INAV pauses all LCs -> no overrides.
  * state 0     -> nothing is overridden; the pilot's sticks pass straight through.

Angle sign convention (INAV pid.c): roll positive = right bank. Pitch
NEGATIVE = nose up (``fixedWingLevelTrim`` comments: positive trim means
nose up and is subtracted from the angle target). Confirm both on the bench.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Protocol, Tuple

from . import codec

# Operations (INAV src/main/programming/logic_condition.h)
OP_EQUAL = 1
OP_GREATER_THAN = 2
OP_HIGH = 6
OP_AND = 7
OP_NOT = 12
OP_ADD = 14
OP_RC_CHANNEL_OVERRIDE = 38
OP_FLIGHT_AXIS_ANGLE_OVERRIDE = 45
OP_DELAY = 48
OP_DELTA = 50

# Operand types
OPERAND_VALUE = 0
OPERAND_RC_CHANNEL = 1
OPERAND_LC = 4

MAX_LOGIC_CONDITIONS = 64
NO_ACTIVATOR = -1

# Mode permanent ids (INAV src/main/fc/fc_msp_box.c)
MODE_HORIZON = 2

STATE_OFF = 0
STATE_ENGAGED = 1           # Pi flies (auto-level after a lost lock), no indicator
STATE_ENGAGED_LOCKED = 2    # Pi flies a locked target, indicator on


@dataclass(frozen=True)
class LogicCondition:
    index: int
    enabled: int
    activator: int
    operation: int
    a_type: int
    a_value: int
    b_type: int
    b_value: int
    flags: int = 0

    def to_payload(self) -> bytes:
        """MSP2_INAV_SET_LOGIC_CONDITIONS payload (15 bytes)."""
        return (bytes([self.index, self.enabled, self.activator & 0xFF, self.operation, self.a_type])
                + int(self.a_value).to_bytes(4, "little", signed=True)
                + bytes([self.b_type])
                + int(self.b_value).to_bytes(4, "little", signed=True)
                + bytes([self.flags]))

    @classmethod
    def from_single_payload(cls, index: int, payload: bytes) -> "LogicCondition":
        """Parse an MSP2_INAV_LOGIC_CONDITIONS_SINGLE response (14 bytes)."""
        if len(payload) < 14:
            raise ValueError(f"logic condition payload too short: {len(payload)} bytes")
        activator = payload[1] - 256 if payload[1] > 127 else payload[1]
        return cls(
            index=index, enabled=payload[0], activator=activator, operation=payload[2],
            a_type=payload[3], a_value=int.from_bytes(payload[4:8], "little", signed=True),
            b_type=payload[8], b_value=int.from_bytes(payload[9:13], "little", signed=True),
            flags=payload[13],
        )

    def cli(self) -> str:
        return (f"logic {self.index} {self.enabled} {self.activator} {self.operation} "
                f"{self.a_type} {self.a_value} {self.b_type} {self.b_value} {self.flags}")


@dataclass(frozen=True)
class LockOnChainConfig:
    switch_channel: int = 7        # radio channel (1-based) of the lock-on switch
    indicator_channel: int = 12    # spare radio channel the chain drives for the indicator
    base_index: int = 0            # first LC slot used
    aux_slot: int = 2              # mode-range slot for the indicator's HORIZON range
    heartbeat_timeout_ms: int = 300
    max_roll_deg: int = 45         # INAV max_angle_inclination_rll / 10
    max_pitch_deg: int = 30        # INAV max_angle_inclination_pit / 10
    roll_sign: int = 1             # +1: positive roll target = right bank
    pitch_sign: int = -1           # INAV: negative pitch target = nose up

    def __post_init__(self):
        if not (5 <= self.switch_channel <= 18 and 5 <= self.indicator_channel <= 18):
            raise ValueError("switch and indicator must be AUX channels (radio channels 5-18)")
        if self.switch_channel == self.indicator_channel:
            raise ValueError("switch and indicator channels must differ")
        if not 0 <= self.base_index <= MAX_LOGIC_CONDITIONS - CHAIN_LENGTH:
            raise ValueError(f"base_index must leave room for {CHAIN_LENGTH} logic conditions")


CHAIN_LENGTH = 14
# offsets of the LCs whose operand the Pi rewrites at run time, and which operand
HEARTBEAT, STATE, ROLL, PITCH = 1, 5, 9, 10
SWITCH, ALIVE, ENGAGED, INDICATOR_ACTIVE = 0, 4, 8, 12
PI_WRITTEN = {HEARTBEAT: "a_value", STATE: "a_value", ROLL: "b_value", PITCH: "b_value"}


def build_chain(cfg: LockOnChainConfig) -> List[LogicCondition]:
    """The saved (disengaged) form of the chain."""
    b = cfg.base_index

    def lc(off, op, a_type, a_value, b_type=OPERAND_VALUE, b_value=0, activator=NO_ACTIVATOR):
        return LogicCondition(b + off, 1, activator, op, a_type, a_value, b_type, b_value)

    return [
        lc(0, OP_HIGH, OPERAND_RC_CHANNEL, cfg.switch_channel),
        lc(1, OP_DELTA, OPERAND_VALUE, 0, OPERAND_VALUE, 1),
        lc(2, OP_NOT, OPERAND_LC, b + 1),
        lc(3, OP_DELAY, OPERAND_LC, b + 2, OPERAND_VALUE, cfg.heartbeat_timeout_ms),
        lc(4, OP_NOT, OPERAND_LC, b + 3),
        lc(5, OP_ADD, OPERAND_VALUE, STATE_OFF, OPERAND_VALUE, 0),
        lc(6, OP_GREATER_THAN, OPERAND_LC, b + 5, OPERAND_VALUE, 0),
        lc(7, OP_AND, OPERAND_LC, b + 0, OPERAND_LC, b + 4),
        lc(8, OP_AND, OPERAND_LC, b + 7, OPERAND_LC, b + 6),
        lc(9, OP_FLIGHT_AXIS_ANGLE_OVERRIDE, OPERAND_VALUE, 0, OPERAND_VALUE, 0, activator=b + 8),
        lc(10, OP_FLIGHT_AXIS_ANGLE_OVERRIDE, OPERAND_VALUE, 1, OPERAND_VALUE, 0, activator=b + 8),
        lc(11, OP_EQUAL, OPERAND_LC, b + 5, OPERAND_VALUE, STATE_ENGAGED_LOCKED),
        lc(12, OP_AND, OPERAND_LC, b + 8, OPERAND_LC, b + 11),
        lc(13, OP_RC_CHANNEL_OVERRIDE, OPERAND_VALUE, cfg.indicator_channel, OPERAND_VALUE, 2000, activator=b + 12),
    ]


def setup_cli_commands(cfg: LockOnChainConfig, msp_uart_identifier: Optional[int] = None) -> List[str]:
    """CLI lines to paste once (then `save`)."""
    lines = []
    if msp_uart_identifier is not None:
        lines.append(f"serial {msp_uart_identifier} 1 115200 115200 0 115200")
    lines += [lc.cli() for lc in build_chain(cfg)]
    # aux <slot> <mode permanent id> <aux channel index> <start> <end>; AUX index 0 = radio channel 5
    lines.append(f"aux {cfg.aux_slot} {MODE_HORIZON} {cfg.indicator_channel - 5} 1700 2100")
    lines.append("save")
    return lines


class ChainMismatchError(RuntimeError):
    """The FC's logic conditions don't match the expected chain."""


class MSPRequester(Protocol):
    def request(self, function: int, payload: bytes = b"") -> codec.MSPFrame: ...


class InavLockOnLink:
    """Run-time side: verifies the chain on the FC, then writes heartbeat,
    state and angle targets."""

    def __init__(self, msp: MSPRequester, cfg: LockOnChainConfig = LockOnChainConfig()):
        self._msp = msp
        self.cfg = cfg
        self._chain = {lc.index: lc for lc in build_chain(cfg)}
        self._heartbeat = 0
        self.last_written: Tuple[int, int, int] = (STATE_OFF, 0, 0)

    def index(self, offset: int) -> int:
        return self.cfg.base_index + offset

    def read_condition(self, index: int) -> LogicCondition:
        frame = self._msp.request(codec.MSP2_INAV_LOGIC_CONDITIONS_SINGLE, bytes([index]))
        return LogicCondition.from_single_payload(index, frame.payload)

    def verify(self) -> None:
        """Raise ChainMismatchError unless every chain slot on the FC matches
        the expected structure (Pi-written operand values are not compared).
        Prevents the Pi from overwriting logic conditions it doesn't own."""
        problems = []
        for index, expected in self._chain.items():
            actual = self.read_condition(index)
            field_name = PI_WRITTEN.get(index - self.cfg.base_index)
            if field_name is not None:
                actual = replace(actual, **{field_name: getattr(expected, field_name)})
            if actual != expected:
                problems.append(f"  LC {index}: on FC `{actual.cli()}` expected `{expected.cli()}`")
        if problems:
            raise ChainMismatchError(
                "logic conditions on the flight controller don't match the lock-on chain:\n"
                + "\n".join(problems)
                + "\nPaste the output of tools/inav_lockon_setup.py into the CLI and save, then retry."
            )

    def _write(self, offset: int, value: int) -> None:
        index = self.index(offset)
        lc = replace(self._chain[index], **{PI_WRITTEN[offset]: int(value)})
        self._msp.request(codec.MSP2_INAV_SET_LOGIC_CONDITIONS, lc.to_payload())

    def write(self, state: int, roll_deg: float = 0.0, pitch_deg: float = 0.0) -> None:
        """Send one update. Must be called at least every heartbeat_timeout_ms
        (300 ms by default) or the FC treats the Pi as dead.

        ``roll_deg``: positive = bank right. ``pitch_deg``: positive = nose
        UP (converted to INAV's sign here)."""
        if state not in (STATE_OFF, STATE_ENGAGED, STATE_ENGAGED_LOCKED):
            raise ValueError(f"invalid state {state}")
        roll = max(-self.cfg.max_roll_deg, min(self.cfg.max_roll_deg, round(roll_deg)))
        pitch = max(-self.cfg.max_pitch_deg, min(self.cfg.max_pitch_deg, round(pitch_deg)))
        if state == STATE_OFF:
            roll = pitch = 0
        self._heartbeat = (self._heartbeat + 1) % 1000
        # targets first, then state, then heartbeat: the FC never sees a new
        # state paired with stale targets within one LC tick
        self._write(ROLL, roll * self.cfg.roll_sign)
        self._write(PITCH, pitch * self.cfg.pitch_sign)
        self._write(STATE, state)
        self._write(HEARTBEAT, self._heartbeat)
        self.last_written = (state, roll, pitch)

    def release(self) -> None:
        """Disengage (used on shutdown). Stops nothing else: the heartbeat
        simply stops, so the FC also times the Pi out."""
        self._write(STATE, STATE_OFF)
        self._write(ROLL, 0)
        self._write(PITCH, 0)
        self.last_written = (STATE_OFF, 0, 0)

    def read_status(self) -> Dict[str, int]:
        """Current values of the key LCs on the FC (for bench tests)."""
        frame = self._msp.request(codec.MSP2_INAV_LOGIC_CONDITIONS_STATUS)
        values = [int.from_bytes(frame.payload[i:i + 4], "little", signed=True)
                  for i in range(0, len(frame.payload), 4)]
        return {name: values[self.index(off)] for name, off in
                (("switch", SWITCH), ("alive", ALIVE), ("state", STATE), ("engaged", ENGAGED),
                 ("indicator", INDICATOR_ACTIVE))}
