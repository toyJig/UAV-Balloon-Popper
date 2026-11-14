"""End-to-end check of the lock-on logic-condition chain against a small
emulator of INAV 6.1.1's programming framework (logic_condition.c):

* all LCs run in index order once per 100 ms tick (TASK_PROGRAMMING_FRAMEWORK, 10 Hz)
* an LC runs only if enabled and its activator LC is true (-1 = always);
  otherwise its value is forced to false
* override flags are cleared at the start of every tick
* LC operands read the current value of LCs already processed this tick

The Pi side is the real InavLockOnLink writing through the fake serial port,
so this exercises the exact records the Pi would send.
"""

from fakes import FakeSerialPort

from lockon.msp.client import MSPClient
from lockon.msp.inav_logic import (
    OP_ADD, OP_AND, OP_DELAY, OP_DELTA, OP_EQUAL, OP_FLIGHT_AXIS_ANGLE_OVERRIDE, OP_GREATER_THAN, OP_HIGH,
    OP_NOT, OP_RC_CHANNEL_OVERRIDE, OPERAND_LC, OPERAND_RC_CHANNEL, OPERAND_VALUE,
    STATE_ENGAGED, STATE_ENGAGED_LOCKED, STATE_OFF, InavLockOnLink, LockOnChainConfig, LogicCondition, build_chain,
)

CFG = LockOnChainConfig()
TICK_MS = 100


class InavLogicEmulator:
    def __init__(self, port: FakeSerialPort, rc_channels):
        self.port = port
        self.rc = rc_channels  # receiver order, 1-based channel n = rc[n-1] (AUX channels are not remapped)
        self.values = [0] * 64
        self.last_value = [0] * 64
        self.timeout = [0] * 64
        self.flags = [0] * 64
        self.angle_override = {}
        self.rc_override = {}

    def _lc(self, i):
        return LogicCondition.from_single_payload(i, self.port.logic[i])

    def _operand(self, t, v):
        if t == OPERAND_VALUE:
            return v
        if t == OPERAND_RC_CHANNEL:
            return self.rc[v - 1]
        if t == OPERAND_LC:
            return self.values[v]
        raise NotImplementedError(t)

    def tick(self, now_ms):
        self.angle_override, self.rc_override = {}, {}
        for i in range(64):
            lc = self._lc(i)
            active = lc.enabled and (lc.activator < 0 or self.values[lc.activator])
            if not active:
                self.values[i] = 0
                continue
            a, b = self._operand(lc.a_type, lc.a_value), self._operand(lc.b_type, lc.b_value)
            op = lc.operation
            if op == OP_HIGH:
                r = int(a > 1666)
            elif op == OP_DELTA:
                diff = self.last_value[i] - a
                self.last_value[i] = a
                r = int(abs(diff) >= b)
            elif op == OP_NOT:
                r = int(not a)
            elif op == OP_DELAY:
                r = 0
                if a:
                    if self.timeout[i] == 0:
                        self.timeout[i] = now_ms + b
                    elif now_ms > self.timeout[i]:
                        self.flags[i] = 1
                        r = 1
                    elif self.flags[i]:
                        r = 1
                else:
                    self.timeout[i] = 0
                    self.flags[i] = 0
            elif op == OP_ADD:
                r = a + b
            elif op == OP_GREATER_THAN:
                r = int(a > b)
            elif op == OP_AND:
                r = int(bool(a and b))
            elif op == OP_EQUAL:
                r = int(a == b)
            elif op == OP_FLIGHT_AXIS_ANGLE_OVERRIDE:
                self.angle_override[a] = b
                r = 1
            elif op == OP_RC_CHANNEL_OVERRIDE:
                self.rc_override[a] = b
                r = 1
            elif op == 0 and not lc.enabled:
                r = 0
            else:
                raise NotImplementedError(op)
            self.values[i] = r


def setup(switch_us=2000):
    port = FakeSerialPort()
    port.load_chain(build_chain(CFG))
    rc = [1500] * 16
    rc[CFG.switch_channel - 1] = switch_us
    emu = InavLogicEmulator(port, rc)
    link = InavLockOnLink(MSPClient(port), CFG)
    return emu, link, rc


def run(emu, link, duration_ms, pi_period_ms=66, pi_action=None, start_ms=0):
    """Advance time in 1 ms steps; the Pi acts every pi_period_ms (None =
    Pi silent), the FC ticks every 100 ms. Returns per-tick snapshots."""
    snapshots = []
    next_pi = start_ms
    for t in range(start_ms, start_ms + duration_ms):
        if pi_action is not None and t >= next_pi:
            pi_action(link)
            next_pi += pi_period_ms
        if t % TICK_MS == 0:
            emu.tick(t)
            snapshots.append((t, dict(emu.angle_override), dict(emu.rc_override)))
    return snapshots


LOCKED = lambda link: link.write(STATE_ENGAGED_LOCKED, roll_deg=20, pitch_deg=10)


def test_engages_with_pi_targets_and_indicator_when_switch_on_and_locked():
    emu, link, rc = setup()
    snaps = run(emu, link, 1000, pi_action=LOCKED)
    _, angles, rcs = snaps[-1]
    assert angles == {0: 20, 1: -10}            # roll right 20, pitch nose-up 10 (INAV sign)
    assert rcs == {CFG.indicator_channel: 2000}  # indicator channel driven -> HORIZON -> "HOR"


def test_switch_off_means_no_override_even_while_the_pi_insists():
    emu, link, rc = setup(switch_us=1000)
    snaps = run(emu, link, 1000, pi_action=LOCKED)
    assert all(not a and not r for _, a, r in snaps)


def test_switch_off_cuts_within_one_tick():
    emu, link, rc = setup()
    run(emu, link, 1000, pi_action=LOCKED)
    rc[CFG.switch_channel - 1] = 1000
    snaps = run(emu, link, 200, pi_action=LOCKED, start_ms=1000)
    assert snaps[0][1] == {} and snaps[0][2] == {}


def test_state_off_never_overrides():
    emu, link, rc = setup()
    snaps = run(emu, link, 1000, pi_action=lambda l: l.write(STATE_OFF))
    assert all(not a and not r for _, a, r in snaps)


def test_engaged_without_lock_flies_level_with_indicator_off():
    emu, link, rc = setup()
    snaps = run(emu, link, 1000, pi_action=lambda l: l.write(STATE_ENGAGED, 0, 0))
    _, angles, rcs = snaps[-1]
    assert angles == {0: 0, 1: 0}
    assert rcs == {}


def test_pi_crash_releases_within_half_a_second():
    emu, link, rc = setup()
    run(emu, link, 1000, pi_action=LOCKED)
    snaps = run(emu, link, 1000, pi_action=None, start_ms=1000)  # Pi stops writing; values stay in RAM
    released_at = next(t for t, a, r in snaps if not a and not r)
    assert released_at - 1000 <= 500
    assert all(not a and not r for t, a, r in snaps if t >= released_at)


def test_saved_values_after_reboot_never_engage():
    emu, link, rc = setup()  # chain loaded, Pi never writes
    snaps = run(emu, link, 2000, pi_action=None)
    assert all(not a and not r for _, a, r in snaps)


def test_a_slow_pi_at_6hz_stays_engaged_without_flicker():
    emu, link, rc = setup()
    snaps = run(emu, link, 3000, pi_period_ms=167, pi_action=LOCKED)
    engaged = [bool(a) for _, a, _ in snaps[5:]]  # after start-up
    assert all(engaged)


def test_pi_slower_than_the_timeout_is_treated_as_dead():
    # With heartbeat_timeout_ms=300 the FC declares the Pi dead after 400-500 ms
    # without a change (first stale 100 ms tick + 300 ms + INAV's strict ">").
    emu, link, rc = setup()
    snaps = run(emu, link, 3000, pi_period_ms=700, pi_action=LOCKED)
    assert not all(bool(a) for _, a, _ in snaps[5:])


def test_pi_at_3hz_is_still_alive():
    emu, link, rc = setup()
    snaps = run(emu, link, 3000, pi_period_ms=333, pi_action=LOCKED)
    assert all(bool(a) for _, a, _ in snaps[5:])
