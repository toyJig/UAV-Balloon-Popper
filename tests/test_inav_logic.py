import pytest

from lockon.msp import codec
from lockon.msp.client import MSPClient
from lockon.msp.inav_logic import (
    CHAIN_LENGTH, HEARTBEAT, PITCH, ROLL, STATE, STATE_ENGAGED_LOCKED, STATE_OFF,
    ChainMismatchError, InavLockOnLink, LockOnChainConfig, LogicCondition, build_chain, setup_cli_commands,
)

from fakes import FakeSerialPort


def make_link(cfg=LockOnChainConfig(), loaded=True):
    port = FakeSerialPort()
    if loaded:
        port.load_chain(build_chain(cfg))
    return InavLockOnLink(MSPClient(port), cfg), port


def decoded_writes(port):
    return [(i, LogicCondition.from_single_payload(i, rec)) for i, rec in port.lc_writes]


def test_logic_condition_payload_roundtrip_including_negative_values():
    lc = LogicCondition(10, 1, -1, 45, 0, 1, 0, -12, 0)
    payload = lc.to_payload()

    assert len(payload) == 15
    assert LogicCondition.from_single_payload(10, payload[1:]) == lc


def test_cli_line_matches_inav_logic_syntax():
    lc = LogicCondition(3, 1, -1, 48, 4, 2, 0, 300, 0)
    assert lc.cli() == "logic 3 1 -1 48 4 2 0 300 0"


def test_setup_commands_cover_serial_chain_mode_and_save():
    cfg = LockOnChainConfig(switch_channel=7, indicator_channel=12, aux_slot=2)
    lines = setup_cli_commands(cfg, msp_uart_identifier=2)

    assert lines[0] == "serial 2 1 115200 115200 0 115200"
    assert sum(1 for l in lines if l.startswith("logic ")) == CHAIN_LENGTH
    assert "logic 0 1 -1 6 1 7 0 0 0" in lines              # HIGH(RC channel 7)
    assert "logic 13 1 12 38 0 12 0 2000 0" in lines        # drive CH12 when locked
    assert "aux 2 2 7 1700 2100" in lines                   # HORIZON on CH12 (AUX index 7)
    assert lines[-1] == "save"


def test_chain_references_stay_consistent_with_a_nonzero_base_index():
    chain = build_chain(LockOnChainConfig(base_index=20))
    assert [lc.index for lc in chain] == list(range(20, 34))
    roll = chain[ROLL]
    assert roll.activator == 28  # ENGAGED at base+8


def test_config_rejects_overlapping_or_non_aux_channels():
    with pytest.raises(ValueError):
        LockOnChainConfig(switch_channel=3)
    with pytest.raises(ValueError):
        LockOnChainConfig(switch_channel=7, indicator_channel=7)
    with pytest.raises(ValueError):
        LockOnChainConfig(base_index=60)


def test_verify_passes_when_the_chain_is_loaded():
    link, port = make_link()
    link.verify()


def test_verify_ignores_values_the_pi_wrote_earlier():
    link, port = make_link()
    link.write(STATE_ENGAGED_LOCKED, 10, 5)
    link.verify()


def test_verify_refuses_an_empty_or_different_chain():
    link, port = make_link(loaded=False)
    with pytest.raises(ChainMismatchError) as exc:
        link.verify()
    assert "LC 0" in str(exc.value)


def test_verify_detects_a_wrong_switch_channel():
    link, port = make_link()
    port.load_chain([LogicCondition(0, 1, -1, 6, 1, 8, 0, 0, 0)])  # switch on CH8 instead of CH7
    with pytest.raises(ChainMismatchError):
        link.verify()


def test_write_sends_targets_then_state_then_heartbeat_with_inav_pitch_sign():
    link, port = make_link()

    link.write(STATE_ENGAGED_LOCKED, roll_deg=20, pitch_deg=10)  # bank right, nose up

    writes = decoded_writes(port)
    assert [i for i, _ in writes] == [ROLL, PITCH, STATE, HEARTBEAT]
    assert writes[0][1].b_value == 20
    assert writes[1][1].b_value == -10   # INAV: negative pitch target = nose up
    assert writes[2][1].a_value == STATE_ENGAGED_LOCKED
    assert writes[3][1].a_value == 1


def test_heartbeat_changes_on_every_write():
    link, port = make_link()
    for _ in range(3):
        link.write(STATE_OFF)
    beats = [lc.a_value for i, lc in decoded_writes(port) if i == HEARTBEAT]
    assert beats == [1, 2, 3]


def test_write_clamps_to_the_configured_max_angles():
    link, port = make_link(LockOnChainConfig(max_roll_deg=45, max_pitch_deg=30))
    link.write(STATE_ENGAGED_LOCKED, roll_deg=-90, pitch_deg=50)
    assert link.last_written == (STATE_ENGAGED_LOCKED, -45, 30)


def test_state_off_zeroes_the_targets():
    link, port = make_link()
    link.write(STATE_OFF, roll_deg=30, pitch_deg=20)
    assert link.last_written == (STATE_OFF, 0, 0)


def test_write_rejects_an_unknown_state():
    link, port = make_link()
    with pytest.raises(ValueError):
        link.write(7)


def test_release_writes_state_off_and_neutral_targets():
    link, port = make_link()
    link.write(STATE_ENGAGED_LOCKED, 20, 10)
    port.lc_writes.clear()

    link.release()

    by_index = dict(decoded_writes(port))
    assert by_index[STATE].a_value == STATE_OFF
    assert by_index[ROLL].b_value == 0 and by_index[PITCH].b_value == 0


def test_writes_keep_the_saved_structure_of_each_condition():
    link, port = make_link()
    link.write(STATE_ENGAGED_LOCKED, 15, -5)
    chain = {lc.index: lc for lc in build_chain(LockOnChainConfig())}
    for i, lc in decoded_writes(port):
        expected = chain[i]
        assert (lc.enabled, lc.activator, lc.operation, lc.a_type, lc.b_type, lc.flags) == \
               (expected.enabled, expected.activator, expected.operation, expected.a_type, expected.b_type, expected.flags)


def test_read_status_maps_the_key_conditions():
    link, port = make_link()
    port.lc_status[0] = 1   # switch
    port.lc_status[4] = 1   # alive
    port.lc_status[5] = 2   # state value
    port.lc_status[8] = 1   # engaged
    port.lc_status[12] = 1  # indicator

    assert link.read_status() == {"switch": 1, "alive": 1, "state": 2, "engaged": 1, "indicator": 1}
