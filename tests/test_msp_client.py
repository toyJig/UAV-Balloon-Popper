import pytest

from lockon.msp import codec
from lockon.msp.client import MSPClient, MSPErrorResponse, MSPTimeoutError

from fakes import FakeSerialPort


def test_read_rc_decodes_pilot_sticks_in_receiver_order():
    port = FakeSerialPort()
    port.rc_channels_to_report = [1600, 1400, 1800, 1500, 988, 1500, 1500, 1500]
    client = MSPClient(port)

    assert client.read_rc() == port.rc_channels_to_report


def test_read_rc_converts_internal_order_to_receiver_order_with_aetr_map():
    port = FakeSerialPort()
    # receiver order AETR: CH1 roll, CH2 pitch, CH3 throttle, CH4 yaw, CH5 AUX1
    port.rc_channels_to_report = [1510, 1520, 1100, 1490, 1800, 1500, 1500, 1500]
    client = MSPClient(port)

    channels = client.read_rc()

    # the FC answered in internal order (roll, pitch, YAW, THROTTLE)...
    assert port.internal_order()[:4] == [1510, 1520, 1490, 1100]
    # ...and the client hands back receiver order, so throttle stays at index 2
    assert channels[:5] == [1510, 1520, 1100, 1490, 1800]


def test_read_rc_handles_a_taer_map():
    port = FakeSerialPort()
    port.rcmap = [1, 2, 3, 0]  # "TAER": CH1 throttle, CH2 roll, CH3 pitch, CH4 yaw
    port.rc_channels_to_report = [1100, 1510, 1520, 1490, 1500, 1500, 1500, 1500]
    client = MSPClient(port)

    assert client.read_rc()[:4] == [1100, 1510, 1520, 1490]


def test_internal_to_receiver_order_rejects_a_bad_map():
    with pytest.raises(ValueError):
        codec.internal_to_receiver_order([1500] * 8, [0, 1, 1, 2])


def test_read_altitude_converts_units_and_ignores_the_extra_inav_field():
    port = FakeSerialPort()
    port.altitude_cm_to_report = 457  # ~15 ft
    client = MSPClient(port)

    alt = client.read_altitude()

    assert alt.altitude_cm == 457
    assert round(alt.altitude_ft, 1) == 15.0


def test_request_raises_on_an_fc_error_frame():
    port = FakeSerialPort()
    port.reject_lc_writes = True
    client = MSPClient(port)

    with pytest.raises(MSPErrorResponse):
        client.request(codec.MSP2_INAV_SET_LOGIC_CONDITIONS, bytes(15))


def test_request_times_out_when_the_fc_is_silent():
    port = FakeSerialPort()
    port.respond = False
    client = MSPClient(port)

    with pytest.raises(MSPTimeoutError):
        client.request(codec.MSP_RC, timeout_s=0.02)
