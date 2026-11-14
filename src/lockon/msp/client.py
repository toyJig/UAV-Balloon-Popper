"""Serial client around the MSP codec.

Talks to the INAV flight controller over the UART set to MSP (UART3 on this
build). ``pyserial`` is the only hardware dependency, and it is injected
through the ``serial_port`` constructor argument so this class is fully
unit-testable with a fake/in-memory port (see tests/fakes.py).

Channel order: channel lists returned by ``read_rc`` are in RECEIVER order
(CH1, CH2, ... as the radio sends them). MSP_RC answers in the FC's
INTERNAL order (Roll, Pitch, Yaw, Throttle, AUX1...), so ``read_rc``
converts using the map from MSP_RX_MAP. With this plane's "AETR" map:

    index 0: Roll (CH1), 1: Pitch (CH2), 2: Throttle (CH3), 3: Yaw (CH4)
    index 4..: AUX channels (index 4 = CH5 = AUX1, index 6 = CH7, ...)

Steering is done through INAV logic conditions (see inav_logic.py), not by
sending RC values.
"""

from __future__ import annotations

import time
from typing import List, Optional, Protocol

from . import codec

# Receiver-order indices for the "AETR" map.
ROLL = 0
PITCH = 1
THROTTLE = 2
YAW = 3

CENTER_US = 1500

REQUEST_TIMEOUT_S = 0.25


class SerialPort(Protocol):
    """The subset of pyserial's Serial interface this client needs, so
    tests can swap in a fake."""

    def write(self, data: bytes) -> int: ...

    def read(self, size: int = 1) -> bytes: ...

    @property
    def in_waiting(self) -> int: ...


class MSPTimeoutError(TimeoutError):
    """Raised when the FC doesn't answer a request in time."""


class MSPErrorResponse(RuntimeError):
    """Raised when the FC answers a request with an MSP error frame
    (unknown command, or a rejected payload)."""


class MSPClient:
    def __init__(self, serial_port: SerialPort):
        self._port = serial_port
        self._parser = codec.FrameParser()
        self._rcmap: Optional[List[int]] = None

    def request(self, function: int, payload: bytes = b"", timeout_s: float = REQUEST_TIMEOUT_S) -> codec.MSPFrame:
        """Send one request (v1 or v2, chosen from the function id) and wait
        for the FC's answer to it. Write commands are answered with an empty
        acknowledgement, so this also confirms writes."""
        self._port.write(codec.encode_request(function, payload))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            n = self._port.in_waiting
            chunk = self._port.read(n if n > 0 else 1)
            if chunk:
                for frame in self._parser.feed(chunk):
                    if frame.function == function:
                        if frame.is_error:
                            raise MSPErrorResponse(f"FC rejected MSP function {function:#x}")
                        return frame
        raise MSPTimeoutError(f"no response to MSP function {function:#x} within {timeout_s}s")

    _request = request  # backwards-compatible alias

    def read_rx_map(self) -> List[int]:
        """Read (once, then cached) the FC's channel map from MSP_RX_MAP."""
        if self._rcmap is None:
            frame = self.request(codec.MSP_RX_MAP)
            self._rcmap = list(frame.payload)
        return self._rcmap

    def read_rc(self) -> List[int]:
        """Return the FC's current RC channel values (us) in receiver order.

        Note: values are read after INAV applies logic-condition RC channel
        overrides, so the indicator channel reads 2000 while it is driven."""
        frame = self.request(codec.MSP_RC)
        internal = codec.decode_rc_channels(frame.payload)
        return codec.internal_to_receiver_order(internal, self.read_rx_map())

    def read_attitude(self) -> codec.RawAttitude:
        """INAV's attitude estimate (gyro-integrated, accelerometer-corrected)."""
        return codec.decode_attitude(self.request(codec.MSP_ATTITUDE).payload)

    def read_gyro_dps(self) -> "tuple[int, int, int]":
        """Body rotation rates straight from the gyro, deg/s, INAV signs."""
        return codec.decode_gyro_dps(self.request(codec.MSP_RAW_IMU).payload)

    def read_altitude(self) -> codec.Altitude:
        """Altitude estimate in cm. On INAV this is relative to where the
        board was powered up / armed, not above sea level."""
        frame = self.request(codec.MSP_ALTITUDE)
        return codec.decode_altitude(frame.payload)
