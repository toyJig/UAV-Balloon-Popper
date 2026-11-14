"""MSP v1 and v2 frame encode/decode.

MSP v1 (function ids < 256: MSP_RC, MSP_RX_MAP, MSP_ALTITUDE):

    '$' 'M' <dir> <size:u8> <function:u8> <payload...> <checksum>
    checksum = XOR of (size, function, *payload)

MSP v2 (16-bit function ids, needed for INAV's MSP2_INAV_* messages):

    '$' 'X' <dir> <flags:u8> <function:u16 LE> <size:u16 LE> <payload...> <crc>
    crc = CRC-8/DVB-S2 (poly 0xD5) over flags, function, size and payload

<dir> is '<' for requests (companion computer -> FC), '>' for responses,
'!' for error responses.

References: INAV src/main/msp/msp_serial.c and src/main/common/crc.c.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

HEADER_V1 = b"$M"
HEADER_V2 = b"$X"
HEADER = HEADER_V1  # kept for backwards compatibility

DIR_REQUEST = ord("<")
DIR_RESPONSE = ord(">")
DIR_ERROR = ord("!")

# MSP v1 function codes used by this project.
MSP_RX_MAP = 64
MSP_RAW_IMU = 102
MSP_RC = 105
MSP_ATTITUDE = 108
MSP_ALTITUDE = 109
MSP_SET_RAW_RC = 200

# MSP v2 (INAV) function codes.
MSP2_INAV_SET_LOGIC_CONDITIONS = 0x2023
MSP2_INAV_LOGIC_CONDITIONS_STATUS = 0x2026
MSP2_INAV_LOGIC_CONDITIONS_SINGLE = 0x203B

MAX_PAYLOAD_SIZE = 255
MAX_PAYLOAD_SIZE_V2 = 4096


class MSPChecksumError(ValueError):
    """Raised when a decoded frame's checksum does not match."""


@dataclass(frozen=True)
class MSPFrame:
    direction: int
    function: int
    payload: bytes
    version: int = 1

    @property
    def is_error(self) -> bool:
        return self.direction == DIR_ERROR


def _checksum(size: int, function: int, payload: bytes) -> int:
    chk = size ^ function
    for b in payload:
        chk ^= b
    return chk & 0xFF


def crc8_dvb_s2(data: bytes, crc: int = 0) -> int:
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode_frame(function: int, payload: bytes = b"", direction: int = DIR_REQUEST) -> bytes:
    """Encode a single MSP v1 frame ready to write to the serial port."""
    if len(payload) > MAX_PAYLOAD_SIZE:
        raise ValueError(f"MSP v1 payload too large: {len(payload)} bytes (max {MAX_PAYLOAD_SIZE})")
    if not 0 <= function <= 255:
        raise ValueError(f"MSP v1 function id must fit in one byte, got {function}; use encode_frame_v2")
    size = len(payload)
    out = bytearray()
    out += HEADER_V1
    out.append(direction)
    out.append(size)
    out.append(function)
    out += payload
    out.append(_checksum(size, function, payload))
    return bytes(out)


def encode_frame_v2(function: int, payload: bytes = b"", direction: int = DIR_REQUEST, flags: int = 0) -> bytes:
    """Encode a single MSP v2 frame."""
    if len(payload) > MAX_PAYLOAD_SIZE_V2:
        raise ValueError(f"MSP v2 payload too large: {len(payload)} bytes")
    body = bytes([flags & 0xFF]) + function.to_bytes(2, "little") + len(payload).to_bytes(2, "little") + payload
    return HEADER_V2 + bytes([direction]) + body + bytes([crc8_dvb_s2(body)])


def encode_request(function: int, payload: bytes = b"") -> bytes:
    """Encode a request frame (companion computer -> FC), picking v1 or v2
    from the function id."""
    if function > 255:
        return encode_frame_v2(function, payload, direction=DIR_REQUEST)
    return encode_frame(function, payload, direction=DIR_REQUEST)


def encode_rc_channels(channel_values_us: List[int]) -> bytes:
    """Encode an MSP_SET_RAW_RC request frame from a list of channel values
    in microseconds (typically 1000-2000, centre ~1500).
    """
    payload = bytearray()
    for value in channel_values_us:
        v = int(value) & 0xFFFF
        payload += v.to_bytes(2, "little")
    return encode_request(MSP_SET_RAW_RC, bytes(payload))


def decode_rc_channels(payload: bytes) -> List[int]:
    """Decode an MSP_RC response payload into a list of channel values (us)."""
    if len(payload) % 2 != 0:
        raise ValueError(f"MSP_RC payload length must be even, got {len(payload)}")
    return [int.from_bytes(payload[i:i + 2], "little") for i in range(0, len(payload), 2)]


def internal_to_receiver_order(internal: List[int], rcmap: List[int]) -> List[int]:
    """Convert channel values from the FC's internal order (what MSP_RC
    returns: Roll, Pitch, Yaw, Throttle, AUX1...) to receiver order (what
    MSP_SET_RAW_RC expects, e.g. A, E, T, R, AUX1... for the common "AETR"
    map). ``rcmap`` is the MSP_RX_MAP payload: rcmap[i] is the receiver
    channel that feeds internal channel i. Channels past the map are AUX
    channels, which are never remapped."""
    n = len(rcmap)
    if sorted(rcmap) != list(range(n)):
        raise ValueError(f"MSP_RX_MAP is not a permutation of 0..{n - 1}: {rcmap}")
    if len(internal) < n:
        raise ValueError(f"need at least {n} channels to apply the map, got {len(internal)}")
    receiver = list(internal)
    for i, raw_index in enumerate(rcmap):
        receiver[raw_index] = internal[i]
    return receiver


@dataclass(frozen=True)
class Altitude:
    altitude_cm: int
    vario_cm_s: int

    @property
    def altitude_m(self) -> float:
        return self.altitude_cm / 100.0

    @property
    def altitude_ft(self) -> float:
        return self.altitude_cm / 30.48


def decode_altitude(payload: bytes) -> Altitude:
    """Decode an MSP_ALTITUDE response payload: int32 altitude (cm), int16
    vario (cm/s). INAV appends an int32 raw baro altitude, which is ignored."""
    if len(payload) < 6:
        raise ValueError(f"MSP_ALTITUDE payload too short: {len(payload)} bytes")
    altitude_cm = int.from_bytes(payload[0:4], "little", signed=True)
    vario_cm_s = int.from_bytes(payload[4:6], "little", signed=True)
    return Altitude(altitude_cm=altitude_cm, vario_cm_s=vario_cm_s)


@dataclass(frozen=True)
class RawAttitude:
    """MSP_ATTITUDE as INAV sends it: roll and pitch in decidegrees, yaw
    (heading) in whole degrees. INAV convention (from its angle controller,
    where stick targets and attitude are compared directly): roll positive =
    right wing down, pitch positive = nose DOWN, heading 0-360."""
    roll_decideg: int
    pitch_decideg: int
    yaw_deg: int


def decode_attitude(payload: bytes) -> RawAttitude:
    if len(payload) < 6:
        raise ValueError(f"MSP_ATTITUDE payload too short: {len(payload)} bytes")
    v = [int.from_bytes(payload[i:i + 2], "little", signed=True) for i in (0, 2, 4)]
    return RawAttitude(*v)


def decode_gyro_dps(payload: bytes) -> "tuple[int, int, int]":
    """Gyro rates (deg/s, body axes roll, pitch, yaw, INAV signs) from an
    MSP_RAW_IMU payload: 3 x int16 acc, 3 x int16 gyro, 3 x int16 mag."""
    if len(payload) < 12:
        raise ValueError(f"MSP_RAW_IMU payload too short: {len(payload)} bytes")
    return tuple(int.from_bytes(payload[i:i + 2], "little", signed=True) for i in (6, 8, 10))


class FrameParser:
    """Stateful incremental parser for a byte stream that may contain MSP v1
    and v2 frames split arbitrarily across reads (as serial reads always are).

    Usage:
        parser = FrameParser()
        for frame in parser.feed(data_just_read):
            ...handle frame...
    """

    _WAIT_DOLLAR = 0
    _WAIT_VERSION = 1
    _WAIT_DIRECTION = 2
    # v1
    _V1_SIZE = 3
    _V1_FUNCTION = 4
    _V1_PAYLOAD = 5
    _V1_CHECKSUM = 6
    # v2
    _V2_HEADER = 7   # flags, function lo/hi, size lo/hi
    _V2_PAYLOAD = 8
    _V2_CRC = 9

    def __init__(self) -> None:
        self._reset()
        self.checksum_errors = 0

    def _reset(self) -> None:
        self._state = self._WAIT_DOLLAR
        self._version = 1
        self._direction = 0
        self._size = 0
        self._function = 0
        self._header = bytearray()
        self._payload = bytearray()

    def feed(self, data: bytes) -> List[MSPFrame]:
        """Feed newly-read bytes into the parser and return any complete,
        valid frames found. Frames that fail their checksum are dropped
        (and counted in ``checksum_errors``) rather than raising, so one bad
        byte on the wire can't stop later, valid frames from being parsed.
        """
        frames: List[MSPFrame] = []
        for byte in data:
            try:
                frame = self._feed_byte(byte)
            except MSPChecksumError:
                continue
            if frame is not None:
                frames.append(frame)
        return frames

    def _feed_byte(self, byte: int) -> Optional[MSPFrame]:
        s = self._state
        if s == self._WAIT_DOLLAR:
            if byte == ord("$"):
                self._state = self._WAIT_VERSION
            return None

        if s == self._WAIT_VERSION:
            if byte == ord("M"):
                self._version, self._state = 1, self._WAIT_DIRECTION
            elif byte == ord("X"):
                self._version, self._state = 2, self._WAIT_DIRECTION
            else:
                self._reset()
            return None

        if s == self._WAIT_DIRECTION:
            if byte in (DIR_REQUEST, DIR_RESPONSE, DIR_ERROR):
                self._direction = byte
                self._payload = bytearray()
                self._header = bytearray()
                self._state = self._V1_SIZE if self._version == 1 else self._V2_HEADER
            else:
                self._reset()
            return None

        # ---- v1 ----
        if s == self._V1_SIZE:
            self._size = byte
            self._state = self._V1_FUNCTION
            return None

        if s == self._V1_FUNCTION:
            self._function = byte
            self._state = self._V1_PAYLOAD if self._size > 0 else self._V1_CHECKSUM
            return None

        if s == self._V1_PAYLOAD:
            self._payload.append(byte)
            if len(self._payload) >= self._size:
                self._state = self._V1_CHECKSUM
            return None

        if s == self._V1_CHECKSUM:
            expected = _checksum(self._size, self._function, bytes(self._payload))
            frame = MSPFrame(self._direction, self._function, bytes(self._payload), version=1)
            self._reset()
            if byte != expected:
                self.checksum_errors += 1
                raise MSPChecksumError(
                    f"checksum mismatch for function {frame.function}: expected {expected:#x}, got {byte:#x}"
                )
            return frame

        # ---- v2 ----
        if s == self._V2_HEADER:
            self._header.append(byte)
            if len(self._header) == 5:
                self._function = int.from_bytes(self._header[1:3], "little")
                self._size = int.from_bytes(self._header[3:5], "little")
                if self._size > MAX_PAYLOAD_SIZE_V2:
                    self._reset()
                    return None
                self._state = self._V2_PAYLOAD if self._size > 0 else self._V2_CRC
            return None

        if s == self._V2_PAYLOAD:
            self._payload.append(byte)
            if len(self._payload) >= self._size:
                self._state = self._V2_CRC
            return None

        if s == self._V2_CRC:
            expected = crc8_dvb_s2(bytes(self._header) + bytes(self._payload))
            frame = MSPFrame(self._direction, self._function, bytes(self._payload), version=2)
            self._reset()
            if byte != expected:
                self.checksum_errors += 1
                raise MSPChecksumError(
                    f"CRC mismatch for MSP v2 function {frame.function:#x}: expected {expected:#x}, got {byte:#x}"
                )
            return frame

        self._reset()
        return None
