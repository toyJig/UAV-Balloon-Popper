"""Shared test fakes for hardware interfaces, used across multiple test
modules so unit tests never need a real serial port, camera, or FC."""

from lockon.msp import codec

DEFAULT_LC = bytes([0, 0xFF, 0, 0]) + (0).to_bytes(4, "little") + bytes([0]) + (0).to_bytes(4, "little") + bytes([0])


class FakeSerialPort:
    """In-memory stand-in for pyserial's Serial, with a scriptable "FC side"
    that reacts to whatever MSPClient writes to it, like INAV 6.1.1 would."""

    def __init__(self):
        self.written_frames = []  # raw bytes of every frame the client sent
        self._rx_buffer = bytearray()
        self._parser = codec.FrameParser()
        # What the receiver sends, in RECEIVER order (CH1, CH2, ...). The
        # fake FC answers MSP_RC in internal order, like a real FC.
        self.rc_channels_to_report = [1500] * 8
        self.rcmap = [0, 1, 3, 2]  # "AETR": internal R,P,Y,T <- receiver CH1,CH2,CH4,CH3
        self.altitude_cm_to_report = 500
        self.vario_cm_s_to_report = 0
        # INAV units/signs: roll, pitch decidegrees (pitch + = nose down), yaw degrees
        self.attitude_to_report = (0, 0, 0)
        self.gyro_dps_to_report = (0, 0, 0)  # roll, pitch, yaw rates, INAV signs
        # logic conditions: index -> 14-byte record (as returned by LOGIC_CONDITIONS_SINGLE)
        self.logic = {i: DEFAULT_LC for i in range(64)}
        self.lc_status = [0] * 64
        self.lc_writes = []  # (index, record) for every SET_LOGIC_CONDITIONS received
        self.reject_lc_writes = False
        self.respond = True

    def internal_order(self):
        rx = list(self.rc_channels_to_report)
        internal = list(rx)
        for i, raw_index in enumerate(self.rcmap):
            internal[i] = rx[raw_index]
        return internal

    def write(self, data: bytes) -> int:
        self.written_frames.append(bytes(data))
        for frame in self._parser.feed(data):
            if self.respond:
                self._handle_request(frame)
        return len(data)

    def _reply(self, frame, payload=b"", direction=codec.DIR_RESPONSE):
        if frame.version == 2:
            self._rx_buffer += codec.encode_frame_v2(frame.function, payload, direction=direction)
        else:
            self._rx_buffer += codec.encode_frame(frame.function, payload, direction=direction)

    def _handle_request(self, frame: codec.MSPFrame) -> None:
        f = frame.function
        if f == codec.MSP_RC:
            self._reply(frame, b"".join(v.to_bytes(2, "little") for v in self.internal_order()))
        elif f == codec.MSP_RX_MAP:
            self._reply(frame, bytes(self.rcmap))
        elif f == codec.MSP_ALTITUDE:
            self._reply(frame, self.altitude_cm_to_report.to_bytes(4, "little", signed=True)
                        + int(self.vario_cm_s_to_report).to_bytes(2, "little", signed=True)
                        + (0).to_bytes(4, "little", signed=True))
        elif f == codec.MSP_ATTITUDE:
            self._reply(frame, b"".join(int(v).to_bytes(2, "little", signed=True) for v in self.attitude_to_report))
        elif f == codec.MSP_RAW_IMU:
            acc = (0, 0, 512)
            mag = (0, 0, 0)
            self._reply(frame, b"".join(int(v).to_bytes(2, "little", signed=True)
                                        for v in (*acc, *self.gyro_dps_to_report, *mag)))
        elif f == codec.MSP2_INAV_SET_LOGIC_CONDITIONS:
            if self.reject_lc_writes or len(frame.payload) != 15:
                self._reply(frame, direction=codec.DIR_ERROR)
                return
            index, record = frame.payload[0], frame.payload[1:]
            self.logic[index] = record
            self.lc_writes.append((index, record))
            self._reply(frame)
        elif f == codec.MSP2_INAV_LOGIC_CONDITIONS_SINGLE:
            self._reply(frame, self.logic[frame.payload[0]])
        elif f == codec.MSP2_INAV_LOGIC_CONDITIONS_STATUS:
            self._reply(frame, b"".join(int(v).to_bytes(4, "little", signed=True) for v in self.lc_status))
        # MSP_SET_RAW_RC and anything else: no response

    def load_chain(self, chain):
        """Pretend the user pasted the setup CLI: store these LCs."""
        for lc in chain:
            self.logic[lc.index] = lc.to_payload()[1:]

    def read(self, size: int = 1) -> bytes:
        n = min(size, len(self._rx_buffer))
        out = bytes(self._rx_buffer[:n])
        del self._rx_buffer[:n]
        return out

    @property
    def in_waiting(self) -> int:
        return len(self._rx_buffer)
