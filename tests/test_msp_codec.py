import pytest

from lockon.msp import codec


def test_encode_decode_rc_channels_roundtrip():
    channels = [1500, 1500, 988, 1500, 1000, 2000, 1500, 1500]
    frame_bytes = codec.encode_rc_channels(channels)

    parser = codec.FrameParser()
    frames = parser.feed(frame_bytes)

    assert len(frames) == 1
    frame = frames[0]
    assert frame.function == codec.MSP_SET_RAW_RC
    assert frame.direction == codec.DIR_REQUEST

    # MSP_SET_RAW_RC payload format is identical to MSP_RC's response
    # payload (array of uint16 channel values), so decode_rc_channels works
    # on either.
    assert codec.decode_rc_channels(frame.payload) == channels


def test_decode_altitude():
    # altitude = 457 cm (15 ft), vario = -12 cm/s
    payload = (457).to_bytes(4, "little", signed=True) + (-12).to_bytes(2, "little", signed=True)
    alt = codec.decode_altitude(payload)
    assert alt.altitude_cm == 457
    assert alt.vario_cm_s == -12
    assert alt.altitude_m == pytest.approx(4.57)
    assert alt.altitude_ft == pytest.approx(15.0, abs=0.01)


def test_parser_handles_fragmented_stream():
    frame_bytes = codec.encode_rc_channels([1500] * 8)
    parser = codec.FrameParser()

    frames = []
    # Feed one byte at a time to simulate a slow/fragmented serial read.
    for b in frame_bytes:
        frames.extend(parser.feed(bytes([b])))

    assert len(frames) == 1
    assert frames[0].function == codec.MSP_SET_RAW_RC


def test_parser_recovers_after_garbage_bytes():
    frame_bytes = codec.encode_rc_channels([1500] * 4)
    noisy = b"\x00\xff\xaa" + frame_bytes + b"\x11\x22"
    parser = codec.FrameParser()
    frames = parser.feed(noisy)
    assert len(frames) == 1
    assert codec.decode_rc_channels(frames[0].payload) == [1500] * 4


def test_parser_drops_frame_with_bad_checksum_but_keeps_parsing():
    good_frame = codec.encode_rc_channels([1500] * 4)
    corrupted = bytearray(good_frame)
    corrupted[-1] ^= 0xFF  # flip the checksum byte -> invalid
    stream = bytes(corrupted) + good_frame  # bad frame, then a good one

    parser = codec.FrameParser()
    frames = parser.feed(stream)

    assert parser.checksum_errors == 1
    assert len(frames) == 1
    assert codec.decode_rc_channels(frames[0].payload) == [1500] * 4


def test_encode_frame_rejects_oversized_payload():
    with pytest.raises(ValueError):
        codec.encode_frame(codec.MSP_SET_RAW_RC, b"\x00" * 256)


def test_decode_rc_channels_rejects_odd_length():
    with pytest.raises(ValueError):
        codec.decode_rc_channels(b"\x00\x01\x02")


def test_crc8_dvb_s2_matches_the_standard_check_value():
    # CRC-8/DVB-S2 catalogue check value for ASCII "123456789"
    assert codec.crc8_dvb_s2(b"123456789") == 0xBC


def test_v2_frame_roundtrip():
    payload = bytes(range(15))
    raw = codec.encode_frame_v2(codec.MSP2_INAV_SET_LOGIC_CONDITIONS, payload)
    assert raw[:3] == b"$X<"

    frames = codec.FrameParser().feed(raw)

    assert len(frames) == 1
    assert frames[0].version == 2
    assert frames[0].function == 0x2023
    assert frames[0].payload == payload


def test_parser_handles_mixed_v1_and_v2_frames_byte_by_byte():
    stream = (codec.encode_frame(codec.MSP_RC, b"\x01\x02", direction=codec.DIR_RESPONSE)
              + codec.encode_frame_v2(0x203B, b"\x05" * 14, direction=codec.DIR_RESPONSE)
              + codec.encode_frame_v2(0x2023, b"", direction=codec.DIR_ERROR))
    parser = codec.FrameParser()
    frames = []
    for b in stream:
        frames.extend(parser.feed(bytes([b])))

    assert [(f.version, f.function) for f in frames] == [(1, 105), (2, 0x203B), (2, 0x2023)]
    assert frames[2].is_error


def test_parser_drops_a_v2_frame_with_a_bad_crc_and_recovers():
    good = codec.encode_frame_v2(0x2023, b"\x00" * 15)
    bad = bytearray(good)
    bad[-1] ^= 0x01
    parser = codec.FrameParser()

    frames = parser.feed(bytes(bad) + good)

    assert parser.checksum_errors == 1
    assert len(frames) == 1


def test_encode_request_picks_the_protocol_version_from_the_function_id():
    assert codec.encode_request(codec.MSP_RC)[:2] == b"$M"
    assert codec.encode_request(codec.MSP2_INAV_LOGIC_CONDITIONS_SINGLE, b"\x00")[:2] == b"$X"
