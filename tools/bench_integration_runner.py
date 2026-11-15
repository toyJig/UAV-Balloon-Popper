#!/usr/bin/env python3
"""Task 11 - first real integration run, PROPS OFF.

Wires the real camera, hybrid detector (ML + fallback), pursuit guidance,
altitude floor clamp and the INAV logic-condition link together through
LockOnController and runs the control loop continuously. This is the first
point in the project where camera input actually moves the elevons.

The INAV CLI must be closed (INAV pauses logic conditions while it is open).

Usage:
    python3 tools/bench_integration_runner.py --msp-port /dev/ttyAMA0 --model-path models/balloon_drone_v1/ncnn_640x480

Safety: PROPS OFF. Throttle is never controlled by the Pi in this build.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import serial  # type: ignore  # noqa: E402

import time  # noqa: E402

from lockon.control.lock_on_controller import LockOnController  # noqa: E402
from lockon.guidance.altitude_clamp import AltitudeGuard  # noqa: E402
from lockon.guidance.intercept import GuidanceParams, InterceptGuidance  # noqa: E402
from lockon.msp.client import MSPClient  # noqa: E402
from lockon.msp.inav_logic import InavLockOnLink, LockOnChainConfig  # noqa: E402
from lockon.vision.capture import CameraCapture  # noqa: E402
from lockon.vision.fallback_detector import ColorMotionDetector  # noqa: E402
from lockon.vision.hybrid import HybridDetector  # noqa: E402
from lockon.vision.ml_detector import MLDetector  # noqa: E402


def _publish(hub, frame_rgb, result, roll, pitch, speed, fps) -> None:
    """Draw the tracked box + the live command overlay and push a JPEG to
    the browser stream."""
    import cv2

    from lockon.vision.types import DetectionSource

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    cv2.drawMarker(bgr, (w // 2, h // 2), (90, 90, 90), cv2.MARKER_CROSS, 20, 1)  # boresight
    det = result.detection
    if det is not None:
        color = (0, 255, 0) if det.source == DetectionSource.ML else (0, 165, 255)
        x, y, bw, bh = int(det.x), int(det.y), int(det.w), int(det.h)
        cv2.rectangle(bgr, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(bgr, f"{det.source.value} {det.class_name} {det.confidence:.2f}",
                    (x, max(14, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    boost = "  FLOOR-CLIMB" if result.altitude_boosting else ""
    line1 = f"{result.state.value}   roll {roll:+d}  pitch {pitch:+d}{boost}"
    line2 = f"V~{speed:.0f} m/s   {fps:4.1f} fps"
    cv2.putText(bgr, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bgr, line2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        hub.publish(buf.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--msp-port", required=True)
    parser.add_argument("--msp-baud", type=int, default=115200)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--resolution", default="640x480")
    parser.add_argument("--floor-ft", type=float, default=15.0)
    parser.add_argument("--switch-channel", type=int, default=7)
    parser.add_argument("--indicator-channel", type=int, default=12)
    parser.add_argument("--speed-mps", type=float, default=30.0,
                        help="assumed attack airspeed (refined in flight from banked turns)")
    parser.add_argument("--wide", action="store_true", help="Camera Module 3 Wide lens")
    parser.add_argument("--mount-pitch-deg", type=float, default=0.0,
                        help="camera tilt above the nose (measure once on the airframe)")
    parser.add_argument("--stream", action="store_true",
                        help="serve the live feed + detection box to a laptop browser (headless demo)")
    parser.add_argument("--http-port", type=int, default=8000)
    args = parser.parse_args()

    w, h = (int(v) for v in args.resolution.split("x"))
    frame_size = (w, h)

    msp_client = MSPClient(serial.Serial(args.msp_port, args.msp_baud, timeout=0.05))
    cfg = LockOnChainConfig(switch_channel=args.switch_channel, indicator_channel=args.indicator_channel)
    link = InavLockOnLink(msp_client, cfg)
    link.verify()

    ml = MLDetector(model_path=args.model_path)
    hybrid = HybridDetector(ml, ColorMotionDetector(), frame_size)
    guidance = InterceptGuidance(GuidanceParams(speed_mps=args.speed_mps))
    guard = AltitudeGuard(floor_ft=args.floor_ft)

    hub = server = None
    if args.stream:
        from lockon.vision.mjpeg import FrameHub, serve  # noqa: E402
        hub = FrameHub()
        server = serve(hub, args.http_port)

    print("Bench integration running, PROPS OFF. Ctrl+C to stop.")
    controller = None
    try:
        with CameraCapture(resolution=frame_size) as cap:
            cap.read_timed()  # prime the camera so camera_model() can read the sensor crop
            camera = cap.camera_model(mount_pitch_deg=args.mount_pitch_deg)
            print(f"Camera: f_px={camera.f_px:.0f} hfov={camera.hfov_deg:.0f} deg")
            controller = LockOnController(
                msp_client, link, hybrid, camera, guidance, guard,
                lock_switch_index=args.switch_channel - 1,
            )
            last_t = time.monotonic()
            fps = 0.0
            for frame, frame_time in cap.timed_frames():
                now = time.monotonic()
                result = controller.step(frame, frame_time=frame_time, now=now)
                state, roll, pitch = link.last_written
                print(f"[t={now:9.2f}] {result.state.value:16s} sent state={state} roll={roll:+3d} "
                      f"pitch={pitch:+3d} V~{guidance.speed_mps:4.1f} floor_boost={result.altitude_boosting}")
                if hub is not None:
                    dt = now - last_t
                    last_t = now
                    if dt > 0:
                        fps = 0.9 * fps + 0.1 / dt
                    _publish(hub, frame, result, roll, pitch, guidance.speed_mps, fps)
    except KeyboardInterrupt:
        pass
    finally:
        if controller is not None:
            controller.shutdown()
        if hub is not None:
            hub.stop()
        if server is not None:
            server.shutdown()
        print("Stopped; control handed back.")


if __name__ == "__main__":
    main()
