# Pi-Assisted Lock-On Autopilot for an INAV Mini-Plane

A Raspberry Pi 5 + Camera Module 3 added to an existing SpeedyBee F405
Wing flying wing running **INAV 6.1.1**, so that flipping a transmitter
switch hands bank and pitch to the Pi, which detects a flying object
(balloon/drone) and flies an intercept course straight at it. The guidance
removes the plane's own roll and pitch from the camera line of sight using
the FC's attitude (gyro-extrapolated to the frame's exposure time) and
leads the target with proportional navigation, so a fast pass stays a
near-straight collision course instead of curving in late. Flipping the
switch off returns control to the pilot within 100 ms; if the Pi dies,
within about half a second. A 15 ft altitude floor is enforced whenever
lock-on is engaged. The Zorro shows the flight-mode text `HOR`
and beeps while a target is locked.

The original plan assumed Betaflight's MSP RC override. The bench
findings showed the board actually runs INAV 6.1.1, which doesn't include
that feature, so the Pi steers through INAV's programming framework
instead - see "How the Pi steers" below.

## What's implemented vs. what you still have to physically do

This repo contains everything that is pure software: the MSP protocol
client, the vision pipeline (ML + fallback detection, tracking, target
selection), the guidance law, the altitude safety clamp, and the
integration controller wiring them all together - all unit tested
without needing the real Pi/FC/camera.

**Hardware setup, INAV CLI configuration, EdgeTX radio configuration,
and actual flight testing are physical actions nobody but you can
perform.** Each one has a step-by-step runbook in `docs/`:

| Task | What | Runbook |
|------|------|---------|
| 1 | Find free FC UART, confirm baro | `docs/bench-findings.md` (filled in) |
| 2 | Set up the INAV logic-condition link | `docs/runbook-task02-msp-override.md` |
| 3 | Wire the Pi, headless bring-up | `docs/runbook-task03-pi-bringup.md` |
| 4 | Camera sanity check | `docs/runbook-task04-camera-sanity.md` |
| 8 | Dataset/training/export for the ML detector | `docs/runbook-task08-ml-detector.md` |
| 11 | First real integration (props off) | `docs/runbook-task11-bench-integration.md` |
| 12 | Zorro indicator setup | `docs/runbook-task12-edgetx-indicator.md` |
| 13 | Props-on integration (secured) | `docs/runbook-task13-props-on.md` |
| 14 | Flight tests, then the balloon-pop test | `docs/runbook-task14-flight-tests.md` |

## Repo layout

```
src/lockon/
  msp/
    codec.py            MSP v1 + v2 frames
    client.py           serial client (RC channels, altitude, raw requests)
    inav_logic.py       Task 2 - INAV logic-condition chain + run-time link
  guidance/
    attitude.py         aircraft attitude from INAV + gyro extrapolation
    altitude_clamp.py   15ft floor: predictive recovery + dive-angle limit
    intercept.py        pursuit + proportional-navigation guidance; speed est.
    sim.py              closed-loop intercept simulator (plane+camera+latency)
  vision/
    types.py            shared Detection type
    camera_model.py     pixel -> line of sight (azimuth/elevation) via attitude
    capture.py          Task 6 - Picamera2 wrapper (guarded import)
    fallback_detector.py Task 7 - HSV color + motion fallback
    ml_detector.py       Task 8 - NCNN inference wrapper (guarded import)
    ncnn_backend.py      Task 8 - real NCNN backend (Pi-only)
    tracker.py           Task 9 - CSRT/KCF wrapper (guarded import)
    selection.py         Task 9 - multi-target scoring
    hybrid.py            Task 9 - ML primary / fallback secondary / tracker bridge
  control/
    state_machine.py     Task 11 - LockState enum
    lock_on_controller.py Task 11 - wires everything into the INAV link

tools/            runnable scripts - bench tests, live demos, training,
                  benchmarking - see each script's docstring for usage
  inav_lockon_setup.py  prints the INAV CLI setup (Task 2)
  bench_logic_test.py   link bench tests (Tasks 2-3)
  edgetx/lockon.lua     radio beep script (Task 12)
docs/             hardware runbooks + bench-findings template
tests/            pytest unit tests for every pure-software module
```

## Dev setup (laptop, no Pi/FC needed)

```
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

All tests should pass; none require real hardware - every hardware
interface (serial port, camera, ML backend, tracker) is injected so
tests run against fakes. The guidance is also checked end-to-end in a
closed-loop simulator; `python tools/intercept_sim.py` prints a miss-
distance table across speeds, latencies and lenses.

## Training setup (laptop with NVIDIA GPU)

See `requirements-train.txt` and `docs/runbook-task08-ml-detector.md`
(datasets, filtering decisions, rebuild commands, v1 results).

The trained v1 model, already exported for the Pi, is in
`models/balloon_drone_v1/` (git-ignored; copy it to the Pi). Use
`--model-path models/balloon_drone_v1/ncnn_640x480` with the Pi tools.

## Pi setup

See `requirements-pi.txt` and `docs/runbook-task03-pi-bringup.md`.

## How the Pi steers

The ELRS receiver stays wired to the flight controller as before. A chain
of 14 INAV logic conditions (entered once from the CLI) applies the Pi's
target bank and pitch angles only while the pilot's lock-on switch (CH7)
is up, the Pi's heartbeat is fresh, and the Pi reports a target. The Pi
changes those values in RAM over MSP and never saves them. INAV runs the
chain at 10 Hz, so targets update 10 times a second in whole degrees,
capped at the INAV max angles (45 deg bank, 30 deg pitch). Throttle stays
on the pilot's stick. Details and the exact CLI lines:
`docs/runbook-task02-msp-override.md` and
`src/lockon/msp/inav_logic.py`.

## Known calibration items (flagged inline in code, not guessed at)

- Angle signs: `roll_sign`/`pitch_sign` in `LockOnChainConfig`
  (`inav_logic.py`) follow INAV's source (pitch target negative = nose
  up). Confirm on the bench (Task 2, step 4.3).
- Attitude/gyro signs: `InavSigns` in `guidance/attitude.py` assumes
  INAV's pitch is positive nose-down and the gyro signs match. Confirm
  with `python tools/check_attitude.py --port ...` before any lock-on
  flight (it tells you which sign to flip if one is wrong).
- Camera optics and tilt: the guidance needs the right focal length.
  `bench_integration_runner.py` reads the lens (standard vs wide) and the
  sensor crop from Picamera2 automatically; set `--mount-pitch-deg` if the
  camera points above the nose.
- Assumed airspeed: guidance assumes 30 m/s (`--speed-mps`) and refines it
  in flight from banked turns. A wrong estimate only scales the turn gain;
  the camera loop corrects the rest.
- Altitude: INAV measures it from where the board powered up, so power
  up at ground level at the field.
- Camera color order: `capture.py` requests Picamera2 `"BGR888"`, which
  yields RGB arrays. Confirm on the Task 4 monitor setup that a red
  balloon looks red (gets boxed by `live_overlay_demo.py --mode fallback`).
- Detector confidence threshold (0.40) and Pi input size (640x480 export,
  416x320 fallback): re-pick both on your own camera footage (Task 8
  runbook).
- `ColorMotionDetector`'s default HSV range targets a bright red/orange
  balloon - retune for your actual balloon color (Task 7).
