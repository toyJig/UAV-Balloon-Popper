# Task 11 Runbook - First Real Integration (props off)

Everything built in Tasks 2-10 comes together here for the first time:
camera -> hybrid detector -> guidance -> altitude clamp -> INAV
logic-condition link (Task 2). **Props removed for this entire task.**
Close the INAV CLI first. Stabilization may need the board armed (see
Task 2, step 4).

```
python3 tools/bench_integration_runner.py --msp-port /dev/ttyAMA0 --model-path models/balloon_drone_v1/ncnn_640x480
```

Each line shows the controller state and what the Pi sent: `state` 0 =
off, 1 = engaged (auto-level), 2 = engaged + locked; `roll`/`pitch` are
target angles in degrees (pitch positive = nose up); `V~` is the current
assumed airspeed. The runner reads the camera lens and sensor crop from
Picamera2 and prints the computed focal length on start; pass
`--mount-pitch-deg N` if the camera is tilted up from the nose, and
`--wide` is detected automatically from the sensor model.

### Before this task: confirm the attitude signs (one time)

The guidance removes the plane's roll/pitch from the camera line of sight
using the FC attitude, and winds it back to the frame's exposure time with
the gyro. The sign conventions for both are assumed in code. Confirm them:

```
python3 tools/check_attitude.py --port /dev/ttyAMA0 --seconds 20
```

Keep the board still for the first second, then roll, pitch and yaw it one
axis at a time through moderate angles. Confirm the live line: roll > 0
right-wing-down, pitch > 0 nose-up, yaw rising when the nose swings right.
If the summary prints `MISMATCH` on any axis, flip the matching
`gyro_*` sign in `InavSigns` (`src/lockon/guidance/attitude.py`) and
re-run until all three say OK.

Note on the altitude floor: INAV's altitude is measured from where the
board powered up, so on the bench it reads about 0 ft - below the 15 ft
floor. While locked, the floor therefore forces the recovery climb
(`floor_boost=True`, `pitch=+15`). To see normal intercept pitch on the
bench, run with `--floor-ft -10`. Never use that outside the bench.

## Procedure

1. Plane secured on the bench, props off, lock-on switch starts **off**.
2. Power up, confirm the script prints `disengaged` steadily.
3. Flip the switch **on** with no balloon in view - confirm
   `state=searching` and the elevons simply follow the transmitter sticks
   (no camera influence yet, since nothing is locked).
4. Hold a balloon in front of the camera - confirm the state flips to
   `state=locked_ml` or `state=locked_fallback`, and moving the balloon by
   hand causes the elevons to visibly deflect toward it. Tilt the whole
   plane (roll/pitch it on the bench) with the balloon held fixed in space:
   the commanded angles should stay pointed at the balloon's real position,
   not swing with the plane - that is the attitude compensation working.
5. Pull the balloon away until nothing is detected - confirm
   `state=auto_leveling` for about a second (elevons return to center,
   not whatever they were doing when lock was lost) and then
   `state=searching` again, still with the switch on, still trying.
6. Flip the switch **off** mid-lock (hold the balloon in view, locked,
   then flip) - confirm `state=disengaged` instantly and the elevons
   snap to whatever the transmitter sticks are currently commanding, with
   no delay.
7. Repeat step 6 a few times from different lock states to be confident
   it's solid before Task 12/13.

Record any tuning changes needed (`nav_constant`/`pursuit_time_s`/
`pitch_lead_s` in `GuidanceParams`, HSV range in `ColorMotionDetector`,
confidence threshold in `MLDetector`) in a dated note. Before changing a
gain, reproduce the behaviour in `python tools/intercept_sim.py` and check
the change helps there first.
