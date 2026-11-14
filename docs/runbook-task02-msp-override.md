# Task 2 Runbook - Lock-on link on INAV 6.1.1 (props off)

The flight controller runs **INAV 6.1.1** (not Betaflight). That build has
no MSP RC override, so the Pi steers through INAV's **programming
framework** instead: a chain of 14 logic conditions you enter once from the
CLI. At run time the Pi only changes a few numbers in that chain over MSP
(in RAM, never saved). Do this whole task with **props removed**.

## How the chain behaves

INAV runs the chain 10 times a second. The Pi's bank/pitch targets are
applied only when all three are true:

- the lock-on switch (CH7) is up - checked by the flight controller itself,
- the Pi's heartbeat number changed within the last 300 ms,
- the Pi has set its state to "engaged".

| Event | Result |
|-------|--------|
| Switch off | Pi cut within 100 ms, whatever the Pi does |
| Pi crashes / hangs / unplugged | Pi cut 400-500 ms later |
| Nothing locked | Nothing overridden; your sticks fly the plane |
| Target locked | Pi's bank (max 45 deg) and pitch (max 30 deg) targets, INAV stabilizes them; CH12 driven high -> HORIZON -> radio shows "HOR" |
| CLI open on any port | INAV pauses all logic conditions -> nothing overridden |
| Reboot | Saved values = disengaged |

Throttle always stays on your stick (throttle control is a Task 13 item).

## Bench findings used here

- Map `AETR`: CH1 roll, CH2 pitch, CH3 throttle, CH4 yaw.
- Receiver on UART4 (`serial 3 64 ...`), Bluetooth/app MSP on UART6
  (`serial 5 1 ...`), USB MSP (`serial 20 1 ...`).
- **Pi on UART3.** UART3 is currently set to GPS (`serial 2 2 ...`). The
  setup below changes it to MSP. If a GPS module is plugged into the TX3/RX3
  pads, stop and say so; UART2 or UART5 are free alternatives.
- Modes: `aux 0` = ARM on CH5 high, `aux 1` = ANGLE on CH6 high. You fly
  Acro, so CH6 is normally low.
- All 64 logic slots are empty; the chain uses slots 0-13.
- `max_angle_inclination_rll = 450`, `_pit = 300` (45 and 30 degrees): the
  Pi's targets are capped there.

## Step 1 - Radio (Zorro, EdgeTX)

1. In the model's **Mixes**, put a free 2-position switch on **CH7**
   (source = that switch, weight 100). Up must read high (~2000 us), down
   low (~1000 us).
2. Leave **CH12** without any mix. The flight controller drives it itself
   while a target is locked.

## Step 2 - Back up, then paste the setup

1. In the INAV CLI, run `diff all` and save the whole output to a text
   file. That is your way back.
2. On the laptop, run `python tools/inav_lockon_setup.py`. It prints:

```
serial 2 1 115200 115200 0 115200
logic 0 1 -1 6 1 7 0 0 0
logic 1 1 -1 50 0 0 0 1 0
logic 2 1 -1 12 4 1 0 0 0
logic 3 1 -1 48 4 2 0 300 0
logic 4 1 -1 12 4 3 0 0 0
logic 5 1 -1 14 0 0 0 0 0
logic 6 1 -1 2 4 5 0 0 0
logic 7 1 -1 7 4 0 4 4 0
logic 8 1 -1 7 4 7 4 6 0
logic 9 1 8 45 0 0 0 0 0
logic 10 1 8 45 0 1 0 0 0
logic 11 1 -1 1 4 5 0 2 0
logic 12 1 -1 7 4 8 4 11 0
logic 13 1 12 38 0 12 0 2000 0
aux 2 2 7 1700 2100
save
```

3. Paste all of it into the CLI. `save` reboots the board.
4. Reconnect and check `logic` and `aux` show those lines.

What each line does is documented at the top of
`src/lockon/msp/inav_logic.py`.

## Step 3 - Wire the link

Pi GPIO14 (TXD) -> FC **RX3** pad, Pi GPIO15 (RXD) -> FC **TX3** pad, Pi
GND -> FC GND. Both sides are 3.3 V, so no level shifter. Pi setup is in
`docs/runbook-task03-pi-bringup.md`. (For an early test from the laptop
instead, a 3.3 V USB-UART adapter on the same pads works, or any other MSP
port of the FC.)

## Step 4 - Bench test (props off)

Close the INAV CLI everywhere first. All commands are from the Pi; use
`--port COMx` instead when running from the laptop.

1. **Link and switch check:**
   ```
   python3 tools/bench_logic_test.py --port /dev/ttyAMA0 --idle
   ```
   It first verifies the chain (it refuses to run if slots 0-13 don't
   match). Then it prints the flight controller's view twice a second:
   - flip the CH7 switch: `switch` follows it (1 = up),
   - `alive=1`, `engaged=0` the whole time,
   - the elevons respond to your sticks exactly as without the Pi.

2. **Steering check:**
   ```
   python3 tools/bench_logic_test.py --port /dev/ttyAMA0 --sweep
   ```
   - Switch **down**: `engaged=0`, elevons follow your sticks only.
   - Switch **up**: `engaged=1`, `indicator=1`, elevons move with the sweep
     and ignore your roll/pitch stick.
   - Flip the switch down mid-sweep: control back to your sticks at once.

   If the elevons don't move at all with the switch up while disarmed,
   **arm** (CH5) with props still off and repeat - INAV may only stabilize
   while armed. The motors will spin; that's why the props are off.

3. **Direction check** (the most important one):
   ```
   python3 tools/bench_logic_test.py --port /dev/ttyAMA0 --hold 15 0
   ```
   Switch up. The elevons must deflect to **bank right** (right elevon
   trailing edge up, left down). Tilt the plane right by hand: the
   deflection should shrink to neutral near 15 degrees. Then:
   ```
   python3 tools/bench_logic_test.py --port /dev/ttyAMA0 --hold 0 10
   ```
   Both trailing edges up (**nose up**), shrinking as you raise the nose by
   hand toward 10 degrees.

   If either axis moves the wrong way, don't touch the INAV servo settings
   (your sticks are already correct). Flip `roll_sign` or `pitch_sign` in
   `LockOnChainConfig` (`src/lockon/msp/inav_logic.py`) instead and repeat.

4. **Pi-crash check:**
   ```
   python3 tools/bench_logic_test.py --port /dev/ttyAMA0 --sweep --crash-after 10
   ```
   Switch up. At 10 s the tool stops writing but keeps reading: `alive`
   and `engaged` must drop to 0 within about half a second, and the
   elevons return to your sticks. Also try pulling the Pi's TX wire during
   a sweep.

5. **Indicator check** (armed, props off): with `--sweep` and the switch up,
   the Zorro's flight-mode telemetry should read `HOR`; it should go back
   to `AIR` or `ACRO` when the switch goes down. INAV only sends the
   flight-mode text while armed (it sends `OK` while disarmed). The audio
   cue is Task 12.

Write down the results (especially the directions from check 3) before
moving on.

## Undo

Paste these lines into the CLI to remove everything this task added, then
`save`:

```
logic 0 0 -1 0 0 0 0 0 0
...            (same for 1-13)
logic 13 0 -1 0 0 0 0 0 0
aux 2 0 0 900 900
serial 2 2 115200 115200 0 115200
```
