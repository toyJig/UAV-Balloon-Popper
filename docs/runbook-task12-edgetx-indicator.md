# Task 12 Runbook - Zorro Indicator Setup

While the Pi has a target locked, the INAV logic-condition chain (Task 2)
drives CH12 high, which switches HORIZON on. INAV then reports the
flight mode as `HOR` in its CRSF telemetry. Normally (Acro) it reports
`AIR` (`ACRO` if air mode is off).

INAV only sends the flight-mode text while **armed** (`OK` or `!ERR` while
disarmed), so test this armed, with props off.

Limits:
- If you switch ANGLE on yourself (CH6), the text reads `ANGL`, which INAV
  shows in preference to `HOR`, so the lock indicator is hidden while
  you're in Angle.
- MANUAL, failsafe and navigation modes also take precedence.

## 1. Telemetry sensor

Zorro -> Model -> Telemetry -> **Discover new sensors** with the plane
powered. A sensor named `FM` (flight mode) should appear. Add the `FM`
value to a telemetry screen so the text is visible.

## 2. Audio cue (Lua function script)

EdgeTX logical switches compare numbers, so a text sensor can't trigger
one directly. `tools/edgetx/lockon.lua` watches the `FM` text and beeps:
two high beeps on lock, one low beep when the lock ends.

1. Copy `tools/edgetx/lockon.lua` to the radio's SD card as
   `/SCRIPTS/FUNCTIONS/lockon.lua` (USB, "USB Storage" mode).
2. Model -> Special Functions -> free slot: switch `ON` (always active),
   action **Lua Script**, value `lockon`.

I haven't been able to run this on a Zorro. If nothing plays, check
Radio -> Tools / the script shows no error, and that `FM` reads text on
the telemetry screen.

## Verification

With the Task 11 bench rig running, armed, props off, switch up:
- present a balloon: two high beeps, and `FM` reads `HOR` at the moment the
  console prints `locked_ml` / `locked_fallback`,
- pull the balloon away: one low beep and the text returns to `AIR`/`ACRO`
  within about a second (after the 1 s auto-level),
- flip the lock-on switch down while locked: low beep at once.
