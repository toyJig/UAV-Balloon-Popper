# Task 13 Runbook - Full Control Integration (props on, secured)

Validates the complete system with motors live, still on the ground. The
plane must be **securely restrained** - bench-clamped or firmly held by a
second person - for this entire task.

## Throttle is not under Pi control yet

In this build the Pi steers bank and pitch only; throttle always stays on
your stick. Adding throttle means one more logic condition (INAV
operation 29, OVERRIDE_THROTTLE) gated by the same "engaged" condition
as the angle targets. It has to be designed and bench-tested before this
task; until then, fly attack runs with your own throttle (you said the
motors can fly straight up, so full throttle during an attack is simple).

## Steps

1. Secure the plane. Clear the area - spinning props, even restrained.
2. Arm per your normal procedure with the lock-on switch down, and confirm
   control is completely normal (re-check this after any code change).
3. Run `tools/bench_integration_runner.py`, flip the lock-on switch up and
   present a balloon: elevons react as in Task 11, throttle follows only
   your stick.
4. Re-check the hand-backs with motors live:
   - switch down mid-lock -> your sticks at once,
   - kill the Pi program (`Ctrl+C`, or pull the Pi's TX wire) while locked
     and switch up -> your sticks within about half a second.

Only proceed to Task 14 once every item above is confirmed solid.
