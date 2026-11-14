"""Lock-on state machine states, matching the design doc's state diagram.

    DISENGAGED    switch is off - nothing overridden, no indicator. The
                  flight controller's own logic conditions also check the
                  switch, so this hand-back happens within 100 ms whatever
                  this software does.
    SEARCHING     switch on, no target locked - nothing overridden, the
                  pilot keeps the sticks, the Pi keeps trying every frame,
                  no indicator.
    LOCKED_ML     switch on, ML detector has a confident target - Pi flies.
    LOCKED_FALLBACK  switch on, color/motion fallback has a target - Pi flies.
    AUTO_LEVELING transitional: just lost the target while the switch is
                  still on - command level flight for a short, fixed
                  duration before dropping back to SEARCHING, so the pilot
                  doesn't catch the plane mid-dive. Indicator already off
                  (lock itself was lost, even though override authority
                  hasn't been handed back).
"""

from __future__ import annotations

from enum import Enum


class LockState(str, Enum):
    DISENGAGED = "disengaged"
    SEARCHING = "searching"
    LOCKED_ML = "locked_ml"
    LOCKED_FALLBACK = "locked_fallback"
    AUTO_LEVELING = "auto_leveling"

    @property
    def is_locked(self) -> bool:
        return self in (LockState.LOCKED_ML, LockState.LOCKED_FALLBACK)
