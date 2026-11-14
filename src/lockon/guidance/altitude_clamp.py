"""Hard altitude floor (15 ft by default) while lock-on is engaged, in the
same degrees as the guidance output.

At attack speed the plane descends fast: a 20 deg dive at 30 m/s sinks
10 m/s (33 ft/s), so the floor must act on where the plane WILL be, not
where it is. Two layers:

1. Dive limit (always): the commanded dive is capped so the current sink
   rate would take at least ``time_to_floor_s`` to reach the floor. Near the
   floor this goes to level flight.
2. Recovery (when predicted altitude = altitude + climb_rate * lookahead
   falls below the floor, or altitude already is below it): pitch at least
   ``recovery_pitch_deg`` nose up and bank limited to
   ``recovery_max_bank_deg`` (a banked wing loses lift). It stays on until
   the plane is ``margin_ft`` above the floor and not descending (hysteresis,
   so it can't flicker at the boundary).

The recovery pitch is deliberately moderate (15 deg): a hard pull with
the pilot's throttle low could stall the wing (~10 m/s stall speed).

Altitude source caveat: INAV's altitude here is barometric, relative to
where the board powered up. At 30 m/s the airflow can shift the baro
reading by metres unless the sensor is shielded from it (foam over the
baro, sealed bay). Check the floor in flight (Task 14a) before trusting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FT_PER_M = 3.28084


@dataclass(frozen=True)
class GuardResult:
    pitch_deg: float
    bank_deg: float
    boosting: bool
    dive_limit_deg: float


class AltitudeGuard:
    def __init__(self, floor_ft: float = 15.0, margin_ft: float = 5.0, lookahead_s: float = 1.5,
                 time_to_floor_s: float = 2.0, recovery_pitch_deg: float = 15.0,
                 recovery_max_bank_deg: float = 15.0):
        if margin_ft < 0 or lookahead_s < 0 or time_to_floor_s <= 0:
            raise ValueError("margin and lookahead must be >= 0, time_to_floor_s > 0")
        self.floor_ft = floor_ft
        self.margin_ft = margin_ft
        self.lookahead_s = lookahead_s
        self.time_to_floor_s = time_to_floor_s
        self.recovery_pitch_deg = recovery_pitch_deg
        self.recovery_max_bank_deg = recovery_max_bank_deg
        self._boosting = False

    @property
    def is_boosting(self) -> bool:
        return self._boosting

    def reset(self) -> None:
        self._boosting = False

    def update(self, altitude_ft: float, climb_rate_ftps: float, pitch_deg: float, bank_deg: float,
               speed_mps: float) -> GuardResult:
        predicted = altitude_ft + min(0.0, climb_rate_ftps) * self.lookahead_s
        if not self._boosting and (altitude_ft < self.floor_ft or predicted < self.floor_ft):
            self._boosting = True
        elif self._boosting and altitude_ft >= self.floor_ft + self.margin_ft and climb_rate_ftps >= 0:
            self._boosting = False

        allowed_sink_mps = max(0.0, (altitude_ft - self.floor_ft) / FT_PER_M / self.time_to_floor_s)
        dive_limit = math.degrees(math.asin(min(1.0, allowed_sink_mps / max(speed_mps, 1.0))))

        if self._boosting:
            bank = max(-self.recovery_max_bank_deg, min(self.recovery_max_bank_deg, bank_deg))
            return GuardResult(max(pitch_deg, self.recovery_pitch_deg), bank, True, dive_limit)
        return GuardResult(max(pitch_deg, -dive_limit), bank_deg, False, dive_limit)
