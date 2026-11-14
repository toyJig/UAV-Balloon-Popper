-- EdgeTX function script: beeps when the lock-on indicator turns on/off.
--
-- The INAV lock-on chain switches HORIZON on while the Pi has a target
-- locked, so the CRSF flight-mode telemetry text ("FM") reads "HOR".
-- This script watches that text and plays a tone on each change.
--
-- Install: copy to /SCRIPTS/FUNCTIONS/lockon.lua on the radio's SD card.
-- Then Model -> Special Functions: switch ON (always), action "Lua Script",
-- value "lockon". See docs/runbook-task12-edgetx-indicator.md.

local LOCK_TEXT = "HOR"
local last = ""

local function run(event)
  local fm = getValue("FM")
  if type(fm) ~= "string" then
    fm = ""
  end
  if fm == LOCK_TEXT and last ~= LOCK_TEXT then
    playTone(1800, 120, 60)   -- lock: two high beeps
    playTone(1800, 120, 0)
  elseif last == LOCK_TEXT and fm ~= LOCK_TEXT then
    playTone(500, 300, 0)     -- lock lost: one low beep
  end
  last = fm
  return 0
end

return { run = run }
