# Task 3 Runbook - Wire the Pi and bring up the link (idle only)

## Wiring

- **Power**: dedicated 5V BEC/UBEC off the flight battery, rated for the
  Pi 5's peak draw (official Pi 5 PSU spec is 5V/5A; size the UBEC for at
  least 5V/3A continuous to be safe under camera+inference load). Do
  **not** power the Pi from the FC's own 5V pad - most FC BECs are only
  rated for ~1-1.5A, well short of what a Pi 5 can pull under load.
- **Data**: Pi GPIO14 (TXD, pin 8) -> FC **RX3**, Pi GPIO15 (RXD, pin 10)
  -> FC **TX3**, Pi GND -> FC GND. Both are 3.3 V logic, so no level
  shifter. UART3 is set to MSP by the Task 2 setup.
- **Do not use the Pi 5's 3-pin JST-SH "UART" connector** (between the
  HDMI ports). It is the debug console: the Pi prints boot messages and
  kernel logs there. While disarmed, INAV treats stray text on an MSP
  port as commands - an `R` reboots the flight controller into its
  firmware-update mode, a `#` turns the port into a CLI (checked in INAV
  6.1.1 `msp/msp_serial.c`). Boot text contains both, so the FC would be
  knocked out at power-up. GPIO14/15 carries nothing but the Pi's MSP
  frames.
- If the FC ever stops answering the Pi on the bench (Configurator
  shows the CLI arming flag, or the board seems dead), power-cycle it.
- **Camera**: Camera Module 3 fixed forward-facing in the nose.

## OS setup (headless Raspberry Pi OS, 64-bit)

1. Flash Raspberry Pi OS Lite (64-bit) with Raspberry Pi Imager; enable
   SSH + set hostname/credentials in the imager's advanced options so it's
   reachable headless from first boot.
2. `sudo raspi-config` -> Interface Options -> Serial Port: login shell
   over serial = **No**, serial port hardware = **Yes**.
3. Make sure `/boot/firmware/config.txt` contains `dtparam=uart0=on`
   (add it if raspi-config didn't), then reboot.
4. Confirm the port: `ls -l /dev/ttyAMA0`. On the Pi 5, GPIO14/15 is
   `/dev/ttyAMA0`. `/dev/serial0` points to the small 3-pin debug
   connector instead, so don't use it.
5. Copy this project to the Pi (`git clone` or `scp -r`), then:

   ```
   python3 -m venv --system-site-packages ~/lockon-venv
   source ~/lockon-venv/bin/activate
   pip install -r requirements-pi.txt
   ```

## Bench test (props off)

Close the INAV CLI everywhere first (it pauses logic conditions).

```
python3 tools/bench_logic_test.py --port /dev/ttyAMA0 --idle
```

The Pi sends only its heartbeat with the state "off". Check:
- It starts with "Logic-condition chain verified" (if it reports a
  mismatch, redo Task 2 step 2).
- `alive=1` the whole time; `engaged=0` with the lock-on switch up or down.
- Fly the sticks with the switch **up** and **down**: no difference from
  having no Pi at all.
- Stop it with `Ctrl+C`, or pull the Pi's TX wire: `alive` goes to 0 (run
  the tool again to watch), and still nothing changes for the sticks.

Then run the Task 2 bench checks (sweep, directions, crash) from the Pi.
Only once all of them pass should Task 4 (camera) and later tasks wire in
vision and guidance.
