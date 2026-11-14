# Bench Findings (Task 1)

Fill this in while connected to the FC via Betaflight Configurator's CLI
tab. These values feed directly into Task 2 and Task 3.

## 1. Free UART

In the CLI tab, run:

```
resource
serial
```

- `resource` lists every pin's current assignment (`SERIAL_TX 1`, `SERIAL_RX 1`,
  etc.) - look for a `UARTx` not already claimed by `SERIALRX` (ELRS receiver),
  `ESC`, `VTX`, etc.
- `serial` lists currently configured ports and their function masks.

Free UART found: `__UART 1,2,3,4,5___`  (e.g. `UART3`)

Pads/pins for that UART on the F405 Wing SpeedyBee (check the board's
pinout silkscreen/diagram): TX = `___3,5___`, RX = `___3,5___`

## 2. Barometer confirmation

Betaflight Configurator -> Sensors tab -> watch the "Altitude" /
barometer readout while gently moving the board up and down by hand;
confirm it responds.

Baro chip found (if shown under Configurator -> CLI -> `status`):
`______It's reading SPL06 and says Baro: OKAY. Rest of the output:
# status
INAV/SPEEDYBEEF405WING 6.1.1 May  8 2024 / 20:16:28 (ba108dd0)
GCC-10.2.1 20201103 (release)
System Uptime: 95 seconds
Current Time: 2041-06-28T01:04:00.000+00:00
Voltage: 23.64V (6S battery - OK)
CPU Clock=168MHz, GYRO=ICM42605, ACC=ICM42605, BARO=SPL06
STM32 system clocks:
  SYSCLK = 168 MHz
  HCLK   = 168 MHz
  PCLK1  = 42 MHz
  PCLK2  = 84 MHz
Sensor status: GYRO=OK, ACC=OK, MAG=NONE, BARO=OK, RANGEFINDER=NONE, OPFLOW=NONE, GPS=NONE
SD card: Startup failed
Stack size: 6144, Stack address: 0x10010000, Heap available: 1828
I2C Errors: 0, config size: 9055, max available config: 131072
ADC channel usage:
   BATTERY : configured = ADC 1, used = ADC 1
      RSSI : configured = ADC 4, used = none
   CURRENT : configured = ADC 2, used = ADC 2
  AIRSPEED : configured = ADC 3, used = none
System load: 6, cycle time: 1021, PID rate: 979, RX rate: 9, System rate: 9
Arming disabled flags: RX CLI
OSD: MAX7456 [30 x 16]
VTX: not detected
_____`

Live altitude reading observed on the bench: `______Can't find______` m / ft

## 3. Firmware confirmation

CLI -> `version` output:

```
(paste here)
# version
# INAV/SPEEDYBEEF405WING 6.1.1 May  8 2024 / 20:16:28 (ba108dd0)
# GCC-10.2.1 20201103 (release)
```

## 4. Lock-on link bench results (Task 2 / 3)

- Steps 1-7 of the bench checklist passed: radio CH7 switch, `diff all`
  backup, logic-condition setup pasted, Pi set up and wired on GPIO14/15
  (not the JST-SH debug port), idle test, steering test.
- **Still to do once the servos are installed:**
  - Step 8, direction check: `bench_logic_test.py --hold 15 0` must bank
    right (right elevon up, left down); `--hold 0 10` must raise both
    elevons (nose up). If either is wrong, flip `roll_sign`/`pitch_sign`
    in `src/lockon/msp/inav_logic.py`, not the INAV servo settings.
- Step 9, Pi-crash check: passed.
- Before flying lock-on: with the plane sitting level, run the
  accelerometer calibration (INAV Configurator, Calibration tab), because
  a 0 deg pitch/bank target means "level" as the accelerometer sees it.
