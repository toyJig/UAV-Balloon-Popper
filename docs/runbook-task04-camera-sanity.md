# Task 4 Runbook - Camera Sanity Check

The camera connects straight to the Pi's CAM/DISP port and has nothing to
do with the flight controller, so you can do this whole task with the
**FC disconnected** - the Pi just needs its own power (see "Powering the
Pi on its own" below).

Two ways to do it, depending on whether you have a monitor:

## A. No monitor (headless, view on your laptop) - recommended here

You can watch the live feed in a browser on your laptop over the network.

1. Power the Pi, SSH in from the laptop.
2. First confirm the OS sees the camera:

   ```
   rpicam-hello --list-cameras
   ```

   It should list one camera (an `imx708` for a Camera Module 3 / clone).
   If it says **"no cameras available"**, fix that first - see
   "If the camera isn't detected" below.
3. Stream the live feed to your laptop:

   ```
   source ~/lockon-venv/bin/activate
   python3 tools/web_preview.py            # raw feed, no detector
   ```

   It prints a URL like `http://192.168.1.42:8000/`. Open that in your
   laptop browser - you'll see the live camera with an fps counter. This
   needs no monitor and no X server. Point the camera at a balloon and
   check focus, framing and that colours look right (a red balloon looks
   red, not blue). Add `--rotate180` if the image is upside down for how
   the camera sits in the nose.
4. To watch detection live, add a detector (once the model is on the Pi):

   ```
   python3 tools/web_preview.py --mode hybrid --model-path models/balloon_drone_v2/ncnn_640x480
   ```

   Boxes are drawn on the stream: green = ML, orange = fallback.
5. Prefer a one-off still instead of a stream? Capture and pull it over:

   ```
   rpicam-jpeg -o ~/camtest.jpg --width 1920 --height 1080
   ```
   then from the laptop: `scp moehenry@MoeHenry.local:~/camtest.jpg .`
   (or just open it through VS Code's Remote-SSH file browser).

## B. With a monitor + keyboard (desktop session)

```
sudo apt update
sudo apt install -y rpicam-apps   # or libcamera-apps on older OS images
chmod +x tools/camera_sanity_check.sh
./tools/camera_sanity_check.sh
```

This shows a live preview window and saves a still + clip in
`~/lockon-camera-check/` to review.

Either way you are confirming: the camera is detected, the ribbon is
seated (clean image, no green/purple noise), and focus/framing/exposure
look sane for a forward-facing outdoor shot.

## If the camera isn't detected ("no cameras available")

Work through these in order:

1. **Reseat the ribbon, both ends.** On the Pi 5 the CAM/DISP connector
   takes the narrow **22-pin** cable; the camera end takes the wide 15-pin.
   Lift the small black tab, slide the cable fully in with the shiny metal
   contacts facing the correct way (on the Pi 5 port, contacts face the
   side away from the tab), press the tab back down. A cable in backwards
   or half-seated is the most common cause.
2. **Third-party Camera Module 3 (e.g. the "TNBA"/Amazon IMX708 board):**
   the Pi only auto-detects genuine Raspberry Pi-branded modules, so a
   clone needs its sensor overlay added by hand. Edit
   `/boot/firmware/config.txt`:

   ```
   camera_auto_detect=0
   dtoverlay=imx708
   ```

   Put the `dtoverlay` line after any `[all]` section, save, and reboot.
   Then re-run `rpicam-hello --list-cameras`. (For a genuine Raspberry Pi
   module, leave `camera_auto_detect=1` instead.)
   Content was rephrased for compliance with licensing restrictions.
3. **Update first if still stuck:** `sudo apt update && sudo apt full-upgrade`
   then `sudo rpi-update` for the latest firmware, and reboot. Older Pi 5
   firmware shipped before some IMX708 support.
4. Confirm which port you used. You said CAM/DISP 0 (J3) - that is the
   correct, default port; no config change is needed to select it.

## Powering the Pi on its own (FC disconnected)

Your friend has the FC to split battery power to the Pi through a buck
converter, so the FC isn't available right now. You do not need it for the
camera: power the Pi from its normal USB-C supply (the official Pi 5
27 W / 5 V-5 A supply, or any good 5 V/3 A+ USB-C source) and SSH in from
the laptop over Wi-Fi. The camera plugs into the Pi directly. When the
buck-converter power path comes back, nothing about the camera test
changes - it was never powered or fed through the FC.
