"""
Record DIGIT tactile sensor + ESP32 HX711/YZC-131 load cell simultaneously.
Time-synced CSV + DIGIT images saved to a timestamped output folder.

The ESP32 sketch requires interactive calibration on startup (tare then
place known weight). This script handles that via stdin prompts before
entering the main recording loop. Set calibration: false in config.yaml
to skip it (ESP must already be calibrated / running).

Usage:
    pip install digit-interface opencv-python pyserial pyyaml
    python record_both_sensors.py                 # uses config.yaml
    python record_both_sensors.py my_config.yaml  # custom config

Press 'q' in the DIGIT preview window (or Ctrl+C in terminal) to stop.
"""

import os
import re
import sys
import csv
import time
import threading
import queue
from datetime import datetime

import yaml
import cv2
import serial
import serial.tools.list_ports
from digit_interface import Digit

# -- LOAD CONFIG ------------------------------------------------------
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yaml")

with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

DIGIT_SERIAL     = CFG["digit"]["serial_id"]
DIGIT_RES        = CFG["digit"]["resolution"]
DIGIT_FPS        = CFG["digit"]["fps"]
SAVE_EVERY_N     = CFG["digit"]["save_every_n_frames"]  # 1 = every frame
LED_INTENSITY    = CFG["digit"].get("led_intensity", 15)  # 0-15

ESP_BAUD         = CFG["loadcell"]["baud_rate"]
ESP_PORT         = CFG["loadcell"]["port"]
DO_CALIBRATION   = CFG["loadcell"]["calibration"]
KNOWN_WEIGHT_G   = CFG["loadcell"]["known_weight_g"]

OUTPUT_DIR       = CFG["recording"]["output_dir"]
IMAGE_FORMAT     = CFG["recording"]["image_format"]
INTERPOLATION    = CFG["recording"].get("interpolation", "linear")  # linear, nearest, last

# Resolve output_dir relative to config file location
CONFIG_DIR = os.path.dirname(os.path.abspath(CONFIG_PATH))
if OUTPUT_DIR and OUTPUT_DIR.lower() != "auto":
    OUTPUT_DIR = os.path.normpath(os.path.join(CONFIG_DIR, OUTPUT_DIR))
# ---------------------------------------------------------------------

WEIGHT_RE = re.compile(
    r"Weight:\s*([-+]?\d+(?:\.\d+)?)\s*(g|kg)", re.IGNORECASE
)
FAST_WEIGHT_RE = re.compile(
    r"^W:(\d+),([-+]?\d+(?:\.\d+)?)$"
)


def find_esp_port():
    """Return configured port, or auto-detect the ESP32 serial port."""
    if ESP_PORT and ESP_PORT.lower() != "auto":
        return ESP_PORT
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description + (p.manufacturer or "")).lower()
        if any(k in desc for k in ["cp210", "ch340", "ch341", "ftdi",
                                    "esp", "silicon labs", "usb-serial",
                                    "usb2.0-serial"]):
            return p.device
    candidates = [p for p in ports if "ttyUSB" in p.device or "ttyACM" in p.device]
    if len(candidates) == 1:
        return candidates[0].device
    return None


def parse_weight(line):
    """Extract weight in grams from a serial line. Returns None if not a weight line.
    Supports both formats:
      Fast:  W:<millis>,<weight_g>
      Old:   Weight: 123.4 g
    """
    m = FAST_WEIGHT_RE.match(line)
    if m:
        return float(m.group(2))
    m = WEIGHT_RE.search(line)
    if m:
        val = float(m.group(1))
        if m.group(2).lower() == "kg":
            val *= 1000.0
        return val
    return None


def get_weight_for_frame(frame_ts, weight_buf, method="linear"):
    """Assign a weight to a DIGIT frame from the load cell buffer.

    Methods:
      "linear"  -- linearly interpolate between the two surrounding readings
      "cubic"   -- cubic spline through all buffered readings (smoother)
      "nearest" -- pick the reading closest in time to the frame
      "last"    -- use the most recent reading before (or at) the frame
    """
    if not weight_buf:
        return 0.0
    if len(weight_buf) == 1:
        return weight_buf[0][1]

    if method == "last":
        best = weight_buf[0][1]
        for t, w in weight_buf:
            if t <= frame_ts:
                best = w
            else:
                break
        return best

    if method == "nearest":
        best_t, best_w = weight_buf[0]
        best_dt = abs(frame_ts - best_t)
        for t, w in weight_buf[1:]:
            dt = abs(frame_ts - t)
            if dt < best_dt:
                best_dt = dt
                best_w = w
        return best_w

    if method == "cubic":
        from scipy.interpolate import CubicSpline
        ts = [t for t, _ in weight_buf]
        ws = [w for _, w in weight_buf]
        if len(weight_buf) < 4:
            # not enough points for cubic, fall through to linear
            pass
        else:
            clamped = max(ts[0], min(frame_ts, ts[-1]))
            cs = CubicSpline(ts, ws, bc_type="clamped")
            return float(cs(clamped))

    # method == "linear" (also fallback for cubic with < 4 points)
    if frame_ts <= weight_buf[0][0]:
        return weight_buf[0][1]
    if frame_ts >= weight_buf[-1][0]:
        return weight_buf[-1][1]

    for i in range(len(weight_buf) - 1):
        t0, w0 = weight_buf[i]
        t1, w1 = weight_buf[i + 1]
        if t0 <= frame_ts <= t1:
            dt = t1 - t0
            if dt == 0:
                return w0
            frac = (frame_ts - t0) / dt
            return w0 + frac * (w1 - w0)

    return weight_buf[-1][1]


def serial_readline(ser, timeout=None):
    """Read one line from serial (blocking). Returns stripped string."""
    old_timeout = ser.timeout
    if timeout is not None:
        ser.timeout = timeout
    try:
        raw = ser.readline()
        return raw.decode("utf-8", errors="replace").strip()
    finally:
        ser.timeout = old_timeout


def esp_calibration(ser):
    """
    Handle ESP32 HX711 startup -- works with both flows:
      1. NVS sketch (saved calibration): quick tare only
      2. Fresh calibration: full tare + known weight
    Prints everything the ESP sends and responds to prompts.
    """
    print("\n" + "=" * 50)
    print("  ESP32 + HX711 LOAD CELL SETUP")
    print("=" * 50)
    print("  (Follow the prompts. If calibration is saved on the ESP,")
    print("   it will only ask for a quick tare.)\n")

    calibration_done = False
    silent_count = 0
    last_prompt_sent = 0  # epoch when we last sent a character

    while not calibration_done:
        line = serial_readline(ser, timeout=1.0)

        if not line:
            silent_count += 1
            if silent_count >= 3:
                print("  [INFO] No data from ESP32 -- it may be waiting for input.")
                input("    >> Press ENTER to send a character to ESP32... ")
                ser.write(b"x\n")
                ser.flush()
                last_prompt_sent = time.time()
                silent_count = 0
            continue

        silent_count = 0
        low = line.lower()

        # Ignore boot garbage (short hex fragments, non-printable junk)
        if len(line) < 4 and not FAST_WEIGHT_RE.match(line):
            continue

        print(f"  [ESP] {line}")

        # Detect end -- works for both NVS ("loaded") and fresh ("complete")
        if any(k in low for k in ["calibration complete", "calibration loaded",
                                   "streaming", "resuming weighing"]):
            calibration_done = True
            continue

        # Detect weight line = already streaming
        if WEIGHT_RE.search(line) or FAST_WEIGHT_RE.match(line):
            calibration_done = True
            continue

        # Detect when ESP asks for user input (full or partial line)
        # Debounce: don't re-prompt within 2 seconds of last send
        if ("send any character" in low or "when ready" in low):
            if time.time() - last_prompt_sent > 2.0:
                input("    >> Press ENTER here to continue... ")
                ser.write(b"x\n")
                ser.flush()
                last_prompt_sent = time.time()

    print("=" * 50)
    print("  Setup done. Starting recording...\n")


def esp_reader(ser, data_queue, stop_event):
    """Thread: continuously read lines from ESP32 serial and enqueue them."""
    while not stop_event.is_set():
        try:
            line = serial_readline(ser, timeout=0.2)
            if line:
                data_queue.put((time.time(), line))
        except serial.SerialException:
            break
        except Exception:
            continue


def main():
    # -- Print loaded config -----------------------------------------
    print(f"[CONFIG] Loaded from {CONFIG_PATH}")

    # -- Connect DIGIT ------------------------------------------------
    print(f"[DIGIT] Connecting to {DIGIT_SERIAL}...")
    digit = Digit(DIGIT_SERIAL)
    digit.connect()
    digit.set_resolution(Digit.STREAMS[DIGIT_RES])
    digit.set_fps(DIGIT_FPS)
    digit.set_intensity(LED_INTENSITY)
    print(f"[DIGIT] Connected. LED intensity: {LED_INTENSITY}/15")

    # -- Connect ESP32 ------------------------------------------------
    esp_port = find_esp_port()
    if esp_port is None:
        print("[ESP32] ERROR: No serial port found. Plug it in and retry.")
        print("        Available ports:",
              [p.device for p in serial.tools.list_ports.comports()])
        digit.disconnect()
        sys.exit(1)

    print(f"[ESP32] Opening {esp_port} @ {ESP_BAUD}...")
    # Open without resetting ESP32 -- it may already be running/calibrated
    ser = serial.Serial(esp_port, ESP_BAUD, timeout=0.5)
    ser.dtr = False          # don't pull reset line
    ser.rts = False
    time.sleep(1)
    ser.reset_input_buffer()
    print("[ESP32] Connected.")

    # -- Run ESP32 calibration (interactive) --------------------------
    if DO_CALIBRATION:
        esp_calibration(ser)
    else:
        print("[ESP32] Skipping calibration (calibration: false in config).")
        ser.reset_input_buffer()

    # -- Create output folder (after calibration so aborted runs don't leave empty dirs)
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_DIR, f"recording_{session_ts}")
    img_dir = os.path.join(out_dir, "digit_frames")
    os.makedirs(img_dir, exist_ok=True)

    # -- Start ESP reader thread --------------------------------------
    stop_event = threading.Event()
    esp_queue = queue.Queue()
    esp_thread = threading.Thread(target=esp_reader,
                                  args=(ser, esp_queue, stop_event),
                                  daemon=True)
    esp_thread.start()

    # -- Open CSV logs ------------------------------------------------
    digit_csv_path = os.path.join(out_dir, "digit_log.csv")
    load_csv_path = os.path.join(out_dir, "loadcell_log.csv")

    digit_csv = open(digit_csv_path, "w", newline="")
    load_csv = open(load_csv_path, "w", newline="")

    digit_writer = csv.writer(digit_csv)
    digit_writer.writerow(["timestamp_epoch", "timestamp_iso",
                           "frame_idx", "image_file",
                           "weight_g_interp"])

    load_writer = csv.writer(load_csv)
    load_writer.writerow(["timestamp_epoch", "timestamp_iso",
                          "weight_g", "raw_line"])

    # -- Main loop ----------------------------------------------------
    # Frames are buffered and their CSV rows written with a small lag
    # so that interpolation has load cell readings on BOTH sides of each frame.
    # Images are still saved to disk immediately.
    grab_count = 0        # total frames grabbed from sensor
    save_count = 0        # frames actually saved to disk
    csv_written = 0       # CSV rows written (lags behind save_count)
    weight_buf = []       # rolling buffer of (timestamp, weight_g)
    WEIGHT_BUF_MAX = 100  # keep last 100 readings (~10s at 10Hz)
    # pending_frames: list of (epoch, iso_str, frame_idx, filename)
    # waiting for a future weight reading before CSV write
    pending_frames = []
    load_readings = 0
    print(f">>> Recording to {out_dir}")
    print(f">>> Saving every {SAVE_EVERY_N} frame(s) "
          f"({DIGIT_FPS / SAVE_EVERY_N:.0f} fps to disk)")
    print(f">>> Weight interpolation: {INTERPOLATION}")
    print(">>> Press 'q' in preview window or Ctrl+C to stop.\n")

    try:
        while True:
            # -- DIGIT frame --
            frame = digit.get_frame()
            now = time.time()
            grab_count += 1

            # -- Drain ESP queue -> build weight buffer --
            while not esp_queue.empty():
                t, line = esp_queue.get_nowait()
                weight = parse_weight(line)
                if weight is not None:
                    weight_buf.append((t, weight))
                    load_readings += 1
                    t_iso = datetime.fromtimestamp(t).strftime(
                        "%Y%m%d_%H%M%S_%f")
                    load_writer.writerow([f"{t:.6f}", t_iso,
                                          f"{weight:.2f}", line])
                    load_csv.flush()
            # trim buffer
            if len(weight_buf) > WEIGHT_BUF_MAX:
                weight_buf = weight_buf[-WEIGHT_BUF_MAX:]

            # -- Save image immediately --
            if grab_count % SAVE_EVERY_N == 0:
                ts_iso = datetime.fromtimestamp(now).strftime(
                    "%Y%m%d_%H%M%S_%f")
                fname = f"digit_{ts_iso}_{save_count:05d}.{IMAGE_FORMAT}"
                cv2.imwrite(os.path.join(img_dir, fname), frame)
                pending_frames.append((now, ts_iso, save_count, fname))
                save_count += 1

            # -- Flush pending frames whose timestamps now have a future
            #    weight reading available (i.e. weight_buf[-1] > frame_ts) --
            if pending_frames and weight_buf:
                latest_weight_ts = weight_buf[-1][0]
                flush_up_to = 0
                for i, (ft, fiso, fidx, ffname) in enumerate(pending_frames):
                    if latest_weight_ts > ft:
                        flush_up_to = i + 1
                    else:
                        break
                for ft, fiso, fidx, ffname in pending_frames[:flush_up_to]:
                    w = get_weight_for_frame(ft, weight_buf, INTERPOLATION)
                    digit_writer.writerow([f"{ft:.6f}", fiso,
                                           fidx, ffname,
                                           f"{w:.2f}"])
                    csv_written += 1
                if flush_up_to > 0:
                    pending_frames = pending_frames[flush_up_to:]
                    digit_csv.flush()

            # -- Preview + keyboard commands --
            cv2.imshow("DIGIT Live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                print("[CMD] Sending tare to ESP32...")
                ser.write(b"t")
            elif key == ord("c"):
                print("[CMD] Sending recalibrate to ESP32...")
                ser.write(b"c")

    except KeyboardInterrupt:
        print("\nStopping...")

    # -- Flush remaining pending frames (use last-known for tail) ----
    for ft, fiso, fidx, ffname in pending_frames:
        w = get_weight_for_frame(ft, weight_buf, INTERPOLATION)
        digit_writer.writerow([f"{ft:.6f}", fiso, fidx, ffname,
                               f"{w:.2f}"])
        csv_written += 1
    digit_csv.flush()

    # -- Cleanup ------------------------------------------------------
    stop_event.set()
    esp_thread.join(timeout=2)
    ser.close()
    digit.disconnect()
    digit_csv.close()
    load_csv.close()
    cv2.destroyAllWindows()

    print(f"\nDone. {save_count} DIGIT frames saved ({grab_count} grabbed)."
          f" {load_readings} load cell readings.")
    print(f"  DIGIT log    : {digit_csv_path}")
    print(f"  LoadCell log : {load_csv_path}")
    print(f"  Frames       : {img_dir}/")


if __name__ == "__main__":
    main()
