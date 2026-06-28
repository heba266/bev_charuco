"""
live_detection_check.py
========================
Live camera preview with REAL-TIME ChArUco marker detection feedback.
No photo-then-check loop -- point the camera, see detection status update
live as you change resolution, distance, or camera settings.

Pairs with marker_size_sanity_check.py (same px/cell thresholds), but
gives you instant visual feedback instead of having to capture and
re-run a script each time.

──────────────────────────────────────────────────────────────────
HOW TO USE
──────────────────────────────────────────────────────────────────
    python3 live_detection_check.py --device /dev/video2 --width 1280 --height 720

    # try several resolutions back to back, just change --width/--height
    python3 live_detection_check.py --device /dev/video2 --width 960 --height 720
    python3 live_detection_check.py --device /dev/video2 --width 640 --height 480

ON-SCREEN DISPLAY:
  - Green border  = markers detected, px/cell ABOVE usable threshold
  - Yellow border = markers detected, but px/cell is MARGINAL
  - Red border    = no markers detected (too small / out of frame / etc)
  - Top-left text shows: markers detected, avg marker size in px,
    estimated px/cell, and current resolution/fps actually being
    delivered by the camera (which may differ from what you requested,
    same as you saw with the ROS launch file -- the driver can silently
    cap fps at certain resolutions)

CONTROLS:
  S       -> save current frame to disk (for later use with
             bev_extrinsic_charuco.py once you've found a good resolution)
  Q / ESC -> quit

NOTE: this uses cv2.VideoCapture directly (NOT through ROS2 / v4l2_camera).
It's meant as a fast way to A/B test resolutions before committing to one
in your ROS launch file. Once you pick a resolution, set it in your
launch file the same way we did for 1920x1080, and redo the FULL
intrinsic + extrinsic calibration at that resolution -- this script does
not replace that, it just helps you choose the resolution faster.
"""

import cv2
import numpy as np
import argparse
import time
import os

# ── Board geometry -- MUST match your actual (corrected) board ─────
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# ── Thresholds (same as marker_size_sanity_check.py) ────────────────
MIN_PX_PER_CELL_USABLE = 4.0
MIN_PX_PER_CELL_MARGINAL = 3.0

GREEN = (0, 210, 0)
YELLOW = (0, 200, 200)
RED = (0, 0, 210)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)


def estimate_px_per_cell(corners):
    if corners is None or len(corners) == 0:
        return None, None
    side_lengths = []
    for c in corners:
        pts = c.reshape(4, 2)
        sides = [
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[3]),
            np.linalg.norm(pts[3] - pts[0]),
        ]
        side_lengths.append(np.mean(sides))
    avg_marker_px = float(np.mean(side_lengths))
    px_per_cell = avg_marker_px / 6.0   # DICT_4X4_50 = 6x6 cells incl. border
    return avg_marker_px, px_per_cell


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0",
                    help="Camera index (e.g. 0) or device path "
                         "(e.g. /dev/video2)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--save_dir", default=".",
                    help="Where to save frames when pressing S")
    args = p.parse_args()

    # cv2.VideoCapture accepts either an int index or a string path
    device = args.device
    try:
        device = int(device)
    except ValueError:
        pass  # keep as string path, e.g. "/dev/video2"

    # Explicitly request the V4L2 backend. Without this, OpenCV may pick
    # a different backend (e.g. GStreamer) that doesn't reliably honour
    # resolution .set() calls on Linux, silently falling back to
    # whatever the device was already streaming.
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera: {args.device}")
        print("  Try a different index/path, or check it's not already "
              "in use by another process (same issue as before -- check "
              "'fuser -v /dev/videoN' if this fails).")
        return

    # Set FOURCC explicitly too -- some drivers only apply a resolution
    # change correctly when the pixel format is (re-)set in the same
    # negotiation. MJPG is the most broadly supported high-res format
    # for UVC webcams (confirmed available on your camera via v4l2-ctl).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Read back what was ACTUALLY negotiated, before any frames are read
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print(f"[DEBUG] Backend reports format: {fourcc_str} @ "
          f"{actual_w}x{actual_h} (requested {args.width}x{args.height})")

    if (actual_w, actual_h) != (args.width, args.height):
        print(f"[WARN] Requested {args.width}x{args.height} but camera "
              f"gave {actual_w}x{actual_h}.")
        print(f"  This usually means one of:")
        print(f"  1. That exact WIDTHxHEIGHT isn't in your camera's "
              f"supported list -- double check with:")
        print(f"       v4l2-ctl -d {args.device} --list-formats-ext")
        print(f"  2. The device is still held by another process -- check:")
        print(f"       fuser -v {args.device}")
        print(f"  3. A stale OpenCV/V4L2 negotiation -- try unplugging and "
              f"replugging the USB camera, or rebooting if this persists.")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    win_name = "Live ChArUco Detection Check  [S=save | Q=quit]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    os.makedirs(args.save_dir, exist_ok=True)
    save_count = 0

    # Rolling fps measurement of what's ACTUALLY being delivered, since
    # the driver can silently cap this regardless of what was requested
    # (same behaviour you saw via the ROS2 launch file).
    frame_times = []

    print(f"Requested {args.width}x{args.height} -- got {actual_w}x{actual_h}")
    print("Press S to save a frame, Q/ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        now = time.time()
        frame_times.append(now)
        frame_times = [t for t in frame_times if now - t < 2.0]
        live_fps = len(frame_times) / 2.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        n_detected = 0 if ids is None else len(ids)

        avg_marker_px, px_per_cell = estimate_px_per_cell(corners)

        if n_detected == 0:
            border_color = RED
            verdict = "NO MARKERS DETECTED"
        elif px_per_cell >= MIN_PX_PER_CELL_USABLE:
            border_color = GREEN
            verdict = "USABLE"
        elif px_per_cell >= MIN_PX_PER_CELL_MARGINAL:
            border_color = YELLOW
            verdict = "MARGINAL"
        else:
            border_color = RED
            verdict = "TOO SMALL"

        display = frame.copy()
        if corners is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids,
                                            borderColor=(0, 255, 100))

        h, w = display.shape[:2]
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), border_color, 8)
        cv2.rectangle(display, (0, 0), (w, 75), DARK, -1)

        cv2.putText(display, f"markers: {n_detected}   verdict: {verdict}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

        if px_per_cell is not None:
            cv2.putText(display,
                        f"avg marker: {avg_marker_px:.1f}px   "
                        f"px/cell: {px_per_cell:.2f}  "
                        f"(need >= {MIN_PX_PER_CELL_USABLE})",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(display, f"px/cell: -- (no markers to measure)",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        cv2.putText(display,
                    f"res: {actual_w}x{actual_h}   live fps: {live_fps:.1f}   "
                    f"rejected candidates: {len(rejected) if rejected is not None else 0}",
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            save_count += 1
            fname = os.path.join(args.save_dir,
                                  f"live_check_{actual_w}x{actual_h}_{save_count:02d}.png")
            cv2.imwrite(fname, frame)
            print(f"[SAVED] {fname}  (markers={n_detected}, "
                  f"px/cell={px_per_cell if px_per_cell else 'N/A'})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
