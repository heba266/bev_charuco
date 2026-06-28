"""
charuco_calibration.py
======================
Camera intrinsic calibration using a ChArUco board.

Board specs (calib.io):
  8 x 11 squares  |  checker = 20 mm  |  marker = 15 mm  |  DICT_4X4_50
  (corrected from misprinted 15mm/11mm -- see SQUARE_SIZE/MARKER_SIZE below)

INTERFACE:
  Green border  = board detected clearly, good to capture
  Red border    = board not detected or too few corners
  Yellow border = board detected but poor quality (too close/far/blurry)

CONTROLS:
  SPACE   → capture current frame
  C       → run calibration (need at least 15 captures)
  D       → delete last captured frame
  Q / ESC → quit

HOW TO USE:
  1. Run:  python3 charuco_calibration.py
  2. Move the board to different positions, angles, distances
  3. When border goes GREEN press SPACE to capture
  4. Aim for 20-30 captures covering all regions of the image
  5. Press C to calibrate and save results

TIPS FOR GOOD CALIBRATION:
  - Board in all corners of the image (top-left, top-right, bottom-left, bottom-right)
  - Board at different distances (close filling half image, medium, far)
  - Board tilted at different angles (left, right, up, down ~30-45 degrees)
  - Avoid motion blur — hold still when capturing
  - Good even lighting, no glare on the markers
  - Aim for RMS < 0.5 px (excellent), < 1.0 px (good)
"""

import cv2
import numpy as np
import json
import os
import time
from datetime import datetime

# ─────────────────────────────────────────────
# BOARD SETTINGS — must match your printed board
# ─────────────────────────────────────────────
SQUARES_X    = 11          # columns  (calib.io "8x11" = 11 cols)
SQUARES_Y    = 8           # rows
SQUARE_SIZE  = 0.02        # metres — 2 cm (corrected, measured value)
MARKER_SIZE  = 0.015       # metres — 15 mm (corrected, measured value)
ARUCO_DICT   = cv2.aruco.DICT_4X4_50

# ─────────────────────────────────────────────
# CAMERA SETTINGS
# ─────────────────────────────────────────────
CAMERA_INDEX     = "/dev/video2"   # change if your camera is on a different index
REQUESTED_WIDTH  = 1920            # explicit resolution request -- the camera
REQUESTED_HEIGHT = 1200            # will NOT necessarily default to this on
                                    # its own; see notes in open_camera() below

# ─────────────────────────────────────────────
# CALIBRATION SETTINGS
# ─────────────────────────────────────────────
MIN_CAPTURES     = 15      # minimum before calibration is allowed
TARGET_CAPTURES  = 25      # recommended number of captures
MIN_CORNERS      = 10      # minimum corners per frame to consider it good
BLUR_THRESHOLD   = 80.0    # Laplacian variance below this = too blurry

# ─────────────────────────────────────────────
# UI COLOURS  (BGR)
# ─────────────────────────────────────────────
GREEN  = (0,   210,  0  )
RED    = (0,   0,    210)
YELLOW = (0,   200,  200)
WHITE  = (255, 255,  255)
BLACK  = (0,   0,    0  )
DARK   = (30,  30,   30 )
CYAN   = (200, 200,  0  )


# ══════════════════════════════════════════════
#  OPEN CAMERA AT EXPLICIT RESOLUTION
# ══════════════════════════════════════════════

def open_camera(device, width, height):
    """
    Opens the camera and explicitly requests a resolution, using the V4L2
    backend directly (more reliable on Linux than letting OpenCV
    auto-select a backend, which can silently ignore .set() calls and
    fall back to whatever the device was last streaming).

    Returns (cap, actual_width, actual_height) -- ALWAYS check the
    actual values returned, since the camera may not support the exact
    resolution requested and will silently give you the nearest mode it
    actually has instead.
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, None, None

    # Request MJPG explicitly -- on this camera, MJPG is required to get
    # full framerate at higher resolutions (YUYV caps hard at 5fps above
    # 1280x720 per this camera's reported v4l2-ctl capabilities). For
    # STATIONARY calibration capture, framerate doesn't matter much, but
    # setting the format explicitly as part of the same negotiation also
    # tends to make the resolution request itself land more reliably.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, actual_w, actual_h


# ══════════════════════════════════════════════
#  BUILD BOARD
# ══════════════════════════════════════════════

def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_SIZE,
        MARKER_SIZE,
        dictionary
    )
    return board, dictionary


# ══════════════════════════════════════════════
#  QUALITY CHECKS
# ══════════════════════════════════════════════

def check_blur(gray):
    """Returns Laplacian variance — low value means blurry."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def check_coverage(corners, img_w, img_h):
    """
    Returns what fraction of the image area the detected corners cover.
    Higher = better — board covers more of the image.
    """
    if corners is None or len(corners) < 4:
        return 0.0
    pts = corners.reshape(-1, 2)
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    return (x_range * y_range) / (img_w * img_h)


def assess_frame_quality(corners, n_corners, blur_val, img_w, img_h):
    """
    Returns:  'good' | 'ok' | 'bad'  and a reason string.
    """
    if n_corners < MIN_CORNERS:
        return 'bad', f"Too few corners ({n_corners}) — show more of the board"

    if blur_val < BLUR_THRESHOLD:
        return 'bad', f"Too blurry ({blur_val:.0f}) — hold still"

    coverage = check_coverage(corners, img_w, img_h)
    if coverage < 0.03:
        return 'ok', "Board too far — move closer for this shot"
    if coverage > 0.85:
        return 'ok', "Board too close — move back slightly"

    if n_corners < 20:
        return 'ok', f"Only {n_corners} corners — tilt board more for more corners"

    return 'good', f"{n_corners} corners detected"


# ══════════════════════════════════════════════
#  DETECT CHARUCO CORNERS
# ══════════════════════════════════════════════

def detect_corners(gray, board, dictionary):
    """
    Detects ChArUco corners in a grayscale image.
    Returns (corners, ids, n_corners).
    """
    params   = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) < 2:
        return None, None, 0

    n, corners, ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )

    if n < 4 or corners is None:
        return None, None, 0

    return corners.reshape(-1, 1, 2).astype(np.float32), ids, n


# ══════════════════════════════════════════════
#  DRAW HUD OVERLAY
# ══════════════════════════════════════════════

def draw_hud(frame, status, reason, n_captures, n_corners, blur_val,
             border_color, last_capture_time):
    h, w = frame.shape[:2]

    # ── Thick border showing detection status ─────────────────
    cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, 8)

    # ── Top bar background ────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 40), DARK, -1)

    # ── Status text ───────────────────────────────────────────
    status_color = GREEN if status == 'good' else \
                   YELLOW if status == 'ok' else RED
    cv2.putText(frame, reason, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

    # ── Capture count ─────────────────────────────────────────
    count_color = GREEN if n_captures >= TARGET_CAPTURES else \
                  YELLOW if n_captures >= MIN_CAPTURES else WHITE
    count_text  = f"Captures: {n_captures}/{TARGET_CAPTURES}"
    cv2.putText(frame, count_text, (w - 220, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, count_color, 2, cv2.LINE_AA)

    # ── Bottom bar background ─────────────────────────────────
    cv2.rectangle(frame, (0, h-50), (w, h), DARK, -1)

    # ── Controls hint ─────────────────────────────────────────
    if n_captures >= MIN_CAPTURES:
        hint = "SPACE=capture  C=calibrate  D=delete last  Q=quit"
    else:
        hint = f"SPACE=capture  (need {MIN_CAPTURES - n_captures} more)  Q=quit"

    cv2.putText(frame, hint, (10, h-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    # ── Blur indicator ────────────────────────────────────────
    blur_color = GREEN if blur_val > BLUR_THRESHOLD * 1.5 else \
                 YELLOW if blur_val > BLUR_THRESHOLD else RED
    cv2.putText(frame, f"Sharp: {blur_val:.0f}", (10, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, blur_color, 1, cv2.LINE_AA)

    # ── "CAPTURED!" flash ─────────────────────────────────────
    if last_capture_time and (time.time() - last_capture_time) < 0.8:
        cv2.putText(frame, "CAPTURED!", (w//2 - 80, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 4, cv2.LINE_AA)

    return frame


def draw_coverage_guide(frame, all_corners_list):
    """
    Draw a small coverage map in bottom-right corner showing
    which regions of the image have been covered by captures.
    """
    h, w = frame.shape[:2]
    map_w, map_h = 120, 80
    map_x = w - map_w - 10
    map_y = h - map_h - 60

    # Background
    cv2.rectangle(frame,
        (map_x, map_y), (map_x + map_w, map_y + map_h),
        (50, 50, 50), -1)
    cv2.rectangle(frame,
        (map_x, map_y), (map_x + map_w, map_y + map_h),
        WHITE, 1)
    cv2.putText(frame, "coverage", (map_x + 15, map_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)

    # Draw dots for each capture's centroid
    for corners in all_corners_list:
        if corners is None:
            continue
        pts = corners.reshape(-1, 2)
        cx  = int(pts[:, 0].mean() / w * map_w) + map_x
        cy  = int(pts[:, 1].mean() / h * map_h) + map_y
        cv2.circle(frame, (cx, cy), 3, CYAN, -1)

    return frame


# ══════════════════════════════════════════════
#  CALIBRATION
# ══════════════════════════════════════════════

def run_calibration(all_corners, all_ids, board, image_size):
    """
    Runs cv2.calibrateCamera on captured frames.
    Returns K, dist, rms.
    """
    print(f"\n{'='*55}")
    print(f"  Running calibration on {len(all_corners)} frames...")
    print(f"{'='*55}")

    obj_points = []
    img_points = []

    for corners, ids in zip(all_corners, all_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is not None and len(obj_pts) >= 4:
            obj_points.append(obj_pts.reshape(-1, 1, 3).astype(np.float32))
            img_points.append(img_pts.reshape(-1, 1, 2).astype(np.float32))

    if len(obj_points) < 6:
        print(f"  ERROR: Only {len(obj_points)} valid frames after filtering.")
        print("  Capture more frames and try again.")
        return None, None, None

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    return K, dist, rms


def print_results(K, dist, rms):
    """Prints calibration results to terminal."""
    print(f"\n{'='*55}")
    print("  CALIBRATION RESULTS")
    print(f"{'='*55}")

    if rms < 0.5:
        quality = "Excellent ✓"
    elif rms < 1.0:
        quality = "Good ✓"
    elif rms < 2.0:
        quality = "Acceptable"
    else:
        quality = "Poor — recapture with more varied angles"

    print(f"\n  RMS re-projection error: {rms:.4f} px  ({quality})")
    print(f"\n  Camera Matrix (K):")
    print(f"    fx = {K[0,0]:.4f} px    fy = {K[1,1]:.4f} px")
    print(f"    cx = {K[0,2]:.4f} px    cy = {K[1,2]:.4f} px")
    print(f"\n  Full K matrix:")
    print(f"    {K}")
    print(f"\n  Distortion coefficients (k1, k2, p1, p2, k3):")
    print(f"    {dist.ravel()}")
    print(f"{'='*55}\n")


def save_results(K, dist, rms, image_size, output_path="intrinsics_pinhole.json"):
    """Saves results to JSON — compatible with bev_pipeline_pinhole.py."""
    data = {
        "model":        "pinhole",
        "rms_px":       rms,
        "image_width":  image_size[0],
        "image_height": image_size[1],
        "K": K.tolist(),
        "dist": dist.tolist(),
        "K_readable": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        },
        "dist_readable": {
            "k1": float(dist.ravel()[0]),
            "k2": float(dist.ravel()[1]),
            "p1": float(dist.ravel()[2]),
            "p2": float(dist.ravel()[3]),
            "k3": float(dist.ravel()[4]),
        },
        "board": {
            "squares_x":     SQUARES_X,
            "squares_y":     SQUARES_Y,
            "square_size_m": SQUARE_SIZE,
            "marker_size_m": MARKER_SIZE,
            "aruco_dict":    "DICT_4X4_50",
        },
        "calibrated_at": datetime.now().isoformat(),
        "note": "Camera outputs hardware-undistorted images — "
                "pinhole model used, no fisheye model needed."
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  [SAVED] {output_path}")
    return output_path


def show_undistortion_preview(frame, K, dist):
    """
    Shows original vs undistorted side by side.
    Lets you visually verify the calibration is correct.
    """
    undistorted = cv2.undistort(frame, K, dist)
    preview = np.hstack([frame, undistorted])

    # Labels
    h = preview.shape[0]
    cv2.rectangle(preview, (0, 0), (frame.shape[1], 30), DARK, -1)
    cv2.putText(preview, "Original", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.rectangle(preview,
        (frame.shape[1], 0), (preview.shape[1], 30), DARK, -1)
    cv2.putText(preview, "Undistorted", (frame.shape[1] + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

    cv2.namedWindow("Undistortion preview — press any key to close",
                    cv2.WINDOW_NORMAL)
    cv2.imshow("Undistortion preview — press any key to close", preview)
    cv2.waitKey(0)
    cv2.destroyWindow(
        "Undistortion preview — press any key to close")


# ══════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════

def main():
    board, dictionary = make_board()

    # ── Open camera at the requested resolution ───────────────
    cap, actual_w, actual_h = open_camera(
        CAMERA_INDEX, REQUESTED_WIDTH, REQUESTED_HEIGHT
    )
    if cap is None:
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX at the top of the script, or "
              "check 'fuser -v <device>' if it's held by another process.")
        return

    if (actual_w, actual_h) != (REQUESTED_WIDTH, REQUESTED_HEIGHT):
        print(f"[WARN] Requested {REQUESTED_WIDTH}x{REQUESTED_HEIGHT} but "
              f"camera gave {actual_w}x{actual_h} instead.")
        print(f"       Check 'v4l2-ctl -d {CAMERA_INDEX} --list-formats-ext' "
              f"to confirm this resolution is actually supported.")
        print(f"       Continuing with {actual_w}x{actual_h} -- this IS what "
              f"will be used for calibration below, so it's still valid, "
              f"just not what you asked for.")

    ret, test_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read from camera.")
        return

    image_size = (test_frame.shape[1], test_frame.shape[0])
    print(f"\n{'='*55}")
    print("  ChArUco Calibration")
    print(f"{'='*55}")
    print(f"  Board:   {SQUARES_X} x {SQUARES_Y} squares")
    print(f"  Square:  {SQUARE_SIZE*1000:.0f} mm  |  "
          f"Marker: {MARKER_SIZE*1000:.0f} mm")
    print(f"  Camera:  {image_size[0]} x {image_size[1]} px")
    print(f"\n  SPACE=capture  C=calibrate  D=delete  Q=quit")
    print(f"{'='*55}\n")

    # ── Storage ───────────────────────────────────────────────
    all_corners       = []   # list of (N,1,2) corner arrays
    all_ids           = []   # list of (N,1) id arrays
    all_corners_raw   = []   # for coverage map (unreshaped)
    last_capture_time = None
    K = dist = rms    = None

    win_name = "ChArUco Calibration  [SPACE=capture | C=calibrate | Q=quit]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_val    = check_blur(gray)
        h, w        = frame.shape[:2]

        # ── Detect ────────────────────────────────────────────
        corners, ids, n_corners = detect_corners(gray, board, dictionary)

        # ── Assess quality ────────────────────────────────────
        status, reason = assess_frame_quality(
            corners, n_corners, blur_val, w, h)

        # ── Border colour ─────────────────────────────────────
        border_color = GREEN  if status == 'good' else \
                       YELLOW if status == 'ok'   else RED

        # ── Draw detected corners on frame ────────────────────
        display = frame.copy()
        if corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                display, corners, ids,
                cornerColor=(0, 255, 100))

        # ── Draw HUD ──────────────────────────────────────────
        draw_hud(display, status, reason,
                 len(all_corners), n_corners, blur_val,
                 border_color, last_capture_time)

        draw_coverage_guide(display, all_corners_raw)

        cv2.imshow(win_name, display)

        # ── Key handling ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        # SPACE — capture
        if key == ord(' '):
            if status == 'bad':
                print(f"  [!] Not captured — {reason}")
            else:
                if corners is not None:
                    all_corners.append(corners)
                    all_ids.append(ids)
                    all_corners_raw.append(corners.copy())
                    last_capture_time = time.time()
                    quality_tag = "GOOD" if status == 'good' else "OK"
                    print(f"  [+] Frame {len(all_corners):02d} captured "
                          f"({quality_tag}, {n_corners} corners, "
                          f"blur={blur_val:.0f})")
                else:
                    print("  [!] No corners detected — move board into view")

        # C — calibrate
        elif key == ord('c'):
            if len(all_corners) < MIN_CAPTURES:
                print(f"  [!] Need at least {MIN_CAPTURES} captures "
                      f"(have {len(all_corners)})")
            else:
                cv2.destroyWindow(win_name)
                K, dist, rms = run_calibration(
                    all_corners, all_ids, board, image_size)

                if K is not None:
                    print_results(K, dist, rms)
                    saved = save_results(K, dist, rms, image_size)

                    # Ask user if they want to see undistortion preview
                    print("  Show undistortion preview? (y/n): ", end="")
                    ans = input().strip().lower()
                    if ans == 'y':
                        show_undistortion_preview(frame, K, dist)

                    print(f"\n  Done! Results saved to: {saved}")
                    print("  You can now use intrinsics_pinhole.json "
                          "in your BEV pipeline.")
                    break
                else:
                    # Re-open window if calibration failed
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # D — delete last capture
        elif key == ord('d'):
            if all_corners:
                all_corners.pop()
                all_ids.pop()
                all_corners_raw.pop()
                print(f"  [-] Deleted last capture "
                      f"({len(all_corners)} remaining)")
            else:
                print("  [!] Nothing to delete")

        # Q or ESC — quit
        elif key in (ord('q'), 27):
            print("  Exiting without calibrating.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
