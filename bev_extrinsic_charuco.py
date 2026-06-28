"""
bev_extrinsic_charuco.py
=========================
Extrinsic (ground-plane homography) calibration using a ChArUco board,
adapted from bev_full_pipeline.py's Part B + Part C.

Use this AFTER you already have intrinsics from charuco_calibration.py
(intrinsics_pinhole.json). This script only does:

  STEP 2 — Extrinsic calibration  (one image, ChArUco board flat on ground)
  STEP 3 — BEV warp

It is built to compare MULTIPLE height/pitch candidates per camera
position (front / left / right), so you can pick the best one before
fixing the cameras permanently on the rover.

──────────────────────────────────────────────────────────────────
BOARD SETTINGS — must match charuco_calibration.py
──────────────────────────────────────────────────────────────────
  8 x 11 squares | checker = 20 mm | marker = 15 mm | DICT_4X4_50

──────────────────────────────────────────────────────────────────
GROUND PLACEMENT (read before shooting)
──────────────────────────────────────────────────────────────────
  - Board must lie FLAT on the ground (tape down paper boards — any
    curl/warp breaks the planarity assumption the homography relies on).
  - Keep the board at the SAME physical ground position for every
    height/pitch candidate of a given camera. You are testing which
    camera pose gives the best homography for a FIXED target zone —
    only the camera moves between shots, not the board.
  - FRONT camera: place board ~1.5-2.5 m directly ahead of the camera's
    ground projection point (centered in the lane). This anchors the
    homography in your real working zone, not at an extreme of the FOV.
  - LEFT/RIGHT cameras: place board ~0.3-1.0 m laterally from the
    chassis, at the near-field distance where you actually expect to
    see lane lines beside the robot.

──────────────────────────────────────────────────────────────────
WHY THE OUTPUT CAN BE MOSTLY BLACK (read this if BEV looks empty)
──────────────────────────────────────────────────────────────────
The ChArUco board's own local coordinate origin (0,0) is one corner of
the physical board -- NOT the point on the ground directly under your
camera. If you place the board ~2m in front of the camera (as the
placement guidance below recommends), then "world (0,0)" is 2 metres
away from where the BEV window assumes the camera is centered. This
script automatically recovers the camera's true ground-projection point
via solvePnP and re-centers all ground coordinates around THAT point
before computing the homography -- so (0,0) in the final output is
always directly under the camera, regardless of where you placed the
board. Without this step, a perfectly correct homography (low
reprojection error, all RANSAC inliers) can still produce a BEV image
that is almost entirely black, because the output window and the
visible ground region are offset from each other and barely overlap.

──────────────────────────────────────────────────────────────────
HOW TO USE
──────────────────────────────────────────────────────────────────
For each camera position and each height/pitch candidate you test:

    python3 bev_extrinsic_charuco.py \
        --ground_image  front_h65_p60.jpg \
        --intrinsics_json intrinsics_pinhole.json \
        --camera front \
        --height_cm 65 \
        --pitch_deg 60

This will:
  1. Load K, DIST from intrinsics_pinhole.json
  2. Undistort the image
  3. Detect ChArUco corners
  4. Compute homography (image pixel -> ground metres) via RANSAC
  5. Print reprojection error (mean / max, in cm)
  6. Save a BEV preview image + append results to results_summary.json

After testing all candidates for all 3 cameras, inspect
results_summary.json to compare reprojection error and pick the winner
per camera. Then re-run once more on the WINNING shot per camera to
produce the final bev_H.json used by your lane-detection pipeline.

    python3 bev_extrinsic_charuco.py \
        --ground_image  front_BEST.jpg \
        --intrinsics_json intrinsics_pinhole.json \
        --camera front \
        --height_cm 65 --pitch_deg 60 \
        --finalize
"""

import cv2
import numpy as np
import json
import os
import argparse
from datetime import datetime

# ─────────────────────────────────────────────
# BOARD SETTINGS — must match charuco_calibration.py
# ─────────────────────────────────────────────
SQUARES_X   = 11          # columns (calib.io "8x11" = 11 cols)
SQUARES_Y   = 8           # rows
SQUARE_SIZE = 0.02       # metres — 2cm
MARKER_SIZE = 0.015       # metres — 11 mm
ARUCO_DICT  = cv2.aruco.DICT_4X4_50

# ─────────────────────────────────────────────
# BEV OUTPUT REGION (per-camera, since front vs side cameras
# care about different zones — adjust to taste)
# ─────────────────────────────────────────────
BEV_REGIONS = {
    "front": dict(x_left=-2.0, x_right=2.0, y_near=0.3, y_far=5.0),
    "left":  dict(x_left=-2.5, x_right=0.3, y_near=-2.0, y_far=2.0),
    "right": dict(x_left=-0.3, x_right=2.5, y_near=-2.0, y_far=2.0),
}
PIXELS_PER_METER = 100

RESULTS_FILE = "results_summary.json"


# ══════════════════════════════════════════════════════════════════
#  VERSION-SAFE ARUCO API (works on both old <4.7 and new >=4.7 cv2)
# ══════════════════════════════════════════════════════════════════

_HAS_NEW_ARUCO_API = hasattr(cv2.aruco, "ArucoDetector")


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    if hasattr(cv2.aruco, "CharucoBoard") and _HAS_NEW_ARUCO_API:
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, dictionary
        )
    else:
        # Old API constructor signature
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_X, SQUARES_Y, SQUARE_SIZE, MARKER_SIZE, dictionary
        )
    return board, dictionary


def detect_charuco_corners(gray, board, dictionary):
    """
    Detects ChArUco corners in a grayscale image.
    Returns (corners (N,1,2) float32, ids (N,1) int, n_corners).
    """
    if _HAS_NEW_ARUCO_API:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=params
        )

    if marker_ids is None or len(marker_ids) < 2:
        return None, None, 0

    n, corners, ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )

    if n < 4 or corners is None:
        return None, None, 0

    return corners.reshape(-1, 1, 2).astype(np.float32), ids, n


def get_charuco_world_points(board):
    """
    Returns a dict mapping charuco corner ID -> (x, y) ground-plane
    position in metres, using the board's own chessboardCorners
    (already in the board's local metric frame, Z=0).
    """
    if hasattr(board, "getChessboardCorners"):
        all_pts = board.getChessboardCorners()   # new API method
    else:
        all_pts = board.chessboardCorners        # old API attribute

    world = {}
    for idx, pt in enumerate(all_pts):
        world[idx] = (float(pt[0]), float(pt[1]))
    return world


def get_camera_ground_origin(img_pts, world_pts_board_frame, K, dist):
    """
    Recovers the camera's true extrinsics (R, t) relative to the BOARD's
    own local frame via solvePnP, then computes where the camera's optical
    axis / straight-down ray actually hits the ground plane (Z=0), IN THE
    BOARD'S COORDINATE FRAME.

    This is the camera's "ground projection point" expressed in board
    coordinates — i.e. how far the camera's true ground-center point is
    from the board's corner (0,0).

    Returns (origin_x, origin_y) in metres, in the board's local frame.

    WHY THIS MATTERS:
    The board's local frame has its origin at one corner of the physical
    board (OpenCV's internal convention), NOT at the camera's own ground
    projection. If the board is placed e.g. 2m in front of the camera (as
    it should be, per placement guidance), then "world point (0,0)" is
    NOT under the camera -- it's 2m away, at the board's corner. Defining
    a BEV window like x in [-2,2], y in [0.3,5] only makes sense if (0,0)
    is roughly at/under the camera. Without this correction, the BEV
    window and the actual visible ground region are offset from each
    other, and most of the warped output samples points the camera never
    saw -- producing a mostly-black result despite a perfectly correct
    homography fit.
    """
    obj_pts_3d = np.array(
        [[x, y, 0.0] for (x, y) in world_pts_board_frame], dtype=np.float64
    )
    img_pts_2d = np.array(img_pts, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts_3d, img_pts_2d, K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    # Camera center in board-frame coordinates: C = -R^T @ t
    cam_center_board_frame = -R.T @ tvec.reshape(3)

    # Camera center's (x, y) directly above/below it on the board's
    # Z=0 plane IS simply its (x, y) components, since the board frame
    # already defines Z=0 as the ground plane.
    origin_x = float(cam_center_board_frame[0])
    origin_y = float(cam_center_board_frame[1])

    return origin_x, origin_y


# ══════════════════════════════════════════════════════════════════
#  INTRINSICS I/O
# ══════════════════════════════════════════════════════════════════

def load_intrinsics(path):
    with open(path) as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"] if "dist" in data else data["DIST"],
                     dtype=np.float64)
    rms = data.get("rms_px", data.get("rms", "?"))
    print(f"[LOADED] Intrinsics from {path}  (calibration RMS was {rms} px)")
    return K, dist


# ══════════════════════════════════════════════════════════════════
#  EXTRINSIC: HOMOGRAPHY FROM CHARUCO ON GROUND
# ══════════════════════════════════════════════════════════════════

def compute_homography_from_ground(img_path, K, dist, board, dictionary):
    """
    STEP 2 — Compute homography H mapping image pixels -> ground metres,
    using a ChArUco board lying flat on the ground.

    Returns (H, img_undist, mean_err_cm, max_err_cm, n_inliers, n_total)
    or (None, None, None, None, None, None) on failure.
    """
    img_raw = cv2.imread(img_path)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    # ── 1. Undistort ──────────────────────────────────────────────
    img_undist = cv2.undistort(img_raw, K, dist)
    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)

    print(f"\n[EXTRINSIC] Image: {img_raw.shape[1]}x{img_raw.shape[0]}")
    print(f"[EXTRINSIC] Looking for ChArUco corners "
          f"({SQUARES_X}x{SQUARES_Y} squares, "
          f"checker={SQUARE_SIZE*1000:.0f}mm, marker={MARKER_SIZE*1000:.0f}mm) ...")

    # ── 2. Detect ChArUco corners ─────────────────────────────────
    corners, ids, n_corners = detect_charuco_corners(gray, board, dictionary)
    if corners is None or n_corners < 4:
        print("[FAIL] ChArUco board not detected (or too few corners) "
              "in the ground image.")
        print("  - Check board must be FULLY visible and FLAT on the ground.")
        print("  - Avoid glare / shadows / motion blur.")
        print("  - Confirm SQUARES_X/SQUARES_Y/SQUARE_SIZE/MARKER_SIZE match "
              "the physical board.")
        return None, None, None, None, None, None

    print(f"[EXTRINSIC] Detected {n_corners} ChArUco corners")

    # ── 3. Map detected corners to known ground-plane (x, y) ───────
    world_lookup = get_charuco_world_points(board)
    img_pts = []
    world_pts = []
    for corner_xy, corner_id in zip(corners.reshape(-1, 2), ids.reshape(-1)):
        wid = int(corner_id)
        if wid in world_lookup:
            img_pts.append(corner_xy)
            world_pts.append(world_lookup[wid])

    img_pts = np.array(img_pts, dtype=np.float32)
    world_pts_board_frame = np.array(world_pts, dtype=np.float32)

    if len(img_pts) < 4:
        print(f"[FAIL] Only {len(img_pts)} corners mapped to known world "
              f"points — need at least 4.")
        return None, None, None, None, None, None

    # ── 4. Recover camera's true ground-projection origin ──────────
    # The board's own (0,0) is a corner of the physical board, not the
    # camera's ground position. Re-center world coordinates so (0,0) is
    # directly under the camera instead -- otherwise the BEV output
    # window (defined relative to the camera) and the actual visible
    # ground (defined relative to the board corner) don't line up, and
    # the warp samples mostly empty space even with a perfect homography.
    cam_origin = get_camera_ground_origin(img_pts, world_pts_board_frame, K, dist)
    if cam_origin is None:
        print("[WARN] Could not recover camera ground origin via solvePnP — "
              "falling back to board-corner-relative coordinates. BEV output "
              "may be misaligned with the expected window.")
        origin_x, origin_y = 0.0, 0.0
    else:
        origin_x, origin_y = cam_origin
        print(f"[EXTRINSIC] Camera ground-projection point, relative to "
              f"board corner: x={origin_x:+.3f}m  y={origin_y:+.3f}m")

    # Shift world points so (0,0) = directly under the camera
    world_pts = world_pts_board_frame.copy()
    world_pts[:, 0] -= origin_x
    world_pts[:, 1] -= origin_y

    # ── 5. RANSAC homography (now in camera-centered ground frame) ──
    H, mask = cv2.findHomography(
        img_pts, world_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=0.01   # metres — tight
    )

    if H is None:
        print("[FAIL] cv2.findHomography returned None.")
        return None, None, None, None, None, None

    n_inliers = int(mask.sum())
    n_total = len(img_pts)
    print(f"[EXTRINSIC] RANSAC inliers: {n_inliers}/{n_total}")

    if n_inliers < 8:
        print("[WARN] Too few inliers — result may be unreliable. "
              "Consider re-shooting this candidate.")

    # ── 6. Reprojection error check ─────────────────────────────────
    errs = []
    for px, gnd in zip(img_pts, world_pts):
        pt = H @ np.array([px[0], px[1], 1.0])
        pt /= pt[2]
        errs.append(np.linalg.norm(pt[:2] - gnd))
    mean_err_cm = float(np.mean(errs) * 100)
    max_err_cm = float(np.max(errs) * 100)
    print(f"[EXTRINSIC] Reprojection error:  "
          f"mean={mean_err_cm:.2f} cm   max={max_err_cm:.2f} cm")

    # ── 7. Visual check ─────────────────────────────────────────────
    vis = img_undist.copy()
    cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids,
                                          cornerColor=(0, 255, 100))
    out_name = f"corners_detected_{os.path.splitext(os.path.basename(img_path))[0]}.png"
    cv2.imwrite(out_name, vis)
    print(f"[SAVED] {out_name} — inspect this to confirm detection!")

    return H, img_undist, mean_err_cm, max_err_cm, n_inliers, n_total


# ══════════════════════════════════════════════════════════════════
#  BEV WARP
# ══════════════════════════════════════════════════════════════════

def warp_to_bev(img_undist, H_img_to_ground, region):
    x_left, x_right = region["x_left"], region["x_right"]
    y_near, y_far = region["y_near"], region["y_far"]

    bev_w = int((x_right - x_left) * PIXELS_PER_METER)
    bev_h = int((y_far - y_near) * PIXELS_PER_METER)

    S = np.array([
        [1.0 / PIXELS_PER_METER, 0, x_left],
        [0, -1.0 / PIXELS_PER_METER, y_far],
        [0, 0, 1]
    ], dtype=np.float64)

    H_bev_to_img = np.linalg.inv(H_img_to_ground) @ S
    H_img_to_bev = np.linalg.inv(H_bev_to_img)

    bev = cv2.warpPerspective(
        img_undist, H_img_to_bev, (bev_w, bev_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(30, 30, 30)
    )

    for d in range(int(np.floor(y_near)), int(np.ceil(y_far)) + 1):
        if y_near <= d <= y_far:
            row = int((y_far - d) * PIXELS_PER_METER)
            cv2.line(bev, (0, row), (bev_w, row), (60, 60, 60), 1)
            cv2.putText(bev, f"{d}m", (4, max(row - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    bar_y = bev_h - 20
    cv2.line(bev, (20, bar_y), (20 + PIXELS_PER_METER, bar_y), (0, 255, 0), 2)
    cv2.putText(bev, "1 m", (20, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return bev


# ══════════════════════════════════════════════════════════════════
#  RESULTS LOG  (compare candidates across height/pitch combos)
# ══════════════════════════════════════════════════════════════════

def append_result(camera, height_cm, pitch_deg, image_path,
                   mean_err_cm, max_err_cm, n_inliers, n_total):
    entry = {
        "camera": camera,
        "height_cm": height_cm,
        "pitch_deg": pitch_deg,
        "image": os.path.basename(image_path),
        "mean_err_cm": round(mean_err_cm, 3),
        "max_err_cm": round(max_err_cm, 3),
        "inliers": n_inliers,
        "total_corners": n_total,
        "timestamp": datetime.now().isoformat(),
    }

    results = []
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []

    results.append(entry)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[LOGGED] Appended result to {RESULTS_FILE}")

    # Show current ranking for this camera so far
    same_cam = [r for r in results if r["camera"] == camera]
    same_cam.sort(key=lambda r: r["mean_err_cm"])
    print(f"\n[RANKING] Candidates so far for camera='{camera}' "
          f"(sorted by mean error, lower=better):")
    for r in same_cam:
        print(f"  h={r['height_cm']:>5}cm  p={r['pitch_deg']:>4}deg  "
              f"mean={r['mean_err_cm']:>6.2f}cm  max={r['max_err_cm']:>6.2f}cm  "
              f"inliers={r['inliers']}/{r['total_corners']}  "
              f"({r['image']})")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="ChArUco-based extrinsic calibration + BEV warp, "
                     "for comparing height/pitch candidates per camera."
    )
    p.add_argument("--ground_image", required=True,
                    help="Image with ChArUco board FLAT on the ground")
    p.add_argument("--intrinsics_json", default="intrinsics_pinhole.json",
                    help="Path to intrinsics JSON from charuco_calibration.py")
    p.add_argument("--camera", required=True, choices=["front", "left", "right"],
                    help="Which camera position this shot is for")
    p.add_argument("--height_cm", type=float, required=True,
                    help="Camera height above ground for this candidate (cm)")
    p.add_argument("--pitch_deg", type=float, required=True,
                    help="Camera pitch for this candidate (deg from horizontal, "
                         "90=straight down)")
    p.add_argument("--finalize", action="store_true",
                    help="Save this candidate's homography into the final "
                         "bev_H.json (use once you've picked the winner)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.intrinsics_json):
        print(f"[ERROR] Intrinsics file not found: {args.intrinsics_json}")
        print("  Run charuco_calibration.py first.")
        return

    K, dist = load_intrinsics(args.intrinsics_json)
    board, dictionary = make_board()

    H, img_undist, mean_err_cm, max_err_cm, n_inliers, n_total = \
        compute_homography_from_ground(args.ground_image, K, dist, board, dictionary)

    if H is None:
        print("[ERROR] Could not compute homography. Nothing saved.")
        return

    append_result(args.camera, args.height_cm, args.pitch_deg,
                   args.ground_image, mean_err_cm, max_err_cm,
                   n_inliers, n_total)

    region = BEV_REGIONS[args.camera]
    bev = warp_to_bev(img_undist, H, region)
    bev_out = f"bev_{args.camera}_h{int(args.height_cm)}_p{int(args.pitch_deg)}.png"
    cv2.imwrite(bev_out, bev)
    print(f"[SAVED] {bev_out}")

    if args.finalize:
        final_path = "bev_H.json"
        final_data = {}
        if os.path.isfile(final_path):
            with open(final_path) as f:
                final_data = json.load(f)
        final_data[args.camera] = {
            "H": H.tolist(),
            "height_cm": args.height_cm,
            "pitch_deg": args.pitch_deg,
            "mean_err_cm": mean_err_cm,
            "max_err_cm": max_err_cm,
            "source_image": os.path.basename(args.ground_image),
        }
        with open(final_path, "w") as f:
            json.dump(final_data, f, indent=2)
        print(f"[FINALIZED] {args.camera} homography saved to {final_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
