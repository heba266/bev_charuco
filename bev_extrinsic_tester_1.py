"""
bev_extrinsic_tester.py
=======================
Tests multiple camera height/pitch combinations using ChArUco boards
laid flat on the ground.

For cameras that output hardware-undistorted images (pinhole model).
No fisheye undistortion needed — image arrives already clean.

WORKFLOW:
  1. Lay ChArUco boards flat on ground in front of camera
  2. Take photo at each height/pitch combination
  3. Run this script on all photos
  4. Script shows BEV result for each — pick the best one
  5. Best H is saved to JSON for use in your BEV node

HOW TO RUN:
  # Test one image:
  python3 bev_extrinsic_tester.py \
      --images front_h090_p20.jpg \
      --intrinsics intrinsics_front.json

  # Test multiple images at once (compare all):
  python3 bev_extrinsic_tester.py \
      --images front_h090_p20.jpg front_h090_p15.jpg front_h100_p20.jpg \
      --intrinsics intrinsics_front.json

  # Test left or right camera:
  python3 bev_extrinsic_tester.py \
      --images left_h090_p20.jpg \
      --intrinsics intrinsics_left.json \
      --camera left

BOARD SETTINGS — must match your calib.io board:
  8 x 11 squares | checker = 15mm | marker = 11mm | DICT_4X4_50
  (SQUARES_X=11 cols, SQUARES_Y=8 rows)
"""

import cv2
import numpy as np
import json
import os
import sys
import argparse


# ─────────────────────────────────────────────
# BOARD SETTINGS — calib.io 8x11 board
# ─────────────────────────────────────────────
SQUARES_X   = 11        # columns  (calib.io "8x11" = 11 cols)
SQUARES_Y   = 8         # rows
SQUARE_SIZE = 0.02     # metres — 15 mm
MARKER_SIZE = 0.015     # metres — 11 mm
ARUCO_DICT  = cv2.aruco.DICT_4X4_50

# ─────────────────────────────────────────────
# BEV OUTPUT REGION (metres, camera-centred)
# Tune these to show the road area you care about
# ─────────────────────────────────────────────
GROUND_X_LEFT    = -2.0   # metres left
GROUND_X_RIGHT   =  2.0   # metres right
GROUND_Y_NEAR    =  0.3   # metres ahead (close edge)
GROUND_Y_FAR     =  5.0   # metres ahead (far edge)
PIXELS_PER_METER = 100    # output resolution


# ══════════════════════════════════════════════
#  BOARD FACTORY
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
#  CORNER DETECTION — works with old and new OpenCV
# ══════════════════════════════════════════════

def detect_charuco(gray, board, dictionary):
    """
    Detects ChArUco corners. Compatible with OpenCV 4.5+ and 4.7+.
    Returns (corners, ids, n) or (None, None, 0) on failure.
    """
    # Detect ArUco markers — use old API if ArucoDetector not available
    try:
        # New API (OpenCV >= 4.7)
        params   = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        # Old API (OpenCV < 4.7)
        params = cv2.aruco.DetectorParameters_create()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=params)

    if marker_ids is None or len(marker_ids) < 2:
        return None, None, 0

    # Interpolate ChArUco corners
    n, corners, ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )

    if n < 4 or corners is None:
        return None, None, 0

    return corners.reshape(-1, 1, 2).astype(np.float32), ids, n


# ══════════════════════════════════════════════
#  LOAD INTRINSICS
# ══════════════════════════════════════════════

def load_intrinsics(json_path):
    """
    Loads intrinsics from JSON saved by charuco_calibration.py.
    Returns K (3x3), dist (1x5).
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] Intrinsics file not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    K    = np.array(data["K"],    dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64)
    rms  = data.get("rms_px", "?")

    print(f"[LOADED] {json_path}")
    print(f"  Model: {data.get('model', 'pinhole')}")
    print(f"  RMS:   {rms}")
    print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
          f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")

    return K, dist


# ══════════════════════════════════════════════
#  COMPUTE HOMOGRAPHY FROM GROUND IMAGE
# ══════════════════════════════════════════════

def compute_H(img_path, K, dist, board, dictionary):
    """
    Detects ChArUco corners in a ground image and computes
    H: image pixels → ground metres using RANSAC.

    No undistortion applied — camera outputs clean images.

    Returns (H, img, stats_dict) or (None, None, None) on failure.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [ERROR] Cannot read: {img_path}")
        return None, None, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"\n[PROCESSING] {os.path.basename(img_path)}  ({w}x{h})")
    print("  NOTE: No undistortion — camera outputs clean images.")

    # Detect corners
    corners, ids, n = detect_charuco(gray, board, dictionary)

    if corners is None or n < 6:
        print(f"  [FAIL] Only {n} corners detected. Need at least 6.")
        print("  → Check boards are flat, fully visible, well-lit")
        print("  → Check SQUARES_X/Y match your board")
        return None, img, None

    print(f"  [OK] {n} ChArUco corners detected")

    # Get world coordinates from corner IDs
    world_pts_3d = board.getChessboardCorners()[ids.flatten()]
    world_pts_2d = world_pts_3d[:, :2].astype(np.float32)
    img_pts      = corners.reshape(-1, 2).astype(np.float32)

    # RANSAC homography
    H, mask = cv2.findHomography(
        img_pts, world_pts_2d,
        method=cv2.RANSAC,
        ransacReprojThreshold=0.01
    )

    if H is None:
        print("  [FAIL] RANSAC failed.")
        return None, img, None

    n_inliers = int(mask.sum())
    pct       = 100.0 * n_inliers / n

    # Reprojection error
    errs = []
    for px, gnd in zip(img_pts, world_pts_2d):
        pt = H @ np.array([px[0], px[1], 1.0])
        pt /= pt[2]
        errs.append(np.linalg.norm(pt[:2] - gnd))

    mean_err_cm = float(np.mean(errs)) * 100
    max_err_cm  = float(np.max(errs))  * 100

    stats = {
        "n_corners":    n,
        "n_inliers":    n_inliers,
        "inlier_pct":   pct,
        "mean_err_cm":  mean_err_cm,
        "max_err_cm":   max_err_cm,
    }

    # Quality assessment
    if pct >= 80 and mean_err_cm < 3.0:
        quality = "EXCELLENT ✓"
    elif pct >= 60 and mean_err_cm < 5.0:
        quality = "GOOD ✓"
    elif pct >= 40:
        quality = "ACCEPTABLE — consider retaking"
    else:
        quality = "POOR — retake photo"

    print(f"  Inliers:  {n_inliers}/{n}  ({pct:.0f}%)")
    print(f"  Error:    mean={mean_err_cm:.1f} cm   max={max_err_cm:.1f} cm")
    print(f"  Quality:  {quality}")

    # Draw detected corners on image
    vis = img.copy()
    try:
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
    except Exception:
        pass

    return H, vis, stats


# ══════════════════════════════════════════════
#  BEV WARP
# ══════════════════════════════════════════════

def warp_to_bev(img, H_img_to_ground, camera="front"):
    """
    Warps image to BEV using the computed homography.
    Adds distance grid lines and scale bar.
    """
    bev_w = int((GROUND_X_RIGHT - GROUND_X_LEFT) * PIXELS_PER_METER)
    bev_h = int((GROUND_Y_FAR   - GROUND_Y_NEAR) * PIXELS_PER_METER)

    # S maps BEV pixel (u,v) → ground metres (x,y)
    S = np.array([
        [1.0 / PIXELS_PER_METER,  0,                    GROUND_X_LEFT],
        [0,                      -1.0 / PIXELS_PER_METER, GROUND_Y_FAR],
        [0,                       0,                     1            ]
    ], dtype=np.float64)

    H_bev_to_img = np.linalg.inv(H_img_to_ground) @ S
    H_img_to_bev = np.linalg.inv(H_bev_to_img)

    bev = cv2.warpPerspective(
        img, H_img_to_bev, (bev_w, bev_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(30, 30, 30)
    )

    # Distance grid lines
    for d in range(1, int(GROUND_Y_FAR) + 1):
        if GROUND_Y_NEAR <= d <= GROUND_Y_FAR:
            row = int((GROUND_Y_FAR - d) * PIXELS_PER_METER)
            cv2.line(bev, (0, row), (bev_w, row), (60, 60, 60), 1)
            cv2.putText(bev, f"{d}m", (4, row - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (160, 160, 160), 1, cv2.LINE_AA)

    # Lateral centre line
    cx = bev_w // 2
    cv2.line(bev, (cx, 0), (cx, bev_h), (50, 50, 80), 1)

    # Scale bar
    bar_y = bev_h - 20
    cv2.line(bev, (20, bar_y),
             (20 + PIXELS_PER_METER, bar_y), (0, 255, 0), 2)
    cv2.putText(bev, "1 m", (20, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Camera label
    cv2.putText(bev, camera.upper(), (bev_w - 80, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1, cv2.LINE_AA)

    return bev


# ══════════════════════════════════════════════
#  COMPARISON DISPLAY
# ══════════════════════════════════════════════

def build_comparison(results):
    """
    Builds a side-by-side comparison image of all BEV results.
    results: list of (label, bev_img, stats)
    """
    if not results:
        return None

    # Resize all BEVs to same height for comparison
    target_h = 450
    panels   = []

    for label, bev, stats in results:
        h, w = bev.shape[:2]
        scale = target_h / h
        bev_r = cv2.resize(bev, (int(w * scale), target_h))

        # Add label + stats bar at top
        bar_h  = 80
        panel  = np.zeros((target_h + bar_h, bev_r.shape[1], 3),
                          dtype=np.uint8)
        panel[bar_h:, :] = bev_r

        # Label
        cv2.putText(panel, label, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

        if stats:
            # Inlier bar colour
            pct   = stats["inlier_pct"]
            color = (0, 220, 0) if pct >= 80 else \
                    (0, 220, 220) if pct >= 60 else \
                    (0, 80, 220)

            cv2.putText(panel,
                f"Inliers: {stats['n_inliers']}/{stats['n_corners']}"
                f"  ({pct:.0f}%)",
                (6, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, color, 1, cv2.LINE_AA)

            err_color = (0, 220, 0) if stats["mean_err_cm"] < 3 else \
                        (0, 220, 220) if stats["mean_err_cm"] < 5 else \
                        (0, 80, 220)

            cv2.putText(panel,
                f"Err: {stats['mean_err_cm']:.1f} cm mean  "
                f"{stats['max_err_cm']:.1f} cm max",
                (6, 64), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, err_color, 1, cv2.LINE_AA)

        # Separator line
        cv2.line(panel, (panel.shape[1]-1, 0),
                 (panel.shape[1]-1, panel.shape[0]),
                 (80, 80, 80), 2)

        panels.append(panel)

    return np.hstack(panels)


# ══════════════════════════════════════════════
#  SAVE BEST H
# ══════════════════════════════════════════════

def save_H(H, img_path, camera, stats, out_dir="."):
    """Saves the homography matrix and metadata to JSON."""
    name     = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"bev_H_{name}.json")

    data = {
        "camera":       camera,
        "source_image": os.path.basename(img_path),
        "H":            H.tolist(),
        "stats":        stats,
        "bev_region": {
            "x_left_m":    GROUND_X_LEFT,
            "x_right_m":   GROUND_X_RIGHT,
            "y_near_m":    GROUND_Y_NEAR,
            "y_far_m":     GROUND_Y_FAR,
            "px_per_meter": PIXELS_PER_METER,
        }
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  [SAVED] {out_path}")
    return out_path


# ══════════════════════════════════════════════
#  INTERACTIVE VIEWER
# ══════════════════════════════════════════════

def interactive_viewer(results, camera):
    """
    Shows each result one by one.
    Keys: N=next  P=prev  S=save this H  Q=quit
    """
    if not results:
        print("[WARN] No results to display.")
        return None

    idx      = 0
    saved    = []
    win_name = "BEV Result  [N=next | P=prev | S=save | Q=quit]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        label, bev, stats, H, img_path = results[idx]

        display = bev.copy()
        h, w    = display.shape[:2]

        # Top info bar
        cv2.rectangle(display, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.putText(display,
            f"[{idx+1}/{len(results)}]  {label}",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2, cv2.LINE_AA)

        if stats:
            pct = stats["inlier_pct"]
            c   = (0,220,0) if pct>=80 else (0,220,220) if pct>=60 else (0,80,220)
            cv2.putText(display,
                f"inliers {pct:.0f}%  err {stats['mean_err_cm']:.1f}cm mean",
                (8, 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, c, 1, cv2.LINE_AA)

        # Saved indicator
        if img_path in saved:
            cv2.putText(display, "SAVED ✓",
                (w - 100, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('n'), ord('N'), 83):    # N or right arrow
            idx = min(idx + 1, len(results) - 1)

        elif key in (ord('p'), ord('P'), 81):  # P or left arrow
            idx = max(idx - 1, 0)

        elif key in (ord('s'), ord('S')):      # S — save this H
            if H is not None:
                out = save_H(H, img_path, camera, stats)
                saved.append(img_path)
                print(f"  Saved H for: {label}")
            else:
                print("  [WARN] No valid H for this image.")

        elif key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break

    cv2.destroyAllWindows()
    return saved


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BEV extrinsic tester — ChArUco on ground, "
                    "hardware-undistorted camera"
    )
    p.add_argument("--images",      nargs="+", required=True,
                   help="Ground images to test "
                        "(e.g. front_h090_p20.jpg front_h090_p15.jpg)")
    p.add_argument("--intrinsics",  required=True,
                   help="Intrinsics JSON from charuco_calibration.py")
    p.add_argument("--camera",      default="front",
                   choices=["front", "left", "right"],
                   help="Which camera (affects BEV orientation label)")
    p.add_argument("--save_all",    action="store_true",
                   help="Save H JSON for all valid results automatically")
    p.add_argument("--out_dir",     default=".",
                   help="Directory to save H JSON files")
    p.add_argument("--compare",     action="store_true",
                   help="Show side-by-side comparison of all results")
    return p.parse_args()


def main():
    args   = parse_args()
    board, dictionary = make_board()

    print(f"\n{'='*60}")
    print("  BEV Extrinsic Tester")
    print(f"{'='*60}")
    print(f"  Board:   {SQUARES_X}×{SQUARES_Y} squares | "
          f"square={SQUARE_SIZE*1000:.0f}mm | "
          f"marker={MARKER_SIZE*1000:.0f}mm")
    print(f"  Camera:  {args.camera}")
    print(f"  Images:  {len(args.images)}")
    print(f"  BEV region: X[{GROUND_X_LEFT},{GROUND_X_RIGHT}]m  "
          f"Y[{GROUND_Y_NEAR},{GROUND_Y_FAR}]m")

    # Load intrinsics
    K, dist = load_intrinsics(args.intrinsics)

    # Process each image
    results = []      # for interactive viewer
    compare = []      # for comparison grid

    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"\n[SKIP] File not found: {img_path}")
            continue

        # Parse label from filename (e.g. front_h090_p20 → h=90cm p=20°)
        label = os.path.splitext(os.path.basename(img_path))[0]

        # Compute H
        H, vis_img, stats = compute_H(
            img_path, K, dist, board, dictionary)

        if H is None:
            # Still add to results so user can see the failed detection
            bev = np.zeros(
                (int((GROUND_Y_FAR-GROUND_Y_NEAR)*PIXELS_PER_METER),
                 int((GROUND_X_RIGHT-GROUND_X_LEFT)*PIXELS_PER_METER), 3),
                dtype=np.uint8)
            cv2.putText(bev, "DETECTION FAILED", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,200), 2)
            cv2.putText(bev, "Check board visibility", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,200), 1)
            results.append((label, bev, None, None, img_path))
            compare.append((label, bev, None))
            continue

        # Save corner detection image for inspection
        det_path = f"corners_{label}.png"
        if vis_img is not None:
            cv2.imwrite(det_path, vis_img)
            print(f"  [SAVED] {det_path} — inspect corner detection")

        # Warp to BEV
        bev = warp_to_bev(vis_img if vis_img is not None
                          else cv2.imread(img_path),
                          H, camera=args.camera)

        # Save BEV preview image
        bev_path = f"bev_{label}.png"
        cv2.imwrite(bev_path, bev)
        print(f"  [SAVED] {bev_path}")

        results.append((label, bev, stats, H, img_path))
        compare.append((label, bev, stats))

        # Auto-save if requested
        if args.save_all and H is not None:
            save_H(H, img_path, args.camera, stats, args.out_dir)

    # Print summary table
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Image':<30} {'Inliers':>10} {'Mean err':>10} {'Quality'}")
    print(f"  {'-'*60}")
    for label, bev, stats, H, img_path in results:
        if stats:
            q = "EXCELLENT" if stats["inlier_pct"]>=80 and \
                               stats["mean_err_cm"]<3 else \
                "GOOD"      if stats["inlier_pct"]>=60 and \
                               stats["mean_err_cm"]<5 else \
                "POOR"
            print(f"  {label:<30} "
                  f"{stats['n_inliers']:>3}/{stats['n_corners']:<3} "
                  f"({stats['inlier_pct']:>3.0f}%)  "
                  f"{stats['mean_err_cm']:>6.1f} cm   {q}")
        else:
            print(f"  {label:<30} {'FAILED':>10}")
    print(f"{'='*60}\n")

    # Show comparison grid if requested
    if args.compare and len(compare) > 1:
        comp_img = build_comparison(compare)
        if comp_img is not None:
            comp_path = f"comparison_{args.camera}.png"
            cv2.imwrite(comp_path, comp_img)
            print(f"[SAVED] {comp_path} — side-by-side comparison")
            cv2.namedWindow("Comparison — press any key to close",
                            cv2.WINDOW_NORMAL)
            cv2.imshow("Comparison — press any key to close", comp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Interactive viewer
    if results:
        print("Opening interactive viewer...")
        print("  N = next image")
        print("  P = previous image")
        print("  S = save H for this image")
        print("  Q = quit")
        saved = interactive_viewer(results, args.camera)
        if saved:
            print(f"\nSaved H matrices for {len(saved)} image(s).")
            print("Use the saved JSON files in your BEV node.")
    else:
        print("[ERROR] No images could be processed.")


if __name__ == "__main__":
    main()
