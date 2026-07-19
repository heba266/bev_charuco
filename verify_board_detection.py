"""
verify_board_detection.py
===========================
Two-stage verification: (1) raw ArUco marker detection, (2) ChArUco
corner interpolation -- the exact two steps bev_extrinsic_charuco.py
needs, run in isolation with clear diagnosis at EACH stage, so you know
immediately which one is failing and why, instead of just seeing one
final pass/fail.

WHY TWO STAGES, NOT ONE:
  Stage A (detectMarkers) only checks: can each individual marker's
  black/white bit pattern be decoded? This does NOT care about board
  flatness, checker/marker size, or relative marker positions.

  Stage B (interpolateCornersCharuco) requires ALL detected markers to
  fit together as one consistent RIGID PLANE, with the correct
  checker/marker size. This is what catches non-flat boards (creases,
  curling, sagging) -- a board can pass Stage A perfectly while still
  failing Stage B if it isn't lying flat.

  If Stage A fails: problem is detection itself (blur, resolution,
  exposure, wrong dictionary).
  If Stage A passes but Stage B fails: problem is almost always
  PHYSICAL FLATNESS, or a checker/marker size mismatch. This script
  prints a planarity/reprojection-error check to tell you which.
67
HOW TO USE:
    python3 verify_board_detection.py --image board_test.png

    # or live from camera, no photo needed:
    python3 verify_board_detection.py --device /dev/video2 --width 960 --height 720
"""

import cv2
import numpy as np
import argparse
import sys

# ── Board geometry -- adjust to your REPRINTED board's measured values ──
SQUARES_X = 11
SQUARES_Y = 8
SQUARE_SIZE = 0.020   # metres -- update after you measure the reprint
MARKER_SIZE = 0.015   # metres -- update after you measure the reprint
ARUCO_DICT = cv2.aruco.DICT_4X4_50

PLANARITY_WARN_M = 0.01    # mean reprojection error above this (metres)
                            # strongly suggests the board isn't flat


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, dictionary
    )
    return board, dictionary


def check_planarity(marker_corners, marker_ids, board):
    """
    Fits a homography from detected marker corners -> board-local
    coordinates. A clean, flat, correctly-sized board gives near-zero
    reprojection error. A non-flat board (creased/curled/sagging) or a
    wrong checker/marker size gives large, inconsistent error even
    though every individual marker decoded successfully.
    """
    obj_points_board = board.getObjPoints()
    board_ids_list = board.getIds().reshape(-1).tolist()

    img_pts, board_pts = [], []
    for mc, mid in zip(marker_corners, marker_ids.reshape(-1)):
        mid = int(mid)
        if mid in board_ids_list:
            idx = board_ids_list.index(mid)
            obj_pts = obj_points_board[idx]
            img_pts.extend(mc.reshape(4, 2))
            board_pts.extend(obj_pts[:, :2])

    if len(img_pts) < 4:
        return None, None, None

    img_pts = np.array(img_pts, dtype=np.float32)
    board_pts = np.array(board_pts, dtype=np.float32)

    H, mask = cv2.findHomography(img_pts, board_pts, method=cv2.RANSAC,
                                   ransacReprojThreshold=0.01)
    if H is None:
        return None, None, None

    errs = []
    for px, gnd in zip(img_pts, board_pts):
        pt = H @ np.array([px[0], px[1], 1.0])
        pt /= pt[2]
        errs.append(np.linalg.norm(pt[:2] - gnd))

    mean_err = float(np.mean(errs))
    max_err = float(np.max(errs))
    n_inliers = int(mask.sum())
    return mean_err, max_err, n_inliers


# def run_check(gray, board, dictionary, verbose=True):
#     """
#     Runs both stages and returns a dict summarizing the result.
#     """
#     params = cv2.aruco.DetectorParameters()
#     detector = cv2.aruco.ArucoDetector(dictionary, params)

#     # ── Stage A: raw marker detection ───────────────────────────────
#     marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
#     n_markers = 0 if marker_ids is None else len(marker_ids)

#     result = {
#         "stage_a_markers": n_markers,
#         "stage_a_rejected": len(rejected) if rejected is not None else 0,
#         "stage_b_corners": 0,
#         "planarity_mean_err_cm": None,
#         "planarity_max_err_cm": None,
#         "verdict": "FAIL",
#         "diagnosis": "",
#     }

#     if n_markers == 0:
#         result["diagnosis"] = (
#             "Stage A FAILED: no markers decoded at all. This is a "
#             "detection-level problem -- check resolution, blur, exposure, "
#             "lighting, or distance (use marker_size_sanity_check.py / "
#             "live_detection_check.py to dig into THIS specifically)."
#         )
#         if verbose:
#             _print_result(result)
#         return result

#     # ── Stage B: ChArUco corner interpolation ───────────────────────
#     n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
#         marker_corners, marker_ids, gray, board
#     )
#     result["stage_b_corners"] = n if n else 0

#     # ── Planarity / consistency diagnostic (runs regardless of B's
#     #    pass/fail, since it explains WHY B failed when A succeeded) ──
#     mean_err, max_err, n_inliers = check_planarity(marker_corners, marker_ids, board)
#     if mean_err is not None:
#         result["planarity_mean_err_cm"] = mean_err * 100
#         result["planarity_max_err_cm"] = max_err * 100
#         result["planarity_inliers"] = n_inliers

#     if n and n >= 4 and ch_corners is not None:
#         result["verdict"] = "PASS"
#         result["diagnosis"] = (
#             f"Both stages passed. {n} ChArUco corners interpolated "
#             f"successfully. Safe to proceed to bev_extrinsic_charuco.py."
#         )
#     else:
#         result["verdict"] = "FAIL"
#         if mean_err is not None and mean_err > PLANARITY_WARN_M:
#             result["diagnosis"] = (
#                 f"Stage A passed ({n_markers} markers decoded) but Stage B "
#                 f"FAILED. Planarity check shows mean error "
#                 f"{mean_err*100:.1f}cm (max {max_err*100:.1f}cm) -- this is "
#                 f"HIGH for a board whose squares are only "
#                 f"{SQUARE_SIZE*100:.1f}cm. This almost always means the "
#                 f"board is NOT physically flat (crease, curl, sag, or "
#                 f"corners lifting off the surface). Check for visible "
#                 f"folds/buckling and re-tape firmly, pressing out air "
#                 f"pockets, on a hard flat surface."
#             )
#         else:
#             result["diagnosis"] = (
#                 f"Stage A passed ({n_markers} markers decoded) but Stage B "
#                 f"FAILED, and the planarity check did NOT show high error "
#                 f"({mean_err*100:.1f}cm if available). This suggests a "
#                 f"SQUARE_SIZE / MARKER_SIZE / SQUARES_X / SQUARES_Y "
#                 f"mismatch between this script's settings and your actual "
#                 f"physical board. Double check the values at the top of "
#                 f"this script against a fresh manual measurement of the "
#                 f"REPRINTED board."
#             )

#     if verbose:
#         _print_result(result)
#     return result


def _print_result(r):
    print(f"\n{'='*60}")
    print(f"  Stage A (marker detection):     {r['stage_a_markers']} markers "
          f"decoded  ({r['stage_a_rejected']} candidates rejected)")
    print(f"  Stage B (ChArUco interpolation): {r['stage_b_corners']} corners "
          f"interpolated")
    if r["planarity_mean_err_cm"] is not None:
        print(f"  Planarity check:  mean={r['planarity_mean_err_cm']:.2f}cm  "
              f"max={r['planarity_max_err_cm']:.2f}cm  "
              f"(>{PLANARITY_WARN_M*100:.0f}cm mean = likely not flat)")
    print(f"\n  VERDICT: {r['verdict']}")
    print(f"  {r['diagnosis']}")
    print(f"{'='*60}\n")



def run_check(gray, board, dictionary, verbose=True):
    """
    Runs both stages and returns a dict summarizing the result.
    """
    # ── OpenCV 4.7+ Detector Setup ──────────────────────────────────
    params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board)
    # charuco_detector.detectorParameters = params
    # charuco_detector.dictionary = dictionary
    charuco_detector.setDetectorParameters(params)

    # ── Let the detector handle both markers and board at once ──────
    ch_corners, ch_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
    
    n_markers = 0 if marker_ids is None else len(marker_ids)
    # The modern detector doesn't easily expose rejected markers from detectBoard, 
    # so we default it to 0 for the dictionary structure.
    rejected_count = 0 

    result = {
        "stage_a_markers": n_markers,
        "stage_a_rejected": rejected_count,
        "stage_b_corners": 0,
        "planarity_mean_err_cm": None,
        "planarity_max_err_cm": None,
        "verdict": "FAIL",
        "diagnosis": "",
    }

    if n_markers == 0:
        result["diagnosis"] = (
            "Stage A FAILED: no markers decoded at all. This is a "
            "detection-level problem -- check resolution, blur, exposure, "
            "lighting, or distance (use marker_size_sanity_check.py / "
            "live_detection_check.py to dig into THIS specifically)."
        )
        if verbose:
            _print_result(result)
        return result

    # ── Stage B: Extract corner count ───────────────────────────────
    n = len(ch_corners) if ch_corners is not None else 0
    result["stage_b_corners"] = n

    # ── Planarity / consistency diagnostic ──────────────────────────
    mean_err, max_err, n_inliers = check_planarity(marker_corners, marker_ids, board)
    if mean_err is not None:
        result["planarity_mean_err_cm"] = mean_err * 100
        result["planarity_max_err_cm"] = max_err * 100
        result["planarity_inliers"] = n_inliers

    if n and n >= 4 and ch_corners is not None:
        result["verdict"] = "PASS"
        result["diagnosis"] = (
            f"Both stages passed. {n} ChArUco corners interpolated "
            f"successfully. Safe to proceed to bev_extrinsic_charuco.py."
        )
    else:
        result["verdict"] = "FAIL"
        if mean_err is not None and mean_err > PLANARITY_WARN_M:
            result["diagnosis"] = (
                f"Stage A passed ({n_markers} markers decoded) but Stage B "
                f"FAILED. Planarity check shows mean error "
                f"{mean_err*100:.1f}cm (max {max_err*100:.1f}cm) -- this is "
                f"HIGH for a board whose squares are only "
                f"{SQUARE_SIZE*100:.1f}cm. This almost always means the "
                f"board is NOT physically flat (crease, curl, sag, or "
                f"corners lifting off the surface). Check for visible "
                f"folds/buckling and re-tape firmly, pressing out air "
                f"pockets, on a hard flat surface."
            )
        else:
            result["diagnosis"] = (
                f"Stage A passed ({n_markers} markers decoded) but Stage B "
                f"FAILED, and the planarity check did NOT show high error "
                f"({mean_err*100:.1f}cm if available). This suggests a "
                f"SQUARE_SIZE / MARKER_SIZE / SQUARES_X / SQUARES_Y "
                f"mismatch between this script's settings and your actual "
                f"physical board. Double check the values at the top of "
                f"this script against a fresh manual measurement of the "
                f"REPRINTED board."
            )

    if verbose:
        _print_result(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None,
                    help="Path to a single image to check")
    p.add_argument("--device", default=None,
                    help="Camera device for LIVE checking instead of a "
                         "saved image (e.g. /dev/video2)")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    board, dictionary = make_board()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"[ERROR] Could not read image: {args.image}")
            sys.exit(1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        run_check(gray, board, dictionary)

    elif args.device:
        device = args.device
        try:
            device = int(device)
        except ValueError:
            pass
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera: {args.device}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        print("Press SPACE to check the current frame, Q to quit.")
        win_name = "verify_board_detection  [SPACE=check | Q=quit]"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera read failed.")
                break
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                run_check(gray, board, dictionary)
        cap.release()
        cv2.destroyAllWindows()

    else:
        print("[ERROR] Provide either --image or --device.")
        sys.exit(1)


if __name__ == "__main__":
    main()
