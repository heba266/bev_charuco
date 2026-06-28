"""
marker_size_sanity_check.py
============================
Batch sanity-checker for ChArUco calibration photos. Run this BEFORE
bev_extrinsic_charuco.py on a whole folder of shots, to immediately see
which images are usable and which have markers too small to ever be
detected -- without wasting time running full calibration on each one.

WHAT IT CHECKS, PER IMAGE:
  1. Sharpness (Laplacian variance) -- rules out blur as the cause
  2. ACTUAL ArUco marker detection count (ground truth, not estimated)
  3. Estimated pixels-per-marker-cell, based on board geometry + how
     large the markers that WERE detected actually are in the image
     (this gives you a real number, not a guess, even for images where
     SOME markers were found but not all)

WHY THIS MATTERS:
  An ArUco marker (DICT_4X4_50) is a 6x6 grid of cells (4x4 data bits +
  1-cell border on each side). Below ~3px per cell, the black/white
  pattern is physically unresolvable by the camera, regardless of how
  in-focus the lens is -- this is a sampling-resolution limit, not a
  blur/focus problem, and cannot be fixed by software after the fact.

HOW TO USE:
    python3 marker_size_sanity_check.py --folder /path/to/your/27_shots

    python3 marker_size_sanity_check.py --folder . --pattern "*.png"

Outputs a sorted table (best to worst) and a CSV you can open in a
spreadsheet, so you know exactly which shots are worth running full
calibration on.
"""

import cv2
import numpy as np
import argparse
import glob
import os
import csv

# ── Board geometry -- MUST match your actual (corrected) board ─────
SQUARES_X = 11
SQUARES_Y = 8
CHECKER_MM = 20.0   # your measured value, not the misprinted 15mm
MARKER_MM = 15.0    # your measured value, not the misprinted 11mm
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# ── Thresholds ───────────────────────────────────────────────────────
BLUR_OK_THRESHOLD = 80.0          # Laplacian variance; below = blurry
MIN_PX_PER_CELL_USABLE = 4.0      # below this, detection is unreliable
MIN_PX_PER_CELL_MARGINAL = 3.0    # below this, essentially undetectable


def estimate_px_per_cell_from_detected(corners):
    """
    Given detected ArUco marker corners (Nx4x2), estimate the average
    marker side length in pixels, then convert to px-per-cell.
    A DICT_4X4_50 marker is a 6x6 cell grid (4x4 data + 1-cell border
    on each side).
    """
    if corners is None or len(corners) == 0:
        return None

    side_lengths = []
    for c in corners:
        pts = c.reshape(4, 2)
        # average of all 4 side lengths of this marker quad
        sides = [
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[3]),
            np.linalg.norm(pts[3] - pts[0]),
        ]
        side_lengths.append(np.mean(sides))

    avg_marker_px = float(np.mean(side_lengths))
    px_per_cell = avg_marker_px / 6.0
    return avg_marker_px, px_per_cell


def estimate_px_per_cell_from_geometry(img_shape, n_boards_visible=1):
    """
    Fallback estimate when ZERO markers were detected: uses the image
    size and an assumed board footprint to guess marker pixel size.
    Much rougher than the detected-corner method, but better than
    nothing when detection totally fails.

    This assumes the board roughly fills a fraction of frame width
    typical of a single ground-placed board in a wide-FOV ground shot.
    It's a sanity estimate, not a precise measurement -- if you have
    ANY image from the same combo where markers WERE detected, prefer
    that number instead.
    """
    h, w = img_shape[:2]
    # Heuristic: assume the board occupies ~1/5 of image width per board
    # visible in frame (rough, just for a ballpark figure)
    assumed_board_px_width = w / (5 * max(n_boards_visible, 1))
    board_phys_width_mm = SQUARES_X * CHECKER_MM
    px_per_mm = assumed_board_px_width / board_phys_width_mm
    marker_px = MARKER_MM * px_per_mm
    px_per_cell = marker_px / 6.0
    return marker_px, px_per_cell


def check_image(path, detector, dictionary):
    img = cv2.imread(path)
    if img is None:
        return {"file": os.path.basename(path), "error": "could not read file"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()

    corners, ids, rejected = detector.detectMarkers(gray)
    n_detected = 0 if ids is None else len(ids)
    n_rejected = len(rejected) if rejected is not None else 0

    if n_detected > 0:
        avg_marker_px, px_per_cell = estimate_px_per_cell_from_detected(corners)
        size_source = "measured"
    else:
        avg_marker_px, px_per_cell = estimate_px_per_cell_from_geometry(img.shape)
        size_source = "estimated (0 detected — rough guess)"

    if px_per_cell >= MIN_PX_PER_CELL_USABLE:
        verdict = "USABLE"
    elif px_per_cell >= MIN_PX_PER_CELL_MARGINAL:
        verdict = "MARGINAL"
    else:
        verdict = "TOO SMALL"

    if blur_val < BLUR_OK_THRESHOLD:
        verdict += " (+BLURRY)"

    return {
        "file": os.path.basename(path),
        "resolution": f"{img.shape[1]}x{img.shape[0]}",
        "blur_var": round(blur_val, 1),
        "markers_detected": n_detected,
        "candidates_rejected": n_rejected,
        "avg_marker_px": round(avg_marker_px, 1),
        "px_per_cell": round(px_per_cell, 2),
        "size_source": size_source,
        "verdict": verdict,
    }


def required_distance_for_target(current_distance_m, current_px_per_cell,
                                   target_px_per_cell=MIN_PX_PER_CELL_USABLE):
    """
    Back-solves: given a board was shot at current_distance_m and measured
    current_px_per_cell, how close would it need to be (at the SAME
    resolution) to reach target_px_per_cell? Apparent size scales ~1/distance
    for a planar target viewed by a pinhole camera, so:
        required_distance = current_distance * (current_px_per_cell / target)
    Returns None if current_px_per_cell is 0 or current_distance is unknown.
    """
    if not current_distance_m or current_px_per_cell <= 0:
        return None
    return current_distance_m * (current_px_per_cell / target_px_per_cell)


def required_resolution_scale(current_px_per_cell,
                                target_px_per_cell=MIN_PX_PER_CELL_USABLE):
    """
    Back-solves: given current_px_per_cell at the CURRENT resolution, what
    linear resolution multiplier (e.g. 2x => double width AND height) would
    be needed to reach target_px_per_cell, if distance stays fixed?
    """
    if current_px_per_cell <= 0:
        return None
    return target_px_per_cell / current_px_per_cell



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True, help="Folder containing your shots")
    p.add_argument("--pattern", default="*.png",
                    help="Glob pattern, e.g. '*.png' or '*.jpg' (default: *.png)")
    p.add_argument("--csv_out", default="marker_size_check_results.csv")
    p.add_argument("--distance_m", type=float, default=None,
                    help="Distance (m) from camera to the board in these "
                         "shots, if known and consistent across the folder. "
                         "Enables back-solved 'move to Xm' / 'need NxN res' "
                         "recommendations in the output.")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if not paths:
        print(f"[ERROR] No files matched {args.pattern} in {args.folder}")
        return

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    results = []
    print(f"Checking {len(paths)} image(s)...\n")
    for path in paths:
        r = check_image(path, detector, dictionary)
        results.append(r)

    # Sort worst-to-best so problems are easy to spot, or best-to-worst —
    # here: usable first, then marginal, then too small
    order = {"USABLE": 0, "MARGINAL": 1, "TOO SMALL": 2}
    def sort_key(r):
        base = next((v for k, v in order.items() if k in r.get("verdict", "")), 3)
        return (base, -r.get("px_per_cell", 0))
    results_sorted = sorted(results, key=sort_key)

    # ── Print table ──────────────────────────────────────────────────
    header = f"{'file':<28} {'res':<10} {'blur':>8} {'markers':>8} {'px/cell':>8}  verdict"
    print(header)
    print("-" * len(header))
    for r in results_sorted:
        if "error" in r:
            print(f"{r['file']:<28} ERROR: {r['error']}")
            continue
        print(f"{r['file']:<28} {r['resolution']:<10} {r['blur_var']:>8.1f} "
              f"{r['markers_detected']:>8} {r['px_per_cell']:>8.2f}  {r['verdict']}")

        if "USABLE" not in r["verdict"] and args.distance_m is not None:
            req_dist = required_distance_for_target(args.distance_m, r["px_per_cell"])
            res_scale = required_resolution_scale(r["px_per_cell"])
            if req_dist is not None:
                w, h = r["resolution"].split("x")
                new_res = f"{int(int(w)*res_scale)}x{int(int(h)*res_scale)}"
                print(f"{'':<28}   -> to reach {MIN_PX_PER_CELL_USABLE}px/cell: "
                      f"move to ~{req_dist:.2f}m (currently ~{args.distance_m:.2f}m), "
                      f"OR increase resolution ~{res_scale:.1f}x (~{new_res})")

    # ── Summary counts ───────────────────────────────────────────────
    n_usable = sum(1 for r in results if "USABLE" in r.get("verdict", ""))
    n_marginal = sum(1 for r in results if "MARGINAL" in r.get("verdict", ""))
    n_too_small = sum(1 for r in results if "TOO SMALL" in r.get("verdict", ""))
    print(f"\n{'='*50}")
    print(f"  USABLE:    {n_usable}")
    print(f"  MARGINAL:  {n_marginal}  (try calibration, may still fail)")
    print(f"  TOO SMALL: {n_too_small}  (will not detect — board too few px)")
    print(f"{'='*50}")
    print(f"\nMinimum px/cell for reliable detection: {MIN_PX_PER_CELL_USABLE}")
    print(f"Below {MIN_PX_PER_CELL_MARGINAL} px/cell: physically unresolvable, "
          f"not a blur/focus issue, cannot be fixed by software.")

    # ── CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(args.folder, args.csv_out)
    fieldnames = ["file", "resolution", "blur_var", "markers_detected",
                  "candidates_rejected", "avg_marker_px", "px_per_cell",
                  "size_source", "verdict"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_sorted:
            if "error" not in r:
                writer.writerow(r)
    print(f"\n[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
