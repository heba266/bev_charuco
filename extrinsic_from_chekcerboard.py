"""
extrinsic_from_checkerboard.py
-------------------------------
Paper's method: lay checkerboard(s) flat on ground in front of camera,
take ONE image, detect corners automatically, get world coords,
compute H via RANSAC → BEV.

HOW TO USE:
  1. Print a checkerboard (or use existing one)
  2. Lay it FLAT on the ground in front of the camera
  3. Take a photo with the camera mounted on the rover
  4. Run: python3 extrinsic_from_checkerboard.py your_image.jpg
  5. If it works, saves H to bev_H.json for use in bev_maker.py

IMPORTANT settings to change:
  SQUARE_SIZE  = real size of one square in METRES (measure with ruler)
  BOARD_W, BOARD_H = number of INNER corners (not squares) on your board
                     e.g. a 6x9 square board has 5x8 inner corners
"""

import cv2
import numpy as np
import json

# ─────────────────────────────────────────────
# SETTINGS — change these to match your checkerboard
# ─────────────────────────────────────────────
SQUARE_SIZE = 0.10        # metres — measure one square on your printed board
BOARD_W     = 5           # inner corners horizontally (columns - 1)
BOARD_H     = 7           # inner corners vertically   (rows - 1)

# Output BEV region (same as ipm script)
GROUND_X_LEFT  = -2.0
GROUND_X_RIGHT =  2.0
GROUND_Y_NEAR  =  0.5
GROUND_Y_FAR   =  5.0
PIXELS_PER_METER = 100

# ─────────────────────────────────────────────
# INTRINSICS (from your calibration)
# ─────────────────────────────────────────────
K = np.array([
    [240.32319761,   0.,         310.18861609],
    [  0.,         241.48387481, 194.53783189],
    [  0.,           0.,           1.        ]
], dtype=np.float64)

DIST = np.array(
    [-0.03394111, 0.06001386, -0.00282427, 0.00279054, -0.0454231],
    dtype=np.float64
)


def detect_corners(img_gray):
    """Detect checkerboard corners in image."""
    pattern = (BOARD_W, BOARD_H)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(img_gray, pattern, flags)

    if not found:
        # Try the newer findChessboardCornersSB which is more robust
        found, corners = cv2.findChessboardCornersSB(img_gray, pattern)

    if found:
        # Refine to sub-pixel accuracy
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), term)

    return found, corners


def build_world_points():
    """
    Build the 3D world coordinates of all inner corners.
    The board lies FLAT on the ground (Z=0).
    Corner (0,0) is at origin, corners spread in X and Y.
    Units = metres (SQUARE_SIZE controls the scale).
    """
    pts = []
    for row in range(BOARD_H):
        for col in range(BOARD_W):
            pts.append([col * SQUARE_SIZE, row * SQUARE_SIZE, 0.0])
    return np.array(pts, dtype=np.float32)


def compute_H_from_checkerboard(img_path: str):
    img_raw  = cv2.imread(img_path)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    # Step 1 — undistort (paper does this first)
    img_undist = cv2.undistort(img_raw, K, DIST)
    img_gray   = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)

    print(f"[INFO] Image: {img_raw.shape[1]}×{img_raw.shape[0]}")
    print(f"[INFO] Looking for {BOARD_W}×{BOARD_H} inner corners "
          f"(square size = {SQUARE_SIZE*100:.0f} cm)")

    # Step 2 — detect corners → pixel coordinates
    found, img_corners = detect_corners(img_gray)
    if not found:
        print("[FAIL] Checkerboard not detected. Check:")
        print("  - BOARD_W and BOARD_H match your actual board inner corners")
        print("  - Board is fully visible and flat on ground")
        print("  - Good lighting, no glare")
        return None, None

    img_pts   = img_corners.reshape(-1, 2)           # shape (N, 2)
    world_pts = build_world_points()                  # shape (N, 3)
    world_pts_2d = world_pts[:, :2]                   # shape (N, 2) — Z=0 so drop it

    print(f"[OK] Detected {len(img_pts)} corners")

    # Step 3 — RANSAC homography (paper's approach: many points → robust H)
    # img_pts  = where corners appear in the undistorted image (pixels)
    # world_pts_2d = where those corners are on the ground (metres)
    H, mask = cv2.findHomography(
        img_pts, world_pts_2d,
        method=cv2.RANSAC,
        ransacReprojThreshold=0.01   # metres — tight threshold
    )

    n_inliers = int(mask.sum())
    print(f"[INFO] RANSAC inliers: {n_inliers}/{len(img_pts)}")

    if n_inliers < 8:
        print("[WARN] Too few inliers — result may be unreliable")

    # Verify: reproject corners and check error
    errors = []
    for px, world2 in zip(img_pts, world_pts_2d):
        pt = H @ np.array([px[0], px[1], 1.0])
        pt /= pt[2]
        errors.append(np.linalg.norm(pt[:2] - world2))
    print(f"[INFO] Reprojection error: mean={np.mean(errors)*100:.1f} cm  "
          f"max={np.max(errors)*100:.1f} cm")

    # Draw detection for visual check
    vis = cv2.drawChessboardCorners(img_undist.copy(),
                                     (BOARD_W, BOARD_H), img_corners, found)
    cv2.imwrite("corners_detected.png", vis)
    print("[SAVED] corners_detected.png — check this looks correct!")

    return H, img_undist


def warp_to_bev(img_undist, H_img_to_ground):
    """
    H_img_to_ground maps:  image pixel → ground position (metres)

    For warpPerspective we need the INVERSE:  ground position → image pixel
    Then we also need a pixel↔metre scaling for the output canvas.
    """
    bev_w = int((GROUND_X_RIGHT - GROUND_X_LEFT) * PIXELS_PER_METER)
    bev_h = int((GROUND_Y_FAR   - GROUND_Y_NEAR) * PIXELS_PER_METER)

    # Matrix that maps BEV pixel → ground metres
    # BEV pixel (u, v):  u=0 is left edge, v=0 is far edge (top of BEV)
    #   ground_x = GROUND_X_LEFT + u / PIXELS_PER_METER
    #   ground_y = GROUND_Y_FAR  - v / PIXELS_PER_METER
    S = np.array([
        [1.0/PIXELS_PER_METER,  0,                    GROUND_X_LEFT],
        [0,                    -1.0/PIXELS_PER_METER,  GROUND_Y_FAR ],
        [0,                     0,                     1            ]
    ], dtype=np.float64)

    # Combined: BEV pixel → ground metres → image pixel
    H_bev_to_img = np.linalg.inv(H_img_to_ground) @ S
    H_img_to_bev = np.linalg.inv(H_bev_to_img)

    bev = cv2.warpPerspective(
        img_undist, H_img_to_bev, (bev_w, bev_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(30, 30, 30)
    )

    # Draw distance lines
    for d in [1, 2, 3, 4]:
        if GROUND_Y_NEAR <= d <= GROUND_Y_FAR:
            row = int((GROUND_Y_FAR - d) / (GROUND_Y_FAR - GROUND_Y_NEAR) * bev_h)
            cv2.line(bev, (0, row), (bev_w, row), (60, 60, 60), 1)
            cv2.putText(bev, f"{d}m", (4, row - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    # Scale bar
    cv2.line(bev, (20, bev_h-20), (20+PIXELS_PER_METER, bev_h-20), (0,255,0), 2)
    cv2.putText(bev, "1 m", (20, bev_h-28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return bev


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "checkerboard_ground.jpg"

    H, img_undist = compute_H_from_checkerboard(img_path)

    if H is not None:
        # Save H for use in bev_maker.py / bev_node.py
        with open("bev_H_from_checkerboard.json", "w") as f:
            json.dump({"front": H.tolist()}, f, indent=2)
        print("\n[SAVED] bev_H_from_checkerboard.json")

        bev = warp_to_bev(img_undist, H)
        cv2.imwrite("bev_checkerboard_result.png", bev)
        print("[SAVED] bev_checkerboard_result.png")

        cv2.namedWindow("BEV result", cv2.WINDOW_NORMAL)
        cv2.imshow("BEV result", bev)
        cv2.waitKey(0)
        cv2.destroyAllWindows()