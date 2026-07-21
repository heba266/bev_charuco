# BEV Calibration via ChArUco Board Detection

> **Fisheye Camera Extrinsic Calibration &amp; Bird's-Eye-View Generation Using ChArUco and Checkerboard Pattern Detection**

---

## 📋 Project Overview

This repository implements a **calibration-based approach** to Bird's-Eye-View (BEV) generation from fisheye cameras. It estimates camera **intrinsic** and **extrinsic** parameters using calibration patterns (checkerboards and ChArUco boards), computes the **homography matrix** mapping the perspective view to a top-down ground plane, and produces a BEV image through inverse perspective mapping (IPM).

### Theoretical Background

In autonomous driving, obtaining a top-down Bird's-Eye-View of the road scene is critical for lane detection, path planning, and spatial reasoning. The BEV transformation requires solving for a **homography matrix** *H* that maps points from the camera's perspective image plane onto the ground plane.

This project follows the methodology described in:

> **Zhou Su et al., "Calibration Method for Fisheye Camera Based on Multi-checkerboard Detection"**, *Automation and Machine Learning*, 2023.

The pipeline consists of:

1. **Fisheye Intrinsic Calibration** — Using the **Kannala-Brandt** camera model to undistort fisheye images.
2. **Multi-Pattern Corner Detection** — Detecting calibration pattern corners using **libcbdetect** (checkerboards) and **OpenCV ArUco/ChArUco** (ChArUco boards).
3. **World Coordinate Mapping** — Using a **depth-first search (DFS)** algorithm to assign world coordinates to detected corners.
4. **Homography Estimation** — Computing the perspective-to-BEV homography matrix via **RANSAC** from world-pixel coordinate pairs.
5. **BEV Generation** — Applying the homography to the undistorted image to produce the top-down view.

### Why ChArUco?

ChArUco boards combine the sub-pixel precision of chessboard corners with the unique identifier robustness of ArUco markers. This makes them more resilient to:
- **Partial occlusions** — Individual markers can be detected even when parts of the board are hidden.
- **Varying lighting conditions** — ArUco markers provide reliable detection under challenging illumination.
- **Automatic corner identification** — No need to manually specify grid dimensions; the board self-identifies.

---

## 🗂 Code Structure

```
bev_charuco/
├── images/                          # Sample calibration images
├── charuco_calibration.py           # Intrinsic calibration using ChArUco board
├── bev_extrinsic_charuco.py         # Extrinsic calibration & BEV from ChArUco detection
├── bev_extrinsic_tester_1.py        # Testing script for BEV extrinsic pipeline
├── extrinsic_from_chekcerboard.py   # Extrinsic calibration using checkerboard (libcbdetect)
├── aruco_detection.py               # ArUco marker detection utilities
├── dict_detection.py                # ArUco dictionary detection & validation
├── live_detection_check.py          # Real-time camera detection & calibration preview
├── intrinsics_front.json            # Calibrated intrinsics — front camera
├── intrinsics_left.json             # Calibrated intrinsics — left camera
├── intrinsics_right.json            # Calibrated intrinsics — right camera
├── intrinsics_pinhole.json          # Pinhole model intrinsics
├── homographies_bev2.json           # Computed BEV homography matrices
└── bev.pdf                          # Reference paper (Su et al., 2023)
```

### Pipeline Execution Order

| Step | Script | Description |
|------|--------|-------------|
| 1 | `charuco_calibration.py` | Run intrinsic calibration using the Kannala-Brandt fisheye model with ChArUco board images. Outputs camera matrix and distortion coefficients to JSON. |
| 2 | `extrinsic_from_chekcerboard.py` | Detect multi-checkerboard corners (libcbdetect), run DFS for world coordinates, estimate homography via RANSAC. |
| 3 | `bev_extrinsic_charuco.py` | Detect ChArUco corners, interpolate sub-pixel positions, compute extrinsic parameters and homography matrix. |
| 4 | `bev_extrinsic_tester_1.py` | Validate the BEV transformation and visualize the result. |
| — | `live_detection_check.py` | *(Optional)* Real-time detection preview from a connected camera feed. |

---

## ⚠️ Challenges &amp; Limitations — Why We Pivoted

While this calibration-based approach produced accurate results in controlled settings, it revealed **significant practical limitations** for real-world autonomous driving deployment:

### Proximity Constraint

- The method required cameras to be placed in **extremely close proximity** to the ChArUco/checkerboard calibration patterns to reliably detect markers and accurately interpolate sub-pixel grid points.
- At close range, the fisheye lens captures sufficient corner detail for robust detection. However, as the distance increases, marker resolution degrades rapidly, leading to detection failures and inaccurate homographies.

### Physical Mounting Incompatibility

- Our autonomous driving platform mounts cameras **high off the ground** to maximize the field of view and coverage area.
- At these elevated mounting positions, the calibration patterns are too far from the camera for reliable ChArUco corner interpolation and checkerboard grid detection.
- **Due to these physical mounting constraints, high-quality BEV output images could not be generated at production camera heights.**

### Scalability

- The method requires a **controlled calibration environment** with a physical board present — making it impractical for in-field recalibration or deployment across multiple vehicles.

### Conclusion

These limitations motivated the transition to a **CNN-based deep learning approach** that can estimate BEV homographies from a single image without requiring physical calibration targets at deployment time.

---

## 🔗 Transition to CNN-Based Approach

For the **production-grade BEV pipeline** that addresses the limitations above, see:

👉 **[bev_cnn](https://github.com/heba266/bev_cnn)** — CNN-based BEV homography estimation with multi-camera fusion and segmentation mapping.

The CNN approach leverages a trained Inception-V4 model to predict vanishing points and estimate the front-camera homography directly from a single image, then extends to multi-camera surround-view BEV through feature matching — **no physical calibration board required**.

---

## 🛠 Dependencies

```
Python 3.6+
OpenCV (with ArUco/ChArUco contrib modules)
NumPy
libcbdetect (for checkerboard approach)
```

---

## 📚 References

1. Su, Z., Zhu, X., &amp; Lu, Y. (2023). *Calibration Method for Fisheye Camera Based on Multi-checkerboard Detection*. Automation and Machine Learning, 4(1), 24-31.
2. Kannala, J. &amp; Brandt, S.S. (2006). *A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses*. IEEE TPAMI, 28(8), 1335-1340.
3. Geiger, A. et al. (2012). *Automatic Camera and Range Sensor Calibration Using a Single Shot*. IEEE ICRA.
4. Abbas, S.A. &amp; Zisserman, A. (2019). *A Geometric Approach to Obtain a Bird's Eye View From an Image*. IEEE ICCV Workshop.

---

## 👤 Author

**Heba El-Afifi** — Computer &amp; Communication Engineering, Alexandria University  
📧 iheba3930@gmail.com | 🐙 [github.com/heba266](https://github.com/heba266)
