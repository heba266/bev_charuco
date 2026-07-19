import cv2
import numpy as np

img  = cv2.imread("right_h70_p60_high_res.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Enhance contrast to compensate for overexposure
clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params     = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(dictionary, params)

# Try on both original and enhanced
for name, img_to_check in [("original", gray), ("enhanced", enhanced)]:
    corners, ids, rejected = detector.detectMarkers(img_to_check)
    n = 0 if ids is None else len(ids)
    print(f"{name}: {n} markers detected, {len(rejected)} rejected")
    if n > 0:
        vis = cv2.cvtColor(img_to_check, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.imwrite(f"detected_{name}.png", vis)
        print(f"  Saved detected_{name}.png")