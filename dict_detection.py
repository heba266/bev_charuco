import cv2
import numpy as np

img  = cv2.imread("live_check_960x720_01.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# Try ALL common 4x4 dictionaries to find which one works
dicts_to_try = {
    "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":  cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50":   cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50":   cv2.aruco.DICT_6X6_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

print("Testing all dictionaries on your board image:\n")
for name, dict_id in dicts_to_try.items():
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params     = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(dictionary, params)

    corners, ids, rejected = detector.detectMarkers(enhanced)
    n = 0 if ids is None else len(ids)

    if n > 0:
        print(f"  ✅ {name}: {n} markers DETECTED ← USE THIS ONE")
        vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.imwrite(f"detected_{name}.png", vis)
    else:
        r = len(rejected) if rejected else 0
        print(f"  ❌ {name}: 0 detected ({r} rejected)")