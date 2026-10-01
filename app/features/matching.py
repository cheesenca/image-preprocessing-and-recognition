import cv2
import numpy as np

# Feature Detection
def detect_and_compute(image: np.ndarray, method='ORB'):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if method == 'ORB':
        detector = cv2.ORB_create()
    else:
        try:
            detector = cv2.SIFT_create()
        except Exception:
            detector = cv2.ORB_create()
    kp, des = detector.detectAndCompute(gray, None)
    return kp, des


# Matching Feature
def match_features(img1: np.ndarray, img2: np.ndarray, method='ORB', match_method='BF'):
    kp1, des1 = detect_and_compute(img1, method)
    kp2, des2 = detect_and_compute(img2, method)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return img1
    if match_method == 'BF':
        norm_type = cv2.NORM_HAMMING if method == 'ORB' else cv2.NORM_L2
        bf = cv2.BFMatcher(norm_type, crossCheck=True)
        matches = bf.match(des1, des2)
    else:
        if method == 'ORB':
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.match(des1, des2)
        else:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_img


# Template Matching
def template_match(main_img: np.ndarray, templ: np.ndarray, threshold=0.8):
    if len(main_img.shape) == 3:
        main_gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
    else:
        main_gray = main_img
    if len(templ.shape) == 3:
        templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
    else:
        templ_gray = templ
    result = cv2.matchTemplate(main_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)
    display = main_img.copy()
    h, w = templ_gray.shape
    for pt in zip(*loc[::-1]):
        cv2.rectangle(display, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    return display