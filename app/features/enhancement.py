import cv2
import numpy as np

# Histogram Equalization
def histogram_equalization(image: np.ndarray):
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    else:
        b, g, r = cv2.split(image)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        return cv2.merge([b_eq, g_eq, r_eq])


# Contrast Stretching
def contrast_stretch(image: np.ndarray, in_low: int, in_high: int):
    img_float = image.astype(np.float32)
    img_stretched = (img_float - in_low) * (255.0 / max(in_high - in_low, 1e-6))
    return np.clip(img_stretched, 0, 255).astype(np.uint8)


# Gamma Correction
def gamma_correction(image: np.ndarray, gamma: float = 1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)