import cv2
import numpy as np

# Global Thresholding
def global_threshold(image: np.ndarray, thresh: int = 128):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return binary


# Adaptive Thresholding
def adaptive_threshold(image: np.ndarray, block_size: int = 11, C: int = 2):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)


# K-Means
def kmeans_segmentation(image: np.ndarray, k: int = 2):
    if len(image.shape) == 2:
        data = image.reshape((-1, 1))
    else:
        data = image.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    if len(image.shape) == 2:
        segmented = segmented.reshape((image.shape[0], image.shape[1]))
    else:
        segmented = segmented.reshape((image.shape[0], image.shape[1], 3))
    return segmented