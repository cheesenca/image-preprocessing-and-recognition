import cv2
import numpy as np

# Binarize
def binarize_image(image: np.ndarray, thresh: int = 127):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return binary


# Morphological
def morphological_op(binary_image: np.ndarray, op_type: str, ksize: int, iterations: int):
    kernel = np.ones((ksize, ksize), np.uint8)
    if op_type == 'erosion':
        return cv2.erode(binary_image, kernel, iterations=iterations)
    elif op_type == 'dilation':
        return cv2.dilate(binary_image, kernel, iterations=iterations)
    elif op_type == 'opening':
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op_type == 'closing':
        return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        return binary_image


# Extract Boundary
def extract_boundary(binary_image: np.ndarray, ksize: int = 3):
    kernel = np.ones((ksize, ksize), np.uint8)
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    return cv2.subtract(binary_image, eroded)


# Skeletonize
def skeletonize_image(binary_image: np.ndarray):
    from skimage.morphology import skeletonize
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    bin_bool = (binary_image > 0)
    skeleton = skeletonize(bin_bool)
    return (skeleton * 255).astype(np.uint8)