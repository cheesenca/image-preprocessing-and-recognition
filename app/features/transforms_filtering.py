import cv2
import numpy as np

# FFT
def fft_image(image: np.ndarray):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# Mean Blur
def mean_blur(image: np.ndarray, ksize: int = 3):
    return cv2.blur(image, (ksize, ksize))


# Gaussian Blur
def gaussian_blur(image: np.ndarray, ksize: int = 3, sigma: float = 1.0):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


# Median Blur
def median_blur(image: np.ndarray, ksize: int = 3):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(image, ksize)


# Sobel Edge
def sobel_edge(image: np.ndarray, ksize: int = 3):
    if ksize % 2 == 0:
        ksize += 1
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(sobelx, sobely)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


# Canny Edge
def canny_edge(image: np.ndarray, t1: float, t2: float):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(image, t1, t2)


# Laplacian Edge
def laplacian_edge(image: np.ndarray, ksize: int = 3):
    if ksize % 2 == 0:
        ksize += 1
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)