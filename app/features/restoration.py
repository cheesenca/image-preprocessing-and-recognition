import cv2
import numpy as np

# Add Noise
def add_noise(image: np.ndarray, mean=0, std=25):
    gauss = np.random.normal(mean, std, image.shape)
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


# Wiener Filter
def wiener_filter(image: np.ndarray, ksize=5, noise_var=0.01):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    local_mean = cv2.filter2D(image, -1, kernel)
    local_var = cv2.filter2D(image ** 2, -1, kernel) - (local_mean ** 2)
    noise_var = max(noise_var, 0.0001)
    result = local_mean + ((local_var - noise_var) / np.maximum(local_var, noise_var)) * (image - local_mean)
    return np.clip(result, 0, 255).astype(np.uint8)


# Gaussian Filter
def gaussian_filter_restoration(image: np.ndarray, ksize=5, sigma=1.0):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


# Inpainting
def inpaint_image(image: np.ndarray, top_left, bottom_right, inpaint_radius=3):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    r1, c1 = top_left
    r2, c2 = bottom_right
    mask[r1:r2, c1:c2] = 255
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)