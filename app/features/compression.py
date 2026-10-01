import numpy as np
import cv2
from scipy.fftpack import dct, idct

# RLE
def rle_encode(image: np.ndarray):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat = image.flatten()
    encoding = []
    prev = flat[0]
    count = 1
    for pixel in flat[1:]:
        if pixel == prev:
            count += 1
        else:
            encoding.append((int(prev), count))
            prev = pixel
            count = 1
    encoding.append((int(prev), count))
    return encoding


def rle_decode(encoding, shape):
    flat_image = []
    for pixel_value, count in encoding:
        flat_image.extend([pixel_value] * count)
    return np.array(flat_image, dtype=np.uint8).reshape(shape)


# DCT
def apply_dct(image, threshold_ratio=0.01):
    dct_transformed = dct(dct(image.T, norm='ortho').T, norm='ortho')
    dct_thresh = dct_transformed * (np.abs(dct_transformed) > threshold_ratio * np.max(dct_transformed))
    img_reconstructed = idct(idct(dct_thresh.T, norm='ortho').T, norm='ortho')
    return np.clip(img_reconstructed, 0, 255).astype(np.uint8)