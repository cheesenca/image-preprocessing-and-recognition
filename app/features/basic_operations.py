import cv2
import numpy as np

# Grayscale
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Negative
def to_negative(image):
    return 255 - image


# Adjust Color
def adjust_color(image, r_factor=1.0, g_factor=1.0, b_factor=1.0):
    b, g, r = cv2.split(image)
    b = np.clip(b * b_factor, 0, 255).astype(np.uint8)
    g = np.clip(g * g_factor, 0, 255).astype(np.uint8)
    r = np.clip(r * r_factor, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


# Flip
def flip_image(image, direction="horizontal"):
    flip_code = {'horizontal': 1, 'vertical': 0, 'diagonal': -1}
    return cv2.flip(image, flip_code.get(direction, 1))


# Translate
def translate(image, x_offset, y_offset):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, x_offset],
                    [0, 1, y_offset]])
    return cv2.warpAffine(image, M, (cols, rows))


# Scale
def scale_image(image, width=None, height=None, keep_aspect_ratio=True):
    h, w = image.shape[:2]
    if keep_aspect_ratio:
        if width is not None:
            ratio = width / float(w)
            height = int(h * ratio)
        elif height is not None:
            ratio = height / float(h)
            width = int(w * ratio)
    if width is None:
        width = w
    if height is None:
        height = h
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


# Rotate
def rotate(image, angle, clockwise=True):
    rows, cols = image.shape[:2]
    center = (cols / 2, rows / 2)
    if clockwise:
        angle = -angle
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    return rotated


# Crop
def crop(image, x, y, w, h):
    return image[y:y+h, x:x+w]


# Blend
def blend(image1, image2, alpha=0.5):
    if image2 is None:
        return image1
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)


# Brightness & Contrast
def adjust_brightness_contrast(image, brightness=50, contrast=1.2):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + brightness, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    image_brightness = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    image_float = image_brightness.astype(np.float32)
    image_contrast = np.clip(contrast * (image_float - 128) + 128, 0, 255)
    return image_contrast.astype(np.uint8)


# Color Filter
def color_filter(image, lower_bound, upper_bound):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return cv2.bitwise_and(image, image, mask=mask)


def apply_sepia(image):
    image_float = image.astype(np.float32)
    sepia_kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ])
    sepia_image = image_float @ sepia_kernel.T
    sepia_image = np.clip(sepia_image, 0, 255)
    return sepia_image.astype(np.uint8)


def apply_cyanotype(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cyan_image = np.zeros_like(image)
    cyan_image[:, :, 0] = np.clip(gray / 2, 0, 255)
    cyan_image[:, :, 1] = np.clip(gray, 0, 255)
    cyan_image[:, :, 2] = 255
    return cyan_image


# Border
def add_border(image, top, bottom, left, right, color_str="(0,0,0)"):
    color = tuple(map(int, color_str.strip("()").split(",")))
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


# Overlay
def overlay(image1, image2, x, y, alpha=0.5):
    if image2 is None:
        return image1
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    if y + h2 > h1 or x + w2 > w1:
        new_h = min(h2, h1 - y)
        new_w = min(w2, w1 - x)
        image2 = cv2.resize(image2, (new_w, new_h))
        h2, w2 = new_h, new_w
    roi = image1[y:y+h2, x:x+w2]
    blended = cv2.addWeighted(roi, 1 - alpha, image2, alpha, 0)
    image1[y:y+h2, x:x+w2] = blended
    return image1