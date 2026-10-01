import cv2
import numpy as np

# Pixelwise Operations
def pixelwise_operation(image1, image2, operation, brightness_factor=1.2):
    if operation == 'Bitwise (NOT)':
        return cv2.bitwise_not(image1)
    
    if image2 is None:
        return image1

    # Resize If Mismatch
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    if operation == 'Add':
        return cv2.add(image1, image2)
    elif operation == 'Subtract':
        return cv2.subtract(image1, image2)
    elif operation == 'Multiply':
        return cv2.multiply(image1, image2)
    elif operation == 'Divide':
        epsilon = 1e-5
        image2 = image2.astype(np.float32) + epsilon
        image1 = image1.astype(np.float32)
        divided = cv2.divide(image1, image2)
        divided_normalized = cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.convertScaleAbs(divided_normalized, alpha=brightness_factor, beta=0)
    elif operation == 'Bitwise (AND)':
        return cv2.bitwise_and(image1, image2)
    elif operation == 'Bitwise (OR)':
        return cv2.bitwise_or(image1, image2)
    elif operation == 'Bitwise (XOR)':
        return cv2.bitwise_xor(image1, image2)
    else:
        return image1