from features.basic_operations import (
    to_grayscale, to_negative, adjust_color, flip_image, translate,
    scale_image, rotate, crop, blend, adjust_brightness_contrast,
    color_filter, apply_sepia, apply_cyanotype, add_border, overlay,
)
from features.math_operations import pixelwise_operation
from features.transforms_filtering import (
    fft_image, mean_blur, gaussian_blur, median_blur,
    sobel_edge, canny_edge, laplacian_edge,
)
from features.enhancement import (
    histogram_equalization, contrast_stretch, gamma_correction,
)
from features.compression import rle_encode, rle_decode, apply_dct
from features.segmentation import (
    global_threshold, adaptive_threshold, kmeans_segmentation,
)
from features.binary_processing import (
    binarize_image, morphological_op, extract_boundary, skeletonize_image,
)
from features.restoration import (
    add_noise, wiener_filter, gaussian_filter_restoration, inpaint_image,
)
from features.matching import match_features, template_match


# Tab: Basic Operations
def tab_basic_ops(img1, img2, operation, r_factor, g_factor, b_factor, flip_dir,
    trans_x, trans_y, scale_w, scale_h, scale_keep_aspect, rot_angle, rot_clockwise,
    crop_x, crop_y, crop_w, crop_h, blend_alpha, bright_val, contrast_val, filter_lower_H,
    filter_lower_S, filter_lower_V, filter_upper_H, filter_upper_S, filter_upper_V,
    border_top, border_bottom, border_left, border_right, border_color_str, overlay_x,
    overlay_y, overlay_alpha):
    if img1 is None:
        return None

    try:
        color_tuple = tuple(map(int, border_color_str.split(',')))
        if len(color_tuple) != 3:
            color_tuple = (0, 0, 0)
    except Exception:
        color_tuple = (0, 0, 0)

    if operation == "Greyscale":
        return to_grayscale(img1)
    elif operation == "Negative":
        return to_negative(img1)
    elif operation == "Adjust Color":
        return adjust_color(img1, r_factor, g_factor, b_factor)
    elif operation == "Flip":
        return flip_image(img1, flip_dir)
    elif operation == "Translate":
        return translate(img1, trans_x, trans_y)
    elif operation == "Scale":
        return scale_image(img1, width=scale_w, height=scale_h, keep_aspect_ratio=scale_keep_aspect)
    elif operation == "Rotate":
        return rotate(img1, rot_angle, clockwise=rot_clockwise)
    elif operation == "Crop":
        return crop(img1, crop_x, crop_y, crop_w, crop_h)
    elif operation == "Blend":
        return blend(img1, img2, alpha=blend_alpha)
    elif operation == "Brightness & Contrast":
        return adjust_brightness_contrast(img1, bright_val, contrast_val)
    elif operation == "Color Filter":
        lowerb = (filter_lower_H, filter_lower_S, filter_lower_V)
        upperb = (filter_upper_H, filter_upper_S, filter_upper_V)
        return color_filter(img1, lowerb, upperb)
    elif operation == "Sepia":
        return apply_sepia(img1)
    elif operation == "Cyanotype":
        return apply_cyanotype(img1)
    elif operation == "Add Border":
        return add_border(img1, border_top, border_bottom, border_left, border_right, color_str=border_color_str)
    elif operation == "Overlay":
        return overlay(img1, img2, overlay_x, overlay_y, alpha=overlay_alpha)
    else:
        return img1


# Mathematical Operations
def tab_pixelwise_op(image1, image2, operation):
    if image1 is None:
        return None
    return pixelwise_operation(image1, image2, operation)


# Transforms & Filtering
def tab_transform_filter(image, operation, ksize=3, sigma=1.0, t1=50, t2=150):
    if image is None:
        return None
    if operation == "FFT":
        return fft_image(image)
    elif operation == "Mean Blur":
        return mean_blur(image, ksize)
    elif operation == "Gaussian Blur":
        return gaussian_blur(image, ksize, sigma)
    elif operation == "Median Blur":
        return median_blur(image, ksize)
    elif operation == "Sobel Edge":
        return sobel_edge(image, ksize)
    elif operation == "Canny Edge":
        return canny_edge(image, t1, t2)
    elif operation == "Laplacian Edge":
        return laplacian_edge(image, ksize)
    else:
        return image


# Image Enhancement
def tab_enhancement(image, operation, in_low=0, in_high=255, gamma_val=1.0):
    if image is None:
        return None
    if operation == "Histogram Equalization":
        return histogram_equalization(image)
    elif operation == "Contrast Stretching":
        return contrast_stretch(image, in_low, in_high)
    elif operation == "Gamma Correction":
        return gamma_correction(image, gamma_val)
    else:
        return image


# Image Compression
def tab_compression(image, operation, threshold_ratio=0.005):
    if image is None:
        return None
    if operation == "RLE":
        encoded = rle_encode(image)
        decoded_image = rle_decode(encoded, image.shape[:2])
        return decoded_image
    elif operation == "DCT":
        compressed_image = apply_dct(image, threshold_ratio=threshold_ratio)
        return compressed_image
    else:
        return image


# Image Segmentation
def tab_segmentation(image, operation, threshold=128, block_size=11, C=2, k=3):
    if image is None:
        return None
    if operation == "Global Thresholding":
        return global_threshold(image, threshold)
    elif operation == "Adaptive Thresholding":
        return adaptive_threshold(image, block_size, C)
    elif operation == "K-Means":
        return kmeans_segmentation(image, k)
    else:
        return image


# Binary Processing
def tab_binary(image, operation, thresh=127, morph_op_='erosion', ksize=3, iterations=1):
    if image is None:
        return None
    if operation == "Binarize":
        return binarize_image(image, thresh)
    elif operation == "Morphological":
        bin_img = binarize_image(image, thresh=127) if len(image.shape) == 3 else image
        return morphological_op(bin_img, morph_op_, ksize, iterations)
    elif operation == "Extract Boundary":
        bin_img = binarize_image(image, thresh=127) if len(image.shape) == 3 else image
        return extract_boundary(bin_img, ksize=ksize)
    elif operation == "Skeletonize":
        bin_img = binarize_image(image, thresh=127) if len(image.shape) == 3 else image
        return skeletonize_image(bin_img)
    else:
        return image


# Image Restoration
def tab_restoration(image, operation, ksize=5, sigma=1.0, noise_var=0.01, top_left_r=0, top_left_c=0, bot_right_r=10, bot_right_c=10, inpaint_radius=3):
    if image is None:
        return None
    if operation == "Add Noise + Wiener Filter":
        noisy = add_noise(image, 0, 25)
        return wiener_filter(noisy, ksize, noise_var)
    elif operation == "Add Noise + Gaussian Filter":
        noisy = add_noise(image, 0, 25)
        return gaussian_filter_restoration(noisy, ksize, sigma)
    elif operation == "Inpainting":
        return inpaint_image(image, (top_left_r, top_left_c), (bot_right_r, bot_right_c), inpaint_radius)
    else:
        return image


# Image Matching
def tab_matching(image1, image2, operation, feature_method='ORB', match_method='BF', templ_thresh=0.8):
    if image1 is None or image2 is None:
        return None
    if operation == "Feature Detection & Matching":
        return match_features(image1, image2, method=feature_method, match_method=match_method)
    elif operation == "Template Matching":
        return template_match(image1, image2, threshold=templ_thresh)
    else:
        return None