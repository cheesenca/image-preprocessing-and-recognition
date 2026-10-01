# Image Processing and Recognition

## Overview
**Image Processing and Recognition** is a desktop-style web application for experimenting with classic digital image processing and computer vision techniques. It provides an interactive Gradio Interface where users can upload an image, choose an operation, tune its parameters with sliders and dropdowns, and instantly see the result. The project is organized into two layers:
- **Feature modules** (`app/features/`): pure functions built on OpenCV, NumPy, SciPy and scikit-image. Each function takes an image (`numpy.ndarray`) plus parameters and returns a processed image.
- **UI layer** (`app/ui/`): the Gradio interface (`gui.py`) and the handler functions (`handlers.py`) that route UI selections to the correct feature function.
**Tech stack:** Python, OpenCV, NumPy, SciPy, scikit-image, scikit-learn, Gradio.

## Features
### 1. Basic Image Operations
- **Greyscale:** Convert a BGR/RGB image to a single-channel greyscale image
- **Negative:** Invert pixel intensities (`255 - pixel`) 
- **Adjust Color:** Scale the Red, Green and Blue channels independently (0.0 – 2.0) 
- **Flip:** Flip horizontally, vertically or diagonally 
- **Translate:** Shift the image along X and Y (-200 to 200 px) 
- **Scale:** Resize by width and/or height, with optional aspect-ratio lock 
- **Rotate:** Rotate by an angle (-180° to 180°), clockwise or counter-clockwise 
- **Crop:** Crop a region defined by X, Y, width and height 
- **Blend:** Weighted blend of two images using an alpha value 
- **Brightness & Contrast:** Adjust brightness (via HSV value channel) and contrast 
- **Color Filter:** Custom HSV range mask, or preset **Sepia** and **Cyanotype** filters 
- **Add Border:** Add a constant-colour border (top, bottom, left, right, RGB colour) 
- **Overlay:** Paste a second image onto the first at (X, Y) with transparency 
### 2. Mathematical Operations
- **Arithmetic:** Add, Subtract, Multiply, Divide (normalized to 0–255)
- **Bitwise:** AND, OR, XOR, NOT (NOT ignores the second image)
### 3. Transforms & Filtering
- **FFT:** log-scaled magnitude spectrum of the 2D Fourier transform
- **Smoothing:** Mean Blur, Gaussian Blur, Median Blur
- **Edge detection:** Sobel, Canny (two thresholds), Laplacian Even kernel sizes are automatically adjusted to the next odd number where required.
### 4. Enhancement
- **Histogram Equalization:** per-channel for colour images
- **Contrast Stretching:** linear mapping between a chosen input low/high range
- **Gamma Correction:** lookup-table based gamma adjustment (0.1 – 5.0)
### 5. Compression
- **RLE (Run-Length Encoding):** lossless encode and decode of the greyscale image
- **DCT (Discrete Cosine Transform):** lossy compression by thresholding DCT coefficients and reconstructing the image
### 6. Segmentation
- **Global Thresholding:** fixed threshold (0–255)
- **Adaptive Thresholding:** local mean threshold with configurable block size and constant `C`
- **K-Means:** colour/intensity clustering with `K` from 2 to 10
### 7. Binary Processing
- **Binarize:** threshold to a binary image
- **Morphological operations:** erosion, dilation, opening, closing (kernel size and iterations adjustable)
- **Extract Boundary:** original minus eroded image
- **Skeletonize:** thin shapes to a one-pixel-wide skeleton (scikit-image)
### 8. Restoration
- **Add Noise + Wiener Filter:** adds Gaussian noise, then denoises with a local-statistics Wiener filter
- **Add Noise + Gaussian Filter:** adds Gaussian noise, then denoises with a Gaussian blur
- **Inpainting:** fills a rectangular region (top-left / bottom-right) using the Telea algorithm
### 9. Matching
- **Feature Detection & Matching:** ORB or SIFT keypoints, matched with Brute-Force or FLANN, displaying the best 20 matches
- **Template Matching:** normalized cross-correlation (`TM_CCOEFF_NORMED`) with an adjustable threshold; matches are drawn as red rectangles

## Folder Structure
```
image-preprocessing-and-recognition/
├── app/
│   ├── features/
│   │   ├── __init__.py
│   │   ├── basic_operations.py
│   │   ├── binary_processing.py
│   │   ├── compression.py
│   │   ├── enhancement.py
│   │   ├── matching.py
│   │   ├── math_operations.py
│   │   ├── restoration.py
│   │   ├── segmentation.py
│   │   └── transforms_filtering.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── gui.py
│   │   └── handlers.py
│   └── main.py
├── venv/
├── README.md
└── requirements.txt
```

## Installation Setup
### 1. Clone or extract the project
```bash
cd image-preprocessing-and-recognition
```
 
### 2. Create and activate a virtual environment
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# macOS / Linux
source venv/bin/activate
```
 
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
 
`requirements.txt` includes:
 
```
opencv-python
numpy
gradio
scikit-learn
scipy
scikit-image
```
 
### 4. Run the application
```bash
cd app
python main.py
```