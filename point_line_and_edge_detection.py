import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the original image
img_original = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg")

# Check if the image was loaded properly
if img_original is None:
    print("Error: Image not loaded.")
else:
    # Convert to grayscale for processing
    img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

    # --- Point Detection using Harris Corner Detection ---
    # Convert to float32 for corner detection
    img_gray_float = np.float32(img_gray)
    # Harris corner detection
    corners = cv.cornerHarris(img_gray_float, blockSize=2, ksize=3, k=0.04)
    # Result is dilated to mark the corners
    img_corners = cv.dilate(corners, None)
    # Threshold to select strong corners
    threshold = 0.01 * corners.max()
    img_corners[img_corners > threshold] = 255

    # --- Line Detection using Hough Line Transform ---
    # Apply Canny edge detection
    edges = cv.Canny(img_gray, 50, 150, apertureSize=3)
    # Use Hough Transform to detect lines
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

    # Create a copy of the original image to draw lines
    img_lines = img_original.copy()
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # Convert polar coordinates to Cartesian coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- Edge Detection using Canny ---
    img_edges = cv.Canny(img_gray, 100, 200)

    # Plotting the results
    plt.figure(figsize=(15, 10))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Point Detection (Corners)
    plt.subplot(2, 2, 2)
    plt.imshow(img_corners, cmap='gray')
    plt.title('Harris Corners')
    plt.axis('off')

    # Line Detection
    plt.subplot(2, 2, 3)
    plt.imshow(cv.cvtColor(img_lines, cv.COLOR_BGR2RGB))
    plt.title('Detected Lines')
    plt.axis('off')

    # Edge Detection
    plt.subplot(2, 2, 4)
    plt.imshow(img_edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
