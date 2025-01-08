import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the original image
img_original = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg")

# Check if the image is loaded
if img_original is None:
    print("Error: Image not loaded.")
else:
    # Convert to RGB for display
    img_rgb = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)

    # --- Apply Different Blurring Techniques ---

    # 1. Averaging (Mean Blurring)
    blur_avg = cv.blur(img_original, (5, 5))

    # 2. Gaussian Blurring
    blur_gaussian = cv.GaussianBlur(img_original, (5, 5), 0)

    # 3. Median Blurring
    blur_median = cv.medianBlur(img_original, 5)

    # 4. Bilateral Filtering (Retains edges while reducing noise)
    blur_bilateral = cv.bilateralFilter(img_original, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert all blurred images to RGB for display
    blur_avg_rgb = cv.cvtColor(blur_avg, cv.COLOR_BGR2RGB)
    blur_gaussian_rgb = cv.cvtColor(blur_gaussian, cv.COLOR_BGR2RGB)
    blur_median_rgb = cv.cvtColor(blur_median, cv.COLOR_BGR2RGB)
    blur_bilateral_rgb = cv.cvtColor(blur_bilateral, cv.COLOR_BGR2RGB)

    # Plotting the Results
    plt.figure(figsize=(15, 10))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # Averaging Blurring
    plt.subplot(2, 3, 2)
    plt.imshow(blur_avg_rgb)
    plt.title("Averaging Blur")
    plt.axis("off")

    # Gaussian Blurring
    plt.subplot(2, 3, 3)
    plt.imshow(blur_gaussian_rgb)
    plt.title("Gaussian Blur")
    plt.axis("off")

    # Median Blurring
    plt.subplot(2, 3, 4)
    plt.imshow(blur_median_rgb)
    plt.title("Median Blur")
    plt.axis("off")

    # Bilateral Filtering
    plt.subplot(2, 3, 5)
    plt.imshow(blur_bilateral_rgb)
    plt.title("Bilateral Filter")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
