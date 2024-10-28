import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the original image
img_original = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg")

# Check if the image was loaded properly
if img_original is None:
    print("Error: Image not loaded.")
else:
    # Convert to grayscale
    img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur (Low-pass filter)
    img_gaussian = cv.GaussianBlur(img_gray, (5, 5), 0)

    # Apply Median Blur
    img_median = cv.medianBlur(img_gray, 5)

    # Apply Bilateral Filter
    img_bilateral = cv.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply Sobel Edge Detection
    sobel_x = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=5)
    img_sobel = cv.magnitude(sobel_x, sobel_y)

    # High-Pass Filter using kernel
    kernel_hp = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])  # A simple high-pass filter kernel
    img_highpass = cv.filter2D(img_gray, -1, kernel_hp)

    # Low-Pass Filters
    # 1. Averaging Filter
    kernel_avg = np.ones((5, 5), np.float32) / 25  # Simple averaging filter
    img_averaging = cv.filter2D(img_gray, -1, kernel_avg)

    # 2. Box Filter
    img_box = cv.boxFilter(img_gray, -1, (5, 5))  # Box filter

    # Plotting the original and filtered images
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # Original Image
    axes[0, 0].imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Grayscale Image
    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title('Grayscale Image')
    axes[0, 1].axis('off')

    # Gaussian Blur
    axes[0, 2].imshow(img_gaussian, cmap='gray')
    axes[0, 2].set_title('Gaussian Blur (Low-pass Filter)')
    axes[0, 2].axis('off')

    # Median Blur
    axes[1, 0].imshow(img_median, cmap='gray')
    axes[1, 0].set_title('Median Blur')
    axes[1, 0].axis('off')

    # Bilateral Filter
    axes[1, 1].imshow(img_bilateral, cmap='gray')
    axes[1, 1].set_title('Bilateral Filter')
    axes[1, 1].axis('off')

    # Averaging Filter
    axes[1, 2].imshow(img_averaging, cmap='gray')
    axes[1, 2].set_title('Averaging Filter (Low-pass)')
    axes[1, 2].axis('off')

    # Box Filter
    axes[2, 0].imshow(img_box, cmap='gray')
    axes[2, 0].set_title('Box Filter (Low-pass)')
    axes[2, 0].axis('off')

    # Sobel Edge Detection
    axes[2, 1].imshow(img_sobel, cmap='gray')
    axes[2, 1].set_title('Sobel Edge Detection')
    axes[2, 1].axis('off')

    # High-Pass Filter
    axes[2, 2].imshow(img_highpass, cmap='gray')
    axes[2, 2].set_title('High-Pass Filter')
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.show()
