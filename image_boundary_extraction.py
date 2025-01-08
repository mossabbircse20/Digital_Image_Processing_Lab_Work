import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the grayscale image
img = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg", cv.IMREAD_GRAYSCALE)

# Check if the image is loaded
if img is None:
    print("Error: Image not loaded.")
else:
    # Threshold the image to get a binary image
    _, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # Perform erosion
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for erosion
    eroded = cv.erode(binary, kernel, iterations=1)

    # Extract the boundary by subtracting the eroded image from the binary image
    boundary = cv.subtract(binary, eroded)

    # Plotting the results
    plt.figure(figsize=(10, 5))

    # Original Grayscale Image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis("off")

    # Binary Image
    plt.subplot(1, 3, 2)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Image")
    plt.axis("off")

    # Boundary Image
    plt.subplot(1, 3, 3)
    plt.imshow(boundary, cmap='gray')
    plt.title("Boundary Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
