import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the grayscale image
img = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg", cv.IMREAD_GRAYSCALE)

# Check if the image was loaded properly
if img is None:
    print("Error: Image not loaded.")
else:
    # Perform histogram equalization
    img_equalized = cv.equalizeHist(img)

    # Plot the original and equalized images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Show the original grayscale image
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Grayscale Image')
    ax1.axis('off')

    # Show the histogram equalized image
    ax2.imshow(img_equalized, cmap='gray')
    ax2.set_title('Histogram Equalized Image')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()