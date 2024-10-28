import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the grayscale image
img = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg", cv.IMREAD_GRAYSCALE)

# Check if the image is loaded
if img is None:
    print("Error: Image not loaded.")
else:
    # Calculate the histogram (probability distribution)
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256], density=True)

    # Create an array for the x-axis (intensity values)
    x = np.arange(0, 256)

    # Plot the grayscale image and probability histogram side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Show the grayscale image
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Grayscale Image')
    ax1.axis('off')  # Hide axes for the image

    # Plot the probability histogram
    ax2.bar(x, hist, color="gray", align="center")
    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Probability')
    ax2.set_title('Probability Histogram of Grayscale Image')

    plt.tight_layout()
    plt.show()
