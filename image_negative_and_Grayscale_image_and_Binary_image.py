
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the original color image
img_original = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg")

# Load the image in grayscale
img = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg", cv.IMREAD_GRAYSCALE)

# Check if the image was loaded properly
if img is None:
    print("Error: Image not loaded.")
else:
    # Get the dimensions of the image
    row, col = img.shape

    # Initialize the histogram array
    y = np.zeros((256), np.uint64)

    # Calculate the histogram
    for i in range(row):
        for j in range(col):
            y[img[i, j]] += 1

    # Create an array for the x-axis (intensity values)
    x = np.arange(0, 256)

    # Create the negative image
    img_negative = 255 - img  # Invert grayscale pixel values for a negative effect

    # Create the binary image using a threshold
    threshold_value = 127
    max_value = 255
    _, img_binary = cv.threshold(img, threshold_value, max_value, cv.THRESH_BINARY)

    # Plot the original image, grayscale image, negative image, histogram, and binary image
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 5))

    # Show the original color image
    ax1.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    ax1.set_title('Original Image')
    ax1.axis('off')  # Hide axes for the image

    # Show the grayscale image
    ax2.imshow(img, cmap='gray')
    ax2.set_title('Grayscale Image')
    ax2.axis('off')

    # Show the negative image
    ax3.imshow(img_negative, cmap='gray')
    ax3.set_title('Negative Image')
    ax3.axis('off')

    # Show the binary image
    ax4.imshow(img_binary, cmap='gray')
    ax4.set_title('Binary Image')
    ax4.axis('off')

    # Plot the histogram
    ax5.bar(x, y, color="gray", align="center")
    ax5.set_xlabel('Pixel Intensity')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Grayscale Histogram')

    plt.tight_layout()
    plt.show()

