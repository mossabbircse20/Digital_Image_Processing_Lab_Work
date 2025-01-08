import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the grayscale image
img = cv.imread(r"D:\CSE\CSE 4-1\DIP\Digital_Image_Processing_Lab_Work\A_letter.png", cv.IMREAD_GRAYSCALE)

# Check if the image is loaded
if img is None:
    print("Error: Image not loaded.")
else:
    # Threshold the image to get a binary image
    _, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # Define a kernel (structuring element)
    kernel = np.ones((7, 7), np.uint8)  # 5x5 kernel

    # --- Morphological Operations ---

    # 1. Dilation
    dilation = cv.dilate(binary, kernel, iterations=4)

    # 2. Erosion
    erosion = cv.erode(binary, kernel, iterations=4)

    # 3. Opening (Erosion followed by Dilation)
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    # 4. Closing (Dilation followed by Erosion)
    closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    # Plotting the Results
    plt.figure(figsize=(15, 10))

    # Original Binary Image
    plt.subplot(2, 3, 1)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Image")
    plt.axis("off")

    # Dilation
    plt.subplot(2, 3, 2)
    plt.imshow(erosion, cmap='gray')
    plt.title("Dilation")
    plt.axis("off")

    # Erosion
    plt.subplot(2, 3, 3)
    plt.imshow(dilation, cmap='gray')
    plt.title("Erosion")
    plt.axis("off")

    # Opening
    plt.subplot(2, 3, 4)
    plt.imshow(opening, cmap='gray')
    plt.title("Opening")
    plt.axis("off")

    # Closing
    plt.subplot(2, 3, 5)
    plt.imshow(closing, cmap='gray')
    plt.title("Closing")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
