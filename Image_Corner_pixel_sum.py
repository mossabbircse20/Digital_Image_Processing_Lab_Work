import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv.imread(r"C:\Users\User\Pictures\Untitled.png")

# Check if the image is loaded properly
if img is None:
    print("Error: Image not loaded.")
else:
    # Convert to grayscale for simplicity (if needed)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Get the image dimensions
    rows, cols = img_gray.shape

    # Extract the corner pixel values
    top_left = img_gray[0, 0]
    top_right = img_gray[0, cols - 1]
    bottom_left = img_gray[rows - 1, 0]
    bottom_right = img_gray[rows - 1, cols - 1]

    # Calculate the sum of the corner pixel values
    corner_sum = top_left + top_right + bottom_left + bottom_right

# Print the corner pixel values and their sum
    print(f"Top-left corner pixel value: {top_left}")
    print(f"Top-right corner pixel value: {top_right}")
    print(f"Bottom-left corner pixel value: {bottom_left}")
    print(f"Bottom-right corner pixel value: {bottom_right}")
    print(f"Sum of corner pixels: {corner_sum}")

    # Plot the image and highlight the corners
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Show the grayscale image
    ax.imshow(img_gray, cmap='gray')
    ax.set_title('Image with Corner Pixels Highlighted')

    # Mark the corner pixels on the image
    ax.plot(0, 0, 'ro')  # top-left
    ax.plot(cols - 1, 0, 'ro')  # top-right
    ax.plot(0, rows - 1, 'ro')  # bottom-left
    ax.plot(cols - 1, rows - 1, 'ro')  # bottom-right

    ax.axis('off')  # Hide axes for the image
    plt.show()

