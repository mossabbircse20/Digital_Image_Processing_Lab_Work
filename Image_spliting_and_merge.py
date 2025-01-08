import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg")

# Check if the image is loaded properly
if img is None:
    print("Error: Image not loaded.")
else:
    # Get the dimensions of the image
    height, width, _ = img.shape

    # Define the midpoint for slicing
    mid_height = height // 2
    mid_width = width // 2

    # Slice the image into 4 parts (top-left, top-right, bottom-left, bottom-right)
    top_left = img[0:mid_height, 0:mid_width]
    top_right = img[0:mid_height, mid_width:width]
    bottom_left = img[mid_height:height, 0:mid_width]
    bottom_right = img[mid_height:height, mid_width:width]

    # Merge the image back together
    top = np.hstack((top_left, top_right))  # Merge top-left and top-right horizontally
    bottom = np.hstack((bottom_left, bottom_right))  # Merge bottom-left and bottom-right horizontally
    merged_image = np.vstack((top, bottom))  # Merge top and bottom vertically

    # Display all images in one figure
    plt.figure(figsize=(12, 12))

    plt.subplot(4, 2, 1)
    plt.imshow(cv.cvtColor(top_left, cv.COLOR_BGR2RGB))
    plt.title('Top Left')
    plt.axis('off')

    plt.subplot(4, 2, 2)
    plt.imshow(cv.cvtColor(top_right, cv.COLOR_BGR2RGB))
    plt.title('Top Right')
    plt.axis('off')

    plt.subplot(4, 2, 3)
    plt.imshow(cv.cvtColor(bottom_left, cv.COLOR_BGR2RGB))
    plt.title('Bottom Left')
    plt.axis('off')

    plt.subplot(4, 2, 4)
    plt.imshow(cv.cvtColor(bottom_right, cv.COLOR_BGR2RGB))
    plt.title('Bottom Right')
    plt.axis('off')

    plt.subplot(4, 2, 5)
    plt.imshow(cv.cvtColor(merged_image, cv.COLOR_BGR2RGB))
    plt.title('Merged Image (Original)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Optionally, save the merged image
    cv.imwrite('merged_image.jpg', merged_image)
