import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the main image (source image)
img = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg", cv.IMREAD_GRAYSCALE)

# Load the template (pattern to search for)
template = cv.imread(r"E:\Mobile Doc\Cat 2021\template_img.jpg", cv.IMREAD_GRAYSCALE)

# Check if the images are loaded
if img is None or template is None:
    print("Error: Image or template not loaded.")
else:
    # Get dimensions of the template
    h, w = template.shape

    # Perform template matching
    result = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)

    # Set a threshold for detection
    threshold = 0.8
    loc = np.where(result >= threshold)

    # Draw rectangles around detected patterns
    img_with_boxes = img.copy()
    for pt in zip(*loc[::-1]):  # Switch x and y coordinates
        cv.rectangle(img_with_boxes, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 2)

    # Plotting the results
    plt.figure(figsize=(15, 10))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # Template (Pattern)
    plt.subplot(1, 3, 2)
    plt.imshow(template, cmap='gray')
    plt.title("Template (Pattern)")
    plt.axis("off")

    # Detected Patterns
    plt.subplot(1, 3, 3)
    plt.imshow(img_with_boxes, cmap='gray')
    plt.title("Detected Patterns")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
