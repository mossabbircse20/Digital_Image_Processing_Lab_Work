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

    # Apply Gaussian Blur
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)

    # Unsharp Masking
    img_sharpened = cv.addWeighted(img_gray, 1.5, img_blur, -0.5, 0)

    # Plotting the original and sharpened images
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Sharpened Image
    plt.subplot(1, 2, 2)
    plt.imshow(img_sharpened, cmap='gray')
    plt.title('Sharpened Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
