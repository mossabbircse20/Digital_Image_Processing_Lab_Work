import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the grayscale image
img = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg", cv.IMREAD_GRAYSCALE)

# Check if the image is loaded
if img is None:
    print("Error: Image not loaded.")
else:
    # Log Transform
    c_log = 255 / np.log(1 + np.max(img))  # Calculate constant c
    img_log = c_log * np.log(1 + img.astype(np.float64))  # Apply log transform
    img_log = np.uint8(np.clip(img_log, 0, 255))  # Convert back to uint8

    # Power-Law Transform (Gamma Correction)
    gamma = 0.5  # Example gamma value < 1 enhances darker pixels
    c_power = 255 / (np.max(img) ** gamma)  # Calculate constant c
    img_power = c_power * (img.astype(np.float64) ** gamma)  # Apply power transform
    img_power = np.uint8(np.clip(img_power, 0, 255))  # Convert back to uint8

    # Plot the original, log transform, and power-law transform images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Show the original grayscale image
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Grayscale Image')
    ax1.axis('off')

    # Show the log transformed image
    ax2.imshow(img_log, cmap='gray')
    ax2.set_title('Log Transform')
    ax2.axis('off')

    # Show the power-law transformed image
    ax3.imshow(img_power, cmap='gray')
    ax3.set_title(f'Power-Law Transform (Gamma={gamma})')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()
