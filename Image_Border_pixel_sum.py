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

    # Define border thickness
    border_thickness = 10 # Change this value for a thicker or thinner border

    # Extract the border pixel values (top row, bottom row, left column, right column)
    top_row = img_gray[0, :]
    bottom_row = img_gray[rows - 1, :]
    left_column = img_gray[:, 0]
    right_column = img_gray[:, cols - 1]

    # Calculate the sum of the border pixel values
    border_sum = np.sum(top_row) + np.sum(bottom_row) + np.sum(left_column) + np.sum(right_column)

    # Print the sum of border pixels
    print(f"Sum of border pixels: {border_sum}")

    # Highlight the border in the original image by marking it in red
    img_border_marked = img.copy()  # Create a copy of the original image

    # Mark the top and bottom rows in red
    img_border_marked[0:border_thickness, :] = [0, 0, 255]  # Top border in red (BGR format)
    img_border_marked[rows - border_thickness:, :] = [0, 0, 255]  # Bottom border in red

    # Mark the left and right columns in red
    img_border_marked[:, 0:border_thickness] = [0, 0, 255]  # Left border in red
    img_border_marked[:, cols - border_thickness:] = [0, 0, 255]  # Right border in red

    # Display the original image with highlighted borders
    plt.figure(figsize=(6, 6))
    plt.imshow(cv.cvtColor(img_border_marked, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
    plt.title('Image with Bold Border Highlighted')
    plt.axis('off')  # Hide axes
    plt.show()

    # Save the output image with the marked border (optional)
    cv.imwrite('output_with_bold_border_marked.jpg', img_border_marked)
