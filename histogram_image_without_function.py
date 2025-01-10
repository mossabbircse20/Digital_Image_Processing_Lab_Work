import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

img = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg", cv.IMREAD_GRAYSCALE)

rows, cols = img.shape

histogram = np.zeros(256, dtype=int)

for i in range(rows):
    for j in range(cols):
        pixel_intensity = img[i,j]
        histogram[pixel_intensity] += 1
        

x = np.arange(256)

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")


plt.subplot(2,2,2)
plt.bar(x, histogram, color = "Green", width= 0.5 )
plt.title("Grayscale Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()