import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 

img = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg", cv.IMREAD_GRAYSCALE)

rows, cols = img.shape
total_pixel = rows * cols

hist = np.zeros(256, dtype=int)

x = np.arange(256)

for i in range(rows):
    for j in range(cols):
        pixel_intensity = img[i,j]
        hist[pixel_intensity] += 1

probability = hist / total_pixel

plt.figure(figsize= (10,5))

plt.subplot(2,2,1)
plt.imshow(img , cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.bar(x, probability , color = "Green" , align= "center", width = 0.5)
plt.title("Probability of Pixel Intensity")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()