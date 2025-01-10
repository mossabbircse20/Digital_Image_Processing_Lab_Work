import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

img = cv.imread(r"E:\Mobile Doc\Cat 2021\IMG_3894.jpg", cv.IMREAD_GRAYSCALE)

rows, cols = img.shape
total_pixel = rows * cols 

hist = np.zeros(256, dtype=int)

for i in range(rows):
    for j in range(cols):
        pixel_intesity = img[i,j]
        hist[pixel_intesity] += 1
        

cdf = np.zeros(256, dtype=float)
csum = 0

for i in range(256):
    csum += hist[i]
    cdf[i] = csum / total_pixel
    

equalized_img = np.zeros_like(img,dtype=np.uint8)
equalized_hist = np.zeros(256, dtype=int)

for i in range(rows):
    for j in range(cols):
        original_intensity = img[i,j]
        equalized_img[i,j] = np.round(cdf[original_intensity] * 255).astype(np.uint8)
        equalized_hist[equalized_img[i,j]] += 1
        


plt.figure(figsize=(15,5))

plt.subplot(2,2,1)
plt.imshow(img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.bar(np.arange(256), hist , color = "Blue" , width= 0.5)
plt.title("Grayscale Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2,2,3)
plt.imshow(equalized_img, cmap="gray")
plt.title("Equalized Grayscale Image")
plt.axis("off")

plt.subplot(2,2,4)
plt.bar(np.arange(256), equalized_hist, color = "Green" , width= 0.5)
plt.title("Equalized Grayscale Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
