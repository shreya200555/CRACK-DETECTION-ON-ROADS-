import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread("crack1.jpg")

if img is None:
    print("Error: Image not found. Place 'jp.jpg' in your folder.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur for noise reduction
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply simple thresholding to get a binary image. The value 50 can be adjusted.
_, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)

# Morphological closing to connect crack segments
kernel = np.ones((5, 5), np.uint8)
closed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw on
output_img = img.copy()

# Draw the cracks on the original image
cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2)

# Show results
plt.figure(figsize=(15, 7))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 4, 2)
plt.title("Thresholded Image")
plt.imshow(thresh, cmap='gray')

plt.subplot(1, 4, 3)
plt.title("After Morphological Operations")
plt.imshow(closed_img, cmap='gray')

plt.subplot(1, 4, 4)
plt.title("Detected Cracks")
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))

plt.show()