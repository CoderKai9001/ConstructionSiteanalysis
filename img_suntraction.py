import cv2
import numpy as np

# Load the two images
image1 = cv2.imread('yolo_training/sub_imgs/before_wc.jpeg')
image2 = cv2.imread('yolo_training/sub_imgs/after_wc.jpeg')

# Ensure both images are the same size
if image1.shape != image2.shape:
    raise ValueError("Images must have the same dimensions.")

# Subtract the images to get the absolute difference
difference = cv2.absdiff(image1, image2)

# Convert the difference to grayscale for MSE calculation
gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

# Calculate the Mean Squared Error (MSE) to quantify the difference
mse = np.sum((gray_diff.astype("float")) ** 2) / float(image1.shape[0] * image1.shape[1])

# Normalize the difference image to get a percentage of how different the images are
max_diff = 255  # Since images are 8-bit per channel
difference_percentage = (np.sum(gray_diff) / (image1.shape[0] * image1.shape[1] * max_diff)) * 100

# Threshold the difference image to highlight only significant differences
_, thresh_diff = cv2.threshold(gray_diff, 100, 255, cv2.THRESH_BINARY)

# Convert thresholded difference back to a BGR image for color display
highlighted_diff = cv2.cvtColor(thresh_diff, cv2.COLOR_GRAY2BGR)

# Optionally highlight the difference in color (e.g., in red) for better visibility
highlighted_diff[np.where((highlighted_diff == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  # Red

# Display the original images and the highlighted difference
cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.imshow('Difference', highlighted_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the difference metrics
print(f"Mean Squared Error: {mse}")
print(f"Difference Percentage: {difference_percentage:.2f}%")