import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS

# Load the image
image_path = '/content/Down.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image loaded successfully
if image is None:
    raise FileNotFoundError(f"Image at path {image_path} could not be loaded. Please check the file path.")

# Resize the image to reduce memory usage
scale_percent = 50  # Scale down to 50% of the original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Reshape the image to a 2D array of pixels and 1 color value (grayscale)
pixel_values = resized_image.reshape((-1, 1))
pixel_values = np.float32(pixel_values)

# Standardize the data
scaler = StandardScaler()
scaled_pixel_values = scaler.fit_transform(pixel_values)

# Apply OPTICS clustering
optics = OPTICS(min_samples=10, max_eps=0.5)
optics_labels = optics.fit_predict(scaled_pixel_values)
optics_segmented_image = optics_labels.reshape(resized_image.shape)

# Color Mapping for OPTICS
colored_optics_segmented_image = np.zeros((resized_image.shape[0], resized_image.shape[1], 3), dtype=np.uint8)
unique_labels = np.unique(optics_labels)
for label in unique_labels:
    if label == -1:
        color = [255, 255, 255]  # White for noise
    else:
        color = [0, 255, 0] if label == 0 else [255, 255, 0]  # Green for one cluster, Yellow for another
    colored_optics_segmented_image[optics_segmented_image == label] = color

# Plot the results
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(resized_image, cmap='gray')
plt.axis('off')

# OPTICS Segmented Image
plt.subplot(1, 2, 2)
plt.title('Clustered Image (OPTICS - Yellow/Green)')
plt.imshow(colored_optics_segmented_image)
plt.axis('off')

plt.show()

# Calculate the area of the oil spill using OPTICS
oil_spill_area_optics = np.sum(optics_segmented_image == 1)

# Convert to percentage of total image area
total_pixels = resized_image.shape[0] * resized_image.shape[1]
oil_spill_percentage_optics = (oil_spill_area_optics / total_pixels) * 100

# Display the calculated percentage
oil_spill_percentage_optics
