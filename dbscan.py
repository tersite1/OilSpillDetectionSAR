import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load the image
image_path = '/content/Down.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image loaded successfully
if image is None:
    raise FileNotFoundError(f"Image at path {image_path} could not be loaded. Please check the file path.")

# Further downsample the image to reduce memory usage
scale_percent = 25  # Further reduce image size to 25% of the original
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Optionally, use only a subset of the image
subset_image = resized_image[0:500, 0:500]  # Use a 500x500 subset

# Reshape the subset image to a 2D array of pixels and 1 color value (grayscale)
pixel_values = subset_image.reshape((-1, 1))
pixel_values = np.float32(pixel_values)

# Standardize the data
scaler = StandardScaler()
scaled_pixel_values = scaler.fit_transform(pixel_values)

# Apply DBSCAN clustering with tuned parameters
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan_labels = dbscan.fit_predict(scaled_pixel_values)
dbscan_segmented_image = dbscan_labels.reshape(subset_image.shape)

# Color Mapping for DBSCAN
colored_dbscan_segmented_image = np.zeros((subset_image.shape[0], subset_image.shape[1], 3), dtype=np.uint8)
unique_labels = np.unique(dbscan_labels)
for label in unique_labels:
    if label == -1:
        color = [255, 255, 255]  # White for noise
    else:
        color = [0, 255, 0] if label == 0 else [255, 255, 0]  # Green for one cluster, Yellow for another
    colored_dbscan_segmented_image[dbscan_segmented_image == label] = color

# Plot the results
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image (Subset)')
plt.imshow(subset_image, cmap='gray')
plt.axis('off')

# DBSCAN Segmented Image
plt.subplot(1, 2, 2)
plt.title('Clustered Image (DBSCAN - Yellow/Green)')
plt.imshow(colored_dbscan_segmented_image)
plt.axis('off')

plt.show()

# Calculate the area of the oil spill using DBSCAN
oil_spill_area_dbscan = np.sum(dbscan_segmented_image == 1)

# Convert to percentage of total image area
total_pixels = subset_image.shape[0] * subset_image.shape[1]
oil_spill_percentage_dbscan = (oil_spill_area_dbscan / total_pixels) * 100

# Display the calculated percentage
oil_spill_percentage_dbscan
