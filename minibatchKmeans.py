import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

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

# Apply MiniBatchKMeans clustering
kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=1000)
kmeans_labels = kmeans.fit_predict(scaled_pixel_values)
kmeans_segmented_image = kmeans_labels.reshape(resized_image.shape)

# Color Mapping for MiniBatchKMeans
colored_kmeans_segmented_image = np.zeros((resized_image.shape[0], resized_image.shape[1], 3), dtype=np.uint8)
colored_kmeans_segmented_image[kmeans_segmented_image == 0] = [0, 255, 0]  # Green for non-oil spill
colored_kmeans_segmented_image[kmeans_segmented_image == 1] = [255, 255, 0]  # Yellow for oil spill

# Plot the results
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(resized_image, cmap='gray')
plt.axis('off')

# MiniBatchKMeans Segmented Image
plt.subplot(1, 2, 2)
plt.title('Clustered Image (MiniBatchKMeans - Yellow/Green)')
plt.imshow(colored_kmeans_segmented_image)
plt.axis('off')

plt.show()

# Calculate the area of the oil spill using MiniBatchKMeans
oil_spill_area_kmeans = np.sum(kmeans_segmented_image == 1)

# Convert to percentage of total image area
total_pixels = resized_image.shape[0] * resized_image.shape[1]
oil_spill_percentage_kmeans = (oil_spill_area_kmeans / total_pixels) * 100

# Display the calculated percentage
oil_spill_percentage_kmeans
