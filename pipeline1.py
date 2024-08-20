import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.ndimage import convolve

# Load the image
image_path = 'test.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. Lee Filter 적용
def lee_filter(img, size=5):
    img_mean = cv2.blur(img, (size, size))
    img_sqr_mean = cv2.blur(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = np.var(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    
    return img_output

lee_filtered_image = lee_filter(image)

# 2. Adaptive K-means 클러스터링 (Pixel Intensity 기반)
# Flatten the image and normalize pixel intensity
scaler = StandardScaler()
pixels = lee_filtered_image.flatten().reshape(-1, 1)
pixels = scaler.fit_transform(pixels)

# Adaptive K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(pixels)

# Reshape the labels back to the original image shape
kmeans_labels_image = kmeans_labels.reshape(lee_filtered_image.shape)

# 3. Erosion 연산 적용
# Define a kernel for erosion
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Convert the clustered image to binary (0s and 1s) for applying erosion
binary_image = (kmeans_labels_image * 255).astype(np.uint8)

# Apply erosion
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# 4. Morphological Operations (Opening) + 5. Connected Component Analysis
# Convert eroded image to binary (0 or 1)
_, binary_eroded_image = cv2.threshold(eroded_image, 127, 1, cv2.THRESH_BINARY)

# Apply Opening (Erosion followed by Dilation) to remove small noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened_image = cv2.morphologyEx(binary_eroded_image, cv2.MORPH_OPEN, kernel)

# Connected Component Analysis to remove small components (noise)
num_labels, labels_im = cv2.connectedComponents(opened_image.astype(np.uint8))

# Filter out small connected components (noise)
min_size = 100  # Minimum size of oil spill region to keep
final_filtered_image = np.zeros_like(labels_im)
for label in range(1, num_labels):  # Skip the background label
    if np.sum(labels_im == label) > min_size:
        final_filtered_image[labels_im == label] = 1

# Save the filtered result
final_filtered_image_path = 'result.png'
cv2.imwrite(final_filtered_image_path, final_filtered_image * 255)

print("Process completed and the final image is saved.")
