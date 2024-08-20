# Apply a lower threshold to detect more regions (potential oil spill areas)
_, lower_thresholded_image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY_INV)

# Flatten the image for clustering
image_flattened = lower_thresholded_image.reshape(-1, 1)

# Apply K-means clustering on the thresholded image
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(image_flattened)
kmeans_clustered_image = kmeans_labels.reshape(lower_thresholded_image.shape)

# Plot the original thresholded image and the K-means clustered image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Lower Thresholded Image')
plt.imshow(lower_thresholded_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('K-means Clustered Image (After Lower Thresholding)')
plt.imshow(kmeans_clustered_image, cmap='gray')
plt.axis('off')

plt.show()
