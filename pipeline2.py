import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
start = time.time()


# Load the image
image_path = 'test.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. 이미지 분해: Gaussian 필터를 사용하여 저주파 성분과 고주파 성분 분리
# 저주파 성분 추출
low_freq = cv2.GaussianBlur(image, (21, 21), 0)

# 고주파 성분 추출: 원본 이미지에서 저주파 성분을 빼서 얻음
high_freq = cv2.subtract(image, low_freq)

# 2. 저주파 성분 클러스터링 (K-means 또는 Adaptive K-means)
scaler = StandardScaler()
low_freq_flat = low_freq.flatten().reshape(-1, 1)
low_freq_scaled = scaler.fit_transform(low_freq_flat)

# K-means 클러스터링
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels_low = kmeans.fit_predict(low_freq_scaled)

# 저주파 결과를 원본 이미지 크기로 변환
kmeans_labels_low_image = kmeans_labels_low.reshape(low_freq.shape)

# 3. 고주파 성분 노이즈 제거
# 고주파 성분의 작은 잡음 제거를 위해 Morphological 연산 적용
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
high_freq_denoised = cv2.morphologyEx(high_freq, cv2.MORPH_OPEN, kernel)

# 4. 저주파 및 고주파 결과 병합
# 저주파에서 탐지된 오일 유출 영역과 고주파에서 제거된 노이즈 결합
kmeans_labels_low_image_uint8 = (kmeans_labels_low_image * 255).astype(np.uint8)
high_freq_denoised_uint8 = cv2.convertScaleAbs(high_freq_denoised)
final_result = cv2.addWeighted(kmeans_labels_low_image_uint8, 0.7, high_freq_denoised_uint8, 0.3, 0)

# 5. 연결된 구성 요소 분석을 통해 작은 영역 제거 (Connected Component Analysis)
_, binary_cleaned_image = cv2.threshold(final_result, 127, 1, cv2.THRESH_BINARY)

# Perform connected component analysis
num_labels, labels_im = cv2.connectedComponents(binary_cleaned_image.astype(np.uint8))

# Filter out small connected components based on size
min_size = 200  # Minimum size of region to keep (you can adjust this based on your needs)
filtered_image = np.zeros_like(labels_im)
for label in range(1, num_labels):  # Skip the background label
    if np.sum(labels_im == label) > min_size:
        filtered_image[labels_im == label] = 1

# Save the final filtered result
final_filtered_result_path = 'result.png'
cv2.imwrite(final_filtered_result_path, filtered_image * 255)

print("Process completed and the final image is saved.")

end = time.time()

finish = end - start
print("소요시간 : ",finish, "초")
