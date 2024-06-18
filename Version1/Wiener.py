import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Load the image
image = cv2.imread('images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif', cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# creating noise of same dimensions as the image
mean = 0
std_dev = 1
noise = np.random.normal(mean, std_dev, image.shape)
print(noise.shape, image.shape)

# adding noise to image
# noisy_image = np.array(np.clip(np.array(noise + img, dtype=np.float64), 0, 255), dtype=np.uint8)
x_pixels = np.random.randint(0, 520, 5000)
y_pixels = np.random.randint(0, 704, 5000)

noisy_image = image.copy()
noisy_image[x_pixels[:2500], y_pixels[:2500]] = 255
noisy_image[x_pixels[2500:], y_pixels[2500:]] = 0

# Display the noisy image
plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

# Apply Wiener filter
denoised_image = wiener(noisy_image.astype(np.float64), (5, 5))

# Convert the denoised image back to uint8
denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)

# Display
plt.subplot(1, 3, 3)
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')

plt.show()

cv2.imwrite('denoised_image_wiener.png',denoised_image)