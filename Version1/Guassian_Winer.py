import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

def add_gaussian_noise(image, mean=0, var=100):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy = cv2.add(image.astype('float32'), gauss)
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype('uint8')

def denoise_image_with_wiener(noisy_image):
    # Applying Wiener filter
    denoised_channels = []
    for i in range(3):  # Assuming a 3-channel (RGB) image
        channel = noisy_image[:, :, i]
        denoised_channel = wiener(channel, mysize=(5, 5))
        denoised_channels.append(denoised_channel)
    denoised_image = np.stack(denoised_channels, axis=-1)
    return denoised_image.astype('uint8')

# Load the image
image_path = 'images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Increase the noise by increasing the variance
noisy_image = add_gaussian_noise(image)

# Denoise the image using Wiener filter
denoised_image = denoise_image_with_wiener(noisy_image)

# Display the original, noisy, and denoised images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Denoised Image (Wiener)')
plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

cv2.imwrite('guassian_denoised_image_Wiener.png',denoised_image)