import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# Load the image (assuming it's a 3D image like a stack of 2D images)
image_path = 'images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create a 3D image from a stack of the same image for demonstration
# In practice, load a 3D volumetric image directly
volume = np.stack([image] * 10, axis=-1)  # Stack the image to create a 3D volume

# Function to add salt-and-pepper noise to a 3D image
def add_salt_and_pepper_noise(volume, amount=0.05):
    noisy_volume = volume.copy()
    num_voxels = volume.size
    num_salt = int(amount * num_voxels / 2)
    num_pepper = int(amount * num_voxels / 2)

    # Add salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in volume.shape]
    noisy_volume[coords[0], coords[1], coords[2]] = 255

    # Add pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in volume.shape]
    noisy_volume[coords[0], coords[1], coords[2]] = 0

    return noisy_volume

# Add salt-and-pepper noise to the 3D image
noisy_volume = add_salt_and_pepper_noise(volume, amount=0.05)

# Apply a 3D median filter for denoising
denoised_volume = median_filter(noisy_volume, size=3)

# Plot a slice of the original, noisy, and denoised volume
slice_index = volume.shape[2] // 2  # Use the middle slice for visualization
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(volume[:, :, slice_index], cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_volume[:, :, slice_index], cmap='gray')
plt.title('Noisy')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(denoised_volume[:, :, slice_index], cmap='gray')
plt.title('Denoised')
plt.axis('off')

plt.show()

cv2.imwrite('denoised_image_box3d.png',denoised_volume[:, :, slice_index])
