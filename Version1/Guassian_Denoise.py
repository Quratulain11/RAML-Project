import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, var=100):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy = cv.add(image.astype('float32'), gauss)
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype('uint8')

# read image into an array
img = cv.imread('images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif')

noisy_image=add_gaussian_noise(img)

denoised_image_box = cv.blur(noisy_image, (3, 3))

# visualize actual image vs noisy image vs denoised image
fig, axs = plt.subplots(3,3, figsize=(25, 20))

axs[0, 0].imshow(img)
axs[0, 0].title.set_text('Original Image')
axs[0, 1].imshow(noisy_image)
axs[0, 1].title.set_text('Noisy Image')
axs[0, 2].imshow(denoised_image_box)
axs[0, 2].title.set_text('Denoised Image - Box')


# plt.show()

# remove noise using the Gaussian filter
denoised_image_gaussian = cv.GaussianBlur(noisy_image, (5,5), 0)
# fig2, axs2 = plt.subplots(1,3, figsize=(15, 10))

axs[1, 0].imshow(img)
axs[1, 0].title.set_text('Original Image')

axs[1, 1].imshow(noisy_image)
axs[1, 1].title.set_text('Noisy Image')

axs[1, 2].imshow(denoised_image_gaussian)
axs[1, 2].title.set_text('Denoised Image - Gaussian')


# plt.show()

# remove noise using Median filter
denoised_image_median = cv.medianBlur(noisy_image, 5)
# fig3, axs3 = plt.subplots(1,3, figsize=(15, 10))

axs[2, 0].imshow(img)
axs[2, 0].title.set_text('Original Image')

axs[2, 1].imshow(noisy_image)
axs[2, 1].title.set_text('Noisy Image')

axs[2, 2].imshow(denoised_image_median)
axs[2, 2].title.set_text('Denoised Image - Median')


plt.show()

cv.imwrite('guassian_denoised_image_box.png',denoised_image_box)
cv.imwrite('guassian_denoised_image_gaussian.png',denoised_image_gaussian)
cv.imwrite('guassian_denoised_image_median.png',denoised_image_median)