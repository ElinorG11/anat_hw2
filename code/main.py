# imports for the HW
import numpy as np
import matplotlib.pyplot as plt
import cv2


# ---------------------------------------------- Q1 ---------------------------------------------- #

# Q1a - Load puppy image grayscale

puppy = cv2.imread(filename='../given_data/puppy.jpg')

puppy_rgb = cv2.cvtColor(puppy, cv2.COLOR_BGR2RGB)
puppy_grayscale = cv2.cvtColor(puppy, cv2.COLOR_BGR2GRAY)

fig11, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(puppy_rgb)
axes[0].set_title("Adorable Puppy")
axes[1].imshow(puppy_grayscale, cmap='gray')
axes[1].set_title("Adorable Puppy in Grayscale")
plt.tight_layout()
plt.show()

# Q1b - Display Grayscale Puppy Histogram

hist, bins = np.histogram(puppy_grayscale.ravel(), 256, [0, 255])

fig = plt.figure()
plt.xlim([0, 255])
plt.plot(hist)
plt.title('Histogram of Puppy in Grayscale')

plt.show()

# Q1c
def gamma_correction(img, gamma):
    """
    Perform gamma correction on a grayscale image.
    :param img: An input grayscale image - ndarray of uint8 type.
    :param gamma: the gamma parameter for the correction.
    :return:
    gamma_img: An output grayscale image after gamma correction -
    uint8 ndarray of size [H x W x 1].
    """
    # ====== YOUR CODE: ======
    gamma_img = np.zeros(np.shape(img))

    # for gamma correction formula we assumed pixel values are in the range of [0,1]
    gamma_img = [(x/255) ** gamma for x in img]
    # now we should perform opposite transform from [0,1] to [0,255]
    gamma_img = np.array([x * 255 for x in gamma_img]).reshape(np.shape(img))
    # ========================

    return gamma_img


corrected_img_1 = gamma_correction(puppy_grayscale, 0.5)
corrected_img_2 = gamma_correction(puppy_grayscale, 1.5)

fig13, axes = plt.subplots(1, 3, figsize=(10, 10))
axes[0].imshow(puppy_grayscale, cmap='gray')
axes[0].set_title("Adorable Puppy in Grayscale")
axes[1].imshow(corrected_img_1, cmap='gray')
axes[1].set_title("Correction with Gamma=0.5")
axes[2].imshow(corrected_img_2, cmap='gray')
axes[2].set_title("Correction with Gamma=1.5")
plt.tight_layout()
plt.show()


# ---------------------------------------------- Q3 ---------------------------------------------- #

# Q3a1 - Plot Keyboard
keyboard = cv2.imread(filename='../given_data/keyboard.jpg')

keyboard_rgb = cv2.cvtColor(keyboard, cv2.COLOR_BGR2RGB)

fig311 = plt.figure(2)
plt.title("Keyboard")
plt.imshow(keyboard_rgb)
plt.show()

# Q3a2 - Define Kernels
kernel_horizontal = np.ones((1, 8), np.uint8)
kernel_vertical = np.ones((1, 8), np.uint8).T

# Q3a3 - Apply Erosion Filtering Using Different Kernels
# We expect that the horizontal filter will smooth vertical changes
# the vertical filter will smooth the horizontal changes
horizontal_erosion = cv2.erode(keyboard_rgb, kernel_horizontal)
vertical_erosion = cv2.erode(keyboard_rgb, kernel_vertical)

fig312, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(horizontal_erosion)
axes[0].set_title("Keyboad Erosion using Horizontal Kernel")
axes[1].imshow(vertical_erosion)
axes[1].set_title("Keyboad Erosion using Vertical Kernel")
plt.tight_layout()
plt.show()

# Q3a4
eroded_summation = cv2.add(horizontal_erosion, vertical_erosion)

fig314 = plt.figure(3)
plt.title("Summation of Eroded pictures")
plt.imshow(eroded_summation)
plt.show()

binary_keyboard = cv2.threshold(eroded_summation, 0.2*255, 255, type=cv2.THRESH_BINARY)[1]

# Q3b
inverse_keyboard = cv2.threshold(eroded_summation, 0.2*255, 255, type=cv2.THRESH_BINARY_INV)[1]

keyboard_median_filtering = cv2.medianBlur(src=inverse_keyboard, ksize=9)

fig3134, axes = plt.subplots(1, 3, figsize=(10, 10))
axes[0].imshow(binary_keyboard, cmap='gray')
axes[0].set_title("Binary Keyboard")
axes[1].imshow(inverse_keyboard, cmap='gray')
axes[1].set_title("Inverse Keyboard")
axes[2].imshow(keyboard_median_filtering)
axes[2].set_title("Filtered Keyboad")
plt.tight_layout()

plt.show()

# Q3c
kernel_window = np.ones((8, 8), np.uint8)
rectangular_erosion = cv2.erode(keyboard_median_filtering, kernel_window)

fig33 = plt.figure(4)
plt.title("Second Erosion with Rectangular Kernel")
plt.imshow(rectangular_erosion)
plt.show()

# Q3d1
keyboard_contour = rectangular_erosion/255

multiplication_result = keyboard_contour.astype(np.uint8) * keyboard.astype(np.uint8)

# Q3d2
sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_keyboard = cv2.filter2D(multiplication_result, -1, sharpening_kernel, cv2.BORDER_CONSTANT)

fig34 = plt.figure(5)
plt.title("Keyboard with Sharpened Keys Contour")
plt.imshow(sharpened_keyboard)
plt.show()


"""
fig3134, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(keyboard_rgb)
axes[0].set_title("1")
axes[1].imshow(sharpened_keyboard)
axes[1].set_title("2")
plt.tight_layout()
plt.show()
"""

threshold = 0.8*255

denoised_sharpened_keyboared = cv2.threshold(sharpened_keyboard, threshold, 255, cv2.THRESH_BINARY)[1]
fig343, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(sharpened_keyboard)
axes[0].set_title("Sharpened Keyboard")
axes[1].imshow(denoised_sharpened_keyboared)
axes[1].set_title("Denoised Sharpened Keyboard")
plt.tight_layout()
plt.show()
