# imports for the HW
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Q1a - Load puppy image grayscale

puppy = cv2.imread(filename='../given_data/puppy.jpg')

puppy_rgb = cv2.cvtColor(puppy, cv2.COLOR_BGR2RGB)
puppy_grayscale = cv2.cvtColor(puppy, cv2.COLOR_BGR2GRAY)

fig11, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(puppy_rgb)
axes[0].set_title("Adorable Puppy")
axes[1].imshow(puppy_grayscale, cmap='gray')
axes[1].set_title("Adorable Pyppy Grayscaled")
plt.tight_layout()
plt.show()


