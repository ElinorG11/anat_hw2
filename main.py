# imports for the HW
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Q1a - Load puppy image grayscale

puppy = cv2.imread(filename='../my_data/building.jpg')

puppy_rgb = cv2.cvtColor(puppy, cv2.COLOR_BGR2RGB)
puppy_grayscale = cv2.cvtColor(puppy, cv2.COLOR_BGR2GRAY)
