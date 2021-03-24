# season-classifier.py
import cv2 # computer vision library
import helpers

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

%matplotlib inline # used for Jupyter notebooks, to display plots and output directly below the code cell that produced it.

# Image data directories
imgs_train = 'img/train/'
imgs_test = 'img/test/'

# 1. INPUT DATA - with load_dataset function from helpers.py
# Load training data
IMG_LIST = helpers.load_dataset(imgs_train)

# 2. PRE-PROCESSING - standardize all training images
STD_LIST = helpers.standardize(IMG_LIST)

# Display one standardized image and its label
# Select an image by index
img_index = 0
selected_img = STD_LIST[img_index][0]
selected_lbl = STD_LIST[img_index][1]

# Display image and data
plt.imshow(selected_img)
print("Shape: ", str(selected_img.shape))
print("Label: ", str(selected_lbl))

# 3. AOI / ROI - not applicable at the moment
# 4. FEATURE EXTRACTION - convert color space, compare
# Find the average Value or brightness of an image
def avg_brightness(rgb_img):
  # Convert to HSV
  hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

  # Add up all the pixel values in the V channel
  sum_brightness = np.sum(hsv[:,:,2])
  area = WIDTH * HEIGHT # pixel size - set elsewhere

  # Average it
  avg = sum_brightness / area 

  return avg 

# Testing average brightness levels
# Look at a number of different images and think abt what average brightness separates the different images
img_index = 3
test_img = STD_LIST[img_index][0]

avg = avg_brightness(test_img)
print('Avg brightness: ', str(avg))
plt.imshow(test_img)

# 5. RECOGNITION / PREDICTION - Estimate / classify the image label
# This function should take in an RGB image input
def estimate_lbl(rgb_img):
  # Extract average brightness from RGB image
  avg = avg_brightness(rgb_img)

  # Use brightness to predict the label (0, 1, 2, 3)
  predicted_lbl = 0 # default

  # Try different threshold values to produce a better result
  threshold = 100
  if ( avg > threshold ):
    predicted_lbl = 1

  return predicted_lbl