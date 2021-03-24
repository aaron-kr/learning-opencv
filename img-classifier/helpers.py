# Helper functions
import os 
import glob # library for loading images from a directory
import matplotlib.image as mpimg 
import cv2 

# load_dataset(dir)
# This function loads in images and labels, and places them in a list
# which contains all images and associated labels.
# Example: im_list[0][:] is the first image-label pair in the list
def load_dataset(img_dir):
  # Populate the empty image list
  im_list = []
  image_types = ["winter", "spring", "summer", "fall"]

  # Iterate through each color folder
  for im_type in image_types:
    # Iterate through each image file in each image_type folder
    # glob reads in any image with the extension "image_dir/im_type/*"
    for file in glob.glob( os.path.join( image_dir, im_type, "*" ) ):
      
      # Read in the image
      im = mpimg.imread(file)

      # Check if the image exists / if it's been read correctly
      if not im is None:
        # Append the image, and its type (winter, spring, summer, fall) to the image list
        im_list.append((im, im_type))

  return im_list 

# Standardize input images
# Resize each image to desired input size: 300x400px (hxw)

# standardize_input(img)
# This function takes in an RGB image and returns a new, standardized version
# 300 height x 400 width
def standardize_input(img):
  # Resize image and pre-process so that all "standard" images are uniform
  std_im = cv2.resize(img, (400, 300))

  return std_im

# Standardize the output
# With each loaded image, we also specify the expected output.

# encode(lbl)
# Use int values: 0/1/2/3 = winter/spring/summer/fall
def encode(lbl):
  num_val = 0
  if (lbl == 'spring'):
    num_val = 1
  elif (lbl == 'summer'):
    num_val = 2
  elif (lbl == 'fall'):
    num_val = 3
  # else it is winter, and remains 0

  return num_val

# Combine both functions from above to standardize imgs and lbls
def standardize(img_list):
  # Empty img data array
  std_list = []

  # Iterate through all img-lbl pairs
  for item in image_list:
    img = item[0]
    lbl = item[1]

    # Standardize the image
    std_img = standardize_input(img)

    # Create numerical label
    std_lbl = encode(lbl)

    # Append img, and its one-hot encoded label to the full, processed list of img data
    std_list.append((std_img, std_lbl))
  
  return std_list