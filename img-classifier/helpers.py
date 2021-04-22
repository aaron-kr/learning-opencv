# Helper functions
import os 
import glob # library for loading images from a directory
import matplotlib.image as mpimg 
import cv2 
import torch
import numpy as np
import random


def load_dataset(image_directory):
    # Populate this empty image list
    im_list = []
    im_types = ["winter", "spring", "summer", "fall"]
    
    # Iterate through each color folder
    for im_type in im_types:
        
        # Iterate through each file in each image_type folder
        # glob reads in any image with the ext "image_directory/im_type/*"
        for file in glob.glob(os.path.join(image_directory, im_type, "*")):
            
            # Read in the image
            im = mpimg.imread(file)
            
            # Check if the image exists / if it's been correctly read in
            if not im is None:
                # Append the image, and its type (winter, spring, summer, fall) to the image list
                im_list.append((im, im_type))
                
    return im_list


## Standardize input images

# Grayscale and Normalize color range to [0,1].
class GrayNorm(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""
    
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        
        im_cp = np.copy(image)
        lb_cp = np.copy(label)
        
        # convert image to grayscale
        im_cp = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0,255] to [0,1]
        im_cp = im_cp / 255.0
        
        # standardize labels?
        
        return {0: im_cp, 1: lb_cp}


# Resize each image to desired input size.
class Rescale(object):
    """Rescale the image in a sample to a given size.
    
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
            
        new_h, new_w = int(new_h), int(new_w)
        
        scaled_img = cv2.resize(image, (new_w, new_h))
        
        return {0: scaled_img, 1: label}

    
# Randomly crop the image in a sample.
def RandCrop(object):
    """Crop randomly the image in a sample.
    
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        cropped_img = image[top:top+new_h, left:left+new_w]
        
        return {0: cropped_img, 1: label}

    
# Randomly rotate the image in a sample.
def RandRotate(object):
    """Rotate randomly the image in a sample."""
    
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        
        rotation = random.choice([0, 90, 180, 270])
        
        # Matrix / affine transforms needed...
        
        return {0: rotated_img, 1: label}
    
    
# Randomly flip the image in a sample.
class RandFlip(object):
    """Flip randomly, horizontally or vertically, the image in a sample."""
    
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        
        h, w, c = image.shape
        
        flip = random.randint(0, 1)
        
        if flip == 1: # only flip sometimes, when flip == 1
            horz = random.randint(0, 1)
            
            if horz == 1: # flip horizontally
                image = image[:, ::-1, :]
            else: # flip vertically
                image = image[::-1, :, :]
                
        return {0: image, 1: label}
    

# Convert the ndarray image in the sample to a Tensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        
        # if image has no grayscale color channel, add one
        if (len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1) # 3 channels for RGB images
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        
        return {0: torch.from_numpy(image), 1: torch.from_numpy(label)}
    

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