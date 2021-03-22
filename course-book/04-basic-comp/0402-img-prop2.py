# 0402.py
import cv2 
## import numpy as np 

img = cv2.imread('../../img/spirit-week.jpg', cv2.IMREAD_GRAYSCALE)
print('img.shape = ', img.shape)

# img = img.reshape(img.shape[0] * img.shape[1])
img = img.flatten()
print('img.shape = ', img.shape)

img = img.reshape( -1, 399, 820 ) # distorts the img - original size is 820 tall x 399 wide
print('img.shape = ', img.shape)

cv2.imshow('img', img[0]) # Display the grayscale img
cv2.waitKey()
cv2.destroyAllWindows()