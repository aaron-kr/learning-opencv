# 0405.py
import cv2 
import numpy as np 

img = cv2.imread('../../img/spirit-week.jpg')

## for y in range(100, 400):
##  for x in range(200, 300):
##    img[y, x, 0] = 255 # Change the blue channel to 255

# img[100:400, 200:300, 0] = 255 # B - blue to 255 
# img[100:400, 200:300, 1] = 255 # G - green to 255 
img[100:400, 200:300, 2] = 255 # R - red to 255 

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()