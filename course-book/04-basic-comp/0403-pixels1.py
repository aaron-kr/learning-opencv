# 0403.py
import cv2 
# import numpy as np 

img = cv2.imread('../../img/spirit-week.jpg', cv2.IMREAD_GRAYSCALE)
img[100,200] = 0 # change pixel (brightness, grayscale)
print(img[100:110, 200:210]) # ROI (Region of interest) access

## for y in range(100, 400):
##  for x in range(200, 300):
##    img[y,x] = 0

img[100:400, 200:300] = 0 # ROI (Region of interest) access

cv2.imshow('img', img)
cv2.waitKey() 
cv2.destroyAllWindows()