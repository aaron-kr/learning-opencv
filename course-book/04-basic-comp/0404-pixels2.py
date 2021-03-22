#0404.py
import cv2
## import numpy as np 

img = cv2.imread('../../img/spirit-week.jpg')
img[100, 200] = [255, 0, 0] # Change color (BGR)
print(img[100, 200:210]) # ROI (Region of interest) access

## for y in range(100, 400):
##  for x in range(200, 300):
##    img[y,x] = [255, 0, 0] # change to blue

img[100:400, 200:300] = [255, 0, 0] # ROI (Region of interest) access

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()