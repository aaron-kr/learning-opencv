# 0407.py
import cv2 
import numpy as np 

src = cv2.imread('../../img/spirit-week.jpg', cv2.IMREAD_GRAYSCALE)
roi = cv2.selectROI(src)
print('roi = ', roi)

img = src[roi[1]:roi[1] + roi[3],
          roi[0]:roi[0] + roi[2]]

cv2.imshow('img', img)
cv2.waitKey() 
cv2.destroyAllWindows()