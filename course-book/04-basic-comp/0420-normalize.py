# 0420.py
import cv2
import numpy as np 

src = cv2.imread('../../img/link-512.jpg', cv2.IMREAD_GRAYSCALE)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src)
print('src: ', minVal, maxVal, minLoc, maxLoc)

dst = cv2.normalize(src, None, 100, 200, cv2.NORM_MINMAX)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dst)
print('src: ', minVal, maxVal, minLoc, maxLoc)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()