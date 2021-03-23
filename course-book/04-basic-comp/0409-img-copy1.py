# 0409.py
import cv2 
src = cv2.imread('../../img/spirit-week.jpg', cv2.IMREAD_GRAYSCALE)

## dst = src # reference
dst = src.copy() # copy
dst[100:400, 200:300] = 0

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()  
cv2.destroyAllWindows()