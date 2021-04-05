# 0501.py
import cv2
import numpy as np 

src = cv2.imread('../../img/link-arrow-512.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', src)

ret, dst = cv2.threshold(src, 60, 255, cv2.THRESH_BINARY)
print('ret = ', ret)
cv2.imshow('dst', dst)

ret2, dst2 = cv2.threshold(src, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print('ret2 = ', ret2)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()