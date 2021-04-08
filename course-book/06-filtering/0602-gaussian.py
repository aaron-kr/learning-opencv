# 0602.py
import cv2 
import numpy as np 

src = cv2.imread('../../img/link-arrow-512.png', cv2.IMREAD_GRAYSCALE)

dst1 = cv2.medianBlur(src, ksize = 7)
dst2 = cv2.blur(src, ksize = (7,7))
dst3 = cv2.GaussianBlur(src, ksize = (7,7), sigmaX = 0.0)
dst4 = cv2.GaussianBlur(src, ksize = (7,7), sigmaX = 10.0) # sigmaX is like stddev

cv2.imshow('dst1 = medianBlur', dst1) # looks a bit like a painting
cv2.imshow('dst2 = blur', dst2) # more blurred than above
cv2.imshow('dst3 = GaussianBlur, sigmaX = 0 (stddev)', dst3)
cv2.imshow('dst4 = GaussianBlur, sigmaX = 10 (stddev)', dst4) # more blurred than above
cv2.waitKey()
cv2.destroyAllWindows()