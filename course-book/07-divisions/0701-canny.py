# 0701.py
import cv2
import numpy as np

src = cv2.imread('../../img/spirit-week.jpg', cv2.IMREAD_GRAYSCALE)

edges1 = cv2.Canny(src, 50, 100)
edges2 = cv2.Canny(src, 50, 200)

cv2.imshow('edges1', edges1)
cv2.imshow('edges2', edges2)
cv2.waitKey(0)
cv2.destroyAllWindows()