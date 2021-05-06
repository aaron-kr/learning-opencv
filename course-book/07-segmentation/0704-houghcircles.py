# 0704.py
import cv2
import numpy as np

#1
src1 = cv2.imread('../../img/circles.jpeg')
gray1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
circles1 = cv2.HoughCircles(gray1, method = cv2.HOUGH_GRADIENT, dp = 1, minDist = 50, param2 = 15)

print('circles1.shape = ', circles1.shape)
for circle in circles1[0,:]:
	cx, cy, r = circle
	cv2.circle(src1, (cx,cy), int(r), (255,0,0), 2)
cv2.imshow('src1', src1)

#2
src2 = cv2.imread('../../img/circles.jpeg')
gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
circles2 = cv2.HoughCircles(gray2, method = cv2.HOUGH_GRADIENT, dp = 1, minDist = 50, param2 = 15, minRadius = 30, maxRadius = 100)

print('circles2.shape = ', circles2.shape)
for circle in circles2[0,:]:
	cx, cy, r = circle
	cv2.circle(src2, (cx,cy), int(r), (255,0,0), 2)
cv2.imshow('src2', src2)

cv2.waitKey(0)
cv2.destroyAllWindows()