# 0707.py
import cv2
import numpy as np

#1
src = np.zeros(shape = (512, 512, 3), dtype = np.uint8)
cv2.rectangle(src, (50, 100), (450, 400), (255,255,255), -1)
cv2.rectangle(src, (100,150), (400, 350), (0,0,0), -1)
cv2.rectangle(src, (200,200), (300, 300), (255,255,0), -1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#2
mode = cv2.RETR_LIST
method = cv2.CHAIN_APPROX_SIMPLE
## method = cv2.CHAIN_APPROX_NONE

# don't need 'image' below - since openCV 3.2 source image is not modified by this function
contours, hierarchy = cv2.findContours(gray, mode, method)
# print('type(contours) = ', type(contours))
# print('type(contours[0]) = ', type(contours[0]))
print('len(contours) = ', len(contours))
print('contours[0].shape = ', contours[0].shape)
print('contours[0] = ', contours[0])

#3
for cnt in contours:
	cv2.drawContours(src, [cnt], 0, (255,0,0), 3) # all shapes

	for pt in cnt: # find shape
		cv2.circle(src, (pt[0][0], pt[0][1]), 5, (0,0,255), -1)

cv2.imshow('src', src)
cv2.waitKey(0)
cv2.destroyAllWindows()