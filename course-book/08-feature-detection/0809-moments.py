# 0809.py
import cv2
import numpy as np

#1
src = cv2.imread('../../img/shapes.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, bImg = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

#2
## M = cv2.moments(bImg)
M = cv2.moments(bImg, True)
for key, value in M.items():
    print('{} = {}'.format(key, value))

#3
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
dst = src.copy()
cv2.circle(dst, (cx,cy), 5, (0,0,255), 2)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()