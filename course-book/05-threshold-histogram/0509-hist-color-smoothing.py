# 0509.py
import cv2
import numpy as np

src = cv2.imread('../../img/link-arrow-512.png')
cv2.imshow('src', src)

#1
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

v2 = cv2.equalizeHist(v)
hsv2 = cv2.merge([h,s,v2])
dst = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
cv2.imshow('dst', dst)

#2
yCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
y,Cr,Cb = cv2.split(yCrCb)

y2 = cv2.equalizeHist(y)
yCrCb2 = cv2.merge([y2,Cr,Cb])
dst2 = cv2.cvtColor(yCrCb2, cv2.COLORYCrCb2BGR)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()